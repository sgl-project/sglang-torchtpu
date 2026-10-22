# SPDX-License-Identifier: Apache-2.0
"""DSpark draft model (dense Qwen3-style block drafter + vanilla Markov head) in plain JAX, TP-sharded on
the megakernel mesh, for the standalone server's on-device speculative loop.

Interface (agreed with the engine; keep it stable):

* ``DraftModel.load(path, mesh, *, target_ckpt=None, embed_mode="full", tp=None)`` reads the draft checkpoint
  (e.g. /work/wenhao/k3_dspark_draft) and shards it over ``mesh`` (axis ``AXIS``). With ``target_ckpt``
  the draft also keeps its own copy of the target embedding / lm_head (tests; the engine passes its own).
* ``draft.init_state(mesh, capacity) -> dstate``: pytree of ``[world, ...]`` arrays sharded ``P(AXIS)``
  (draft weights + the draft context KV for ``capacity`` positions). Inside ``shard_map`` the caller strips
  the rank axis (``tree.map(lambda v: v[0], dstate)``) and passes the local view to the functions below.
* ``draft.reset(ds) -> ds`` (local view): identity. Stale KV beyond the anchor position is never read, so
  a new request needs no clearing.
* ``draft.append_context(ds, captures [n, C, H], positions [n], n_valid) -> ds`` (local view, jittable,
  usable in ``lax.fori_loop``): the captured target hidden states (C = 5 layers in capture order, i.e.
  layer order [7, 23, 51, 67, 83], or [0, 3, 6, 9, 12] on the 13-layer stack) of tokens at ``positions``
  go into the draft KV. Each capture is what sglang's KimiK3LinearModel._dspark_capture_stream feeds the
  projector: the attention-residual mixture before layer l + 1's input norm, i.e. the megakernel's
  ``capture_mixture=True`` (the sglang reference dumps were taken that way); rows ``>= n_valid`` are dropped. ``n`` is static (8 in the engine).
* ``draft.propose(ds, anchor, pos, embed=None, lm_head=None) -> int32 [ndraft]`` (local view, jittable):
  greedy drafts d_1..d_ndraft for positions pos+1..pos+ndraft (``ndraft`` = load arg, default 7 = sglang's
  gamma with speculative_num_draft_tokens 8; the block width changes the drafts, so 6 does not equal 7[:6]), where ``anchor`` (the last committed / bonus token)
  sits at ``pos`` and the draft KV holds positions [0, pos). ``embed`` is the engine's per-rank embedding
  (``[V, H]`` full or ``[V/world, H]`` sharded) and ``lm_head`` its per-rank ``[V/world, H]`` rows; ``None``
  uses the copies loaded with ``target_ckpt``.

Semantics (sglang TpuBlockSpecWorker + TpuDsparkDraftWorker + DSparkDraftModel, greedy, sample_from_anchor):

* context: ``c = hidden_norm(fc(concat_l capture_l))``; per draft layer ``k = rope(k_norm(k_proj(c)))``,
  ``v = v_proj(c)`` at the token's own position (prefill: every prompt row; a round: the a + 1 committed
  verify rows [anchor, d_1..d_a]). The draft layers never run on context tokens.
* block: ``[anchor, mask x (ndraft - 1)]`` (mask 163824) at positions pos..pos+ndraft-1 through the 5 layers with target
  embeddings; attention is bidirectional over context [0, pos) plus the whole block.
* drafts: base logits ``l_j = norm(h_j) @ lm_head^T`` (row j-1 predicts position pos+j);
  ``d_j = argmax(l_j + w2 @ w1[d_{j-1}])``, ``d_0 = anchor`` (sglang rounds both terms and the sum to bf16).
* sglang parity: the sglang TPU path's later rounds also attend to every earlier round's block anchor-row K/V
  (a second entry at the anchor's position next to its committed context K/V); ``init_state(...,
  sglang_compat=True)`` + ``propose_state`` reproduce that for the replay of sglang reference dumps
  (test/manual/k3_megakernel/test_dspark_draft.py --compat). Serving uses the clean semantics above.
* R-invariance: ``propose`` / ``append_context`` are fenced by ``optimization_barrier`` and RoPE reads a
  host-built cos|sin table, so the drafts do not depend on the program the draft is compiled into (the
  engine's rounds-per-call R).
* the confidence head is not used by the greedy serving path (sglang verifies all gamma drafts); not loaded.
"""

from __future__ import annotations

import json
import math
import os
import struct
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sglang_tpu.k3_megakernel.mesh import AXIS, local_ranks
from sglang_tpu.k3_megakernel.model_io import put_rows

CHUNK = 2048  # context rows per attention step (arbitrary; capacity must be a multiple)
# Up to this capacity the context is attended in one masked shot over all rows instead of a loop of CHUNK-row
# steps: at small batch the draft is bound by per-op latency, not by the 2 x cap x hd bytes it reads.
ONESHOT_MAX = 16384
MARKOV_KERNEL = (
    True  # the Markov chain as one Pallas kernel (draft_kernels.markov_chain) on TPU
)
NDRAFT = 7  # sglang DSpark default: speculative_num_draft_tokens 8 = anchor + 7 drafts

_ST = {
    "BF16": "bfloat16",
    "F32": "float32",
    "F16": "float16",
    "I64": "int64",
    "I32": "int32",
}


def _st_dtype(name):
    if name == "BF16":
        import ml_dtypes

        return np.dtype(ml_dtypes.bfloat16)
    return np.dtype(_ST[name])


class _File:
    """Memory-mapped reader of one safetensors file."""

    def __init__(self, path):
        self.path = path
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            self.base, self.header = 8 + n, json.loads(f.read(n))

    def get(self, name):
        m = self.header[name]
        a, b = m["data_offsets"]
        return np.memmap(
            self.path,
            dtype=_st_dtype(m["dtype"]),
            mode="r",
            offset=self.base + a,
            shape=tuple(m["shape"]),
        )


def _target_tensor(ckpt, name):
    with open(os.path.join(ckpt, "model.safetensors.index.json")) as f:
        wm = json.load(f)["weight_map"]
    for full in (name, "language_model." + name):
        if full in wm:
            return _File(os.path.join(ckpt, wm[full])).get(full)
    raise KeyError(name)


def yarn_inv_freq_mscale(cfg):
    """sglang YaRNScalingRotaryEmbedding (beta_fast 32, beta_slow 1, truncate) or plain RoPE."""
    rp = cfg.get("rope_parameters") or cfg.get("rope_scaling") or {}
    base = float(rp.get("rope_theta", cfg.get("rope_theta", 10000.0)))
    dim = int(cfg["head_dim"])
    pos_freqs = base ** (np.arange(0, dim, 2, dtype=np.float32) / dim)
    if rp.get("rope_type", rp.get("type")) != "yarn":
        return (1.0 / pos_freqs).astype(np.float32), 1.0
    factor = float(rp["factor"])
    omax = int(rp["original_max_position_embeddings"])

    def corr(rot):
        return dim * math.log(omax / (rot * 2 * math.pi)) / (2 * math.log(base))

    low, high = max(math.floor(corr(32)), 0), min(math.ceil(corr(1)), dim - 1)
    if low == high:
        high += 0.001
    ramp = np.clip((np.arange(dim // 2, dtype=np.float32) - low) / (high - low), 0, 1)
    mask = 1 - ramp
    inv = (1.0 / (factor * pos_freqs)) * (1 - mask) + (1.0 / pos_freqs) * mask
    mscale = 0.1 * math.log(factor) + 1.0 if factor > 1 else 1.0
    return inv.astype(np.float32), float(mscale)


class DraftDims(NamedTuple):
    hidden: int
    heads: int
    kv_heads: int
    head_dim: int
    ffn: int
    layers: int
    vocab: int
    n_caps: int
    rank: int
    eps: float
    mask_token: int
    world: int

    @property
    def qpr(self):
        return self.heads // self.world

    @property
    def kvpr(self):
        return max(1, self.kv_heads // self.world)

    @property
    def group(self):
        return self.heads // self.kv_heads


def _rms(x, w, eps):
    x32 = x.astype(jnp.float32)
    y = x32 * jax.lax.rsqrt(jnp.mean(x32 * x32, axis=-1, keepdims=True) + eps)
    return (y * w.astype(jnp.float32)).astype(jnp.bfloat16)


def _dot(spec, a, b):
    """einsum with f32 accumulation (CPU dots lack bf16 x bf16 -> f32: upcast there)."""
    if jax.default_backend() == "cpu":
        a, b = a.astype(jnp.float32), b.astype(jnp.float32)
    return jnp.einsum(spec, a, b, preferred_element_type=jnp.float32)


COLLECTIVE_ID = 9  # Pallas barrier id of the draft's all-reduce; the megakernel uses 4


def _pallas_allreduce(rows, world, wire=jnp.bfloat16):
    """Standalone megakernel AllReduce (``wire`` transport, f32 sum) of a [rows, 128] f32 block over the
    ``world`` consecutive ranks of this rank's draft group (peers are ``rank ^ offset``, so a group of 8 is
    one host)."""
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    from sglang_tpu.k3_megakernel.config import Topology
    from sglang_tpu.k3_megakernel.kernel.collectives import (
        AllReduce,
        AllReduceSpec,
        entry_barrier,
    )

    topo = Topology(world // 8, 8) if world >= 8 else Topology(1, world)
    spec = AllReduceSpec(topo, rows, wire)

    def kernel(x_ref, o_ref, *scratch):
        me = jax.lax.axis_index(AXIS)
        ar = AllReduce(spec, me, *scratch)
        ar.init()
        entry_barrier(me, topo)
        o_ref[...] = ar(x_ref[...])

    vm = pl.BlockSpec(memory_space=pltpu.VMEM)
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((rows, 128), jnp.float32),
        in_specs=[vm],
        out_specs=vm,
        scratch_shapes=spec.scratch_shapes(),
        compiler_params=pltpu.CompilerParams(collective_id=COLLECTIVE_ID),
    )


def _allreduce_bf16(x, world):
    """Sum of x [R, N] over the ``world`` ranks of the draft group, bf16 out (bf16 partials summed in f32,
    like sglang's bf16 RowParallel).

    [7, 7168] on 8 cores of one host: XLA psum ~40 us, all_gather + sum ~19 us, Pallas AllReduce ~15 us."""
    if world == 1:
        return x.astype(jnp.bfloat16)
    if jax.default_backend() == "cpu" or x.size % 1024:
        assert world == jax.lax.psum(1, AXIS), (
            "draft groups need the TPU (Pallas) collectives"
        )
        every = jax.lax.all_gather(x.astype(jnp.bfloat16), AXIS)
        return every.astype(jnp.float32).sum(0).astype(jnp.bfloat16)
    rows = x.size // 128
    y = _pallas_allreduce(rows, world)(
        x.astype(jnp.bfloat16).astype(jnp.float32).reshape(rows, 128)
    )
    return y.reshape(x.shape).astype(jnp.bfloat16)


def _gather_pair(packed, world):
    """[world, 2] f32 of every group rank's ``packed`` [2] (exact). On TPU a Pallas f32 sum of one-hot rows:
    ~10 us vs ~13-19 us for XLA's all_gather in a latency-bound chain (8 cores of one host)."""
    if world == 1:
        return packed[None]
    if jax.default_backend() == "cpu":
        assert world == jax.lax.psum(1, AXIS), (
            "draft groups need the TPU (Pallas) collectives"
        )
        return jax.lax.all_gather(packed, AXIS)
    rows = -(-world // 8) * 8
    me = jax.lax.axis_index(AXIS) % world
    blk = jnp.zeros((rows, 128), jnp.float32).at[me, :2].set(packed)
    return _pallas_allreduce(rows, world, jnp.float32)(blk)[:world, :2]


def _gather_cols(x, world):
    """[n, world * m] of every group rank's x [n, m] (exact: one-hot placement summed by the bf16 AllReduce)."""
    if world == 1:
        return x
    if jax.default_backend() == "cpu":
        assert world == jax.lax.psum(1, AXIS), (
            "draft groups need the TPU (Pallas) collectives"
        )
        return jax.lax.all_gather(x, AXIS, axis=1, tiled=True)
    n, m = x.shape
    me = jax.lax.axis_index(AXIS) % world
    blk = jnp.zeros((n, world, m), jnp.bfloat16)
    blk = jax.lax.dynamic_update_slice_in_dim(
        blk, x.astype(jnp.bfloat16)[:, None], me, 1
    )
    return _allreduce_bf16(blk.reshape(n, world * m), world)


def _mm(x, w):
    """[.., K] @ [K, N] in bf16 with f32 accumulation, bf16 out (torch bf16 linear)."""
    return _dot("...k,kn->...n", x, w).astype(jnp.bfloat16)


class DraftModel:
    capacity_multiple = (
        CHUNK  # init_state's capacity must be a multiple (the engine rounds up)
    )

    def __init__(
        self, dims: DraftDims, weights: dict, inv_freq, mscale, mesh, ndraft=NDRAFT
    ):
        self.d, self.w, self.mesh = dims, weights, mesh
        self.ndraft = ndraft  # block width = drafts per round
        self.inv_freq, self.mscale = inv_freq, mscale

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(
        cls, path, mesh, *, target_ckpt=None, embed_mode="full", ndraft=NDRAFT, tp=None
    ):
        """``tp``: draft tensor-parallel group size (default: the whole mesh). With ``tp`` < world every
        group of ``tp`` consecutive ranks (8 = one host) holds and runs a full copy of the draft, so no
        draft collective leaves the group; the inputs (anchor, captures) are equal on every rank."""
        with open(os.path.join(path, "config.json")) as f:
            cfg = json.load(f)
        world = tp or mesh.devices.size
        assert mesh.devices.size % world == 0, (mesh.devices.size, world)
        d = DraftDims(
            hidden=cfg["hidden_size"],
            heads=cfg["num_attention_heads"],
            kv_heads=cfg["num_key_value_heads"],
            head_dim=cfg["head_dim"],
            ffn=cfg["intermediate_size"],
            layers=cfg["num_hidden_layers"],
            vocab=cfg["vocab_size"],
            n_caps=len(cfg["dflash_config"]["target_layer_ids"]),
            rank=cfg["markov_rank"],
            eps=cfg["rms_norm_eps"],
            mask_token=cfg["dflash_config"]["mask_token_id"],
            world=world,
        )
        assert cfg.get("markov_head_type", "vanilla") == "vanilla", cfg.get(
            "markov_head_type"
        )
        assert cfg.get("sample_from_anchor", True), "only sample_from_anchor drafts"
        assert (
            d.heads % world == 0
            and d.vocab % world == 0
            and d.ffn % world == 0
            and d.hidden % world == 0
        )
        assert d.qpr % d.group == 0 or d.group % d.qpr == 0
        f = _File(os.path.join(path, "model.safetensors"))
        ranks = local_ranks(mesh)
        H, hd, qpr, kvpr = d.hidden, d.head_dim, d.qpr, d.kvpr
        fs, vs, hs = d.ffn // world, d.vocab // world, H // world

        def kv0(r):  # first global kv head of rank r
            return (r * qpr) // d.group

        def rows(fn, dtype=None):  # fn(group rank) -> this rank's array
            by_group_rank = {}
            for r in ranks:
                if r % world not in by_group_rank:
                    by_group_rank[r % world] = np.asarray(fn(r % world))
            per = {r: by_group_rank[r % world] for r in ranks}
            a = per[ranks[0]]
            return put_rows(mesh, per, a.shape, a.dtype if dtype is None else dtype)

        def stack(name, fn):
            return rows(
                lambda r: np.stack(
                    [fn(f.get(f"layers.{i}.{name}"), r) for i in range(d.layers)]
                )
            )

        w = {}
        fc = f.get("fc.weight")  # [H, C*H]
        w["fc"] = rows(lambda r: fc[r * hs : (r + 1) * hs].T)  # [C*H, H/world]
        w["hidden_norm"] = rows(lambda r: f.get("hidden_norm.weight"))
        w["norm"] = rows(lambda r: f.get("norm.weight"))

        def qkv(
            i, r
        ):  # [H, (qpr + 2 kvpr) hd]: this rank's q heads, then its k and v heads
            p = f"layers.{i}.self_attn."
            kvs = slice(kv0(r) * hd, (kv0(r) + kvpr) * hd)
            return np.concatenate(
                [
                    f.get(p + "q_proj.weight")[r * qpr * hd : (r + 1) * qpr * hd],
                    f.get(p + "k_proj.weight")[kvs],
                    f.get(p + "v_proj.weight")[kvs],
                ]
            ).T

        w["wqkv"] = rows(lambda r: np.stack([qkv(i, r) for i in range(d.layers)]))
        # append_context's k|v columns as their own array: slicing them out of wqkv costs a strided copy
        w["wkv"] = rows(
            lambda r: np.concatenate(
                [qkv(i, r)[:, qpr * hd :] for i in range(d.layers)], 1
            )
        )
        w["wo"] = stack(
            "self_attn.o_proj.weight",
            lambda a, r: a[:, r * qpr * hd : (r + 1) * qpr * hd].T,
        )
        w["q_norm"] = stack("self_attn.q_norm.weight", lambda a, r: a)
        w["k_norm"] = stack("self_attn.k_norm.weight", lambda a, r: a)
        w["ln1"] = stack("input_layernorm.weight", lambda a, r: a)
        w["ln2"] = stack("post_attention_layernorm.weight", lambda a, r: a)
        w["wgu"] = rows(
            lambda r: np.stack(
                [
                    np.concatenate(
                        [  # [H, 2 fs]: gate then up columns
                            f.get(f"layers.{i}.mlp.gate_proj.weight")[
                                r * fs : (r + 1) * fs
                            ],
                            f.get(f"layers.{i}.mlp.up_proj.weight")[
                                r * fs : (r + 1) * fs
                            ],
                        ]
                    ).T
                    for i in range(d.layers)
                ]
            )
        )
        w["wd"] = stack(
            "mlp.down_proj.weight", lambda a, r: a[:, r * fs : (r + 1) * fs].T
        )
        w1 = np.asarray(
            f.get("markov_head.markov_w1.weight")
        )  # [V, R] (embedding, replicated)
        w["m_w1"] = rows(lambda r: w1)
        w2 = f.get(
            "markov_head.markov_w2.weight"
        )  # [V, R] (linear R -> V), rows of this rank's vocab shard
        w["m_w2"] = rows(lambda r: w2[r * vs : (r + 1) * vs])
        w["m_w2t"] = rows(
            lambda r: np.ascontiguousarray(w2[r * vs : (r + 1) * vs].T)
        )  # [R, V/world] (kernel)
        if target_ckpt:
            head = _target_tensor(target_ckpt, "lm_head.weight")
            emb = _target_tensor(target_ckpt, "model.embed_tokens.weight")
            w["lm_head_t"] = rows(
                lambda r: np.ascontiguousarray(head[r * vs : (r + 1) * vs].T)
            )
            assert embed_mode == "full" or world == mesh.devices.size
            w["embed"] = rows(
                lambda r: emb if embed_mode == "full" else emb[r * vs : (r + 1) * vs]
            )
        jax.block_until_ready(w)
        inv, ms = yarn_inv_freq_mscale(cfg)
        return cls(d, w, inv, ms, mesh, ndraft)

    def attach_lm_head(self, lm_head):
        """Keep a transposed copy ``[world, H, V/tp]`` (group rank g: vocab rows [g V/tp, (g + 1) V/tp)) of the
        engine's ``[world, V/world, H]`` head (+293 MB per rank at TP8); propose then ignores its ``lm_head``
        argument. Call before init_state."""
        sh = NamedSharding(self.mesh, P(AXIS))
        world, tp = lm_head.shape[0], self.d.world

        def regroup(a):
            t = jnp.swapaxes(a.reshape(tp, -1, a.shape[-1]), 1, 2)  # [tp, H, V/tp]
            return jnp.tile(t, (world // tp, 1, 1))

        self.w["lm_head_t"] = jax.jit(regroup, out_shardings=sh)(lm_head)

    # ------------------------------------------------------------------ state
    def init_state(self, mesh, capacity, sglang_compat=False):
        """``sglang_compat``: also keep every round's block anchor-row K/V as extra context (see propose_state)."""
        world = mesh.devices.size
        d = self.d
        shape = (world, d.layers, capacity, d.kvpr, d.head_dim)

        def fresh():
            return jnp.zeros(shape, jnp.bfloat16), jnp.zeros(shape, jnp.bfloat16)

        sh = NamedSharding(mesh, P(AXIS))
        k, v = jax.jit(fresh, out_shardings=(sh, sh))()
        # A private copy: callers donate dstate (weights included) to their round loop.
        w = jax.jit(lambda t: jax.tree.map(jnp.copy, t))(self.w)
        rope = self._rope_table(
            capacity + 2 * self.ndraft
        )  # block rows reach pos + ndraft - 1
        ds = {
            "w": w,
            "k": k,
            "v": v,
            "rope": put_rows(
                mesh, {r: rope for r in local_ranks(mesh)}, rope.shape, rope.dtype
            ),
        }
        if sglang_compat:
            ak, av = jax.jit(fresh, out_shardings=(sh, sh))()
            am = jax.jit(
                lambda: jnp.zeros((world, capacity), jnp.bool_), out_shardings=sh
            )()
            ds.update(ak=ak, av=av, am=am)
        return ds

    def reset(self, ds):
        return ds

    # ------------------------------------------------------------------ math
    def _rope_table(self, n):
        """[n, hd] f32 cos | sin rows (YaRN mscale folded in), like sglang's cos_sin_cache. Built on the host:
        an in-graph cos/sin is not bit-stable across compiled programs (R=1 vs R=8 drafts diverged)."""
        f = np.arange(n, dtype=np.float32)[:, None] * self.inv_freq[None]
        return np.concatenate(
            [np.cos(f) * self.mscale, np.sin(f) * self.mscale], -1
        ).astype(np.float32)

    def _rope(self, x, pos, table):
        """Neox RoPE of x [n, heads, hd] at positions [n] from the ``table`` rows."""
        cs = jnp.take(table, pos, axis=0)  # [n, hd]
        h = x.shape[-1] // 2
        cos, sin = cs[:, None, :h], cs[:, None, h:]
        x32 = x.astype(jnp.float32)
        a, b = x32[..., :h], x32[..., h:]
        return jnp.concatenate([a * cos - b * sin, b * cos + a * sin], -1).astype(
            jnp.bfloat16
        )

    def _headnorm(self, x, w):
        return _rms(x, w, self.d.eps)

    def append_context(self, ds, captures, positions, n_valid):
        # Barriers keep XLA from fusing the draft into the caller's ops: the fusion (so the bf16 rounding)
        # would otherwise depend on the enclosing program (engine R=1 vs R=8 drafts diverged).
        ds, captures, positions, n_valid = jax.lax.optimization_barrier(
            (ds, captures, positions, n_valid)
        )
        return jax.lax.optimization_barrier(
            self._append_context(ds, captures, positions, n_valid)
        )

    def _append_context(self, ds, captures, positions, n_valid):
        d, w = self.d, ds["w"]
        n = captures.shape[0]
        x = captures.reshape(n, d.n_caps * d.hidden).astype(jnp.bfloat16)
        c = _mm(x, w["fc"])  # [n, H/world]
        c = _gather_cols(c, d.world)  # [n, H]
        c = _rms(c, w["hidden_norm"], d.eps)
        kvs = d.kvpr * d.head_dim
        kv = (
            _mm(c, w["wkv"]).reshape(n, d.layers, 2 * kvs).transpose(1, 0, 2)
        )  # one dot for all layers
        k, v = kv[..., :kvs], kv[..., kvs:]
        k = k.reshape(d.layers, n, d.kvpr, d.head_dim)
        v = v.reshape(d.layers, n, d.kvpr, d.head_dim)
        k = _rms(k, w["k_norm"][:, None, None, :], d.eps)
        k = jax.vmap(lambda kk: self._rope(kk, positions, ds["rope"]))(k)
        cap = ds["k"].shape[1]
        idx = jnp.where(jnp.arange(n) < n_valid, positions, cap).astype(jnp.int32)
        kc = ds["k"].at[:, idx].set(k, mode="drop")
        vc = ds["v"].at[:, idx].set(v, mode="drop")
        return dict(ds, k=kc, v=vc)

    def _embed(self, table, ids):
        d = self.d
        if table.shape[0] == d.vocab:
            return jnp.take(table, ids, axis=0)
        vs = table.shape[0]
        local = ids - jax.lax.axis_index(AXIS) * vs
        mine = (local >= 0) & (local < vs)
        rows = jnp.take(table, jnp.clip(local, 0, vs - 1), axis=0)
        rows = jnp.where(mine[:, None], rows.astype(jnp.float32), 0.0)
        return jax.lax.psum(rows, AXIS).astype(jnp.bfloat16)

    def _attend(self, ds, l, q, kb, vb, pos):
        """Softmax attention of the block queries q [B, qpr, hd] over context [0, pos) of layer l plus the
        block's own kb / vb [B, kvl, hd] (GQA, kvl local kv heads); the context is scanned in CHUNK-row
        pieces up to pos only (online softmax). -> [B, qpr * hd] bf16."""
        d = self.d
        B = q.shape[0]
        kvl = kb.shape[1]
        G = d.qpr // kvl
        qg = q.reshape(B, kvl, G, d.head_dim)
        scale = 1.0 / math.sqrt(d.head_dim)
        compat = "ak" in ds
        cap = ds["k"].shape[1]
        C = min(CHUNK, cap)
        assert cap % C == 0, (cap, C)

        def scores(k):  # [c, kvl, hd] -> [kvl, G, B, c]
            return _dot("bkgd,ckd->kgbc", qg, k) * scale

        def pv(p, v):  # [kvl, G, B, c] x [c, kvl, hd] -> [kvl, G, B, hd]
            return _dot("kgbc,ckd->kgbd", p.astype(jnp.bfloat16), v)

        sb = scores(kb)
        if not compat and cap <= ONESHOT_MAX:
            sc = jnp.where(jnp.arange(cap) < pos, scores(ds["k"][l]), -jnp.inf)
            m = jnp.maximum(sb.max(-1), sc.max(-1))
            pb, pc = jnp.exp(sb - m[..., None]), jnp.exp(sc - m[..., None])
            o = (pv(pb, vb) + pv(pc, ds["v"][l])) / (pb.sum(-1) + pc.sum(-1))[..., None]
            return (
                jnp.transpose(o, (2, 0, 1, 3))
                .astype(jnp.bfloat16)
                .reshape(B, d.qpr * d.head_dim)
            )
        m = sb.max(-1)
        pb = jnp.exp(sb - m[..., None])
        den = pb.sum(-1)
        acc = pv(pb, vb)

        def chunk(i, carry):
            m, den, acc = carry
            start = i * C
            ok = (start + jnp.arange(C)) < pos
            kc = jax.lax.dynamic_slice_in_dim(ds["k"][l], start, C, 0)
            vc = jax.lax.dynamic_slice_in_dim(ds["v"][l], start, C, 0)
            sc = jnp.where(ok, scores(kc), -jnp.inf)
            if compat:  # earlier rounds' anchor rows, attended as extra context
                am = jax.lax.dynamic_slice_in_dim(ds["am"], start, C, 0) & ok
                ka = jax.lax.dynamic_slice_in_dim(ds["ak"][l], start, C, 0)
                va = jax.lax.dynamic_slice_in_dim(ds["av"][l], start, C, 0)
                sc = jnp.concatenate([sc, jnp.where(am, scores(ka), -jnp.inf)], -1)
                vc = jnp.concatenate([vc, va], 0)
            mn = jnp.maximum(m, sc.max(-1))
            a = jnp.exp(m - mn)
            pc = jnp.exp(sc - mn[..., None])
            return mn, den * a + pc.sum(-1), acc * a[..., None] + pv(pc, vc)

        n = (pos + C - 1) // C
        m, den, acc = jax.lax.fori_loop(0, n, chunk, (m, den, acc))
        o = acc / den[..., None]  # [kvl, G, B, hd]
        return (
            jnp.transpose(o, (2, 0, 1, 3))
            .astype(jnp.bfloat16)
            .reshape(B, d.qpr * d.head_dim)
        )

    def block_hidden(self, ds, anchor, pos, embed=None, return_anchor_kv=False):
        """Final-normed draft hidden [ndraft, H] of the block [anchor, mask x (ndraft - 1)] from pos."""
        d, w = self.d, ds["w"]
        table = w["embed"] if embed is None else embed
        B = self.ndraft
        ids = jnp.concatenate(
            [
                jnp.reshape(anchor, (1,)).astype(jnp.int32),
                jnp.full((B - 1,), d.mask_token, jnp.int32),
            ]
        )
        bpos = pos + jnp.arange(B, dtype=jnp.int32)
        x = self._embed(table, ids)  # [B, H] bf16
        resid = x
        anchor_kv = []
        for l in range(d.layers):
            hn = _rms(resid, w["ln1"][l], d.eps)
            qkv = _mm(hn, w["wqkv"][l])
            qs, kvs = d.qpr * d.head_dim, d.kvpr * d.head_dim
            q = qkv[:, :qs].reshape(B, d.qpr, d.head_dim)
            k = qkv[:, qs : qs + kvs].reshape(B, d.kvpr, d.head_dim)
            v = qkv[:, qs + kvs :].reshape(B, d.kvpr, d.head_dim)
            q = self._rope(_rms(q, w["q_norm"][l], d.eps), bpos, ds["rope"])
            k = self._rope(_rms(k, w["k_norm"][l], d.eps), bpos, ds["rope"])
            anchor_kv.append((k[0], v[0]))
            o = self._attend(ds, l, q, k, v, pos)
            ao = _dot("bk,kn->bn", o, w["wo"][l])
            ao = _allreduce_bf16(ao, d.world)
            resid = (resid.astype(jnp.float32) + ao.astype(jnp.float32)).astype(
                jnp.bfloat16
            )
            hn = _rms(resid, w["ln2"][l], d.eps)
            gu = _mm(hn, w["wgu"][l]).astype(jnp.float32)
            gt, up = gu[:, : gu.shape[1] // 2], gu[:, gu.shape[1] // 2 :]
            act = (jax.nn.silu(gt) * up).astype(jnp.bfloat16)
            mo = _dot("bk,kn->bn", act, w["wd"][l])
            mo = _allreduce_bf16(mo, d.world)
            resid = (resid.astype(jnp.float32) + mo.astype(jnp.float32)).astype(
                jnp.bfloat16
            )
        h = _rms(resid, w["norm"], d.eps)
        return (h, anchor_kv) if return_anchor_kv else h

    def propose(self, ds, anchor, pos, embed=None, lm_head=None):
        ds, anchor, pos, embed, lm_head = jax.lax.optimization_barrier(
            (ds, anchor, pos, embed, lm_head)
        )
        return jax.lax.optimization_barrier(
            self._propose(ds, anchor, pos, embed, lm_head)[0]
        )

    def propose_state(self, ds, anchor, pos, embed=None, lm_head=None):
        """propose + (sglang_compat state) record this round's anchor-row K/V at ``pos``.

        Reproduces the sglang TPU DSpark path, whose next rounds attend to the draft block's anchor-row K/V
        in addition to the committed context K/V of the same position (observed in reference dumps).
        """
        drafts, anchor_kv = self._propose(ds, anchor, pos, embed, lm_head)
        if "ak" not in ds:
            return drafts, ds
        ak = jnp.stack([k for k, _ in anchor_kv])  # [L, kvpr, hd]
        av = jnp.stack([v for _, v in anchor_kv])
        return drafts, dict(
            ds,
            ak=ds["ak"].at[:, pos].set(ak),
            av=ds["av"].at[:, pos].set(av),
            am=ds["am"].at[pos].set(True),
        )

    def _propose(self, ds, anchor, pos, embed, lm_head):
        d, w = self.d, ds["w"]
        rank = jax.lax.axis_index(AXIS) % d.world
        h, anchor_kv = self.block_hidden(ds, anchor, pos, embed, return_anchor_kv=True)
        if (
            "lm_head_t" in w
        ):  # own [H, V/world] copy: ~25 us faster than [V/world, H] rows at TP8
            vs = w["lm_head_t"].shape[1]
            base = _dot("bh,hv->bv", h, w["lm_head_t"]).astype(jnp.bfloat16)
        else:
            assert d.world == jax.lax.psum(1, AXIS), "draft groups need attach_lm_head"
            vs = lm_head.shape[0]
            base = _dot("bh,vh->bv", h, lm_head).astype(jnp.bfloat16)
        if MARKOV_KERNEL and jax.default_backend() != "cpu":
            from sglang_tpu.k3_megakernel.server.draft_kernels import markov_chain

            drafts = markov_chain(
                base, w["m_w2t"], w["m_w1"], anchor, group=d.world, ndraft=self.ndraft
            )
            return drafts, anchor_kv
        prev = jnp.reshape(anchor, ()).astype(jnp.int32)
        out = []
        for j in range(self.ndraft):
            e = jax.lax.dynamic_index_in_dim(w["m_w1"], prev, 0, keepdims=False)  # [R]
            bias = _dot("vr,r->v", w["m_w2"], e).astype(jnp.bfloat16)
            z = (
                (base[j].astype(jnp.float32) + bias.astype(jnp.float32))
                .astype(jnp.bfloat16)
                .astype(jnp.float32)
            )
            li = jnp.argmax(z).astype(jnp.int32)
            packed = jnp.stack([z[li], (li + rank * vs).astype(jnp.float32)])
            every = _gather_pair(packed, d.world)  # [world, 2]
            prev = every[jnp.argmax(every[:, 0]), 1].astype(jnp.int32)
            out.append(prev)
        return jnp.stack(out), anchor_kv
