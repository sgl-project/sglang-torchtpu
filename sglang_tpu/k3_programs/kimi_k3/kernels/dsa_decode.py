# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3.1's DSA indexer inside the decode megakernel's MLA layers.

Per MLA layer, before the DCP attention:

1. Projections. Rank ``r`` computes index head ``r`` of every row's query
   (``d_wqb`` is head-sharded; ranks load their weights by mesh position, so
   DCP member ``d`` holds head ``d``) from the normalized q-latent, and sends
   it into slot ``r`` of every peer (remote copies, as the DCP query gather).
   Every rank computes the rows' head gates (``weights_proj``) and new index
   keys (``wk``, LayerNorm) from the full hidden state while the gather is in
   flight. Rounding follows sglang's native indexer: BF16 linear outputs,
   ``bf16(LN(bf16 key))``, gate ``bf16(bf16(g) * bf16(32^-0.5))``, query
   heads FP8 at their UE8M0 scales, ``w = gate * 2^e_q * 128^-0.5``
   (``sglang_tpu/k3_megakernel/kernel/stages/dsa.py``).
2. Selection (``dsa_dcp.IndexSelect``): each rank scores its own index pages
   and the ranks agree on every row's top-k threshold.
3. The rows' new key records go out with the layer's latents (``records``,
   committed after the step by ``dsa_dcp.commit_index_rows``), and the DCP
   attention takes the selection (``mla_dcp.DcpSelection``).
"""

from __future__ import annotations

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sglang_tpu.k3_programs.kimi_k3.kernels import dsa_dcp, mla_dcp

HEADS = 32
DIM = 128
Q_LORA = 1536
FAMILIES = ("d_wqb", "d_wk", "d_wproj", "d_knorm")
WEIGHT_SCALE = DIM**-0.5  # the indexer's softmax scale on the head weights
# torch multiplies a BF16 tensor by a Python float in BF16: gate * bf16(32^-0.5).
HEAD_GATE_SCALE = float(np.float32(ml_dtypes.bfloat16(HEADS**-0.5)))
FP8_MAX = 448.0
CHUNK = 512  # contraction rows of a streamed weight chunk (wq_b, wk, weights_proj)
RING = 3  # chunk slots of the weight stream
LANES = dsa_dcp.LANES
RECORD_SHAPE = (2, LANES)  # key bytes | UE8M0 byte in lane 0 (int32 lanes)


def _bf16(x):
    return x.astype(jnp.bfloat16).astype(jnp.float32)


def _halving_sum(v):
    """Sum over the last axis by halving (XLA's lane-reduction order)."""
    n = v.shape[-1]
    while n > 1:
        n //= 2
        v = v[:, :n] + v[:, n:2 * n]
    return v


def layer_norm(x, weight, bias, eps):
    """torch_tpu's F.layer_norm in f32 (``var = E[x^2] - mean^2``, halving sums)."""
    inv_n = 1.0 / x.shape[-1]
    mean = _halving_sum(x) * inv_n
    var = _halving_sum(x * x) * inv_n - mean * mean
    return (x - mean) * jax.lax.rsqrt(var + eps) * weight + bias


def ue8m0_exponent(abs_max):
    """The UE8M0 scale exponent of an f32 abs max (``quantize_fp8_ue8m0``)."""
    bits = jax.lax.bitcast_convert_type(abs_max, jnp.int32)
    biased = bits >> 23
    exp = biased - (127 + 8) + ((bits & 0x7FFFFF) > 0x600000).astype(jnp.int32)
    exp = jnp.where(biased == 0, jnp.where(bits == 0, 0, -126), exp)
    return jnp.clip(exp, -126, 127)


def _pow2(exp):
    return jax.lax.bitcast_convert_type((exp + 127) << 23, jnp.float32)


def quantize(x):
    """f32 ``[R, 128]`` -> (fp8 values as f32, exponent i32 ``[R, 1]``)."""
    exp = ue8m0_exponent(jnp.max(jnp.abs(x), axis=1, keepdims=True))
    values = jnp.clip(x * _pow2(-exp), -FP8_MAX, FP8_MAX).astype(jnp.float8_e4m3fn)
    return values, exp


# ------------------------------------------------------------------ scratch
PERSISTENT_NAMES = ("dsa query gather", "dsa query gather semaphores", "dsa count exchange",
                    "dsa count exchange semaphores")


def persistent_scratch(batch, group=32):
    """Buffers peers' remote copies land in (live for the whole kernel). One
    query-gather buffer suffices: a rank reads its gathered queries before
    its first count exchange, and no peer sends the next layer's queries
    before that exchange completes."""
    return (
        pltpu.VMEM((group, batch, DIM), jnp.bfloat16),
        pltpu.SemaphoreType.DMA((2,)),
        pltpu.VMEM((2, group, dsa_dcp.COUNT_ROWS, LANES), jnp.int32),
        pltpu.SemaphoreType.DMA((2, 2)),
    )


def scoped_scratch(batch, *, hidden, index_block_shape, key_rows, page_tokens, sequence_rows=1, page_plan=None):
    """The MLA region's indexer scratch: ``{name: spec}`` (``bind`` order).
    ``page_plan``: the DCP attention's ``mla_dcp.page_plan`` (span, buffers)."""
    select = dsa_dcp.select_scratch(batch, index_block_shape, key_rows)
    for name in ("exchange", "exchange_sems"):
        select.pop(name)
    key_buf, key_sems = mla_dcp.selection_scratch(
        batch, page_tokens=page_tokens, sequence_rows=sequence_rows,
        cache_dtype=jnp.float8_e4m3fn, **(page_plan or {}))
    if hidden % CHUNK:
        raise ValueError(f"hidden size {hidden} is not a multiple of {CHUNK}")
    return {
        "weight_ring": pltpu.VMEM((RING, CHUNK, DIM), jnp.bfloat16),  # wq_b, then wk / weights_proj chunks
        "ring_sems": pltpu.SemaphoreType.DMA((RING,)),
        "knorm": pltpu.VMEM((2, DIM), jnp.bfloat16),
        "knorm_sem": pltpu.SemaphoreType.DMA((1,)),
        "q8": pltpu.VMEM((batch, HEADS, DIM), jnp.float8_e4m3fn),
        "w_t": pltpu.VMEM((HEADS, LANES), jnp.float32),
        "new_k8": pltpu.VMEM((batch, DIM), jnp.float32),
        "new_exp": pltpu.SMEM((batch,), jnp.int32),
        "records": pltpu.VMEM((batch, *RECORD_SHAPE), jnp.int32),
        "record_sem": pltpu.SemaphoreType.DMA((1,)),
        "tau": pltpu.SMEM((batch,), jnp.int32),
        "p_tie": pltpu.SMEM((batch,), jnp.int32),
        "own": pltpu.SMEM((batch,), jnp.int32),
        "attend_keys": key_buf,
        "attend_key_sems": key_sems,
        **select,
    }


@dataclasses.dataclass
class DecodeIndexer:
    """One MLA layer's indexer (see the module docstring).

    ``weights`` maps each of ``FAMILIES`` to this layer's HBM weight ref;
    ``pools`` are the 24 index-key pools (picked by the traced ``layer``);
    ``keys`` the HBM keys scratch ``[B, key_rows, 128]`` int32 and
    ``records_out`` the HBM record handoff ``[B, 24, 2, 128]`` int32 (kernel
    outputs); ``persistent`` / ``scoped`` the scratch above."""

    weights: dict
    pools: tuple
    layer: Any
    keys: Any
    records_out: Any
    persistent: tuple
    scoped: dict
    topk: int = 2048
    eps: float = 1e-5
    sequence_rows: int = 1

    def _plan(self):
        """The weight stream: ``(family, chunk)`` in consumption order (wq_b's
        chunks, then wk and weights_proj chunk by chunk)."""
        hidden = self.weights["d_wk"].shape[0]
        return ([("d_wqb", c) for c in range(Q_LORA // CHUNK)]
                + [(name, c) for c in range(hidden // CHUNK) for name in ("d_wk", "d_wproj")])

    def _stream(self, index):
        name, chunk = self._plan()[index]
        slot = index % RING
        return pltpu.make_async_copy(self.weights[name].at[pl.ds(chunk * CHUNK, CHUNK)],
                                     self.scoped["weight_ring"].at[slot], self.scoped["ring_sems"].at[slot])

    def _knorm(self):
        return pltpu.make_async_copy(self.weights["d_knorm"], self.scoped["knorm"], self.scoped["knorm_sem"].at[0])

    def start(self):
        """Start the layer's indexer weight copies (as early as possible):
        the key norm and the stream's first ``RING - 1`` chunks."""
        self._knorm().start()
        for index in range(RING - 1):
            self._stream(index).start()

    def _take(self, index):
        """Chunk ``index`` of the stream (its slot's value), the next refill started."""
        if index + RING - 1 < len(self._plan()):
            self._stream(index + RING - 1).start()
        self._stream(index).wait()
        return self.scoped["weight_ring"][index % RING]

    def _query(self, query_latent):
        """``query_latent @ wq_b`` (this rank's index head, f32)."""
        q = None
        for c in range(Q_LORA // CHUNK):
            part = jnp.dot(query_latent[:, c * CHUNK:(c + 1) * CHUNK], self._take(c),
                           preferred_element_type=jnp.float32)
            q = part if q is None else q + part
        return q

    def _key_and_gate(self, x):
        """``x @ wk`` and ``x @ weights_proj`` (f32), streamed after wq_b."""
        first = Q_LORA // CHUNK
        key = gate = None
        for c in range(x.shape[1] // CHUNK):
            part = x[:, c * CHUNK:(c + 1) * CHUNK]
            k = jnp.dot(part, self._take(first + 2 * c), preferred_element_type=jnp.float32)
            g = jnp.dot(part, self._take(first + 2 * c + 1), preferred_element_type=jnp.float32)
            key, gate = (k, g) if key is None else (key + k, gate + g)
        return key, gate

    def selection(self, cache, position_ref, hidden, query_latent, parity) -> mla_dcp.DcpSelection:
        """``hidden`` [B, hidden] and ``query_latent`` [B, 1536] (normalized,
        BF16) values; ``cache`` the layer's ``mla_dcp.DcpCache``."""
        s, batch = self.scoped, hidden.shape[0]
        group = cache.group
        if group != HEADS:
            raise ValueError("the indexer gathers its 32 query heads over a DCP32 group")
        rank = jax.lax.axis_index("tp")
        member = rank % group
        base = rank - member
        owner = member if cache.owner is None else cache.owner
        gather, gather_sems, exchange, exchange_sems = self.persistent

        # 1. This rank's index head of every row, out to every peer's slot.
        q_own = self._query(query_latent.astype(jnp.bfloat16)).astype(jnp.bfloat16)  # [B, 128]
        gather[member] = q_own
        for peer in range(group):
            @pl.when(peer != member)
            def send(peer=peer):
                pltpu.make_async_remote_copy(
                    gather.at[member], gather.at[member], gather_sems.at[0], gather_sems.at[1],
                    device_id=(base + peer,), device_id_type=pl.DeviceIdType.MESH,
                ).start()

        # 2. The rows' new keys (records) and head gates, while it is in flight.
        self._knorm().wait()
        x = hidden.astype(jnp.bfloat16)
        key_raw, gate_raw = self._key_and_gate(x)
        key_raw = _bf16(key_raw)
        norm = s["knorm"][...].astype(jnp.float32)
        key = _bf16(layer_norm(key_raw, norm[0:1], norm[1:2], self.eps))
        values, exp = quantize(key)
        s["new_k8"][...] = values.astype(jnp.float32)
        lane = jax.lax.broadcasted_iota(jnp.int32, (batch, LANES), 1)
        key_bytes = jax.lax.bitcast_convert_type(values, jnp.uint8).astype(jnp.int32)
        s["records"][:, 0, :] = key_bytes
        s["records"][:, 1, :] = jnp.where(lane == 0, exp + 127, 0)
        row = jax.lax.broadcasted_iota(jnp.int32, exp.shape, 0)
        for b in range(batch):
            s["new_exp"][b] = jnp.sum(jnp.where(row == b, exp + 127, 0))
        gate = _bf16(_bf16(gate_raw) * HEAD_GATE_SCALE)  # [B, 128]: heads 0..31
        if batch < LANES:
            gate = jnp.concatenate([gate, jnp.zeros((LANES - batch, DIM), jnp.float32)], axis=0)
        gate_t = gate.T[:HEADS]  # [32 heads, 128 rows]
        record_copy = pltpu.make_async_copy(s["records"], self.records_out.at[:, self.layer],
                                            s["record_sem"].at[0])
        record_copy.start()

        # 3. Every row's 32 query heads: FP8 at their scales; the head weights.
        others = gather.at[pl.ds(0, group - 1)]
        pltpu.make_async_copy(others, others, gather_sems.at[0]).wait()
        pltpu.make_async_copy(others, others, gather_sems.at[1]).wait()
        head_lane = jax.lax.broadcasted_iota(jnp.int32, (HEADS, LANES), 1)
        w_t = jnp.zeros((HEADS, LANES), jnp.float32)
        for b in range(batch):
            heads = gather[:, pl.ds(b, 1), :].reshape(HEADS, DIM).astype(jnp.float32)
            q8, q_exp = quantize(heads)
            s["q8"][b] = q8
            column = jnp.sum(jnp.where(head_lane == b, gate_t, 0.0), axis=1, keepdims=True)  # [32, 1]
            w_t = jnp.where(head_lane == b, column * _pow2(q_exp) * WEIGHT_SCALE, w_t)
        s["w_t"][...] = w_t

        # 4. The selection.
        refs = {name: s[name] for name in dsa_dcp.select_scratch(1, (33, 4, 128), 8) if name in s}
        refs.update(exchange=exchange, exchange_sems=exchange_sems, keys=self.keys,
                    tau=s["tau"], p_tie=s["p_tie"], own=s["own"])
        select = dsa_dcp.IndexSelect(
            pools=self.pools, layer=self.layer, table_ref=cache.table_ref, pages=cache.pages,
            page_tokens=cache.page_tokens, group=group, owner=owner, member=member, base=base,
            position_ref=position_ref, new_exp_ref=s["new_exp"], batch=batch, topk=self.topk,
            refs=refs, sequence_rows=self.sequence_rows, global_pages=cache.global_pages,
            table_row=cache.table_row)
        def query(row):  # [32, 128] FP8 query heads of a row (traced or not)
            return s["q8"][row]

        select.plan()
        select.in_flight(query, s["w_t"], s["new_k8"])
        select.score(query, s["w_t"])
        select.select()
        record_copy.wait()
        return mla_dcp.DcpSelection(self.keys, s["tau"], s["p_tie"], s["own"],
                                    s["attend_keys"], s["attend_key_sems"])
