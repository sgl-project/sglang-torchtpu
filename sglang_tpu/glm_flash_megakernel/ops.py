# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash decode megakernel in serving (experimental, opt-in).

``SGLANG_TPU_GLM_FLASH_MEGAKERNEL=1`` runs single-request target
steps through one Pallas call per device instead of the decoder: the MTP verify
(all draft tokens in one call) with speculative decoding, the plain decode step
without it. Prefill, batches of more than one request and the draft model keep
the native path.

Prototype scope:
- The kernel streams its own packed weights (``..._PACKED``, written by
  ``python -m sglang_tpu.glm_flash_megakernel.pack``), loaded per rank right
  after the model, so the KV pool is sized after them. Serving's weights stay resident beside them.
- At a request's first step its caches are gathered from serving's pools (fp8
  MLA latent x k_scale, the raw indexer key | gate, the committed KDA state) into
  the kernel's contiguous per-request layout; later steps keep them in the
  kernel. Serving's pools are not updated during these steps, so the radix cache
  must be off.
- With MTP the KDA states hold one checkpoint per draft token, like serving's
  mamba pool: a verify resumes from the checkpoint of the last token the
  previous one accepted, found on the device from the two steps' positions.
  Rejected tokens' KV rows are overwritten by the next verify.
- The kernel computes the logits and the draft's hidden (final norm of the mean
  residual stream) itself; the logits are all-gathered over TP.
"""

from __future__ import annotations

import json
import logging
import os

import numpy as np
import torch

from sglang_tpu.envs import envs as tpu_envs

logger = logging.getLogger(__name__)

_AXIS = "tp"


def enabled() -> bool:
    return tpu_envs.SGLANG_TPU_GLM_FLASH_MEGAKERNEL.get()


def _torch_from_npy(a: np.ndarray, dtype_name: str, device, chunk_bytes=1 << 30):
    """A memory-mapped .npy to a device tensor, a slab of rows at a time so the
    host never holds a whole (multi-GB) array."""
    tdtype, view = {
        "bfloat16": (torch.bfloat16, np.uint16),
        "float8_e4m3fn": (torch.float8_e4m3fn, np.uint8),
        "float32": (torch.float32, np.float32),
    }[dtype_name]
    raw = a.view(view)
    out = torch.empty(raw.shape, dtype=tdtype, device=device)
    rows = max(1, chunk_bytes // max(1, raw[:1].nbytes))
    for i in range(0, raw.shape[0], rows):
        part = torch.from_numpy(np.ascontiguousarray(raw[i : i + rows]))
        out[i : i + rows].copy_(part.view(tdtype))
    return out


def _manifest(packed_dir: str) -> dict:
    """The pack manifest, with the per-channel experts (which serving's kernel
    config reads) merged in."""
    manifest = json.load(open(os.path.join(packed_dir, "manifest.json")))
    for r in range(manifest["tp"]):
        manifest["ranks"][str(r)].update(json.load(open(
            os.path.join(packed_dir, f"manifest_ch.rank{r}.json"))))
    return manifest


def _sources() -> dict:
    """Kernel weight name -> packed array name (the runner's ``src``)."""
    from sglang_tpu.glm_flash_megakernel.kernel import runner as R
    from sglang_tpu.glm_flash_megakernel.weights import CH_REPLACES

    return {n: CH_REPLACES.get(n, n) for n in R.WEIGHT_ORDER if n != "kda_scale"}


def load_packed_weights(packed_dir: str, device) -> dict:
    """This rank's packed weight shards as device tensors, by packed name.

    Called before serving loads its own weights: the largest array is one
    ~24 GiB buffer, which an HBM already fragmented by serving cannot place."""
    from sglang.srt.runtime_context import get_parallel
    from sglang_tpu.glm_flash_megakernel.weights import SHARD_AXIS, _load_npy

    manifest = _manifest(packed_dir)
    rank = get_parallel().tp_rank
    weights = {}
    for name in _sources().values():
        e = manifest["ranks"][str(rank if SHARD_AXIS[name] is not None else 0)][name]
        weights[name] = _torch_from_npy(
            _load_npy(os.path.join(packed_dir, e["file"]), e["dtype"]), e["dtype"], device)
    nbytes = sum(t.numel() * t.element_size() for t in weights.values())
    logger.info("GLM-Flash megakernel: %.1f GiB of packed weights on rank %d",
                nbytes / 2**30, rank)
    return weights


class GlmFlashMegakernel:
    def __init__(self, model, model_path: str, packed_dir: str, max_len: int,
                 weights: dict, n_query: int = 1):
        import jax
        from jax.sharding import NamedSharding
        from jax.sharding import PartitionSpec as P
        from torch_tpu._internal.pallas import get_pallas_mesh
        from torch_tpu._internal.pallas import pallas as pallas_impl

        from sglang_tpu.glm_flash_megakernel.config import Glm53FlashSpec
        from sglang_tpu.glm_flash_megakernel.kernel import decode as D
        from sglang_tpu.glm_flash_megakernel.kernel import runner as R
        from sglang_tpu.glm_flash_megakernel.layout import PackedLayout
        from sglang_tpu.glm_flash_megakernel.weights import SHARD_AXIS, PackedWeights
        from sglang_tpu.pallas_export import tpu_info_for_cpu_export
        from sglang_tpu.pallas_jax_op import _CpuExportJaxCallable

        spec = Glm53FlashSpec.from_checkpoint(model_path)
        manifest = _manifest(packed_dir)
        tp = manifest["tp"]
        device = next(iter(weights.values())).device
        self.spec, self.tp, self.max_len, self.Q = spec, tp, max_len, n_query
        self.vocab = spec.vocab_size

        # Shapes only: the runner reads ranks and the slab index, not the arrays.
        global_arrays = {}
        for name in weights:
            ax = SHARD_AXIS[name]
            e = manifest["ranks"]["0"][name]
            shape = list(e["shape"])
            if ax is not None:
                shape[ax] *= tp
            global_arrays[name] = jax.ShapeDtypeStruct(tuple(shape), jax.numpy.dtype(e["dtype"]))
        small_map = {k: (v[0], v[1], tuple(v[2])) for k, v in manifest["small_map"].items()}
        small_f32_map = {
            int(k): {kk: (vv[0], vv[1]) for kk, vv in v.items()}
            for k, v in manifest["small_f32_map"].items()
        }
        meta = PackedWeights(global_arrays, small_map, small_f32_map,
                             PackedLayout(spec, tp), None, manifest)
        mesh = get_pallas_mesh((_AXIS,), (tp,))
        # The kernel reads embedding rows from serving's 2-D table (8-row tiles)
        # instead of a re-tiled [V, 1, D] copy.
        self.cfg = D.KernelConfig(max_len=max_len, n_query=n_query, embed_2d=True)
        with tpu_info_for_cpu_export():
            run = R.make_decode(spec, meta, mesh, self.cfg)
        self.run = run

        def export(name, fn, in_specs, out_specs, donate=()):
            out_sh = tuple(NamedSharding(mesh, s) for s in out_specs)
            jit_fn = jax.jit(fn, donate_argnums=donate, keep_unused=True,
                             out_shardings=out_sh)
            key = pallas_impl._get_kernel_invocation_key(name, [], {"donate": donate})
            call = _CpuExportJaxCallable(
                name=name, jit_fn=jit_fn, trace_key=key, mesh=mesh,
                input_partition_specs=tuple(in_specs), static_argnums=(),
                donate_argnums=donate)
            call.sglang_output_shardings = out_sh
            return call

        # The derived weights (K-major packing, fp8 KDA scales, channels-last
        # conv), once.
        derived = R.DERIVED
        prep = export("jax_glm_flash_mk_prep", lambda *a: run.prep(*a),
                      run.prep_in_specs, [run.wspec[n] for n in derived])
        out = prep(weights["kda_qkv"], weights["kda_o"], weights["indexer_wq_b"],
                   weights["lm_head"], weights["kda_conv"])
        ready = {run.src.get(n, n): w for n, w in weights.items()}
        ready.update(dict(zip(derived, out)))
        self.weights = [ready[n if n in derived else run.src[n]] for n in R.WEIGHT_ORDER]
        del weights, ready, out
        # A real 2-D tensor, not a view of the packed [V, 1, D] one: a view is
        # materialised again on every call.
        ie = R.WEIGHT_ORDER.index("embed_tokens")
        self.weights[ie] = self.weights[ie].reshape(self.weights[ie].shape[0], -1).clone()

        # The pooled indexer keys, derived from the raw caches at a request's load.
        rep3 = P(None, None, None)
        self.init_ip = export("jax_glm_flash_mk_init_ip", lambda *a: (run.init_ip(*a),),
                              [rep3, rep3, run.wspec["smalls"]], [rep3])
        self.smalls = self.weights[R.WEIGHT_ORDER.index("smalls")]

        # The request's caches, in the kernel's layout.
        n_dsa, n_kda = len(spec.dsa_layers) + 1, len(spec.kda_layers)
        bf16, f32 = torch.bfloat16, torch.float32
        ck = (n_query,) if n_query > 1 else ()
        self.dsa_caches = [
            torch.zeros((n_dsa, max_len, spec.kv_lora_rank), dtype=bf16, device=device),
            torch.zeros((n_dsa, max_len, spec.index_head_dim), dtype=bf16, device=device),
            torch.zeros((n_dsa, max_len, spec.index_head_dim), dtype=bf16, device=device),
            torch.zeros((n_dsa, max_len // spec.index_kpool, spec.index_head_dim),
                        dtype=f32, device=device),
        ]
        self.kda_caches = [
            torch.zeros((n_kda, *ck, spec.linear_num_heads // tp, spec.linear_head_dim,
                         spec.linear_head_dim), dtype=f32, device=device),
            torch.zeros((n_kda, *ck, spec.linear_conv_kernel_dim - 1,
                         3 * spec.kda_qkv_dim // tp), dtype=f32, device=device),
        ]
        assert list(run.dsa_names) == ["kv", "idx_key", "idx_gate", "idx_pool"], run.dsa_names
        assert list(run.kda_names) == ["recur", "conv"], run.kda_names
        self.init_streams = torch.zeros((n_query * spec.hc_mult, spec.hidden_size),
                                        dtype=bf16, device=device)
        self.k_scales = []
        for L in spec.dsa_layers:
            attn = getattr(model.model.layers[L].self_attn, "attn_mqa", None)
            ks = getattr(attn, "k_scale_float", None)
            self.k_scales.append(1.0 if ks is None else float(ks))

        nw, nd = len(R.WEIGHT_ORDER), len(run.dsa_names)

        Q = n_query

        # The per-step bookkeeping (the checkpoint offset) happens inside this launch: as separate eager ops each would be its
        # own program launch, and the decode loop is host-bound once the kernel is
        # fast.  The checkpoint of the last token the previous verify accepted is
        # pos - prev - 1 (that verify ended at prev + accepted - 1 = pos - 1).
        def step(tok, positions, prev, init, *rest):
            ws = dict(zip(R.WEIGHT_ORDER, rest[:nw]))
            dc = dict(zip(run.dsa_names, rest[nw : nw + nd]))
            kc = dict(zip(run.kda_names, rest[nw + nd :]))
            pos = positions.reshape(-1)[:1].astype(jax.numpy.int32)
            pos_in = jax.numpy.concatenate([pos, pos - prev - 1]) if Q > 1 else pos
            logits, _, hidden, out = run.shard(
                tok.reshape(Q).astype(jax.numpy.int32), pos_in, init, ws, dc, kc)
            return (logits, hidden, pos) + tuple(out[n] for n in run.step_caches)

        in_specs = ([P(), P(), P(), P(None, None)] + [run.wspec[n] for n in R.WEIGHT_ORDER]
                    + [run.dspec[n] for n in run.dsa_names]
                    + [run.kspec[n] for n in run.kda_names])
        cspec = {**run.dspec, **run.kspec}
        self.call = export(
            f"jax_glm_flash_mk_L{max_len}_q{n_query}", step, in_specs,
            # Logits stay vocab-sharded and are gathered by serving's collective:
            # replicated in this launch, tensors derived from them carry mesh
            # shardings into serving's own compiled ops, which then fail to compile.
            [P(None, _AXIS), P(None, None), P()] + [cspec[n] for n in run.step_caches],
            donate=tuple(range(4 + nw, 4 + nw + nd)))
        self.req, self.prev_pos = None, None

    # ------------------------------------------------------------------ serving
    def handles(self, forward_batch, input_ids) -> bool:
        if forward_batch.batch_size != 1 or input_ids.shape[0] != self.Q:
            return False
        mode = forward_batch.forward_mode
        if self.Q > 1:
            return mode.is_target_verify()
        return mode.is_decode() and forward_batch.spec_info is None

    @property
    def active(self) -> bool:
        """A request's caches live in the kernel (serving's pools lag behind)."""
        return self.req is not None

    def invalidate(self):
        """A target step the kernel did not run (a prefill): reload next time."""
        self.req = None

    def _load_request(self, forward_batch, req: int, pos: int):
        """Gather the request's caches from serving's pools (once per request)."""
        from sglang.srt.model_executor.forward_context import get_attn_backend

        spec = self.spec
        if pos + self.Q > self.max_len:
            raise ValueError(f"position {pos} exceeds the megakernel's max_len {self.max_len}")
        backend = get_attn_backend()
        full = getattr(backend, "full_attn_backend", backend)
        lin = getattr(backend, "linear_attn_backend", backend)
        pool, req_pool = full.token_to_kv_pool, lin.req_to_token_pool
        # req_to_token is host-resident on TPU.
        device = self.dsa_caches[0].device
        locs = req_pool.req_to_token[req, :pos].to(device=device, dtype=torch.int64)
        kv, ik, ig, _ = self.dsa_caches
        idh, kvl = spec.index_head_dim, spec.kv_lora_rank
        for d, L in enumerate(spec.dsa_layers):
            nope = pool.get_nope_cache(L).view(-1, kvl).index_select(0, locs)
            rows = nope.view(torch.float8_e4m3fn).to(torch.float32) * self.k_scales[d]
            kv[d, :pos] = rows.to(torch.bfloat16)
            raw = pool.get_kpool_raw_cache(L).view(-1, 2 * idh).index_select(0, locs)
            ik[d, :pos] = raw[:, :idh]
            ig[d, :pos] = raw[:, idh:]
        self.dsa_caches[3] = self.init_ip(ik, ig, self.smalls)[0]
        # With MTP the committed state is checkpoint read_offsets[slot] of the slot.
        slot = lin._slot_indices(forward_batch.req_pool_indices)[0][:1].to(device)
        row = slot
        if self.Q > 1:
            row = slot * self.Q + lin._read_offsets.to(device).index_select(0, slot)
        recur, conv = self.kda_caches
        k1 = spec.linear_conv_kernel_dim - 1
        for k, L in enumerate(spec.kda_layers):
            c = req_pool.mamba2_layer_cache(L)
            st = c.temporal.index_select(0, row)[0]
            cv = c.conv[0].index_select(0, row)[0].reshape(k1, -1)   # [K-1, C], as the kernel
            if self.Q > 1:
                # Every checkpoint: the first verify resumes from checkpoint 0.
                recur[k] = st.unsqueeze(0).expand(self.Q, *st.shape)
                conv[k] = cv.unsqueeze(0).expand(self.Q, *cv.shape)
            else:
                recur[k] = st
                conv[k] = cv
        self.req = req

    def forward(self, input_ids, positions, forward_batch):
        from sglang.srt.distributed import tensor_model_parallel_all_gather
        from sglang.srt.layers.logits_processor import LogitsProcessorOutput

        req = int(forward_batch.req_pool_indices_cpu[0])
        if req != self.req:
            pos = int(positions.reshape(-1)[0].item())
            self._load_request(forward_batch, req, pos)
            # The first verify resumes from checkpoint 0: prev = pos - 1.
            self.prev_pos = torch.full((1,), pos - 1, dtype=torch.int32,
                                       device=positions.device)
        if input_ids.dtype != torch.int32:
            input_ids = input_ids.to(torch.int32)
        if positions.dtype != torch.int32:
            positions = positions.to(torch.int32)
        out = self.call(input_ids, positions, self.prev_pos, self.init_streams,
                        *self.weights, *self.dsa_caches, *self.kda_caches)
        logits, hidden, self.prev_pos, nd = out[0], out[1], out[2], len(self.dsa_caches)
        caches = list(out[3:])
        self.dsa_caches, self.kda_caches = caches[:nd], caches[nd:]
        logits = tensor_model_parallel_all_gather(logits, dim=-1)[:, : self.vocab]
        return LogitsProcessorOutput(next_token_logits=logits, hidden_states=hidden)
