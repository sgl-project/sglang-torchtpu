# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash decode megakernel in serving (experimental, opt-in).

Runs single-request target steps (the MTP verify, or plain decode without MTP) as
one Pallas call per device; every other step runs the model as usual. A request's
caches are copied from serving's pools at its first kernel step and then live in the
kernel. Before anything else reads serving's pools for that request, `flush` writes
the kernel's rows and states back in serving's layout. With MTP the KDA states hold
one checkpoint per draft token, and a verify resumes from the last accepted one.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

from sglang_tpu.envs import envs as tpu_envs

logger = logging.getLogger(__name__)

_AXIS = "tp"
# Rows per copy between the kernel's caches and serving's pools: one compiled shape.
_CHUNK = 4096


def _owned(t: torch.Tensor) -> torch.Tensor:
    # A device-side clone or copy keeps the view; a host round trip uploads a plain buffer.
    return t.cpu().to(t.device)


def enabled() -> bool:
    return tpu_envs.SGLANG_TPU_GLM_FLASH_MEGAKERNEL.get()


class GlmFlashMegakernel:
    def __init__(self, model, model_path: str, max_len: int, n_query: int = 1):
        import jax
        from jax.sharding import NamedSharding
        from jax.sharding import PartitionSpec as P
        from torch_tpu._internal.pallas import get_pallas_mesh
        from torch_tpu._internal.pallas import pallas as pallas_impl

        from sglang.srt.runtime_context import get_parallel
        from sglang_tpu.glm_flash_megakernel.build import build_weights, slab_maps
        from sglang_tpu.glm_flash_megakernel.config import Glm53FlashSpec
        from sglang_tpu.glm_flash_megakernel.kernel import decode as D
        from sglang_tpu.glm_flash_megakernel.kernel import runner as R
        from sglang_tpu.glm_flash_megakernel.layout import KernelLayout
        from sglang_tpu.pallas_export import tpu_info_for_cpu_export
        from sglang_tpu.pallas_jax_op import _CpuExportJaxCallable

        spec = Glm53FlashSpec.from_checkpoint(model_path)
        par = get_parallel()
        tp, rank = par.tp_size, par.tp_rank
        device = model.model.norm.weight.device
        max_len = -(-max_len // _CHUNK) * _CHUNK
        self.spec, self.tp, self.max_len, self.Q = spec, tp, max_len, n_query
        self.vocab = spec.vocab_size
        layout = KernelLayout(spec, tp)
        small_map, small_f32_map = slab_maps(spec)
        mesh = get_pallas_mesh((_AXIS,), (tp,))
        # embed_2d: build_weights holds the 2-D table as a real tensor, since a torch
        # view would be re-materialised per call.
        self.cfg = D.KernelConfig(max_len=max_len, n_query=n_query, embed_2d=True)
        with tpu_info_for_cpu_export():
            run = R.make_decode(spec, layout, small_map, small_f32_map, mesh, self.cfg)
        self.run = run

        weights = build_weights(model, spec, layout, self.cfg, tp, rank)
        # Packing leaves dtype and shape views, which every call would re-materialise.
        self.weights = [_owned(weights[n]) for n in R.PACKED_ORDER]
        self.experts = [w for n in R.EXPERT_LISTS for w in weights[n]]
        nbytes = sum(t.numel() * t.element_size() for t in self.weights)
        logger.info(
            "GLM-Flash megakernel: %.1f GiB of packed weights on rank %d (experts are sglang's)",
            nbytes / 2**30,
            rank,
        )
        del weights

        def export(name, fn, in_specs, out_specs, donate=()):
            out_sh = tuple(NamedSharding(mesh, s) for s in out_specs)
            jit_fn = jax.jit(
                fn, donate_argnums=donate, keep_unused=True, out_shardings=out_sh
            )
            key = pallas_impl._get_kernel_invocation_key(name, [], {"donate": donate})
            call = _CpuExportJaxCallable(
                name=name,
                jit_fn=jit_fn,
                trace_key=key,
                mesh=mesh,
                input_partition_specs=tuple(in_specs),
                static_argnums=(),
                donate_argnums=donate,
            )
            call.sglang_output_shardings = out_sh
            return call

        rep3 = P(None, None, None)
        self.init_ip = export(
            "jax_glm_flash_mk_init_ip",
            lambda *a: (run.init_ip(*a),),
            [rep3, rep3, run.wspec["smalls"]],
            [rep3],
        )
        self.smalls = self.weights[R.PACKED_ORDER.index("smalls")]
        self.pool_records = export(
            "jax_glm_flash_mk_pool_records",
            lambda *a: (run.pool_records(*a),),
            [rep3, rep3, run.wspec["smalls"], P(None)],
            [rep3],
        )

        n_dsa, n_kda = len(spec.dsa_layers) + 1, len(spec.kda_layers)
        bf16, f32 = torch.bfloat16, torch.float32
        ck = (n_query,) if n_query > 1 else ()
        self.dsa_caches = [
            torch.zeros((n_dsa, max_len, spec.kv_lora_rank), dtype=bf16, device=device),
            torch.zeros(
                (n_dsa, max_len, spec.index_head_dim), dtype=bf16, device=device
            ),
            torch.zeros(
                (n_dsa, max_len, spec.index_head_dim), dtype=bf16, device=device
            ),
            torch.zeros(
                (n_dsa, max_len // spec.index_kpool, spec.index_head_dim),
                dtype=f32,
                device=device,
            ),
        ]
        self.kda_caches = [
            torch.zeros(
                (
                    n_kda,
                    *ck,
                    spec.linear_num_heads // tp,
                    spec.linear_head_dim,
                    spec.linear_head_dim,
                ),
                dtype=f32,
                device=device,
            ),
            torch.zeros(
                (
                    n_kda,
                    *ck,
                    spec.linear_conv_kernel_dim - 1,
                    3 * spec.kda_qkv_dim // tp,
                ),
                dtype=f32,
                device=device,
            ),
        ]
        assert list(run.dsa_names) == ["kv", "idx_key", "idx_gate", "idx_pool"], (
            run.dsa_names
        )
        assert list(run.kda_names) == ["recur", "conv"], run.kda_names
        self.init_streams = torch.zeros(
            (n_query * spec.hc_mult, spec.hidden_size), dtype=bf16, device=device
        )
        self.k_scales = []
        for L in spec.dsa_layers:
            attn = getattr(model.model.layers[L].self_attn, "attn_mqa", None)
            ks = getattr(attn, "k_scale_float", None)
            self.k_scales.append(1.0 if ks is None else float(ks))

        nw, nd, nk = len(R.PACKED_ORDER), len(run.dsa_names), len(run.kda_names)
        n_moe = len(self.experts) // len(R.EXPERT_LISTS)

        Q = n_query

        # In-launch bookkeeping: separate eager ops would each be a host-bound launch.
        # The last verify ended at pos - 1, so its last accepted checkpoint is pos - prev - 1.
        def step(tok, positions, prev, init, *rest):
            ws = dict(zip(R.PACKED_ORDER, rest[:nw]))
            dc = dict(zip(run.dsa_names, rest[nw : nw + nd]))
            kc = dict(zip(run.kda_names, rest[nw + nd : nw + nd + nk]))
            experts = rest[nw + nd + nk :]
            for i, n in enumerate(R.EXPERT_LISTS):
                ws[n] = list(experts[i * n_moe : (i + 1) * n_moe])
            pos = positions.reshape(-1)[:1].astype(jax.numpy.int32)
            pos_in = jax.numpy.concatenate([pos, pos - prev - 1]) if Q > 1 else pos
            logits, _, hidden, out = run.shard(
                tok.reshape(Q).astype(jax.numpy.int32), pos_in, init, ws, dc, kc
            )
            return (logits, hidden, pos) + tuple(out[n] for n in run.step_caches)

        in_specs = (
            [P(), P(), P(), P(None, None)]
            + [run.wspec[n] for n in R.PACKED_ORDER]
            + [run.dspec[n] for n in run.dsa_names]
            + [run.kspec[n] for n in run.kda_names]
            + [spec for n in R.EXPERT_LISTS for spec in run.wspec[n]]
        )
        cspec = {**run.dspec, **run.kspec}
        self.call = export(
            f"jax_glm_flash_mk_L{max_len}_q{n_query}",
            step,
            in_specs,
            # Keep logits vocab-sharded: replicated here, their mesh sharding leaks into
            # serving's compiled ops, which then fail to compile.
            [P(None, _AXIS), P(None, None), P()] + [cspec[n] for n in run.step_caches],
            donate=tuple(range(4 + nw, 4 + nw + nd)),
        )
        self.req, self.prev_pos, self.last_steps = None, None, None
        self.slot: Optional[torch.Tensor] = None
        self.req_rows: Optional[torch.Tensor] = None

    def handles(self, forward_batch, input_ids) -> bool:
        if forward_batch.batch_size != 1 or input_ids.shape[0] != self.Q:
            return False
        # A request longer than the kernel's caches continues on the model.
        seq_lens_cpu = forward_batch.seq_lens_cpu
        if seq_lens_cpu is not None and int(seq_lens_cpu[0]) + self.Q > self.max_len:
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
        self.req = None

    def _bind_pools(self):
        from sglang.srt.model_executor.forward_context import get_attn_backend

        backend = get_attn_backend()
        full = getattr(backend, "full_attn_backend", backend)
        lin = getattr(backend, "linear_attn_backend", backend)
        self._pool, self._lin = full.token_to_kv_pool, lin
        self._req_pool = lin.req_to_token_pool

    def _rows(self, a: int, device):
        return a + torch.arange(_CHUNK, dtype=torch.int64, device=device)

    def _locs(self, rows, valid):
        table = self._req_pool.req_to_token[self.req].to(rows.device, torch.int64)
        locs = table.index_select(0, rows.clamp(max=table.shape[0] - 1))
        return torch.where(valid, locs, torch.zeros_like(locs))

    def _load_request(self, forward_batch, req: int, pos: int):
        spec = self.spec
        if pos + self.Q > self.max_len:
            raise ValueError(
                f"position {pos} exceeds the megakernel's max_len {self.max_len}"
            )
        self._bind_pools()
        pool, req_pool, lin = self._pool, self._req_pool, self._lin
        self.req = req
        device = self.dsa_caches[0].device
        kv, ik, ig, _ = self.dsa_caches
        idh, kvl = spec.index_head_dim, spec.kv_lora_rank
        for a in range(0, pos, _CHUNK):
            rows = self._rows(a, device)
            locs = self._locs(rows, rows < pos)
            for d, L in enumerate(spec.dsa_layers):
                nope = pool.get_nope_cache(L).view(-1, kvl).index_select(0, locs)
                vals = (
                    nope.view(torch.float8_e4m3fn).to(torch.float32) * self.k_scales[d]
                )
                kv[d].index_copy_(0, rows, vals.to(torch.bfloat16))
                raw = (
                    pool.get_kpool_raw_cache(L).view(-1, 2 * idh).index_select(0, locs)
                )
                ik[d].index_copy_(0, rows, raw[:, :idh].contiguous())
                ig[d].index_copy_(0, rows, raw[:, idh:].contiguous())
        self.dsa_caches[3] = self.init_ip(ik, ig, self.smalls)[0]
        self.slot = lin._slot_indices(forward_batch.req_pool_indices)[0][:1].to(device)
        self.req_rows = forward_batch.req_pool_indices[:1].to(device)
        row = req_pool.mamba_pool.committed_state_rows(self.slot, self.req_rows)
        recur, conv = self.kda_caches
        k1 = spec.linear_conv_kernel_dim - 1
        for k, L in enumerate(spec.kda_layers):
            c = req_pool.mamba2_layer_cache(L)
            st = c.temporal.index_select(0, row)[0]
            cv = c.conv[0].index_select(0, row)[0].reshape(k1, -1)
            if self.Q > 1:
                recur[k] = st.unsqueeze(0).expand(self.Q, *st.shape)
                conv[k] = cv.unsqueeze(0).expand(self.Q, *cv.shape)
            else:
                recur[k] = st
                conv[k] = cv
        self.start, self.steps, self.last_steps = pos, 0, None

    def record_commit(self, last_steps: torch.Tensor):
        """The verify's last accepted step, in place of serving's checkpoint commit."""
        self.last_steps = last_steps

    def flush(self):
        """Write the request's kernel-side rows and states into serving's pools."""
        if self.req is None:
            return
        spec, Q = self.spec, self.Q
        pool, req_pool = self._pool, self._req_pool
        device = self.dsa_caches[0].device
        last = self.prev_pos.to(torch.int64)
        if Q == 1:
            committed = last + 1
        elif self.last_steps is not None:
            committed = last + self.last_steps[:1].to(device, torch.int64) + 1
        else:
            committed = last
        hi = min(self.start + self.steps * Q, self.max_len)
        kv, ik, ig, _ = self.dsa_caches
        idh, kvl, kp = spec.index_head_dim, spec.kv_lora_rank, spec.index_kpool
        ps = pool.page_size
        for a in range(self.start - self.start % _CHUNK, hi, _CHUNK):
            rows = self._rows(a, device)
            valid = (rows >= self.start) & (rows < committed) & (rows < self.max_len)
            locs = self._locs(rows, valid)
            idx = (locs // ps, locs % ps)
            src = rows.clamp(max=self.max_len - 1)
            for d, L in enumerate(spec.dsa_layers):
                vals = kv[d].index_select(0, src).to(torch.float32) / self.k_scales[d]
                nope = (
                    vals.to(torch.float8_e4m3fn).view(torch.uint8).view(_CHUNK, -1, 128)
                )
                pool.get_nope_cache(L).index_put_(idx, nope)
                raw = torch.cat(
                    [ik[d].index_select(0, src), ig[d].index_select(0, src)], 1
                )
                pool.get_kpool_raw_cache(L).index_put_(idx, raw)
        # Pools closed while the kernel held the request; pool p sits at position p.
        p_lo, p_hi = self.start // kp, -(-hi // kp)
        for a in range(p_lo - p_lo % _CHUNK, p_hi, _CHUNK):
            pools = self._rows(a, device)
            end = pools * kp + kp - 1
            valid = (
                (end >= self.start) & (end < committed) & (pools * kp < self.max_len)
            )
            locs = self._locs(pools, valid)
            idx = (locs // ps, (locs % ps) // 4, locs % 4)
            src = pools.clamp(max=self.max_len // kp - 1).to(torch.int32)
            records = self.pool_records(ik, ig, self.smalls, src)[0]
            for d, L in enumerate(spec.dsa_layers):
                pool.get_index_key_cache(L).index_put_(idx, records[d])
        recur, conv = self.kda_caches
        if Q > 1:
            # The verify checkpoints of the request's window, as serving writes them.
            rows = req_pool.mamba_pool.checkpoint_rows(self.req_rows).reshape(-1)
        else:
            rows = self.slot
        for k, L in enumerate(spec.kda_layers):
            c = req_pool.mamba2_layer_cache(L)
            c.temporal.index_copy_(
                0, rows, recur[k].reshape(rows.shape[0], *c.temporal.shape[1:])
            )
            c.conv[0].index_copy_(
                0, rows, conv[k].reshape(rows.shape[0], *c.conv[0].shape[1:])
            )
        if Q > 1 and self.last_steps is not None:
            req_pool.mamba_pool.select_checkpoints(self.slot, self.last_steps[:1])
        self.req = None

    def forward(self, input_ids, positions, forward_batch):
        from sglang.srt.distributed import tensor_model_parallel_all_gather
        from sglang.srt.layers.logits_processor import LogitsProcessorOutput

        req = int(forward_batch.req_pool_indices_cpu[0])
        if req != self.req:
            self.flush()
            pos = int(positions.reshape(-1)[0].item())
            self._load_request(forward_batch, req, pos)
            # The first verify resumes from checkpoint 0: prev = pos - 1.
            self.prev_pos = torch.full(
                (1,), pos - 1, dtype=torch.int32, device=positions.device
            )
        if input_ids.dtype != torch.int32:
            input_ids = input_ids.to(torch.int32)
        if positions.dtype != torch.int32:
            positions = positions.to(torch.int32)
        out = self.call(
            input_ids,
            positions,
            self.prev_pos,
            self.init_streams,
            *self.weights,
            *self.dsa_caches,
            *self.kda_caches,
            *self.experts,
        )
        self.steps += 1
        self.last_steps = None
        logits, hidden, self.prev_pos, nd = out[0], out[1], out[2], len(self.dsa_caches)
        caches = list(out[3:])
        self.dsa_caches, self.kda_caches = caches[:nd], caches[nd:]
        logits = tensor_model_parallel_all_gather(logits, dim=-1)[:, : self.vocab]
        return LogitsProcessorOutput(next_token_logits=logits, hidden_states=hidden)
