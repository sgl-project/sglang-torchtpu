# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash decode megakernel in serving (experimental, opt-in).

Runs single-request target steps (the MTP verify, or plain decode without MTP) as
one Pallas call per device. A request's caches are gathered from serving's pools at
its first step and then live in the kernel; serving's pools are not updated, so the
radix cache must be off. With MTP the KDA states hold one checkpoint per draft
token, and a verify resumes from the last token the previous verify accepted.
"""

from __future__ import annotations

import logging

import torch

from sglang_tpu.envs import envs as tpu_envs

logger = logging.getLogger(__name__)

_AXIS = "tp"


def enabled() -> bool:
    return tpu_envs.SGLANG_TPU_GLM_FLASH_MEGAKERNEL.get()


class GlmFlashMegakernel:
    def __init__(
        self, model, model_path: str, max_len: int, experts: dict, n_query: int = 1
    ):
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

        weights = build_weights(model, spec, layout, self.cfg, tp, rank, experts)
        self.weights = [weights[n] for n in R.WEIGHT_ORDER]
        nbytes = sum(t.numel() * t.element_size() for t in self.weights)
        logger.info(
            "GLM-Flash megakernel: %.1f GiB of kernel weights on rank %d",
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
        self.smalls = self.weights[R.WEIGHT_ORDER.index("smalls")]

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

        nw, nd = len(R.WEIGHT_ORDER), len(run.dsa_names)

        Q = n_query

        # In-launch bookkeeping: separate eager ops would each be a host-bound launch.
        # The last verify ended at pos - 1, so its last accepted checkpoint is pos - prev - 1.
        def step(tok, positions, prev, init, *rest):
            ws = dict(zip(R.WEIGHT_ORDER, rest[:nw]))
            dc = dict(zip(run.dsa_names, rest[nw : nw + nd]))
            kc = dict(zip(run.kda_names, rest[nw + nd :]))
            pos = positions.reshape(-1)[:1].astype(jax.numpy.int32)
            pos_in = jax.numpy.concatenate([pos, pos - prev - 1]) if Q > 1 else pos
            logits, _, hidden, out = run.shard(
                tok.reshape(Q).astype(jax.numpy.int32), pos_in, init, ws, dc, kc
            )
            return (logits, hidden, pos) + tuple(out[n] for n in run.step_caches)

        in_specs = (
            [P(), P(), P(), P(None, None)]
            + [run.wspec[n] for n in R.WEIGHT_ORDER]
            + [run.dspec[n] for n in run.dsa_names]
            + [run.kspec[n] for n in run.kda_names]
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
        self.req, self.prev_pos = None, None

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
        self.req = None

    def _load_request(self, forward_batch, req: int, pos: int):
        from sglang.srt.model_executor.forward_context import get_attn_backend

        spec = self.spec
        if pos + self.Q > self.max_len:
            raise ValueError(
                f"position {pos} exceeds the megakernel's max_len {self.max_len}"
            )
        backend = get_attn_backend()
        full = getattr(backend, "full_attn_backend", backend)
        lin = getattr(backend, "linear_attn_backend", backend)
        pool, req_pool = full.token_to_kv_pool, lin.req_to_token_pool
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
            offsets = req_pool.mamba_pool.checkpoint_read_offsets
            row = slot * self.Q + offsets.to(device).index_select(0, slot)
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
        self.req = req

    def forward(self, input_ids, positions, forward_batch):
        from sglang.srt.distributed import tensor_model_parallel_all_gather
        from sglang.srt.layers.logits_processor import LogitsProcessorOutput

        req = int(forward_batch.req_pool_indices_cpu[0])
        if req != self.req:
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
        )
        logits, hidden, self.prev_pos, nd = out[0], out[1], out[2], len(self.dsa_caches)
        caches = list(out[3:])
        self.dsa_caches, self.kda_caches = caches[:nd], caches[nd:]
        logits = tensor_model_parallel_all_gather(logits, dim=-1)[:, : self.vocab]
        return LogitsProcessorOutput(next_token_logits=logits, hidden_states=hidden)
