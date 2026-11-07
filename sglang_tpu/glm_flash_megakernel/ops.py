# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash decode megakernel in serving (experimental, opt-in).

Runs target steps (the MTP verify, or plain decode without MTP) of up to
`max_rows` rows as one Pallas call per device; every other step runs the model.
The kernel reads serving's KV pools and KDA state rows in place through the
step's page table and writes the step's rows and checkpoints back into them, so
serving's own verify commit and radix snapshots apply unchanged.
"""

from __future__ import annotations

import logging

import torch

from sglang.srt.compilation.compiled_op import compiled_op, warmup_compiled_op
from sglang_tpu.envs import envs as tpu_envs

logger = logging.getLogger(__name__)

_AXIS = "tp"


def _pad(t: torch.Tensor, n: int) -> torch.Tensor:
    return torch.nn.functional.pad(t, (0, n - t.shape[0]))


@compiled_op()
def _step_inputs_op(
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    page_indices: torch.Tensor,
    req_pool_indices: torch.Tensor,
    slot_map: torch.Tensor,
    v2p: torch.Tensor | None,
    read_offsets: torch.Tensor | None,
    *,
    q: int,
    b0: int,
    n: int,
    bucket: int,
    pages_per_seq: int,
    window_start: int,
    checkpoint_width: int,
):
    """The kernel's int32 SMEM inputs for requests [b0, b0 + n), padded to
    `bucket` requests: one program instead of ~15 eager launches per call."""
    rows = bucket * q
    r0, r1 = b0 * q, (b0 + n) * q
    tok = _pad(input_ids[r0:r1].to(torch.int32), rows)
    pos = _pad(positions[r0:r1].to(torch.int32).reshape(n, q)[:, 0], bucket)
    tab = page_indices.reshape(-1, pages_per_seq)[b0 : b0 + n].reshape(-1)
    tab = _pad(tab.to(torch.int32), bucket * pages_per_seq)
    req_rows = req_pool_indices[b0 : b0 + n]
    slots = slot_map[req_rows]
    if v2p is not None:
        slots = v2p[slots]
    slots = slots.to(torch.int64)
    if read_offsets is None:
        # No checkpoint windows (no spec): the slot's own row is read and written.
        srow, ckrow = slots, slots
    else:
        bases = window_start + req_rows.clamp(min=0) * checkpoint_width
        offsets = read_offsets.index_select(0, slots)
        srow = torch.where(offsets < 0, slots, bases + offsets)
        steps = torch.arange(
            checkpoint_width, dtype=req_rows.dtype, device=req_rows.device
        )
        ckrow = (bases.unsqueeze(1) + steps).reshape(-1)
    return tok, pos, tab, _pad(srow.to(torch.int32), bucket), _pad(ckrow.to(torch.int32), rows)


def _owned(t: torch.Tensor) -> torch.Tensor:
    # A device-side clone or copy keeps the view; a host round trip uploads a plain buffer.
    return t.cpu().to(t.device)


def enabled() -> bool:
    return tpu_envs.SGLANG_TPU_GLM_FLASH_MEGAKERNEL.get()


def _buckets(max_req: int) -> list[int]:
    """Request-count buckets 1, 2, 4, ... up to max_req."""
    out, b = [], 1
    while b < max_req:
        out.append(b)
        b *= 2
    return out + [max_req]


class GlmFlashMegakernel:
    def __init__(
        self,
        model,
        model_path: str,
        max_len: int,
        n_query: int,
        max_rows: int,
        backend,
        call_rows: int | None = None,
        max_requests: int | None = None,
    ):
        import jax
        from jax.sharding import NamedSharding
        from jax.sharding import PartitionSpec as P
        from torch_tpu._internal.pallas import get_pallas_mesh
        from torch_tpu._internal.pallas import pallas as pallas_impl

        from sglang.srt.runtime_context import get_parallel
        from sglang_tpu.attention_pallas_rpa_base import _get_pages_per_seq
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
        self.spec, self.tp, self.Q = spec, tp, n_query
        self.vocab = spec.vocab_size
        self.device = device
        layout = KernelLayout(spec, tp)
        small_map, small_f32_map = slab_maps(spec)
        mesh = get_pallas_mesh((_AXIS,), (tp,))

        full = backend.full_attn_backend
        self._full, self._lin = full, backend.linear_attn_backend
        self._pool = full.token_to_kv_pool
        self._req_pool = self._lin.req_to_token_pool
        self._mamba = self._req_pool.mamba_pool
        page_size = self._pool.page_size
        table_pages = _get_pages_per_seq(full.max_context_len, page_size)
        max_len = min(max_len, table_pages * page_size)
        max_len = -(-max_len // D.KernelConfig.attn_tile) * D.KernelConfig.attn_tile
        self.max_len = max_len
        self.pages_per_seq = table_pages
        self.max_req = max(1, max_rows // n_query)
        # Requests per kernel call; a larger batch is split into calls of this size.
        call_rows = max_rows if call_rows is None else min(call_rows, max_rows)
        self.call_req = max(1, call_rows // n_query)
        # The scheduler never batches past its concurrency cap, so larger buckets
        # would only be compiled (one Pallas kernel each), never run.
        if max_requests is not None:
            self.call_req = max(1, min(self.call_req, max_requests))
        km_tile = 512 if self.max_req * n_query == 1 else 256

        def cfg_for(b):
            # Larger steps trade DMA ring depth for the VMEM their row-scaled buffers need.
            big = b * n_query > 16
            return D.KernelConfig(
                max_len=max_len,
                n_query=n_query,
                n_req=b,
                page_size=page_size,
                table_pages=table_pages,
                embed_2d=True,
                km_tile=km_tile,
                slots=3 if big else 4,
                prefetch=1 if big else 2,
                km_slots=3 if big else 4,
                km_issue=1 if big else 2,
                unroll_periods=big,
                single_slab=big,
            )

        self.cfgs = {b: cfg_for(b) for b in _buckets(self.call_req)}
        cfg0 = next(iter(self.cfgs.values()))
        weights = build_weights(model, spec, layout, cfg0, tp, rank)
        # Packing leaves dtype and shape views, which every call would re-materialise.
        self.weights = [_owned(weights[n]) for n in R.PACKED_ORDER]
        self.experts = [w for n in R.EXPERT_LISTS for w in weights[n]]
        nbytes = sum(t.numel() * t.element_size() for t in self.weights)
        logger.info(
            "GLM-Flash megakernel: %.1f GiB of packed weights on rank %d (experts are "
            "sglang's); request buckets %s x %d rows",
            nbytes / 2**30,
            rank,
            sorted(self.cfgs),
            n_query,
        )
        del weights

        self.pools = {
            "nope": [self._pool.get_nope_cache(L) for L in spec.dsa_layers],
            "raw": [self._pool.get_kpool_raw_cache(L) for L in spec.dsa_layers],
            "records": [self._pool.get_index_key_cache(L) for L in spec.dsa_layers],
            "temporal": [
                self._req_pool.mamba2_layer_cache(L).temporal for L in spec.kda_layers
            ],
            "conv": [
                self._req_pool.mamba2_layer_cache(L).conv[0] for L in spec.kda_layers
            ],
        }
        self.pool_list = [p for n in R.POOL_ORDER for p in self.pools[n]]
        k_scales = []
        for L in spec.dsa_layers:
            attn = getattr(model.model.layers[L].self_attn, "attn_mqa", None)
            ks = getattr(attn, "k_scale_float", None)
            k_scales.append(1.0 if ks is None else float(ks))
        self.k_scales = torch.tensor(k_scales, dtype=torch.float32, device=device)

        n_dsa, n_kda = len(spec.dsa_layers), len(spec.kda_layers)
        nw, n_moe = len(R.PACKED_ORDER), len(self.experts) // len(R.EXPERT_LISTS)
        n_pool = 3 * n_dsa + 2 * n_kda
        # tok, pos, meta, tab, srow, ckrow, k_scales, init_streams
        n_lead = 8

        def export(name, fn, in_specs, out_specs, donate=(), aliases=None):
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
                input_output_aliases=aliases or {},
            )
            call.sglang_output_shardings = out_sh
            return call

        self.calls, self.init_streams = {}, {}
        for b, cfg in self.cfgs.items():
            with tpu_info_for_cpu_export():
                run = R.make_decode(spec, layout, small_map, small_f32_map, mesh, cfg)
            pspec = [s for n in R.POOL_ORDER for s in run.pspec[n]]

            def step(tok, pos, meta, tab, srow, ckrow, ks, init, *rest, run=run):
                ws = dict(zip(R.PACKED_ORDER, rest[:nw]))
                pools, i = {}, nw
                for n in R.POOL_ORDER:
                    cnt = len(run.pspec[n])
                    pools[n] = list(rest[i : i + cnt])
                    i += cnt
                experts = rest[i:]
                for j, n in enumerate(R.EXPERT_LISTS):
                    ws[n] = list(experts[j * n_moe : (j + 1) * n_moe])
                logits, hidden, new = run.shard(
                    tok, pos, meta, tab, srow, ckrow, ks, init, ws, pools
                )
                return (logits, hidden) + tuple(p for n in R.POOL_ORDER for p in new[n])

            in_specs = (
                [P()] * 7
                + [P(None, None)]
                + [run.wspec[n] for n in R.PACKED_ORDER]
                + pspec
                + [s for n in R.EXPERT_LISTS for s in run.wspec[n]]
            )
            pool0 = n_lead + nw
            self.calls[b] = export(
                f"jax_glm_flash_mk_L{max_len}_q{n_query}_b{b}",
                step,
                in_specs,
                # Keep logits vocab-sharded: replicated here, their mesh sharding leaks into
                # serving's compiled ops, which then fail to compile.
                [P(None, _AXIS), P(None, None)] + pspec,
                donate=tuple(range(pool0, pool0 + n_pool)),
                aliases={pool0 + i: 2 + i for i in range(n_pool)},
            )
            self.init_streams[b] = torch.zeros(
                (cfg.rows * spec.hc_mult, spec.hidden_size),
                dtype=torch.bfloat16,
                device=device,
            )
        self._meta = {
            b: torch.tensor([b] + [0] * 7, dtype=torch.int32, device=device)
            for b in range(1, self.call_req + 1)
        }
        # Set when the radix cache is on: only then does serving snapshot KDA state.
        self.radix_snapshots = False
        self._warmup_step_inputs(device)

    def handles(self, forward_batch, input_ids) -> bool:
        bs = forward_batch.batch_size
        if bs < 1 or bs > self.max_req or input_ids.shape[0] != bs * self.Q:
            return False
        # A request longer than the kernel's page table continues on the model.
        seq_lens_cpu = forward_batch.seq_lens_cpu
        if (
            seq_lens_cpu is not None
            and int(seq_lens_cpu[:bs].max()) + self.Q > self.max_len
        ):
            return False
        mode = forward_batch.forward_mode
        if self.Q > 1:
            return mode.is_target_verify()
        return mode.is_decode() and forward_batch.spec_info is None

    def _bucket(self, bs: int) -> int:
        return min(b for b in self.cfgs if b >= bs)

    def _calls(self, bs: int) -> list[tuple[int, int]]:
        """(first request, count) of each kernel call covering `bs` requests."""
        out, b0 = [], 0
        while b0 < bs:
            n = min(self.call_req, bs - b0)
            out.append((b0, n))
            b0 += n
        return out

    def _step_input_tables(self):
        """(slot map, virtual->physical slot table, checkpoint read offsets) and the
        window constants the input program reads."""
        mamba = self._mamba
        return dict(
            slot_map=self._req_pool.req_index_to_mamba_index_mapping,
            v2p=self._req_pool.mamba_v2p_table,
            read_offsets=mamba.checkpoint_read_offsets,
            window_start=mamba.window_start or 0,
            checkpoint_width=mamba.checkpoint_width,
        )

    def _warmup_step_inputs(self, device) -> None:
        """Compile the input program for every call size; a cold compile at the
        first request of a size would otherwise land inside serving."""
        Q = self.Q
        for n in range(1, self.call_req + 1):
            bucket = self._bucket(n)
            zeros = lambda *shape, dtype=torch.int64: torch.zeros(
                shape, dtype=dtype, device=device
            )
            for padded_bs in sorted({n, bucket}):
                warmup_compiled_op(
                    _step_inputs_op,
                    zeros(n * Q),
                    zeros(n * Q),
                    zeros(padded_bs * self.pages_per_seq, dtype=torch.int32),
                    zeros(n),
                    q=Q,
                    b0=0,
                    n=n,
                    bucket=bucket,
                    pages_per_seq=self.pages_per_seq,
                    **self._step_input_tables(),
                )

    def _step_inputs(
        self, input_ids, positions, forward_batch, b0: int, n: int, bucket: int
    ):
        """The kernel's SMEM inputs for requests [b0, b0 + n), padded to `bucket` requests."""
        rows = bucket * self.Q
        tok, pos, tab, srow, ckrow = _step_inputs_op(
            input_ids,
            positions,
            self._full.forward_metadata.page_indices,
            forward_batch.req_pool_indices,
            q=self.Q,
            b0=b0,
            n=n,
            bucket=bucket,
            pages_per_seq=self.pages_per_seq,
            **self._step_input_tables(),
        )
        assert ckrow.shape[0] == rows, (ckrow.shape, rows)
        return (
            tok,
            pos,
            self._meta[n],
            tab,
            srow,
            ckrow,
            self.k_scales,
            self.init_streams[bucket],
        )

    def _decode_snapshot(self, forward_batch):
        """Radix snapshot after a decode step: the slot's committed row into its track
        slot, as the GDN backend's decode op does. The mask is a host tensor on TPU."""
        mask = forward_batch.mamba_track_mask
        if mask is None or not bool(mask.any()):
            return
        from sglang_tpu.mamba_state_copy import compiled_state_copy, copy_bucket_size

        slots, _ = self._lin._slot_indices(forward_batch.req_pool_indices)
        dst = self._lin._translate_mamba_indices(forward_batch.mamba_track_indices)
        dst = torch.where(mask.to(dst.device), dst, torch.zeros_like(dst)).to(
            torch.int32
        )
        src = slots.to(torch.int32)
        bucket = copy_bucket_size(src.shape[0])
        count = torch.tensor([src.shape[0]], dtype=torch.int32, device=src.device)
        compiled_state_copy()(
            self._mamba._copy_states(),
            self._pad(src, bucket),
            self._pad(dst, bucket),
            count,
        )

    def forward(self, input_ids, positions, forward_batch):
        from sglang.srt.distributed import tensor_model_parallel_all_gather
        from sglang.srt.layers.logits_processor import LogitsProcessorOutput

        bs = forward_batch.batch_size
        logits, hidden = [], []
        # A batch beyond one call's rows runs as several calls (each streams the weights).
        for b0, n in self._calls(bs):
            bucket = self._bucket(n)
            inputs = self._step_inputs(
                input_ids, positions, forward_batch, b0, n, bucket
            )
            out = self.calls[bucket](
                *inputs, *self.weights, *self.pool_list, *self.experts
            )
            logits.append(out[0][: n * self.Q])
            hidden.append(out[1][: n * self.Q])
        logits = torch.cat(logits) if len(logits) > 1 else logits[0]
        hidden = torch.cat(hidden) if len(hidden) > 1 else hidden[0]
        logits = tensor_model_parallel_all_gather(logits, dim=-1)[:, : self.vocab]
        if self.radix_snapshots and self.Q == 1:
            self._decode_snapshot(forward_batch)
        return LogitsProcessorOutput(next_token_logits=logits, hidden_states=hidden)
