"""Prepare an EAGLE step before its dependent model submissions."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import msgspec
import torch
import torch.nn.functional as F

from sglang.srt.hardware_backend.tpu.attention_pallas_rpa_base import (
    BaseTpuPallasRpaAttnBackend,
    BaseTpuPallasRpaMultiStepDraftBackend,
    PallasForwardMetadata,
    _get_pages_per_seq,
    _validate_kv_len_capacity,
)
from sglang.srt.hardware_backend.tpu.compilation.compiled_forward_shim import (
    _REAL_B_KEY,
    _REAL_Q_KEY,
)
from sglang.srt.hardware_backend.tpu.compilation.spec_decode_compile_policy import (
    _pad_first_dim,
)
from sglang.srt.hardware_backend.tpu.compilation.spec_decode_compile_policy_eagle3 import (
    Eagle3SpecDecodeCompilePolicyMixin,
)
from sglang.srt.hardware_backend.tpu.eagle_step_inputs import (
    prepare_eagle_step_inputs,
)
from sglang.srt.model_executor.forward_batch_info import (
    ForwardMode,
    PreparedExtendMetadata,
)
from sglang.srt.runtime_context import get_spec
from sglang.srt.speculative.eagle_info import EagleDraftInput

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch


class _Layout(msgspec.Struct, frozen=True):
    req_pool_indices_cpu: torch.Tensor
    req_pool_indices: torch.Tensor
    logical_bs: torch.Tensor
    query_lens: torch.Tensor
    extend_start_loc: torch.Tensor
    retrieve_index: torch.Tensor
    decode_cu_q_lens: torch.Tensor
    verify_cu_q_lens: torch.Tensor
    decode_active_tokens: torch.Tensor
    verify_active_tokens: torch.Tensor


class _Pages(msgspec.Struct, frozen=True):
    cpu_indices: torch.Tensor
    cpu_swa_indices: torch.Tensor | None
    cu_kv_lens: torch.Tensor
    indices: torch.Tensor
    swa_indices: torch.Tensor | None


def _pad_sampling(value, rows, fill):
    if value is None or value.shape[0] == rows:
        return value
    return F.pad(
        value, [0, 0] * (value.ndim - 1) + [0, rows - value.shape[0]], value=fill
    )


class EagleStepPreparer:
    @classmethod
    def create(cls, *, target_runner, draft_runner, draft_attn_backend, server_args):
        target_policy = target_runner.compile_policy
        draft_policy = draft_runner.compile_policy
        spec = get_spec()
        if (
            not isinstance(target_policy, Eagle3SpecDecodeCompilePolicyMixin)
            or not isinstance(draft_policy, Eagle3SpecDecodeCompilePolicyMixin)
            or not isinstance(draft_attn_backend, BaseTpuPallasRpaMultiStepDraftBackend)
            or server_args.enable_dp_attention
            or server_args.enable_hierarchical_cache
            or spec.speculative_adaptive
            or spec.speculative_use_rejection_sampling
            or spec.speculative_eagle_topk != 1
            or spec.speculative_num_steps < 2
            or spec.speculative_num_draft_tokens != spec.speculative_num_steps + 1
            or target_runner.model_config.is_multimodal
            or target_runner.model_config.model_is_mrope
            or target_runner.canary_manager is not None
            or draft_runner.canary_manager is not None
            or target_runner.ngram_embedding_manager.enabled
            or draft_runner.ngram_embedding_manager.enabled
            or target_policy.tracer != "direct"
            or draft_policy.tracer != "direct"
        ):
            return None
        backends = (
            *draft_attn_backend.attn_backends,
            target_runner.attn_backend,
            draft_runner.attn_backend,
        )
        if not all(
            isinstance(backend, BaseTpuPallasRpaAttnBackend)
            and backend.kernel_spec.uses_draft_extend_v2
            and backend.req_to_token_pool.req_to_token.device.type == "cpu"
            for backend in backends
        ):
            return None
        return cls(
            target_policy=target_policy,
            draft_policy=draft_policy,
            draft_backends=tuple(draft_attn_backend.attn_backends),
            target_backend=target_runner.attn_backend,
            extend_backend=draft_runner.attn_backend,
            verify_width=spec.speculative_num_draft_tokens,
        )

    def __init__(
        self,
        *,
        target_policy,
        draft_policy,
        draft_backends,
        target_backend,
        extend_backend,
        verify_width,
    ):
        self.policies = (target_policy, draft_policy)
        self.draft_backends = draft_backends
        self.target_backend = target_backend
        self.extend_backend = extend_backend
        self.backends = tuple(
            dict.fromkeys((*draft_backends, target_backend, extend_backend))
        )
        self.verify_width = verify_width
        self.bucketer = draft_policy.b_bucketer
        self._layouts = {}
        self._pages = {}
        self._page_columns = {}
        self._page_groups = {}
        for backend in self.backends:
            group = (
                id(backend.req_to_token_pool),
                id(backend.token_to_kv_pool) if backend._is_hybrid_swa else None,
                backend.page_size,
                backend.max_context_len,
                str(backend.device),
            )
            self._page_groups[backend] = group
            pages = _get_pages_per_seq(backend.max_context_len, backend.page_size)
            self._page_columns[group] = torch.arange(pages) * backend.page_size

    def prepare(self, batch: ScheduleBatch):
        draft = batch.spec_info
        if (
            not batch.forward_mode.is_decode()
            or not isinstance(draft, EagleDraftInput)
            or draft.tpu_relay_inputs is None
            or batch.seq_lens_cpu is None
            or batch.return_logprob
            or batch.has_grammar
            or batch.input_embeds is not None
            or batch.encoder_lens is not None
            or batch.mamba_track_indices is not None
            or draft.dsa_topk_indices is not None
            or any(req.lora_id is not None for req in batch.reqs)
        ):
            return None
        relay = draft.tpu_relay_inputs
        bs = batch.seq_lens.shape[0]
        physical_bs = self.bucketer.next_bucket(bs)
        if (
            bs == 0
            or relay.logical_seq_lens is not batch.seq_lens
            or relay.seq_lens.shape != (physical_bs,)
            or relay.cache_locs.shape != (physical_bs, self.verify_width)
            or any(
                policy.b_bucketer.next_bucket(bs) != physical_bs
                for policy in self.policies
            )
            or batch.seq_lens_cpu.shape != (bs,)
            or int(batch.seq_lens_cpu.min()) <= 0
        ):
            return None
        layout = self._prepare_layout(batch, physical_bs=physical_bs)
        prefix_cpu = F.pad(batch.seq_lens_cpu.to(torch.int32), (0, physical_bs - bs))
        prefix = prefix_cpu.to(batch.seq_lens.device, non_blocking=True)
        inputs = prepare_eagle_step_inputs(
            relay.seq_lens,
            relay.topk_p,
            relay.topk_index,
            relay.bonus_tokens,
            relay.hidden_states,
            relay.cache_locs,
            prefix,
            layout.logical_bs,
            self.verify_width,
        )
        pages = {}
        for backend in self.backends:
            group = self._page_groups[backend]
            if group not in pages:
                pages[group] = self._prepare_pages(backend, layout.req_pool_indices_cpu)
        return PreparedEagleStep(
            owner=self,
            batch=batch,
            layout=layout,
            inputs=inputs,
            pages=pages,
            prefix_cpu=prefix_cpu,
            prefix=prefix,
        )

    def _prepare_layout(self, batch, *, physical_bs):
        bs = batch.seq_lens.shape[0]
        rows = batch.req_pool_indices_cpu
        cached = self._layouts.get(bs)
        if cached is not None and torch.equal(cached.req_pool_indices_cpu[:bs], rows):
            return cached
        device, width = batch.seq_lens.device, self.verify_width
        padded_rows = F.pad(rows, (0, physical_bs - bs)).clone()
        retrieve = F.pad(
            torch.arange(bs * width, dtype=torch.int64).view(bs, width),
            (0, 0, 0, physical_bs - bs),
            value=-1,
        )
        layout = _Layout(
            req_pool_indices_cpu=padded_rows,
            req_pool_indices=padded_rows.to(
                device, dtype=batch.req_pool_indices.dtype, non_blocking=True
            ),
            logical_bs=torch.tensor(bs, dtype=torch.int32, device=device),
            query_lens=torch.full(
                (physical_bs,), width, dtype=torch.int32, device=device
            ),
            extend_start_loc=F.pad(
                torch.arange(bs, dtype=torch.int32) * width, (0, physical_bs - bs)
            ).to(device, non_blocking=True),
            retrieve_index=retrieve.to(device, non_blocking=True),
            decode_cu_q_lens=torch.arange(physical_bs + 1, dtype=torch.int32).to(
                device, non_blocking=True
            ),
            verify_cu_q_lens=(
                torch.arange(physical_bs + 1, dtype=torch.int32) * width
            ).to(device, non_blocking=True),
            decode_active_tokens=torch.tensor(bs, dtype=torch.int32, device=device),
            verify_active_tokens=torch.tensor(
                bs * width, dtype=torch.int32, device=device
            ),
        )
        # At most one layout per live batch size; old in-flight tensors stay owned by their step.
        self._layouts[bs] = layout
        return layout

    def _prepare_pages(self, backend, rows):
        group = self._page_groups[backend]
        page_size = backend.page_size
        columns = self._page_columns[group]
        indices = (
            (
                backend.req_to_token_pool.req_to_token[
                    rows.long()[:, None], columns[None, :]
                ]
                // page_size
            )
            .to(torch.int32)
            .flatten()
        )
        swa_indices = None
        if backend._is_hybrid_swa:
            mapping = backend.token_to_kv_pool.full_to_swa_index_mapping
            assert mapping.device.type == "cpu"
            swa_indices = (mapping[indices.long() * page_size] // page_size).to(
                torch.int32
            )
        previous = self._pages.get(group)
        if (
            previous is not None
            and torch.equal(previous.cpu_indices, indices)
            and (
                swa_indices is None
                and previous.cpu_swa_indices is None
                or swa_indices is not None
                and previous.cpu_swa_indices is not None
                and torch.equal(previous.cpu_swa_indices, swa_indices)
            )
        ):
            return previous
        # Compare owned CPU values, not tensor versions: pool writers may use aliases.
        pages = _Pages(
            cpu_indices=indices,
            cpu_swa_indices=swa_indices,
            cu_kv_lens=(
                torch.arange(rows.numel() + 1, dtype=torch.int32)
                * columns.numel()
                * page_size
            ).to(backend.device, non_blocking=True),
            indices=indices.to(backend.device, non_blocking=True),
            swa_indices=(
                swa_indices.to(backend.device, non_blocking=True)
                if swa_indices is not None
                else None
            ),
        )
        self._pages[group] = pages
        return pages


class PreparedEagleStep:
    def __init__(self, *, owner, batch, layout, inputs, pages, prefix_cpu, prefix):
        self.owner, self.layout, self.inputs = owner, layout, inputs
        self.bs = batch.seq_lens.shape[0]
        self.physical_bs = inputs.seq_lens.shape[0]
        self.width = owner.verify_width
        self.logical_seq_lens = batch.seq_lens
        self.seq_lens_cpu = batch.seq_lens_cpu
        self.draft_seq_lens_cpu = F.pad(
            batch.seq_lens_cpu, (0, self.physical_bs - self.bs)
        )
        self.extend_seq_lens_cpu = F.pad(
            batch.seq_lens_cpu + self.width, (0, self.physical_bs - self.bs)
        )
        self.extend_metadata = PreparedExtendMetadata(
            seq_lens=layout.query_lens,
            prefix_lens=prefix,
            positions=inputs.extend_positions,
            start_loc=layout.extend_start_loc,
            seq_lens_cpu=[self.width] * self.physical_bs,
            prefix_lens_cpu=prefix_cpu.tolist(),
        )
        sampling = batch.sampling_info
        self.sampling = (
            _pad_sampling(sampling.temperatures, self.physical_bs, 1.0),
            _pad_sampling(sampling.top_ks, self.physical_bs, 1),
            _pad_sampling(sampling.top_ps, self.physical_bs, 1.0),
        )
        self.verify_tokens = None
        self._metadata = {}
        self._fused_draft_metadata = {}
        for backend in owner.backends:
            page_data = pages[owner._page_groups[backend]]
            max_len = int(batch.seq_lens_cpu.max()) + self.width
            _validate_kv_len_capacity(
                max_len,
                owner._page_columns[owner._page_groups[backend]].numel()
                * backend.page_size,
            )
            for mode, kv_lens in self._phase_lengths(backend):
                decode = mode.is_decode()
                self._metadata[backend, mode] = PallasForwardMetadata(
                    kv_lens=kv_lens,
                    cu_q_lens=(
                        layout.decode_cu_q_lens if decode else layout.verify_cu_q_lens
                    ),
                    cu_kv_lens=page_data.cu_kv_lens,
                    page_indices=page_data.indices,
                    distribution=backend._get_cached_rpa_distribution(
                        decode, self.bs, backend.device
                    ),
                    active_num_tokens=(
                        layout.decode_active_tokens
                        if decode
                        else layout.verify_active_tokens
                    ),
                    swa_page_indices=page_data.swa_indices,
                )
                if decode:
                    # The fused graph owns its per-step casts and increments.
                    self._fused_draft_metadata[backend] = replace(
                        self._metadata[backend, mode], kv_lens=inputs.seq_lens
                    )

    def _phase_lengths(self, backend):
        if backend is self.owner.target_backend:
            yield ForwardMode.TARGET_VERIFY, self.inputs.verify_kv_lens
        if backend is self.owner.extend_backend:
            yield ForwardMode.DRAFT_EXTEND_V2, self.inputs.extend_kv_lens
        if backend in self.owner.draft_backends:
            yield ForwardMode.DECODE, self.inputs.draft_kv_lens[
                backend.speculative_step_id
            ]

    def __enter__(self):
        for owner in (*self.owner.policies, *self.owner.backends):
            assert owner.prepared_eagle_step is None
            owner.prepared_eagle_step = self
        return self

    def __exit__(self, exc_type, exc, tb):
        for owner in (*self.owner.policies, *self.owner.backends):
            owner.prepared_eagle_step = None

    def metadata_for(self, backend, forward_batch, *, fused_draft=False):
        expected = (
            self.inputs.extend_seq_lens
            if forward_batch.forward_mode.is_draft_extend_v2()
            else self.inputs.seq_lens
        )
        assert (
            forward_batch.seq_lens is expected
        ), "prepared EAGLE lengths must belong to this step"
        assert forward_batch.batch_size == self.physical_bs
        if fused_draft:
            assert forward_batch.forward_mode.is_decode()
            return self._fused_draft_metadata[backend]
        return self._metadata[backend, forward_batch.forward_mode]

    def _bind_common(self, forward_batch, *, extend=False):
        forward_batch.batch_size = self.physical_bs
        forward_batch.req_pool_indices = self.layout.req_pool_indices
        forward_batch.req_pool_indices_cpu = self.layout.req_pool_indices_cpu
        forward_batch.seq_lens = (
            self.inputs.extend_seq_lens if extend else self.inputs.seq_lens
        )
        setattr(forward_batch, _REAL_B_KEY, self.bs)
        if forward_batch.mm_inputs is not None:
            forward_batch.mm_inputs = [None] * self.physical_bs

    def prepare_draft(self, forward_batch):
        self._bind_common(forward_batch)
        forward_batch.seq_lens_cpu = self.draft_seq_lens_cpu
        forward_batch.positions = self.inputs.seq_lens
        forward_batch.out_cache_loc = self.inputs.draft_cache_locs
        draft = forward_batch.spec_info
        draft.tpu_relay_inputs = None
        draft.topk_p, draft.topk_index = self.inputs.topk_p, self.inputs.topk_index
        draft.bonus_tokens, draft.hidden_states = (
            self.inputs.bonus_tokens,
            self.inputs.hidden_states,
        )
        draft.record_padding(
            logical_batch_size=self.bs,
            physical_batch_size=self.physical_bs,
            logical_num_tokens=self.bs,
            physical_num_tokens=self.physical_bs,
        )

    def prepare_verify(self, forward_batch):
        self._bind_common(forward_batch)
        forward_batch.mm_inputs = [None] * self.physical_bs
        forward_batch.extend_seq_lens_cpu = [0] * self.physical_bs
        forward_batch.extend_prefix_lens_cpu = [0] * self.physical_bs
        forward_batch.extend_logprob_start_lens_cpu = [0] * self.physical_bs
        self.verify_tokens = _pad_first_dim(
            forward_batch.input_ids, self.physical_bs * self.width
        )
        forward_batch.input_ids = self.verify_tokens
        forward_batch.positions = self.inputs.verify_positions
        forward_batch.out_cache_loc = self.inputs.verify_cache_locs
        forward_batch.orig_seq_lens = _pad_first_dim(
            forward_batch.orig_seq_lens, self.physical_bs
        )
        setattr(forward_batch, _REAL_Q_KEY, self.bs * self.width)
        forward_batch.spec_info.record_padding(
            logical_batch_size=self.bs,
            physical_batch_size=self.physical_bs,
            logical_num_tokens=self.bs * self.width,
            physical_num_tokens=self.physical_bs * self.width,
        )

    def prepare_extend(self, forward_batch):
        self._bind_common(forward_batch, extend=True)
        forward_batch.seq_lens_cpu = self.extend_seq_lens_cpu
        forward_batch.seq_lens_sum = int(self.extend_seq_lens_cpu.sum())
        forward_batch._original_num_tokens = self.bs * self.width
        forward_batch.positions = self.inputs.extend_positions
        forward_batch.out_cache_loc = self.inputs.verify_cache_locs
        forward_batch.extend_seq_lens = self.extend_metadata.seq_lens
        forward_batch.extend_prefix_lens = self.extend_metadata.prefix_lens
        forward_batch.extend_start_loc = self.extend_metadata.start_loc
        forward_batch.extend_seq_lens_cpu = list(self.extend_metadata.seq_lens_cpu)
        forward_batch.extend_prefix_lens_cpu = list(
            self.extend_metadata.prefix_lens_cpu
        )
        forward_batch.lora_ids = [None] * self.physical_bs
        forward_batch.input_ids = _pad_first_dim(
            forward_batch.input_ids, self.physical_bs * self.width
        )
        spec = forward_batch.spec_info
        forward_batch.output_cache_loc_backup = forward_batch.out_cache_loc
        forward_batch.hidden_states_backup = spec.hidden_states
        spec.hidden_states = _pad_first_dim(
            spec.hidden_states, self.physical_bs * self.width
        )
        spec.num_correct_drafts = _pad_first_dim(
            spec.num_correct_drafts, self.physical_bs
        )
        spec.num_accept_tokens = _pad_first_dim(
            spec.num_accept_tokens, self.physical_bs
        )
        spec.record_padding(
            logical_batch_size=self.bs,
            physical_batch_size=self.physical_bs,
            logical_num_tokens=self.bs * self.width,
            physical_num_tokens=self.physical_bs * self.width,
        )
