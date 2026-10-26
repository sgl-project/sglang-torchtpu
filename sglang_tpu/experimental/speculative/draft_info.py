"""Overlap state a TpuSpecWorker relays between decode steps; bucketed addressing is the plugin's."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import torch

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.mem_cache.allocation_sizing import page_aligned_decode_alloc_lens
from sglang.srt.runtime_context import (
    get_server_args,
    get_spec,
    max_speculative_num_draft_tokens,
)
from sglang.srt.speculative.spec_info import SpecInput, SpecInputType
from sglang.srt.speculative.spec_utils import allocate_and_assign_decode_cache
from sglang_tpu.dflash_utils_tpu import gather_dflash_candidate_window


@dataclass
class HostPaddedDraftInput(SpecInput):
    """Draft state of one TpuSpecWorker step; algorithms subclass it with their draft seed.

    prepare_for_decode reserves 2 * (N + 1) KV slots past the committed length and
    plans the verify window on host from it, so the host length mirror stays one
    unread commit behind the device (tpu_host_seq_lens_is_committed).
    """

    SPEC_INPUT_TYPE: ClassVar[SpecInputType]
    # Per-request tensors a subclass relays; filter/merge slice them before FutureMap owns the rows.
    RELAYED_FIELDS: ClassVar[tuple[str, ...]] = ("bonus_tokens", "new_seq_lens")

    bonus_tokens: torch.Tensor
    new_seq_lens: torch.Tensor
    future_indices: torch.Tensor | None = None
    nxt_kv_lens_cpu: torch.Tensor | None = None
    nxt_kv_lens_sum: int | None = None
    tpu_host_seq_lens_is_committed: bool = False
    tpu_verify_cache_candidates: torch.Tensor | None = None
    tpu_committed_lens: torch.Tensor | None = None
    tpu_req_pool_indices: torch.Tensor | None = None
    # Device tensor read after the step's last KV write; the async result copy
    # copies it so the TPU KV free fence's copy_done covers every KV write.
    kv_fence_tail: torch.Tensor | None = None

    def __post_init__(self):
        super().__init__(spec_input_type=self.SPEC_INPUT_TYPE)
        self.num_tokens_per_req = 1
        self.num_tokens_for_logprob_per_req = 1

    def prepare_for_decode(self, batch: ScheduleBatch) -> None:
        bs = batch.batch_size()
        if bs == 0:
            return
        if not batch.req_to_token_pool.host_resident:
            raise RuntimeError("TpuSpecWorker needs a host-resident req_to_token pool.")
        batch.maybe_evict_swa()

        # The previous, not-yet-copied commit may be wider than the next step.
        # Reserve both windows at capacity so either adaptive transition fits.
        num_verify_rows = int(max_speculative_num_draft_tokens())
        cur_kv_lens, nxt_kv_lens, num_needed_tokens = page_aligned_decode_alloc_lens(
            batch.reqs,
            reserve=2 * num_verify_rows,
            page_size=batch.token_to_kv_pool_allocator.page_size,
        )
        nxt_kv_lens_cpu = torch.tensor(nxt_kv_lens, dtype=torch.int32)
        # Reservations are page-aligned, so most steps need no new slot; then
        # the lookup, allocation and table write would all be empty.
        if num_needed_tokens > 0:
            allocate_and_assign_decode_cache(
                tree_cache=batch.tree_cache,
                req_to_token_pool=batch.req_to_token_pool,
                req_pool_indices=batch.req_pool_indices,
                cur_kv_lens_cpu=torch.tensor(cur_kv_lens, dtype=torch.int32),
                nxt_kv_lens_cpu=nxt_kv_lens_cpu,
                num_needed_tokens=num_needed_tokens,
                batch_size=bs,
                device=batch.device,
                host_req_pool_indices=batch.req_pool_indices_cpu,
            )
        for req, nxt in zip(batch.reqs, nxt_kv_lens):
            req.kv.kv_allocated_len = max(req.kv.kv_allocated_len, nxt)
            req.decode_batch_idx += 1

        committed_lens_cpu = torch.tensor(
            [req.kv.kv_committed_len for req in batch.reqs], dtype=torch.int64
        )
        batch.seq_lens_cpu = committed_lens_cpu
        batch.seq_lens_sum = int(committed_lens_cpu.sum())
        self.nxt_kv_lens_cpu = nxt_kv_lens_cpu
        self.nxt_kv_lens_sum = int(nxt_kv_lens_cpu.sum())
        self._plan_verify_window(
            batch,
            bs=bs,
            block_size=num_verify_rows,
            committed_lens_cpu=committed_lens_cpu,
        )

    def _plan_verify_window(
        self, batch: ScheduleBatch, *, bs: int, block_size: int, committed_lens_cpu
    ) -> None:
        # Kept on host: TpuSpecVerifyInput.from_batch uploads it at the forward's
        # bucket, which DP attention only fixes after this point.
        if not _verify_window_eligible(batch):
            self.tpu_verify_cache_candidates = None
            self.tpu_committed_lens = None
            self.tpu_req_pool_indices = None
            self.tpu_host_seq_lens_is_committed = False
            return
        # The 2 * block_size reservation above maps the whole window.
        assert all(
            int(committed_lens_cpu[i]) + 2 * block_size <= req.kv.kv_allocated_len
            for i, req in enumerate(batch.reqs)
        ), "TpuSpecWorker verify window needs two windows of reserved KV"
        req_pool_indices_cpu = batch.req_pool_indices_cpu[:bs].to(torch.int64)
        self.tpu_verify_cache_candidates = gather_dflash_candidate_window(
            batch.req_to_token_pool.req_to_token,
            req_pool_indices_cpu,
            committed_lens_cpu,
            block_size,
        ).to(torch.int64)
        self.tpu_committed_lens = committed_lens_cpu
        self.tpu_req_pool_indices = req_pool_indices_cpu
        self.tpu_host_seq_lens_is_committed = True

    def _drop_window(self):
        # prepare_for_decode replans bucketed windows after composition changes.
        # Preserve the committed-base flag: the host lengths still lag the device.
        self.tpu_verify_cache_candidates = None
        self.tpu_committed_lens = None
        self.tpu_req_pool_indices = None

    def filter_batch(self, new_indices, new_indices_cpu=None, *, future_indices=None):
        self._drop_window()
        if self.nxt_kv_lens_cpu is not None:
            keep = new_indices_cpu if new_indices_cpu is not None else new_indices.cpu()
            self.nxt_kv_lens_cpu = self.nxt_kv_lens_cpu[keep]
            self.nxt_kv_lens_sum = int(self.nxt_kv_lens_cpu.sum())
        if future_indices is not None and self.future_indices is not None:
            self.future_indices = future_indices
            return
        if self.future_indices is not None:
            self.future_indices = self.future_indices[new_indices]
            return
        # A PD prefill without overlap keeps its seq_lens on host.
        keep_cpu = new_indices_cpu if new_indices_cpu is not None else new_indices.cpu()
        for name in self.RELAYED_FIELDS:
            value = getattr(self, name)
            if value is not None:
                setattr(
                    self,
                    name,
                    value[keep_cpu if value.device.type == "cpu" else new_indices],
                )

    def merge_batch(self, spec_info, *, future_indices=None):
        assert type(spec_info) is type(self), (type(self), type(spec_info))
        self._drop_window()
        if self.nxt_kv_lens_cpu is not None:
            assert spec_info.nxt_kv_lens_cpu is not None
            self.nxt_kv_lens_cpu = torch.cat(
                [self.nxt_kv_lens_cpu, spec_info.nxt_kv_lens_cpu]
            )
            self.nxt_kv_lens_sum = int(self.nxt_kv_lens_cpu.sum())
        elif spec_info.nxt_kv_lens_cpu is not None:
            self.nxt_kv_lens_cpu = spec_info.nxt_kv_lens_cpu
            self.nxt_kv_lens_sum = spec_info.nxt_kv_lens_sum
        self.tpu_host_seq_lens_is_committed &= spec_info.tpu_host_seq_lens_is_committed
        if future_indices is not None and self.future_indices is not None:
            assert spec_info.future_indices is not None
            self.future_indices = future_indices
            return
        if self.future_indices is not None:
            assert spec_info.future_indices is not None
            self.future_indices = torch.cat(
                [self.future_indices, spec_info.future_indices]
            )
            return
        for name in self.RELAYED_FIELDS:
            left, right = getattr(self, name), getattr(spec_info, name)
            if left is None and right is None:
                continue
            if left is None or right is None:
                raise ValueError(f"Cannot merge missing relay field {name}")
            setattr(self, name, torch.cat([left, right]))


def _verify_window_eligible(batch: ScheduleBatch) -> bool:
    # A verify commits at most one window, the single offset the plan enumerates.
    # grammar_retained_tokens can commit less, and the compact draft cache maps
    # its own window from host lengths.
    return (
        batch.enable_overlap
        and isinstance(batch.req_pool_indices_cpu, torch.Tensor)
        and get_spec().speculative_draft_window_size is None
        and (
            not batch.return_logprob
            # Use the worker's request capability even for draft inputs built
            # directly by the PD handoff, before the worker has run a step.
            or batch.spec_algorithm.create_worker(get_server_args()).supports_logprobs
        )
        and not batch.has_grammar
        and all(req.lora_id is None for req in batch.reqs)
    )
