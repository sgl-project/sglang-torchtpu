"""Overlap state a TpuSpecWorker relays between decode steps; bucketed addressing is the plugin's."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import torch

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.mem_cache.allocation_sizing import page_aligned_decode_alloc_lens
from sglang.srt.runtime_context import get_spec
from sglang.srt.speculative.spec_info import SpecInput, SpecInputType
from sglang.srt.speculative.spec_utils import allocate_and_assign_decode_cache


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

        num_verify_rows = int(get_spec().speculative_num_draft_tokens)
        cur_kv_lens, nxt_kv_lens, num_needed_tokens = page_aligned_decode_alloc_lens(
            batch.reqs,
            reserve=2 * num_verify_rows,
            page_size=batch.token_to_kv_pool_allocator.page_size,
        )
        nxt_kv_lens_cpu = torch.tensor(nxt_kv_lens, dtype=torch.int32)
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
        from sglang_tpu.dflash_utils_tpu import plan_dflash_device_accept

        plan = plan_dflash_device_accept(
            batch,
            bs=bs,
            block_size=block_size,
            committed_lens_cpu=committed_lens_cpu,
            compact_draft_cache=get_spec().speculative_draft_window_size is not None,
        )
        self.tpu_verify_cache_candidates = (
            plan.verify_cache_candidates if plan is not None else None
        )
        self.tpu_committed_lens = plan.committed_lens if plan is not None else None
        self.tpu_req_pool_indices = plan.req_pool_indices if plan is not None else None
        self.tpu_host_seq_lens_is_committed = plan is not None

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
        for name in self.RELAYED_FIELDS:
            setattr(self, name, getattr(self, name)[new_indices])

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
            setattr(
                self, name, torch.cat([getattr(self, name), getattr(spec_info, name)])
            )
