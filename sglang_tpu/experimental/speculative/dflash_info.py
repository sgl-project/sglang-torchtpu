"""DSpark overlap state; bucketed addressing is owned by the TPU plugin."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sglang.srt.runtime_context import get_spec
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2


@dataclass
class HostPaddedDraftInput(DFlashDraftInputV2):
    tpu_req_pool_indices: torch.Tensor | None = None

    def _maybe_prepare_device_accept_tpu(
        self, batch, *, bs, block_size, committed_lens_cpu
    ):
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
        if self.future_indices is None or future_indices is None:
            return super().filter_batch(new_indices, new_indices_cpu)
        assert new_indices_cpu is not None
        if self.nxt_kv_lens_cpu is not None:
            self.nxt_kv_lens_cpu = self.nxt_kv_lens_cpu[new_indices_cpu]
            self.nxt_kv_lens_sum = int(self.nxt_kv_lens_cpu.sum())
        self.future_indices = future_indices

    def merge_batch(self, spec_info, *, future_indices=None):
        assert isinstance(spec_info, HostPaddedDraftInput)
        self._drop_window()
        if self.future_indices is None or future_indices is None:
            return super().merge_batch(spec_info)
        assert spec_info.future_indices is not None
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
        self.future_indices = future_indices
