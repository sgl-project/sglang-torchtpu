# SPDX-License-Identifier: Apache-2.0
"""Host and device state layout for bucketed chain verification."""

from __future__ import annotations

import contextlib
from typing import Any

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang_tpu.experimental.speculative.spec_types import (
    SpecState,
    TpuSpecVerifyInput,
)
from sglang_tpu.experimental.speculative.spec_utils import overridden_fields


class SpecStateLayout:
    @staticmethod
    def bucket_view(
        batch: ScheduleBatch,
        *,
        bucket_state: SpecState,
        verify_input: TpuSpecVerifyInput,
        kv_rows: int,
        **overrides: Any,
    ) -> contextlib.AbstractContextManager[ScheduleBatch]:
        """batch with its row fields at the bucket, restored on exit, for code that reads them.

        init_new snapshots these; feeding the bucket rows leaves the shim nothing
        to pad. Padded rows hold pool row 0 and host KV length 0, inactive in RPA;
        seq_lens_cpu bounds the KV length once kv_rows rows past seq_lens are
        written. overrides set (and restore) further batch fields.
        """
        padded_bs = bucket_state.seq_lens.shape[0]
        assert verify_input.req_pool_indices_cpu.shape[0] == padded_bs
        seq_lens_cpu, seq_lens_sum = verify_input.padded_kv_lens_after_rows(kv_rows)
        multimodal_inputs = batch.multimodal_inputs
        if multimodal_inputs is not None:
            # Spec M-RoPE positions read one entry per (bucket) row.
            multimodal_inputs = multimodal_inputs + [None] * (
                padded_bs - len(multimodal_inputs)
            )
        view_fields = {
            "multimodal_inputs": multimodal_inputs,
            "seq_lens": bucket_state.seq_lens,
            "seq_lens_cpu": seq_lens_cpu,
            "seq_lens_sum": seq_lens_sum,
            # Ignored by direct compile on TPU; None keeps the shim from padding it per bs.
            "orig_seq_lens": None,
            "req_pool_indices": bucket_state.req_pool_indices,
            "req_pool_indices_cpu": verify_input.req_pool_indices_cpu,
            **overrides,
        }
        return overridden_fields(batch, **view_fields)
