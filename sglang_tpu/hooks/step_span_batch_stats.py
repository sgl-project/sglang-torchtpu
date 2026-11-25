# SPDX-License-Identifier: Apache-2.0
"""Attach batch-composition TraceMe metadata to per-step profiler spans.

Why: Upstream ``step[...]`` spans report batch size and extend token count,
     but TPU performance analysis also needs prefill/decode token splits,
     phase classification, padded bucket token count, minimum KV length, and
     a ``batch_id`` that joins 1:1 with auto-profile batch stats JSONL rows.
How: AFTER ``sglang.srt.observability.profile_utils.build_step_span_name``;
     no-op when ``is_draft_worker=True`` or ``torch.autograd._profiler_enabled()``
     is False, otherwise appends ``#num_reqs=...,phase=...,batch_id=...,...#``
     (deduplicating keys already present in an existing trailing ``#...#`` block).
Upstream PR or issue: none (TraceMe metadata syntax and TPU bucket metrics).
Remove when: core exposes a platform hook for step-span metadata.
"""

from __future__ import annotations

from typing import Any

import torch

from sglang_tpu.profile_batch_stats import (
    append_span_metadata,
    consume_active_batch_id,
    get_batch_composition_stats,
)


def append_batch_stats_to_step_span(
    result: str,
    forward_batch: Any,
    *args: Any,
    **kwargs: Any,
) -> str:
    """AFTER hook on ``build_step_span_name`` to embed batch composition stats."""
    if kwargs.get("is_draft_worker", False) or not torch.autograd._profiler_enabled():
        return result
    batch_id = consume_active_batch_id()
    if batch_id is None:
        forward_id = kwargs.get("forward_id")
        batch_id = int(forward_id) if forward_id is not None else 0
    stats = get_batch_composition_stats(forward_batch, batch_id=batch_id)
    metadata = {
        ("min_kv_length" if k == "min_kv_len" else k): v for k, v in stats.items()
    }
    return append_span_metadata(result, metadata)
