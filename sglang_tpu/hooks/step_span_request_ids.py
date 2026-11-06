# SPDX-License-Identifier: Apache-2.0
"""Attach request-ID TraceMe metadata to per-step profiler spans.

Why: Upstream ``step[...]`` spans only report batch-level aggregates; embedding
     ``Name#k1=v1,k2=v2#`` TraceMe metadata surfaces active ``ForwardBatch.rids``
     as structured event attributes in XProf Trace Viewer so tail latency and
     stragglers can be traced back to specific requests.
How: AFTER ``sglang.srt.observability.profile_utils.build_step_span_name``;
     no-op when ``is_draft_worker=True`` or ``torch.autograd._profiler_enabled()``
     is False, otherwise appends ``#num_reqs=N,request_id1=...,...#``.
Upstream PR or issue: none (TraceMe metadata syntax for TPU profiling).
Remove when: core exposes a platform hook for step-span metadata.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

import torch

# Delimiters that would corrupt ``Name#k1=v1,k2=v2#`` TraceMe metadata parsing.
_METADATA_DELIMITER_RE = re.compile(r"[,#=\s]+")


def clean_req_id(req_id: Any) -> str:
    """Sanitize a value so it can be embedded safely in a TraceMe metadata block.

    Preserves SGLang ``rid`` strings (including parallel-sampling ``_0``, ``_1``
    suffixes) while replacing ``,``, ``#``, ``=``, and whitespace with ``_``.
    """
    text = str(req_id).strip()
    if not text:
        return ""
    return _METADATA_DELIMITER_RE.sub("_", text)


def extract_request_ids_for_tracing(
    rids: Sequence[Any] | None,
) -> dict[str, str]:
    """Convert ``ForwardBatch.rids`` into 1-indexed ``request_idN`` entries."""
    if not rids:
        return {}
    cleaned_ids = [
        cleaned for rid in rids if rid is not None and (cleaned := clean_req_id(rid))
    ]
    return {
        f"request_id{idx}": cleaned for idx, cleaned in enumerate(cleaned_ids, start=1)
    }


def append_span_metadata(
    span_name: str,
    metadata: Mapping[str, Any],
    *,
    sanitize: bool = True,
) -> str:
    """Append ``#k1=v1,k2=v2#`` TraceMe metadata to ``span_name``.

    If ``span_name`` already ends with a ``#...#`` metadata block, merges the
    new pairs into that trailing block so the parser sees a single block. Set
    ``sanitize=False`` when string values are already cleaned.
    """
    if not metadata:
        return span_name
    encoded = ",".join(
        f"{k}={clean_req_id(v) if sanitize and isinstance(v, str) else v}"
        for k, v in metadata.items()
    )
    if span_name.endswith("##"):
        return f"{span_name[:-1]}{encoded}#"
    if span_name.endswith("#") and "#" in span_name[:-1]:
        return f"{span_name[:-1]},{encoded}#"
    return f"{span_name}#{encoded}#"


def append_request_ids_to_step_span(
    result: str,
    forward_batch: Any,
    *args: Any,
    **kwargs: Any,
) -> str:
    """AFTER hook on ``build_step_span_name`` to embed active request IDs."""
    if kwargs.get("is_draft_worker", False) or not torch.autograd._profiler_enabled():
        return result
    req_ids = extract_request_ids_for_tracing(getattr(forward_batch, "rids", None))
    if not req_ids:
        return result
    return append_span_metadata(
        result, {"num_reqs": len(req_ids), **req_ids}, sanitize=False
    )
