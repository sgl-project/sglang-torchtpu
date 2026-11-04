# SPDX-License-Identifier: Apache-2.0
"""Input-logprob chunks on the chunked-prefill grid (SGLANG_TPU_LOGPROB_GRID_ROWS).

Registered only when the variable is above 0; unset, no hook exists and core
keeps its uniform chunk split.

Why: the eager input-logprob epilogue (lm_head on power-of-two row buckets, the
     logsumexp normalizer) rounds differently for different row counts. A lone
     scoring request whose scored rows cross a grid boundary is split there by
     grid-sized prefill chunks (2048) but not by larger ones (4096), so its
     prompt logprobs moved by ulps between the two chunk sizes.
How: AFTER LogitsMetadata.from_forward_batch: when the forward computes input
     logprobs for exactly one request, record its grid offsets in the generic
     LogitsMetadata.input_logprob_chunk_cuts_cpu; InputLogprobProcessor splits
     there before chunking. With several requests the scheduler shares one
     prefill budget, so their grid-sized forwards do not start on each
     request's own grid multiples: those forwards get no cuts (the env-off
     chunking).
Upstream PR or issue: not filed; core's side is the optional
     LogitsMetadata.input_logprob_chunk_cuts_cpu field.
Remove when: core takes input-logprob chunk cuts from a platform method.
"""

from __future__ import annotations

from typing import Any, List, Optional

from sglang_tpu.envs import envs


def logprob_grid_cuts(
    start_lens: Optional[List[int]], extend_lens: Optional[List[int]], grid: int
) -> List[int]:
    """Offsets into a forward's pruned rows where the input-logprob chunking must
    cut: every multiple of ``grid`` from the request's first row in the forward.

    Only a forward holding one request is cut. Its pruned rows are
    ``[start, extend)`` (``LogitsProcessor._get_pruned_states``; with
    ``start == extend`` it keeps the last row alone, which has nothing to cut).
    """
    if grid <= 0 or not extend_lens or len(extend_lens) != 1:
        return []
    start, extend = start_lens[0], extend_lens[0]
    return list(range((start // grid + 1) * grid - start, extend - start, grid))


def record_logprob_grid_cuts(metadata: Any, *args: Any, **kwargs: Any) -> None:
    """AFTER LogitsMetadata.from_forward_batch: the cuts of a forward that
    computes input logprobs. The field stays None when there is no cut."""
    if not metadata.extend_return_logprob:
        return None
    cuts = logprob_grid_cuts(
        metadata.extend_logprob_start_lens_cpu,
        metadata.extend_seq_lens_cpu,
        envs.SGLANG_TPU_LOGPROB_GRID_ROWS.get(),
    )
    if cuts:
        metadata.input_logprob_chunk_cuts_cpu = cuts
    return None


def hook_specs():
    if envs.SGLANG_TPU_LOGPROB_GRID_ROWS.get() <= 0:
        return []

    from sglang.srt.plugins.hook_registry import HookType

    return [
        (
            "sglang.srt.layers.logits_processor.LogitsMetadata.from_forward_batch",
            record_logprob_grid_cuts,
            HookType.AFTER,
        )
    ]
