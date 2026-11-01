# SPDX-License-Identifier: Apache-2.0
"""Scoped SMEM capacity checks for TPU Pallas ragged-paged-attention backends."""

from __future__ import annotations

from collections.abc import Callable

from sglang.srt.runtime_context import get_exec, get_parallel
from sglang_tpu._logging import get_logger

logger = get_logger(__name__)

# Scoped SMEM available to one Pallas kernel invocation on TPU (1 MiB).
SCOPED_SMEM_BYTES = 1 << 20


def check_compiled_smem_budget(
    *,
    context_len: int,
    pages_per_seq: int,
    exact_fn: Callable[[int, int], int] | None,
) -> None:
    """Fail fast (or warn without an exact model) when compiled shapes overflow SMEM."""
    # Deferred: decoder_compile_policy imports model_runner at module load.
    from sglang_tpu.compilation.decoder_compile_policy import (
        attention_dp_enabled,
        compiled_attention_row_families,
    )

    families = compiled_attention_row_families(context_len=context_len)
    if not families:
        return

    smem_fn = exact_fn or page_table_smem_bytes
    overflows = [
        family
        for family in families
        if smem_fn(family.rows, pages_per_seq) > SCOPED_SMEM_BYTES
    ]
    if not overflows:
        return

    max_seqs = _max_smem_seqs(estimate_fn=smem_fn, pages_per_seq=pages_per_seq)
    parallel = get_parallel()
    attn_dp = max(parallel.attn_dp_size, 1)
    attn_tp = max(parallel.attn_tp_size, 1) if attention_dp_enabled() else 1
    has_tpu_compile_bs = bool(get_exec().graph.tpu_compile_bs)

    worst_families = [
        min(
            (f for f in overflows if f.limit_flag == flag),
            key=lambda f: (max_seqs // f.rows_per_req, -f.rows),
        )
        for flag in dict.fromkeys(f.limit_flag for f in overflows)
    ]

    messages = []
    for family in worst_families:
        smem_bytes = smem_fn(family.rows, pages_per_seq)
        remedy = _smem_overflow_remedy(
            limit_flag=family.limit_flag,
            raw_cap=max_seqs // family.rows_per_req,
            attn_dp=attn_dp,
            attn_tp=attn_tp,
            has_tpu_compile_bs=has_tpu_compile_bs,
        )
        if exact_fn is not None:
            messages.append(
                f"RPA kernel for {family.rows} sequences x {pages_per_seq} "
                f"pages needs {smem_bytes / 2**20:.2f} MiB of SMEM, above "
                f"the {SCOPED_SMEM_BYTES / 2**20:.0f} MiB scoped budget; "
                f"{remedy}"
            )
        else:
            logger.warning(
                "RPA page table for %d sequences x %d pages needs %.2f MiB "
                "of SMEM, above the %.0f MiB scoped budget (the kernel "
                "compile may fail); %s",
                family.rows,
                pages_per_seq,
                smem_bytes / 2**20,
                SCOPED_SMEM_BYTES / 2**20,
                remedy,
            )

    if messages:
        raise ValueError(" ".join(messages))


def _smem_overflow_remedy(
    *,
    limit_flag: str,
    raw_cap: int,
    attn_dp: int,
    attn_tp: int,
    has_tpu_compile_bs: bool,
) -> str:
    if limit_flag == "--max-running-requests":
        per_worker_cap = (raw_cap // attn_tp) * attn_tp
        if per_worker_cap < 1:
            return "raise --page-size or lower --context-length."
        mrr_cap = per_worker_cap * attn_dp
        bs_prefix = (
            f"drop any --tpu-compile-bs entries above {per_worker_cap} and "
            if has_tpu_compile_bs
            else ""
        )
        cap_clause = (
            f"lower --max-running-requests to <= {mrr_cap} "
            f"(and --torch-compile-max-bs to <= {per_worker_cap} if set)"
            if mrr_cap != per_worker_cap
            else (
                "lower --max-running-requests (and "
                f"--torch-compile-max-bs if set) to <= {per_worker_cap}"
            )
        )
        return f"{bs_prefix}{cap_clause} at this context length, or raise --page-size."

    if raw_cap < 1:
        return "raise --page-size or lower --context-length."
    return (
        f"lower --prefill-max-requests to <= {raw_cap} at "
        "this context length, or raise --page-size."
    )


def _max_smem_seqs(
    *,
    estimate_fn: Callable[[int, int], int],
    pages_per_seq: int,
) -> int:
    # Returns 0 when even a single sequence row exceeds SCOPED_SMEM_BYTES.
    if pages_per_seq <= 0:
        raise ValueError(f"pages_per_seq must be positive, got {pages_per_seq}.")
    if estimate_fn(1, pages_per_seq) > SCOPED_SMEM_BYTES:
        return 0
    lo, hi = 1, max(1, SCOPED_SMEM_BYTES // (4 * pages_per_seq))
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if estimate_fn(mid, pages_per_seq) <= SCOPED_SMEM_BYTES:
            lo = mid
        else:
            hi = mid - 1
    return lo


def page_table_smem_bytes(max_num_seqs: int, pages_per_seq: int) -> int:
    # Scalar-prefetch int32[max_num_seqs * pages_per_seq], padded to 128 lanes.
    return align_to_128(max_num_seqs * pages_per_seq) * 4


def align_to_128(n: int) -> int:
    return ((n + 127) // 128) * 128
