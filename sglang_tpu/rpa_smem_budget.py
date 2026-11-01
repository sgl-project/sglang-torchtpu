# SPDX-License-Identifier: Apache-2.0
"""Scoped SMEM capacity checks for TPU Pallas ragged-paged-attention backends."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from sglang.srt.runtime_context import (
    get_disagg,
    get_exec,
    get_parallel,
    get_schedule,
)
from sglang_tpu._logging import get_logger

if TYPE_CHECKING:
    from sglang_tpu.compilation.decoder_compile_policy import CompiledAttnFamily

logger = get_logger(__name__)

# Scoped SMEM available to one Pallas kernel invocation on TPU (1 MiB).
SCOPED_SMEM_BYTES = 1 << 20


def check_compiled_smem_budget(
    *,
    context_len: int,
    pages_per_seq: int,
    exact_fn: Callable[[int, int], int] | None,
    compiled: bool,
    is_generation: bool,
) -> None:
    """Fail fast on compiled SMEM overflow; warn on eager/non-compiled caps."""
    # Deferred: decoder_compile_policy imports model_runner at module load, and
    # dp_attention imports triton.
    from sglang.srt.layers.dp_attention import deployment_attn_dp_size
    from sglang_tpu.compilation.decoder_compile_policy import (
        attention_dp_enabled,
        compiled_attention_row_families,
    )

    families = (
        compiled_attention_row_families(context_len=context_len)
        if compiled
        else _scheduler_attention_row_families(is_generation=is_generation)
    )
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
    dp_enabled = attention_dp_enabled()
    attn_dp = (
        max(deployment_attn_dp_size(), parallel.attn_dp_size, 1) if dp_enabled else 1
    )
    attn_tp = max(parallel.attn_tp_size, 1) if (compiled and dp_enabled) else 1
    has_tpu_compile_bs = bool(get_exec().graph.tpu_compile_bs)

    worst_families = [
        min(
            (f for f in overflows if f.limit_flag == flag),
            key=lambda f: (max_seqs // f.rows_per_req, -f.rows),
        )
        for flag in dict.fromkeys(f.limit_flag for f in overflows)
    ]

    messages = []
    budget_str = _format_smem_budget()
    for family in worst_families:
        smem_bytes = smem_fn(family.rows, pages_per_seq)
        need_str = _format_smem_mib(smem_bytes)
        remedy = _smem_overflow_remedy(
            limit_flag=family.limit_flag,
            raw_cap=max_seqs // family.rows_per_req,
            attn_dp=attn_dp,
            attn_tp=attn_tp,
            has_tpu_compile_bs=has_tpu_compile_bs,
            compiled=compiled,
        )
        if compiled and exact_fn is not None:
            messages.append(
                f"RPA kernel for {family.rows} sequences x {pages_per_seq} "
                f"pages needs {need_str} of SMEM, above the {budget_str} "
                f"scoped budget; {remedy}"
            )
        elif compiled:
            logger.warning(
                "RPA page table for %d sequences x %d pages needs %s of SMEM, "
                "above the %s scoped budget (the kernel compile may fail); %s",
                family.rows,
                pages_per_seq,
                need_str,
                budget_str,
                remedy,
            )
        else:
            label = "RPA kernel" if exact_fn is not None else "RPA page table"
            logger.warning(
                "%s for %d sequences x %d pages needs %s of SMEM, above the "
                "%s scoped budget (batches at this concurrency may fail in "
                "Pallas compilation); %s",
                label,
                family.rows,
                pages_per_seq,
                need_str,
                budget_str,
                remedy,
            )

    if messages:
        raise ValueError(" ".join(messages))


def format_cp_smem_overflow_error(
    *,
    num_rows: int,
    pages_per_seq: int,
    smem_bytes: int,
    exact_fn: Callable[[int, int], int] | None,
) -> str:
    from sglang_tpu.compilation.decoder_compile_policy import cp_limit_flag

    smem_fn = exact_fn or page_table_smem_bytes
    max_seqs = _max_smem_seqs(estimate_fn=smem_fn, pages_per_seq=pages_per_seq)
    remedy = _smem_overflow_remedy(
        limit_flag=cp_limit_flag(),
        raw_cap=max_seqs // 2,
        attn_dp=1,
        attn_tp=1,
        has_tpu_compile_bs=bool(get_exec().graph.tpu_compile_bs),
        compiled=True,
    )
    label = "RPA kernel" if exact_fn is not None else "RPA page table"
    return (
        f"{label} for {num_rows} context-parallel rows x {pages_per_seq} pages "
        f"needs {_format_smem_mib(smem_bytes)} of SMEM, above the "
        f"{_format_smem_budget()} scoped budget; {remedy}"
    )


def _scheduler_attention_row_families(
    *, is_generation: bool
) -> tuple[CompiledAttnFamily, ...]:
    from sglang.srt.layers.dp_attention import deployment_attn_dp_size
    from sglang_tpu.compilation.compiled_forward_shim import (
        DEFAULT_TPU_COMPILE_MAX_BS,
    )
    from sglang_tpu.compilation.decoder_compile_policy import (
        CompiledAttnFamily,
        attention_dp_enabled,
    )

    schedule = get_schedule()
    if schedule.max_running_requests == 0:
        return ()
    mrr = (
        schedule.max_running_requests
        if schedule.max_running_requests is not None
        else DEFAULT_TPU_COMPILE_MAX_BS
    )
    dp_enabled = attention_dp_enabled()
    attn_dp = (
        max(deployment_attn_dp_size(), get_parallel().attn_dp_size, 1)
        if dp_enabled
        else 1
    )
    per_worker_mrr = max(mrr // attn_dp, 1)
    prefill_only = not is_generation or (
        get_disagg().disaggregation_mode == "prefill"
        and not schedule.enable_mixed_chunk
    )
    pmr = schedule.prefill_max_requests
    if prefill_only and not dp_enabled and pmr is not None and pmr <= per_worker_mrr:
        return (
            CompiledAttnFamily(
                rows=pmr, rows_per_req=1, limit_flag="--prefill-max-requests"
            ),
        )
    return (
        CompiledAttnFamily(
            rows=per_worker_mrr,
            rows_per_req=1,
            limit_flag="--max-running-requests",
        ),
    )


def _format_smem_mib(smem_bytes: int) -> str:
    return f"{smem_bytes / 2**20:.2f} MiB ({smem_bytes:,} B)"


def _format_smem_budget() -> str:
    return f"{SCOPED_SMEM_BYTES // 2**20} MiB ({SCOPED_SMEM_BYTES:,} B)"


def _smem_overflow_remedy(
    *,
    limit_flag: str,
    raw_cap: int,
    attn_dp: int,
    attn_tp: int,
    has_tpu_compile_bs: bool,
    compiled: bool,
) -> str:
    if limit_flag == "--max-running-requests":
        per_worker_cap = (raw_cap // attn_tp) * attn_tp
        if per_worker_cap < 1:
            return "raise --page-size or lower --context-length."
        mrr_cap = per_worker_cap * attn_dp
        if not compiled:
            return (
                f"lower --max-running-requests to <= {mrr_cap} at this context "
                "length, or raise --page-size."
            )
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
