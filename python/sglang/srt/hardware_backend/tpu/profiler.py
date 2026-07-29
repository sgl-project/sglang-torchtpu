"""TPU xprof profiler."""

from __future__ import annotations

import dataclasses
from contextlib import AbstractContextManager
from pathlib import Path
from types import ModuleType

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.tpu._logging import get_logger

logger = get_logger(__name__)


def start_tpu_profile(
    *, output_dir: Path, with_cpu: bool = True
) -> AbstractContextManager:
    """Start an xprof capture writing to ``output_dir``.

    Returns the entered profiler. Pass it to :func:`stop_tpu_profile` to end
    the session. ``with_cpu`` adds the host-CPU activity to the device trace.
    """
    from torch_tpu._internal.profiler import profiler_api

    _install_tpu_profile_options_patch(profiler_api)

    output_dir.mkdir(parents=True, exist_ok=True)
    handler = profiler_api.xprof_trace_handler(dir_name=str(output_dir))
    activities = [profiler_api.ProfilerActivity.TPU]
    if with_cpu:
        activities.append(profiler_api.ProfilerActivity.CPU)
    profiler = profiler_api.profile(activities=activities, on_trace_ready=handler)
    profiler.__enter__()
    return profiler


def stop_tpu_profile(profiler: AbstractContextManager) -> None:
    """Stop an xprof session started by :func:`start_tpu_profile`."""
    # Exiting the context stops the xprof session.
    profiler.__exit__(None, None, None)


def _install_tpu_profile_options_patch(profiler_api: ModuleType) -> None:
    """Add libtpu trace-limit fields to torch_tpu profile options."""
    if getattr(profiler_api, "_sglang_tpu_profile_options_patch_installed", False):
        return

    num_chips = envs.SGLANG_TPU_PROFILE_NUM_CHIPS.get()
    num_sparse_cores = envs.SGLANG_TPU_PROFILE_NUM_SPARSE_CORES.get()
    chip_limit = num_chips if num_chips > 0 else None
    sparse_core_limit = num_sparse_cores if num_sparse_cores > 0 else None
    # When tracing is narrowed to a single SparseCore, also narrow to a
    # single tile within it to reduce the number of trace events to allow
    # tracing more steps, otherwise we may run into dropping trace.
    sparse_core_tile_limit = 1 if sparse_core_limit == 1 else None
    if chip_limit is None and sparse_core_limit is None:
        profiler_api._sglang_tpu_profile_options_patch_installed = True
        return

    try:
        from torch_tpu._internal.profiler import _impl as _tpu_impl

        orig_get_options = profiler_api._get_profile_options
        if not dataclasses.is_dataclass(_tpu_impl.ProfileOptions):
            raise TypeError("ProfileOptions is no longer a dataclass")

        @dataclasses.dataclass
        class _ProfileOptions(_tpu_impl.ProfileOptions):
            tpu_num_chips_to_profile_per_task: int | None = None
            tpu_num_sparse_cores_to_trace: int | None = None
            tpu_num_sparse_core_tiles_to_trace: int | None = None

    except (ImportError, AttributeError, TypeError):
        logger.warning(
            "[TPU profiler] this torch_tpu does not expose the expected "
            "profile-options hook. Tracing all chips/SparseCores "
            "(traces may be large)."
        )
        profiler_api._sglang_tpu_profile_options_patch_installed = True
        return

    def _get_options(
        activities: list[profiler_api.ProfilerActivity],
    ) -> _tpu_impl.ProfileOptions:
        base = orig_get_options(activities)
        try:
            # Merge into a dict first. A future ProfileOptions that grows one
            # of these fields would make **asdict raise on duplicate keywords.
            fields = dataclasses.asdict(base)
            fields.update(
                tpu_num_chips_to_profile_per_task=chip_limit,
                tpu_num_sparse_cores_to_trace=sparse_core_limit,
                tpu_num_sparse_core_tiles_to_trace=sparse_core_tile_limit,
            )
            return _ProfileOptions(**fields)
        except Exception:
            # Restore the original hook so the failure is logged once.
            profiler_api._get_profile_options = orig_get_options
            logger.warning(
                "[TPU profiler] failed to extend ProfileOptions. Tracing all "
                "chips/SparseCores (traces may be large).",
                exc_info=True,
            )
            return base

    profiler_api._get_profile_options = _get_options
    profiler_api._sglang_tpu_profile_options_patch_installed = True
    # The tile cap is only derived when tracing a single SparseCore; omit the
    # clause otherwise so the log doesn't read "(all tile(s))" for a branch
    # that was never entered.
    tile_clause = (
        f" ({sparse_core_tile_limit} tile(s))"
        if sparse_core_tile_limit is not None
        else ""
    )
    logger.info(
        "[TPU profiler] limiting trace to %s chip(s), %s SparseCore(s)%s per task",
        chip_limit or "all",
        sparse_core_limit or "all",
        tile_clause,
    )
