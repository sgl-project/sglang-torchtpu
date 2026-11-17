"""TPU xprof profiler."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import msgspec

from sglang.srt.platforms import current_platform
from sglang.srt.runtime_context import get_parallel
from sglang.utils import LazyImport
from sglang_tpu._logging import get_logger
from sglang_tpu.envs import envs

logger = get_logger(__name__)
tpu_profiler = LazyImport("torch_tpu._internal.profiler", "_impl")


def start_tpu_profile(*, output_dir: Path, with_cpu: bool = True) -> _TpuTrace | None:
    """Start an xprof capture writing to ``output_dir``.

    Returns a handle on rank 0, otherwise None. Pass the handle to
    :func:`stop_tpu_profile` to end the session.
    ``with_cpu`` selects the host plane.

    Goes through _impl.start_trace rather than torch_tpu's own profile()
    context manager: profile() takes no options argument, so narrowing a
    capture to a few chips means patching the module that was renamed
    profiler_api -> _api between the 0817 and 0913 pins, while _impl.py was
    byte-identical across both.
    """
    parallel = get_parallel()
    if any(
        (parallel.tp_rank, parallel.dp_rank, parallel.pp_rank, parallel.moe_ep_rank)
    ):
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    tpu_profiler.start_trace(str(output_dir), _tpu_profile_options(with_cpu=with_cpu))
    return _TpuTrace(output_dir=output_dir)


def stop_tpu_profile(trace: _TpuTrace | None) -> None:
    """Stop an xprof session started by :func:`start_tpu_profile`."""
    if trace is None:
        return
    # Non-staged/manual stops must also capture queued device work.
    current_platform.synchronize()
    tpu_profiler.stop_trace()


class _TpuTrace(msgspec.Struct, frozen=True):
    """Handle for an in-flight capture. torch_tpu keeps the session itself."""

    output_dir: Path


def _tpu_profile_options(*, with_cpu: bool):
    """Build libtpu trace options, or None to fall back to backend defaults."""
    # torch_tpu's profile() derives these from the activity list; start_trace
    # does not, and an unset level takes the backend default (host 2, device 1,
    # python 0). python_tracer_level is the expensive one -- profile() turns it
    # on whenever CPU activity is requested.
    levels = {
        "device_tracer_level": 1,
        "host_tracer_level": 2 if with_cpu else 0,
        "python_tracer_level": 1 if envs.SGLANG_TPU_PROFILE_PYTHON_TRACER.get() else 0,
    }

    num_chips = envs.SGLANG_TPU_PROFILE_NUM_CHIPS.get()
    num_sparse_cores = envs.SGLANG_TPU_PROFILE_NUM_SPARSE_CORES.get()
    chip_limit = num_chips if num_chips > 0 else None
    sparse_core_limit = num_sparse_cores if num_sparse_cores > 0 else None
    # When tracing is narrowed to a single SparseCore, also narrow to a
    # single tile within it to reduce the number of trace events to allow
    # tracing more steps, otherwise we may run into dropping trace.
    sparse_core_tile_limit = 1 if sparse_core_limit == 1 else None

    try:
        if not dataclasses.is_dataclass(tpu_profiler.ProfileOptions):
            raise TypeError("ProfileOptions is no longer a dataclass")

        options_cls, scope = tpu_profiler.ProfileOptions, {}
        if chip_limit is not None or sparse_core_limit is not None:

            @dataclasses.dataclass
            class _ProfileOptions(tpu_profiler.ProfileOptions):
                tpu_num_chips_to_profile_per_task: int | None = None
                tpu_num_sparse_cores_to_trace: int | None = None
                tpu_num_sparse_core_tiles_to_trace: int | None = None

            options_cls = _ProfileOptions
            scope = {
                "tpu_num_chips_to_profile_per_task": chip_limit,
                "tpu_num_sparse_cores_to_trace": sparse_core_limit,
                "tpu_num_sparse_core_tiles_to_trace": sparse_core_tile_limit,
            }
        options = options_cls(**levels, **scope)
    except (AttributeError, TypeError):
        logger.warning(
            "[TPU profiler] this torch_tpu does not expose the expected profile "
            "options. Tracing at backend defaults (all chips/SparseCores)."
        )
        return None

    if chip_limit is None and sparse_core_limit is None:
        return options

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
    return options
