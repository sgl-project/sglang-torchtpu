"""TPU xprof profiler."""

from __future__ import annotations

import dataclasses
import os
import shutil
from pathlib import Path
from typing import Optional

import msgspec

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.tpu._logging import get_logger
from sglang.srt.hardware_backend.tpu.xplane import rebase_to_wall_clock

logger = get_logger(__name__)


def start_tpu_profile(
    *,
    output_dir: Path,
    with_cpu: bool = True,
    session_dir: Optional[Path] = None,
    file_suffix: str = "",
) -> _TpuTrace:
    """Start an xprof capture writing to ``output_dir``.

    Returns a handle. Pass it to :func:`stop_tpu_profile` to end the session.
    ``with_cpu`` selects the host plane.

    With ``session_dir`` set, ``output_dir`` is only a staging area: on stop the
    trace moves into ``session_dir`` as ``<host><file_suffix>.xplane.pb``,
    rebased onto the wall clock. XProf loads every file in a session folder as
    one multi-host trace, so an engine's ranks publishing into the same
    ``session_dir`` with distinct suffixes show up on one timeline.

    Goes through _impl.start_trace rather than torch_tpu's own profile()
    context manager: profile() takes no options argument, so narrowing a
    capture to a few chips means patching the module that was renamed
    profiler_api -> _api between the 0817 and 0913 pins, while _impl.py was
    byte-identical across both.
    """
    # start_trace/stop_trace live in _impl; the package root re-exports only
    # profile, xprof_trace_handler and the remote-capture server.
    from torch_tpu._internal.profiler import _impl as tpu_profiler

    output_dir.mkdir(parents=True, exist_ok=True)
    tpu_profiler.start_trace(str(output_dir), _tpu_profile_options(with_cpu=with_cpu))
    return _TpuTrace(
        output_dir=output_dir, session_dir=session_dir, file_suffix=file_suffix
    )


def stop_tpu_profile(trace: Optional[_TpuTrace]) -> Optional[Path]:
    """Stop an xprof session started by :func:`start_tpu_profile`.

    Returns the published trace file, or None when the capture was not
    published (no ``session_dir``, or publishing failed and the trace was left
    in its staging directory).
    """
    from torch_tpu._internal.profiler import _impl as tpu_profiler

    # The session is process-global; only one may run at a time. stop_trace
    # returns once the trace file is written.
    tpu_profiler.stop_trace()
    if trace is None or trace.session_dir is None:
        return None
    return _publish_trace(trace)


def _publish_trace(trace: _TpuTrace) -> Optional[Path]:
    """Move a staged trace into its session folder, rebased onto wall clock."""
    staged = sorted(trace.output_dir.glob("plugins/profile/*/*.xplane.pb"))
    if len(staged) != 1:
        logger.warning(
            "[TPU profiler] expected one trace under %s, found %d; leaving it "
            "unpublished.",
            trace.output_dir,
            len(staged),
        )
        return None

    src = staged[0]
    # libtpu names the file after the host, which every rank on it shares.
    host = src.name.removesuffix(".xplane.pb")
    dst = trace.session_dir / f"{host}{trace.file_suffix}.xplane.pb"
    try:
        data, start_ns = rebase_to_wall_clock(src.read_bytes())
        if start_ns is None:
            logger.warning(
                "[TPU profiler] %s has no profile_start_time; publishing it "
                "unrebased, so it will not line up with other ranks.",
                src,
            )
        trace.session_dir.mkdir(parents=True, exist_ok=True)
        # XProf may be watching the folder; never let it see a partial file.
        tmp = dst.with_name(f".{dst.name}.tmp")
        tmp.write_bytes(data)
        os.replace(tmp, dst)
    except (OSError, ValueError, IndexError) as e:
        logger.warning(
            "[TPU profiler] could not publish %s to %s (%s); the per-rank trace "
            "is still at %s.",
            src,
            trace.session_dir,
            e,
            trace.output_dir,
        )
        return None

    shutil.rmtree(trace.output_dir, ignore_errors=True)
    # Ranks share the staging parent; whichever publishes last removes it.
    try:
        trace.output_dir.parent.rmdir()
    except OSError:
        pass
    return dst


class _TpuTrace(msgspec.Struct, frozen=True):
    """Handle for an in-flight capture. torch_tpu keeps the session itself."""

    output_dir: Path
    session_dir: Optional[Path] = None
    file_suffix: str = ""


def _tpu_profile_options(*, with_cpu: bool):
    """Build libtpu trace options, or None to fall back to backend defaults."""
    from torch_tpu._internal.profiler import _impl as tpu_profiler

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
