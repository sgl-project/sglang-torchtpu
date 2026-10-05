"""TPU xprof profiler."""

from __future__ import annotations

from pathlib import Path

import msgspec

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.tpu._logging import get_logger

logger = get_logger(__name__)


def start_tpu_profile(*, output_dir: Path, with_cpu: bool = True) -> _TpuTrace:
    """Start an xprof capture writing to ``output_dir``.

    Returns a handle. Pass it to :func:`stop_tpu_profile` to end the session.
    ``with_cpu`` selects the host plane.

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
    return _TpuTrace(output_dir=output_dir)


def stop_tpu_profile(trace: _TpuTrace) -> None:
    """Stop an xprof session started by :func:`start_tpu_profile`."""
    del trace  # the session is process-global; only one may run at a time
    from torch_tpu._internal.profiler import _impl as tpu_profiler

    tpu_profiler.stop_trace()


class _TpuTrace(msgspec.Struct, frozen=True):
    """Handle for an in-flight capture. torch_tpu keeps the session itself."""

    output_dir: Path


def _tpu_profile_options(*, with_cpu: bool):
    """Build libtpu trace options, or None to fall back to backend defaults."""
    from torch_tpu._internal.profiler import _impl as tpu_profiler

    # torch_tpu's profile() derives these from the activity list; start_trace
    # does not, and an unset level takes the backend default (host 2, device 1,
    # python 0). python_tracer_level is the expensive one -- profile() turns it
    # on whenever CPU activity is requested.
    #
    # These three are the only keys the trace backend reads: its pybind entry
    # point does a contains/get/cast<int> for each by name, then hands the proto
    # to tsl::ProfilerSession. Anything else in the dict is dropped in C++, so
    # there is no way to narrow a capture to fewer chips or SparseCores from
    # here. That needs custom_profiler_config via TpuProfilerConfig, which is
    # the native torch.profiler route -- available once the TPU stack is on
    # torch >= 2.12.
    try:
        return tpu_profiler.ProfileOptions(
            device_tracer_level=1,
            host_tracer_level=2 if with_cpu else 0,
            python_tracer_level=(
                1 if envs.SGLANG_TPU_PROFILE_PYTHON_TRACER.get() else 0
            ),
        )
    except (AttributeError, TypeError):
        logger.warning(
            "[TPU profiler] this torch_tpu does not expose the expected profile "
            "options. Tracing at backend defaults."
        )
        return None
