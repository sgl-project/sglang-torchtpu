"""TPU xprof profiler."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import msgspec
import torch

from sglang_tpu._logging import get_logger
from sglang_tpu.envs import envs

logger = get_logger(__name__)


def start_tpu_profile(*, output_dir: Path, with_cpu: bool = True) -> _TpuTrace:
    """Start an xprof capture writing to ``output_dir``.

    Returns a handle. Pass it to :func:`stop_tpu_profile` to end the session.
    ``with_cpu`` selects the host plane. The xplane lands under
    ``output_dir/plugins/profile/<ts>/<host>.xplane.pb``.

    Goes through torch.profiler with a TpuProfilerConfig: torch_tpu's
    _impl.start_trace forwards only the three tracer levels to libtpu, so the
    trace mode and the chip/SparseCore scope would be dropped.
    """
    from torch_tpu._internal.profiler import TpuProfilerConfig

    output_dir.mkdir(parents=True, exist_ok=True)
    levels = _tracer_levels(with_cpu=with_cpu)
    experimental_options = _experimental_options()
    activities = [torch.profiler.ProfilerActivity.PrivateUse1]
    if with_cpu:
        activities.append(torch.profiler.ProfilerActivity.CPU)
    profile = torch.profiler.profile(
        activities=activities,
        experimental_config=TpuProfilerConfig(
            run_dir=output_dir,
            experimental_options=experimental_options,
            **levels,
        ),
    )
    profile.__enter__()
    logger.info(
        "[TPU profiler] tracing to %s with %s", output_dir, experimental_options
    )
    return _TpuTrace(output_dir=output_dir, profile=profile)


def stop_tpu_profile(trace: _TpuTrace) -> None:
    """Stop an xprof session started by :func:`start_tpu_profile`."""
    # torch_tpu's Kineto session drains every TPU device and writes the xplane here.
    trace.profile.__exit__(None, None, None)


class _TpuTrace(msgspec.Struct, frozen=True):
    """Handle for an in-flight capture."""

    output_dir: Path
    profile: Any  # torch.profiler.profile


def _tracer_levels(*, with_cpu: bool) -> dict[str, int]:
    # python_tracer_level is the expensive one; torch_tpu's own profile()
    # turns it on whenever CPU activity is requested.
    return {
        "device_tracer_level": 1,
        "host_tracer_level": 2 if with_cpu else 0,
        "python_tracer_level": 1 if envs.SGLANG_TPU_PROFILE_PYTHON_TRACER.get() else 0,
    }


def _experimental_options() -> dict[str, int | str]:
    """libtpu advanced_configuration; keys libtpu does not know fail the start."""
    options: dict[str, int | str] = {}
    trace_mode = envs.SGLANG_TPU_PROFILE_TRACE_MODE.get()
    if trace_mode:
        options["tpu_trace_mode"] = trace_mode
    num_chips = envs.SGLANG_TPU_PROFILE_NUM_CHIPS.get()
    if num_chips > 0:
        options["tpu_num_chips_to_profile_per_task"] = num_chips
    num_sparse_cores = envs.SGLANG_TPU_PROFILE_NUM_SPARSE_CORES.get()
    if num_sparse_cores > 0:
        options["tpu_num_sparse_cores_to_trace"] = num_sparse_cores
    # One SparseCore's tiles still fill most of a capture; one tile leaves
    # room for more steps under the 2 GB trace limit.
    if num_sparse_cores == 1:
        options["tpu_num_sparse_core_tiles_to_trace"] = 1
    return options
