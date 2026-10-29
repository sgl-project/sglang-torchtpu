"""TPU profiler on the native torch.profiler API.

TPU device activity reaches torch.profiler through torch_tpu's Kineto plugin,
which registers with PyTorch's PrivateUse1ProfilerRegistry (torch >= 2.12).
One ``torch.profiler.profile`` records both the Kineto CPU trace and the TPU
xplane.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import msgspec

from sglang.srt.environ import envs


def start_tpu_profile(*, output_dir: Path, with_cpu: bool = True) -> _TpuTrace:
    """Start a torch.profiler capture with TPU device activity.

    Returns a handle. Pass it to :func:`stop_tpu_profile` to end the capture.
    ``with_cpu`` adds ProfilerActivity.CPU. The Kineto trace
    (``*.pt.trace.json.gz``) is written to ``output_dir``.
    """
    import torch

    activities = [_tpu_profiler_activity()]
    if with_cpu:
        activities.insert(0, torch.profiler.ProfilerActivity.CPU)

    output_dir.mkdir(parents=True, exist_ok=True)
    profiler = torch.profiler.profile(
        activities=activities,
        # with_stack makes the plugin set python_tracer_level=1, and also adds
        # Python stacks to the Kineto CPU trace. Off by default: it dominates
        # host busy time.
        with_stack=envs.SGLANG_TPU_PROFILE_PYTHON_TRACER.get(),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            str(output_dir), use_gzip=True
        ),
    )
    profiler.start()
    return _TpuTrace(output_dir=output_dir, profiler=profiler)


def stop_tpu_profile(trace: _TpuTrace) -> None:
    """Stop a capture started by :func:`start_tpu_profile`."""
    # The plugin synchronizes the TPU devices, then writes the xplane; the
    # Kineto trace is exported by on_trace_ready.
    trace.profiler.stop()


class _TpuTrace(msgspec.Struct, frozen=True):
    """Handle for an in-flight capture."""

    output_dir: Path
    profiler: Any


def _tpu_profiler_activity():
    import torch

    # torch_tpu adds ProfilerActivity.TPU (an alias of PrivateUse1) on import.
    activity = getattr(torch.profiler.ProfilerActivity, "TPU", None)
    if activity is None:
        raise RuntimeError(
            "torch.profiler.ProfilerActivity.TPU is not defined. TPU profiling "
            "needs torch_tpu imported and torch >= 2.12."
        )
    return activity
