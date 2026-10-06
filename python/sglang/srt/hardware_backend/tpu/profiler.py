"""TPU profiler on the native torch.profiler API.

TPU device activity reaches torch.profiler through torch_tpu's Kineto plugin,
which registers with PyTorch's PrivateUse1ProfilerRegistry (torch >= 2.12).
One ``torch.profiler.profile`` records both the Kineto CPU trace and the TPU
xplane.
"""

from __future__ import annotations

import mmap
import os
import time
from pathlib import Path
from typing import Any

import msgspec

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.tpu._logging import get_logger

logger = get_logger(__name__)

# Without a custom profiler config, torch_tpu's plugin writes the xplane to
# <base>/plugins/profile/<YYYY_MM_DD_HH_MM_SS>/<hostname>.xplane.pb, where
# <base> is its output-dir env var if set, else $TMPDIR, else /tmp. The env var
# is TPU_PROFILER_OUTPUT_DIR in the plugin source; both names are checked.
_XPLANE_OUTPUT_DIR_ENV_VARS = (
    "TPU_PROFILER_OUTPUT_DIR",
    "TORCH_TPU_PROFILER_OUTPUT_DIR",
)
_TPU_DEVICE_PLANE_PREFIX = b"/device:TPU:"
# The xplane the previous capture in this process reported.
_last_xplane: Path | None = None


def start_tpu_profile(*, output_dir: Path, with_cpu: bool = True) -> _TpuTrace:
    """Start a torch.profiler capture with TPU device activity.

    Returns a handle. Pass it to :func:`stop_tpu_profile` to end the capture.
    ``with_cpu`` adds ProfilerActivity.CPU. The Kineto trace
    (``*.pt.trace.json.gz``) is written to ``output_dir``; the TPU xplane goes
    where torch_tpu's plugin puts it (see :func:`_xplane_base_dirs`).
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
    started_at = time.time()
    profiler.start()
    return _TpuTrace(output_dir=output_dir, profiler=profiler, started_at=started_at)


def stop_tpu_profile(trace: _TpuTrace) -> None:
    """Stop a capture started by :func:`start_tpu_profile`."""
    # The plugin synchronizes the TPU devices, then writes the xplane; the
    # Kineto trace is exported by on_trace_ready.
    trace.profiler.stop()
    try:
        _check_xplane(trace.started_at)
    except Exception:  # a diagnostic must never fail /stop_profile
        logger.warning("[TPU profiler] could not check the xplane", exc_info=True)


class _TpuTrace(msgspec.Struct, frozen=True):
    """Handle for an in-flight capture."""

    output_dir: Path
    profiler: Any
    started_at: float


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


def _xplane_base_dirs() -> list[Path]:
    """Candidate base directories for the plugin's xplane output."""
    candidates = [os.environ.get(name) for name in _XPLANE_OUTPUT_DIR_ENV_VARS]
    candidates.append(os.environ.get("TMPDIR") or "/tmp")
    base_dirs: list[Path] = []
    for candidate in candidates:
        if candidate and Path(candidate) not in base_dirs:
            base_dirs.append(Path(candidate))
    return base_dirs


def _find_xplane(started_at: float) -> Path | None:
    """Newest xplane written by the plugin since ``started_at``, if any.

    Ranks on one host share the plugin's output directory, so with several
    ranks this may be another rank's file.
    """
    newest: tuple[float, Path] | None = None
    for base_dir in _xplane_base_dirs():
        for path in base_dir.glob("plugins/profile/*/*.xplane.pb"):
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue
            if mtime >= started_at and (newest is None or mtime > newest[0]):
                newest = (mtime, path)
    return newest[1] if newest else None


def _check_xplane(started_at: float) -> None:
    """Log where the xplane went, or an error if the capture came back empty."""
    global _last_xplane
    xplane = _find_xplane(started_at)
    if xplane is not None and xplane == _last_xplane:
        # The file name only has the host and a 1-second timestamp directory,
        # e.g. profile_by_stage's EXTEND and DECODE stopping in the same second.
        logger.warning(
            "[TPU profiler] %s was rewritten by this capture: the previous "
            "capture's xplane, which had the same path, is lost.",
            xplane,
        )
    _last_xplane = xplane
    if xplane is None:
        logger.error(
            "[TPU profiler] no .xplane.pb was written under %s. The capture "
            "has no TPU device activity; check that torch_tpu's Kineto plugin "
            "is registered.",
            " or ".join(str(d / "plugins" / "profile") for d in _xplane_base_dirs()),
        )
        return
    try:
        has_device_events = _has_tpu_device_events(xplane)
    except (OSError, ValueError, IndexError) as e:
        logger.warning("[TPU profiler] could not inspect %s: %s", xplane, e)
        has_device_events = True
    if not has_device_events:
        logger.error(
            "[TPU profiler] %s has no /device:TPU plane with events. The "
            "capture recorded no TPU device activity.",
            xplane,
        )
        return
    logger.info("[TPU profiler] TPU xplane written to %s", xplane)


def _has_tpu_device_events(path: Path) -> bool:
    """Whether an XSpace file has a ``/device:TPU:<n>`` plane with events.

    Walks the protobuf wire format (XSpace.planes = 1, XPlane.name = 2,
    XPlane.lines = 3, XLine.events = 4) instead of depending on the xplane
    protos. Event payloads are skipped, not decoded.
    """
    with open(path, "rb") as f:
        if os.fstat(f.fileno()).st_size == 0:
            return False
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as buf:
            for field, start, end in _length_delimited_fields(buf, 0, len(buf)):
                if field == 1 and _is_tpu_plane_with_events(buf, start, end):
                    return True
    return False


def _is_tpu_plane_with_events(buf, start: int, end: int) -> bool:
    name = b""
    has_events = False
    for field, field_start, field_end in _length_delimited_fields(buf, start, end):
        if field == 2:
            name = buf[field_start:field_end]
        elif field == 3 and not has_events:
            has_events = any(
                line_field == 4
                for line_field, _, _ in _length_delimited_fields(
                    buf, field_start, field_end
                )
            )
    return name.startswith(_TPU_DEVICE_PLANE_PREFIX) and has_events


def _length_delimited_fields(buf, start: int, end: int):
    """Yield (field number, start, end) of each length-delimited field."""
    pos = start
    while pos < end:
        key, pos = _read_varint(buf, pos)
        wire_type = key & 0x7
        if wire_type == 0:
            _, pos = _read_varint(buf, pos)
        elif wire_type == 1:
            pos += 8
        elif wire_type == 5:
            pos += 4
        elif wire_type == 2:
            length, pos = _read_varint(buf, pos)
            if pos + length > end:
                raise ValueError("truncated protobuf field")
            yield key >> 3, pos, pos + length
            pos += length
        else:
            raise ValueError(f"unsupported protobuf wire type {wire_type}")
    if pos != end:
        raise ValueError("truncated protobuf message")


def _read_varint(buf, pos: int) -> tuple[int, int]:
    result = 0
    for shift in range(0, 64, 7):
        byte = buf[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, pos
    raise ValueError("malformed protobuf varint")
