"""TPU profiler on the native torch.profiler API.

TPU device activity reaches torch.profiler through torch_tpu's Kineto plugin,
which registers with PyTorch's PrivateUse1ProfilerRegistry (torch >= 2.12).
One ``torch.profiler.profile`` records both the Kineto CPU trace and the TPU
xplane. torch_tpu's ``TpuProfilerConfig``, passed as ``experimental_config``,
carries the TPU options: tracer levels, experimental (libtpu) options, and the
``run_dir`` that puts the xplane next to the Kineto trace.

Options come from, lowest to highest precedence:

1. Built-in defaults and the SGLANG_TPU_PROFILE_PYTHON_TRACER,
   SGLANG_TPU_PROFILE_NUM_CHIPS and SGLANG_TPU_PROFILE_NUM_SPARSE_CORES env
   vars.
2. SGLANG_TPU_PROFILE_OPTIONS, a ``key:value;key:value`` string.
3. A /start_profile ``profile_prefix`` in the same ``key:value;...`` form
   (see :func:`resolve_tpu_profile_options`).

``host_tracer_level``, ``device_tracer_level``, ``python_tracer_level`` and
``check_experimental_options`` are TpuProfilerConfig arguments. Any other key
(for example ``tpu_trace_mode:TRACE_COMPUTE`` or ``tpu_perf_counters:true``)
is passed to libtpu as an experimental option. For the chip / SparseCore
narrowing options (``tpu_num_chips_to_profile_per_task``,
``tpu_num_sparse_cores_to_trace``, ``tpu_num_sparse_core_tiles_to_trace``) a
value <= 0 means trace all, and drops the option.
"""

from __future__ import annotations

import glob
import math
import mmap
import os
import re
import time
from pathlib import Path
from typing import Any

import msgspec

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.tpu._logging import get_logger

logger = get_logger(__name__)

# torch_tpu's plugin writes the xplane to
# <base>/plugins/profile/<YYYY_MM_DD_HH_MM_SS>/<host>[_<worker_rank>].xplane.pb,
# where <base> is this env var if set, else the config's run_dir.
_XPLANE_OUTPUT_DIR_ENV_VAR = "TPU_PROFILER_OUTPUT_DIR"
_TPU_DEVICE_PLANE_PREFIX = b"/device:TPU:"

# Option keys that are TpuProfilerConfig arguments. Every other key goes to
# experimental_options.
_LEVEL_KEYS = ("host_tracer_level", "device_tracer_level", "python_tracer_level")
_CHECK_KEY = "check_experimental_options"
# Experimental options that narrow what is traced. A value <= 0 means trace
# all, and removes the option.
_NUM_CHIPS_KEY = "tpu_num_chips_to_profile_per_task"
_NUM_SPARSE_CORES_KEY = "tpu_num_sparse_cores_to_trace"
_NUM_SPARSE_CORE_TILES_KEY = "tpu_num_sparse_core_tiles_to_trace"
_NARROWING_KEYS = (_NUM_CHIPS_KEY, _NUM_SPARSE_CORES_KEY, _NUM_SPARSE_CORE_TILES_KEY)
# Set by SGLang for every capture, so not user options.
_RESERVED_KEYS = frozenset({"run_dir", "worker_rank"})
_OPTION_PATTERN = re.compile(r"^([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*(.+)$")
_ALLOWED_VALUE_CHARS = re.compile(r"^[a-zA-Z0-9_./,:\-\s]*$")
# The plugin splits its config string on unquoted commas and does not quote
# run_dir, so these characters break it.
_UNSUPPORTED_RUN_DIR_CHARS = (",", '"', "\\")


class TpuProfileOptions(msgspec.Struct, frozen=True):
    """TPU options for one capture. See the module docstring."""

    # None: 2 when CPU activity is captured, 0 (TPU-only) otherwise.
    host_tracer_level: int | None = None
    device_tracer_level: int = 1
    python_tracer_level: int = 0
    check_experimental_options: bool = True
    experimental_options: dict[str, int | bool | str] = msgspec.field(
        default_factory=dict
    )


def resolve_tpu_profile_options(
    profile_prefix: str | None,
) -> tuple[str | None, TpuProfileOptions]:
    """Split a /start_profile ``profile_prefix`` into a trace name and options.

    A ``profile_prefix`` containing ``:`` or ``;`` is an options string, e.g.
    ``"host_tracer_level:0;tpu_trace_mode:TRACE_COMPUTE"``, and the returned
    name is None. Any other prefix is returned as the name, with the default
    options. Raises ValueError for malformed options.
    """
    values = _default_option_values()
    env_options = envs.SGLANG_TPU_PROFILE_OPTIONS.get()
    if env_options:
        values.update(_parse_options(env_options, "SGLANG_TPU_PROFILE_OPTIONS"))
    name = profile_prefix or None
    if profile_prefix and (":" in profile_prefix or ";" in profile_prefix):
        values.update(_parse_options(profile_prefix, "profile_prefix"))
        name = None
    # With a single SparseCore, also trace a single tile in it unless told
    # otherwise: fewer trace events, so more steps fit before the trace drops
    # events.
    num_sparse_cores = values.get(_NUM_SPARSE_CORES_KEY)
    if num_sparse_cores == 1 and not isinstance(num_sparse_cores, bool):
        values.setdefault(_NUM_SPARSE_CORE_TILES_KEY, 1)
    return name, _options_from_values(values)


def start_tpu_profile(
    *,
    output_dir: Path,
    with_cpu: bool = True,
    options: TpuProfileOptions | None = None,
    worker_rank: str | None = None,
) -> _TpuTrace:
    """Start a torch.profiler capture with TPU device activity.

    Returns a handle. Pass it to :func:`stop_tpu_profile` to end the capture.
    ``with_cpu`` adds ProfilerActivity.CPU. ``options`` defaults to
    ``resolve_tpu_profile_options(None)``. ``output_dir`` gets the Kineto trace
    (``*.pt.trace.json.gz``) and is the xplane's run_dir: the xplane goes to
    ``output_dir/plugins/profile/<timestamp>/<host>[_<worker_rank>].xplane.pb``.

    Raises RuntimeError if torch_tpu's profiler API is missing, and ValueError
    for bad options or an ``output_dir`` the plugin can't take.
    """
    import torch

    if options is None:
        options = resolve_tpu_profile_options(None)[1]
    activities = [_tpu_profiler_activity()]
    if with_cpu:
        activities.insert(0, torch.profiler.ProfilerActivity.CPU)

    output_dir = output_dir.absolute()
    _check_run_dir(output_dir)
    host_tracer_level = options.host_tracer_level
    if host_tracer_level is None:
        host_tracer_level = 2 if with_cpu else 0
    config = _tpu_profiler_config_cls()(
        run_dir=str(output_dir),
        worker_rank=worker_rank,
        host_tracer_level=host_tracer_level,
        device_tracer_level=options.device_tracer_level,
        python_tracer_level=options.python_tracer_level,
        experimental_options=dict(options.experimental_options) or None,
        check_experimental_options=options.check_experimental_options,
    )
    logger.info(
        "[TPU profiler] host_tracer_level=%d device_tracer_level=%d "
        "python_tracer_level=%d experimental_options=%s",
        host_tracer_level,
        options.device_tracer_level,
        options.python_tracer_level,
        options.experimental_options,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    profiler = torch.profiler.profile(
        activities=activities,
        # Adds Python stacks to the Kineto CPU trace. The xplane's python
        # tracer follows python_tracer_level in the config instead. Off by
        # default: it dominates host busy time.
        with_stack=envs.SGLANG_TPU_PROFILE_PYTHON_TRACER.get(),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            str(output_dir), use_gzip=True
        ),
        experimental_config=config,
    )
    started_at = time.time()
    profiler.start()
    return _TpuTrace(
        output_dir=output_dir,
        profiler=profiler,
        started_at=started_at,
        worker_rank=worker_rank,
    )


def stop_tpu_profile(trace: _TpuTrace) -> None:
    """Stop a capture started by :func:`start_tpu_profile`."""
    # The plugin synchronizes the TPU devices, then writes the xplane; the
    # Kineto trace is exported by on_trace_ready.
    trace.profiler.stop()
    try:
        _check_xplane(trace.output_dir, trace.started_at, trace.worker_rank)
    except Exception:  # a diagnostic must never fail /stop_profile
        logger.warning("[TPU profiler] could not check the xplane", exc_info=True)


class _TpuTrace(msgspec.Struct, frozen=True):
    """Handle for an in-flight capture."""

    output_dir: Path
    profiler: Any
    started_at: float
    worker_rank: str | None = None


def _default_option_values() -> dict[str, Any]:
    return {
        "python_tracer_level": int(envs.SGLANG_TPU_PROFILE_PYTHON_TRACER.get()),
        _NUM_CHIPS_KEY: envs.SGLANG_TPU_PROFILE_NUM_CHIPS.get(),
        _NUM_SPARSE_CORES_KEY: envs.SGLANG_TPU_PROFILE_NUM_SPARSE_CORES.get(),
    }


def _parse_options(text: str, source: str) -> dict[str, Any]:
    """Parse ``key:value;key:value``. Values become bools, ints or strings."""
    if not text.isascii():
        raise ValueError(f"{source}: profiler options must be ASCII: {text!r}")
    values: dict[str, Any] = {}
    for part in text.split(";"):
        part = part.strip()
        if not part:
            continue
        match = _OPTION_PATTERN.match(part)
        if not match:
            raise ValueError(
                f"{source}: invalid profiler option {part!r}; expected "
                "'key:value' items separated by ';'."
            )
        key, value = match.group(1), match.group(2).strip()
        if key in _RESERVED_KEYS:
            raise ValueError(
                f"{source}: SGLang sets {key!r} for every capture; it can't be "
                "overridden."
            )
        values[key] = _parse_option_value(key, value, source)
    return values


def _parse_option_value(key: str, value: str, source: str) -> int | bool | str:
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    try:
        return int(value)
    except ValueError:
        pass
    if not _ALLOWED_VALUE_CHARS.match(value):
        raise ValueError(
            f"{source}: invalid characters in value {value!r} of profiler "
            f"option {key!r}."
        )
    return value


def _options_from_values(values: dict[str, Any]) -> TpuProfileOptions:
    values = dict(values)
    levels = {}
    for key in _LEVEL_KEYS:
        if key not in values:
            continue
        level = values.pop(key)
        if isinstance(level, bool) or not isinstance(level, int) or level < 0:
            raise ValueError(
                f"profiler option {key!r} must be a non-negative integer, "
                f"got {level!r}."
            )
        levels[key] = level
    check = values.pop(_CHECK_KEY, True)
    if not isinstance(check, bool):
        raise ValueError(f"profiler option {_CHECK_KEY!r} must be true or false.")
    for key in _NARROWING_KEYS:
        # <= 0 traces all, like the env vars: drop the narrowing default.
        value = values.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value <= 0:
            del values[key]
    return TpuProfileOptions(
        **levels, check_experimental_options=check, experimental_options=values
    )


def _check_run_dir(run_dir: Path) -> None:
    path = str(run_dir)
    if not path.isascii() or any(c in path for c in _UNSUPPORTED_RUN_DIR_CHARS):
        raise ValueError(
            f"TPU profiler output directory {path!r} must be ASCII and must "
            "not contain ',', '\"' or '\\': torch_tpu's profiler config can't "
            "carry it."
        )


def _tpu_profiler_config_cls():
    import torch

    # torch_tpu adds torch.tpu.profiler on import; TpuProfilerConfig is
    # imported lazily on first access.
    profiler_api = getattr(getattr(torch, "tpu", None), "profiler", None)
    try:
        config_cls = getattr(profiler_api, "TpuProfilerConfig", None)
    except ImportError as e:
        raise RuntimeError(
            f"could not import torch.tpu.profiler.TpuProfilerConfig: {e}"
        ) from e
    if config_cls is None:
        raise RuntimeError(
            "torch.tpu.profiler.TpuProfilerConfig is not defined. TPU profiling "
            "needs torch_tpu imported, and a torch_tpu that provides it."
        )
    return config_cls


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


def _xplane_base_dir(run_dir: Path) -> Path:
    """Where the plugin puts the xplane of a capture with ``run_dir``."""
    return Path(os.environ.get(_XPLANE_OUTPUT_DIR_ENV_VAR) or run_dir)


def _find_xplane(
    run_dir: Path, started_at: float, worker_rank: str | None = None
) -> Path | None:
    """Newest xplane of this capture written since ``started_at``, if any.

    With ``worker_rank`` only ``<host>_<worker_rank>.xplane.pb`` matches, so
    another rank's xplane in a shared TPU_PROFILER_OUTPUT_DIR doesn't count.
    """
    name = f"*_{glob.escape(worker_rank)}.xplane.pb" if worker_rank else "*.xplane.pb"
    # Some filesystems keep whole-second mtimes.
    since = math.floor(started_at)
    newest: tuple[float, Path] | None = None
    for path in _xplane_base_dir(run_dir).glob(f"plugins/profile/*/{name}"):
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        if mtime >= since and (newest is None or mtime > newest[0]):
            newest = (mtime, path)
    return newest[1] if newest else None


def _check_xplane(
    run_dir: Path, started_at: float, worker_rank: str | None = None
) -> None:
    """Log where the xplane went, or an error if the capture came back empty."""
    base_dir = _xplane_base_dir(run_dir)
    if base_dir != run_dir:
        logger.warning(
            "[TPU profiler] %s=%s takes precedence over the capture directory "
            "%s for the xplane.",
            _XPLANE_OUTPUT_DIR_ENV_VAR,
            base_dir,
            run_dir,
        )
    xplane = _find_xplane(run_dir, started_at, worker_rank)
    if xplane is None:
        logger.error(
            "[TPU profiler] no .xplane.pb was written under %s. The capture "
            "has no TPU device activity; check that torch_tpu's Kineto plugin "
            "is registered.",
            base_dir / "plugins" / "profile",
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
