"""TPU profiler on the native torch.profiler API.

TPU device activity reaches torch.profiler through torch_tpu's Kineto plugin,
which registers with PyTorch's PrivateUse1ProfilerRegistry. One
``torch.profiler.profile`` records both the Kineto CPU trace and the TPU
xplane. torch_tpu's ``TpuProfilerConfig``, passed as ``experimental_config``,
carries the TPU options: tracer levels and experimental (libtpu) options.

Options come from, lowest to highest precedence:

1. Built-in defaults and the SGLANG_TPU_PROFILE_PYTHON_TRACER,
   SGLANG_TPU_PROFILE_NUM_CHIPS and SGLANG_TPU_PROFILE_NUM_SPARSE_CORES env
   vars.
2. SGLANG_TPU_PROFILE_OPTIONS, a ``key:value;key:value`` string.
3. A /start_profile ``profile_prefix`` in the same ``key:value;...`` form.
4. A /start_profile ``profiler_kwargs`` dictionary (highest precedence; see
   :func:`resolve_tpu_profile_options`).

``host_tracer_level``, ``device_tracer_level``, ``python_tracer_level`` and
``check_experimental_options`` are TpuProfilerConfig arguments. Any other key
(for example ``tpu_trace_mode:TRACE_COMPUTE`` or ``tpu_perf_counters:true``)
is passed to libtpu as an experimental option. For the chip / SparseCore
narrowing options (``tpu_num_chips_to_profile_per_task``,
``tpu_num_sparse_cores_to_trace``, ``tpu_num_sparse_core_tiles_to_trace``) a
value <= 0 means trace all, and drops the option.
"""

from __future__ import annotations

import contextlib
import contextvars
import os
import re
from pathlib import Path
from typing import Any

import msgspec

from sglang_tpu._logging import get_logger
from sglang_tpu.envs import envs as tpu_envs

logger = get_logger(__name__)

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
_KEY_PATTERN = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")
_OPTION_PATTERN = re.compile(r"^([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*(.+)$")
_ALLOWED_VALUE_CHARS = re.compile(r"^[a-zA-Z0-9_./,:\-\s]*$")


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


_SCOPED_OPTIONS: contextvars.ContextVar[TpuProfileOptions | None] = (
    contextvars.ContextVar("tpu_scoped_profile_options", default=None)
)


@contextlib.contextmanager
def scoped_profile_options(options: TpuProfileOptions | None):
    """Bind default ``TpuProfileOptions`` for :func:`start_tpu_profile` within a block."""
    token = _SCOPED_OPTIONS.set(options)
    try:
        yield
    finally:
        _SCOPED_OPTIONS.reset(token)


def resolve_tpu_profile_options(
    profile_prefix: str | None = None,
    profiler_kwargs: dict[str, Any] | None = None,
) -> tuple[str | None, TpuProfileOptions]:
    """Resolve a /start_profile ``profile_prefix`` and ``profiler_kwargs``.

    A ``profile_prefix`` containing ``:`` or ``;`` is parsed as an options
    string, e.g. ``"host_tracer_level:0;tpu_trace_mode:TRACE_COMPUTE"``, and
    the returned name is None. Any other prefix is returned as the name.
    ``profiler_kwargs`` (when provided) takes highest precedence over both
    ``SGLANG_TPU_PROFILE_OPTIONS`` and ``profile_prefix``. Raises ValueError
    for malformed options.
    """
    values = _default_option_values()
    env_options = tpu_envs.SGLANG_TPU_PROFILE_OPTIONS.get()
    if env_options:
        values.update(_parse_options(env_options, "SGLANG_TPU_PROFILE_OPTIONS"))
    name = profile_prefix or None
    if profile_prefix and (":" in profile_prefix or ";" in profile_prefix):
        values.update(_parse_options(profile_prefix, "profile_prefix"))
        name = None
    if profiler_kwargs is not None:
        values.update(_parse_kwargs(profiler_kwargs, "profiler_kwargs"))
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
) -> _TpuTrace:
    """Start a torch.profiler capture with TPU device activity.

    Returns a handle. Pass it to :func:`stop_tpu_profile` to end the capture.
    ``with_cpu`` adds ProfilerActivity.CPU. ``options`` defaults to
    ``resolve_tpu_profile_options(None)[1]``. The Kineto trace
    (``*.pt.trace.json.gz``) is written to ``output_dir``.

    Raises RuntimeError if torch_tpu's profiler API is missing, and ValueError
    for bad options.
    """
    import torch

    if "," in os.fspath(output_dir):
        raise ValueError(
            f"output_dir cannot contain ',' for TpuProfilerConfig: {output_dir!r}"
        )
    if options is None:
        options = _SCOPED_OPTIONS.get()
    if options is None:
        options = resolve_tpu_profile_options(None)[1]
    activities = [_tpu_profiler_activity()]
    if with_cpu:
        activities.insert(0, torch.profiler.ProfilerActivity.CPU)

    host_tracer_level = options.host_tracer_level
    if host_tracer_level is None:
        host_tracer_level = 2 if with_cpu else 0
    config = _tpu_profiler_config_cls()(
        host_tracer_level=host_tracer_level,
        device_tracer_level=options.device_tracer_level,
        python_tracer_level=options.python_tracer_level,
        run_dir=output_dir,
        experimental_options=dict(options.experimental_options) or None,
        check_experimental_options=options.check_experimental_options,
    )
    logger.info(
        "[TPU profiler] run_dir=%s host_tracer_level=%d device_tracer_level=%d "
        "python_tracer_level=%d experimental_options=%s",
        output_dir,
        host_tracer_level,
        options.device_tracer_level,
        options.python_tracer_level,
        options.experimental_options,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    profiler = torch.profiler.profile(
        activities=activities,
        # Adds Python stacks to the Kineto CPU trace whenever python_tracer_level > 0
        # (seeded from SGLANG_TPU_PROFILE_PYTHON_TRACER unless overridden). Off by
        # default: it dominates host busy time.
        with_stack=options.python_tracer_level > 0,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            str(output_dir), use_gzip=True
        ),
        experimental_config=config,
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


def _default_option_values() -> dict[str, Any]:
    return {
        "python_tracer_level": int(tpu_envs.SGLANG_TPU_PROFILE_PYTHON_TRACER.get()),
        _NUM_CHIPS_KEY: tpu_envs.SGLANG_TPU_PROFILE_NUM_CHIPS.get(),
        _NUM_SPARSE_CORES_KEY: tpu_envs.SGLANG_TPU_PROFILE_NUM_SPARSE_CORES.get(),
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


def _parse_kwargs(kwargs: dict[str, Any], source: str) -> dict[str, Any]:
    """Validate and normalize a ``profiler_kwargs`` dictionary."""
    if not isinstance(kwargs, dict):
        raise ValueError(
            f"{source}: expected a dictionary of profiler options, "
            f"got {type(kwargs).__name__}."
        )
    values: dict[str, Any] = {}
    for key, value in kwargs.items():
        if not isinstance(key, str) or not key.isascii() or not _KEY_PATTERN.match(key):
            raise ValueError(f"{source}: invalid profiler option key {key!r}.")
        if key in _RESERVED_KEYS:
            raise ValueError(
                f"{source}: SGLang sets {key!r} for every capture; it can't be "
                "overridden."
            )
        if isinstance(value, (bool, int)):
            values[key] = value
        elif isinstance(value, str):
            if not value.isascii():
                raise ValueError(
                    f"{source}: profiler option {key!r} value must be ASCII: {value!r}"
                )
            values[key] = _parse_option_value(key, value.strip(), source)
        else:
            raise ValueError(
                f"{source}: profiler option {key!r} must be a bool, int, or str, "
                f"got {type(value).__name__}."
            )
    return values


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
        if key not in values:
            continue
        value = values[key]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(
                f"profiler option {key!r} must be an integer, got {value!r}."
            )
        # <= 0 traces all, like the env vars: drop the narrowing default.
        if value <= 0:
            del values[key]
    return TpuProfileOptions(
        **levels, check_experimental_options=check, experimental_options=values
    )


def _tpu_profiler_activity():
    import torch

    # torch >= 2.12 + torch_tpu exposes ProfilerActivity.TPU (an alias of
    # PrivateUse1); torch 2.11 uses ProfilerActivity.PrivateUse1 directly.
    activities = torch.profiler.ProfilerActivity
    activity = getattr(activities, "TPU", None) or getattr(
        activities, "PrivateUse1", None
    )
    if activity is None:
        raise RuntimeError(
            "torch.profiler.ProfilerActivity.TPU / PrivateUse1 is not defined. "
            "TPU profiling needs torch_tpu imported and torch >= 2.11."
        )
    return activity


def _tpu_profiler_config_cls():
    import torch

    tpu_mod = getattr(torch, "tpu", None)
    profiler_mod = getattr(tpu_mod, "profiler", None)
    cls = getattr(profiler_mod, "TpuProfilerConfig", None)
    if cls is not None:
        return cls
    try:
        from torch_tpu._internal.profiler import TpuProfilerConfig
    except ImportError as e:
        raise RuntimeError(
            "TpuProfilerConfig is not available via torch.tpu.profiler or "
            "torch_tpu._internal.profiler. TPU profiling needs torch_tpu >= 0.2.0."
        ) from e
    return TpuProfilerConfig
