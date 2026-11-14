# SPDX-License-Identifier: Apache-2.0
"""Export hit/miss and static-arg cardinality stats for cpu_export_jax_op.

Every distinct static value exports its own kernel, so a per-layer static
creates one kernel per layer. Diagnostic only: static values are never
rewritten or logged.
"""

import collections
import hashlib
import threading
from collections.abc import Iterable, Mapping, Sequence

import msgspec

from sglang_tpu._logging import get_logger
from sglang_tpu.envs import envs as tpu_envs

logger = get_logger(__name__)


class ExportStats(msgspec.Struct, frozen=True):
    """Counters for one custom op since the last reset."""

    hits: int
    misses: int  # calls that exported a kernel
    static_variants: int  # distinct static-arg signatures
    shape_variants: int  # distinct input shape/dtype signatures
    max_static_variants_per_shape: int  # what the warning threshold checks
    varying_static_args: dict[str, int]  # arg -> distinct values, if > 1


class _Site:
    __slots__ = ("hits", "misses", "statics_by_shape", "values_by_arg", "warned")

    def __init__(self) -> None:
        self.hits = 0
        self.misses = 0
        # Input shape/dtype signature -> static signatures exported with it.
        self.statics_by_shape = collections.defaultdict(set)
        # Static arg name -> digests of the values it took.
        self.values_by_arg = collections.defaultdict(set)
        self.warned = False


_LOCK = threading.Lock()
_SITES = collections.defaultdict(_Site)  # custom-op name -> _Site


def record_export_call(
    name: str,
    *,
    exported: bool,
    param_names: Sequence[str],
    static_argnums: Sequence[int],
    args: Sequence[object],
    kwargs: Mapping[str, object],
) -> None:
    """Count one call to custom op ``name``; ``exported`` means a cache miss."""
    with _LOCK:
        site = _SITES[name]
        if not exported:
            # Hot path: signatures are built only on misses, which already
            # paid for a JAX export.
            site.hits += 1
            return

        static_sig, shape_sig = _signatures(param_names, static_argnums, args, kwargs)
        cause = _miss_cause(site, static_sig, shape_sig)
        site.misses += 1
        same_shape = site.statics_by_shape[shape_sig]
        same_shape.add(static_sig)
        for arg_name, digest in static_sig:
            site.values_by_arg[arg_name].add(digest)
        logger.debug("[pallas export] %s miss %d: %s", name, site.misses, cause)

        threshold = tpu_envs.SGLANG_TPU_PALLAS_EXPORT_STATIC_VARIANT_THRESHOLD.get()
        if site.warned or not 0 < threshold < len(same_shape):
            return
        site.warned = True
        # Only args that vary at this one shape are named; args that follow
        # the shape bucket are expected variants.
        logger.warning(
            "[pallas export] %s exported %d kernels for one input shape/dtype "
            "(threshold %d); varying static args: %s. Pass per-layer or "
            "per-request values as tensors or bounded IDs. "
            "misses=%d hits=%d shape_variants=%d",
            name,
            len(same_shape),
            threshold,
            _varying_args(same_shape),
            site.misses,
            site.hits,
            len(site.statics_by_shape),
        )


def get_export_stats(name: str) -> ExportStats | None:
    """Counters for custom op ``name``, or None if not called since reset."""
    with _LOCK:
        site = _SITES.get(name)
        if site is None:
            return None
        per_shape = site.statics_by_shape.values()
        return ExportStats(
            hits=site.hits,
            misses=site.misses,
            static_variants=len(set().union(*per_shape)),
            shape_variants=len(per_shape),
            max_static_variants_per_shape=max(map(len, per_shape), default=0),
            varying_static_args={
                arg_name: len(digests)
                for arg_name, digests in site.values_by_arg.items()
                if len(digests) > 1
            },
        )


def reset_export_stats() -> None:
    """Drop all counters and re-arm every warning."""
    with _LOCK:
        _SITES.clear()


def _signatures(
    param_names: Sequence[str],
    static_argnums: Sequence[int],
    args: Sequence[object],
    kwargs: Mapping[str, object],
) -> tuple[tuple, tuple]:
    """Return ((static arg name, value digest), ...) and input shapes/dtypes."""
    static_sig, shape_sig = [], []
    for index, arg in enumerate(args):
        if index in static_argnums:
            arg_name = param_names[index] if index < len(param_names) else f"arg{index}"
            static_sig.append((arg_name, _digest(arg)))
        else:
            shape_sig.append(_shape_dtype(arg))
    # Keyword arguments are static too.
    static_sig += [(key, _digest(kwargs[key])) for key in sorted(kwargs)]
    return tuple(static_sig), tuple(shape_sig)


def _shape_dtype(arg: object) -> object:
    # Host-side tensor metadata: reading it does not sync the device.
    if getattr(arg, "shape", None) is None:
        return None if arg is None else type(arg).__qualname__
    return tuple(arg.shape), str(arg.dtype)


def _digest(value: object) -> str:
    # Opaque so static values never reach logs; only distinct counts matter.
    text = f"{type(value).__qualname__}:{value!r}"
    return hashlib.blake2b(
        text.encode(errors="backslashreplace"), digest_size=8
    ).hexdigest()


def _miss_cause(site: _Site, static_sig: tuple, shape_sig: tuple) -> str:
    # Called before the miss is recorded, so "new" means new to this op.
    causes = [
        f"new {arg_name} value"
        for arg_name, digest in static_sig
        if digest not in site.values_by_arg.get(arg_name, ())
    ]
    if shape_sig not in site.statics_by_shape:
        causes.append("new input shape/dtype")
    return ", ".join(causes) or "no new static value or input shape/dtype"


def _varying_args(static_sigs: Iterable[tuple]) -> str:
    """Format the static args that take more than one value as name=count."""
    values = collections.defaultdict(set)
    for static_sig in static_sigs:
        for arg_name, digest in static_sig:
            values[arg_name].add(digest)
    return ", ".join(f"{arg}={len(d)}" for arg, d in values.items() if len(d) > 1)
