"""Module to integrate torch_tpu's Dynamo backend with PyTorch's AOTAutogradCache."""

from __future__ import annotations

import inspect

from sglang.srt.hardware_backend.tpu._logging import get_logger

logger = get_logger(__name__)

_installed = False


def _replace_dynamo_backend(name: str, compiler_fn) -> None:
    """Re-registers a Dynamo backend under the given name.

    Not atomic: pops the prior entry before calling register_backend. If
    register_backend raises, the registry is left without an entry for
    ``name``. The caller is responsible for restoring state in that case.
    """
    from torch._dynamo import register_backend
    from torch._dynamo.backends import registry

    registry._COMPILER_FNS.pop(name, None)
    register_backend(name=name, compiler_fn=compiler_fn)


def _forward_backend_kwargs(new_cls, old_backend, overrides: dict) -> dict:
    """Build kwargs for ``new_cls`` by mirroring state from ``old_backend``.

    Walks ``new_cls.__init__``'s signature. ``self``, ``*args``, and
    ``**kwargs`` are ignored. Each remaining keyword-accepting parameter is
    resolved in order:

    * if its name is a key in ``overrides``, use ``overrides[name]``;
    * else if ``old_backend`` exposes a matching ``_<name>`` attribute
      (torch_tpu's private-attr convention), forward that value;
    * else if ``old_backend`` exposes a matching ``<name>`` attribute
      (catches backends that expose state without the underscore prefix),
      forward that value;
    * else raise ``TypeError`` — we cannot infer the value from any source
      (including the case where the param has a default; we don't assume
      ``old_backend`` was constructed at the default).

    Raises ``TypeError`` if:

    * any parameter is positional-only — we only return a kwargs dict, so
      such args can't be forwarded;
    * any ``overrides`` key is not an accepted parameter name — e.g. a
      future torch_tpu removes ``enable_serialization``, in which case
      silently dropping the override would degrade cache behavior unnoticed.
    """
    sig = inspect.signature(new_cls.__init__)
    accepted = {}
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        if param.kind == inspect.Parameter.POSITIONAL_ONLY:
            # We only build a kwargs dict, so a positional-only arg can't
            # be forwarded even if we knew its value. Fail loudly rather
            # than silently dropping it.
            raise TypeError(
                f"Cannot construct {new_cls.__name__}: positional-only "
                f"argument {name!r} is not forwardable"
            )
        accepted[name] = param

    unknown_overrides = set(overrides) - set(accepted)
    if unknown_overrides:
        raise TypeError(
            f"{new_cls.__name__} does not accept overrides: "
            f"{sorted(unknown_overrides)}"
        )

    kwargs = {}
    for name in accepted:
        if name in overrides:
            kwargs[name] = overrides[name]
        elif old_backend is not None and hasattr(old_backend, "_" + name):
            kwargs[name] = getattr(old_backend, "_" + name)
        elif old_backend is not None and hasattr(old_backend, name):
            kwargs[name] = getattr(old_backend, name)
        else:
            raise TypeError(
                f"Cannot construct {new_cls.__name__}: cannot infer "
                f"argument {name!r} from old_backend or overrides"
            )
    return kwargs


def install() -> None:
    """Enables AOTAutogradCache for TPU compilation; rolls back atomically on failure."""
    global _installed
    if _installed:
        return

    try:
        from torch._dynamo.backends import registry
        from torch._functorch import config as _ft_config
        from torch_tpu._internal.compile._backend import TpuBackend

        old_backend = registry._COMPILER_FNS.get("tpu")
        old_enable = _ft_config.enable_autograd_cache
        old_bundled = _ft_config.bundled_autograd_cache

        backend_kwargs = _forward_backend_kwargs(
            TpuBackend, old_backend, overrides={"enable_serialization": True}
        )

        try:
            _replace_dynamo_backend("tpu", TpuBackend(**backend_kwargs))
            _ft_config.enable_autograd_cache = True
            _ft_config.bundled_autograd_cache = True
        except Exception:
            if old_backend is not None:
                registry._COMPILER_FNS["tpu"] = old_backend
            _ft_config.enable_autograd_cache = old_enable
            _ft_config.bundled_autograd_cache = old_bundled
            raise
    except Exception as exc:  # noqa: BLE001
        logger.exception(
            "AOTAutogradCache install failed: %s. Server will continue with "
            "compile-cache MISS on every launch.",
            exc,
        )
        return

    _installed = True
    logger.info("AOTAutogradCache successfully enabled for TPU backend.")
