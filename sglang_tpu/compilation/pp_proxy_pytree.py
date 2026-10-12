"""JAX pytree view of ``PPProxyTensors`` for the direct TPU tracer.

``direct_compile`` flattens every input and output with ``jax.tree_util``; the
pipeline-stage proxy (hidden states and residual of a non-final stage) is a
plain object holding a dict of tensors, so it needs an explicit pytree
definition. The flatten and unflatten functions here take no JAX dependency;
``register_pp_proxy_pytree`` wires them into a ``tree_util`` module.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable

from sglang.srt.model_executor.forward_batch_info import PPProxyTensors


def flatten_pp_proxy_with_keys(
    proxy: PPProxyTensors, make_key: Callable[[str], Any]
) -> tuple[list[tuple[Any, Any]], tuple[str, ...]]:
    """Return ``([(key, tensor), ...], keys)`` in the proxy's key order."""
    keys = tuple(proxy.tensors.keys())
    return [(make_key(key), proxy.tensors[key]) for key in keys], keys


def unflatten_pp_proxy(keys: Iterable[str], values: Iterable[Any]) -> PPProxyTensors:
    """Rebuild a proxy from the key order and the leaves."""
    return PPProxyTensors(dict(zip(keys, values)))


def register_pp_proxy_pytree(tree_util: Any) -> None:
    """Register ``PPProxyTensors`` as a keyed pytree on ``tree_util``.

    Children are addressed with ``DictKey`` so a key path resolves through
    ``PPProxyTensors.__getitem__`` at run time.
    """
    tree_util.register_pytree_with_keys(
        PPProxyTensors,
        lambda proxy: flatten_pp_proxy_with_keys(proxy, tree_util.DictKey),
        unflatten_pp_proxy,
    )
