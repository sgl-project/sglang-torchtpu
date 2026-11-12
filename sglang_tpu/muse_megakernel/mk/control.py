# SPDX-License-Identifier: Apache-2.0
"""Branch trees and selects over traced indices."""

import functools

import jax
import jax.numpy as jnp


def _branch(idx, fns, lo=0):
    """Run ``fns[idx - lo]`` for a traced ``idx`` through a balanced tree of two-way branches.

    ``lax.switch`` over ~60 layers costs time linear in the branch count; a tree
    costs log2(n)
    compares.
    """
    if len(fns) == 1:
        fns[0]()
        return
    mid = len(fns) // 2
    jax.lax.cond(
        idx < lo + mid,
        lambda: _branch(idx, fns[:mid], lo),
        lambda: _branch(idx, fns[mid:], lo + mid),
    )


def _select(idx, keys, fn):
    """Run ``fn(k)`` for the ``k`` in sorted ``keys`` equal to ``idx`` (a traced index must be

    one of them) through a balanced compare tree.
    """
    if isinstance(idx, int):
        fn(idx)
        return
    if len(keys) == 1:
        fn(keys[0])
        return
    mid = len(keys) // 2
    jax.lax.cond(
        idx < keys[mid],
        lambda: _select(idx, keys[:mid], fn),
        lambda: _select(idx, keys[mid:], fn),
    )


def _dispatch(idx, refs, fn, only=None):
    """Run ``fn(ref)`` on ``refs[idx]``; a traced index goes through ``_branch``.

    ``only``: the sorted indices ``idx`` can take, so the tree has a leaf per
    index rather than per ref.
    """
    if isinstance(idx, int):
        fn(refs[idx])
        return
    if only is None:
        _branch(idx, [functools.partial(fn, r) for r in refs])
        return
    _select(idx, list(only), lambda k: fn(refs[k]))


def _sel(flag, a, b):
    """``where(flag, a, b)``, resolved at trace time for a Python-bool flag."""
    if isinstance(flag, bool):
        return a if flag else b
    return jnp.where(flag, a, b)


def _in_window(is_global, cond):
    """``is_global or cond``; None when every key is in range."""
    if isinstance(is_global, bool):
        return None if is_global else cond
    return jnp.logical_or(is_global, cond)
