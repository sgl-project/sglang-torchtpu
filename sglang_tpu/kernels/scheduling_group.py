# SPDX-License-Identifier: Apache-2.0
"""Process-unique XLA scheduling groups; ops sharing an ID are scheduled together."""

import threading

import jax
from jax.experimental import xla_metadata

_next_id = 1
_lock = threading.Lock()


def reserve(count: int) -> int:
    """Reserve ``count`` consecutive IDs and return the first."""
    global _next_id
    if count < 1:
        raise ValueError(f"count ({count}) must be positive")
    with _lock:
        first, _next_id = _next_id, _next_id + count
    return first


def tag(tree, group: int | None):
    """Attach ``group`` to every array in ``tree``; ``None`` leaves it untagged."""
    if group is None:
        return tree
    return jax.tree.map(
        lambda x: xla_metadata.set_xla_metadata(x, _scheduling_group_id=group), tree
    )
