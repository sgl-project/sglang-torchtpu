# SPDX-License-Identifier: Apache-2.0
"""One upload per dtype for a forward's host-built attention metadata.

Each non-blocking host-to-device copy costs the host about 0.1 ms to issue, and
the DSV4.1 metadata made ~16 of them per forward, on the host path between two
programs. ``pack_upload`` concatenates the host tensors of each dtype, uploads
one buffer per dtype and returns the fields as views of those buffers.
"""

from typing import Dict, List, Sequence, Tuple

import torch


def pack_upload(
    tensors: Sequence[torch.Tensor], *, device: torch.device
) -> List[torch.Tensor]:
    """``tensors`` on ``device``, in order, as views of one upload per dtype."""
    groups: Dict[torch.dtype, List[int]] = {}
    for i, t in enumerate(tensors):
        groups.setdefault(t.dtype, []).append(i)
    out: List[torch.Tensor] = [None] * len(tensors)  # type: ignore[list-item]
    for indices in groups.values():
        host = torch.cat([tensors[i].reshape(-1) for i in indices])
        buffer = host.to(device=device, non_blocking=True)
        offset = 0
        for i in indices:
            numel = tensors[i].numel()
            out[i] = buffer[offset : offset + numel].view(tensors[i].shape)
            offset += numel
    return out


def _is_leaf(x) -> bool:
    return x is None or isinstance(x, torch.Tensor)


def lift_views(values: Sequence[object]) -> Tuple[list, tuple, tuple]:
    """Split the packed views out of ``values`` (pytrees of tensors).

    Returns ``values`` with each contiguous view of a 1-D base replaced by None,
    the distinct bases, and per view ``(value index, leaf index, base index,
    offset, shape)``. TorchTPU materializes every view argument of an executable
    on every call (~19 us each); the bases go in as plain arguments instead and
    ``restore_views`` rebuilds the views inside the graph.
    """
    from jax import tree_util

    bases: List[torch.Tensor] = []
    base_index: Dict[int, int] = {}
    specs = []
    out = []
    for vi, value in enumerate(values):
        leaves, treedef = tree_util.tree_flatten(value, is_leaf=_is_leaf)
        lifted = False
        for li, t in enumerate(leaves):
            base = t._base if isinstance(t, torch.Tensor) else None
            if base is None or base.dim() != 1 or not t.is_contiguous():
                continue
            bi = base_index.get(id(base))
            if bi is None:
                bi = base_index[id(base)] = len(bases)
                bases.append(base)
            offset = t.storage_offset() - base.storage_offset()
            specs.append((vi, li, bi, offset, tuple(t.shape)))
            leaves[li] = None
            lifted = True
        out.append(tree_util.tree_unflatten(treedef, leaves) if lifted else value)
    return out, tuple(bases), tuple(specs)


def restore_views(
    values: Sequence[object], bases: Sequence[torch.Tensor], specs
) -> list:
    """Inverse of ``lift_views``: the views as slices of ``bases``."""
    from jax import tree_util

    out = list(values)
    by_value: Dict[int, list] = {}
    for spec in specs:
        by_value.setdefault(spec[0], []).append(spec)
    for vi, group in by_value.items():
        leaves, treedef = tree_util.tree_flatten(out[vi], is_leaf=_is_leaf)
        for _, li, bi, offset, shape in group:
            numel = 1
            for d in shape:
                numel *= d
            leaves[li] = bases[bi].narrow(0, offset, numel).view(shape)
        out[vi] = tree_util.tree_unflatten(treedef, leaves)
    return out
