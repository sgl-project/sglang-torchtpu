"""Hand a live tensor its functional update without a TPU op."""

import torch


def rebind_tensor(tensor: torch.Tensor, update: torch.Tensor) -> None:
    """Make ``tensor`` hold ``update``, a same-shape result computed from it.

    A base tensor swaps in the new storage through ``.data``: TensorImpl
    metadata only, so no TPU op runs and every holder of ``tensor`` sees the
    update. A view, or a swap that would flip the inference flag, copies.
    """
    if tensor._base is None and tensor.is_inference() == update.is_inference():
        tensor.data = update
    else:
        tensor.copy_(update)
