"""Hand a live tensor its functional update without a TPU op."""

import torch


def _shares_storage(tensor: torch.Tensor) -> bool:
    """Whether another tensor (a view) holds ``tensor``'s storage.

    The storage is referenced by ``tensor`` and by the Python storage object
    created here; anything above that is a view, which a ``.data`` swap would
    leave on the old storage. Backends without an inspectable storage report
    no sharing.
    """
    try:
        storage = tensor.untyped_storage()
    except (RuntimeError, NotImplementedError):
        return False
    return torch._C._storage_Use_Count(storage._cdata) > 2


def rebind_tensor(tensor: torch.Tensor, update: torch.Tensor) -> bool:
    """Make ``tensor`` hold ``update``, a same-shape result computed from it.

    A base tensor swaps in the new storage through ``.data``: TensorImpl
    metadata only, so no TPU op runs and every holder of ``tensor`` sees the
    update. A view, a tensor other views share, or a swap that would flip the
    inference flag copies instead. Returns whether it rebound.
    """
    if (
        tensor._base is None
        and tensor.is_inference() == update.is_inference()
        and tensor.shape == update.shape
        and tensor.dtype == update.dtype
        and not _shares_storage(tensor)
    ):
        tensor.data = update
        return True
    tensor.copy_(update)
    return False
