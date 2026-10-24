"""The MoE expert-parallel rank as a rank-lifted trace input."""

from typing import Optional

import torch

from sglang.srt.platforms import current_platform
from sglang.srt.runtime_context import get_parallel


def traced_moe_ep_rank() -> Optional[torch.Tensor]:
    """This rank's MoE EP index as a traced tensor, or None if it is not lifted.

    Rank lifting requires the TP group to span the world, so the traced input
    is the TP rank. Within a TP group, ranks are laid out as
    ``(moe_dp_rank * moe_ep_size + moe_ep_rank) * moe_tp_size + moe_tp_rank``.
    """
    # Probe before get_parallel(): untraced callers may not have published it.
    # Every rank is rank 0 of a size-1 group, so the probe never fails the rank
    # check and is None exactly when no rank input is lifted.
    if current_platform.get_traced_rank(0, group_size=1) is None:
        return None
    parallel = get_parallel()
    if parallel.moe_ep_size == 1:
        return None
    tp_rank = current_platform.get_traced_rank(
        parallel.tp_rank, group_size=parallel.tp_size
    )
    return tp_rank // parallel.moe_tp_size % parallel.moe_ep_size
