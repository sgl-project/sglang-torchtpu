"""Device-specific seams in the DFlash draft loop.

The defaults here are the CUDA/NPU behavior: no device-resident verify
addressing, so the draft loop reads its block's slots back through the host
req_to_token mapping and the draft forward carries no extend split. TPU
overrides both to keep a per-step seq_lens D2H off the decode path.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Protocol

import msgspec
import torch

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.speculative.spec_utils import assign_extend_cache_locs_func


class DFlashBackendState(Protocol):
    """Per-step device-resident addressing a backend carries on the draft input.

    Opaque to the generic draft input, which only routes it through batch
    filter/merge. Absence of a state is the generic path, so there is no
    separate "is the fast path live" flag to keep in sync.
    """

    def filter(self, new_indices: torch.Tensor) -> DFlashBackendState:
        """Keep the rows in new_indices, in that order."""
        ...

    def merge(self, other: DFlashBackendState) -> DFlashBackendState:
        """Concatenate other's rows after this one's."""
        ...


class DFlashExtendSplit(msgspec.Struct, frozen=True):
    """How a backend wants the draft block planned.

    Attributes:
      seq_lens: Sequence lengths the draft forward runs against, [bs].
      forward_fields: Extra ForwardBatch kwargs, splatted by the caller. Empty
        unless the backend plans the block as an extend.
    """

    seq_lens: torch.Tensor
    forward_fields: dict[str, Any] = {}


class DFlashBackend:
    """Device seams of the DFlash draft loop, resolved through the platform.

    Subclasses own whatever staging buffers their overrides need, so the worker
    holds one collaborator instead of a buffer per device path.
    """

    def __init__(
        self,
        *,
        device: torch.device,
        block_size: int,
        pad_bs: Callable[[int], int],
        req_to_token_pool: ReqToTokenPool,
    ) -> None:
        self.device = device
        self.block_size = int(block_size)
        self.pad_bs = pad_bs
        # The pool, not its tensor: clear_cache_pool rebinds req_to_token.
        self.req_to_token_pool = req_to_token_pool

    @classmethod
    def plan_decode(
        cls,
        batch: ScheduleBatch,
        *,
        bs: int,
        block_size: int,
        committed_lens_cpu: torch.Tensor,
        compact_draft_cache: bool,
    ) -> Optional[DFlashBackendState]:
        """Build this step's device-resident addressing, or None for the host path.

        A classmethod because prepare_for_decode runs scheduler-side, where no
        worker and therefore no backend instance exists.
        """
        return None

    def host_len_slack(self, state: Optional[DFlashBackendState]) -> int:
        """Extra headroom every bound derived from the host seq_lens mirror needs."""
        return 0

    def fill_verify_block_cache_locs(
        self,
        *,
        req_pool_indices: torch.Tensor,
        prefix_lens: torch.Tensor,
        state: Optional[DFlashBackendState],
        out_2d: torch.Tensor,
        bs: int,
    ) -> None:
        """Write the verify block's KV slots into out_2d, [bs, block_size]."""
        locs = assign_extend_cache_locs_func(
            req_pool_indices=req_pool_indices,
            req_to_token=self.req_to_token_pool.req_to_token,
            start_offset=prefix_lens,
            end_offset=prefix_lens + self.block_size,
            batch_size=bs,
            draft_token_num=self.block_size,
            device=self.device,
        )
        out_2d.copy_(locs.view(bs, self.block_size))

    def draft_extend_split(
        self,
        *,
        bs: int,
        draft_prefix_lens: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
    ) -> DFlashExtendSplit:
        """Plan the draft block for the draft forward."""
        return DFlashExtendSplit(seq_lens=draft_prefix_lens)


def filter_backend_state(
    state: Optional[DFlashBackendState], new_indices: torch.Tensor
) -> Optional[DFlashBackendState]:
    """Route a batch filter through the state, if there is one."""
    return None if state is None else state.filter(new_indices)


def merge_backend_state(
    state: Optional[DFlashBackendState], other: Optional[DFlashBackendState]
) -> Optional[DFlashBackendState]:
    """Merge two batches' states; mismatched generations drop to the host path."""
    return None if state is None or other is None else state.merge(other)
