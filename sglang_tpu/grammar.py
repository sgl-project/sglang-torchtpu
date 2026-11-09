# SPDX-License-Identifier: Apache-2.0
"""TPU kernels and host synchronization for grammar-guided sampling."""

from __future__ import annotations

import torch


@torch.compile(dynamic=False, fullgraph=True)
def _mask_logits(logits: torch.Tensor, bitmask: torch.Tensor) -> None:
    token_ids = torch.arange(logits.shape[-1], device=logits.device, dtype=torch.int32)
    words = torch.index_select(bitmask, -1, token_ids // 32)
    allowed = ((words >> (token_ids % 32)) & 1).to(torch.bool)
    logits.masked_fill_(~allowed, float("-inf"))


def apply_vocab_mask(logits: torch.Tensor, bitmask: torch.Tensor) -> None:
    # Grammar batches arrive after decoder warmup and can introduce new mask
    # shapes. Permit compilation of this independent kernel only; restore the
    # decoder's fail_on_recompile stance even if compilation raises.
    with (
        torch.compiler.set_stance("default"),
        torch._dynamo.config.patch(suppress_errors=False),
    ):
        _mask_logits(logits, bitmask)


def synchronize_token_ids(token_ids: torch.Tensor, group) -> None:
    if group.world_size == 1:
        return
    # A blocking host copy materializes deferred TPU sampling before the
    # collective. Gloo then broadcasts from the subgroup's global source rank.
    # Only the sampled IDs cross the host boundary, not vocabulary-sized logits.
    host_ids = token_ids.to("cpu", non_blocking=False)
    torch.distributed.broadcast(host_ids, src=group.ranks[0], group=group.cpu_group)
    token_ids.copy_(host_ids.to(token_ids.device))
