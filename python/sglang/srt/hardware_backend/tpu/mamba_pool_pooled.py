# SPDX-License-Identifier: Apache-2.0
"""Byte-addressed Gated-DeltaNet state pool for the context-parallel path.

The dense pool keeps conv and recurrent state in two tensors shaped to the
kernel's VMEM tiles. The fused context-parallel kernel does not take those: it
reads and writes both regions through a single indexed byte buffer, so that
the state traffic rides the same DMA pipeline as the head-shard exchange
instead of being separate operands.

Layout. The buffer is a flat array of 256-token kernel blocks, each block one
``(1, 4, 128)`` fp8 row per token, i.e. 512 bytes per token. One state index
owns a window of ``split`` consecutive blocks:

    [ recurrent region: whole blocks         ][ conv region: rows ][ pad ]
     n_v * d_k * d_v * 4 bytes                 (K-1) * conv_dim * 2

``pool_state.build_pool_state_plan`` turns that geometry into the static plan
the kernel indexes with. The dims here are already *head-shard local*: under
context parallelism a rank owns ``n_v / cp_size`` value heads, so its state
covers only those, and the caller passes the divided counts.
"""

from __future__ import annotations

import math
from typing import List

import torch

from sglang.srt.hardware_backend.tpu._logging import get_logger

logger = get_logger(__name__)

GB = 1024**3

# The kernel addresses the pool in blocks of this many tokens, each token one
# (1, 4, 128) fp8 row. Both are fixed by the kernel's pipeline geometry.
POOL_KERNEL_BLOCK_TOKENS = 256
POOL_TOKEN_BYTES = 4 * 128


def derive_pool_block_tokens(
    *,
    n_v: int,
    conv_dim: int,
    d_k: int,
    d_v: int,
    kernel_size: int,
) -> int:
    """Tokens one state index needs, rounded up to whole kernel blocks.

    The recurrent region must be a whole number of pool tokens and the conv
    region must divide it, so the conv rows start on a boundary the kernel can
    address with a single static offset.
    """
    ssm_bytes = n_v * d_k * d_v * 4
    conv_bytes = (kernel_size - 1) * conv_dim * 2
    if ssm_bytes % POOL_TOKEN_BYTES:
        raise ValueError(
            f"recurrent state ({ssm_bytes} B) must occupy whole "
            f"{POOL_TOKEN_BYTES} B pool tokens; got n_v={n_v} d_k={d_k} d_v={d_v}."
        )
    ssm_tokens = ssm_bytes // POOL_TOKEN_BYTES
    conv_tokens = 1
    while conv_tokens * POOL_TOKEN_BYTES < conv_bytes or ssm_tokens % conv_tokens:
        conv_tokens *= 2
        if conv_tokens > ssm_tokens:
            raise ValueError(
                f"conv state ({conv_bytes} B) cannot be placed after a "
                f"{ssm_bytes} B recurrent region in this pool geometry."
            )
    needed = ssm_tokens + conv_tokens
    return math.ceil(needed / POOL_KERNEL_BLOCK_TOKENS) * POOL_KERNEL_BLOCK_TOKENS


class TpuPooledGDNState:
    """Per-layer byte-addressed GDN state buffers.

    Slot bookkeeping stays with the dense pool's allocator: this owns storage
    only, indexed by the same state indices, so a request's slot means the
    same thing on both paths.
    """

    def __init__(
        self,
        *,
        size: int,
        mamba_layer_ids: List[int],
        n_v: int,
        conv_dim: int,
        d_k: int,
        d_v: int,
        kernel_size: int,
        device: str,
    ):
        self.size = size
        self.layer_ids = list(mamba_layer_ids)
        self.pool_block_tokens = derive_pool_block_tokens(
            n_v=n_v, conv_dim=conv_dim, d_k=d_k, d_v=d_v, kernel_size=kernel_size
        )
        self.split = self.pool_block_tokens // POOL_KERNEL_BLOCK_TOKENS
        # Slot 0 is reserved, matching the dense pool.
        nblocks = size + 1
        # One tensor per layer, not a view into a stacked one: the kernel
        # takes its layer's pool as a donated argument, and donation needs a
        # whole tensor. Donating a view leaves the storage backing every other
        # layer in an invalid state.
        shape = (nblocks * self.split, POOL_KERNEL_BLOCK_TOKENS, 1, 4, 128)
        self.pool = [
            torch.zeros(shape, dtype=torch.float8_e4m3fn, device=device)
            for _ in self.layer_ids
        ]
        self._layer_id_to_index = {lid: i for i, lid in enumerate(self.layer_ids)}
        self.mem_usage = sum(t.numel() * t.element_size() for t in self.pool) / GB
        logger.info(
            "TPU pooled GDN state allocated: max_size=%d, layers=%d, "
            "block_tokens=%d, split=%d, shape=%s (mem=%.3fGB)",
            size,
            len(self.layer_ids),
            self.pool_block_tokens,
            self.split,
            [len(self.pool), *shape],
            self.mem_usage,
        )

    def layer_state(self, layer_id: int) -> torch.Tensor:
        """This layer's pool, shaped as the kernel's ``state_source``."""
        return self.pool[self._layer_id_to_index[layer_id]]
