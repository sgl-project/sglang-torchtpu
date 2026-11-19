# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import dataclasses

LANE = 128


def pad_to(n: int, m: int) -> int:
    return -(-n // m) * m


@dataclasses.dataclass(frozen=True)
class KernelLayout:
    spec: object
    tp: int

    @property
    def vocab_padded(self) -> int:
        return pad_to(self.spec.vocab_size, self.tp * LANE)

    def small_rows(self) -> int:
        """Height of the per-layer bf16 slab, in rows of hidden_size values."""
        from sglang_tpu.glm_flash_megakernel.build import slab_plan

        return slab_plan(self.spec, self.spec.hidden_size)["_rows"][0]


# Pool arithmetic shared with serving's CPU unit test: plain operators only, so it
# works on numpy ints and on traced values alike without importing jax.
def record_bytes(head_dim):
    """Bytes per pooled record: key in whole 128-byte lanes plus one scale byte per lane, lane-rounded."""
    lane = 128
    key = -(-head_dim // lane) * lane
    return -(-(key + key // lane) // lane) * lane


def closing_pools(pos, Q, kp):
    """(first pool, count) of the pools whose last token is one of pos .. pos+Q-1."""
    first = (pos + kp) // kp - 1
    first = first + ((first + 1) * kp - 1 < pos)
    count = (pos + Q) // kp - 1 - first + 1
    return first, count * (count > 0)


def pool_owner(pool, pools_per_page, pools_per_share):
    """Rank of a split latent cache (kv_shard.py) holding pool `pool`'s tokens;
    power-of-two counts, so it lowers to a mask and a shift on vectors."""
    return (pool & (pools_per_page - 1)) >> (pools_per_share.bit_length() - 1)


def pool_block_row(pool, kpool, page_size, share, block):
    """Row, in its owner's `share`-token slice of the page, of the `block`-row gather
    block holding pool `pool` (share == page_size: the cache is whole)."""
    return (pool * kpool) % page_size % share // block * block
