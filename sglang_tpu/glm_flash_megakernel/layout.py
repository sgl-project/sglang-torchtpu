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
