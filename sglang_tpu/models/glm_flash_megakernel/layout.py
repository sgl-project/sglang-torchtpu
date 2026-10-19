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
        from sglang_tpu.models.glm_flash_megakernel.build import slab_plan

        return slab_plan(self.spec, self.spec.hidden_size)["_rows"][0]
