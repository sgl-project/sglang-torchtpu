# SPDX-License-Identifier: Apache-2.0
"""DMA views. The pipeline owns start/wait order and buffer lifetimes."""

from dataclasses import dataclass
from typing import Any

from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


@dataclass(frozen=True)
class ShortQueryIO:
    desc: Any
    raw: Any
    output: Any
    raw_buffer: Any
    output_buffer: Any
    read_sem: Any
    old_sem: Any
    write_sem: Any

    def destination(self, i):
        return self.output.at[self.desc[2, i], pl.ds(self.desc[3, i], 1), 0, :]

    def read(self, i, slot):
        return pltpu.make_async_copy(
            self.raw.at[self.desc[0, i], pl.ds(self.desc[1, i] * 16, 16), :],
            self.raw_buffer.at[slot],
            self.read_sem.at[slot],
        )

    def read_old(self, i, slot):
        return pltpu.make_async_copy(
            self.destination(i), self.output_buffer.at[slot], self.old_sem.at[slot]
        )

    def write(self, i, slot):
        return pltpu.make_async_copy(
            self.output_buffer.at[slot], self.destination(i), self.write_sem.at[slot]
        )


@dataclass(frozen=True)
class LongQueryIO:
    desc: Any
    raw: Any
    output: Any
    raw_buffer: Any
    output_buffer: Any
    read_sem: Any
    write_sem: Any
    old_sem: Any
    tile_rows: int

    def destination(self, i):
        return self.output.at[
            self.desc[1, i], pl.ds(self.desc[2, i], self.tile_rows // 16), 0, :
        ]

    def read(self, i):
        # Store tile indices in scalar metadata so alignment remains explicit.
        return pltpu.make_async_copy(
            self.raw.at[
                self.desc[0, i],
                pl.ds(self.desc[5, i] * self.tile_rows, self.tile_rows),
                :,
            ],
            self.raw_buffer.at[i % 2],
            self.read_sem.at[i % 2],
        )

    def read_old(self, i, slot):
        return pltpu.make_async_copy(
            self.destination(i), self.output_buffer.at[slot], self.old_sem
        )

    def write(self, i):
        return pltpu.make_async_copy(
            self.output_buffer.at[i % 2], self.destination(i), self.write_sem.at[i % 2]
        )
