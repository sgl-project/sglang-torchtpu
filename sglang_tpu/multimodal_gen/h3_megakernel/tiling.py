# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""How the MiniMax H3 megakernel cuts a device's rows into tiles.

``Tiling`` holds upper bounds that ``fit`` cuts down to divide a shape;
``padded_rows`` picks the packed row count those tiles cut well.
"""

from __future__ import annotations

import dataclasses
import math

from sglang_tpu.multimodal_gen.h3_megakernel import reference as ref


@dataclasses.dataclass(frozen=True)
class Tiling:
    """Upper bounds on how the kernel cuts the work.

    ``fit`` turns each bound into the largest size under it that divides
    what it cuts, so a shape needs only 128-row granularity.

    Attributes:
      qkv_rows: Token rows phase A holds; QKV weights stream once per tile.
      mlp_rows: Token rows phase C holds; its weights stream once per tile.
      block_q: Query rows per attention tile.
      split_q_tiles: Also allow query tiles that cross a device boundary at
        their middle (a multiple of 256 that divides all rows, whose half
        divides a device's rows), e.g. 3840 at 9600 rows a device: each
        output tile then goes out as two halves. A device's rows may then be
        a multiple of 16 that no 128-row tile divides, its query tiles
        multiples of 128 with such halves (3,200 at 4,800 rows a device).
      block_kv: Key rows per attention DMA step (see ``kv_divisor``).
      key_tile: Key rows per attention DMA step whatever the rows (a multiple
        of 16), in place of ``block_kv``'s divisor (0): ``stack``'s dense walk
        takes a key tile that does not divide the rows (its q, k, v buffer
        pads to whole key tiles); other walks need one that does.
      block_kv_compute: Key rows per unrolled compute step within one.
      block_kv_compute_online: The same for online-softmax layers (1024 is
        3% faster at 74k rows, 5% slower at 32k).
      sub_rows: Rows per MXU call inside a tile.
      n_chunk: Output columns per MXU call.
      chunk_rows: Rows per pass of the norm, SwiGLU and residual loops.
      epilogue_rows: Rows per pass of phase A's q/k norm and RoPE.
      stage_slots: Steps phase A's q, k, v sends may stay in flight.
      vmem_limit_bytes: Mosaic's scoped-VMEM ceiling for the call.
    """

    qkv_rows: int = 768
    mlp_rows: int = 512
    block_q: int = 4096
    split_q_tiles: bool = False
    block_kv: int = 2560
    key_tile: int = 0
    block_kv_compute: int = 2048
    block_kv_compute_online: int = 2048
    sub_rows: int = 256
    n_chunk: int = 768
    chunk_rows: int = 128
    epilogue_rows: int = 64
    stage_slots: int = 16
    vmem_limit_bytes: int = 64 * 2**20

    def fit(self, config: ref.Config, rows: int, devices: int) -> Tiling:
        """These bounds cut down to divide ``rows`` local rows on ``devices``.

        Row counts are sublane dims (multiples of 16); ``block_q`` and the
        column tiles are lane dims (multiples of 128).

        Raises:
          ValueError: If ``rows`` is not a multiple of 128 (of 16 with
            ``split_q_tiles`` and a query tile that fits), or the model does
            not split over ``devices``.
        """
        off_lanes = rows % 128 != 0
        widths = (config.hidden, config.ffn)
        if (
            rows % 16
            or (off_lanes and not self.split_q_tiles)
            or any(width % 128 for width in widths)
            or self.key_tile % 16
            or config.heads % devices
            or config.head_dim != 128
            or config.rope_dims % 2
            or config.rope_dims > config.head_dim
        ):
            raise ValueError(
                f"MiniMax H3 megakernel takes a multiple of 128 rows a device and"
                f" hidden and ffn widths, 128 head dims, heads that split"
                f" {devices} ways and a key tile of 16-row multiples; got {rows}"
                f" rows, key tile {self.key_tile} and {config}"
            )
        qkv = divisor(rows, self.qkv_rows, 16)
        mlp = divisor(rows, self.mlp_rows, 16)
        block_q = 0 if off_lanes else divisor(rows, self.block_q, 128)
        if self.split_q_tiles:
            # Off whole 128-row multiples a device, 128-row steps whose halves
            # take whole 16-row tiles.
            step = 128 if off_lanes else 256
            halves = [
                b
                for b in range(step, min(self.block_q, rows * devices) + 1, step)
                if (rows * devices) % b == 0
                and rows % (b // 2) == 0
                and (b // 2) % 16 == 0
            ]
            block_q = max([block_q] + halves)
        if not block_q:
            raise ValueError(
                f"MiniMax H3 megakernel finds no query tile for {rows} rows a"
                f" device on {devices} devices under {self.block_q} rows"
            )
        return dataclasses.replace(
            self,
            qkv_rows=qkv,
            mlp_rows=mlp,
            block_q=block_q,
            block_kv=self.key_tile or kv_divisor(rows * devices, self.block_kv),
            # Elementwise passes walk both phases' row tiles.
            chunk_rows=divisor(math.gcd(qkv, mlp), self.chunk_rows, 16),
            epilogue_rows=divisor(qkv, self.epilogue_rows, 16),
        )

    def sub(self, tile: int) -> int:
        """Rows per MXU call within a ``tile``-row tile."""
        return divisor(tile, self.sub_rows, 16)


def divisor(size: int, cap: int, multiple: int) -> int:
    """Largest multiple of ``multiple`` that divides ``size``, under ``cap``."""
    top = max(min(size, cap), multiple)
    return max(d for d in range(multiple, top + 1, multiple) if size % d == 0)


def kv_divisor(size: int, cap: int) -> int:
    """``divisor(size, cap, 16)``, or the largest multiple of 256 under
    ``cap`` that divides ``size`` if that is at least half as large.

    A key tile off the MXU's 256 leaves part of a pass empty in every PV
    matmul: at 76,800 rows 2560-row tiles beat 1920-row ones by 4% of the
    step, and at 33,792 rows 1536 beat 2112 by 2%.
    """
    rows = divisor(size, cap, 16)
    if size % 256 or cap < 256:
        return rows
    aligned = divisor(size, cap, 256)
    return aligned if 2 * aligned >= rows else rows


def padded_rows(
    tokens: int,
    devices: int,
    tiling: Tiling,
    config: ref.Config | None = None,
) -> int:
    """The fewest packed rows, ``tokens`` or more, that ``stack`` tiles well.

    A device's rows grow by 128 until its row tiles reach 384 rows, its
    MXU calls 128 and its query tiles 256 (or their bounds), for up to 64
    steps; bounds no row count meets then take the fewest rows.
    """
    config = config or ref.Config()
    least = -(-tokens // (devices * 128)) * 128
    for rows in range(least, least + 64 * 128, 128):
        fitted = tiling.fit(config, rows, devices)
        floors = (
            (fitted.qkv_rows, tiling.qkv_rows, 384, 16),
            (fitted.mlp_rows, tiling.mlp_rows, 384, 16),
            (fitted.sub(fitted.qkv_rows), tiling.sub_rows, 128, 16),
            (fitted.sub(fitted.mlp_rows), tiling.sub_rows, 128, 16),
            (fitted.block_q, tiling.block_q, 256, 128),
        )
        # A bound reaches at most its largest multiple of the granule.
        if all(
            got >= min(max(bound // granule * granule, granule), floor, rows)
            for got, bound, floor, granule in floors
        ):
            return rows * devices
    return least * devices
