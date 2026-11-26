# SPDX-License-Identifier: Apache-2.0
"""How the Wan megakernel cuts its work (see ``kernel``)."""

from __future__ import annotations

import dataclasses
import math

# VMEM of phases C and C2 beyond their buffers and the layer vectors
# (Tiling.check): modulation rows, internal scratch and spills, ~3.2 MiB in
# Mosaic's allocations on v7x.
GEMM_VMEM_EXTRA = 3328 << 10


@dataclasses.dataclass(frozen=True)
class Tiling:
    # Phase A (norm, modulation, QKV, q/k norm, RoPE, sends): rows a tile.
    qkv_rows: int = 1024
    # Phase C (projections, cross-attention, MLP): rows a tile.
    mlp_rows: int = 1024
    # Column width of every streamed weight tile of a [H, N] projection
    # (QKV, output projections, cross q, MLP up) and row height of the MLP
    # down projection's tiles.
    cols: int = 512
    # Phase B (self-attention): query rows a tile, key rows a tile, and key
    # rows a score block inside a key tile.
    block_q: int = 4096
    block_kv: int = 2048
    sub_kv: int = 1024
    # Rows an elementwise pass holds at once.
    chunk_rows: int = 64
    # Row-chunk DMAs a weight tile streams in (several in flight at once).
    weight_splits: int = 4
    # Phase A row tiles whose q and k buffers can be in flight (1: a row tile's
    # q sends finish before the next one's q tiles are stored).
    qk_slots: int = 1
    # Phase C's MLP: "acc" (per row tile, (up, down) tile pairs into an f32
    # accumulator) or "twopass" (phase C2: up + GELU into a [rows, F] buffer,
    # then the down projection by down_cols-wide output tiles).
    mlp_mode: str = "twopass"
    down_cols: int = 256
    # Cross-attention key counts with their own code path (``kernel.stack``): a
    # prompt of n tokens, zero-padded to the text length, attends to the smallest
    # of these above n (its n keys and copies of the padding key, see
    # ``pack.cross_keys``), a longer one to every text row. Each count adds a
    # cross-attention loop to phase C's row-tile loop.
    cross_keys: tuple[int, ...] = (64, 128, 256)
    # Cross-attention heads a step of a K2/V2 head group's loop (1, 2, 4 or 8).
    # Phase C's row-tile loop holds the loops of every key count, so it grows with
    # this: Wan2.2-A14B at 4 is ~30K bundles, under the ~32K a hot loop's body
    # fits one instruction overlay in; at 8 (no step loop) ~43K.
    cross_step: int = 4
    # Each streamed phase C projection (output, cross q, cross output) and MLP up
    # projection hands its tile's epilogue to the next tile's step, in that step's
    # basic block after its matmul is issued, so the epilogue runs under the MXU
    # stream instead of after the tile's last matmul; a tile's step only rounds
    # its result into a [mlp_rows, cols] bf16 buffer (the cross q into its place).
    # The same values: the same bits.
    pipelined_epilogues: bool = False
    # Mosaic's DMA bounds checks (a trap on an out-of-bounds copy, scalar work at
    # every copy the kernel starts or waits). Off: the same copies without them.
    dma_bounds_checks: bool = True
    # Mosaic's semaphore checks. Off: the same waits and signals without them.
    semaphore_checks: bool = True
    # A masked attention sub-block (the key tile holding the last real row) on
    # the fixed path (the path ~98% of items take) subtracts a per-key column
    # from its scores, 0 at a real key and +inf at a padding key, instead of
    # selecting its scores: a real score is unchanged (s - 0 is s, signed zeros
    # included, also fused with the scale's multiply), a padding one -inf, whose
    # exp2 is the select's 0. The same probabilities, so the same bits. The
    # rowfix and online paths keep the select.
    mask_scores: bool = False
    # Each attention item's running sum and output are zeroed in the item
    # before's epilogue, after its output is read and ahead of its send, where
    # the stores issue beside the output's transposes, instead of at the item's
    # start ahead of its first key tile (the first item's before the item loop).
    # The fixed path then starts from the same zero state: the same bits.
    zero_ahead: bool = False
    vmem_limit_bytes: int = 64 * 1024 * 1024

    def opt_suffix(self) -> str:
        """The op name's mark of the exact layout options in effect."""
        return (
            ("_pe" if self.pipelined_epilogues else "")
            + ("" if self.dma_bounds_checks else "_nb")
            + ("" if self.semaphore_checks else "_ns")
            + ("_ms" if self.mask_scores else "")
            + ("_za" if self.zero_ahead else "")
        )

    def row_unit(self) -> int:
        """Rows a device holds must be a multiple of this."""
        return math.lcm(self.qkv_rows, self.mlp_rows, self.block_q)

    def check(
        self,
        rows: int,
        devices: int,
        hidden: int,
        ffn: int,
        heads: int,
        text_len: int = 512,
    ) -> None:
        """Raises unless this tiling cuts ``rows`` rows a device evenly and fits
        VMEM."""
        total = rows * devices

        def need(cond, msg):
            if not cond:
                raise ValueError(f"{self}: {msg}")

        need(
            rows % self.row_unit() == 0,
            f"{rows} rows a device: need a multiple of qkv_rows, mlp_rows and block_q",
        )
        need(
            self.qkv_rows % self.chunk_rows == 0
            and self.mlp_rows % self.chunk_rows == 0,
            "chunk_rows must divide the row tiles",
        )
        need(
            self.cols % 128 == 0 and hidden % self.cols == 0 and ffn % self.cols == 0,
            f"cols must be a multiple of 128 dividing {hidden} and {ffn}",
        )
        need(total % self.block_kv == 0, f"block_kv must divide the {total} rows")
        need(
            self.block_kv % self.sub_kv == 0 and self.sub_kv % 128 == 0,
            "sub_kv must divide block_kv",
        )
        need(heads % devices == 0, f"{devices} devices must divide {heads} heads")
        need(
            self.cols % (8 * self.weight_splits) == 0
            and hidden % self.weight_splits == 0,
            "weight_splits must divide the weight tiles' rows",
        )
        need(self.mlp_mode in ("acc", "twopass"), f"mlp_mode {self.mlp_mode}")
        need(
            all(nk > 0 and nk % 16 == 0 for nk in self.cross_keys)
            and list(self.cross_keys) == sorted(set(self.cross_keys)),
            "cross_keys must be ascending multiples of 16 (bf16 K/V row tiles)",
        )
        need(self.cross_step in (1, 2, 4, 8), "cross_step must be 1, 2, 4 or 8")
        # Phase B holds the most VMEM: the f32 [sub_kv, block_q] score block, the
        # query, output and staging buffers a query row, the key/value tiles and
        # small vectors (matches Mosaic's count on v7x to ~0.1 MiB, e.g. 69.4 MiB
        # at 19456 / 512 / 512, which fails to compile).
        phase_b = (
            4 * self.sub_kv * self.block_q
            + 1632 * self.block_q
            + 1024 * self.block_kv
            + (680 << 10)
        )
        need(
            phase_b <= self.vmem_limit_bytes - (4 << 20),
            f"phase B needs ~{phase_b / 2**20:.1f} MiB of VMEM",
        )
        need(
            hidden % self.down_cols == 0
            and self.down_cols % 128 == 0
            and ffn % self.weight_splits == 0,
            "down_cols must divide the hidden size",
        )
        # Phases C and C2. C: a, n2 / a2 / n3, x for two row tiles (the next one's
        # prefetched), the weight ring and two K2/V2 head groups. C2: h [mlp_rows,
        # ffn], beside pass 1's up ring (2 slots) and n3, which stays through pass 2
        # to take the next row tile's, or pass 2's down ring and x tiles. A
        # [rows, 1] f32 column takes 512 bytes a row. With the f32 layer vectors,
        # the cross keys' weights and GEMM_VMEM_EXTRA this matches Mosaic's count
        # on v7x for phase C to ~0.1 MiB and overestimates C2 by ~3 MiB.
        tc = self.mlp_rows
        row = 2 * tc * hidden
        group = next(c for c in (8, 4, 2, 1) if heads % c == 0)
        # pipelined_epilogues: a rounded [mlp_rows, cols] tile in phase C and in
        # C2's first pass.
        stage = 2 * tc * self.cols if self.pipelined_epilogues else 0
        phase_c = (
            4 * row
            + 4 * hidden * self.cols
            + 8 * group * text_len * (hidden // heads)
            + 512 * tc
            + stage
        )
        if self.mlp_mode == "acc":
            phase_c += 4 * tc * hidden + 2 * tc * self.cols + 4 * self.cols * hidden
            phase_c2 = 0
        else:
            h = 2 * tc * ffn
            down = 4 * ffn * self.down_cols + 6 * tc * self.down_cols + row
            up = 4 * hidden * self.cols + row + stage
            phase_c2 = h + max(up, down)
        key_weights = 512 * max(
            [nk for nk in self.cross_keys if nk < text_len], default=8
        )
        gemm = (
            max(phase_c, phase_c2)
            + 4 * (12 * hidden + ffn)
            + key_weights
            + GEMM_VMEM_EXTRA
        )
        need(
            gemm <= self.vmem_limit_bytes - (4 << 20),
            f"phases C and C2 need ~{gemm / 2**20:.1f} MiB of VMEM",
        )


# Phase B on v7x (Wan2.2-A14B at 1280x720x81, 75600 tokens, ms a layer): the
# per-score cost is flat from 4096-row to ~10K-row query tiles (4096 / 2048 /
# 1024 over 20480 rows a device: 60.8; 9728 / 1024 / 512 over 19456 rows: 57.7,
# the difference being the 5% more rows), larger query tiles cost more (19456 /
# 512 / 256: 59.0) and so do larger key-tile steps (9728 / 2048 / 512: 58.5).
MAX_BLOCK_Q = 10240
MAX_STEP_SCORES = 10240 * 1024  # block_q x block_kv


def _attention_tiles(rows: int, devices: int, base: Tiling) -> Tiling:
    """The largest query tile dividing ``rows`` (up to MAX_BLOCK_Q), then the
    largest key tile whose steps stay within MAX_STEP_SCORES."""
    bq = max(d for d in range(512, min(rows, MAX_BLOCK_Q) + 1, 512) if rows % d == 0)
    total = rows * devices
    bkv = next(
        k for k in (2048, 1024, 512) if total % k == 0 and bq * k <= MAX_STEP_SCORES
    )
    return dataclasses.replace(base, block_q=bq, block_kv=bkv, sub_kv=bkv // 2)


def for_tokens(
    tokens: int, devices: int, base: Tiling | None = None
) -> tuple[Tiling, int]:
    """The tiling to run ``tokens`` real rows over ``devices`` with, and its rows
    a device (padding included). Rows a device are a multiple of the GEMM row
    tiles; the fewest that do are taken unless their best query tile is small
    (a prime-heavy count), then a few more rows buy a larger one: the choice
    minimizes rows x per-score cost (1 from 4096-row query tiles, 1.2 at 2048,
    1.6 below; the real keys are the same either way)."""
    base = base or Tiling()
    unit = math.lcm(base.qkv_rows, base.mlp_rows)
    fewest = -(-tokens // (devices * unit)) * unit
    best = None
    for step in (unit, 2048, 4096, 8192):
        rows = -(-fewest // math.lcm(unit, step)) * math.lcm(unit, step)
        tiling = _attention_tiles(rows, devices, base)
        q = tiling.block_q
        cost = rows * (1.0 if q >= 4096 else 1.2 if q >= 2048 else 1.6)
        if best is None or cost < best[0]:
            best = (cost, tiling, rows)
    return best[1], best[2]
