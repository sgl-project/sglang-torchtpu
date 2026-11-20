"""MXINT8 pages on the Raiden pipeline transfer.

An MXINT8 layer page is one int8 tensor ``[pages, R, T]`` (``sglang_tpu.kv_mxint8``):
each (kv-head, K/V) pair's payload rows, then one exponent row per (kv-head,
K/V, group). The kv-heads are folded into R, the tiled second-minor axis, so a
kv-head slice of a page is not one byte run. The transfer stages pages as raw
device bytes, so the spans here address the page's tiled bytes.

int8 pages are tiled ``T(32,128)(4,1)``: 32-row x 128-token tiles of 4096
bytes, row band first, then column tile; within a tile, rows ``4s..4s+3`` of
token column ``c`` are the four bytes of the 32-bit word at ``(128 * s + c) * 4``.
Four rows of one 128-token column tile (a sublane group) are therefore one
contiguous 512-byte run. Since R is a multiple of 32 and T of 128, a page's
tiled bytes equal ``R * T``.

Two page layouts:

- merged, ``[pages, R, page_size]``: payload rows, then the exponent rows padded
  to a whole 32-row band;
- blocked, ``[pages, R, 128]``: ``page_size / 128`` token blocks of payload rows,
  then exponent rows padded to a whole 32-row band, where row ``4 r + j`` holds
  scale row ``r`` of token block ``j`` (a scale row's four blocks fill one
  sublane group).

A destination page holds its rank's kv-heads only. Its exponent padding rows are
never read: the decode kernels read exponent rows below ``scale_rows`` only, and
a write-back can overwrite padding bytes. The transfer must still cover every
destination byte exactly once, so the padding is covered by filler spans copied
from the source page's exponent rows: should a kernel ever decode a padding byte,
it reads a valid exponent rather than payload.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Protocol, runtime_checkable

import msgspec

from sglang_tpu import kv_mxint8
from sglang_tpu.disaggregation.raiden.reshard_schedule import head_slice_params

TILE_ROWS = 32
TILE_TOKENS = 128
SUBLANE_ROWS = 4
TILE_BYTES = TILE_ROWS * TILE_TOKENS
SUBLANE_BYTES = SUBLANE_ROWS * TILE_TOKENS
# (minor_to_major, tiles) of the int8 page tiling the spans assume.
INT8_TILING = ((2, 1, 0), ((TILE_ROWS, TILE_TOKENS), (SUBLANE_ROWS, 1)))


class Mxint8PageFormat(msgspec.Struct, frozen=True, kw_only=True):
    """Geometry of one pool's MXINT8 pages, without the per-rank kv-head count."""

    layout: str
    head_dim: int
    group_size: int
    page_size: int

    def __post_init__(self) -> None:
        # Validates the geometry and the layout's page_size bounds.
        kv_mxint8.cache_bytes_per_token(
            1, self.head_dim, self.page_size, self.group_size, self.layout
        )

    @property
    def width(self) -> int:
        return self.page_size if self.layout == "merged" else kv_mxint8.TOKEN_BLOCK

    @property
    def token_blocks(self) -> int:
        return 1 if self.layout == "merged" else self.page_size // kv_mxint8.TOKEN_BLOCK

    @property
    def head_payload_rows(self) -> int:
        return kv_mxint8.payload_rows(1, self.head_dim)

    @property
    def head_scale_rows(self) -> int:
        return kv_mxint8.scale_rows(1, self.head_dim, self.group_size)

    def rows(self, heads: int) -> int:
        if self.layout == "merged":
            return kv_mxint8.merged_rows(heads, self.head_dim, self.group_size)
        return kv_mxint8.blocked_page_rows(
            heads, self.head_dim, self.page_size, self.group_size
        )

    def page_bytes(self, heads: int) -> int:
        return self.rows(heads) * self.width

    def heads(self, tensor: Any) -> int:
        """The kv-head count of a page tensor of this format."""
        if len(tensor.shape) != 3 or int(tensor.shape[2]) != self.width:
            raise ValueError(
                f"Raiden pipeline MXINT8 pages must be [pages, R, {self.width}], got "
                f"{tuple(tensor.shape)}"
            )
        rows = int(tensor.shape[1])
        # rows(heads) grows with heads, and payload alone bounds it from above.
        for heads in range(rows // (self.token_blocks * self.head_payload_rows), 0, -1):
            if self.rows(heads) == rows:
                return heads
        raise ValueError(
            f"Raiden pipeline MXINT8 page rows {rows} match no kv-head count"
        )

    def check_head_slices(self) -> None:
        """Refuse a format whose kv-heads cannot be sliced as whole sublane groups."""
        if self.layout == "merged" and self.head_scale_rows % SUBLANE_ROWS:
            raise ValueError(
                "Raiden pipeline MXINT8 merged pages need an even head_dim / "
                "group_size to move to a tensor-parallel decode, so no two "
                "kv-heads share an exponent sublane group"
            )


@runtime_checkable
class Mxint8KVPool(Protocol):
    """A KV pool that keeps the KV cache configurator's MXINT8 pool arguments."""

    kv_cache_dtype_str: str
    kv_layout: str
    kv_group_size: int
    head_dim: int
    page_size: int


def page_format(pool: Any) -> Mxint8PageFormat | None:
    """The MXINT8 page format of a KV pool, or None for any other KV dtype."""
    if not isinstance(pool, Mxint8KVPool) or pool.kv_cache_dtype_str != "mxint8":
        return None
    return Mxint8PageFormat(
        layout=pool.kv_layout,
        head_dim=int(pool.head_dim),
        group_size=int(pool.kv_group_size),
        page_size=int(pool.page_size),
    )


def check_page_shape(tensor: Any) -> None:
    """Refuse an int8 page whose rows or tokens are not whole tiles, so its
    tiled bytes are its R * T elements."""
    _, rows, width = (int(n) for n in tensor.shape)
    if str(tensor.dtype) != "torch.int8" or rows % TILE_ROWS or width % TILE_TOKENS:
        raise ValueError(
            "Raiden pipeline MXINT8 pages must be int8 [pages, R, T] with R a "
            f"multiple of {TILE_ROWS} and T of {TILE_TOKENS}, got "
            f"{tensor.dtype} {tuple(tensor.shape)}"
        )


def check_tiling(layout: Any) -> None:
    """Refuse a materialized page layout other than the one the spans assume."""
    major, tiles, _ = layout
    found = (
        tuple(int(n) for n in major),
        tuple(tuple(int(n) for n in tile) for tile in tiles),
    )
    if found != INT8_TILING:
        raise ValueError(
            f"Raiden pipeline MXINT8 pages need the int8 tiling {INT8_TILING}, got "
            f"{found}"
        )


def _sublane_offset(row: int, width: int) -> int:
    """Tiled byte offset of a sublane-aligned ``row`` in column tile 0."""
    band, sublane = divmod(row, TILE_ROWS)
    return band * TILE_ROWS * width + sublane // SUBLANE_ROWS * SUBLANE_BYTES


def _row_runs(
    src_row: int, dst_row: int, rows: int, width: int
) -> list[tuple[int, int, int, int]]:
    """Byte runs moving ``rows`` sublane-aligned rows from ``src_row`` to
    ``dst_row``: (src offset, dst offset, size, count), the count repeating
    every column tile at a TILE_BYTES stride. A run stays within one row band
    of both pages; a one-tile-wide page is linear, so it needs one run."""
    if src_row % SUBLANE_ROWS or dst_row % SUBLANE_ROWS or rows % SUBLANE_ROWS:
        raise ValueError("Raiden pipeline MXINT8 rows must be whole sublane groups")
    if rows and width == TILE_TOKENS:
        return [
            (
                src_row * width,
                dst_row * width,
                rows // SUBLANE_ROWS * SUBLANE_BYTES,
                1,
            )
        ]
    runs = []
    while rows:
        step = min(
            rows, TILE_ROWS - src_row % TILE_ROWS, TILE_ROWS - dst_row % TILE_ROWS
        )
        runs.append(
            (
                _sublane_offset(src_row, width),
                _sublane_offset(dst_row, width),
                step // SUBLANE_ROWS * SUBLANE_BYTES,
                width // TILE_TOKENS,
            )
        )
        src_row, dst_row, rows = src_row + step, dst_row + step, rows - step
    return runs


def _page_runs(
    fmt: Mxint8PageFormat,
    *,
    src_heads: int,
    dst_heads: int,
    src_head: int,
    num_heads: int,
    dst_head: int,
    fill_padding: bool,
) -> list[tuple[int, int, int, int, int, int]]:
    """(src offset, dst offset, size, src stride, dst stride, count) runs that
    move kv-heads ``[src_head, src_head + num_heads)`` of a source page holding
    ``src_heads`` to ``dst_head`` onward of a destination page holding
    ``dst_heads``, plus the destination's exponent padding when asked."""
    width = fmt.width
    head_rows = fmt.head_payload_rows
    src_payload = kv_mxint8.payload_rows(src_heads, fmt.head_dim)
    dst_payload = kv_mxint8.payload_rows(dst_heads, fmt.head_dim)
    blocks = fmt.token_blocks
    # A kv-head's payload is whole row bands (2 * align32(D) rows), and the
    # bands of a page are consecutive bytes; blocked pages repeat it per block.
    runs = [
        (
            src_head * head_rows * width,
            dst_head * head_rows * width,
            num_heads * head_rows * width,
            src_payload * width,
            dst_payload * width,
            blocks,
        )
    ]
    # Exponent rows follow every block's payload; in the blocked tile each
    # scale row is one sublane group (its four token blocks).
    per_scale_row = 1 if fmt.layout == "merged" else kv_mxint8.BLOCKS_PER_WORD
    head_exp_rows = fmt.head_scale_rows * per_scale_row
    src_exp = blocks * src_payload
    dst_exp = blocks * dst_payload
    for src_off, dst_off, size, count in _row_runs(
        src_exp + src_head * head_exp_rows,
        dst_exp + dst_head * head_exp_rows,
        num_heads * head_exp_rows,
        width,
    ):
        runs.append((src_off, dst_off, size, TILE_BYTES, TILE_BYTES, count))
    if fill_padding:
        used = dst_exp + dst_heads * head_exp_rows
        padding = fmt.rows(dst_heads) - used
        src_exp_rows = src_heads * head_exp_rows
        while padding:
            rows = min(padding, src_exp_rows)
            for src_off, dst_off, size, count in _row_runs(src_exp, used, rows, width):
                runs.append((src_off, dst_off, size, TILE_BYTES, TILE_BYTES, count))
            used, padding = used + rows, padding - rows
    return runs


def slice_spans(
    fmt: Mxint8PageFormat,
    *,
    num_pages: int,
    src_heads: int,
    dst_heads: int,
    src_head: int,
    num_heads: int,
    dst_head: int,
    fill_padding: bool,
    dst_unit_ordinal: int,
) -> list[Any]:
    """Spans moving a kv-head slice of each of ``num_pages`` source pages into
    the matching destination page of rank ``dst_unit_ordinal``; ``fill_padding``
    also covers the destination's exponent padding, which exactly one source
    of each destination page must do."""
    if src_head + num_heads > src_heads or dst_head + num_heads > dst_heads:
        raise ValueError("Raiden pipeline MXINT8 head slice exceeds a page")
    dst_page = fmt.page_bytes(dst_heads)
    runs = _page_runs(
        fmt,
        src_heads=src_heads,
        dst_heads=dst_heads,
        src_head=src_head,
        num_heads=num_heads,
        dst_head=dst_head,
        fill_padding=fill_padding,
    )
    return [
        SimpleNamespace(
            src_block_ordinal=index,
            src_offset_bytes=src_off,
            dst_block_index=0,
            dst_offset_bytes=index * dst_page + dst_off,
            size_bytes=size,
            src_stride_bytes=src_stride if count > 1 else 0,
            dst_stride_bytes=dst_stride if count > 1 else 0,
            count=count,
            dst_unit_ordinal=dst_unit_ordinal,
        )
        for index in range(num_pages)
        for src_off, dst_off, size, src_stride, dst_stride, count in runs
    ]


def head_slice_part(
    fmt: Mxint8PageFormat,
    *,
    tag: str,
    tensor: Any,
    blocks: list[int],
    dst_attn_tp: int,
    total_kv_heads: int,
) -> Any:
    """Per-rank kv-head slices of one TP=1 producer pool's MXINT8 pages, in the
    pool-spans form of the pipeline transfer."""
    fmt.check_head_slices()
    if int(tensor.shape[1]) != fmt.rows(total_kv_heads):
        raise ValueError(
            f"Raiden pipeline MXINT8 pool {tag} pages of {int(tensor.shape[1])} "
            f"rows do not hold all {total_kv_heads} kv-heads"
        )
    spans = []
    for rank in range(dst_attn_tp):
        src_head, num_heads, dst_head, _, dst_heads = head_slice_params(
            src_tp=1,
            dst_tp=dst_attn_tp,
            src_rank=0,
            dst_rank=rank,
            total_kv_heads=total_kv_heads,
        )
        spans.extend(
            slice_spans(
                fmt,
                num_pages=len(blocks),
                src_heads=total_kv_heads,
                dst_heads=dst_heads,
                src_head=src_head,
                num_heads=num_heads,
                dst_head=dst_head,
                # The source that fills slot 0 of a destination page also
                # covers its exponent padding.
                fill_padding=dst_head == 0,
                dst_unit_ordinal=rank,
            )
        )
    return SimpleNamespace(
        tag=tag,
        block_ids=list(blocks),
        # Ranks that share a kv-head each receive it.
        declared_bytes=sum(span.size_bytes * span.count for span in spans),
        dst_space_version=1,
        spans=spans,
    )
