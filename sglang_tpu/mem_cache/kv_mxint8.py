# SPDX-License-Identifier: Apache-2.0
"""MXINT8 storage geometry, independent of backend and kernel implementations.

The cache is one INT8 page tensor ``[pages, R, tokens]``: ``2*H*align32(D)``
payload rows followed by one biased-exponent byte (``e + 127``) per
(head, K/V, group) row, padded with 127 to a whole 32-row int8 sublane tile.
"""

import torch

GROUP_SIZE = 32
EXPONENT_BIAS = 127
EXPONENT_PAD_BYTE = EXPONENT_BIAS
QUANT_DTYPE = torch.int8
ROW_ALIGNMENT = 32


def _validate_geometry(
    head_num: int, head_dim: int, page_size: int, group_size: int = GROUP_SIZE
) -> None:
    if group_size not in (16, 32, 64):
        raise ValueError("MXINT8 group_size must be 16, 32, or 64")
    if head_num <= 0 or head_dim <= 0 or head_dim % group_size:
        raise ValueError(
            "MXINT8 requires positive heads and head_dim divisible by group_size"
        )
    if group_size != GROUP_SIZE and head_dim not in (64, 128):
        raise ValueError("MXINT8 group_size 16/64 requires head_dim 64 or 128")
    if page_size < 128 or page_size & (page_size - 1):
        raise ValueError("MXINT8 page_size must be a power of two and at least 128")


def _align(rows: int) -> int:
    return (rows + ROW_ALIGNMENT - 1) // ROW_ALIGNMENT * ROW_ALIGNMENT


def payload_rows(head_num: int, head_dim: int) -> int:
    """Rows of INT8 payload; row = (2 * head + K0V1) * align32(D) + channel."""
    return 2 * head_num * _align(head_dim)


def scale_rows(head_num: int, head_dim: int, group_size: int = GROUP_SIZE) -> int:
    """Exponent-byte rows actually used; row = (2 * head + K0V1) * G + group."""
    return 2 * head_num * (head_dim // group_size)


def merged_rows(head_num: int, head_dim: int, group_size: int = GROUP_SIZE) -> int:
    return payload_rows(head_num, head_dim) + _align(
        scale_rows(head_num, head_dim, group_size)
    )


def merged_cache_shape(
    num_blocks: int,
    head_num: int,
    head_dim: int,
    page_size: int,
    group_size: int = GROUP_SIZE,
) -> tuple[int, int, int]:
    """[pages, merged rows, tokens] for the single INT8 MXINT8 page tensor."""
    _validate_geometry(head_num, head_dim, page_size, group_size)
    if num_blocks <= 0:
        raise ValueError("MXINT8 num_blocks must be positive (including dummy page)")
    return num_blocks, merged_rows(head_num, head_dim, group_size), page_size


# Page layouts: "merged" keeps payload rows then tile-padded exponent rows in a
# [pages, R, tokens] page; "blocked" stores a page as S/128 token blocks of
# payload rows followed by one fully used 32-row exponent tile in a
# [pages, R, 128] page (row r*4 + j of the tile = scale row r, token block j).
KV_LAYOUTS = ("merged", "blocked")
DEFAULT_KV_LAYOUT = "merged"
TOKEN_BLOCK = 128
BLOCKS_PER_WORD = 4
MAX_BLOCKED_PAGE_SIZE = TOKEN_BLOCK * BLOCKS_PER_WORD


def validate_kv_layout(layout: str, page_size: int | None = None) -> None:
    if layout not in KV_LAYOUTS:
        raise ValueError(
            f"MXINT8 KV layout must be one of {KV_LAYOUTS}, got {layout!r}"
        )
    if layout == "blocked" and page_size is not None:
        if (
            page_size < TOKEN_BLOCK
            or page_size & (page_size - 1)
            or page_size > MAX_BLOCKED_PAGE_SIZE
        ):
            raise ValueError(
                "MXINT8 blocked layout requires a power-of-two page_size between "
                f"{TOKEN_BLOCK} and {MAX_BLOCKED_PAGE_SIZE}"
            )


def blocked_page_rows(
    head_num: int, head_dim: int, page_size: int, group_size: int = GROUP_SIZE
) -> int:
    _validate_geometry(head_num, head_dim, page_size, group_size)
    validate_kv_layout("blocked", page_size)
    tile_rows = _align(BLOCKS_PER_WORD * scale_rows(head_num, head_dim, group_size))
    return page_size // TOKEN_BLOCK * payload_rows(head_num, head_dim) + tile_rows


def cache_bytes_per_token(
    head_num: int,
    head_dim: int,
    page_size: int,
    group_size: int = GROUP_SIZE,
    layout: str = DEFAULT_KV_LAYOUT,
) -> int:
    """Physical bytes per layer, excluding the extra dummy page's token count."""
    _validate_geometry(head_num, head_dim, page_size, group_size)
    validate_kv_layout(layout, page_size)
    if layout == "blocked":
        rows = blocked_page_rows(head_num, head_dim, page_size, group_size)
        return rows * TOKEN_BLOCK // page_size
    return merged_rows(head_num, head_dim, group_size)
