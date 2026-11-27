# Adapted from vllm-project/vllm-torchtpu (Apache-2.0).
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Packed FP8 index keys: four channels per word, followed by one FP32 scale.
Byte b of field f at slot t is stored at cache[page, f, b, t].
"""

WORD_BYTES = 4
TOKENS_PER_ROW = 128


def key_fields(head_dim: int) -> int:
    return head_dim // WORD_BYTES + 1


def cache_shape(pages: int, page_size: int, head_dim: int) -> tuple[int, ...]:
    fields = key_fields(head_dim)
    if head_dim % WORD_BYTES or page_size % TOKENS_PER_ROW:
        raise ValueError("Packed keys require whole words and pages of whole TPU rows")
    if pages * fields * page_size >= 1 << 31:
        raise ValueError("Packed FP8 cache word addresses exceed int32")
    return pages, fields, WORD_BYTES, page_size


def page_rows(shape: tuple[int, ...]) -> int:
    return shape[1] * shape[-1] // TOKENS_PER_ROW
