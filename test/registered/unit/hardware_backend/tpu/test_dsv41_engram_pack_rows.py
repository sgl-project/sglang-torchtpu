"""CPU tests for the NumPy Engram row gather (DSV4.1 host Engram)."""

import sys

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu.engram import _FP8_BLOCK_SIZE, _INVALID_SCALE, pack_rows

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _reference_pack(
    weight: torch.Tensor,
    scale: torch.Tensor,
    indices: torch.Tensor,
    *,
    row_start: int,
    padded_tokens: int,
    block_size: int = _FP8_BLOCK_SIZE,
) -> torch.Tensor:
    """Torch reference for the packed upload bytes."""
    if sys.byteorder != "little":
        raise RuntimeError("Packed Engram staging requires a little-endian host")
    if any(t.device.type != "cpu" for t in (weight, scale, indices)):
        raise ValueError("Engram row gathering requires CPU tensors")
    tokens, heads = indices.shape
    rows, dim = weight.shape
    if padded_tokens < tokens or dim % block_size or dim % 4:
        raise ValueError("Invalid Engram staging shape")
    blocks = dim // block_size
    if scale.shape != (rows, blocks):
        raise ValueError("Engram scale shape does not match the FP8 table")
    raw_bytes = heads * dim
    scale_bytes = heads * blocks
    # Interleave payload and scales per token, so graph padding has whole rows.
    row_bytes = raw_bytes + ((scale_bytes + 3) // 4) * 4
    packed = torch.zeros((padded_tokens, row_bytes), dtype=torch.uint8, device="cpu")
    packed[:, raw_bytes:].fill_(_INVALID_SCALE)
    if tokens and rows:
        local = indices.to(torch.int64) - row_start
        owned = (local >= 0) & (local < rows)
        local = local.clamp(0, rows - 1).reshape(-1)
        raw = weight.view(torch.uint8).index_select(0, local).view(tokens, -1)
        scales = scale.view(torch.uint8).index_select(0, local)
        scales = scales.view(tokens, heads, blocks)
        scales.masked_fill_(~owned.unsqueeze(-1), _INVALID_SCALE)
        packed[:tokens, :raw_bytes].copy_(raw)
        packed[:tokens, raw_bytes : raw_bytes + scale_bytes].copy_(
            scales.reshape(tokens, -1)
        )
    return packed.view(torch.int32).reshape(-1)


def _table(rows, dim=64, seed=1):
    gen = torch.Generator().manual_seed(seed)
    weight = torch.randint(0, 256, (rows, dim), dtype=torch.uint8, generator=gen)
    scale = torch.randint(100, 140, (rows, dim // 32), dtype=torch.uint8, generator=gen)
    return weight.view(torch.float8_e4m3fn), scale.view(torch.float8_e8m0fnu)


@pytest.mark.parametrize("row_start", [0, 50])
@pytest.mark.parametrize("tokens,padded", [(0, 4), (1, 1), (3, 8), (16, 16)])
def test_pack_rows_is_byte_identical_to_torch_gather(row_start, tokens, padded):
    weight, scale = _table(rows=100)
    gen = torch.Generator().manual_seed(tokens)
    indices = torch.randint(0, 200, (tokens, 4), dtype=torch.int64, generator=gen)
    expected = _reference_pack(
        weight, scale, indices, row_start=row_start, padded_tokens=padded
    )
    actual = pack_rows(
        weight, scale, indices, row_start=row_start, padded_tokens=padded
    )
    assert actual.dtype == expected.dtype == torch.int32
    assert torch.equal(actual, expected)
    empty_w, empty_s = _table(rows=0)
    assert torch.equal(
        pack_rows(empty_w, empty_s, indices, row_start=0, padded_tokens=padded),
        _reference_pack(empty_w, empty_s, indices, row_start=0, padded_tokens=padded),
    )
