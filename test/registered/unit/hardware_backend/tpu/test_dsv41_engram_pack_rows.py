# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the NumPy Engram row pack/unpack and signed_sqrt (DSV4.1 host Engram)."""

import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu.models.deepseek_v41.engram_host import (
    _FP8_BLOCK_SIZE,
    _INVALID_SCALE,
    local_row_ids,
    pack_rows,
    signed_sqrt,
    staged_lookup,
    unpack_rows,
)

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


def _reference_dequant(
    weight: torch.Tensor, scale: torch.Tensor, indices: torch.Tensor
) -> torch.Tensor:
    rows = weight.view(torch.uint8)[indices].view(torch.float8_e4m3fn).float()
    scales = scale.view(torch.uint8)[indices].to(torch.int32)
    power = (scales << 23).view(torch.float32)
    values = rows.unflatten(-1, (scales.shape[-1], -1)) * power.unsqueeze(-1)
    return values.flatten(-2).to(torch.bfloat16)


def _table(rows, dim=64, seed=1):
    gen = torch.Generator().manual_seed(seed)
    # Exclude FP8 NaN bytes (0x7F, 0xFF) for finite dequant checks.
    weight = torch.randint(0, 127, (rows, dim), dtype=torch.uint8, generator=gen)
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


def test_unpack_rows_matches_reference_and_zeros_invalid_nan_fp8():
    weight, scale = _table(rows=32, dim=128, seed=7)
    # Put FP8 NaN bytes (0x7F and 0xFF) in row 0 to ensure invalid rows still
    # unpack to exact 0.0 (not NaN * 0.0 == NaN).
    weight.view(torch.uint8)[0, :64] = 0x7F
    weight.view(torch.uint8)[0, 64:] = 0xFF
    gen = torch.Generator().manual_seed(11)
    # Mix valid indices (1..31) and unowned indices (99) with non-4B-aligned
    # scale_bytes (heads=3, blocks=4 -> 12B, and heads=3, dim=64 -> 6B).
    indices = torch.randint(1, 32, (5, 3), dtype=torch.int64, generator=gen)
    packed = pack_rows(weight, scale, indices, row_start=0, padded_tokens=8)
    unpacked = unpack_rows(packed, tokens=8, heads=3, dim=128)
    expected_valid = _reference_dequant(weight, scale, indices)
    assert unpacked.dtype == torch.bfloat16
    assert torch.equal(unpacked[:5], expected_valid)
    assert torch.all(unpacked[5:] == 0)

    # Unowned row clamps to local row 0 (containing FP8 NaN) with _INVALID_SCALE.
    unowned_indices = torch.tensor([[0, 99, 5]], dtype=torch.int64)
    packed_unowned = pack_rows(
        weight, scale, unowned_indices, row_start=1, padded_tokens=2
    )
    unpacked_unowned = unpack_rows(packed_unowned, tokens=2, heads=3, dim=128)
    assert not torch.isnan(unpacked_unowned).any()
    assert torch.all(unpacked_unowned[0, 0] == 0)
    assert torch.all(unpacked_unowned[0, 1] == 0)
    assert torch.equal(
        unpacked_unowned[0, 2],
        _reference_dequant(weight, scale, torch.tensor([[4]]))[0, 0],
    )
    assert torch.all(unpacked_unowned[1] == 0)

    # Also test non-4-byte-aligned scale bytes (heads=3, dim=64 -> scale_bytes=6).
    w64, s64 = _table(rows=16, dim=64, seed=3)
    idx64 = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int64)
    p64 = pack_rows(w64, s64, idx64, row_start=0, padded_tokens=4)
    u64 = unpack_rows(p64, tokens=4, heads=3, dim=64)
    assert torch.equal(u64[:2], _reference_dequant(w64, s64, idx64))
    assert torch.all(u64[2:] == 0)


def test_staged_lookup_supports_full_and_late_layer_tail_subset():
    weight, scale = _table(rows=40, dim=64, seed=13)
    indices = torch.tensor(
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
        dtype=torch.int64,
    )
    packed = pack_rows(weight, scale, indices, row_start=0, padded_tokens=6)
    embedding = SimpleNamespace(dim=64, tpu_hash_index=0)
    forward_batch = SimpleNamespace(tpu_engram_staged=(packed,))

    full_ids = local_row_ids(6, 1, 4, "cpu")
    full_out = staged_lookup(embedding, full_ids[:, 0], forward_batch)
    expected_full = unpack_rows(packed, tokens=6, heads=4, dim=64)
    assert torch.equal(full_out, expected_full)

    # Simulate late_layer_tail slicing dim 0 to tokens [3, 1, 5].
    tail_rows = torch.tensor([3, 1, 5], dtype=torch.int64)
    tail_ids = full_ids[tail_rows]
    tail_out = staged_lookup(embedding, tail_ids[:, 0], forward_batch)
    assert torch.equal(tail_out, expected_full[tail_rows])


def test_signed_sqrt_preserves_negative_zero():
    dot = torch.tensor([-4.0, -0.0, 0.0, 4.0, -1e-12, 1e-12], dtype=torch.float32)
    clamp = 1e-6
    got = signed_sqrt(dot, clamp)
    expected = torch.copysign(dot.abs().clamp_min(clamp).sqrt(), dot)
    assert torch.equal(got.view(torch.int32), expected.view(torch.int32))
    assert got[1].item() < 0.0 and got[2].item() > 0.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-p", "no:cacheprovider"]))
