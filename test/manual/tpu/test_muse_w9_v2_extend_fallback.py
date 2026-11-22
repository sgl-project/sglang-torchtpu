# SPDX-License-Identifier: Apache-2.0
"""Token-order reassembly for the token-sharded sparse_v2 EXTEND fallback."""

from unittest import mock

import pytest

torch = pytest.importorskip("torch")
muse_spark = pytest.importorskip("sglang_tpu.models.muse_spark")


@pytest.mark.parametrize(
    "n_tok,tp", [(1, 8), (64, 8), (200, 8), (256, 8), (700, 8), (8192, 8), (300, 1)]
)
def test_gather_token_sharded_chunks_restores_token_order(n_tok, tp):
    hidden = 16
    step_b = 256
    # Ground truth per padded chunk, as the replicated op would return it.
    full_chunks, chunk_rows = [], []
    for s in range(0, n_tok, step_b):
        rem = min(step_b, n_tok - s)
        b_pad = 64 if rem <= 64 else (128 if rem <= 128 else 256)
        y = torch.arange(s, s + b_pad, dtype=torch.float32)[:, None].repeat(1, hidden)
        y[rem:] = -1.0  # padded rows must never surface
        full_chunks.append(y)
        chunk_rows.append((b_pad // tp, rem))

    # What each rank's token-sharded op returns for every chunk.
    def rank_local(r):
        return torch.cat(
            [y[r * rpr : (r + 1) * rpr] for y, (rpr, _) in zip(full_chunks, chunk_rows)]
        )

    my_chunks = [y[0:rpr] for y, (rpr, _) in zip(full_chunks, chunk_rows)]
    gathered = torch.cat([rank_local(r) for r in range(tp)], dim=0)
    with mock.patch.object(
        muse_spark, "tensor_model_parallel_all_gather", return_value=gathered
    ) as ag:
        out = muse_spark._gather_token_sharded_chunks(my_chunks, chunk_rows, tp)
    if tp > 1:
        ag.assert_called_once()
    assert out.shape == (n_tok, hidden)
    assert torch.equal(out[:, 0], torch.arange(n_tok, dtype=torch.float32))
