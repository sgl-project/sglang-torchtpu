# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for TPU DeepSeek-V4.1 Engram hash sentinel masking."""

import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.kernels.ops.embeddings.engram_hash import MODE_DECODE, MODE_EXTEND
from sglang.srt.managers.schedule_batch import MM_PAD_SHIFT_VALUE
from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu.models.deepseek_v41.hooks import tpu_dsv41_torch_hash_ids

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _fake_hasher(image_token_id=None, max_ngram_size=3):
    return SimpleNamespace(
        max_ngram_size=max_ngram_size,
        image_token_id=image_token_id,
        pad_id=0,
        token_map=torch.arange(64, dtype=torch.int32),
        multipliers=torch.ones((2, max_ngram_size), dtype=torch.int64),
        primes=torch.full((2, max_ngram_size - 1, 2), 17, dtype=torch.int64),
        offsets=torch.zeros((2, (max_ngram_size - 1) * 2), dtype=torch.int64),
    )


@pytest.mark.parametrize("image_token_id", [None, 99])
def test_torch_hash_ids_blocks_negative_and_mm_sentinels(image_token_id):
    captured = {}

    def _capture(tokens, blocked, pad_id, token_map, multipliers, primes, offsets):
        captured["tokens"] = tokens.clone()
        captured["blocked"] = blocked.clone()
        return torch.zeros((tokens.shape[0], 2, 4), dtype=torch.int32)

    hasher = _fake_hasher(image_token_id=image_token_id, max_ngram_size=3)
    # Token 0: normal at pos 5 with normal history [10, 11] -> unblocked [F, F, F]
    # Token 1: negative sentinel -1 at pos 6 with history [10, 11] -> all blocked [T, T, T]
    # Token 2: normal token 12 with negative predecessor in history [-5, 11] -> [F, F, T]
    # Token 3: normal token 13 with MM sentinel in history [10, MM_PAD+1] -> [F, T, T]
    input_ids = torch.tensor([12, -1, 12, 13], dtype=torch.int32)
    positions = torch.tensor([5, 6, 7, 8], dtype=torch.int32)
    history = torch.tensor(
        [
            [10, 11],
            [10, 11],
            [-5, 11],
            [10, MM_PAD_SHIFT_VALUE + 1],
        ],
        dtype=torch.int32,
    )
    with patch(
        "sglang.srt.layers.engram.compute_engram_hash_ids", side_effect=_capture
    ):
        tpu_dsv41_torch_hash_ids(
            lambda *a: None,
            hasher,
            input_ids,
            positions,
            MODE_DECODE,
            history,
            4,
            1,
            None,
            None,
        )

    assert captured["blocked"].tolist() == [
        [False, False, False],
        [True, True, True],
        [False, False, True],
        [False, True, True],
    ]


def test_torch_hash_ids_extend_preserves_normal_positions_mask():
    captured = {}

    def _capture(tokens, blocked, pad_id, token_map, multipliers, primes, offsets):
        captured["blocked"] = blocked.clone()
        return torch.zeros((tokens.shape[0], 2, 4), dtype=torch.int32)

    hasher = _fake_hasher(image_token_id=None, max_ngram_size=3)
    input_ids = torch.tensor([5, 6, 7], dtype=torch.int32)
    positions = torch.tensor([0, 1, 2], dtype=torch.int32)
    history = torch.zeros((1, 2), dtype=torch.int32)
    row = torch.zeros(3, dtype=torch.int32)
    starts = torch.zeros(1, dtype=torch.int32)
    with patch(
        "sglang.srt.layers.engram.compute_engram_hash_ids", side_effect=_capture
    ):
        tpu_dsv41_torch_hash_ids(
            lambda *a: None,
            hasher,
            input_ids,
            positions,
            MODE_EXTEND,
            history,
            3,
            1,
            row,
            starts,
        )

    assert captured["blocked"].tolist() == [
        [False, True, True],
        [False, False, True],
        [False, False, False],
    ]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-p", "no:cacheprovider"]))
