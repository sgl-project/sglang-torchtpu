# SPDX-License-Identifier: Apache-2.0
"""Batched-RPA non-causal schedule and validation, run in Pallas interpret mode on CPU."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import numpy as np

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="base-a-test-cpu")

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp  # noqa: E402

from sglang_tpu.attention.pallas.batched_rpa import configs, schedule  # noqa: E402
from sglang_tpu.pallas_export import tpu_info_for_cpu_export  # noqa: E402

PAGE_SIZE = 256
BQ = 256
BKV = 512
WINDOW = 2048


def _tpu_info():
    # Interpret mode still sizes SMEM and lanes from the TPU7x hardware table.
    return (
        patch(
            "sglang_tpu.pallas_export.get_torch_tpu_device_kind", return_value="TPU v7"
        ),
        tpu_info_for_cpu_export(),
    )


def _cfgs(
    *,
    causal: bool,
    sliding_window: int | None,
    num_seqs: int,
    pages_per_seq: int,
    kv_layout: configs.KVLayout = configs.KVLayout.SEQ_ALONG_LANE,
    has_attention_sink: bool = False,
) -> configs.RpaConfigs:
    bf16 = jnp.dtype(jnp.bfloat16)
    return configs.RpaConfigs(
        block=configs.BlockSizes(
            bq_sz=BQ, bq_c_sz=BQ, bkv_sz=BKV, batch_size=1, n_buffer=3
        ),
        model=configs.ModelConfigs(
            num_q_heads=8,
            num_kv_heads=1,
            head_dim=128,
            sliding_window=sliding_window,
            sm_scale=1.0,
            mask_value=-1e30,
            has_attention_sink=has_attention_sink,
        ),
        serve=configs.ServingConfigs(
            num_seqs=num_seqs,
            page_size=PAGE_SIZE,
            total_q_tokens=2048,
            num_page_indices=num_seqs * pages_per_seq,
            dtype_q=bf16,
            dtype_kv=bf16,
            dtype_out=bf16,
            kv_layout=kv_layout,
            causal=causal,
        ),
        mode=configs.RpaCase.MIXED,
        vmem_limit_bytes=64 << 20,
    )


def _schedule_steps(
    seqs: tuple[tuple[int, int], ...], *, causal: bool, sliding_window: int | None
) -> list[tuple[int, int, int, int]]:
    """(seq, q block, k block, do_writeback) of every scheduled step."""
    kv_lens = [prefix + q_len for prefix, q_len in seqs] + [0]
    cu_q_lens = np.concatenate([[0], np.cumsum([q for _, q in seqs])]).tolist()
    cu_q_lens.append(cu_q_lens[-1])
    pages_per_seq = -(-max(kv_lens) // PAGE_SIZE)
    device_kind, tpu_info = _tpu_info()
    with device_kind, tpu_info:
        cfgs = _cfgs(
            causal=causal,
            sliding_window=sliding_window,
            num_seqs=len(kv_lens),
            pages_per_seq=pages_per_seq,
        )
        result = schedule.generate_rpa_metadata(
            jnp.asarray(cu_q_lens, jnp.int32),
            jnp.asarray(kv_lens, jnp.int32),
            jnp.asarray([0, 0, len(seqs)], jnp.int32),
            cfgs,
            interpret=True,
        )
        steps = int(result.actual_steps[0])
        fields = (result.s_idx, result.q_idx, result.k_idx, result.do_writeback)
        columns = [np.asarray(f.data)[:steps].tolist() for f in fields]
    return [tuple(step) for step in zip(*columns) if step[0] >= 0]


def _expected_k_blocks(
    prefix: int, q_len: int, q_idx: int, *, causal: bool, sliding_window: int | None
) -> range:
    kv_len = prefix + q_len
    first_q = prefix + q_idx * BQ
    start = 0
    if sliding_window is not None:
        start = max(0, first_q - sliding_window + 1) // BKV
    if causal:
        last_q = min(first_q + BQ, kv_len) - 1
        return range(start, last_q // BKV + 1)
    return range(start, -(-kv_len // BKV))


class TestNonCausalSchedule(unittest.TestCase):
    # Prefix off a page, q blocks spanning several k blocks; a short block after.
    SEQS = ((3000, 600), (437, 77))

    def test_every_q_block_runs_to_kv_len_and_q_block_zero_writes_back(self):
        """Non-causal q blocks visit every k block up to kv_len (the window sets
        only the first), and each new k block is written back exactly once."""
        for sliding_window in (None, WINDOW):
            with self.subTest(sliding_window=sliding_window):
                steps = _schedule_steps(
                    self.SEQS, causal=False, sliding_window=sliding_window
                )
                for s_idx, (prefix, q_len) in enumerate(self.SEQS):
                    for q_idx in range(-(-q_len // BQ)):
                        visited = [
                            k for s, q, k, _ in steps if (s, q) == (s_idx, q_idx)
                        ]
                        self.assertEqual(
                            visited,
                            list(
                                _expected_k_blocks(
                                    prefix,
                                    q_len,
                                    q_idx,
                                    causal=False,
                                    sliding_window=sliding_window,
                                )
                            ),
                        )
                    written = sorted(
                        (q, k) for s, q, k, wb in steps if s == s_idx and wb
                    )
                    new_k_blocks = range(prefix // BKV, -(-(prefix + q_len) // BKV))
                    self.assertEqual(written, [(0, k) for k in new_k_blocks])

    def test_causal_schedule_is_unchanged(self):
        steps = _schedule_steps(self.SEQS, causal=True, sliding_window=WINDOW)
        prefix, q_len = self.SEQS[0]
        for q_idx in range(-(-q_len // BQ)):
            visited = [k for s, q, k, _ in steps if (s, q) == (0, q_idx)]
            self.assertEqual(
                visited,
                list(
                    _expected_k_blocks(
                        prefix, q_len, q_idx, causal=True, sliding_window=WINDOW
                    )
                ),
            )


class TestNonCausalValidation(unittest.TestCase):
    def test_unsupported_combinations_raise(self):
        cases = (
            {"kv_layout": configs.KVLayout.HEAD_ALONG_SUBLANE},
            {"has_attention_sink": True},
        )
        device_kind, tpu_info = _tpu_info()
        with device_kind, tpu_info:
            for kwargs in cases:
                with self.subTest(**{k: str(v) for k, v in kwargs.items()}):
                    cfgs = _cfgs(
                        causal=False,
                        sliding_window=WINDOW,
                        num_seqs=2,
                        pages_per_seq=4,
                        **kwargs,
                    )
                    with self.assertRaisesRegex(NotImplementedError, "Non-causal"):
                        cfgs._validate_non_causal()


if __name__ == "__main__":
    unittest.main()
