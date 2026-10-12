"""Batched attention schedule bounds for pipeline prefill and decode."""

import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu import pp_attention_capacity as capacity

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestResolvedPipelineGuard(CustomTestCase):
    def test_resolved_pipeline_enables_capacity_guard(self):
        from sglang_tpu.attention_pallas_rpa_batched import (
            BaseTpuPallasRpaAttnBackend,
            TpuPallasRpaBatchedAttnBackend,
        )

        with (
            get_context().override_server_args(pp_size=8) as raw,
            patch.object(BaseTpuPallasRpaAttnBackend, "__init__", return_value=None),
        ):
            self.assertEqual(raw.pp_size, 1)
            runner = NS(server_args=raw)
            backend = TpuPallasRpaBatchedAttnBackend(runner)
            self.assertIs(backend._pp_schedule_runner, runner)
            get_context().override("test.single_stage", pp_size=1)
            self.assertIsNone(
                TpuPallasRpaBatchedAttnBackend(runner)._pp_schedule_runner
            )


class TestScheduleLengths(CustomTestCase):
    def test_mixed_requests_share_one_schedule_budget(self):
        capacity.check_schedule_lengths([4, 8], [12, 8], (6, 4, 4), "mixed")
        with self.assertRaisesRegex(ValueError, "needs 7.*schedule 6"):
            capacity.check_schedule_lengths([4, 8], [16, 8], (6, 4, 4), "mixed")

    def test_decode_uses_its_own_query_and_kv_tiles(self):
        capacity.check_schedule_lengths([1, 1], [9, 17], (5, 1, 8), "decode")
        with self.assertRaisesRegex(ValueError, "Batched decode attention needs 5"):
            capacity.check_schedule_lengths([1, 1], [9, 17], (4, 1, 8), "decode")

    def test_partial_query_tile_counts_only_its_causal_kv_prefix(self):
        capacity.check_schedule_lengths([5], [12], (6, 4, 4), "mixed")
        with self.assertRaisesRegex(ValueError, "needs 6"):
            capacity.check_schedule_lengths([5], [12], (5, 4, 4), "mixed")

    def test_empty_rows_do_not_consume_pairs(self):
        capacity.check_schedule_lengths([0, 4, 0], [0, 4, 0], (1, 4, 4), "mixed")

    def test_invalid_lengths_are_rejected(self):
        for query, kv in [([4], [3]), ([-1], [4]), ([1, 2], [3])]:
            with self.subTest(query=query, kv=kv), self.assertRaises(ValueError):
                capacity.check_schedule_lengths(query, kv, (100, 4, 4), "mixed")


class TestHbmScheduleCapacity(CustomTestCase):
    def test_hbm_windows_and_pipeline_tail_are_included(self):
        cfg = NS(
            max_steps_ub=128,
            max_schedule_size_multiplier=16,
            block=NS(n_buffer=3, batch_size=1, bq_sz=256, bkv_sz=512),
        )
        self.assertEqual(capacity._capacity_from_config(cfg, 64, 256), (2044, 256, 512))

    def test_multiple_lanes_reserve_the_longest_sequence_imbalance(self):
        cfg = NS(
            max_steps_ub=128,
            max_schedule_size_multiplier=16,
            block=NS(n_buffer=3, batch_size=4, bq_sz=1, bkv_sz=512),
        )
        self.assertEqual(capacity._capacity_from_config(cfg, 65, 256), (8044, 1, 512))


class TestLiveBatchCapacity(CustomTestCase):
    def _batch(self, decode=False):
        return NS(
            forward_mode=NS(is_decode=lambda: decode),
            seq_lens=torch.zeros(8, dtype=torch.int64),
            input_ids=torch.zeros(32, dtype=torch.int64),
            seq_lens_cpu=torch.tensor([12, 8, 0, 0, 0, 0, 0, 0]),
            extend_seq_lens_cpu=[4, 8, 0, 0, 0, 0, 0, 0],
        )

    def test_capacity_uses_padded_shape_and_counts_only_active_requests(self):
        runner = object()
        backend = NS(_non_empty_batch_size=lambda batch: 2)
        batch = self._batch()
        with patch.object(
            capacity, "capacity_for_batch", return_value=(6, 4, 4)
        ) as get:
            capacity.check_batch_schedule(runner, backend, batch)
        get.assert_called_once_with(
            runner, backend, mode="mixed", num_seqs=8, total_q_tokens=32
        )

    def test_decode_length_bound_runs_before_device_metadata(self):
        backend = NS(_non_empty_batch_size=lambda batch: 2)
        with (
            patch.object(capacity, "capacity_for_batch", return_value=(1, 1, 8)),
            self.assertRaisesRegex(ValueError, "Batched decode attention needs 3"),
        ):
            capacity.check_batch_schedule(object(), backend, self._batch(decode=True))

    def test_idle_batch_does_not_create_a_kernel_shape(self):
        backend = NS(_non_empty_batch_size=lambda batch: 0)
        with patch.object(capacity, "capacity_for_batch") as get:
            capacity.check_batch_schedule(object(), backend, self._batch())
        get.assert_not_called()

    def test_missing_host_lengths_fail_without_a_device_read(self):
        batch = self._batch()
        batch.seq_lens_cpu = None
        with self.assertRaisesRegex(ValueError, "requires host sequence lengths"):
            capacity.check_batch_schedule(
                object(), NS(_non_empty_batch_size=lambda batch: 2), batch
            )


if __name__ == "__main__":
    unittest.main()
