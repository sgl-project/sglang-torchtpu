# SPDX-License-Identifier: Apache-2.0

import unittest
import weakref
from types import SimpleNamespace

import torch

from sglang.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sglang.srt.layers.logprob_processor import InputLogprobProcessor
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu.input_logprobs import (
    TpuInputLogprobProcessor,
    _copy_top_logprobs,
    get_top_logprobs_chunk,
)

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestInputTopLogprobs(unittest.TestCase):
    def test_bounded_pipeline_preserves_chunked_prompt_results(self):
        for fast in (False, True):
            for lengths, starts, chunk_size in (
                ([512], [1], 512),
                ([2048, 3, 16], [0, 3, 1], 2048),
                ([129, 4, 136], [1, 4, 2], 128),
            ):
                with self.subTest(fast=fast, lengths=lengths):
                    states = (
                        torch.arange(sum(lengths) * 17)
                        .reshape(-1, 17)
                        .remainder(37)
                        .float()
                    )
                    ranges = []
                    offset = 0
                    for length, start in zip(lengths, starts):
                        ranges.append(
                            (offset + min(start, length - 1), offset + length)
                        )
                        offset += length
                    results = []
                    for cls, skip_chunking in (
                        (InputLogprobProcessor, False),
                        (TpuInputLogprobProcessor, False),
                        (TpuInputLogprobProcessor, True),
                    ):
                        processor = (
                            cls(17, get_top_logprobs_chunk_fn=get_top_logprobs_chunk)
                            if cls is InputLogprobProcessor
                            else cls(17, skip_chunking=skip_chunking)
                        )
                        processor.enable_logprobs_chunk = True
                        processor.logprobs_chunk_size = chunk_size
                        processor.enable_fast_input_logprobs = fast
                        metadata = LogitsMetadata(
                            forward_mode=ForwardMode.EXTEND,
                            extend_return_logprob=True,
                            extend_return_top_logprob=True,
                            extend_token_ids_logprob=True,
                            extend_seq_lens_cpu=lengths,
                            extend_logprob_start_lens_cpu=starts,
                            extend_logprob_pruned_lens_cpu=[
                                length - start for length, start in zip(lengths, starts)
                            ],
                            top_logprobs_nums=[5, 0, 3][: len(lengths)],
                            token_ids_logprobs=[[3, 1, 3], None, [16]][: len(lengths)],
                            extend_input_logprob_token_ids_gpu=torch.arange(
                                sum(
                                    length - start
                                    for length, start in zip(lengths, starts)
                                )
                            ).remainder(17),
                        )
                        metadata.extend_input_logprob_token_ids_cpu = (
                            metadata.extend_input_logprob_token_ids_gpu
                        )
                        holder = SimpleNamespace(input_logprob_processor=processor)
                        pruned, before_norm, aux, samples, inputs, sequence = (
                            LogitsProcessor._get_pruned_states(
                                holder, states, states + 1, [states + 2], metadata
                            )
                        )
                        expected_pruned = torch.cat(
                            [states[start:end] for start, end in ranges]
                        )
                        torch.testing.assert_close(
                            pruned[: len(expected_pruned)],
                            expected_pruned,
                            rtol=0,
                            atol=0,
                        )
                        torch.testing.assert_close(
                            before_norm[: len(expected_pruned)],
                            expected_pruned + 1,
                            rtol=0,
                            atol=0,
                        )
                        torch.testing.assert_close(
                            aux[0][: len(expected_pruned)],
                            expected_pruned + 2,
                            rtol=0,
                            atol=0,
                        )
                        projected = []

                        def get_logits(hidden, head, meta, **kwargs):
                            # A completed chunk must not keep a vocabulary-sized
                            # output alive while the next projection allocates.
                            self.assertTrue(all(ref() is None for ref in projected))
                            logits = hidden.clone()
                            projected.append(weakref.ref(logits))
                            return logits

                        results.append(
                            processor.forward(
                                pruned,
                                samples,
                                inputs,
                                sequence,
                                None,
                                get_logits,
                                metadata,
                                skip_chunking_for_dp_attn=skip_chunking,
                            )
                        )
                    expected = results[0]
                    for actual in results[1:]:
                        torch.testing.assert_close(
                            actual[1], expected[1], rtol=0, atol=0
                        )
                        torch.testing.assert_close(
                            actual[0].token_logprobs,
                            expected[0].token_logprobs,
                            rtol=0,
                            atol=0,
                        )
                        for name in (
                            "top_logprobs_val",
                            "top_logprobs_idx",
                            "token_ids_logprobs_val",
                            "token_ids_logprobs_idx",
                        ):
                            self.assertEqual(
                                getattr(actual[0], name), getattr(expected[0], name)
                            )

    def test_chunks_and_width_buckets_preserve_values_and_stable_ties(self):
        for rows, k in ((0, 1), (1, 0), (127, 1), (129, 3), (511, 5), (512, 17)):
            with self.subTest(rows=rows, k=k):
                logits = torch.arange(rows * 17).reshape(rows, 17).remainder(7).float()
                expected_values, expected_indices = torch.sort(
                    logits, descending=True, stable=True
                )
                values, indices = _copy_top_logprobs(logits, k, None, None)
                self.assertEqual(values, expected_values[:, :k].tolist())
                self.assertEqual(indices, expected_indices[:, :k].tolist())

    def test_normalization_and_split_sequence_assembly(self):
        logits = torch.arange(129 * 17).reshape(129, 17).remainder(7).float()
        row_max = logits.max(dim=1).values
        row_log_sum = torch.logsumexp(logits - row_max[:, None], dim=1)
        values, indices = torch.sort(logits, descending=True, stable=True)
        normalized = (values - row_max[:, None]) - row_log_sum[:, None]
        for normalizer in (None, (row_max, row_log_sum)):
            for precomputed in (None, (values[:, :5], indices[:, :5])):
                with self.subTest(
                    normalized=normalizer is not None,
                    precomputed=precomputed is not None,
                ):
                    actual_values, actual_indices = _copy_top_logprobs(
                        logits, 5, normalizer, precomputed
                    )
                    expected = normalized if normalizer is not None else values
                    self.assertEqual(actual_values, expected[:, :5].tolist())
                    self.assertEqual(actual_indices, indices[:, :5].tolist())
        actual_values, actual_indices = [[[-99.0]]], [[[16]]]
        split = get_top_logprobs_chunk(
            logits, [3, 5], [130, 10], actual_values, actual_indices, 1
        )
        self.assertEqual(split, 0)
        self.assertEqual(actual_values, [[[-99.0], *values[:, :3].tolist()], []])
        self.assertEqual(actual_indices, [[[16], *indices[:, :3].tolist()], []])


if __name__ == "__main__":
    unittest.main()
