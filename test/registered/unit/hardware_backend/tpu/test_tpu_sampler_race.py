import unittest
from unittest.mock import Mock, patch

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.sampler import sampling_from_probs_torch
from sglang.test.ci.ci_register import register_tpu_ci
from sglang_tpu.layers.sampler import sample_exponential_race

register_tpu_ci(est_time=10, suite="stage-a-test-tpu-unit", chips=1)


class TestExponentialRaceSampler(unittest.TestCase):
    def test_unseeded_dispatch_and_seed_precedence(self):
        probs = torch.tensor([[0.25, 0.75], [0.6, 0.4]])
        expected = torch.tensor([[1], [0]], dtype=torch.int64)
        callback = Mock(return_value=expected)
        with envs.SGLANG_OPT_USE_GUMBEL_SAMPLE.override(False):
            with patch(
                "sglang.srt.layers.sampler.torch.multinomial", return_value=expected
            ) as default:
                result = sampling_from_probs_torch(probs)
                default.assert_called_once_with(probs, num_samples=1)
                self.assertEqual(result.tolist(), [1, 0])
                self.assertEqual(result.dtype, torch.int32)
        with patch("sglang.srt.layers.sampler.torch.multinomial") as default:
            result = sampling_from_probs_torch(probs, unseeded_sampling_fn=callback)
            callback.assert_called_once_with(probs)
            default.assert_not_called()
            self.assertEqual(result.tolist(), [1, 0])
        callback.reset_mock()
        seeds, positions = torch.tensor([10, 20]), torch.tensor([5, 6])
        with patch(
            "sglang.srt.layers.sampler.multinomial_with_seed", return_value=expected
        ) as seeded:
            result = sampling_from_probs_torch(
                probs, seeds, positions, unseeded_sampling_fn=callback
            )
            callback.assert_not_called()
            seeded.assert_called_once()
            logprobs, actual_seeds, actual_positions = seeded.call_args.args
            torch.testing.assert_close(logprobs, torch.log(probs))
            self.assertIs(actual_seeds, seeds)
            self.assertIs(actual_positions, positions)
            self.assertEqual(result.tolist(), [1, 0])
            self.assertEqual(result.dtype, torch.int32)

    def test_matches_the_target_distribution(self):
        torch.manual_seed(0)
        p = torch.tensor([0.5, 0.3, 0.15, 0.05, 0.0])
        n = 40000
        draws = sample_exponential_race(p.expand(n, -1).contiguous())
        self.assertEqual(draws.dtype, torch.int32)
        freq = torch.bincount(draws.long(), minlength=5).float() / n
        # 3-sigma band for n=40000 is about +-0.0075 on the largest entry
        torch.testing.assert_close(freq[:4], p[:4], atol=0.012, rtol=0)
        self.assertEqual(
            int(freq[4] * n), 0
        )  # zero-probability entries are never drawn

    def test_rows_are_independent_and_shape_preserving(self):
        torch.manual_seed(1)
        p = torch.zeros(6, 10)
        p[torch.arange(6), torch.arange(6)] = 1.0  # one-hot rows
        draws = sample_exponential_race(p)
        self.assertEqual(tuple(draws.shape), (6,))
        self.assertEqual(draws.tolist(), list(range(6)))

    def test_all_zero_row_does_not_produce_nan(self):
        draws = sample_exponential_race(torch.zeros(2, 8))
        self.assertEqual(tuple(draws.shape), (2,))
        self.assertTrue(bool(((draws >= 0) & (draws < 8)).all()))


if __name__ == "__main__":
    unittest.main()
