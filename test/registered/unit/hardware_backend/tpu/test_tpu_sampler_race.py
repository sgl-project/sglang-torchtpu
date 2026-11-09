import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.sampler import sampling_from_probs_torch
from sglang.srt.model_executor.forward_batch_info import ForwardMode, LogitsOutputMode
from sglang.test.ci.ci_register import register_tpu_ci
from sglang_tpu import sampler as tpu_sampler
from sglang_tpu.sampler import sample_exponential_race

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


_TP = 16


class _GatherAcrossRanks:
    """TP group stand-in: records each rank's [rows, 2] pairs, then returns them all."""

    def __init__(self):
        self.pairs = []
        self.gathered = None

    def all_gather(self, tensor, dim=0):
        assert dim == 0
        if self.gathered is None:
            self.pairs.append(tensor.clone())
            return torch.cat([tensor] * _TP, dim=0)
        return self.gathered


def _sample_on_every_rank(logits, temperatures, vocab_size, seed):
    """Run sample_vocab_shard on every simulated rank from the same RNG state."""
    width = -(-logits.shape[1] // _TP)
    padded = torch.nn.functional.pad(logits, (0, width * _TP - logits.shape[1]))
    group = _GatherAcrossRanks()

    def run_all():
        tokens = []
        for rank in range(_TP):
            start = rank * width
            valid = max(0, min(width, vocab_size - start))
            torch.manual_seed(seed)
            tokens.append(
                tpu_sampler.sample_vocab_shard(
                    padded[:, start : start + width],
                    (start, valid, vocab_size),
                    temperatures,
                )
            )
        return tokens

    with patch("sglang.srt.distributed.get_tp_group", return_value=group):
        run_all()
        group.gathered = torch.cat(group.pairs, dim=0)
        tokens = run_all()
    for other in tokens[1:]:
        torch.testing.assert_close(other, tokens[0])
    return tokens[0]


class TestVocabShardDraw(unittest.TestCase):
    def test_greedy_matches_full_vocab_argmax_including_cross_shard_ties(self):
        torch.manual_seed(0)
        vocab = 4096
        logits = torch.randn(8, vocab) * 3
        width = vocab // _TP
        logits[2, 7] = logits[2, width + 7] = logits[2].max() + 1.0
        tokens = _sample_on_every_rank(logits, None, vocab, seed=0)
        torch.testing.assert_close(tokens, torch.argmax(logits, dim=-1))
        self.assertEqual(tokens[2].item(), 7)

    def test_same_uniforms_draw_the_exponential_race_token(self):
        """Each rank reads its columns of one full-width uniform draw, so the token
        equals the replicated exponential race over softmax(z / T)."""
        torch.manual_seed(1)
        vocab = 4096
        logits = torch.randn(16, vocab) * 4
        temperatures = torch.full((16, 1), 0.7)
        for seed in range(4):
            tokens = _sample_on_every_rank(logits, temperatures, vocab, seed=seed)
            torch.manual_seed(seed)
            probs = torch.softmax(logits / temperatures, dim=-1)
            expected = tpu_sampler.sample_exponential_race(probs).to(torch.int64)
            torch.testing.assert_close(tokens, expected)

    def test_padding_columns_never_win(self):
        vocab = 4000
        width = -(-vocab // _TP)
        logits = torch.full((4, width * _TP), -10.0)
        logits[:, vocab:] = 1e9
        logits[:, 123] = 5.0
        greedy = _sample_on_every_rank(logits, None, vocab, seed=0)
        sampled = _sample_on_every_rank(logits, torch.ones(4, 1), vocab, seed=3)
        self.assertTrue(bool((greedy < vocab).all()))
        self.assertTrue(bool((sampled < vocab).all()))


def _batch(**sampling):
    info = SimpleNamespace(
        is_all_greedy=False,
        need_top_k_sampling=False,
        need_top_p_sampling=False,
        need_min_p_sampling=False,
        sampling_seed=None,
        has_custom_logit_processor=False,
        logit_bias=None,
        grammars=None,
        sampling_mask_batch_indices=None,
        acc_additive_penalties=None,
        acc_scaling_penalties=None,
        penalizer_orchestrator=None,
    )
    for key, value in sampling.items():
        setattr(info, key, value)
    return SimpleNamespace(
        spec_info=None,
        spec_algorithm=None,
        forward_mode=ForwardMode.DECODE,
        sampling_info=info,
        return_logprob=False,
        logits_output_mode=LogitsOutputMode.FULL_VOCAB,
    )


class TestVocabShardGate(unittest.TestCase):
    def setUp(self):
        patches = [
            patch.object(tpu_sampler, "_VOCAB_SHARD_SAMPLING", True),
            patch.object(
                tpu_sampler,
                "get_parallel",
                return_value=SimpleNamespace(tp_size=_TP, attn_dp_size=1),
            ),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def test_temperature_only_and_all_greedy_batches_take_the_shard(self):
        self.assertTrue(tpu_sampler.vocab_shard_sampling_for_batch(_batch()))
        self.assertTrue(
            tpu_sampler.vocab_shard_sampling_for_batch(
                _batch(is_all_greedy=True, need_top_k_sampling=True)
            )
        )

    def test_batches_that_read_or_edit_full_logits_keep_the_gather(self):
        refused = {
            "top_k": _batch(need_top_k_sampling=True),
            "top_p": _batch(need_top_p_sampling=True),
            "min_p": _batch(need_min_p_sampling=True),
            "seed": _batch(sampling_seed=torch.zeros(1)),
            "custom processor": _batch(has_custom_logit_processor=True),
            "logit bias": _batch(logit_bias=torch.zeros(1)),
            "grammar": _batch(grammars=[object()]),
            "additive penalty": _batch(acc_additive_penalties=torch.zeros(1)),
            "scaling penalty": _batch(acc_scaling_penalties=torch.zeros(1)),
            "penalizer": _batch(
                penalizer_orchestrator=SimpleNamespace(is_required=True)
            ),
        }
        for name, batch in refused.items():
            with self.subTest(name):
                self.assertFalse(tpu_sampler.vocab_shard_sampling_for_batch(batch))
        logprob = _batch()
        logprob.return_logprob = True
        self.assertFalse(tpu_sampler.vocab_shard_sampling_for_batch(logprob))
        extend = _batch()
        extend.forward_mode = ForwardMode.EXTEND
        self.assertFalse(tpu_sampler.vocab_shard_sampling_for_batch(extend))
        speculative = _batch()
        speculative.spec_info = object()
        self.assertFalse(tpu_sampler.vocab_shard_sampling_for_batch(speculative))

    def test_off_without_tp_or_with_dp_attention(self):
        for parallel in (
            SimpleNamespace(tp_size=1, attn_dp_size=1),
            SimpleNamespace(tp_size=_TP, attn_dp_size=2),
        ):
            with (
                self.subTest(parallel=parallel),
                patch.object(tpu_sampler, "get_parallel", return_value=parallel),
            ):
                self.assertFalse(tpu_sampler.vocab_shard_sampling_for_batch(_batch()))


if __name__ == "__main__":
    unittest.main()
