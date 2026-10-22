# SPDX-License-Identifier: Apache-2.0
"""Stateless TPU sampling: shared hash bits, rounding boundaries and row identity."""

import random
import struct
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu import sampling_mask
from sglang_tpu.experimental.speculative.spec_ops import (
    accept_drafts_sampled,
    sample_tokens,
)
from sglang_tpu.sampler import TpuSampler
from sglang_tpu.sampling import seeded
from sglang_tpu.sampling.seeded import (
    DRAFT_COLUMN,
    hash_to_uniform,
    murmur_hash32,
    sample_categorical,
    sample_draft_with_seed,
    sample_with_seed,
    verify_coins,
)

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _reference_hash(seed, position, column):
    data = struct.pack(
        "<QII", seed & ((1 << 64) - 1), position & 0xFFFFFFFF, column & 0xFFFFFFFF
    )
    state = 0
    for (block,) in struct.iter_unpack("<I", data):
        block = block * 0xCC9E2D51 & 0xFFFFFFFF
        block = ((block << 15) | (block >> 17)) & 0xFFFFFFFF
        block = block * 0x1B873593 & 0xFFFFFFFF
        state ^= block
        state = ((state << 13) | (state >> 19)) & 0xFFFFFFFF
        state = (state * 5 + 0xE6546B64) & 0xFFFFFFFF
    state ^= len(data)
    state ^= state >> 16
    state = state * 0x85EBCA6B & 0xFFFFFFFF
    state ^= state >> 13
    state = state * 0xC2B2AE35 & 0xFFFFFFFF
    return state ^ (state >> 16)


class TestSeededSampling(CustomTestCase):
    def test_hash_preserves_seed_high_bits_and_unsigned_position_words(self):
        seeds = [0, 1, -1, -(1 << 63), (1 << 32) + 17]
        positions = [0, 31, (1 << 32) + 9, 19, (1 << 62) + 7]
        columns = [0, 1, 7, 31, (1 << 32) + 3]
        actual = murmur_hash32(
            torch.tensor(seeds), torch.tensor(positions), torch.tensor(columns)
        )
        expected = [
            [_reference_hash(seed, pos, col) for col in columns]
            for seed, pos in zip(seeds, positions)
        ]
        self.assertEqual(actual.tolist(), expected)

    def test_float32_coins_match_shared_float64_intermediate_at_boundaries(self):
        # 4294966655 exposes double rounding: rounding the rational directly
        # chooses the neighboring float32 instead of the shared two-stage result.
        maximum = (1 << 32) - 1
        values = {0, maximum, 4294966655}
        for exponent in range(33):
            center = 1 << exponent
            values.update(range(max(0, center - 1024), min(maximum + 1, center + 1024)))
        rng = random.Random(371)
        values.update(rng.randrange(maximum + 1) for _ in range(20000))
        hashed = torch.tensor(sorted(values), dtype=torch.int64)
        expected = (hashed.double() / maximum).float().clamp_max(1.0 - 2**-24)
        actual = hash_to_uniform(hashed)
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.all((actual >= 0) & (actual < 1)))

    def test_seeded_draws_follow_request_identity_across_padding_and_reorder(self):
        weights = torch.tensor([[0.1, 0.0, 0.9], [0.8, 0.2, 0.0], [0.0, 0.3, 0.7]])
        seeds = torch.tensor([4, -17, 1 << 40])
        positions = torch.tensor([8, 102, 78])
        baseline = sample_with_seed(weights, seeds, positions)
        order = torch.tensor([2, 0, 1])
        padded = sample_with_seed(
            torch.cat([weights[order], torch.ones(1, 3)]),
            torch.cat([seeds[order], torch.zeros(1, dtype=torch.int64)]),
            torch.cat([positions[order], torch.zeros(1, dtype=torch.int64)]),
        )
        self.assertTrue(torch.equal(padded[:3], baseline[order]))
        # The draft sampler draws per request and position too, from the draft column.
        actual = sample_tokens(
            weights.log(),
            torch.ones(3),
            torch.full((3,), 3),
            sampling_seed=seeds,
            positions=positions,
        )
        self.assertTrue(
            torch.equal(actual, sample_draft_with_seed(weights, seeds, positions))
        )

    def test_draft_draws_never_reuse_a_verify_or_target_coin(self):
        from sglang_tpu.experimental.speculative.block_spec_worker import (
            TpuBlockSpecWorker,
        )

        gamma, vocab = 3, 5
        num_verify_rows = gamma + 1
        seeds = torch.tensor([7, -3])
        seq_lens = torch.tensor([10, 20])
        keys = {"draft": set(), "verify": set(), "target": set()}
        site = []

        def recording(seed, positions, columns):
            keys[site[-1]].update(
                (s, p, c)
                for s, p in zip(seed.tolist(), positions.tolist())
                for c in columns.tolist()
            )
            return hash_to_uniform(murmur_hash32(seed, positions, columns))

        class _BlockDraft:
            def sample_drafts(self, *, block_hidden, bonus_tokens, sampler):
                del bonus_tokens
                drafts = [sampler(block_hidden[:, k], k) for k in range(gamma)]
                return torch.stack(drafts, dim=1), block_hidden

        worker = SimpleNamespace(
            _block_draft_worker=_BlockDraft(),
            _tp_sync=SimpleNamespace(sync=lambda _site, tokens: tokens),
        )
        with patch.object(seeded, "_uniforms", recording):
            # One row rejects at every draft index in turn, the other accepts all.
            for rejected_at in range(gamma + 1):
                site.append("target")
                sample_with_seed(torch.ones(2, vocab), seeds, seq_lens - 1)
                site.append("draft")
                TpuBlockSpecWorker._sample_draft_block(
                    worker,
                    torch.zeros(2, gamma, vocab),
                    torch.zeros(2, dtype=torch.int64),
                    torch.ones(2),
                    torch.full((2,), vocab),
                    sampling_seed=seeds,
                    seq_lens=seq_lens,
                )
                site.append("verify")
                verify_coins(seeds, seq_lens, num_verify_rows)
                seq_lens = seq_lens + torch.tensor([rejected_at, gamma]) + 1
        self.assertTrue(keys["draft"])
        self.assertEqual({column for *_, column in keys["draft"]}, {DRAFT_COLUMN})
        self.assertFalse(keys["draft"] & (keys["verify"] | keys["target"]))

    def test_verify_coins_key_a_chain_as_gpu_eagle_does(self):
        """A chain of N drafts verifies N + 1 rows. Given that width, as GPU
        EAGLE is given draft_token_num, draft j's rejection coin hashes column
        j and the final coin column N + 1."""
        num_drafts = 3
        seeds, seq_lens = torch.tensor([7, -3]), torch.tensor([10, 20])
        coins, final = verify_coins(seeds, seq_lens, num_drafts + 1)

        def column(c):
            columns = torch.tensor([c])
            return hash_to_uniform(murmur_hash32(seeds, seq_lens, columns))[:, 0]

        for j in range(num_drafts):
            torch.testing.assert_close(coins[:, j], column(j))
        torch.testing.assert_close(final, column(num_drafts + 1))

    def test_inverse_cdf_skips_zero_mass_bins_at_both_edges(self):
        weights = torch.tensor([[0.0, 2.0, 0.0, 1.0, 0.0]]).repeat(3, 1)
        actual = sample_categorical(weights, torch.tensor([0.0, 0.5, 1.0 - 2**-24]))
        self.assertEqual(actual.tolist(), [1, 1, 3])

    def test_seeded_acceptance_does_not_consume_ambient_rng(self):
        target = torch.tensor(
            [[2.0, 1.0, 0.0], [0.0, 1.0, 2.0], [1.0, 2.0, 0.0]]
        ).repeat(2, 1)
        drafts = torch.tensor([[0, 1], [1, 2]])
        draft_logits = torch.tensor([[[1.0, 2.0, 0.0], [2.0, 0.0, 1.0]]]).repeat(
            2, 1, 1
        )
        coins, final = verify_coins(torch.tensor([123, -9]), torch.tensor([10, 20]), 3)
        outputs = []
        with torch.random.fork_rng():
            for seed in (1, 999):
                torch.manual_seed(seed)
                before = torch.random.get_rng_state()
                outputs.append(
                    accept_drafts_sampled(
                        target,
                        drafts,
                        torch.tensor([2, 2]),
                        draft_logits,
                        torch.ones(2),
                        torch.full((2,), 3),
                        coins=coins,
                        final_sampling_coins=final,
                    )
                )
                self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        for left, right in zip(*outputs):
            self.assertTrue(torch.equal(left, right))

    def test_greedy_row_takes_argmax_beside_a_sampled_row(self):
        """top_k=1 keeps every token tied at the maximum, so without a greedy
        override a greedy row's token would depend on its companion's mode."""

        def tie_keeping_filter(probs, top_ks, top_ps, min_ps, **need):
            del top_ps, min_ps, need
            k = top_ks.clamp(1, probs.shape[1]).long() - 1
            descending = probs.sort(dim=-1, descending=True).values
            return torch.where(probs >= descending.gather(1, k[:, None]), probs, 0.0)

        sampler = object.__new__(TpuSampler)
        probs = torch.softmax(torch.tensor([[1.0, 1.0, -1.0]]).repeat(2, 1), dim=-1)
        sampled_second = set()
        for seed in range(16):
            info = SimpleNamespace(
                sampling_seed=torch.tensor([seed, seed]),
                top_ks=torch.tensor([1, 2]),
                is_any_greedy=True,
                top_ps=torch.ones(2),
                min_ps=torch.zeros(2),
                need_top_k_sampling=True,
                need_top_p_sampling=False,
                need_min_p_sampling=False,
                sampling_mask_batch_indices=torch.tensor([0, 1]),
            )
            with patch.object(sampling_mask, "tpu_filter_probs", tie_keeping_filter):
                tokens, capture = sampler._sample_from_probs(
                    probs, info, torch.tensor([1, 1]), False
                )
            self.assertEqual(tokens[0].item(), 0)
            self.assertEqual(capture.weights[0].nonzero().flatten().tolist(), [0])
            sampled_second.add(tokens[1].item())
        # The sampled row still draws from both tied tokens.
        self.assertEqual(sampled_second, {0, 1})

        # A batch without greedy rows samples the filtered probabilities as they are.
        info.top_ks, info.is_any_greedy = torch.tensor([2, 2]), False
        with (
            patch.object(sampling_mask, "tpu_filter_probs", tie_keeping_filter),
            patch.object(torch.Tensor, "argmax", side_effect=AssertionError),
        ):
            tokens, _ = sampler._sample_from_probs(
                probs, info, torch.tensor([1, 1]), False
            )
        expected = sample_with_seed(probs, info.sampling_seed, torch.tensor([1, 1]))
        self.assertEqual(tokens.tolist(), expected.tolist())


if __name__ == "__main__":
    unittest.main()
