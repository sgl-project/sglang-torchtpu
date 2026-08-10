"""Unit tests for the TPU torch fallbacks used by EAGLE3 non-greedy spec decode."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.speculative.eagle_utils import (
    _top_k_renorm_prob_torch,
    _top_p_renorm_prob_torch,
    _tree_speculative_sampling_target_only_torch,
)


def _sample_output0(target_probs_row, draft_token, n_samples, num_draft, seed=0):
    """Run the topk=1 rejection sampler ``n_samples`` times (batched) and return
    the position-0 output token for each sample.
    """
    torch.manual_seed(seed)
    bs = n_samples
    vocab = target_probs_row.shape[-1]

    candidates = torch.zeros(bs, num_draft, dtype=torch.long)
    candidates[:, 1:] = draft_token  # the proposed draft token at every step

    target_probs = target_probs_row.view(1, 1, vocab).expand(bs, num_draft, vocab)
    target_probs = target_probs.contiguous()

    retrive_index = (
        torch.arange(bs).unsqueeze(1) * num_draft + torch.arange(num_draft).unsqueeze(0)
    ).to(torch.long)

    uniform_samples = torch.rand(bs, num_draft)
    uniform_final = torch.rand(bs)

    predicts = torch.zeros(bs * num_draft, dtype=torch.long)
    accept_index = torch.full((bs, num_draft), -1, dtype=torch.int32)
    accept_token_num = torch.zeros(bs, dtype=torch.int32)
    draft_probs = torch.zeros_like(target_probs, dtype=torch.float32)

    _tree_speculative_sampling_target_only_torch(
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accept_token_num,
        candidates=candidates,
        retrive_index=retrive_index,
        uniform_samples=uniform_samples,
        uniform_samples_for_final_sampling=uniform_final,
        target_probs=target_probs,
        draft_probs=draft_probs,
        threshold_single=1.0,
        threshold_acc=1.0,
        topk=1,
    )
    # retrive_index[i, 0] == i * num_draft, so output position 0 of sample i is
    # predicts.view(bs, num_draft)[i, 0].
    return predicts.view(bs, num_draft)[:, 0]


def _empirical(tokens, vocab):
    counts = torch.bincount(tokens, minlength=vocab).to(torch.float64)
    return counts / counts.sum(), counts


def _run_sampler(
    candidates,
    target_probs,
    uniform_samples,
    uniform_final,
    threshold_single=1.0,
    threshold_acc=1.0,
):
    bs, num_draft = candidates.shape
    retrive_index = (
        torch.arange(bs).unsqueeze(1) * num_draft + torch.arange(num_draft).unsqueeze(0)
    ).to(torch.long)
    predicts = torch.full((bs * num_draft,), -1, dtype=torch.int32)
    accept_index = torch.full((bs, num_draft), -1, dtype=torch.int32)
    accept_token_num = torch.full((bs,), -1, dtype=torch.int32)
    draft_probs = torch.zeros_like(target_probs, dtype=torch.float32)

    _tree_speculative_sampling_target_only_torch(
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accept_token_num,
        candidates=candidates,
        retrive_index=retrive_index,
        uniform_samples=uniform_samples,
        uniform_samples_for_final_sampling=uniform_final,
        target_probs=target_probs,
        draft_probs=draft_probs,
        threshold_single=threshold_single,
        threshold_acc=threshold_acc,
        topk=1,
    )
    return predicts, accept_index, accept_token_num, draft_probs


class TestEagleSamplingTorchFallbacks(unittest.TestCase):
    def _assert_matches_target(self, p, draft_token, num_draft, n=40000):
        out = _sample_output0(p, draft_token, n, num_draft)
        freq, counts = _empirical(out, p.numel())
        # Deterministic (seeded). Correct code lands well within ~0.01 of p;
        # a wrong residual / accept rule shifts a bin by >= ~0.05.
        max_dev = (freq - p.to(torch.float64)).abs().max().item()
        self.assertLess(
            max_dev,
            0.02,
            f"output distribution deviates from target: max|freq-p|={max_dev:.4f}, "
            f"freq={freq.tolist()}, p={p.tolist()}",
        )
        # Chi-square as a second, scale-free guard (df = vocab-1).
        exp = p.to(torch.float64) * n
        chi2 = (((counts - exp) ** 2) / exp).sum().item()
        self.assertLess(chi2, 50.0, f"chi-square too large: {chi2:.1f}")

    def test_rejection_sampling_preserves_target_distribution(self):
        p = torch.tensor([0.40, 0.25, 0.15, 0.10, 0.07, 0.03])
        # High-prob, mid, and low-prob draft tokens -> accept-heavy and
        # reject-heavy regimes; all must yield output ~ p.
        for draft_token in (0, 3, 5):
            with self.subTest(draft_token=draft_token):
                self._assert_matches_target(p, draft_token, num_draft=2)

    def test_rejection_sampling_multi_step_chain(self):
        # A longer chain (num_draft=4) exercises accept-then-continue; the
        # position-0 marginal is still target_probs[:, 0].
        p = torch.tensor([0.5, 0.2, 0.15, 0.1, 0.05])
        self._assert_matches_target(p, draft_token=1, num_draft=4)

    def test_rejection_sampling_respects_zero_support(self):
        # A token with zero target probability must never be emitted.
        p = torch.tensor([0.6, 0.4, 0.0, 0.0])
        out = _sample_output0(p, draft_token=0, n_samples=20000, num_draft=2)
        self.assertEqual((out == 2).sum().item(), 0)
        self.assertEqual((out == 3).sum().item(), 0)

    def test_rejection_sampling_deterministic_with_seed(self):
        p = torch.tensor([0.4, 0.3, 0.2, 0.1])
        a = _sample_output0(p, 1, 1000, num_draft=2, seed=123)
        b = _sample_output0(p, 1, 1000, num_draft=2, seed=123)
        self.assertTrue(torch.equal(a, b))

    def test_rejection_sampling_exact_outputs_and_bookkeeping(self):
        candidates = torch.tensor([[0, 1, 2, 3], [0, 4, 3, 2]], dtype=torch.long)
        target_probs = torch.zeros(2, 4, 6, dtype=torch.float32)
        target_probs[0, 0] = torch.tensor([0.05, 0.80, 0.05, 0.04, 0.03, 0.03])
        target_probs[0, 1] = torch.tensor([0.10, 0.20, 0.20, 0.30, 0.10, 0.10])
        target_probs[0, 2:] = 1.0 / 6.0
        target_probs[1, 0] = torch.tensor([0.20, 0.10, 0.10, 0.10, 0.20, 0.30])
        target_probs[1, 1:] = 1.0 / 6.0
        uniform_samples = torch.tensor(
            [
                [0.70, 0.90, 0.05, 0.05],  # accept token 1, then reject token 2
                [0.90, 0.05, 0.05, 0.05],  # reject token 4 immediately
            ],
            dtype=torch.float32,
        )
        uniform_final = torch.tensor([0.50, 0.70], dtype=torch.float32)

        predicts, accept_index, accept_token_num, draft_probs = _run_sampler(
            candidates, target_probs, uniform_samples, uniform_final
        )

        self.assertEqual(
            predicts.view(2, 4).tolist(), [[1, 3, -1, -1], [5, -1, -1, -1]]
        )
        self.assertEqual(accept_index.tolist(), [[0, 1, -1, -1], [4, -1, -1, -1]])
        self.assertEqual(accept_token_num.tolist(), [1, 0])
        self.assertTrue(torch.equal(draft_probs, torch.zeros_like(draft_probs)))

    def test_rejection_sampling_threshold_acc_zero_matches_cuda_cap(self):
        candidates = torch.tensor([[0, 1]], dtype=torch.long)
        target_probs = torch.tensor([[[0.5, 0.0, 0.5], [0.1, 0.8, 0.1]]])
        uniform_samples = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
        uniform_final = torch.tensor([0.05], dtype=torch.float32)

        predicts, accept_index, accept_token_num, draft_probs = _run_sampler(
            candidates,
            target_probs,
            uniform_samples,
            uniform_final,
            threshold_single=0.0,
            threshold_acc=0.0,
        )

        self.assertEqual(predicts.view(1, 2).tolist(), [[1, 0]])
        self.assertEqual(accept_index.tolist(), [[0, 1]])
        self.assertEqual(accept_token_num.tolist(), [1])
        self.assertTrue(torch.equal(draft_probs, torch.zeros_like(draft_probs)))

    def test_rejection_sampling_all_accepted_uses_full_bonus_distribution(self):
        candidates = torch.tensor([[0, 1, 2]], dtype=torch.long)
        target_probs = torch.zeros(1, 3, 4, dtype=torch.float32)
        target_probs[0, 0] = torch.tensor([0.0, 1.0, 0.0, 0.0])
        target_probs[0, 1] = torch.tensor([0.0, 0.0, 1.0, 0.0])
        target_probs[0, 2] = torch.tensor([0.10, 0.20, 0.30, 0.40])
        uniform_samples = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
        uniform_final = torch.tensor([0.95], dtype=torch.float32)

        predicts, accept_index, accept_token_num, draft_probs = _run_sampler(
            candidates, target_probs, uniform_samples, uniform_final
        )

        self.assertEqual(predicts.view(1, 3).tolist(), [[1, 2, 3]])
        self.assertEqual(accept_index.tolist(), [[0, 1, 2]])
        self.assertEqual(accept_token_num.tolist(), [2])
        self.assertTrue(torch.equal(draft_probs, torch.zeros_like(draft_probs)))

    def test_rejection_sampling_padded_batch_rows_do_not_change_real_rows(self):
        real_candidates = torch.tensor([[0, 1, 2, 3], [0, 4, 3, 2]], dtype=torch.long)
        padded_candidates = torch.tensor(
            [
                [0, 1, 2, 3],
                [0, 4, 3, 2],
                [0, 1, 1, 1],  # dummy padded row
                [0, 2, 2, 2],  # dummy padded row
            ],
            dtype=torch.long,
        )
        real_target_probs = torch.zeros(2, 4, 6, dtype=torch.float32)
        real_target_probs[0, 0] = torch.tensor([0.05, 0.80, 0.05, 0.04, 0.03, 0.03])
        real_target_probs[0, 1] = torch.tensor([0.10, 0.20, 0.20, 0.30, 0.10, 0.10])
        real_target_probs[0, 2:] = 1.0 / 6.0
        real_target_probs[1, 0] = torch.tensor([0.20, 0.10, 0.10, 0.10, 0.20, 0.30])
        real_target_probs[1, 1:] = 1.0 / 6.0
        padded_target_probs = torch.cat(
            [real_target_probs, torch.full((2, 4, 6), 1.0 / 6.0)], dim=0
        )
        real_uniform_samples = torch.tensor(
            [
                [0.70, 0.90, 0.05, 0.05],
                [0.90, 0.05, 0.05, 0.05],
            ],
            dtype=torch.float32,
        )
        padded_uniform_samples = torch.cat(
            [real_uniform_samples, torch.full((2, 4), 0.99)], dim=0
        )
        real_uniform_final = torch.tensor([0.50, 0.70], dtype=torch.float32)
        padded_uniform_final = torch.cat(
            [real_uniform_final, torch.tensor([0.10, 0.20], dtype=torch.float32)]
        )

        real_predicts, real_accept_index, real_accept_token_num, _ = _run_sampler(
            real_candidates,
            real_target_probs,
            real_uniform_samples,
            real_uniform_final,
        )
        padded_predicts, padded_accept_index, padded_accept_token_num, _ = _run_sampler(
            padded_candidates,
            padded_target_probs,
            padded_uniform_samples,
            padded_uniform_final,
        )

        self.assertEqual(
            padded_predicts.view(4, 4)[:2].tolist(),
            real_predicts.view(2, 4).tolist(),
        )
        self.assertEqual(padded_accept_index[:2].tolist(), real_accept_index.tolist())
        self.assertEqual(
            padded_accept_token_num[:2].tolist(), real_accept_token_num.tolist()
        )
        # Padded rows are processed as dummy rows. This is fine only if downstream
        # slices/ignores them, but they must stay inside their own flat row ranges.
        self.assertEqual(
            padded_accept_index[2:].tolist(),
            [[8, -1, -1, -1], [12, -1, -1, -1]],
        )

    def test_rejection_sampling_token_axis_padding_after_rejection_is_marked_invalid(
        self,
    ):
        candidates = torch.tensor([[0, 1, 2, 3, 4, 5]], dtype=torch.long)
        target_probs = torch.full((1, 6, 6), 1.0 / 6.0, dtype=torch.float32)
        target_probs[0, 0] = torch.tensor([0.20, 0.10, 0.20, 0.20, 0.20, 0.10])
        uniform_samples = torch.tensor([[0.90, 0.05, 0.05, 0.05, 0.05, 0.05]])
        uniform_final = torch.tensor([0.40], dtype=torch.float32)

        predicts, accept_index, accept_token_num, draft_probs = _run_sampler(
            candidates, target_probs, uniform_samples, uniform_final
        )

        self.assertEqual(predicts.view(1, 6).tolist(), [[2, -1, -1, -1, -1, -1]])
        self.assertEqual(accept_index.tolist(), [[0, -1, -1, -1, -1, -1]])
        self.assertEqual(accept_token_num.tolist(), [0])
        self.assertTrue(torch.equal(draft_probs, torch.zeros_like(draft_probs)))

    def test_top_k_renorm(self):
        probs = torch.tensor([[0.4, 0.3, 0.2, 0.1]])
        out = _top_k_renorm_prob_torch(probs, torch.tensor([2]))
        expected = torch.tensor([[0.4, 0.3, 0.0, 0.0]]) / 0.7
        self.assertTrue(torch.allclose(out, expected, atol=1e-6))

    def test_top_k_renorm_batched_with_ties_and_clamp(self):
        probs = torch.tensor(
            [
                [0.4, 0.3, 0.2, 0.1],
                [0.4, 0.3, 0.3, 0.0],
                [0.1, 0.2, 0.3, 0.4],
            ]
        )
        out = _top_k_renorm_prob_torch(probs, torch.tensor([1, 2, 99]))
        expected = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.4, 0.3, 0.3, 0.0],
                [0.1, 0.2, 0.3, 0.4],
            ]
        )
        self.assertTrue(torch.allclose(out, expected, atol=1e-6))

    def test_top_p_renorm(self):
        probs = torch.tensor([[0.5, 0.3, 0.15, 0.05]])
        out = _top_p_renorm_prob_torch(probs, torch.tensor([0.7]))
        # nucleus keeps the tokens whose prefix mass is < 0.7 -> {0.5, 0.3}.
        expected = torch.tensor([[0.5, 0.3, 0.0, 0.0]]) / 0.8
        self.assertTrue(torch.allclose(out, expected, atol=1e-6))

    def test_top_p_renorm_batched(self):
        probs = torch.tensor(
            [
                [0.5, 0.3, 0.15, 0.05],
                [0.4, 0.35, 0.15, 0.10],
                [0.1, 0.2, 0.3, 0.4],
            ]
        )
        out = _top_p_renorm_prob_torch(probs, torch.tensor([0.7, 0.4, 1.0]))
        expected = torch.tensor(
            [
                [0.5, 0.3, 0.0, 0.0],
                [0.4, 0.0, 0.0, 0.0],
                [0.1, 0.2, 0.3, 0.4],
            ]
        )
        expected = expected / expected.sum(dim=-1, keepdim=True)
        self.assertTrue(torch.allclose(out, expected, atol=1e-6))


class TestGreedyVerifyTorchEquivalence(unittest.TestCase):
    """The fused greedy verify must match the staged-buffer implementation."""

    @staticmethod
    def _reference(next_token_logits, candidates, retrieve_index):
        # Independent staged-buffer oracle for the functional implementation.
        bs, ntv = candidates.shape
        predicts = torch.zeros((bs * ntv,), dtype=torch.int32)
        accept_index = torch.full((bs, ntv), -1, dtype=torch.int32)
        accept_token_num = torch.empty((bs,), dtype=torch.int32)
        target_predict = torch.argmax(next_token_logits, dim=-1).reshape(bs, ntv)
        predicts[: bs * ntv] = target_predict.flatten().to(predicts.dtype)

        matches = target_predict[:, :-1] == candidates[:, 1:]
        if matches.shape[1] > 0:
            all_match = matches.all(dim=1)
            first_mismatch = (~matches).to(torch.int32).argmax(dim=1)
            num_correct = torch.where(all_match, ntv - 1, first_mismatch)
        else:
            num_correct = torch.zeros((bs,), dtype=torch.long)
        accept_len = num_correct + 1

        ri_valid = retrieve_index != -1
        col_idx = torch.arange(ntv, dtype=torch.long)
        within_accept = col_idx.unsqueeze(0) < accept_len.unsqueeze(1)
        invalid_within = within_accept & ~ri_valid
        has_invalid = invalid_within.any(dim=1)
        first_invalid = invalid_within.to(torch.int32).argmax(dim=1)
        accept_len = torch.where(has_invalid, first_invalid, accept_len)
        num_correct = accept_len - 1

        copy_mask = col_idx.unsqueeze(0) < accept_len.unsqueeze(1)
        accept_index[:bs] = torch.where(
            copy_mask,
            retrieve_index.to(accept_index.dtype),
            torch.tensor(-1, dtype=accept_index.dtype),
        )
        accept_token_num[:bs] = num_correct.to(accept_token_num.dtype)
        return predicts, accept_index, accept_token_num

    def test_matches_reference_on_random_cases(self):
        from sglang.srt.speculative.eagle_utils import _verify_tree_greedy_torch

        torch.manual_seed(0)
        vocab = 32
        for bs, ntv in ((3, 1), (1, 4), (2, 4), (3, 2), (2, 8)):
            for trial in range(8):
                logits = torch.randn(bs * ntv, vocab)
                candidates = torch.randint(0, vocab, (bs, ntv), dtype=torch.int32)
                retrieve_index = torch.arange(bs * ntv, dtype=torch.int32).reshape(
                    bs, ntv
                )
                if trial % 3 == 1 and ntv > 1:
                    # Exercise the -1 clamp inside the accepted prefix.
                    retrieve_index[:, -1] = -1

                want_p, want_ai, want_n = self._reference(
                    logits, candidates, retrieve_index
                )
                got_p, got_ai, got_lens = _verify_tree_greedy_torch(
                    logits, candidates, retrieve_index
                )

                with self.subTest(bs=bs, ntv=ntv, trial=trial):
                    self.assertTrue(torch.equal(got_p, want_p))
                    self.assertTrue(torch.equal(got_ai, want_ai))
                    # fused returns accept_len (incl. bonus); reference counts drafts
                    self.assertTrue(torch.equal(got_lens - 1, want_n))

    def test_handles_single_verify_token(self):
        from sglang.srt.speculative.eagle_utils import _verify_tree_greedy_torch

        logits = torch.randn(3, 16)
        candidates = torch.randint(0, 16, (3, 1), dtype=torch.int32)
        retrieve_index = torch.arange(3, dtype=torch.int32).reshape(3, 1)

        predicts, accept_index, accept_lens = _verify_tree_greedy_torch(
            logits, candidates, retrieve_index
        )

        self.assertEqual(predicts.shape, (3,))
        self.assertEqual(accept_index.shape, (3, 1))
        # Only the bonus token is ever accepted when there are no drafts.
        self.assertEqual(accept_lens.tolist(), [1, 1, 1])

    def test_logical_logits_view_leaves_the_bucketed_tail_untouched(self):
        from sglang.srt.speculative.spec_info import SpecInput

        logits = torch.zeros((16, 8))
        spec_info = SimpleNamespace(
            padding_state=SimpleNamespace(
                logical_num_tokens=12,
                physical_num_tokens=16,
            )
        )

        logical_logits = SpecInput.logical_token_view(spec_info, logits)
        logical_logits.add_(1)

        self.assertEqual(logical_logits.shape, (12, 8))
        self.assertTrue(torch.all(logits[:12] == 1))
        self.assertTrue(torch.all(logits[12:] == 0))


class TestVerifyBookkeepingEquivalence(unittest.TestCase):
    """Fused bookkeeping must match seq-len add + gather + bonus fill."""

    def test_matches_the_unfused_sequence(self):
        from sglang.srt.speculative.eagle_utils import fused_verify_bookkeeping

        torch.manual_seed(1)
        for bs, width, ndt in ((1, 4, 4), (3, 4, 4), (2, 3, 6)):
            seq_lens = torch.randint(8, 64, (bs,), dtype=torch.int32)
            accept_lens = torch.randint(1, width + 1, (bs,), dtype=torch.int32)
            predict = torch.randint(0, 1000, (bs * ndt,), dtype=torch.int32)
            accept_index = torch.randint(0, bs * ndt, (bs, width), dtype=torch.int32)

            got_seq, got_bonus = fused_verify_bookkeeping(
                seq_lens, accept_lens, predict, accept_index
            )

            want_seq = seq_lens + accept_lens
            accept_tokens = predict[accept_index]
            want_bonus = torch.empty_like(accept_lens, dtype=torch.int32)
            for i in range(bs):
                want_bonus[i] = accept_tokens.reshape(-1)[
                    width * i + int(accept_lens[i]) - 1
                ]

            with self.subTest(bs=bs, width=width):
                self.assertTrue(torch.equal(got_seq, want_seq))
                self.assertTrue(torch.equal(got_bonus, want_bonus))


class TestDraftExtendSelectFused(unittest.TestCase):
    """Fused draft-extend select must match the eager gather + argmax."""

    def test_matches_eager_select_and_argmax(self):
        from sglang.srt.speculative.eagle_utils import (
            fused_draft_extend_select_argmax_func,
        )

        torch.manual_seed(3)
        for bs, ndt, vocab, hdim in ((1, 4, 64, 8), (3, 4, 128, 16), (2, 6, 32, 4)):
            logits = torch.randn(bs * ndt, vocab)
            hidden = torch.randn(bs * ndt, hdim)
            accept_lens = torch.randint(1, ndt + 1, (bs,), dtype=torch.int32)
            select_index = (
                torch.arange(0, bs * ndt, ndt, dtype=torch.int32) + accept_lens - 1
            ).long()

            sel_l, sel_h, tp, ti = fused_draft_extend_select_argmax_func(
                logits, hidden, select_index
            )

            want_l = logits[select_index]
            want_h = hidden[select_index]
            want_ti = torch.argmax(want_l, dim=-1, keepdim=True)
            with self.subTest(bs=bs, ndt=ndt):
                self.assertTrue(torch.equal(sel_l, want_l))
                self.assertTrue(torch.equal(sel_h, want_h))
                self.assertTrue(torch.equal(ti, want_ti))
                self.assertTrue(torch.all(tp == 1.0))
                self.assertEqual(tp.shape, want_ti.shape)

    def test_none_hidden_states_passes_through(self):
        from sglang.srt.speculative.eagle_utils import (
            fused_draft_extend_select_argmax_func,
        )

        logits = torch.randn(8, 16)
        sel_l, sel_h, _, _ = fused_draft_extend_select_argmax_func(
            logits, None, torch.tensor([1, 5])
        )
        self.assertIsNone(sel_h)
        self.assertEqual(sel_l.shape, (2, 16))

    def test_draft_extend_input_builds_a_safe_bucketed_selector(self):
        from sglang.srt.speculative.eagle_info import EagleDraftExtendInput

        draft_extend_input = EagleDraftExtendInput(
            num_accept_tokens=torch.tensor([2, 3, 2, 0], dtype=torch.int32),
            num_tokens_per_req=4,
        )

        # The compile policy pads accept counts with zero. Its inert request
        # selects the first row in that request rather than an invalid -1 row.
        self.assertEqual(
            draft_extend_input.last_accepted_indices().tolist(), [1, 6, 9, 12]
        )


if __name__ == "__main__":
    unittest.main()
