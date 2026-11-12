"""CPU contracts for grammar-constrained requests on the experimental TPU spec worker."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
import xgrammar as xgr

from sglang.srt.constrained.xgrammar_backend import XGrammarGrammar
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.experimental.sampling.sampling_batch_info import (
    HostPaddedSamplingInfo,
    RowPaddedGrammarMask,
)
from sglang_tpu.experimental.speculative import draft_info, spec_ops
from sglang_tpu.experimental.speculative.base_spec_worker import (
    _grammar_mask_enabled,
)
from sglang_tpu.experimental.speculative.block_spec_worker import TpuBlockSpecWorker
from sglang_tpu.experimental.speculative.spec_ops import (
    accept_drafts_sampled,
    accept_drafts_target_only,
    apply_grammar_bitmask,
)
from sglang_tpu.experimental.speculative.spec_types import DraftResult
from sglang_tpu.experimental.speculative.spec_utils import grammar_verify_bitmask
from sglang_tpu.sampler import _supports_padded_rows, supports_padded_sampling

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

# Tokens a, b, c, x, <eos>; the grammar is "ab+c".
A, B, C, X, EOS = range(5)
VOCAB = ["a", "b", "c", "x", "<eos>"]


def _grammar() -> XGrammarGrammar:
    info = xgr.TokenizerInfo(VOCAB, stop_token_ids=[EOS])
    ctx = xgr.GrammarCompiler(info).compile_regex("ab+c")
    return XGrammarGrammar(
        xgr.GrammarMatcher(ctx, max_rollback_tokens=200), len(VOCAB), ctx, None
    )


def _allowed(bitmask: torch.Tensor) -> list[list[int]]:
    """Allowed token ids per bitmask row."""
    return [
        [t for t in range(len(VOCAB)) if (int(row[t // 32]) >> (t % 32)) & 1]
        for row in bitmask
    ]


class TestExperimentalSpecGrammar(CustomTestCase):
    def test_bitmask_walks_the_draft_chain_and_restores_the_grammar(self):
        """Row j masks the token after d_1..d_j; rows past a rejected draft, of
        grammar-free requests and of padding allow every token."""
        grammar = _grammar()
        grammar.accept_token(A)
        reqs = [SimpleNamespace(grammar=grammar), SimpleNamespace(grammar=None)]
        # Rows: [r, d_1, d_2, d_3]; d_2 = x breaks the grammar.
        verify_tokens = torch.tensor(
            [[A, B, X, C], [A, A, A, A], [0, 0, 0, 0]], dtype=torch.int64
        )
        bitmask = grammar_verify_bitmask(reqs, verify_tokens, len(VOCAB), "cpu")
        everything = list(range(len(VOCAB)))
        self.assertEqual(bitmask.dtype, torch.int32)
        self.assertEqual(bitmask.shape, (12, 1))
        self.assertEqual(_allowed(bitmask[:4]), [[B], [B, C], everything, everything])
        self.assertEqual(_allowed(bitmask[4:]), [everything] * 8)
        self.assertEqual(grammar.accepted_tokens, [A])

    def test_no_live_grammar_builds_no_mask(self):
        grammar = _grammar()
        grammar.finished = True
        reqs = [SimpleNamespace(grammar=None), SimpleNamespace(grammar=grammar)]
        self.assertIsNone(
            grammar_verify_bitmask(
                reqs, torch.zeros((2, 3), dtype=torch.int64), 5, "cpu"
            )
        )

    def test_apply_bitmask_masks_every_cleared_bit(self):
        """Bits past 31 and a vocab that is not a multiple of 32 keep their rows."""
        vocab = 40
        allowed = [[0, 31, 39], [5]]
        bitmask = torch.zeros((2, 2), dtype=torch.int32)
        for row, tokens in enumerate(allowed):
            for t in tokens:
                word = int(bitmask[row, t // 32]) | (1 << (t % 32))
                bitmask[row, t // 32] = torch.tensor(word).to(torch.int32)
        masked = apply_grammar_bitmask(torch.zeros((2, vocab)), bitmask)
        for row, tokens in enumerate(allowed):
            self.assertEqual(
                torch.isfinite(masked[row]).nonzero().view(-1).tolist(), tokens
            )

    def _verify(self, *, barrier_tokens, target_logits, bucket_extra=None):
        worker = object.__new__(TpuBlockSpecWorker)
        worker.num_verify_rows = 3
        worker.device = "cpu"
        worker._simulate_acc_len = 0
        worker._tp_sync = MagicMock()
        worker._verify_top_k_width = None
        worker._rank_check_interval = 0
        worker._rng_step = torch.zeros((1,), dtype=torch.int64)
        worker._block_verification = False
        grammar = _grammar()
        # bs 1 at bucket 2: drafts [b, x] after r = a.
        draft = DraftResult(
            torch.tensor([[B, X], [0, 0]]),
            None,
            torch.tensor([2, 2], dtype=torch.int32),
            torch.tensor([A, B, X, 0, 0, 0]),
        )
        bucket = SimpleNamespace(
            state=SimpleNamespace(seq_lens=torch.tensor([4, 0])),
            live_mask=torch.tensor([True, False]),
            temperatures=None,
            top_ks=None,
            **(bucket_extra or {}),
        )
        barrier = MagicMock(
            side_effect=lambda: [grammar.accept_token(t) for t in barrier_tokens]
        )
        batch = SimpleNamespace(
            has_grammar=True, reqs=[SimpleNamespace(grammar=grammar)]
        )
        with (
            patch.object(worker, "make_verify_input"),
            patch.object(
                worker,
                "_forward_target_verify",
                return_value=SimpleNamespace(
                    next_token_logits=target_logits,
                    has_next_token_global_top_k=False,
                ),
            ),
        ):
            _, accept = worker.verify(
                batch, bucket, None, draft, None, grammar_barrier=barrier
            )
        barrier.assert_called_once()
        self.assertEqual(grammar.accepted_tokens, list(barrier_tokens))
        return accept

    def test_greedy_verify_accepts_only_grammar_tokens(self):
        """The barrier lands the previous step's a before the walk; the target
        prefers x everywhere, but the grammar leaves b then {b, c} with c > b."""
        logits = torch.zeros((6, len(VOCAB)))
        logits[:, X] = 10.0
        logits[:, C] = 5.0
        logits[:, B] = 4.0
        accept = self._verify(barrier_tokens=[A], target_logits=logits)
        self.assertEqual(accept.num_correct_drafts.tolist()[0], 1)
        self.assertEqual(accept.bonus_tokens.tolist()[0], C)
        self.assertEqual(accept.accept_tokens[0].tolist(), [B, C, -1])

    def test_sampled_acceptance_never_takes_a_zero_probability_draft(self):
        """A grammar-masked draft has p = 0 and must be rejected even when u = 0
        or the acceptance threshold is relaxed to 0."""
        logits = torch.zeros((2, 4))
        logits[0, 1] = float("-inf")
        drafts = torch.tensor([[1]])
        rng_step = torch.zeros((1,), dtype=torch.int64)

        def zero_uniforms(step, stream, shape):
            return torch.zeros(shape)

        # threshold_acc 0 accepts any draft the target can produce.
        with (
            patch.object(spec_ops, "hash_uniforms", zero_uniforms),
            patch.object(spec_ops, "tpu_filter_probs", lambda probs, *a, **k: probs),
        ):
            num_correct, bonus = accept_drafts_target_only(
                logits,
                drafts,
                torch.ones(1, dtype=torch.int32),
                torch.ones(1),
                torch.full((1,), 4, dtype=torch.int32),
                torch.ones(1),
                torch.zeros(1),
                1.0,
                0.0,
                rng_step,
            )
        self.assertEqual(num_correct.tolist(), [0])
        self.assertNotEqual(bonus.tolist(), [1])
        with patch.object(spec_ops, "hash_uniforms", zero_uniforms):
            num_correct, bonus = accept_drafts_sampled(
                logits,
                drafts,
                torch.ones(1, dtype=torch.int32),
                torch.zeros((1, 1, 4)),
                torch.ones(1),
                torch.full((1,), 4, dtype=torch.int32),
                rng_step,
            )
        self.assertEqual(num_correct.tolist(), [0])
        self.assertNotEqual(bonus.tolist(), [1])

    def test_window_planning_keeps_grammar_batches(self):
        batch = SimpleNamespace(
            enable_overlap=True,
            req_pool_indices_cpu=torch.tensor([1]),
            return_logprob=False,
            has_grammar=True,
            reqs=[SimpleNamespace(lora_id=None)],
            spec_algorithm=SimpleNamespace(
                create_worker=lambda _: SimpleNamespace(supports_grammar=True)
            ),
        )
        with (
            patch.object(
                draft_info,
                "get_spec",
                return_value=SimpleNamespace(speculative_draft_window_size=None),
            ),
            patch.object(draft_info, "get_server_args", return_value=None),
        ):
            self.assertTrue(draft_info._verify_window_eligible(batch))

    def test_only_packed_bitmask_backends_start(self):
        self.assertTrue(_grammar_mask_enabled("xgrammar"))
        self.assertFalse(_grammar_mask_enabled("none"))
        for backend in ("outlines", "llguidance"):
            with self.assertRaisesRegex(NotImplementedError, "grammar-backend"):
                _grammar_mask_enabled(backend)

    def test_host_padded_grammar_mask_covers_extend_cap_rows(self):
        """Extend logits can sit at a cap wider than the bucket the mask covers;
        the extra rows stay unconstrained instead of failing the broadcast."""
        grammar = _grammar()
        info = HostPaddedSamplingInfo(
            temperatures=torch.ones((2, 1)),
            top_ps=torch.ones(2),
            top_ks=torch.ones(2, dtype=torch.int32),
            min_ps=torch.zeros(2),
            is_all_greedy=True,
            is_any_greedy=True,
            need_top_p_sampling=False,
            need_top_k_sampling=True,
            need_min_p_sampling=False,
            vocab_size=len(VOCAB),
            device="cpu",
            real_batch_size=1,
            grammars=[grammar],
            sampling_params_on_cpu=True,
        )
        info.update_regex_vocab_mask()
        self.assertIsInstance(info.grammar_mask, RowPaddedGrammarMask)
        logits = torch.zeros((4, len(VOCAB)))
        info.grammar_mask.apply(logits)
        self.assertEqual(torch.isfinite(logits[0]).nonzero().view(-1).tolist(), [A])
        self.assertTrue(torch.isfinite(logits[1:]).all())
        # Only HOST_PADDED masks pad their own rows.
        self.assertTrue(_supports_padded_rows(info))
        self.assertFalse(supports_padded_sampling(info))

    def test_grammar_mask_warmup_covers_every_bucket_and_width(self):
        worker = object.__new__(TpuBlockSpecWorker)
        worker.device = "cpu"
        worker.num_verify_rows = 4
        worker._target_worker = SimpleNamespace(
            model_runner=SimpleNamespace(model_config=SimpleNamespace(vocab_size=40))
        )
        calls = []
        with (
            patch.object(TpuBlockSpecWorker, "verify_widths", return_value=(2, 4)),
            patch.object(
                TpuBlockSpecWorker,
                "_b_bucketer",
                return_value=SimpleNamespace(buckets=[1, 4]),
            ),
            patch(
                "sglang_tpu.experimental.speculative.base_spec_worker."
                "warmup_compiled_op",
                lambda op, *args: calls.append((op, args)),
            ),
        ):
            worker._warmup_grammar_mask()
        self.assertTrue(all(op is apply_grammar_bitmask for op, _ in calls))
        self.assertEqual(
            sorted(
                (tuple(logits.shape), logits.dtype, tuple(mask.shape), mask.dtype)
                for _, (logits, mask) in calls
            ),
            sorted(
                ((rows, 40), torch.float32, (rows, 2), torch.int32)
                for rows in (2, 8, 4, 16)
            ),
        )


if __name__ == "__main__":
    unittest.main()
