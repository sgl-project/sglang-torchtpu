"""SGLANG_SIMULATE_ACC_TOKEN_MODE=real-draft-token: TpuSpecWorker's compiled forced accept."""

import unittest

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.experimental.speculative.spec_ops import (
    accept_real_draft_step,
    accept_real_draft_top_k_step,
    finalize_accept,
)

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _eager_reference(target_logits, draft_tokens, forced, num_verify, seq_lens, live):
    """The eager path the compiled ops replace: argmax, clamp, gather, then finalize."""
    bs, n = draft_tokens.shape
    target_argmax = torch.argmax(target_logits, dim=-1).reshape(bs, n + 1)
    num_correct = torch.minimum(forced, num_verify.to(forced.dtype))
    bonus = torch.gather(
        target_argmax.to(draft_tokens.dtype), 1, num_correct.reshape(bs, 1)
    ).reshape(bs)
    return finalize_accept(draft_tokens, num_correct, bonus, seq_lens, live)


def _tied_logits(rows: int, vocab: int, seed: int) -> torch.Tensor:
    # Coarse values make the row maximum tie across several columns.
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(0, 4, (rows, vocab), generator=gen).to(torch.bfloat16).float()


def _assert_accept_equal(test, actual, expected):
    for field in expected._fields:
        a, e = getattr(actual, field), getattr(expected, field)
        test.assertEqual(a.dtype, e.dtype, field)
        test.assertTrue(torch.equal(a, e), f"{field}: {a} != {e}")


class TestRealDraftAccept(CustomTestCase):
    def _inputs(self, *, bs=5, n=3, vocab=257, seed=0):
        gen = torch.Generator().manual_seed(seed)
        logits = _tied_logits(bs * (n + 1), vocab, seed)
        drafts = torch.randint(0, vocab, (bs, n), generator=gen, dtype=torch.int64)
        forced = torch.full((bs,), 2, dtype=torch.int64)
        # Caps below, at and above the forced length; the last row is padding.
        num_verify = torch.tensor([3, 2, 1, 0, 3], dtype=torch.int32)[:bs]
        seq_lens = torch.randint(1, 1000, (bs,), generator=gen, dtype=torch.int64)
        live = torch.tensor([True, True, True, True, False])[:bs]
        return logits, drafts, forced, num_verify, seq_lens, live

    def test_full_logits_matches_eager_path_with_ties_and_caps(self):
        for seed in range(8):
            inputs = self._inputs(seed=seed)
            _assert_accept_equal(
                self, accept_real_draft_step(*inputs), _eager_reference(*inputs)
            )

    def test_top_k_ids_match_eager_path(self):
        for width in (1, 64):
            logits, *rest = self._inputs(seed=width)
            # Descending top-k ids whose first column is torch.argmax's lowest-id winner.
            order = torch.argsort(-logits, dim=-1, stable=True)[:, :width]
            top_ids = order.to(torch.int32)
            self.assertTrue(torch.equal(top_ids[:, 0].long(), logits.argmax(-1)))
            _assert_accept_equal(
                self,
                accept_real_draft_top_k_step(top_ids, *rest),
                _eager_reference(logits, *rest),
            )

    def test_bonus_is_target_token_at_forced_row(self):
        # bs=3, N=3 drafts -> 4 verify rows per request; argmax of row j is 10 * (b + 1) + j.
        target = torch.tensor(
            [[10, 11, 12, 13], [20, 21, 22, 23], [30, 31, 32, 33]], dtype=torch.int32
        )
        drafts = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.int64)
        accept = accept_real_draft_top_k_step(
            target.reshape(-1, 1),
            drafts,
            torch.tensor([2, 0, 3], dtype=torch.int64),
            torch.tensor([3, 3, 1], dtype=torch.int32),
            torch.tensor([100, 200, 300], dtype=torch.int64),
            torch.tensor([True, True, True]),
        )
        self.assertEqual(accept.num_correct_drafts.tolist(), [2, 0, 1])
        self.assertEqual(accept.bonus_tokens.tolist(), [12, 20, 31])
        self.assertEqual(
            accept.accept_tokens.tolist(),
            [[1, 2, 12, -1], [20, -1, -1, -1], [7, 31, -1, -1]],
        )
        self.assertEqual(accept.next_seq_lens.tolist(), [103, 201, 302])


if __name__ == "__main__":
    unittest.main()
