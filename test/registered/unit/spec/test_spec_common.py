"""CPU contracts for the shared speculative-decoding functions in spec_common."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.speculative.spec_common import (
    adjust_verify_logits,
    build_dp_idle_draft_batch,
    build_trivial_verify_input,
    update_target_recurrent_state_after_verify,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestAdjustVerifyLogits(CustomTestCase):
    def test_per_request_terms_cover_its_verify_rows_and_skip_bucket_padding(self):
        bs, draft_token_num, vocab, padding = 2, 3, 4, 2
        logits = torch.linspace(-3.0, 3.0, (bs * draft_token_num + padding) * vocab)
        logits = logits.view(-1, vocab)
        original = logits.clone()
        additive = torch.tensor([[0.5, -0.5, 1.0, 0.0], [2.0, 0.0, -1.0, 0.5]])
        scaling = torch.tensor([[2.0, 1.0, 4.0, 1.0], [1.0, 2.0, 1.0, 8.0]])
        bias = torch.tensor([[0.0, 1.0, 0.0, -1.0], [3.0, 0.0, 0.0, 0.0]])

        adjust_verify_logits(
            next_token_logits=logits,
            sampling_info=SimpleNamespace(
                acc_additive_penalties=additive,
                acc_scaling_penalties=scaling,
                logit_bias=bias,
            ),
            draft_token_num=draft_token_num,
        )

        for row in range(bs * draft_token_num):
            req = row // draft_token_num
            shifted = original[row] + additive[req]
            # Scaling penalties shrink positive logits and push negative ones further.
            scaled = torch.where(
                shifted < 0, shifted * scaling[req], shifted / scaling[req]
            )
            torch.testing.assert_close(logits[row], scaled + bias[req])
        torch.testing.assert_close(
            logits[bs * draft_token_num :], original[bs * draft_token_num :]
        )

    def test_per_request_terms_broadcast_without_repeating_rows(self):
        from torch.utils._python_dispatch import TorchDispatchMode

        bs, draft_token_num, vocab = 2, 3, 4
        logits = torch.zeros(bs * draft_token_num, vocab)
        ops = []

        class RecordOps(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                ops.append(func.overloadpacket.__name__)
                return func(*args, **(kwargs or {}))

        with RecordOps():
            adjust_verify_logits(
                next_token_logits=logits,
                sampling_info=SimpleNamespace(
                    acc_additive_penalties=torch.ones(bs, vocab),
                    acc_scaling_penalties=torch.full((bs, vocab), 2.0),
                    logit_bias=torch.ones(bs, vocab),
                ),
                draft_token_num=draft_token_num,
            )
        # A repeated [bs * draft_token_num, vocab] copy costs a vocab-sized
        # allocation per verify step.
        self.assertNotIn("repeat_interleave", ops)
        torch.testing.assert_close(logits, torch.full_like(logits, 1.5))


def _decode_batch(*, seq_lens, seq_lens_sum=None, seq_lens_cpu=None):
    return SimpleNamespace(
        forward_mode=SimpleNamespace(is_idle=lambda: False),
        spec_info=SimpleNamespace(bonus_tokens=torch.tensor([7, 8])),
        seq_lens=torch.tensor(seq_lens),
        seq_lens_sum=seq_lens_sum,
        seq_lens_cpu=seq_lens_cpu,
    )


class TestTrivialVerifyInput(CustomTestCase):
    def build(self, batch, verify_mask=None):
        backend = SimpleNamespace(verify_mask=verify_mask, max_context_len=100)
        return build_trivial_verify_input(
            batch=batch, topk=1, device="cpu", target_attn_backend=backend
        )

    def test_one_node_tree_rooted_at_the_bonus_token(self):
        verify_input = self.build(_decode_batch(seq_lens=[3, 5]))
        torch.testing.assert_close(verify_input.draft_token, torch.tensor([7, 8]))
        torch.testing.assert_close(verify_input.positions, torch.tensor([3, 5]))
        torch.testing.assert_close(
            verify_input.retrieve_index, torch.tensor([[0], [1]])
        )
        self.assertTrue(bool((verify_input.retrieve_next_token == -1).all()))
        self.assertTrue(bool((verify_input.retrieve_next_sibling == -1).all()))
        self.assertEqual(verify_input.draft_token_num, 1)

    def test_fallback_mask_covers_each_prefix_plus_its_root(self):
        cases = (
            (dict(seq_lens_sum=8), 8 + 2),
            (dict(seq_lens_cpu=torch.tensor([3, 5])), 8 + 2),
            # No length mirror: the context-length upper bound per request.
            (dict(), 2 * 100 + 2),
        )
        for lengths, expected in cases:
            with self.subTest(lengths=sorted(lengths)):
                mask = self.build(_decode_batch(seq_lens=[3, 5], **lengths)).custom_mask
                self.assertEqual(mask.numel(), expected)
                self.assertTrue(bool(mask.all()))

    def test_a_fitting_backend_mask_is_reused_and_fully_visible(self):
        buffer = torch.zeros(16, dtype=torch.bool)
        verify_mask = SimpleNamespace(buffer=buffer, fits=lambda bs: bs <= 4)
        mask = self.build(_decode_batch(seq_lens=[3, 5]), verify_mask).custom_mask
        self.assertIs(mask, buffer)
        self.assertTrue(bool(buffer.all()))


class TestDpIdleDraftBatch(CustomTestCase):
    def test_idle_draft_batch_carries_the_peer_token_counts(self):
        idle = MagicMock()
        batch = MagicMock(
            global_num_tokens=[0, 7], global_num_tokens_for_logprob=[0, 1]
        )
        with patch.object(ScheduleBatch, "init_new", return_value=idle) as init_new:
            draft_batch = build_dp_idle_draft_batch(batch)
        self.assertIs(draft_batch, idle)
        self.assertEqual(init_new.call_args.args[0], [])
        idle.prepare_for_idle.assert_called_once_with()
        self.assertEqual(draft_batch.global_num_tokens, [0, 7])
        self.assertEqual(draft_batch.global_num_tokens_for_logprob, [0, 1])


class TestRecurrentStateAfterVerify(CustomTestCase):
    def commit(self, *, pre, commit_lens, track):
        calls = []
        backend = SimpleNamespace(
            update_mamba_state_after_mtp_verify=lambda **kwargs: calls.append(kwargs)
        )
        pre = torch.tensor(pre)
        commit_lens = torch.tensor(commit_lens, dtype=torch.int32)
        with patch(
            "sglang.srt.speculative.spec_common.mamba_track_grid", return_value=4
        ):
            update_target_recurrent_state_after_verify(
                attn_backend=backend,
                model="model",
                mamba_track_indices=torch.arange(4) if track else None,
                tree_page_size=1,
                # One bucket-padding row past the committed requests.
                req_pool_indices=torch.tensor([10, 11, 12, 13, 99]),
                seq_lens_pre_verify=pre,
                seq_lens_post_verify=pre + commit_lens,
                commit_lens=commit_lens,
            )
        (kwargs,) = calls
        return kwargs

    def test_commits_last_accepted_step_and_the_step_that_crosses_a_track_boundary(
        self,
    ):
        kwargs = self.commit(pre=[6, 6, 3, 7], commit_lens=[3, 1, 2, 1], track=True)
        torch.testing.assert_close(
            kwargs["last_correct_step_indices"], torch.tensor([2, 0, 1, 0])
        )
        # Interval 4: 6->9 crosses 8 at step 1; 6->7 crosses nothing;
        # 3->5 crosses 4 at step 0; 7->8 lands on 8 at its only step.
        torch.testing.assert_close(
            kwargs["mamba_steps_to_track"], torch.tensor([1, -1, 0, 0])
        )
        torch.testing.assert_close(
            kwargs["req_pool_indices"], torch.tensor([10, 11, 12, 13])
        )

    def test_without_checkpoint_tracking_nothing_extra_is_saved(self):
        kwargs = self.commit(pre=[6], commit_lens=[3], track=False)
        self.assertIsNone(kwargs["mamba_steps_to_track"])
        self.assertIsNone(kwargs["mamba_track_indices"])


if __name__ == "__main__":
    unittest.main()
