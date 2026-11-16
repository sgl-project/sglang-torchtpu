# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for the DeepSeek-V4.1 DSpark draft pieces on TPU.

Guards the conventions TpuBlockSpecWorker and the CSA2 backend rely on: which
verify rows the draft context-KV commit writes, the target Engram history after
a partial accept (against core EngramHasher.commit_after_verify), the in-graph
one-hot Markov chain (against the gather form), the draft attention handing
the backend output to wo_a without a second inverse RoPE, and the draft graph
input shim never reaching for the target-only Engram hasher.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.layers.engram import EngramHasher
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.models.deepseek_v41 import dspark, hooks


class TestSwaCommitRecords(CustomTestCase):
    def test_verify_commit_writes_only_accepted_columns(self):
        """A rejected draft's SWA row must not be published: its slot is reused
        by the next window and a stale row would sit inside the 128 window."""
        cache_loc_2d = torch.tensor([[64, 65, 66, 67], [128, 129, 130, 131]])
        records = dspark.swa_commit_records(
            cache_loc_2d.reshape(-1), cache_loc_2d, torch.tensor([2, 0])
        )
        self.assertEqual(records.tolist(), [64, 65, -1, -1, -1, -1, -1, -1])
        self.assertEqual(records.dtype, torch.int32)

    def test_extend_writes_every_row(self):
        cache_loc = torch.tensor([64, 65, 0])
        records = dspark.swa_commit_records(cache_loc, None, None)
        self.assertEqual(records.tolist(), [64, 65, 0])


def _hasher(num_slots: int, n: int) -> EngramHasher:
    hasher = object.__new__(EngramHasher)
    hasher.max_ngram_size = n
    hasher.history = torch.zeros(num_slots + 1, n - 1, dtype=torch.int32)
    hasher.pad_row = num_slots
    return hasher


class TestCommitEngramHistory(CustomTestCase):
    def test_matches_core_commit_after_verify(self):
        """After a verify of [r, d_1..d_N] the history must be core's: the last
        n - 1 of history + r + the correct drafts. The bonus is not committed."""
        n, num_slots = 4, 6
        history = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int32)
        req = torch.tensor([2, 5])
        bonus = torch.tensor([10, 20])
        drafts = torch.tensor([[11, 12, 13], [21, 22, 23]])
        num_correct = torch.tensor([2, 0])
        accept_lens = (num_correct + 1).to(torch.int32)
        # accept_tokens: correct drafts, then the new bonus, padded with -1.
        accept_tokens = torch.tensor([[11, 12, 99, -1], [77, -1, -1, -1]])

        reference = _hasher(num_slots, n)
        reference.history[req] = history
        verify_ids = torch.cat([bonus[:, None], drafts], dim=1)
        EngramHasher.commit_after_verify(reference, verify_ids, req, accept_lens)

        hasher = _hasher(num_slots, n)
        hasher.history[req] = history
        dspark.commit_engram_history(
            hasher,
            bonus_tokens=bonus,
            accept_tokens=accept_tokens,
            accept_lens=accept_lens,
            req_pool_indices=req,
        )
        torch.testing.assert_close(hasher.history[req], reference.history[req])
        self.assertEqual(hasher.history[2].tolist(), [10, 11, 12])

    def test_padded_rows_write_the_pad_row(self):
        """Bucket-padding rows carry req slot 0, a live slot; they must not
        overwrite it."""
        hasher = _hasher(4, 3)
        hasher.history[0] = torch.tensor([7, 8], dtype=torch.int32)
        dspark.commit_engram_history(
            hasher,
            bonus_tokens=torch.tensor([5]),
            accept_tokens=torch.tensor([[-1, -1]]),
            accept_lens=torch.tensor([0], dtype=torch.int32),
            req_pool_indices=torch.tensor([0]),
        )
        self.assertEqual(hasher.history[0].tolist(), [7, 8])


class TestDeferredEngramCommit(CustomTestCase):
    """The verify commit must not read the verify outputs back on the commit
    path (it blocked the host for the whole verify); the history must still be
    core's by the time the next target forward or a PD prebuilt batch reads it."""

    def _commit(self, hasher):
        worker = SimpleNamespace(
            target_worker=SimpleNamespace(
                model_runner=SimpleNamespace(
                    model=SimpleNamespace(model=SimpleNamespace(engram_hasher=hasher))
                )
            )
        )
        state = SimpleNamespace(
            bonus_tokens=torch.tensor([10, 20]), req_pool_indices=torch.tensor([2, 5])
        )
        accept = SimpleNamespace(
            accept_tokens=torch.tensor([[11, 12, 99, -1], [77, -1, -1, -1]]),
            accept_lens=torch.tensor([3, 1], dtype=torch.int32),
        )
        original = mock.MagicMock(return_value="committed")
        out = hooks.tpu_dsv41_commit_engram_history(
            original, worker, None, state, None, None, None, accept
        )
        self.assertEqual(out, "committed")

    def _reference(self, n, num_slots, history):
        reference = _hasher(num_slots, n)
        reference.history[torch.tensor([2, 5])] = history
        EngramHasher.commit_after_verify(
            reference,
            torch.tensor([[10, 11, 12, 13], [20, 21, 22, 23]]),
            torch.tensor([2, 5]),
            torch.tensor([3, 1], dtype=torch.int32),
        )
        return reference.history

    def test_commit_waits_for_the_next_history_read(self):
        n, num_slots = 4, 6
        history = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int32)
        hasher = _hasher(num_slots, n)
        hasher.history[torch.tensor([2, 5])] = history
        before = hasher.history.clone()
        self._commit(hasher)
        torch.testing.assert_close(hasher.history, before)
        dspark.flush_engram_commits(hasher)
        torch.testing.assert_close(
            hasher.history, self._reference(n, num_slots, history)
        )
        self.assertEqual(hasher.tpu_pending_commits, ())

    def test_queued_commits_apply_in_verify_order(self):
        hasher = _hasher(4, 3)
        for bonus in (5, 6):
            dspark.defer_engram_commit(
                hasher,
                bonus_tokens=torch.tensor([bonus]),
                accept_tokens=torch.tensor([[bonus + 1, -1]]),
                accept_lens=torch.tensor([1], dtype=torch.int32),
                req_pool_indices=torch.tensor([1]),
            )
        dspark.flush_engram_commits(hasher)
        # Each commit appends its anchor: [0, 0] -> [0, 5] -> [5, 6].
        self.assertEqual(hasher.history[1].tolist(), [5, 6])

    def test_prebuilt_history_write_follows_queued_commits(self):
        hasher = _hasher(4, 3)
        dspark.defer_engram_commit(
            hasher,
            bonus_tokens=torch.tensor([5]),
            accept_tokens=torch.tensor([[-1, -1]]),
            accept_lens=torch.tensor([1], dtype=torch.int32),
            req_pool_indices=torch.tensor([1]),
        )
        seen = {}

        def original(manager, batch):
            seen["history"] = manager.engram_hasher.history[1].tolist()

        manager = SimpleNamespace(engram_hasher=hasher)
        batch = SimpleNamespace(forward_mode=SimpleNamespace(is_prebuilt=lambda: True))
        hooks.tpu_dsv41_flush_engram_before_prebuilt(original, manager, batch)
        self.assertEqual(seen["history"], [0, 5])


class _MarkovModel:
    def __init__(self, *, gamma, vocab, hidden, sample_from_anchor):
        g = torch.Generator().manual_seed(0)
        self.gamma = gamma
        self.sample_from_anchor = sample_from_anchor
        self.lm = torch.randn(vocab, hidden, generator=g)
        self.lm_head = None
        self.markov_head = SimpleNamespace(
            vocab_size=vocab,
            markov_w1=SimpleNamespace(weight=torch.randn(vocab, 8, generator=g)),
            markov_w2=SimpleNamespace(weight=torch.randn(vocab, 8, generator=g)),
        )

    def compute_base_logits(self, x):
        return x @ self.lm.T, x


def _gather_reference(model, hidden, input_ids, width):
    bs = hidden.shape[0] // width
    h = hidden.view(bs, width, -1)
    h = h if model.sample_from_anchor else h[:, 1:]
    w1 = model.markov_head.markov_w1.weight
    w2 = model.markov_head.markov_w2.weight
    prev = input_ids.view(bs, width)[:, 0]
    out = []
    for step in range(model.gamma):
        logits = h[:, step] @ model.lm.T + w1[prev] @ w2.T
        prev = logits.argmax(-1)
        out.append(prev)
    return torch.stack(out, dim=1)


class TestGreedyMarkovDrafts(CustomTestCase):
    def test_one_hot_chain_matches_gather_chain(self):
        """The one-hot Markov embedding (used because the in-graph gather of an
        unsharded table is wrong on TPU) drafts what the gather form drafts,
        anchored on the first block row, for both block geometries."""
        for anchor in (True, False):
            with self.subTest(sample_from_anchor=anchor):
                model = _MarkovModel(
                    gamma=4, vocab=50, hidden=16, sample_from_anchor=anchor
                )
                width = dspark.block_width(model)
                self.assertEqual(width, 4 if anchor else 5)
                g = torch.Generator().manual_seed(1)
                hidden = torch.randn(3 * width, 16, generator=g)
                input_ids = torch.randint(0, 50, (3 * width,), generator=g)
                drafts = dspark.greedy_markov_drafts(model, hidden, input_ids)
                torch.testing.assert_close(
                    drafts, _gather_reference(model, hidden, input_ids, width)
                )


class TestDraftBaseLogits(CustomTestCase):
    def test_attention_tp1_head_keeps_this_ranks_logits(self):
        """Under attention DP8 the draft gathered vocab logits over the 8-rank TP
        group and cropped to one vocab width, so every rank drafted from TP rank
        0's rows (accept ~1.05 on 7 of 8 ranks). A full-vocab head must project
        this rank's rows locally and never call the gathering path."""
        g = torch.Generator().manual_seed(0)
        vocab, padded, hidden = 10, 12, 4
        lm_head = SimpleNamespace(
            tp_size=1,
            use_attn_tp_group=True,
            org_vocab_size=vocab,
            weight=torch.randn(padded, hidden, generator=g),
            quant_method=None,
        )
        norm_scale = torch.randn(hidden, generator=g)

        def gathering_path(x):
            raise AssertionError("compute_base_logits gathers over the TP group")

        model = SimpleNamespace(
            lm_head=lm_head,
            stages=[SimpleNamespace(norm=lambda x: x * norm_scale)],
            collapse_hc_head=lambda x: x,
            compute_base_logits=gathering_path,
        )
        x = torch.randn(3, hidden, generator=g)
        out = dspark.draft_base_logits(model, x)
        torch.testing.assert_close(out, (x * norm_scale) @ lm_head.weight[:vocab].T)


class TestDsparkAttention(CustomTestCase):
    def test_backend_output_reaches_wo_a_unrotated(self):
        """CSA2 already inverse-rotates; the draft must not rotate again, and
        must ask for the ratio-0 path with the stage's sink."""
        tokens, heads, head_dim, groups, rank = 3, 4, 8, 2, 5
        g = torch.Generator().manual_seed(0)
        backend_out = torch.randn(tokens, heads, head_dim, generator=g)
        wo_a = torch.randn(groups * rank, heads // groups * head_dim, generator=g)
        backend = mock.MagicMock()
        backend.prepare_mqa_qkv.return_value = ("ql", "q", "kv")
        backend.forward.return_value = backend_out
        sink = torch.zeros(heads)
        attn = SimpleNamespace(
            wq_a=lambda x: ("q_lora", None),
            attn="radix",
            n_local_heads=heads,
            n_local_groups=groups,
            o_lora_rank=rank,
            wo_a=SimpleNamespace(weight=wo_a),
            wo_b=lambda x: (x, None),
            _local_attn_sink=lambda: sink,
        )
        with mock.patch(
            "sglang.srt.model_executor.forward_context.get_attn_backend",
            return_value=backend,
        ):
            out = dspark.dspark_attention_forward(attn, "pos", "x", "fb")
        backend.prepare_mqa_qkv.assert_called_once_with(
            attn, "x", "q_lora", None, "pos"
        )
        kwargs = backend.forward.call_args.kwargs
        self.assertEqual(kwargs["compress_ratio"], 0)
        self.assertIs(kwargs["attn_sink"], sink)
        expected = torch.einsum(
            "tgd,grd->tgr",
            backend_out.reshape(tokens, groups, -1),
            wo_a.view(groups, rank, -1),
        ).reshape(tokens, -1)
        torch.testing.assert_close(out, expected)


class TestDrafterSelection(CustomTestCase):
    def test_only_v41_dspark_targets_get_the_v41_drafter(self):
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
        from sglang_tpu.models.deepseek_v41 import dspark_drafter

        def runner(model_type):
            return SimpleNamespace(
                model_config=SimpleNamespace(
                    hf_config=SimpleNamespace(model_type=model_type)
                )
            )

        original = mock.MagicMock(return_value="built-in")
        with mock.patch.object(
            dspark_drafter, "TpuDsv41DsparkDraftWorker", return_value="v41"
        ):
            pick = hooks.tpu_dsv41_create_draft_worker
            dspark_alg = SpeculativeAlgorithm.DSPARK
            self.assertEqual(
                pick(
                    original, algorithm=dspark_alg, target_runner=runner("deepseek_v41")
                ),
                "v41",
            )
            self.assertEqual(
                pick(original, algorithm=dspark_alg, target_runner=runner("qwen3")),
                "built-in",
            )
            self.assertEqual(
                pick(
                    original,
                    algorithm=SpeculativeAlgorithm.DFLASH,
                    target_runner=runner("deepseek_v41"),
                ),
                "built-in",
            )


class TestDraftEpLocalNames(CustomTestCase):
    def test_keeps_local_experts_and_every_replicated_tensor(self):
        from sglang_tpu.models.deepseek_v41 import loader

        names = {
            f"mtp.{s}.ffn.experts.{e}.w{w}.{k}"
            for s in range(2)
            for e in range(16)
            for w in (1, 2, 3)
            for k in ("weight", "scale")
        } | {"mtp.0.attn.wq_a.weight", "mtp.1.ffn.shared_experts.w1.weight"}
        config = SimpleNamespace(dspark_n_routed_experts=16, n_routed_experts=256)
        parallel = SimpleNamespace(moe_ep_size=8, moe_ep_rank=3)
        with mock.patch(
            "sglang.srt.runtime_context.get_parallel", return_value=parallel
        ):
            kept, experts = loader._draft_ep_local_names(names, config=config)
        kept_experts = {int(n.split(".")[4]) for n in kept if ".experts." in n}
        self.assertEqual(kept_experts, {6, 7})
        self.assertEqual(experts, 2 * 2 * 3 * 2)
        self.assertIn("mtp.0.attn.wq_a.weight", kept)
        self.assertIn("mtp.1.ffn.shared_experts.w1.weight", kept)

    def test_without_ep_reads_everything(self):
        from sglang_tpu.models.deepseek_v41 import loader

        names = {"mtp.0.ffn.experts.5.w1.weight", "mtp.0.norm.weight"}
        config = SimpleNamespace(dspark_n_routed_experts=0, n_routed_experts=8)
        parallel = SimpleNamespace(moe_ep_size=1, moe_ep_rank=0)
        with mock.patch(
            "sglang.srt.runtime_context.get_parallel", return_value=parallel
        ):
            kept, experts = loader._draft_ep_local_names(names, config=config)
        self.assertEqual(kept, names)
        self.assertEqual(experts, 1)


class TestDraftPlacement(CustomTestCase):
    def test_v41_dspark_draft_keeps_the_target_placement(self):
        from sglang_tpu.experimental.speculative.draft_worker import TpuDraftWorker
        from sglang_tpu.models.deepseek_v41.dspark_drafter import (
            TpuDsv41DsparkDraftWorker,
        )

        self.assertFalse(TpuDraftWorker._keeps_target_placement)
        self.assertTrue(TpuDsv41DsparkDraftWorker._keeps_target_placement)


class TestGetLastLocOffCuda(CustomTestCase):
    def test_host_tensors_skip_the_triton_kernel(self):
        """Spec decode allocation on TPU called the Triton get_last_loc
        ("0 active drivers"); host tensors must take the torch lookup."""
        req_to_token = torch.arange(12, dtype=torch.int32).view(3, 4)
        rows = torch.tensor([2, 0])
        lens = torch.tensor([3, 0], dtype=torch.int32)

        def triton(*args):
            raise AssertionError("Triton get_last_loc called off CUDA")

        out = hooks.tpu_dsv41_get_last_loc(triton, req_to_token, rows, lens)
        self.assertEqual(out.tolist(), [10, -1])


if __name__ == "__main__":
    unittest.main()
