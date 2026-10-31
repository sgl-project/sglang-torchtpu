"""CPU contracts for the chain drafters (EAGLE3, MTP) on the experimental TPU spec worker."""

from __future__ import annotations

import contextlib
import unittest
from types import SimpleNamespace
from typing import NamedTuple
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    HybridLinearAttnBackend,
)
from sglang.srt.layers.attention.index_topk_share import IndexTopKShareState
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_tp_sync import SpecTpSyncSite
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.envs import envs
from sglang_tpu.experimental.compilation.chain_compile_policy import (
    TpuChainSpecDecodeCompilePolicyMixin,
    _hidden_only_logits,
)
from sglang_tpu.experimental.sampling.sampling_batch_info import HostPaddedSamplingInfo
from sglang_tpu.experimental.speculative import (
    base_spec_worker,
    chain_spec_worker,
    spec_ops,
)
from sglang_tpu.experimental.speculative.base_spec_worker import TpuSpecWorker
from sglang_tpu.experimental.speculative.chain_draft_input import (
    chain_draft_forward_input,
)
from sglang_tpu.experimental.speculative.chain_draft_worker import (
    TpuChainDraftWorker,
)
from sglang_tpu.experimental.speculative.chain_info import HostPaddedChainDraftInput
from sglang_tpu.experimental.speculative.chain_spec_worker import (
    TpuChainSpecWorker,
    _attach_pd_seeds,
    fill_sampled_tails,
    fold_prompt_tail_embeds,
    shift_prompt_tokens,
)
from sglang_tpu.experimental.speculative.eagle3 import eagle3_draft_worker
from sglang_tpu.experimental.speculative.eagle3.eagle3_draft_worker import (
    TpuEagle3DraftWorker,
)
from sglang_tpu.experimental.speculative.spec_ops import (
    _filtered_target_probs,
    accept_candidates_step,
    accept_drafts_candidates,
    accept_drafts_sampled,
    accept_drafts_sampled_filtered,
    accept_drafts_target_only,
    accept_greedy_step,
    accept_greedy_top_k_step,
    accept_top_k_candidates_step,
    finalize_accept,
)
from sglang_tpu.experimental.speculative.spec_types import (
    DraftResult,
    SpecState,
    TpuSpecVerifyInput,
)
from sglang_tpu.experimental.speculative.spec_utils import pad_rows
from sglang_tpu.hooks.experimental_spec import (
    keep_host_padded_extend_outputs,
)
from sglang_tpu.hooks.spec_worker import select_experimental_spec_worker

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _OptionalCarry(NamedTuple):
    hidden_states: torch.Tensor
    optional: torch.Tensor | None = None


class _AdaptiveWarmupWorker(TpuChainSpecWorker):
    def verify_widths(self):
        return (2, 4)


def _chain_worker() -> TpuChainSpecWorker:
    """A TpuChainSpecWorker carrying only the attributes the tested methods read."""
    worker = object.__new__(TpuChainSpecWorker)
    worker.device = "cpu"
    worker.num_verify_rows = 3
    return worker


def _cpu_joint_filter(probs, top_ks, top_ps, min_ps, **need):
    """CPU reference of tpu_filter_probs with every filter enabled (ties kept)."""
    del need
    desc = probs.sort(dim=-1, descending=True).values
    rows = torch.arange(probs.shape[0])
    k = top_ks.clamp(1, probs.shape[1]).long() - 1
    nucleus = (desc.cumsum(-1) < top_ps[:, None]).sum(-1).clamp_max(probs.shape[1] - 1)
    cutoff = torch.maximum(desc[rows, k], desc[rows, nucleus])
    cutoff = torch.maximum(cutoff, desc[:, 0] * min_ps)
    return torch.where(probs >= cutoff[:, None], probs, 0.0)


def _unfiltered(rows: int) -> dict[str, torch.Tensor]:
    """Sampling params that keep the whole target distribution at temperature 1."""
    return dict(
        temperatures=torch.ones(rows),
        top_ps=torch.ones(rows),
        min_ps=torch.zeros(rows),
    )


class TestExperimentalEagle3(CustomTestCase):
    def test_verify_warmups_keep_capacity_across_active_widths(self):
        worker = object.__new__(_AdaptiveWarmupWorker)
        worker.device = "cpu"
        worker.num_drafts = 3
        worker.num_verify_rows = 4
        worker._dp_attention = False
        worker._sampled_drafts = False
        worker._target_worker = SimpleNamespace(
            model_runner=SimpleNamespace(max_running_requests=2)
        )
        worker._chain_draft_worker = SimpleNamespace(
            carry_size=3, carry=torch.zeros((3, 3)), dsa_seed_topk_table=None
        )
        worker._draft_runner = SimpleNamespace(dtype=torch.float32)
        buckets = SimpleNamespace(buckets=[1, 2], max_val=2, next_bucket=lambda bs: bs)
        selections = []
        drafts = []

        def select(candidates, committed, seq_lens, width):
            selections.append((tuple(candidates.shape), width))
            self.assertIsNot(committed, seq_lens)
            return spec_ops.select_verify_rows(candidates, committed, seq_lens, width)

        def warmup(op, *args):
            return select(*args) if op is spec_ops.select_verify_rows else op(*args)

        def draft(**kwargs):
            rows = kwargs["rows"]
            bs, width = rows.positions.shape
            drafts.append((bs, width))
            self.assertEqual(kwargs["seq_lens_cpu"].tolist(), [width] * bs)
            return DraftResult(
                torch.zeros((bs, width - 1), dtype=torch.int64),
                None,
                torch.full((bs,), width - 1, dtype=torch.int32),
            )

        expected = {((bs, 8), width) for bs in (1, 2) for width in (2, 4)}
        with (
            patch.object(torch, "tpu", create=True) as tpu,
            patch.object(
                base_spec_worker, "max_speculative_num_draft_tokens", return_value=4
            ),
            patch.object(worker, "_b_bucketer", return_value=buckets),
            patch.object(base_spec_worker, "warmup_compiled_op", side_effect=warmup),
            patch.object(chain_spec_worker, "warmup_compiled_op", side_effect=warmup),
            patch.object(chain_spec_worker, "select_verify_rows", side_effect=select),
            patch.object(worker, "_draft_chain", side_effect=draft),
        ):
            self.assertEqual(worker.verify_window_rows, 4)
            for run in (
                worker._warmup_logical_bs_ops,
                worker._warmup_draft_ops,
                worker._warmup_verify_mrope,
            ):
                selections.clear()
                run()
                self.assertEqual(set(selections), expected)
                self.assertEqual(len(selections), len(expected))
            idle = TpuSpecVerifyInput.idle(
                verify_window_rows=worker.verify_window_rows, padded_bs=2, device="cpu"
            )
        tpu.current_stream.assert_called_once_with()
        tpu.current_stream.return_value.synchronize.assert_called_once_with()
        self.assertEqual(idle.verify_cache_candidates.shape, (2, 8))
        self.assertEqual(
            set(drafts), {(bs, width) for bs in (1, 2) for width in (2, 4)}
        )
        self.assertEqual((worker.num_drafts, worker.num_verify_rows), (3, 4))
        self.assertEqual(_chain_worker().verify_widths(), (3,))

    def _radix_track_commit(self, *, host_lens, device_lens, last_steps):
        """The backend's verify commit kwargs for one step (track grid 64, slack 4)."""
        commits = []
        backend = object.__new__(HybridLinearAttnBackend)
        backend.update_mamba_state_after_mtp_verify = lambda **kw: commits.append(kw)
        worker = _chain_worker()
        worker._target_has_linear_attn = True
        worker._target_worker = SimpleNamespace(
            model_runner=SimpleNamespace(attn_backend=backend, model=None)
        )
        accept = SimpleNamespace(last_steps=torch.tensor(last_steps))
        worker.draft = lambda *args: None
        worker.verify = lambda *args: (None, accept)
        worker.commit = lambda *args: None
        batch = SimpleNamespace(
            mamba_track_indices=torch.tensor([3, 0]),
            tree_cache=SimpleNamespace(page_size=64),
        )
        bucket = SimpleNamespace(
            state=SimpleNamespace(seq_lens=torch.tensor(device_lens))
        )
        zeros = torch.zeros(2, dtype=torch.int64)
        verify_input = TpuSpecVerifyInput(
            verify_cache_candidates=zeros,
            verify_cache_candidates_cpu=zeros,
            committed_lens=zeros,
            req_pool_indices=zeros,
            req_pool_indices_cpu=zeros,
            seq_lens_cpu=torch.tensor(host_lens),
            host_len_slack=4,
            future_indices=None,
        )
        with patch.object(base_spec_worker, "mamba_track_grid", return_value=64):
            TpuSpecWorker.step(worker, batch, bucket, None, verify_input)
        return commits[0]

    def test_radix_track_step_counts_from_the_device_length(self):
        """Under overlap the host committed base trails the device by the last
        commit; the snapshot step must count from the device length (63 + 2 -> step 0)."""
        commit = self._radix_track_commit(
            host_lens=[60, 0], device_lens=[63, 0], last_steps=[1, -1]
        )
        self.assertEqual(commit["mamba_steps_to_track"].tolist(), [0, -1])

    def test_radix_tracking_skips_a_verify_that_cannot_cross_a_boundary(self):
        """A verify whose host bounds (base 8, base + 4 rows + slack 4) stay inside one
        track interval snapshots nothing, so it must not launch the snapshot."""
        commit = self._radix_track_commit(
            host_lens=[8, 0], device_lens=[11, 0], last_steps=[3, -1]
        )
        self.assertIsNone(commit["mamba_track_indices"])
        self.assertIsNone(commit["mamba_steps_to_track"])

    def test_flag_routes_eagle3_and_leaves_other_algorithms_alone(self):
        with (
            envs.SGLANG_TPU_ENABLE_EXPERIMENTAL_SPEC_WORKER.override(True),
            patch(
                "sglang_tpu.hooks.spec_worker.get_parallel",
                return_value=SimpleNamespace(
                    pp_group=SimpleNamespace(is_last_rank=True)
                ),
            ),
        ):
            self.assertIs(
                select_experimental_spec_worker(
                    None, SpeculativeAlgorithm.EAGLE3, None
                ),
                TpuChainSpecWorker,
            )
            # EAGLE is MTP (core resolves NEXTN to it), served by the chain worker.
            self.assertIs(
                select_experimental_spec_worker(None, SpeculativeAlgorithm.EAGLE, None),
                TpuChainSpecWorker,
            )
        with (
            envs.SGLANG_TPU_ENABLE_EXPERIMENTAL_SPEC_WORKER.override(False),
            envs.SGLANG_TPU_HOST_PADDING_NO_D2H.override(False),
        ):
            self.assertIsNone(
                select_experimental_spec_worker(None, SpeculativeAlgorithm.EAGLE3, None)
            )

    def test_dp_ranks_agree_on_a_bucket_the_draft_padding_keeps(self):
        """Every DP rank derives the same decode bucket from the synced sizes, equal
        to the draft's DP peer rows (aligned to attention TP), even when idle."""
        worker = _chain_worker()
        worker._dp_attention = True
        worker._attn_tp_size = 4
        buckets = [1, 2, 4, 8, 16]
        worker._b_bucketer = lambda: SimpleNamespace(
            buckets=buckets, next_bucket=lambda n: next(b for b in buckets if b >= n)
        )
        worker._pad_bs = lambda n: worker._b_bucketer().next_bucket(n)
        synced = SimpleNamespace(global_num_tokens=[1, 5, 0])
        self.assertEqual(
            [TpuSpecWorker._decode_bucket(worker, synced, bs) for bs in (1, 5, 0)],
            [8, 8, 8],
        )
        self.assertEqual(TpuSpecWorker._decode_buckets_for(worker, 1), [4, 8, 16])
        worker._dp_attention = False
        self.assertEqual(TpuSpecWorker._decode_bucket(worker, synced, 1), 1)
        self.assertEqual(TpuSpecWorker._decode_buckets_for(worker, 3), [4])

    def test_dp_unpadding_keeps_host_padded_extend_outputs(self):
        """DP attention's post-forward unpadding trims outputs to live rows; a
        host-padded extend keeps the bucket rows its samplers and draft read."""

        def trim(forward_batch, out):
            out.next_token_logits = out.next_token_logits[:1]
            out.hidden_states = out.hidden_states[:6]

        def outputs():
            return LogitsProcessorOutput(
                next_token_logits=torch.zeros((4, 3)),
                hidden_states=torch.zeros((16, 2)),
            )

        batch = SimpleNamespace(
            sampling_info=SimpleNamespace(), forward_mode=ForwardMode.EXTEND
        )
        kept, trimmed = outputs(), outputs()
        with patch(
            "sglang_tpu.experimental.sampling.sampling_batch_info.is_host_padded",
            side_effect=lambda info: info is batch.sampling_info,
        ):
            keep_host_padded_extend_outputs(trim, batch, kept)
            batch.forward_mode = ForwardMode.TARGET_VERIFY
            keep_host_padded_extend_outputs(trim, batch, trimmed)
        self.assertEqual(kept.next_token_logits.shape[0], 4)
        self.assertEqual(kept.hidden_states.shape[0], 16)
        self.assertEqual(trimmed.hidden_states.shape[0], 6)

    def test_bucket_state_repads_when_the_shared_bucket_moves(self):
        worker = _chain_worker()
        live = SpecState(
            bonus_tokens=torch.tensor([7]),
            seq_lens=torch.tensor([11]),
            req_pool_indices=torch.tensor([2]),
        )
        worker.get_spec_state = lambda batch: live
        sampling_info = HostPaddedSamplingInfo(
            temperatures=torch.full((1, 1), 0.5),
            top_ps=torch.full((1,), 0.9),
            top_ks=torch.full((1,), 3, dtype=torch.int32),
            min_ps=torch.zeros(1),
            is_all_greedy=False,
            is_any_greedy=False,
            need_top_p_sampling=True,
            need_top_k_sampling=True,
            need_min_p_sampling=False,
            vocab_size=16,
            device="cpu",
            real_batch_size=1,
        )
        key = torch.tensor([2])
        batch = SimpleNamespace(req_pool_indices=key, sampling_info=sampling_info)

        def bucket(rows):
            return TpuSpecWorker._bucket_state(
                worker,
                batch,
                rows,
                SimpleNamespace(
                    req_pool_indices=pad_rows(key, rows), future_indices=key
                ),
            )

        worker._bucket = None
        worker._bucket = bucket(1)
        # Same composition, but DP peers grew the shared bucket: re-pad.
        grown = bucket(4)
        self.assertEqual(grown.state.seq_lens.tolist(), [11, 0, 0, 0])
        torch.testing.assert_close(grown.temperatures, torch.tensor([0.5, 1, 1, 1]))
        self.assertEqual(grown.top_ks.tolist(), [3, 1, 1, 1])
        worker._bucket = grown
        self.assertIs(bucket(4), grown)
        worker._bucket = None
        live = live._replace(draft_carry=_OptionalCarry(torch.ones((1, 2))))
        carry = bucket(4).state.draft_carry
        self.assertIsNone(carry.optional)
        self.assertEqual(carry.hidden_states.tolist(), [[1, 1], [0, 0], [0, 0], [0, 0]])

    def test_target_only_acceptance_commits_target_distribution(self):
        """The first committed token follows p_0 whatever d_1 is, with the
        rejected draft excluded from the bonus; top_k <= 1 rows stay greedy."""
        torch.manual_seed(0)
        p0 = torch.tensor([0.1, 0.2, 0.3, 0.4])
        rows = 40000
        probs = torch.stack([p0, torch.full((4,), 0.25)]).repeat(rows, 1)
        drafts = torch.full((rows, 1), 2)
        with patch.object(spec_ops, "tpu_filter_probs", _cpu_joint_filter):
            num_correct, bonus = accept_drafts_target_only(
                probs.log(),
                drafts,
                torch.ones(rows, dtype=torch.int32),
                top_ks=torch.full((rows,), 4, dtype=torch.int32),
                threshold_single=1.0,
                threshold_acc=1.0,
                rng_step=torch.zeros(1, dtype=torch.int64),
                **_unfiltered(rows),
            )
        first = torch.where(num_correct > 0, drafts[:, 0], bonus)
        freq = torch.bincount(first, minlength=4).float() / rows
        torch.testing.assert_close(freq, p0, atol=0.01, rtol=0)
        rejected = num_correct == 0
        self.assertFalse(bool((bonus[rejected] == 2).any()))

        with patch.object(spec_ops, "tpu_filter_probs", _cpu_joint_filter):
            greedy_correct, greedy_bonus = accept_drafts_target_only(
                probs[:4].log(),
                torch.tensor([[3], [1]]),
                torch.ones(2, dtype=torch.int32),
                top_ks=torch.ones(2, dtype=torch.int32),
                threshold_single=1.0,
                threshold_acc=1.0,
                rng_step=torch.zeros(1, dtype=torch.int64),
                **_unfiltered(2),
            )
        self.assertEqual(greedy_correct.tolist(), [1, 0])
        self.assertEqual(greedy_bonus.tolist(), [0, 3])

    def test_sampled_draft_acceptance_commits_filtered_target(self):
        """With a sampled draft, filtering only p stays lossless: the first committed
        token follows the filtered p_0, including tokens the filter dropped from q."""
        torch.manual_seed(0)
        rows = 40000
        target = torch.tensor([0.1, 0.4, 0.4, 0.1])
        p0 = torch.tensor([0.0, 0.5, 0.5, 0.0])  # top-k 2 of the target
        q_logits = torch.log(torch.tensor([0.4, 0.3, 0.2, 0.1]))
        drafts = torch.multinomial(q_logits.exp(), rows, replacement=True)[:, None]
        target_logits = torch.stack([target, torch.full((4,), 0.25)]).log()
        with patch.object(spec_ops, "tpu_filter_probs", _cpu_joint_filter):
            num_correct, bonus = accept_drafts_sampled_filtered(
                target_logits.repeat(rows, 1),
                drafts,
                torch.ones(rows, dtype=torch.int32),
                q_logits.expand(rows, 1, 4),
                top_ks=torch.full((rows,), 2, dtype=torch.int32),
                rng_step=torch.zeros(1, dtype=torch.int64),
                **_unfiltered(rows),
            )
        first = torch.where(num_correct > 0, drafts[:, 0], bonus)
        freq = torch.bincount(first, minlength=4).float() / rows
        torch.testing.assert_close(freq, p0, atol=0.01, rtol=0)

    def test_candidates_acceptance_is_the_filtered_full_vocab_rule(self):
        """On a top-k-only target and a q on K candidates, the candidates rule
        decides the same correct prefix as the full-vocabulary filtered rule
        from the same uniforms, and its bonus keeps the first committed token on
        the filtered p_0."""
        torch.manual_seed(0)
        bs, n, vocab, k_draft = 4, 3, 32, 5
        logits = torch.randn(bs * (n + 1), vocab)
        draft_ids = torch.stack(
            [torch.randperm(vocab)[:k_draft] for _ in range(bs * n)]
        ).view(bs, n, k_draft)
        draft_logits = torch.randn(bs, n, k_draft)
        drafts = draft_ids[..., 0]
        dense_q = torch.full((bs, n, vocab), float("-inf")).scatter(
            -1, draft_ids, draft_logits
        )
        args = dict(
            draft_tokens=drafts,
            num_verify_drafts=torch.full((bs,), n, dtype=torch.int32),
            temperatures=torch.tensor([0.7, 1.0, 1.3, 1.0]),
            top_ks=torch.tensor([4, 8, 2, 1], dtype=torch.int32),
            rng_step=torch.tensor([3]),
        )
        with (
            patch.object(spec_ops, "tpu_filter_probs", _cpu_joint_filter),
            patch.object(spec_ops, "TARGET_CANDIDATES", 8),
        ):
            full_correct, _ = accept_drafts_sampled_filtered(
                logits,
                draft_logits=dense_q,
                top_ps=torch.ones(bs),
                min_ps=torch.zeros(bs),
                **args,
            )
            cand_correct, cand_bonus = accept_drafts_candidates(
                *torch.topk(logits, 8),
                draft_ids=draft_ids,
                draft_logits=draft_logits,
                **args,
            )
        self.assertEqual(cand_correct.tolist(), full_correct.tolist())
        # The greedy row (top_k 1) commits the raw argmax.
        greedy_bonus = logits.view(bs, n + 1, -1)[3].argmax(-1)[cand_correct[3]]
        self.assertEqual(cand_bonus[3].item(), greedy_bonus.item())

        rows = 40000
        target = torch.tensor([0.1, 0.4, 0.3, 0.2]).log()
        p0 = torch.tensor([0.0, 4 / 7, 3 / 7, 0.0])  # top-k 2 of the target
        q_ids = torch.tensor([0, 1, 3])
        q_logits = torch.tensor([0.5, 0.3, 0.2]).log()
        picks = torch.multinomial(q_logits.exp(), rows, replacement=True)
        with patch.object(spec_ops, "TARGET_CANDIDATES", 4):
            num_correct, bonus = accept_drafts_candidates(
                *torch.topk(torch.stack([target, torch.zeros(4)]).repeat(rows, 1), 4),
                q_ids[picks][:, None],
                torch.ones(rows, dtype=torch.int32),
                q_ids.expand(rows, 1, 3),
                q_logits.expand(rows, 1, 3),
                torch.ones(rows),
                torch.full((rows,), 2, dtype=torch.int32),
                torch.tensor([0]),
            )
        first = torch.where(num_correct > 0, q_ids[picks], bonus)
        freq = torch.bincount(first, minlength=4).float() / rows
        torch.testing.assert_close(freq, p0, atol=0.01, rtol=0)

    def test_block_verification_is_lossless_and_accepts_more(self):
        """With a context-free target p and draft q, block verification's output
        stream has its first two tokens distributed p x p, and it accepts at least
        as many drafts as token-wise rejection."""
        # A q far from p: an unscaled correction (r_tau = 1) misses by ~0.007.
        rows, n, vocab = 100000, 3, 4
        p_row = torch.tensor([0.05, 0.6, 0.25, 0.1])
        q_row = torch.tensor([0.5, 0.05, 0.15, 0.3])
        gen = torch.Generator().manual_seed(0)
        ids = torch.arange(vocab)

        def step(rng_step, block_verification):
            drafts = torch.multinomial(
                q_row, rows * n, replacement=True, generator=gen
            ).view(rows, n)
            with patch.object(spec_ops, "TARGET_CANDIDATES", vocab):
                correct, bonus = accept_drafts_candidates(
                    p_row.log().expand(rows * (n + 1), vocab),
                    ids.expand(rows * (n + 1), vocab).int(),
                    drafts,
                    torch.full((rows,), n, dtype=torch.int32),
                    ids.expand(rows, n, vocab),
                    q_row.log().expand(rows, n, vocab),
                    torch.ones(rows),
                    torch.full((rows,), vocab, dtype=torch.int32),
                    torch.tensor([rng_step]),
                    block_verification=block_verification,
                )
            stream = torch.cat([drafts, bonus[:, None]], dim=1)
            # Committed: the correct drafts, then the bonus.
            slots = torch.arange(n + 1)[None, :]
            stream = torch.where(slots < correct[:, None], stream, bonus[:, None])
            return correct, stream

        token_correct, _ = step(1, False)
        block_correct, first_step = step(2, True)
        _, second_step = step(3, True)
        first = first_step[:, 0]
        second = torch.where(block_correct >= 1, first_step[:, 1], second_step[:, 0])
        joint = torch.bincount(first * vocab + second, minlength=vocab * vocab)
        torch.testing.assert_close(
            joint.float().view(vocab, vocab) / rows,
            torch.outer(p_row, p_row),
            atol=0.004,
            rtol=0,
        )
        self.assertGreater(
            block_correct.float().mean().item(), token_correct.float().mean().item()
        )

    def test_hash_uniforms_match_the_uint32_reference(self):
        """The int32-lane hash keeps MurmurHash3's uint32 bits (masked int64 reference)."""
        mask = 0xFFFFFFFF

        def fmix(x):
            x = x ^ (x >> 16)
            x = (x * 0x85EBCA6B) & mask
            x = x ^ (x >> 13)
            x = (x * 0xC2B2AE35) & mask
            return x ^ (x >> 16)

        for step in (0, 7, 2**30 + 3):
            for stream in (0, 1, 2):
                key = fmix((step * 0x9E3779B1 + stream * 0x7F4A7C15) & mask)
                index = torch.arange(3 * 64, dtype=torch.int64)
                bits = fmix(fmix((index * 0x632BE5AB + key) & mask) ^ key)
                expected = ((bits >> 8).float() / (1 << 24)).view(3, 64)
                got = spec_ops.hash_uniforms(torch.tensor([step]), stream, (3, 64))
                self.assertTrue(torch.equal(got, expected))

    def test_top_k_accept_steps_match_full_vocab_steps(self):
        """The verify's distributed top-k (descending logits and ids) commits what
        the full-vocabulary steps commit from the same rng_step."""
        torch.manual_seed(1)
        bs, n, vocab, k_draft, width = 3, 4, 50, 6, 8
        logits = torch.randn(bs * (n + 1), vocab)
        top_logits, top_ids = torch.topk(logits, width)
        draft_ids = torch.stack(
            [torch.randperm(vocab)[:k_draft] for _ in range(bs * n)]
        ).view(bs, n, k_draft)
        draft_tokens = draft_ids[..., 0]
        num_verify_drafts = torch.full((bs,), n, dtype=torch.int32)
        seq_lens = torch.tensor([10, 20, 30])
        live_mask = torch.tensor([True, True, False])
        sampled = (
            draft_ids,
            torch.randn(bs, n, k_draft),
            torch.tensor([1.0, 0.8, 1.0]),
            torch.tensor([width, 3, 1], dtype=torch.int32),
            seq_lens,
            live_mask,
            torch.tensor([5]),
        )
        with patch.object(spec_ops, "TARGET_CANDIDATES", width):
            full, full_step = accept_candidates_step(
                logits, draft_tokens, num_verify_drafts, *sampled, False
            )
            top, top_step = accept_top_k_candidates_step(
                top_logits,
                top_ids.int(),
                draft_tokens,
                num_verify_drafts,
                *sampled,
                False,
            )
        for a, b in zip(full, top):
            self.assertTrue(torch.equal(a, b))
        self.assertEqual(full_step.item(), top_step.item())

        greedy = accept_greedy_step(
            logits, draft_tokens, num_verify_drafts, seq_lens, live_mask
        )
        greedy_top = accept_greedy_top_k_step(
            top_ids.int(), draft_tokens, num_verify_drafts, seq_lens, live_mask
        )
        for a, b in zip(greedy, greedy_top):
            self.assertTrue(torch.equal(a, b))

    def test_greedy_rows_in_a_sampled_batch_compare_raw_logits(self):
        """A greedy row (top_k = 1) sharing a batch with sampled rows takes the
        greedy rule on the raw target logits; probabilities can round equal."""
        # Temperature 1e9 makes both target probs 0.5; the logits still pick 1.
        target_logits = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
        args = dict(
            draft_tokens=torch.tensor([[1]]),
            num_verify_drafts=torch.ones(1, dtype=torch.int32),
            draft_logits=torch.zeros(1, 1, 2),
            temperatures=torch.tensor([1e9]),
            top_ks=torch.ones(1, dtype=torch.int32),
            rng_step=torch.zeros(1, dtype=torch.int64),
        )
        with patch.object(spec_ops, "tpu_filter_probs", _cpu_joint_filter):
            filtered = accept_drafts_sampled_filtered(
                target_logits, top_ps=torch.ones(1), min_ps=torch.zeros(1), **args
            )
        for num_correct, bonus in (
            accept_drafts_sampled(target_logits, **args),
            filtered,
        ):
            self.assertEqual((num_correct.item(), bonus.item()), (1, 1))

    def test_greedy_rows_do_not_switch_the_batch_onto_the_filter(self):
        """Greedy rows set core's need_top_k_sampling (top_k = 1); only sampled
        rows' filters may route verify through the full-vocab target filter."""
        from sglang_tpu.experimental.speculative.base_spec_worker import (
            _sampled_row_filters,
        )

        def filters(top_ks, top_ps=(1.0, 1.0), min_ps=(0.0, 0.0)):
            return _sampled_row_filters(
                SimpleNamespace(
                    vocab_size=32,
                    top_ks=torch.tensor(top_ks, dtype=torch.int32),
                    top_ps=torch.tensor(top_ps),
                    min_ps=torch.tensor(min_ps),
                )
            )

        none = {
            "need_top_k": False,
            "need_top_p": False,
            "need_min_p": False,
            "target_candidates": False,
        }
        self.assertEqual(filters([1, 1 << 30]), none)
        self.assertEqual(filters([1, 1 << 30], top_ps=(0.5, 1.0)), none)
        self.assertEqual(filters([1, 8])["need_top_k"], True)
        self.assertEqual(filters([1, 1 << 30], min_ps=(0.0, 0.1))["need_min_p"], True)
        # Candidates acceptance covers a top-k-only target within TARGET_CANDIDATES.
        self.assertTrue(filters([1, 64])["target_candidates"])
        self.assertFalse(filters([1, 65])["target_candidates"])
        self.assertFalse(filters([1, 8], top_ps=(1.0, 0.9))["target_candidates"])

    def test_sampled_verify_filters_target_like_the_target_sampler(self):
        """Sampled verify keeps what the target sampler keeps: every cutoff on the
        unfiltered p, each request's values on all its N + 1 rows, one renorm."""
        p = torch.tensor([[0.40, 0.25, 0.20, 0.15], [0.50, 0.30, 0.15, 0.05]])
        with patch.object(spec_ops, "tpu_filter_probs", _cpu_joint_filter):
            filtered = _filtered_target_probs(
                p.log().repeat_interleave(2, dim=0),
                torch.ones(2),
                # Renormalizing after top-k would push 0.40 past top_p and drop 0.25.
                torch.tensor([2, 1 << 30], dtype=torch.int32),
                torch.tensor([0.6, 1.0]),
                torch.tensor([0.0, 0.5]),
                num_verify_rows=2,
            )
        expected = torch.tensor(
            [[0.40 / 0.65, 0.25 / 0.65, 0, 0], [0.50 / 0.80, 0.30 / 0.80, 0, 0]]
        ).repeat_interleave(2, dim=0)
        torch.testing.assert_close(filtered, expected)

    def test_draft_extend_reads_valid_tokens_at_post_write_lengths(self):
        """Draft extend rewrites the whole verify window: rows past the accepted
        prefix must be valid ids, and the length counts every window row."""
        accept = finalize_accept(
            torch.tensor([[5, 7], [8, 9]]),
            torch.tensor([1, 0]),
            torch.tensor([6, 4]),
            torch.tensor([10, 0]),
            torch.tensor([True, False]),
        )
        self.assertEqual(accept.window_tokens.tolist(), [5, 6, 0, 4, 0, 0])
        self.assertEqual(accept.window_seq_lens.tolist(), [13, 3])

    def test_filtered_accept_warmup_passes_unaliased_params(self):
        """Serving uploads each sampling param separately; a warmup passing one
        tensor for two of them leaves a Dynamo alias guard serving never meets."""
        worker = _chain_worker()
        worker.num_drafts = 2
        worker._target_worker = SimpleNamespace(
            model_runner=SimpleNamespace(model_config=SimpleNamespace(vocab_size=8))
        )
        worker._all_drafts = {4: torch.full((4,), 2, dtype=torch.int32)}
        worker._accept_threshold_single = worker._accept_threshold_acc = 1.0
        worker._rng_step = torch.zeros(1, dtype=torch.int64)
        calls = []
        with (
            patch.object(
                TpuChainSpecWorker,
                "_b_bucketer",
                return_value=SimpleNamespace(buckets=[4]),
            ),
            patch(
                "sglang_tpu.experimental.speculative.base_spec_worker."
                "warmup_compiled_op",
                lambda op, *args: calls.append(args),
            ),
        ):
            worker._warmup_target_only_accept()
        logits, _, _, temperatures, top_ks, top_ps, min_ps, *_ = calls[0]
        self.assertEqual((logits.shape, logits.dtype), ((12, 8), torch.float32))
        self.assertEqual(len({id(t) for t in (temperatures, top_ps, min_ps)}), 3)
        self.assertEqual(
            [t.dtype for t in (temperatures, top_ks, top_ps, min_ps)],
            [torch.float32, torch.int32, torch.float32, torch.float32],
        )

    def test_extend_sampling_warmup_broadcasts_every_sampled_shape(self):
        """extend() TP-broadcasts its sampled tokens; a sampled shape warmup never
        broadcast compiles that collective at serving time."""
        worker = _chain_worker()
        sampled: set[tuple[tuple[int, ...], torch.dtype]] = set()

        def sampler(logits_output, *_args):
            tokens = torch.zeros((logits_output.next_token_logits.shape[0],))
            tokens = tokens.to(torch.int32)
            sampled.add((tuple(tokens.shape), tokens.dtype))
            return tokens

        broadcast: set[tuple[SpecTpSyncSite, tuple[int, ...], torch.dtype]] = set()

        def sync(site, values):
            broadcast.add((site, tuple(values.shape), values.dtype))
            return values

        worker._tp_sync = SimpleNamespace(sync=sync)
        worker._dp_attention = False
        worker._target_worker = SimpleNamespace(
            model_runner=SimpleNamespace(
                sampler=sampler, model_config=SimpleNamespace(vocab_size=8)
            )
        )
        policy = SimpleNamespace(
            _extend_compile_cap=lambda: 4, _compiles_logits_processor=lambda: True
        )
        buckets = SimpleNamespace(next_bucket=lambda bs: 1 << (bs - 1).bit_length())
        with (
            patch.object(torch, "tpu", create=True) as tpu,
            patch.object(TpuChainSpecWorker, "_compile_policy", return_value=policy),
            patch.object(TpuChainSpecWorker, "_b_bucketer", return_value=buckets),
        ):
            worker._warmup_extend_sampling()
        tpu.current_stream.assert_called_once_with()
        tpu.current_stream.return_value.synchronize.assert_called_once_with()
        self.assertEqual({shape for shape, _ in sampled}, {(1,), (2,), (4,)})
        self.assertEqual(
            broadcast,
            {(SpecTpSyncSite.SPEC_TARGET, *shape_dtype) for shape_dtype in sampled},
        )

    def test_draft_prompt_shift_keeps_chunks_consistent(self):
        """Each request's last row takes its own sampled token, except a non-final
        chunk, which takes its next prompt token; placeholder ids stay in vocab."""
        # 99999 stands in for an image placeholder hash id.
        shifted, row_reqs, is_sampled = shift_prompt_tokens(
            torch.tensor([10, 99999, 12, 20, 21]),
            extend_lens=[3, 2],
            padded_tokens=8,
            chunked_row=1,
            next_prompt_token=22,
            vocab_size=64,
        )
        filled = fill_sampled_tails(
            shifted, row_reqs, is_sampled, torch.tensor([7, 8, 0, 0])
        )
        self.assertEqual(filled.tolist(), [63, 12, 7, 21, 22, 0, 0, 0])

    def test_eagle3_draft_maps_draft_vocab_to_target(self):
        """d2t maps draft ids to target ids; q puts the draft mass on those target
        ids and nothing elsewhere, so sampled acceptance stays lossless."""
        worker = object.__new__(TpuEagle3DraftWorker)
        worker.target_vocab_size = 6
        worker._draft_model = SimpleNamespace(hot_token_id=torch.tensor([4, 1, 5]))
        self.assertEqual(worker.target_tokens(torch.tensor([0, 2])).tolist(), [4, 5])
        dense = worker.target_logits(torch.tensor([[1.0, 2.0, 3.0]]))
        self.assertEqual(dense[0, [4, 1, 5]].tolist(), [1.0, 2.0, 3.0])
        self.assertTrue(torch.isneginf(dense[0, [0, 2, 3]]).all())
        worker._draft_model = SimpleNamespace(hot_token_id=None)
        ids = torch.tensor([3])
        self.assertIs(worker.target_tokens(ids), ids)

    def test_eagle3_first_draft_norms_the_pre_norm_carry(self):
        """The carry is the draft's pre-norm hidden (its next-step input); d_1's
        logits come from the normed hidden, unless the draft already outputs normed."""
        worker = object.__new__(TpuEagle3DraftWorker)
        norm = lambda hidden: hidden * 2  # noqa: E731
        worker._draft_model = SimpleNamespace(
            model=SimpleNamespace(norm=norm), lm_head=SimpleNamespace()
        )
        carry = torch.ones((1, 2))
        with (
            patch.object(
                eagle3_draft_worker,
                "project_through_lm_head",
                lambda hidden, lm_head: hidden,
            ),
            patch.object(
                eagle3_draft_worker,
                "gather_and_crop_vocab",
                lambda logits, lm_head: logits,
            ),
        ):
            worker._carry_is_normed = False
            self.assertEqual(worker.first_draft_logits(carry).tolist(), [[2.0, 2.0]])
            worker._carry_is_normed = True
            self.assertEqual(worker.first_draft_logits(carry).tolist(), [[1.0, 1.0]])

    def test_draft_rows_return_the_eagle3_aux_hidden(self):
        """Commit's draft rows skip the lm_head, but must hand back what FULL capture
        would: an EAGLE3 draft's aux (pre-norm) hidden, not its normed logits input."""
        normed, aux = torch.zeros((2, 3)), torch.ones((2, 3))
        out = _hidden_only_logits(None, normed, None, None, [aux])
        self.assertIs(out.hidden_states, aux)
        self.assertIs(
            _hidden_only_logits(None, normed, None, None).hidden_states, normed
        )

    def test_pd_prefill_ships_the_carried_hidden(self):
        """A PD prefill hands core one seed row per request: the carried hidden at
        the extend cap (copied to host whole), topk as host placeholders."""
        draft_input = HostPaddedChainDraftInput(
            bonus_tokens=torch.tensor([1, 2]), new_seq_lens=torch.tensor([3, 4])
        )
        tail_hidden = torch.arange(8.0).view(4, 2)
        _attach_pd_seeds(draft_input, tail_hidden=tail_hidden, bs=2)
        self.assertIs(draft_input.hidden_states, tail_hidden)
        self.assertEqual(draft_input.topk_p.shape, (2, 1))
        self.assertEqual(draft_input.topk_index.device.type, "cpu")

    def test_multimodal_draft_prompt_reads_the_next_rows_embeds(self):
        """Row p of the draft prompt embeds token p + 1 from the target's embeds;
        each request's last row is re-embedded from the token appended to it."""
        embeds = torch.arange(10.0).view(5, 2)
        tail_embeds = torch.tensor([[-1.0, -1.0], [-2.0, -2.0], [-1.0, -1.0]])
        # Requests end at rows 1 and 4; the padded slot repeats request 0's row.
        out = fold_prompt_tail_embeds(embeds, torch.tensor([1, 4, 1]), tail_embeds)
        self.assertEqual(
            out.tolist(),
            [[2.0, 3.0], [-1.0, -1.0], [6.0, 7.0], [8.0, 9.0], [-2.0, -2.0]],
        )

    def test_dp_attention_drafts_greedy(self):
        """Under DP attention the draft never samples: peers and idle ranks must
        issue the same draft sync, and sampled requests take target-only accept."""
        worker = _chain_worker()
        worker.num_drafts = 2
        seen = []
        worker._draft_chain = lambda **kwargs: seen.append(kwargs["temperatures"])
        bucket = SimpleNamespace(
            state=None, temperatures=torch.ones(4), top_ks=torch.ones(4)
        )
        verify_input = SimpleNamespace(
            req_pool_indices_cpu=None, padded_kv_lens_after_rows=lambda rows: (None, 0)
        )
        for sampled in (True, False):
            worker._sampled_drafts = sampled
            worker.draft(None, bucket, None, verify_input)
        self.assertIs(seen[0], bucket.temperatures)
        self.assertIsNone(seen[1])

    def _fused_drafter(self, fused_draft):
        drafter = object.__new__(TpuEagle3DraftWorker)
        calls = []
        shim = SimpleNamespace(wait_for_hicache_reload=lambda: calls.append("wait"))
        drafter._compile_policy = lambda: SimpleNamespace(compiled_forward_shim=shim)
        drafter.model_runner = SimpleNamespace(
            attn_backend=SimpleNamespace(init_forward_metadata=lambda batch: None)
        )
        drafter._carry_table = torch.zeros((1, 1))
        drafter._fused_draft = lambda *args: fused_draft(calls)
        return drafter, calls

    @staticmethod
    def _run_fused(drafter):
        drafter.run_fused_draft(
            SimpleNamespace(mark_forward_metadata_ready=lambda: None),
            req_pool_indices=None,
            positions=None,
            cache_locs=None,
            seq_lens=None,
            temperatures=None,
            top_ks=None,
            uniforms=None,
        )

    def test_fused_draft_runs_as_a_decode_program(self):
        """The fused draft guards on is_extend_in_batch; a stale True from the last
        extend under DP attention would recompile it under fail_on_recompile."""
        from sglang.srt.layers.dp_attention import (
            get_is_extend_in_batch,
            set_is_extend_in_batch,
        )

        seen = []
        drafter, _ = self._fused_drafter(
            lambda calls: seen.append(get_is_extend_in_batch()) or ("d", None)
        )
        set_is_extend_in_batch(True)
        self._run_fused(drafter)
        self.assertEqual(seen, [False])

    def test_fused_draft_waits_for_the_hicache_reload_first(self):
        """The fused draft program inlines the forward shim, whose HiCache reload gate
        only runs eagerly; without the eager wait a PD decode drafts on draft KV
        a host reload has not landed yet."""
        drafter, calls = self._fused_drafter(
            lambda calls: calls.append("draft") or ("d", None)
        )
        self._run_fused(drafter)
        self.assertEqual(calls, ["wait", "draft"])


class _IdentityChainDrafter(TpuChainDraftWorker):
    """A chain drafter whose logits are its hidden, over the target vocab."""

    def first_draft_logits(self, carry):
        return carry

    def target_tokens(self, draft_ids):
        return draft_ids

    def target_logits(self, draft_logits):
        return draft_logits


def _publish_topk(*, forward_mode, spec_info, topk, reuse_dsa_topk_indices=False):
    """What the NextN model does with its indexer rows (DeepseekModelNextN.forward)."""
    forward_batch = SimpleNamespace(
        forward_mode=forward_mode,
        spec_info=spec_info,
        reuse_dsa_topk_indices=reuse_dsa_topk_indices,
    )
    share = IndexTopKShareState.from_mtp_carry(forward_batch)
    share.update(topk)
    share.publish()


def _capture_layout(spec_info):
    """(shape, dtype) of the DSA top-k capture and its row select, or None."""
    return tuple(
        None if t is None else (tuple(t.shape), t.dtype)
        for t in (spec_info.dsa_seed_topk_capture, spec_info.dsa_seed_topk_select)
    )


class TestChainDsaIndexShare(CustomTestCase):
    """MTP index sharing: draft steps reuse the indexer top-k of the last committed row."""

    WIDTH = 4  # DSA top-k columns
    HIDDEN = 2

    def _seed_table(self, rows: int = 5) -> torch.Tensor:
        return torch.full((rows, self.WIDTH), -1, dtype=torch.int32)

    def _drafter(self, seed_table, rows: int = 5):
        return SimpleNamespace(
            draft_context=contextlib.nullcontext,
            carry=torch.zeros((rows, self.HIDDEN)),
            dsa_seed_topk_width=None if seed_table is None else self.WIDTH,
            dsa_seed_topk_table=seed_table,
        )

    def _commit(self, *, seed_table, rows_topk, num_correct_drafts, req_pool_indices):
        """TpuChainSpecWorker.commit around a draft forward whose indexer yields
        rows_topk; returns the forward's spec_info, flag and hidden rows, and the
        drafter's carry and seed table after commit."""
        padded_bs = req_pool_indices.shape[0]
        width = rows_topk.shape[0] // padded_bs
        worker = object.__new__(TpuChainSpecWorker)
        worker.device = "cpu"
        worker.num_verify_rows = width
        drafter = self._drafter(seed_table)
        worker._chain_draft_worker = drafter
        worker._draft_runner = worker._spec_algorithm = None
        served = []

        def forward_draft_rows(*, spec_info, publish_dsa_topk=False, **_):
            _publish_topk(
                forward_mode=ForwardMode.TARGET_VERIFY,
                spec_info=spec_info,
                topk=rows_topk,
                reuse_dsa_topk_indices=publish_dsa_topk,
            )
            hidden = torch.arange(
                rows_topk.shape[0] * self.HIDDEN, dtype=torch.float32
            ).view(-1, self.HIDDEN)
            served.append(
                SimpleNamespace(
                    spec_info=spec_info,
                    publish_dsa_topk=publish_dsa_topk,
                    rows_hidden=hidden,
                )
            )
            return LogitsProcessorOutput(next_token_logits=None, hidden_states=hidden)

        accept = SimpleNamespace(
            window_tokens=None,
            live_mask=None,
            num_correct_drafts=num_correct_drafts,
            bonus_tokens=torch.zeros(padded_bs, dtype=torch.int64),
            next_seq_lens=torch.zeros(padded_bs, dtype=torch.int64),
        )
        state = SpecState(
            bonus_tokens=accept.bonus_tokens,
            seq_lens=accept.next_seq_lens,
            req_pool_indices=req_pool_indices,
        )
        with (
            patch.object(chain_spec_worker, "build_chain_block_inputs"),
            patch.object(chain_spec_worker, "forward_draft_rows", forward_draft_rows),
        ):
            worker.commit(
                None,
                state,
                None,
                None,
                SimpleNamespace(hidden_states=torch.zeros((rows_topk.shape[0], 1))),
                accept,
            )
        (forward,) = served
        forward.carry = drafter.carry
        forward.seed_table = drafter.dsa_seed_topk_table
        return forward

    def _extend(self, *, seed_table, prompt_topk, extend_lens, req_pool_indices, cap):
        """TpuChainSpecWorker._extend_draft around a draft forward that publishes prompt_topk."""
        worker = object.__new__(TpuChainSpecWorker)
        worker.device = "cpu"
        worker._chain_draft_worker = self._drafter(seed_table)
        worker._compile_policy = lambda: SimpleNamespace(
            _extend_compile_cap=lambda: cap
        )
        worker._draft_prompt_ids = lambda batch, **_: None
        served = []

        def forward(forward_batch):
            served.append(forward_batch.spec_info)
            _publish_topk(
                forward_mode=ForwardMode.EXTEND,
                spec_info=forward_batch.spec_info,
                topk=prompt_topk,
            )
            tail = torch.zeros((cap, self.HIDDEN))
            return SimpleNamespace(
                logits_output=SimpleNamespace(
                    next_token_logits=tail, hidden_states=tail
                )
            )

        worker._draft_runner = SimpleNamespace(forward=forward)
        batch = SimpleNamespace(
            extend_lens=extend_lens, req_pool_indices_cpu=req_pool_indices
        )
        init_new = lambda *args, **kwargs: SimpleNamespace(  # noqa: E731
            forward_mode=ForwardMode.EXTEND, global_num_tokens_cpu=None
        )
        with patch.object(chain_spec_worker.ForwardBatch, "init_new", init_new):
            worker._extend_draft(
                batch,
                target_hidden=torch.zeros((prompt_topk.shape[0], 1)),
                bucket_tokens=None,
                mm_input_embeds=None,
            )
        return served[0]

    def test_draft_steps_attend_over_the_committed_row_topk(self):
        """Every draft step hands the NextN layer the seed of its request's pool row
        as the carried top-k, so the layer skips its indexer; a draft that does
        not share (EAGLE3, or the flag off) keeps running its indexer each step."""
        table = torch.arange(5 * self.WIDTH, dtype=torch.int32).view(5, self.WIDTH)
        req_pool_indices = torch.tensor([2, 0])
        for seed_table, expected in ((table, table[[2, 0]]), (None, None)):
            seen = []

            def forward(input_ids, positions, forward_batch):
                del positions
                share = IndexTopKShareState.from_mtp_carry(forward_batch)
                seen.append(share.topk_indices)
                logits = torch.zeros((input_ids.shape[0], self.HIDDEN))
                return SimpleNamespace(next_token_logits=logits, hidden_states=logits)

            drafter = object.__new__(_IdentityChainDrafter)
            drafter.num_drafts = 3
            drafter.model_runner = SimpleNamespace(
                model=SimpleNamespace(forward=forward)
            )
            forward_batch = SimpleNamespace(
                spec_info=chain_draft_forward_input(
                    hidden_states=torch.zeros((2, self.HIDDEN)), width=1
                ),
                reuse_dsa_topk_indices=False,
            )
            with self.subTest(shares=seed_table is not None):
                drafter._fused_draft_steps(
                    forward_batch,
                    torch.zeros((5, self.HIDDEN)),
                    req_pool_indices,
                    torch.zeros((2, 3), dtype=torch.int64),
                    torch.zeros((2, 3), dtype=torch.int64),
                    torch.zeros(2, dtype=torch.int64),
                    None,
                    None,
                    None,
                    seed_table,
                )
                # d_1 comes from the carry; the other two drafts run the layer.
                self.assertEqual(len(seen), 2)
                for topk in seen:
                    if expected is None:
                        self.assertIsNone(topk)
                    else:
                        self.assertTrue(torch.equal(topk, expected))

    def test_commit_carries_and_seeds_the_last_accepted_row(self):
        """Commit runs the draft over all N + 1 verify rows; each request's last
        accepted row (num_correct_drafts) gives its carry and, with sharing, its
        seed top-k, both written at its pool row. Without sharing no top-k is
        published and the carry still moves."""
        rows_topk = torch.arange(6 * self.WIDTH, dtype=torch.int32).view(6, self.WIDTH)
        for seed_table in (self._seed_table(), None):
            with self.subTest(shares=seed_table is not None):
                served = self._commit(
                    seed_table=seed_table,
                    rows_topk=rows_topk,
                    num_correct_drafts=torch.tensor([2, 0]),
                    req_pool_indices=torch.tensor([3, 1]),
                )
                # Three verify rows per request: request 0 keeps row 2, request 1 row 3.
                carry, hidden = served.carry, served.rows_hidden
                self.assertEqual(carry[3].tolist(), hidden[2].tolist())
                self.assertEqual(carry[1].tolist(), hidden[3].tolist())
                self.assertTrue((carry[[0, 2, 4]] == 0).all())
                self.assertEqual(served.publish_dsa_topk, seed_table is not None)
                table = served.seed_table
                if seed_table is None:
                    self.assertIsNone(table)
                    continue
                self.assertEqual(table[3].tolist(), rows_topk[2].tolist())
                self.assertEqual(table[1].tolist(), rows_topk[3].tolist())
                self.assertTrue((table[[0, 2, 4]] == -1).all())

    def test_extend_seeds_each_request_from_its_last_prompt_row(self):
        """The draft prompt's seed is each request's last row (a chunk's last row
        for a chunked request); padded slots land in the reserved pool row 0."""
        table = self._seed_table()
        prompt_topk = torch.arange(8 * self.WIDTH, dtype=torch.int32).view(
            8, self.WIDTH
        )
        self._extend(
            seed_table=table,
            prompt_topk=prompt_topk,
            extend_lens=[3, 2],
            req_pool_indices=torch.tensor([4, 2]),
            cap=4,
        )
        self.assertEqual(table[4].tolist(), prompt_topk[2].tolist())
        self.assertEqual(table[2].tolist(), prompt_topk[4].tolist())
        self.assertTrue((table[[1, 3]] == -1).all())

    def test_warmup_traces_the_capture_serving_passes(self):
        """Commit's draft forward publishes its top-k, extend's fills a top-k
        capture; the warmup must trace the same flag and capture layout, or the
        forward recompiles at serving time."""
        padded_bs, width, cap = 2, 3, 4
        served_commit = self._commit(
            seed_table=self._seed_table(),
            rows_topk=torch.zeros((padded_bs * width, self.WIDTH), dtype=torch.int32),
            num_correct_drafts=torch.zeros(padded_bs, dtype=torch.int64),
            req_pool_indices=torch.tensor([1, 0]),
        )
        served_extend = self._extend(
            seed_table=self._seed_table(),
            prompt_topk=torch.zeros((8, self.WIDTH), dtype=torch.int32),
            extend_lens=[8],
            req_pool_indices=torch.tensor([1]),
            cap=cap,
        )
        policy = object.__new__(TpuChainSpecDecodeCompilePolicyMixin)
        policy.draft_token_num = width
        policy.b_bucketer = SimpleNamespace(buckets=[padded_bs])
        policy.t_bucketer = SimpleNamespace(buckets=[8])
        policy._target_is_multimodal = False
        policy._extend_compile_cap = lambda: cap
        policy.model_runner = SimpleNamespace(
            device="cpu",
            dtype=torch.float32,
            spec_algorithm=SpeculativeAlgorithm.EAGLE,
        )
        policy._precompile_shapes = lambda desc, items, run: [run(i) for i in items]
        warmed = []
        policy._run_one = lambda **kwargs: warmed.append(kwargs)
        policy._precompile_draft_rows(
            None, target_feature_size=1, dsa_seed_topk_width=self.WIDTH
        )
        policy._precompile_draft_prompt_extend(1, dsa_seed_topk_width=self.WIDTH)
        warm_commit, warm_extend = warmed
        self.assertTrue(served_commit.publish_dsa_topk)
        self.assertEqual(
            warm_commit.get("publish_dsa_topk", False), served_commit.publish_dsa_topk
        )
        self.assertEqual(
            _capture_layout(warm_commit["spec_info"]),
            _capture_layout(served_commit.spec_info),
        )
        self.assertIsNotNone(served_extend.dsa_seed_topk_select)
        self.assertEqual(
            _capture_layout(warm_extend["spec_info"]), _capture_layout(served_extend)
        )


if __name__ == "__main__":
    unittest.main()
