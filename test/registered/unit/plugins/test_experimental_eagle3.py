"""CPU contracts for EAGLE3 on the experimental TPU spec worker."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers import overlap_utils
from sglang.srt.managers.overlap_utils import FutureMap, RelayPayload
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardMode,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.envs import envs
from sglang_tpu.experimental.compilation.eagle3_compile_policy import (
    TpuEagle3SpecDecodeCompilePolicyMixin,
)
from sglang_tpu.experimental.sampling.sampling_batch_info import HostPaddedSamplingInfo
from sglang_tpu.experimental.speculative import base_spec_worker
from sglang_tpu.experimental.speculative.base_spec_worker import TpuSpecWorker
from sglang_tpu.experimental.speculative.eagle.eagle_worker import (
    EagleSeed,
    TpuEagle3Worker,
    _shifted_prompt,
    draft_extend_inputs,
    fill_prefill_tail,
)
from sglang_tpu.experimental.speculative.eagle_info import HostPaddedEagleDraftInput
from sglang_tpu.experimental.speculative.spec_ops import (
    accept_drafts_sampled,
    accept_drafts_sampled_filtered,
    accept_drafts_target_only,
)
from sglang_tpu.experimental.speculative.spec_types import (
    AcceptResult,
    BucketState,
    SpecState,
)
from sglang_tpu.experimental.speculative.spec_utils import pad_rows
from sglang_tpu.hooks.experimental_spec import (
    keep_host_padded_extend_outputs,
)
from sglang_tpu.hooks.spec_worker import select_experimental_spec_worker
from sglang_tpu.overlap_utils_tpu import (
    device_accept_resolved_tpu,
    resolve_device_accept_seq_lens_tpu,
)

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

HIDDEN = 4


def _eagle_worker() -> TpuEagle3Worker:
    """A TpuEagle3Worker carrying only the attributes its state conversions read."""
    worker = object.__new__(TpuEagle3Worker)
    worker.device = "cpu"
    worker._hidden_size = HIDDEN
    worker._hidden_dtype = torch.bfloat16
    worker.num_verify_rows = 3
    return worker


def _seed(rows: int, offset: int = 0) -> EagleSeed:
    return EagleSeed(
        topk_p=torch.ones((rows, 1)),
        topk_index=torch.arange(rows).view(-1, 1) + offset,
        hidden_states=(torch.arange(rows * HIDDEN).view(rows, HIDDEN) + offset).to(
            torch.bfloat16
        ),
    )


class TestExperimentalEagle3(CustomTestCase):
    def test_flag_routes_eagle3_and_leaves_other_algorithms_alone(self):
        with envs.SGLANG_TPU_ENABLE_EXPERIMENTAL_SPEC_WORKER.override(True):
            self.assertIs(
                select_experimental_spec_worker(
                    None, SpeculativeAlgorithm.EAGLE3, None
                ),
                TpuEagle3Worker,
            )
            self.assertIsNone(
                select_experimental_spec_worker(None, SpeculativeAlgorithm.EAGLE, None)
            )
        with envs.SGLANG_TPU_ENABLE_EXPERIMENTAL_SPEC_WORKER.override(False):
            self.assertIsNone(
                select_experimental_spec_worker(None, SpeculativeAlgorithm.EAGLE3, None)
            )

    def test_seed_survives_relay_and_host_lengths_stay_committed(self):
        """The relayed seed reaches the next decode, and resolve keeps the committed
        host lengths the verify window was planned from (no seq_lens D2H)."""
        worker = _eagle_worker()
        pool = SimpleNamespace(req_to_token=torch.zeros((8, 4), dtype=torch.int32))
        spec = SimpleNamespace(
            speculative_algorithm="EAGLE3", enable_multi_layer_eagle=False
        )
        with (
            patch("sglang.srt.speculative.spec_utils.get_spec", return_value=spec),
            # The TPU relay gathers with torch ops; CPU would launch Triton.
            patch("sglang.srt.speculative.spec_utils._use_torch_native", True),
        ):
            future_map = FutureMap("cpu", SpeculativeAlgorithm.EAGLE3, pool)
            rows = torch.tensor([2, 5])
            future_map.publish(rows, torch.tensor([11, 21]))
            state = SpecState(
                bonus_tokens=torch.tensor([7, 9]),
                seq_lens=torch.tensor([11, 21]),
                req_pool_indices=rows,
                draft_carry=_seed(2, offset=3),
            )
            future_map.stash(
                rows, RelayPayload.from_draft_input(worker.state_to_spec_info(state))
            )

        next_input = worker.state_to_spec_info(
            SpecState(rows, rows, rows, worker.make_dummy_carry(2))
        )
        next_input.future_indices = rows
        next_input.tpu_host_seq_lens_is_committed = True
        committed = torch.tensor([10, 20])
        batch = SimpleNamespace(
            spec_info=next_input, seq_lens_cpu=committed, host_pad_bs=lambda bs: bs
        )
        with (
            patch.object(overlap_utils, "_is_tpu", True),
            # sglang_tpu.patches installs this on TPU.
            patch.object(FutureMap, "tpu_direct_handoff", None, create=True),
            patch.object(
                overlap_utils,
                "resolve_device_accept_seq_lens_tpu",
                resolve_device_accept_seq_lens_tpu,
                create=True,
            ),
            patch.object(
                overlap_utils,
                "device_accept_resolved_tpu",
                device_accept_resolved_tpu,
                create=True,
            ),
            patch("sglang.srt.speculative.spec_utils._use_torch_native", True),
        ):
            future_map.resolve_seq_lens_cpu(batch)
            future_map._resolve_spec_extras(batch)
        self.assertIs(batch.seq_lens_cpu, committed)
        self.assertEqual(batch.seq_lens.tolist(), [11, 21])
        resolved = worker.get_spec_state(batch_with(next_input, batch.seq_lens, rows))
        self.assertEqual(resolved.bonus_tokens.tolist(), [7, 9])
        for got, want in zip(resolved.draft_carry, state.draft_carry):
            torch.testing.assert_close(got, want.to(got.dtype))

    def test_seed_survives_bucketing_and_scheduler_export(self):
        worker = _eagle_worker()
        live = SpecState(
            bonus_tokens=torch.tensor([7, 9]),
            seq_lens=torch.tensor([11, 21]),
            req_pool_indices=torch.tensor([2, 5]),
            draft_carry=_seed(2, offset=3),
        )
        sampling_info = HostPaddedSamplingInfo(
            temperatures=torch.ones((4, 1)),
            top_ps=torch.ones(4),
            top_ks=torch.ones(4, dtype=torch.int32),
            min_ps=torch.zeros(4),
            is_all_greedy=True,
            is_any_greedy=True,
            need_top_p_sampling=False,
            need_top_k_sampling=False,
            need_min_p_sampling=False,
            vocab_size=16,
            device="cpu",
            real_batch_size=2,
        )
        rows = torch.tensor([2, 5, 0, 0])
        worker._bucket = None
        worker.get_spec_state = lambda batch: live
        bucket = TpuSpecWorker._bucket_state(
            worker,
            SimpleNamespace(req_pool_indices=rows[:2], sampling_info=sampling_info),
            4,
            SimpleNamespace(req_pool_indices=rows, future_indices=None),
        )
        padded = bucket.state.draft_carry
        self.assertEqual(padded.hidden_states.shape, (4, HIDDEN))
        torch.testing.assert_close(padded.topk_index[:2], live.draft_carry.topk_index)
        result = TpuSpecWorker._pack_decode_result(
            worker,
            bs=2,
            next_state=bucket.state,
            logits_output=SimpleNamespace(next_token_logits=torch.zeros(12, 16)),
            accept=AcceptResult(
                num_correct_drafts=torch.zeros(4, dtype=torch.int64),
                bonus_tokens=torch.zeros(4, dtype=torch.int64),
                accept_tokens=torch.zeros((4, 3), dtype=torch.int64),
                accept_lens=torch.ones(4, dtype=torch.int32),
                next_seq_lens=torch.zeros(4, dtype=torch.int64),
            ),
        )
        exported = result.next_draft_input
        self.assertIsInstance(exported, HostPaddedEagleDraftInput)
        torch.testing.assert_close(exported.topk_index, live.draft_carry.topk_index)
        torch.testing.assert_close(
            exported.hidden_states, live.draft_carry.hidden_states
        )
        # FutureMap's TPU EAGLE resolvers read these on every EAGLE3 input.
        self.assertIsNone(exported.device_accept_selected_cache_locs)
        self.assertIsNone(exported.device_accept_candidate_locs)

    def test_dp_ranks_agree_on_a_bucket_the_draft_padding_keeps(self):
        """Every DP rank derives the same decode bucket from the synced sizes, equal
        to the draft's DP peer rows (aligned to attention TP), even when idle."""
        worker = _eagle_worker()
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
        worker = _eagle_worker()
        live = SpecState(
            bonus_tokens=torch.tensor([7]),
            seq_lens=torch.tensor([11]),
            req_pool_indices=torch.tensor([2]),
            draft_carry=_seed(1),
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
        self.assertEqual(grown.state.draft_carry.hidden_states.shape, (4, HIDDEN))
        worker._bucket = grown
        self.assertIs(bucket(4), grown)

    def test_target_only_acceptance_commits_target_distribution(self):
        """The first committed token follows p_0 whatever d_1 is, with the
        rejected draft excluded from the bonus; top_k <= 1 rows stay greedy."""
        torch.manual_seed(0)
        p0 = torch.tensor([0.1, 0.2, 0.3, 0.4])
        rows = 40000
        probs = torch.stack([p0, torch.full((4,), 0.25)]).repeat(rows, 1)
        drafts = torch.full((rows, 1), 2)
        num_correct, bonus = accept_drafts_target_only(
            probs,
            drafts,
            torch.ones(rows, dtype=torch.int32),
            torch.full((rows,), 4, dtype=torch.int32),
            1.0,
            1.0,
        )
        first = torch.where(num_correct > 0, drafts[:, 0], bonus)
        freq = torch.bincount(first, minlength=4).float() / rows
        torch.testing.assert_close(freq, p0, atol=0.01, rtol=0)
        rejected = num_correct == 0
        self.assertFalse(bool((bonus[rejected] == 2).any()))

        greedy_correct, greedy_bonus = accept_drafts_target_only(
            probs[:4],
            torch.tensor([[3], [1]]),
            torch.ones(2, dtype=torch.int32),
            torch.ones(2, dtype=torch.int32),
            1.0,
            1.0,
        )
        self.assertEqual(greedy_correct.tolist(), [1, 0])
        self.assertEqual(greedy_bonus.tolist(), [0, 3])

    def test_sampled_draft_acceptance_commits_filtered_target(self):
        """With a sampled draft, filtering only p stays lossless: the first committed
        token follows the filtered p_0, including tokens the filter dropped from q."""
        torch.manual_seed(0)
        rows = 40000
        p0 = torch.tensor([0.0, 0.5, 0.5, 0.0])  # e.g. top-k 2 of the target
        q_logits = torch.log(torch.tensor([0.4, 0.3, 0.2, 0.1]))
        drafts = torch.multinomial(q_logits.exp(), rows, replacement=True)[:, None]
        target_probs = torch.stack([p0, torch.full((4,), 0.25)]).repeat(rows, 1)
        num_correct, bonus = accept_drafts_sampled_filtered(
            target_probs,
            target_probs.log(),
            drafts,
            torch.ones(rows, dtype=torch.int32),
            q_logits.expand(rows, 1, 4),
            torch.ones(rows),
            torch.full((rows,), 2, dtype=torch.int32),
        )
        first = torch.where(num_correct > 0, drafts[:, 0], bonus)
        freq = torch.bincount(first, minlength=4).float() / rows
        torch.testing.assert_close(freq, p0, atol=0.01, rtol=0)

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
        )
        probs = torch.full((2, 2), 0.5)
        for num_correct, bonus in (
            accept_drafts_sampled(target_logits, **args),
            accept_drafts_sampled_filtered(probs, target_logits, **args),
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

        none = {"need_top_k": False, "need_top_p": False, "need_min_p": False}
        self.assertEqual(filters([1, 1 << 30]), none)
        self.assertEqual(filters([1, 1 << 30], top_ps=(0.5, 1.0)), none)
        self.assertEqual(filters([1, 8])["need_top_k"], True)
        self.assertEqual(filters([1, 1 << 30], min_ps=(0.0, 0.1))["need_min_p"], True)

    def test_sampled_verify_filters_target_like_the_target_sampler(self):
        """Sampled verify keeps what the target sampler keeps: every cutoff on the
        unfiltered p, each request's values on all its N + 1 rows, one renorm."""
        worker = _eagle_worker()
        worker.num_verify_rows = 2

        def joint_filter(probs, top_ks, top_ps, min_ps, **need):
            # CPU reference of tpu_filter_probs with every filter enabled.
            del need
            desc = probs.sort(dim=-1, descending=True).values
            rows = torch.arange(probs.shape[0])
            k = top_ks.clamp(1, probs.shape[1]).long() - 1
            nucleus = (
                (desc.cumsum(-1) < top_ps[:, None])
                .sum(-1)
                .clamp_max(probs.shape[1] - 1)
            )
            cutoff = torch.maximum(desc[rows, k], desc[rows, nucleus])
            cutoff = torch.maximum(cutoff, desc[:, 0] * min_ps)
            return torch.where(probs >= cutoff[:, None], probs, 0.0)

        p = torch.tensor([[0.40, 0.25, 0.20, 0.15], [0.50, 0.30, 0.15, 0.05]])
        bucket = BucketState(
            key=None,
            state=None,
            live_mask=None,
            temperatures=torch.ones(2),
            # Renormalizing after top-k would push 0.40 past top_p and drop 0.25.
            top_ks=torch.tensor([2, 1 << 30], dtype=torch.int32),
            top_ps=torch.tensor([0.6, 1.0]),
            min_ps=torch.tensor([0.0, 0.5]),
            need_top_k=True,
            need_top_p=True,
            need_min_p=True,
        )
        with patch.object(base_spec_worker, "tpu_filter_probs", joint_filter):
            filtered = worker._filtered_target_probs(
                p.log().repeat_interleave(2, dim=0), bucket
            )
        expected = torch.tensor(
            [[0.40 / 0.65, 0.25 / 0.65, 0, 0], [0.50 / 0.80, 0.30 / 0.80, 0, 0]]
        ).repeat_interleave(2, dim=0)
        torch.testing.assert_close(filtered, expected)

    def test_draft_prefill_shift_keeps_chunks_consistent(self):
        """Each request's last row takes its own sampled token, except a non-final
        chunk, which takes its next prompt token; placeholder ids stay in vocab."""
        chunked = SimpleNamespace()
        batch = SimpleNamespace(
            # 99999 stands in for an image placeholder hash id.
            input_ids=torch.tensor([10, 99999, 12, 20, 21]),
            extend_lens=[3, 2],
            chunked_req_next_prompt_token=22,
            chunked_req=chunked,
            reqs=[SimpleNamespace(), chunked],
        )
        input_ids, tail_src = _shifted_prompt(
            batch=batch, padded_tokens=8, vocab_size=64, pin_memory=False
        )
        filled = fill_prefill_tail(input_ids, tail_src, torch.tensor([7, 8, 0, 0]))
        self.assertEqual(filled.tolist(), [63, 12, 7, 21, 22, 0, 0, 0])

    def test_draft_extend_reads_valid_tokens_at_post_write_lengths(self):
        accept_tokens = torch.tensor([[5, 6, -1], [-1, -1, -1]])
        input_ids, post = draft_extend_inputs(accept_tokens, torch.tensor([10, 0]))
        self.assertEqual(input_ids.tolist(), [5, 6, 0, 0, 0, 0])
        self.assertEqual(post.tolist(), [13, 3])

    def test_eagle3_verify_warmup_builds_the_served_structure(self):
        """The dynamo tracer guards the spec_info structure, so the EAGLE3 compile
        policy's TARGET_VERIFY warmup must build what serving builds."""
        worker = _eagle_worker()
        worker.num_drafts = 2
        served = worker.make_verify_input(
            batch=SimpleNamespace(),
            verify_tokens=torch.arange(6),
            rows=SimpleNamespace(flat_positions=torch.arange(6)),
        )
        policy = object.__new__(TpuEagle3SpecDecodeCompilePolicyMixin)
        policy.draft_token_num = 3
        policy.model_runner = SimpleNamespace(device="cpu")
        warmed = policy.make_verify_spec_info()

        def structure(spec_info):
            return {
                name: (
                    value
                    if isinstance(value, (int, bool)) and value is not None
                    else type(value).__name__
                )
                for name, value in vars(spec_info).items()
            }

        self.assertEqual(type(warmed), type(served))
        self.assertEqual(structure(warmed), structure(served))

    def test_host_padded_prefill_keeps_bucket_rows(self):
        policy = object.__new__(TpuEagle3SpecDecodeCompilePolicyMixin)
        sampling_info = SimpleNamespace(temperatures=torch.ones((4, 1)))
        for capture, hidden_rows in (
            (CaptureHiddenMode.LAST, 4),
            (CaptureHiddenMode.FULL, 16),
        ):
            forward_batch = SimpleNamespace(
                sampling_info=sampling_info,
                forward_mode=ForwardMode.EXTEND,
                capture_hidden_mode=capture,
                spec_info=None,
                _tpu_real_q=5,
                _tpu_real_b=2,
            )
            outputs = LogitsProcessorOutput(
                next_token_logits=torch.zeros((8, 3)),
                hidden_states=torch.zeros((16 if capture.is_full() else 8, 2)),
            )
            with patch(
                "sglang_tpu.experimental.compilation.eagle3_compile_policy.is_host_padded",
                return_value=True,
            ):
                outputs = policy.finalize_forward_batch(outputs, forward_batch)
            self.assertEqual(outputs.next_token_logits.shape[0], 4)
            self.assertEqual(outputs.hidden_states.shape[0], hidden_rows)
            self.assertEqual(forward_batch.batch_size, 2)

        # A multimodal wrapper's inner model returns (hidden, aux) for its eager
        # logits processor: rows stay at the token bucket, the batch is restored.
        forward_batch = SimpleNamespace(
            sampling_info=sampling_info,
            forward_mode=ForwardMode.EXTEND,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            spec_info=None,
            input_ids=torch.zeros(16, dtype=torch.int64),
            _tpu_real_q=5,
            _tpu_real_b=2,
        )
        inner = (torch.zeros((16, 2)), [torch.zeros((16, 2))])
        with patch(
            "sglang_tpu.experimental.compilation.eagle3_compile_policy.is_host_padded",
            return_value=True,
        ):
            hidden, aux = policy.finalize_forward_batch(inner, forward_batch)
        self.assertEqual((hidden.shape[0], aux[0].shape[0]), (16, 16))
        self.assertEqual(forward_batch.input_ids.shape[0], 5)


def batch_with(spec_info, seq_lens, req_pool_indices):
    return SimpleNamespace(
        spec_info=spec_info, seq_lens=seq_lens, req_pool_indices=req_pool_indices
    )


if __name__ == "__main__":
    unittest.main()
