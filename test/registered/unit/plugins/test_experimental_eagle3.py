"""CPU contracts for EAGLE3 on the experimental TPU spec worker."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_tp_sync import SpecTpSyncSite
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.envs import envs
from sglang_tpu.experimental.compilation.chain_compile_policy import (
    _hidden_only_logits,
)
from sglang_tpu.experimental.sampling.sampling_batch_info import HostPaddedSamplingInfo
from sglang_tpu.experimental.speculative import spec_ops
from sglang_tpu.experimental.speculative.base_spec_worker import TpuSpecWorker
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
    accept_drafts_sampled,
    accept_drafts_sampled_filtered,
    accept_drafts_target_only,
    finalize_accept,
)
from sglang_tpu.experimental.speculative.spec_types import SpecState
from sglang_tpu.experimental.speculative.spec_utils import pad_rows
from sglang_tpu.hooks.experimental_spec import (
    keep_host_padded_extend_outputs,
)
from sglang_tpu.hooks.spec_worker import select_experimental_spec_worker

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


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
        with envs.SGLANG_TPU_ENABLE_EXPERIMENTAL_SPEC_WORKER.override(False):
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
                **_unfiltered(rows),
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

        none = {"need_top_k": False, "need_top_p": False, "need_min_p": False}
        self.assertEqual(filters([1, 1 << 30]), none)
        self.assertEqual(filters([1, 1 << 30], top_ps=(0.5, 1.0)), none)
        self.assertEqual(filters([1, 8])["need_top_k"], True)
        self.assertEqual(filters([1, 1 << 30], min_ps=(0.0, 0.1))["need_min_p"], True)

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
            patch.object(TpuChainSpecWorker, "_compile_policy", return_value=policy),
            patch.object(TpuChainSpecWorker, "_b_bucketer", return_value=buckets),
        ):
            worker._warmup_extend_sampling()
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


if __name__ == "__main__":
    unittest.main()
