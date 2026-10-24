"""CPU contracts for plugin-owned host padding and scheduler composition."""

from __future__ import annotations

import pkgutil
import subprocess
import sys
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.plugins.hook_registry import HookRegistry, _propagate_patch
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, enter_scope
from sglang_tpu.envs import envs
from sglang_tpu.experimental import register_host_batch_padder, uses_host_batch
from sglang_tpu.experimental.sampling.sampling_batch_info import HostPaddedSamplingInfo
from sglang_tpu.experimental.speculative.block_info import (
    HostPaddedBlockDraftInput,
)
from sglang_tpu.experimental.speculative.chain_info import HostPaddedChainDraftInput
from sglang_tpu.hooks.experimental_spec import (
    alloc_extend,
    hook_specs,
    initialize_host_batch,
    kimi_spec_defaults,
)

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _pad(bs):
    return 4 if bs <= 4 else 8


def _module_init(self):
    # Sampler.__init__ needs the TP group.
    torch.nn.Module.__init__(self)


def _tpu_sampler():
    from sglang.srt.layers.sampler import Sampler
    from sglang_tpu.sampler import TpuSampler

    with patch.object(Sampler, "__init__", _module_init):
        return TpuSampler()


def _req(temperature=0.5):
    return SimpleNamespace(
        sampling_params=SamplingParams(temperature=temperature),
        eos_token_ids=None,
        tokenizer=SimpleNamespace(eos_token_id=None, additional_stop_token_ids=None),
        custom_logit_processor=None,
        return_sampling_mask=False,
        return_logprob=False,
        return_hidden_states=False,
        return_hidden_states_mode=CaptureHiddenMode.NULL,
        grammar=None,
        is_prefill_only=False,
    )


def _state(rows):
    bs = len(rows)
    return HostPaddedBlockDraftInput(
        bonus_tokens=torch.ones(bs, dtype=torch.int64),
        new_seq_lens=torch.arange(bs),
        future_indices=torch.tensor(rows),
        nxt_kv_lens_cpu=torch.arange(bs) + 10,
        tpu_host_seq_lens_is_committed=True,
        tpu_verify_cache_candidates=torch.zeros(_pad(bs), 8, dtype=torch.int64),
        tpu_committed_lens=torch.zeros(_pad(bs), dtype=torch.int64),
        tpu_req_pool_indices=torch.zeros(_pad(bs), dtype=torch.int64),
    )


class TestExperimentalSpecHooks(CustomTestCase):
    def setUp(self):
        super().setUp()
        # enterContext is 3.11+; CI runs 3.10.
        stack = enter_scope(self, ExitStack())
        context = SimpleNamespace(
            deterministic=SimpleNamespace(enable_deterministic_inference=False),
            features=SimpleNamespace(enable_custom_logit_processor=False),
        )
        stack.enter_context(
            patch(
                "sglang.srt.sampling.sampling_batch_info.get_exec", return_value=context
            )
        )
        stack.enter_context(
            envs.SGLANG_TPU_ENABLE_EXPERIMENTAL_SPEC_WORKER.override(True)
        )
        stack.enter_context(patch("sglang_tpu.experimental._host_batch_padder", None))
        register_host_batch_padder(_pad)
        # Exercise the real registry's descriptor handling and import propagation,
        # then restore every patched binding so this test cannot affect other tests.
        for target, hook, kind in hook_specs():
            owner_path, name = target.rsplit(".", 1)
            owner = pkgutil.resolve_name(owner_path)
            raw = vars(owner).get(name)
            original = getattr(owner, name)
            HookRegistry._apply_target(target, [(kind, hook, None)])
            wrapped = getattr(owner, name)

            def restore(
                owner=owner, name=name, raw=raw, original=original, wrapped=wrapped
            ):
                if raw is None:
                    delattr(owner, name)
                else:
                    setattr(owner, name, raw)
                _propagate_patch(wrapped, original, owner)

            stack.callback(restore)

    def batch(self, rows, spec_algorithm=SpeculativeAlgorithm.DSPARK, overlap=True):
        batch = ScheduleBatch.init_new(
            [_req(0.1 * (i + 1)) for i in range(len(rows))],
            SimpleNamespace(device="cpu"),
            None,
            None,
            SimpleNamespace(is_encoder_decoder=False, vocab_size=32),
            enable_overlap=overlap,
            spec_algorithm=spec_algorithm,
        )
        batch.req_pool_indices_cpu = torch.tensor(rows)
        batch.req_pool_indices = torch.tensor(rows)
        batch.seq_lens = torch.arange(len(rows)) + 10
        batch.orig_seq_lens = batch.seq_lens.clone()
        batch.seq_lens_cpu = batch.seq_lens.clone()
        batch.sampling_info = SamplingBatchInfo.from_schedule_batch(batch, 32)
        if uses_host_batch(batch):
            batch.spec_info = _state(rows)
        return batch

    def test_filter_merge_and_snapshot_keep_live_rows_and_relay_identity(self):
        batch = self.batch([2, 5, 8])
        self.assertIsInstance(batch.sampling_info, HostPaddedSamplingInfo)
        self.assertEqual(batch.sampling_info.temperatures.shape, (4, 1))
        snapshot = batch.copy()
        original_rows = snapshot.req_pool_indices
        # No device gather should be attempted: the authoritative rows are on host.
        batch.seq_lens = batch.orig_seq_lens = None
        batch.filter_batch(keep_indices=[0, 2])
        self.assertEqual(batch.req_pool_indices.tolist(), [2, 8])
        self.assertIs(batch.spec_info.future_indices, batch.req_pool_indices)
        self.assertIsNone(batch.spec_info.tpu_verify_cache_candidates)
        self.assertIsNone(batch.seq_lens)
        torch.testing.assert_close(
            batch.sampling_info.temperatures[:, 0], torch.tensor([0.1, 0.3, 1.0, 1.0])
        )
        self.assertIs(snapshot.host_pad_bs, batch.host_pad_bs)
        self.assertEqual(original_rows.tolist(), [2, 5, 8])
        other = self.batch([3, 4, 6])
        batch.merge_batch(other)
        self.assertEqual(batch.req_pool_indices.tolist(), [2, 8, 3, 4, 6])
        self.assertIs(batch.spec_info.future_indices, batch.req_pool_indices)
        self.assertEqual(batch.spec_info.nxt_kv_lens_cpu.tolist(), [10, 12, 10, 11, 12])
        self.assertEqual(len(batch.sampling_info), 5)
        torch.testing.assert_close(
            batch.sampling_info.temperatures[:, 0],
            torch.tensor([0.1, 0.3, 0.1, 0.2, 0.3, 1.0, 1.0, 1.0]),
        )
        forwarded = batch.sampling_info.copy_for_forward()
        self.assertIsInstance(forwarded, HostPaddedSamplingInfo)
        self.assertEqual(len(forwarded), 5)
        self.assertIsNone(forwarded.penalizer_orchestrator)
        self.assertEqual(forwarded.on_device().temperatures.shape, (8, 1))

    def test_pd_prefill_without_overlap_filters_core_rows(self):
        """A PP PD prefill filters its chunked extend batch without overlap; FutureMap
        owns no rows there, so the filter must keep core's rows and the seed's."""
        with patch(
            "sglang_tpu.hooks.experimental_spec.get_disagg",
            return_value=SimpleNamespace(disaggregation_mode="prefill"),
        ):
            batch = self.batch([2, 5, 8], overlap=False)
        batch.spec_info = HostPaddedBlockDraftInput(
            bonus_tokens=torch.tensor([7, 8, 9]), new_seq_lens=batch.seq_lens
        )
        batch.filter_batch(keep_indices=[0, 2])
        self.assertEqual(batch.req_pool_indices.tolist(), [2, 8])
        self.assertEqual(batch.seq_lens.tolist(), [10, 12])
        self.assertEqual(batch.spec_info.bonus_tokens.tolist(), [7, 9])
        self.assertEqual(batch.spec_info.new_seq_lens.tolist(), [10, 12])

    def test_host_padding_follows_the_worker_not_the_algorithm(self):
        """Without a TpuSpecWorker (no registered padder) batches keep core's rows;
        with one, every algorithm's batches are host padded."""
        with patch("sglang_tpu.experimental._host_batch_padder", None):
            batch = self.batch([2, 5, 8])
        self.assertIs(type(batch.sampling_info), SamplingBatchInfo)
        self.assertFalse(hasattr(batch, "host_pad_bs"))
        batch.input_ids = torch.tensor([20, 50, 80])
        batch.filter_batch(keep_indices=[0, 2])
        self.assertEqual(batch.seq_lens.tolist(), [10, 12])
        self.assertEqual(batch.input_ids.tolist(), [20, 80])
        self.assertEqual(batch.sampling_info.temperatures.shape, (2, 1))
        for algorithm in (SpeculativeAlgorithm.EAGLE, SpeculativeAlgorithm.EAGLE3):
            served = self.batch([1], spec_algorithm=algorithm)
            self.assertIs(served.host_pad_bs, _pad)
            self.assertIsInstance(served.sampling_info, HostPaddedSamplingInfo)

    def test_prefill_staging_and_allocation_remain_on_host(self):
        from sglang.srt.managers.overlap_utils import resolve_forward_inputs
        from sglang.srt.model_executor.forward_batch_info import _extend_metadata_device

        batch = self.batch([2])
        prompt = batch.prefill_input_ids_cpu = torch.tensor([4, 5, 6])
        batch.device = (
            "meta"  # An accidental upload is observable without TPU hardware.
        )
        resolve_forward_inputs(
            batch, SimpleNamespace(_resolve_spec_extras=lambda batch: None)
        )
        self.assertIs(batch.input_ids, prompt)
        self.assertIsNone(batch.prefill_input_ids_cpu)
        self.assertEqual(_extend_metadata_device(batch, "meta"), "cpu")
        with patch("sglang_tpu.experimental._host_batch_padder", None):
            plain = self.batch([2])
        self.assertEqual(_extend_metadata_device(plain, "meta"), "meta")
        captured = {}

        def allocate(**kwargs):
            captured.update(kwargs)
            return kwargs["out_cache_loc_host"]

        slots = alloc_extend(
            allocate, None, None, None, None, None, None, 3, batch=batch
        )
        self.assertEqual(slots.device.type, "cpu")
        self.assertEqual(slots.shape, (3,))
        self.assertIsNone(
            alloc_extend(allocate, None, None, None, None, None, None, 3, batch=plain)
        )
        batch.prefill_input_ids_cpu = torch.tensor([1])
        batch.mix_running_indices = torch.tensor([0])
        with self.assertRaisesRegex(AssertionError, "mixed chunks"):
            resolve_forward_inputs(
                batch, SimpleNamespace(_resolve_spec_extras=lambda batch: None)
            )

    def test_worker_publishes_plugin_owned_draft_state(self):
        from sglang_tpu.experimental.speculative.block_spec_worker import (
            TpuBlockSpecWorker,
        )
        from sglang_tpu.experimental.speculative.spec_types import SpecState

        state = SpecState(
            torch.tensor([4, 5]), torch.tensor([10, 20]), torch.tensor([1, 2])
        )
        draft = TpuBlockSpecWorker.state_to_spec_info(None, state)
        self.assertIsInstance(draft, HostPaddedBlockDraftInput)
        torch.testing.assert_close(draft.bonus_tokens, state.bonus_tokens)
        torch.testing.assert_close(draft.new_seq_lens, state.seq_lens)

    def test_model_hook_uses_anchor_width_without_changing_parent(self):
        from sglang.srt.models.dflash import DFlashDraftModel
        from sglang.srt.models.dspark import DSparkDraftModel
        from sglang_tpu.hooks.experimental_spec import sample_draft_tokens

        self.assertIsNot(
            DSparkDraftModel.sample_draft_tokens_tpu,
            DFlashDraftModel.sample_draft_tokens_tpu,
        )
        for anchor in (True, False):
            width = 2 if anchor else 3
            hidden = torch.arange(2 * width * 4, dtype=torch.float32).view(-1, 4)
            tokens = torch.arange(2 * width).view(-1)

            def sample_block(
                logits, *, first_prev_tokens, hidden_states, sampler, collect_corrected
            ):
                torch.testing.assert_close(first_prev_tokens, torch.tensor([0, width]))
                expected = hidden.view(2, width, 4)
                if not anchor:
                    expected = expected[:, 1:]
                torch.testing.assert_close(hidden_states, expected)
                return torch.stack(
                    [sampler(logits[:, i], i) for i in range(2)], dim=1
                ), None

            model = SimpleNamespace(
                gamma=2,
                sample_from_anchor=anchor,
                compute_base_logits=lambda values: (values, None),
                markov_head=SimpleNamespace(sample_block=sample_block),
            )
            result = DSparkDraftModel.sample_draft_tokens_tpu(model, hidden, tokens)
            torch.testing.assert_close(result, torch.full((2, 2), 3, dtype=torch.int64))
        with envs.SGLANG_TPU_ENABLE_EXPERIMENTAL_SPEC_WORKER.override(False):
            sentinel = object()
            self.assertIs(
                sample_draft_tokens(lambda *args: sentinel, None, None, None), sentinel
            )

    def test_either_flag_enables_the_experimental_worker(self):
        from sglang_tpu.experimental import (
            is_experimental_mtp,
            is_experimental_spec_worker,
        )

        for experimental, no_d2h in ((True, False), (False, True), (False, False)):
            with (
                envs.SGLANG_TPU_ENABLE_EXPERIMENTAL_SPEC_WORKER.override(experimental),
                envs.SGLANG_TPU_HOST_PADDING_NO_D2H.override(no_d2h),
            ):
                for algo in (SpeculativeAlgorithm.DFLASH, SpeculativeAlgorithm.DSPARK):
                    self.assertEqual(
                        is_experimental_spec_worker(algo), experimental or no_d2h
                    )
                self.assertEqual(
                    is_experimental_mtp(SpeculativeAlgorithm.EAGLE),
                    experimental or no_d2h,
                )

    def test_overlap_utils_tpu_imports_without_the_hook_package(self):
        """Core overlap_utils imports overlap_utils_tpu while the scheduler module
        is still initializing; pulling in sglang_tpu.hooks there (whose PP hooks
        import Scheduler) killed every TPU scheduler with a circular import."""
        code = (
            "import sys, sglang_tpu.overlap_utils_tpu; "
            "sys.exit(int(any(m.startswith('sglang_tpu.hooks') for m in sys.modules)))"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr.decode()[-2000:])

    def test_hot_path_hooks_register_only_with_the_flag_on(self):
        """With both flags off the default spec path must run no extra wrappers."""
        hot = {
            "sglang.srt.speculative.spec_utils.spec_prepare_for_decode",
            "sglang.srt.speculative.eagle_disaggregation.build_eagle_disagg_draft_input",
            "sglang.srt.speculative.dflash_disaggregation.build_dflash_disagg_draft_input",
            "sglang.srt.speculative.dspark_disaggregation.build_dspark_disagg_draft_input",
            "sglang.srt.model_executor.forward_batch_info.ForwardBatch."
            "post_forward_mlp_sync_batch",
        }
        with (
            envs.SGLANG_TPU_ENABLE_EXPERIMENTAL_SPEC_WORKER.override(False),
            envs.SGLANG_TPU_HOST_PADDING_NO_D2H.override(False),
        ):
            off = {target for target, _, _ in hook_specs()}
        on = {target for target, _, _ in hook_specs()}
        self.assertFalse(hot & off)
        self.assertEqual(hot & on, hot)

    def test_chain_draft_prompt_reads_the_next_token_of_each_row(self):
        from sglang_tpu.experimental.speculative.chain_spec_worker import (
            fill_sampled_tails,
            shift_prompt_tokens,
        )

        # Request 0 finishes its prompt; request 1 is a chunk followed by token 22.
        shifted, row_reqs, is_sampled = shift_prompt_tokens(
            torch.tensor([10, 11, 12, 20, 21]),
            extend_lens=[3, 2],
            padded_tokens=8,
            chunked_row=1,
            next_prompt_token=22,
            vocab_size=100,
        )
        torch.testing.assert_close(row_reqs, torch.tensor([0, 0, 0, 1, 1, 0, 0, 0]))
        self.assertEqual(is_sampled.nonzero().view(-1).tolist(), [2])
        draft_ids = fill_sampled_tails(
            shifted, row_reqs, is_sampled, torch.tensor([7, 8, 0, 0])
        )
        torch.testing.assert_close(draft_ids[:5], torch.tensor([11, 12, 7, 21, 22]))

    def test_chain_draft_rows_hand_rpa_the_pre_row_length(self):
        """TARGET_VERIFY RPA adds the row width itself; a post-row length let draft
        steps attend stale draft KV and cut chain acceptance by ~13%."""
        from sglang_tpu.experimental.speculative.spec_ops import (
            build_chain_block_inputs,
            build_draft_block_inputs,
        )
        from sglang_tpu.experimental.speculative.spec_types import VerifyRows

        seq_lens = torch.tensor([10, 20, 0])
        live = torch.tensor([True, True, False])
        positions = seq_lens[:, None] + torch.arange(4)
        rows = VerifyRows(
            positions=positions,
            cache_locs=positions,
            flat_positions=positions.reshape(-1),
            flat_cache_locs=positions.reshape(-1),
        )
        step = build_chain_block_inputs(
            torch.tensor([1, 2, 3]), rows, seq_lens, live, 2
        )
        self.assertEqual(step.seq_lens.tolist(), [12, 22, 0])
        self.assertEqual(step.positions.tolist(), [12, 22, 2])
        extend = build_chain_block_inputs(
            torch.zeros((3, 4), dtype=torch.int64), rows, seq_lens, live, 0
        )
        self.assertEqual(extend.seq_lens.tolist(), [10, 20, 0])
        block = build_draft_block_inputs(
            torch.tensor([1, 2, 3]), rows, seq_lens, live, 0, 4
        )
        self.assertEqual(block.seq_lens.tolist(), [14, 24, 0])

    def test_mtp_takes_the_chain_paths_only_when_enabled(self):
        from sglang_tpu.experimental import is_experimental_spec_worker
        from sglang_tpu.experimental.speculative.block_spec_worker import (
            TpuBlockSpecWorker,
        )
        from sglang_tpu.experimental.speculative.chain_spec_worker import (
            TpuChainSpecWorker,
        )
        from sglang_tpu.hooks.experimental_spec import (
            need_topk,
            prepare_spec_decode,
            spec_need_hidden_states,
            supports_kv_free_fence,
            supports_tpu_direct_compile,
        )
        from sglang_tpu.hooks.spec_worker import select_experimental_spec_worker

        mtp = SpeculativeAlgorithm.from_string("NEXTN")
        spec_args = SimpleNamespace(speculative_algorithm="NEXTN")
        prepared = []
        draft_input = MagicMock(spec=HostPaddedChainDraftInput)
        draft_input.prepare_for_decode.side_effect = prepared.append
        batch = SimpleNamespace(
            spec_algorithm=mtp, host_pad_bs=lambda bs: bs, spec_info=draft_input
        )
        core_batch = SimpleNamespace(spec_algorithm=mtp, spec_info=draft_input)
        core_prepare = lambda batch: prepared.append("core")  # noqa: E731
        core_fence = lambda algo, *, copies_draft_tail: True  # noqa: E731
        last_stage = SimpleNamespace(pp_group=SimpleNamespace(is_last_rank=True))
        with (
            envs.SGLANG_TPU_ENABLE_EXPERIMENTAL_SPEC_WORKER.override(True),
            patch("sglang_tpu.hooks.spec_worker.get_parallel", return_value=last_stage),
        ):
            self.assertFalse(is_experimental_spec_worker(SpeculativeAlgorithm.EAGLE3))
            self.assertIs(
                select_experimental_spec_worker(None, mtp, None), TpuChainSpecWorker
            )
            self.assertIs(
                select_experimental_spec_worker(
                    None, SpeculativeAlgorithm.DFLASH, None
                ),
                TpuBlockSpecWorker,
            )
            prepare_spec_decode(core_prepare, batch)
            # commit writes KV after the accept outputs; only the copied
            # kv_fence_tail lets the fence cover it, for every family.
            for algo in (mtp, SpeculativeAlgorithm.DFLASH, SpeculativeAlgorithm.EAGLE3):
                for copies_draft_tail in (True, False):
                    self.assertEqual(
                        supports_kv_free_fence(
                            core_fence, algo, copies_draft_tail=copies_draft_tail
                        ),
                        copies_draft_tail,
                    )
            self.assertFalse(need_topk(lambda algo: True, mtp))
            self.assertTrue(supports_tpu_direct_compile(lambda algo: False, mtp))
            with patch(
                "sglang_tpu.hooks.experimental_spec.get_spec", return_value=spec_args
            ):
                # The carry stays in the draft's table, so FutureMap relays no hidden.
                self.assertFalse(spec_need_hidden_states(lambda: True))
        self.assertEqual(prepared, [batch])
        with envs.SGLANG_TPU_ENABLE_EXPERIMENTAL_SPEC_WORKER.override(False):
            self.assertIsNone(select_experimental_spec_worker(None, mtp, None))
            # Off the flag, init_new never sets host_pad_bs, so core prepares.
            prepare_spec_decode(core_prepare, core_batch)
            self.assertTrue(
                supports_kv_free_fence(core_fence, mtp, copies_draft_tail=True)
            )
            self.assertTrue(need_topk(lambda algo: True, mtp))
            self.assertFalse(supports_tpu_direct_compile(lambda algo: False, mtp))
            with patch(
                "sglang_tpu.hooks.experimental_spec.get_spec", return_value=spec_args
            ):
                self.assertTrue(spec_need_hidden_states(lambda: True))
        self.assertEqual(prepared, [batch, "core"])

    def test_experimental_block_drafts_opt_into_the_pd_decode_swa_hicache(self):
        """DSpark joins DFlash under the PD decode SWA HiCache on the experimental worker."""
        from sglang_tpu.hooks.experimental_spec import supports_pd_decode_swa_hicache

        def core(algo):
            return algo.supports_pd_decode_swa_hicache()

        with envs.SGLANG_TPU_ENABLE_EXPERIMENTAL_SPEC_WORKER.override(True):
            self.assertTrue(
                supports_pd_decode_swa_hicache(core, SpeculativeAlgorithm.DSPARK)
            )
            self.assertTrue(
                supports_pd_decode_swa_hicache(core, SpeculativeAlgorithm.EAGLE3)
            )
        with envs.SGLANG_TPU_ENABLE_EXPERIMENTAL_SPEC_WORKER.override(False):
            self.assertFalse(
                supports_pd_decode_swa_hicache(core, SpeculativeAlgorithm.DSPARK)
            )

    def test_host_relayed_draft_state_filters_with_host_indices(self):
        """A PP PD prefill keeps the relayed EAGLE draft state on host; filtering
        it with device indices would compile per batch on the device."""
        from sglang.srt.speculative.eagle_info import EagleDraftInput
        from sglang_tpu.hooks.experimental_spec import filter_spec_input

        spec_info = EagleDraftInput(
            topk_p=torch.ones(3, 1),
            topk_index=torch.zeros(3, 1, dtype=torch.int64),
            hidden_states=torch.zeros(3, 4),
            bonus_tokens=torch.arange(3),
        )
        batch = SimpleNamespace(spec_info=spec_info, enable_overlap=False)
        seen = []
        filter_spec_input(
            lambda b, keep, keep_device: seen.append(keep_device),
            batch,
            [0, 2],
            "device-indices",
        )
        self.assertEqual(seen[0].tolist(), [0, 2])
        batch.spec_info = None
        filter_spec_input(
            lambda b, keep, keep_device: seen.append(keep_device),
            batch,
            [0],
            "device-indices",
        )
        self.assertEqual(seen[1], "device-indices")

    def test_pp_stages_before_the_last_get_no_draft(self):
        """Under PP only the last stage hosts the draft; an earlier stage building one
        would bind the target embedding or lm_head its stage does not hold."""
        from sglang_tpu.experimental.speculative.pipeline_stage_worker import (
            TpuPipelineStageWorker,
        )
        from sglang_tpu.hooks.spec_worker import select_experimental_spec_worker

        earlier_stage = SimpleNamespace(pp_group=SimpleNamespace(is_last_rank=False))
        algorithms = (
            SpeculativeAlgorithm.EAGLE3,
            SpeculativeAlgorithm.DFLASH,
            SpeculativeAlgorithm.DSPARK,
        )
        with patch(
            "sglang_tpu.hooks.spec_worker.get_parallel", return_value=earlier_stage
        ):
            with envs.SGLANG_TPU_ENABLE_EXPERIMENTAL_SPEC_WORKER.override(True):
                for algorithm in algorithms:
                    self.assertIs(
                        select_experimental_spec_worker(None, algorithm, None),
                        TpuPipelineStageWorker,
                    )
            with envs.SGLANG_TPU_ENABLE_EXPERIMENTAL_SPEC_WORKER.override(False):
                for algorithm in algorithms:
                    self.assertIsNone(
                        select_experimental_spec_worker(None, algorithm, None)
                    )

    def test_only_a_pd_prefill_host_pads_without_overlap(self):
        """PP runs a PD prefill without overlap; decode relays through FutureMap,
        so a non-overlap decode must still refuse host padding."""
        batch = SimpleNamespace(enable_overlap=False)
        with patch("sglang_tpu.experimental._host_batch_padder", _pad):
            for mode, allowed in (
                ("prefill", True),
                ("decode", False),
                ("null", False),
            ):
                with patch(
                    "sglang_tpu.hooks.experimental_spec.get_disagg",
                    return_value=SimpleNamespace(disaggregation_mode=mode),
                ):
                    if allowed:
                        self.assertIs(initialize_host_batch(batch).host_pad_bs, _pad)
                    else:
                        with self.assertRaisesRegex(ValueError, "overlap"):
                            initialize_host_batch(SimpleNamespace(enable_overlap=False))

    def test_kimi_spec_defaults_yield_only_to_experimental_dspark(self):
        def original(server_args):
            raise ValueError("core refused")

        for algo, experimental, passes in (
            ("DSPARK", True, True),
            ("DSPARK", False, False),
            ("DFLASH", True, False),
        ):
            args = SimpleNamespace(speculative_algorithm=algo)
            with (
                self.subTest(algo=algo, experimental=experimental),
                envs.SGLANG_TPU_ENABLE_EXPERIMENTAL_SPEC_WORKER.override(experimental),
                patch(
                    "sglang.srt.arg_groups.overrides.resolving_view",
                    side_effect=lambda server_args: server_args,
                ),
            ):
                if passes:
                    self.assertIsNone(kimi_spec_defaults(original, args))
                else:
                    with self.assertRaisesRegex(ValueError, "core refused"):
                        kimi_spec_defaults(original, args)

    def test_draft_merge_preserves_right_hand_length_metadata(self):
        left, right = _state([1]), _state([2, 3])
        left.nxt_kv_lens_cpu = left.nxt_kv_lens_sum = None
        right.nxt_kv_lens_sum = int(right.nxt_kv_lens_cpu.sum())
        merged_rows = torch.tensor([1, 2, 3])
        left.merge_batch(right, future_indices=merged_rows)
        self.assertIs(left.nxt_kv_lens_cpu, right.nxt_kv_lens_cpu)
        self.assertEqual(left.nxt_kv_lens_sum, right.nxt_kv_lens_sum)
        self.assertTrue(left.tpu_host_seq_lens_is_committed)
        self.assertIs(left.future_indices, merged_rows)

    def test_sampler_samples_live_logits_at_the_host_bucket(self):
        """An outer logits processor emits 3 live rows; HOST_PADDED sampling
        params sit at bucket 4, and sampling still returns the bucket rows."""
        from sglang_tpu.sampler import TpuSampler

        sampler = _tpu_sampler()
        info = HostPaddedSamplingInfo(
            temperatures=torch.ones((4, 1)),
            top_ps=torch.ones(4),
            top_ks=torch.ones(4, dtype=torch.int32),
            min_ps=torch.zeros(4),
            is_all_greedy=True,
            is_any_greedy=True,
            need_top_p_sampling=False,
            need_top_k_sampling=True,
            need_min_p_sampling=False,
            vocab_size=8,
            device="cpu",
            host_pad_bs=_pad,
            real_batch_size=3,
        )
        logits = torch.randn(3, 8)
        tokens = sampler(
            LogitsProcessorOutput(next_token_logits=logits),
            info,
            False,
            None,
            None,
            torch.zeros(4, dtype=torch.int64),
        )
        self.assertIsInstance(sampler, TpuSampler)
        self.assertEqual(tokens.shape, (4,))
        self.assertEqual(tokens[:3].tolist(), logits.argmax(-1).tolist())

    def test_worker_installs_the_tpu_sampler(self):
        """A plain Sampler (--sampling-backend pytorch) cannot read HOST_PADDED
        host params, so the worker swaps in TpuSampler."""
        from sglang.srt.layers.sampler import Sampler
        from sglang_tpu.experimental.speculative import base_spec_worker
        from sglang_tpu.experimental.speculative.block_spec_worker import (
            TpuBlockSpecWorker,
        )
        from sglang_tpu.sampler import TpuSampler

        class _Stop(Exception):
            pass

        def init(custom_processors):
            runner = SimpleNamespace(sampler=object.__new__(Sampler))
            worker = object.__new__(TpuBlockSpecWorker)
            with (
                patch.object(
                    base_spec_worker,
                    "get_parallel",
                    return_value=SimpleNamespace(pp_size=1),
                ),
                patch.object(Sampler, "__init__", _module_init),
                # Stop right after the sampler is installed.
                patch.object(base_spec_worker, "get_dp_tp_group", side_effect=_Stop),
            ):
                with self.assertRaises(_Stop):
                    base_spec_worker.TpuSpecWorker.__init__(
                        worker,
                        SimpleNamespace(model_runner=runner, device="cpu"),
                        None,
                        3,
                    )
            return runner

        self.assertIsInstance(init(False).sampler, TpuSampler)
        # Enabling the server feature alone must not stop startup.
        self.assertIsInstance(init(True).sampler, TpuSampler)

    def test_worker_rejects_mixed_phase_dp_steps(self):
        """Under DP attention, prefill coordination or a skipped MLP sync lets one
        rank prefill while another decodes; the worker must refuse, not desync."""
        from sglang.srt.environ import envs as srt_envs
        from sglang_tpu.experimental.speculative import base_spec_worker
        from sglang_tpu.experimental.speculative.block_spec_worker import (
            TpuBlockSpecWorker,
        )

        class _Stop(Exception):
            pass

        class _Parallel:
            pp_size = 1

            @property
            def attn_tp_size(self):
                # Reached only past the mixed-phase check.
                raise _Stop

        def init(*, dp_attention, coordination=False, skip_mlp_sync=False):
            runner = SimpleNamespace(sampler=_tpu_sampler())
            worker = object.__new__(TpuBlockSpecWorker)
            with (
                patch.object(
                    base_spec_worker, "get_parallel", return_value=_Parallel()
                ),
                patch.object(base_spec_worker, "get_dp_tp_group", return_value=None),
                patch.object(base_spec_worker, "SpecTpSync", lambda group: None),
                patch.object(
                    base_spec_worker,
                    "is_dp_attention_enabled",
                    return_value=dp_attention,
                ),
                patch.object(
                    base_spec_worker,
                    "get_spec",
                    return_value=SimpleNamespace(
                        speculative_skip_dp_mlp_sync=skip_mlp_sync
                    ),
                ),
                srt_envs.SGLANG_ENABLE_DP_SPEC_PREFILL_COORDINATION.override(
                    coordination
                ),
            ):
                base_spec_worker.TpuSpecWorker.__init__(
                    worker,
                    SimpleNamespace(model_runner=runner, device="cpu"),
                    None,
                    3,
                )

        for flags in ({"coordination": True}, {"skip_mlp_sync": True}):
            with self.assertRaisesRegex(NotImplementedError, "mixed prefill/decode"):
                init(dp_attention=True, **flags)
            with self.assertRaises(_Stop):
                init(dp_attention=False, **flags)
        with self.assertRaises(_Stop):
            init(dp_attention=True)

    def test_idle_dp_rank_mirrors_the_whole_step(self):
        """An idle DP rank runs the active step's forwards and syncs in order (the
        greedy draft, verify, commit) on all-padding rows; a skipped one leaves
        its peers hanging in the collective."""
        from sglang_tpu.experimental.speculative.block_spec_worker import (
            TpuBlockSpecWorker,
        )
        from sglang_tpu.experimental.speculative.spec_types import (
            DraftResult,
            SpecState,
        )

        worker = object.__new__(TpuBlockSpecWorker)
        calls = []
        drafts = torch.zeros((4, 2), dtype=torch.int64)

        def draft(batch, bucket, rows, verify_input):
            calls.append(("draft", bucket.temperatures, bucket.live_mask.any().item()))
            return DraftResult(
                draft_tokens=drafts, draft_logits=None, num_verify_drafts=None
            )

        worker.draft = draft
        worker._forward_idle_verify = lambda **kwargs: (
            calls.append(("verify", kwargs["verify_tokens"].shape)) or "out"
        )
        worker.commit = lambda *args: calls.append(
            ("commit", args[4], args[5].accept_lens.tolist())
        )
        worker._device_zeros = lambda rows: torch.zeros((rows,), dtype=torch.int64)
        zeros = torch.zeros((4,), dtype=torch.int64)
        state = SpecState(bonus_tokens=zeros, seq_lens=zeros, req_pool_indices=zeros)
        worker._forward_idle_step(batch=None, state=state, rows=None, verify_input=None)
        self.assertEqual(
            calls,
            [
                ("draft", None, False),
                # Flat [pbs * (gamma + 1)] verify rows.
                ("verify", torch.Size([12])),
                ("commit", "out", [0, 0, 0, 0]),
            ],
        )

    def test_target_token_sync_has_one_shape_on_every_dp_rank(self):
        """Every DP rank must issue extend's target-token sync at one shape; an idle
        rank (no rows) syncing a different shape than its peers hangs them."""
        from sglang_tpu.experimental.speculative.block_spec_worker import (
            TpuBlockSpecWorker,
        )

        worker = object.__new__(TpuBlockSpecWorker)
        synced = []
        worker._tp_sync = SimpleNamespace(
            sync=lambda site, values: synced.append(tuple(values.shape)) or values
        )
        worker._compile_policy = lambda: SimpleNamespace(_extend_compile_cap=lambda: 8)
        tokens = torch.tensor([5, 6, 7])
        worker._dp_attention = True
        self.assertEqual(worker.sync_target_tokens(tokens).tolist(), [5, 6, 7])
        self.assertEqual(worker.sync_target_tokens(tokens[:0]).tolist(), [])
        self.assertEqual(synced, [(8,), (8,)])
        worker._dp_attention = False
        self.assertIs(worker.sync_target_tokens(tokens), tokens)

    def test_idle_draft_prefill_runs_the_peers_token_bucket(self):
        """An idle DP rank's draft prefill pads to the token bucket its peers run;
        the synced live counts would pick other attention collectives and hang."""
        from sglang_tpu.experimental.speculative import chain_spec_worker
        from sglang_tpu.experimental.speculative.chain_spec_worker import (
            TpuChainSpecWorker,
        )

        worker = object.__new__(TpuChainSpecWorker)
        worker.device = "cpu"
        forwarded = []
        worker._draft_runner = SimpleNamespace(forward=forwarded.append)
        worker._chain_draft_worker = SimpleNamespace(draft_context=ExitStack)
        idle_batch = SimpleNamespace(global_num_tokens_cpu=[780, 0])
        target_hidden = torch.zeros((8192, 4))
        with patch.object(
            chain_spec_worker.ForwardBatch, "init_new", return_value=idle_batch
        ):
            worker._forward_idle_draft_prefill(None, target_hidden=target_hidden)
        (forward_batch,) = forwarded
        self.assertEqual(forward_batch.global_num_tokens_cpu, [8192, 8192])
        self.assertEqual(forward_batch.global_num_tokens_gpu.tolist(), [8192, 8192])
        # Core's DP unpadding of the relabeled idle rank reads this backup.
        self.assertEqual(tuple(forward_batch.hidden_states_backup.shape), (0, 4))

    def test_dp_attention_rejects_a_draft_head_sharded_over_full_tp(self):
        """A draft sharing a target head sharded over the full TP group drafts from one
        shard inside its attention-TP group: acceptance collapses with no error. A
        head sharded over the draft's own group (built in its scope) is fine."""
        from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
        from sglang_tpu.experimental.speculative.base_spec_worker import (
            _check_draft_lm_head_group,
        )

        def head(*, use_attn_tp_group, tp_size):
            lm_head = object.__new__(VocabParallelEmbedding)
            lm_head.__dict__.update(
                enable_tp=True, use_attn_tp_group=use_attn_tp_group, tp_size=tp_size
            )
            return lm_head

        def check(lm_head, attn_tp_size):
            _check_draft_lm_head_group(
                SimpleNamespace(lm_head=lm_head), attn_tp_size=attn_tp_size
            )

        for attn_tp_size in (1, 4):
            with self.assertRaisesRegex(ValueError, "--enable-dp-lm-head"):
                check(head(use_attn_tp_group=False, tp_size=8), attn_tp_size)
        # Target head under --enable-dp-lm-head; a draft head built in the draft's
        # scope, where the narrowed group reports use_attn_tp_group False.
        check(head(use_attn_tp_group=True, tp_size=4), 4)
        check(head(use_attn_tp_group=False, tp_size=4), 4)
        check(head(use_attn_tp_group=False, tp_size=1), 1)
        check(None, 4)

    def test_block_dp_attention_drafts_greedy(self):
        """Under DP attention the block draft never samples: peers and idle ranks
        must issue the same draft sync, and sampled requests take target-only accept."""
        from sglang_tpu.experimental.speculative.block_spec_worker import (
            TpuBlockSpecWorker,
        )

        worker = object.__new__(TpuBlockSpecWorker)
        greedy = torch.ones((4, 2), dtype=torch.int64)
        worker._forward_draft_block = lambda **kwargs: (None, greedy)
        worker._sync_greedy_drafts = lambda tokens: tokens
        worker._sample_draft_block = lambda *args: ("sampled", "logits")
        worker._all_drafts = {4: None}
        bucket = SimpleNamespace(
            state=SimpleNamespace(seq_lens=torch.zeros(4), bonus_tokens=None),
            live_mask=None,
            temperatures=torch.ones(4),
            top_ks=torch.ones(4),
        )
        results = {}
        for sampled in (True, False):
            worker._sampled_drafts = sampled
            results[sampled] = worker.draft(None, bucket, None, None)
        self.assertEqual(results[True].draft_tokens, "sampled")
        self.assertIs(results[False].draft_tokens, greedy)
        self.assertIsNone(results[False].draft_logits)

    def test_tpu_opts_experimental_spec_into_dp_attention(self):
        """Core rejects DFLASH with DP attention unless the platform opts in; every
        experimental worker serves DP attention, core's DFlash worker does not."""
        from sglang_tpu.tpu_platform import TpuSRTPlatform

        platform = TpuSRTPlatform()
        for enabled in (True, False):
            with envs.SGLANG_TPU_ENABLE_EXPERIMENTAL_SPEC_WORKER.override(enabled):
                for algorithm in ("DFLASH", "DSPARK", "EAGLE3", "NEXTN"):
                    with self.subTest(algorithm=algorithm, enabled=enabled):
                        self.assertEqual(
                            platform.supports_speculative_dp_attention(algorithm),
                            enabled,
                        )

    def test_pd_seeds_reach_decode_as_host_padded_inputs(self):
        """A PD decode server's prebuilt batch must carry the experimental input
        type (decode prep and filter/merge reject core's), built on host: core's
        device-side seed ops compile once per live bs."""
        from sglang_tpu.experimental.speculative.chain_info import (
            HostPaddedChainDraftInput,
        )
        from sglang_tpu.hooks.experimental_spec import (
            build_disagg_block_input,
            build_disagg_eagle_input,
        )

        def core(*args):
            raise AssertionError("core seed builder must not run")

        reqs = [
            SimpleNamespace(
                output_topk_p=torch.tensor([0.5]),
                output_topk_index=torch.tensor([7]),
                hidden_states_tensor=torch.full((4,), float(i)),
                output_dsa_topk_indices=None,
            )
            for i in range(2)
        ]
        rows = torch.tensor([3, 5])

        def batch(algorithm):
            return SimpleNamespace(
                host_pad_bs=_pad,
                enable_overlap=True,
                device="cpu",
                reqs=reqs,
                spec_algorithm=algorithm,
                seq_lens=torch.tensor([11, 21], dtype=torch.int32),
                req_pool_indices=rows,
                req_pool_indices_cpu=rows,
                req_to_token_pool=None,
            )

        future_map = MagicMock()
        bonus = torch.tensor([7, 9])

        def check_relayed(draft):
            self.assertIs(draft.future_indices, rows)
            self.assertEqual(draft.new_seq_lens.tolist(), [11, 21])
            self.assertEqual(draft.new_seq_lens.dtype, torch.int64)
            self.assertFalse(draft.tpu_host_seq_lens_is_committed)
            future_map.publish.assert_called_once()
            payload = future_map.stash.call_args.args[1]
            self.assertIs(payload.bonus_tokens, bonus)
            future_map.reset_mock()
            return payload

        # MTP (core resolves NEXTN to EAGLE) and EAGLE3 both keep the seed in the
        # chain carry table.
        for algorithm in (SpeculativeAlgorithm.EAGLE, SpeculativeAlgorithm.EAGLE3):
            seeded = []
            with patch(
                "sglang_tpu.experimental._pd_carry_seeder",
                lambda indices, hidden: seeded.append((indices, hidden)),
            ):
                chain = build_disagg_eagle_input(
                    core, batch(algorithm), bonus, future_map
                )
            self.assertIsInstance(chain, HostPaddedChainDraftInput)
            check_relayed(chain)
            ((indices, hidden),) = seeded
            self.assertIs(indices, rows)
            self.assertEqual(hidden[:, 0].tolist(), [0.0, 1.0])

        for algorithm in (SpeculativeAlgorithm.DFLASH, SpeculativeAlgorithm.DSPARK):
            block = build_disagg_block_input(core, batch(algorithm), bonus, future_map)
            self.assertIsInstance(block, HostPaddedBlockDraftInput)
            check_relayed(block)

        reqs[0].output_dsa_topk_indices = torch.zeros(4)
        with self.assertRaisesRegex(NotImplementedError, "DSA"):
            build_disagg_eagle_input(
                core, batch(SpeculativeAlgorithm.EAGLE3), bonus, future_map
            )
        plain = batch(SpeculativeAlgorithm.EAGLE3)
        plain.host_pad_bs = None
        sentinel = object()
        self.assertIs(
            build_disagg_eagle_input(lambda *a: sentinel, plain, bonus, future_map),
            sentinel,
        )
        self.assertIs(
            build_disagg_block_input(lambda *a: sentinel, plain, bonus, future_map),
            sentinel,
        )

    def test_pd_seed_resets_reused_gdn_checkpoint_offsets(self):
        """A reused decode slot keeps its previous occupant's verify read offset;
        the transferred GDN state sits in checkpoint row 0, so the seed resets it."""
        from sglang.srt.configs.mamba_utils import (
            Mamba2CacheParams,
            Mamba2StateDType,
            Mamba2StateShape,
        )
        from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
        from sglang_tpu.hooks.experimental_spec import _reset_pd_checkpoints
        from sglang_tpu.mamba_pool import TpuMambaPool

        shape = Mamba2StateShape.create(
            tp_world_size=1,
            intermediate_size=8,
            n_groups=1,
            num_heads=2,
            head_dim=4,
            state_size=4,
            conv_kernel=4,
        )
        mamba_pool = TpuMambaPool(
            size=3,
            spec_state_size=3,
            cache_params=Mamba2CacheParams(
                dtype=Mamba2StateDType(conv=torch.float32, temporal=torch.float32),
                layers=[0],
                shape=shape,
            ),
            mamba_layer_ids=[0],
            device="cpu",
            speculative_num_draft_tokens=4,
            speculative_eagle_topk=1,
        )
        mamba_pool.checkpoint_read_offsets.fill_(2)
        pool = MagicMock(spec=HybridReqToTokenPool)
        pool.mamba_pool = mamba_pool
        reqs = [
            # Allocation stores a 0-d slot tensor.
            SimpleNamespace(kv=SimpleNamespace(mamba_pool_idx=torch.tensor(i)))
            for i in (1, 3)
        ]
        _reset_pd_checkpoints(SimpleNamespace(req_to_token_pool=pool, reqs=reqs))
        self.assertEqual(mamba_pool.checkpoint_read_offsets.tolist(), [2, -1, 2, -1])

    def test_pd_carry_seed_writes_at_the_batch_bucket(self):
        """The decode-side carry write must keep the warmed bucket shape; padded
        rows land on the reserved pool row 0."""
        from sglang_tpu.experimental.speculative.chain_spec_worker import (
            TpuChainSpecWorker,
        )

        worker = object.__new__(TpuChainSpecWorker)
        worker.device = "cpu"
        worker._pad_bs = _pad
        worker._draft_runner = SimpleNamespace(dtype=torch.float32)
        carry = torch.zeros((8, 2))
        worker._chain_draft_worker = SimpleNamespace(carry=MagicMock(wraps=carry))
        worker._seed_carry(torch.tensor([3, 5]), torch.ones((2, 2)))
        (dim, rows, hidden), _ = worker._chain_draft_worker.carry.index_copy_.call_args
        self.assertEqual(rows.tolist(), [3, 5, 0, 0])
        self.assertEqual(hidden.shape, (4, 2))
        self.assertEqual(carry[3].tolist(), [1.0, 1.0])
        self.assertEqual(carry[0].tolist(), [0.0, 0.0])

    def _retraction_worker(self, draft_pool, *, mode="decode", backup="cpu_tensor"):
        from sglang_tpu.experimental.speculative.block_spec_worker import (
            TpuBlockSpecWorker,
        )

        worker = object.__new__(TpuBlockSpecWorker)
        worker._draft_worker = SimpleNamespace(
            draft_runners=(SimpleNamespace(token_to_kv_pool=draft_pool),)
        )
        disagg = SimpleNamespace(
            disaggregation_mode=mode, disaggregation_decode_retraction_backup=backup
        )
        get_disagg = patch(
            "sglang_tpu.experimental.speculative.base_spec_worker.get_disagg",
            return_value=disagg,
        )
        return worker, get_disagg

    def test_cpu_retraction_restores_draft_kv_at_resumed_slots(self):
        """A PD decode request retracted with a CPU backup must resume with its
        own draft KV, not whatever its newly allocated slots hold."""
        from sglang.srt.managers.schedule_batch import Req
        from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

        draft_pool = MHATokenToKVPool(
            size=16,
            page_size=1,
            dtype=torch.float32,
            head_num=1,
            head_dim=2,
            layer_num=1,
            device="cpu",
            enable_memory_saver=False,
            enable_alt_stream=False,
        )
        old_slots, new_slots = torch.tensor([3, 4, 5]), torch.tensor([9, 10, 11])
        draft_pool.k_buffer[0][old_slots] = 7.0
        draft_pool.v_buffer[0][old_slots] = 8.0
        req_to_token = torch.zeros((4, 8), dtype=torch.int64)
        req_to_token[1, :3] = old_slots
        req_to_token[2, :3] = new_slots
        req = object.__new__(Req)
        req.origin_input_ids = [1, 2, 3]
        req.output_ids = [4]
        req.kv = SimpleNamespace(req_pool_idx=1, mamba_pool_idx=None, holds_mamba=False)
        target_allocator = MagicMock()
        target_allocator.get_kvcache.return_value.cpu_copy_carries_mamba = True
        req_to_token_pool = SimpleNamespace(req_to_token=req_to_token)

        worker, get_disagg = self._retraction_worker(draft_pool)
        with get_disagg:
            draft_kv_pools = worker.retraction_draft_kv_pools
        req.offload_kv_cache(
            req_to_token_pool, target_allocator, draft_kv_pools=draft_kv_pools
        )
        # Resume lands on another req pool row whose slots held other KV.
        req.kv.req_pool_idx = 2
        draft_pool.k_buffer[0][new_slots] = -1.0
        draft_pool.v_buffer[0][new_slots] = -1.0
        req.load_kv_cache(req_to_token_pool, target_allocator)
        self.assertTrue(bool((draft_pool.k_buffer[0][new_slots] == 7.0).all()))
        self.assertTrue(bool((draft_pool.v_buffer[0][new_slots] == 8.0).all()))

    def test_unbackable_draft_pool_refuses_cpu_retraction_at_startup(self):
        """A draft pool without a CPU copy must fail when the scheduler reads the
        pools at startup, not resume requests over stale draft KV."""
        from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool

        mla_pool = object.__new__(MLATokenToKVPool)
        worker, get_disagg = self._retraction_worker(mla_pool)
        with get_disagg, self.assertRaisesRegex(NotImplementedError, "draft KV"):
            worker.retraction_draft_kv_pools
        # No CPU backup is taken off a decode server or with another backend.
        for mode, backup in (("prefill", "cpu_tensor"), ("decode", "none")):
            worker, get_disagg = self._retraction_worker(
                mla_pool, mode=mode, backup=backup
            )
            with get_disagg:
                self.assertEqual(worker.retraction_draft_kv_pools, (mla_pool,))

    def test_per_batch_checks_refuse_only_unsupported_requests(self):
        """Filtered sampling needs the sort-free filter (the sort-based target
        keeps different boundary ties), but greedy rows (top_k = 1) do not; custom
        logit processors are refused per batch, not at startup."""
        from sglang_tpu.experimental.speculative.block_spec_worker import (
            TpuBlockSpecWorker,
        )

        worker = object.__new__(TpuBlockSpecWorker)
        worker._bsearch_filtering = False

        def batch(top_k=1, top_p=1.0, custom=False):
            sampling_info = SimpleNamespace(
                logit_bias=None,
                acc_additive_penalties=None,
                acc_scaling_penalties=None,
                penalizer_orchestrator=None,
                sampling_seed=None,
                has_custom_logit_processor=custom,
                vocab_size=32,
                top_ks=torch.tensor([1, top_k], dtype=torch.int32),
                top_ps=torch.tensor([1.0, top_p]),
                min_ps=torch.zeros(2),
            )
            return SimpleNamespace(
                has_grammar=False, return_logprob=False, sampling_info=sampling_info
            )

        check = worker._check_supported
        check(batch=batch(), pp_proxy_tensors=None)
        # Temperature-only sampled row: no filter needed.
        check(batch=batch(top_k=1 << 30), pp_proxy_tensors=None)
        with self.assertRaisesRegex(NotImplementedError, "BSEARCH"):
            check(batch=batch(top_k=1 << 30, top_p=0.9), pp_proxy_tensors=None)
        with self.assertRaisesRegex(NotImplementedError, "custom logit processors"):
            check(batch=batch(custom=True), pp_proxy_tensors=None)
        worker._bsearch_filtering = True
        check(batch=batch(top_k=1 << 30, top_p=0.9), pp_proxy_tensors=None)

    def test_block_draft_input_relays_through_the_future_map(self):
        """RelayPayload.from_draft_input reads the EAGLE seed slots of every draft
        input; the block input left them out and every DFlash/DSpark startup died."""
        from sglang.srt.managers.overlap_utils import RelayPayload

        payload = RelayPayload.from_draft_input(_state([1, 2]))
        self.assertEqual(payload.bonus_tokens.tolist(), [1, 1])
        self.assertIsNone(payload.topk_p)
        self.assertIsNone(payload.hidden_states)

    def test_decode_prep_reserves_two_windows_and_keeps_committed_host_lengths(self):
        """prepare_for_decode reserves 2 * N tokens past the committed length,
        plans the verify window on host from the committed lengths, and refuses
        a device-resident pool."""
        from sglang_tpu.experimental.speculative import draft_info

        page_size, num_draft_tokens = 4, 3
        reqs = [
            SimpleNamespace(
                kv=SimpleNamespace(kv_committed_len=c, kv_allocated_len=a),
                decode_batch_idx=0,
                lora_id=None,
            )
            for c, a in ((5, 8), (10, 20))
        ]
        batch = SimpleNamespace(
            reqs=reqs,
            batch_size=lambda: len(reqs),
            maybe_evict_swa=lambda: None,
            req_to_token_pool=SimpleNamespace(
                host_resident=True,
                req_to_token=torch.arange(3 * 32).reshape(3, 32),
            ),
            token_to_kv_pool_allocator=SimpleNamespace(page_size=page_size),
            tree_cache=None,
            req_pool_indices=torch.tensor([1, 2]),
            req_pool_indices_cpu=torch.tensor([1, 2]),
            device="cpu",
            enable_overlap=True,
            return_logprob=False,
            has_grammar=False,
        )
        allocations = []
        spec = SimpleNamespace(
            speculative_num_draft_tokens=num_draft_tokens,
            speculative_draft_window_size=None,
        )
        draft = HostPaddedBlockDraftInput(
            bonus_tokens=torch.zeros(2, dtype=torch.int64),
            new_seq_lens=torch.zeros(2, dtype=torch.int64),
        )
        with (
            patch.object(draft_info, "get_spec", return_value=spec),
            patch.object(
                draft_info,
                "allocate_and_assign_decode_cache",
                lambda **kwargs: allocations.append(kwargs),
            ),
        ):
            draft.prepare_for_decode(batch)
            # Committed 5 + 6 rounds up to 12; 20 already covers 10 + 6.
            self.assertEqual(allocations[0]["nxt_kv_lens_cpu"].tolist(), [12, 20])
            self.assertEqual(allocations[0]["num_needed_tokens"], 4)
            self.assertEqual([r.kv.kv_allocated_len for r in reqs], [12, 20])
            self.assertEqual([r.decode_batch_idx for r in reqs], [1, 1])
            self.assertEqual(batch.seq_lens_cpu.tolist(), [5, 10])
            self.assertEqual(batch.seq_lens_sum, 15)
            self.assertTrue(draft.tpu_host_seq_lens_is_committed)
            # Window [committed, committed + 2N) of each request's pool row.
            self.assertEqual(
                draft.tpu_verify_cache_candidates.tolist(),
                [list(range(32 + 5, 32 + 11)), list(range(64 + 10, 64 + 16))],
            )
            self.assertEqual(draft.tpu_committed_lens.tolist(), [5, 10])
            self.assertEqual(draft.tpu_req_pool_indices.tolist(), [1, 2])

            # Logprobs can commit past the window's bound: stay on the sync path.
            batch.return_logprob = True
            draft.prepare_for_decode(batch)
            self.assertFalse(draft.tpu_host_seq_lens_is_committed)
            self.assertIsNone(draft.tpu_verify_cache_candidates)

            batch.req_to_token_pool.host_resident = False
            with self.assertRaisesRegex(RuntimeError, "host-resident"):
                draft.prepare_for_decode(batch)


if __name__ == "__main__":
    unittest.main()
