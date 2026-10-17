"""CPU contracts for plugin-owned host padding and scheduler composition."""

from __future__ import annotations

import pkgutil
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

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
from sglang_tpu.experimental import register_host_batch_padder
from sglang_tpu.experimental.sampling.sampling_batch_info import HostPaddedSamplingInfo
from sglang_tpu.experimental.speculative.block_info import (
    HostPaddedBlockDraftInput,
)
from sglang_tpu.hooks.experimental_spec import alloc_extend, hook_specs

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _pad(bs):
    return 4 if bs <= 4 else 8


def _module_init(self):
    # Sampler.__init__ needs the TP group.
    torch.nn.Module.__init__(self)


def _host_padded_sampler():
    from sglang.srt.layers.sampler import Sampler
    from sglang_tpu.experimental.sampling.sampler import HostPaddedTpuSampler

    with patch.object(Sampler, "__init__", _module_init):
        return HostPaddedTpuSampler()


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

    def batch(self, rows, experimental=True):
        batch = ScheduleBatch.init_new(
            [_req(0.1 * (i + 1)) for i in range(len(rows))],
            SimpleNamespace(device="cpu"),
            None,
            None,
            SimpleNamespace(is_encoder_decoder=False, vocab_size=32),
            enable_overlap=True,
            spec_algorithm=SpeculativeAlgorithm.DSPARK
            if experimental
            else SpeculativeAlgorithm.NONE,
        )
        batch.req_pool_indices_cpu = torch.tensor(rows)
        batch.req_pool_indices = torch.tensor(rows)
        batch.seq_lens = torch.arange(len(rows)) + 10
        batch.orig_seq_lens = batch.seq_lens.clone()
        batch.seq_lens_cpu = batch.seq_lens.clone()
        batch.sampling_info = SamplingBatchInfo.from_schedule_batch(batch, 32)
        if experimental:
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

    def test_unselected_batches_keep_default_sampling_and_device_rows(self):
        batch = self.batch([2, 5, 8], experimental=False)
        self.assertIs(type(batch.sampling_info), SamplingBatchInfo)
        self.assertFalse(hasattr(batch, "host_pad_bs"))
        batch.input_ids = torch.tensor([20, 50, 80])
        batch.filter_batch(keep_indices=[0, 2])
        self.assertEqual(batch.seq_lens.tolist(), [10, 12])
        self.assertEqual(batch.input_ids.tolist(), [20, 80])
        self.assertEqual(batch.sampling_info.temperatures.shape, (2, 1))
        with envs.SGLANG_TPU_ENABLE_EXPERIMENTAL_SPEC_WORKER.override(False):
            dspark = self.batch([1])
        self.assertFalse(hasattr(dspark, "host_pad_bs"))
        self.assertIs(type(dspark.sampling_info), SamplingBatchInfo)

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
        plain = self.batch([2], experimental=False)
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
        from sglang.srt.environ import envs as srt_envs
        from sglang_tpu.experimental import is_experimental_spec_worker

        for experimental, no_d2h in ((True, False), (False, True), (False, False)):
            with (
                envs.SGLANG_TPU_ENABLE_EXPERIMENTAL_SPEC_WORKER.override(experimental),
                srt_envs.SGLANG_TPU_HOST_PADDING_NO_D2H.override(no_d2h),
            ):
                for algo in (SpeculativeAlgorithm.DFLASH, SpeculativeAlgorithm.DSPARK):
                    self.assertEqual(
                        is_experimental_spec_worker(algo), experimental or no_d2h
                    )

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
        from sglang_tpu.experimental.sampling.sampler import HostPaddedTpuSampler

        sampler = _host_padded_sampler()
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
        self.assertIsInstance(sampler, HostPaddedTpuSampler)
        self.assertEqual(tokens.shape, (4,))
        self.assertEqual(tokens[:3].tolist(), logits.argmax(-1).tolist())

    def test_worker_installs_the_host_padded_sampler_and_rejects_processors(self):
        """A plain Sampler (--sampling-backend pytorch) cannot read HOST_PADDED
        host params; custom logit processors expect live rows."""
        from sglang.srt.layers.sampler import Sampler
        from sglang_tpu.experimental.sampling.sampler import HostPaddedTpuSampler
        from sglang_tpu.experimental.speculative import base_spec_worker
        from sglang_tpu.experimental.speculative.block_spec_worker import (
            TpuBlockSpecWorker,
        )

        class _Stop(Exception):
            pass

        def init(custom_processors):
            runner = SimpleNamespace(sampler=object.__new__(Sampler))
            worker = object.__new__(TpuBlockSpecWorker)
            features = SimpleNamespace(enable_custom_logit_processor=custom_processors)
            with (
                patch.object(
                    base_spec_worker,
                    "get_parallel",
                    return_value=SimpleNamespace(pp_size=1),
                ),
                patch.object(
                    base_spec_worker,
                    "get_exec",
                    return_value=SimpleNamespace(features=features),
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

        self.assertIsInstance(init(False).sampler, HostPaddedTpuSampler)
        with self.assertRaisesRegex(NotImplementedError, "custom logit processors"):
            init(True)

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
            runner = SimpleNamespace(sampler=_host_padded_sampler())
            worker = object.__new__(TpuBlockSpecWorker)
            with (
                patch.object(
                    base_spec_worker, "get_parallel", return_value=_Parallel()
                ),
                patch.object(
                    base_spec_worker,
                    "get_exec",
                    return_value=SimpleNamespace(
                        features=SimpleNamespace(enable_custom_logit_processor=False)
                    ),
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

    def test_partial_bucket_results_are_copied_on_the_forward_thread(self):
        """Rows the result-copy thread reads below a full bucket are fresh tensors,
        not views that need a program queued behind the next step."""
        from sglang_tpu.experimental.speculative.base_spec_worker import _live_rows

        full = torch.arange(8).reshape(4, 2)
        self.assertIs(_live_rows(full, 4), full)
        live = _live_rows(full, 3)
        self.assertEqual(live.tolist(), full[:3].tolist())
        self.assertIsNone(live._base)
        self.assertNotEqual(live.data_ptr(), full.data_ptr())

    def test_filtered_target_warmup_matches_serving_inputs(self):
        """Serving uploads temperatures, top-p and min-p as separate tensors; an
        aliased warmup would leave Dynamo an alias guard serving never satisfies."""
        from sglang_tpu.experimental.speculative import base_spec_worker
        from sglang_tpu.experimental.speculative.block_spec_worker import (
            TpuBlockSpecWorker,
        )

        worker = object.__new__(TpuBlockSpecWorker)
        worker.num_verify_rows = 2
        worker.device = "cpu"
        worker._target_worker = SimpleNamespace(
            model_runner=SimpleNamespace(model_config=SimpleNamespace(vocab_size=8))
        )
        calls = []
        with (
            patch.object(
                TpuBlockSpecWorker,
                "_b_bucketer",
                return_value=SimpleNamespace(buckets=[4]),
            ),
            patch.object(
                base_spec_worker,
                "warmup_compiled_op",
                lambda op, *args: calls.append(args),
            ),
            patch.object(base_spec_worker, "_joint_filter_renorm", lambda *a: None),
            patch.object(torch.accelerator, "synchronize", lambda: None),
        ):
            worker._warmup_filtered_target()
        probs, temperatures, top_ks, top_ps, min_ps, rows = calls[0]
        self.assertEqual(len({id(t) for t in (temperatures, top_ps, min_ps)}), 3)
        self.assertEqual(
            (temperatures.shape, temperatures.dtype), ((4,), torch.float32)
        )
        self.assertEqual(top_ks.dtype, torch.int32)
        self.assertEqual(probs.shape, (8, 8))

    def test_pd_seed_reaches_decode_as_the_host_padded_input(self):
        """A PD decode server's prebuilt batch must carry the experimental input
        type: experimental decode prep and filter/merge reject the core one."""
        from sglang_tpu.experimental.speculative.eagle_info import (
            HostPaddedEagleDraftInput,
        )
        from sglang_tpu.hooks.experimental_spec import build_disagg_eagle_input

        rows = torch.tensor([3, 5])
        core = SimpleNamespace(
            bonus_tokens=torch.tensor([7, 9]),
            topk_p=torch.ones(2, 1),
            topk_index=torch.zeros(2, 1, dtype=torch.int64),
            hidden_states=torch.zeros(2, 4),
            dsa_topk_indices=None,
            future_indices=rows,
        )
        batch = SimpleNamespace(
            host_pad_bs=_pad, seq_lens=torch.tensor([11, 21], dtype=torch.int32)
        )
        draft = build_disagg_eagle_input(lambda *a: core, batch, None, None)
        self.assertIsInstance(draft, HostPaddedEagleDraftInput)
        self.assertIs(draft.future_indices, rows)
        self.assertIs(draft.topk_p, core.topk_p)
        self.assertIs(draft.hidden_states, core.hidden_states)
        self.assertEqual(draft.new_seq_lens.tolist(), [11, 21])
        self.assertFalse(draft.tpu_host_seq_lens_is_committed)

        core.dsa_topk_indices = torch.zeros(2, 4)
        with self.assertRaisesRegex(NotImplementedError, "DSA"):
            build_disagg_eagle_input(lambda *a: core, batch, None, None)
        batch.host_pad_bs = None
        self.assertIs(
            build_disagg_eagle_input(lambda *a: core, batch, None, None), core
        )

    def test_filtered_sampling_needs_the_sort_free_filter(self):
        """Verify always filters sort-free; with the target on the sort-based
        fallback the two keep different boundary ties, so filtering is refused."""
        from sglang_tpu.experimental.speculative.block_spec_worker import (
            TpuBlockSpecWorker,
        )

        worker = object.__new__(TpuBlockSpecWorker)
        worker._bsearch_filtering = False

        def batch(**need):
            sampling_info = SimpleNamespace(
                logit_bias=None,
                acc_additive_penalties=None,
                acc_scaling_penalties=None,
                penalizer_orchestrator=None,
                sampling_seed=None,
                need_top_k_sampling=False,
                need_top_p_sampling=False,
                need_min_p_sampling=False,
            )
            sampling_info.__dict__.update(need)
            return SimpleNamespace(
                has_grammar=False, return_logprob=False, sampling_info=sampling_info
            )

        worker._check_supported(batch=batch(), pp_proxy_tensors=None)
        with self.assertRaisesRegex(NotImplementedError, "BSEARCH"):
            worker._check_supported(
                batch=batch(need_top_p_sampling=True), pp_proxy_tensors=None
            )
        worker._bsearch_filtering = True
        worker._check_supported(
            batch=batch(need_top_p_sampling=True), pp_proxy_tensors=None
        )

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
