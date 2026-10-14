"""CPU contracts for plugin-owned host padding and scheduler composition."""

from __future__ import annotations

import pkgutil
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.plugins.hook_registry import HookRegistry, _propagate_patch
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.envs import envs
from sglang_tpu.experimental import register_host_batch_padder
from sglang_tpu.experimental.draft_input import HostPaddedDraftInput
from sglang_tpu.experimental.host_sampling import HostPaddedSamplingInfo
from sglang_tpu.hooks.experimental_spec import alloc_extend, hook_specs

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _pad(bs):
    return 4 if bs <= 4 else 8


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
    return HostPaddedDraftInput(
        topk_p=torch.empty(bs, 0),
        topk_index=torch.empty(bs, 0, dtype=torch.int64),
        bonus_tokens=torch.ones(bs, dtype=torch.int64),
        new_seq_lens=torch.arange(bs),
        hidden_states=torch.empty(bs, 0),
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
        stack = self.enterContext(ExitStack())
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
        from sglang_tpu.experimental.tpu_dspark_worker import TpuDsparkWorker
        from sglang_tpu.experimental.tpu_spec_types import SpecState

        state = SpecState(
            torch.tensor([4, 5]), torch.tensor([10, 20]), torch.tensor([1, 2])
        )
        draft = TpuDsparkWorker.state_to_spec_info(None, state)
        self.assertIsInstance(draft, HostPaddedDraftInput)
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


if __name__ == "__main__":
    unittest.main()
