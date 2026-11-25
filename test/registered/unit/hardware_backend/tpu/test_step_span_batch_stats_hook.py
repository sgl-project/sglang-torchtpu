"""CPU tests for the step-span batch-composition stats hook."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import contextlib
import sys
import types
import unittest
from unittest.mock import patch

import torch

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.observability import profile_utils as profile_utils_mod
from sglang.srt.plugins.hook_registry import HookRegistry, HookType
from sglang.srt.runtime_context import get_parallel, publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils.nvtx_utils import profile_range
from sglang.test.test_utils import CustomTestCase
from sglang_tpu import compile_cache, profile_batch_stats
from sglang_tpu.compilation.bucketer import PowerBucketer
from sglang_tpu.hooks import HOOKS
from sglang_tpu.hooks import step_span_batch_stats as hook_mod

_TARGET = "sglang.srt.observability.profile_utils.build_step_span_name"


@contextlib.contextmanager
def _applied_step_span_hook(*extra_importers: types.ModuleType):
    """Apply the AFTER hook through HookRegistry and restore bindings after."""
    original = profile_utils_mod.build_step_span_name
    saved_hooks = dict(HookRegistry._hooks)
    saved_patched = set(HookRegistry._patched)
    HookRegistry.reset()
    try:
        HookRegistry.register(
            _TARGET, hook_mod.append_batch_stats_to_step_span, HookType.AFTER
        )
        HookRegistry.apply_hooks()
        yield original
    finally:
        profile_utils_mod.build_step_span_name = original
        for mod in extra_importers:
            mod.build_step_span_name = original
        HookRegistry.reset()
        HookRegistry._hooks.update(saved_hooks)
        HookRegistry._patched.update(saved_patched)


class TestStepSpanBatchStatsHook(CustomTestCase):
    def setUp(self):
        super().setUp()
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy"), role="scheduler")
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(get_parallel().override(attn_dp_enabled=False))
        profile_batch_stats.set_target_compile_policy(None)
        profile_batch_stats.set_active_batch_id(None)
        self.addCleanup(profile_batch_stats.set_target_compile_policy, None)
        self.addCleanup(profile_batch_stats.set_active_batch_id, None)

    def test_hook_registered_in_tpu_hooks(self):
        self.assertIn(
            (_TARGET, hook_mod.append_batch_stats_to_step_span, HookType.AFTER),
            HOOKS,
        )

    def test_decode_gate_envs_do_not_key_compile_cache(self):
        for env_name in (
            "SGLANG_TPU_PROFILE_AUTO_DECODE_STEPS_TO_SKIP",
            "SGLANG_TPU_PROFILE_AUTO_DECODE_KV_LEN_THRESHOLD",
        ):
            self.assertIn(env_name, compile_cache._ENV_FIELDS_NOT_KEYED)

    def test_append_span_metadata_deduplicates_existing_keys(self):
        # Composes cleanly when an earlier hook already emitted num_reqs & request_ids.
        base = "step[DECODE bs=2]#num_reqs=2,request_id1=r1,request_id2=r2#"
        merged = profile_batch_stats.append_span_metadata(
            base, {"num_reqs": 2, "phase": "DECODE_ONLY", "batch_id": 19}
        )
        self.assertEqual(
            merged,
            "step[DECODE bs=2]#num_reqs=2,request_id1=r1,request_id2=r2,phase=DECODE_ONLY,batch_id=19#",
        )
        self.assertEqual(merged.count("num_reqs="), 1)
        self.assertEqual(
            profile_batch_stats.append_span_metadata(
                "step[DECODE bs=1]##", {"phase": "MIXED"}
            ),
            "step[DECODE bs=1]#phase=MIXED#",
        )

    def test_batch_composition_stats_prefill_decode_mixed_and_padding(self):
        policy = types.SimpleNamespace(
            t_bucketer=PowerBucketer("t", min_val=16, max_val=256),
            b_bucketer=PowerBucketer("b", min_val=1, max_val=64),
            extend_bs=8,
        )
        profile_batch_stats.set_target_compile_policy(policy)

        prefill_batch = types.SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            batch_size=2,
            extend_num_tokens=20,
            extend_prefix_lens_cpu=[4, 12],
            extend_seq_lens_cpu=[10, 10],
            spec_info=None,
        )
        prefill_stats = profile_batch_stats.get_batch_composition_stats(
            prefill_batch, batch_id=1
        )
        self.assertEqual(prefill_stats["phase"], "PREFILL_ONLY")
        self.assertEqual(prefill_stats["num_prefill_tokens"], 20)
        self.assertEqual(prefill_stats["num_decode_tokens"], 0)
        self.assertEqual(prefill_stats["total_num_scheduled_tokens"], 20)
        self.assertEqual(prefill_stats["padded_total_num_scheduled_tokens"], 32)
        self.assertEqual(prefill_stats["min_kv_len"], 14)

        decode_batch = types.SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            batch_size=3,
            seq_lens_cpu=torch.tensor([32, 16, 48]),
            spec_info=None,
        )
        decode_stats = profile_batch_stats.get_batch_composition_stats(
            decode_batch, batch_id=2
        )
        self.assertEqual(decode_stats["phase"], "DECODE_ONLY")
        self.assertEqual(decode_stats["num_decode_tokens"], 3)
        self.assertEqual(decode_stats["padded_total_num_scheduled_tokens"], 4)
        self.assertEqual(decode_stats["min_kv_len"], 16)

        mixed_batch = types.SimpleNamespace(
            forward_mode=ForwardMode.MIXED,
            batch_size=3,
            # On MIXED ScheduleBatch, seq_lens_cpu only covers prefill reqs,
            # while prefix_lens / extend_lens cover prefill + running decode reqs.
            seq_lens_cpu=torch.tensor([100]),
            prefix_lens=torch.tensor([90, 5, 20]),
            extend_lens=torch.tensor([10, 1, 1]),
            spec_info=None,
        )
        mixed_stats = profile_batch_stats.get_batch_composition_stats(
            mixed_batch, batch_id=3
        )
        self.assertEqual(mixed_stats["phase"], "MIXED")
        self.assertEqual(mixed_stats["num_prefill_tokens"], 10)
        self.assertEqual(mixed_stats["num_decode_tokens"], 2)
        self.assertEqual(mixed_stats["total_num_scheduled_tokens"], 12)
        self.assertEqual(mixed_stats["padded_total_num_scheduled_tokens"], 16)
        self.assertEqual(mixed_stats["min_kv_len"], 6)

    def test_non_cpu_tensor_is_ignored_without_d2h(self):
        meta_tensor = torch.empty(2, device="meta")
        batch = types.SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            batch_size=2,
            seq_lens_cpu=meta_tensor,
        )
        stats = profile_batch_stats.get_batch_composition_stats(batch, batch_id=5)
        self.assertEqual(stats["min_kv_len"], 0)
        self.assertEqual(stats["phase"], "DECODE_ONLY")

    def test_step_span_records_batch_stats_with_cpu_profiler(self):
        policy = types.SimpleNamespace(
            t_bucketer=PowerBucketer("t", min_val=16, max_val=256),
            b_bucketer=PowerBucketer("b", min_val=1, max_val=64),
            extend_bs=8,
        )
        profile_batch_stats.set_target_compile_policy(policy)

        fake_importer = types.ModuleType("_test_fake_batch_stats_importer")
        fake_importer.build_step_span_name = profile_utils_mod.build_step_span_name

        decode_batch = types.SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            batch_size=3,
            seq_lens_cpu=torch.tensor([32, 16, 48]),
            spec_info=None,
        )

        with (
            patch.dict(sys.modules, {fake_importer.__name__: fake_importer}),
            _applied_step_span_hook(fake_importer),
        ):
            # Off-profile fast path leaves step[...] untouched.
            self.assertEqual(
                fake_importer.build_step_span_name(
                    decode_batch, detailed_annotations=False, forward_id=7
                ),
                "step[DECODE bs=3]",
            )

            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU]
            ) as prof:
                # Auto-profile batch_id handoff takes priority over forward_id.
                profile_batch_stats.set_active_batch_id(19)
                # Draft worker must not consume the target batch_id or annotate.
                with profile_range(
                    fake_importer.build_step_span_name(
                        decode_batch,
                        detailed_annotations=False,
                        is_draft_worker=True,
                        forward_id=99,
                    )
                ):
                    pass
                with profile_range(
                    fake_importer.build_step_span_name(
                        decode_batch,
                        detailed_annotations=False,
                        forward_id=99,
                    )
                ):
                    pass
                # Manual /start_profile (no active handoff) falls back to forward_id.
                with profile_range(
                    fake_importer.build_step_span_name(
                        decode_batch,
                        detailed_annotations=False,
                        forward_id=42,
                    )
                ):
                    pass

        step_keys = sorted(
            evt.key for evt in prof.key_averages() if evt.key.startswith("step[")
        )
        self.assertEqual(
            step_keys,
            sorted(
                [
                    "step[DRAFT bs=3]",
                    "step[DECODE bs=3]#num_reqs=3,num_prefill_tokens=0,num_decode_tokens=3,phase=DECODE_ONLY,batch_id=19,total_num_scheduled_tokens=3,padded_total_num_scheduled_tokens=4,min_kv_length=16#",
                    "step[DECODE bs=3]#num_reqs=3,num_prefill_tokens=0,num_decode_tokens=3,phase=DECODE_ONLY,batch_id=42,total_num_scheduled_tokens=3,padded_total_num_scheduled_tokens=4,min_kv_length=16#",
                ]
            ),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
