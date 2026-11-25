"""CPU tests for the request-ID step-span hook."""

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
from sglang.srt.utils.nvtx_utils import profile_range
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.hooks import HOOKS
from sglang_tpu.hooks import step_span_request_ids as hook_mod

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
            _TARGET, hook_mod.append_request_ids_to_step_span, HookType.AFTER
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


class TestStepSpanRequestIdsHook(CustomTestCase):
    def test_hook_registered_in_tpu_hooks(self):
        self.assertIn(
            (_TARGET, hook_mod.append_request_ids_to_step_span, HookType.AFTER),
            HOOKS,
        )

    def test_clean_and_extract_request_ids(self):
        self.assertEqual(hook_mod.clean_req_id("chatcmpl-abc_0"), "chatcmpl-abc_0")
        self.assertEqual(hook_mod.clean_req_id("a,b#c=d e\tf"), "a_b_c_d_e_f")
        self.assertEqual(hook_mod.clean_req_id("   "), "")

        self.assertEqual(hook_mod.extract_request_ids_for_tracing(None), {})
        self.assertEqual(hook_mod.extract_request_ids_for_tracing([]), {})
        self.assertEqual(
            hook_mod.extract_request_ids_for_tracing(
                ["req-1_0", None, "  ", "req,2#3=4"]
            ),
            {"request_id1": "req-1_0", "request_id2": "req_2_3_4"},
        )

    def test_append_span_metadata(self):
        self.assertEqual(
            hook_mod.append_span_metadata("step[DECODE]", {}), "step[DECODE]"
        )
        self.assertEqual(
            hook_mod.append_span_metadata(
                "step[EXTEND bs=1 toks=4]", {"num_reqs": 1, "request_id1": "r1"}
            ),
            "step[EXTEND bs=1 toks=4]#num_reqs=1,request_id1=r1#",
        )
        self.assertEqual(
            hook_mod.append_span_metadata(
                "step[DECODE bs=2]#num_reqs=2,request_id1=r_1,request_id2=r_2#",
                {"phase": "DECODE_ONLY"},
            ),
            "step[DECODE bs=2]#num_reqs=2,request_id1=r_1,request_id2=r_2,phase=DECODE_ONLY#",
        )
        self.assertEqual(
            hook_mod.append_span_metadata("step[DECODE bs=1]##", {"k": "a,b#c"}),
            "step[DECODE bs=1]#k=a_b_c#",
        )
        self.assertEqual(
            hook_mod.append_span_metadata(
                "step[DECODE bs=1]", {"k": "pre_cleaned"}, sanitize=False
            ),
            "step[DECODE bs=1]#k=pre_cleaned#",
        )

    def test_step_span_hook_sanitizes_once_per_rid(self):
        batch = types.SimpleNamespace(rids=["r1", "r2"])
        with (
            patch.object(torch.autograd, "_profiler_enabled", return_value=True),
            patch.object(
                hook_mod, "clean_req_id", wraps=hook_mod.clean_req_id
            ) as spy_clean,
        ):
            out = hook_mod.append_request_ids_to_step_span("step[DECODE bs=2]", batch)
        self.assertEqual(
            out, "step[DECODE bs=2]#num_reqs=2,request_id1=r1,request_id2=r2#"
        )
        self.assertEqual(spy_clean.call_count, 2)

    def test_step_span_records_request_ids_with_cpu_profiler(self):
        # Simulate an importer that did ``from ...profile_utils import build_step_span_name``
        # before HookRegistry.apply_hooks() ran, verifying _propagate_patch updates it too.
        fake_importer = types.ModuleType("_test_fake_step_span_importer")
        fake_importer.build_step_span_name = profile_utils_mod.build_step_span_name

        ext_batch = types.SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            batch_size=2,
            extend_num_tokens=32,
            rids=["chatcmpl-111_0", "chatcmpl-111_1"],
        )
        dec_batch = types.SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            batch_size=1,
            seq_lens_cpu=torch.tensor([64]),
            spec_info=None,
            rids=["req,with#delim=1"],
        )
        none_rids_batch = types.SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            batch_size=1,
            rids=None,
        )
        no_rids_attr_batch = types.SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            batch_size=2,
        )

        with (
            patch.dict(sys.modules, {fake_importer.__name__: fake_importer}),
            _applied_step_span_hook(fake_importer),
        ):
            # Off-profile fast path leaves step[...] untouched on both bindings.
            self.assertEqual(
                profile_utils_mod.build_step_span_name(
                    ext_batch, detailed_annotations=False
                ),
                "step[EXTEND bs=2 toks=32]",
            )
            self.assertEqual(
                fake_importer.build_step_span_name(
                    ext_batch, detailed_annotations=False
                ),
                "step[EXTEND bs=2 toks=32]",
            )

            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU]
            ) as prof:
                with profile_range(
                    fake_importer.build_step_span_name(
                        ext_batch, detailed_annotations=False
                    )
                ):
                    pass
                with profile_range(
                    fake_importer.build_step_span_name(
                        dec_batch, detailed_annotations=True, forward_id=7
                    )
                ):
                    pass
                with profile_range(
                    fake_importer.build_step_span_name(
                        ext_batch,
                        detailed_annotations=False,
                        is_draft_worker=True,
                    )
                ):
                    pass
                with profile_range(
                    fake_importer.build_step_span_name(
                        none_rids_batch, detailed_annotations=False
                    )
                ):
                    pass
                with profile_range(
                    fake_importer.build_step_span_name(
                        no_rids_attr_batch, detailed_annotations=False
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
                    "step[EXTEND bs=2 toks=32]#num_reqs=2,request_id1=chatcmpl-111_0,request_id2=chatcmpl-111_1#",
                    "step[DECODE bs=1 g_sq=1 g_sqsq=1 g_sqsk=64 g_sk=64 forward_id=7]#num_reqs=1,request_id1=req_with_delim_1#",
                    "step[DRAFT bs=2 toks=32]",
                    "step[DECODE bs=1]",
                    "step[DECODE bs=2]",
                ]
            ),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
