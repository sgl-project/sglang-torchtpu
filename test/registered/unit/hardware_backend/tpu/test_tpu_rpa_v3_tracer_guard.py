"""apply_server_args_defaults resolves the compile tracer for RPA v3 at wide TP.

Under the Dynamo tracer the compiled extend program returns a wrong first token
for prefill chunks longer than one RPA v3 KV tile when the attention TP group
has four or more ranks (sgl-project/sglang-torchtpu#1700). The platform
resolves such configurations to the direct tracer unless a Dynamo-only path
is in use or SGLANG_TPU_FORCE_DYNAMO_TRACER=1.

Mocks torch_tpu like test_tpu_platform_unit.py, so it runs on CPU runners.
"""

from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups.overrides import resolution_result
from sglang.srt.model_executor.cuda_graph_config import default_cuda_graph_config
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.envs import envs as tpu_envs
from sglang_tpu.tpu_platform import TpuSRTPlatform

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _args(**overrides):
    # Platform defaults run after handle_deprecated_dp_attention, so they only
    # ever see attn_dp_size=N (test_tpu_platform_unit.py explains why).
    assert not overrides.get("enable_dp_attention"), "pass attn_dp_size=N"
    defaults = dict(
        dtype=None,
        attention_backend=None,
        sampling_backend=None,
        disable_cuda_graph=False,
        cuda_graph_config=default_cuda_graph_config(),
        enable_torch_compile=True,
        disable_overlap_schedule=False,
        mem_fraction_static=None,
        tp_size=8,
        pp_size=1,
        dp_size=1,
        attn_dp_size=1,
        enable_dp_attention=False,
        enable_dp_lm_head=False,
        enable_dynamic_chunking=False,
        page_size=None,
        kv_cache_dtype="auto",
        use_ray=False,
        hicache_io_backend=None,
        max_total_tokens=None,
        speculative_algorithm=None,
        speculative_eagle_topk=None,
        speculative_use_rejection_sampling=False,
        disaggregation_mode="null",
        disaggregation_transfer_backend="mooncake",
        optimistic_prefill_attempts=0,
        max_queued_requests=None,
        enable_priority_scheduling=False,
        disaggregation_enable_kv_checksum=False,
        dcp_size=1,
        attn_cp_size=1,
        enable_prefill_cp=False,
        enable_hierarchical_cache=False,
        tpu_compile_tracer="dynamo",
        unsafe_save_torch_compile_bundle=None,
        unsafe_load_torch_compile_bundle=None,
        json_model_override_args=None,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _apply(args):
    with patch("sglang_tpu.patches._prepare_tpu_env"):
        TpuSRTPlatform().apply_server_args_defaults(args)
    return resolution_result(args, "tpu_compile_tracer")


class TestRpaV3DirectTracerGuard(CustomTestCase):
    def test_rpa_v3_at_tp8_resolves_to_the_direct_tracer(self):
        args = _args()
        self.assertEqual(_apply(args), "direct")
        # The default backend was resolved first; the guard keyed on it.
        self.assertEqual(
            resolution_result(args, "attention_backend"), "pallas_tpu_rpa_v3"
        )

    def test_explicit_rpa_v3_at_tp4_resolves_to_the_direct_tracer(self):
        args = _args(tp_size=4, attention_backend="pallas_tpu_rpa_v3")
        self.assertEqual(_apply(args), "direct")

    def test_narrow_attention_tp_keeps_dynamo(self):
        for overrides in (
            dict(tp_size=1),
            dict(tp_size=2),
            # attn_dp_size=8 leaves one rank per attention TP group.
            dict(tp_size=8, attn_dp_size=8),
        ):
            with self.subTest(**overrides):
                self.assertEqual(_apply(_args(**overrides)), "dynamo")

    def test_other_backends_keep_dynamo(self):
        args = _args(attention_backend="pallas_tpu_rpa_batched")
        self.assertEqual(_apply(args), "dynamo")

    def test_eager_keeps_dynamo(self):
        # Without torch.compile there is no compiled extend program to guard.
        self.assertEqual(_apply(_args(enable_torch_compile=False)), "dynamo")

    def test_explicit_direct_is_left_alone(self):
        args = _args(tpu_compile_tracer="direct")
        self.assertEqual(_apply(args), "direct")
        sources = [s for s, d in args._resolved_overrides if "tpu_compile_tracer" in d]
        self.assertEqual(sources, [], "no declaration needed for an explicit direct")

    def test_dynamo_only_paths_keep_dynamo(self):
        # Speculative algorithms without direct-compile support, and
        # torch.compile bundles, are rejected by handle_tpu_compile_tracer
        # under the direct tracer; the guard leaves them on Dynamo.
        for overrides in (
            dict(speculative_algorithm="EAGLE"),
            dict(unsafe_load_torch_compile_bundle="/tmp/bundle"),
            dict(unsafe_save_torch_compile_bundle="/tmp/bundle"),
        ):
            with self.subTest(**overrides):
                self.assertEqual(_apply(_args(**overrides)), "dynamo")

    def test_direct_capable_speculative_algorithm_switches(self):
        self.assertEqual(_apply(_args(speculative_algorithm="EAGLE3")), "direct")

    def test_force_dynamo_env_keeps_dynamo(self):
        with tpu_envs.SGLANG_TPU_FORCE_DYNAMO_TRACER.override(True):
            self.assertEqual(_apply(_args()), "dynamo")


if __name__ == "__main__":
    import unittest

    unittest.main()
