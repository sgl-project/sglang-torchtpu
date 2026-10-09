"""Only backends that write native TPU checkpoints may enable prefix reuse."""

import contextlib
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

from sglang.srt.arg_groups import mamba_hook, overrides
from sglang.srt.mem_cache import kv_cache_builder as builder
from sglang.srt.platforms import current_platform
from sglang.srt.platforms.interface import SRTPlatform
from sglang_tpu.tpu_platform import TpuSRTPlatform


class TestTpuPrefixCachePolicy(unittest.TestCase):
    def test_lazy_strategy_is_rejected_only_on_tpu(self):
        for tpu in (True, False):
            for strategy in ("extra_buffer", "extra_buffer_lazy"):
                with self.subTest(tpu=tpu, strategy=strategy):
                    platform = SimpleNamespace(
                        is_tpu=tpu,
                        is_cuda=not tpu,
                        is_musa=False,
                        is_npu=False,
                        is_hip=False,
                        is_xpu=False,
                    )
                    view = SimpleNamespace(
                        mamba_radix_cache_strategy=strategy,
                        enable_hierarchical_cache=False,
                        enable_int8_mamba_checkpoint=False,
                        linear_attn_backend="triton",
                        disaggregation_mode="null",
                        speculative_num_draft_tokens=None,
                        page_size=64,
                        mamba_track_interval=64,
                        chunked_prefill_size=256,
                    )
                    config = SimpleNamespace(architectures=["Qwen3NextForCausalLM"])
                    with (
                        patch.object(mamba_hook, "get_platform", return_value=platform),
                        patch.object(
                            mamba_hook,
                            "current_platform",
                            TpuSRTPlatform() if tpu else SRTPlatform(),
                        ),
                        patch.object(overrides, "get_platform", return_value=platform),
                        self.assertRaisesRegex(
                            ValueError,
                            "extra_buffer_lazy.*TPU.*"
                            "--mamba-radix-cache-strategy extra_buffer",
                        )
                        if tpu and strategy == "extra_buffer_lazy"
                        else contextlib.nullcontext(),
                    ):
                        mamba_hook.validate_mamba_extra_buffer(
                            view, config, mamba_cache_chunk_size_of=lambda: 64
                        )

    def test_backend_guard_preserves_other_devices_and_explicit_disable(self):
        for tpu, recurrent, gdn, disabled, expected in (
            (True, True, False, False, True),  # KDA / other recurrent backends
            (True, True, True, False, False),
            (True, True, True, True, True),
            (True, False, False, False, False),  # Ordinary attention
            (False, True, False, False, False),  # GPU KDA remains supported
        ):
            with self.subTest(tpu=tpu, recurrent=recurrent, gdn=gdn, disabled=disabled):
                worker = MagicMock()
                worker.is_hybrid_swa = False
                worker.get_memory_pool.return_value = (MagicMock(), MagicMock())
                model = SimpleNamespace(is_multimodal=False, hf_config=object())
                memory = MagicMock(disable_radix_cache=disabled)
                parallel = MagicMock(dcp_enabled=False, enable_dp_attention=False)
                with contextlib.ExitStack() as stack:
                    for name, value in {
                        "get_memory": memory,
                        "get_parallel": parallel,
                        "get_dp_tp_group": MagicMock(),
                        "get_exec": MagicMock(),
                        "get_disagg": SimpleNamespace(
                            disaggregation_decode_enable_radix_cache=False
                        ),
                        "get_schedule": SimpleNamespace(chunked_prefill_size=256),
                        "get_resolved_model_impl": builder.ModelImpl.SGLANG,
                        "uses_ssm_state": recurrent,
                        "hybrid_gdn_config": object() if gdn else None,
                        "is_deepseek_dsa": False,
                        "resolve_decode_retraction_backup": "cpu_tensor",
                    }.items():
                        stack.enter_context(
                            patch.object(builder, name, return_value=value)
                        )
                    stack.enter_context(
                        patch.object(current_platform, "is_tpu", return_value=tpu)
                    )
                    stack.enter_context(
                        patch.object(
                            builder,
                            "auto_size_hicache",
                            return_value=contextlib.nullcontext(),
                        )
                    )
                    create = stack.enter_context(
                        patch.object(builder, "create_tree_cache")
                    )
                    stack.enter_context(
                        patch.object(builder, "init_mm_embedding_cache")
                    )
                    result = builder.build_kv_cache(
                        server_args=MagicMock(),
                        model_config=model,
                        tp_worker=worker,
                        page_size=256,
                        spec_algorithm=MagicMock(),
                        enable_metrics=False,
                        enable_kv_cache_events=False,
                        enable_hierarchical_cache=False,
                    )
                self.assertEqual(result.disable_radix_cache, expected)
                self.assertEqual(create.call_args.args[0].params.disable, expected)


if __name__ == "__main__":
    unittest.main()
