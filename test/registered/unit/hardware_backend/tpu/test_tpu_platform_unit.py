"""Unit tests for TpuSRTPlatform.

These tests mock torch_tpu so they can run on any machine without TPU hardware.
"""

import builtins
import os
import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch
import torch.compiler.config as compiler_config

from sglang.srt.environ import envs
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    default_cuda_graph_config,
)
from sglang.srt.platforms.device_mixin import PlatformEnum
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.kv_pool import TpuFusedMHATokenToKVPool
from sglang_tpu.tpu_platform import (
    TpuSRTPlatform,
    _validate_decode_context_parallel,
)

# CPU suite: mocks torch_tpu, so it runs on the CPU runners -- never a TPU node.
register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestTpuPlatformIdentity(CustomTestCase):
    """Platform identity and capability flags."""

    def setUp(self):
        self.platform = TpuSRTPlatform()

    def test_enum_is_tpu(self):
        self.assertEqual(self.platform._enum, PlatformEnum.TPU)

    def test_device_name(self):
        self.assertEqual(self.platform.device_name, "tpu")
        self.assertEqual(self.platform.device_type, "tpu")

    def test_is_tpu(self):
        self.assertTrue(self.platform.is_tpu())
        self.assertFalse(self.platform.is_cuda())
        self.assertFalse(self.platform.is_npu())

    def test_supports_fp8_false(self):
        self.assertFalse(self.platform.supports_fp8())

    def test_pin_memory_unavailable(self):
        self.assertFalse(self.platform.is_pin_memory_available())

    def test_cuda_graph_disabled(self):
        self.assertFalse(self.platform.support_cuda_graph())
        self.assertFalse(self.platform.support_piecewise_cuda_graph())

    def test_decode_context_parallel_shards_every_kv_pool(self):
        self.assertTrue(self.platform.supports_decode_context_parallel())
        self.assertTrue(self.platform.dcp_shards_every_kv_pool())

    def test_dispatch_key(self):
        self.assertEqual(self.platform.get_dispatch_key_name(), "tpu")

    def test_compile_backend(self):
        self.assertEqual(self.platform.get_compile_backend(), "tpu")

    def test_distributed_backend(self):
        self.assertEqual(self.platform.get_torch_distributed_backend_str(), "tpu_dist")

    def test_default_attention_backend(self):
        self.assertEqual(
            self.platform.get_default_attention_backend(), "pallas_tpu_rpa_v3"
        )


class TestTpuPlatformInitialization(CustomTestCase):
    def setUp(self):
        self.platform = TpuSRTPlatform()

    def test_rejects_backend_initialization_without_worker_world_size(self):
        with (
            patch.dict(os.environ, {}, clear=False),
            patch("sglang_tpu.patches._prepare_tpu_env") as prepare_tpu_env,
            patch("sglang_tpu.patches.apply_tpu_patches") as apply_tpu_patches,
        ):
            os.environ.pop("WORLD_SIZE", None)
            with self.assertRaisesRegex(RuntimeError, "requires worker WORLD_SIZE"):
                self.platform.init_backend()

        prepare_tpu_env.assert_not_called()
        apply_tpu_patches.assert_not_called()

    def test_worker_world_size_prepares_environment_before_runtime_patches(self):
        calls = []
        fake_torch_tpu = ModuleType("torch_tpu")

        def prepare(world_size, **kwargs):
            calls.append(("prepare", world_size, kwargs))
            os.environ["LIBTPU_INIT_ARGS"] = "--xla_user=1 --xla_override=0"

        with (
            patch.dict(
                os.environ,
                {"WORLD_SIZE": "16", "LIBTPU_INIT_ARGS": "--xla_user=1"},
                clear=False,
            ),
            patch.dict(sys.modules, {"torch_tpu": fake_torch_tpu}),
            compiler_config.patch(cache_key_tag=""),
            patch(
                "sglang_tpu.patches._prepare_tpu_env",
                side_effect=prepare,
            ),
            patch(
                "sglang_tpu.compile_cache.add_tpu_kernel_inputs_to_compile_cache_key",
                side_effect=lambda: calls.append(
                    ("cache_key", os.environ["LIBTPU_INIT_ARGS"])
                ),
            ),
            patch(
                "sglang_tpu.patches.apply_tpu_patches",
                side_effect=lambda: calls.append(("patch", None)),
            ),
            patch(
                "sglang_tpu.tpu_platform.rank_slice._load",
                side_effect=lambda: calls.append(("rank_slice", None)),
            ),
            patch(
                "sglang_tpu.tpu_platform.traced_rank._load",
                side_effect=lambda: calls.append(("traced_rank", None)),
            ),
        ):
            self.platform.init_backend()

        # The compile cache key must see the flags _prepare_tpu_env appends.
        self.assertEqual(
            calls,
            [
                (
                    "prepare",
                    16,
                    {"resolve_single_host_topology": True},
                ),
                ("cache_key", "--xla_user=1 --xla_override=0"),
                ("patch", None),
                ("rank_slice", None),
                ("traced_rank", None),
            ],
        )

    def test_reentry_during_model_registration_registers_once(self):
        """init_backend re-entered while registering models (model_runner's OOT import
        hook does this) must not register the TPU models a second time."""
        registered = []

        def register(package_name):
            registered.append(package_name)
            if len(registered) == 1:
                self.platform.init_backend()

        with (
            patch.dict(os.environ, {"WORLD_SIZE": "1"}, clear=False),
            patch.dict(sys.modules, {"torch_tpu": ModuleType("torch_tpu")}),
            patch("sglang_tpu.patches._prepare_tpu_env"),
            patch(
                "sglang_tpu.compile_cache.add_tpu_kernel_inputs_to_compile_cache_key"
            ),
            patch("sglang_tpu.patches.apply_tpu_patches"),
            patch("sglang_tpu.tpu_platform.rank_slice._load"),
            patch("sglang_tpu.tpu_platform.traced_rank._load"),
            patch(
                "sglang.srt.models.registry.ModelRegistry.register",
                side_effect=register,
            ),
        ):
            self.platform.init_backend()

        self.assertEqual(registered, ["sglang_tpu.models"])


class TestTpuCompileCache(CustomTestCase):
    def test_reports_tier1_cache_misses(self):
        with patch.object(
            torch,
            "tpu",
            SimpleNamespace(_get_cache_misses=lambda: 37),
            create=True,
        ):
            self.assertEqual(TpuSRTPlatform().get_compilation_cache_misses(), 37)

    def test_omits_cache_misses_when_runtime_has_no_counter(self):
        with patch.object(torch, "tpu", SimpleNamespace(), create=True):
            self.assertIsNone(TpuSRTPlatform().get_compilation_cache_misses())


class TestTpuPlatformMemory(CustomTestCase):
    """Device memory methods delegate to torch.accelerator.get_memory_info.

    The per-process HBM limit is read from the runtime (``(free, limit)``), not
    a static per-device-name table, so these mock get_memory_info + get_device
    to run without a real TPU.
    """

    def setUp(self):
        self.platform = TpuSRTPlatform()
        self.platform._backend_initialized = True

    def _patch_mem(self, free, limit):
        # get_device() builds torch.device("tpu", ...) which is invalid off-TPU;
        # stub it, and stub the runtime memory query to (free, limit).
        return (
            patch.object(self.platform, "get_device", return_value=None),
            patch("torch.accelerator.get_memory_info", return_value=(free, limit)),
        )

    def test_uninitialized_returns_zero_without_calling_runtime(self):
        uninit_platform = TpuSRTPlatform()
        with patch("torch.accelerator.get_memory_info") as mock_mem:
            self.assertEqual(uninit_platform.get_device_total_memory(), 0)
            self.assertEqual(uninit_platform.get_current_memory_usage(), 0.0)
            mock_mem.assert_not_called()

    def test_total_memory_returns_reported_limit(self):
        dev, mem = self._patch_mem(10 << 30, 96 << 30)
        with dev, mem:
            self.assertEqual(self.platform.get_device_total_memory(), 96 << 30)

    def test_current_memory_usage_is_total_minus_free(self):
        dev, mem = self._patch_mem(10 << 30, 96 << 30)
        with dev, mem:
            self.assertEqual(self.platform.get_current_memory_usage(), float(86 << 30))

    def test_available_memory_returns_free_and_total(self):
        dev, mem = self._patch_mem(10 << 30, 96 << 30)
        with dev, mem:
            free, total = self.platform.get_available_memory()
            self.assertEqual(total, 96 << 30)
            self.assertEqual(free, 10 << 30)


def _require_resolved_dp_layout(overrides):
    """Platform defaults only ever see the resolved DP layout.

    handle_deprecated_dp_attention rewrites `--dp-size N --enable-dp-attention`
    to attn_dp_size=N, dp_size=1, enable_dp_attention=False before platform
    defaults run. Handing the hook the deprecated spelling tests a state
    production never reaches: these cases stayed green while the DP-attention
    refusals they cover had stopped firing.
    """
    if overrides.get("enable_dp_attention"):
        raise AssertionError(
            "pass attn_dp_size=N: the pipeline resolves enable_dp_attention "
            "before platform defaults run"
        )


class TestTpuPlatformServerDefaults(CustomTestCase):
    """apply_server_args_defaults sets correct values."""

    def _args(self, **overrides):
        _require_resolved_dp_layout(overrides)
        # apply_server_args_defaults declares rather than writes, so the assertions read the
        # decisions back with resolution_result, as the projection does.
        defaults = dict(
            dtype=None,
            attention_backend=None,
            sampling_backend=None,
            disable_cuda_graph=False,
            cuda_graph_config=default_cuda_graph_config(),
            enable_torch_compile=False,
            disable_overlap_schedule=False,
            mem_fraction_static=None,
            tp_size=1,
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
            json_model_override_args=None,
            enable_unified_memory=False,
            enable_page_major_kv_layout=False,
        )
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    def _apply(self, args):
        platform = TpuSRTPlatform()
        with patch("sglang_tpu.patches._prepare_tpu_env"):
            platform.apply_server_args_defaults(args)

    def _resolved(self, args, field):
        """What resolution decided for ``field``, else what the caller supplied."""
        from sglang.srt.arg_groups.overrides import resolution_result

        return resolution_result(args, field)

    def test_rejects_optimistic_prefill_on_raiden_pd(self):
        """Optimistic prefill on raiden parks KV for requests the decode has
        no pages for, and deadlocked the PD pools under burst load."""
        for mode in ("prefill", "decode"):
            with self.subTest(mode=mode):
                args = self._args(
                    disaggregation_mode=mode,
                    disaggregation_transfer_backend="raiden",
                    optimistic_prefill_attempts=1,
                )
                with self.assertRaisesRegex(
                    ValueError, "--optimistic-prefill-attempts must be 0"
                ):
                    self._apply(args)

    def test_rejects_max_queued_requests_under_pd(self):
        """The two sides admit independently, so a queue cap cannot be enforced
        consistently; refuse at start-up instead of silently ignoring it."""
        for mode in ("prefill", "decode"):
            with self.subTest(mode=mode):
                args = self._args(disaggregation_mode=mode, max_queued_requests=4)
                with self.assertRaisesRegex(
                    ValueError, "--max-queued-requests is not supported"
                ):
                    self._apply(args)

    def test_queue_limit_allowed_outside_pd(self):
        for overrides in (
            dict(max_queued_requests=4, enable_priority_scheduling=True),
            dict(max_queued_requests=0),
            dict(disaggregation_mode="decode"),
        ):
            with self.subTest(**overrides):
                self._apply(self._args(**overrides))

    def test_optimistic_prefill_allowed_outside_raiden_pd(self):
        for overrides in (
            dict(disaggregation_mode="prefill", optimistic_prefill_attempts=1),
            dict(
                disaggregation_mode="prefill",
                disaggregation_transfer_backend="raiden",
                optimistic_prefill_attempts=0,
                max_queued_requests=None,
                enable_priority_scheduling=False,
            ),
            dict(
                disaggregation_transfer_backend="raiden",
                optimistic_prefill_attempts=1,
            ),
        ):
            with self.subTest(**overrides):
                self._apply(self._args(**overrides))

    def test_rejects_kv_checksum(self):
        """The KV checksum is CUDA-only. On TPU the prefill passed its health
        checks and then every rank died on the first real request (#1177)."""
        for mode in ("prefill", "decode", "null"):
            with self.subTest(mode=mode):
                args = self._args(
                    disaggregation_mode=mode,
                    disaggregation_transfer_backend="raiden",
                    disaggregation_enable_kv_checksum=True,
                )
                with self.assertRaisesRegex(
                    ValueError,
                    "--disaggregation-enable-kv-checksum is not supported on TPU",
                ):
                    self._apply(args)

    def test_kv_checksum_off_allowed(self):
        self._apply(
            self._args(
                disaggregation_mode="prefill",
                disaggregation_transfer_backend="raiden",
                disaggregation_enable_kv_checksum=False,
            )
        )

    def test_defaults_applied(self):
        args = self._args(enable_torch_compile=False)
        self._apply(args)

        self.assertEqual(self._resolved(args, "dtype"), "bfloat16")
        self.assertEqual(self._resolved(args, "attention_backend"), "pallas_tpu_rpa_v3")
        self.assertEqual(self._resolved(args, "sampling_backend"), "tpu")
        self.assertTrue(self._resolved(args, "disable_cuda_graph"))
        graph_config = self._resolved(args, "cuda_graph_config")
        self.assertEqual(graph_config.decode.backend, Backend.DISABLED)
        self.assertEqual(graph_config.prefill.backend, Backend.DISABLED)
        # enable_torch_compile is opt-in on TPU and must not be forced off.
        self.assertFalse(self._resolved(args, "enable_torch_compile"))
        # Overlap is enabled on TPU via the thread-based path in
        # scheduler_overlap_mixin; the platform must NOT force-disable it.
        self.assertFalse(self._resolved(args, "disable_overlap_schedule"))
        self.assertEqual(self._resolved(args, "mem_fraction_static"), 0.40)
        # Without --enable-unified-memory the static pools stay selected.
        self.assertFalse(self._resolved(args, "tpu_unified_kv_pool"))

    def test_unified_memory_selects_the_tpu_pool_and_clears_the_generic_flag(self):
        # The generic flag's shared code paths build a pool the Pallas kernels
        # cannot read; only the TPU field may stay set past resolution.
        args = self._args(enable_unified_memory=True)
        self._apply(args)
        self.assertFalse(self._resolved(args, "enable_unified_memory"))
        self.assertTrue(self._resolved(args, "tpu_unified_kv_pool"))

    def test_page_major_kv_layout_is_rejected(self):
        args = self._args(enable_unified_memory=True, enable_page_major_kv_layout=True)
        with self.assertRaisesRegex(
            ValueError, "enable-page-major-kv-layout is not supported on TPU"
        ):
            self._apply(args)

    def test_resolved_cuda_graphs_disabled_without_disabling_tpu_compile(self):
        graph_config = default_cuda_graph_config()
        graph_config.decode.backend = Backend.FULL
        graph_config.prefill.backend = Backend.BREAKABLE
        graph_config.decode.bs = [1, 4]
        args = self._args(cuda_graph_config=graph_config, enable_torch_compile=True)
        self._apply(args)

        resolved = self._resolved(args, "cuda_graph_config")
        self.assertEqual(resolved.decode.backend, Backend.DISABLED)
        self.assertEqual(resolved.prefill.backend, Backend.DISABLED)
        self.assertEqual(resolved.decode.bs, [1, 4])
        self.assertTrue(self._resolved(args, "enable_torch_compile"))
        self.assertEqual(graph_config.decode.backend, Backend.FULL)
        self.assertEqual(graph_config.prefill.backend, Backend.BREAKABLE)

    def test_rpa_backends_default_to_fp8_kv_cache(self):
        for backend in ("pallas_tpu_rpa_v3", "pallas_tpu_rpa_batched"):
            with self.subTest(backend=backend):
                args = self._args(attention_backend=backend)
                self._apply(args)
                self.assertEqual(self._resolved(args, "kv_cache_dtype"), "fp8_e4m3")

    def test_explicit_kv_cache_dtype_not_overridden(self):
        args = self._args(
            attention_backend="pallas_tpu_rpa_batched", kv_cache_dtype="bfloat16"
        )
        self._apply(args)
        self.assertEqual(self._resolved(args, "kv_cache_dtype"), "bfloat16")

    def test_non_rpa_backend_keeps_auto_kv_cache_dtype(self):
        # pallas_tpu_mla rejects an fp8 KV cache, so the fp8 default is scoped
        # to the RPA backends and must not leak to it.
        args = self._args(attention_backend="pallas_tpu_mla")
        self._apply(args)
        self.assertEqual(self._resolved(args, "kv_cache_dtype"), "auto")
        self.assertEqual(self._resolved(args, "page_size"), 64)

    def test_sparse_mla_backend_defaults_kv_cache_to_fp8(self):
        # Its kernels read an FP8 cache and nothing else, so "auto" cannot be
        # left to mean the model dtype.
        args = self._args(attention_backend="pallas_tpu_mla_sparse")
        self._apply(args)
        self.assertEqual(self._resolved(args, "kv_cache_dtype"), "fp8_e4m3")

    def test_batched_rpa_page_size_default_meets_the_kernel_floor(self):
        args = self._args(attention_backend="pallas_tpu_rpa_batched")
        self._apply(args)
        self.assertEqual(self._resolved(args, "page_size"), 256)

    def test_batched_rpa_rejects_a_page_size_below_the_floor(self):
        args = self._args(attention_backend="pallas_tpu_rpa_batched", page_size=64)
        with self.assertRaisesRegex(ValueError, "page-size >= 256"):
            self._apply(args)

    def test_explicit_page_size_above_the_floor_is_kept(self):
        args = self._args(attention_backend="pallas_tpu_rpa_batched", page_size=512)
        self._apply(args)
        self.assertEqual(self._resolved(args, "page_size"), 512)

    def test_enable_torch_compile_preserved_when_opted_in(self):
        args = self._args(enable_torch_compile=True)
        self._apply(args)
        self.assertTrue(self._resolved(args, "enable_torch_compile"))

    def test_registered_spec_algorithm_name_is_admitted(self):
        # Plugin registration may happen after platform defaults. Unknown names
        # pass this backend gate and are validated by the shared registry later.
        args = self._args(speculative_algorithm="MY_PLUGIN")
        self._apply(args)

    def test_unsupported_builtin_spec_algorithm_is_rejected(self):
        args = self._args(speculative_algorithm="NGRAM")
        with self.assertRaisesRegex(NotImplementedError, "only EAGLE"):
            self._apply(args)

    def test_dp_attention_requires_dp_lm_head_when_attn_tp_gt_1(self):
        args = self._args(
            tp_size=8,
            attn_dp_size=2,
            enable_dp_lm_head=False,
        )
        with self.assertRaisesRegex(
            ValueError, "When attention TP > 1 under DP attention on TPU"
        ):
            self._apply(args)

    def test_dp_attention_allows_valid_dp_lm_head_combos(self):
        # attn_tp_size > 1 with enable_dp_lm_head=True
        args = self._args(
            tp_size=8,
            attn_dp_size=2,
            enable_dp_lm_head=True,
        )
        self._apply(args)

        # attn_tp_size == 1 with enable_dp_lm_head=False
        args = self._args(
            tp_size=8,
            attn_dp_size=8,
            enable_dp_lm_head=False,
        )
        self._apply(args)

    def test_shared_head_experimental_drafts_need_dp_lm_head_at_attn_tp_1(self):
        """At attention TP 1 the target head shards over the full TP group, but these
        drafts share it inside the attention-TP group: they would draft from one shard."""
        from sglang_tpu.envs import envs as tpu_envs

        for algorithm, enabled, error in (
            ("DFLASH", True, True),
            ("NEXTN", True, True),
            ("EAGLE3", True, False),
            ("DFLASH", False, False),
        ):
            args = self._args(
                tp_size=8,
                attn_dp_size=8,
                enable_dp_lm_head=False,
                speculative_algorithm=algorithm,
            )
            with (
                self.subTest(algorithm=algorithm, enabled=enabled),
                tpu_envs.SGLANG_TPU_ENABLE_EXPERIMENTAL_SPEC_WORKER.override(enabled),
            ):
                if error:
                    with self.assertRaisesRegex(ValueError, "--enable-dp-lm-head"):
                        self._apply(args)
                else:
                    self._apply(args)

    def test_configures_export_mesh_from_tp_size(self):
        args = self._args(tp_size=16)
        platform = TpuSRTPlatform()

        with patch("sglang_tpu.patches._prepare_tpu_env") as prepare_tpu_env:
            platform.apply_server_args_defaults(args)

        prepare_tpu_env.assert_called_once_with(16, setup_slicebuilder=True)

    def test_parent_defaults_do_not_import_torch_tpu(self):
        args = self._args(tp_size=8)
        platform = TpuSRTPlatform()
        imported_modules = []
        real_import = builtins.__import__

        def track_import(name, *args, **kwargs):
            imported_modules.append(name)
            return real_import(name, *args, **kwargs)

        with (
            patch.dict(os.environ, {}, clear=True),
            patch("sglang_tpu.env_override.apply_tpu_env_overrides"),
            patch("builtins.__import__", side_effect=track_import),
        ):
            platform.apply_server_args_defaults(args)

        self.assertFalse(
            any(
                name == "torch_tpu" or name.startswith("torch_tpu.")
                for name in imported_modules
            )
        )

    def test_explicit_dtype_not_overridden(self):
        args = self._args(dtype="float32")
        self._apply(args)
        self.assertEqual(self._resolved(args, "dtype"), "float32")

    def test_explicit_mem_fraction_not_overridden(self):
        args = self._args(mem_fraction_static=0.60)
        self._apply(args)
        self.assertEqual(self._resolved(args, "mem_fraction_static"), 0.60)

    def test_explicit_sampling_backend_not_overridden(self):
        args = self._args(sampling_backend="pytorch")
        self._apply(args)
        self.assertEqual(self._resolved(args, "sampling_backend"), "pytorch")


class TestPrefillContextParallelValidation(CustomTestCase):
    """apply_server_args_defaults gates the prefill context-parallel path."""

    _args = TestTpuPlatformServerDefaults._args
    _apply = TestTpuPlatformServerDefaults._apply
    _resolved = TestTpuPlatformServerDefaults._resolved

    def _apply_cp(self, architectures=("Qwen3MoeForCausalLM",), **overrides):
        fields = dict(
            tp_size=8,
            attn_cp_size=8,
            enable_prefill_cp=True,
            cp_strategy="zigzag",
            moe_dp_size=8,
            enable_torch_compile=True,
            tpu_compile_tracer="dynamo",
            model_path="model",
            trust_remote_code=False,
            revision=None,
        )
        fields.update(overrides)
        args = self._args(**fields)
        hf_config = SimpleNamespace(architectures=list(architectures))
        with patch(
            "sglang.srt.utils.hf_transformers.config.get_config",
            return_value=hf_config,
        ):
            self._apply(args)
        return args

    def test_resolves_the_cp_backend_and_turns_off_sparse_core_gathers(self):
        """On the current libtpu the context-parallel forward returns wrong tokens
        or halts with SparseCore gathers; a user's own libtpu flag value stays."""
        user_flag = "--xla_tpu_offload_gather_to_sparsecore=true"
        env = {
            "SGLANG_TPU_REQ_TO_TOKEN_ON_HOST": "0",
            "SGLANG_TPU_FUSED_MOE_SC_GATHER": "1",
            "LIBTPU_INIT_ARGS": user_flag,
        }
        with patch.dict(os.environ, env):
            args = self._apply_cp()
            self.assertEqual(
                self._resolved(args, "attention_backend"), "pallas_tpu_rpa_batched"
            )
            self.assertEqual(self._resolved(args, "sampling_backend"), "pytorch")
            self.assertTrue(envs.SGLANG_TPU_REQ_TO_TOKEN_ON_HOST.get())
            self.assertFalse(envs.SGLANG_TPU_FUSED_MOE_SC_GATHER.get())
            libtpu_args = os.environ["LIBTPU_INIT_ARGS"].split()
        self.assertIn(user_flag, libtpu_args)
        self.assertNotIn("--xla_tpu_offload_gather_to_sparsecore=false", libtpu_args)
        self.assertIn(
            "--xla_tpu_enable_sparse_core_collective_offload_all_gather=false",
            libtpu_args,
        )

    def test_without_context_parallelism_sparse_core_stays_on(self):
        """The SparseCore workarounds would slow every other server down."""
        with patch.dict(os.environ, {"LIBTPU_INIT_ARGS": ""}):
            os.environ.pop("SGLANG_TPU_FUSED_MOE_SC_GATHER", None)
            self._apply(self._args(tp_size=8))
            self.assertEqual(os.environ["LIBTPU_INIT_ARGS"], "")
            self.assertTrue(envs.SGLANG_TPU_FUSED_MOE_SC_GATHER.get())

    def test_rejections_name_the_missing_piece(self):
        cases = [
            (dict(enable_prefill_cp=False), {}, "requires --enable-prefill-cp"),
            (dict(cp_strategy="interleave"), {}, "requires --cp-strategy zigzag"),
            (
                dict(architectures=("Qwen3VLMoeForConditionalGeneration",)),
                {},
                "supports Qwen3MoeForCausalLM and GptOssForCausalLM",
            ),
            (
                dict(attention_backend="pallas_tpu_rpa_v3"),
                {},
                "requires --attention-backend pallas_tpu_rpa_batched",
            ),
            (
                {},
                {"SGLANG_TPU_USE_SEQ_ALONG_LANE_KV": "1"},
                "unset SGLANG_TPU_USE_SEQ_ALONG_LANE_KV",
            ),
            (
                dict(architectures=("GptOssForCausalLM",), kv_cache_dtype="bfloat16"),
                {},
                "bf16 KV cache fails in warmup for gpt-oss",
            ),
            (dict(speculative_algorithm="EAGLE3"), {}, "speculative decoding"),
            (dict(attn_dp_size=8), {}, "DP attention"),
            (dict(moe_dp_size=1), {}, "requires --moe-dp-size 8"),
            (dict(enable_torch_compile=False), {}, "requires --enable-torch-compile"),
            (dict(tpu_compile_tracer="direct"), {}, "requires --tpu-compile-tracer"),
            (dict(sampling_backend="tpu"), {}, "requires --sampling-backend pytorch"),
            (
                {},
                {"SGLANG_TPU_ENABLE_PROMPT_LOGPROB": "1"},
                r"input \(prompt\) logprobs",
            ),
        ]
        for overrides, env, message in cases:
            with self.subTest(message=message), patch.dict(os.environ, env):
                with self.assertRaisesRegex(NotImplementedError, message):
                    self._apply_cp(**overrides)

    def test_gpt_oss_bf16_kv_runs_at_cp_two(self):
        """The bf16 refusal covers only the CP sizes that fail."""
        self._apply_cp(
            architectures=("GptOssForCausalLM",),
            kv_cache_dtype="bfloat16",
            tp_size=2,
            attn_cp_size=2,
            moe_dp_size=2,
        )


class TestDsaValidation(CustomTestCase):
    """apply_server_args_defaults rejects the DSA setups TPU would serve wrongly."""

    _args = TestTpuPlatformServerDefaults._args
    _apply = TestTpuPlatformServerDefaults._apply

    def _apply_dsa(
        self, *, attention_backend, tpu_compile_tracer="dynamo", **config_fields
    ):
        args = self._args(
            attention_backend=attention_backend,
            tpu_compile_tracer=tpu_compile_tracer,
            model_path="model",
            trust_remote_code=False,
            revision=None,
        )
        hf_config = SimpleNamespace(
            architectures=["GlmMoeDsaForCausalLM"], index_topk=2048, **config_fields
        )
        with patch(
            "sglang.srt.utils.hf_transformers.config.get_config",
            return_value=hf_config,
        ):
            self._apply(args)

    def test_dsa_needs_the_sparse_backend_and_a_plain_top_k(self):
        """TPU has one DSA backend, and its top-k kernel cannot force the
        init/local positions in. Either mismatch must stop the launch: served
        anyway, the model would attend over a different selection."""
        self._apply_dsa(attention_backend="pallas_tpu_mla_sparse")
        with self.assertRaisesRegex(ValueError, "pallas_tpu_mla_sparse"):
            self._apply_dsa(attention_backend="pallas_tpu_mla")
        for field in ("index_init_tokens", "index_local_tokens"):
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, field):
                    self._apply_dsa(
                        attention_backend="pallas_tpu_mla_sparse", **{field: 4}
                    )

    def test_dsa_needs_the_dynamo_tracer(self):
        """The direct tracer reads the KV cache through the pool's get_state,
        which the split MLA pool does not implement. The launch must say so
        instead of failing at the first compile."""
        with self.assertRaisesRegex(ValueError, "tpu-compile-tracer dynamo"):
            self._apply_dsa(
                attention_backend="pallas_tpu_mla_sparse",
                tpu_compile_tracer="direct",
            )


class TestTpuPlatformDecodeContextParallel(CustomTestCase):
    """Decode context parallel is admitted only for the configuration the TPU
    batched RPA backend implements; every other combination is refused by name."""

    def _args(self, **overrides):
        _require_resolved_dp_layout(overrides)
        defaults = dict(
            attention_backend="pallas_tpu_rpa_batched",
            tp_size=8,
            dcp_size=2,
            pp_size=1,
            dp_size=1,
            attn_dp_size=1,
            enable_dp_attention=False,
            enable_dp_lm_head=False,
            attn_cp_size=1,
            enable_prefill_cp=False,
            disaggregation_mode="null",
            disaggregation_transfer_backend="mooncake",
            optimistic_prefill_attempts=0,
            max_queued_requests=None,
            enable_priority_scheduling=False,
            disaggregation_enable_kv_checksum=False,
            enable_hierarchical_cache=False,
            dcp_comm_backend="ag_rs",
            dcp_replicate_q_proj=False,
            enable_torch_compile=True,
            tpu_compile_tracer="dynamo",
            speculative_algorithm=None,
            speculative_eagle_topk=None,
            speculative_use_rejection_sampling=False,
            enable_mixed_chunk=False,
            dtype=None,
            sampling_backend=None,
            disable_cuda_graph=False,
            cuda_graph_config=default_cuda_graph_config(),
            disable_overlap_schedule=False,
            mem_fraction_static=None,
            page_size=256,
            chunked_prefill_size=None,
            kv_cache_dtype="auto",
            use_ray=False,
            hicache_io_backend=None,
            max_total_tokens=None,
            enable_unified_memory=False,
            enable_page_major_kv_layout=False,
        )
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    def _validate(self, args, env=None):
        environment = {"SGLANG_TPU_REQ_TO_TOKEN_ON_HOST": "1"}
        environment.update(env or {})
        with patch.dict(os.environ, environment):
            _validate_decode_context_parallel(args)

    def _assert_refused(self, pattern, **overrides):
        with self.assertRaisesRegex(ValueError, pattern):
            self._validate(self._args(**overrides))

    def test_accepts_the_benchmark_configuration(self):
        # pallas_tpu_rpa_batched, tp 8, dcp 2, no DP attention, torch compile on.
        self._validate(self._args())

    def test_accepts_the_host_req_to_token_default_when_unset(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SGLANG_TPU_REQ_TO_TOKEN_ON_HOST", None)
            _validate_decode_context_parallel(self._args())

    def test_dcp_disabled_admits_every_other_flag(self):
        self._validate(
            self._args(
                dcp_size=1,
                attention_backend="pallas_tpu_rpa_v3",
                pp_size=2,
                attn_dp_size=2,
                enable_torch_compile=False,
            )
        )

    def test_refuses_another_attention_backend(self):
        for backend in ("pallas_tpu_rpa_v3", "pallas_tpu_flash", "torch_native"):
            with self.subTest(backend=backend):
                self._assert_refused(
                    "(?i)attention[-_ ]backend", attention_backend=backend
                )

    def test_refuses_a_tp_size_the_dcp_group_does_not_divide(self):
        self._assert_refused(
            "(?i)tp[-_ ]size|tensor[-_ ]parallel", tp_size=6, dcp_size=4
        )

    def test_refuses_pipeline_parallelism(self):
        self._assert_refused("(?i)pp[-_ ]size|pipeline", pp_size=2)

    def test_refuses_speculative_decoding(self):
        self._assert_refused("(?i)specul", speculative_algorithm="EAGLE3")

    def test_refuses_mixed_chunk(self):
        self._assert_refused("(?i)mixed[-_ ]chunk", enable_mixed_chunk=True)

    def test_refuses_a_quantized_kv_cache(self):
        # Only the bf16 cache has been validated under DCP.
        for dtype in ("fp8_e4m3", "fp8_e5m2", "mxint8"):
            with self.subTest(kv_cache_dtype=dtype):
                self._assert_refused("(?i)kv[-_ ]cache[-_ ]dtype", kv_cache_dtype=dtype)

    def test_refuses_dp_attention(self):
        self._assert_refused("(?i)dp[-_ ]attention", attn_dp_size=2)

    def test_refuses_prefill_context_parallelism(self):
        self._assert_refused(
            "(?i)attn[-_ ]cp[-_ ]size|context[-_ ]parallel", attn_cp_size=2
        )
        self._assert_refused("(?i)prefill[-_ ]cp", enable_prefill_cp=True)

    def test_refuses_pd_disaggregation(self):
        for mode in ("prefill", "decode"):
            with self.subTest(mode=mode):
                self._assert_refused("(?i)disaggregation", disaggregation_mode=mode)

    def test_refuses_hierarchical_cache(self):
        self._assert_refused("(?i)hierarchical", enable_hierarchical_cache=True)

    def test_refuses_another_dcp_communication_backend(self):
        for backend in ("a2a", "fi_a2a"):
            with self.subTest(backend=backend):
                self._assert_refused("(?i)dcp[-_ ]comm", dcp_comm_backend=backend)

    def test_refuses_a_replicated_q_projection(self):
        self._assert_refused("(?i)replicate[-_ ]q", dcp_replicate_q_proj=True)

    def test_refuses_eager_mode(self):
        self._assert_refused("(?i)torch[-_ ]compile", enable_torch_compile=False)

    def test_refuses_the_direct_tracer(self):
        self._assert_refused("(?i)tracer", tpu_compile_tracer="direct")

    def test_refuses_a_device_resident_req_to_token(self):
        with self.assertRaisesRegex(ValueError, "SGLANG_TPU_REQ_TO_TOKEN_ON_HOST"):
            self._validate(self._args(), env={"SGLANG_TPU_REQ_TO_TOKEN_ON_HOST": "0"})

    def test_refuses_the_seq_along_lane_kv_layout(self):
        # The owner-masked KV write addresses the token-major fused cache.
        with self.assertRaisesRegex(ValueError, "SGLANG_TPU_USE_SEQ_ALONG_LANE_KV"):
            self._validate(self._args(), env={"SGLANG_TPU_USE_SEQ_ALONG_LANE_KV": "1"})

    def test_refuses_a_chunk_below_one_dcp_page(self):
        # A DCP page holds page_size * dcp_size tokens and prefill chunks end on
        # page boundaries, so a smaller chunk would leave every request waiting.
        self._assert_refused(
            "(?i)chunked[-_ ]prefill",
            dcp_size=2,
            page_size=256,
            chunked_prefill_size=511,
        )
        for chunk in (512, -1, None):
            with self.subTest(chunked_prefill_size=chunk):
                self._validate(
                    self._args(dcp_size=2, page_size=256, chunked_prefill_size=chunk)
                )
        # An unset chunk resolves to 4096 tokens after this check runs.
        self._assert_refused(
            "(?i)chunked[-_ ]prefill",
            dcp_size=2,
            page_size=4096,
            chunked_prefill_size=None,
        )

    def test_refuses_more_than_two_dcp_ranks(self):
        # DCP4 and DCP8 stall in warmup or return wrong first tokens on the
        # pinned stack; only DCP2 passes the e2e gate.
        for dcp_size in (4, 8):
            with self.subTest(dcp_size=dcp_size):
                self._assert_refused("(?i)must be 1 or 2", dcp_size=dcp_size)

    def test_chunk_check_uses_the_resolved_page_size(self):
        # An unset page resolves to the batched kernel's 256-token floor first.
        args = self._args(dcp_size=8, page_size=None, chunked_prefill_size=1024)
        with (
            patch("sglang_tpu.patches._prepare_tpu_env"),
            patch.dict(os.environ, {"SGLANG_TPU_REQ_TO_TOKEN_ON_HOST": "1"}),
        ):
            with self.assertRaisesRegex(ValueError, r"256 x 8 = 2048"):
                TpuSRTPlatform().apply_server_args_defaults(args)

    def test_unset_backend_resolves_to_batched_rpa(self):
        # The platform default (RPA v3) is refused under DCP, so an unset
        # backend has to resolve to batched RPA and take its page floor.
        from sglang.srt.arg_groups.overrides import resolution_result

        args = self._args(attention_backend=None, page_size=None)
        with (
            patch("sglang_tpu.patches._prepare_tpu_env"),
            patch.dict(os.environ, {"SGLANG_TPU_REQ_TO_TOKEN_ON_HOST": "1"}),
        ):
            TpuSRTPlatform().apply_server_args_defaults(args)
        self.assertEqual(
            resolution_result(args, "attention_backend"), "pallas_tpu_rpa_batched"
        )
        self.assertEqual(resolution_result(args, "page_size"), 256)

    def test_unset_kv_dtype_resolves_to_bf16(self):
        # The RPA backends' fp8 default is refused under DCP.
        from sglang.srt.arg_groups.overrides import resolution_result

        args = self._args()
        with (
            patch("sglang_tpu.patches._prepare_tpu_env"),
            patch.dict(os.environ, {"SGLANG_TPU_REQ_TO_TOKEN_ON_HOST": "1"}),
        ):
            TpuSRTPlatform().apply_server_args_defaults(args)
        self.assertEqual(resolution_result(args, "kv_cache_dtype"), "bfloat16")

    def test_platform_defaults_run_the_validator(self):
        # The gate has to be wired into apply_server_args_defaults, or a bad
        # configuration reaches the model runner.
        platform = TpuSRTPlatform()
        args = self._args(attn_dp_size=2, enable_dp_lm_head=True)
        with (
            patch("sglang_tpu.patches._prepare_tpu_env"),
            patch.dict(os.environ, {"SGLANG_TPU_REQ_TO_TOKEN_ON_HOST": "1"}),
        ):
            with self.assertRaisesRegex(ValueError, "(?i)dp[-_ ]attention"):
                platform.apply_server_args_defaults(args)


class TestTpuPlatformFactoryMethods(CustomTestCase):
    """Factory methods return the expected classes."""

    def setUp(self):
        self.platform = TpuSRTPlatform()

    def test_graph_runner_cls(self):
        cls = self.platform.get_graph_runner_cls()
        self.assertEqual(cls.__name__, "TpuGraphRunner")

    def test_paged_allocator_cls(self):
        cls = self.platform.get_paged_allocator_cls()
        self.assertEqual(cls.__name__, "TpuPagedTokenToKVPoolAllocator")

    def test_mha_kv_pool_cls(self):
        # get_mha_kv_pool_cls reads the exec bag; publish the backend.
        with get_context().override_server_args(attention_backend="torch_native"):
            cls = self.platform.get_mha_kv_pool_cls()
        self.assertEqual(cls.__name__, "MHATokenToKVPool")

        with get_context().override_server_args(attention_backend="pallas_tpu_rpa_v3"):
            cls = self.platform.get_mha_kv_pool_cls()
        self.assertEqual(cls.__name__, "TpuFusedMHATokenToKVPool")

    def test_dsa_pool_is_the_split_pool(self):
        cls = self.platform.get_dsa_kv_pool_cls()
        self.assertEqual(cls.__name__, "TpuSplitMLATokenToKVPool")

    def test_draft_backends_run_their_target_backend_per_step(self):
        from sglang.srt.layers.attention import attention_registry as registry
        from sglang_tpu.attention_pallas_mla import TpuPallasMlaAttnBackend
        from sglang_tpu.attention_pallas_mla_sparse import (
            TpuPallasSparseMlaAttnBackend,
        )
        from sglang_tpu.attention_pallas_rpa_batched import (
            TpuPallasRpaBatchedAttnBackend,
        )
        from sglang_tpu.attention_pallas_rpa_v3 import TpuPallasRpaV3AttnBackend
        from sglang_tpu.tpu_platform import _register_attention_backends

        expected = {
            "pallas_tpu_rpa_v3": TpuPallasRpaV3AttnBackend,
            "pallas_tpu_rpa_batched": TpuPallasRpaBatchedAttnBackend,
            "pallas_tpu_mla": TpuPallasMlaAttnBackend,
            "pallas_tpu_mla_sparse": TpuPallasSparseMlaAttnBackend,
        }
        with (
            patch.dict(registry.ATTENTION_BACKENDS),
            patch.dict(registry.DRAFT_ATTENTION_BACKENDS, clear=True),
        ):
            _register_attention_backends()
            self.assertEqual(set(registry.DRAFT_ATTENTION_BACKENDS), set(expected))
            for name, attn_backend_cls in expected.items():
                with self.subTest(backend=name):
                    # The draft extend builds the target backend of the same name.
                    self.assertIn(name, registry.ATTENTION_BACKENDS)
                    with patch.object(
                        attn_backend_cls, "__init__", return_value=None
                    ) as init:
                        container = registry.DRAFT_ATTENTION_BACKENDS[name](
                            "runner", 1, 4
                        )
                    self.assertEqual(len(container.attn_backends), 3)
                    self.assertTrue(
                        all(
                            type(c) is attn_backend_cls for c in container.attn_backends
                        )
                    )
                    self.assertEqual(
                        [c.kwargs["speculative_step_id"] for c in init.call_args_list],
                        [0, 1, 2],
                    )

    def test_piecewise_backend_raises(self):
        with self.assertRaises(NotImplementedError):
            self.platform.get_piecewise_backend_cls()


def _built_bytes_per_token(pool_cls, *, num_kv_heads, head_dim, v_head_dim, dtype):
    """Per-token, per-layer bytes of a pool actually built on the CPU."""
    page_size = 16
    pool = pool_cls(
        size=4 * page_size,
        page_size=page_size,
        dtype=dtype,
        head_num=num_kv_heads,
        head_dim=head_dim,
        v_head_dim=v_head_dim,
        layer_num=1,
        device="cpu",
        enable_memory_saver=False,
    )
    buf = pool.kv_buffer[0]
    return buf.numel() * buf.element_size() // (pool.num_blocks * page_size)


class TestTpuKvPoolBytesPerToken(CustomTestCase):
    """The KV budget is spent before any pool exists, so each pool's
    ``kv_bytes_per_token_per_layer`` must track its ``_create_buffers``."""

    def _check(self, pool_cls, expected, **shape):
        dtype = shape.pop("dtype")
        per_token = pool_cls.kv_bytes_per_token_per_layer(**shape, store_dtype=dtype)
        self.assertEqual(per_token, expected)
        self.assertEqual(
            per_token, _built_bytes_per_token(pool_cls, **shape, dtype=dtype)
        )

    def test_head_dim_64_is_priced_at_the_padded_width(self):
        """Bug regression: sizing priced heads * (head_dim + v_head_dim) while the
        fused pool pads head_dim 64 to 128, so the pool was twice its budget."""
        self._check(
            TpuFusedMHATokenToKVPool,
            8 * 2 * 128 * 2,
            num_kv_heads=8,
            head_dim=64,
            v_head_dim=64,
            dtype=torch.bfloat16,
        )

    def test_fp8_rounds_the_head_axis_up_to_the_lane_packing(self):
        """fp8 packs 4 per 32-bit lane, so one KV head's K/V axis rounds 2 up to 4
        on top of the head_dim padding."""
        for heads in (1, 2):
            self._check(
                TpuFusedMHATokenToKVPool,
                4 * 128,
                num_kv_heads=heads,
                head_dim=64,
                v_head_dim=64,
                dtype=torch.float8_e4m3fn,
            )

    def test_v_is_stored_at_the_padded_qk_width(self):
        """The fused pool stores V in a slot of the padded qk width, so the
        footprint is flat in v_head_dim, unlike head_dim + v_head_dim."""
        for v_head_dim in (192, 128):
            self._check(
                TpuFusedMHATokenToKVPool,
                8 * 256 * 2,
                num_kv_heads=4,
                head_dim=192,
                v_head_dim=v_head_dim,
                dtype=torch.bfloat16,
            )

    def test_platform_prices_only_the_padding_pools(self):
        """A pool without padding reports None, so sizing keeps the generic formula."""
        platform = TpuSRTPlatform()
        shape = dict(
            num_kv_heads=8, head_dim=64, v_head_dim=64, kv_cache_dtype=torch.bfloat16
        )
        with get_context().override_server_args(attention_backend="torch_native"):
            self.assertIsNone(platform.get_mha_kv_bytes_per_token(**shape))
        with get_context().override_server_args(attention_backend="pallas_tpu_rpa_v3"):
            self.assertEqual(platform.get_mha_kv_bytes_per_token(**shape), 4096)


if __name__ == "__main__":
    unittest.main()
