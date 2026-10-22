"""CPU tests for the optional MXINT8 backend integration boundary."""

import ast
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.arg_groups.overrides import resolving_view
from sglang.srt.mem_cache.memory_pool import (
    MHATokenToKVPool,
    PageMajorMHATokenToKVPool,
)
from sglang.srt.mem_cache.page_interleave import make_page_shard_spec
from sglang.srt.runtime_context import (
    get_context,
    get_exec,
    get_parallel,
    get_schedule,
    get_server_args,
    get_spec,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _load_configurator(**dependencies):
    # Compile the original dataclass, including its constructor and methods,
    # without initializing accelerator-only registries. Configuration is real;
    # only model-family discovery and pool constructors are test doubles.
    path = (
        Path(__file__).resolve().parents[4]
        / "python/sglang/srt/mem_cache/kv_cache_configurator.py"
    )
    tree = ast.parse(path.read_text())
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "KVCacheConfigurator"
    )
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    scope = {
        "__name__": __name__,
        "dataclass": dataclass,
        "field": field,
        "torch": torch,
        "resolving_view": resolving_view,
        "get_parallel": get_parallel,
        "get_schedule": get_schedule,
        "get_exec": get_exec,
        "get_spec": get_spec,
        # The QSA pool build reads the indexer dtype off the model namespace.
        "get_model": lambda: SimpleNamespace(qsa_indexer_dtype="auto"),
        "mambaish_config": lambda _: None,
        "hybrid_gdn_config": lambda _: None,
        "hybrid_kda_config": lambda _: None,
        # Logical-page KV sharding stays off unless the parallel bag enables it.
        "make_page_shard_spec": make_page_shard_spec,
        # The hybrid full-attention pool choice branches on NPU first.
        "_is_npu": False,
        # The out-of-tree MHA build tells the in-tree pools from a platform's.
        "MHATokenToKVPool": MHATokenToKVPool,
        "PageMajorMHATokenToKVPool": PageMajorMHATokenToKVPool,
        **dependencies,
    }
    module = ast.fix_missing_locations(ast.Module(body=[future, cls], type_ignores=[]))
    exec(compile(module, str(path), "exec"), scope)
    return scope[cls.name]


class TestMXINT8Configurator(CustomTestCase):
    def setUp(self):
        super().setUp()
        override = get_context().override_server_args(
            page_size=128, speculative_draft_kv_cache_dtype=None
        )
        override.install()
        self.addCleanup(override.restore)
        self.platform_pool = Mock(name="platform_pool")
        self.platform = SimpleNamespace(
            get_kv_pool_cls=Mock(return_value=self.platform_pool),
            is_cpu=lambda: False,
        )
        self.swa_pool = Mock(name="swa_pool")
        self.hybrid_pool = Mock(name="hybrid_pool")
        self.configurator = _load_configurator(
            current_platform=self.platform,
            SWAKVPool=self.swa_pool,
            HybridLinearKVPool=self.hybrid_pool,
            is_float4_e2m1fn_x2=lambda _: False,
        )

    def _make(self, **overrides):
        kwargs = dict(
            device="tpu",
            gpu_id=0,
            attn_dp_size=1,
            pp_size=1,
            pp_group=None,
            model=None,
            model_config=SimpleNamespace(
                get_num_kv_heads=lambda *_: 3,
                head_dim=64,
                v_head_dim=64,
                swa_head_dim=128,
                swa_v_head_dim=128,
                hf_text_config=SimpleNamespace(
                    num_key_value_heads=3, swa_num_key_value_heads=5
                ),
                swa_attention_layer_ids=[1],
                full_attention_layer_ids=[0],
                hf_config=SimpleNamespace(),
            ),
            server_args=get_server_args(),
            kv_cache_dtype=torch.int8,
            kv_cache_dtype_str="mxint8",
            kv_group_size=16,
            model_dtype=torch.bfloat16,
            page_size=128,
            sliding_window_size=None,
            spec_algorithm=SpeculativeAlgorithm.from_string("EAGLE"),
            is_draft_worker=False,
            post_capture_kv_active=False,
            spec_aux_config=None,
            is_hybrid_swa=False,
            is_hybrid_swa_compress=False,
            use_mla_backend=False,
            layer_info=SimpleNamespace(
                start_layer=0,
                end_layer=2,
                num_effective_layers=2,
                swa_attention_layer_ids=[1],
                full_attention_layer_ids=[0],
                is_hybrid_swa_mtp_draft=False,
            ),
            forward_stream=None,
            req_to_token_pool=None,
            token_to_kv_pool_allocator=None,
            memory_pool_config=None,
            speculative_draft_attention_backend="runner_backend",
        )
        kwargs.update(overrides)
        return self.configurator(**kwargs)

    def _assert_preflight_error(
        self, message="unsupported.*backend extension", **kwargs
    ):
        with patch.object(self.configurator, "_profile_available_bytes") as profile:
            with self.assertRaisesRegex(ValueError, message):
                kvc = self._make(**kwargs)
                kvc.configure(pre_model_load_memory=1)
            profile.assert_not_called()

    def test_current_mxint8_rejected_before_profiling(self):
        for device in ("cpu", "cuda", "tpu", "tpu:0"):
            for is_draft in (False, True):
                with self.subTest(device=device, draft=is_draft):
                    self._assert_preflight_error(
                        device=device, is_draft_worker=is_draft
                    )

    def test_future_mxint8_draft_rejected_before_target_profiling(self):
        for algorithm in ("EAGLE", "EAGLE3", "STANDALONE"):
            for name, dtype in (
                ("bf16", torch.bfloat16),
                ("fp8_e4m3", torch.float8_e4m3fn),
                ("fp8_e5m2", torch.float8_e5m2),
            ):
                with (
                    self.subTest(algorithm=algorithm, format=name),
                    get_context().override_server_args(
                        speculative_draft_kv_cache_dtype="mxint8"
                    ),
                ):
                    self._assert_preflight_error(
                        kv_cache_dtype_str=name,
                        kv_cache_dtype=dtype,
                        spec_algorithm=SpeculativeAlgorithm.from_string(algorithm),
                    )

    def test_unused_draft_format_does_not_reject_ordinary_runner(self):
        with get_context().override_server_args(
            speculative_draft_kv_cache_dtype="mxint8"
        ):
            for algorithm, is_draft in ((None, False), ("EAGLE", True)):
                with self.subTest(algorithm=algorithm, draft=is_draft):
                    kvc = self._make(
                        kv_cache_dtype_str="bf16",
                        kv_cache_dtype=torch.bfloat16,
                        spec_algorithm=SpeculativeAlgorithm.from_string(algorithm),
                        is_draft_worker=is_draft,
                    )
                    self.assertIsNone(kvc.kv_group_size)

    def test_ordinary_formats_keep_platform_factory(self):
        for name, dtype in (
            ("auto", torch.bfloat16),
            ("bf16", torch.bfloat16),
            ("fp8_e4m3", torch.float8_e4m3fn),
            ("fp8_e5m2", torch.float8_e5m2),
        ):
            with self.subTest(format=name):
                kvc = self._make(kv_cache_dtype_str=name, kv_cache_dtype=dtype)
                self.assertIsNone(kvc.kv_group_size)
                self.assertEqual(kvc._mxint8_pool_kwargs(), {})
                self.assertIs(kvc._tpu_mha_pool_class(), self.platform_pool)
        self.assertEqual(self.platform.get_kv_pool_cls.call_count, 4)

    def test_validation_override_retains_default_group_and_platform_factory(self):
        with patch.object(self.configurator, "_validate_mxint8", return_value=None):
            kvc = self._make(kv_group_size=None)
        self.assertEqual(kvc.kv_group_size, 32)
        self.assertIs(kvc._tpu_mha_pool_class(), self.platform_pool)
        self.platform.get_kv_pool_cls.assert_called_once_with(kind="mha")

    def test_bare_int8_rejected_before_profiling(self):
        for name in (None, "auto", "bf16"):
            with self.subTest(format=name):
                self._assert_preflight_error("explicit mxint8", kv_cache_dtype_str=name)

    def test_backend_overrides_preserve_runner_metadata_and_pool_kwargs(self):
        backend_pool = Mock(name="backend_pool")
        observed = []

        def validate(kvc):
            observed.append(
                (kvc.kv_group_size, kvc.speculative_draft_attention_backend)
            )

        with (
            get_parallel().override(attn_tp_size=1, attn_dcp_size=1),
            patch.object(self.configurator, "_validate_mxint8", validate),
            patch.object(
                self.configurator, "_tpu_mha_pool_class", lambda _: backend_pool
            ),
        ):
            kvc = self._make(is_draft_worker=True)
            self.assertEqual(observed, [(16, "runner_backend")])
            self.assertIs(
                kvc._build_oot_mha_kv_pool(max_total_num_tokens=512),
                backend_pool.return_value,
            )
            self.assertEqual(backend_pool.call_args.args, (512,))
            self.assertEqual(
                backend_pool.call_args.kwargs["kv_cache_dtype_str"], "mxint8"
            )
            self.assertEqual(backend_pool.call_args.kwargs["kv_group_size"], 16)
            kvc._build_tpu_swa_kv_pool(
                full_max_total_num_tokens=512, swa_max_total_num_tokens=256
            )
            kwargs = self.swa_pool.call_args.kwargs
            self.assertIs(kwargs["token_to_kv_pool_class"], backend_pool)
            for label, heads, dim in (("full", 3, 64), ("swa", 5, 128)):
                subpool = kwargs[f"{label}_kv_pool_kwargs"]
                self.assertEqual(subpool["kv_cache_dtype_str"], "mxint8")
                self.assertEqual(subpool["kv_group_size"], 16)
                self.assertEqual(
                    (subpool["head_num"], subpool["head_dim"]), (heads, dim)
                )
            kvc.mambaish_config = SimpleNamespace(full_attention_layer_ids=[0])
            mla_pool, dsa_pool = Mock(name="mla_pool"), Mock(name="dsa_pool")
            kvc._build_hybrid_linear_kv_pool(
                max_total_num_tokens=512,
                req_to_token_pool=SimpleNamespace(mamba_pool=object()),
                mha_pool_class=self.platform_pool,
                mla_pool_class=mla_pool,
                dsa_pool_class=dsa_pool,
            )
            mla_pool.assert_not_called()
            dsa_pool.assert_not_called()
            self.assertIs(
                self.hybrid_pool.call_args.kwargs["full_kv_pool"],
                backend_pool.return_value,
            )
            self.assertEqual(backend_pool.call_args.kwargs["layer_num"], 1)
            self.platform.get_kv_pool_cls.assert_not_called()
        self._assert_preflight_error()


if __name__ == "__main__":
    unittest.main()
