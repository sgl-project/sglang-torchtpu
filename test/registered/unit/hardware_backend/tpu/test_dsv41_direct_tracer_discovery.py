# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for DeepSeek-V4.1 direct-tracer tensor discovery and pruning.

The direct tracer captures any tensor the forward reads that it was not told
about as a constant, and lifts every registered tensor unless told to prune.
Guards the discovery of plain tensor attributes (every path, including aliases
of plain and of registered tensors), and that the V4.1 policy opts into pruning
so the host-resident Engram tables are not lifted. Discovery tests follow the attention-TP port of this package.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

import dataclasses
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn

from sglang.test.test_utils import CustomTestCase
from sglang_tpu.models.deepseek_v41 import direct_tracer


class _Helper:
    def __init__(self):
        self.scale = torch.ones(2)
        self.name = "helper"


class _Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(2, 2)
        self.register_buffer("buf", torch.zeros(2))
        self.plain = torch.ones(3)
        self.plain_alias = self.plain  # a second path to an unregistered tensor
        self.alias = self.buf  # same tensor as a registered buffer
        self.helper = _Helper()
        self.cache = {0: torch.ones(1)}
        self.child = nn.Module()
        self.child.table = torch.ones(4)


_real_discover = direct_tracer.discover_extra_tensor_attrs


def _discover_cpu(model):
    return _real_discover(model, device_type="cpu")


class TestDiscoverExtras(CustomTestCase):
    def test_plain_helper_and_alias_attrs(self):
        found, unreachable = direct_tracer.discover_extra_tensor_attrs(
            _Toy(), device_type="cpu"
        )
        self.assertEqual(
            set(found),
            {"plain", "plain_alias", "alias", "helper.scale", "child.table"},
        )
        self.assertEqual(unreachable, ("cache[(1,):float32]",))

    def test_config_object_alias_of_a_parameter_is_listed(self):
        """The MoE TopKConfig.correction_bias aliases gate.e_score_correction_bias."""

        @dataclasses.dataclass
        class Cfg:
            correction_bias: torch.Tensor = None
            top_k: int = 8

        m = nn.Module()
        m.gate = nn.Module()
        m.gate.e_score_correction_bias = nn.Parameter(
            torch.zeros(128), requires_grad=False
        )
        m.topk = nn.Module()
        m.topk.topk_config = Cfg(correction_bias=m.gate.e_score_correction_bias)
        found, _ = direct_tracer.discover_extra_tensor_attrs(m, device_type="cpu")
        self.assertIn("topk.topk_config.correction_bias", found)

    def test_device_filter(self):
        self.assertEqual(
            direct_tracer.discover_extra_tensor_attrs(_Toy(), device_type="tpu"),
            ((), ()),
        )

    def test_paths_resolve_like_direct_compile(self):
        model = _Toy()
        found, _ = direct_tracer.discover_extra_tensor_attrs(model, device_type="cpu")
        for path in found:
            obj = model
            for part in path.split("."):
                obj = getattr(obj, part)
            self.assertIsInstance(obj, torch.Tensor)


class TestPrepareDirectCompile(CustomTestCase):
    def _policy(self, model):
        return SimpleNamespace(
            model_runner=SimpleNamespace(model=model),
            model_extra_tensor_attrs=("existing",),
            get_model_extra_tensor_attrs=lambda: ("existing",),
            direct_compile_prune_unused_inputs=False,
            name="target/forward",
            spmd_safe=True,
        )

    def test_registers_attrs_and_opts_into_pruning(self):
        model = _Toy()
        policy = self._policy(model)
        with mock.patch.object(
            direct_tracer, "discover_extra_tensor_attrs", side_effect=_discover_cpu
        ):
            direct_tracer.prepare_direct_compile(policy)
        self.assertEqual(policy.model_extra_tensor_attrs[0], "existing")
        self.assertIn("child.table", policy.model_extra_tensor_attrs)
        self.assertTrue(policy.direct_compile_prune_unused_inputs)

    def test_base_policy_does_not_prune(self):
        from sglang_tpu.compilation.decoder_compile_policy import (
            DecoderTpuCompilePolicy,
        )

        self.assertFalse(DecoderTpuCompilePolicy.direct_compile_prune_unused_inputs)


class TestDeepseekV4Policy(CustomTestCase):
    def _policy(self, model_type):
        from sglang_tpu.compilation.decoder_compile_policy import (
            DecoderTpuCompilePolicy,
        )
        from sglang_tpu.models.deepseek_v41 import compile_policy
        from sglang_tpu.models.deepseek_v41.compile_policy import (
            DeepseekV4DecoderTpuCompilePolicy,
        )

        runner = SimpleNamespace(
            model=SimpleNamespace(),
            is_draft_worker=False,
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(model_type=model_type), use_engram=False
            ),
        )

        def base_init(self, *, model_runner, **kwargs):
            self.model_runner = model_runner

        with (
            mock.patch.object(DecoderTpuCompilePolicy, "__init__", base_init),
            mock.patch.object(
                compile_policy,
                "get_parallel",
                return_value=SimpleNamespace(attn_dp_size=8),
            ),
        ):
            return DeepseekV4DecoderTpuCompilePolicy(model_runner=runner)

    def test_v41_is_spmd_safe_and_v4_keeps_the_defaults(self):
        self.assertTrue(self._policy("deepseek_v41").spmd_safe)
        v4 = self._policy("deepseek_v4")
        self.assertFalse(v4.spmd_safe)
        self.assertFalse(v4.is_v41)


class TestHostResidentTensors(CustomTestCase):
    def test_largest_first_and_aliases_once(self):
        model = nn.Module()
        model.small = nn.Parameter(torch.zeros(4), requires_grad=False)
        model.table = nn.Parameter(torch.zeros(64, 8), requires_grad=False)
        model.register_buffer("scale", torch.zeros(16, dtype=torch.uint8))
        model.alias = model.table
        self.assertEqual(
            direct_tracer.host_resident_tensors(model),
            [("table", 64 * 8 * 4), ("small", 16), ("scale", 16)],
        )


if __name__ == "__main__":
    unittest.main()
