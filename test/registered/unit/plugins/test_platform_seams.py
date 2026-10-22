# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the wave-1 platform seams: PlatformCapabilities, OotReplaceable,
HelperCompilePolicy / maybe_compile. Runs without TPU hardware."""

import copy
import os
import pickle
import sys
import types
import unittest
from unittest.mock import patch

import msgspec

import sglang.srt.platforms as platforms_module
from sglang.srt.compilation import helper_compile
from sglang.srt.compilation.helper_compile import HelperCompilePolicy, maybe_compile
from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8LinearMethod
from sglang.srt.platforms.interface import PlatformCapabilities, SRTPlatform
from sglang.srt.plugins.oot_classes import (
    OotReplaceable,
    clear_oot_replacements,
    oot_base_class,
    register_oot_replacement,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir, os.pardir)
)

# The CPU gate installs python/ only; an installed sglang_tpu still wins.
if _REPO_ROOT not in sys.path:
    sys.path.append(_REPO_ROOT)


# Module level so pickle can find them by qualified name.
class _Base(OotReplaceable):
    def __init__(self, value, *, tag="base"):
        self.value = value
        self.tag = tag


class _Oot(_Base):
    def __init__(self, value, *, tag="base"):
        super().__init__(value, tag=tag)
        self.oot_init_ran = True


class _SubOfBase(_Base):
    pass


class _TpuLikeFp8LinearMethod(Fp8LinearMethod):
    pass


def _fp8_block_config():
    return Fp8Config(is_checkpoint_fp8_serialized=True, weight_block_size=[128, 128])


class TestPlatformCapabilities(CustomTestCase):
    def test_upstream_fields_come_first_in_upstream_order(self):
        """The struct copies sgl-project/sglang#35594; a reorder breaks the sync merge."""
        self.assertEqual(
            PlatformCapabilities.__struct_fields__[:4],
            (
                "supports_triton",
                "graph_capture",
                "piecewise_graph",
                "hicache_device_kernels",
            ),
        )

    def test_fork_fields_default_to_non_tpu_behaviour(self):
        """An adopting PR swaps `_is_tpu` for the field; a wrong default changes CUDA."""
        caps = SRTPlatform.capabilities
        self.assertFalse(caps.uniform_dp_padding)
        self.assertFalse(caps.rank_uniform_graphs)
        self.assertTrue(caps.device_p2p)
        self.assertTrue(caps.traces_raw_collectives)

    def test_record_is_frozen_and_keyword_only(self):
        caps = PlatformCapabilities()
        with self.assertRaises(AttributeError):
            caps.uniform_dp_padding = True
        with self.assertRaises(TypeError):
            PlatformCapabilities(True)
        with self.assertRaises(TypeError):
            PlatformCapabilities(no_such_field=True)


class TestTpuCapabilities(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        from sglang_tpu.platform import TpuOOTPlatform
        from sglang_tpu.tpu_platform import TpuSRTPlatform

        cls.TpuSRTPlatform = TpuSRTPlatform
        cls.TpuOOTPlatform = TpuOOTPlatform

    def test_modes_a_and_b_share_one_record(self):
        self.assertIs(
            self.TpuOOTPlatform.capabilities, self.TpuSRTPlatform.capabilities
        )

    def test_tpu_record_differs_from_default_exactly_here(self):
        """Snapshot: a flag flipped here flips every adopted core site on TPU."""
        tpu = msgspec.structs.asdict(self.TpuSRTPlatform.capabilities)
        default = msgspec.structs.asdict(PlatformCapabilities())
        changed = {name: tpu[name] for name in tpu if tpu[name] != default[name]}
        self.assertEqual(
            changed,
            {
                "uniform_dp_padding": True,
                "rank_uniform_graphs": True,
                "device_p2p": False,
                "traces_raw_collectives": False,
            },
        )

    def test_tpu_declares_no_device_graphs(self):
        # The capability record is the only graph-capture switch core reads.
        caps = self.TpuOOTPlatform().capabilities
        self.assertFalse(caps.graph_capture)
        self.assertFalse(caps.piecewise_graph)


class TestOotReplaceable(CustomTestCase):
    def setUp(self):
        clear_oot_replacements()
        self.addCleanup(clear_oot_replacements)

    def test_construction_without_registration_is_unchanged(self):
        obj = _Base(1, tag="x")
        self.assertIs(type(obj), _Base)
        self.assertEqual((obj.value, obj.tag), (1, "x"))

    def test_registered_base_builds_the_replacement_with_its_init(self):
        register_oot_replacement(_Base, _Oot)
        obj = _Base(7, tag="t")
        self.assertIs(type(obj), _Oot)
        self.assertEqual((obj.value, obj.tag, obj.oot_init_ran), (7, "t", True))

    def test_lookup_is_exact_type(self):
        """Subclasses of the base and the replacement itself construct unchanged."""
        register_oot_replacement(_Base, _Oot)
        self.assertIs(type(_SubOfBase(1)), _SubOfBase)
        self.assertIs(type(_Oot(1)), _Oot)

    def test_deepcopy_and_pickle_keep_the_concrete_class(self):
        register_oot_replacement(_Base, _Oot)
        obj = _Base(3, tag="p")
        for clone in (copy.deepcopy(obj), pickle.loads(pickle.dumps(obj))):
            self.assertIs(type(clone), _Oot)
            self.assertEqual((clone.value, clone.tag), (3, "p"))

    def test_registration_rejects_invalid_pairs(self):
        with self.assertRaises(TypeError):
            register_oot_replacement(_Base, object)
        with self.assertRaises(TypeError):
            register_oot_replacement(_Base, _Base)
        with self.assertRaises(TypeError):
            register_oot_replacement(dict, type("D", (dict,), {}))

    def test_double_registration_is_refused(self):
        register_oot_replacement(_Base, _Oot)
        with self.assertRaises(ValueError):
            register_oot_replacement(_Base, type("Other", (_Base,), {}))

    def test_registering_the_same_pair_again_is_a_no_op(self):
        register_oot_replacement(_Base, _Oot)
        register_oot_replacement(_Base, _Oot)
        self.assertIs(type(_Base(1)), _Oot)
        self.assertIs(oot_base_class(_Oot), _Base)

    def test_one_class_cannot_replace_two_bases(self):
        both = type("Both", (_SubOfBase,), {})
        register_oot_replacement(_Base, both)
        with self.assertRaises(ValueError):
            register_oot_replacement(_SubOfBase, both)

    def test_replacement_class_maps_back_to_its_base(self):
        register_oot_replacement(_Base, _Oot)
        self.assertIs(oot_base_class(_Oot), _Base)
        self.assertIs(oot_base_class(_SubOfBase), _SubOfBase)

    def test_quant_and_scheme_bases_are_replaceable(self):
        from sglang.srt.layers.quantization.base_config import QuantizeMethodBase
        from sglang.srt.layers.quantization.base_scheme import (
            BaseLinearScheme,
            BaseMoEScheme,
        )

        for base in (QuantizeMethodBase, BaseLinearScheme, BaseMoEScheme):
            self.assertTrue(issubclass(base, OotReplaceable), base.__name__)

    def test_substituted_fp8_linear_method_survives_deepcopy_and_pickle(self):
        register_oot_replacement(Fp8LinearMethod, _TpuLikeFp8LinearMethod)
        method = Fp8LinearMethod(_fp8_block_config())
        self.assertIs(type(method), _TpuLikeFp8LinearMethod)
        for clone in (copy.deepcopy(method), pickle.loads(pickle.dumps(method))):
            self.assertIs(type(clone), _TpuLikeFp8LinearMethod)
            self.assertTrue(clone.block_quant)

    def test_substitute_keeps_its_base_weight_loader(self):
        """A substituted Fp8LinearMethod must still get weight_loader_v2."""
        from sglang.srt.layers.linear import _uses_weight_loader_v2

        register_oot_replacement(Fp8LinearMethod, _TpuLikeFp8LinearMethod)
        method = Fp8LinearMethod(_fp8_block_config())
        self.assertTrue(_uses_weight_loader_v2(quant_method=method, scheme=None))

    def test_substitute_still_accepts_the_fused_fp8_tuple(self):
        """qwen3_5 must route the pre-quantized AR tuple to a substituted fp8 linear."""
        from sglang.srt.models.qwen3_5 import _linear_accepts_fp8_tuple

        register_oot_replacement(Fp8LinearMethod, _TpuLikeFp8LinearMethod)
        linear = types.SimpleNamespace(
            quant_method=Fp8LinearMethod(_fp8_block_config())
        )
        self.assertTrue(_linear_accepts_fp8_tuple(linear))


class _RecordingPolicy(HelperCompilePolicy):
    def __init__(self):
        self.calls = []

    def compile_helper(self, fn, *, dynamic, backend, options):
        self.calls.append((fn, dynamic, backend, dict(options)))
        return "compiled"


class _FakePlatform(SRTPlatform):
    def __init__(self, policy):
        self.policy = policy

    def get_helper_compile_policy(self):
        return self.policy


def _helper(x):
    return x


class TestMaybeCompile(CustomTestCase):
    def _with_platform(self, platform):
        patcher = patch.object(platforms_module, "_current_platform", platform)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_default_policy_is_torch_compile_as_written(self):
        self._with_platform(SRTPlatform())
        with patch.object(helper_compile.torch, "compile", return_value="c") as tc:
            self.assertEqual(maybe_compile(_helper, dynamic=True), "c")
            tc.assert_called_once_with(_helper, dynamic=True)
            tc.reset_mock()
            self.assertEqual(
                maybe_compile(dynamic=False, backend="eager", fullgraph=True)(_helper),
                "c",
            )
            tc.assert_called_once_with(
                _helper, dynamic=False, backend="eager", fullgraph=True
            )

    def test_disable_returns_fn_without_asking_the_platform(self):
        policy = _RecordingPolicy()
        self._with_platform(_FakePlatform(policy))
        self.assertIs(maybe_compile(_helper, disable=True), _helper)
        self.assertEqual(policy.calls, [])

    def test_platform_policy_decides(self):
        policy = _RecordingPolicy()
        self._with_platform(_FakePlatform(policy))
        decorated = maybe_compile(dynamic=True, backend="b", mode="m")(_helper)
        self.assertEqual(decorated, "compiled")
        self.assertEqual(policy.calls, [(_helper, True, "b", {"mode": "m"})])

    def test_tpu_policy_leaves_dynamic_helpers_eager(self):
        """Same as today's @torch.compile(disable=_is_tpu): the function itself."""
        from sglang_tpu.tpu_platform import TpuSRTPlatform

        self._with_platform(TpuSRTPlatform())
        self.assertIs(maybe_compile(_helper, dynamic=True), _helper)


if __name__ == "__main__":
    unittest.main()
