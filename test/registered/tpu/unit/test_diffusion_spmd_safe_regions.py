"""TPU CI entry for the SPMD-safe regional-compile tests.

Mostly host-only checks, registered here because the CPU CI image does not
install the diffusion runtime; the torch_tpu wrapper check needs torch_tpu.
"""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=15, suite="stage-a-test-tpu-unit", chips=1)

import importlib.util
import sys
import unittest
from types import ModuleType
from unittest.mock import patch

import torch.compiler.config as compiler_config

from sglang.multimodal_gen.test.unit.test_spmd_safe_regions import (  # noqa: F401
    TestSpmdSafeRegions,
    TestWanBlockInputsAreRankInvariant,
)
from sglang_tpu.multimodal_gen.platform import TpuOOTPlatform

_CACHE_TAG = "spmd-safe-regions-v1"


def _torch_tpu_stubs(spmd_safe=None):
    names = (
        "torch_tpu",
        "torch_tpu._internal",
        "torch_tpu._internal.distributed",
        "torch_tpu._internal.distributed.spmd_util",
    )
    stubs = {name: ModuleType(name) for name in names}
    if spmd_safe is not None:
        stubs[names[-1]].spmd_safe = spmd_safe
    return stubs


class TestTpuSpmdRegionWrapper(unittest.TestCase):
    def test_wrapper_tags_the_compile_cache_once(self):
        spmd_safe = object()
        stubs = _torch_tpu_stubs(spmd_safe)
        for user_tag, expected in (("", _CACHE_TAG), ("user", f"user|{_CACHE_TAG}")):
            with (
                self.subTest(user_tag=user_tag),
                patch.dict(sys.modules, stubs),
                compiler_config.patch(cache_key_tag=user_tag),
            ):
                for _ in range(2):
                    self.assertIs(
                        TpuOOTPlatform().prepare_spmd_region_wrapper(), spmd_safe
                    )
                self.assertEqual(compiler_config.cache_key_tag, expected)

    def test_torch_tpu_without_spmd_safe_leaves_regions_unwrapped(self):
        # spmd_safe is a private torch_tpu API: without it, regions compile
        # with split collectives and the cache key keeps its tags.
        with (
            patch.dict(sys.modules, _torch_tpu_stubs()),
            compiler_config.patch(cache_key_tag="user"),
        ):
            self.assertIsNone(TpuOOTPlatform().prepare_spmd_region_wrapper())
            self.assertEqual(compiler_config.cache_key_tag, "user")

    @unittest.skipUnless(importlib.util.find_spec("torch_tpu"), "requires torch_tpu")
    def test_wrapper_is_torch_tpu_spmd_safe(self):
        from torch_tpu._internal.distributed.spmd_util import spmd_safe

        with compiler_config.patch(cache_key_tag=""):
            self.assertIs(TpuOOTPlatform().prepare_spmd_region_wrapper(), spmd_safe)


if __name__ == "__main__":
    unittest.main()
