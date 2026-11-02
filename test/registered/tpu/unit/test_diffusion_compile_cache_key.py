"""TPU CI entry for the diffusion torch.compile cache-key tests."""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=20, suite="stage-a-test-tpu-unit", chips=1)

import sys
import unittest
from types import ModuleType
from unittest.mock import patch

import torch.compiler.config as compiler_config

from sglang.multimodal_gen.test.unit.test_compile_cache_key import (  # noqa: F401
    TestCompileCacheHook,
)
from sglang_tpu.multimodal_gen.platform import TpuOOTPlatform

_KERNEL_INPUTS_TAG = r"^tpu-kernel-inputs-v1:[0-9a-f]{64}$"
_RANK_ASSIGNMENT_TAG = r"^tpu-rank-assignment-v1:[0-9a-f]{64}$"


class TestTpuCompileCacheKey(unittest.TestCase):
    """The TPU worker adds sglang_tpu's compile-cache tags; what each tag
    digests is covered by sglang_tpu's compile_cache tests."""

    def _tags(self, world_size=1, existing="user-tag"):
        device_mesh = ModuleType("torch_tpu._internal.distributed.device_mesh")
        device_mesh.get_device_assignment = lambda: list(range(world_size))
        with (
            patch.dict(sys.modules, {device_mesh.__name__: device_mesh}),
            patch("torch.distributed.get_world_size", return_value=world_size),
            patch("torch.distributed.get_rank", return_value=0),
            compiler_config.patch(cache_key_tag=existing),
        ):
            TpuOOTPlatform().configure_compile_cache()
            return compiler_config.cache_key_tag.split("|")

    def test_one_worker_keeps_user_tags_and_adds_the_kernel_inputs(self):
        user, kernel_inputs = self._tags()
        self.assertEqual(user, "user-tag")
        self.assertRegex(kernel_inputs, _KERNEL_INPUTS_TAG)

    def test_multi_worker_launch_adds_the_rank_assignment(self):
        user, rank_assignment, kernel_inputs = self._tags(world_size=4)
        self.assertEqual(user, "user-tag")
        self.assertRegex(rank_assignment, _RANK_ASSIGNMENT_TAG)
        self.assertRegex(kernel_inputs, _KERNEL_INPUTS_TAG)

    def test_reconfiguring_keeps_one_tag_of_each_kind(self):
        tags = self._tags(world_size=4)
        self.assertEqual(self._tags(world_size=4, existing="|".join(tags)), tags)


if __name__ == "__main__":
    unittest.main()
