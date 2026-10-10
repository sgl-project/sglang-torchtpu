"""CPU tests for the TPU collective cache compatibility contract."""

import os
import sys
import unittest
from types import ModuleType
from unittest.mock import Mock, patch

import torch.compiler.config as compiler_config

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import enter_scope
from sglang_tpu import compile_cache

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestCollectiveCompileCache(unittest.TestCase):
    def setUp(self):
        enter_scope(self, patch.dict(os.environ, TORCH_TPU_TOPOLOGY="2,2,1,1"))
        enter_scope(self, compiler_config.patch(cache_key_tag=""))
        self.world_size = enter_scope(
            self, patch("torch.distributed.get_world_size", return_value=4)
        )
        self.rank = enter_scope(
            self, patch("torch.distributed.get_rank", return_value=0)
        )
        device_mesh = ModuleType("torch_tpu._internal.distributed.device_mesh")
        self.assignment = device_mesh.get_device_assignment = Mock(
            return_value=[0, 1, 2, 3]
        )
        enter_scope(self, patch.dict(sys.modules, {device_mesh.__name__: device_mesh}))

    def install(self, existing="user|pallas-sources"):
        compiler_config.cache_key_tag = existing
        compile_cache.add_tpu_rank_assignment_to_compile_cache_key()
        return compiler_config.cache_key_tag

    def test_assignment_is_stable_and_preserves_existing_tags(self):
        original = self.install()
        self.assertTrue(
            original.startswith("user|pallas-sources|tpu-rank-assignment-v1:")
        )
        self.assertEqual(self.install(), original)
        # Installing over an already-guarded tag is a no-op, even with later tags.
        for existing in (original, original + "|later-tag"):
            with self.subTest(existing=existing):
                self.assertEqual(self.install(existing), existing)
        # A remote swap matters even when this worker stays on device zero, and
        # is rechecked on an already-guarded tag.
        self.assignment.return_value = [0, 2, 1, 3]
        swapped = self.install()
        self.assertNotEqual(swapped, original)
        self.assertNotEqual(self.install(original), original)
        self.assignment.return_value = [0, 1, 2, 3]
        self.assertEqual(self.install(), original)

    def test_rank_and_topology_are_part_of_the_key(self):
        original = self.install()
        self.rank.return_value = 1
        self.assertNotEqual(self.install(), original)
        self.rank.return_value = 0
        os.environ["TORCH_TPU_TOPOLOGY"] = "4,1,1,1"
        self.assertNotEqual(self.install(), original)

    def test_empty_user_tag(self):
        self.assertEqual(self.install(None), self.install(""))
        self.assertTrue(self.install("").startswith("tpu-rank-assignment-v1:"))

    def test_invalid_native_assignment_fails_before_cache_selection(self):
        for mapping in ([], [0], [0, 0, 2, 3], [0, None, 2, 3], [0, -1, 2, 3]):
            with self.subTest(mapping=mapping):
                self.assignment.return_value = mapping
                with self.assertRaisesRegex(
                    RuntimeError, "Invalid TPU rank assignment"
                ):
                    self.install()
                self.assertEqual(compiler_config.cache_key_tag, "user|pallas-sources")

    def test_single_rank_does_not_query_native_assignment(self):
        self.world_size.return_value = 1
        self.assertEqual(self.install(), "user|pallas-sources")
        self.assignment.assert_not_called()


if __name__ == "__main__":
    unittest.main()
