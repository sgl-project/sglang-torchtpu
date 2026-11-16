"""Early checkpoint prefetch in the TPU worker bootstrap."""

import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu import checkpoint_prefetch

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _cfg(model_path, **overrides):
    values = dict(
        model_path=model_path,
        weight_loader_prefetch_checkpoints=True,
        weight_loader_prefetch_num_threads=3,
        weight_loader_disable_mmap=False,
        load_format="auto",
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class TestEarlyCheckpointPrefetch(CustomTestCase):
    def setUp(self):
        checkpoint_prefetch._handle = None
        self.addCleanup(setattr, checkpoint_prefetch, "_handle", None)
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.shards = []
        for i in range(5):
            path = os.path.join(self.tmpdir.name, f"model-{i}.safetensors")
            open(path, "wb").close()
            self.shards.append(path)

    def test_prefetches_this_ranks_share_once(self):
        calls = []
        fake = patch.object(
            checkpoint_prefetch,
            "_prefetch_all_checkpoints",
            side_effect=lambda files, num_threads: (
                calls.append((files, num_threads)) or object()
            ),
        )
        with fake, patch.dict(os.environ, {"RANK": "5", "LOCAL_WORLD_SIZE": "4"}):
            first = checkpoint_prefetch.start_early_checkpoint_prefetch(
                _cfg(self.tmpdir.name)
            )
            second = checkpoint_prefetch.start_early_checkpoint_prefetch(
                _cfg(self.tmpdir.name)
            )
        self.assertIs(first, second)
        self.assertEqual(calls, [([self.shards[1]], 3)])

    def test_skipped_when_loader_prefetch_is_off_or_path_is_not_local(self):
        with patch.object(checkpoint_prefetch, "_prefetch_all_checkpoints") as fake:
            for cfg in (
                _cfg(self.tmpdir.name, weight_loader_prefetch_checkpoints=False),
                _cfg(self.tmpdir.name, weight_loader_disable_mmap=True),
                _cfg(self.tmpdir.name, load_format="dummy"),
                _cfg(self.tmpdir.name, load_format="pt"),
                _cfg("org/repo-id"),
            ):
                self.assertIsNone(
                    checkpoint_prefetch.start_early_checkpoint_prefetch(cfg)
                )
        fake.assert_not_called()


if __name__ == "__main__":
    unittest.main()
