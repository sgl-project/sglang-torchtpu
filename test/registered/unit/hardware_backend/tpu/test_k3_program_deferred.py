"""The deferred-write guard: sglang's pending torch_tpu writes are flushed
before a program step (sglang_deferred), here over CPU tensors."""

import unittest
from types import SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.k3_programs import sglang_deferred as deferred

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDeferredGuard(CustomTestCase):
    def setUp(self):
        self.mapping = torch.zeros(8, dtype=torch.int32)
        self.seq_lens = torch.ones(4)
        sampling = SimpleNamespace(temperatures=torch.ones(4), top_ks=[torch.ones(4)])
        self.batch = SimpleNamespace(
            seq_lens=self.seq_lens,
            sampling_info=sampling,
            reqs=[SimpleNamespace(big=torch.ones(1))],  # never walked
        )
        self.pool = SimpleNamespace(req_index_to_mamba_index_mapping=self.mapping)
        self.roots = (("batch", self.batch), ("pool", self.pool), ("none", None))

    def test_walks_the_steps_objects(self):
        names = [n for n, _ in deferred.device_tensors(self.roots, "cpu")]
        self.assertEqual(
            sorted(names),
            [
                "batch.sampling_info.temperatures",
                "batch.sampling_info.top_ks[0]",
                "batch.seq_lens",
                "pool.req_index_to_mamba_index_mapping",
            ],
        )
        self.assertEqual(deferred.device_tensors(self.roots), [])  # no TPU tensors

    def test_flushes_only_pending_tensors(self):
        flushed = []
        pending = {id(self.mapping)}
        guard = deferred.DeferredGuard(
            device_type="cpu",
            sync=(
                lambda t: id(t) not in pending,
                lambda ts, wait: flushed.append((ts, wait)),
            ),
        )
        self.assertEqual(guard(self.roots), 1)
        ((tensors, wait),) = flushed
        self.assertIs(tensors[0], self.mapping)
        self.assertFalse(wait)
        pending.clear()
        self.assertEqual(guard(self.roots), 0)
        self.assertEqual(guard.flushes, 1)
        self.assertEqual(deferred.DeferredGuard(enabled=False)(self.roots), 0)


if __name__ == "__main__":
    unittest.main()
