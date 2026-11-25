"""SGLANG_TPU_K3_LATENT_UP_ATTN_RES keeps per-forward Python state that Dynamo
traces. A forward that starts from the state the previous forward left must
not recompile: CompiledForwardShim empties it eagerly before every call."""

import unittest
from types import SimpleNamespace

import torch
from torch._dynamo.testing import CompileCounter

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.compilation import compiled_forward_shim
from sglang_tpu.kimi_k3 import latent_up_tail

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _forward(x):
    # Each attention-residual aggregation records its bank (hooks/attn_residual.py).
    latent_up_tail.record_aggregation(x, 1)
    return x + 1


class TestLatentUpTailReset(CustomTestCase):
    def setUp(self):
        torch._dynamo.reset()
        latent_up_tail.reset()

    def tearDown(self):
        latent_up_tail.reset()

    def _graphs(self, eager_reset):
        counter = CompileCounter()
        fn = torch.compile(_forward, backend=counter)
        for _ in range(3):
            if eager_reset:
                latent_up_tail.reset()
            fn(torch.zeros(4, 8))
        return counter.frame_count

    def test_leftover_state_recompiles(self):
        self.assertEqual(self._graphs(eager_reset=False), 2)

    def test_reset_keeps_one_graph(self):
        self.assertEqual(self._graphs(eager_reset=True), 1)

    def test_shim_runs_registered_resets_before_the_forward(self):
        compiled_forward_shim.register_eager_reset(latent_up_tail.reset)
        latent_up_tail.record_aggregation(torch.zeros(1), 1)  # a prior forward's
        seen = []
        shim = SimpleNamespace(
            wait_for_hicache_reload=lambda: None,
            _default_kwargs={},
            _recompilation_checker=None,
            _forward=lambda fn, *args, **kwargs: fn(*args, **kwargs),
        )
        compiled_forward_shim.CompiledForwardShim._call(
            shim,
            lambda *args: seen.append(list(latent_up_tail._STATE.agg)),
            None,
            None,
            None,
        )
        self.assertEqual(seen, [[]])


if __name__ == "__main__":
    unittest.main()
