"""Unit tests for symbolic dynamo decoder warmup."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import unittest

import torch
import torch._dynamo
from torch._dynamo.backends.common import aot_autograd

from sglang_tpu.compilation import symbolic_warmup
from sglang_tpu.compilation.symbolic_warmup import (
    PerShapeBackend,
    mark_bucket_dims_dynamic,
)


class _StaticRecorder:
    """Stands in for the TPU backend: refuses SymInts like it does."""

    def __init__(self) -> None:
        self.shapes: list[list] = []

    def __call__(self, graph, example_inputs, **kwargs):
        for value in example_inputs:
            assert not isinstance(value, torch.SymInt), "inner got a SymInt"
            if isinstance(value, torch.Tensor):
                assert all(isinstance(d, int) for d in value.shape)
        self.shapes.append(
            [tuple(v.shape) for v in example_inputs if isinstance(v, torch.Tensor)]
        )
        return aot_autograd(fw_compiler=lambda g, _: g, bw_compiler=lambda g, _: g)(
            graph, example_inputs
        )


class _Batch:
    def __init__(self, tokens: int, bs: int) -> None:
        self.input_ids = torch.randint(0, 100, (tokens,))
        self.positions = torch.arange(tokens)
        self.seq_lens = torch.full((bs,), 7)
        self.cu_q_lens = torch.arange(bs + 1)


def _forward(weight: torch.Tensor, batch: _Batch) -> torch.Tensor:
    hidden = weight[batch.input_ids] * (batch.positions[:, None] + 1)
    per_req = hidden.sum(0) * batch.seq_lens.sum() + batch.cu_q_lens[-1]
    return torch.nn.functional.silu(per_req)


class TestSymbolicWarmup(unittest.TestCase):
    def setUp(self) -> None:
        torch._dynamo.reset()
        symbolic_warmup.STATS = symbolic_warmup.SymbolicWarmupStats()

    def _run(self, buckets, *, mark: bool):
        inner = _StaticRecorder()
        compiled = torch.compile(
            _forward, backend=PerShapeBackend(inner), fullgraph=True, dynamic=False
        )
        weight = torch.randn(100, 8)
        for tokens, bs in buckets:
            batch = _Batch(tokens, bs)
            if mark:
                mark_bucket_dims_dynamic((tokens, bs), objects=(batch,))
            torch.testing.assert_close(compiled(weight, batch), _forward(weight, batch))
        return inner

    def test_one_trace_one_static_compile_per_bucket(self) -> None:
        buckets = [(64, 64), (128, 64), (4096, 64), (128, 64)]
        inner = self._run(buckets, mark=True)
        stats = symbolic_warmup.STATS
        self.assertEqual(stats.dynamo_traces, 1)
        self.assertEqual(stats.symbolic_traces, 1)
        # The repeated (128, 64) bucket reuses its executable.
        self.assertEqual(len(inner.shapes), 3)
        # Each static compile sees its own token bucket on input_ids/positions.
        token_dims = [
            sorted({s[0] for s in compile_ if len(s) == 1} - {64, 65})
            for compile_ in inner.shapes
        ]
        self.assertEqual(token_dims, [[], [128], [4096]])

    def test_unmarked_traces_every_bucket(self) -> None:
        inner = self._run([(64, 64), (128, 64), (4096, 64)], mark=False)
        self.assertEqual(symbolic_warmup.STATS.symbolic_traces, 0)
        self.assertEqual(len(inner.shapes), 3)
        self.assertEqual(symbolic_warmup.STATS.dynamo_traces, 3)

    def test_new_shape_after_freeze_fails(self) -> None:
        inner = _StaticRecorder()
        compiled = torch.compile(
            _forward, backend=PerShapeBackend(inner), fullgraph=True, dynamic=False
        )
        weight = torch.randn(100, 8)
        batch = _Batch(64, 64)
        mark_bucket_dims_dynamic((64, 64), objects=(batch,))
        compiled(weight, batch)
        torch.compiler.set_stance("fail_on_recompile")
        try:
            compiled(weight, batch)  # warmed shape still runs
            with self.assertRaisesRegex(RuntimeError, "fail_on_recompile"):
                compiled(weight, _Batch(256, 64))
        finally:
            torch.compiler.set_stance("default")

    def test_marking_skips_size_one_and_unrelated_dims(self) -> None:
        batch = _Batch(64, 1)
        batch.other = torch.zeros(5)
        marked = mark_bucket_dims_dynamic((64, 1), objects=(batch,))
        # input_ids and positions (64); seq_lens (1) and cu_q_lens (2) are not.
        self.assertEqual(marked, 2)


if __name__ == "__main__":
    unittest.main()
