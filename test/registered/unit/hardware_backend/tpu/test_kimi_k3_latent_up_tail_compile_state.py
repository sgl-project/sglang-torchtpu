"""Latent-tail bookkeeping must not invalidate a warmed Dynamo graph (CPU)."""

import unittest
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu.kimi_k3 import latent_up_tail as tail

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _two_aggregations(x):
    tail.record_aggregation(x + 1, 1)
    bank, count = tail._STATE.agg[-1]
    first = bank * count
    tail.record_aggregation(x + 3, 2)
    bank, count = tail._STATE.agg[-1]
    return first + bank * count


class TestLatentTailCompileState(unittest.TestCase):
    def test_repeated_shapes_reuse_graph_and_record_latest_bank(self):
        graphs = []

        def backend(graph, inputs):
            graphs.append(graph)
            return graph.forward

        torch._dynamo.reset()
        try:
            with patch.object(tail, "_STATE", tail._State()):
                forward = torch.compile(
                    _two_aggregations, backend=backend, fullgraph=True, dynamic=False
                )
                seen = set()
                # Start with an empty record, revisit the same shape after it
                # becomes populated, then alternate prefill and decode shapes.
                for index, rows in enumerate((2048, 2048, 4096, 2048, 1, 1, 2048)):
                    with self.subTest(rows=rows, call=index):
                        x = torch.arange(rows * 4, dtype=torch.float32).reshape(rows, 4)
                        x = x + index
                        result = forward(x)
                        torch.testing.assert_close(result, 3 * x + 7, rtol=0, atol=0)
                        bank, count = tail._STATE.agg[-1]
                        torch.testing.assert_close(bank, x + 3, rtol=0, atol=0)
                        self.assertEqual(count, 2)
                        seen.add(rows)
                        self.assertEqual(len(graphs), len(seen))
        finally:
            torch._dynamo.reset()


if __name__ == "__main__":
    unittest.main()
