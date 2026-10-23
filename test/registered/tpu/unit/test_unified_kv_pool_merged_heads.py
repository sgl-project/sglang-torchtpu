"""The unified KV pool's merged sliding-window heads on TPU.

merge_swa_kv_heads and split_swa_output must give on TPU, eagerly and under
torch.compile, what they give on CPU: torch_tpu returned wrong values for an
earlier split that stacked sliced views.
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=60, suite="stage-a-test-tpu-unit")


def _has_torch_tpu() -> bool:
    try:
        import torch_tpu  # noqa: F401
    except Exception:
        return False
    return True


@unittest.skipIf(not _has_torch_tpu(), "Requires torch_tpu")
class TestMergedSlidingWindowHeadsOnTpu(unittest.TestCase):
    def test_merge_and_split_match_cpu(self):
        from sglang_tpu.unified_kv_heads import merge_swa_kv_heads, split_swa_output

        torch.manual_seed(0)
        tokens, q_heads, kv_heads, head_dim = 40, 4, 2, 256
        q = torch.randn(tokens, q_heads, head_dim, dtype=torch.bfloat16)
        k = torch.randn(tokens, kv_heads, head_dim, dtype=torch.bfloat16)
        v = torch.randn(tokens, kv_heads, head_dim, dtype=torch.bfloat16)
        out = torch.randn(tokens, q_heads, kv_heads * head_dim, dtype=torch.bfloat16)

        def both(q, k, v, out):
            return (*merge_swa_kv_heads(q, k, v), split_swa_output(out, kv_heads))

        expected = both(q, k, v, out)
        on_tpu = [x.to("tpu") for x in (q, k, v, out)]
        torch._dynamo.reset()
        compiled = torch.compile(both, backend="tpu", fullgraph=True, dynamic=False)
        for results in (both(*on_tpu), compiled(*on_tpu)):
            for got, want in zip(results, expected):
                torch.testing.assert_close(got.cpu(), want, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
