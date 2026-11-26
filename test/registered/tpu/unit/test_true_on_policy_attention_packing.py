"""Packing invariance of the true on-policy attention op (sglang_tpu/rl/true_on_policy/attention_kernel.py)."""

import importlib.util
import unittest

from sglang.test.ci.ci_register import register_tpu_ci

# About 55 s of attention kernel compiles; the other kernel cases are in test_true_on_policy_kernels.py.
register_tpu_ci(
    est_time=70,
    suite="nightly-tpu-4-v7",
    nightly=True,
    chips=1,
    labels=["singlehost", "infra"],
)

import torch  # noqa: E402

from sglang.test.test_utils import CustomTestCase  # noqa: E402


def _tpu_available() -> bool:
    if importlib.util.find_spec("torch_tpu") is None:
        return False
    try:
        import torch_tpu  # noqa: F401

        torch.zeros(1).to("tpu")
        return True
    except Exception:
        return False


def _rnd(*shape, scale=1.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(*shape, generator=g) * scale).to(torch.bfloat16).to("tpu")


@unittest.skipIf(not _tpu_available(), "needs a TPU")
class TestTrueOnPolicyAttentionPacking(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        from sglang_tpu.rl import true_on_policy

        cls.k = true_on_policy

    def test_packed_attention_does_not_depend_on_packing(self):
        lens = (300, 700, 41)
        q, k, v = (
            _rnd(1041, 16, 128, scale=2.0),
            _rnd(1041, 8, 128, seed=1),
            _rnd(1041, 8, 128, seed=2),
        )

        def attend(*args):
            return self.k.packed_causal_attention(*args, 128**-0.5).cpu()

        cu = torch.tensor([0, 300, 1000, 1041], dtype=torch.int32, device="tpu")
        packed = attend(q, k, v, cu)
        # each sequence alone, padded to another length, with empty sequences after it
        start = 0
        for n in lens:
            sl = slice(start, start + n)
            pad = 1024 - n
            qs, ks, vs = (
                torch.nn.functional.pad(t[sl], (0, 0, 0, 0, 0, pad)) for t in (q, k, v)
            )
            cu_alone = torch.tensor([0, n, n, n], dtype=torch.int32, device="tpu")
            alone = attend(qs, ks, vs, cu_alone)[:n]
            self.assertTrue(torch.equal(alone, packed[sl]), f"sequence of {n} differs")
            start += n


if __name__ == "__main__":
    unittest.main()
