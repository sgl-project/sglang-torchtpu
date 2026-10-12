"""Correctness tests for the TPU ``aten::masked_scatter_`` fallback kernel.

torch_tpu cannot lower ``masked_scatter_`` — its source-offset prefix scan runs
in int64, which XLA:TPU can't materialize in vector registers ("Unsupported
conversion vmreg/vreg to U64"). ``hardware_backend/tpu/patches.py`` registers a
``PrivateUse1`` kernel that does the scan in int32 instead.

These tests are differential: the TPU kernel's output must equal PyTorch's own
CPU ``masked_scatter_`` (the ground truth) for every case, across dtypes, mask
patterns, broadcasting, source lengths, and edge cases (plus a randomized
fuzz). Requires a real TPU device; skipped otherwise.

Run standalone with
``python -m unittest test.registered.tpu.test_masked_scatter_tpu`` on a TPU host.
"""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=120, suite="stage-a-test-tpu-unit", chips=1)

import unittest

import torch

from sglang.test.test_utils import CustomTestCase


class TestMaskedScatterTpu(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch_tpu  # noqa: F401

            from sglang_tpu import patches
            from sglang_tpu.env_override import (
                apply_tpu_env_overrides,
            )

            # Serving applies these in _prepare_tpu_env(); this test reaches
            # torch_tpu directly, so it opts in itself. Must precede the probe:
            # libtpu reads LIBTPU_INIT_ARGS when PJRT starts, at the first device op.
            apply_tpu_env_overrides()

            cls.dev = "tpu"
            torch.zeros(1, device=cls.dev)  # probe: a real TPU is present
            patches._register_aten_fallbacks()  # installs the kernel under test
        except Exception as e:  # no torch_tpu / no device -> skip, don't fail
            raise unittest.SkipTest(f"TPU not available: {e}")

    def _assert_matches_cpu(self, base, mask, source):
        """TPU masked_scatter_ must match CPU masked_scatter_ bit-for-bit."""
        ref = base.clone()
        ref.masked_scatter_(mask, source)
        got = base.to(self.dev)
        got.masked_scatter_(mask.to(self.dev), source.to(self.dev))
        got = got.cpu()
        self.assertTrue(
            torch.equal(got.float(), ref.float()),
            f"TPU != CPU for base={tuple(base.shape)}/{base.dtype} "
            f"mask={tuple(mask.shape)} source={tuple(source.shape)}",
        )

    def test_dtypes_row_wise(self):
        # The multimodal-embed case: a [N,1] mask broadcast across the hidden dim.
        N, H, K = 64, 512, 40
        for dt in (torch.bfloat16, torch.float32, torch.int32):
            mask = torch.zeros(N, 1, dtype=torch.bool)
            mask[torch.randperm(N)[:K]] = True
            src = (torch.arange(K * H) % 97).reshape(K, H).to(dt)
            self._assert_matches_cpu(
                torch.zeros(N, H, dtype=dt), mask.expand(N, H), src
            )

    def test_mask_patterns(self):
        # general 2D
        self._assert_matches_cpu(
            torch.randn(32, 48), torch.rand(32, 48) > 0.5, torch.randn(32 * 48)
        )
        # 3D
        self._assert_matches_cpu(
            torch.randn(4, 8, 16), torch.rand(4, 8, 16) > 0.7, torch.randn(4 * 8 * 16)
        )
        # broadcast mask [1,H] over [N,H]
        self._assert_matches_cpu(
            torch.randn(16, 32), torch.rand(1, 32) > 0.5, torch.randn(16 * 32)
        )
        # all-True (full copy)
        self._assert_matches_cpu(
            torch.zeros(20), torch.ones(20, dtype=torch.bool), torch.arange(20.0)
        )
        # all-False (no-op)
        self._assert_matches_cpu(
            torch.arange(20.0), torch.zeros(20, dtype=torch.bool), torch.randn(5)
        )

    def test_edge_cases(self):
        # source longer than needed: only the first mask.sum() elements are used.
        mask = torch.zeros(10, dtype=torch.bool)
        mask[[1, 3, 5]] = True
        self._assert_matches_cpu(torch.zeros(10), mask, torch.arange(100.0))
        # all-False + EMPTY source -> no-op (regression guard: used to IndexError).
        self._assert_matches_cpu(
            torch.arange(20.0), torch.zeros(20, dtype=torch.bool), torch.randn(0)
        )
        # single element
        self._assert_matches_cpu(
            torch.zeros(1), torch.tensor([True]), torch.tensor([7.0])
        )

    def test_empty_source_with_true_mask_raises(self):
        # Empty source + a True mask is invalid input; both CPU and the TPU
        # kernel must raise (the kernel must not silently no-op).
        base = torch.arange(20.0)
        mask = torch.zeros(20, dtype=torch.bool)
        mask[3] = True
        source = torch.randn(0)
        with self.assertRaises(RuntimeError):
            base.clone().masked_scatter_(mask, source)
        with self.assertRaises(RuntimeError):
            base.to(self.dev).masked_scatter_(mask.to(self.dev), source.to(self.dev))

    def test_large_count_exceeds_float32(self):
        # More than 2**24 True elements: the index scan must stay int-exact.
        # A float32 scan would round here and scatter the wrong source values;
        # int32 is exact. Guards against regressing back to a float32 scan.
        n = 2**24 + 50_000
        source = torch.arange(n, dtype=torch.int32) % 997
        self._assert_matches_cpu(
            torch.zeros(n, dtype=torch.int32), torch.ones(n, dtype=torch.bool), source
        )

    def test_fuzz(self):
        torch.manual_seed(0)
        for _ in range(40):
            ndim = int(torch.randint(1, 4, (1,)).item())
            dims = torch.randint(1, 20, (ndim,)).tolist()
            base = torch.randn(dims)
            mask = torch.rand(dims) > float(torch.rand(1).item())
            n_true = int(mask.sum())
            source = torch.randn(n_true + int(torch.randint(0, 5, (1,)).item()))
            self._assert_matches_cpu(base, mask, source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
