"""Unit tests for the TPU GDN attention backend's metadata (CPU-runnable).

These cover the pure-Python/torch part of
``sglang.srt.hardware_backend.tpu.attention_gdn_pallas.TpuGDNAttnBackend`` that
needs no JAX / torch_tpu / TPU device: the ``distribution`` dispatch vector the
fused kernel keys on.

``distribution = [decode_end, prefill_end, mixed_end]`` decides how the kernel
splits a ragged batch into decode / prefill regions. A wrong vector silently
routes tokens through the wrong phase (plausible-but-wrong output, not a crash),
so it is worth pinning directly: ``[bs, bs, bs]`` for a pure-decode batch,
``[0, bs, bs]`` for extend/prefill (matching the RPA v3 backend's convention),
int32, and cached per ``(is_decode, bs, device)``.

Kernel/op numerics (including mixed prefill+decode batches) are gated on TPU by
``test/registered/tpu/test_gdn_attention_v3.py`` (kernel vs. eager reference)
and ``test_gdn_attention_op.py`` (bridge op + state write-back). This file only
pins the CPU-side metadata contract.

The backend's ``__init__`` builds the JAX op eagerly and needs a ``ModelRunner``;
we instantiate a thin subclass that skips both — ``_get_distribution`` only
touches ``self._distribution_cache``.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

import unittest

import torch

from sglang.srt.hardware_backend.tpu.attention_gdn_pallas import TpuGDNAttnBackend
from sglang.test.test_utils import CustomTestCase


class _BackendNoInit(TpuGDNAttnBackend):
    """TpuGDNAttnBackend without the JAX-op build / ModelRunner dependency.

    ``_get_distribution`` reads only ``self._distribution_cache``, so a bare
    instance with that dict is enough to exercise the metadata logic on CPU.
    """

    def __init__(self):  # noqa: D107  (deliberately skips super().__init__)
        self._distribution_cache = {}


class TestGDNDistribution(CustomTestCase):
    def setUp(self):
        self.backend = _BackendNoInit()
        self.device = torch.device("cpu")

    def test_decode_distribution(self):
        # Pure-decode batch: every sequence is one step -> [bs, bs, bs].
        for bs in (1, 4, 37):
            dist = self.backend._get_distribution(True, bs, self.device)
            self.assertEqual(dist.tolist(), [bs, bs, bs])
            self.assertEqual(dist.dtype, torch.int32)
            self.assertEqual(dist.device.type, "cpu")

    def test_extend_distribution(self):
        # Extend/prefill batch: no BATCHED decode seqs -> [0, bs, bs].
        for bs in (1, 4, 37):
            dist = self.backend._get_distribution(False, bs, self.device)
            self.assertEqual(dist.tolist(), [0, bs, bs])
            self.assertEqual(dist.dtype, torch.int32)

    def test_decode_and_extend_differ(self):
        bs = 8
        decode = self.backend._get_distribution(True, bs, self.device)
        extend = self.backend._get_distribution(False, bs, self.device)
        self.assertNotEqual(decode.tolist(), extend.tolist())

    def test_cached_identity_per_key(self):
        # Same (is_decode, bs, device) returns the cached tensor (not rebuilt);
        # a different key builds a distinct one.
        a = self.backend._get_distribution(True, 4, self.device)
        b = self.backend._get_distribution(True, 4, self.device)
        self.assertIs(a, b)
        c = self.backend._get_distribution(True, 5, self.device)
        self.assertIsNot(a, c)
        d = self.backend._get_distribution(False, 4, self.device)
        self.assertIsNot(a, d)


if __name__ == "__main__":
    unittest.main()
