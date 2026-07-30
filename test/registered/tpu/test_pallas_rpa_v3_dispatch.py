"""Regression tests for the TPU Pallas-backend selection (RPA v3 vs flash).

Two TPU attention backends share the ``pallas_tpu_*`` namespace:

* ``pallas_tpu_rpa_v3`` (default) — wraps the vendored
  :mod:`sglang.srt.layers.attention.pallas.ragged_paged_attention_v3` kernel
  and needs the fused 5D KV layout from
  :class:`sglang.srt.hardware_backend.tpu.kv_pool.TpuFusedMHATokenToKVPool`.
* ``pallas_tpu_flash`` — wraps ``flash_attention`` for prefill/extend and
  ``jax.experimental.pallas.ops.tpu.paged_attention`` for decode. Reads
  separate K/V from the standard :class:`MHATokenToKVPool`.

The behaviour being guarded here:

  * Both names are registered with the right factory class.
  * The TPU platform selects :class:`TpuFusedMHATokenToKVPool` for the RPA
    backends and the standard pool for flash and torch-native attention.

These tests run on CPU (no torch_tpu / JAX needed) — they only exercise
the patch-and-dispatch glue.
"""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=120, suite="stage-a-test-tpu-unit")

import types
import unittest
from unittest.mock import patch

from sglang.test.test_utils import CustomTestCase


class TestPallasBackendRegistration(CustomTestCase):
    """Both Pallas backends must be registered and resolve to distinct
    factory classes that match the documented design. Registration happens
    at attention_registry import time (same as ascend / intel_xpu)."""

    def test_both_backends_registered(self):
        from sglang.srt.layers.attention.attention_registry import (
            ATTENTION_BACKENDS,
        )

        self.assertIn("pallas_tpu_flash", ATTENTION_BACKENDS)
        self.assertIn("pallas_tpu_rpa_v3", ATTENTION_BACKENDS)
        self.assertTrue(callable(ATTENTION_BACKENDS["pallas_tpu_flash"]))
        self.assertTrue(callable(ATTENTION_BACKENDS["pallas_tpu_rpa_v3"]))
        # The two factories must be distinct — registering both names to the
        # same factory would silently make the flag meaningless.
        self.assertIsNot(
            ATTENTION_BACKENDS["pallas_tpu_flash"],
            ATTENTION_BACKENDS["pallas_tpu_rpa_v3"],
        )


class TestMHAKVPoolSelection(CustomTestCase):
    def _pool_cls_for(self, attention_backend):
        from sglang.srt.hardware_backend.tpu.tpu_platform import TpuSRTPlatform

        with patch(
            "sglang.srt.server_args.get_global_server_args",
            return_value=types.SimpleNamespace(attention_backend=attention_backend),
        ):
            return TpuSRTPlatform().get_mha_kv_pool_cls()

    def test_fused_pool_for_rpa_backends(self):
        from sglang.srt.hardware_backend.tpu.kv_pool import (
            TpuFusedMHATokenToKVPool,
        )

        for backend in ("pallas_tpu_rpa_v3", "pallas_tpu_rpa_batched"):
            with self.subTest(backend=backend):
                self.assertIs(self._pool_cls_for(backend), TpuFusedMHATokenToKVPool)

    def test_flat_pool_for_non_rpa_backends(self):
        from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

        for backend in ("pallas_tpu_flash", "torch_native"):
            with self.subTest(backend=backend):
                self.assertIs(self._pool_cls_for(backend), MHATokenToKVPool)


if __name__ == "__main__":
    unittest.main()
