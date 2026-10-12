"""Regression tests for the TPU Pallas-backend selection (RPA v3 vs flash).

Two TPU attention backends share the ``pallas_tpu_*`` namespace:

* ``pallas_tpu_rpa_v3`` (default) — wraps the vendored
  :mod:`sglang_tpu.attention.pallas.ragged_paged_attention_v3` kernel
  and needs the fused 5D KV layout from
  :class:`sglang_tpu.kv_pool.TpuFusedMHATokenToKVPool`.
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

register_tpu_ci(est_time=120, suite="stage-a-test-tpu-unit", chips=1)

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
        from sglang.srt.runtime_context import get_context
        from sglang_tpu.tpu_platform import TpuSRTPlatform

        # get_mha_kv_pool_cls reads the exec bag; publish the backend.
        with get_context().override_server_args(attention_backend=attention_backend):
            return TpuSRTPlatform().get_mha_kv_pool_cls()

    def test_fused_pool_for_rpa_backends(self):
        from sglang_tpu.kv_pool import (
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


class TestBatchedRpaSpecBackend(CustomTestCase):
    def _factory(self):
        from sglang.srt.speculative.draft_utils import DraftBackendFactory

        self.backend_context = patch(
            "sglang.srt.speculative.draft_utils.attention_backends",
            return_value=("pallas_tpu_rpa_batched", "pallas_tpu_rpa_batched"),
        )
        self.backend_context.start()
        self.addCleanup(self.backend_context.stop)
        spec_context = patch(
            "sglang.srt.speculative.draft_utils.get_spec",
            return_value=types.SimpleNamespace(speculative_attention_mode="prefill"),
        )
        spec_context.start()
        self.addCleanup(spec_context.stop)
        return DraftBackendFactory(
            types.SimpleNamespace(
                draft_attention_backend=None,
                model_config=types.SimpleNamespace(
                    hf_config=object(), hf_text_config=object()
                ),
            ),
            topk=1,
            speculative_num_steps=3,
        )

    def test_decode_backend_registered(self):
        factory = self._factory()
        expected = types.SimpleNamespace(attn_backends=[])
        with patch.object(
            factory,
            "_create_tpu_pallas_rpa_batched_decode_backend",
            return_value=("tpu_pallas_rpa_batched", expected),
        ):
            self.assertIs(factory.create_decode_backend(), expected)

    def test_draft_extend_backend_registered(self):
        factory = self._factory()
        expected = types.SimpleNamespace(attn_backends=[])
        with patch.object(
            factory,
            "_create_tpu_pallas_rpa_batched_prefill_backend",
            return_value=("tpu_pallas_rpa_batched", expected),
        ):
            self.assertIs(factory.create_draft_extend_backend(), expected)


if __name__ == "__main__":
    unittest.main()
