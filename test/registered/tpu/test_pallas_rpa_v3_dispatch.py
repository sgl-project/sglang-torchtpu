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
  * ``set_active_attention_backend("pallas_tpu_rpa_v3")`` +
    ``_patch_mha_kv_pool_for_rpa_v3()`` swaps the pool class in
    ``model_runner_kv_cache_mixin`` to :class:`TpuFusedMHATokenToKVPool`.
  * Same patch with the flash backend active is a no-op — the flash
    backend would mis-read the fused layout, so the swap must NOT fire.

These tests run on CPU (no torch_tpu / JAX needed) — they only exercise
the patch-and-dispatch glue.
"""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=120, suite="stage-a-test-tpu-unit")

import importlib
import unittest

from sglang.test.test_utils import CustomTestCase


def _reload_patches_module():
    """Reload the patches module so module-level state resets between tests."""
    import sglang.srt.hardware_backend.tpu.patches as patches_mod

    return importlib.reload(patches_mod)


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


class TestMHAKVPoolSwapForRpaV3(CustomTestCase):
    """``_patch_mha_kv_pool_for_rpa_v3`` must swap ``MHATokenToKVPool`` in
    the mixin's namespace iff the active backend is the RPA v3 one. Any
    other backend (flash, torch_native) needs the original flat pool."""

    def setUp(self):
        # Reload patches so its ``_active_attention_backend`` /
        # ``_mha_kv_pool_patched`` module globals reset between tests.
        self.patches_mod = _reload_patches_module()

        import sglang.srt.model_executor.model_runner_kv_cache_mixin as mix
        from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

        # Restore the stock pool symbol if a prior test left it swapped.
        mix.MHATokenToKVPool = MHATokenToKVPool
        self.mix = mix
        self.original_pool_cls = MHATokenToKVPool

    def test_swap_when_active_backend_is_rpa_v3(self):
        from sglang.srt.hardware_backend.tpu.kv_pool import (
            TpuFusedMHATokenToKVPool,
        )

        self.patches_mod.set_active_attention_backend("pallas_tpu_rpa_v3")
        self.patches_mod._patch_mha_kv_pool_for_rpa_v3()

        self.assertIs(self.mix.MHATokenToKVPool, TpuFusedMHATokenToKVPool)

        # Idempotent: a second call must not undo the swap or raise.
        self.patches_mod._patch_mha_kv_pool_for_rpa_v3()
        self.assertIs(self.mix.MHATokenToKVPool, TpuFusedMHATokenToKVPool)

    def test_no_swap_when_active_backend_is_flash(self):
        self.patches_mod.set_active_attention_backend("pallas_tpu_flash")
        self.patches_mod._patch_mha_kv_pool_for_rpa_v3()

        # The flash backend reads K and V via the flat layout; the fused
        # pool would silently mis-feed it. The patch must leave the symbol
        # as the upstream class.
        self.assertIs(self.mix.MHATokenToKVPool, self.original_pool_cls)

    def test_no_swap_when_active_backend_is_torch_native(self):
        self.patches_mod.set_active_attention_backend("torch_native")
        self.patches_mod._patch_mha_kv_pool_for_rpa_v3()

        self.assertIs(self.mix.MHATokenToKVPool, self.original_pool_cls)

    def test_no_swap_when_active_backend_unset(self):
        # ``apply_server_args_defaults`` is the only thing that populates
        # the active-backend flag. If a code path imports patches without
        # going through the platform (e.g. a unit test that skips the
        # platform's startup hook), the swap must default to no-op so
        # nothing silently breaks for non-RPA-v3 users.
        # The reload in setUp already cleared the flag back to None.
        self.assertIsNone(self.patches_mod._active_attention_backend)
        self.patches_mod._patch_mha_kv_pool_for_rpa_v3()
        self.assertIs(self.mix.MHATokenToKVPool, self.original_pool_cls)


if __name__ == "__main__":
    unittest.main()
