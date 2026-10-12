"""Unit tests for the TPU-native recurrent (GDN) state pool.

``TpuMambaPool`` only allocates plain torch tensors, so these tests run on
CPU (``device="cpu"``) and need no TPU — they pin the kernel-native buffer
layout and slot bookkeeping that the fused Pallas GDN kernel depends on:

* **one tensor per layer, never a stacked ``[num_layers, ...]`` buffer**, and
  ``at_layer_idx`` hands back a tensor that *owns its storage*. This is a
  performance invariant with no functional symptom, so nothing else catches a
  regression: torch_tpu's Pallas bridge writes donated state back with
  ``tensor.copy_(result)``, which XLA elides against a whole buffer but must
  lower to slice + dynamic-update-slice against a view of a stacked pool. On
  Qwen3.5-397B that view cost ~0.70 ms per decode step — 1.4x the GDN kernel
  itself — while producing byte-identical output;
* conv buffer is ``[size+1, K-1, conv_dim]`` per layer (trailing axes
  swapped vs. the GPU pool's ``[..., conv_dim, K-1]``);
* recurrent buffer is ``[size+1, n_v, d_k, d_v]`` per layer (swapped vs.
  the GPU ``[..., n_v, d_v, d_k]``);
* slot 0 is reserved; ``alloc``/``free`` cycle real slots ``1..size``;
* ``clear_slots`` is a no-op (kernel masks stale slots via
  ``has_initial_state``) and ``copy_from`` raises (radix CoW unsupported);
* ReplaySSM is disabled here, but its flags must still be *present* --
  the shared req-slot alloc path reads them without a hasattr guard.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

import unittest

import torch

from sglang.srt.configs.mamba_utils import (
    Mamba2CacheParams,
    Mamba2StateDType,
    Mamba2StateShape,
)
from sglang.srt.mem_cache.memory_pool import MambaPool
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.mamba_pool import TpuMambaPool


def _make_cache_params(n_kq, n_v, d_k, d_v, K, layers):
    shape = Mamba2StateShape.create(
        tp_world_size=1,
        intermediate_size=n_v * d_v,
        n_groups=n_kq,
        num_heads=n_v,
        head_dim=d_v,
        state_size=d_k,
        conv_kernel=K,
    )
    return Mamba2CacheParams(
        dtype=Mamba2StateDType(conv=torch.float32, temporal=torch.float32),
        layers=layers,
        shape=shape,
    )


class TestTpuMambaPool(CustomTestCase):
    N_KQ, N_V, D_K, D_V, K = 2, 8, 128, 128, 4
    LAYERS = [1, 3, 5]
    SIZE = 16

    def _pool(self):
        cp = _make_cache_params(
            self.N_KQ, self.N_V, self.D_K, self.D_V, self.K, self.LAYERS
        )
        return TpuMambaPool(
            size=self.SIZE,
            spec_state_size=0,
            cache_params=cp,
            mamba_layer_ids=self.LAYERS,
            device="cpu",
        )

    def test_disabled_replayssm_surface_covers_base(self):
        """Disabling ReplaySSM must mean "present and inert", not "absent".

        ``HybridReqToTokenPool.alloc`` resets the ReplaySSM ring on every
        freshly assigned slot and reads the pool's flags with bare attribute
        access -- ``self.mamba_pool.replayssm_cache_base is not None`` -- with
        no hasattr guard. A pool that simply omits an attribute it does not
        use therefore raises AttributeError on the first prefill of any hybrid
        model, long after startup has reported success.

        Comparing against a live ``MambaPool`` rather than a hardcoded name
        list is deliberate: the failure mode this guards is "the base grew
        another disabled-state flag and the TPU pool was not updated", which a
        fixed list would not catch.
        """
        cp = _make_cache_params(
            self.N_KQ, self.N_V, self.D_K, self.D_V, self.K, self.LAYERS
        )
        common = dict(
            size=self.SIZE,
            spec_state_size=0,
            cache_params=cp,
            mamba_layer_ids=self.LAYERS,
            device="cpu",
        )
        base_attrs = {n for n in vars(MambaPool(**common)) if "replayssm" in n}
        tpu_attrs = {n for n in vars(TpuMambaPool(**common)) if "replayssm" in n}
        self.assertTrue(
            base_attrs,
            "no ReplaySSM attributes found on MambaPool -- this test has "
            "drifted from the base pool and no longer guards anything.",
        )
        self.assertEqual(
            set(),
            base_attrs - tpu_attrs,
            "TpuMambaPool is missing ReplaySSM attributes that MambaPool "
            "defines; unconditional readers such as "
            "HybridReqToTokenPool.alloc will raise AttributeError.",
        )

    def test_kernel_native_buffer_layout(self):
        pool = self._pool()
        num_layers = len(self.LAYERS)
        conv_dim = self.N_KQ * self.D_K * 2 + self.N_V * self.D_V
        # One tensor per layer: the layer axis is the list index, not a
        # tensor dimension.
        self.assertEqual(len(pool.mamba_cache.conv[0]), num_layers)
        self.assertEqual(len(pool.mamba_cache.temporal), num_layers)
        for layer in range(num_layers):
            self.assertEqual(
                list(pool.mamba_cache.conv[0][layer].shape),
                [self.SIZE + 1, self.K - 1, conv_dim],
            )
            self.assertEqual(
                list(pool.mamba_cache.temporal[layer].shape),
                [self.SIZE + 1, self.N_V, self.D_K, self.D_V],
            )

    def test_layers_are_independent_allocations(self):
        """Every layer's state must own its buffer.

        A stacked ``[num_layers, ...]`` pool passes every *functional* test in
        this file while silently costing ~0.70 ms/decode step, so the allocation
        shape has to be asserted directly. ``_base`` is what discriminates:
        distinct ``data_ptr()`` does not, because iterating a stacked tensor
        also yields one view per layer, each at its own offset.
        """
        pool = self._pool()
        for layer, temporal in enumerate(pool.mamba_cache.temporal):
            self.assertIsNone(temporal._base, f"recurrent layer {layer} is a view")
        for kind, conv_kind in enumerate(pool.mamba_cache.conv):
            for layer, conv in enumerate(conv_kind):
                self.assertIsNone(conv._base, f"conv[{kind}] layer {layer} is a view")

    def test_per_layer_cache_owns_its_storage(self):
        pool = self._pool()
        conv_dim = self.N_KQ * self.D_K * 2 + self.N_V * self.D_V
        # dense layer index 0 corresponds to global layer id LAYERS[0].
        per = pool.mamba2_layer_cache(0)
        self.assertEqual(list(per.conv[0].shape), [self.SIZE + 1, self.K - 1, conv_dim])
        self.assertEqual(
            list(per.temporal.shape), [self.SIZE + 1, self.N_V, self.D_K, self.D_V]
        )
        # The invariant the Pallas bridge's in-place write-back depends on:
        # a whole buffer, not a view into a stacked pool. `_base is None` is
        # exactly what distinguishes the two.
        self.assertIsNone(per.temporal._base)
        self.assertIsNone(per.conv[0]._base)
        # ...and it must still be the pool's own tensor, not a copy, or the
        # kernel's state write-back would land nowhere.
        self.assertIs(per.temporal, pool.mamba_cache.temporal[0])
        per.temporal.fill_(4.0)
        self.assertTrue(torch.all(pool.mamba_cache.temporal[0] == 4.0))
        self.assertTrue(torch.all(pool.mamba_cache.temporal[1] == 0.0))

    def test_total_bytes_match_stacked_layout(self):
        """Per-layer allocation must not change the memory footprint."""
        pool = self._pool()
        num_layers = len(self.LAYERS)
        conv_dim = self.N_KQ * self.D_K * 2 + self.N_V * self.D_V
        expected = (
            num_layers
            * (self.SIZE + 1)
            * 4
            * ((self.K - 1) * conv_dim + self.N_V * self.D_K * self.D_V)
        )
        self.assertEqual(int(pool.mamba_cache.mem_usage_bytes()), expected)

    def test_clear_slots_is_noop(self):
        pool = self._pool()
        for temporal in pool.mamba_cache.temporal:
            temporal.fill_(7.0)
        pool.clear_slots(torch.tensor([1, 2]))
        # Kernel handles fresh-slot masking; pool must not touch the buffer.
        self.assertTrue(all(torch.all(t == 7.0) for t in pool.mamba_cache.temporal))

    def test_copy_from_unsupported(self):
        pool = self._pool()
        with self.assertRaises(NotImplementedError):
            pool.copy_from(torch.tensor([1]), torch.tensor([2]))

    def test_speculative_unsupported(self):
        cp = _make_cache_params(
            self.N_KQ, self.N_V, self.D_K, self.D_V, self.K, self.LAYERS
        )
        with self.assertRaises(NotImplementedError):
            TpuMambaPool(
                size=self.SIZE,
                spec_state_size=4,
                cache_params=cp,
                mamba_layer_ids=self.LAYERS,
                device="cpu",
                speculative_num_draft_tokens=2,
            )


if __name__ == "__main__":
    unittest.main()
