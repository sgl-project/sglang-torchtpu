"""Unit tests for the TPU-native recurrent (GDN) state pool.

``TpuMambaPool`` only allocates plain torch tensors, so these tests run on
CPU (``device="cpu"``) and need no TPU — they pin the kernel-native buffer
layout and slot bookkeeping that the fused Pallas GDN kernel depends on:

* conv buffer is ``[num_layers, size+1, K-1, conv_dim]`` (trailing axes
  swapped vs. the GPU pool's ``[..., conv_dim, K-1]``);
* recurrent buffer is ``[num_layers, size+1, n_v, d_k, d_v]`` (swapped vs.
  the GPU ``[..., n_v, d_v, d_k]``);
* slot 0 is reserved; ``alloc``/``free`` cycle real slots ``1..size``;
* ``clear_slots`` is a no-op (kernel masks stale slots via
  ``has_initial_state``) and ``copy_from`` raises (radix CoW unsupported).
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
from sglang.srt.hardware_backend.tpu.mamba_pool import TpuMambaPool
from sglang.test.test_utils import CustomTestCase


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

    def test_kernel_native_buffer_layout(self):
        pool = self._pool()
        num_layers = len(self.LAYERS)
        conv_dim = self.N_KQ * self.D_K * 2 + self.N_V * self.D_V
        self.assertEqual(
            list(pool.mamba_cache.conv[0].shape),
            [num_layers, self.SIZE + 1, self.K - 1, conv_dim],
        )
        self.assertEqual(
            list(pool.mamba_cache.temporal.shape),
            [num_layers, self.SIZE + 1, self.N_V, self.D_K, self.D_V],
        )

    def test_per_layer_cache_slice(self):
        pool = self._pool()
        conv_dim = self.N_KQ * self.D_K * 2 + self.N_V * self.D_V
        # dense layer index 0 corresponds to global layer id LAYERS[0].
        per = pool.mamba2_layer_cache(0)
        self.assertEqual(
            list(per.conv[0].shape), [self.SIZE + 1, self.K - 1, conv_dim]
        )
        self.assertEqual(
            list(per.temporal.shape), [self.SIZE + 1, self.N_V, self.D_K, self.D_V]
        )

    def test_clear_slots_is_noop(self):
        pool = self._pool()
        pool.mamba_cache.temporal.fill_(7.0)
        pool.clear_slots(torch.tensor([1, 2]))
        # Kernel handles fresh-slot masking; pool must not touch the buffer.
        self.assertTrue(torch.all(pool.mamba_cache.temporal == 7.0))

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
