"""Typed VMEM arena views (stages/arena.py): buffers of different dtypes and
shapes placed in one raw scratch buffer must round-trip DMA -> load exactly,
feed MXU dots like ordinary scratch, and not clobber each other."""

import unittest

import ml_dtypes
import numpy as np

from sglang.test.test_utils import CustomTestCase

try:
    import jax
    import jax.numpy as jnp

    _TPU = any(d.platform == "tpu" for d in jax.devices())
except Exception:  # pragma: no cover
    _TPU = False


@unittest.skipUnless(_TPU, "needs a TPU")
class TestArena(CustomTestCase):
    def test_views_roundtrip_and_dot(self):
        from jax.experimental import pallas as pl
        from jax.experimental.pallas import tpu as pltpu

        from sglang_tpu.k3_megakernel.kernel.stages.arena import Arena

        rng = np.random.default_rng(0)
        w_bf = rng.standard_normal((3, 256, 768)).astype(ml_dtypes.bfloat16)
        w_f8 = rng.standard_normal((2, 3584, 768)).astype(ml_dtypes.float8_e4m3fn)
        w_u8 = rng.integers(0, 255, (2, 12, 3584)).astype(np.uint8)
        w_f32 = rng.standard_normal((2, 1, 768)).astype(np.float32)
        x = rng.standard_normal((1, 256)).astype(ml_dtypes.bfloat16)

        arena = Arena()
        # Two overlapping generations: (bf16 ring) reuses the bytes of (f8, u8, f32) after they are read.
        f8_at, f8_end = arena.place(0, w_f8.shape, w_f8.dtype)
        u8_at, u8_end = arena.place(f8_end, w_u8.shape, w_u8.dtype)
        f32_at, _ = arena.place(u8_end, w_f32.shape, w_f32.dtype)
        bf_at, _ = arena.place(4096 * 3, w_bf.shape, w_bf.dtype)

        def kernel(
            bf_h, f8_h, u8_h, f32_h, x_ref, o_bf, o_f8, o_u8, o_f32, o_dot, ar, sem
        ):
            views = [
                Arena.view(ar, at, a.shape, a.dtype)
                for at, a in ((f8_at, w_f8), (u8_at, w_u8), (f32_at, w_f32))
            ]
            for i, (h, v) in enumerate(zip((f8_h, u8_h, f32_h), views)):
                pltpu.make_async_copy(h, v, sem.at[i]).start()
            for i, (h, v) in enumerate(zip((f8_h, u8_h, f32_h), views)):
                pltpu.make_async_copy(h, v, sem.at[i]).wait()
            o_f8[...] = views[0][...]
            o_u8[...] = views[1][...]
            o_f32[...] = views[2][...]
            ring = Arena.view(ar, bf_at, w_bf.shape, w_bf.dtype)
            for i in range(3):
                pltpu.make_async_copy(bf_h.at[i], ring.at[i], sem.at[i]).start()
            for i in range(3):
                pltpu.make_async_copy(bf_h.at[i], ring.at[i], sem.at[i]).wait()
            o_bf[...] = ring[...]
            o_dot[...] = jax.lax.dot_general(
                x_ref[...],
                ring.at[1][...],
                (((1,), (0,)), ((), ())),
                preferred_element_type=jnp.float32,
            )

        hbm, vm = (
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.VMEM),
        )
        outs = pl.pallas_call(
            kernel,
            out_shape=tuple(
                jax.ShapeDtypeStruct(a.shape, a.dtype)
                for a in (w_bf, w_f8, w_u8, w_f32)
            )
            + (jax.ShapeDtypeStruct((1, 768), jnp.float32),),
            in_specs=[hbm] * 4 + [vm],
            out_specs=(vm,) * 5,
            scratch_shapes=[arena.scratch_shape(), pltpu.SemaphoreType.DMA((3,))],
        )(w_bf, w_f8, w_u8, w_f32, x)
        o_bf, o_f8, o_u8, o_f32, o_dot = (np.asarray(o) for o in outs)
        np.testing.assert_array_equal(o_f8.view(np.uint8), w_f8.view(np.uint8))
        np.testing.assert_array_equal(o_u8, w_u8)
        np.testing.assert_array_equal(o_f32, w_f32)
        np.testing.assert_array_equal(o_bf.view(np.uint16), w_bf.view(np.uint16))
        ref = x.astype(np.float32) @ w_bf[1].astype(np.float32)
        np.testing.assert_allclose(o_dot, ref, rtol=1e-5, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
