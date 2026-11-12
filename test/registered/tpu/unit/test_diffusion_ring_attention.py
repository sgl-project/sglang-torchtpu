# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the TPU Pallas diffusion ring attention kernel and tile selector."""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=45, suite="stage-a-test-tpu-unit", chips=1)

import functools
import importlib.util
import itertools
import os
import subprocess
import sys
import unittest

if "--xla_force_host_platform_device_count=" not in os.environ.get("XLA_FLAGS", ""):
    os.environ["XLA_FLAGS"] = (
        f"{os.environ.get('XLA_FLAGS', '')} --xla_force_host_platform_device_count=8"
    ).strip()

import numpy as np

from sglang_tpu.multimodal_gen import diffusion_ring_attention


def _has_jax() -> bool:
    return importlib.util.find_spec("jax") is not None


def _has_tpu_v7x_pair() -> bool:
    if not _has_jax():
        return False
    import jax

    return (
        jax.default_backend() == "tpu"
        and len(jax.devices()) >= 2
        and "7x" in jax.devices()[0].device_kind.lower()
    )


def _fp32_attention_reference(q_bf16, k_bf16, v_bf16, scale: float) -> np.ndarray:
    qf = np.asarray(q_bf16, dtype=np.float32).transpose(0, 2, 1, 3)
    rep = q_bf16.shape[2] // k_bf16.shape[2]
    kf = np.repeat(np.asarray(k_bf16, dtype=np.float32), rep, axis=2).transpose(
        0, 2, 1, 3
    )
    vf = np.repeat(np.asarray(v_bf16, dtype=np.float32), rep, axis=2).transpose(
        0, 2, 1, 3
    )
    scores = np.matmul(qf, np.swapaxes(kf, -1, -2)) * scale
    scores -= scores.max(axis=-1, keepdims=True)
    probs = np.exp(scores)
    probs /= probs.sum(axis=-1, keepdims=True)
    return np.matmul(probs, vf).transpose(0, 2, 1, 3)


class TestRingAttentionBlocks(unittest.TestCase):
    def test_tuned_wan22_blocks_match_table(self):
        blocks = diffusion_ring_attention.Blocks.for_shard(18900, 4, use_fixed_m=True)
        self.assertEqual(
            (
                blocks.block_q,
                blocks.block_kv,
                blocks.block_kv_compute,
                blocks.block_q_sub,
            ),
            (19200, 3328, 3328, 3200),
        )
        online = diffusion_ring_attention.Blocks.for_shard(18900, 4, use_fixed_m=False)
        self.assertEqual(
            (
                online.block_q,
                online.block_kv,
                online.block_kv_compute,
                online.block_q_sub,
            ),
            (19200, 2816, 2816, 3840),
        )

    def test_untuned_wan22_sweep_shapes_fit_vmem(self):
        # All [480p, 720p] x [1..15]s shards at ring=4 and ring=8 resolve valid tiles within v7x VMEM.
        for ring_size in (4, 8):
            for tokens_per_latent_frame in (390 * 4, 900 * 4):
                for sec in range(1, 16):
                    latent_frames = 4 * sec + 1
                    shard_len = (latent_frames * tokens_per_latent_frame) // ring_size
                    for use_fixed_m in (True, False):
                        blocks = diffusion_ring_attention.Blocks.for_shard(
                            shard_len, ring_size, use_fixed_m=use_fixed_m
                        )
                        q_sub = (
                            blocks.block_q_sub
                            or diffusion_ring_attention.auto_block_q_sub(blocks.block_q)
                        )
                        self.assertEqual(blocks.block_q % q_sub, 0)
                        self.assertEqual(q_sub % 128, 0)
                        self.assertEqual(blocks.block_kv % blocks.block_kv_compute, 0)
                        ubq, ubkv, ubkvc, uqs = (
                            diffusion_ring_attention.select_untuned_blocks(
                                shard_len, use_fixed_m=use_fixed_m
                            )
                        )
                        if shard_len > 34_000:
                            self.assertLess(ubq, shard_len)
                        single_buf = diffusion_ring_attention.is_single_buffered_q(
                            ubq, shard_len
                        )
                        est = diffusion_ring_attention.estimate_vmem_bytes(
                            ubq,
                            uqs,
                            ubkv,
                            ubkvc,
                            single_buffered_q=single_buf,
                        )
                        self.assertLessEqual(
                            est, diffusion_ring_attention.VMEM_LIMIT_BYTES
                        )

    def test_single_buffered_q_and_head_dim_scaling(self):
        self.assertFalse(diffusion_ring_attention.is_single_buffered_q(4608, 4500))
        self.assertTrue(diffusion_ring_attention.is_single_buffered_q(5120, 5070))
        self.assertFalse(diffusion_ring_attention.is_single_buffered_q(23040, 45900))
        v64 = diffusion_ring_attention.estimate_vmem_bytes(
            8192, 2048, 2560, head_dim=64
        )
        v128 = diffusion_ring_attention.estimate_vmem_bytes(
            8192, 2048, 2560, head_dim=128
        )
        self.assertLess(v64, v128)
        b64 = diffusion_ring_attention.Blocks.for_shard(
            5070, 4, head_dim=64, use_fixed_m=True
        )
        self.assertEqual(b64.block_q % (b64.block_q_sub or 128), 0)


@unittest.skipUnless(_has_jax(), "jax not installed")
class TestRingFixedMOperands(unittest.TestCase):
    def test_constant_row0_and_gate_eligibility(self):
        import jax.numpy as jnp

        qn = jnp.array([5.0, 12.0, 9.5], dtype=jnp.float32)
        kn = jnp.array([10.0, 10.0, 10.0], dtype=jnp.float32)
        mk = np.asarray(diffusion_ring_attention.ring_fixed_m_operands(qn, kn))
        self.assertEqual(mk.shape, (2, 3))
        np.testing.assert_allclose(mk[0], [95.0, 95.0, 95.0])
        np.testing.assert_allclose(mk[1], [1.0, 0.0, 1.0])

    def test_fixed_m_window_keeps_pv_normal(self):
        """A head at the fixed-m bound keeps exp2 arguments in [-102, 89): below -102,
        p * v with |v| down to 2**-24 would be subnormal, which TPU flushes to zero."""
        import jax.numpy as jnp

        bound = diffusion_ring_attention.FIXED_M_RING_SAFE_BOUND
        mk = np.asarray(
            diffusion_ring_attention.ring_fixed_m_operands(
                jnp.array([bound], jnp.float32), jnp.array([1.0], jnp.float32)
            )
        )
        self.assertEqual(mk[1, 0], 1.0)
        m = mk[0, 0] - diffusion_ring_attention.FIXED_M_RECENTER
        self.assertGreaterEqual(-bound - m, -102.0)
        self.assertLess(bound - m, 89.0)


def _assert_per_head_close(test, out, ref, tol, label):
    test.assertTrue(np.all(np.isfinite(out)), label)
    for head in range(ref.shape[2]):
        err = np.linalg.norm(out[:, :, head] - ref[:, :, head])
        rel_l2 = float(err / np.linalg.norm(ref[:, :, head]))
        test.assertLess(rel_l2, tol, f"{label} head {head}")


def _rerun_with_8_cpu_devices(test: unittest.TestCase) -> None:
    xla_flags = " ".join(
        tok
        for tok in os.environ.get("XLA_FLAGS", "").split()
        if not tok.startswith("--xla_force_host_platform_device_count=")
    )
    env = {
        **os.environ,
        "JAX_PLATFORMS": "cpu",
        "XLA_FLAGS": f"{xla_flags} --xla_force_host_platform_device_count=8".strip(),
    }
    test_name = f"{type(test).__name__}.{test._testMethodName}"
    proc = subprocess.run(
        [sys.executable, os.path.abspath(__file__), test_name],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    test.assertEqual(proc.returncode, 0, f"{proc.stdout}\n{proc.stderr}")


@unittest.skipUnless(_has_jax(), "jax not installed")
class TestRingAttentionInterpret(unittest.TestCase):
    def test_ring4_interpret_multiblock_matches_fp32_reference(self):
        """Ring 4 matches fp32 per head with fixed-m on and off; head 1's base-2 logits
        (~147) overflow exp2 if it is gated to fixed-m."""
        import jax

        cpu_devices = jax.devices("cpu")
        if len(cpu_devices) < 4:
            # Another test initialized JAX's CPU backend before XLA_FLAGS took effect.
            _rerun_with_8_cpu_devices(self)
            return

        import jax.numpy as jnp
        from jax._src.pallas.mosaic.interpret import interpret_pallas_call as ipc
        from jax.experimental.pallas import tpu as pltpu
        from jax.sharding import Mesh, NamedSharding, PartitionSpec

        rng = np.random.default_rng(0)
        ring_size = 4
        shard_len = 310
        seq_len = ring_size * shard_len
        head_dim = 128
        scale = head_dim**-0.5
        # ring_size=4 separates upstream from downstream. 310 rows with bq=256,
        # q_sub=128, bkv=128: 3 KV blocks with a ragged tail, 2 Q passes of 2 query
        # chunks, and the last KV block in the online body for fixed-m heads. Two KV
        # heads, so each query head reloads its own K/V.
        pinned = {
            "block_q": 256,
            "block_kv": 128,
            "block_kv_compute": 128,
            "block_q_sub": 128,
        }

        k_np = rng.standard_normal((1, seq_len, 2, head_dim), dtype=np.float32)
        v_np = rng.standard_normal((1, seq_len, 2, head_dim), dtype=np.float32)
        q_np = rng.standard_normal((1, seq_len, 2, head_dim), dtype=np.float32)
        q_np[:, :, 1, :] = 9.0 * k_np[:, :, 1, :]

        q_bf16 = jnp.asarray(q_np, dtype=jnp.bfloat16)
        k_bf16 = jnp.asarray(k_np, dtype=jnp.bfloat16)
        v_bf16 = jnp.asarray(v_np, dtype=jnp.bfloat16)
        ref = _fp32_attention_reference(q_bf16, k_bf16, v_bf16, scale)

        axis_names = ("batch", "ring", "uly")
        mesh = Mesh(
            np.asarray(cpu_devices[:ring_size]).reshape(1, ring_size, 1), axis_names
        )
        spec = PartitionSpec(*axis_names)
        sharding = NamedSharding(mesh, spec)

        def _put(x):
            reshaped = x.reshape(ring_size, shard_len, x.shape[2], head_dim)
            return jax.device_put(reshaped[None, :, None, None], sharding)

        outs = {}
        for use_fixed_m in (True, False):
            pltpu.reset_tpu_interpret_mode_state()
            # Both launches run the same DMA schedule; race-check one of them.
            interpret = pltpu.InterpretParams(detect_races=use_fixed_m)
            fn = jax.jit(
                jax.shard_map(
                    functools.partial(
                        diffusion_ring_attention.ring_attention_sharded,
                        ring_axis="ring",
                        axis_names=axis_names,
                        scale=scale,
                        use_fixed_m=use_fixed_m,
                        pinned=pinned,
                        interpret=interpret,
                    ),
                    mesh=mesh,
                    in_specs=(spec, spec, spec),
                    out_specs=spec,
                    check_vma=False,
                )
            )
            out = np.asarray(fn(_put(q_bf16), _put(k_bf16), _put(v_bf16)))
            outs[use_fixed_m] = out.astype(np.float32).reshape(1, seq_len, 2, head_dim)
            if use_fixed_m:
                self.assertFalse(
                    ipc.races is not None and ipc.races.races_found,
                    "Pallas interpret detected a DMA or semaphore race",
                )

        for use_fixed_m, out in outs.items():
            _assert_per_head_close(self, out, ref, 4e-3, f"fixed_m={use_fixed_m}")
        # Head 0 takes the fixed-m body only when fixed-m is on.
        self.assertFalse(np.array_equal(outs[True][:, :, 0], outs[False][:, :, 0]))


@unittest.skipUnless(_has_tpu_v7x_pair(), "requires TPU v7x with >= 2 TensorCores")
class TestRingAttentionOnDevice(unittest.TestCase):
    def test_sharded_ring_matches_fp32_reference(self):
        import jax
        import jax.numpy as jnp
        from jax.sharding import Mesh, NamedSharding, PartitionSpec

        rng = np.random.default_rng(0)
        shard_len = 390
        ring_size = 2
        seq_len = ring_size * shard_len
        head_dim = 128
        scale = head_dim**-0.5
        # Batch 2 exercises the batch term of the kernel's KV-head index; the pinned
        # tile adds mid-row K/V reloads (4 KV blocks), which read that index.
        batch = 2
        tiles = (
            None,
            {
                "block_q": 256,
                "block_kv": 128,
                "block_kv_compute": 128,
                "block_q_sub": 128,
            },
        )

        q_np = rng.standard_normal((batch, seq_len, 4, head_dim), dtype=np.float32)
        # Head 3 exceeds FIXED_M_RING_SAFE_BOUND so one launch exercises both fixed-m and online fallback.
        q_np[:, :, 3, :] *= 8.0
        k_np = rng.standard_normal((batch, seq_len, 2, head_dim), dtype=np.float32)
        v_np = rng.standard_normal((batch, seq_len, 2, head_dim), dtype=np.float32)

        q_bf16 = jnp.asarray(q_np, dtype=jnp.bfloat16)
        k_bf16 = jnp.asarray(k_np, dtype=jnp.bfloat16)
        v_bf16 = jnp.asarray(v_np, dtype=jnp.bfloat16)
        ref = _fp32_attention_reference(q_bf16, k_bf16, v_bf16, scale)

        axis_names = ("batch", "ring", "uly")
        mesh = Mesh(np.asarray(jax.devices()[:2]).reshape(1, 2, 1), axis_names)
        spec = PartitionSpec(*axis_names)
        sharding = NamedSharding(mesh, spec)

        def _put(x):
            blocks = x.reshape(batch, ring_size, shard_len, x.shape[2], head_dim)
            return jax.device_put(jnp.swapaxes(blocks, 0, 1)[None, :, None], sharding)

        q_in, k_in, v_in = _put(q_bf16), _put(k_bf16), _put(v_bf16)

        for pinned, use_fixed_m in itertools.product(tiles, (True, False)):
            with self.subTest(pinned=pinned, use_fixed_m=use_fixed_m):
                fn = jax.jit(
                    jax.shard_map(
                        functools.partial(
                            diffusion_ring_attention.ring_attention_sharded,
                            ring_axis="ring",
                            axis_names=axis_names,
                            scale=scale,
                            use_fixed_m=use_fixed_m,
                            pinned=pinned,
                        ),
                        mesh=mesh,
                        in_specs=(spec, spec, spec),
                        out_specs=spec,
                        check_vma=False,
                    )
                )
                out = np.asarray(fn(q_in, k_in, v_in), dtype=np.float32)[0, :, 0]
                out = np.swapaxes(out, 0, 1).reshape(batch, seq_len, 4, head_dim)
                _assert_per_head_close(self, out, ref, 4e-3, f"fixed_m={use_fixed_m}")


if __name__ == "__main__":
    unittest.main()
