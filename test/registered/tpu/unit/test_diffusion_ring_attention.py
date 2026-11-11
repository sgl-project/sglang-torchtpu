# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the TPU Pallas diffusion ring attention kernel and tile selector."""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=45, suite="stage-a-test-tpu-unit", chips=1)

import functools
import importlib.util
import math
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
        """Admitted heads keep every exp2 argument in [-102, 88].

        Below -102, p * v with |v| < 2**-24 is subnormal, which TPU flushes to
        zero: the output shrinks while l keeps the term. Above 88, l and o can
        overflow float32 when summed over ~2**17 keys.
        """
        bound = diffusion_ring_attention.FIXED_M_RING_SAFE_BOUND
        recenter = diffusion_ring_attention.FIXED_M_RECENTER
        m = math.floor(bound) - recenter
        self.assertLessEqual(bound - m, recenter + 1.0)
        self.assertGreaterEqual(-bound - m, -102.0)


def _assert_per_head_close(test, out, ref, tol, label):
    test.assertTrue(np.all(np.isfinite(out)), label)
    for head in range(ref.shape[2]):
        err = np.linalg.norm(out[:, :, head] - ref[:, :, head])
        rel_l2 = float(err / np.linalg.norm(ref[:, :, head]))
        test.assertLess(rel_l2, tol, f"{label} head {head}")


@unittest.skipUnless(_has_jax(), "jax not installed")
class TestRingAttentionInterpret(unittest.TestCase):
    def test_ring4_interpret_multiblock_matches_fp32_reference(self):
        """Ring 4 in Pallas interpret mode against an fp32 reference, per head.

        ring_size=4 separates upstream from downstream. shard_len=310 with
        bq=256, q_sub=256 and bkv=128 runs 3 KV blocks with a ragged tail, 2 Q
        passes and the n_q == 1 hybrid last body. GQA 2:1: head 0 is gated to
        fixed-m; head 1's query is 9x its key, so its base-2 logits (~147)
        would overflow exp2 if it were gated to fixed-m.
        """
        import jax

        cpu_devices = jax.devices("cpu")
        if len(cpu_devices) < 4:
            # Another test initialized JAX's CPU backend before XLA_FLAGS took effect.
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
            proc = subprocess.run(
                [
                    sys.executable,
                    os.path.abspath(__file__),
                    "TestRingAttentionInterpret.test_ring4_interpret_multiblock_matches_fp32_reference",
                ],
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, f"{proc.stdout}\n{proc.stderr}")
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
        pinned = {
            "block_q": 256,
            "block_kv": 128,
            "block_kv_compute": 128,
            "block_q_sub": 256,
        }

        k_np = rng.standard_normal((1, seq_len, 1, head_dim), dtype=np.float32)
        v_np = rng.standard_normal((1, seq_len, 1, head_dim), dtype=np.float32)
        q_np = rng.standard_normal((1, seq_len, 2, head_dim), dtype=np.float32)
        q_np[:, :, 1, :] = 9.0 * k_np[:, :, 0, :]

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
        # Head 0 takes the fixed-m body only when fixed-m is on; head 1 takes
        # the online body in both launches.
        self.assertFalse(np.array_equal(outs[True][:, :, 0], outs[False][:, :, 0]))
        np.testing.assert_array_equal(outs[True][:, :, 1], outs[False][:, :, 1])


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

        q_np = rng.standard_normal((1, seq_len, 4, head_dim), dtype=np.float32)
        # Head 3 exceeds FIXED_M_RING_SAFE_BOUND so one launch exercises both fixed-m and online fallback.
        q_np[:, :, 3, :] *= 8.0
        k_np = rng.standard_normal((1, seq_len, 2, head_dim), dtype=np.float32)
        v_np = rng.standard_normal((1, seq_len, 2, head_dim), dtype=np.float32)

        q_bf16 = jnp.asarray(q_np, dtype=jnp.bfloat16)
        k_bf16 = jnp.asarray(k_np, dtype=jnp.bfloat16)
        v_bf16 = jnp.asarray(v_np, dtype=jnp.bfloat16)
        ref = _fp32_attention_reference(q_bf16, k_bf16, v_bf16, scale)

        axis_names = ("batch", "ring", "uly")
        mesh = Mesh(np.asarray(jax.devices()[:2]).reshape(1, 2, 1), axis_names)
        spec = PartitionSpec(*axis_names)
        sharding = NamedSharding(mesh, spec)

        q_in = jax.device_put(
            q_bf16.reshape(1, ring_size, shard_len, 4, head_dim)[
                :, :, None, None, :, :, :
            ],
            sharding,
        )
        k_in = jax.device_put(
            k_bf16.reshape(1, ring_size, shard_len, 2, head_dim)[
                :, :, None, None, :, :, :
            ],
            sharding,
        )
        v_in = jax.device_put(
            v_bf16.reshape(1, ring_size, shard_len, 2, head_dim)[
                :, :, None, None, :, :, :
            ],
            sharding,
        )

        for use_fixed_m in (True, False):
            with self.subTest(use_fixed_m=use_fixed_m):
                fn = jax.jit(
                    jax.shard_map(
                        functools.partial(
                            diffusion_ring_attention.ring_attention_sharded,
                            ring_axis="ring",
                            axis_names=axis_names,
                            scale=scale,
                            use_fixed_m=use_fixed_m,
                        ),
                        mesh=mesh,
                        in_specs=(spec, spec, spec),
                        out_specs=spec,
                        check_vma=False,
                    )
                )
                out = np.asarray(fn(q_in, k_in, v_in), dtype=np.float32)[
                    0, :, 0, 0
                ].reshape(1, seq_len, 4, head_dim)
                _assert_per_head_close(self, out, ref, 4e-3, f"fixed_m={use_fixed_m}")


class TestRingMeshSpec(unittest.TestCase):
    def _check_spec(
        self,
        world_size: int,
        ring_ranks: list[int],
        replica_ranks: list[int] | None = None,
    ):
        from types import SimpleNamespace
        from unittest.mock import patch

        from sglang_tpu.multimodal_gen import tpu_splash_attn

        if replica_ranks is None:
            replica_ranks = list(range(world_size))
        replica = SimpleNamespace(
            ranks=replica_ranks,
            world_size=len(replica_ranks),
            device_group=None,
        )
        sp = SimpleNamespace(ring_world_size=len(ring_ranks), ring_group=object())
        with (
            patch("torch.distributed.is_initialized", return_value=True),
            patch.object(
                tpu_splash_attn, "model_parallel_is_initialized", return_value=True
            ),
            patch.object(tpu_splash_attn, "get_sp_group", return_value=sp),
            patch.object(tpu_splash_attn, "get_replica_group", return_value=replica),
            patch("torch.distributed.get_world_size", return_value=world_size),
            patch("torch.distributed.get_process_group_ranks", return_value=ring_ranks),
        ):
            return tpu_splash_attn._resolve_ring_mesh_spec()

    def test_regular_grids_resolve(self):
        self.assertEqual(self._check_spec(8, [0, 1, 2, 3]), (2, 4, 1))
        self.assertEqual(self._check_spec(8, [4, 5, 6, 7]), (2, 4, 1))
        self.assertEqual(self._check_spec(8, [0, 2]), (2, 2, 2))
        self.assertEqual(self._check_spec(8, [5, 7]), (2, 2, 2))
        self.assertEqual(self._check_spec(8, list(range(8))), (1, 8, 1))

    def test_replica_submesh_resolves_per_dp_replica(self):
        # dp_size=2 over world_size=8: each replica has 4 ranks.
        self.assertEqual(
            self._check_spec(8, [0, 1, 2, 3], replica_ranks=[0, 1, 2, 3]), (1, 4, 1)
        )
        self.assertEqual(
            self._check_spec(8, [4, 5, 6, 7], replica_ranks=[4, 5, 6, 7]), (1, 4, 1)
        )
        self.assertEqual(
            self._check_spec(8, [5, 7], replica_ranks=[4, 5, 6, 7]), (1, 2, 2)
        )
        # Ring ranks outside the replica group are rejected.
        self.assertIsNone(self._check_spec(8, [0, 1, 2, 3], replica_ranks=[4, 5, 6, 7]))

    def test_non_grid_or_trivial_returns_none(self):
        from sglang_tpu.multimodal_gen import tpu_splash_attn

        self.assertIsNone(tpu_splash_attn._resolve_ring_mesh_spec())
        self.assertIsNone(self._check_spec(8, [0]))
        self.assertIsNone(self._check_spec(8, [2, 4]))
        self.assertIsNone(self._check_spec(8, [0, 3, 5, 7]))


class TestRingAttentionDispatch(unittest.TestCase):
    def test_capability_flags(self):
        from sglang.multimodal_gen.runtime.layers.attention.backends.sdpa import (
            SDPAImpl,
        )
        from sglang_tpu.multimodal_gen.tpu_splash_attn import TpuSplashAttentionImpl

        self.assertTrue(TpuSplashAttentionImpl.supports_ring())
        self.assertFalse(SDPAImpl.supports_ring())

    def test_env_switch_disables_ring_kernel(self):
        from unittest.mock import patch

        from sglang_tpu.envs import envs as tpu_envs
        from sglang_tpu.multimodal_gen import tpu_splash_attn
        from sglang_tpu.multimodal_gen.tpu_splash_attn import TpuSplashAttentionImpl

        with (
            tpu_envs.SGLANG_TPU_ENABLE_DIFFUSION_RING_KERNEL.override(False),
            patch.object(tpu_splash_attn, "_check_splash_available", return_value=True),
            patch.object(tpu_splash_attn, "_register_ops", side_effect=lambda: None),
            patch.object(tpu_splash_attn, "_FLASH_OP", object()),
            patch.object(
                tpu_splash_attn, "_resolve_ring_mesh_spec"
            ) as mock_resolve_mesh,
        ):
            impl = TpuSplashAttentionImpl(
                num_heads=4, head_size=128, causal=False, softmax_scale=128**-0.5
            )
            self.assertIsNone(impl._ring_op)
            self.assertIsNone(impl._short_2d_ring_op)
            mock_resolve_mesh.assert_not_called()

    def test_splash_impl_forward_ring_guards_and_reshape(self):
        from types import SimpleNamespace

        import torch

        from sglang_tpu.multimodal_gen.tpu_splash_attn import TpuSplashAttentionImpl

        class _FakeTpuTensor(torch.Tensor):
            @property
            def device(self):
                return SimpleNamespace(type="tpu")

            @device.setter
            def device(self, _value):
                pass

        seen = {}

        def fake_op(q, k, v):
            seen["shapes"] = (tuple(q.shape), tuple(k.shape), tuple(v.shape))
            return q + 1

        impl = TpuSplashAttentionImpl.__new__(TpuSplashAttentionImpl)
        impl.causal = False
        impl._ring_op = fake_op
        impl._short_2d_ring_op = None

        cpu_q = torch.zeros((1, 256, 4, 128), dtype=torch.bfloat16)
        self.assertIsNone(impl.forward_ring(cpu_q, cpu_q, cpu_q))

        tpu_q = cpu_q.as_subclass(_FakeTpuTensor)
        tpu_k_short = torch.zeros((1, 128, 4, 128), dtype=torch.bfloat16).as_subclass(
            _FakeTpuTensor
        )
        self.assertIsNone(impl.forward_ring(tpu_q, tpu_k_short, tpu_k_short))
        self.assertIsNone(impl.forward_ring(tpu_k_short, tpu_k_short, tpu_k_short))

        tpu_gqa_k = torch.zeros((1, 256, 2, 128), dtype=torch.bfloat16).as_subclass(
            _FakeTpuTensor
        )
        out = impl.forward_ring(tpu_q, tpu_gqa_k, tpu_gqa_k)
        self.assertIsNotNone(out)
        self.assertEqual(tuple(out.shape), (1, 256, 4, 128))
        self.assertEqual(
            seen["shapes"],
            (
                (1, 1, 1, 1, 256, 4, 128),
                (1, 1, 1, 1, 256, 2, 128),
                (1, 1, 1, 1, 256, 2, 128),
            ),
        )

        # Short shards (q_seq <= 4096) with even head counts dispatch to _short_2d_ring_op.
        impl._short_2d_ring_op = lambda q, k, v: q + 2
        short_q = torch.zeros((1, 3510, 4, 128), dtype=torch.bfloat16).as_subclass(
            _FakeTpuTensor
        )
        self.assertTrue(torch.all(impl.forward_ring(short_q, short_q, short_q) == 2))
        long_q = torch.zeros((1, 4290, 4, 128), dtype=torch.bfloat16).as_subclass(
            _FakeTpuTensor
        )
        self.assertTrue(torch.all(impl.forward_ring(long_q, long_q, long_q) == 1))
        odd_k = torch.zeros((1, 1950, 1, 128), dtype=torch.bfloat16).as_subclass(
            _FakeTpuTensor
        )
        odd_q = torch.zeros((1, 1950, 3, 128), dtype=torch.bfloat16).as_subclass(
            _FakeTpuTensor
        )
        self.assertTrue(torch.all(impl.forward_ring(odd_q, odd_k, odd_k) == 1))

    def test_usp_attention_prefers_forward_ring_and_falls_back(self):
        from types import SimpleNamespace
        from unittest.mock import patch

        import torch

        from sglang.multimodal_gen.runtime.layers.attention.layer import USPAttention
        from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum

        layer_mod = "sglang.multimodal_gen.runtime.layers.attention.layer"
        obj = USPAttention.__new__(USPAttention)
        obj.skip_sequence_parallel = False
        obj.sp_attention_mode = "ulysses"
        obj.sp_attention_mode_is_auto = False
        obj.softmax_scale = 0.125
        obj.backend = AttentionBackendEnum.OOT
        obj.causal = False
        obj.dropout_p = 0.0
        obj.enable_packed_qkv_input_a2a = False

        q = torch.randn(1, 8, 2, 8)
        calls = []

        def run_with_ring_return(ring_ret):
            obj.attn_impl = SimpleNamespace(
                supports_ring=lambda: True,
                forward_ring=lambda qc, kc, vc: calls.append("ring") or ring_ret,
                supports_ring_resume=lambda: True,
            )
            with (
                patch(
                    f"{layer_mod}.get_forward_context",
                    return_value=SimpleNamespace(attn_metadata=None),
                ),
                patch(f"{layer_mod}.get_sequence_parallel_world_size", return_value=4),
                patch(f"{layer_mod}.get_ulysses_parallel_world_size", return_value=1),
                patch(f"{layer_mod}.get_ring_parallel_world_size", return_value=4),
                patch(
                    f"{layer_mod}._usp_input_all_to_all_qkv",
                    side_effect=lambda qc, kc, vc: (qc, kc, vc),
                ),
                patch(
                    f"{layer_mod}._usp_output_all_to_all",
                    side_effect=lambda t, head_dim: t,
                ),
                patch(
                    f"{layer_mod}.allgather_kv_ring_attn",
                    side_effect=lambda qc, kc, vc, attn_impl: (
                        calls.append("allgather") or torch.zeros_like(qc)
                    ),
                ),
            ):
                return obj.forward(q, q, q)

        out_fast = run_with_ring_return(torch.ones_like(q))
        self.assertEqual(calls, ["ring"])
        self.assertTrue(torch.all(out_fast == 1))

        calls.clear()
        out_fallback = run_with_ring_return(None)
        self.assertEqual(calls, ["ring", "allgather"])
        self.assertTrue(torch.all(out_fallback == 0))


if __name__ == "__main__":
    unittest.main()
