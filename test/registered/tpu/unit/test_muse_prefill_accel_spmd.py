# SPDX-License-Identifier: Apache-2.0
"""Unit and SPMD-export regression tests for Muse Spark prefill acceleration."""

import os
import unittest

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.kernels.xor1_bridge_all_gather import xor1_bridge_all_gather
from sglang_tpu.muse_prefill_kernels.attn_kernels import entry_exit
from sglang_tpu.muse_prefill_kernels.integration import _a1_body
from sglang_tpu.muse_prefill_kernels.moe_bridge_kernels import kernel_b2

register_tpu_ci(
    est_time=120,
    suite="stage-a-test-tpu-unit",
    chips=4,
    labels=["singlehost", "muse-spark-open-816b-a42b", "tp"],
)


class TestMusePrefillAccelSPMD(CustomTestCase):
    def setUp(self):
        super().setUp()
        devices = jax.devices()
        if len(devices) < 8:
            self.skipTest(f"Requires 8 TPU TensorCores, got {len(devices)}")
        self.devs = list(devices[:8])
        self.mesh = Mesh(np.array(self.devs), ("tp",))

    def test_v6_s8str128_bf16kv_export_has_zero_partition_id_and_rpa_shapes(self):
        tp = 8
        t_len = 2048
        t_shard = t_len // tp
        hidden = 8192
        hq_loc, hkv_loc = 16, 2
        q_sz, kv_sz = hq_loc * 64, hkv_loc * 64

        a1_fn = _a1_body(hq_loc, hkv_loc, "v6_s8str128_bf16kv", rpa_layout=True)

        def _wrapped(x_sh, wqg, wqgs, wkv, wkvs, gain, inv_gain, cos, sin):
            return jax.shard_map(
                lambda x, a, b, c, d, e, f, g, h: a1_fn(
                    x, a[0], b[0], c[0], d[0], e[0], f[0], g[0], h[0]
                ),
                mesh=self.mesh,
                in_specs=(P("tp", None),) + (P("tp"),) * 8,
                out_specs=(P("tp"),) * 3,
                check_vma=False,
            )(x_sh, wqg, wqgs, wkv, wkvs, gain, inv_gain, cos, sin)

        shd_row = NamedSharding(self.mesh, P("tp", None))
        shd_tp = NamedSharding(self.mesh, P("tp"))
        args = (
            jax.ShapeDtypeStruct((t_len, hidden), jnp.bfloat16, sharding=shd_row),
            jax.ShapeDtypeStruct((tp, hidden, 2 * q_sz), jnp.float8_e4m3fn, sharding=shd_tp),
            jax.ShapeDtypeStruct((tp, 1, 2 * q_sz), jnp.float32, sharding=shd_tp),
            jax.ShapeDtypeStruct((tp, tp * hidden, 2 * kv_sz), jnp.float8_e4m3fn, sharding=shd_tp),
            jax.ShapeDtypeStruct((tp, tp, 2 * kv_sz), jnp.float32, sharding=shd_tp),
            jax.ShapeDtypeStruct((tp, 1, hidden), jnp.bfloat16, sharding=shd_tp),
            jax.ShapeDtypeStruct((tp, 1, hidden), jnp.bfloat16, sharding=shd_tp),
            jax.ShapeDtypeStruct((tp, t_len, 128), jnp.float32, sharding=shd_tp),
            jax.ShapeDtypeStruct((tp, t_len, 128), jnp.float32, sharding=shd_tp),
        )
        exported = jax.export.export(jax.jit(_wrapped))(*args)
        mlir_text = exported.mlir_module()
        self.assertNotIn("partition_id", mlir_text)
        self.assertEqual(exported.out_avals[0].shape, (tp, t_len, 8, 2, 128))
        self.assertEqual(exported.out_avals[1].shape, (tp * t_len, q_sz))
        self.assertEqual(exported.out_avals[2].shape, (tp * t_len, 1, 4, 128))

    def test_sharded_rope_and_xor1_xytorus_zero_partition_id_and_bit_exact(self):
        tp = 8
        t_shard = 256
        x_np = np.arange(tp * t_shard * 512, dtype=np.float32).reshape(tp * t_shard, 512) * 0.001
        x_shd = jax.device_put(jnp.asarray(x_np, jnp.bfloat16), NamedSharding(self.mesh, P("tp", None)))

        fn_ref = jax.jit(
            jax.shard_map(
                lambda s: lax.all_gather(s, "tp", axis=0, tiled=True),
                mesh=self.mesh,
                in_specs=P("tp", None),
                out_specs=P(),
                check_vma=False,
            )
        )
        fn_xytorus = jax.jit(
            jax.shard_map(
                lambda s: xor1_bridge_all_gather(s, "tp", tiled=True, xy_torus=True),
                mesh=self.mesh,
                in_specs=P("tp", None),
                out_specs=P(),
                check_vma=False,
            )
        )
        exported = jax.export.export(fn_xytorus)(x_shd)
        self.assertNotIn("partition_id", exported.mlir_module())

        y_ref = fn_ref(x_shd)
        y_xy = fn_xytorus(x_shd)
        for s_ref, s_xy in zip(y_ref.addressable_shards, y_xy.addressable_shards, strict=True):
            self.assertTrue(np.array_equal(np.asarray(s_ref.data), np.asarray(s_xy.data)))

        rope_fn = jax.jit(
            jax.shard_map(
                lambda pos, tab: entry_exit.sharded_rope_gather_128(pos, tab, tp_axis="tp"),
                mesh=self.mesh,
                in_specs=(P(), P()),
                out_specs=(P(), P()),
                check_vma=False,
            )
        )
        rep = NamedSharding(self.mesh, P())
        rope_exp = jax.export.export(rope_fn)(
            jax.ShapeDtypeStruct((tp * t_shard,), jnp.int32, sharding=rep),
            jax.ShapeDtypeStruct((4096, 256), jnp.float32, sharding=rep),
        )
        self.assertNotIn("partition_id", rope_exp.mlir_module())

    def test_b2_single_write_matches_default_nstream(self):
        rng = np.random.default_rng(42)
        h_sh = jnp.asarray(rng.standard_normal((256, 8192)), jnp.bfloat16)
        wp = jnp.asarray(rng.standard_normal((8192, 4096)) * 0.02, jnp.float8_e4m3fn)
        wps = jnp.ones((4096,), jnp.float32) * 0.02
        pe_w = jnp.zeros((4096,), jnp.bfloat16)

        prev = os.environ.get("SGLANG_TPU_MUSE_KERNEL_B2_SINGLE_WRITE")
        try:
            os.environ["SGLANG_TPU_MUSE_KERNEL_B2_SINGLE_WRITE"] = "0"
            q0, s0 = kernel_b2.pallas_wpool_prenorm_int8_quant_nstream(h_sh, wp, wps, pe_w)
            os.environ["SGLANG_TPU_MUSE_KERNEL_B2_SINGLE_WRITE"] = "1"
            q1, s1 = kernel_b2.pallas_wpool_prenorm_int8_quant_nstream(h_sh, wp, wps, pe_w)
            self.assertTrue(np.array_equal(np.asarray(q0), np.asarray(q1)))
            self.assertTrue(np.array_equal(np.asarray(s0), np.asarray(s1)))
        finally:
            if prev is None:
                os.environ.pop("SGLANG_TPU_MUSE_KERNEL_B2_SINGLE_WRITE", None)
            else:
                os.environ["SGLANG_TPU_MUSE_KERNEL_B2_SINGLE_WRITE"] = prev


if __name__ == "__main__":
    unittest.main()
