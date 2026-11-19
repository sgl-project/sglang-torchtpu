# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash decode megakernel on a split latent cache.

Random weights and pools at the checkpoint's dimensions (one DSA layer), tp8 on 4 chips:

* split: the kernel on whole pages vs on each rank's in-page share (rank r holds offsets
  [r * P/8, (r + 1) * P/8)). Each rank writes the step's latent rows at its own offsets
  only, at its share row, byte-identical to its own whole-page copy; the logits agree to
  the merge order of the attention partials.

Usage:
    python test/registered/tpu/unit/test_glm_flash_megakernel_split.py
"""

import importlib.util
import os
import sys
import unittest

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=420, suite="nightly-tpu-4-v7", nightly=True, chips=4)

assert "jax" not in sys.modules
os.environ["JAX_PLATFORMS"] = "tpu"

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from sglang.test.test_utils import CustomTestCase

TP, PS, MAX_LEN = 8, 128, 8192
F8 = jnp.float8_e4m3fn


def _spec():
    path = os.path.join(
        os.path.dirname(__file__),
        "../../unit/hardware_backend/tpu/test_glm_flash_megakernel_config.py",
    )
    mod_spec = importlib.util.spec_from_file_location("_mk_config_test", path)
    mod = importlib.util.module_from_spec(mod_spec)
    mod_spec.loader.exec_module(mod)
    return mod._spec()


class _Harness:
    def __init__(self):
        from sglang_tpu.glm_flash_megakernel.build import slab_maps
        from sglang_tpu.glm_flash_megakernel.kernel import decode as D
        from sglang_tpu.glm_flash_megakernel.kernel import pools as PL
        from sglang_tpu.glm_flash_megakernel.kernel import runner as R
        from sglang_tpu.glm_flash_megakernel.layout import KernelLayout

        self.D, self.PL, self.R = D, PL, R
        self.spec = _spec()
        self.layout = KernelLayout(self.spec, TP)
        self.maps = slab_maps(self.spec)
        self.mesh = Mesh(np.array(jax.devices()[:TP]), (D.AXIS,))
        self.pages = 1 + 4 * (MAX_LEN // PS)
        self.pools = self._pool_values()
        self.weights = None

    def cfg(self, q, b, shards):
        return self.D.KernelConfig(
            max_len=MAX_LEN,
            n_query=q,
            n_req=b,
            n_layers=4,
            embed_2d=True,
            km_tile=256,
            context_parallel=True,
            mla_shards=shards,
        )

    def _pool_values(self):
        s, rng = self.spec, np.random.default_rng(1)
        kp, idh, PL = s.index_kpool, s.index_head_dim, self.PL
        n_dsa, n_kda = len(s.dsa_layers), len(s.kda_layers)
        hkda, dh, qsh = s.linear_num_heads // TP, s.linear_head_dim, s.kda_qkv_dim // TP

        def f8bytes(shape):
            v = jnp.asarray(rng.standard_normal(shape, dtype=np.float32)).astype(F8)
            return np.asarray(jax.lax.bitcast_convert_type(v, jnp.uint8))

        nope = [f8bytes((self.pages, PS, 4, 128))] * n_dsa
        raw = [
            (0.5 * rng.standard_normal((self.pages, PS, 2 * idh), np.float32)).astype(
                jnp.bfloat16
            )
        ] * n_dsa
        keys = jnp.asarray(
            rng.standard_normal((self.pages * PS // kp, idh), np.float32)
        )
        rec = np.asarray(PL.pool_records(keys, PL.record_bytes(idh)))
        rec = [rec.reshape(self.pages, PS // kp // 4, 4, PL.record_bytes(idh))] * n_dsa
        tmp = [0.05 * rng.standard_normal((40, hkda * TP, dh, dh), np.float32)] * n_kda
        cnv = [
            0.1
            * rng.standard_normal(
                (40, s.linear_conv_kernel_dim - 1, 1, 3 * qsh * TP), np.float32
            )
        ] * n_kda
        return dict(nope=nope, raw=raw, records=rec, temporal=tmp, conv=cnv)

    def _weights(self, run, cfg):
        D, R, s = self.D, self.R, self.spec
        H, tp = s.hidden_size, TP
        n_dsa, n_kda = len(s.dsa_layers), len(s.kda_layers)
        n_moe = sum(t == "sparse" for t in s.mlp_layer_types)
        nh, qsh = s.num_attention_heads // tp, s.kda_qkv_dim // tp
        hkda, dh = s.linear_num_heads // tp, s.linear_head_dim
        idh, kvl = s.index_head_dim, s.kv_lora_rank
        moesh, ffnsh = s.moe_intermediate_size // tp, s.intermediate_size // tp
        ql, qk, vd, L = s.q_lora_rank, s.qk_head_dim, s.v_head_dim, s.num_hidden_layers
        rows8, cols = 2 * H, cfg.km_cols
        sds = jax.ShapeDtypeStruct
        pk = lambda w, rows: jax.eval_shape(
            lambda: D.pack_kmajor(jnp.zeros(w.shape, w.dtype), rows, cols)
        )
        shapes = {
            "embed_tokens": sds((s.vocab_size, H // tp), jnp.bfloat16),
            "lm_head": pk(sds((1, self.layout.vocab_padded // tp, H), jnp.bfloat16), H),
            "final_norm": sds((1, H), jnp.bfloat16),
            "smalls": sds((L + 1, self.layout.small_rows(), H), jnp.bfloat16),
            "smalls_f32": sds((L + 1, 1, H), jnp.float32),
            "kda_qkv": pk(sds((n_kda, 3 * qsh, H), F8), rows8),
            "kda_o": pk(sds((n_kda, H, qsh), F8), rows8),
            "kda_fg_b": sds((n_kda, 2 * qsh, dh), jnp.bfloat16),
            "kda_dt_bias": sds((n_kda, 1, qsh), jnp.float32),
            "kda_conv": sds(
                (n_kda, s.linear_conv_kernel_dim, 1, 3 * qsh), jnp.bfloat16
            ),
            "kda_b": sds((n_kda, hkda, H), jnp.bfloat16),
            "kda_a_log": sds((n_kda, 1, hkda), jnp.float32),
            "kda_scale": sds((n_kda, 1, 3 * qsh + H), jnp.float32),
            "dsa_qakv": pk(sds((n_dsa, ql + kvl, H), F8), rows8),
            "dsa_q_b": pk(sds((n_dsa, nh * qk, ql), F8), rows8),
            "dsa_kv_b": sds((n_dsa, nh * (qk + vd), kvl), jnp.bfloat16),
            "dsa_o": pk(sds((n_dsa, H, nh * vd), F8), rows8),
            "dsa_scale": sds((n_dsa, 1, D.proj_scale_cols(s, tp)), jnp.float32),
            "indexer_wq_b": pk(
                sds((n_dsa, s.index_n_heads * idh, ql), jnp.bfloat16), H
            ),
            "moe_router": sds((n_moe, s.n_routed_experts, H), jnp.bfloat16),
            "expert_gate_up": sds((s.n_routed_experts, H, 2 * moesh), F8),
            "expert_gate_up_scale": sds(
                (n_moe * s.n_routed_experts, 2 * moesh), jnp.float32
            ),
            "expert_down": sds((s.n_routed_experts, moesh, H), F8),
            "expert_down_scale": sds((n_moe * s.n_routed_experts, H), jnp.float32),
            "shared_gate_up": sds((n_moe, H, 2 * moesh), F8),
            "shared_gate_up_scale": sds((n_moe, 2 * moesh), jnp.float32),
            "shared_down": sds((n_moe, moesh, H), F8),
            "shared_down_scale": sds((n_moe, H), jnp.float32),
            "dense_gate_up": pk(sds((3, 2 * ffnsh, H), F8), rows8),
            "dense_down": pk(sds((3, H, ffnsh), F8), rows8),
            "dense_scale": sds((3, 1, D.proj_scale_cols(s, tp)), jnp.float32),
        }
        scale = {"embed_tokens": 1.0, "smalls": 0.1, "final_norm": 1.0, "kda_conv": 0.2}

        def rand(i, name, x, spec_):
            shape = tuple(
                d * (tp if p == D.AXIS else 1)
                for d, p in zip(x.shape, tuple(spec_) + (None,) * 8)
            )
            key = jax.random.PRNGKey(i)

            def f():
                v = jax.random.normal(key, shape, jnp.float32)
                if x.dtype == F8:
                    return v.astype(F8)
                if "scale" in name:
                    return (0.015 * (1 + 0.1 * v)).astype(x.dtype)
                return (scale.get(name, 0.05) * v).astype(x.dtype)

            return jax.jit(f, out_shardings=NamedSharding(self.mesh, spec_))()

        out = {}
        for i, (name, x) in enumerate(shapes.items()):
            if name in R.EXPERT_LISTS:  # one MoE layer runs; the rest repeat it
                out[name] = [rand(i, name, x, run.wspec[name][0])] * n_moe
            else:
                out[name] = rand(i, name, x, run.wspec[name])
        return out

    def run(self, q, b, shards, positions, tokens):
        D, R = self.D, self.R
        cfg = self.cfg(q, b, shards)
        run = R.make_decode(self.spec, self.layout, *self.maps, self.mesh, cfg)
        if self.weights is None:
            self.weights = self._weights(run, cfg)
        devs = self.mesh.devices.reshape(-1)
        pools = {}
        for n, arrs in self.pools.items():
            sh = NamedSharding(self.mesh, run.pspec[n][0])
            if n == "nope" and shards > 1:
                sub = PS // shards
                pools[n] = [
                    jax.make_array_from_single_device_arrays(
                        (x.shape[0], sub) + x.shape[2:],
                        sh,
                        [
                            jax.device_put(
                                np.ascontiguousarray(x[:, r * sub : (r + 1) * sub]), d
                            )
                            for r, d in enumerate(devs)
                        ],
                    )
                    for x in arrs
                ]
            else:
                pools[n] = [jax.device_put(jnp.asarray(x), sh) for x in arrs]
        pps = cfg.pages_per_seq
        perm = np.random.default_rng(2).permutation(np.arange(1, self.pages))
        tab = perm[: b * pps].astype(np.int32)
        small = NamedSharding(self.mesh, P())
        scal = [
            jnp.asarray(tokens[:b, :q].reshape(-1), jnp.int32),
            jnp.asarray(positions[:b], jnp.int32),
            jnp.asarray([b] + [0] * 7, jnp.int32),
            jnp.asarray(tab),
            jnp.arange(1, b + 1, dtype=jnp.int32),
            jnp.arange(16, 16 + b * q, dtype=jnp.int32),
            jnp.full((len(self.spec.dsa_layers),), 0.5, jnp.float32),
        ]
        scal = [jax.device_put(x, small) for x in scal]
        init = jax.device_put(
            jnp.zeros(
                (cfg.rows * self.spec.hc_mult, self.spec.hidden_size), jnp.bfloat16
            ),
            NamedSharding(self.mesh, P(None, None)),
        )
        logits, _, new = jax.jit(run.shard)(*scal, init, self.weights, pools)
        nope = [
            np.asarray(s.data)
            for s in sorted(
                new["nope"][0].addressable_shards, key=lambda s: s.device.id
            )
        ]
        rows = [
            (
                int(tab.reshape(b, pps)[i, (positions[i] + j) // PS]),
                (positions[i] + j) % PS,
            )
            for i in range(b)
            for j in range(q)
        ]
        return np.asarray(logits, np.float32), nope, rows


@unittest.skipIf(len(jax.devices()) < TP, "needs 8 TPU devices")
class TestGlmFlashMegakernelSplit(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.h = _Harness()
        rng = np.random.default_rng(7)
        cls.tokens = rng.integers(1000, 150000, size=(4, 16))
        # Sparse (> 2048 tokens of history); rows cross a page and a share boundary.
        cls.positions = [5000 + 70, 4093, 6014, 3010]

    def _close(self, x, y):
        rel = np.mean(np.abs(x - y)) / np.mean(np.abs(y))
        self.assertTrue(np.isfinite(x).all())
        self.assertLess(rel, 1e-2)
        self.assertGreaterEqual(np.mean(x.argmax(-1) == y.argmax(-1)), 0.99)

    def _split_case(self, q, b):
        whole, nope_w, rows = self.h.run(q, b, 1, self.positions, self.tokens)
        split, nope_s, _ = self.h.run(q, b, TP, self.positions, self.tokens)
        self._close(split, whole)
        sub = PS // TP
        for r in range(TP):
            # Rank r keeps exactly its own copy of the rows at its offsets.
            np.testing.assert_array_equal(
                nope_s[r], nope_w[r][:, r * sub : (r + 1) * sub]
            )
        owners = {off // sub for _, off in rows}
        self.assertGreater(len(owners), 1)  # the step's rows span several ranks

    def test_split_matches_whole_pages(self):
        self._split_case(4, 2)


if __name__ == "__main__":
    unittest.main()
