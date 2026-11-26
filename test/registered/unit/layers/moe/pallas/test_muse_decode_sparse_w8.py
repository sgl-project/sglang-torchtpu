# SPDX-License-Identifier: Apache-2.0
"""CPU unit, scratch-layout, & numerical accuracy tests for Wave-8 sparse decode MoE."""

import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

try:
    from sglang.test.ci.ci_register import register_cpu_ci

    register_cpu_ci(est_time=5, suite="base-a-test-cpu")
except Exception:
    pass

_REPO_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "python" / "sglang").is_dir()
)
_MUSE_DIR = _REPO_ROOT / "sglang_tpu/moe/pallas/fused_rs_muse"


def _load_pkg_module(mod_name: str, file_path: Path):
    for parent_pkg in (
        "sglang_tpu",
        "sglang_tpu.moe",
        "sglang_tpu.moe.pallas",
        "sglang_tpu.moe.pallas.fused_rs_muse",
    ):
        if parent_pkg not in sys.modules:
            pkg = types.ModuleType(parent_pkg)
            pkg.__path__ = [str(_REPO_ROOT / parent_pkg.replace(".", "/"))]
            sys.modules[parent_pkg] = pkg
    full_name = f"sglang_tpu.moe.pallas.fused_rs_muse.{mod_name}"
    if full_name in sys.modules:
        return sys.modules[full_name]
    spec = importlib.util.spec_from_file_location(full_name, file_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = mod
    spec.loader.exec_module(mod)
    return mod


_PLANNER_MOD = _load_pkg_module(
    "moe_imbalance_planner", _MUSE_DIR / "moe_imbalance_planner.py"
)
_BREAKERS_MOD = _load_pkg_module(
    "decode_hbm_breakers", _MUSE_DIR / "decode_hbm_breakers.py"
)

DecodeRoutingPruneConfig = _BREAKERS_MOD.DecodeRoutingPruneConfig
dequant_nvfp4_u32_np = _BREAKERS_MOD.dequant_nvfp4_u32_np
parse_m_buckets_str = _BREAKERS_MOD.parse_m_buckets_str
quantize_and_pack_w3_u32_np = _BREAKERS_MOD.quantize_and_pack_w3_u32_np
read_sparse_v2_env_overrides = _BREAKERS_MOD.read_sparse_v2_env_overrides
repack_w1_s1_part_major_np = _BREAKERS_MOD.repack_w1_s1_part_major_np
route_weights_with_breakers_np = _BREAKERS_MOD.route_weights_with_breakers_np
select_cold_local_slots_np = _BREAKERS_MOD.select_cold_local_slots_np
unpack_w3_u32_np = _BREAKERS_MOD.unpack_w3_u32_np

build_preloaded_replica_plan = _PLANNER_MOD.build_preloaded_replica_plan
plan_preloaded_replicas_ref = _PLANNER_MOD.plan_preloaded_replicas_ref
validate_replica_plan = _PLANNER_MOD.validate_replica_plan


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    af = a.astype(np.float64).ravel()
    bf = b.astype(np.float64).ravel()
    denom = np.linalg.norm(af) * np.linalg.norm(bf)
    return float(np.dot(af, bf) / max(denom, 1e-30))


def _quantize_nvfp4_u32_np(
    w_f32: np.ndarray, scale_block: int = 256
) -> tuple[np.ndarray, np.ndarray]:
    """Reference NVFP4 block quantizer returning ``(u32 [K//8, N], scale [K//blk, N])``."""
    k, n = w_f32.shape
    wb = w_f32.reshape(k // scale_block, scale_block, n)
    sb = (np.max(np.abs(wb), axis=1) / 6.0).astype(np.float32)
    sb_safe = np.where(sb > 0.0, sb, 1.0)[:, None, :]
    ab = np.abs(wb) / sb_safe
    mids = (
        (0.25, False),
        (0.75, True),
        (1.25, False),
        (1.75, True),
        (2.5, False),
        (3.5, True),
        (5.0, False),
    )
    idx = np.zeros_like(ab, dtype=np.uint32)
    for m_val, tie_up in mids:
        idx += (ab >= m_val if tie_up else ab > m_val).astype(np.uint32)
    code4 = (idx | ((wb < 0.0).astype(np.uint32) << np.uint32(3))).reshape(
        k // 8, 8, n
    )
    u32 = np.zeros((k // 8, n), dtype=np.uint32)
    for b in range(8):
        u32 |= code4[:, b, :] << np.uint32(4 * b)
    return u32, sb


class TestMuseDecodeSparseW8(unittest.TestCase):
    def test_baseline_routing_parity_and_nan_safety(self):
        rng = np.random.default_rng(123)
        logits = rng.standard_normal((64, 256)).astype(np.float32)
        logits[60:, :] = np.nan  # Padded tail rows
        bias = (rng.standard_normal((1, 256)) * 0.1).astype(np.float32)

        dense, counts, free_mask = route_weights_with_breakers_np(
            logits, bias, DecodeRoutingPruneConfig()
        )
        self.assertEqual(dense.shape, (64, 256))
        self.assertFalse(np.any(free_mask))
        # Live rows each pick exactly 8 experts whose weights sum to 1.0.
        live_nnz = np.sum(dense[:60] > 0.0, axis=1)
        np.testing.assert_array_equal(live_nnz, np.full((60,), 8))
        np.testing.assert_allclose(
            np.sum(dense[:60], axis=1), np.ones((60,)), atol=1e-6
        )
        # NaN tail rows contribute 0 weights and 0 expert counts.
        np.testing.assert_array_equal(dense[60:], np.zeros((4, 256)))
        self.assertEqual(int(np.sum(counts)), 60 * 8)

    def test_batch_coupled_free_an_expert_invariants_and_hbm_savings(self):
        rng = np.random.default_rng(2026)
        # Skewed Zipf pre-sigmoid logit prior + realistic small correction bias (B=64, E=256)
        ranks = np.arange(1, 257, dtype=np.float32)
        perm = rng.permutation(256)
        logit_prior = (-4.0 + 1.1 * np.log(16.0 / np.power(ranks, 0.65))).reshape(
            1, 256
        )[:, perm]
        logits = (
            logit_prior + rng.standard_normal((64, 256)) * 1.35
        ).astype(np.float32)
        zipf_bias = (rng.standard_normal((1, 256)) * 0.04).astype(np.float32)

        base_w, base_counts, _ = route_weights_with_breakers_np(
            logits, zipf_bias, DecodeRoutingPruneConfig()
        )
        cfg = DecodeRoutingPruneConfig(
            free_expert_rows=2,
            free_min_rank=6,
            free_max_weight=0.105,
        )
        pruned_w, pruned_counts, free_mask = route_weights_with_breakers_np(
            logits, zipf_bias, cfg
        )

        # 1. Every freed expert had 1..2 rows in baseline and now has 0 rows!
        freed_ids = np.flatnonzero(free_mask)
        self.assertGreater(len(freed_ids), 10)
        self.assertTrue(np.all(base_counts[freed_ids] >= 1))
        self.assertTrue(np.all(base_counts[freed_ids] <= 2))
        self.assertTrue(np.all(pruned_counts[freed_ids] == 0))

        # 2. Every non-freed expert retains its EXACT hit count (zero wasted drops!).
        kept_ids = np.flatnonzero(~free_mask)
        np.testing.assert_array_equal(pruned_counts[kept_ids], base_counts[kept_ids])

        # 3. Every token retains >= 6 experts and sums to 1.0.
        self.assertTrue(np.all(np.sum(pruned_w > 0.0, axis=1) >= 6))
        np.testing.assert_allclose(
            np.sum(pruned_w, axis=1), np.ones((64,)), atol=1e-6
        )

        # 4. High MoE & residual-stream cosine similarity (strictly beating static Top-6!).
        e_outs = rng.standard_normal((256, 256)).astype(np.float32)
        e_outs /= np.linalg.norm(e_outs, axis=1, keepdims=True)
        residual = rng.standard_normal((64, 256)).astype(np.float32) * 2.5
        y_base = base_w @ e_outs
        y_pruned = pruned_w @ e_outs
        static_top6_w, _, _ = route_weights_with_breakers_np(
            logits, zipf_bias, DecodeRoutingPruneConfig(), topk=6
        )
        y_top6 = static_top6_w @ e_outs
        self.assertGreater(_cosine_sim(y_base, y_pruned), 0.988)
        self.assertGreater(_cosine_sim(y_base, y_pruned), _cosine_sim(y_base, y_top6))
        self.assertGreater(_cosine_sim(residual + y_base, residual + y_pruned), 0.9999)

    def test_entropy_mass_and_min_p_dynamic_topk(self):
        # Construct 2 tokens: row 0 is spiky (low entropy), row 1 is flat (high entropy)
        logits = np.full((2, 256), -8.0, dtype=np.float32)
        logits[0, :8] = np.array([6.0, 5.0, -2.0, -3.0, -3.5, -4.0, -4.5, -5.0])
        logits[1, :8] = np.array([1.2, 1.15, 1.1, 1.05, 1.0, 0.95, 0.9, 0.85])
        bias = np.zeros((1, 256), dtype=np.float32)

        cfg = DecodeRoutingPruneConfig(min_p=0.025, cum_mass=0.92, min_k=2)
        w, counts, _ = route_weights_with_breakers_np(logits, bias, cfg)
        k_spiky = int(np.sum(w[0] > 0.0))
        k_flat = int(np.sum(w[1] > 0.0))
        self.assertLessEqual(k_spiky, 3)
        self.assertGreaterEqual(k_flat, 7)
        np.testing.assert_allclose(np.sum(w, axis=1), np.ones((2,)), atol=1e-6)
        self.assertEqual(int(np.sum(counts)), k_spiky + k_flat)

    def test_cold_expert_w3_pack_unpack_and_moe_layer_accuracy(self):
        rng = np.random.default_rng(77)
        k_dim, i_dim, h_dim = 512, 256, 512  # Scaled geometry divisible by 256
        w_f32 = (rng.standard_normal((k_dim, i_dim)) * 0.04).astype(np.float32)

        # 1. Verify 25% byte reduction (384/512 ratio = 0.75) and exact code roundtrip
        packed_w3, sb_w3 = quantize_and_pack_w3_u32_np(w_f32, 256, alpha=0.75)
        self.assertEqual(packed_w3.shape, ((k_dim // 8) * 3 // 4, i_dim))
        u32_unpacked = unpack_w3_u32_np(packed_w3)
        self.assertEqual(u32_unpacked.shape, (k_dim // 8, i_dim))
        # Every nibble must have bit 0 == 1 -> E2M1 codes in {1,3,5,7,9,11,13,15}
        self.assertTrue(
            np.all((u32_unpacked & np.uint32(0x11111111)) == np.uint32(0x11111111))
        )

        # Re-packing unpacked W3 words must reproduce packed_w3 bit-for-bit!
        chunks = u32_unpacked.reshape(-1, 4, 8, i_dim)
        p0 = (chunks[:, 0] & np.uint32(0xEEEEEEEE)) | (
            (chunks[:, 3] >> np.uint32(1)) & np.uint32(0x11111111)
        )
        p1 = (chunks[:, 1] & np.uint32(0xEEEEEEEE)) | (
            (chunks[:, 3] >> np.uint32(2)) & np.uint32(0x11111111)
        )
        p2 = (chunks[:, 2] & np.uint32(0xEEEEEEEE)) | (
            (chunks[:, 3] >> np.uint32(3)) & np.uint32(0x11111111)
        )
        repacked = np.concatenate([p0, p1, p2], axis=1).reshape(packed_w3.shape)
        np.testing.assert_array_equal(repacked, packed_w3)

        # 2. Simulate 32-expert local SwiGLU + RMSNorm MoE (8 hot W4 + 24 cold W3)
        n_exp = 32
        counts = np.exp(-0.18 * np.arange(n_exp)) * 100.0
        mask_i32, cold_slots = select_cold_local_slots_np(
            counts, owner=0, e_local=32, cold_per_owner=24, replicated_eids=(0, 1)
        )
        self.assertEqual(len(cold_slots), 24)
        self.assertNotIn(0, cold_slots)
        self.assertNotIn(1, cold_slots)
        for le in range(32):
            bit_set = ((np.int32(mask_i32).view(np.uint32) >> np.uint32(le)) & np.uint32(1)) != 0
            self.assertEqual(bit_set, le in cold_slots)

        x = rng.standard_normal((16, h_dim)).astype(np.float32)
        p_exp = counts / counts.sum()
        y_w4 = np.zeros((16, h_dim), dtype=np.float32)
        y_mix = np.zeros((16, h_dim), dtype=np.float32)
        for le in range(n_exp):
            wg = (rng.standard_normal((h_dim, i_dim)) * 0.03).astype(np.float32)
            wu = (rng.standard_normal((h_dim, i_dim)) * 0.03).astype(np.float32)
            wd = (rng.standard_normal((i_dim, h_dim)) * 0.03).astype(np.float32)

            wg4 = dequant_nvfp4_u32_np(*_quantize_nvfp4_u32_np(wg, 256))
            wu4 = dequant_nvfp4_u32_np(*_quantize_nvfp4_u32_np(wu, 256))
            wd4 = dequant_nvfp4_u32_np(*_quantize_nvfp4_u32_np(wd, 256))

            def _swiglu_norm(w_g, w_u, w_d):
                g = x @ w_g
                u = x @ w_u
                act = (g / (1.0 + np.exp(-g))) * u
                out = act @ w_d
                rms = np.sqrt(np.mean(out * out, axis=-1, keepdims=True) + 1e-8)
                return out / rms

            out_w4 = _swiglu_norm(wg4, wu4, wd4)
            if le in cold_slots:
                pwg, swg = quantize_and_pack_w3_u32_np(wg4, 256, alpha=0.75)
                pwu, swu = quantize_and_pack_w3_u32_np(wu4, 256, alpha=0.75)
                pwd, swd = quantize_and_pack_w3_u32_np(wd4, 256, alpha=0.75)
                out_m = _swiglu_norm(
                    dequant_nvfp4_u32_np(unpack_w3_u32_np(pwg), swg),
                    dequant_nvfp4_u32_np(unpack_w3_u32_np(pwu), swu),
                    dequant_nvfp4_u32_np(unpack_w3_u32_np(pwd), swd),
                )
            else:
                out_m = out_w4

            y_w4 += float(p_exp[le]) * out_w4
            y_mix += float(p_exp[le]) * out_m

        self.assertGreater(_cosine_sim(y_w4, y_mix), 0.996)

    def test_contiguous_8mib_part_major_repack_proj_and_dual(self):
        rng = np.random.default_rng(99)
        w1 = rng.integers(0, 2**32, size=(4, 512, 8192), dtype=np.uint32)
        s1 = rng.standard_normal((4, 16, 8192)).astype(np.float32)

        # Proj part-major (CONTIG_PARTS=2)
        w_pm, s_pm = repack_w1_s1_part_major_np(w1, s1, dual_halves=False)
        self.assertEqual(w_pm.shape, (8, 512, 4096))
        self.assertEqual(s_pm.shape, (8, 16, 4096))
        for e in range(4):
            np.testing.assert_array_equal(w_pm[2 * e + 0], w1[e, :, 0:4096])
            np.testing.assert_array_equal(w_pm[2 * e + 1], w1[e, :, 4096:8192])
            np.testing.assert_array_equal(s_pm[2 * e + 0], s1[e, :, 0:4096])
            np.testing.assert_array_equal(s_pm[2 * e + 1], s1[e, :, 4096:8192])

        # Dual-half-major (CONTIG_PARTS=3)
        w_dm, s_dm = repack_w1_s1_part_major_np(w1, s1, dual_halves=True)
        for e in range(4):
            np.testing.assert_array_equal(w_dm[2 * e + 0, :, 0:2048], w1[e, :, 0:2048])
            np.testing.assert_array_equal(
                w_dm[2 * e + 0, :, 2048:4096], w1[e, :, 4096:6144]
            )
            np.testing.assert_array_equal(
                w_dm[2 * e + 1, :, 0:2048], w1[e, :, 2048:4096]
            )
            np.testing.assert_array_equal(
                w_dm[2 * e + 1, :, 2048:4096], w1[e, :, 6144:8192]
            )

    def test_planner_greedy_peak_first_and_chip_aware_invariants(self):
        # Construct an adversarial R=1 table where static eid order lets medium
        # rank 0 steal holder rank 4's slot before bottleneck rank 2 (load 28)!
        counts = np.zeros((256,), dtype=np.int32)
        counts[0 * 32 : 0 * 32 + 21] = 2   # Owner 0 load = 21
        counts[1 * 32 : 1 * 32 + 20] = 2   # Owner 1 load = 20
        counts[2 * 32 : 2 * 32 + 28] = 2   # Owner 2 load = 28 (GLOBAL BOTTLENECK!)
        counts[3 * 32 : 3 * 32 + 18] = 2   # Owner 3 load = 18
        counts[4 * 32 : 4 * 32 + 19] = 2   # Owner 4 load = 19
        counts[5 * 32 : 5 * 32 + 24] = 2   # Owner 5 load = 24
        counts[6 * 32 : 6 * 32 + 20] = 2   # Owner 6 load = 20
        counts[7 * 32 : 7 * 32 + 20] = 2   # Owner 7 load = 20

        # Both eid=0 (home=0) and eid=64 (home=2) compete for holder=4's slot 0?
        # Even simpler: eid=0 (home=0 -> holder=4, slot=0) makes load[4]=20;
        # if another replica eid=64 (home=2 -> holder=3, slot=0) and eid=65 (home=2 -> holder=4, slot=1)
        calib = counts.astype(np.float64) + 1.0
        plan = build_preloaded_replica_plan(
            calib,
            num_owners=8,
            num_replicas_per_owner=2,
            owner_to_chip=[r >> 1 for r in range(8)],
            selection_mode="bottleneck_owner_b64",
        )
        validate_replica_plan(plan)

        dec_lin = plan_preloaded_replicas_ref(
            counts,
            plan.replica_meta,
            num_owners=8,
            num_replicas_per_owner=2,
            cost_mode="active",
            greedy_peak_first=False,
        )
        dec_greedy = plan_preloaded_replicas_ref(
            counts,
            plan.replica_meta,
            num_owners=8,
            num_replicas_per_owner=2,
            cost_mode="active",
            greedy_peak_first=True,
        )
        dec_chip = plan_preloaded_replicas_ref(
            counts,
            plan.replica_meta,
            num_owners=8,
            num_replicas_per_owner=2,
            cost_mode="active",
            greedy_peak_first=True,
            chip_aware=True,
        )
        self.assertLessEqual(
            int(dec_greedy.owner_loads_after.max()),
            int(dec_lin.owner_loads_after.max()),
        )
        # Verify total active experts across all 8 owners is conserved (unique execution!).
        total_active = int(np.sum(counts > 0))
        self.assertEqual(int(dec_lin.owner_loads_after.sum()), total_active)
        self.assertEqual(int(dec_greedy.owner_loads_after.sum()), total_active)
        self.assertEqual(int(dec_chip.owner_loads_after.sum()), total_active)

    def test_granular_env_overrides_parsing(self):
        env_patch = {
            "SGLANG_TPU_MUSE_KERNEL_DECODE_V2_RING": "ring6_dual",
            "SGLANG_TPU_MUSE_KERNEL_DECODE_PAIR_RS": "1",
            "SGLANG_TPU_MUSE_KERNEL_DECODE_VMEM_DEQ": "1",
            "SGLANG_TPU_MUSE_KERNEL_DECODE_FP4_UNPACK": "lut_u4",
            "SGLANG_TPU_MUSE_KERNEL_DECODE_ASYNC_GATHER": "1",
            "SGLANG_TPU_MUSE_KERNEL_DECODE_FUSED_NORM_SCATTER": "1",
            "SGLANG_TPU_MUSE_KERNEL_DECODE_M_BUCKETS": "8,16,32",
            "SGLANG_TPU_MUSE_KERNEL_DECODE_LHS_FP8": "1",
            "SGLANG_TPU_MUSE_KERNEL_DECODE_HOIST_SCALES": "1",
            "SGLANG_TPU_MUSE_KERNEL_DECODE_V2_CONTIG_PARTS": "3",
            "SGLANG_TPU_MUSE_DECODE_MIN_P_TOPK": "0.022",
            "SGLANG_TPU_MUSE_DECODE_TOPK_MASS": "0.93",
            "SGLANG_TPU_MUSE_DECODE_FREE_EXPERT_ROWS": "2",
        }
        with mock.patch.dict(os.environ, env_patch, clear=False):
            ov = read_sparse_v2_env_overrides()
        self.assertEqual(ov.ring_mode, "ring6_dual")
        self.assertEqual(ov.pair_rs, 1)
        self.assertEqual(ov.vmem_deq, 1)
        self.assertEqual(ov.fp4_unpack_mode, "lut_u4")
        self.assertEqual(ov.async_gather, 1)
        self.assertEqual(ov.fused_norm_scatter, 1)
        self.assertEqual(ov.m_buckets, (8, 16, 32))
        self.assertTrue(ov.lhs_fp8)
        self.assertTrue(ov.hoist_scales)
        self.assertEqual(ov.contig_parts, 3)
        self.assertAlmostEqual(ov.routing_prune.min_p, 0.022)
        self.assertAlmostEqual(ov.routing_prune.cum_mass, 0.93)
        self.assertEqual(ov.routing_prune.free_expert_rows, 2)
        self.assertEqual(parse_m_buckets_str("32, 8, 16"), (8, 16, 32))

        # Verify #1466 cross-branch compact syntax ("2:0.12:6" and MIN_P_TOPK >= 0.5 -> cum_mass)
        with mock.patch.dict(
            os.environ,
            {
                "SGLANG_TPU_MUSE_DECODE_FREE_EXPERT_ROWS": "2:0.12:6",
                "SGLANG_TPU_MUSE_DECODE_MIN_P_TOPK": "0.74",
                "SGLANG_TPU_MUSE_DECODE_TOPK_MASS": "",
            },
            clear=False,
        ):
            ov2 = read_sparse_v2_env_overrides()
        self.assertEqual(ov2.routing_prune.free_expert_rows, 2)
        self.assertAlmostEqual(ov2.routing_prune.free_max_weight, 0.12)
        self.assertEqual(ov2.routing_prune.free_min_rank, 6)
        self.assertAlmostEqual(ov2.routing_prune.min_p, 0.0)
        self.assertAlmostEqual(ov2.routing_prune.cum_mass, 0.74)

    def test_v2_scratch_iterator_parity_and_64mib_vmem_budget(self):
        """Exhaustively verify all_refs next(it) count == call_args + out + scratch_shapes
        and total VMEM < 64 MiB across 96 SparseDecodeV2Config combinations!"""

        class _BarrierReached(Exception):
            pass

        class _MockArr:
            def __init__(self, shape, dtype):
                self.shape = tuple(shape)
                self.dtype = str(dtype)
                self.ndim = len(self.shape)

            def reshape(self, *s):
                if len(s) == 1 and isinstance(s[0], tuple):
                    s = s[0]
                if -1 in s:
                    known = 1
                    for d in s:
                        if d != -1:
                            known *= d
                    tot = 1
                    for d in self.shape:
                        tot *= d
                    s = tuple((tot // known) if d == -1 else d for d in s)
                return _MockArr(s, self.dtype)

            def astype(self, dt):
                return _MockArr(self.shape, dt)

        class _MemSpec:
            def __init__(self, space, shape, dtype=None):
                self.space = space
                self.shape = tuple(shape) if isinstance(shape, tuple) else shape
                self.dtype = str(dtype) if dtype is not None else ""

            @property
            def nbytes(self) -> int:
                if self.space != "VMEM" or not isinstance(self.shape, tuple):
                    return 0
                bpe = {
                    "uint32": 4,
                    "int32": 4,
                    "float32": 4,
                    "bfloat16": 2,
                    "float8_e4m3fn": 1,
                }[self.dtype]
                elems = 1
                for d in self.shape:
                    elems *= int(d)
                return elems * bpe

        def _stop_at_barrier():
            raise _BarrierReached()

        # Build lightweight mock jax/pallas if jax is unavailable on bare CPU host
        saved_mods = {}
        mock_names = (
            "jax",
            "jax.numpy",
            "jax.lax",
            "jax.experimental",
            "jax.experimental.pallas",
            "jax.experimental.pallas.tpu",
            "sglang_tpu.moe.pallas.fused_rs_muse.fused_decode_routing",
            "sglang_tpu.moe.pallas.fused_rs_muse.fused_decode_sparse_v2",
        )
        for mn in mock_names:
            saved_mods[mn] = sys.modules.get(mn)

        try:
            jnp_m = types.ModuleType("jax.numpy")
            for dt in (
                "int32",
                "uint32",
                "uint16",
                "uint4",
                "float32",
                "bfloat16",
                "float8_e4m3fn",
                "float4_e2m1fn",
                "bool_",
            ):
                setattr(jnp_m, dt, dt)
            lax_m = types.ModuleType("jax.lax")
            pl_m = types.ModuleType("jax.experimental.pallas")
            pl_m.BlockSpec = lambda memory_space=None: ("BlockSpec", memory_space)
            pl_m.PrefetchScalarGridSpec = None

            pltpu_m = types.ModuleType("jax.experimental.pallas.tpu")
            pltpu_m.VMEM = lambda shape, dtype: _MemSpec("VMEM", shape, dtype)
            pltpu_m.SMEM = lambda shape, dtype: _MemSpec("SMEM", shape, dtype)
            pltpu_m.HBM = "HBM"

            class _Sem:
                def __call__(self, shape):
                    return _MemSpec("SEM", shape)

            pltpu_m.SemaphoreType = types.SimpleNamespace(DMA=_Sem())
            pltpu_m.PrefetchScalarGridSpec = lambda **kw: kw
            pltpu_m.CompilerParams = lambda **kw: kw
            pltpu_m.get_barrier_semaphore = _stop_at_barrier

            captured = {}

            def _mock_pallas_call(kernel_fn, *, out_shape, grid_spec, compiler_params, name):
                def _runner(*call_args):
                    scratch = grid_spec["scratch_shapes"]
                    vmem_bytes = sum(s.nbytes for s in scratch if isinstance(s, _MemSpec))
                    captured["vmem_bytes"] = vmem_bytes
                    captured["name"] = name
                    # Construct exact refs tuple: (*prefix, *in_args, output_ref, *scratch)
                    all_refs = list(call_args) + ["OUT_REF"] + list(scratch)
                    consumed = [0]

                    def _iter_spy(seq):
                        for item in seq:
                            consumed[0] += 1
                            yield item

                    with mock.patch("builtins.iter", side_effect=_iter_spy):
                        try:
                            kernel_fn(*all_refs)
                        except _BarrierReached:
                            pass
                    if consumed[0] != len(all_refs):
                        raise AssertionError(
                            f"{name}: kernel next(it) consumed {consumed[0]} refs, "
                            f"but pallas_call supplied {len(all_refs)}!"
                        )
                    return out_shape

                return _runner

            pl_m.pallas_call = _mock_pallas_call
            exp_m = types.ModuleType("jax.experimental")
            exp_m.pallas = pl_m

            jax_m = types.ModuleType("jax")
            jax_m.Array = _MockArr
            jax_m.numpy = jnp_m
            jax_m.lax = lax_m
            jax_m.experimental = exp_m
            jax_m.ShapeDtypeStruct = lambda shape, dtype: _MockArr(shape, dtype)

            sys.modules["jax"] = jax_m
            sys.modules["jax.numpy"] = jnp_m
            sys.modules["jax.lax"] = lax_m
            sys.modules["jax.experimental"] = exp_m
            sys.modules["jax.experimental.pallas"] = pl_m
            sys.modules["jax.experimental.pallas.tpu"] = pltpu_m

            sys.modules.pop(
                "sglang_tpu.moe.pallas.fused_rs_muse.fused_decode_routing", None
            )
            sys.modules.pop(
                "sglang_tpu.moe.pallas.fused_rs_muse.fused_decode_sparse_v2", None
            )
            _load_pkg_module(
                "fused_decode_routing", _MUSE_DIR / "fused_decode_routing.py"
            )
            v2_mod = _load_pkg_module(
                "fused_decode_sparse_v2", _MUSE_DIR / "fused_decode_sparse_v2.py"
            )

            checked = 0
            for bsz in (16, 64, 128, 384):
                for blk in (256, 16):
                    for ring in (
                        "ring3_proj",
                        "ring4_dual",
                        "ring6_dual",
                        "ring3_khalf",
                        "ring4_khalf",
                    ):
                        for has_rep in (False, True):
                            for has_w3 in (False, True):
                                for sharded in (True, False):
                                    if bsz == 16 and ring != "ring3_proj":
                                        continue
                                    pm = ring == "ring4_dual" and has_w3
                                    w_rows = 64 if pm else 32
                                    w_cols = 4096 if pm else 8192
                                    rank = _MockArr((1,), "int32")
                                    logits = _MockArr((bsz, 256), "float32")
                                    bias = _MockArr((1, 256), "float32")
                                    x = _MockArr((bsz, 4096), "bfloat16")
                                    w1 = _MockArr((w_rows, 4096, w_cols), "float4_e2m1fn")
                                    s1 = _MockArr((w_rows, 4096 // blk, w_cols), "bfloat16")
                                    w2 = _MockArr((32, 4096, 4096), "float4_e2m1fn")
                                    s2 = _MockArr((32, 4096 // blk, 4096), "bfloat16")
                                    norm = _MockArr((1, 4096), "bfloat16")
                                    kw = {}
                                    if has_rep:
                                        rw_rows = 8 if pm else 4
                                        kw.update(
                                            rep_w1=_MockArr((rw_rows, 4096, w_cols), "float4_e2m1fn"),
                                            rep_s1=_MockArr((rw_rows, 4096 // blk, w_cols), "bfloat16"),
                                            rep_w2=_MockArr((4, 4096, 4096), "float4_e2m1fn"),
                                            rep_s2=_MockArr((4, 4096 // blk, 4096), "bfloat16"),
                                            replica_meta=_MockArr((4, 32), "int32"),
                                        )
                                    if has_w3:
                                        kw["w3_mask"] = _MockArr((1,), "int32")

                                    cfg = v2_mod.SparseDecodeV2Config(
                                        row_mode="dense" if bsz == 16 else "sparse",
                                        ring_mode=ring,
                                        tile_m=16 if bsz == 16 else 64,
                                        m_buckets=(16,) if bsz == 16 else (8, 32, 64),
                                        multi_m32_mode="vmem_deq" if (bsz == 128 and not ring.endswith("_khalf")) else "none",
                                        lhs_fp8=(blk == 256 and has_w3),
                                        async_gather=(bsz == 64),
                                        fused_norm_scatter=True,
                                        rs_wire_bf16=(bsz >= 64),
                                        pair_rs=(bsz >= 64),
                                        pair_ag_relay=(bsz >= 64),
                                        static_holder_mask=has_rep,
                                        spec_prime_parts=2,
                                        hoist_scales=True,
                                        contig_parts_mode=3 if pm else (1 if ring.endswith("_khalf") else 0),
                                    )
                                    v2_mod.fused_decode_sparse_v2_from_logits(
                                        rank,
                                        logits,
                                        bias,
                                        x,
                                        w1,
                                        s1,
                                        w2,
                                        s2,
                                        norm,
                                        ep=8,
                                        return_sharded=sharded,
                                        config=cfg,
                                        **kw,
                                    )
                                    self.assertLess(
                                        captured["vmem_bytes"],
                                        64 * 1024**2,
                                        msg=f"{captured['name']} exceeded 64 MiB VMEM: {captured['vmem_bytes']}",
                                    )
                                    checked += 1
            self.assertGreaterEqual(checked, 100)
        finally:
            for mn, prev in saved_mods.items():
                if prev is None:
                    sys.modules.pop(mn, None)
                else:
                    sys.modules[mn] = prev

    def test_real_torch_and_jax_pallas_trace_when_available(self):
        """When running inside the sglang-torchtpu container, verify real Torch
        float4_e2m1fn_x2 W3/part-major packers and real JAX/Pallas kernel tracing!"""
        try:
            import jax
            import jax.numpy as jnp
            import torch
        except ImportError:
            self.skipTest("Real torch/jax only available inside sglang-torchtpu image")

        from sglang_tpu.moe.pallas.fused_rs_muse.decode_hbm_breakers import (
            DecodeRoutingPruneConfig,
            quantize_cold_slots_inplace_torch,
            repack_w1_s1_part_major_torch,
        )
        from sglang_tpu.moe.pallas.fused_rs_muse.fused_decode_sparse_v2 import (
            SparseDecodeV2Config,
            fused_decode_sparse_v2_from_logits,
        )

        # 1. Real PyTorch float4_e2m1fn_x2 Cold-W3 + Dual-Half Part-Major repack
        fp4x2 = getattr(torch, "float4_e2m1fn_x2", None)
        if fp4x2 is not None:
            g = torch.Generator().manual_seed(11)
            w1_u8 = torch.randint(0, 256, (4, 4096, 4096), dtype=torch.uint8, generator=g)
            w1_t = torch.empty((4, 4096, 4096), dtype=fp4x2)
            w1_t.view(torch.uint8).copy_(w1_u8)
            s1_t = (torch.rand((4, 16, 8192), generator=g) * 0.1 + 0.01).to(torch.bfloat16)

            rel_err = quantize_cold_slots_inplace_torch(w1_t, s1_t, [1, 3], alpha=0.75)
            self.assertGreater(rel_err, 0.0)
            self.assertLess(rel_err, 0.35)
            # Tail rows 3072:4096 of cold slots must be zeroed
            self.assertEqual(int(w1_t.view(torch.uint8)[1, 3072:, :].abs().sum().item()), 0)
            # Hot slot 0 must be untouched
            self.assertTrue(torch.equal(w1_t.view(torch.uint8)[0], w1_u8[0]))

            w1_pm, s1_pm = repack_w1_s1_part_major_torch(w1_t, s1_t, dual_halves=True)
            self.assertEqual(tuple(w1_pm.shape), (8, 4096, 2048))
            self.assertEqual(tuple(s1_pm.shape), (8, 16, 4096))

        # 2. Real JAX/Pallas jax.eval_shape / jaxpr trace across all new Wave-8 paths!
        rank_s = jax.ShapeDtypeStruct((1,), jnp.int32)
        logits_s = jax.ShapeDtypeStruct((64, 256), jnp.float32)
        bias_s = jax.ShapeDtypeStruct((1, 256), jnp.float32)
        x_s = jax.ShapeDtypeStruct((64, 4096), jnp.bfloat16)
        w1_s = jax.ShapeDtypeStruct((32, 4096, 8192), jnp.float4_e2m1fn)
        s1_s = jax.ShapeDtypeStruct((32, 16, 8192), jnp.bfloat16)
        w2_s = jax.ShapeDtypeStruct((32, 4096, 4096), jnp.float4_e2m1fn)
        s2_s = jax.ShapeDtypeStruct((32, 16, 4096), jnp.bfloat16)
        norm_s = jax.ShapeDtypeStruct((1, 4096), jnp.bfloat16)
        rep_w1_s = jax.ShapeDtypeStruct((4, 4096, 8192), jnp.float4_e2m1fn)
        rep_s1_s = jax.ShapeDtypeStruct((4, 16, 8192), jnp.bfloat16)
        rep_w2_s = jax.ShapeDtypeStruct((4, 4096, 4096), jnp.float4_e2m1fn)
        rep_s2_s = jax.ShapeDtypeStruct((4, 16, 4096), jnp.bfloat16)
        rep_meta_s = jax.ShapeDtypeStruct((4, 32), jnp.int32)
        w3_mask_s = jax.ShapeDtypeStruct((1,), jnp.int32)

        configs_to_trace = [
            # (a) ring4_dual + W4A8/W3A8 + Free-an-Expert + Min-P + SPEC_PRIME + pair_rs + hoisted scales
            (
                SparseDecodeV2Config(
                    row_mode="sparse",
                    ring_mode="ring4_dual",
                    tile_m=32,
                    m_buckets=(8, 32),
                    lhs_fp8=True,
                    async_gather=True,
                    fused_norm_scatter=True,
                    rs_wire_bf16=True,
                    pair_rs=True,
                    static_holder_mask=True,
                    spec_prime_parts=2,
                    hoist_scales=True,
                    routing_prune=DecodeRoutingPruneConfig(
                        min_p=0.05, cum_mass=0.90, free_expert_rows=2
                    ),
                ),
                True,
                False,
            ),
            # (b) ring4_khalf zero-copy contiguous 8-MiB burst mode + W4A8
            (
                SparseDecodeV2Config(
                    row_mode="sparse",
                    ring_mode="ring4_khalf",
                    tile_m=32,
                    m_buckets=(8, 32),
                    lhs_fp8=True,
                    async_gather=False,
                    fused_norm_scatter=True,
                    rs_wire_bf16=True,
                    pair_rs=True,
                    static_holder_mask=True,
                    spec_prime_parts=2,
                    hoist_scales=True,
                    contig_parts_mode=1,
                ),
                False,
                False,
            ),
            # (c) ring6_dual + K256 vmem_deq + part-major 8-MiB repack
            (
                SparseDecodeV2Config(
                    row_mode="sparse",
                    ring_mode="ring6_dual",
                    tile_m=64,
                    m_buckets=(8, 32, 64),
                    multi_m32_mode="vmem_deq",
                    lhs_fp8=False,
                    async_gather=True,
                    fused_norm_scatter=True,
                    rs_wire_bf16=True,
                    pair_rs=True,
                    static_holder_mask=True,
                    spec_prime_parts=3,
                    hoist_scales=True,
                    contig_parts_mode=3,
                ),
                True,
                True,
            ),
        ]

        mesh_devs = jax.devices()
        use_8dev_mosaic = len(mesh_devs) >= 8
        mesh = jax.sharding.Mesh(
            np.array(mesh_devs[:8] if use_8dev_mosaic else mesh_devs[:1]),
            ("model",),
        )
        P = jax.sharding.PartitionSpec

        from contextlib import ExitStack
        from unittest.mock import patch as mock_patch

        with ExitStack() as stack:
            if use_8dev_mosaic:
                from sglang_tpu.pallas_export import tpu_info_for_cpu_export

                stack.enter_context(
                    mock_patch(
                        "sglang_tpu.pallas_export.get_torch_tpu_device_kind",
                        return_value="TPU v7",
                    )
                )
                stack.enter_context(tpu_info_for_cpu_export())

            for cfg, use_w3, use_pm in configs_to_trace:
                w1_cur = (
                    jax.ShapeDtypeStruct((64, 4096, 4096), jnp.float4_e2m1fn)
                    if use_pm
                    else w1_s
                )
                s1_cur = (
                    jax.ShapeDtypeStruct((64, 16, 4096), jnp.bfloat16)
                    if use_pm
                    else s1_s
                )
                rw1_cur = (
                    jax.ShapeDtypeStruct((8, 4096, 4096), jnp.float4_e2m1fn)
                    if use_pm
                    else rep_w1_s
                )
                rs1_cur = (
                    jax.ShapeDtypeStruct((8, 16, 4096), jnp.bfloat16)
                    if use_pm
                    else rep_s1_s
                )

                def _fn(
                    rk,
                    lg,
                    bi,
                    xv,
                    w1v,
                    s1v,
                    w2v,
                    s2v,
                    nm,
                    rw1,
                    rs1,
                    rw2,
                    rs2,
                    rm,
                    w3m,
                    _c=cfg,
                    _w3=use_w3,
                ):
                    return fused_decode_sparse_v2_from_logits(
                        rk,
                        lg,
                        bi,
                        xv,
                        w1v,
                        s1v,
                        w2v,
                        s2v,
                        nm,
                        rw1,
                        rs1,
                        rw2,
                        rs2,
                        rm,
                        w3m if _w3 else None,
                        ep=8,
                        return_sharded=True,
                        ep_axis_name="model",
                        config=_c,
                    )

                in_shapes = (
                    rank_s,
                    logits_s,
                    bias_s,
                    x_s,
                    w1_cur,
                    s1_cur,
                    w2_s,
                    s2_s,
                    norm_s,
                    rw1_cur,
                    rs1_cur,
                    rep_w2_s,
                    rep_s2_s,
                    rep_meta_s,
                    w3_mask_s,
                )
                if use_8dev_mosaic:
                    sm = jax.jit(
                        jax.shard_map(
                            _fn,
                            mesh=mesh,
                            in_specs=(P(),) * len(in_shapes),
                            out_specs=P(),
                            check_vma=False,
                        )
                    )
                    lowered = sm.trace(*in_shapes).lower(lowering_platforms=("tpu",))
                    stablehlo = lowered.as_text("stablehlo")
                    hlo = lowered.as_text("hlo")
                    self.assertIn("tpu_custom_call", hlo)
                    self.assertIn("muse_sparse_v2_ep8_b64_k256", stablehlo)
                else:
                    with mesh:
                        out_sd = jax.eval_shape(_fn, *in_shapes)
                    self.assertEqual(out_sd.shape, (8, 4096))
                    self.assertEqual(out_sd.dtype, jnp.bfloat16)

    def test_w9_bucket_scaled_free_expert_pruning(self):
        """Verify #2407 bucket-scaled FREE pruning across v1 FP4, v1 FP8, and v2."""
        scale_free_expert_for_batch = _BREAKERS_MOD.scale_free_expert_for_batch
        route_weights_with_breakers_np = _BREAKERS_MOD.route_weights_with_breakers_np

        # Direct unit checks across b64, b64_ms16, ge32, and off modes.
        self.assertEqual(scale_free_expert_for_batch(7, 1, 16, mode="b64"), (2, 2))
        self.assertEqual(scale_free_expert_for_batch(7, 1, 32, mode="b64"), (4, 2))
        self.assertEqual(scale_free_expert_for_batch(7, 1, 64, mode="b64"), (7, 1))
        self.assertEqual(scale_free_expert_for_batch(10, 1, 16, mode="b64"), (2, 2))
        self.assertEqual(scale_free_expert_for_batch(10, 1, 32, mode="b64"), (5, 2))
        self.assertEqual(scale_free_expert_for_batch(10, 1, 64, mode="b64"), (10, 1))
        self.assertEqual(scale_free_expert_for_batch(7, 1, 32, mode="b64_ms16"), (4, 1))
        self.assertEqual(scale_free_expert_for_batch(7, 1, 16, mode="ge32"), (0, 1))
        self.assertEqual(scale_free_expert_for_batch(7, 1, 32, mode="ge32"), (4, 2))
        self.assertEqual(scale_free_expert_for_batch(7, 1, 16, mode="off"), (7, 1))
        self.assertEqual(scale_free_expert_for_batch(0, 1, 16, mode="b64"), (0, 1))
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SGLANG_TPU_MUSE_DECODE_FREE_SCALE_MODE", None)
            self.assertEqual( _BREAKERS_MOD.read_free_expert_scale_mode(), "b64_ms16")
            self.assertEqual(scale_free_expert_for_batch(7, 1, 16), (2, 2))
            self.assertEqual(scale_free_expert_for_batch(7, 1, 32), (4, 1))
            self.assertEqual(scale_free_expert_for_batch(7, 1, 64), (7, 1))
            self.assertEqual(scale_free_expert_for_batch(6, 2, 32), (3, 2))

        # Verify tail-batch protection (5 real tokens + 11 identical dummy rows at B=16):
        # unscaled f7 (7:0.20:1) collapses real tokens to top-1, whereas b64-scaled (2:0.20:2)
        # preserves >= 2 experts on every real token!
        rng = np.random.default_rng(2407)
        real_lg = rng.normal(-3.0, 0.8, size=(5, 256)).astype(np.float32)
        dummy_row = rng.normal(-3.0, 0.8, size=(1, 256)).astype(np.float32)
        logits_b16 = np.concatenate([real_lg, np.repeat(dummy_row, 11, axis=0)], axis=0)
        bias = np.zeros((1, 256), dtype=np.float32)

        f_rows_eff, ms_eff = scale_free_expert_for_batch(7, 1, 16, mode="b64")
        w_scaled, _, _ = route_weights_with_breakers_np(
            logits_b16,
            bias,
            DecodeRoutingPruneConfig(
                free_expert_rows=f_rows_eff,
                free_min_rank=ms_eff,
                free_max_weight=0.20,
            ),
        )
        real_kept_counts = np.sum(w_scaled[:5] > 0.0, axis=1)
        self.assertTrue(np.all(real_kept_counts >= 2), f"got {real_kept_counts}")
        np.testing.assert_allclose(np.sum(w_scaled[:5], axis=1), 1.0, atol=1e-6)

        try:
            v1_mod = _load_pkg_module(
                "fused_decode_sparse", _MUSE_DIR / "fused_decode_sparse.py"
            )
            v2_mod = _load_pkg_module(
                "fused_decode_sparse_v2", _MUSE_DIR / "fused_decode_sparse_v2.py"
            )
        except Exception:
            return

        with mock.patch.dict(
            os.environ,
            {
                "SGLANG_TPU_MUSE_DECODE_FREE_EXPERT_ROWS": "7:0.20:1",
                "SGLANG_TPU_MUSE_DECODE_MIN_P_TOPK": "0.80",
                "SGLANG_TPU_MUSE_DECODE_FREE_SCALE_MODE": "b64",
            },
            clear=False,
        ):
            c16 = v1_mod.default_sparse_decode_config(16, 16, 512)
            c32 = v1_mod.default_sparse_decode_config(32, 16, 512)
            c64 = v1_mod.default_sparse_decode_config(64, 16, 512)
            self.assertEqual((c16.free_max_rows, c16.free_min_slot), (2, 2))
            self.assertEqual((c32.free_max_rows, c32.free_min_slot), (4, 2))
            self.assertEqual((c64.free_max_rows, c64.free_min_slot), (7, 1))

            fp8_16 = v1_mod.default_fp8_sparse_decode_config(16, 16)
            fp8_64 = v1_mod.default_fp8_sparse_decode_config(64, 16)
            self.assertEqual((fp8_16.free_max_rows, fp8_16.free_min_slot), (2, 2))
            self.assertEqual((fp8_64.free_max_rows, fp8_64.free_min_slot), (7, 1))

            v2_16 = v2_mod.default_sparse_decode_v2_config(16, 16, 512)
            v2_32 = v2_mod.default_sparse_decode_v2_config(32, 16, 512)
            v2_64 = v2_mod.default_sparse_decode_v2_config(64, 16, 512)
            self.assertEqual(
                (v2_16.routing_prune.free_expert_rows, v2_16.routing_prune.free_min_rank),
                (2, 2),
            )
            self.assertEqual(
                (v2_32.routing_prune.free_expert_rows, v2_32.routing_prune.free_min_rank),
                (4, 2),
            )
            self.assertEqual(
                (v2_64.routing_prune.free_expert_rows, v2_64.routing_prune.free_min_rank),
                (7, 1),
            )

            # Verify cand_w8_zi legacy defaults ("wide", all W9 sub-knobs off) & opt-in toggles.
            self.assertEqual(c16.k256_strip_mode, "wide")
            self.assertEqual(c64.k256_strip_mode, "wide")
            self.assertEqual(v2_64.k256_strip_mode, "wide")
            self.assertFalse(
                c64.nc_split512 or c64.k512_chain or c64.unroll_strips or c64.route_rb16_mode
            )
            self.assertFalse(c16.pair_rs or c16.bridge_ag)
            self.assertFalse(c32.pair_rs or c32.bridge_ag)
            self.assertFalse(c64.pair_rs or c64.bridge_ag)
            self.assertFalse(fp8_64.pair_rs or fp8_64.bridge_ag)
            self.assertTrue(fp8_64.static_holder_mask)
            self.assertFalse(v2_32.pair_rs or v2_32.pair_ag_relay)
            self.assertFalse(v2_64.pair_rs or v2_64.pair_ag_relay)

            c64_ep8 = v1_mod.default_sparse_decode_config(64, 8, 512)
            self.assertFalse(c64_ep8.pair_rs)
            with mock.patch.dict(
                os.environ,
                {
                    "SGLANG_TPU_MUSE_KERNEL_DECODE_PAIR_RS": "1",
                    "SGLANG_TPU_MUSE_KERNEL_DECODE_STRIP_MODE": "adaptive",
                    "SGLANG_TPU_MUSE_KERNEL_DECODE_NC_SPLIT512": "1",
                    "SGLANG_TPU_MUSE_KERNEL_DECODE_K512_CHAIN": "1",
                    "SGLANG_TPU_MUSE_KERNEL_DECODE_UNROLL_STRIPS": "1",
                    "SGLANG_TPU_MUSE_KERNEL_DECODE_ROUTE_RB16": "2",
                },
                clear=False,
            ):
                c16_on = v1_mod.default_sparse_decode_config(16, 16, 512)
                c32_on = v1_mod.default_sparse_decode_config(32, 16, 512)
                c64_on = v1_mod.default_sparse_decode_config(64, 16, 512)
                fp8_64_on = v1_mod.default_fp8_sparse_decode_config(64, 16)
                v2_64_on = v2_mod.default_sparse_decode_v2_config(64, 16, 512)
                self.assertEqual(c64_on.k256_strip_mode, "adaptive")
                self.assertTrue(
                    c64_on.nc_split512 and c64_on.k512_chain and c64_on.unroll_strips
                )
                self.assertEqual(c64_on.route_rb16_mode, 2)
                self.assertEqual(fp8_64_on.route_rb16_mode, 2)
                self.assertFalse(c16_on.pair_rs or c16_on.bridge_ag)
                self.assertTrue(c32_on.pair_rs and c32_on.bridge_ag)
                self.assertTrue(c64_on.pair_rs and c64_on.bridge_ag)
                self.assertTrue(fp8_64_on.pair_rs and fp8_64_on.bridge_ag)
                self.assertTrue(v2_64_on.pair_rs and v2_64_on.pair_ag_relay)
                self.assertTrue(v1_mod.default_sparse_decode_config(64, 8, 512).pair_rs)

        # Verify default (unset env) uses b64_ms16 across v1, FP8, and v2 (#2552).
        with mock.patch.dict(
            os.environ,
            {
                "SGLANG_TPU_MUSE_DECODE_FREE_EXPERT_ROWS": "7:0.20:1",
                "SGLANG_TPU_MUSE_DECODE_MIN_P_TOPK": "0.80",
            },
            clear=False,
        ):
            os.environ.pop("SGLANG_TPU_MUSE_DECODE_FREE_SCALE_MODE", None)
            d16 = v1_mod.default_sparse_decode_config(16, 16, 512)
            d32 = v1_mod.default_sparse_decode_config(32, 16, 512)
            d64 = v1_mod.default_sparse_decode_config(64, 16, 512)
            self.assertEqual((d16.free_max_rows, d16.free_min_slot), (2, 2))
            self.assertEqual((d32.free_max_rows, d32.free_min_slot), (4, 1))
            self.assertEqual((d64.free_max_rows, d64.free_min_slot), (7, 1))
            dfp8_32 = v1_mod.default_fp8_sparse_decode_config(32, 16)
            self.assertEqual((dfp8_32.free_max_rows, dfp8_32.free_min_slot), (4, 1))
            dv2_32 = v2_mod.default_sparse_decode_v2_config(32, 16, 512)
            self.assertEqual(
                (dv2_32.routing_prune.free_expert_rows, dv2_32.routing_prune.free_min_rank),
                (4, 1),
            )

    def test_w9_routing_rb16_two_pass_bit_exact_vs_legacy_rb64(self):
        """Bit-exact selection-mask, weight-bits, counts_smem & freed-expert parity vs legacy rb=64."""
        try:
            import jax
            import jax.numpy as jnp
            from jax import lax
            from jax.experimental import pallas as pl
            from jax.experimental.pallas import tpu as pltpu

            sys.modules.pop(
                "sglang_tpu.moe.pallas.fused_rs_muse.fused_decode_routing", None
            )
            routing_mod = _load_pkg_module(
                "fused_decode_routing", _MUSE_DIR / "fused_decode_routing.py"
            )
        except Exception as exc:
            self.skipTest(f"JAX/Pallas TPU unavailable: {exc}")

        def _legacy_route_rb64(
            logits_ref,
            bias_ref,
            scratch,
            dma_sem,
            *,
            my_owner,
            e_per_owner: int,
            holder_cand_eids: tuple,
            topk: int = 8,
            route_eps: float = 1e-15,
            route_scale: float = 1.0,
            route_chunk_rows: int = 32,
            min_p_topk: float = 0.0,
            free_max_rows: int = 0,
            free_max_w: float = 0.06,
            free_min_slot: int = 5,
        ):
            # Exact cand_w9_base (349b604d63) pre-Wave-9 routing implementation.
            batch_size, num_experts = logits_ref.shape
            eff_chunk = (
                max(route_chunk_rows, min(batch_size, 64))
                if (free_max_rows > 0 and batch_size <= 64)
                else route_chunk_rows
            )
            rb = min(batch_size, eff_chunk)
            if batch_size % rb != 0:
                rb = 8 if batch_size % 8 == 0 else batch_size

            first = my_owner * e_per_owner
            last_excl = first + e_per_owner
            bias_row = bias_ref[...].reshape(1, num_experts).astype(jnp.float32)
            e_iota = lax.broadcasted_iota(jnp.int32, (rb, num_experts), 1)
            col128 = lax.broadcasted_iota(jnp.int32, (rb, 128), 1)
            counts_init = jnp.zeros((1, num_experts), jnp.int32)

            n_chunks = batch_size // rb
            eff_min_s = max(1, int(free_min_slot))
            first_v = jnp.broadcast_to(first, (rb, 1))
            last_excl_v = jnp.broadcast_to(last_excl, (rb, 1))
            rep_vecs = tuple(
                jnp.broadcast_to(jnp.where(eid >= 0, eid, jnp.int32(-1)), (rb, 1))
                for eid in holder_cand_eids
            )
            free_mask_global = None
            if free_max_rows > 0 and n_chunks > 1:

                def _scan_free_sparse(ci, carry):
                    tot_c, drp_c = carry
                    r0 = pl.multiple_of(ci * rb, rb)
                    sb, sw, kf = routing_mod._select_and_filter_chunk_vpu(
                        logits_ref[pl.ds(r0, rb), :].astype(jnp.float32),
                        bias_row,
                        e_iota,
                        num_experts,
                        topk=topk,
                        route_eps=route_eps,
                        min_p_topk=min_p_topk,
                    )
                    df = routing_mod._droppable_flags_for_chunk(
                        sw,
                        kf,
                        route_eps=route_eps,
                        free_max_w=free_max_w,
                        free_min_slot=free_min_slot,
                    )
                    for b_s, k_s, d_s in zip(sb, kf, df):
                        pk = e_iota == b_s
                        tot_c = tot_c + jnp.sum(
                            jnp.logical_and(pk, k_s).astype(jnp.int32),
                            axis=0,
                            keepdims=True,
                        )
                        drp_c = drp_c + jnp.sum(
                            jnp.logical_and(pk, d_s).astype(jnp.int32),
                            axis=0,
                            keepdims=True,
                        )
                    return tot_c, drp_c

                tot_all, drp_all = lax.fori_loop(
                    0, n_chunks, _scan_free_sparse, (counts_init, counts_init)
                )
                free_mask_global = (
                    (tot_all > 0) & (tot_all <= free_max_rows) & (drp_all == tot_all)
                )

            def _route_chunk_sparse_dyn(r0, counts_acc):
                row_s = pl.ds(r0, rb)
                slot_bests, slot_ws, keep_flags = (
                    routing_mod._select_and_filter_chunk_vpu(
                        logits_ref[row_s, :].astype(jnp.float32),
                        bias_row,
                        e_iota,
                        num_experts,
                        topk=topk,
                        route_eps=route_eps,
                        min_p_topk=min_p_topk,
                    )
                )
                if free_max_rows > 0:
                    drop_flags = routing_mod._droppable_flags_for_chunk(
                        slot_ws,
                        keep_flags,
                        route_eps=route_eps,
                        free_max_w=free_max_w,
                        free_min_slot=free_min_slot,
                    )
                    if free_mask_global is None:
                        tot_c = counts_init
                        drp_c = counts_init
                        for b_s, k_s, d_s in zip(slot_bests, keep_flags, drop_flags):
                            pk = e_iota == b_s
                            tot_c = tot_c + jnp.sum(
                                jnp.logical_and(pk, k_s).astype(jnp.int32),
                                axis=0,
                                keepdims=True,
                            )
                            drp_c = drp_c + jnp.sum(
                                jnp.logical_and(pk, d_s).astype(jnp.int32),
                                axis=0,
                                keepdims=True,
                            )
                        fmask = (
                            (tot_c > 0)
                            & (tot_c <= free_max_rows)
                            & (drp_c == tot_c)
                        )
                    else:
                        fmask = free_mask_global
                    for s in range(eff_min_s, topk):
                        freed_e = jnp.any(
                            (e_iota == slot_bests[s]) & fmask,
                            axis=1,
                            keepdims=True,
                        )
                        keep_flags[s] = keep_flags[s] & (~(drop_flags[s] & freed_e))

                pack_ids = jnp.zeros((rb, 128), dtype=jnp.int32)
                w_loc_raw = jnp.zeros((rb, 128), dtype=jnp.float32)
                loc_cnt = jnp.zeros((rb, 1), dtype=jnp.int32)
                hit = jnp.zeros((rb, num_experts), dtype=jnp.bool_)
                w_sum = jnp.zeros((rb, 1), dtype=jnp.float32)
                for best, w_slot, keep_s in zip(slot_bests, slot_ws, keep_flags):
                    pk_kept = jnp.logical_and(e_iota == best, keep_s)
                    hit = jnp.logical_or(hit, pk_kept)
                    w_sum = w_sum + jnp.where(keep_s, w_slot, 0.0)
                    cand_slot = jnp.where(
                        (best >= first_v) & (best < last_excl_v),
                        best - first_v,
                        jnp.int32(-1),
                    )
                    for r_slot, rep_v in enumerate(rep_vecs):
                        cand_slot = jnp.where(
                            best == rep_v,
                            jnp.int32(e_per_owner + r_slot),
                            cand_slot,
                        )
                    is_cand = keep_s & (cand_slot >= 0)
                    id_col = jnp.int32(1) + loc_cnt
                    w_col = jnp.int32(16) + loc_cnt
                    pack_ids = jnp.where(
                        jnp.logical_and(is_cand, col128 == id_col),
                        cand_slot,
                        pack_ids,
                    )
                    w_loc_raw = jnp.where(
                        jnp.logical_and(is_cand, col128 == w_col),
                        w_slot,
                        w_loc_raw,
                    )
                    loc_cnt = loc_cnt + is_cand.astype(jnp.int32)
                pack_ids = jnp.where(col128 == 0, loc_cnt, pack_ids)
                w_loc_norm = w_loc_raw / (w_sum + route_eps)
                if route_scale != 1.0:
                    w_loc_norm = w_loc_norm * route_scale
                w_bits = lax.bitcast_convert_type(w_loc_norm, jnp.int32)
                scratch.route_pack_vmem[row_s, :] = jnp.where(
                    col128 < 16, pack_ids, w_bits
                )
                return counts_acc + jnp.sum(
                    hit.astype(jnp.int32), axis=0, keepdims=True
                )

            if n_chunks == 1:
                counts_acc = _route_chunk_sparse_dyn(0, counts_init)
            else:
                counts_acc = lax.fori_loop(
                    0,
                    n_chunks,
                    lambda ci, acc: _route_chunk_sparse_dyn(
                        pl.multiple_of(ci * rb, rb), acc
                    ),
                    counts_init,
                )
            scratch.counts_vmem[...] = jnp.broadcast_to(
                counts_acc, scratch.counts_vmem.shape
            )
            cp1 = pltpu.make_async_copy(
                scratch.counts_vmem, scratch.counts_smem, dma_sem
            )
            cp2 = pltpu.make_async_copy(
                scratch.route_pack_vmem, scratch.route_pack_smem, dma_sem
            )
            cp1.start()
            cp2.start()
            cp1.wait()
            cp2.wait()

        def _make_runner(fn, batch_size: int, ep: int, rep_eids: tuple[int, ...], **kw):
            e_per_owner = 256 // ep
            r_shapes = routing_mod.routing_scratch_shapes(
                batch_size, e_per_owner + len(rep_eids)
            )
            v_spec = pl.BlockSpec(memory_space=pltpu.VMEM)
            s_spec = pl.BlockSpec(memory_space=pltpu.SMEM)

            def _kern(owner_ref, lg_ref, b_ref, cnt_out_ref, pack_out_ref, r_scr, sem):
                fn(
                    lg_ref,
                    b_ref,
                    r_scr,
                    sem,
                    my_owner=owner_ref[0],
                    e_per_owner=e_per_owner,
                    holder_cand_eids=tuple(jnp.int32(x) for x in rep_eids),
                    **kw,
                )
                cnt_out_ref[...] = r_scr.counts_vmem[...]
                pack_out_ref[...] = r_scr.route_pack_vmem[...]
                # Cross-check SMEM async DMA landed identical contents.
                diff_cnt = jnp.sum(
                    jnp.abs(r_scr.counts_smem[0, :] - r_scr.counts_vmem[0, :])
                )
                cnt_out_ref[1, 0] = diff_cnt

            return jax.jit(
                pl.pallas_call(
                    _kern,
                    out_shape=(
                        jax.ShapeDtypeStruct((8, 256), jnp.int32),
                        jax.ShapeDtypeStruct((batch_size, 128), jnp.int32),
                    ),
                    grid_spec=pltpu.PrefetchScalarGridSpec(
                        num_scalar_prefetch=1,
                        grid=(1,),
                        in_specs=[v_spec, v_spec],
                        out_specs=[v_spec, v_spec],
                        scratch_shapes=[r_shapes, pltpu.SemaphoreType.DMA],
                    ),
                    interpret=True,
                )
            )

        scale_free = _BREAKERS_MOD.scale_free_expert_for_batch
        presets = [
            ("f7m80", 7, 0.20, 1, 0.80),
            ("f10m84", 10, 0.20, 1, 0.84),
            ("qf4m90", 4, 0.18, 2, 0.90),
            ("qf6m90", 6, 0.18, 2, 0.90),
        ]
        rng = np.random.default_rng(20261123)
        rep_eids = (200, 47, -1, 131)
        total_freed_experts = 0

        for b_sz in (16, 32, 64, 128):
            # Regime 1: tie-heavy quantized + duplicated token rows + tied router bias.
            q_levels = rng.integers(-6, 7, size=(b_sz // 2, 64)).astype(np.float32) * 0.25
            lg_ties = np.repeat(q_levels, 4, axis=1)  # 4-way exact expert score ties!
            lg_ties = np.concatenate([lg_ties, lg_ties[::-1]], axis=0)  # duplicated rows!
            bias_ties = (
                np.repeat(
                    rng.integers(-2, 3, size=(1, 64)).astype(np.float32) * 0.125,
                    4,
                    axis=1,
                )
            )

            # Regime 2: NaN pad rows + -inf rows + boundary FREE tail hits.
            lg_nan = rng.normal(-2.5, 0.9, size=(b_sz, 256)).astype(np.float32)
            # Force specific experts (e.g. eid=5, 21, 200) to have rare low-weight tail hits
            # and eid=6 to have 1 slot-0 hit + tail hits (must NOT be freed when min_slot>=1).
            lg_nan[:, 5] = -8.0
            lg_nan[0, 5] = -0.8
            lg_nan[0, :4] = 2.5
            lg_nan[:, 6] = -8.0
            lg_nan[0, 6] = 4.0  # slot-0 primary expert on row 0 -> un-droppable!
            if b_sz > 2:
                lg_nan[1, 6] = -0.9
                lg_nan[1, :4] = 2.5
            # Inject NaN pad rows (tail & interleaved) and an all -inf row.
            lg_nan[-5:, :] = np.nan
            lg_nan[2, :] = np.nan
            lg_nan[3, :128] = np.nan  # partial-NaN row!
            lg_nan[4, :] = -np.inf
            bias_zero = np.zeros((1, 256), dtype=np.float32)

            for name, f_rows, f_max_w, f_min_s, min_p in presets:
                for scale_mode in ("b64", "off"):
                    f_eff, ms_eff = scale_free(f_rows, f_min_s, b_sz, mode=scale_mode)
                    kw = dict(
                        route_chunk_rows=32,
                        min_p_topk=min_p,
                        free_max_rows=f_eff,
                        free_max_w=f_max_w,
                        free_min_slot=ms_eff,
                    )
                    kw_nofree = dict(
                        route_chunk_rows=32,
                        min_p_topk=min_p,
                        free_max_rows=0,
                        free_max_w=f_max_w,
                        free_min_slot=ms_eff,
                    )
                    old_run = _make_runner(_legacy_route_rb64, b_sz, 16, rep_eids, **kw)
                    new_run = _make_runner(
                        routing_mod.route_topk_and_counts_in_kernel,
                        b_sz,
                        16,
                        rep_eids,
                        route_rb16_mode=2 if scale_mode == "off" else 1,
                        **kw,
                    )
                    fast_run = _make_runner(
                        routing_mod.route_topk_and_counts_in_kernel,
                        b_sz,
                        16,
                        rep_eids,
                        route_fast=True,
                        **kw,
                    )
                    nofree_run = _make_runner(
                        _legacy_route_rb64, b_sz, 16, rep_eids, **kw_nofree
                    )
                    nofree_fast_run = _make_runner(
                        routing_mod.route_topk_and_counts_in_kernel,
                        b_sz,
                        16,
                        rep_eids,
                        route_fast=True,
                        **kw_nofree,
                    )

                    for reg_name, lg_np, b_np in (
                        ("ties", lg_ties, bias_ties),
                        ("nan_pad", lg_nan, bias_zero),
                    ):
                        lg_j = jnp.asarray(lg_np, dtype=jnp.bfloat16)
                        b_j = jnp.asarray(b_np, dtype=jnp.float32)
                        cnt_pre, pack_pre = nofree_run(jnp.array([0], jnp.int32), lg_j, b_j)
                        cnt_pre_fast, pack_pre_fast = nofree_fast_run(
                            jnp.array([0], jnp.int32), lg_j, b_j
                        )
                        pre_counts = np.asarray(cnt_pre[0])
                        self.assertTrue(
                            np.array_equal(np.asarray(cnt_pre[0]), np.asarray(cnt_pre_fast[0]))
                        )
                        self.assertTrue(
                            np.array_equal(np.asarray(pack_pre), np.asarray(pack_pre_fast))
                        )

                        for owner in (0, 11):
                            own_j = jnp.array([owner], dtype=jnp.int32)
                            cnt_old, pack_old = old_run(own_j, lg_j, b_j)
                            cnt_new, pack_new = new_run(own_j, lg_j, b_j)
                            cnt_fast, pack_fast = fast_run(own_j, lg_j, b_j)
                            cnt_old_np = np.asarray(cnt_old)
                            cnt_new_np = np.asarray(cnt_new)
                            cnt_fast_np = np.asarray(cnt_fast)
                            pack_old_np = np.asarray(pack_old)
                            pack_new_np = np.asarray(pack_new)
                            pack_fast_np = np.asarray(pack_fast)

                            ctx = f"{name}/{scale_mode}/B={b_sz}/{reg_name}/owner={owner}"
                            self.assertEqual(int(cnt_old_np[1, 0]), 0, ctx)
                            self.assertEqual(int(cnt_new_np[1, 0]), 0, ctx)
                            self.assertEqual(int(cnt_fast_np[1, 0]), 0, ctx)
                            # 1. Bit-exact post-FREE expert counts (counts_smem[0, :]).
                            self.assertTrue(
                                np.array_equal(cnt_old_np[0], cnt_new_np[0]),
                                f"counts mismatch in {ctx}",
                            )
                            self.assertTrue(
                                np.array_equal(cnt_old_np[0], cnt_fast_np[0]),
                                f"route_fast counts mismatch in {ctx}",
                            )
                            # 2. Bit-exact freed-expert set.
                            freed_old = (pre_counts > 0) & (cnt_old_np[0] == 0)
                            freed_new = (pre_counts > 0) & (cnt_new_np[0] == 0)
                            freed_fast = (pre_counts > 0) & (cnt_fast_np[0] == 0)
                            self.assertTrue(
                                np.array_equal(freed_old, freed_new),
                                f"freed-set mismatch in {ctx}",
                            )
                            self.assertTrue(
                                np.array_equal(freed_old, freed_fast),
                                f"route_fast freed-set mismatch in {ctx}",
                            )
                            total_freed_experts += int(np.sum(freed_new))
                            if reg_name == "nan_pad":
                                # Expert 6 has a slot-0 hit on row 0 -> never freed!
                                self.assertFalse(bool(freed_new[6]), ctx)
                                self.assertFalse(bool(freed_fast[6]), ctx)
                            # 3. Bit-exact loc_cnt, compacted topk slot IDs, AND IEEE-754 FP32 weight bits!
                            self.assertTrue(
                                np.array_equal(pack_old_np, pack_new_np),
                                f"route_pack_smem bit mismatch in {ctx}",
                            )
                            self.assertTrue(
                                np.array_equal(pack_old_np, pack_fast_np),
                                f"route_fast route_pack_smem bit mismatch in {ctx}",
                            )

                        # Cross-check against standalone route_weights reference (0-ULP).
                        w_ref = np.asarray(
                            routing_mod.route_weights(
                                lg_j,
                                b_j,
                                min_p_topk=min_p,
                                free_max_rows=f_eff,
                                free_max_w=f_max_w,
                                free_min_slot=ms_eff,
                            )
                        )
                        ref_counts = np.sum(w_ref > 0.0, axis=0, dtype=np.int32)
                        self.assertTrue(
                            np.array_equal(cnt_new_np[0], ref_counts),
                            f"route_weights count mismatch in {name}/{scale_mode}/B={b_sz}/{reg_name}",
                        )

        self.assertGreater(total_freed_experts, 0)

    def test_w9_ep_tp2_k512_route_fast_and_sparse_fast_gs_parity(self):
        """Verify EPxTP2 k512 + ROUTE_FAST + FAST_PLAN + FAST_GS + PAIR_RS + return_sharded
        Mosaic TPU lowering and CPU Pallas interpret 0.0-ULP race-free parity."""
        import dataclasses
        import sys
        from functools import partial

        for mod_name in (
            "sglang_tpu.moe.pallas.fused_rs_muse.fused_decode_routing",
            "sglang_tpu.moe.pallas.fused_rs_muse.fused_decode_ep_tp2",
            "sglang_tpu.moe.pallas.fused_rs_muse.fused_decode_op",
            "sglang_tpu.moe.pallas.fused_rs_muse.fused_decode_sparse",
        ):
            sys.modules.pop(mod_name, None)

        import jax
        import jax.numpy as jnp
        from jax.experimental import pallas as pl
        from jax.experimental.pallas import tpu as pltpu
        from jax.sharding import Mesh, PartitionSpec as P
        from sglang_tpu.moe.pallas.fused_rs_muse import fused_decode_routing as r_mod
        from sglang_tpu.moe.pallas.fused_rs_muse.fused_decode_ep_tp2 import (
            fused_decode_ep_tp2_from_logits,
        )
        from sglang_tpu.moe.pallas.fused_rs_muse.fused_decode_op import (
            ep_tp2_fallback_from_topk,
            muse_fuse_all_decode_op_name,
        )
        from sglang_tpu.moe.pallas.fused_rs_muse.fused_decode_sparse import (
            default_fp8_sparse_decode_config,
            default_sparse_decode_config,
            fused_decode_sparse_from_logits,
        )

        with mock.patch.dict(
            os.environ,
            {
                "SGLANG_TPU_MUSE_KERNEL_DECODE_SPEC_PRIME": "2",
                "SGLANG_TPU_MUSE_KERNEL_DECODE_ROUTE_FAST": "1",
            },
            clear=False,
        ):
            self.assertEqual(default_sparse_decode_config(64, 8, 512).spec_prime_parts, 0)
            self.assertEqual(default_fp8_sparse_decode_config(64, 8).spec_prime_parts, 0)

        # 1. Verify fused_decode_op k512 + return_sharded naming/validation for ep_tp2.
        op_sh = muse_fuse_all_decode_op_name(
            mode="ep_tp2",
            ep=8,
            tp=2,
            block=512,
            num_replicas=8,
            replica_cost_mode="tiles",
            return_sharded=True,
        )
        self.assertIn("muse_decode_ep8_tp2_k512_r8_tiles_sh", op_sh)

        # 2. CPU Pallas interpret (with detect_races=True) 0.0-ULP parity on:
        #    (a) pack_active_tokens_smem(fast_gs=True) vs fast_gs=False with replicas + suppressed hits
        #    (b) FAST_GS branchless 8-row gather + trash-row scatter (cur_rows <= 8 & cur_rows > 8 + NaN pad!)
        rng = np.random.default_rng(20261124)
        b_sz = 32
        max_exec = 20
        r_shapes = r_mod.routing_scratch_shapes(b_sz, max_exec)
        max_pk = int(r_shapes.packed_tok_ids.shape[0])

        # Build deterministic route_pack_smem, cand_exec_idx (with -1 suppressed replicas!), and exec_start.
        cand_map = np.full((max_exec,), -1, dtype=np.int32)
        active_cands = [0, 2, 5, 9, 14, 18]  # 6 active experts; others (-1) suppressed!
        counts_per_act = [11, 5, 1, 8, 3, 2]  # covers cur_rows=1,2,3,5,8 AND >8 (11)!
        starts = np.zeros((max_exec + 1,), dtype=np.int32)
        cur_off = 0
        for k_i, (c_id, cnt_v) in enumerate(zip(active_cands, counts_per_act)):
            cand_map[c_id] = k_i
            starts[k_i] = cur_off
            cur_off += cnt_v
        starts[len(active_cands)] = cur_off

        pack_arr = np.zeros((b_sz, 128), dtype=np.int32)
        rem_per_cand = {c: n for c, n in zip(active_cands, counts_per_act)}
        suppressed_cands = [1, 3, 7, 12, 19]
        for t in range(b_sz - 2):  # last 2 tokens are NaN pad tokens (0 hits)
            hits_c = []
            for c_id in active_cands:
                if rem_per_cand[c_id] > 0 and (rng.random() < 0.45 or t >= b_sz - 12):
                    hits_c.append(c_id)
                    rem_per_cand[c_id] -= 1
                    if len(hits_c) == 6:
                        break
            # Inject 1-2 suppressed replica hits (k == -1) on every token!
            hits_c.append(int(rng.choice(suppressed_cands)))
            pack_arr[t, 0] = len(hits_c)
            for j, c_id in enumerate(hits_c):
                pack_arr[t, 1 + j] = c_id
                w_f32 = np.float32(rng.uniform(0.05, 0.4))
                pack_arr[t, 16 + j] = w_f32.view(np.int32)
        # Drain any remaining counts onto early tokens
        for c_id, rem_n in list(rem_per_cand.items()):
            for t in range(b_sz - 2):
                if rem_n <= 0:
                    break
                cur_n = int(pack_arr[t, 0])
                existing = set(pack_arr[t, 1 : 1 + cur_n].tolist())
                if c_id not in existing and cur_n < 8:
                    pack_arr[t, 1 + cur_n] = c_id
                    pack_arr[t, 16 + cur_n] = np.float32(0.125).view(np.int32)
                    pack_arr[t, 0] = cur_n + 1
                    rem_n -= 1
            self.assertEqual(rem_n, 0)

        x_in = rng.normal(0.0, 1.0, size=(b_sz, 2, 128)).astype(np.float32)
        x_in[-2:, :, :] = np.nan  # NaN pad rows must NEVER leak into valid rows!
        x_bf16 = jnp.asarray(x_in, dtype=jnp.bfloat16)

        def _make_gs_kernel(use_fast_gs: bool):
            acc_rows = (b_sz + 1) if use_fast_gs else b_sz

            def _kern(pack_in_ref, cand_in_ref, start_in_ref, x_ref, out_acc_ref, io_buf, acc_buf, p_ids, p_w, w_ptr):
                # Initialize SMEM state
                for m_i in range(max_exec + 1):
                    w_ptr[m_i] = start_in_ref[m_i]
                for p_i in range(max_pk):
                    p_ids[p_i] = jnp.int32(b_sz - 1)  # poison with NaN pad token ID!
                    p_w[p_i] = jnp.float32(999.0)
                fake_scratch = r_mod.RoutingScratch(
                    counts_vmem=None,
                    counts_smem=None,
                    route_pack_vmem=None,
                    route_pack_smem=pack_in_ref,
                    exec_expert_id=None,
                    exec_weight_src=None,
                    exec_weight_slot=None,
                    exec_start=start_in_ref,
                    exec_write_ptr=w_ptr,
                    cand_exec_idx=cand_in_ref,
                    packed_tok_ids=p_ids,
                    packed_tok_w=p_w,
                )
                r_mod.pack_active_tokens_smem(
                    b_sz, fake_scratch, has_replicas=True, fast_gs=use_fast_gs
                )
                # Poison scratch VMEM with NaNs before gather/scatter (#2770 guardrail)!
                io_buf[...] = jnp.full((16, 2, 128), jnp.nan, dtype=jnp.bfloat16)
                acc_buf[...] = jnp.full((acc_rows, 2, 128), jnp.nan, dtype=jnp.float32)
                acc_buf[pl.ds(0, b_sz), :, :] = jnp.zeros((b_sz, 2, 128), dtype=jnp.float32)
                if use_fast_gs:
                    acc_buf[pl.ds(b_sz, 1), :, :] = jnp.zeros((1, 2, 128), dtype=jnp.float32)

                def _exp_loop(k_idx, _):
                    base_p = start_in_ref[k_idx]
                    cur_r = start_in_ref[k_idx + 1] - base_p
                    if use_fast_gs:
                        safe_b = jnp.minimum(base_p, jnp.int32(max_pk - 1))
                        ts8 = [
                            jnp.clip(
                                p_ids[jnp.where(jnp.int32(s) < cur_r, base_p + s, safe_b)],
                                jnp.int32(0),
                                jnp.int32(b_sz - 1),
                            )
                            for s in range(8)
                        ]
                        io_buf[pl.ds(0, 8), :, :] = jnp.concatenate(
                            [x_ref[pl.ds(t, 1), :, :] for t in ts8], axis=0
                        )

                        @pl.when(cur_r > 8)
                        def _ghi():
                            def _gt(i, _c):
                                tok = p_ids[base_p + i]
                                io_buf[pl.ds(i, 1), :, :] = x_ref[pl.ds(tok, 1), :, :]
                                return _c

                            jax.lax.fori_loop(8, cur_r, _gt, jnp.int32(0))

                        # Simulate expert transform on gathered rows (2x scale in bf16)
                        io_buf[...] = (io_buf[...].astype(jnp.float32) * 1.5).astype(jnp.bfloat16)
                        y8_f = io_buf[pl.ds(0, 8), :, :].astype(jnp.float32)
                        for s in range(8):
                            valid_s = jnp.int32(s) < cur_r
                            p_s = jnp.where(valid_s, base_p + s, safe_b)
                            tok = jnp.where(valid_s, p_ids[p_s], jnp.int32(b_sz))
                            w = jnp.where(valid_s, p_w[p_s], jnp.float32(0.0))
                            acc_buf[pl.ds(tok, 1), :, :] = (
                                acc_buf[pl.ds(tok, 1), :, :] + y8_f[s : s + 1, :, :] * w
                            )

                        @pl.when(cur_r > 8)
                        def _shi():
                            def _st(i, _c):
                                tok = p_ids[base_p + i]
                                w = p_w[base_p + i]
                                yr = io_buf[pl.ds(i, 1), :, :].astype(jnp.float32)
                                acc_buf[pl.ds(tok, 1), :, :] = acc_buf[pl.ds(tok, 1), :, :] + yr * w
                                return _c

                            jax.lax.fori_loop(8, cur_r, _st, jnp.int32(0))
                    else:
                        def _g_ref(i, _c):
                            tok = p_ids[base_p + i]
                            io_buf[pl.ds(i, 1), :, :] = x_ref[pl.ds(tok, 1), :, :]
                            return _c

                        jax.lax.fori_loop(0, cur_r, _g_ref, jnp.int32(0))
                        io_buf[...] = (io_buf[...].astype(jnp.float32) * 1.5).astype(jnp.bfloat16)

                        def _s_ref(i, _c):
                            tok = p_ids[base_p + i]
                            w = p_w[base_p + i]
                            yr = io_buf[pl.ds(i, 1), :, :].astype(jnp.float32)
                            acc_buf[pl.ds(tok, 1), :, :] = acc_buf[pl.ds(tok, 1), :, :] + yr * w
                            return _c

                        jax.lax.fori_loop(0, cur_r, _s_ref, jnp.int32(0))
                    return _

                jax.lax.fori_loop(0, len(active_cands), _exp_loop, jnp.int32(0))
                out_acc_ref[...] = acc_buf[pl.ds(0, b_sz), :, :]

            return pl.pallas_call(
                _kern,
                out_shape=jax.ShapeDtypeStruct((b_sz, 2, 128), jnp.float32),
                grid_spec=pltpu.PrefetchScalarGridSpec(
                    num_scalar_prefetch=0,
                    grid=(1,),
                    in_specs=(
                        pl.BlockSpec(memory_space=pltpu.SMEM),
                        pl.BlockSpec(memory_space=pltpu.SMEM),
                        pl.BlockSpec(memory_space=pltpu.SMEM),
                        pl.BlockSpec(memory_space=pltpu.VMEM),
                    ),
                    out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
                    scratch_shapes=(
                        pltpu.VMEM((16, 2, 128), jnp.bfloat16),
                        pltpu.VMEM((acc_rows, 2, 128), jnp.float32),
                        pltpu.SMEM((max_pk,), jnp.int32),
                        pltpu.SMEM((max_pk,), jnp.float32),
                        pltpu.SMEM((max_exec + 1,), jnp.int32),
                    ),
                ),
                interpret=pltpu.InterpretParams(
                    detect_races=True,
                    **(
                        {"uninitialized_memory": "nan"}
                        if "uninitialized_memory"
                        in pltpu.InterpretParams.__annotations__
                        else {}
                    ),
                ),
            )(
                jnp.asarray(pack_arr, dtype=jnp.int32),
                jnp.asarray(cand_map, dtype=jnp.int32),
                jnp.asarray(starts, dtype=jnp.int32),
                x_bf16,
            )

        out_ref_gs = np.asarray(_make_gs_kernel(False))
        out_fast_gs = np.asarray(_make_gs_kernel(True))
        self.assertFalse(np.isnan(out_fast_gs[:-2]).any())
        self.assertTrue(
            np.array_equal(out_ref_gs, out_fast_gs),
            f"FAST_GS gather/scatter/pack must be 0-ULP bit-exact; max diff={np.max(np.abs(out_ref_gs - out_fast_gs))}",
        )

        # 3. Verify Mosaic TPU lowering of EP4xTP2 k512 (Wave-9 bundle + return_sharded + pair_rs + fallback)
        #    and muse_sparse FAST_GS=1 across 8 XLA devices!
        from jax.experimental.topologies import get_topology_desc

        tpu_devs = get_topology_desc("tpu7x:2x2x1", platform="tpu").devices
        if len(tpu_devs) >= 8:
            mesh = Mesh(
                np.array(
                    sorted(tpu_devs, key=lambda d: (d.coords[1], d.coords[0], d.core_on_chip))
                ),
                ("model",),
            )
            h, i_half, block, r_slots = 4096, 2048, 512, 2
            e_chip = 64
            in_shapes_tp2 = (
                jax.ShapeDtypeStruct((1,), jnp.int32),
                jax.ShapeDtypeStruct((64, 256), jnp.float32),
                jax.ShapeDtypeStruct((1, 256), jnp.float32),
                jax.ShapeDtypeStruct((64, h), jnp.bfloat16),
                jax.ShapeDtypeStruct((e_chip, h, 2 * i_half), jnp.float4_e2m1fn),
                jax.ShapeDtypeStruct((e_chip, h // block, 2 * i_half), jnp.bfloat16),
                jax.ShapeDtypeStruct((e_chip, i_half, h), jnp.float4_e2m1fn),
                jax.ShapeDtypeStruct((e_chip, i_half // block, h), jnp.bfloat16),
                jax.ShapeDtypeStruct((1, h), jnp.bfloat16),
                jax.ShapeDtypeStruct((r_slots, h, 2 * i_half), jnp.float4_e2m1fn),
                jax.ShapeDtypeStruct((r_slots, h // block, 2 * i_half), jnp.bfloat16),
                jax.ShapeDtypeStruct((r_slots, i_half, h), jnp.float4_e2m1fn),
                jax.ShapeDtypeStruct((r_slots, i_half // block, h), jnp.bfloat16),
                jax.ShapeDtypeStruct((4, 4 * r_slots), jnp.int32),
            )
            for pair_rs in (False, True):
                for ret_sh in (False, True):
                    sm_tp2 = jax.jit(
                        jax.shard_map(
                            partial(
                                fused_decode_ep_tp2_from_logits,
                                ep=4,
                                tp=2,
                                replica_cost_mode="tiles",
                                return_sharded=ret_sh,
                                min_p_topk=0.90,
                                free_max_rows=6,
                                free_max_w=0.18,
                                free_min_slot=1,
                                static_holder_mask=True,
                                route_fast=True,
                                fast_plan=2,
                                unroll_strips=True,
                                fast_gs=True,
                                pair_rs=pair_rs,
                            ),
                            mesh=mesh,
                            in_specs=(P(),) * len(in_shapes_tp2),
                            out_specs=P(),
                            check_vma=False,
                        )
                    )
                    hlo_tp2 = (
                        sm_tp2.trace(*in_shapes_tp2)
                        .lower(lowering_platforms=("tpu",))
                        .as_text("hlo")
                    )
                    self.assertIn("tpu_custom_call", hlo_tp2)

            # Verify ep_tp2_fallback_from_topk k512 Mosaic TPU lowering (B=8 warmup).
            fb_shapes = (
                jax.ShapeDtypeStruct((1,), jnp.int32),
                jax.ShapeDtypeStruct((8, h), jnp.bfloat16),
                jax.ShapeDtypeStruct((8, 8), jnp.float32),
                jax.ShapeDtypeStruct((8, 8), jnp.int32),
                in_shapes_tp2[4],
                in_shapes_tp2[5],
                in_shapes_tp2[6],
                in_shapes_tp2[7],
                in_shapes_tp2[8],
            )
            sm_fb = jax.jit(
                jax.shard_map(
                    partial(ep_tp2_fallback_from_topk, ep=4, tp=2, chipmate_xor=1),
                    mesh=mesh,
                    in_specs=(P(),) * len(fb_shapes),
                    out_specs=P(),
                    check_vma=False,
                )
            )
            hlo_fb = sm_fb.trace(*fb_shapes).lower(lowering_platforms=("tpu",)).as_text("hlo")
            self.assertIn("tpu_custom_call", hlo_fb)

            # Verify muse_sparse FAST_GS=1 Mosaic TPU lowering on EP=8 k512!
            cfg_fgs = dataclasses.replace(
                default_sparse_decode_config(64, 8, 512),
                static_holder_mask=True,
                route_fast=True,
                fast_plan=2,
                unroll_strips=True,
                fast_gs=True,
            )
            in_shapes_sp = (
                jax.ShapeDtypeStruct((1,), jnp.int32),
                jax.ShapeDtypeStruct((64, 256), jnp.float32),
                jax.ShapeDtypeStruct((1, 256), jnp.float32),
                jax.ShapeDtypeStruct((64, h), jnp.bfloat16),
                jax.ShapeDtypeStruct((32, h, 8192), jnp.float4_e2m1fn),
                jax.ShapeDtypeStruct((32, h // 512, 8192), jnp.bfloat16),
                jax.ShapeDtypeStruct((32, 4096, h), jnp.float4_e2m1fn),
                jax.ShapeDtypeStruct((32, 4096 // 512, h), jnp.bfloat16),
                jax.ShapeDtypeStruct((1, h), jnp.bfloat16),
                jax.ShapeDtypeStruct((r_slots, h, 8192), jnp.float4_e2m1fn),
                jax.ShapeDtypeStruct((r_slots, h // 512, 8192), jnp.bfloat16),
                jax.ShapeDtypeStruct((r_slots, 4096, h), jnp.float4_e2m1fn),
                jax.ShapeDtypeStruct((r_slots, 4096 // 512, h), jnp.bfloat16),
                jax.ShapeDtypeStruct((4, 8 * r_slots), jnp.int32),
            )
            sm_sp = jax.jit(
                jax.shard_map(
                    partial(
                        fused_decode_sparse_from_logits,
                        ep=8,
                        return_sharded=True,
                        replica_cost_mode="tiles",
                        config=cfg_fgs,
                    ),
                    mesh=mesh,
                    in_specs=(P(),) * len(in_shapes_sp),
                    out_specs=P(),
                    check_vma=False,
                )
            )
            hlo_sp = sm_sp.trace(*in_shapes_sp).lower(lowering_platforms=("tpu",)).as_text("hlo")
            self.assertIn("tpu_custom_call", hlo_sp)

    def test_w9_spec_prime_fast_with_fast_plan_parity_and_lowering(self):
        """Verify SPEC_PRIME_FAST opt-in enables SPEC_PRIME alongside FAST_PLAN=2 with 0-ULP parity."""
        import dataclasses
        from functools import partial

        v1_mod = _load_pkg_module(
            "fused_decode_sparse", _MUSE_DIR / "fused_decode_sparse.py"
        )
        routing_mod = _load_pkg_module(
            "fused_decode_routing", _MUSE_DIR / "fused_decode_routing.py"
        )

        # 1. Verify default-OFF gating of SPEC_PRIME_FAST & SPEC_HOME_SLOT.
        with mock.patch.dict(
            os.environ,
            {
                "SGLANG_TPU_MUSE_KERNEL_DECODE_ROUTE_FAST": "1",
                "SGLANG_TPU_MUSE_KERNEL_DECODE_FAST_PLAN": "2",
                "SGLANG_TPU_MUSE_KERNEL_DECODE_SPEC_PRIME": "2",
                "SGLANG_TPU_MUSE_KERNEL_DECODE_SPEC_PRIME_FAST": "0",
            },
            clear=False,
        ):
            cfg_off = v1_mod.default_sparse_decode_config(64, 8, 512)
            self.assertEqual(cfg_off.spec_prime_parts, 0)
            self.assertEqual(cfg_off.spec_home_slots, ())

        with mock.patch.dict(
            os.environ,
            {
                "SGLANG_TPU_MUSE_KERNEL_DECODE_ROUTE_FAST": "1",
                "SGLANG_TPU_MUSE_KERNEL_DECODE_FAST_PLAN": "2",
                "SGLANG_TPU_MUSE_KERNEL_DECODE_SPEC_PRIME": "2",
                "SGLANG_TPU_MUSE_KERNEL_DECODE_SPEC_PRIME_FAST": "1",
                "SGLANG_TPU_MUSE_KERNEL_DECODE_SPEC_HOME_SLOT": "29",
            },
            clear=False,
        ):
            cfg_on = v1_mod.default_sparse_decode_config(64, 8, 512)
            self.assertEqual(cfg_on.spec_prime_parts, 2)
            self.assertEqual(cfg_on.spec_home_slots, (29,) * 8)
            fp8_on = v1_mod.default_fp8_sparse_decode_config(64, 8)
            self.assertEqual(fp8_on.spec_prime_parts, 2)
            self.assertEqual(fp8_on.spec_home_slots, (29,) * 8)
            # EP=16 e_local=16 rejects out-of-range slot=29 safely.
            cfg_ep16 = v1_mod.default_sparse_decode_config(64, 16, 512)
            self.assertEqual(cfg_ep16.spec_home_slots, ())

        try:
            import jax
            import jax.numpy as jnp
            from jax.experimental import pallas as pl
            from jax.experimental.pallas import tpu as pltpu
            from jax.experimental.topologies import get_topology_desc
            from jax.sharding import Mesh, PartitionSpec as P
        except Exception as exc:
            self.skipTest(f"JAX/Pallas TPU unavailable: {exc}")

        # 2. SMEM exec-plan parity across hit/suppressed/zero/empty/slot0/slot31/disabled under nan-scratch.
        interp_kw = {
            "detect_races": True,
            **(
                {"uninitialized_memory": "nan"}
                if "uninitialized_memory" in pltpu.InterpretParams.__annotations__
                else {}
            ),
        }

        def _check_plan(fast_p: bool, with_rep: bool, spec_val: int, cnt_np, sup_np, rep_np):
            e_loc, num_rep = 32, 8
            max_exec = e_loc + (num_rep if with_rep else 0)
            r_sh = routing_mod.routing_scratch_shapes(64, max_exec)

            def _k(me_ref, sp_ref, cnt_ref, sup_ref, rep_ref, out_ref, r_sc):
                r_sc.counts_smem[...] = cnt_ref[...]
                if with_rep:
                    ne = routing_mod.build_exec_plan_from_masks_smem(
                        me_ref[0],
                        e_loc,
                        num_rep,
                        sup_ref,
                        rep_ref,
                        r_sc,
                        spec_slot=sp_ref[0],
                        fast_plan=fast_p,
                    )
                else:
                    ne = routing_mod.build_default_exec_plan_smem(
                        me_ref[0],
                        e_loc,
                        r_sc,
                        spec_slot=sp_ref[0],
                        fast_plan=fast_p,
                    )
                out_ref[0, 0] = ne
                for i in range(max_exec):
                    out_ref[1, i] = r_sc.cand_exec_idx[i]
                    out_ref[2, i] = jnp.where(
                        i < ne, r_sc.exec_expert_id[i], jnp.int32(-1)
                    )
                    out_ref[3, i] = jnp.where(
                        i < ne, r_sc.exec_weight_src[i], jnp.int32(-1)
                    )
                    out_ref[4, i] = jnp.where(
                        i < ne, r_sc.exec_weight_slot[i], jnp.int32(-1)
                    )
                for i in range(max_exec + 1):
                    out_ref[5, i] = jnp.where(
                        i <= ne, r_sc.exec_start[i], jnp.int32(-1)
                    )

            fn = pl.pallas_call(
                _k,
                out_shape=jax.ShapeDtypeStruct((6, max_exec + 1), jnp.int32),
                grid_spec=pltpu.PrefetchScalarGridSpec(
                    num_scalar_prefetch=5,
                    grid=(1,),
                    in_specs=(),
                    out_specs=pl.BlockSpec(memory_space=pltpu.SMEM),
                    scratch_shapes=(r_sh,),
                ),
                interpret=pltpu.InterpretParams(**interp_kw),
            )
            return np.asarray(
                fn(
                    jnp.array([2], jnp.int32),
                    jnp.array([spec_val], jnp.int32),
                    jnp.asarray(cnt_np, jnp.int32),
                    jnp.asarray(sup_np, jnp.int32),
                    jnp.asarray(rep_np, jnp.int32),
                )
            )

        rng = np.random.default_rng(1234)
        for mode_name, sp_v in (
            ("normal_hit", 17),
            ("suppressed_miss", 5),
            ("zero_cnt_miss", 12),
            ("all_zero", 3),
            ("disabled_neg1", -1),
            ("slot0_hit", 0),
            ("slot31_hit", 31),
        ):
            cnt = np.zeros((8, 256), dtype=np.int32)
            if mode_name != "all_zero":
                cnt[0] = rng.choice([0, 0, 1, 4, 15], size=256)
                if mode_name in ("normal_hit", "slot0_hit", "slot31_hit", "suppressed_miss"):
                    cnt[0, 2 * 32 + max(0, sp_v)] = 9
                if mode_name == "zero_cnt_miss":
                    cnt[0, 2 * 32 + sp_v] = 0
            sup = (rng.random(32) < 0.25).astype(np.int32)
            if mode_name == "suppressed_miss":
                sup[sp_v] = 1
            elif sp_v >= 0:
                sup[sp_v] = 0
            rep = np.array([10, -1, 200, 145, -1, 3, -1, 250], dtype=np.int32)
            for w_rep in (True, False):
                o_ref = _check_plan(False, w_rep, sp_v, cnt, sup, rep)
                o_fast = _check_plan(True, w_rep, sp_v, cnt, sup, rep)
                self.assertTrue(
                    np.array_equal(o_ref, o_fast),
                    f"spec_prime fast_plan SMEM mismatch on {mode_name} (rep={w_rep}, sp={sp_v})",
                )

        # 3. Verify TPU7x Mosaic lowering for sp=1 (overlapping prime-rest) & sp=2 with fast_plan=2.
        tpu_devs = get_topology_desc("tpu7x:2x2x1", platform="tpu").devices
        if len(tpu_devs) >= 8:
            mesh = Mesh(
                np.array(
                    sorted(tpu_devs, key=lambda d: (d.coords[1], d.coords[0], d.core_on_chip))
                ),
                ("model",),
            )
            h, r_slots = 4096, 8
            in_shapes_sp = (
                jax.ShapeDtypeStruct((1,), jnp.int32),
                jax.ShapeDtypeStruct((64, 256), jnp.float32),
                jax.ShapeDtypeStruct((1, 256), jnp.float32),
                jax.ShapeDtypeStruct((64, h), jnp.bfloat16),
                jax.ShapeDtypeStruct((32, h, 8192), jnp.float4_e2m1fn),
                jax.ShapeDtypeStruct((32, h // 512, 8192), jnp.bfloat16),
                jax.ShapeDtypeStruct((32, 4096, h), jnp.float4_e2m1fn),
                jax.ShapeDtypeStruct((32, 4096 // 512, h), jnp.bfloat16),
                jax.ShapeDtypeStruct((1, h), jnp.bfloat16),
                jax.ShapeDtypeStruct((r_slots, h, 8192), jnp.float4_e2m1fn),
                jax.ShapeDtypeStruct((r_slots, h // 512, 8192), jnp.bfloat16),
                jax.ShapeDtypeStruct((r_slots, 4096, h), jnp.float4_e2m1fn),
                jax.ShapeDtypeStruct((r_slots, 4096 // 512, h), jnp.bfloat16),
                jax.ShapeDtypeStruct((4, 8 * r_slots), jnp.int32),
            )
            base_cfg = v1_mod.default_sparse_decode_config(64, 8, 512)
            for sp_parts in (1, 2):
                cfg_sp = dataclasses.replace(
                    base_cfg,
                    static_holder_mask=True,
                    route_fast=True,
                    fast_plan=2,
                    unroll_strips=True,
                    spec_prime_parts=sp_parts,
                    spec_home_slots=(29,) * 8,
                )
                sm_sp = jax.jit(
                    jax.shard_map(
                        partial(
                            v1_mod.fused_decode_sparse_from_logits,
                            ep=8,
                            return_sharded=True,
                            replica_cost_mode="tiles",
                            config=cfg_sp,
                        ),
                        mesh=mesh,
                        in_specs=(P(),) * len(in_shapes_sp),
                        out_specs=P(),
                        check_vma=False,
                    )
                )
                comp_sp = sm_sp.trace(*in_shapes_sp).lower(lowering_platforms=("tpu",)).compile()
                hlo_sp = comp_sp.as_text()
                self.assertIn("tpu_custom_call", hlo_sp)
                self.assertGreater(comp_sp.memory_analysis().generated_code_size_in_bytes, 0)

    def test_w9_per_layer_hot_spec_slot_selection_and_lowering(self):
        """Verify per-(layer, owner) hot-slot selection, op naming, and runtime SMEM lowering."""
        import dataclasses
        from functools import partial
        import jax
        import jax.numpy as jnp
        from jax.experimental.topologies import get_topology_desc
        from jax.sharding import Mesh, PartitionSpec as P
        import numpy as np

        from sglang_tpu.moe.pallas.fused_rs_muse import (
            fused_decode_op as op_mod,
            fused_decode_sparse as v1_mod,
            moe_imbalance_planner as planner,
        )

        # 1. Per-owner hottest non-replicated slot selection (placement + replica_meta + ties).
        counts = np.zeros(256, dtype=np.float64)
        counts[17] = 100.0   # owner 0 hottest, replicated below
        counts[5] = 80.0     # owner 0 hottest non-replicated -> expected slot 5
        counts[64 + 10] = 50.0
        counts[64 + 3] = 50.0  # tie on owner 2 -> smallest slot (3)
        placement = np.arange(256, dtype=np.int32)
        placement[32 + 9], placement[32 + 12] = placement[32 + 12], placement[32 + 9]
        counts[32 + 12] = 90.0  # placed at runtime slot (1, 9)
        replica_meta = np.full((4, 16), -1, dtype=np.int32)
        replica_meta[0, 0] = 17  # runtime expert 17 (owner 0, slot 17) is replicated
        # Replicate all 32 slots of owner 7 -> fallback selects hottest overall on owner 7.
        all_o7_meta = np.full((4, 32), -1, dtype=np.int32)
        all_o7_meta[0, :32] = np.arange(7 * 32, 8 * 32, dtype=np.int32)
        counts[7 * 32 + 21] = 75.0
        combined_meta = np.concatenate([replica_meta, all_o7_meta], axis=1)

        hot = planner.select_hot_non_replicated_home_slots(
            counts,
            num_owners=8,
            replica_meta=combined_meta,
            placement=placement,
        )
        self.assertEqual(hot.dtype, np.int32)
        self.assertEqual(hot.shape, (8,))
        self.assertEqual(int(hot[0]), 5)
        self.assertEqual(int(hot[1]), 9)
        self.assertEqual(int(hot[2]), 3)
        self.assertEqual(int(hot[7]), 21)

        # 2. TorchTPU op name and mode validation for has_spec_slot.
        ss_name = op_mod.muse_fuse_all_decode_op_name(
            mode="sparse",
            ep=8,
            tp=1,
            block=512,
            num_replicas=8,
            replica_cost_mode="tiles",
            return_sharded=True,
            has_spec_slot=True,
        )
        self.assertEqual(
            ss_name,
            "sglang_tpu_pallas::muse_decode_sparse_ep8_k512_r8_tiles_ss_sh",
        )
        with self.assertRaises(ValueError):
            op_mod.muse_fuse_all_decode_op_name(
                mode="sparse_v2",
                ep=8,
                tp=1,
                block=512,
                has_spec_slot=True,
            )

        # 3. TPU7x Mosaic lowering with runtime spec_home_slot SMEM arg (R=0 and R=8).
        tpu_devs = get_topology_desc("tpu7x:2x2x1", platform="tpu").devices
        if len(tpu_devs) >= 8:
            mesh = Mesh(
                np.array(
                    sorted(tpu_devs, key=lambda d: (d.coords[1], d.coords[0], d.core_on_chip))
                ),
                ("model",),
            )
            h, r_slots = 4096, 8
            base_shapes = (
                jax.ShapeDtypeStruct((1,), jnp.int32),
                jax.ShapeDtypeStruct((64, 256), jnp.float32),
                jax.ShapeDtypeStruct((1, 256), jnp.float32),
                jax.ShapeDtypeStruct((64, h), jnp.bfloat16),
                jax.ShapeDtypeStruct((32, h, 8192), jnp.float4_e2m1fn),
                jax.ShapeDtypeStruct((32, h // 512, 8192), jnp.bfloat16),
                jax.ShapeDtypeStruct((32, 4096, h), jnp.float4_e2m1fn),
                jax.ShapeDtypeStruct((32, 4096 // 512, h), jnp.bfloat16),
                jax.ShapeDtypeStruct((1, h), jnp.bfloat16),
            )
            rep_shapes = (
                jax.ShapeDtypeStruct((r_slots, h, 8192), jnp.float4_e2m1fn),
                jax.ShapeDtypeStruct((r_slots, h // 512, 8192), jnp.bfloat16),
                jax.ShapeDtypeStruct((r_slots, 4096, h), jnp.float4_e2m1fn),
                jax.ShapeDtypeStruct((r_slots, 4096 // 512, h), jnp.bfloat16),
                jax.ShapeDtypeStruct((4, 8 * r_slots), jnp.int32),
            )
            ss_shape = jax.ShapeDtypeStruct((1,), jnp.int32)
            base_cfg = v1_mod.default_sparse_decode_config(64, 8, 512)
            cfg_dyn = dataclasses.replace(
                base_cfg,
                static_holder_mask=True,
                route_fast=True,
                fast_plan=2,
                unroll_strips=True,
                spec_prime_parts=2,
                spec_home_slots=(),
            )
            for with_rep in (False, True):
                shapes = (
                    (*base_shapes, *rep_shapes, ss_shape)
                    if with_rep
                    else (*base_shapes, ss_shape)
                )
                def _call_dyn(*args):
                    return v1_mod.fused_decode_sparse_from_logits(
                        *args[:-1],
                        spec_home_slot=args[-1],
                        ep=8,
                        return_sharded=True,
                        replica_cost_mode="tiles",
                        config=cfg_dyn,
                    )

                sm_dyn = jax.jit(
                    jax.shard_map(
                        _call_dyn,
                        mesh=mesh,
                        in_specs=(P(),) * len(shapes),
                        out_specs=P(),
                        check_vma=False,
                    )
                )
                comp_dyn = sm_dyn.trace(*shapes).lower(lowering_platforms=("tpu",)).compile()
                self.assertIn("tpu_custom_call", comp_dyn.as_text())
                self.assertGreater(
                    comp_dyn.memory_analysis().generated_code_size_in_bytes, 0
                )

    def test_w9_dflash_verify_b256_b384_route_fast_w4a8_free_and_ep16(self):
        """Verify ROUTE_FAST B>=256 parity, VERIFY FREE/MIN_P, LHS_FP8/TM128, and EP=16 B=256/384 lowering."""
        import dataclasses
        from functools import partial
        import os
        from unittest import mock
        import jax
        import jax.numpy as jnp
        from jax.experimental import pallas as pl
        from jax.experimental.pallas import tpu as pltpu
        from jax.experimental.topologies import get_topology_desc
        from jax.sharding import Mesh, PartitionSpec as P
        import numpy as np

        from sglang_tpu.moe.pallas.fused_rs_muse import (
            fused_decode_routing as rt_mod,
            fused_decode_sparse as v1_mod,
            fused_decode_sparse_v2 as v2_mod,
        )

        # 1. Verify-scaled FREE + MIN_P_TOPK + LHS_FP8 + TILE_M128 config resolution.
        verify_env = {
            "SGLANG_TPU_MUSE_DECODE_FREE_EXPERT_ROWS": "10:0.20:1",
            "SGLANG_TPU_MUSE_DECODE_MIN_P_TOPK": "0.84",
            "SGLANG_TPU_MUSE_DECODE_VERIFY_FREE_EXPERT_ROWS": "off",
            "SGLANG_TPU_MUSE_DECODE_VERIFY_MIN_P_TOPK": "0.0",
            "SGLANG_TPU_MUSE_KERNEL_DECODE_LHS_FP8": "1",
            "SGLANG_TPU_MUSE_KERNEL_DECODE_TILE_M128": "1",
            "SGLANG_TPU_MUSE_KERNEL_DECODE_ROUTE_FAST": "1",
            "SGLANG_TPU_MUSE_KERNEL_DECODE_FAST_PLAN": "2",
            "SGLANG_TPU_MUSE_KERNEL_DECODE_PAIR_RS": "1",
        }
        with mock.patch.dict(os.environ, verify_env, clear=False):
            cfg64 = v1_mod.default_sparse_decode_config(64, 8, 512)
            self.assertEqual(cfg64.free_max_rows, 10)
            self.assertAlmostEqual(cfg64.min_p_topk, 0.84)
            self.assertFalse(cfg64.lhs_fp8)
            self.assertEqual(cfg64.tile_m, 32)

            cfg256 = v1_mod.default_sparse_decode_config(256, 16, 512)
            self.assertEqual(cfg256.free_max_rows, 0)
            self.assertAlmostEqual(cfg256.min_p_topk, 0.0)
            self.assertTrue(cfg256.lhs_fp8)
            self.assertEqual(cfg256.tile_m, 128)
            self.assertEqual(cfg256.m_buckets, (16, 32, 64, 128))
            self.assertTrue(cfg256.pair_rs)
            self.assertTrue(cfg256.bridge_ag)

            cfg_fp8_256 = v1_mod.default_fp8_sparse_decode_config(256, 16)
            self.assertEqual(cfg_fp8_256.free_max_rows, 0)
            self.assertAlmostEqual(cfg_fp8_256.min_p_topk, 0.0)
            self.assertTrue(cfg_fp8_256.pair_rs)
            self.assertTrue(cfg_fp8_256.bridge_ag)

            cfg_v2_256 = v2_mod.default_sparse_decode_v2_config(256, 8, 512)
            self.assertEqual(cfg_v2_256.routing_prune.free_expert_rows, 0)
            self.assertEqual(cfg_v2_256.routing_prune.cum_mass, 0.0)

        explicit_verify_env = {
            "SGLANG_TPU_MUSE_DECODE_FREE_EXPERT_ROWS": "10:0.20:1",
            "SGLANG_TPU_MUSE_DECODE_MIN_P_TOPK": "0.84",
            "SGLANG_TPU_MUSE_DECODE_VERIFY_FREE_EXPERT_ROWS": "24:0.19:2",
            "SGLANG_TPU_MUSE_DECODE_VERIFY_MIN_P_TOPK": "0.88",
        }
        with mock.patch.dict(os.environ, explicit_verify_env, clear=False):
            cfg_exp = v1_mod.default_sparse_decode_config(256, 8, 512)
            self.assertEqual(cfg_exp.free_max_rows, 24)
            self.assertAlmostEqual(cfg_exp.free_max_w, 0.19)
            self.assertEqual(cfg_exp.free_min_slot, 2)
            self.assertAlmostEqual(cfg_exp.min_p_topk, 0.88)

        # 2. Bit-exact CPU InterpretParams(detect_races=True, uninitialized_memory='nan')
        #    for ROUTE_FAST B in (256, 384) vs legacy rb=32 across free=0 and free=20.
        interp = pltpu.InterpretParams(
            detect_races=True,
            **(
                {"uninitialized_memory": "nan"}
                if "uninitialized_memory" in pltpu.InterpretParams.__annotations__
                else {}
            ),
        )
        rng = np.random.default_rng(2026)

        def _run_route_probe(batch_sz: int, ep: int, use_fast: bool, free_rows: int, lg_np, bi_np, rep_np):
            e_local = 256 // ep
            r_slots = len(rep_np)
            max_exec = e_local + r_slots
            shapes = rt_mod.routing_scratch_shapes(batch_sz, max_exec, topk=8, num_experts=256)

            def _kern2(lg_r, bi_r, rp_r, out_cnt, out_pk, *sc_refs):
                scr = rt_mod.RoutingScratch(*sc_refs[:-1])
                sem = sc_refs[-1]
                cands = tuple(rp_r[i] for i in range(r_slots))
                rt_mod.route_topk_and_counts_in_kernel(
                    lg_r,
                    bi_r,
                    scr,
                    sem,
                    my_owner=jnp.int32(1),
                    e_per_owner=e_local,
                    holder_cand_eids=cands,
                    min_p_topk=0.82,
                    free_max_rows=free_rows,
                    free_max_w=0.20,
                    free_min_slot=1,
                    route_chunk_rows=32,
                    route_rb16_mode=0,
                    route_fast=use_fast,
                )
                out_cnt[...] = scr.counts_smem[0:1, :]
                out_pk[...] = scr.route_pack_smem[...]

            res_cnt, res_pk = pl.pallas_call(
                _kern2,
                out_shape=(
                    jax.ShapeDtypeStruct((1, 256), jnp.int32),
                    jax.ShapeDtypeStruct((batch_sz, 128), jnp.int32),
                ),
                grid_spec=pltpu.PrefetchScalarGridSpec(
                    num_scalar_prefetch=0,
                    grid=(1,),
                    in_specs=(
                        pl.BlockSpec(memory_space=pltpu.VMEM),
                        pl.BlockSpec(memory_space=pltpu.VMEM),
                        pl.BlockSpec(memory_space=pltpu.SMEM),
                    ),
                    out_specs=(
                        pl.BlockSpec(memory_space=pltpu.SMEM),
                        pl.BlockSpec(memory_space=pltpu.SMEM),
                    ),
                    scratch_shapes=tuple(list(shapes) + [pltpu.SemaphoreType.DMA]),
                ),
                interpret=interp,
            )(jnp.asarray(lg_np), jnp.asarray(bi_np), jnp.asarray(rep_np, jnp.int32))
            return np.asarray(res_cnt), np.asarray(res_pk)

        bi_np = rng.normal(0.0, 0.2, size=(1, 256)).astype(np.float32)
        rep_np = np.array([5, 42, -1, 120, 201, -1, 88, 250], dtype=np.int32)
        for batch_sz, ep in ((256, 8), (384, 16)):
            lg_np = rng.normal(0.0, 1.8, size=(batch_sz, 256)).astype(np.float32)
            lg_np[-2:, :] = np.nan
            for free_rows in (0, 20):
                c_ref, p_ref = _run_route_probe(batch_sz, ep, False, free_rows, lg_np, bi_np, rep_np)
                c_fast, p_fast = _run_route_probe(batch_sz, ep, True, free_rows, lg_np, bi_np, rep_np)
                self.assertTrue(
                    np.array_equal(c_ref, c_fast),
                    f"counts mismatch B={batch_sz}, EP={ep}, free={free_rows}",
                )
                self.assertTrue(
                    np.array_equal(p_ref[:, :24], p_fast[:, :24]),
                    f"route_pack mismatch B={batch_sz}, EP={ep}, free={free_rows}",
                )

        # 3. Offline TPU7x Mosaic lowering for EP=8 B=256 (LHS_FP8 + TM128) and
        #    EP=16 B=256/384 (PAIR_RS + return_sharded=True, incl. rows_per_rank=24).
        for topo_str, ep, batches in (
            ("tpu7x:2x2x1", 8, (256,)),
            ("tpu7x:2x2x2", 16, (256, 384)),
        ):
            tpu_devs = get_topology_desc(topo_str, platform="tpu").devices
            if len(tpu_devs) < ep:
                continue
            mesh = Mesh(
                np.array(
                    sorted(tpu_devs, key=lambda d: (d.coords[1], d.coords[0], d.core_on_chip))
                ),
                ("model",),
            )
            e_loc = 256 // ep
            h, r_slots = 4096, 8
            for batch_sz in batches:
                shapes_v = (
                    jax.ShapeDtypeStruct((1,), jnp.int32),
                    jax.ShapeDtypeStruct((batch_sz, 256), jnp.float32),
                    jax.ShapeDtypeStruct((1, 256), jnp.float32),
                    jax.ShapeDtypeStruct((batch_sz, h), jnp.bfloat16),
                    jax.ShapeDtypeStruct((e_loc, h, 8192), jnp.float4_e2m1fn),
                    jax.ShapeDtypeStruct((e_loc, h // 512, 8192), jnp.bfloat16),
                    jax.ShapeDtypeStruct((e_loc, 4096, h), jnp.float4_e2m1fn),
                    jax.ShapeDtypeStruct((e_loc, 4096 // 512, h), jnp.bfloat16),
                    jax.ShapeDtypeStruct((1, h), jnp.bfloat16),
                    jax.ShapeDtypeStruct((r_slots, h, 8192), jnp.float4_e2m1fn),
                    jax.ShapeDtypeStruct((r_slots, h // 512, 8192), jnp.bfloat16),
                    jax.ShapeDtypeStruct((r_slots, 4096, h), jnp.float4_e2m1fn),
                    jax.ShapeDtypeStruct((r_slots, 4096 // 512, h), jnp.bfloat16),
                    jax.ShapeDtypeStruct((4, ep * r_slots), jnp.int32),
                )
                base_cfg = v1_mod.default_sparse_decode_config(batch_sz, ep, 512)
                for tm128 in ((False, True) if batch_sz == 256 else (False,)):
                    cfg_v = dataclasses.replace(
                        base_cfg,
                        tile_m=128 if tm128 else 64,
                        m_buckets=(16, 32, 64, 128) if tm128 else (16, 32, 64),
                        lhs_fp8=True,
                        route_fast=True,
                        fast_plan=2,
                        unroll_strips=True,
                        fast_gs=True,
                        static_holder_mask=True,
                        pair_rs=(ep == 16),
                        bridge_ag=True,
                    )
                    sm_v = jax.jit(
                        jax.shard_map(
                            partial(
                                v1_mod.fused_decode_sparse_from_logits,
                                ep=ep,
                                return_sharded=True,
                                replica_cost_mode="tiles",
                                config=cfg_v,
                            ),
                            mesh=mesh,
                            in_specs=(P(),) * len(shapes_v),
                            out_specs=P(),
                            check_vma=False,
                        )
                    )
                    comp_v = sm_v.trace(*shapes_v).lower(lowering_platforms=("tpu",)).compile()
                    self.assertIn("tpu_custom_call", comp_v.as_text())
                    self.assertGreater(
                        comp_v.memory_analysis().generated_code_size_in_bytes, 0
                    )

    def test_w9_split_vmem_phase_helpers_parity_and_tpu7x_lowering(self):
        """Verify split VMEM MoE phase helpers match monolithic core bit-exactly and compile on TPU7x."""
        import dataclasses
        from functools import partial

        try:
            import jax
            import jax.numpy as jnp
            from jax.experimental import pallas as pl
            from jax.experimental.pallas import tpu as pltpu
            from jax.experimental.topologies import get_topology_desc
            from jax.sharding import Mesh, PartitionSpec as P
        except Exception as exc:
            self.skipTest(f"JAX/Pallas TPU unavailable: {exc}")

        v1_mod = _load_pkg_module(
            "fused_decode_sparse", _MUSE_DIR / "fused_decode_sparse.py"
        )
        routing_mod = _load_pkg_module(
            "fused_decode_routing", _MUSE_DIR / "fused_decode_routing.py"
        )

        # 1. CPU Pallas InterpretParams(detect_races=True, uninitialized_memory='nan')
        #    bit-exact parity between monolithic _run_muse_sparse_from_vmem_core and
        #    split (plan_and_prime(..., zero_and_pack=False) -> zero_and_pack ->
        #    run_muse_sparse_experts_and_rs_from_vmem(..., x_is_3d=True)).
        ep = 8
        batch_size = 64
        scale_block = 512
        e_local = 256 // ep
        h, n1, i_dim = 4096, 8192, 4096
        base_cfg = v1_mod.default_sparse_decode_config(batch_size, ep, scale_block)
        cfg = dataclasses.replace(
            base_cfg,
            m_buckets=(16, 32),
            route_chunk_rows=32,
            static_holder_mask=True,
            route_fast=True,
            fast_plan=2,
            unroll_strips=True,
            fast_gs=True,
            pair_rs=True,
            bridge_ag=True,
            fast_b64_epi_pro=False,
            free_max_rows=10,
            free_max_w=0.20,
            free_min_slot=1,
            min_p_topk=0.84,
            spec_prime_parts=0,
        )

        interp_kw = {
            "detect_races": True,
            **(
                {"uninitialized_memory": "nan"}
                if "uninitialized_memory" in pltpu.InterpretParams.__annotations__
                else {}
            ),
        }

        def _unpack_vmem_scratch(refs_iter, num_replicas: int, return_sharded: bool):
            r_sc = routing_mod.RoutingScratch(
                *(next(refs_iter) for _ in range(len(routing_mod.RoutingScratch._fields)))
            )
            if num_replicas > 0:
                ol_sm = next(refs_iter)
                mf_sm = next(refs_iter)
                ls_sm = next(refs_iter)
                hs_sm = next(refs_iter)
            else:
                ol_sm = mf_sm = ls_sm = hs_sm = None
            w1_b = next(refs_iter)
            s1_b = next(refs_iter)
            w2_b = next(refs_iter)
            s2_b = next(refs_iter)
            w1_sm = next(refs_iter)
            w2_sm = next(refs_iter)
            misc_sm = next(refs_iter)
            x_3d_b = next(refs_iter)
            io_3d_b = next(refs_iter)
            lhs_2d_b = next(refs_iter)
            act_2d_b = next(refs_iter)
            y_st_b = next(refs_iter)
            gain_b = next(refs_iter)
            acc_3d = next(refs_iter)
            rs_send_bf = next(refs_iter) if cfg.rs_wire_bf16 else None
            rs_p_r = next(refs_iter)
            rs_c_s = next(refs_iter)
            rs_c_r = next(refs_iter)
            rs_p_ss = next(refs_iter)
            rs_p_rs = next(refs_iter)
            rs_c_ss = next(refs_iter)
            rs_c_rs = next(refs_iter)
            own_3d = next(refs_iter)
            fan_ss = next(refs_iter) if not return_sharded else None
            fan_rs = next(refs_iter) if not return_sharded else None
            fan_s2_ss = next(refs_iter) if (not return_sharded and cfg.bridge_ag) else None
            fan_s2_rs = next(refs_iter) if (not return_sharded and cfg.bridge_ag) else None
            return dict(
                r_scratch=r_sc,
                owner_load_smem=ol_sm,
                moved_flag_smem=mf_sm,
                local_suppress_smem=ls_sm,
                holder_slot_expert_smem=hs_sm,
                w1_buf=w1_b,
                s1_buf=s1_b,
                w2_buf=w2_b,
                s2_buf=s2_b,
                w1_sem=w1_sm,
                w2_sem=w2_sm,
                misc_sem=misc_sm,
                x_3d_buf=x_3d_b,
                io_3d_buf=io_3d_b,
                lhs_2d_buf=lhs_2d_b,
                act_2d_buf=act_2d_b,
                y_stage_buf=y_st_b,
                gain_f32_buf=gain_b,
                accum_3d=acc_3d,
                rs_send_bf16=rs_send_bf,
                rs_pair_recv=rs_p_r,
                rs_chip_send=rs_c_s,
                rs_chip_recv=rs_c_r,
                rs_pair_ssem=rs_p_ss,
                rs_pair_rsem=rs_p_rs,
                rs_chip_ssem=rs_c_ss,
                rs_chip_rsem=rs_c_rs,
                owner_3d=own_3d,
                fan_send_sem=fan_ss,
                fan_recv_sem=fan_rs,
                fan_s2_send_sem=fan_s2_ss,
                fan_s2_recv_sem=fan_s2_rs,
            )

        rng = np.random.default_rng(20261125)
        rank_np = np.array([2], dtype=np.int32)
        logits_np = rng.normal(0.0, 2.0, size=(batch_size, 256)).astype(np.float32)
        bias_np = (rng.normal(0.0, 0.2, size=(1, 256))).astype(np.float32)
        x_np = (rng.normal(0.0, 0.25, size=(batch_size, h))).astype(np.float32)
        norm_np = (rng.normal(0.0, 0.05, size=(1, h))).astype(np.float32)

        def _make_u32_bank(rows: int):
            w1_u = jnp.asarray(
                rng.integers(0, 0x88888888, size=(rows, h // 8, n1), dtype=np.uint32)
            )
            w2_u = jnp.asarray(
                rng.integers(0, 0x88888888, size=(rows, i_dim // 8, h), dtype=np.uint32)
            )
            s1_bf = jnp.asarray(
                rng.uniform(0.05, 0.25, size=(rows, h // scale_block, n1)), jnp.bfloat16
            )
            s2_bf = jnp.asarray(
                rng.uniform(0.05, 0.25, size=(rows, i_dim // scale_block, h)), jnp.bfloat16
            )
            return w1_u, s1_bf, w2_u, s2_bf

        w1_j, s1_j, w2_j, s2_j = _make_u32_bank(e_local)
        rep_w1_j, rep_s1_j, rep_w2_j, rep_s2_j = _make_u32_bank(8)
        meta_np = np.full((4, ep * 8), -1, dtype=np.int32)
        meta_np[0, 2 * 8 + 0] = 64
        meta_np[1, 2 * 8 + 0] = 1
        meta_np[2, 2 * 8 + 0] = 64

        v_spec = pl.BlockSpec(memory_space=pltpu.VMEM)
        h_spec = pl.BlockSpec(memory_space=pltpu.HBM)

        for num_replicas in (0, 8):
            scratch_shapes = v1_mod.build_muse_sparse_scratch_shapes(
                ep=ep,
                batch_size=batch_size,
                scale_block=scale_block,
                s1_dtype=jnp.bfloat16,
                s2_dtype=jnp.bfloat16,
                num_replicas=num_replicas,
                return_sharded=True,
                config=cfg,
            )

            def _run_parity(split_mode: bool):
                def _kern(*all_refs):
                    it = iter(all_refs)
                    rank_r = next(it)
                    meta_r = next(it) if num_replicas > 0 else None
                    logits_r = next(it)
                    bias_r = next(it)
                    x_r = next(it)
                    w1_r = next(it)
                    s1_r = next(it)
                    w2_r = next(it)
                    s2_r = next(it)
                    norm_r = next(it)
                    if num_replicas > 0:
                        rw1_r = next(it)
                        rs1_r = next(it)
                        rw2_r = next(it)
                        rs2_r = next(it)
                    else:
                        rw1_r = rs1_r = rw2_r = rs2_r = None
                    out_r = next(it)
                    sc = _unpack_vmem_scratch(it, num_replicas, return_sharded=True)
                    me = rank_r[0]
                    w1_u = w1_r
                    w2_u = w2_r
                    rw1_u = rw1_r
                    rw2_u = rw2_r

                    def _dump_accum():
                        out_r[...] = sc["accum_3d"][pl.ds(0, batch_size), :, :].reshape(
                            batch_size, h
                        ).astype(jnp.bfloat16)

                    if not split_mode:
                        v1_mod._run_muse_sparse_from_vmem_core(
                            me,
                            logits_r,
                            bias_r,
                            x_r,
                            w1_u,
                            s1_r,
                            w2_u,
                            s2_r,
                            norm_r,
                            d_scratch=None,
                            r_scratch=sc["r_scratch"],
                            w1_buf=sc["w1_buf"],
                            s1_buf=sc["s1_buf"],
                            w2_buf=sc["w2_buf"],
                            s2_buf=sc["s2_buf"],
                            w1_sem=sc["w1_sem"],
                            w2_sem=sc["w2_sem"],
                            misc_sem=sc["misc_sem"],
                            w1_acc_buf=None,
                            x_3d_buf=sc["x_3d_buf"],
                            io_3d_buf=sc["io_3d_buf"],
                            lhs_2d_buf=sc["lhs_2d_buf"],
                            act_2d_buf=sc["act_2d_buf"],
                            y_stage_buf=sc["y_stage_buf"],
                            lhs_fp8_buf=None,
                            lhs_sc_buf=None,
                            masked_x_buf=None,
                            bd_sc_buf=None,
                            gain_f32_buf=sc["gain_f32_buf"],
                            accum_3d=sc["accum_3d"],
                            accum_2d=None,
                            fan_3d_buf=None,
                            rs_send_bf16=sc["rs_send_bf16"],
                            rs_pair_recv=sc["rs_pair_recv"],
                            rs_chip_send=sc["rs_chip_send"],
                            rs_chip_recv=sc["rs_chip_recv"],
                            rs_pair_ssem=sc["rs_pair_ssem"],
                            rs_pair_rsem=sc["rs_pair_rsem"],
                            rs_chip_ssem=sc["rs_chip_ssem"],
                            rs_chip_rsem=sc["rs_chip_rsem"],
                            rs_recv_3d=None,
                            rs_send_sem=None,
                            rs_recv_sem=None,
                            owner_3d=sc["owner_3d"],
                            fan_send_sem=None,
                            fan_recv_sem=None,
                            fan_s2_send_sem=None,
                            fan_s2_recv_sem=None,
                            barrier_sem=None,
                            output_ref=out_r,
                            ep=ep,
                            batch_size=batch_size,
                            scale_block=scale_block,
                            num_replicas=num_replicas,
                            return_sharded=True,
                            replica_cost_mode="active",
                            ep_axis_name="model",
                            cfg=cfg,
                            rep_w1_u32=rw1_u,
                            rep_s1_ref=rs1_r,
                            rep_w2_u32=rw2_u,
                            rep_s2_ref=rs2_r,
                            replica_meta_ref=meta_r,
                            owner_load_smem=sc["owner_load_smem"],
                            moved_flag_smem=sc["moved_flag_smem"],
                            local_suppress_smem=sc["local_suppress_smem"],
                            holder_slot_expert_smem=sc["holder_slot_expert_smem"],
                            holder_cand_smem=None,
                            x_is_3d=False,
                            wait_entry_barrier=False,
                            write_output_ref=False,
                            stop_stage="experts",
                            early_exit_fn=_dump_accum,
                        )
                    else:
                        plan_res = v1_mod.plan_and_prime_muse_sparse_from_vmem(
                            me,
                            logits_r,
                            bias_r,
                            norm_r,
                            w1_u,
                            s1_r,
                            w2_u,
                            s2_r,
                            d_scratch=None,
                            r_scratch=sc["r_scratch"],
                            w1_buf=sc["w1_buf"],
                            s1_buf=sc["s1_buf"],
                            w2_buf=sc["w2_buf"],
                            s2_buf=sc["s2_buf"],
                            w1_sem=sc["w1_sem"],
                            w2_sem=sc["w2_sem"],
                            misc_sem=sc["misc_sem"],
                            gain_f32_buf=sc["gain_f32_buf"],
                            accum_3d=sc["accum_3d"],
                            io_3d_buf=sc["io_3d_buf"],
                            accum_2d=None,
                            rep_w1_u32=rw1_u,
                            rep_s1_ref=rs1_r,
                            rep_w2_u32=rw2_u,
                            rep_s2_ref=rs2_r,
                            replica_meta_ref=meta_r,
                            owner_load_smem=sc["owner_load_smem"],
                            moved_flag_smem=sc["moved_flag_smem"],
                            local_suppress_smem=sc["local_suppress_smem"],
                            holder_slot_expert_smem=sc["holder_slot_expert_smem"],
                            holder_cand_smem=None,
                            ep=ep,
                            batch_size=batch_size,
                            scale_block=scale_block,
                            num_replicas=num_replicas,
                            replica_cost_mode="active",
                            cfg=cfg,
                            zero_and_pack=False,
                        )
                        v1_mod.zero_and_pack_muse_sparse_from_vmem(
                            plan_res,
                            r_scratch=sc["r_scratch"],
                            accum_3d=sc["accum_3d"],
                            io_3d_buf=sc["io_3d_buf"],
                            accum_2d=None,
                            batch_size=batch_size,
                            cfg=cfg,
                        )
                        for r0 in range(0, batch_size, 16):
                            for h_i in range(2):
                                sc["x_3d_buf"][
                                    pl.ds(r0, 16), pl.ds(h_i * 16, 16), :
                                ] = x_r[
                                    pl.ds(r0, 16), pl.ds(h_i * 2048, 2048)
                                ].reshape(16, 16, 128)
                        v1_mod.run_muse_sparse_experts_and_rs_from_vmem(
                            me,
                            x_r,
                            plan_res,
                            d_scratch=None,
                            r_scratch=sc["r_scratch"],
                            w1_buf=sc["w1_buf"],
                            s1_buf=sc["s1_buf"],
                            w2_buf=sc["w2_buf"],
                            s2_buf=sc["s2_buf"],
                            w1_acc_buf=None,
                            x_3d_buf=sc["x_3d_buf"],
                            io_3d_buf=sc["io_3d_buf"],
                            lhs_2d_buf=sc["lhs_2d_buf"],
                            act_2d_buf=sc["act_2d_buf"],
                            y_stage_buf=sc["y_stage_buf"],
                            lhs_fp8_buf=None,
                            lhs_sc_buf=None,
                            masked_x_buf=None,
                            bd_sc_buf=None,
                            gain_f32_buf=sc["gain_f32_buf"],
                            accum_3d=sc["accum_3d"],
                            accum_2d=None,
                            fan_3d_buf=None,
                            rs_send_bf16=sc["rs_send_bf16"],
                            rs_pair_recv=sc["rs_pair_recv"],
                            rs_chip_send=sc["rs_chip_send"],
                            rs_chip_recv=sc["rs_chip_recv"],
                            rs_pair_ssem=sc["rs_pair_ssem"],
                            rs_pair_rsem=sc["rs_pair_rsem"],
                            rs_chip_ssem=sc["rs_chip_ssem"],
                            rs_chip_rsem=sc["rs_chip_rsem"],
                            rs_recv_3d=None,
                            rs_send_sem=None,
                            rs_recv_sem=None,
                            owner_3d=sc["owner_3d"],
                            fan_send_sem=None,
                            fan_recv_sem=None,
                            fan_s2_send_sem=None,
                            fan_s2_recv_sem=None,
                            barrier_sem=None,
                            output_ref=out_r,
                            ep=ep,
                            batch_size=batch_size,
                            scale_block=scale_block,
                            return_sharded=True,
                            ep_axis_name="model",
                            cfg=cfg,
                            x_is_3d=True,
                            wait_entry_barrier=False,
                            write_output_ref=False,
                            stop_stage="experts",
                            early_exit_fn=_dump_accum,
                        )

                in_specs = (
                    (v_spec,) * 3
                    + (h_spec,) * 4
                    + (v_spec,)
                    + ((h_spec,) * 4 if num_replicas > 0 else ())
                )
                num_sc_pre = 2 if num_replicas > 0 else 1
                fn = pl.pallas_call(
                    _kern,
                    out_shape=jax.ShapeDtypeStruct((batch_size, h), jnp.bfloat16),
                    grid_spec=pltpu.PrefetchScalarGridSpec(
                        num_scalar_prefetch=num_sc_pre,
                        grid=(1,),
                        in_specs=in_specs,
                        out_specs=v_spec,
                        scratch_shapes=scratch_shapes,
                    ),
                    interpret=pltpu.InterpretParams(**interp_kw),
                )
                args = (
                    (
                        jnp.asarray(rank_np),
                        jnp.asarray(meta_np),
                    )
                    if num_replicas > 0
                    else (jnp.asarray(rank_np),)
                ) + (
                    jnp.asarray(logits_np),
                    jnp.asarray(bias_np),
                    jnp.asarray(x_np, jnp.bfloat16),
                    w1_j,
                    s1_j,
                    w2_j,
                    s2_j,
                    jnp.asarray(norm_np, jnp.bfloat16),
                ) + (
                    (rep_w1_j, rep_s1_j, rep_w2_j, rep_s2_j) if num_replicas > 0 else ()
                )
                return np.asarray(fn(*args), dtype=np.float32)

            ref_acc = _run_parity(False)
            split_acc = _run_parity(True)
            self.assertGreater(float(np.max(np.abs(ref_acc))), 0.0)
            self.assertTrue(np.all(np.isfinite(split_acc)))
            self.assertEqual(
                float(np.max(np.abs(ref_acc - split_acc))),
                0.0,
                f"Split VMEM helpers parity mismatch for num_replicas={num_replicas}",
            )

        # 2. Offline TPU7x Mosaic compilation for full split flow + on_experts_done_fn
        #    aliasing into w2_buf + write_output_ref=False:
        #      - EP=8 return_sharded=True + spec_prime_parts=1 -> owner_3d
        #      - EP=16 return_sharded=False (B=64 rpr=4) + output_ref=None -> fan_3d_buf
        for topo_str, cur_ep, cur_ret_sh, cur_sp_parts in (
            ("tpu7x:2x2x1", 8, True, 1),
            ("tpu7x:2x2x2", 16, False, 0),
        ):
            tpu_devs = get_topology_desc(topo_str, platform="tpu").devices
            if len(tpu_devs) < cur_ep:
                continue
            mesh = Mesh(
                np.array(
                    sorted(tpu_devs, key=lambda d: (d.coords[1], d.coords[0], d.core_on_chip))
                ),
                ("model",),
            )
            cur_eloc = 256 // cur_ep
            cur_out_rows = (batch_size // cur_ep) if cur_ret_sh else batch_size
            cur_cfg = dataclasses.replace(
                v1_mod.default_sparse_decode_config(batch_size, cur_ep, scale_block),
                m_buckets=(16, 32),
                route_chunk_rows=32,
                static_holder_mask=True,
                route_fast=True,
                fast_plan=2,
                unroll_strips=True,
                fast_gs=True,
                pair_rs=True,
                bridge_ag=True,
                fast_b64_epi_pro=False,
                spec_prime_parts=cur_sp_parts,
                spec_home_slots=(0,) * cur_ep if cur_sp_parts > 0 else (),
            )
            cur_scratch = v1_mod.build_muse_sparse_scratch_shapes(
                ep=cur_ep,
                batch_size=batch_size,
                scale_block=scale_block,
                s1_dtype=jnp.bfloat16,
                s2_dtype=jnp.bfloat16,
                num_replicas=8,
                return_sharded=cur_ret_sh,
                config=cur_cfg,
            )
            cur_shapes = (
                jax.ShapeDtypeStruct((1,), jnp.int32),
                jax.ShapeDtypeStruct((4, cur_ep * 8), jnp.int32),
                jax.ShapeDtypeStruct((batch_size, 256), jnp.float32),
                jax.ShapeDtypeStruct((1, 256), jnp.float32),
                jax.ShapeDtypeStruct((batch_size, h), jnp.bfloat16),
                jax.ShapeDtypeStruct((cur_eloc, h, n1), jnp.float4_e2m1fn),
                jax.ShapeDtypeStruct((cur_eloc, h // scale_block, n1), jnp.bfloat16),
                jax.ShapeDtypeStruct((cur_eloc, i_dim, h), jnp.float4_e2m1fn),
                jax.ShapeDtypeStruct((cur_eloc, i_dim // scale_block, h), jnp.bfloat16),
                jax.ShapeDtypeStruct((1, h), jnp.bfloat16),
                jax.ShapeDtypeStruct((8, h, n1), jnp.float4_e2m1fn),
                jax.ShapeDtypeStruct((8, h // scale_block, n1), jnp.bfloat16),
                jax.ShapeDtypeStruct((8, i_dim, h), jnp.float4_e2m1fn),
                jax.ShapeDtypeStruct((8, i_dim // scale_block, h), jnp.bfloat16),
            )

            def _split_super_ep(*args):
                def _k(*all_refs):
                    it = iter(all_refs)
                    rank_r, meta_r = next(it), next(it)
                    logits_r, bias_r, x_r = next(it), next(it), next(it)
                    w1_r, s1_r, w2_r, s2_r, norm_r = (
                        next(it),
                        next(it),
                        next(it),
                        next(it),
                        next(it),
                    )
                    rw1_r, rs1_r, rw2_r, rs2_r = next(it), next(it), next(it), next(it)
                    out_r = next(it)
                    sc = _unpack_vmem_scratch(it, 8, return_sharded=cur_ret_sh)
                    bsem = pltpu.get_barrier_semaphore()
                    for d in range(cur_ep):
                        pl.semaphore_signal(
                            bsem,
                            device_id={"model": jnp.int32(d)},
                            device_id_type=pl.DeviceIdType.MESH,
                        )
                    me = rank_r[0]
                    w1_u = w1_r.bitcast(jnp.uint32)
                    w2_u = w2_r.bitcast(jnp.uint32)
                    rw1_u = rw1_r.bitcast(jnp.uint32)
                    rw2_u = rw2_r.bitcast(jnp.uint32)
                    plan_res = v1_mod.plan_and_prime_muse_sparse_from_vmem(
                        me,
                        logits_r,
                        bias_r,
                        norm_r,
                        w1_u,
                        s1_r,
                        w2_u,
                        s2_r,
                        d_scratch=None,
                        r_scratch=sc["r_scratch"],
                        w1_buf=sc["w1_buf"],
                        s1_buf=sc["s1_buf"],
                        w2_buf=sc["w2_buf"],
                        s2_buf=sc["s2_buf"],
                        w1_sem=sc["w1_sem"],
                        w2_sem=sc["w2_sem"],
                        misc_sem=sc["misc_sem"],
                        gain_f32_buf=sc["gain_f32_buf"],
                        accum_3d=sc["accum_3d"],
                        io_3d_buf=sc["io_3d_buf"],
                        accum_2d=None,
                        rep_w1_u32=rw1_u,
                        rep_s1_ref=rs1_r,
                        rep_w2_u32=rw2_u,
                        rep_s2_ref=rs2_r,
                        replica_meta_ref=meta_r,
                        owner_load_smem=sc["owner_load_smem"],
                        moved_flag_smem=sc["moved_flag_smem"],
                        local_suppress_smem=sc["local_suppress_smem"],
                        holder_slot_expert_smem=sc["holder_slot_expert_smem"],
                        holder_cand_smem=None,
                        ep=cur_ep,
                        batch_size=batch_size,
                        scale_block=scale_block,
                        num_replicas=8,
                        replica_cost_mode="tiles",
                        cfg=cur_cfg,
                        zero_and_pack=False,
                    )
                    v1_mod.zero_and_pack_muse_sparse_from_vmem(
                        plan_res,
                        r_scratch=sc["r_scratch"],
                        accum_3d=sc["accum_3d"],
                        io_3d_buf=sc["io_3d_buf"],
                        accum_2d=None,
                        batch_size=batch_size,
                        cfg=cur_cfg,
                    )
                    for r0 in range(0, batch_size, 32):
                        for h_i in range(2):
                            sc["x_3d_buf"][
                                pl.ds(r0, 32), pl.ds(h_i * 16, 16), :
                            ] = x_r[pl.ds(r0, 32), pl.ds(h_i * 2048, 2048)].reshape(
                                32, 16, 128
                            )

                    def _on_experts_done():
                        sc["w2_buf"][:, :128] = jnp.zeros(
                            (i_dim // 8, 128), dtype=jnp.uint32
                        )

                    v1_mod.run_muse_sparse_experts_and_rs_from_vmem(
                        me,
                        x_r,
                        plan_res,
                        d_scratch=None,
                        r_scratch=sc["r_scratch"],
                        w1_buf=sc["w1_buf"],
                        s1_buf=sc["s1_buf"],
                        w2_buf=sc["w2_buf"],
                        s2_buf=sc["s2_buf"],
                        w1_acc_buf=None,
                        x_3d_buf=sc["x_3d_buf"],
                        io_3d_buf=sc["io_3d_buf"],
                        lhs_2d_buf=sc["lhs_2d_buf"],
                        act_2d_buf=sc["act_2d_buf"],
                        y_stage_buf=sc["y_stage_buf"],
                        lhs_fp8_buf=None,
                        lhs_sc_buf=None,
                        masked_x_buf=None,
                        bd_sc_buf=None,
                        gain_f32_buf=sc["gain_f32_buf"],
                        accum_3d=sc["accum_3d"],
                        accum_2d=None,
                        fan_3d_buf=sc["x_3d_buf"] if not cur_ret_sh else None,
                        rs_send_bf16=sc["rs_send_bf16"],
                        rs_pair_recv=sc["rs_pair_recv"],
                        rs_chip_send=sc["rs_chip_send"],
                        rs_chip_recv=sc["rs_chip_recv"],
                        rs_pair_ssem=sc["rs_pair_ssem"],
                        rs_pair_rsem=sc["rs_pair_rsem"],
                        rs_chip_ssem=sc["rs_chip_ssem"],
                        rs_chip_rsem=sc["rs_chip_rsem"],
                        rs_recv_3d=None,
                        rs_send_sem=None,
                        rs_recv_sem=None,
                        owner_3d=sc["owner_3d"],
                        fan_send_sem=sc["fan_send_sem"],
                        fan_recv_sem=sc["fan_recv_sem"],
                        fan_s2_send_sem=sc["fan_s2_send_sem"],
                        fan_s2_recv_sem=sc["fan_s2_recv_sem"],
                        barrier_sem=bsem,
                        output_ref=out_r if cur_ret_sh else None,
                        ep=cur_ep,
                        batch_size=batch_size,
                        scale_block=scale_block,
                        return_sharded=cur_ret_sh,
                        ep_axis_name="model",
                        cfg=cur_cfg,
                        x_is_3d=True,
                        wait_entry_barrier=True,
                        write_output_ref=False,
                        on_experts_done_fn=_on_experts_done,
                    )
                    if cur_ret_sh:
                        out_r[...] = sc["owner_3d"][...].reshape(cur_out_rows, h)
                    else:
                        for r0 in range(0, batch_size, 16):
                            out_r[pl.ds(r0, 16), :] = sc["x_3d_buf"][
                                pl.ds(r0, 16), :, :
                            ].reshape(16, h)

                return pl.pallas_call(
                    _k,
                    out_shape=jax.ShapeDtypeStruct((cur_out_rows, h), jnp.bfloat16),
                    grid_spec=pltpu.PrefetchScalarGridSpec(
                        num_scalar_prefetch=2,
                        grid=(1,),
                        in_specs=(
                            (v_spec,) * 3 + (h_spec,) * 4 + (v_spec,) + (h_spec,) * 4
                        ),
                        out_specs=v_spec,
                        scratch_shapes=cur_scratch,
                    ),
                    compiler_params=pltpu.CompilerParams(
                        vmem_limit_bytes=64 * 1024**2,
                        collective_id=0,
                    ),
                )(*args)

            sm_split = jax.jit(
                jax.shard_map(
                    _split_super_ep,
                    mesh=mesh,
                    in_specs=(P(),) * len(cur_shapes),
                    out_specs=P(),
                    check_vma=False,
                )
            )
            comp_split = sm_split.trace(*cur_shapes).lower(lowering_platforms=("tpu",)).compile()
            self.assertIn("tpu_custom_call", comp_split.as_text())
            self.assertGreater(
                comp_split.memory_analysis().generated_code_size_in_bytes, 0
            )

    def test_w9_b16_sparse_and_dynamic_free_params_parity_and_lowering(self):
        """Verify B=16 sparse + fast_plan=dense, nonuniform FREE, dynamic free_params, & TPU7x compile."""
        import dataclasses
        import os

        try:
            import jax
            import jax.numpy as jnp
            from jax.experimental import pallas as pl
            from jax.experimental.pallas import tpu as pltpu
            from jax.experimental.topologies import get_topology_desc
            from jax.sharding import Mesh, PartitionSpec as P
        except Exception as exc:
            self.skipTest(f"JAX/Pallas TPU unavailable: {exc}")

        from sglang_tpu.envs import envs as tpu_envs

        breakers_mod = _load_pkg_module(
            "decode_hbm_breakers", _MUSE_DIR / "decode_hbm_breakers.py"
        )
        routing_mod = _load_pkg_module(
            "fused_decode_routing", _MUSE_DIR / "fused_decode_routing.py"
        )
        v1_mod = _load_pkg_module(
            "fused_decode_sparse", _MUSE_DIR / "fused_decode_sparse.py"
        )

        # 1. Default-OFF env & nonuniform layer FREE resolution checks.
        self.assertIs(tpu_envs.SGLANG_TPU_MUSE_KERNEL_DECODE_B16_SPARSE.get(), False)
        self.assertEqual(tpu_envs.SGLANG_TPU_MUSE_DECODE_NONUNIFORM_FREE.get(), "")

        prev_b16 = os.environ.get("SGLANG_TPU_MUSE_KERNEL_DECODE_B16_SPARSE")
        try:
            os.environ["SGLANG_TPU_MUSE_KERNEL_DECODE_B16_SPARSE"] = "1"
            cfg_b16 = v1_mod.default_sparse_decode_config(16, 8, 512)
            self.assertEqual(cfg_b16.row_mode, "sparse")
            self.assertEqual(cfg_b16.tile_m, 16)
            self.assertEqual(cfg_b16.m_buckets, (8, 16))
            self.assertEqual(cfg_b16.route_chunk_rows, 16)
            cfg_b8 = v1_mod.default_sparse_decode_config(8, 8, 512)
            self.assertEqual(cfg_b8.row_mode, "sparse")
            self.assertEqual(cfg_b8.tile_m, 16)
            self.assertEqual(cfg_b8.m_buckets, (8,))
            self.assertEqual(cfg_b8.route_chunk_rows, 8)
        finally:
            if prev_b16 is None:
                os.environ.pop("SGLANG_TPU_MUSE_KERNEL_DECODE_B16_SPARSE", None)
            else:
                os.environ["SGLANG_TPU_MUSE_KERNEL_DECODE_B16_SPARSE"] = prev_b16

        self.assertEqual(
            breakers_mod.resolve_nonuniform_free_for_layer(
                1, 62, 64,
                min_p_topk=0.86, free_rows=8, free_max_w=0.20, free_min_slot=1,
                parse_free_spec_fn=v1_mod.parse_free_expert_rows_spec,
                spec="mid_f10p82",
            ),
            (0.88, 6, 0.18, 2),
        )
        self.assertEqual(
            breakers_mod.resolve_nonuniform_free_for_layer(
                30, 62, 64,
                min_p_topk=0.86, free_rows=8, free_max_w=0.20, free_min_slot=1,
                parse_free_spec_fn=v1_mod.parse_free_expert_rows_spec,
                spec="mid_f10p82",
            ),
            (0.82, 10, 0.22, 1),
        )
        # EP-aware clamping at B=16: <=2 on EP=8, <=1 on EP=16.
        _, r_ep8, _, _ = breakers_mod.resolve_nonuniform_free_for_layer(
            30, 62, 16,
            min_p_topk=0.86, free_rows=8, free_max_w=0.20, free_min_slot=1,
            parse_free_spec_fn=v1_mod.parse_free_expert_rows_spec,
            ep=8, scale_mode="b64_ms16", spec="mid_f10p82",
        )
        _, r_ep16, _, _ = breakers_mod.resolve_nonuniform_free_for_layer(
            30, 62, 16,
            min_p_topk=0.86, free_rows=8, free_max_w=0.20, free_min_slot=1,
            parse_free_spec_fn=v1_mod.parse_free_expert_rows_spec,
            ep=16, scale_mode="b64_ms16", spec="mid_f10p82",
        )
        self.assertLessEqual(r_ep8, 2)
        self.assertLessEqual(r_ep16, 1)

        rng = np.random.default_rng(20261125)
        h, n1, i_dim, scale_block = 4096, 8192, 4096, 512
        ep = 8
        e_local = 256 // ep
        w1_u = jnp.asarray(
            rng.integers(0, 2**31, size=(e_local, h // 8, n1), dtype=np.int32).view(
                np.uint32
            )
        )
        w2_u = jnp.asarray(
            rng.integers(0, 2**31, size=(e_local, i_dim // 8, h), dtype=np.int32).view(
                np.uint32
            )
        )
        s1_j = jnp.asarray(rng.uniform(0.02, 0.08, size=(e_local, h // scale_block, n1)), jnp.bfloat16)
        s2_j = jnp.asarray(rng.uniform(0.02, 0.08, size=(e_local, i_dim // scale_block, h)), jnp.bfloat16)
        bias_j = jnp.zeros((1, 256), jnp.float32)
        norm_j = jnp.asarray(rng.uniform(0.8, 1.2, size=(1, h)), jnp.bfloat16)

        interp_kw = {
            "detect_races": True,
            **(
                {"uninitialized_memory": "nan"}
                if "uninitialized_memory" in pltpu.InterpretParams.__annotations__
                else {}
            ),
        }
        v_sp, h_sp = pl.BlockSpec(memory_space=pltpu.VMEM), pl.BlockSpec(
            memory_space=pltpu.HBM
        )

        # 2. CPU Interpret parity: B=16 dense (fast_plan=2) vs sparse m_buckets=(8, 16)
        #    (0.0 max_abs) AND dynamic free_params vs static SparseDecodeConfig (0.0 max_abs).
        def _run_core_1d(cur_b: int, cfg_in, lg_arr, x_arr, free_arr=None):
            sc = v1_mod.build_muse_sparse_scratch_shapes(
                ep=ep,
                batch_size=cur_b,
                scale_block=scale_block,
                s1_dtype=jnp.bfloat16,
                s2_dtype=jnp.bfloat16,
                num_replicas=0,
                return_sharded=True,
                config=cfg_in,
            )
            is_d = cfg_in.row_mode == "dense"
            has_fp = free_arr is not None

            def _k(*all_refs):
                it = iter(all_refs)
                rk_r = next(it)
                fp_r = next(it) if has_fp else None
                lg_r, bi_r, x_r = next(it), next(it), next(it)
                w1_r, s1_r, w2_r, s2_r, nm_r = (next(it) for _ in range(5))
                out_r = next(it)
                if is_d:
                    d_sc = routing_mod.DenseRoutingScratch(
                        *(next(it) for _ in range(len(routing_mod.DenseRoutingScratch._fields)))
                    )
                    r_sc = None
                else:
                    d_sc = None
                    r_sc = routing_mod.RoutingScratch(
                        *(next(it) for _ in range(len(routing_mod.RoutingScratch._fields)))
                    )
                w1_b, s1_b, w2_b, s2_b, w1_sm, w2_sm, m_sm = (next(it) for _ in range(7))
                if is_d:
                    a2, ys, gb, ac2, ac3 = (next(it) for _ in range(5))
                    x3 = io3 = l2 = None
                else:
                    x3, io3, l2, a2, ys, gb, ac3 = (next(it) for _ in range(7))
                    ac2 = None
                rs_r3, rs_ss, rs_rs, ow3 = (next(it) for _ in range(4))

                def _dump():
                    out_r[...] = ac3[pl.ds(0, cur_b), :, :].reshape(cur_b, h).astype(
                        jnp.bfloat16
                    )

                v1_mod._run_muse_sparse_from_vmem_core(
                    rk_r[0], lg_r, bi_r, x_r, w1_r, s1_r, w2_r, s2_r, nm_r,
                    d_scratch=d_sc, r_scratch=r_sc,
                    w1_buf=w1_b, s1_buf=s1_b, w2_buf=w2_b, s2_buf=s2_b,
                    w1_sem=w1_sm, w2_sem=w2_sm, misc_sem=m_sm, w1_acc_buf=None,
                    x_3d_buf=x3, io_3d_buf=io3, lhs_2d_buf=l2,
                    act_2d_buf=a2, y_stage_buf=ys,
                    lhs_fp8_buf=None, lhs_sc_buf=None,
                    masked_x_buf=None, bd_sc_buf=None,
                    gain_f32_buf=gb, accum_3d=ac3, accum_2d=ac2, fan_3d_buf=None,
                    rs_send_bf16=None, rs_pair_recv=None,
                    rs_chip_send=None, rs_chip_recv=None,
                    rs_pair_ssem=None, rs_pair_rsem=None,
                    rs_chip_ssem=None, rs_chip_rsem=None,
                    rs_recv_3d=rs_r3, rs_send_sem=rs_ss, rs_recv_sem=rs_rs,
                    owner_3d=ow3, fan_send_sem=None, fan_recv_sem=None,
                    fan_s2_send_sem=None, fan_s2_recv_sem=None,
                    barrier_sem=None, output_ref=out_r,
                    free_params_ref=fp_r,
                    ep=ep, batch_size=cur_b, scale_block=scale_block, num_replicas=0,
                    replica_cost_mode="active", return_sharded=True,
                    ep_axis_name="model", cfg=cfg_in,
                    wait_entry_barrier=False,
                    stop_stage="experts", early_exit_fn=_dump,
                )

            sp_args = (jnp.array([0], jnp.int32),) + (
                (free_arr.astype(jnp.float32),) if has_fp else ()
            )
            return np.asarray(
                pl.pallas_call(
                    _k,
                    out_shape=jax.ShapeDtypeStruct((cur_b, h), jnp.bfloat16),
                    grid_spec=pltpu.PrefetchScalarGridSpec(
                        num_scalar_prefetch=len(sp_args),
                        grid=(1,),
                        in_specs=(v_sp, v_sp, v_sp, h_sp, h_sp, h_sp, h_sp, v_sp),
                        out_specs=v_sp,
                        scratch_shapes=sc,
                    ),
                    interpret=pltpu.InterpretParams(**interp_kw),
                )(*sp_args, lg_arr, bias_j, x_arr, w1_u, s1_j, w2_u, s2_j, norm_j),
                dtype=np.float32,
            )

        lg16 = jnp.asarray(rng.normal(0.0, 2.5, size=(16, 256)), jnp.bfloat16)
        x16 = jnp.asarray(rng.normal(0.0, 0.25, size=(16, h)), jnp.bfloat16)
        base16 = v1_mod.default_sparse_decode_config(16, ep, scale_block)
        y16_dense = _run_core_1d(
            16,
            dataclasses.replace(
                base16, row_mode="dense", tile_m=16, m_buckets=(16,),
                route_fast=True, fast_plan=2, unroll_strips=True,
                fast_gs=True, pair_rs=False, bridge_ag=False,
                free_max_rows=0, min_p_topk=0.0,
            ),
            lg16, x16,
        )
        y16_sparse = _run_core_1d(
            16,
            dataclasses.replace(
                base16, row_mode="sparse", tile_m=16, m_buckets=(8, 16),
                route_chunk_rows=16, route_fast=True, fast_plan=2,
                unroll_strips=True, fast_gs=True, pair_rs=False, bridge_ag=False,
                free_max_rows=0, min_p_topk=0.0,
            ),
            lg16, x16,
        )
        self.assertEqual(float(np.max(np.abs(y16_dense - y16_sparse))), 0.0)

        lg32 = jnp.asarray(rng.normal(0.0, 2.5, size=(32, 256)), jnp.bfloat16)
        x32 = jnp.asarray(rng.normal(0.0, 0.25, size=(32, h)), jnp.bfloat16)
        base32 = v1_mod.default_sparse_decode_config(32, ep, scale_block)
        for route_fast_flag in (True, False):
            dyn_base = dataclasses.replace(
                base32, row_mode="sparse", tile_m=32, m_buckets=(16, 32),
                route_chunk_rows=32, route_fast=route_fast_flag, fast_plan=2,
                unroll_strips=True, fast_gs=True, pair_rs=False, bridge_ag=False,
            )
            for tier_mp, tier_rows, tier_w, tier_ms in (
                (0.88, 6, 0.18, 2),
                (0.82, 10, 0.22, 1),
                (0.0, 0, 0.20, 1),
            ):
                y_stat = _run_core_1d(
                    32,
                    dataclasses.replace(
                        dyn_base, min_p_topk=tier_mp, free_max_rows=tier_rows,
                        free_max_w=tier_w, free_min_slot=tier_ms,
                    ),
                    lg32, x32,
                )
                y_dyn = _run_core_1d(
                    32, dyn_base, lg32, x32,
                    free_arr=jnp.array([tier_mp, tier_rows, tier_w, tier_ms], jnp.float32),
                )
                max_d = float(np.max(np.abs(y_stat - y_dyn)))
                # route_fast=False tier=0 compares _route_chunk_hoisted tree sum vs sequential sum (<= 1 BF16 ULP).
                if route_fast_flag or tier_mp > 0.0:
                    self.assertEqual(
                        max_d,
                        0.0,
                        f"Dynamic free_params mismatch route_fast={route_fast_flag} tier={(tier_mp, tier_rows, tier_w, tier_ms)}",
                    )
                else:
                    self.assertLessEqual(
                        max_d,
                        1e-4,
                        f"Dynamic free_params mismatch route_fast={route_fast_flag} tier={(tier_mp, tier_rows, tier_w, tier_ms)}",
                    )

        # 3. Offline TPU7x Mosaic compilation for B=16 sparse & B=64 dynamic free_params.
        for topo_str, cur_ep in (("tpu7x:2x2x1", 8), ("tpu7x:2x2x2", 16)):
            tpu_devs = get_topology_desc(topo_str, platform="tpu").devices
            if len(tpu_devs) < cur_ep:
                continue
            mesh = Mesh(
                np.array(
                    sorted(tpu_devs, key=lambda d: (d.coords[1], d.coords[0], d.core_on_chip))
                ),
                ("model",),
            )
            cur_eloc = 256 // cur_ep
            for cur_b, cur_row_mode, cur_bkts, cur_ret_sh, use_fp in (
                (16, "sparse", (8, 16), False, False),
                (64, "sparse", (16, 32), True, True),
            ):
                cur_cfg = dataclasses.replace(
                    v1_mod.default_sparse_decode_config(cur_b, cur_ep, scale_block),
                    row_mode=cur_row_mode,
                    tile_m=cur_b if cur_b == 16 else 32,
                    m_buckets=cur_bkts,
                    route_chunk_rows=cur_b if cur_b == 16 else 32,
                    static_holder_mask=True,
                    route_fast=True,
                    fast_plan=2,
                    unroll_strips=True,
                    fast_gs=True,
                    pair_rs=True,
                    bridge_ag=True,
                    k256_strip_mode="adaptive",
                )
                shapes = (
                    jax.ShapeDtypeStruct((1,), jnp.int32),
                    jax.ShapeDtypeStruct((cur_b, 256), jnp.bfloat16),
                    jax.ShapeDtypeStruct((1, 256), jnp.float32),
                    jax.ShapeDtypeStruct((cur_b, h), jnp.bfloat16),
                    jax.ShapeDtypeStruct((cur_eloc, h, n1), jnp.float4_e2m1fn),
                    jax.ShapeDtypeStruct((cur_eloc, h // scale_block, n1), jnp.bfloat16),
                    jax.ShapeDtypeStruct((cur_eloc, i_dim, h), jnp.float4_e2m1fn),
                    jax.ShapeDtypeStruct((cur_eloc, i_dim // scale_block, h), jnp.bfloat16),
                    jax.ShapeDtypeStruct((1, h), jnp.bfloat16),
                ) + ((jax.ShapeDtypeStruct((4,), jnp.float32),) if use_fp else ())

                def _aot_fn(*args, _cfg=cur_cfg, _ep=cur_ep, _ret=cur_ret_sh, _fp=use_fp):
                    fp_arg = args[9] if _fp else None
                    return v1_mod.fused_decode_sparse_from_logits(
                        *args[:9],
                        free_params=fp_arg,
                        ep=_ep,
                        return_sharded=_ret,
                        config=_cfg,
                    )

                comp = (
                    jax.jit(
                        jax.shard_map(
                            _aot_fn,
                            mesh=mesh,
                            in_specs=(P(),) * len(shapes),
                            out_specs=P(),
                            check_vma=False,
                        )
                    )
                    .trace(*shapes)
                    .lower(lowering_platforms=("tpu",))
                    .compile()
                )
                self.assertIn("tpu_custom_call", comp.as_text())
                self.assertGreater(
                    comp.memory_analysis().generated_code_size_in_bytes, 0
                )

    def test_w9_pareto_nonuniform_free_and_dual_scale_placement(self):
        breakers_mod = _load_pkg_module(
            "decode_hbm_breakers", _MUSE_DIR / "decode_hbm_breakers.py"
        )
        v1_mod = _load_pkg_module(
            "fused_decode_sparse", _MUSE_DIR / "fused_decode_sparse.py"
        )
        planner_mod = _load_pkg_module(
            "moe_imbalance_planner", _MUSE_DIR / "moe_imbalance_planner.py"
        )

        # 1. Verify Pareto 2-tier presets preserve tuple return type and layer splits.
        for name, expected_mid in (
            ("pareto_f4w22p88", (0.88, 4, 0.22, 2)),
            ("pareto_f6w24p84", (0.84, 6, 0.24, 1)),
            ("pareto_f8w24p84", (0.84, 8, 0.24, 1)),
            ("pareto_f10w24p82", (0.82, 10, 0.24, 1)),
            ("pareto_f10w26p80", (0.80, 10, 0.26, 1)),
            ("pareto_f4w24p86", (0.86, 4, 0.24, 2)),
            ("pareto_f3w28p88", (0.88, 3, 0.28, 2)),
        ):
            parsed = breakers_mod.parse_nonuniform_free_spec(name)
            self.assertIsInstance(parsed, tuple)
            self.assertEqual(
                breakers_mod.resolve_nonuniform_free_for_layer(
                    30, 62, 64,
                    min_p_topk=0.86, free_rows=8, free_max_w=0.18, free_min_slot=2,
                    parse_free_spec_fn=v1_mod.parse_free_expert_rows_spec,
                    spec=name,
                ),
                expected_mid,
            )

        # 2. Verify 3-tier entropy presets & explicit multi-rule per-layer syntax.
        for spec_name, skew_exp, mid_exp in (
            ("pareto_3tier_safe", (0.88, 4, 0.22, 2), (0.86, 4, 0.24, 2)),
            ("pareto_3tier_fast", (0.88, 4, 0.22, 2), (0.88, 3, 0.28, 2)),
            ("pareto_3tier_ms1", (0.84, 6, 0.24, 1), (0.80, 10, 0.26, 1)),
        ):
            parsed_d = breakers_mod.parse_nonuniform_free_spec(spec_name)
            self.assertIsInstance(parsed_d, dict)
            # Boundary layer (L1) vs skewed layer (L26) vs broad middle layer (L30).
            self.assertEqual(
                breakers_mod.resolve_nonuniform_free_for_layer(
                    1, 62, 64,
                    min_p_topk=0.86, free_rows=8, free_max_w=0.18, free_min_slot=2,
                    parse_free_spec_fn=v1_mod.parse_free_expert_rows_spec,
                    spec=spec_name,
                ),
                (0.88, 6, 0.18, 2),
            )
            self.assertEqual(
                breakers_mod.resolve_nonuniform_free_for_layer(
                    26, 62, 64,
                    min_p_topk=0.86, free_rows=8, free_max_w=0.18, free_min_slot=2,
                    parse_free_spec_fn=v1_mod.parse_free_expert_rows_spec,
                    spec=spec_name,
                ),
                skew_exp,
            )
            self.assertEqual(
                breakers_mod.resolve_nonuniform_free_for_layer(
                    30, 62, 64,
                    min_p_topk=0.86, free_rows=8, free_max_w=0.18, free_min_slot=2,
                    parse_free_spec_fn=v1_mod.parse_free_expert_rows_spec,
                    spec=spec_name,
                ),
                mid_exp,
            )

        custom = "default=6:0.18:2:0.88;10-12,26=4:0.24:2:0.86"
        self.assertEqual(
            breakers_mod.resolve_nonuniform_free_for_layer(
                11, 62, 64,
                min_p_topk=0.86, free_rows=8, free_max_w=0.18, free_min_slot=2,
                parse_free_spec_fn=v1_mod.parse_free_expert_rows_spec,
                spec=custom,
            ),
            (0.86, 4, 0.24, 2),
        )
        with self.assertRaises(ValueError):
            breakers_mod.parse_nonuniform_free_spec("10-12=4:0.24:2:0.86")

        # 3. Dual-scale placement (`sub_owners=16` paired into `num_owners=8`) avoids
        #    EP=16 even-vs-odd rank hot/cold skew while balancing both EP=8 and EP=16.
        rng = np.random.default_rng(17)
        ranks = np.arange(1, 257, dtype=np.float64)
        rng.shuffle(ranks)
        skew_counts = 1.0 / np.power(ranks, 1.1)
        skew_counts = (skew_counts / skew_counts.sum()) * 100000.0
        chip_map = np.array([r >> 1 for r in range(8)], dtype=np.int32)

        ep8_only = planner_mod.build_static_placement(
            skew_counts, num_owners=8, owner_to_chip=chip_map,
            objective="decode_active", min_gain=0.0, hot_slot0_order=True,
        )
        dual_scale = planner_mod.build_static_placement(
            skew_counts, num_owners=8, owner_to_chip=chip_map,
            objective="decode_active", min_gain=0.0, hot_slot0_order=True,
            sub_owners=16,
        )
        self.assertEqual(
            sorted(dual_scale.tolist()), list(range(256))
        )
        w = planner_mod._decode_objective_weights(
            skew_counts, objective="decode_active", calib_batch=64
        )
        ep8_loads_16 = w[ep8_only].reshape(16, 16).sum(axis=1)
        dual_loads_16 = w[dual_scale].reshape(16, 16).sum(axis=1)
        dual_loads_8 = w[dual_scale].reshape(8, 32).sum(axis=1)
        self.assertLess(
            dual_loads_16.max() / dual_loads_16.mean(),
            ep8_loads_16.max() / ep8_loads_16.mean(),
        )
        self.assertLess(dual_loads_16.max() / dual_loads_16.mean(), 1.03)
        self.assertLess(dual_loads_8.max() / dual_loads_8.mean(), 1.02)
        # Slot 0 of every EP=16 sub-owner holds that sub-owner's hottest expert.
        for s in range(16):
            sub_slice = dual_scale[s * 16 : (s + 1) * 16]
            self.assertEqual(
                skew_counts[sub_slice[0]], skew_counts[sub_slice].max()
            )

    def test_w9_b8_native_and_d1_large_batch_free_parity_and_lowering(self):
        import dataclasses
        from unittest import mock
        try:
            import jax
            import jax.numpy as jnp
            from jax.experimental import pallas as pl
            from jax.experimental.pallas import tpu as pltpu
            from jax.experimental.topologies import get_topology_desc
            from jax.sharding import Mesh, PartitionSpec as P
        except Exception as exc:
            self.skipTest(f"JAX/Pallas TPU unavailable: {exc}")

        from sglang_tpu.envs import envs as tpu_envs
        from sglang_tpu.moe.pallas.fused_rs_muse import fused_decode_op as op_mod

        breakers_mod = _load_pkg_module(
            "decode_hbm_breakers", _MUSE_DIR / "decode_hbm_breakers.py"
        )
        routing_mod = _load_pkg_module(
            "fused_decode_routing", _MUSE_DIR / "fused_decode_routing.py"
        )
        v1_mod = _load_pkg_module(
            "fused_decode_sparse", _MUSE_DIR / "fused_decode_sparse.py"
        )

        # 1. Default-OFF checks for DECODE_B8 and DECODE_D1_LARGE_BATCH_FREE.
        self.assertIs(tpu_envs.SGLANG_TPU_MUSE_KERNEL_DECODE_B8.get(), False)
        self.assertIs(tpu_envs.SGLANG_TPU_MUSE_DECODE_D1_LARGE_BATCH_FREE.get(), False)
        self.assertNotIn(8, op_mod.supported_fuse_all_batches())

        prev_b8 = os.environ.get("SGLANG_TPU_MUSE_KERNEL_DECODE_B8")
        try:
            os.environ["SGLANG_TPU_MUSE_KERNEL_DECODE_B8"] = "1"
            self.assertIn(8, op_mod.supported_fuse_all_batches(ep=8))
            self.assertNotIn(8, op_mod.supported_fuse_all_batches(ep=16))
        finally:
            if prev_b8 is None:
                os.environ.pop("SGLANG_TPU_MUSE_KERNEL_DECODE_B8", None)
            else:
                os.environ["SGLANG_TPU_MUSE_KERNEL_DECODE_B8"] = prev_b8

        # D1 large-batch scaling vs verify defaults.
        self.assertEqual(
            breakers_mod.resolve_nonuniform_free_for_layer(
                30, 62, 128,
                min_p_topk=0.86, free_rows=8, free_max_w=0.18, free_min_slot=2,
                parse_free_spec_fn=v1_mod.parse_free_expert_rows_spec,
                spec="pareto_f6w24p84", d1_large_batch=False,
            ),
            (0.86, 8, 0.18, 2),
        )
        self.assertEqual(
            breakers_mod.resolve_nonuniform_free_for_layer(
                30, 62, 128,
                min_p_topk=0.86, free_rows=8, free_max_w=0.18, free_min_slot=2,
                parse_free_spec_fn=v1_mod.parse_free_expert_rows_spec,
                spec="pareto_f6w24p84", d1_large_batch=True,
            ),
            (0.84, 12, 0.24, 1),
        )
        self.assertEqual(
            breakers_mod.resolve_free_and_min_p_for_batch(
                256,
                min_p_topk=0.86, free_rows=8, free_max_w=0.18, free_min_slot=2,
                parse_free_spec_fn=v1_mod.parse_free_expert_rows_spec,
                d1_large_batch=True,
            ),
            (0.86, 32, 0.18, 2),
        )
        with mock.patch.dict(os.environ, {"SGLANG_TPU_MUSE_DECODE_D1_LARGE_BATCH_FREE": "1"}):
            with mock.patch("sglang.srt.runtime_context.get_spec", return_value=mock.Mock(speculative_algorithm=None)):
                self.assertIs(breakers_mod.read_d1_large_batch_free(), True)
            with mock.patch("sglang.srt.runtime_context.get_spec", return_value=mock.Mock(speculative_algorithm="DSPARK")):
                self.assertIs(breakers_mod.read_d1_large_batch_free(), False)

        # B=8 requires EP <= 8 (EP=16 fails cleanly).
        with self.assertRaises(ValueError):
            v1_mod.fused_decode_sparse_from_logits(
                jnp.zeros((1,), jnp.int32),
                jnp.zeros((8, 256), jnp.bfloat16),
                jnp.zeros((1, 256), jnp.float32),
                jnp.zeros((8, 4096), jnp.bfloat16),
                jnp.zeros((16, 4096, 4096), jnp.float4_e2m1fn),
                jnp.zeros((16, 8, 4096), jnp.bfloat16),
                jnp.zeros((16, 4096, 4096), jnp.float4_e2m1fn),
                jnp.zeros((16, 8, 4096), jnp.bfloat16),
                jnp.ones((1, 4096), jnp.bfloat16),
                ep=16,
            )

        # 2. CPU Interpret parity for B=8 (ep=8, tile_m=16, m_buckets=(8,)) and
        #    B=128 (d=1 scaled FREE free_params=(0.86, 16, 0.18, 2)).
        rng = np.random.default_rng(20261126)
        h, n1, i_dim, scale_block, ep = 4096, 8192, 4096, 512, 8
        e_local = 256 // ep
        w1_u = jnp.asarray(
            rng.integers(0, 2**31, size=(e_local, h // 8, n1), dtype=np.int32).view(np.uint32)
        )
        w2_u = jnp.asarray(
            rng.integers(0, 2**31, size=(e_local, i_dim // 8, h), dtype=np.int32).view(np.uint32)
        )
        s1_j = jnp.asarray(rng.uniform(0.02, 0.08, size=(e_local, h // scale_block, n1)), jnp.bfloat16)
        s2_j = jnp.asarray(rng.uniform(0.02, 0.08, size=(e_local, i_dim // scale_block, h)), jnp.bfloat16)
        bias_j = jnp.zeros((1, 256), jnp.float32)
        norm_j = jnp.asarray(rng.uniform(0.8, 1.2, size=(1, h)), jnp.bfloat16)
        interp_kw = {
            "detect_races": True,
            **(
                {"uninitialized_memory": "nan"}
                if "uninitialized_memory" in pltpu.InterpretParams.__annotations__
                else {}
            ),
        }
        v_sp, h_sp = pl.BlockSpec(memory_space=pltpu.VMEM), pl.BlockSpec(memory_space=pltpu.HBM)

        lg_b8 = jnp.asarray(rng.normal(0.0, 2.0, size=(8, 256)), jnp.bfloat16)
        x_b8 = jnp.asarray(rng.normal(0.0, 0.25, size=(8, h)), jnp.bfloat16)
        lg_b128 = jnp.asarray(rng.normal(0.0, 2.0, size=(128, 256)), jnp.bfloat16)
        x_b128 = jnp.asarray(rng.normal(0.0, 0.25, size=(128, h)), jnp.bfloat16)

        for cur_b, rk_id, fp_vals, cfg_in, lg_j, x_j, expect_zero in (
            (
                8,
                0,
                (0.88, 1, 0.18, 2),
                dataclasses.replace(
                    v1_mod.default_sparse_decode_config(8, ep, scale_block),
                    route_fast=True,
                    fast_plan=2,
                    unroll_strips=True,
                    fast_gs=True,
                    pair_rs=False,
                    bridge_ag=False,
                ),
                lg_b8,
                x_b8,
                True,
            ),
            (
                8,
                1,
                (0.88, 1, 0.18, 2),
                dataclasses.replace(
                    v1_mod.default_sparse_decode_config(8, ep, scale_block),
                    route_fast=True,
                    fast_plan=2,
                    unroll_strips=True,
                    fast_gs=True,
                    pair_rs=False,
                    bridge_ag=False,
                ),
                lg_b8,
                x_b8,
                False,
            ),
            (
                128,
                0,
                (0.86, 16, 0.18, 2),
                dataclasses.replace(
                    v1_mod.default_sparse_decode_config(128, ep, scale_block),
                    route_fast=True,
                    fast_plan=2,
                    unroll_strips=True,
                    fast_gs=True,
                    pair_rs=False,
                    bridge_ag=False,
                ),
                lg_b128,
                x_b128,
                False,
            ),
        ):

            def _run_e2e_core(cur_cfg, fp_arr=None):
                sc = v1_mod.build_muse_sparse_scratch_shapes(
                    ep=ep, batch_size=cur_b, scale_block=scale_block,
                    s1_dtype=jnp.bfloat16, s2_dtype=jnp.bfloat16,
                    num_replicas=0, return_sharded=True, config=cur_cfg,
                )
                has_fp = fp_arr is not None

                def _k(*all_refs):
                    it = iter(all_refs)
                    rk_r = next(it)
                    fp_r = next(it) if has_fp else None
                    lg_r, bi_r, x_r = next(it), next(it), next(it)
                    w1_r, s1_r, w2_r, s2_r, nm_r = (next(it) for _ in range(5))
                    out_r = next(it)
                    r_sc = routing_mod.RoutingScratch(
                        *(next(it) for _ in range(len(routing_mod.RoutingScratch._fields)))
                    )
                    w1_b, s1_b, w2_b, s2_b, w1_sm, w2_sm, m_sm = (next(it) for _ in range(7))
                    x3, io3, l2, a2, ys, gb, ac3 = (next(it) for _ in range(7))
                    rs_r3, rs_ss, rs_rs, ow3 = (next(it) for _ in range(4))

                    def _dump():
                        out_r[...] = ac3[pl.ds(0, cur_b), :, :].reshape(cur_b, h).astype(jnp.bfloat16)

                    v1_mod._run_muse_sparse_from_vmem_core(
                        rk_r[0], lg_r, bi_r, x_r, w1_r, s1_r, w2_r, s2_r, nm_r,
                        d_scratch=None, r_scratch=r_sc,
                        w1_buf=w1_b, s1_buf=s1_b, w2_buf=w2_b, s2_buf=s2_b,
                        w1_sem=w1_sm, w2_sem=w2_sm, misc_sem=m_sm, w1_acc_buf=None,
                        x_3d_buf=x3, io_3d_buf=io3, lhs_2d_buf=l2,
                        act_2d_buf=a2, y_stage_buf=ys,
                        lhs_fp8_buf=None, lhs_sc_buf=None,
                        masked_x_buf=None, bd_sc_buf=None,
                        gain_f32_buf=gb, accum_3d=ac3, accum_2d=None, fan_3d_buf=None,
                        rs_send_bf16=None, rs_pair_recv=None,
                        rs_chip_send=None, rs_chip_recv=None,
                        rs_pair_ssem=None, rs_pair_rsem=None,
                        rs_chip_ssem=None, rs_chip_rsem=None,
                        rs_recv_3d=rs_r3, rs_send_sem=rs_ss, rs_recv_sem=rs_rs,
                        owner_3d=ow3, fan_send_sem=None, fan_recv_sem=None,
                        fan_s2_send_sem=None, fan_s2_recv_sem=None,
                        barrier_sem=None, output_ref=out_r,
                        free_params_ref=fp_r,
                        ep=ep, batch_size=cur_b, scale_block=scale_block, num_replicas=0,
                        replica_cost_mode="active", return_sharded=True,
                        ep_axis_name="model", cfg=cur_cfg,
                        wait_entry_barrier=False,
                        stop_stage="experts", early_exit_fn=_dump,
                    )

                sp_args = (jnp.array([rk_id], jnp.int32),) + (
                    (fp_arr.astype(jnp.float32),) if has_fp else ()
                )
                return np.asarray(
                    pl.pallas_call(
                        _k,
                        out_shape=jax.ShapeDtypeStruct((cur_b, h), jnp.bfloat16),
                        grid_spec=pltpu.PrefetchScalarGridSpec(
                            num_scalar_prefetch=len(sp_args),
                            grid=(1,),
                            in_specs=(v_sp, v_sp, v_sp, h_sp, h_sp, h_sp, h_sp, v_sp),
                            out_specs=v_sp,
                            scratch_shapes=sc,
                        ),
                        interpret=pltpu.InterpretParams(**interp_kw),
                    )(*sp_args, lg_j, bias_j, x_j, w1_u, s1_j, w2_u, s2_j, norm_j),
                    dtype=np.float32,
                )

            y_dyn = _run_e2e_core(cfg_in, jnp.asarray(fp_vals, jnp.float32))
            y_stat = _run_e2e_core(
                dataclasses.replace(
                    cfg_in,
                    min_p_topk=fp_vals[0],
                    free_max_rows=fp_vals[1],
                    free_max_w=fp_vals[2],
                    free_min_slot=fp_vals[3],
                )
            )
            self.assertEqual(
                float(np.max(np.abs(y_dyn - y_stat))), 0.0,
                msg=f"B={cur_b} rk={rk_id} dynamic free_params vs static mismatch",
            )
            if expect_zero:
                self.assertEqual(float(np.linalg.norm(y_dyn)), 0.0)
            else:
                self.assertGreater(float(np.linalg.norm(y_dyn)), 0.0)

        # 3. Real TPU7x 2x2x1 AOT .compile() for both B=8 and B=128 with free_params!
        tpu_devs = get_topology_desc("tpu7x:2x2x1", platform="tpu").devices
        if len(tpu_devs) >= 8:
            mesh = Mesh(
                np.array(
                    sorted(tpu_devs, key=lambda d: (d.coords[1], d.coords[0], d.core_on_chip))
                ),
                ("model",),
            )
            for cur_b, cur_ret_sh in ((8, False), (128, True)):
                cur_cfg = dataclasses.replace(
                    v1_mod.default_sparse_decode_config(cur_b, ep, scale_block),
                    static_holder_mask=True, route_fast=True, fast_plan=2,
                    unroll_strips=True, fast_gs=True, pair_rs=True,
                    bridge_ag=True, k256_strip_mode="adaptive",
                )
                shapes = (
                    jax.ShapeDtypeStruct((1,), jnp.int32),
                    jax.ShapeDtypeStruct((cur_b, 256), jnp.bfloat16),
                    jax.ShapeDtypeStruct((1, 256), jnp.float32),
                    jax.ShapeDtypeStruct((cur_b, h), jnp.bfloat16),
                    jax.ShapeDtypeStruct((e_local, h, n1), jnp.float4_e2m1fn),
                    jax.ShapeDtypeStruct((e_local, h // scale_block, n1), jnp.bfloat16),
                    jax.ShapeDtypeStruct((e_local, i_dim, h), jnp.float4_e2m1fn),
                    jax.ShapeDtypeStruct((e_local, i_dim // scale_block, h), jnp.bfloat16),
                    jax.ShapeDtypeStruct((1, h), jnp.bfloat16),
                    jax.ShapeDtypeStruct((4,), jnp.float32),
                )

                def _aot_fn(*args, _cfg=cur_cfg, _ret=cur_ret_sh):
                    return v1_mod.fused_decode_sparse_from_logits(
                        *args[:9], free_params=args[9], ep=8, return_sharded=_ret, config=_cfg,
                    )

                comp = (
                    jax.jit(
                        jax.shard_map(
                            _aot_fn, mesh=mesh,
                            in_specs=(P(),) * len(shapes), out_specs=P(), check_vma=False,
                        )
                    )
                    .trace(*shapes)
                    .lower(lowering_platforms=("tpu",))
                    .compile()
                )
                self.assertIn("tpu_custom_call", comp.as_text())
                self.assertGreater(
                    comp.memory_analysis().generated_code_size_in_bytes, 0
                )

    def test_w9_rank_capped_free_parity_and_aot(self):
        """Verify Wave-9.18b progressive rank-capped FREE & R=12/16 across router paths and AOT."""
        import dataclasses
        import jax
        import jax.numpy as jnp
        from sglang_tpu.moe.pallas.fused_rs_muse import fused_decode_sparse as v1_mod
        from jax.experimental import pallas as pl
        from jax.experimental.pallas import tpu as pltpu
        from jax.experimental.topologies import get_topology_desc
        from jax.sharding import Mesh, PartitionSpec as P
        from sglang_tpu.moe.pallas.fused_rs_muse import decode_hbm_breakers as brk
        from sglang_tpu.moe.pallas.fused_rs_muse import fused_decode_routing as routing_mod
        from sglang_tpu.moe.pallas.fused_rs_muse import moe_imbalance_planner as mip

        # 1. Spec parsing and batch scaling invariants.
        self.assertEqual(brk.parse_free_rank_cap_spec(""), (0, 0, 0.0))
        self.assertEqual(brk.parse_free_rank_cap_spec("off"), (0, 0, 0.0))
        self.assertEqual(brk.parse_free_rank_cap_spec("11:2:0.32"), (11, 2, 0.32))
        self.assertEqual(brk.scale_free_rank_cap_for_batch(11, 2, 16), (0, 0))
        self.assertEqual(brk.scale_free_rank_cap_for_batch(11, 2, 64), (11, 2))
        self.assertEqual(
            brk.scale_free_rank_cap_for_batch(11, 2, 128, d1_large_batch=False), (0, 0)
        )
        self.assertEqual(
            brk.scale_free_rank_cap_for_batch(11, 2, 128, d1_large_batch=True), (16, 4)
        )

        # 2. Construct deterministic skewed logits where rank 0 has 15 active experts (5 x 1-row
        #    tail experts with w ~ 0.30) while rank 7 has only 4 active experts (< cap=11).
        rng = np.random.default_rng(20261128)
        ep, e_local = 8, 32
        b_tok = 64
        logits_64 = np.full((b_tok, 256), -12.0, dtype=np.float32)
        bias_np = np.zeros((1, 256), dtype=np.float32)
        lsig = lambda p: float(np.log(p / (1.0 - p)))
        for t in range(b_tok):
            # Slot 0 on sparse rank 7 (e=224..227, well below cap=11) or hot rank 0 (e=0..3)
            e0 = (t % 4) if (t < 32) else (224 + (t % 4))
            e1 = 4 + (t % 5) if (t < 32) else (32 + (t % 8))
            logits_64[t, e0] = lsig(0.38)
            logits_64[t, e1] = lsig(0.31)
            # Every token gets a 3rd slot (w ~ 0.295): tokens 0..10 put distinct 1-row experts
            # on rank 0 (e=10..20) pushing rank 0 above cap=11!
            if t < 11:
                e2 = 10 + t
            elif t < 20:
                e2 = 224 + ((t + 1) % 4)
            else:
                e2 = 64 + (t % 16)
            logits_64[t, e2] = lsig(0.295)

        ref_w_rc = np.asarray(
            routing_mod.route_weights(
                jnp.asarray(logits_64, jnp.bfloat16),
                jnp.asarray(bias_np, jnp.float32),
                min_p_topk=0.80,
                free_max_rows=10,
                free_max_w=0.26,
                free_min_slot=1,
                free_rank_cap=11,
                free_tail_rows=2,
                free_tail_w=0.32,
            ),
            dtype=np.float32,
        )
        ref_w_base = np.asarray(
            routing_mod.route_weights(
                jnp.asarray(logits_64, jnp.bfloat16),
                jnp.asarray(bias_np, jnp.float32),
                min_p_topk=0.80,
                free_max_rows=10,
                free_max_w=0.26,
                free_min_slot=1,
            ),
            dtype=np.float32,
        )
        r0_act_base = int(np.sum(ref_w_base[:, :32].sum(axis=0) > 0))
        r0_act_rc = int(np.sum(ref_w_rc[:, :32].sum(axis=0) > 0))
        r7_act_base = int(np.sum(ref_w_base[:, 224:256].sum(axis=0) > 0))
        r7_act_rc = int(np.sum(ref_w_rc[:, 224:256].sum(axis=0) > 0))
        self.assertGreater(r0_act_base, 11)
        self.assertLessEqual(r0_act_rc, 11)
        self.assertEqual(r7_act_rc, r7_act_base)

        # Verify InterpretParams(detect_races=True, uninitialized_memory='nan') parity across
        # route_fast in {False, True} and B in {64, 128}!
        logits_128 = np.concatenate([logits_64, logits_64], axis=0)
        ref_w_128 = np.asarray(
            routing_mod.route_weights(
                jnp.asarray(logits_128, jnp.bfloat16),
                jnp.asarray(bias_np, jnp.float32),
                min_p_topk=0.80,
                free_max_rows=20,
                free_max_w=0.26,
                free_min_slot=1,
                free_rank_cap=11,
                free_tail_rows=4,
                free_tail_w=0.32,
            ),
            dtype=np.float32,
        )

        v_sp = pl.BlockSpec(memory_space=pltpu.VMEM)
        interp_kw = dict(dma_execution_mode="eager", detect_races=True, uninitialized_memory="nan")
        for cur_b, cur_lg, cur_ref, cur_fr, cur_tr in (
            (64, logits_64, ref_w_rc, 10, 2),
            (128, logits_128, ref_w_128, 20, 4),
        ):
            for r_fast in (False, True):
                sc = routing_mod.routing_scratch_shapes(cur_b, e_local, topk=8, num_experts=256)
                out_shapes = (
                    jax.ShapeDtypeStruct((8, 256), jnp.int32),
                    jax.ShapeDtypeStruct((cur_b, 128), jnp.int32),
                )

                def _rt_k(rk_r, lg_r, bi_r, cnt_o, pk_o, *sc_refs,
                          _rf=r_fast, _fr=cur_fr, _tr=cur_tr, _cb=cur_b):
                    r_sc = routing_mod.RoutingScratch(*sc_refs[:-1])
                    d_sem = sc_refs[-1]
                    routing_mod.route_topk_and_counts_in_kernel(
                        lg_r, bi_r, r_sc, d_sem.at[0],
                        my_owner=rk_r[0], e_per_owner=32,
                        topk=8, route_chunk_rows=32,
                        min_p_topk=0.80, free_max_rows=_fr, free_max_w=0.26, free_min_slot=1,
                        route_fast=_rf, free_rank_cap=11, free_tail_rows=_tr, free_tail_w=0.32,
                    )
                    cnt_o[...] = r_sc.counts_vmem[...]
                    pk_o[...] = r_sc.route_pack_vmem[:_cb, :]

                cnt_out, pk_out = pl.pallas_call(
                    _rt_k,
                    out_shape=out_shapes,
                    grid_spec=pltpu.PrefetchScalarGridSpec(
                        num_scalar_prefetch=1,
                        grid=(1,),
                        in_specs=(v_sp, v_sp),
                        out_specs=(v_sp, v_sp),
                        scratch_shapes=list(sc) + [pltpu.SemaphoreType.DMA((1,))],
                    ),
                    interpret=pltpu.InterpretParams(**interp_kw),
                )(
                    jnp.array([0], jnp.int32),
                    jnp.asarray(cur_lg, jnp.bfloat16),
                    jnp.asarray(bias_np, jnp.float32),
                )
                cnt_np = np.asarray(cnt_out[0], dtype=np.int32)
                exp_cnt = (cur_ref > 0).sum(axis=0).astype(np.int32)
                np.testing.assert_array_equal(
                    cnt_np, exp_cnt,
                    err_msg=f"counts mismatch B={cur_b} route_fast={r_fast}",
                )
                pk_i32 = np.asarray(pk_out, dtype=np.int32)
                pk_f32 = pk_i32.view(np.float32)
                dense_got = np.zeros((cur_b, 32), dtype=np.float32)
                for t in range(cur_b):
                    n_loc = int(pk_i32[t, 0])
                    for j in range(n_loc):
                        dense_got[t, int(pk_i32[t, 1 + j])] = float(pk_f32[t, 16 + j])
                np.testing.assert_allclose(
                    dense_got, cur_ref[:, :32], atol=1e-5, rtol=1e-5,
                    err_msg=f"weight parity mismatch B={cur_b} route_fast={r_fast}",
                )

        # 3. Verify R=12 replica plan build + TPU7x AOT .compile() for rank-capped FREE & R=12.
        plan_r12 = mip.build_preloaded_replica_plan(
            np.arange(1, 257, dtype=np.float64),
            num_owners=8, num_replicas_per_owner=12,
            owner_to_chip=[r >> 1 for r in range(8)],
            objective="decode_hybrid", selection_mode="bottleneck_owner",
        )
        self.assertEqual(plan_r12.replica_meta.shape, (4, 8 * 12))
        mip.validate_replica_plan(plan_r12)

        tpu_devs = get_topology_desc("tpu7x:2x2x1", platform="tpu").devices
        if len(tpu_devs) >= 8:
            mesh = Mesh(
                np.array(
                    sorted(tpu_devs, key=lambda d: (d.coords[1], d.coords[0], d.core_on_chip))
                ),
                ("model",),
            )
            h, i_dim, n1, sb = 4096, 4096, 8192, 512
            for cur_b, r_fast, rep_n in ((64, False, 0), (64, True, 12), (128, False, 0)):
                cur_cfg = dataclasses.replace(
                    v1_mod.default_sparse_decode_config(cur_b, ep, sb),
                    route_fast=r_fast, pair_rs=True,
                    min_p_topk=0.80, free_max_rows=(10 if cur_b == 64 else 20),
                    free_max_w=0.26, free_min_slot=1,
                    free_rank_cap=11, free_tail_rows=(2 if cur_b == 64 else 4), free_tail_w=0.32,
                )
                shapes = (
                    jax.ShapeDtypeStruct((1,), jnp.int32),
                    jax.ShapeDtypeStruct((cur_b, 256), jnp.bfloat16),
                    jax.ShapeDtypeStruct((1, 256), jnp.float32),
                    jax.ShapeDtypeStruct((cur_b, h), jnp.bfloat16),
                    jax.ShapeDtypeStruct((e_local, h, n1), jnp.float4_e2m1fn),
                    jax.ShapeDtypeStruct((e_local, h // sb, n1), jnp.bfloat16),
                    jax.ShapeDtypeStruct((e_local, i_dim, h), jnp.float4_e2m1fn),
                    jax.ShapeDtypeStruct((e_local, i_dim // sb, h), jnp.bfloat16),
                    jax.ShapeDtypeStruct((1, h), jnp.bfloat16),
                )
                if rep_n > 0:
                    shapes = shapes + (
                        jax.ShapeDtypeStruct((rep_n, h, n1), jnp.float4_e2m1fn),
                        jax.ShapeDtypeStruct((rep_n, h // sb, n1), jnp.bfloat16),
                        jax.ShapeDtypeStruct((rep_n, i_dim, h), jnp.float4_e2m1fn),
                        jax.ShapeDtypeStruct((rep_n, i_dim // sb, h), jnp.bfloat16),
                        jax.ShapeDtypeStruct((4, ep * rep_n), jnp.int32),
                    )

                def _aot_rc(*args, _cfg=cur_cfg, _rn=rep_n):
                    kw = {}
                    if _rn > 0:
                        kw = dict(
                            rep_w1=args[9], rep_s1=args[10], rep_w2=args[11],
                            rep_s2=args[12], replica_meta=args[13],
                        )
                    return v1_mod.fused_decode_sparse_from_logits(
                        *args[:9], **kw, ep=8, return_sharded=True, config=_cfg,
                    )

                comp = (
                    jax.jit(
                        jax.shard_map(
                            _aot_rc, mesh=mesh,
                            in_specs=(P(),) * len(shapes), out_specs=P(), check_vma=False,
                        )
                    )
                    .trace(*shapes)
                    .lower(lowering_platforms=("tpu",))
                    .compile()
                )
                self.assertIn("tpu_custom_call", comp.as_text())
                self.assertGreater(
                    comp.memory_analysis().generated_code_size_in_bytes, 0
                )

    def test_w10_r16_fast_plan_cur1_epi_and_tail_planner(self):
        """Verify R<=16 unrolled fast_plan, tail-excess replica planner, and FAST_B64=4 cur1 parity + AOT."""
        import dataclasses
        import numpy as np

        mip = _load_pkg_module(
            "moe_imbalance_planner", _MUSE_DIR / "moe_imbalance_planner.py"
        )
        v1_mod = _load_pkg_module(
            "fused_decode_sparse", _MUSE_DIR / "fused_decode_sparse.py"
        )

        # 1. Verify minimax tail-excess planner modes & multi-holder (_h2) across EP=8/16 & R=8/12.
        rng = np.random.default_rng(20261130)
        skew_c = np.full(256, 20.0, dtype=np.float64)
        skew_c[:32] = np.linspace(500.0, 2200.0, 32)
        skew_c[128:144] = np.linspace(250.0, 900.0, 16)
        for ep, r_slots in ((8, 12), (16, 8)):
            o2c = [r >> 1 for r in range(ep)]
            for mode in (
                "bottleneck_owner",
                "bottleneck_owner_tail_p3",
                "bottleneck_owner_tail_p3_h2",
            ):
                plan = mip.build_preloaded_replica_plan(
                    skew_c,
                    num_owners=ep,
                    num_replicas_per_owner=r_slots,
                    owner_to_chip=o2c,
                    objective="decode_active",
                    selection_mode=mode,
                )
                mip.validate_replica_plan(plan)
                self.assertEqual(plan.replica_meta.shape, (4, ep * r_slots))

        # 2. Verify FAST_B64=4 single-knob config vs legacy FAST_B64=0 defaults.
        clean_env = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith("SGLANG_TPU_MUSE_")
        }
        with mock.patch.dict(os.environ, clean_env, clear=True):
            cfg_legacy = v1_mod.default_sparse_decode_config(64, 8, 512)
            self.assertFalse(cfg_legacy.nc_split512 or cfg_legacy.route_fast)
            self.assertEqual(cfg_legacy.fast_plan, 0)
            self.assertFalse(cfg_legacy.fast_b64_epi_pro)

        with mock.patch.dict(
            os.environ,
            {**clean_env, "SGLANG_TPU_MUSE_KERNEL_DECODE_FAST_B64": "4"},
            clear=True,
        ):
            c32 = v1_mod.default_sparse_decode_config(32, 8, 512)
            c64 = v1_mod.default_sparse_decode_config(64, 8, 512)
            c128 = v1_mod.default_sparse_decode_config(128, 8, 512)
            self.assertTrue(c64.nc_split512 and c64.route_fast and c64.fast_b64_epi_pro)
            self.assertFalse(c64.fast_gs or c64.unroll_strips)
            self.assertEqual(c64.fast_plan, 2)
            self.assertEqual(c64.m_buckets, (16, 32))
            self.assertEqual(c32.m_buckets, (16, 32))
            self.assertEqual(c64.weight_mode, "stream3")
            self.assertEqual(c128.weight_mode, "half_n")
            self.assertTrue(c128.fast_b64_epi_pro and c128.route_fast)
            self.assertEqual(c128.fast_plan, 2)

        try:
            import jax
            import jax.numpy as jnp
            from jax.experimental import pallas as pl
            from jax.experimental.pallas import tpu as pltpu
            from jax.experimental.topologies import get_topology_desc
            from jax.sharding import Mesh, PartitionSpec as P
        except Exception as exc:
            self.skipTest(f"JAX/Pallas TPU unavailable: {exc}")

        # 3. Eager race/nan-detect Interpret parity for R=12/16 fast_plan, nc_split512, and cur_rows==1 gather/scatter.
        from jax import lax

        routing_mod = _load_pkg_module(
            "fused_decode_routing", _MUSE_DIR / "fused_decode_routing.py"
        )
        interp_p = {
            "dma_execution_mode": "eager",
            "detect_races": True,
            **({"uninitialized_memory": "nan"} if "uninitialized_memory" in pltpu.InterpretParams.__annotations__ else {}),
        }

        # 3a. R=12 & R=16 unrolled vs fori_loop planner + exec_plan bit-exact parity.
        for cur_ep, cur_r in ((8, 12), (8, 16), (16, 12)):
            e_loc = 256 // cur_ep
            max_ex = e_loc + cur_r
            cur_plan = mip.build_preloaded_replica_plan(
                skew_c,
                num_owners=cur_ep,
                num_replicas_per_owner=cur_r,
                owner_to_chip=[r >> 1 for r in range(cur_ep)],
                objective="decode_active",
                selection_mode="bottleneck_owner_tail_p3",
            )
            counts_np = rng.integers(0, 6, size=(1, 256), dtype=np.int32)

            def _run_plan_kernel(fast_u: int):
                sc_spec = routing_mod.routing_scratch_shapes(
                    64, max_ex, topk=8, num_experts=256
                )
                extra_specs = (
                    pltpu.SMEM((cur_ep,), jnp.int32),
                    pltpu.SMEM((8,), jnp.int32),
                    pltpu.SMEM((e_loc,), jnp.int32),
                    pltpu.SMEM((cur_r,), jnp.int32),
                )

                def _kern(rm_ref, cnt_ref, sup_out, exec_id_out, exec_src_out, exec_cnt_out, *sc_refs):
                    rs = routing_mod.RoutingScratch(*sc_refs[: len(sc_spec)])
                    ol, mv, sup, hs = sc_refs[len(sc_spec) :]

                    @pl.loop(0, 256)
                    def _cp(e):
                        rs.counts_smem[0, e] = cnt_ref[0, e]

                    mip.plan_preloaded_replicas_smem(
                        jnp.int32(0), e_loc, cur_ep, cur_r,
                        rs.counts_smem, rm_ref, sup, hs, ol,
                        cost_mode="tiles", tile_m=32, moved_flag_smem=mv,
                        fast_unroll=fast_u,
                    )
                    for le in range(e_loc):
                        sup_out[le] = sup[le]
                    n_ex = routing_mod.build_exec_plan_from_masks_smem(
                        jnp.int32(0), e_loc, cur_r, sup, hs, rs,
                        fast_plan=bool(fast_u),
                    )
                    exec_cnt_out[0] = n_ex

                    @pl.loop(0, max_ex)
                    def _out(i):
                        ok = i < n_ex
                        exec_id_out[i] = jnp.where(ok, rs.exec_expert_id[i], jnp.int32(-1))
                        exec_src_out[i] = jnp.where(ok, rs.exec_weight_src[i], jnp.int32(-1))

                sm = pl.BlockSpec(memory_space=pltpu.SMEM)
                return pl.pallas_call(
                    _kern,
                    out_shape=(
                        jax.ShapeDtypeStruct((e_loc,), jnp.int32),
                        jax.ShapeDtypeStruct((max_ex,), jnp.int32),
                        jax.ShapeDtypeStruct((max_ex,), jnp.int32),
                        jax.ShapeDtypeStruct((1,), jnp.int32),
                    ),
                    grid_spec=pltpu.PrefetchScalarGridSpec(
                        num_scalar_prefetch=0,
                        grid=(1,),
                        in_specs=(sm, sm),
                        out_specs=(sm, sm, sm, sm),
                        scratch_shapes=list(sc_spec) + list(extra_specs),
                    ),
                    interpret=pltpu.InterpretParams(**interp_p),
                )(jnp.asarray(cur_plan.replica_meta, jnp.int32), jnp.asarray(counts_np, jnp.int32))

            s_ref, id_ref, src_ref, n_ref = _run_plan_kernel(0)
            s_fast, id_fast, src_fast, n_fast = _run_plan_kernel(2)
            self.assertEqual(int(n_ref[0]), int(n_fast[0]))
            self.assertTrue(np.array_equal(np.asarray(s_ref), np.asarray(s_fast)))
            self.assertTrue(np.array_equal(np.asarray(id_ref), np.asarray(id_fast)))
            self.assertTrue(np.array_equal(np.asarray(src_ref), np.asarray(src_fast)))

        # 3b. Lightweight Interpret parity for nc_split512 (_fp4_gmm_chunked).
        m_rows, k_dim, n_col, sb = 16, 4096, 1024, 512
        x_small = jnp.asarray(rng.normal(0.0, 0.25, size=(m_rows, k_dim)), jnp.bfloat16)
        w_u32_small = jnp.asarray(
            rng.integers(0, 0xFFFFFFFF, size=(k_dim // 8, n_col), dtype=np.uint32)
        )
        sc_small = jnp.asarray(rng.uniform(0.01, 0.05, size=(k_dim // sb, n_col)), jnp.bfloat16)

        def _run_gmm_test(nc_split: bool):
            def _kern(x_r, w_r, s_r, o_r):
                o_r[...] = v1_mod._fp4_gmm_chunked(
                    x_r, w_r, s_r, m_rows, k_dim, 0, n_col, True, sb,
                    hoist_scales=True, nc_split512=nc_split, k512_chain=False,
                )

            vm = pltpu.VMEM
            return pl.pallas_call(
                _kern,
                out_shape=jax.ShapeDtypeStruct((m_rows, n_col), jnp.float32),
                grid_spec=pltpu.PrefetchScalarGridSpec(
                    num_scalar_prefetch=0, grid=(1,),
                    in_specs=(pl.BlockSpec(memory_space=vm),) * 3,
                    out_specs=pl.BlockSpec(memory_space=vm),
                ),
                interpret=pltpu.InterpretParams(**interp_p),
            )(x_small, w_u32_small, sc_small)

        gmm_ref = np.asarray(_run_gmm_test(False))
        gmm_nc = np.asarray(_run_gmm_test(True))
        np.testing.assert_allclose(gmm_nc, gmm_ref, atol=1e-3, rtol=1e-3)

        # 3c. Lightweight Interpret parity for cur_rows==1 vs <=8 gather/scatter under uninitialized_memory="nan".
        b_gs = 16
        x_3d_in = jnp.asarray(rng.normal(0.0, 0.3, size=(b_gs, 32, 128)), jnp.bfloat16)
        p_ids = np.array([3, 7, 11, 2], dtype=np.int32)  # exp0: cur_rows=1, exp1: cur_rows=3
        p_w = np.array([0.8, 0.4, 0.35, 0.25], dtype=np.float32)
        starts_np = np.array([0, 1, 4], dtype=np.int32)

        def _run_cur1_gs_test(cur1_fast: bool):
            def _kern2(x_r, ids_r, w_r, st_r, out_r):
                def _body(io_b, acc):
                    acc[...] = jnp.zeros((b_gs + 1, 32, 128), jnp.float32)

                    def _step(k, _c):
                        bp = st_r[k]
                        cr = st_r[k + 1] - bp
                        if cur1_fast:
                            @pl.when(cr == 1)
                            def _g1():
                                t0 = ids_r[bp]
                                io_b[pl.ds(0, 1), :, :] = x_r[pl.ds(t0, 1), :, :]

                            @pl.when((cr > 1) & (cr <= 8))
                            def _gm():
                                ts8 = [ids_r[jnp.where(s < cr, bp + s, bp)] for s in range(8)]
                                io_b[pl.ds(0, 8), :, :] = jnp.concatenate(
                                    [x_r[pl.ds(t, 1), :, :] for t in ts8], axis=0
                                )
                        else:
                            ts8 = [ids_r[jnp.where(s < cr, bp + s, bp)] for s in range(8)]
                            io_b[pl.ds(0, 8), :, :] = jnp.concatenate(
                                [x_r[pl.ds(t, 1), :, :] for t in ts8], axis=0
                            )

                        # Simulate m_rows=16 compute transforming row 0..cr and writing io_b[0..16]
                        io_b[pl.ds(0, 16), :, :] = (
                            io_b[pl.ds(0, 16), :, :].astype(jnp.float32) * 1.25
                        ).astype(jnp.bfloat16)

                        if cur1_fast:
                            @pl.when(cr == 1)
                            def _s1():
                                tok = ids_r[bp]
                                wt = w_r[bp]
                                ys = io_b[pl.ds(0, 1), :, :].astype(jnp.float32) * wt
                                acc[pl.ds(tok, 1), :, :] = acc[pl.ds(tok, 1), :, :] + ys

                            @pl.when((cr > 1) & (cr <= 8))
                            def _sm():
                                y8 = io_b[pl.ds(0, 8), :, :]
                                for s in range(8):
                                    @pl.when(s < cr)
                                    def _sc(_s=s):
                                        tok = ids_r[bp + _s]
                                        wt = w_r[bp + _s]
                                        ys = y8[_s : _s + 1, :, :].astype(jnp.float32) * wt
                                        acc[pl.ds(tok, 1), :, :] = acc[pl.ds(tok, 1), :, :] + ys
                        else:
                            y8 = io_b[pl.ds(0, 8), :, :]
                            for s in range(8):
                                @pl.when(s < cr)
                                def _sc(_s=s):
                                    tok = ids_r[bp + _s]
                                    wt = w_r[bp + _s]
                                    ys = y8[_s : _s + 1, :, :].astype(jnp.float32) * wt
                                    acc[pl.ds(tok, 1), :, :] = acc[pl.ds(tok, 1), :, :] + ys
                        return _c

                    lax.fori_loop(0, 2, _step, jnp.int32(0))
                    out_r[...] = acc[pl.ds(0, b_gs), :, :]

                pl.run_scoped(
                    _body,
                    pltpu.VMEM((16, 32, 128), jnp.bfloat16),
                    pltpu.VMEM((b_gs + 1, 32, 128), jnp.float32),
                )

            vm = pl.BlockSpec(memory_space=pltpu.VMEM)
            sm = pl.BlockSpec(memory_space=pltpu.SMEM)
            return pl.pallas_call(
                _kern2,
                out_shape=jax.ShapeDtypeStruct((b_gs, 32, 128), jnp.float32),
                grid_spec=pltpu.PrefetchScalarGridSpec(
                    num_scalar_prefetch=0, grid=(1,),
                    in_specs=(vm, sm, sm, sm),
                    out_specs=vm,
                ),
                interpret=pltpu.InterpretParams(**interp_p),
            )(x_3d_in, jnp.asarray(p_ids, jnp.int32), jnp.asarray(p_w, jnp.float32), jnp.asarray(starts_np, jnp.int32))

        gs_ref = np.asarray(_run_cur1_gs_test(False))
        gs_cur1 = np.asarray(_run_cur1_gs_test(True))
        self.assertTrue(np.all(np.isfinite(gs_cur1)))
        self.assertTrue(np.array_equal(gs_ref, gs_cur1))

        # 4. tpu7x:2x2x1 AOT compilation for FAST_B64=4 across B=64 (R=12) and B=128 (R=8).
        ep, h, n1, i_dim, sb = 8, 4096, 8192, 4096, 512
        e_local = 256 // ep
        tpu_devs = get_topology_desc("tpu7x:2x2x1", platform="tpu").devices
        if len(tpu_devs) >= 8:
            tpu_mesh = Mesh(
                np.array(
                    sorted(tpu_devs, key=lambda d: (d.coords[1], d.coords[0], d.core_on_chip))
                ),
                ("model",),
            )
            for cur_b, cur_r in ((64, 12), (128, 8)):
                with mock.patch.dict(
                    os.environ,
                    {**clean_env, "SGLANG_TPU_MUSE_KERNEL_DECODE_FAST_B64": "4"},
                    clear=True,
                ):
                    cur_cfg = dataclasses.replace(
                        v1_mod.default_sparse_decode_config(cur_b, ep, sb),
                        pair_rs=True,
                        free_max_rows=10,
                        free_max_w=0.26,
                        free_min_slot=1,
                        min_p_topk=0.80,
                    )
                shapes = (
                    jax.ShapeDtypeStruct((1,), jnp.int32),
                    jax.ShapeDtypeStruct((cur_b, 256), jnp.bfloat16),
                    jax.ShapeDtypeStruct((1, 256), jnp.float32),
                    jax.ShapeDtypeStruct((cur_b, h), jnp.bfloat16),
                    jax.ShapeDtypeStruct((e_local, h, n1), jnp.float4_e2m1fn),
                    jax.ShapeDtypeStruct((e_local, h // sb, n1), jnp.bfloat16),
                    jax.ShapeDtypeStruct((e_local, i_dim, h), jnp.float4_e2m1fn),
                    jax.ShapeDtypeStruct((e_local, i_dim // sb, h), jnp.bfloat16),
                    jax.ShapeDtypeStruct((1, h), jnp.bfloat16),
                    jax.ShapeDtypeStruct((cur_r, h, n1), jnp.float4_e2m1fn),
                    jax.ShapeDtypeStruct((cur_r, h // sb, n1), jnp.bfloat16),
                    jax.ShapeDtypeStruct((cur_r, i_dim, h), jnp.float4_e2m1fn),
                    jax.ShapeDtypeStruct((cur_r, i_dim // sb, h), jnp.bfloat16),
                    jax.ShapeDtypeStruct((4, ep * cur_r), jnp.int32),
                )

                def _aot_w10(*args, _cfg=cur_cfg):
                    return v1_mod.fused_decode_sparse_from_logits(
                        *args[:9],
                        rep_w1=args[9],
                        rep_s1=args[10],
                        rep_w2=args[11],
                        rep_s2=args[12],
                        replica_meta=args[13],
                        ep=ep,
                        return_sharded=True,
                        config=_cfg,
                    )

                comp = (
                    jax.jit(
                        jax.shard_map(
                            _aot_w10,
                            mesh=tpu_mesh,
                            in_specs=(P(),) * len(shapes),
                            out_specs=P(),
                            check_vma=False,
                        )
                    )
                    .trace(*shapes)
                    .lower(lowering_platforms=("tpu",))
                    .compile()
                )
                self.assertIn("tpu_custom_call", comp.as_text())
                self.assertGreater(
                    comp.memory_analysis().generated_code_size_in_bytes, 0
                )


if __name__ == "__main__":
    unittest.main()
