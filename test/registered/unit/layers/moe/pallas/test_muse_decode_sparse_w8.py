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


if __name__ == "__main__":
    unittest.main()
