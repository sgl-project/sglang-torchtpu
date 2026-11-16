# SPDX-License-Identifier: Apache-2.0
"""Device-free checks of the experimental PCP ring kernel package's host side.

Covers sglang_tpu.attention.pallas.batched_rpa_pcp (used only by ring-mode
prefill context parallelism, SGLANG_TPU_PCP_MODE=ring): the ring
block-size calculator (ported from google3 wrapper_test.test_cp_block_sizes,
cl/985434096), the cache-phase tile chosen by the step-time model, the
whole-stripe new-token start-mask skip, the fused configs / host schedule check
on the Qwen3 serving shape, and a CPU lowering of the fused ring kernel under
shard_map. The production batched_rpa package keeps its own tests.
"""

import importlib.util
import unittest
from unittest import mock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=60, suite="base-a-test-cpu")

_HAS_JAX = importlib.util.find_spec("jax") is not None
_HAS_TORCH_TPU = importlib.util.find_spec("torch_tpu") is not None


def _mock_tpu_info(
    generation,
    chip_version,
    *,
    num_sublanes=8,
    mxu_column_size=256,
    vmem_capacity_bytes=64 * 1024 * 1024,
):
    info = mock.MagicMock()
    info.generation = generation
    info.chip_version = chip_version
    info.num_lanes = 128
    info.num_sublanes = num_sublanes
    info.mxu_column_size = mxu_column_size
    info.vmem_capacity_bytes = vmem_capacity_bytes
    info.smem_capacity_bytes = 1024 * 1024
    if generation == 8:
        info.bf16_ops_per_second = 1.1e15
        info.fp8_ops_per_second = 8.8e15
    elif generation == 7:
        info.bf16_ops_per_second = 2.3e15
        info.fp8_ops_per_second = 4.6e15
    else:
        info.bf16_ops_per_second = 1e15
        info.fp8_ops_per_second = 0
    return info


def _tpu7x():
    """Physical TPU7x metadata for Pallas on a CPU-only JAX."""
    from sglang_tpu import pallas_export

    stack = mock.patch.object(
        pallas_export, "get_pallas_tpu_device_kind", return_value="TPU7x"
    )
    return stack, pallas_export.tpu_info_for_cpu_export


class _Tpu7xCase(CustomTestCase):
    def setUp(self):
        super().setUp()
        patcher, export_ctx = _tpu7x()
        patcher.start()
        self.addCleanup(patcher.stop)
        ctx = export_ctx()
        ctx.__enter__()
        self.addCleanup(ctx.__exit__, None, None, None)


@unittest.skipUnless(_HAS_JAX, "requires JAX")
class TestPcpRingBlockSizes(CustomTestCase):
    """Tile choice and host schedule check of the fused ring kernel."""

    def test_ring_block_size_calculator_matches_google3_cp_case(self):
        """Ported from google3 ``wrapper_test.test_cp_block_sizes`` (cl/985434096):
        `RingBlockSizeCalculator` picks `(bq=512, bkv=1024, batch=1)` on a 64 MiB
        v6p-like chip for a PCP=4 cache-only ring pass."""
        import jax.numpy as jnp

        from sglang_tpu.attention.pallas.batched_rpa_pcp import configs, wrapper

        model_cfg = configs.ModelConfigs(
            num_q_heads=16, num_kv_heads=1, head_dim=256, mask_value=-1e9
        )
        serve_cfg = configs.ServingConfigs(
            num_seqs=3,
            page_size=128,
            total_q_tokens=1024,
            num_page_indices=1024,
            dtype_q=jnp.bfloat16,
            dtype_kv=jnp.float8_e4m3fn,
            dtype_out=jnp.bfloat16,
            cp_group_size=4,
            ring_axis_name="pcp",
            kv_layout=configs.KVLayout.SEQ_ALONG_LANE,
            attention_scope=configs.AttentionScope.CACHE_ONLY,
            return_lse=False,
        )
        with mock.patch(
            "jax.experimental.pallas.tpu.get_tpu_info",
            return_value=_mock_tpu_info(6, "v6p"),
        ):
            calc = wrapper.RingBlockSizeCalculator(model_cfg, serve_cfg, 64 << 20)
            prefill = calc.prefill_block_sizes()
        self.assertEqual(
            (prefill.bq_sz, prefill.bkv_sz, prefill.batch_size), (512, 1024, 1)
        )

    def test_ring_block_sizes_follow_the_step_cost_model_on_qwen3(self):
        """Qwen3-30B-A3B fp8 on v7x: PCP4 x TP2 (16 q / 2 KV heads) is
        compute-bound per ring step, so bq=512 (no padding of 2k/4k-row chunks)
        beats bq=768 (fewer steps, 12.5% padded rows); PCP8 (32 q / 4 KV heads)
        is hop-bound below bq=384, so it keeps bq=384 over the unpadded 256."""
        import jax.numpy as jnp

        from sglang_tpu.attention.pallas.batched_rpa_pcp import wrapper

        def cache_bq(nq, nkv, cp, total_q):
            fused = wrapper.pcp_fused_configs(
                num_q_heads=nq,
                num_kv_heads=nkv,
                head_dim=128,
                dtype_q=jnp.bfloat16,
                dtype_kv=jnp.float8_e4m3fn,
                page_size=256,
                cp_group_size=cp,
                total_q_tokens=total_q,
                num_cache_rows=8,
                cache_pages_per_row=-(-70000 // (256 * cp)),
                num_new_rows=16,
                new_pages_per_row=-(-70000 // (256 * cp)),
                k_scale=1.0,
                v_scale=1.0,
                vmem_limit_bytes=64 << 20,
            )
            return fused.cfgs_cache.bq_sz, fused.cfgs_cache.bkv_sz

        with mock.patch(
            "jax.experimental.pallas.tpu.get_tpu_info",
            return_value=_mock_tpu_info(7, "v7x"),
        ):
            for total_q in (1024, 2048, 4096):
                self.assertEqual(cache_bq(16, 2, 4, total_q), (512, 1024), total_q)
                self.assertEqual(cache_bq(32, 4, 8, total_q), (384, 1024), total_q)

    def test_whole_stripe_cache_lens_drop_the_new_token_start_mask(self):
        """With cached lengths in whole stripes (page_size * cp) the new-token
        phase's KV blocks never hold a cached key, so its start mask is skipped;
        the host check refuses a batch that breaks the promise."""
        import jax.numpy as jnp

        from sglang_tpu.attention.pallas.batched_rpa_pcp import wrapper

        common = dict(
            num_q_heads=16,
            num_kv_heads=2,
            head_dim=128,
            page_size=256,
            cp_group_size=4,
            total_q_tokens=1024,
        )
        with mock.patch(
            "jax.experimental.pallas.tpu.get_tpu_info",
            return_value=_mock_tpu_info(7, "v7x"),
        ):
            for aligned in (False, True):
                fused = wrapper.pcp_fused_configs(
                    **common,
                    dtype_q=jnp.bfloat16,
                    dtype_kv=jnp.float8_e4m3fn,
                    num_cache_rows=1,
                    cache_pages_per_row=8,
                    num_new_rows=2,
                    new_pages_per_row=8,
                    vmem_limit_bytes=64 << 20,
                    cache_lens_cp_page_aligned=aligned,
                )
                self.assertEqual(
                    fused.cfgs_new.serve.cached_lens_block_aligned, aligned
                )
                self.assertFalse(fused.cfgs_cache.serve.cached_lens_block_aligned)
            # One request, 1024 cached + 1024 new tokens (C = 128 per block).
            batch = dict(
                **common,
                dtype_q="bfloat16",
                dtype_kv="float8_e4m3fn",
                total_kv_tokens=4 * 1024,
                cache_cu_q_lens=[0, 256],
                cache_kv_lens=[1024 + 256],
                cache_pages_per_row=8,
                new_cu_q_lens=[0, 128, 256],
                new_kv_lens=[2048, 2048],
                kv_cache_lens=[1024, 1024],
                q_positions=[1024, 1024 + 7 * 128],
                new_pages_per_row=8,
                new_kv_pages_per_row=16,
                num_cache_rows_live=1,
                num_new_rows_live=2,
                vmem_limit_bytes=64 << 20,
                cache_lens_cp_page_aligned=True,
            )
            wrapper.check_pcp_fused_schedule(**batch)
            batch.update(kv_cache_lens=[768, 768], q_positions=[768, 768 + 896])
            with self.assertRaisesRegex(ValueError, "whole stripes of 1024"):
                wrapper.check_pcp_fused_schedule(**batch)

    def test_pcp_fused_configs_and_schedule_check_on_qwen3_serving_shape(self):
        """`pcp_fused_configs` and `check_pcp_fused_schedule` agree on Qwen3-30B-A3B
        PCP=8 (32 q heads, 4 KV heads, hd 128, fp8 KV, page 256) and reject a
        schedule overflow."""
        import jax.numpy as jnp

        from sglang_tpu.attention.pallas.batched_rpa_pcp import wrapper

        with mock.patch(
            "jax.experimental.pallas.tpu.get_tpu_info",
            return_value=_mock_tpu_info(7, "v7x"),
        ):
            fused = wrapper.pcp_fused_configs(
                num_q_heads=32,
                num_kv_heads=4,
                head_dim=128,
                dtype_q=jnp.bfloat16,
                dtype_kv=jnp.float8_e4m3fn,
                page_size=256,
                cp_group_size=8,
                total_q_tokens=2048,
                num_cache_rows=8,
                cache_pages_per_row=20,
                num_new_rows=16,
                new_pages_per_row=20,
                k_scale=0.5,
                v_scale=0.25,
                vmem_limit_bytes=64 << 20,
            )
            self.assertEqual(fused.cfgs_cache.bq_sz, fused.cfgs_new.bq_sz)
            self.assertEqual(fused.cfgs_cache.batch_size, 1)
            self.assertEqual(fused.cfgs_new.batch_size, 1)
            self.assertLessEqual(fused.cfgs_new.bkv_sz, fused.cfgs_cache.bkv_sz)
            self.assertEqual(fused.cfgs_cache.bkv_sz, 1024)
            self.assertEqual(fused.cfgs_new.bkv_sz, 512)

            # One live request: 4096 cached tokens + 4096 new tokens (C = 256).
            common = dict(
                num_q_heads=32,
                num_kv_heads=4,
                head_dim=128,
                dtype_q="bfloat16",
                dtype_kv="float8_e4m3fn",
                page_size=256,
                cp_group_size=8,
                total_q_tokens=2048,
                total_kv_tokens=8 * 2048,
                cache_cu_q_lens=[0, 512] + [512] * 7,
                cache_kv_lens=[4096 + 512] + [0] * 7,
                cache_pages_per_row=20,
                new_cu_q_lens=[0, 256, 512] + [512] * 14,
                new_kv_lens=[8192, 8192] + [0] * 14,
                kv_cache_lens=[4096, 4096] + [0] * 14,
                q_positions=[4096, 4096 + 15 * 256] + [0] * 14,
                new_pages_per_row=20,
                new_kv_pages_per_row=64,
                num_cache_rows_live=1,
                num_new_rows_live=2,
                vmem_limit_bytes=64 << 20,
            )
            wrapper.check_pcp_fused_schedule(**common)

            # An oversized cached prefix overflows the ring schedule capacity.
            overflow = dict(common, cache_kv_lens=[1 << 26] + [0] * 7)
            with self.assertRaisesRegex(ValueError, "cache \\(ring\\) phase needs"):
                wrapper.check_pcp_fused_schedule(**overflow)


@unittest.skipUnless(_HAS_JAX and _HAS_TORCH_TPU, "requires JAX and TorchTPU")
class TestPcpFusedLowersOnCpu(_Tpu7xCase):
    """Lower the fused ring kernel through the CPU export path (no Mosaic compile)."""

    def test_pcp_fused_ring_lowers_under_shard_map(self):
        """Lower `ragged_paged_attention_pcp_fused` (fp8 KV, read-only and
        in-kernel writeback) under `shard_map` over the available CPU devices."""
        import jax
        import jax.numpy as jnp
        import numpy as np
        from jax.sharding import Mesh
        from jax.sharding import PartitionSpec as P

        from sglang_tpu.attention.pallas.batched_rpa_pcp import wrapper

        devs = jax.devices()
        if len(devs) < 2:
            return
        pcp = 2 if len(devs) < 8 else 8
        mesh = Mesh(np.array(devs[:pcp]), ("model",))
        page, t_local, nq, nkv, hd = 256, 512, 32, 4, 128
        s_c, s_n, cpp, npp, vpp, num_pages = 4, 8, 8, 8, pcp * 2, 32
        fp8, bf16, i32 = jnp.float8_e4m3fn, jnp.bfloat16, jnp.int32
        cache_shape = wrapper.get_kv_cache_shape(num_pages, page, nkv, hd, fp8)
        row = P("model")

        for update_kv_cache in (False, True):

            def per_rank(
                q,
                k,
                v,
                cache,
                ckl,
                cpi,
                ccu,
                cdist,
                nkl,
                npi,
                ncu,
                ndist,
                nkpi,
                qpos,
                kcl,
                wr,
                *,
                _upd=update_kv_cache,
            ):
                cp_rank = jnp.reshape(jax.lax.axis_index("model"), (1,)).astype(i32)
                k_all = jax.lax.all_gather(k, "model", axis=0, tiled=True)
                v_all = jax.lax.all_gather(v, "model", axis=0, tiled=True)
                res = wrapper.ragged_paged_attention_pcp_fused(
                    q,
                    k_all,
                    v_all,
                    cache,
                    cache_kv_lens=ckl,
                    cache_page_indices=cpi,
                    cache_cu_q_lens=ccu,
                    cache_distribution=cdist,
                    new_kv_lens=nkl,
                    new_page_indices=npi,
                    new_cu_q_lens=ncu,
                    new_distribution=ndist,
                    new_kv_page_indices=nkpi,
                    q_positions=qpos,
                    kv_cache_lens=kcl,
                    cp_rank=cp_rank,
                    new_kv_write_rows=wr if _upd else None,
                    cp_group_size=pcp,
                    ring_axis_name="model",
                    ring_mesh_axis_names=("model",),
                    k_scale=0.5,
                    v_scale=0.25,
                    update_kv_cache=_upd,
                )
                return res if _upd else (res,)

            in_specs = (
                (row, row, row, row) + (P(),) * 6 + (row, P(), P(), row, P(), P())
            )
            out_specs = (row, row) if update_kv_cache else (row,)
            fn = jax.shard_map(
                per_rank,
                mesh=mesh,
                in_specs=in_specs,
                out_specs=out_specs,
                check_vma=False,
            )
            sds = jax.ShapeDtypeStruct
            args = (
                sds((pcp * t_local, nq, hd), bf16),
                sds((pcp * t_local, nkv, hd), fp8),
                sds((pcp * t_local, nkv, hd), fp8),
                sds((pcp * cache_shape[0],) + tuple(cache_shape[1:]), fp8),
                sds((s_c,), i32),
                sds((s_c * cpp,), i32),
                sds((s_c + 1,), i32),
                sds((3,), i32),
                sds((s_n,), i32),
                sds((s_n * npp,), i32),
                sds((pcp * (s_n + 1),), i32),
                sds((3,), i32),
                sds((s_n * vpp,), i32),
                sds((pcp * s_n,), i32),
                sds((s_n,), i32),
                sds((s_n,), i32),
            )
            exported = jax.export.export(
                jax.jit(fn),
                platforms=["tpu"],
                disabled_checks=[jax.export.DisabledSafetyCheck.custom_call("ALL")],
            )(*args)
            self.assertEqual(
                (tuple(exported.out_avals[0].shape), str(exported.out_avals[0].dtype)),
                ((pcp * t_local, nq, hd), "bfloat16"),
            )
            self.assertEqual(len(exported.out_avals), 2 if update_kv_cache else 1)


if __name__ == "__main__":
    unittest.main()
