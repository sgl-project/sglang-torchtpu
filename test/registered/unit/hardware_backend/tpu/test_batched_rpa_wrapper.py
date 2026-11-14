# SPDX-License-Identifier: Apache-2.0
"""Device-free checks of the batched RPA kernel's host-side contract.

The block-size cases are ported from google3's ``unit_test/wrapper_test.py``
(the kernel is synced to google3; see ``batched_rpa/__init__.py``): they run
the tile search against mocked chip metadata. The rest pins what SGLang layers
on top of google3's kernel and what its KV pools rely on: the fixed pipeline
slot counts, the batch order of ``distribution``, the cache shape (the
lane-folded fp8 layout exists but is off), and the decode-context-parallel
schedulers' host-side length math. Two small kernels are also lowered through
the CPU export path, so a Pallas API drift shows up without a TPU.
"""

import importlib.util
import unittest
from unittest import mock

import numpy as np
import torch

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
class TestBlockSizeSearch(CustomTestCase):
    """Ported from google3 ``wrapper_test.ConfigsBlockSizesTest``."""

    def _block_sizes(self, tpu_info, *, dtype, decode_query_size=1, **model):
        import jax.numpy as jnp

        from sglang_tpu.attention.pallas.batched_rpa import configs, wrapper

        model_cfg = configs.ModelConfigs(mask_value=-1e9, **model)
        serve_cfg = configs.ServingConfigs(
            num_seqs=8,
            page_size=256,
            total_q_tokens=8 * decode_query_size,
            num_page_indices=8 * 16,
            dtype_q=dtype,
            dtype_kv=dtype,
            dtype_out=jnp.bfloat16,
            decode_query_size=decode_query_size,
        )
        with mock.patch(
            "jax.experimental.pallas.tpu.get_tpu_info", return_value=tpu_info
        ):
            return wrapper.calculate_block_sizes(
                model_cfg,
                serve_cfg,
                tpu_info.vmem_capacity_bytes,
                decode_query_size=decode_query_size,
            )

    def test_block_sizes_on_every_chip_generation(self):
        import jax.numpy as jnp
        from jax.experimental.pallas import tpu as pltpu

        chips = (
            (8, pltpu.ChipVersion.TPU_8I),
            (8, "v8i"),
            (8, "8i"),
            (8, "v8t"),
            (7, "v7x"),
            (5, "v5p"),
        )
        for generation, chip_version in chips:
            for dtype in (jnp.float8_e4m3fn, jnp.bfloat16):
                label = f"gen={generation} chip={chip_version} {dtype.__name__}"
                with self.subTest(label):
                    decode, prefill = self._block_sizes(
                        _mock_tpu_info(generation, chip_version),
                        dtype=dtype,
                        num_q_heads=32,
                        num_kv_heads=8,
                        head_dim=128,
                    )
                    self.assertEqual(decode.bq_sz, decode.bq_c_sz)
                    self.assertGreaterEqual(decode.bq_c_sz, 1)
                    self.assertGreaterEqual(prefill.bq_sz, prefill.bq_c_sz)
                    self.assertGreaterEqual(prefill.bq_c_sz, 1)
                    self.assertEqual(prefill.bq_sz % prefill.bq_c_sz, 0)
                    for blocks in (decode, prefill):
                        self.assertEqual(blocks.bkv_sz % 256, 0)
                        self.assertGreaterEqual(blocks.batch_size, 1)

    def test_decode_compute_chunk_always_equals_the_q_block(self):
        """In decode, bq_c_sz is never downsized below bq_sz, whatever the
        decode query width (Qwen3.5 at TP=2: 16 query heads per core)."""
        import jax.numpy as jnp

        for decode_query_size in (1, 2, 4, 8):
            with self.subTest(decode_query_size=decode_query_size):
                decode, _ = self._block_sizes(
                    _mock_tpu_info(7, "v7x"),
                    dtype=jnp.float8_e4m3fn,
                    decode_query_size=decode_query_size,
                    num_q_heads=16,
                    num_kv_heads=2,
                    head_dim=256,
                )
                self.assertEqual(decode.bq_sz, decode_query_size)
                self.assertEqual(decode.bq_c_sz, decode_query_size)


@unittest.skipUnless(_HAS_JAX and _HAS_TORCH_TPU, "requires JAX and TorchTPU")
class TestFixedPipelineSlots(_Tpu7xCase):
    """Three input and two output buffers for every kernel; a shape that does
    not fit them gets a smaller q block, not fewer buffers."""

    def _prefill(self, *, layout, dtype_kv, return_lse=False, sink=False, **model):
        import jax.numpy as jnp

        from sglang_tpu.attention.pallas.batched_rpa import configs, wrapper

        model_cfg = configs.ModelConfigs(
            mask_value=float("-inf"), has_attention_sink=sink, **model
        )
        serve_cfg = configs.ServingConfigs(
            num_seqs=32,
            num_page_indices=32 * 17,
            total_q_tokens=4096,
            dtype_q=jnp.bfloat16,
            dtype_kv=dtype_kv,
            dtype_out=jnp.bfloat16,
            page_size=256,
            kv_layout=layout,
            return_lse=return_lse,
        )
        return wrapper.calculate_block_sizes(model_cfg, serve_cfg, 64 << 20)

    def test_slot_counts_are_constants(self):
        import dataclasses

        from sglang_tpu.attention.pallas.batched_rpa import configs

        self.assertEqual(configs.NUM_INPUT_SLOTS, 3)
        self.assertEqual(configs.NUM_OUTPUT_SLOTS, 2)
        fields = {f.name for f in dataclasses.fields(configs.BlockSizes)}
        self.assertEqual(fields, {"bq_sz", "bq_c_sz", "bkv_sz", "batch_size"})

    def test_serving_shapes_keep_their_tiles(self):
        """Shapes that already ran with three input buffers are unchanged."""
        import jax.numpy as jnp

        from sglang_tpu.attention.pallas.batched_rpa import configs

        has = configs.KVLayout.HEAD_ALONG_SUBLANE
        fp8, bf16 = jnp.float8_e4m3fn, jnp.bfloat16
        cases = (
            # Qwen3 at TP8.
            (dict(num_q_heads=4, num_kv_heads=1, head_dim=128), fp8, {}, 2048, 2048),
            # gpt-oss DCP2 / DCP8 prefix pass (the group's gathered q heads).
            (
                dict(num_q_heads=16, num_kv_heads=2, head_dim=64, sliding_window=128),
                bf16,
                dict(return_lse=True),
                768,
                768,
            ),
            (
                dict(num_q_heads=64, num_kv_heads=8, head_dim=64),
                bf16,
                dict(return_lse=True),
                256,
                256,
            ),
            # 128 query heads on one chip (PP8 with TP=1).
            (dict(num_q_heads=128, num_kv_heads=16, head_dim=64), fp8, {}, 64, 512),
        )
        for model, dtype_kv, flags, bq_sz, bkv_sz in cases:
            with self.subTest(model=model):
                _, prefill = self._prefill(
                    layout=has, dtype_kv=dtype_kv, **flags, **model
                )
                self.assertEqual((prefill.bq_sz, prefill.bkv_sz), (bq_sz, bkv_sz))
                self.assertEqual(prefill.batch_size, 1)

    def test_shapes_that_needed_fewer_buffers_get_a_smaller_q_block(self):
        """These prefill shapes ran with one or two input buffers at a 256-token
        q block; with three buffers the q block is halved instead."""
        import jax.numpy as jnp

        from sglang_tpu.attention.pallas.batched_rpa import configs

        has = configs.KVLayout.HEAD_ALONG_SUBLANE
        sal = configs.KVLayout.SEQ_ALONG_LANE
        fp8, bf16 = jnp.float8_e4m3fn, jnp.bfloat16
        cases = (
            # gpt-oss DCP8 prefix pass on the lane-major cache.
            (sal, bf16, dict(return_lse=True), 64, 8, 64, (128, 16, 512)),
            (has, fp8, dict(return_lse=True, sink=True), 64, 16, 64, (128, 32, 256)),
            (has, fp8, dict(return_lse=True), 96, 8, 64, (128, 8, 512)),
            (has, bf16, dict(return_lse=True), 40, 8, 256, (128, 32, 256)),
        )
        for layout, dtype_kv, flags, nq, nkv, head_dim, want in cases:
            label = f"{nq}q/{nkv}kv hd{head_dim} {layout.name} {flags}"
            with self.subTest(label):
                _, prefill = self._prefill(
                    layout=layout,
                    dtype_kv=dtype_kv,
                    num_q_heads=nq,
                    num_kv_heads=nkv,
                    head_dim=head_dim,
                    **flags,
                )
                self.assertEqual((prefill.bq_sz, prefill.bq_c_sz, prefill.bkv_sz), want)
                self.assertEqual(prefill.batch_size, 1)


@unittest.skipUnless(_HAS_JAX and _HAS_TORCH_TPU, "requires JAX and TorchTPU")
class TestCacheShape(_Tpu7xCase):
    """The kernel's default cache shape is the KV pool's buffer shape."""

    # (num_kv_heads, head_dim): head_dim > 128 is where google3 folds fp8 heads.
    SHAPES = ((1, 256), (2, 256), (4, 256), (8, 256), (1, 192), (2, 384), (1, 128))

    def test_default_shape_is_the_unfolded_token_major_layout(self):
        import jax.numpy as jnp

        from sglang_tpu.attention.pallas.batched_rpa import configs, wrapper

        for num_kv_heads, head_dim in self.SHAPES:
            for dtype, packing in ((jnp.float8_e4m3fn, 4), (jnp.bfloat16, 2)):
                with self.subTest(kv=num_kv_heads, head_dim=head_dim, packing=packing):
                    shape = wrapper.get_kv_cache_shape(
                        7, 256, num_kv_heads, head_dim, dtype
                    )
                    heads_x2 = -(-num_kv_heads * 2 // packing) * packing
                    self.assertEqual(
                        shape,
                        (
                            7,
                            256,
                            heads_x2 // packing,
                            packing,
                            -(-head_dim // 128) * 128,
                        ),
                    )
                    self.assertEqual(
                        configs.num_lane_folds_in_head_dim(
                            head_dim, dtype, configs.KVLayout.HEAD_ALONG_SUBLANE, 128
                        ),
                        1,
                    )

    def test_default_shape_matches_the_fused_pool(self):
        import jax.numpy as jnp

        from sglang_tpu.attention.pallas.batched_rpa import wrapper
        from sglang_tpu.kv_pool import TpuFusedMHATokenToKVPool

        page_size = 16
        for num_kv_heads, head_dim in self.SHAPES:
            for torch_dtype, jax_dtype in (
                (torch.float8_e4m3fn, jnp.float8_e4m3fn),
                (torch.bfloat16, jnp.bfloat16),
            ):
                with self.subTest(
                    kv=num_kv_heads, head_dim=head_dim, dtype=torch_dtype
                ):
                    pool = TpuFusedMHATokenToKVPool(
                        size=4 * page_size,
                        page_size=page_size,
                        dtype=torch_dtype,
                        head_num=num_kv_heads,
                        head_dim=head_dim,
                        layer_num=1,
                        device="cpu",
                        enable_memory_saver=False,
                    )
                    self.assertEqual(
                        tuple(pool.kv_buffer[0].shape),
                        wrapper.get_kv_cache_shape(
                            pool.num_blocks,
                            page_size,
                            num_kv_heads,
                            head_dim,
                            jax_dtype,
                        ),
                    )

    def test_lane_fold_is_opt_in_and_round_trips(self):
        """google3's lane-folded fp8 layout is reachable only on request."""
        import jax.numpy as jnp

        from sglang_tpu.attention.pallas.batched_rpa import wrapper

        fp8 = jnp.float8_e4m3fn
        unfolded = wrapper.get_kv_cache_shape(3, 256, 1, 256, fp8)
        folded = wrapper.get_kv_cache_shape(
            3, 256, 1, 256, fp8, lane_fold_kv_head_dim=True
        )
        self.assertEqual(unfolded, (3, 256, 1, 4, 256))
        self.assertEqual(folded, (3, 256, 1, 4, 128))
        # bf16 and head_dim <= 128 never fold.
        for dtype, head_dim in ((jnp.bfloat16, 256), (fp8, 128)):
            self.assertEqual(
                wrapper.get_kv_cache_shape(
                    3, 256, 2, head_dim, dtype, lane_fold_kv_head_dim=True
                ),
                wrapper.get_kv_cache_shape(3, 256, 2, head_dim, dtype),
            )

        rng = np.random.default_rng(0)
        cache = np.zeros(wrapper.get_kv_cache_shape(3, 16, 2, 256, fp8), np.float32)
        cache.reshape(3, 16, -1, 256)[:, :, :4] = rng.integers(
            -8, 8, (3, 16, 4, 256)
        ).astype(np.float32)
        cache = jnp.asarray(cache, dtype=fp8)
        as_folded = wrapper.fold_kv_cache_head_dim(cache, 2, 256)
        self.assertEqual(
            as_folded.shape,
            wrapper.get_kv_cache_shape(3, 16, 2, 256, fp8, lane_fold_kv_head_dim=True),
        )
        back = wrapper.unfold_kv_cache_head_dim(as_folded, 2, 256)
        np.testing.assert_array_equal(
            np.asarray(back.astype(jnp.float32)), np.asarray(cache.astype(jnp.float32))
        )


@unittest.skipUnless(_HAS_JAX and _HAS_TORCH_TPU, "requires JAX and TorchTPU")
class TestSchedulerContract(_Tpu7xCase):
    """What the public arguments turn into for the kernels."""

    def _cfgs(self, mode):
        import jax.numpy as jnp

        from sglang_tpu.attention.pallas.batched_rpa import configs

        return configs.RpaConfigs(
            block=configs.BlockSizes(bq_sz=1, bq_c_sz=1, bkv_sz=256, batch_size=1),
            model=configs.ModelConfigs(
                num_q_heads=8, num_kv_heads=1, head_dim=64, mask_value=0.0
            ),
            serve=configs.ServingConfigs(
                num_seqs=8,
                page_size=256,
                total_q_tokens=8,
                num_page_indices=16,
                dtype_q=jnp.bfloat16,
                dtype_kv=jnp.bfloat16,
                dtype_out=jnp.bfloat16,
            ),
            mode=mode,
            vmem_limit_bytes=64 << 20,
        )

    def test_distribution_lists_mixed_rows_first_and_decode_rows_last(self):
        from sglang_tpu.attention.pallas.batched_rpa import configs

        distribution = np.asarray([3, 5, 9], np.int32)
        self.assertEqual(configs.RpaCase.MIXED.get_range(distribution), (0, 3))
        self.assertEqual(configs.RpaCase.PREFILL.get_range(distribution), (3, 5))
        self.assertEqual(configs.RpaCase.DECODE.get_range(distribution), (5, 9))
        # The MIXED launch also runs the prefill-only rows.
        self.assertEqual(
            self._cfgs(configs.RpaCase.MIXED).seq_range(distribution), (0, 5)
        )
        self.assertEqual(
            self._cfgs(configs.RpaCase.DECODE).seq_range(distribution), (5, 9)
        )

    def test_scope_selects_the_scheduler(self):
        from sglang_tpu.attention.pallas.batched_rpa import configs
        from sglang_tpu.attention.pallas.batched_rpa import scheduler as scheduler_lib
        from sglang_tpu.attention.pallas.batched_rpa import scheduler_cp, wrapper

        scope = configs.AttentionScope
        plain = {
            scope.FULL: scheduler_lib.BaseScheduler,
            scope.CACHE_ONLY: scheduler_lib.CacheOnlyScheduler,
            scope.NEW_TOKENS_ONLY: scheduler_lib.NewTokensOnlyScheduler,
        }
        striped = {
            scope.CACHE_ONLY: scheduler_cp.StripedCacheOnlyScheduler,
            scope.NEW_TOKENS_ONLY: scheduler_cp.StripedChunkScheduler,
            scope.STRIPED: scheduler_cp.StripedDecodeScheduler,
        }
        for attention_scope, cls in plain.items():
            scheduler = wrapper._scheduler_for_scope(attention_scope, None)
            self.assertIs(type(scheduler), cls)
            self.assertEqual(scheduler.scope, attention_scope)
            self.assertIsNone(scheduler.get_cp_group_size())
        for attention_scope, cls in striped.items():
            scheduler = wrapper._scheduler_for_scope(attention_scope, 4)
            self.assertIs(type(scheduler), cls)
            self.assertEqual(scheduler.scope, attention_scope)
            self.assertEqual(scheduler.get_cp_group_size(), 4)

    def test_plain_schedulers_split_the_lengths(self):
        import jax.numpy as jnp

        from sglang_tpu.attention.pallas.batched_rpa import scheduler as scheduler_lib

        kv_lens = jnp.asarray([10, 300, 0, 7], jnp.int32)
        q_lens = jnp.asarray([1, 100, 0, 7], jnp.int32)

        cache, new, offsets = scheduler_lib.BaseScheduler().get_kv_lens(kv_lens, q_lens)
        np.testing.assert_array_equal(cache, [9, 200, 0, 0])
        np.testing.assert_array_equal(new, q_lens)
        np.testing.assert_array_equal(offsets, [9, 200, 0, 0])

        # Cache-only: every key is cached and every query follows every key.
        cache, new, offsets = scheduler_lib.CacheOnlyScheduler().get_kv_lens(
            kv_lens, q_lens
        )
        np.testing.assert_array_equal(cache, kv_lens)
        np.testing.assert_array_equal(new, [0, 0, 0, 0])
        np.testing.assert_array_equal(offsets, kv_lens)

    def test_striped_decode_stitches_the_new_token_on_its_owner_only(self):
        """The owner of global page (n - 1) // page gets one new key; the first
        query's global position reaches the kernels as the query offset."""
        import jax.numpy as jnp

        from sglang_tpu.attention.pallas.batched_rpa import scheduler_cp, utils

        page, cp = 256, 4
        # 256 / 257 put the new token on the last slot of a page and on the
        # first slot of the next; 1 is a fresh sequence; the last row is a
        # multi-token (extend) row, which attends its stripe only.
        global_lens = np.asarray([2600, 1, 173, 256, 257, 1024, 1025, 900], np.int32)
        q_lens = np.asarray([1, 1, 1, 1, 1, 1, 1, 300], np.int32)
        scheduler = scheduler_cp.StripedDecodeScheduler(cp_group_size=cp)
        owners = 0
        for rank in range(cp):
            owns = (((global_lens - 1) // page) % cp == rank) & (q_lens == 1)
            cached_global = global_lens - q_lens
            local_cache = np.asarray(
                utils.cp_local_cache_len(jnp.asarray(cached_global), cp, rank, page)
            )
            kv_lens = local_cache + owns
            cache, new, offsets = scheduler.get_kv_lens(
                jnp.asarray(kv_lens, jnp.int32),
                jnp.asarray(q_lens),
                page,
                extra_scalars=(jnp.asarray([rank], jnp.int32),),
                kv_lens_global=jnp.asarray(global_lens),
            )
            np.testing.assert_array_equal(new, owns.astype(np.int32), f"rank {rank}")
            np.testing.assert_array_equal(cache, local_cache, f"rank {rank}")
            np.testing.assert_array_equal(offsets, global_lens - q_lens)
            owners += owns
        np.testing.assert_array_equal(owners, (q_lens == 1).astype(np.int32))

        # The cache-only stripe walk never takes a key from keys/values.
        cache_only = scheduler_cp.StripedCacheOnlyScheduler(cp_group_size=cp)
        cache, new, _ = cache_only.get_kv_lens(
            jnp.asarray(global_lens // cp, jnp.int32),
            jnp.asarray(q_lens),
            page,
            extra_scalars=(jnp.asarray([0], jnp.int32),),
            kv_lens_global=jnp.asarray(global_lens),
        )
        np.testing.assert_array_equal(new, np.zeros_like(q_lens))
        np.testing.assert_array_equal(cache, global_lens // cp)

    def test_context_parallel_accepts_the_window_and_the_sink(self):
        """google3 refuses both under context parallelism; SGLang's DCP runs
        gpt-oss, which has both."""
        import dataclasses

        import jax.numpy as jnp

        from sglang_tpu.attention.pallas.batched_rpa import configs, wrapper

        cfgs = self._cfgs(configs.RpaCase.DECODE)
        cfgs = dataclasses.replace(
            cfgs,
            model=dataclasses.replace(
                cfgs.model, sliding_window=128, has_attention_sink=True
            ),
            serve=dataclasses.replace(
                cfgs.serve,
                cp_group_size=2,
                attention_scope=configs.AttentionScope.STRIPED,
                return_lse=True,
            ),
        )
        inputs = dict(
            q=jnp.zeros((8, 8, 64), jnp.bfloat16),
            k=jnp.zeros((8, 1, 64), jnp.bfloat16),
            v=jnp.zeros((8, 1, 64), jnp.bfloat16),
            kv_cache=jnp.zeros(
                wrapper.get_kv_cache_shape(16, 256, 1, 64, jnp.bfloat16), jnp.bfloat16
            ),
            kv_lens=jnp.zeros((8,), jnp.int32),
            page_indices=jnp.zeros((16,), jnp.int32),
            cu_q_lens=jnp.arange(9, dtype=jnp.int32),
            distribution=jnp.asarray([0, 0, 8], jnp.int32),
            attention_sink=jnp.zeros((8,), jnp.float32),
        )
        cp_operands = dict(
            cp_rank=jnp.zeros((1,), jnp.int32),
            kv_lens_global=jnp.zeros((8,), jnp.int32),
        )
        cfgs.validate_inputs(**inputs, **cp_operands)
        with self.assertRaises(ValueError):
            cfgs.validate_inputs(**inputs)  # the CP operands are required
        full = dataclasses.replace(
            cfgs,
            serve=dataclasses.replace(
                cfgs.serve, attention_scope=configs.AttentionScope.FULL
            ),
        )
        with self.assertRaises(ValueError):
            full.validate_inputs(**inputs, **cp_operands)


@unittest.skipUnless(_HAS_JAX and _HAS_TORCH_TPU, "requires JAX and TorchTPU")
class TestKernelLowersOnCpu(_Tpu7xCase):
    """Lower the kernels through the CPU export path (no Mosaic compile)."""

    def _export(self, *, num_q_heads, num_kv_heads, cp=None, sink=False, **static):
        import jax
        import jax.numpy as jnp

        from sglang_tpu.attention.pallas.batched_rpa import wrapper

        num_seqs, pages_per_seq, page_size, head_dim, tokens = 8, 4, 256, 64, 512
        num_pages = num_seqs * pages_per_seq
        if cp is not None:
            static["cp_group_size"] = cp

        def attention(q, k, v, cache, kv_lens, pages, cu_q_lens, dist, *extra):
            extra = list(extra)
            operands = {}
            if sink:
                operands["attention_sink"] = extra.pop(0)
            if cp is not None:
                operands["cp_rank"], operands["kv_lens_global"] = extra
            return wrapper.ragged_paged_attention(
                q, k, v, cache, kv_lens, pages, cu_q_lens, dist, **operands, **static
            )

        bf16 = jnp.bfloat16
        args = [
            jax.ShapeDtypeStruct((tokens, num_q_heads, head_dim), bf16),
            jax.ShapeDtypeStruct((tokens, num_kv_heads, head_dim), bf16),
            jax.ShapeDtypeStruct((tokens, num_kv_heads, head_dim), bf16),
            jax.ShapeDtypeStruct(
                wrapper.get_kv_cache_shape(
                    num_pages, page_size, num_kv_heads, head_dim, bf16
                ),
                bf16,
            ),
            jax.ShapeDtypeStruct((num_seqs,), jnp.int32),
            jax.ShapeDtypeStruct((num_pages,), jnp.int32),
            jax.ShapeDtypeStruct((num_seqs + 1,), jnp.int32),
            jax.ShapeDtypeStruct((3,), jnp.int32),
        ]
        if sink:
            args.append(jax.ShapeDtypeStruct((num_q_heads,), jnp.float32))
        if cp is not None:
            args.append(jax.ShapeDtypeStruct((1,), jnp.int32))
            args.append(jax.ShapeDtypeStruct((num_seqs,), jnp.int32))
        exported = jax.export.export(
            jax.jit(attention),
            platforms=["tpu"],
            disabled_checks=[jax.export.DisabledSafetyCheck.custom_call("ALL")],
        )(*args)
        return [(tuple(a.shape), str(a.dtype)) for a in exported.out_avals]

    def test_plain_decode_and_extend_with_sink_and_window(self):
        outs = self._export(
            num_q_heads=8, num_kv_heads=1, sink=True, sliding_window=128
        )
        self.assertEqual(
            outs,
            [((512, 8, 64), "bfloat16"), ((32, 256, 1, 2, 128), "bfloat16")],
        )

    def test_striped_decode_returns_a_float32_lse_per_query_head(self):
        outs = self._export(
            num_q_heads=16,
            num_kv_heads=2,
            cp=2,
            attention_scope=3,  # STRIPED
            return_lse=True,
            sliding_window=128,
        )
        self.assertEqual(
            outs,
            [
                ((512, 16, 64), "bfloat16"),
                ((32, 256, 2, 2, 128), "bfloat16"),
                ((512, 16), "float32"),
            ],
        )


if __name__ == "__main__":
    unittest.main()
