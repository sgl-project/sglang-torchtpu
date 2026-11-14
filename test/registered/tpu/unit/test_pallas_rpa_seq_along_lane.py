# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the SEQ_ALONG_LANE KV layout on the batched-RPA backend.

The kernel reads either KV layout; these cover the sglang-side wiring that has
to agree with whichever one is chosen -- the page-size floor, the fp8 KV
default, and the decode and prefill tiles the wrapper derives for either layout.
CPU-runnable via the Pallas export shim.
"""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=5, suite="stage-a-test-tpu-unit", chips=1)

import itertools
import types
import unittest
from unittest.mock import patch

import torch

from sglang.srt.runtime_context import get_context
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.attention_pallas_rpa_batched import (
    RPA_BATCHED_SPEC,
    kv_layout,
)
from sglang_tpu.envs import envs


class TestKvLayoutSelection(CustomTestCase):
    """One flag picks the layout, and the KV pool has to follow it."""

    def test_defaults_to_token_major(self):
        from sglang_tpu.attention.pallas.batched_rpa import configs

        self.assertEqual(kv_layout(), configs.KVLayout.HEAD_ALONG_SUBLANE)

    def test_flag_selects_seq_along_lane(self):
        from sglang_tpu.attention.pallas.batched_rpa import configs

        with envs.SGLANG_TPU_USE_SEQ_ALONG_LANE_KV.override(True):
            self.assertEqual(kv_layout(), configs.KVLayout.SEQ_ALONG_LANE)

    def test_pool_class_follows_the_flag(self):
        from sglang_tpu.kv_pool import (
            TpuFusedMHATokenToKVPool,
            TpuSeqAlongLaneKVPool,
        )
        from sglang_tpu.tpu_platform import TpuSRTPlatform

        # get_mha_kv_pool_cls reads the exec bag; publish the backend.
        with get_context().override_server_args(
            attention_backend="pallas_tpu_rpa_batched"
        ):
            self.assertIs(
                TpuSRTPlatform().get_mha_kv_pool_cls(), TpuFusedMHATokenToKVPool
            )
            with envs.SGLANG_TPU_USE_SEQ_ALONG_LANE_KV.override(True):
                self.assertIs(
                    TpuSRTPlatform().get_mha_kv_pool_cls(), TpuSeqAlongLaneKVPool
                )


class TestPoolBytesPerToken(CustomTestCase):
    """The KV budget is spent before the pool exists, at this layout's price."""

    def test_head_dim_rounds_to_a_sublane_tile_only(self):
        """SEQ_ALONG_LANE rounds head_dim to num_sublanes * packing, not to 128,
        and never rounds the K/V head axis, so the fused formula over-counts it."""
        from sglang_tpu import pallas_export
        from sglang_tpu.kv_pool import TpuSeqAlongLaneKVPool

        page_size = 16
        with patch.object(
            pallas_export, "get_pallas_tpu_device_kind", return_value="TPU7x"
        ):
            for head_dim, dtype, expected in (
                (64, torch.bfloat16, 1 * 2 * 64 * 2),
                (80, torch.float8_e4m3fn, 1 * 2 * 96),
            ):
                per_token = TpuSeqAlongLaneKVPool.kv_bytes_per_token_per_layer(
                    num_kv_heads=1,
                    head_dim=head_dim,
                    v_head_dim=head_dim,
                    store_dtype=dtype,
                )
                self.assertEqual(per_token, expected)
                pool = TpuSeqAlongLaneKVPool(
                    size=4 * page_size,
                    page_size=page_size,
                    dtype=dtype,
                    head_num=1,
                    head_dim=head_dim,
                    layer_num=1,
                    device="cpu",
                    enable_memory_saver=False,
                )
                self.assertEqual(
                    pool.kv_buffer[0].nbytes, pool.num_blocks * page_size * per_token
                )


class TestPageSizeFloor(CustomTestCase):
    """SEQ_ALONG_LANE tiles pages onto lanes, so a page must be >= the floor."""

    def _server_args(self, page_size):
        # declare_late_resolution only refuses the published config; a namespace is enough.
        return types.SimpleNamespace(
            attention_backend="pallas_tpu_rpa_batched", page_size=page_size
        )

    def _resolved(self, sa, field):
        """What resolution decided, i.e. the declaration if one was made."""
        from sglang.srt.arg_groups.overrides import resolution_result

        return resolution_result(sa, field)

    def test_default_is_raised_to_the_floor(self):
        from sglang_tpu.tpu_platform import (
            _apply_page_size_default,
        )

        sa = self._server_args(None)
        _apply_page_size_default(sa)
        self.assertEqual(
            self._resolved(sa, "page_size"), RPA_BATCHED_SPEC.min_page_size
        )

    def test_explicit_below_floor_is_rejected(self):
        from sglang_tpu.tpu_platform import (
            _apply_page_size_default,
        )

        with self.assertRaises(ValueError):
            _apply_page_size_default(self._server_args(64))

    def test_explicit_above_floor_is_kept(self):
        from sglang_tpu.tpu_platform import (
            _apply_page_size_default,
        )

        sa = self._server_args(512)
        _apply_page_size_default(sa)
        self.assertEqual(self._resolved(sa, "page_size"), 512)


class TestKvCacheDtypeDefault(CustomTestCase):
    """fp8 KV halves bytes per token, but only where the kernel reads it."""

    def _server_args(self, kv_cache_dtype, backend="pallas_tpu_rpa_batched"):
        return types.SimpleNamespace(
            kv_cache_dtype=kv_cache_dtype, attention_backend=backend
        )

    def _resolved(self, sa, field):
        """What resolution decided, i.e. the declaration if one was made."""
        from sglang.srt.arg_groups.overrides import resolution_result

        return resolution_result(sa, field)

    def test_auto_becomes_fp8_for_the_rpa_backends(self):
        from sglang_tpu.tpu_platform import (
            _apply_kv_cache_dtype_default,
        )

        for backend in ("pallas_tpu_rpa_batched", "pallas_tpu_rpa_v3"):
            sa = self._server_args("auto", backend)
            _apply_kv_cache_dtype_default(sa)
            self.assertEqual(self._resolved(sa, "kv_cache_dtype"), "fp8_e4m3", backend)

    def test_backends_that_reject_fp8_keep_auto(self):
        # pallas_tpu_mla raises on an fp8 KV cache and pallas_tpu_flash has no
        # dequantization scale, so the default must not reach them -- the
        # DeepSeek recipe launches MLA with no --kv-cache-dtype.
        from sglang_tpu.tpu_platform import (
            _apply_kv_cache_dtype_default,
        )

        for backend in ("pallas_tpu_mla", "pallas_tpu_flash"):
            sa = self._server_args("auto", backend)
            _apply_kv_cache_dtype_default(sa)
            self.assertEqual(self._resolved(sa, "kv_cache_dtype"), "auto", backend)

    def test_explicit_choice_is_left_alone(self):
        from sglang_tpu.tpu_platform import (
            _apply_kv_cache_dtype_default,
        )

        for given in ("bf16", "fp8_e5m2"):
            sa = self._server_args(given)
            _apply_kv_cache_dtype_default(sa)
            self.assertEqual(self._resolved(sa, "kv_cache_dtype"), given)


class TestPrefillBlockSizes(CustomTestCase):
    """The prefill tile: one cell, and an operator override for the k block."""

    def _resolve(
        self,
        num_q_heads=4,
        num_kv_heads=1,
        head_dim=128,
        page_size=256,
        ctx_pages=133,
        sliding_window=None,
    ):
        import jax.numpy as jnp

        from sglang_tpu.attention.pallas.batched_rpa import configs, wrapper
        from sglang_tpu.pallas_export import (
            tpu_info_for_cpu_export,
        )

        model = configs.ModelConfigs(
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            mask_value=float("-inf"),
            sliding_window=sliding_window,
        )
        with tpu_info_for_cpu_export():
            serve = configs.ServingConfigs(
                num_seqs=32,
                num_page_indices=32 * ctx_pages,
                total_q_tokens=4096,
                dtype_q=jnp.bfloat16,
                dtype_kv=jnp.float8_e4m3fn,
                dtype_out=jnp.bfloat16,
                page_size=page_size,
                kv_layout=configs.KVLayout.SEQ_ALONG_LANE,
            )
            _, prefill = wrapper.calculate_block_sizes(model, serve, 67108864)
        return prefill

    def test_prefill_uses_one_cell(self):
        """A second cell halves the VMEM per cell and adds no parallel compute.

        Measured on v7-8 (Qwen3-Coder-30B, tp8, one 32k MIXED step, 4k chunk):
        two cells pick bq=bkv=1280 and run 1194us, one cell picks bq=bkv=2048
        and runs 880us, and bq=2048 with two cells does not fit VMEM.
        """
        self.assertEqual(self._resolve().batch_size, 1)

    def test_compute_chunk_divides_the_q_block(self):
        blocks = self._resolve()
        self.assertEqual(blocks.bq_sz % blocks.bq_c_sz, 0)

    def test_compute_chunk_is_power_of_two_and_meets_threshold(self):
        for nq, nkv, hd in [
            (64, 8, 64),
            (32, 4, 64),
            (16, 2, 64),
            (8, 1, 64),
            (4, 1, 128),
        ]:
            blocks = self._resolve(num_q_heads=nq, num_kv_heads=nkv, head_dim=hd)
            self.assertEqual(blocks.bq_sz % blocks.bq_c_sz, 0)
            self.assertLessEqual(blocks.bq_sz // blocks.bq_c_sz, 32)
            self.assertEqual(blocks.bq_c_sz & (blocks.bq_c_sz - 1), 0)
        # 64q/8kv grows bkv_sz to 512, where bq_c_sz=16 already has compute buffer
        # time 2048 > 1500; 16 is the floor (num_lanes // 8 q heads per KV head).
        self.assertEqual(
            self._resolve(num_q_heads=64, num_kv_heads=8, head_dim=64).bq_c_sz, 16
        )
        # 8q/1kv with bq_sz=1280 must keep power-of-2 bq_c_sz=64 instead of 1280//32=40.
        self.assertEqual(
            self._resolve(num_q_heads=8, num_kv_heads=1, head_dim=64).bq_c_sz, 64
        )

    def test_k_block_is_a_page_multiple(self):
        # A k block below one page gives an empty DMA schedule leaf and fails
        # Mosaic compilation.
        self.assertEqual(self._resolve().bkv_sz % 256, 0)

    def test_wider_head_config_gets_a_smaller_tile(self):
        """A query token costs VMEM per q head, so 12 heads must tile smaller."""
        self.assertLess(
            self._resolve(num_q_heads=12).bq_sz, self._resolve(num_q_heads=4).bq_sz
        )

    def test_high_head_count_grows_bkv_past_bq(self):
        """At 64q/8kv (PP8 with TP=1), bq_sz caps at 256 while bkv_sz grows to 512."""
        blocks = self._resolve(num_q_heads=64, num_kv_heads=8, head_dim=64)
        self.assertEqual(blocks.bq_sz, 256)
        self.assertEqual(blocks.bkv_sz, 512)
        self.assertEqual(blocks.bkv_sz % blocks.bq_sz, 0)
        # Lower head counts (32q/4kv, 4q/1kv) remain square (bq_sz == bkv_sz).
        tp2_blocks = self._resolve(num_q_heads=32, num_kv_heads=4, head_dim=64)
        self.assertEqual(tp2_blocks.bq_sz, 512)
        self.assertEqual(tp2_blocks.bkv_sz, 512)

    def test_128_head_pp8_shrinks_bq_and_grows_bkv(self):
        """Muse Spark PP8 (128q/16kv, d=64) overflows VMEM at bq_sz=256 and
        bq_sz=128; prefill shrinks bq_sz to 64 while preserving triple
        buffering and grows full-attention bkv_sz to 512 (keeping SWA at 256)."""
        import jax.numpy as jnp

        from sglang_tpu.attention.pallas.batched_rpa import configs, wrapper
        from sglang_tpu.pallas_export import tpu_info_for_cpu_export

        for kv_layout in (
            configs.KVLayout.SEQ_ALONG_LANE,
            configs.KVLayout.HEAD_ALONG_SUBLANE,
        ):
            for dtype_kv in (jnp.float8_e4m3fn, jnp.bfloat16):
                with tpu_info_for_cpu_export():
                    full_model = configs.ModelConfigs(
                        num_q_heads=128,
                        num_kv_heads=16,
                        head_dim=64,
                        mask_value=float("-inf"),
                    )
                    swa_model = configs.ModelConfigs(
                        num_q_heads=128,
                        num_kv_heads=16,
                        head_dim=64,
                        mask_value=float("-inf"),
                        sliding_window=2048,
                    )
                    serve = configs.ServingConfigs(
                        num_seqs=8,
                        num_page_indices=8 * 256,
                        total_q_tokens=1024,
                        dtype_q=jnp.bfloat16,
                        dtype_kv=dtype_kv,
                        dtype_out=jnp.bfloat16,
                        page_size=256,
                        kv_layout=kv_layout,
                    )
                    _, full_prefill = wrapper.calculate_block_sizes(
                        full_model, serve, 67108864
                    )
                    _, swa_prefill = wrapper.calculate_block_sizes(
                        swa_model, serve, 67108864
                    )
                self.assertEqual(
                    (
                        full_prefill.bq_sz,
                        full_prefill.bq_c_sz,
                        full_prefill.bkv_sz,
                        full_prefill.batch_size,
                        full_prefill.n_buffer,
                    ),
                    (64, 16, 512, 1, 3),
                )
                self.assertEqual(
                    (swa_prefill.bq_sz, swa_prefill.bkv_sz, swa_prefill.n_buffer),
                    (64, 256, 3),
                )

    def test_k_block_grows_by_at_most_one_q_block(self):
        """Wider bkv_sz / bq_sz ratios are unmeasured, so 32q/2kv stops at 512
        even though VMEM would fit a wider k block."""
        blocks = self._resolve(num_q_heads=32, num_kv_heads=2, head_dim=256)
        self.assertEqual((blocks.bq_sz, blocks.bkv_sz), (256, 512))

    def test_k_block_grows_only_after_the_first_lockstep_step(self):
        """16q/2kv reaches a 512 square tile in lockstep, so it stays square."""
        blocks = self._resolve(num_q_heads=16, num_kv_heads=2, head_dim=256)
        self.assertEqual((blocks.bq_sz, blocks.bkv_sz), (512, 512))

    def test_k_block_grows_only_with_8_or_more_q_heads_per_kv_head(self):
        """With fewer q heads per KV head, the bq_c_sz picked for a 512 k block
        made bf16 KV slower, so 40q/8kv and 24q/4kv keep the lockstep tile."""
        for nq, nkv, hd in [(40, 8, 128), (24, 4, 256)]:
            blocks = self._resolve(num_q_heads=nq, num_kv_heads=nkv, head_dim=hd)
            self.assertEqual(
                (blocks.bq_sz, blocks.bq_c_sz, blocks.bkv_sz), (256, 32, 256)
            )

    def test_k_block_does_not_grow_for_sliding_window_layers(self):
        """A k block wider than bq_sz reads more keys outside the window."""
        for window in (128, 1024):
            blocks = self._resolve(
                num_q_heads=64, num_kv_heads=8, head_dim=64, sliding_window=window
            )
            self.assertEqual((blocks.bq_sz, blocks.bkv_sz), (256, 256))

    def test_k_block_growth_needs_sequences_that_fill_it(self):
        """In a pool of one-page sequences a 512-token k block saves no KV steps;
        it only makes each schedule step larger."""
        blocks = self._resolve(num_q_heads=64, num_kv_heads=8, head_dim=64, ctx_pages=1)
        self.assertEqual((blocks.bq_sz, blocks.bkv_sz), (256, 256))

    def test_k_block_growth_keeps_whole_pages(self):
        """The grown k block is a whole number of pages, so cover_whole_pages never
        cuts it after bq_c_sz is picked for it."""
        one_page = self._resolve(
            num_q_heads=32, num_kv_heads=2, head_dim=256, page_size=512
        )
        self.assertEqual(
            (one_page.bq_sz, one_page.bq_c_sz, one_page.bkv_sz), (256, 16, 512)
        )
        # Half a 1024-token page is not a whole page: no growth, and
        # cover_whole_pages widens the lockstep tile as it did before.
        half_page = self._resolve(
            num_q_heads=32, num_kv_heads=4, head_dim=64, page_size=1024
        )
        self.assertEqual(
            (half_page.bq_sz, half_page.bq_c_sz, half_page.bkv_sz), (256, 32, 1024)
        )

    def test_override_pins_the_k_block(self):
        with envs.SGLANG_TPU_FORCE_RPA_PREFILL_BKV.override(4096):
            self.assertEqual(self._resolve().bkv_sz, 4096)

    def test_override_is_rounded_up_to_a_page(self):
        with envs.SGLANG_TPU_FORCE_RPA_PREFILL_BKV.override(5000):
            self.assertEqual(self._resolve().bkv_sz, 5120)


class TestKernelBlockSizes(CustomTestCase):
    """Block sizes the batched-RPA kernel can run."""

    # (num_q_heads, num_kv_heads, head_dim, cp_group_size). A cp_group_size makes
    # the call cache-only with LSE, as context-parallel decode issues it.
    GEOMETRIES = ((16, 2, 64, 2), (64, 8, 64, 8), (8, 1, 64, None), (4, 1, 128, None))

    def _block_sizes(
        self, *, page_size, kv_layout, dtype_kv, geometry, vmem_limit_bytes=1 << 26
    ):
        import jax.numpy as jnp

        from sglang_tpu.attention.pallas.batched_rpa import configs, wrapper
        from sglang_tpu.pallas_export import tpu_info_for_cpu_export

        num_q_heads, num_kv_heads, head_dim, cp_group_size = geometry
        model = configs.ModelConfigs(
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            mask_value=float("-inf"),
        )
        with tpu_info_for_cpu_export():
            serve = configs.ServingConfigs(
                num_seqs=16,
                num_page_indices=16 * 32,
                total_q_tokens=16,
                dtype_q=jnp.bfloat16,
                dtype_kv=dtype_kv,
                dtype_out=jnp.bfloat16,
                page_size=page_size,
                kv_layout=kv_layout,
                cache_only=cp_group_size is not None,
                skip_kv_write=cp_group_size is not None,
                return_lse=cp_group_size is not None,
                cp_group_size=cp_group_size,
            )
            return wrapper.calculate_block_sizes(model, serve, vmem_limit_bytes)

    def test_searched_k_blocks_cover_whole_pages(self):
        """A k block over cached pages covers whole pages; a partial one attends
        keys from the wrong pages from its second block on."""
        import jax.numpy as jnp

        from sglang_tpu.attention.pallas.batched_rpa import configs

        layouts = (configs.KVLayout.HEAD_ALONG_SUBLANE, configs.KVLayout.SEQ_ALONG_LANE)
        for page_size, kv_layout, dtype_kv, geometry in itertools.product(
            (512, 1024), layouts, (jnp.bfloat16, jnp.float8_e4m3fn), self.GEOMETRIES
        ):
            if geometry[3] is not None and kv_layout != layouts[0]:
                continue  # cache_only requires HEAD_ALONG_SUBLANE
            label = f"page={page_size} {kv_layout.name} {dtype_kv.__name__} {geometry}"
            with self.subTest(label):
                block_sizes = self._block_sizes(
                    page_size=page_size,
                    kv_layout=kv_layout,
                    dtype_kv=dtype_kv,
                    geometry=geometry,
                )
                for blocks in block_sizes:
                    self.assertEqual(blocks.bkv_sz % page_size, 0)
                    self.assertGreaterEqual(blocks.bkv_sz, page_size)
                    self.assertEqual(blocks.bq_sz % blocks.bq_c_sz, 0)
                    self.assertEqual(blocks.bq_c_sz & (blocks.bq_c_sz - 1), 0)
                    self.assertGreaterEqual(blocks.batch_size, 1)
                    self.assertGreaterEqual(blocks.n_buffer, 1)
                self.assertGreater(block_sizes[1].bq_sz, 1)

    def test_grown_mixed_q_block_is_never_one_token(self):
        """A mixed q block shrunk to fit a one-page k block stays above one token;
        the kernel reads bq_sz == 1 as a decode-only block."""
        import jax.numpy as jnp

        from sglang_tpu.attention.pallas.batched_rpa import configs

        shrunk = 0
        for vmem_mib in range(30, 46):
            try:
                _, prefill = self._block_sizes(
                    page_size=1024,
                    kv_layout=configs.KVLayout.SEQ_ALONG_LANE,
                    dtype_kv=jnp.bfloat16,
                    geometry=(16, 16, 64, None),
                    vmem_limit_bytes=vmem_mib << 20,
                )
            except ValueError:
                continue
            self.assertGreater(prefill.bq_sz, 1, f"{vmem_mib} MiB")
            shrunk += prefill.bq_sz < 256
        self.assertGreater(shrunk, 0)  # some budget must exercise the shrinking
        # When bkv_sz grows from 768 to page_size=1024 on 16q/2kv and bq_sz shrinks
        # to 384, bq_c_sz re-evaluates the compute threshold against bkv_sz=1024
        # (stepping down from 32 to 16) rather than keeping gcd(32, 384) == 32.
        _, grown = self._block_sizes(
            page_size=1024,
            kv_layout=configs.KVLayout.SEQ_ALONG_LANE,
            dtype_kv=jnp.bfloat16,
            geometry=(16, 2, 64, None),
        )
        self.assertEqual((grown.bq_sz, grown.bq_c_sz, grown.bkv_sz), (384, 16, 1024))

    def test_explicit_partial_page_k_block_is_refused(self):
        """An explicit k block must cover whole pages too when the call reads
        cached pages; a prefill-only call reads none."""
        import jax.numpy as jnp

        from sglang_tpu.attention.pallas.batched_rpa import configs, wrapper
        from sglang_tpu.pallas_export import tpu_info_for_cpu_export

        page_size, num_seqs, pages_per_seq = 512, 4, 4
        model = configs.ModelConfigs(
            num_q_heads=8, num_kv_heads=1, head_dim=64, mask_value=float("-inf")
        )
        with tpu_info_for_cpu_export():
            serve = configs.ServingConfigs(
                num_seqs=num_seqs,
                num_page_indices=num_seqs * pages_per_seq,
                total_q_tokens=num_seqs,
                dtype_q=jnp.bfloat16,
                dtype_kv=jnp.bfloat16,
                dtype_out=jnp.bfloat16,
                page_size=page_size,
            )
            kv_shape = wrapper.get_kv_cache_shape(
                num_seqs * pages_per_seq, page_size, 1, 64, jnp.bfloat16
            )
        inputs = dict(
            q=jnp.zeros((num_seqs, 8, 64), jnp.bfloat16),
            k=jnp.zeros((num_seqs, 1, 64), jnp.bfloat16),
            v=jnp.zeros((num_seqs, 1, 64), jnp.bfloat16),
            kv_cache=jnp.zeros(kv_shape, jnp.bfloat16),
            kv_lens=jnp.zeros((num_seqs,), jnp.int32),
            page_indices=jnp.zeros((num_seqs * pages_per_seq,), jnp.int32),
            cu_q_lens=jnp.arange(num_seqs + 1, dtype=jnp.int32),
            distribution=jnp.asarray([0, 0, num_seqs], jnp.int32),
        )

        def validate(bkv_sz, mode):
            block = configs.BlockSizes(
                bq_sz=1, bq_c_sz=1, bkv_sz=bkv_sz, batch_size=1, n_buffer=2
            )
            configs.RpaConfigs(
                block=block,
                model=model,
                serve=serve,
                mode=mode,
                vmem_limit_bytes=1 << 26,
            ).validate_inputs(**inputs)

        with tpu_info_for_cpu_export():
            with self.assertRaises(ValueError):
                validate(bkv_sz=1280, mode=configs.RpaCase.DECODE)
            validate(bkv_sz=1024, mode=configs.RpaCase.DECODE)
            validate(bkv_sz=1280, mode=configs.RpaCase.PREFILL)

    def test_explicit_blocks_are_used_where_the_search_refuses(self):
        """Explicit decode and prefill block sizes are used as given, without
        waiting on a block search that cannot answer the shape."""
        import functools

        import jax
        import jax.numpy as jnp

        from sglang_tpu.attention.pallas.batched_rpa import configs, wrapper
        from sglang_tpu.pallas_export import tpu_info_for_cpu_export

        page_size, num_seqs, pages_per_seq, vmem_limit_bytes = 256, 4, 2, 1 << 20
        with self.assertRaises(ValueError):
            self._block_sizes(
                page_size=page_size,
                kv_layout=configs.KVLayout.HEAD_ALONG_SUBLANE,
                dtype_kv=jnp.bfloat16,
                geometry=(8, 1, 64, None),
                vmem_limit_bytes=vmem_limit_bytes,
            )
        attention = functools.partial(
            wrapper.ragged_paged_attention,
            decode_block_sizes=configs.BlockSizes(
                bq_sz=1, bq_c_sz=1, bkv_sz=page_size, batch_size=1, n_buffer=2
            ),
            prefill_block_sizes=configs.BlockSizes(
                bq_sz=16, bq_c_sz=16, bkv_sz=page_size, batch_size=1, n_buffer=2
            ),
            vmem_limit_bytes=vmem_limit_bytes,
        )
        with tpu_info_for_cpu_export():
            kv_shape = wrapper.get_kv_cache_shape(
                num_seqs * pages_per_seq, page_size, 1, 64, jnp.bfloat16
            )
            jax.eval_shape(
                attention,
                jax.ShapeDtypeStruct((8, 8, 64), jnp.bfloat16),
                jax.ShapeDtypeStruct((8, 1, 64), jnp.bfloat16),
                jax.ShapeDtypeStruct((8, 1, 64), jnp.bfloat16),
                jax.ShapeDtypeStruct(kv_shape, jnp.bfloat16),
                jax.ShapeDtypeStruct((num_seqs,), jnp.int32),
                jax.ShapeDtypeStruct((num_seqs * pages_per_seq,), jnp.int32),
                jax.ShapeDtypeStruct((num_seqs + 1,), jnp.int32),
                jax.ShapeDtypeStruct((3,), jnp.int32),
            )

    def test_budget_below_one_lane_and_one_buffer_is_refused(self):
        """A VMEM budget too small for one lane and one buffer is refused, not
        answered with a buffer count below one."""
        import jax.numpy as jnp

        from sglang_tpu.attention.pallas.batched_rpa import configs

        for vmem_limit_bytes in (1 << 18, 1 << 19, 1 << 20, 1 << 21):
            with self.subTest(vmem_limit_bytes=vmem_limit_bytes):
                with self.assertRaises(ValueError):
                    self._block_sizes(
                        page_size=256,
                        kv_layout=configs.KVLayout.SEQ_ALONG_LANE,
                        dtype_kv=jnp.float8_e4m3fn,
                        geometry=(4, 1, 128, None),
                        vmem_limit_bytes=vmem_limit_bytes,
                    )


if __name__ == "__main__":
    unittest.main()
