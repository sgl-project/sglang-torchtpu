"""Unit tests for the TPU Pallas RPA v3 backend."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.runtime_context import get_context
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.attention_pallas_rpa_base import (
    _zero_inactive_rpa_rows,
)
from sglang_tpu.attention_pallas_rpa_v3 import (
    TpuPallasRpaV3AttnBackend,
    _invoke_v3_kernel,
)
from sglang_tpu.envs import envs

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


from sglang_tpu.attention_pallas_rpa_base import (
    _SCOPED_SMEM_BYTES,
    _get_pages_per_seq,
    _page_table_smem_bytes,
)


class TestDistributionWarmup(CustomTestCase):
    def test_distribution_has_fixed_shape_for_all_live_counts(self):
        backend = object.__new__(TpuPallasRpaV3AttnBackend)
        backend._distribution_cache = {}
        backend.warmup_distributions(256, "cpu:0")
        self.assertEqual(len(backend._distribution_cache), 2)
        for rows in (0, 1, 3, 7, 255, 256):
            for decode in (False, True):
                dist = backend._get_cached_rpa_distribution(decode, rows, "cpu")
                self.assertEqual(dist.shape, (3,))
                self.assertEqual(dist.tolist(), [rows if decode else 0, rows, rows])


class TestPageTableSmemBudget(unittest.TestCase):
    def test_1m_context_default_batch_cap_exceeds_smem(self):
        # 1,052,672 tokens at page 64 -> 16,448 pages/seq; the default compile
        # cap of 64 sequences needs 4.2 MB, far above the 1 MiB scoped SMEM.
        pages = _get_pages_per_seq(1052672, 64)
        self.assertEqual(pages, 16448)
        self.assertGreater(_page_table_smem_bytes(64, pages), _SCOPED_SMEM_BYTES)
        self.assertLessEqual(_page_table_smem_bytes(8, pages), _SCOPED_SMEM_BYTES)

    def test_entries_padded_to_128(self):
        self.assertEqual(_page_table_smem_bytes(1, 1), 128 * 4)
        self.assertEqual(_page_table_smem_bytes(2, 100), 256 * 4)


class TestAttentionPallasRpaV3(unittest.TestCase):
    CONTEXT_LEN = 131072
    POOL_WIDTH = 262144
    PAGE_SIZE = 64
    DRAFT_TOKENS = 8

    def _make_backend(
        self,
        speculative_step_id=0,
        pool_width=None,
        req_to_token_device="cpu",
        cache_metadata=True,
    ):
        pool_width = self.POOL_WIDTH if pool_width is None else pool_width
        override = get_context().override_server_args(
            speculative_num_draft_tokens=self.DRAFT_TOKENS,
            speculative_algorithm="EAGLE3",
            speculative_num_steps=None,
            speculative_eagle_topk=1,
            page_size=self.PAGE_SIZE,
        )
        override.install()
        self.addCleanup(override.restore)
        server_args = SimpleNamespace(
            speculative_num_draft_tokens=self.DRAFT_TOKENS,
            speculative_algorithm="EAGLE3",
            speculative_num_steps=None,
            speculative_eagle_topk=1,
            page_size=self.PAGE_SIZE,
        )
        model_runner = SimpleNamespace(
            device=torch.device("cpu"),
            page_size=self.PAGE_SIZE,
            model_config=SimpleNamespace(context_len=self.CONTEXT_LEN),
            server_args=server_args,
            req_to_token_pool=SimpleNamespace(
                req_to_token=torch.zeros(
                    1,
                    pool_width,
                    dtype=torch.int32,
                    device=req_to_token_device,
                )
            ),
            token_to_kv_pool=SimpleNamespace(),
            is_hybrid_swa=False,
            kv_cache_dtype=torch.bfloat16,
            is_draft_worker=False,
            spec_algorithm=SpeculativeAlgorithm.EAGLE3,
        )
        with (
            patch(
                "sglang_tpu.attention_pallas_rpa_base._build_pallas_kernel_op",
                return_value=None,
            ),
            envs.SGLANG_TPU_ENABLE_ATTN_METADATA_CACHE.override(cache_metadata),
        ):
            return TpuPallasRpaV3AttnBackend(
                model_runner, speculative_step_id=speculative_step_id
            )

    def _build_metadata(self, backend, forward_mode, spec_info, seq_len=None):
        seq_len = self.CONTEXT_LEN - 2 if seq_len is None else seq_len
        seq_lens = torch.tensor([seq_len], dtype=torch.int32)
        forward_batch = SimpleNamespace(
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens.clone(),
            req_pool_indices=torch.tensor([0], dtype=torch.int64),
            req_pool_indices_cpu=torch.tensor([0], dtype=torch.int64),
            forward_mode=forward_mode,
            extend_seq_lens=None,
            extend_seq_lens_cpu=None,
            spec_info=spec_info,
        )
        return backend._build_metadata(forward_batch, non_empty_bs=1)

    def _build_padded_decode_metadata(self, backend, active_bs):
        physical_bs = 128
        seq_lens = torch.full((physical_bs,), 100, dtype=torch.int32)
        req_pool_indices = torch.zeros(physical_bs, dtype=torch.int64)
        forward_batch = SimpleNamespace(
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens.clone(),
            req_pool_indices=req_pool_indices,
            req_pool_indices_cpu=req_pool_indices.clone(),
            forward_mode=ForwardMode.DECODE,
            extend_seq_lens=None,
            spec_info=None,
        )
        return backend._build_metadata(forward_batch, non_empty_bs=active_bs)

    def test_pages_per_seq_uses_context_plus_spec_headroom(self):
        backend = self._make_backend()
        # extra = max(4 + draft, 2*max(steps*topk, draft) + page - 1) = 79
        self.assertEqual(backend.max_context_len, self.CONTEXT_LEN + 79)
        self.assertEqual(
            (backend.max_context_len + self.PAGE_SIZE - 1) // self.PAGE_SIZE,
            2050,
        )
        self.assertLess(backend.max_context_len, self.POOL_WIDTH)

    def test_metadata_is_rebuilt_when_cache_is_off(self):
        """With the cache disabled, metadata is uploaded fresh every forward."""
        backend = self._make_backend(cache_metadata=False)
        self.assertIsNone(backend._host_metadata_cache)
        first = self._build_metadata(backend, ForwardMode.DECODE, None, seq_len=63)
        second = self._build_metadata(backend, ForwardMode.DECODE, None, seq_len=63)
        self.assertIsNot(first.cu_q_lens, second.cu_q_lens)
        self.assertIsNot(first.cu_kv_lens, second.cu_kv_lens)
        self.assertIsNot(first.active_num_tokens, second.active_num_tokens)
        self.assertIsNot(first.page_indices, second.page_indices)
        self.assertEqual(first.cu_q_lens.tolist(), second.cu_q_lens.tolist())

    def test_fixed_offsets_do_not_cache_request_state(self):
        """A reused layout must still observe a new KV length and page owner."""
        backend = self._make_backend()
        first = self._build_metadata(backend, ForwardMode.DECODE, None, seq_len=63)
        backend.req_to_token_pool.req_to_token[0, 0] = 2 * self.PAGE_SIZE
        second = self._build_metadata(backend, ForwardMode.DECODE, None, seq_len=65)
        self.assertIs(first.cu_q_lens, second.cu_q_lens)
        self.assertIs(first.cu_kv_lens, second.cu_kv_lens)
        self.assertIs(first.active_num_tokens, second.active_num_tokens)
        self.assertEqual(first.kv_lens.tolist(), [63])
        self.assertEqual(second.kv_lens.tolist(), [65])
        self.assertEqual(first.page_indices[0].item(), 0)
        self.assertEqual(second.page_indices[0].item(), 2)

    def test_fixed_metadata_tracks_physical_and_live_widths(self):
        """Bucket reuse must not bake the preceding batch's live rows or width."""
        backend = self._make_backend()
        for bs, live, width in ((4, 4, 1), (4, 2, 1), (4, 2, 4), (8, 4, 4), (4, 0, 1)):
            with self.subTest(bs=bs, live=live, width=width):
                q, kv, active = backend._get_query_metadata(
                    bs=bs,
                    kv_stride=192,
                    non_empty_bs=live,
                    device="cpu",
                    query_width=width,
                )
                self.assertEqual(q.tolist(), [i * width for i in range(bs + 1)])
                self.assertEqual(kv.tolist(), [i * 192 for i in range(bs + 1)])
                self.assertEqual(active.shape, ())
                self.assertEqual(active.item(), live * width)
                self.assertNotEqual(
                    active.untyped_storage().data_ptr(), q.untyped_storage().data_ptr()
                )

    def test_fixed_offsets_reused_across_live_batch_sizes(self):
        """Changing only the live prefix must not duplicate bucket offsets."""
        backend = self._make_backend()
        first = backend._get_query_metadata(
            bs=256,
            kv_stride=192,
            non_empty_bs=0,
            device="cpu",
            query_width=4,
        )
        for live in range(257):
            q, kv, active = backend._get_query_metadata(
                bs=256,
                kv_stride=192,
                non_empty_bs=live,
                device="cpu",
                query_width=4,
            )
            self.assertIs(q, first[0])
            self.assertIs(kv, first[1])
            self.assertEqual(active.item(), 4 * live)
        self.assertEqual(first[2].item(), 0)

    def test_active_scalar_reused_across_query_layouts(self):
        """Equal active-token counts can share storage across different layouts."""
        backend = self._make_backend()
        decode = backend._get_query_metadata(
            bs=256,
            kv_stride=192,
            non_empty_bs=256,
            device="cpu",
            query_width=1,
        )
        verify = backend._get_query_metadata(
            bs=128,
            kv_stride=192,
            non_empty_bs=64,
            device="cpu",
            query_width=4,
        )
        self.assertIs(decode[2], verify[2])
        self.assertEqual(verify[2].item(), 256)
        self.assertEqual(decode[0][-1].item(), 256)
        self.assertEqual(verify[0][-1].item(), 512)

    def test_extend_offsets_remain_request_dependent(self):
        """Same-bucket prefills with different query lengths cannot share offsets."""
        backend = self._make_backend()
        batch = SimpleNamespace(
            seq_lens=torch.tensor([10, 20]),
            seq_lens_cpu=torch.tensor([10, 20]),
            req_pool_indices_cpu=torch.tensor([0, 0]),
            forward_mode=ForwardMode.EXTEND,
            extend_seq_lens=torch.tensor([2, 3]),
            extend_seq_lens_cpu=None,
            spec_info=None,
        )
        first = backend._build_metadata(batch, non_empty_bs=2)
        batch.extend_seq_lens = torch.tensor([3, 1])
        second = backend._build_metadata(batch, non_empty_bs=2)
        self.assertEqual(first.cu_q_lens.tolist(), [0, 2, 5])
        self.assertEqual(second.cu_q_lens.tolist(), [0, 3, 4])
        self.assertEqual(first.active_num_tokens.item(), 5)
        self.assertEqual(second.active_num_tokens.item(), 4)

    def test_rejects_context_capacity_larger_than_request_table(self):
        with self.assertRaisesRegex(
            ValueError, "exceeds the request-to-token table width"
        ):
            self._make_backend(pool_width=self.CONTEXT_LEN)

    def test_target_verify_bump_fits_rectangular_page_stride(self):
        backend = self._make_backend()
        metadata = self._build_metadata(
            backend,
            ForwardMode.TARGET_VERIFY,
            SimpleNamespace(num_tokens_per_req=self.DRAFT_TOKENS),
        )
        capacity = metadata.cu_kv_lens[1].item()
        self.assertEqual(metadata.kv_lens.item(), self.CONTEXT_LEN + 6)
        self.assertLessEqual(metadata.kv_lens.item(), capacity)
        self.assertEqual(metadata.page_indices.numel(), 2050)

    def test_target_verify_rejects_bump_above_page_stride(self):
        for req_to_token_device in ("cpu", "meta"):
            with self.subTest(req_to_token_device=req_to_token_device):
                backend = self._make_backend(req_to_token_device=req_to_token_device)
                with self.assertRaisesRegex(
                    ValueError, "exceeds the per-request page-table capacity"
                ):
                    self._build_metadata(
                        backend,
                        ForwardMode.TARGET_VERIFY,
                        SimpleNamespace(num_tokens_per_req=self.DRAFT_TOKENS),
                        seq_len=131194,
                    )

    def test_multistep_decode_bump_fits_rectangular_page_stride(self):
        backend = self._make_backend(speculative_step_id=self.DRAFT_TOKENS - 1)
        metadata = self._build_metadata(
            backend,
            ForwardMode.DECODE,
            SimpleNamespace(),
        )
        capacity = metadata.cu_kv_lens[1].item()
        self.assertEqual(metadata.kv_lens.item(), self.CONTEXT_LEN + 6)
        self.assertLessEqual(metadata.kv_lens.item(), capacity)

    def test_multistep_decode_rejects_bump_above_page_stride(self):
        for req_to_token_device in ("cpu", "meta"):
            with self.subTest(req_to_token_device=req_to_token_device):
                backend = self._make_backend(
                    speculative_step_id=self.DRAFT_TOKENS - 1,
                    req_to_token_device=req_to_token_device,
                )
                with self.assertRaisesRegex(
                    ValueError, "exceeds the per-request page-table capacity"
                ):
                    self._build_metadata(
                        backend,
                        ForwardMode.DECODE,
                        SimpleNamespace(),
                        seq_len=131194,
                    )

    def test_seq_lens_cpu_list_and_empty_inputs(self):
        backend = self._make_backend()

        # 1. seq_lens_cpu as a Python list
        fb_list = SimpleNamespace(
            seq_lens=torch.tensor([100], dtype=torch.int32),
            seq_lens_cpu=[100],
            req_pool_indices=torch.tensor([0], dtype=torch.int64),
            req_pool_indices_cpu=torch.tensor([0], dtype=torch.int64),
            forward_mode=ForwardMode.DECODE,
            extend_seq_lens=None,
            spec_info=None,
        )
        meta_list = backend._build_metadata(fb_list, non_empty_bs=1)
        self.assertIsNotNone(meta_list)

        # 2. seq_lens_cpu as empty Tensor (e.g. idle rank forced into decode for DP collectives)
        fb_empty_tensor = SimpleNamespace(
            seq_lens=torch.empty(0, dtype=torch.int32),
            seq_lens_cpu=torch.empty(0, dtype=torch.int64),
            req_pool_indices=torch.empty(0, dtype=torch.int64),
            req_pool_indices_cpu=torch.empty(0, dtype=torch.int64),
            forward_mode=ForwardMode.DECODE,
            extend_seq_lens=None,
            spec_info=None,
        )
        meta_empty_tensor = backend._build_metadata(fb_empty_tensor, non_empty_bs=0)
        self.assertIsNotNone(meta_empty_tensor)

        # 3. seq_lens_cpu as an empty list
        fb_empty_list = SimpleNamespace(
            seq_lens=torch.empty(0, dtype=torch.int32),
            seq_lens_cpu=[],
            req_pool_indices=torch.empty(0, dtype=torch.int64),
            req_pool_indices_cpu=torch.empty(0, dtype=torch.int64),
            forward_mode=ForwardMode.DECODE,
            extend_seq_lens=None,
            spec_info=None,
        )
        meta_empty_list = backend._build_metadata(fb_empty_list, non_empty_bs=0)
        self.assertIsNotNone(meta_empty_list)

    def test_active_num_tokens_owns_scalar_storage_for_padded_decode(self):
        for req_to_token_device in ("cpu", "meta"):
            with self.subTest(req_to_token_device=req_to_token_device):
                backend = self._make_backend(req_to_token_device=req_to_token_device)
                for active_bs in (128, 125):
                    with self.subTest(active_bs=active_bs):
                        metadata = self._build_padded_decode_metadata(
                            backend, active_bs
                        )

                        self.assertEqual(metadata.cu_q_lens.shape, (129,))
                        self.assertEqual(metadata.active_num_tokens.shape, ())
                        self.assertEqual(metadata.active_num_tokens.item(), active_bs)
                        self.assertNotEqual(
                            metadata.active_num_tokens.untyped_storage().data_ptr(),
                            metadata.cu_q_lens.untyped_storage().data_ptr(),
                        )

                        metadata.cu_q_lens[active_bs] = -1
                        self.assertEqual(metadata.active_num_tokens.item(), active_bs)


class TestInactiveRpaRows(unittest.TestCase):
    def test_zeros_only_inactive_nan_rows(self):
        output = torch.arange(32, dtype=torch.float32).reshape(4, 8)
        live_rows = output[:2].clone()
        output[2:] = float("nan")

        output = _zero_inactive_rpa_rows(output, torch.tensor(2))

        self.assertTrue(torch.equal(output[:2], live_rows))
        self.assertTrue(torch.equal(output[2:], torch.zeros(2, 8)))

    def test_preserves_non_finite_live_values(self):
        output = torch.ones(3, 4)
        output[0, 0] = float("nan")
        output[2] = float("nan")

        output = _zero_inactive_rpa_rows(output, torch.tensor(2))

        self.assertTrue(torch.isnan(output[0, 0]))
        self.assertTrue(torch.equal(output[2], torch.zeros(4)))


class TestStationaryQPrefillEnvPrecedence(unittest.TestCase):
    def _invoke(
        self,
        *,
        q_shape=(8192, 16, 128),
        sliding_window=None,
        prepared_inputs=False,
        skip_decode=False,
        a3_out_layout=False,
        pages_per_seq=128,
        page_size=128,
        k_scale=1.0,
        causal=True,
        bs=2,
        soft_cap=None,
        attention_sink=None,
    ):
        calls = []
        _invoke_v3_kernel(
            lambda *args, **kwargs: calls.append(kwargs),
            queries=SimpleNamespace(shape=q_shape, ndim=len(q_shape), dtype="bfloat16"),
            keys=SimpleNamespace(shape=(q_shape[0], 1, 64)),
            values=None,
            kv_cache_fused=SimpleNamespace(
                shape=(9, page_size, 1, 4, 128), dtype="float8_e4m3fn"
            ),
            kv_lens=SimpleNamespace(shape=(bs,)),
            page_indices=SimpleNamespace(shape=(bs * pages_per_seq,)),
            cu_q_lens=None,
            cu_kv_lens=None,
            distribution=None,
            attention_sink=attention_sink,
            sm_scale=0.125,
            sliding_window=sliding_window,
            soft_cap=soft_cap,
            k_scale=k_scale,
            use_causal_mask=causal,
            v_scale=1.0,
            prepared_inputs=prepared_inputs,
            skip_decode=skip_decode,
            a3_out_layout=a3_out_layout,
        )
        return calls[0]

    def test_native_tp16_long_prefill_selects_by_live_metadata(self):
        import jax
        import jax.numpy as jnp

        def chosen(lens, cu_q, distribution):
            return _invoke_v3_kernel(
                lambda *args, **kw: jnp.int32(kw["m_block_sizes"][0]),
                queries=SimpleNamespace(shape=(8192, 8, 64), ndim=3, dtype="bfloat16"),
                keys=SimpleNamespace(shape=(8192, 1, 64)),
                values=None,
                kv_cache_fused=SimpleNamespace(
                    shape=(1, 128, 1, 4, 128), dtype="float8_e4m3fn"
                ),
                kv_lens=lens,
                page_indices=SimpleNamespace(shape=(2 * 2128,)),
                cu_q_lens=cu_q,
                cu_kv_lens=None,
                distribution=distribution,
                attention_sink=None,
                sm_scale=0.125,
                sliding_window=None,
                soft_cap=None,
                k_scale=None,
                v_scale=None,
                skip_decode=True,
            )

        with (
            envs.SGLANG_TPU_RPA_V3_PREFILL_KERNEL_MODE.override("m0_ring_fuse"),
            envs.SGLANG_TPU_RPA_V3_TP16_NATIVE_PREFILL.override(True),
            envs.SGLANG_TPU_RPA_V3_TP16_LONG_PREFILL_TILES.override(True),
        ):
            cpu = jax.devices("cpu")[0]
            with jax.default_device(cpu):
                inputs = tuple(jnp.zeros(n, jnp.int32) for n in (2, 3, 3))
                executable = jax.jit(chosen).lower(*inputs).compile()
                for prefix, live, active, decode, expected in (
                    (262144, 8192, 1, 0, 2048),
                    (131072, 6887, 1, 0, 1024),
                    (131072, 6145, 1, 0, 512),
                    (65536, 6144, 1, 0, 2048),
                    (65535, 8192, 1, 0, 512),
                    (262144, 6000, 1, 0, 512),
                    (262144, 7168, 2, 0, 512),
                    (262144, 8192, 1, 1, 512),
                    (0, 0, 0, 0, 512),
                ):
                    args = (
                        jnp.array([prefix + live, 0], jnp.int32),
                        jnp.array([0, live, live], jnp.int32),
                        jnp.array([decode, active, active], jnp.int32),
                    )
                    with self.subTest(
                        prefix=prefix, live=live, active=active, decode=decode
                    ):
                        self.assertEqual(int(executable(*args)), expected)

    def test_native_tp16_prefill(self):
        with (
            envs.SGLANG_TPU_RPA_V3_PREFILL_KERNEL_MODE.override("m0_ring_fuse"),
            envs.SGLANG_TPU_RPA_V3_TP16_NATIVE_PREFILL.override(True),
        ):
            q = {"q_shape": (8192, 8, 64), "skip_decode": True}
            for window, tiles in (
                (None, (512, 2048, 512, 256)),
                (2048, (256, 1024, 256, 256)),
                (2049, (256, 1024, 256, 256)),
            ):
                kw = self._invoke(**q, sliding_window=window)
                self.assertEqual(kw["m_block_sizes"], tiles)
                self.assertTrue(
                    {"mask_skip", "visible_kv_tile", "fold_scale"}
                    <= kw["_prefill_opts"]
                )
            self.assertNotIn(
                "fold_scale", self._invoke(**q, k_scale=0.75)["_prefill_opts"]
            )
            for case in (
                {},
                {**q, "q_shape": (4096, 8, 64)},
                {**q, "skip_decode": False},
                {**q, "sliding_window": 512},
                {**q, "causal": False},
                {**q, "page_size": 64},
            ):
                self.assertNotIn(
                    "visible_kv_tile", self._invoke(**case)["_prefill_opts"]
                )
            with envs.SGLANG_TPU_RPA_V3_STATIONARY_Q_BLOCK_SIZES.override(
                "256,4096,256,256"
            ):
                kw = self._invoke(**q)
                self.assertEqual(kw["m_block_sizes"], (256, 4096, 256, 256))
                self.assertNotIn("visible_kv_tile", kw["_prefill_opts"])

    def test_prefill_mode_overrides_legacy_ma_mixed_envs_and_strips_opts(self):
        with (
            envs.SGLANG_TPU_RPA_V3_MIXED_BLOCK_SIZES.override("256,2048,128,512"),
            envs.SGLANG_TPU_RPA_V3_MIXED_BLOCK_SIZES_SW.override("256,2048,256,256"),
            envs.SGLANG_TPU_RPA_V3_PREFILL_FOLD_SCALE.override(True),
            envs.SGLANG_TPU_RPA_V3_PREFILL_MASK_SKIP.override(True),
        ):
            # Mode unset -> exact legacy ma behavior preserved.
            kw_off_full = self._invoke(sliding_window=None)
            self.assertEqual(kw_off_full["m_block_sizes"], (256, 2048, 128, 512))
            self.assertEqual(
                kw_off_full["_prefill_opts"], frozenset({"fold_scale", "mask_skip"})
            )

            # Mode="m0" -> overrides legacy MIXED_BLOCK_SIZES and strips fold_scale/mask_skip.
            with envs.SGLANG_TPU_RPA_V3_PREFILL_KERNEL_MODE.override("m0"):
                kw_sw = self._invoke(sliding_window=2049)
                self.assertEqual(kw_sw["m_block_sizes"], (512, 2048, 256, 256))
                self.assertEqual(
                    kw_sw["_prefill_opts"],
                    frozenset({"stationary_q_m0", "skip_empty_cdma", "mixed_s3"}),
                )

                kw_full_std = self._invoke(sliding_window=None, pages_per_seq=128)
                self.assertEqual(kw_full_std["m_block_sizes"], (512, 2048, 256, 256))
                self.assertEqual(
                    kw_full_std["_prefill_opts"],
                    frozenset(
                        {
                            "stationary_q_m0",
                            "bkv_full_unroll",
                            "skip_empty_cdma",
                            "mixed_s3",
                        }
                    ),
                )

                # T=256 uses bq=256; T<256 (e.g. decode B=128) stays on legacy path.
                kw_t256 = self._invoke(q_shape=(256, 16, 128), sliding_window=2049)
                self.assertEqual(kw_t256["m_block_sizes"], (256, 2048, 256, 256))
                kw_b128 = self._invoke(q_shape=(128, 16, 128), sliding_window=2049)
                self.assertEqual(kw_b128["m_block_sizes"], (256, 2048, 256, 256))
                self.assertEqual(
                    kw_b128["_prefill_opts"], frozenset({"fold_scale", "mask_skip"})
                )

    def test_auto_bkv4096_for_long_context_full_and_explicit_override(self):
        with envs.SGLANG_TPU_RPA_V3_PREFILL_KERNEL_MODE.override("m0"):
            # T=16384 even at standard context -> bkv=4096 on Full, 2048 on SW.
            self.assertEqual(
                self._invoke(q_shape=(16384, 16, 128), sliding_window=None)[
                    "m_block_sizes"
                ],
                (512, 4096, 256, 256),
            )
            self.assertEqual(
                self._invoke(q_shape=(16384, 16, 128), sliding_window=2049)[
                    "m_block_sizes"
                ],
                (512, 2048, 256, 256),
            )
            # Long-context page table (e.g. 64K / 128K -> pages_per_seq=512, cap=65536 > 18432)
            # -> auto-selects bkv=4096 at T=8192 and T=2048 on Full!
            self.assertEqual(
                self._invoke(
                    q_shape=(1, 8192, 8, 2, 128),
                    sliding_window=None,
                    prepared_inputs=True,
                    pages_per_seq=512,
                )["m_block_sizes"],
                (512, 4096, 256, 256),
            )
            # Explicit STATIONARY_Q_BLOCK_SIZES[_SW] overrides auto selection.
            with (
                envs.SGLANG_TPU_RPA_V3_STATIONARY_Q_BLOCK_SIZES.override(
                    "512,2048,256,256"
                ),
                envs.SGLANG_TPU_RPA_V3_STATIONARY_Q_BLOCK_SIZES_SW.override(
                    "256,2048,256,256"
                ),
            ):
                self.assertEqual(
                    self._invoke(sliding_window=None, pages_per_seq=512)[
                        "m_block_sizes"
                    ],
                    (512, 2048, 256, 256),
                )
                self.assertEqual(
                    self._invoke(sliding_window=2049)["m_block_sizes"],
                    (256, 2048, 256, 256),
                )

    def test_prepared_skip_empty_decode_and_kernel_op_selection(self):
        with (
            envs.SGLANG_TPU_RPA_V3_PREFILL_KERNEL_MODE.override("pilot"),
            envs.SGLANG_TPU_RPA_V3_SKIP_EMPTY_DECODE.override(True),
        ):
            kw = self._invoke(
                q_shape=(1, 8192, 8, 2, 128),
                sliding_window=2049,
                prepared_inputs=True,
            )
            self.assertTrue(kw["_prepared_inputs"])
            self.assertTrue(kw["_skip_decode"])
            self.assertEqual(
                kw["_prefill_opts"],
                frozenset({"stationary_q_pilot", "skip_empty_cdma", "mixed_s3"}),
            )

        backend = object.__new__(TpuPallasRpaV3AttnBackend)
        backend._op = "v3"
        backend._batched_prefill_op = None
        backend._fused_out_op = None
        backend._skip_decode_op = "skip_dec"
        backend._dcp = 1
        backend.forward_metadata = SimpleNamespace(
            decode_query_size=1, cp_write_segments=None
        )
        self.assertEqual(
            backend._kernel_op(SimpleNamespace(forward_mode=ForwardMode.EXTEND), True),
            "skip_dec",
        )
        self.assertEqual(
            backend._kernel_op(SimpleNamespace(forward_mode=ForwardMode.DECODE), True),
            "v3",
        )

    def test_stationary_q_v2_modes_routing(self):
        for mode in (
            "m0_ring",
            "m0_f32e",
            "m0_ring_fuse",
            "m0_f32e_fuse",
        ):
            with (
                envs.SGLANG_TPU_RPA_V3_PREFILL_KERNEL_MODE.override(mode),
                envs.SGLANG_TPU_RPA_V3_SKIP_EMPTY_DECODE.override(True),
                envs.SGLANG_TPU_RPA_V3_PREFILL_FOLD_SCALE.override(True),
                envs.SGLANG_TPU_RPA_V3_PREFILL_MASK_SKIP.override(True),
            ):
                kw_sw = self._invoke(
                    q_shape=(1, 8192, 8, 2, 128),
                    sliding_window=2049,
                    prepared_inputs=True,
                )
                self.assertEqual(kw_sw["m_block_sizes"], (256, 2048, 256, 256))
                self.assertEqual(
                    kw_sw["_prefill_opts"],
                    frozenset({f"stationary_q_{mode}", "skip_empty_cdma", "mixed_s3"}),
                )
                # Standard <=32K capacity Full -> (256, 2048, 256, 256)
                kw_full_std = self._invoke(
                    q_shape=(1, 8192, 8, 2, 128),
                    sliding_window=None,
                    prepared_inputs=True,
                    pages_per_seq=128,
                )
                self.assertEqual(kw_full_std["m_block_sizes"], (256, 2048, 256, 256))
                # >32K capacity Full (pages_per_seq=512 -> 64K) -> streaming (512, 4096, 256, 256)
                kw_full = self._invoke(
                    q_shape=(1, 8192, 8, 2, 128),
                    sliding_window=None,
                    prepared_inputs=True,
                    pages_per_seq=512,
                )
                self.assertEqual(kw_full["m_block_sizes"], (512, 4096, 256, 256))
                self.assertEqual(
                    kw_full["_prefill_opts"],
                    frozenset(
                        {
                            f"stationary_q_{mode}",
                            "bkv_full_unroll",
                            "skip_empty_cdma",
                            "mixed_s3",
                        }
                    ),
                )
                # >32K capacity SW still stays (256, 2048, 256, 256)
                kw_sw_long = self._invoke(
                    q_shape=(1, 8192, 8, 2, 128),
                    sliding_window=2049,
                    prepared_inputs=True,
                    pages_per_seq=512,
                )
                self.assertEqual(kw_sw_long["m_block_sizes"], (256, 2048, 256, 256))
                # Direct 2D A3 layout composes with all Stationary-Q v2 modes
                kw_a3 = self._invoke(
                    q_shape=(1, 8192, 8, 2, 128),
                    sliding_window=2049,
                    prepared_inputs=True,
                    skip_decode=True,
                    a3_out_layout=2,
                )
                self.assertTrue(kw_a3.get("_a3_out_layout", False))


if __name__ == "__main__":
    unittest.main()
