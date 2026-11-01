"""Unit tests for the TPU Pallas RPA v3 backend."""

import contextlib
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.attention_pallas_rpa_base import (
    _get_pages_per_seq,
    _zero_inactive_rpa_rows,
)
from sglang_tpu.attention_pallas_rpa_v3 import (
    RPA_V3_SPEC,
    TpuPallasRpaV3AttnBackend,
    _get_smem_estimate_bytes,
)
from sglang_tpu.envs import envs
from sglang_tpu.rpa_smem_budget import (
    SCOPED_SMEM_BYTES,
    _max_smem_seqs,
    page_table_smem_bytes,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


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


class TestRpaSmemBudget(CustomTestCase):
    def test_v3_smem_golden_bytes_and_max_seqs(self):
        # Hand-computed 9-buffer scalar-prefetch totals in bytes:
        # (align128(n) + align128(n*p) + 3*align128(n+1) + 512) * 4.
        for n, p, expected_bytes in (
            (1, 1, 4608),
            (63, 4097, 1036800),
            (64, 4097, 1053184),
            (63, 4090, 1035264),
            (64, 4090, 1051136),
            (15, 16385, 987648),
            (16, 16385, 1053184),
        ):
            with self.subTest(n=n, p=p):
                self.assertEqual(_get_smem_estimate_bytes(n, p), expected_bytes)

        # At (64, 4090), page_indices alone is 1,047,040 B <= 1 MiB, while the
        # full 9-buffer kernel estimate is 1,051,136 B > 1 MiB (max_smem_seqs=63).
        self.assertEqual(page_table_smem_bytes(64, 4090), 1047040)
        self.assertLessEqual(page_table_smem_bytes(64, 4090), SCOPED_SMEM_BYTES)
        self.assertGreater(_get_smem_estimate_bytes(64, 4090), SCOPED_SMEM_BYTES)
        estimator = RPA_V3_SPEC.smem_estimate_bytes
        self.assertEqual(_max_smem_seqs(estimate_fn=estimator, pages_per_seq=4090), 63)
        self.assertEqual(_max_smem_seqs(estimate_fn=estimator, pages_per_seq=4097), 63)
        self.assertEqual(_max_smem_seqs(estimate_fn=estimator, pages_per_seq=16385), 15)
        self.assertEqual(_max_smem_seqs(estimate_fn=estimator, pages_per_seq=262144), 0)
        for bad_pages in (0, -4):
            with self.assertRaisesRegex(ValueError, "pages_per_seq must be positive"):
                _max_smem_seqs(estimate_fn=estimator, pages_per_seq=bad_pages)

    def test_entries_padded_to_128(self):
        self.assertEqual(page_table_smem_bytes(1, 1), 128 * 4)
        self.assertEqual(page_table_smem_bytes(1, 128), 128 * 4)
        self.assertEqual(page_table_smem_bytes(1, 129), 256 * 4)
        self.assertEqual(page_table_smem_bytes(2, 100), 256 * 4)


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
        context_len=None,
        torch_compile_max_bs=32,
        scoped_override=False,
        **server_arg_overrides,
    ):
        context_len = self.CONTEXT_LEN if context_len is None else context_len
        pool_width = self.POOL_WIDTH if pool_width is None else pool_width
        merged_args = dict(
            speculative_num_draft_tokens=self.DRAFT_TOKENS,
            speculative_algorithm="EAGLE3",
            speculative_num_steps=None,
            speculative_eagle_topk=1,
            page_size=self.PAGE_SIZE,
            torch_compile_max_bs=torch_compile_max_bs,
            tpu_compile_tokens=None,
        )
        merged_args.update(server_arg_overrides)
        override = get_context().override_server_args(**merged_args)
        if not scoped_override:
            override.install()
            self.addCleanup(override.restore)
        server_args = SimpleNamespace(**merged_args)
        model_runner = SimpleNamespace(
            device=torch.device("cpu"),
            page_size=self.PAGE_SIZE,
            model_config=SimpleNamespace(context_len=context_len),
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
            override if scoped_override else contextlib.nullcontext(),
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

    def _assert_smem_budget(
        self,
        *,
        ctx: int,
        bs: int,
        err: str | tuple[str, ...] | None = None,
        absent: str | None = None,
        cp: int = 1,
        **overrides,
    ) -> None:
        dp = overrides.get("attn_dp_size", 1)
        par_ctx = (
            get_parallel().override(
                tp_size=8,
                attn_tp_size=8 // (dp * cp),
                attn_dp_size=dp,
                attn_dp_enabled=dp > 1,
                attn_cp_size=cp,
                moe_tp_size=8 // cp,
                moe_dp_size=cp,
                moe_ep_size=1,
            )
            if (cp > 1 or dp > 1)
            else get_parallel().override()
        )
        kwargs = dict(
            context_len=ctx,
            pool_width=ctx * 2,
            req_to_token_device="meta",
            torch_compile_max_bs=bs,
            scoped_override=True,
            **overrides,
        )
        with par_ctx:
            if err is None:
                self._make_backend(**kwargs)
                return
            with self.assertRaises(ValueError) as cm:
                self._make_backend(**kwargs)
        msg = str(cm.exception)
        for fragment in (err,) if isinstance(err, str) else err:
            self.assertIn(fragment, msg)
        if absent is not None:
            self.assertNotIn(absent, msg)

    def _make_minimal_runner(
        self,
        *,
        context_len: int,
        page_size: int = 64,
        pool_width: int = 524288,
        pool_device: str = "cpu",
    ) -> SimpleNamespace:
        return SimpleNamespace(
            device=torch.device("cpu"),
            page_size=page_size,
            model_config=SimpleNamespace(
                context_len=context_len, hf_config=SimpleNamespace()
            ),
            req_to_token_pool=SimpleNamespace(
                req_to_token=torch.zeros(
                    1, pool_width, dtype=torch.int32, device=pool_device
                )
            ),
            token_to_kv_pool=SimpleNamespace(),
            is_hybrid_swa=False,
            kv_cache_dtype=torch.bfloat16,
            is_draft_worker=False,
            spec_algorithm=SpeculativeAlgorithm.NONE,
        )

    def test_smem_budget_colocated_and_explicit_bs_buckets(self):
        for ctx, max_ok, bad_bs in (
            (65472, 254, 255),
            (261632, 63, 64),
            (262144, 63, 64),
            (1048576, 15, 32),
        ):
            with self.subTest(ctx=ctx):
                self._assert_smem_budget(ctx=ctx, bs=max_ok)
                self._assert_smem_budget(
                    ctx=ctx,
                    bs=bad_bs,
                    err=(
                        "lower --max-running-requests (and "
                        f"--torch-compile-max-bs if set) to <= {max_ok}"
                    ),
                )
        self._assert_smem_budget(
            ctx=262144,
            bs=16,
            tpu_compile_bs=[1, 16, 64],
            err=(
                "drop any --tpu-compile-bs entries above 63",
                "lower --max-running-requests (and --torch-compile-max-bs if set) to <= 63",
            ),
        )

    def test_smem_budget_attention_dp_remedy_rounding(self):
        # dp=2, attn_tp=4, max_seqs=15 -> per-worker <= 12, global mrr <= 24.
        self._assert_smem_budget(ctx=1048576, bs=12, attn_dp_size=2)
        self._assert_smem_budget(
            ctx=1048576,
            bs=16,
            attn_dp_size=2,
            err=(
                "lower --max-running-requests to <= 24",
                "--torch-compile-max-bs to <= 12",
            ),
        )
        # ctx=4M -> max_seqs=3 < attn_tp=4 rounds per-worker cap to 0.
        self._assert_smem_budget(
            ctx=4194304,
            bs=4,
            attn_dp_size=2,
            err="raise --page-size or lower --context-length",
        )

    def test_smem_budget_disaggregated_prefill_and_decode(self):
        self._assert_smem_budget(
            ctx=1048576, bs=64, disaggregation_mode="prefill", prefill_max_requests=8
        )
        self._assert_smem_budget(
            ctx=1048576,
            bs=64,
            disaggregation_mode="prefill",
            prefill_max_requests=32,
            err="lower --prefill-max-requests to <= 15",
        )
        self._assert_smem_budget(
            ctx=1048576,
            bs=32,
            disaggregation_mode="prefill",
            prefill_max_requests=8,
            attn_dp_size=2,
            err=(
                "lower --max-running-requests to <= 24",
                "--torch-compile-max-bs to <= 12",
            ),
        )
        self._assert_smem_budget(
            ctx=1048576,
            bs=32,
            disaggregation_mode="prefill",
            prefill_max_requests=8,
            enable_mixed_chunk=True,
            err="lower --max-running-requests (and --torch-compile-max-bs if set) to <= 15",
        )
        self._assert_smem_budget(
            ctx=1048576,
            bs=32,
            disaggregation_mode="decode",
            prefill_max_requests=4,
            err="lower --max-running-requests (and --torch-compile-max-bs if set) to <= 15",
        )

    def test_smem_budget_colocated_context_parallel(self):
        mrr_63 = (
            "lower --max-running-requests (and --torch-compile-max-bs if set) to <= 63"
        )
        self._assert_smem_budget(ctx=131072, bs=64, cp=2, err=mrr_63)
        self._assert_smem_budget(
            ctx=131072,
            bs=80,
            cp=2,
            prefill_max_requests=64,
            err="lower --prefill-max-requests to <= 63",
        )
        self._assert_smem_budget(
            ctx=262144,
            bs=64,
            cp=2,
            err=(
                "128 sequences",
                "lower --max-running-requests (and --torch-compile-max-bs if set) to <= 31",
            ),
        )
        self._assert_smem_budget(
            ctx=262144,
            bs=64,
            cp=2,
            prefill_max_requests=60,
            err=("lower --prefill-max-requests to <= 31", mrr_63),
        )
        self._assert_smem_budget(
            ctx=262144,
            bs=63,
            cp=2,
            prefill_max_requests=60,
            err="lower --prefill-max-requests to <= 31",
            absent="--max-running-requests",
        )
        self._assert_smem_budget(
            ctx=262144,
            bs=64,
            cp=2,
            prefill_max_requests=31,
            err=mrr_63,
            absent="--prefill-max-requests",
        )
        self._assert_smem_budget(ctx=262144, bs=63, cp=2, prefill_max_requests=31)
        self._assert_smem_budget(
            ctx=131072,
            bs=64,
            cp=2,
            prefill_max_requests=8,
            attn_dp_size=2,
            err=(
                "lower --max-running-requests to <= 124",
                "--torch-compile-max-bs to <= 62",
            ),
        )

    def test_smem_budget_prefill_cp_and_zero_cap_edges(self):
        self._assert_smem_budget(
            ctx=1048576,
            bs=64,
            cp=2,
            disaggregation_mode="prefill",
            prefill_max_requests=7,
        )
        self._assert_smem_budget(
            ctx=1048576,
            bs=64,
            cp=2,
            disaggregation_mode="prefill",
            prefill_max_requests=8,
            err="lower --prefill-max-requests to <= 7",
        )
        page_remedy = "raise --page-size or lower --context-length"
        self._assert_smem_budget(ctx=33554432, bs=1, err=page_remedy, absent="<=")
        self._assert_smem_budget(
            ctx=33554432,
            bs=1,
            disaggregation_mode="prefill",
            prefill_max_requests=1,
            err=page_remedy,
            absent="<=",
        )
        # Tied zero caps report the larger family (256 decode vs 128 CP rows).
        self._assert_smem_budget(
            ctx=33554432,
            bs=256,
            cp=2,
            tpu_compile_tokens=[64],
            err=("RPA kernel for 256 sequences", page_remedy),
            absent="for 128 sequences",
        )
        self._assert_smem_budget(ctx=33554432, bs=0)

    def _assert_cp_metadata_bound(
        self,
        backend,
        *,
        fits_pad_b: int,
        pages_per_seq: int,
        error_regex: str,
    ) -> None:
        from sglang_tpu.cp_layout import build_tpu_cp_layout

        for pad_b, should_raise in ((fits_pad_b, False), (fits_pad_b + 1, True)):
            host = build_tpu_cp_layout(
                extend_lens=[16],
                prefix_lens=[0],
                cp_size=2,
                cp_rank=0,
                pad_b=pad_b,
                local_rows_padded=16,
            )
            cp_layout = SimpleNamespace(host=host)
            fb = SimpleNamespace(
                seq_lens=torch.ones(pad_b, dtype=torch.int32),
                req_pool_indices_cpu=torch.zeros(pad_b, dtype=torch.int64),
            )
            if should_raise:
                with self.assertRaisesRegex(ValueError, error_regex):
                    backend._build_cp_metadata(fb, cp_layout)
            else:
                meta = backend._build_cp_metadata(fb, cp_layout)
                self.assertEqual(
                    meta.page_indices.numel(), 2 * fits_pad_b * pages_per_seq
                )

    def test_cp_metadata_uses_v3_smem_estimate(self):
        # At 261632 (+ EAGLE3 extra -> 4090 pages/seq), 2 * 31 = 62 rows fits
        # while 2 * 32 = 64 rows raises ValueError under the 9-buffer v3 model.
        backend = self._make_backend(
            context_len=261632,
            pool_width=262144,
            req_to_token_device="cpu",
            torch_compile_max_bs=32,
        )
        self.assertEqual(_get_pages_per_seq(backend.max_context_len, 64), 4090)
        self._assert_cp_metadata_bound(
            backend,
            fits_pad_b=31,
            pages_per_seq=4090,
            error_regex=r"RPA kernel for 64 context-parallel rows x 4090 pages.*above",
        )

    def test_unmodeled_subclasses_warn_instead_of_raise(self):
        from sglang_tpu.attention_pallas_mla import TpuPallasMlaAttnBackend
        from sglang_tpu.attention_pallas_rpa_batched import (
            TpuPallasRpaBatchedAttnBackend,
        )

        for cls, page_size, patch_target in (
            (
                TpuPallasRpaBatchedAttnBackend,
                256,
                "sglang_tpu.attention_pallas_rpa_base._build_pallas_kernel_op",
            ),
            (
                TpuPallasMlaAttnBackend,
                64,
                "sglang_tpu.attention_pallas_mla._build_pallas_mla_kernel_op",
            ),
        ):
            with self.subTest(cls=cls.__name__):
                runner = self._make_minimal_runner(
                    context_len=4194304,
                    page_size=page_size,
                    pool_width=8388608,
                    pool_device="meta",
                )
                with (
                    get_context().override_server_args(
                        page_size=page_size, torch_compile_max_bs=64
                    ),
                    patch(patch_target, return_value=None),
                    self.assertLogs(
                        "sglang_tpu.rpa_smem_budget", level="WARNING"
                    ) as logs,
                ):
                    cls(runner)
                self.assertTrue(
                    any("above the 1 MiB scoped budget" in m for m in logs.output)
                )

    def test_mla_cp_uses_page_table_bound(self):
        from sglang_tpu.attention_pallas_mla import TpuPallasMlaAttnBackend

        # MLA overrides _smem_estimator() -> None, warning in __init__ and using
        # page-table sizing in _build_cp_metadata.
        mla_cp_runner = self._make_minimal_runner(context_len=261696)
        with (
            get_context().override_server_args(
                page_size=64,
                torch_compile_max_bs=40,
                prefill_max_requests=35,
            ),
            get_parallel().override(
                tp_size=4,
                attn_tp_size=2,
                attn_dp_size=1,
                attn_dp_enabled=False,
                attn_cp_size=2,
                moe_tp_size=2,
                moe_dp_size=2,
                moe_ep_size=1,
            ),
            patch(
                "sglang_tpu.attention_pallas_mla._build_pallas_mla_kernel_op",
                return_value=None,
            ),
            self.assertLogs("sglang_tpu.rpa_smem_budget", level="WARNING") as mla_logs,
        ):
            mla_inst = TpuPallasMlaAttnBackend(mla_cp_runner)
        self.assertTrue(any("--prefill-max-requests" in m for m in mla_logs.output))
        self.assertEqual(_get_pages_per_seq(mla_inst.max_context_len, 64), 4090)
        self._assert_cp_metadata_bound(
            mla_inst,
            fits_pad_b=32,
            pages_per_seq=4090,
            error_regex=(
                r"RPA page table for 66 context-parallel rows x 4090 pages.*above"
            ),
        )

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


if __name__ == "__main__":
    unittest.main()
