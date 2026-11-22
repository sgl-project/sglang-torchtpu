# SPDX-License-Identifier: Apache-2.0
"""Unit and CPU Pallas interpret-mode tests for Wave-9 host/scheduler & shared8 decode opts."""

from __future__ import annotations

import os
import py_compile
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang_tpu.envs import envs as tpu_envs


@contextmanager
def _temp_env(**kwargs: str):
    old = {k: os.environ.get(k) for k in kwargs}
    try:
        for k, v in kwargs.items():
            os.environ[k] = v
        yield
    finally:
        for k, prev in old.items():
            if prev is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = prev


def test_w9_syntax_and_default_off_flags():
    for rel in (
        "sglang_tpu/envs.py",
        "sglang_tpu/attention_shared_prefix.py",
        "sglang_tpu/attention_pallas_rpa_base.py",
        "sglang_tpu/host_metadata_cache.py",
        "sglang_tpu/compilation/compiled_forward_shim.py",
        "sglang_tpu/sampler.py",
        "python/sglang/srt/mem_cache/memory_pool.py",
        "python/sglang/srt/mem_cache/allocation.py",
        "python/sglang/srt/managers/schedule_batch.py",
    ):
        py_compile.compile(rel, doraise=True)

    assert tpu_envs.SGLANG_TPU_DECODE_FAST_HOST.get() is False
    assert tpu_envs.SGLANG_TPU_RPA_SHARED_BS8.get() is False


def test_shared8_gating_and_distribution():
    from sglang_tpu.attention_pallas_rpa_base import BaseTpuPallasRpaAttnBackend
    from sglang_tpu.attention_shared_prefix import (
        compute_shared_mismatch,
        shared_decode_distribution,
    )

    pages = np.broadcast_to(np.arange(160, dtype=np.int32), (8, 160)).copy()
    pages[:, 128:] += np.arange(8, dtype=np.int32)[:, None] * 100
    lengths = np.full((8,), 128 * 256 + 64, dtype=np.int32)
    mm = compute_shared_mismatch(pages, 8, group_size=8)
    assert mm.tolist() == [128]
    plan = shared_decode_distribution(
        pages, lengths, 8, 256, group_size=8, cached_mismatch=mm
    )
    assert plan is not None
    assert plan.tolist() == [8, 8, 8, 128]

    backend = object.__new__(BaseTpuPallasRpaAttnBackend)
    backend._dcp = 1
    backend.page_size = 256
    backend.max_context_len = 256 * 256
    backend.kernel_spec = SimpleNamespace(supports_shared_decode=True)
    backend.req_to_token_pool = SimpleNamespace(
        req_to_token=torch.zeros((1,), device="cpu")
    )

    with (
        patch("sglang_tpu.attention_pallas_rpa_base.get_parallel", return_value=SimpleNamespace(attn_dp_size=1)),
        _temp_env(
            SGLANG_TPU_RPA_V3_DECODE_FAST="shared64,c4096",
            SGLANG_TPU_RPA_V3_SKIP_EMPTY_MIXED="1",
        ),
    ):
        assert backend.shared_decode_group_size(8) == 0
        assert backend.supports_shared_decode_batch(8) is False
        assert backend.supports_shared_decode_batch(16) is True

        with _temp_env(SGLANG_TPU_RPA_SHARED_BS8="1"):
            assert backend.shared_decode_group_size(4) == 0
            assert backend.supports_shared_decode_batch(4) is False
            assert backend.shared_decode_group_size(8) == 8
            assert backend.supports_shared_decode_batch(8) is True
            assert backend.shared_decode_group_size(16) == 16
            assert backend.shared_decode_group_size(24) == 8
            assert backend.shared_decode_group_size(64) == 64


def test_shared8_interpret_parity_ps128_and_ps256():
    from sglang_tpu.attention.pallas.rpa_v3_decode_fast import run_decode_fast_rpa

    rng = np.random.default_rng(42)
    bs = 8
    for page_size in (128, 256):
        shared_pgs = 2048 // page_size
        pps = shared_pgs + 4
        phys_rows = page_size // 2
        n_phys = 128
        d2_cache = jnp.asarray(
            (rng.standard_normal((n_phys, phys_rows, 1, 4, 128)) * 0.35).astype(np.float32),
            dtype=jnp.float8_e4m3fn,
        )
        page_indices = np.zeros((bs, pps), dtype=np.int32)
        page_indices[:, :shared_pgs] = np.arange(1, 1 + shared_pgs, dtype=np.int32)
        nxt = 1 + shared_pgs
        for s in range(bs):
            for p in range(shared_pgs, pps):
                page_indices[s, p] = nxt
                nxt += 1

        kv_lens = np.full((bs,), shared_pgs * page_size + 211, dtype=np.int32)
        kv_lens[0] += 13
        kv_lens[3] -= 19
        q = jnp.asarray(rng.standard_normal((bs, 16, 128)) * 0.35, dtype=jnp.bfloat16)
        k_new = jnp.asarray(rng.standard_normal((bs, 1, 128)) * 0.35, dtype=jnp.float8_e4m3fn)
        v_new = jnp.asarray(rng.standard_normal((bs, 1, 128)) * 0.35, dtype=jnp.float8_e4m3fn)
        gate = jnp.asarray(rng.standard_normal((bs, 1024)) * 0.35, dtype=jnp.bfloat16)
        cu_q = jnp.arange(bs + 1, dtype=jnp.int32)
        cu_kv = jnp.arange(bs + 1, dtype=jnp.int32) * (pps * page_size)
        dist = jnp.asarray([bs, bs, bs], dtype=jnp.int32)

        out_ref, _ = run_decode_fast_rpa(
            q, k_new, v_new, d2_cache, jnp.asarray(kv_lens), jnp.asarray(page_indices.ravel()),
            cu_q, cu_kv, dist, sm_scale=0.125, sliding_window=None, k_scale=1.0, v_scale=1.0,
            mode_str="w8,c4096,pbf16", decode_query_size=1, fused_out_pack=2,
            gate=gate, out_norm_eps=1e-6, kv_pack_dense2=True, interpret=True, shared_pages=None,
        )
        out_sh8, _ = run_decode_fast_rpa(
            q, k_new, v_new, d2_cache, jnp.asarray(kv_lens), jnp.asarray(page_indices.ravel()),
            cu_q, cu_kv, dist, sm_scale=0.125, sliding_window=None, k_scale=1.0, v_scale=1.0,
            mode_str="shared64,c4096,pbf16,w8", decode_query_size=1, fused_out_pack=2,
            gate=gate, out_norm_eps=1e-6, kv_pack_dense2=True, interpret=True,
            shared_pages=jnp.asarray([shared_pgs], dtype=jnp.int32),
        )
        ref_f = np.asarray(out_ref, dtype=np.float32).ravel()
        sh_f = np.asarray(out_sh8, dtype=np.float32).ravel()
        max_abs = float(np.max(np.abs(ref_f - sh_f)))
        cos = float(np.dot(ref_f, sh_f) / (np.linalg.norm(ref_f) * np.linalg.norm(sh_f)))
        assert max_abs < 0.02, f"ps={page_size} max_abs={max_abs}"
        assert cos > 0.9999, f"ps={page_size} cos={cos}"


def _make_backend(is_hybrid_swa: bool, cache):
    from sglang_tpu.attention_pallas_rpa_base import BaseTpuPallasRpaAttnBackend

    backend = object.__new__(BaseTpuPallasRpaAttnBackend)
    backend._dcp = 1
    backend._page_stride = 128
    backend._host_page_cols = None
    backend._is_hybrid_swa = is_hybrid_swa
    backend._swa_group_pages = None
    backend._host_metadata_cache = cache
    backend._last_query_metadata = None
    backend._is_extend_including_draft = lambda mode: False
    backend.supports_shared_decode_batch = lambda bs: False
    backend.token_to_kv_pool = SimpleNamespace(
        full_to_swa_index_mapping=(
            (torch.arange(4096, dtype=torch.int64) ^ 128) if is_hybrid_swa else None
        )
    )
    return backend


def test_incremental_decode_page_table_patch_matches_full_gather():
    from sglang_tpu.host_metadata_cache import HostMetadataCache

    for is_hybrid in (False, True):
        pool = ReqToTokenPool(size=8, max_context_len=512, device="tpu", enable_memory_saver=False)
        pool.write((1, slice(0, 128)), torch.arange(128, 256, dtype=torch.int32))
        pool.write((2, slice(0, 128)), torch.arange(256, 384, dtype=torch.int32))
        pool.write((3, slice(0, 256)), torch.arange(384, 640, dtype=torch.int32))

        fast_be = _make_backend(is_hybrid, HostMetadataCache())
        ref_be = _make_backend(is_hybrid, None)

        req_rows = torch.tensor([1, 2, 3], dtype=torch.int32)
        dist = torch.tensor([3, 3, 3], dtype=torch.int32)

        def _fb(lens):
            return SimpleNamespace(
                forward_mode=ForwardMode.DECODE,
                spec_info=None,
                seq_lens=torch.tensor(lens, dtype=torch.int32),
                seq_lens_cpu=torch.tensor(lens, dtype=torch.int32),
                req_pool_indices_cpu=req_rows,
            )

        with _temp_env(SGLANG_TPU_RPA_CACHE_DECODE_PAGE_TABLE="1"):
            # Step 1: cold miss
            m1 = fast_be._build_metadata_host(_fb([128, 128, 256]), 128, "cpu", 3, pool.req_to_token, 4, dist, 3)
            r1 = ref_be._build_metadata_host(_fb([128, 128, 256]), 128, "cpu", 3, pool.req_to_token, 4, dist, 3)
            assert torch.equal(m1.page_indices, r1.page_indices)
            if is_hybrid:
                assert torch.equal(m1.swa_page_indices, r1.swa_page_indices)

            # Step 2: decode step crossing page boundary on rows 1 and 3 (locs=128 and 256)
            locs = torch.tensor([128, 64, 256], dtype=torch.int32)
            vals = torch.tensor([1024, 320, 1536], dtype=torch.int32)
            pool.write((req_rows, locs), vals, bump_page_table=True, is_decode_step=True)
            assert pool.req_to_token._tpu_last_decode_patch is not None

            m2 = fast_be._build_metadata_host(_fb([129, 129, 257]), 128, "cpu", 3, pool.req_to_token, 4, dist, 3)
            r2 = ref_be._build_metadata_host(_fb([129, 129, 257]), 128, "cpu", 3, pool.req_to_token, 4, dist, 3)
            assert torch.equal(m2.page_indices, r2.page_indices)
            if is_hybrid:
                assert torch.equal(m2.swa_page_indices, r2.swa_page_indices)

            # Step 3: intra-page decode step (bump_page_table=False) -> cache hit
            pool.write(
                (req_rows, torch.tensor([129, 65, 257], dtype=torch.int32)),
                torch.tensor([1025, 321, 1537], dtype=torch.int32),
                bump_page_table=False,
                is_decode_step=True,
            )
            m3 = fast_be._build_metadata_host(_fb([130, 130, 258]), 128, "cpu", 3, pool.req_to_token, 4, dist, 3)
            assert m3.page_indices is m2.page_indices

            # Step 4: non-decode write invalidates _tpu_last_decode_patch -> clean fallback
            pool.write((1, slice(256, 384)), torch.arange(2048, 2176, dtype=torch.int32), bump_page_table=True)
            assert pool.req_to_token._tpu_last_decode_patch is None
            m4 = fast_be._build_metadata_host(_fb([260, 130, 258]), 128, "cpu", 3, pool.req_to_token, 4, dist, 3)
            r4 = ref_be._build_metadata_host(_fb([260, 130, 258]), 128, "cpu", 3, pool.req_to_token, 4, dist, 3)
            assert torch.equal(m4.page_indices, r4.page_indices)
            if is_hybrid:
                assert torch.equal(m4.swa_page_indices, r4.swa_page_indices)


def test_host_metadata_cache_own_cpu_and_skip_equal():
    from sglang_tpu.host_metadata_cache import HostMetadataCache

    c = HostMetadataCache()
    t1 = torch.tensor([1, 2, 3], dtype=torch.int32)
    r1 = c.get_by_value("k", t1, torch.device("cpu"), own_cpu=True)
    assert c._by_value["k"][0] is t1
    t2 = torch.tensor([1, 2, 3], dtype=torch.int32)
    r2 = c.get_by_value("k", t2, torch.device("cpu"), own_cpu=True, skip_equal=True)
    assert r2 is not r1
    assert c._by_value["k"][0] is t2


def test_pad_batch_fields_skips_fused_eager_when_unpadded():
    from sglang_tpu.compilation.bucketer import Bucketer
    from sglang_tpu.compilation.compiled_forward_shim import (
        _pad_forward_batch_inputs,
        _restore_forward_batch_inputs,
    )

    t_b = Bucketer("t", [64, 128])
    b_b = Bucketer("b", [32, 64])

    def _fb(bs):
        fb = ForwardBatch(
            forward_mode=ForwardMode.DECODE,
            batch_size=bs,
            input_ids=torch.zeros((bs,), dtype=torch.int32),
            req_pool_indices=torch.arange(bs, dtype=torch.int32),
            seq_lens=torch.full((bs,), 512, dtype=torch.int32),
            out_cache_loc=torch.arange(bs, dtype=torch.int32),
            seq_lens_sum=bs * 512,
            return_logprob=False,
            capture_hidden_mode=CaptureHiddenMode.NULL,
        )
        fb.positions = torch.full((bs,), 511, dtype=torch.int32)
        fb.orig_seq_lens = torch.full((bs,), 512, dtype=torch.int32)
        fb.seq_lens_cpu = torch.full((bs,), 512, dtype=torch.int32)
        fb.req_pool_indices_cpu = torch.arange(bs, dtype=torch.int32)
        return fb

    calls = 0

    @contextmanager
    def _fake_fused(enabled=True):
        nonlocal calls
        calls += 1
        yield False

    with patch("sglang_tpu.compilation.compiled_forward_shim.fused_eager_ops", side_effect=_fake_fused):
        fb64 = _fb(64)
        rq, rb = _pad_forward_batch_inputs(fb64, t_b, b_b)
        assert calls == 0
        _restore_forward_batch_inputs(fb64, rq, rb)

        fb63 = _fb(63)
        rq, rb = _pad_forward_batch_inputs(fb63, t_b, b_b)
        assert calls == 1
        assert fb63.input_ids.shape[0] == 64
        _restore_forward_batch_inputs(fb63, rq, rb)
        assert fb63.input_ids.shape[0] == 63


def test_decode_fast_host_prepare_for_decode_and_sampler_parity():
    from sglang_tpu.sampler import sample_from_global_top_k_logits

    def _make_sb():
        sb = ScheduleBatch(
            reqs=[],
            req_to_token_pool=None,
            token_to_kv_pool_allocator=None,
            tree_cache=None,
            model_config=SimpleNamespace(is_encoder_decoder=False),
            sampling_info=SimpleNamespace(
                penalizer_orchestrator=SimpleNamespace(is_required=False)
            ),
            spec_algorithm=SpeculativeAlgorithm.NONE,
            device="tpu",
            seq_lens=torch.tensor([10, 20], dtype=torch.int32),
            seq_lens_cpu=torch.tensor([10, 20], dtype=torch.int32),
            orig_seq_lens=torch.tensor([10, 20], dtype=torch.int32),
        )
        sb.hisparse_coordinator = None
        return sb

    with (
        patch("sglang.srt.managers.schedule_batch.alloc_for_decode", return_value=torch.tensor([1, 2])),
        patch("sglang.srt.managers.schedule_batch.strip_beam_tail"),
        patch("sglang.srt.managers.schedule_batch.append_beam_tail"),
        patch(
            "sglang.srt.managers.schedule_batch.get_exec",
            return_value=SimpleNamespace(mamba=SimpleNamespace(enable_mamba_extra_buffer=False)),
        ),
    ):
        sb0 = _make_sb()
        with _temp_env(SGLANG_TPU_DECODE_FAST_HOST="0"):
            sb0.prepare_for_decode()
        assert sb0.seq_lens.tolist() == [11, 21]
        assert sb0.orig_seq_lens.tolist() == [11, 21]

        sb1 = _make_sb()
        with _temp_env(SGLANG_TPU_DECODE_FAST_HOST="1"):
            sb1.prepare_for_decode()
        assert sb1.seq_lens.tolist() == [11, 21]
        assert sb1.seq_lens_cpu.tolist() == [11, 21]
        assert sb1.orig_seq_lens.tolist() == [10, 20]

    # Sampler parity (mock device.type == "tpu" check via fast_host helper path)
    logits = torch.randn((8, 16), dtype=torch.float32)
    temps = torch.full((8, 1), 0.7, dtype=torch.float32)
    top_ks = torch.tensor([1, 4, 8, 16, 2, 6, 12, 16], dtype=torch.int32)
    torch.manual_seed(123)
    ref = sample_from_global_top_k_logits(logits, temps, top_ks)
    with _temp_env(SGLANG_TPU_DECODE_FAST_HOST="1"):
        torch.manual_seed(123)
        fast = sample_from_global_top_k_logits(logits, temps, top_ks)
    assert torch.equal(ref, fast)


def test_piggyback_single_token_prefill_with_real_req():
    from types import SimpleNamespace
    from unittest.mock import MagicMock, patch
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.sampling.sampling_params import SamplingParams

    req = Req(
        rid="r0",
        origin_input_text="hi",
        origin_input_ids=[42],
        sampling_params=SamplingParams(max_new_tokens=4),
    )
    req.extend_end = 1
    req.prefix_len = 0
    assert req.extend_len == 1

    sched = object.__new__(Scheduler)
    sched.is_generation = True
    sched.spec_algorithm = SpeculativeAlgorithm.NONE
    sched.require_mlp_sync = False
    sched.enable_hisparse = False
    sched.dllm_config = None
    sched.chunked_reqs = []
    sched.max_running_requests = 64
    sched.future_map = MagicMock()
    sched.token_to_kv_pool_allocator = None

    new_batch = ScheduleBatch(
        reqs=[req],
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
        tree_cache=None,
        model_config=SimpleNamespace(is_encoder_decoder=False),
        spec_algorithm=SpeculativeAlgorithm.NONE,
        device="cpu",
        forward_mode=ForwardMode.EXTEND,
        req_pool_indices=torch.tensor([3], dtype=torch.int64),
        req_pool_indices_cpu=torch.tensor([3], dtype=torch.int64),
        seq_lens=torch.tensor([1], dtype=torch.int32),
        seq_lens_cpu=torch.tensor([1], dtype=torch.int32),
        orig_seq_lens=torch.tensor([1], dtype=torch.int32),
        out_cache_loc=torch.tensor([10], dtype=torch.int32),
    )
    new_batch.prefill_input_ids_cpu = torch.tensor([42], dtype=torch.int64)
    new_batch.multimodal_inputs = None

    running_batch = ScheduleBatch(
        reqs=[],
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
        tree_cache=None,
        model_config=SimpleNamespace(is_encoder_decoder=False),
        spec_algorithm=SpeculativeAlgorithm.NONE,
        device="cpu",
    )

    with (
        patch("sglang.srt.managers.scheduler.current_platform", SimpleNamespace(is_tpu=lambda: True)),
        patch(
            "sglang.srt.managers.scheduler.get_parallel",
            return_value=SimpleNamespace(pp_size=1, attn_cp_size=1, attn_dcp_size=1),
        ),
        patch(
            "sglang.srt.managers.scheduler.get_exec",
            return_value=SimpleNamespace(mamba=SimpleNamespace(enable_mamba_extra_buffer=False)),
        ),
        _temp_env(SGLANG_TPU_PIGGYBACK_SINGLE_TOKEN_PREFILL="1"),
    ):
        nb, rb, piggy = sched._maybe_piggyback_single_token_prefill(new_batch, running_batch)
    assert nb is None
    assert piggy is rb
    assert rb.forward_mode == ForwardMode.DECODE
    assert getattr(req, "_tpu_piggybacked_prefill", False) is True


def test_pack_prefill_to_bucket_with_real_req():
    from types import SimpleNamespace
    from unittest.mock import patch
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.sampling.sampling_params import SamplingParams

    chunked_req = Req(
        rid="c0",
        origin_input_text="x",
        origin_input_ids=list(range(10000)),
        sampling_params=SamplingParams(max_new_tokens=4),
    )
    chunked_req.prefix_len = 8192

    waiting_req = Req(
        rid="w0",
        origin_input_text="y",
        origin_input_ids=list(range(1000)),
        sampling_params=SamplingParams(max_new_tokens=4),
    )
    waiting_req.num_matched_prefix_tokens = 0

    running_req = Req(
        rid="r_run",
        origin_input_text="z",
        origin_input_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_new_tokens=4),
    )

    sched = object.__new__(Scheduler)
    sched.chunked_prefill_size = 16384
    sched.page_size = 256
    sched.is_mixed_chunk = False
    sched.require_mlp_sync = False
    sched.max_running_requests = 64
    sched.chunked_reqs = [chunked_req]
    sched.waiting_queue = [waiting_req]
    sched._tpu_prefill_delay_steps = 0
    sched._tpu_prefill_drain_grace_steps = 0

    running_batch = SimpleNamespace(
        reqs=[running_req] * 62,
        is_empty=lambda: False,
        is_prefill_only=False,
    )

    with (
        patch("sglang.srt.managers.scheduler.current_platform", SimpleNamespace(is_tpu=lambda: True)),
        _temp_env(
            SGLANG_TPU_PACK_PREFILL_TO_BUCKET="1",
            SGLANG_TPU_PACK_PREFILL_MAX_DELAY_STEPS="14",
        ),
    ):
        # chunk_rem = 10000 - 8192 = 1808, queue_rem = 1000 -> pending = 2808 < 8448 -> defer!
        assert sched._should_defer_underfilled_prefill(running_batch) is True
        # Once prefix_len = 0 -> chunk_rem = 10000 >= 8448 -> do not defer!
        chunked_req.prefix_len = 0
        assert sched._should_defer_underfilled_prefill(running_batch) is False


def test_build_metadata_host_with_shared_decode_repeated_steps():
    from sglang_tpu.host_metadata_cache import HostMetadataCache

    bs, page_size, pps = 8, 256, 132
    pool = ReqToTokenPool(size=16, max_context_len=pps * page_size, device="tpu", enable_memory_saver=False)
    shared_tokens = torch.arange(128 * page_size, dtype=torch.int32) + page_size
    for r in range(bs):
        pool.write((r, slice(0, 128 * page_size)), shared_tokens)
        pool.write((r, slice(128 * page_size, 129 * page_size)), torch.arange(page_size, dtype=torch.int32) + (200 + r) * page_size)

    fast_be = _make_backend(False, HostMetadataCache())
    fast_be._page_stride = page_size
    fast_be.page_size = page_size
    fast_be.max_context_len = pps * page_size
    fast_be.supports_shared_decode_batch = lambda b: b >= 8 and b % 8 == 0
    fast_be.shared_decode_group_size = lambda b: 8

    req_rows = torch.arange(bs, dtype=torch.int32)
    dist = torch.tensor([bs, bs, bs], dtype=torch.int32)

    def _fb(lens):
        return SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            spec_info=None,
            seq_lens=torch.tensor(lens, dtype=torch.int32),
            seq_lens_cpu=torch.tensor(lens, dtype=torch.int32),
            req_pool_indices_cpu=req_rows,
        )

    with _temp_env(SGLANG_TPU_RPA_CACHE_DECODE_PAGE_TABLE="1"):
        # Step 1: cold miss -> _build_shared_dist populates shared_dist_cache
        m1 = fast_be._build_metadata_host(_fb([128 * page_size + 10] * bs), page_size, "cpu", bs, pool.req_to_token, pps, dist, bs)
        assert m1.full_decode_distribution is not None
        assert m1.full_decode_distribution.tolist() == [bs, bs, bs, 128]

        # Step 2: cache hit -> hits cached_dist is not None and np.array_equal(cached_dist[0], plan)
        m2 = fast_be._build_metadata_host(_fb([128 * page_size + 11] * bs), page_size, "cpu", bs, pool.req_to_token, pps, dist, bs)
        assert m2.full_decode_distribution is m1.full_decode_distribution

        # Step 3: incremental decode patch crossing page boundary
        locs = torch.full((bs,), 129 * page_size, dtype=torch.int32)
        vals = torch.tensor([(300 + i) * page_size for i in range(bs)], dtype=torch.int32)
        pool.write((req_rows, locs), vals, bump_page_table=True, is_decode_step=True)
        m3 = fast_be._build_metadata_host(_fb([129 * page_size + 1] * bs), page_size, "cpu", bs, pool.req_to_token, pps, dist, bs)
        assert m3.full_decode_distribution.tolist() == [bs, bs, bs, 128]


def test_async_d2h_non_blocking_when_pin_memory_unavailable():
    from sglang.srt.utils import device as device_utils

    calls = []

    class _FakeDeviceTensor:
        device = SimpleNamespace(type="tpu")

        def to(self, target, non_blocking=False):
            calls.append((target, non_blocking))
            return ("cpu_copy", target, non_blocking)

    with patch.object(device_utils, "is_pin_memory_available", return_value=False):
        res = device_utils.async_d2h(_FakeDeviceTensor())

    assert res == ("cpu_copy", "cpu", True)
    assert calls == [("cpu", True)]

    cpu_t = torch.tensor([1, 2], dtype=torch.int32)
    assert device_utils.async_d2h(cpu_t) is cpu_t


def test_hybrid_swa_live_mapping_reflects_alloc_without_version_bump():
    """Verify full_to_swa_index_mapping writes in alloc_extend/alloc_decode (which do not bump
    _tpu_swa_map_version) are immediately visible to both EXTEND and incremental DECODE metadata.
    """
    from sglang_tpu.host_metadata_cache import HostMetadataCache

    bs, page_size, pps = 2, 128, 4
    pool = ReqToTokenPool(size=4, max_context_len=pps * page_size, device="tpu", enable_memory_saver=False)
    full_to_swa = torch.zeros(16 * page_size, dtype=torch.int64)
    # Initial mapping: full page 1 -> SWA page 3, full page 2 -> SWA page 4.
    full_to_swa[1 * page_size : 2 * page_size] = torch.arange(3 * page_size, 4 * page_size)
    full_to_swa[2 * page_size : 3 * page_size] = torch.arange(4 * page_size, 5 * page_size)

    be = _make_backend(True, HostMetadataCache())
    be._page_stride = page_size
    be.page_size = page_size
    be.max_context_len = pps * page_size
    be._swa_group_pages = None
    be.token_to_kv_pool.full_to_swa_index_mapping = full_to_swa

    req_rows = torch.tensor([0, 1], dtype=torch.int32)
    pool.write((req_rows, torch.tensor([0, 0])), torch.tensor([1 * page_size, 2 * page_size], dtype=torch.int32))
    dist = torch.tensor([bs, bs, bs], dtype=torch.int32)

    with _temp_env(SGLANG_TPU_RPA_CACHE_DECODE_PAGE_TABLE="1"):
        # Step 1 (EXTEND): populates any caches at _tpu_swa_map_version == 0
        m_ext1 = be._build_metadata_host(
            SimpleNamespace(
                forward_mode=ForwardMode.EXTEND,
                spec_info=None,
                seq_lens=torch.tensor([10, 10], dtype=torch.int32),
                seq_lens_cpu=torch.tensor([10, 10], dtype=torch.int32),
                extend_seq_lens_cpu=torch.tensor([10, 10], dtype=torch.int32),
                req_pool_indices_cpu=req_rows,
            ),
            page_size, "cpu", bs, pool.req_to_token, pps, dist, bs,
        )
        assert m_ext1.swa_page_indices.tolist() == [3, 0, 0, 0, 4, 0, 0, 0]

        # Step 2 (alloc_extend without bumping _tpu_swa_map_version): full pages 5, 6 -> SWA pages 9, 10
        full_to_swa[5 * page_size : 6 * page_size] = torch.arange(9 * page_size, 10 * page_size)
        full_to_swa[6 * page_size : 7 * page_size] = torch.arange(10 * page_size, 11 * page_size)
        pool.write((req_rows, torch.tensor([0, 0])), torch.tensor([5 * page_size, 6 * page_size], dtype=torch.int32))

        m_ext2 = be._build_metadata_host(
            SimpleNamespace(
                forward_mode=ForwardMode.EXTEND,
                spec_info=None,
                seq_lens=torch.tensor([10, 10], dtype=torch.int32),
                seq_lens_cpu=torch.tensor([10, 10], dtype=torch.int32),
                extend_seq_lens_cpu=torch.tensor([10, 10], dtype=torch.int32),
                req_pool_indices_cpu=req_rows,
            ),
            page_size, "cpu", bs, pool.req_to_token, pps, dist, bs,
        )
        assert m_ext2.swa_page_indices.tolist() == [9, 0, 0, 0, 10, 0, 0, 0]

        # Step 3 (DECODE priming + incremental decode page boundary alloc without _tpu_swa_map_version bump)
        m_dec1 = be._build_metadata_host(
            SimpleNamespace(
                forward_mode=ForwardMode.DECODE,
                spec_info=None,
                seq_lens=torch.tensor([128, 128], dtype=torch.int32),
                seq_lens_cpu=torch.tensor([128, 128], dtype=torch.int32),
                req_pool_indices_cpu=req_rows,
            ),
            page_size, "cpu", bs, pool.req_to_token, pps, dist, bs,
        )
        assert m_dec1.swa_page_indices.tolist() == [9, 0, 0, 0, 10, 0, 0, 0]

        full_to_swa[7 * page_size : 8 * page_size] = torch.arange(11 * page_size, 12 * page_size)
        full_to_swa[8 * page_size : 9 * page_size] = torch.arange(12 * page_size, 13 * page_size)
        pool.write(
            (req_rows, torch.tensor([page_size, page_size])),
            torch.tensor([7 * page_size, 8 * page_size], dtype=torch.int32),
            bump_page_table=True,
            is_decode_step=True,
        )
        m_dec2 = be._build_metadata_host(
            SimpleNamespace(
                forward_mode=ForwardMode.DECODE,
                spec_info=None,
                seq_lens=torch.tensor([129, 129], dtype=torch.int32),
                seq_lens_cpu=torch.tensor([129, 129], dtype=torch.int32),
                req_pool_indices_cpu=req_rows,
            ),
            page_size, "cpu", bs, pool.req_to_token, pps, dist, bs,
        )
        assert m_dec2.swa_page_indices.tolist() == [9, 11, 0, 0, 10, 12, 0, 0]
