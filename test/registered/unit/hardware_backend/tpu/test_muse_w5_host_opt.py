# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Wave-5 TPU speculative host/dispatch optimizations (default-off)."""

from __future__ import annotations

import os
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from sglang.srt.compilation.compiled_op import (
    _canonical_tensor,
    _canonicalize_positional_args_fast,
)
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.overlap_utils import FutureMap, RelayPayload
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
    LogitsOutputMode,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang_tpu.envs import envs as tpu_envs
from sglang_tpu.experimental.speculative.block_info import HostPaddedBlockDraftInput
from sglang_tpu.experimental.speculative.block_spec_worker import TpuBlockSpecWorker
from sglang_tpu.experimental.speculative.spec_ops import (
    TARGET_CANDIDATES,
    accept_greedy_top_k_step,
    build_draft_block_inputs,
    build_verify_tokens,
    prepare_dflash_block_step,
    select_verify_rows,
)
from sglang_tpu.experimental.speculative.spec_types import (
    BucketState,
    DraftResult,
    SpecState,
    TpuSpecVerifyInput,
    VerifyRows,
)
from sglang_tpu.overlap_utils_tpu import (
    device_accept_resolved_tpu,
    flush_direct_handoff_tpu,
    resolve_device_accept_seq_lens_tpu,
)
from sglang_tpu.patches import _patch_future_map_direct_handoff


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


def test_w5_flags_default_off():
    for name in (
        "SGLANG_TPU_VERIFY_SKIP_GUARD_EVAL",
        "SGLANG_TPU_FAST_COMPILED_OP",
        "SGLANG_TPU_SPEC_DIRECT_HANDOFF",
        "SGLANG_TPU_DFLASH_FUSE_STEP_OPS",
        "SGLANG_TPU_SPEC_VERIFY_GREEDY_TOPK1",
        "SGLANG_TPU_RPA_CACHE_DECODE_PAGE_TABLE",
        "SGLANG_TPU_FAST_RESULT_COPY",
        "SGLANG_TPU_FAST_FORWARD_ISOLATION",
        "SGLANG_TPU_FAST_FILTER_FINISHED_BATCH",
        "SGLANG_TPU_SPEC_STEP_TIMERS",
    ):
        field = getattr(tpu_envs, name)
        assert field.get() is False, f"{name} must default to False"


def test_p1_canonical_tensor_fast_matches_reference():
    t1_contig = torch.arange(16, dtype=torch.int64)
    assert _canonical_tensor(t1_contig) is t1_contig

    t1_sliced = torch.arange(32, dtype=torch.int64)[4:20]
    c1 = _canonical_tensor(t1_sliced)
    assert c1.storage_offset() == 0 and c1.stride() == (1,)
    assert torch.equal(c1, t1_sliced)

    # Size-1 leading row with non-canonical stride (99, 1): PyTorch reports
    # is_contiguous()=True, whereas canonical placeholder stride must be (5, 1)!
    t2_size1_view = torch.arange(100, dtype=torch.int64).as_strided((1, 5), (99, 1))
    assert t2_size1_view.is_contiguous()
    assert t2_size1_view.stride() == (99, 1)
    c2 = _canonical_tensor(t2_size1_view)
    assert c2.storage_offset() == 0 and c2.stride() == (5, 1)
    assert torch.equal(c2, t2_size1_view)

    t2_contig = torch.zeros((8, 10), dtype=torch.int64)
    rows = VerifyRows(
        positions=t2_contig,
        cache_locs=t2_size1_view,
        flat_positions=t1_contig,
        flat_cache_locs=t1_sliced,
    )
    out = _canonicalize_positional_args_fast((t1_contig, rows, 5))
    assert out[0] is t1_contig
    assert isinstance(out[1], VerifyRows)
    assert out[1].positions is t2_contig
    assert out[1].cache_locs.stride() == (5, 1)
    assert out[1].flat_cache_locs.storage_offset() == 0
    assert out[2] == 5


def test_p1_verify_skip_guard_stance_covers_target_verify():
    from sglang_tpu.compilation.compiled_forward_shim import CompiledForwardShim

    entered = []

    class FakeStance:
        def armed(self):
            return True

        def __enter__(self):
            entered.append("enter")

        def __exit__(self, *exc):
            entered.append("exit")

    policy = SimpleNamespace(
        model_runner=SimpleNamespace(
            is_draft_worker=False,
            pp_group=SimpleNamespace(is_first_rank=True),
            token_to_kv_pool=SimpleNamespace(layer_transfer_counter=None),
            model=SimpleNamespace(),
        ),
        keep_outputs_padded=lambda fb: True,
        finalize_forward_batch=lambda out, fb: out,
        keeps_padded_mm_embeds=lambda fb: False,
    )

    def make_fb(mode: ForwardMode):
        return SimpleNamespace(
            forward_mode=mode,
            mm_input_embeds=None,
            spec_info=SimpleNamespace(keep_outputs_padded=True),
            input_embeds=None,
            is_prefill_only=False,
            return_logprob=False,
            top_logprobs_nums=None,
            token_ids_logprobs=None,
            extend_input_logprob_token_ids_gpu=None,
            extend_logprob_start_lens_cpu=None,
            input_ids=torch.zeros(10, dtype=torch.int64),
            seq_lens=torch.ones(2, dtype=torch.int64),
        )

    with _temp_env(SGLANG_TPU_VERIFY_SKIP_GUARD_EVAL="0"):
        shim_off = CompiledForwardShim(lambda *a, **kw: "ok", policy=policy)
        shim_off._skip_guard_stance = FakeStance()
        shim_off(torch.zeros(10, dtype=torch.int64), torch.zeros(10, dtype=torch.int64), make_fb(ForwardMode.TARGET_VERIFY))
        assert entered == []

    with _temp_env(SGLANG_TPU_VERIFY_SKIP_GUARD_EVAL="1"):
        shim_on = CompiledForwardShim(lambda *a, **kw: "ok", policy=policy)
        shim_on._skip_guard_stance = FakeStance()
        shim_on(torch.zeros(10, dtype=torch.int64), torch.zeros(10, dtype=torch.int64), make_fb(ForwardMode.TARGET_VERIFY))
        assert entered == ["enter", "exit"]
        entered.clear()
        # EXTEND must NOT enter decode/verify skip-guard stance.
        shim_on(torch.zeros(10, dtype=torch.int64), torch.zeros(10, dtype=torch.int64), make_fb(ForwardMode.EXTEND))
        assert entered == []


def test_p1_greedy_topk1_verify_mode_and_accept_parity():
    torch.manual_seed(0)
    bs, gamma = 8, 4
    rows = bs * (gamma + 1)
    # Build descending top-64 IDs and top-1 slice
    top64_ids = torch.randint(0, 1000, (rows, TARGET_CANDIDATES), dtype=torch.int32)
    top1_ids = top64_ids[:, :1].contiguous()
    draft_tokens = top64_ids[:, 0].view(bs, gamma + 1)[:, :gamma].to(torch.int64).clone()
    # Force mismatches at varying positions
    draft_tokens[1, 2] += 1
    draft_tokens[3, 0] += 1
    num_verify = torch.full((bs,), gamma, dtype=torch.int32)
    seq_lens = torch.full((bs,), 100, dtype=torch.int64)
    seq_lens[-2:] = 0
    live_mask = seq_lens > 0

    acc64 = accept_greedy_top_k_step(top64_ids, draft_tokens, num_verify, seq_lens, live_mask)
    acc1 = accept_greedy_top_k_step(top1_ids, draft_tokens, num_verify, seq_lens, live_mask)
    for f64, f1 in zip(acc64, acc1):
        assert torch.equal(f64, f1)

    # Check _verify_logits_output_mode routing on greedy vs sampled
    dummy_worker = SimpleNamespace(
        _verify_top_k_width=TARGET_CANDIDATES,
        _verify_greedy_topk1=True,
    )
    from sglang_tpu.experimental.speculative.base_spec_worker import TpuSpecWorker

    greedy_bucket = BucketState(
        key=torch.arange(bs),
        state=SpecState(
            bonus_tokens=torch.zeros(bs, dtype=torch.int64),
            seq_lens=seq_lens,
            req_pool_indices=torch.arange(bs),
        ),
        live_mask=live_mask,
        temperatures=None,
        top_ks=None,
    )
    sampled_bucket = greedy_bucket._replace(
        temperatures=torch.ones(bs),
        top_ks=torch.full((bs,), 8, dtype=torch.int32),
        target_candidates=True,
    )
    dr_greedy = DraftResult(draft_tokens=draft_tokens, draft_logits=None, num_verify_drafts=num_verify)
    dr_cand = DraftResult(
        draft_tokens=draft_tokens,
        draft_logits=None,
        num_verify_drafts=num_verify,
        draft_candidates=(torch.zeros((bs, gamma, 8), dtype=torch.int64), torch.zeros((bs, gamma, 8))),
    )
    assert (
        TpuSpecWorker._verify_logits_output_mode(dummy_worker, greedy_bucket, dr_greedy)
        is LogitsOutputMode.DISTRIBUTED_GREEDY
    )
    assert (
        TpuSpecWorker._verify_logits_output_mode(dummy_worker, sampled_bucket, dr_cand)
        is LogitsOutputMode.DISTRIBUTED_TOP_K
    )


def test_p2_dflash_direct_handoff_and_adversarial_transitions():
    saved_fmap_attrs = {
        k: getattr(FutureMap, k)
        for k in (
            "stash",
            "publish",
            "stash_bonus_tokens",
            "resolve_mixed_spec_tails",
            "_resolve_spec_extras",
        )
    }
    had_direct = hasattr(FutureMap, "tpu_direct_handoff")
    had_pub = hasattr(FutureMap, "tpu_published_seq_lens")
    had_orig_pub = hasattr(FutureMap, "_tpu_orig_publish")
    _patch_future_map_direct_handoff()
    try:
        _run_p2_dflash_direct_handoff_assertions()
    finally:
        for k, v in saved_fmap_attrs.items():
            setattr(FutureMap, k, v)
        if not had_direct and hasattr(FutureMap, "tpu_direct_handoff"):
            delattr(FutureMap, "tpu_direct_handoff")
        if not had_pub and hasattr(FutureMap, "tpu_published_seq_lens"):
            delattr(FutureMap, "tpu_published_seq_lens")
        if not had_orig_pub and hasattr(FutureMap, "_tpu_orig_publish"):
            delattr(FutureMap, "_tpu_orig_publish")


def _run_p2_dflash_direct_handoff_assertions():
    pool = ReqToTokenPool(size=16, max_context_len=256, device="cpu", enable_memory_saver=False)
    fmap = FutureMap(
        device=torch.device("cpu"),
        spec_algo=SpeculativeAlgorithm.DFLASH,
        req_to_token_pool=pool,
        needs_cpu_seq_lens=False,
    )
    store_calls = []
    orig_store = fmap._store

    def tracked_store(buf, indices, values):
        store_calls.append((buf, indices.clone(), values.clone()))
        return orig_store(buf, indices, values)

    fmap._store = tracked_store

    with (
        _temp_env(SGLANG_TPU_SPEC_DIRECT_HANDOFF="1"),
        patch("sglang.srt.speculative.spec_utils.spec_need_hidden_states", return_value=False),
        patch("sglang.srt.managers.overlap_utils._is_tpu", True),
    ):
        idx_ab = torch.tensor([1, 2], dtype=torch.int64)
        seq_ab = torch.tensor([50, 60], dtype=torch.int64)
        bonus_ab = torch.tensor([101, 202], dtype=torch.int64)

        # Step 1: publish + stash on steady-state DFLASH decode -> ZERO _store calls!
        fmap.publish(idx_ab, seq_ab)
        fmap.stash(idx_ab, RelayPayload(bonus_tokens=bonus_ab))
        assert len(store_calls) == 0
        assert fmap.tpu_direct_handoff is not None

        # Step 2: unchanged batch resolves directly -> STILL ZERO _store calls!
        draft2 = HostPaddedBlockDraftInput(bonus_tokens=bonus_ab, new_seq_lens=seq_ab)
        draft2.future_indices = idx_ab
        draft2.tpu_host_seq_lens_is_committed = True
        batch2 = SimpleNamespace(
            spec_info=draft2,
            forward_mode=ForwardMode.DECODE,
            is_extend_in_batch=False,
            host_pad_bs=lambda x: x,
            seq_lens=None,
        )
        assert resolve_device_accept_seq_lens_tpu(fmap, batch2) is True
        assert torch.equal(batch2.seq_lens, seq_ab)
        assert torch.equal(draft2.bonus_tokens, bonus_ab)
        assert device_accept_resolved_tpu(fmap, batch2) is True
        assert len(store_calls) == 0

        # Step 2 finishes and publishes updated values [54, 63], [111, 222]
        seq_ab2 = torch.tensor([54, 63], dtype=torch.int64)
        bonus_ab2 = torch.tensor([111, 222], dtype=torch.int64)
        fmap.publish(idx_ab, seq_ab2)
        fmap.stash(idx_ab, RelayPayload(bonus_tokens=bonus_ab2))
        assert len(store_calls) == 0

        # Adversarial transition A: interleaved prefill of req [3] while [1, 2] is parked!
        idx_c = torch.tensor([3], dtype=torch.int64)
        seq_c = torch.tensor([80], dtype=torch.int64)
        bonus_c = torch.tensor([333], dtype=torch.int64)
        fmap.publish(idx_c, seq_c)
        fmap.stash(idx_c, RelayPayload(bonus_tokens=bonus_c))
        # Publishing [3] MUST have flushed [1, 2] into FutureMap buffers first!
        assert len(store_calls) == 2

        # Adversarial transition B: merge_batch([2, 3]) after filtering req 1!
        idx_bc = torch.tensor([2, 3], dtype=torch.int64)
        draft_merged = HostPaddedBlockDraftInput(
            bonus_tokens=torch.zeros(2, dtype=torch.int64),
            new_seq_lens=torch.zeros(2, dtype=torch.int64),
        )
        draft_merged.future_indices = idx_bc
        draft_merged.tpu_host_seq_lens_is_committed = True
        batch_merged = SimpleNamespace(
            spec_info=draft_merged,
            forward_mode=ForwardMode.DECODE,
            is_extend_in_batch=False,
            host_pad_bs=lambda x: x,
            seq_lens=None,
        )
        fmap.resolve_seq_lens_cpu(batch_merged)
        fmap._resolve_spec_extras(batch_merged)
        assert torch.equal(batch_merged.seq_lens, torch.tensor([63, 80], dtype=torch.int64))
        assert torch.equal(draft_merged.bonus_tokens, torch.tensor([222, 333], dtype=torch.int64))


def test_p2_packed_addressing_and_fused_prepare_dflash_block_step():
    torch.manual_seed(42)
    bs, padded_bs, width = 5, 8, 5
    cand_cols = 2 * width
    candidates = torch.arange(100, 100 + bs * cand_cols, dtype=torch.int64).view(bs, cand_cols)
    swa_map = torch.arange(1000, -1, -1, dtype=torch.int64)  # nontrivial mapping
    committed_lens = torch.tensor([10, 20, 30, 40, 50], dtype=torch.int64)
    seq_lens = torch.tensor([13, 20, 34, 41, 52, 0, 0, 0], dtype=torch.int64)
    bonus_tokens = torch.tensor([7, 8, 9, 10, 11, 0, 0, 0], dtype=torch.int64)
    live_mask = torch.tensor([True] * bs + [False] * (padded_bs - bs))
    mask_token_id = 999

    draft_in = HostPaddedBlockDraftInput(
        bonus_tokens=bonus_tokens[:bs], new_seq_lens=seq_lens[:bs]
    )
    draft_in.tpu_verify_cache_candidates = candidates
    draft_in.tpu_committed_lens = committed_lens
    draft_in.tpu_req_pool_indices = torch.arange(1, bs + 1, dtype=torch.int64)
    draft_in.tpu_verify_window_rows = width
    draft_in.tpu_host_seq_lens_is_committed = True
    batch = SimpleNamespace(
        spec_info=draft_in,
        req_pool_indices=torch.arange(1, bs + 1, dtype=torch.int64),
        req_pool_indices_cpu=torch.arange(1, bs + 1, dtype=torch.int64),
        seq_lens_cpu=committed_lens,
        device="cpu",
    )

    vi_ref = TpuSpecVerifyInput.from_batch(batch, width, padded_bs, pack_addressing=False)
    vi_fused = TpuSpecVerifyInput.from_batch(
        batch, width, padded_bs, pack_addressing=True, swa_mapping=swa_map
    )
    assert vi_fused.packed_addressing is not None
    assert vi_fused.verify_cache_candidates is None
    assert vi_fused.committed_lens is None

    ref_rows = select_verify_rows(
        vi_ref.verify_cache_candidates, vi_ref.committed_lens, seq_lens, width
    )
    swa_cands_padded = torch.zeros((padded_bs, cand_cols), dtype=torch.int64)
    swa_cands_padded[:bs] = swa_map[candidates]
    ref_swa_rows = select_verify_rows(
        swa_cands_padded, vi_ref.committed_lens, seq_lens, width
    )
    ref_draft_in = build_draft_block_inputs(
        bonus_tokens, ref_rows, seq_lens, live_mask, mask_token_id, width
    )

    fused_rows, fused_swa_rows, fused_draft_in = prepare_dflash_block_step(
        vi_fused.packed_addressing,
        bonus_tokens,
        seq_lens,
        live_mask,
        width,
        mask_token_id,
        width,
        True,
    )
    for a, b in zip(ref_rows, fused_rows):
        assert torch.equal(a, b)
    assert fused_swa_rows is not None
    for a, b in zip(ref_swa_rows, fused_swa_rows):
        assert torch.equal(a, b)
    for a, b in zip(ref_draft_in, fused_draft_in):
        assert torch.equal(a, b)


def test_p2_unpack_draft_lattice_handles_dflash_and_dflash2():
    bonus = torch.tensor([11, 22, 33, 44], dtype=torch.int64)
    drafts = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=torch.int64)
    verify_flat = build_verify_tokens(bonus, drafts)

    # DFlash (no lattice)
    lat, vt = TpuBlockSpecWorker._unpack_draft_lattice((None, verify_flat))
    assert lat is None and torch.equal(vt, verify_flat)

    # DFlash2 ((cand_ids [B, gamma, K], scores [B, gamma, K, K]), verify_flat)
    cands = torch.zeros((4, 3, 8), dtype=torch.int64)
    scores = torch.zeros((4, 3, 8, 8), dtype=torch.float32)
    lat2, vt2 = TpuBlockSpecWorker._unpack_draft_lattice(((cands, scores), verify_flat))
    assert isinstance(lat2, tuple) and lat2[0] is cands and lat2[1] is scores
    assert torch.equal(vt2, verify_flat)

    # Unfused DFlash2 ((cand_ids, scores) without verify_flat)
    lat3, vt3 = TpuBlockSpecWorker._unpack_draft_lattice((cands, scores))
    assert isinstance(lat3, tuple) and lat3[0] is cands and vt3 is None


def test_p3_req_to_token_version_and_rpa_page_table_cache():
    pool = ReqToTokenPool(size=8, max_context_len=512, device="tpu", enable_memory_saver=False)
    assert pool.req_to_token.device.type == "cpu"
    v0 = pool.req_to_token._tpu_page_table_version
    rows = pool.alloc_rows(2)
    v1 = pool.req_to_token._tpu_page_table_version
    assert v1 > v0
    pool.write((rows[0], slice(0, 128)), torch.arange(128, dtype=torch.int32))
    v2 = pool.req_to_token._tpu_page_table_version
    assert v2 > v1
    pool.free_rows(rows)
    v3 = pool.req_to_token._tpu_page_table_version
    assert v3 > v2
    pool.clear()
    assert pool.req_to_token._tpu_page_table_version > v3


def test_p4_deduplicate_aliased_block_accept_lens_d2h():
    toks = MagicMock(spec=torch.Tensor)
    toks.device = SimpleNamespace(type="tpu")
    toks.is_cuda = False
    lens = MagicMock(spec=torch.Tensor)
    lens.device = SimpleNamespace(type="tpu")
    lens.is_cuda = False

    res = GenerationBatchResult(
        logits_output=LogitsProcessorOutput(next_token_logits=None),
        next_token_ids=toks,
        accept_lens=lens,
        block_accept_lens=lens,
    )
    res.copy_done = MagicMock()

    copied = []

    def fake_d2h(t):
        copied.append(t)
        return f"cpu_{len(copied)}"

    with (
        _temp_env(SGLANG_TPU_FAST_RESULT_COPY="1"),
        patch("sglang.srt.managers.utils._async_d2h", side_effect=fake_d2h),
    ):
        res.copy_to_cpu(return_logprob=False, return_hidden_states=False)

    assert len(copied) == 2  # toks + lens once (NOT twice!)
    assert res.accept_lens == "cpu_2"
    assert res.block_accept_lens == "cpu_2"


def test_p3_build_metadata_host_caches_until_version_or_rows_change():
    from sglang_tpu.attention_pallas_rpa_base import BaseTpuPallasRpaAttnBackend
    from sglang_tpu.host_metadata_cache import HostMetadataCache

    backend = object.__new__(BaseTpuPallasRpaAttnBackend)
    backend._dcp = 1
    backend._page_stride = 128
    backend._host_page_cols = None
    backend._is_hybrid_swa = True
    backend._swa_group_pages = None
    backend._metadata_cache = HostMetadataCache()
    value_calls = []
    orig_val_meta = backend._metadata_cache.get_by_value

    def tracked_val_meta(name, value, device):
        value_calls.append(name)
        return orig_val_meta(name=name, value=value, device=device)

    backend._value_metadata = tracked_val_meta
    backend._layout_metadata = lambda *, key, build, device: backend._metadata_cache.get_by_layout(
        key, build, device
    )
    backend._is_extend_including_draft = lambda mode: False
    backend.supports_shared_decode_batch = lambda bs: False
    backend.token_to_kv_pool = SimpleNamespace(
        full_to_swa_index_mapping=torch.arange(4096, dtype=torch.int64)
    )

    pool = ReqToTokenPool(size=8, max_context_len=512, device="tpu", enable_memory_saver=False)
    pool.write((1, slice(0, 256)), torch.arange(128, 384, dtype=torch.int32))
    pool.write((2, slice(0, 256)), torch.arange(512, 768, dtype=torch.int32))

    fb = SimpleNamespace(
        forward_mode=ForwardMode.TARGET_VERIFY,
        spec_info=SimpleNamespace(num_tokens_per_req=5),
        seq_lens=torch.tensor([140, 200], dtype=torch.int64),
        seq_lens_cpu=torch.tensor([140, 200], dtype=torch.int64),
        req_pool_indices_cpu=torch.tensor([1, 2], dtype=torch.int64),
    )
    dist = torch.tensor([2, 2, 2], dtype=torch.int32)

    with _temp_env(SGLANG_TPU_RPA_CACHE_DECODE_PAGE_TABLE="1"):
        m1 = backend._build_metadata_host(
            fb, 128, "cpu", 2, pool.req_to_token, 4, dist, 2, decode_query_size=5
        )
        assert value_calls == ["swa_page_indices", "page_indices"]

        # Step 2: seq_lens advance within page, req_to_token untouched -> O(1) cache HIT!
        fb.seq_lens = torch.tensor([144, 204], dtype=torch.int64)
        fb.seq_lens_cpu = torch.tensor([144, 204], dtype=torch.int64)
        m2 = backend._build_metadata_host(
            fb, 128, "cpu", 2, pool.req_to_token, 4, dist, 2, decode_query_size=5
        )
        assert value_calls == ["swa_page_indices", "page_indices"]
        assert m2.page_indices is m1.page_indices
        assert m2.swa_page_indices is m1.swa_page_indices
        assert torch.equal(m2.kv_lens, torch.tensor([149, 209], dtype=torch.int32))

        # Step 3: new page written to req_to_token -> version bumps -> cache MISS!
        pool.write((1, slice(256, 384)), torch.arange(1024, 1152, dtype=torch.int32))
        m3 = backend._build_metadata_host(
            fb, 128, "cpu", 2, pool.req_to_token, 4, dist, 2, decode_query_size=5
        )
        assert len(value_calls) == 4
        assert not torch.equal(m3.page_indices, m1.page_indices)


def test_p4_fast_forward_isolation_restores_mutated_fields():
    from sglang.srt.managers.scheduler import Scheduler

    sched = object.__new__(Scheduler)
    sched.batch_record_ct = 0
    sched.batch_record_buf = [None, None]

    batch = SimpleNamespace(
        spec_algorithm=SpeculativeAlgorithm.DFLASH,
        forward_mode=ForwardMode.DECODE,
        input_ids=None,
        seq_lens=torch.tensor([10, 20]),
        seq_lens_cpu=torch.tensor([10, 20]),
        seq_lens_sum=30,
        orig_seq_lens=None,
        req_pool_indices=torch.tensor([1, 2]),
        req_pool_indices_cpu=torch.tensor([1, 2]),
        out_cache_loc=None,
        spec_info="orig_spec_info",
        sampling_info=SimpleNamespace(copy_for_forward=lambda: "fwd_sampling_info"),
        mamba_track_indices=None,
        mamba_track_mask=None,
        mamba_track_seqlens=None,
        untouched_extra="keep",
    )
    Scheduler._SB_ATTR_NAMES = tuple(vars(batch).keys())
    with (
        _temp_env(SGLANG_TPU_FAST_FORWARD_ISOLATION="1"),
        patch("sglang.srt.managers.scheduler.current_platform.is_tpu", return_value=True),
    ):
        with sched._forward_isolation(batch, overlap=True):
            assert batch.sampling_info == "fwd_sampling_info"
            batch.forward_mode = ForwardMode.TARGET_VERIFY
            batch.input_ids = torch.tensor([1, 2, 3])
            batch.spec_info = "mutated_spec"
            batch.out_cache_loc = torch.tensor([99])

    assert batch.forward_mode is ForwardMode.DECODE
    assert batch.input_ids is None
    assert batch.spec_info == "orig_spec_info"
    assert batch.out_cache_loc is None
    assert batch.untouched_extra == "keep"


def test_nonspec_d1_async_copy_worker_off_supported():
    from sglang_tpu.allocator_tpu import TpuPagedTokenToKVPoolAllocator
    from sglang_tpu.kv_free_fence import kv_free_fence_unsupported_reason

    alloc = object.__new__(TpuPagedTokenToKVPoolAllocator)
    # Non-speculative (d=1) supports KVFreeFence with SGLANG_TPU_ASYNC_COPY_WORKER=0
    assert (
        kv_free_fence_unsupported_reason(
            alloc,
            spec_algorithm=SpeculativeAlgorithm.NONE,
            copies_draft_tail=False,
        )
        is None
    )
    # EAGLE requires SGLANG_TPU_ASYNC_COPY_WORKER=1 (copies_draft_tail=True),
    # whereas non-speculative NONE works cleanly with SGLANG_TPU_ASYNC_COPY_WORKER=0.
    assert (
        kv_free_fence_unsupported_reason(
            alloc,
            spec_algorithm=SpeculativeAlgorithm.EAGLE,
            copies_draft_tail=False,
        )
        is not None
    )
    assert (
        kv_free_fence_unsupported_reason(
            alloc,
            spec_algorithm=SpeculativeAlgorithm.EAGLE,
            copies_draft_tail=True,
        )
        is None
    )


def test_nonspec_d1_host_padding_future_map_direct_handoff():
    from sglang.srt.managers import overlap_utils
    from sglang_tpu.compilation.bucketer import Bucketer
    from sglang_tpu.host_padding_relay import HostPaddingFutureMap

    pool = ReqToTokenPool(size=16, max_context_len=64, device="cpu", enable_memory_saver=False)
    pool._tpu_relay_bucketer = Bucketer("relay", [4, 8])

    with (
        _temp_env(SGLANG_TPU_SPEC_DIRECT_HANDOFF="1"),
        patch.object(overlap_utils, "_DEBUG_ASSERT", False),
    ):
        fm = HostPaddingFutureMap(
            torch.device("cpu"),
            SpeculativeAlgorithm.NONE,
            pool,
            needs_cpu_seq_lens=False,
        )
        assert fm._is_noop_publish is True
        index_copy_calls = []
        orig_idx_copy = fm.output_tokens_buf.index_copy_

        def spy_idx_copy(dim, idx, val):
            index_copy_calls.append((idx.clone(), val.clone()))
            return orig_idx_copy(dim, idx, val)

        fm.output_tokens_buf.index_copy_ = spy_idx_copy

        # Step 1 -> Step 2 steady-state decode: zero index_copy_ and zero gather!
        idx = torch.tensor([2, 5, 7], dtype=torch.int64)
        vals = torch.tensor([101, 202, 303, 0], dtype=torch.int64)
        fm.stash_bonus_tokens(idx, vals)
        assert len(index_copy_calls) == 0
        out = fm.gather(idx.clone())
        assert len(index_copy_calls) == 0
        assert torch.equal(out, vals)

        # Step 2 -> Step 3 composition change (filtered row 5 dropped): flushes cleanly!
        vals2 = torch.tensor([404, 505, 606, 0], dtype=torch.int64)
        fm.stash_bonus_tokens(idx, vals2)
        assert len(index_copy_calls) == 0
        idx_filtered = torch.tensor([2, 7], dtype=torch.int64)
        out_filtered = fm.gather(idx_filtered)
        assert len(index_copy_calls) == 1
        assert torch.equal(out_filtered[:2], torch.tensor([404, 606], dtype=torch.int64))


def test_nonspec_d1_distributed_greedy_topk1_routing():
    from sglang_tpu import sampler as tpu_sampler
    from sglang_tpu.compilation.decoder_compile_policy import DecoderTpuCompilePolicy
    from sglang_tpu.models.muse_spark import MuseSparkForConditionalGeneration

    fb = SimpleNamespace(
        forward_mode=ForwardMode.DECODE,
        spec_algorithm=SpeculativeAlgorithm.NONE,
        spec_info=None,
        return_logprob=False,
        batch_size=64,
        sampling_info=SimpleNamespace(
            is_all_greedy=True,
            is_any_greedy=True,
            sampling_support_logprobs_capture_indices=None,
            has_custom_logit_processor=False,
            logit_bias=None,
            grammars=None,
            sampling_mask_batch_indices=None,
            acc_additive_penalties=None,
            acc_scaling_penalties=None,
            penalizer_orchestrator=None,
        ),
        logits_output_mode=LogitsOutputMode.FULL_VOCAB,
    )
    with (
        _temp_env(
            SGLANG_TPU_DISTRIBUTED_GREEDY_TOPK="1",
            SGLANG_TPU_SPEC_VERIFY_GREEDY_TOPK1="1",
        ),
        patch.object(
            tpu_sampler, "distributed_top_k_candidate_width_for_model", return_value=64
        ),
    ):
        assert (
            tpu_sampler.select_logits_output_mode(fb, object())
            is LogitsOutputMode.DISTRIBUTED_GREEDY
        )

    with _temp_env(
        SGLANG_TPU_DISTRIBUTED_GREEDY_TOPK="1",
        SGLANG_TPU_SPEC_VERIFY_GREEDY_TOPK1="0",
    ), patch.object(
        tpu_sampler, "distributed_top_k_candidate_width_for_model", return_value=64
    ):
        assert (
            tpu_sampler.select_logits_output_mode(fb, object())
            is LogitsOutputMode.DISTRIBUTED_TOP_K
        )

    # Check DecoderTpuCompilePolicy warmup modes include DISTRIBUTED_GREEDY when enabled
    pol = object.__new__(DecoderTpuCompilePolicy)
    pol.model_runner = SimpleNamespace(
        spec_algorithm=SpeculativeAlgorithm.NONE,
        device="cpu",
        model_config=SimpleNamespace(vocab_size=1024),
    )
    pol._distributed_top_k_width = lambda: 64
    with _temp_env(
        SGLANG_TPU_DISTRIBUTED_GREEDY_TOPK="1",
        SGLANG_TPU_SPEC_VERIFY_GREEDY_TOPK1="1",
    ):
        modes = pol._warmup_logits_output_modes(64)
        assert LogitsOutputMode.DISTRIBUTED_GREEDY in modes
        g_info = pol._make_greedy_warmup_sampling_info(64)
        assert g_info.is_all_greedy is True
        assert g_info.uniform_top_k == 1

    # Check MuseSparkForConditionalGeneration._forward_pruned_logits passes top_k=1
    captured_k = []
    dummy_lp = SimpleNamespace(
        compute_input_logprobs=False,
        _get_distributed_top_k=lambda states, head, k: (
            captured_k.append(k) or (torch.zeros(4, k), torch.zeros(4, k, dtype=torch.int32))
        ),
    )
    dummy_model = SimpleNamespace(logits_processor=dummy_lp, lm_head=object())
    fb_pruned = SimpleNamespace(
        logits_output_mode=LogitsOutputMode.DISTRIBUTED_GREEDY,
        forward_mode=ForwardMode.EXTEND,
        mm_input_embeds=None,
        return_logprob=False,
        top_logprobs_nums=None,
        token_ids_logprobs=None,
        extend_seq_lens=torch.tensor([4]),
        extend_prefix_lens=torch.tensor([0]),
        extend_logprob_start_lens=None,
        extend_input_logprob_token_ids_gpu=None,
        is_prefill_only=False,
    )
    with (
        patch(
            "sglang_tpu.sampler.distributed_top_k_candidate_width", return_value=64
        ),
        patch(
            "sglang.srt.layers.logits_processor.LogitsMetadata.from_forward_batch",
            return_value=SimpleNamespace(mm_input_embeds=None),
        ),
    ):
        out = MuseSparkForConditionalGeneration._forward_pruned_logits(
            dummy_model, torch.zeros(4, 16), fb_pruned
        )
    assert captured_k == [1]
    assert out.next_token_global_top_k_token_ids.shape == (4, 1)


def test_nonspec_d1_fast_filter_finished_batch_zombie_reqs():
    r_done1 = SimpleNamespace(finished=lambda: True)
    r_done2 = SimpleNamespace(finished=lambda: True)
    r_live = SimpleNamespace(finished=lambda: False)

    class FakeBatch:
        def __init__(self, reqs):
            self.reqs = list(reqs)
            self.batch_is_full = True
            self.is_prefill_only = False
            self.filter_calls = 0

        def is_empty(self):
            return len(self.reqs) == 0

        def filter_batch(self):
            self.filter_calls += 1
            self.reqs = [r for r in self.reqs if not r.finished()]

    # All-finished (8k/1 overlap zombie pattern) clears immediately when enabled
    b_zombies = FakeBatch([r_done1, r_done2])
    with _temp_env(SGLANG_TPU_FAST_FILTER_FINISHED_BATCH="1"):
        if (
            not b_zombies.is_empty()
            and all(r.finished() for r in b_zombies.reqs)
            and tpu_envs.SGLANG_TPU_FAST_FILTER_FINISHED_BATCH.get()
        ):
            b_zombies.filter_batch()
            b_zombies.batch_is_full = False
    assert b_zombies.is_empty()
    assert b_zombies.batch_is_full is False
    assert b_zombies.filter_calls == 1

    # Mixed live + finished (1/1k or 8k/1k active decode) is a strict no-op
    b_active = FakeBatch([r_done1, r_live])
    with _temp_env(SGLANG_TPU_FAST_FILTER_FINISHED_BATCH="1"):
        if (
            not b_active.is_empty()
            and all(r.finished() for r in b_active.reqs)
            and tpu_envs.SGLANG_TPU_FAST_FILTER_FINISHED_BATCH.get()
        ):
            b_active.filter_batch()
            b_active.batch_is_full = False
    assert len(b_active.reqs) == 2
    assert b_active.batch_is_full is True
    assert b_active.filter_calls == 0


