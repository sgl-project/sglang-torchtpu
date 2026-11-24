# SPDX-License-Identifier: Apache-2.0
"""CPU unit & torch.compile parity tests for Wave-9.11 DFlash step fusion (v2)."""

from __future__ import annotations

import ast
import contextlib
import dataclasses
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.speculative.draft_worker_common import make_draft_block_spec_info
from sglang.srt.speculative.dflash_info import DFlashVerifyInput
from sglang_tpu.attention_pallas_rpa_base import PallasForwardMetadata
from sglang_tpu.envs import envs as tpu_envs
from sglang_tpu.experimental.compilation.block_compile_policy import (
    TpuBlockSpecDecodeCompilePolicyMixin,
)
from sglang_tpu.experimental.speculative.block_draft_worker import (
    TpuBlockDraftWorker,
)
from sglang_tpu.experimental.speculative.block_info import HostPaddedBlockDraftInput
from sglang_tpu.experimental.speculative.block_spec_worker import TpuBlockSpecWorker
from sglang_tpu.experimental.speculative.draft_forward import prep_and_draft_rows
from sglang_tpu.experimental.speculative.spec_ops import (
    TARGET_CANDIDATES,
    accept_greedy_top_k_step,
    accept_top_k_candidates_step,
    prepare_dflash_block_step,
    verify_tokens_of,
)
from sglang_tpu.experimental.speculative.spec_types import (
    AcceptResult,
    BucketState,
    SpecState,
    TpuSpecVerifyInput,
)

REPO_ROOT = Path(__file__).resolve().parents[5]


@contextlib.contextmanager
def _temp_env(**overrides: str):
    prev = {k: os.environ.get(k) for k in overrides}
    try:
        for k, v in overrides.items():
            os.environ[k] = v
        yield
    finally:
        for k, v in prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _make_dummy_policy() -> TpuBlockSpecDecodeCompilePolicyMixin:
    policy = object.__new__(TpuBlockSpecDecodeCompilePolicyMixin)
    policy.tracer = "dynamo"
    policy.backend = "eager"
    policy.fullgraph = True
    policy.dynamic = False
    policy.is_draft_worker = True
    policy.owner = None
    policy.model_runner = SimpleNamespace(
        model=SimpleNamespace(forward=lambda *a, **kw: None),
        device="cpu",
        dtype=torch.bfloat16,
    )
    policy._eager_model_forward = None
    policy._kv_writer = None
    policy._compiled_prep_and_draft = None
    policy._canonical_draft_spec_info = None
    policy._compiled_accept_write_greedy_top_k = None
    policy._compiled_accept_write_top_k_candidates = None
    policy._block_skip_guard_stance = None
    return policy


def test_envs_v2_default_off_and_get_hygiene():
    assert tpu_envs.SGLANG_TPU_DFLASH_FUSE_STEP_OPS_V2.get() is False
    targets = [
        REPO_ROOT / "sglang_tpu/experimental/compilation/block_compile_policy.py",
        REPO_ROOT / "sglang_tpu/experimental/speculative/block_spec_worker.py",
        REPO_ROOT / "sglang_tpu/experimental/speculative/block_draft_worker.py",
        REPO_ROOT / "sglang_tpu/experimental/speculative/draft_forward.py",
        REPO_ROOT / "sglang_tpu/experimental/speculative/spec_types.py",
        REPO_ROOT / "sglang_tpu/experimental/speculative/spec_ops.py",
    ]
    for path in targets:
        tree = ast.parse(path.read_text(), filename=str(path))
        parents: dict[int, ast.AST] = {}
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                parents[id(child)] = parent
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id in ("envs", "tpu_envs")
                and node.attr.startswith("SGLANG_")
            ):
                parent = parents.get(id(node))
                assert (
                    isinstance(parent, ast.Attribute)
                    and parent.attr == "get"
                    and isinstance(parents.get(id(parent)), ast.Call)
                ), f"Bare env access {node.value.id}.{node.attr} without .get() in {path}"


def test_packed_addressing_reusable_pinned_staging_zeroes_tail():
    padded_bs, width = 8, 4
    cand_cols = 2 * width
    swa_map = torch.arange(500, -1, -1, dtype=torch.int64)
    staging_cache: dict[tuple[int, int, str], torch.Tensor] = {}

    def _make_batch(bs_live: int, offset: int):
        cands = torch.arange(
            offset, offset + bs_live * cand_cols, dtype=torch.int64
        ).view(bs_live, cand_cols)
        committed = torch.arange(1, bs_live + 1, dtype=torch.int64) * 5
        draft_in = HostPaddedBlockDraftInput(
            bonus_tokens=torch.zeros(bs_live, dtype=torch.int64),
            new_seq_lens=committed + 1,
        )
        draft_in.tpu_verify_cache_candidates = cands
        draft_in.tpu_committed_lens = committed
        draft_in.tpu_req_pool_indices = torch.arange(1, bs_live + 1, dtype=torch.int64)
        draft_in.tpu_verify_window_rows = width
        draft_in.tpu_host_seq_lens_is_committed = True
        return SimpleNamespace(
            spec_info=draft_in,
            req_pool_indices=draft_in.tpu_req_pool_indices,
            req_pool_indices_cpu=draft_in.tpu_req_pool_indices,
            seq_lens_cpu=committed,
            device="cpu",
        )

    b1 = _make_batch(bs_live=5, offset=10)
    vi1 = TpuSpecVerifyInput.from_batch(
        b1, width, padded_bs, pack_addressing=True, swa_mapping=swa_map, staging_cache=staging_cache
    )
    assert vi1.packed_addressing is not None
    expected1 = vi1.packed_addressing.clone()
    key = (padded_bs, cand_cols * 2 + 1, "cpu")
    assert key in staging_cache
    buf_ptr = staging_cache[key].data_ptr()

    # Second step with smaller live batch must reuse the exact same buffer and zero stale tail rows [2:5].
    b2 = _make_batch(bs_live=2, offset=200)
    vi2 = TpuSpecVerifyInput.from_batch(
        b2, width, padded_bs, pack_addressing=True, swa_mapping=swa_map, staging_cache=staging_cache
    )
    assert vi2.packed_addressing is not None
    assert staging_cache[key].data_ptr() == buf_ptr
    assert torch.equal(vi2.packed_addressing[2:], torch.zeros((6, cand_cols * 2 + 1), dtype=torch.int64))
    assert torch.equal(vi2.packed_addressing[:2, :cand_cols], b2.spec_info.tpu_verify_cache_candidates)
    assert not torch.equal(expected1[2:5], torch.zeros((3, cand_cols * 2 + 1), dtype=torch.int64))


@pytest.mark.parametrize("has_swa", [False, True])
@pytest.mark.parametrize("with_lattice", [False, True])
def test_launch1_prep_and_draft_parity_and_fullgraph_compile(
    has_swa: bool, with_lattice: bool
):
    torch.manual_seed(11)
    bs, padded_bs, width = 3, 4, 4
    cand_cols = 2 * width
    mask_token_id = 999

    packed_cols = cand_cols * (2 if has_swa else 1) + 1
    packed = torch.randint(0, 200, (padded_bs, packed_cols), dtype=torch.int64)
    packed[:bs, -1] = torch.tensor([10, 20, 30], dtype=torch.int64)
    packed[bs:] = 0
    bonus_tokens = torch.tensor([7, 8, 9, 0], dtype=torch.int64)
    seq_lens = torch.tensor([12, 21, 33, 0], dtype=torch.int64)
    live_mask = torch.tensor([True, True, True, False], dtype=torch.bool)

    seen_kv_lens: list[torch.Tensor] = []
    attn_backend = SimpleNamespace(
        forward_metadata=PallasForwardMetadata(
            kv_lens=torch.zeros(padded_bs, dtype=torch.int32),
            cu_q_lens=torch.arange(padded_bs + 1, dtype=torch.int32) * width,
            cu_kv_lens=torch.zeros(padded_bs + 1, dtype=torch.int32),
            page_indices=torch.zeros(padded_bs * 4, dtype=torch.int32),
            distribution=torch.tensor([bs, bs, bs], dtype=torch.int32),
            active_num_tokens=torch.tensor(bs * width, dtype=torch.int32),
        )
    )

    def fake_draft_forward(input_ids, positions, forward_batch):
        draft_kv_lens = getattr(forward_batch.spec_info, "rpa_draft_kv_lens", None)
        seen_kv_lens.append(
            draft_kv_lens.clone()
            if draft_kv_lens is not None
            else attn_backend.forward_metadata.kv_lens.clone()
        )
        hidden = (
            input_ids.to(torch.float32)[:, None] * 0.25
            + positions.to(torch.float32)[:, None]
            + forward_batch.out_cache_loc.to(torch.float32)[:, None]
            + forward_batch.extend_prefix_lens.repeat_interleave(width).to(torch.float32)[:, None]
        ).expand(-1, 8).clone()
        next_draft = (input_ids.view(padded_bs, width)[:, : width - 1] + 13).clone()
        if with_lattice:
            cand_ids = next_draft[:, :, None].expand(-1, -1, 4).clone()
            scores = hidden.view(padded_bs, width, 8)[:, : width - 1, :4].unsqueeze(-1).expand(-1, -1, -1, 4).clone()
            lattice = (cand_ids, scores)
        else:
            lattice = None
        verify_toks = verify_tokens_of(input_ids.view(padded_bs, width)[:, 0], next_draft)
        return LogitsProcessorOutput(
            next_token_logits=None,
            next_draft_tokens=next_draft,
            next_draft_lattice=(lattice, verify_toks),
            hidden_states=hidden,
        )

    # Unfused baseline
    ref_rows, ref_swa_rows, ref_di = prepare_dflash_block_step(
        packed, bonus_tokens, seq_lens, live_mask, width, mask_token_id, width, has_swa
    )
    ref_fb = ForwardBatch(
        forward_mode=ForwardMode.TARGET_VERIFY,
        out_cache_loc_is_physical=True,
        batch_size=padded_bs,
        input_ids=ref_di.input_ids,
        req_pool_indices=torch.arange(padded_bs, dtype=torch.int64),
        seq_lens=ref_di.seq_lens,
        seq_lens_sum=int(ref_di.seq_lens.sum().item()),
        out_cache_loc=ref_di.cache_locs,
        positions=ref_di.positions,
        extend_prefix_lens=ref_di.extend_prefix_lens,
        extend_seq_lens=ref_di.extend_seq_lens,
    )
    attn_backend.forward_metadata = dataclasses.replace(
        attn_backend.forward_metadata,
        kv_lens=(ref_di.seq_lens + width).to(torch.int32),
    )
    ref_out = fake_draft_forward(ref_di.input_ids, ref_di.positions, ref_fb)

    # Fused + torch.compile(backend="eager", fullgraph=True)
    policy = _make_dummy_policy()
    policy._eager_model_forward = fake_draft_forward
    fused_fb = dataclasses.replace(
        ref_fb,
        input_ids=None,
        positions=None,
        out_cache_loc=None,
        seq_lens=seq_lens,
        extend_prefix_lens=None,
        extend_seq_lens=None,
    )
    with (
        patch("sglang_tpu.experimental.compilation.block_compile_policy.is_tpu", return_value=True),
        patch.object(
            tpu_envs,
            "SGLANG_TPU_VERIFY_SKIP_GUARD_EVAL",
            SimpleNamespace(get=lambda: False),
        ),
        forward_context(ForwardContext(attn_backend=attn_backend), forward_batch=fused_fb),
    ):
        from sglang.srt.environ import envs

        with patch.object(envs, "SGLANG_ENABLE_TORCH_COMPILE", SimpleNamespace(get=lambda: True)):
            fused_rows, fused_swa_rows, fused_hidden, fused_drafts, fused_lattice, fused_verify = (
                policy.run_prep_and_draft(
                    forward_batch=fused_fb,
                    packed_addressing=packed,
                    bonus_tokens=bonus_tokens,
                    state_seq_lens=seq_lens,
                    live_mask=live_mask,
                    num_verify_rows=width,
                    mask_token_id=mask_token_id,
                    draft_width=width,
                    has_swa=has_swa,
                )
            )

    for a, b in zip(ref_rows, fused_rows):
        assert torch.equal(a, b)
    if has_swa:
        assert ref_swa_rows is not None and fused_swa_rows is not None
        for a, b in zip(ref_swa_rows, fused_swa_rows):
            assert torch.equal(a, b)
    else:
        assert fused_swa_rows is None
    assert torch.equal(fused_hidden, ref_out.hidden_states.view(padded_bs, width, -1))
    assert torch.equal(fused_drafts, ref_out.next_draft_tokens)
    assert torch.equal(fused_verify, ref_out.next_draft_lattice[1])
    if with_lattice:
        assert torch.equal(fused_lattice[0], ref_out.next_draft_lattice[0][0])
        assert torch.equal(fused_lattice[1], ref_out.next_draft_lattice[0][1])
    else:
        assert fused_lattice is None
    assert torch.equal(seen_kv_lens[0], seen_kv_lens[1])


@pytest.mark.parametrize("top_k_width", [1, TARGET_CANDIDATES])
def test_launch3_accept_and_write_draft_kv_greedy_parity_and_compile(top_k_width: int):
    torch.manual_seed(17)
    bs, padded_bs, n = 3, 4, 3
    verify_rows = n + 1
    num_tokens = padded_bs * verify_rows
    hidden_dim = 8

    top_ids = torch.randint(0, 100, (num_tokens, top_k_width), dtype=torch.int32)
    draft_tokens = top_ids[:, 0].view(padded_bs, verify_rows)[:, :n].to(torch.int64).clone()
    draft_tokens[0, 2] += 1  # reject at pos 2
    draft_tokens[2, 0] += 1  # reject at pos 0
    num_verify_drafts = torch.full((padded_bs,), n, dtype=torch.int32)
    seq_lens = torch.tensor([10, 20, 30, 0], dtype=torch.int64)
    live_mask = torch.tensor([True, True, True, False], dtype=torch.bool)
    target_hidden = torch.randn((num_tokens, hidden_dim), dtype=torch.float32)
    positions = (seq_lens[:, None] + torch.arange(verify_rows)).reshape(-1)
    cache_loc_2d = torch.arange(1, num_tokens + 1, dtype=torch.int64).view(padded_bs, verify_rows)
    cache_loc = cache_loc_2d.reshape(-1)

    kv_ref = torch.zeros((num_tokens + 2, hidden_dim), dtype=torch.float32)
    kv_fused = torch.zeros_like(kv_ref)

    def make_writer(kv_buf: torch.Tensor):
        def writer(*, target_hidden, cache_loc, positions, cache_loc_2d, commit_lens):
            del cache_loc
            cols = torch.arange(cache_loc_2d.shape[1], device=cache_loc_2d.device)
            mask = cols[None, :] < commit_lens[:, None]
            flat_locs = torch.where(mask, cache_loc_2d, 0).reshape(-1)
            kv_buf.index_copy_(0, flat_locs, target_hidden + positions.to(torch.float32)[:, None] * 0.01)
            return kv_buf[:1].sum(dim=-1)

        return writer

    ref_accept = accept_greedy_top_k_step(
        top_ids, draft_tokens, num_verify_drafts, seq_lens, live_mask
    )
    ref_tail = make_writer(kv_ref)(
        target_hidden=target_hidden,
        cache_loc=cache_loc,
        positions=positions,
        cache_loc_2d=cache_loc_2d,
        commit_lens=ref_accept.accept_lens,
    )

    policy = _make_dummy_policy()
    from sglang.srt.environ import envs

    with (
        patch("sglang_tpu.experimental.compilation.block_compile_policy.is_tpu", return_value=True),
        patch.object(envs, "SGLANG_ENABLE_TORCH_COMPILE", SimpleNamespace(get=lambda: True)),
    ):
        fused_accept, fused_tail = policy.accept_and_write_draft_kv_greedy_top_k(
            top_ids=top_ids,
            draft_tokens=draft_tokens,
            num_verify_drafts=num_verify_drafts,
            seq_lens=seq_lens,
            live_mask=live_mask,
            target_hidden=target_hidden,
            positions=positions,
            cache_loc=cache_loc,
            cache_loc_2d=cache_loc_2d,
            kv_writer=make_writer(kv_fused),
        )

    for a, b in zip(ref_accept, fused_accept):
        assert torch.equal(a, b)
    assert torch.equal(fused_tail, ref_tail)
    assert torch.equal(kv_fused, kv_ref)


@pytest.mark.parametrize("block_verification", [False, True])
def test_launch3_accept_and_write_draft_kv_candidates_parity_and_compile(
    block_verification: bool,
):
    torch.manual_seed(23)
    bs, padded_bs, n, cand_k = 3, 4, 3, 8
    verify_rows = n + 1
    num_tokens = padded_bs * verify_rows
    hidden_dim = 8

    top_ids = torch.arange(TARGET_CANDIDATES, dtype=torch.int32).repeat(num_tokens, 1)
    top_logits = torch.randn((num_tokens, TARGET_CANDIDATES), dtype=torch.float32)
    draft_ids = torch.arange(cand_k, dtype=torch.int64).view(1, 1, cand_k).expand(padded_bs, n, cand_k).clone()
    draft_logits = torch.randn((padded_bs, n, cand_k), dtype=torch.float32)
    draft_tokens = draft_ids[:, :, 0].clone()
    num_verify_drafts = torch.full((padded_bs,), n, dtype=torch.int32)
    temperatures = torch.tensor([0.0, 0.8, 0.6, 1.0], dtype=torch.float32)
    top_ks = torch.tensor([1, 8, 4, 1], dtype=torch.int32)
    seq_lens = torch.tensor([5, 12, 19, 0], dtype=torch.int64)
    live_mask = torch.tensor([True, True, True, False], dtype=torch.bool)
    rng_step = torch.tensor([42], dtype=torch.int64)
    target_hidden = torch.randn((num_tokens, hidden_dim), dtype=torch.float32)
    positions = (seq_lens[:, None] + torch.arange(verify_rows)).reshape(-1)
    cache_loc_2d = torch.arange(1, num_tokens + 1, dtype=torch.int64).view(padded_bs, verify_rows)
    cache_loc = cache_loc_2d.reshape(-1)

    kv_ref = torch.zeros((num_tokens + 2, hidden_dim), dtype=torch.float32)
    kv_fused = torch.zeros_like(kv_ref)

    def make_writer(kv_buf: torch.Tensor):
        def writer(*, target_hidden, cache_loc, positions, cache_loc_2d, commit_lens):
            del cache_loc
            cols = torch.arange(cache_loc_2d.shape[1], device=cache_loc_2d.device)
            mask = cols[None, :] < commit_lens[:, None]
            flat_locs = torch.where(mask, cache_loc_2d, 0).reshape(-1)
            kv_buf.index_copy_(0, flat_locs, target_hidden + positions.to(torch.float32)[:, None] * 0.01)
            return kv_buf[:1].sum(dim=-1)

        return writer

    ref_accept, ref_next_rng = accept_top_k_candidates_step(
        top_logits,
        top_ids,
        draft_tokens,
        num_verify_drafts,
        draft_ids,
        draft_logits,
        temperatures,
        top_ks,
        seq_lens,
        live_mask,
        rng_step,
        block_verification,
    )
    ref_tail = make_writer(kv_ref)(
        target_hidden=target_hidden,
        cache_loc=cache_loc,
        positions=positions,
        cache_loc_2d=cache_loc_2d,
        commit_lens=ref_accept.accept_lens,
    )

    policy = _make_dummy_policy()
    from sglang.srt.environ import envs

    with (
        patch("sglang_tpu.experimental.compilation.block_compile_policy.is_tpu", return_value=True),
        patch.object(envs, "SGLANG_ENABLE_TORCH_COMPILE", SimpleNamespace(get=lambda: True)),
    ):
        fused_accept, fused_next_rng, fused_tail = (
            policy.accept_and_write_draft_kv_top_k_candidates(
                top_logits=top_logits,
                top_ids=top_ids,
                draft_tokens=draft_tokens,
                num_verify_drafts=num_verify_drafts,
                draft_ids=draft_ids,
                draft_logits=draft_logits,
                temperatures=temperatures,
                top_ks=top_ks,
                seq_lens=seq_lens,
                live_mask=live_mask,
                rng_step=rng_step,
                target_hidden=target_hidden,
                positions=positions,
                cache_loc=cache_loc,
                cache_loc_2d=cache_loc_2d,
                block_verification=block_verification,
                kv_writer=make_writer(kv_fused),
            )
        )

    for a, b in zip(ref_accept, fused_accept):
        assert torch.equal(a, b)
    assert torch.equal(fused_next_rng, ref_next_rng)
    assert torch.equal(fused_tail, ref_tail)
    assert torch.equal(kv_fused, kv_ref)


def test_block_spec_worker_step_v2_dispatches_three_steps_and_falls_back():
    bs = 4
    width = 4
    n = width - 1
    calls: list[str] = []

    worker = object.__new__(TpuBlockSpecWorker)
    worker._fuse_step_ops_v2 = True
    worker._fuse_step_ops = True
    worker._step_timers_enabled = False
    worker._target_has_linear_attn = False
    worker._simulate_acc_len = 0
    worker._sampled_drafts = False
    worker._verify_top_k_width = TARGET_CANDIDATES
    worker._verify_greedy_topk1 = True
    worker._block_verification = False
    worker._rank_check_interval = 0
    worker._mask_token_id = 999
    worker._draft_block_width = width
    worker.num_verify_rows = width
    worker.num_drafts = n
    worker._spec_algorithm = SimpleNamespace(name="DFLASH")
    worker._draft_block_spec_info = SimpleNamespace()
    worker._all_drafts = {bs: torch.full((bs,), n, dtype=torch.int32)}
    worker._rng_step = torch.tensor([0], dtype=torch.int64)
    worker.device = "cpu"
    worker._target_worker = SimpleNamespace(
        model_runner=SimpleNamespace(
            ngram_embedding_manager=SimpleNamespace(engram_hasher=None)
        )
    )
    worker._draft_runner = SimpleNamespace(
        compile_policy=object.__new__(TpuBlockSpecDecodeCompilePolicyMixin)
    )
    worker._sync_greedy_drafts = lambda t: t

    packed = torch.arange(bs * 9, dtype=torch.int64).view(bs, 9)
    bonus = torch.tensor([1, 2, 3, 4], dtype=torch.int64)
    seq_lens = torch.tensor([10, 11, 12, 13], dtype=torch.int64)
    req_pool = torch.arange(bs, dtype=torch.int64)
    live_mask = torch.ones(bs, dtype=torch.bool)
    bucket = BucketState(
        key=req_pool,
        state=SpecState(bonus_tokens=bonus, seq_lens=seq_lens, req_pool_indices=req_pool),
        live_mask=live_mask,
        temperatures=None,
        top_ks=None,
        top_ps=None,
        min_ps=None,
        need_top_k=False,
        need_top_p=False,
        need_min_p=False,
        target_candidates=False,
    )
    verify_input = SimpleNamespace(
        packed_addressing=packed,
        seq_lens_cpu=seq_lens,
        req_pool_indices_cpu=req_pool,
    )

    dummy_rows, _, _ = prepare_dflash_block_step(
        packed, bonus, seq_lens, live_mask, width, 999, width, False
    )
    dummy_drafts = torch.full((bs, n), 42, dtype=torch.int64)
    dummy_hidden = torch.zeros((bs * width, 8), dtype=torch.float32)
    dummy_verify_toks = verify_tokens_of(bonus, dummy_drafts)
    dummy_accept = accept_greedy_top_k_step(
        torch.full((bs * width, 1), 42, dtype=torch.int32),
        dummy_drafts,
        torch.full((bs,), n, dtype=torch.int32),
        seq_lens,
        live_mask,
    )
    dummy_tail = torch.tensor([1.0], dtype=torch.float32)

    worker._block_draft_worker = SimpleNamespace(
        full_to_swa=None,
        swa_ring_pool=None,
        draft_context=contextlib.nullcontext,
        accept_and_write_draft_kv_greedy_top_k=lambda **kw: (
            calls.append("launch3_accept_write") or (dummy_accept, dummy_tail)
        ),
    )
    worker.make_verify_input = lambda **kw: SimpleNamespace()
    worker._verify_logits_output_mode = lambda *a: None
    worker._forward_target_verify = lambda **kw: (
        calls.append("launch2_verify")
        or LogitsProcessorOutput(
            next_token_logits=None,
            hidden_states=dummy_hidden,
            next_token_global_top_k_logits=torch.zeros(
                (bs * width, 1), dtype=torch.float32
            ),
            next_token_global_top_k_token_ids=torch.zeros(
                (bs * width, 1), dtype=torch.int32
            ),
        )
    )
    worker._check_rank_agreement = lambda *a: None

    with patch(
        "sglang_tpu.experimental.speculative.block_spec_worker.prep_and_draft_rows",
        side_effect=lambda **kw: (
            calls.append("launch1_prep_draft")
            or (dummy_rows, None, dummy_hidden.view(bs, width, 8), dummy_drafts, None, dummy_verify_toks)
        ),
    ):
        committed, _logits, accept = worker.step(None, bucket, verify_input)

    assert calls == ["launch1_prep_draft", "launch2_verify", "launch3_accept_write"]
    assert torch.equal(committed.kv_fence_tail, dummy_tail)
    assert torch.equal(accept.bonus_tokens, dummy_accept.bonus_tokens)

    # Full-vocab logits fallback cleanly delegates to _accept + commit without Launch 3 fusion.
    calls.clear()
    fallback_accept = dummy_accept
    fallback_commit = SimpleNamespace(state=bucket.state, kv_fence_tail=dummy_tail)
    worker._forward_target_verify = lambda **kw: (
        calls.append("launch2_verify_fullvocab")
        or LogitsProcessorOutput(next_token_logits=torch.zeros((bs * width, 16)), hidden_states=dummy_hidden)
    )
    worker._accept = lambda *a: (calls.append("fallback_accept") or fallback_accept)
    worker.commit = lambda *a, **kw: (calls.append("fallback_commit") or fallback_commit)
    with patch(
        "sglang_tpu.experimental.speculative.block_spec_worker.prep_and_draft_rows",
        side_effect=lambda **kw: (
            calls.append("launch1_prep_draft")
            or (dummy_rows, None, dummy_hidden.view(bs, width, 8), dummy_drafts, None, dummy_verify_toks)
        ),
    ):
        committed2, _logits2, _accept2 = worker.step(None, bucket, verify_input)
    assert calls == [
        "launch1_prep_draft",
        "launch2_verify_fullvocab",
        "fallback_accept",
        "fallback_commit",
    ]
    assert committed2 is fallback_commit


def test_prep_and_draft_rows_sets_real_bounds_and_skips_prepare_forward_batch():
    bs, padded_bs, width = 2, 4, 4
    mask_token_id = 999
    packed = torch.arange(padded_bs * (2 * width + 1), dtype=torch.int64).view(
        padded_bs, 2 * width + 1
    )
    bonus = torch.tensor([1, 2, 0, 0], dtype=torch.int64)
    state_seq_lens = torch.tensor([10, 14, 0, 0], dtype=torch.int64)
    live_mask = torch.tensor([True, True, False, False], dtype=torch.bool)
    req_pool = torch.tensor([1, 2, 0, 0], dtype=torch.int64)
    verify_input = TpuSpecVerifyInput(
        verify_cache_candidates=None,
        verify_cache_candidates_cpu=None,
        committed_lens=None,
        req_pool_indices=req_pool,
        req_pool_indices_cpu=req_pool,
        seq_lens_cpu=state_seq_lens[:bs],
        host_len_slack=0,
        future_indices=None,
        packed_addressing=packed,
    )
    spec_info = make_draft_block_spec_info(draft_token_num=width, device="cpu")
    policy = _make_dummy_policy()

    def fake_draft_forward(input_ids, positions, forward_batch):
        del positions
        next_draft = torch.zeros((padded_bs, width - 1), dtype=torch.int64)
        return LogitsProcessorOutput(
            next_token_logits=None,
            next_draft_tokens=next_draft,
            next_draft_lattice=(
                None,
                verify_tokens_of(input_ids.view(padded_bs, width)[:, 0], next_draft),
            ),
            hidden_states=torch.zeros((padded_bs * width, 4), dtype=torch.float32),
        )

    policy._eager_model_forward = fake_draft_forward
    recorded_init_batches: list[tuple[int, int, int, int, torch.Tensor | None]] = []

    def init_forward_metadata(fb: ForwardBatch):
        recorded_init_batches.append(
            (
                getattr(fb, "_tpu_real_b", -1),
                getattr(fb, "_tpu_real_q", -1),
                fb.spec_info.padding_state.logical_batch_size,
                fb.spec_info.padding_state.physical_batch_size,
                fb.input_ids,
            )
        )
        draft_runner.attn_backend.forward_metadata = PallasForwardMetadata(
            kv_lens=(fb.seq_lens + width).to(torch.int32),
            cu_q_lens=torch.arange(padded_bs + 1, dtype=torch.int32) * width,
            cu_kv_lens=torch.zeros(padded_bs + 1, dtype=torch.int32),
            page_indices=torch.zeros(padded_bs * 4, dtype=torch.int32),
            distribution=torch.tensor([bs, bs, bs], dtype=torch.int32),
            active_num_tokens=torch.tensor(bs * width, dtype=torch.int32),
        )

    def forbidden_prepare_forward_batch(_fb):
        raise AssertionError(
            "prep_and_draft_rows must not call prepare_forward_batch with input_ids=None"
        )

    draft_runner = SimpleNamespace(
        compile_policy=policy,
        model=SimpleNamespace(prepare_forward_batch=forbidden_prepare_forward_batch),
        attn_backend=SimpleNamespace(
            forward_metadata=None, init_forward_metadata=init_forward_metadata
        ),
    )

    with (
        patch(
            "sglang_tpu.experimental.speculative.draft_forward.is_dp_attention_enabled",
            return_value=False,
        ),
        patch(
            "sglang_tpu.experimental.speculative.draft_forward.enable_num_token_non_padded",
            return_value=False,
        ),
    ):
        rows, swa_rows, hidden, drafts, lattice, verify_toks = prep_and_draft_rows(
            draft_runner=draft_runner,
            spec_algorithm=SimpleNamespace(name="DFLASH"),
            packed_addressing=packed,
            bonus_tokens=bonus,
            state_seq_lens=state_seq_lens,
            live_mask=live_mask,
            req_pool_indices=req_pool,
            verify_input=verify_input,
            num_verify_rows=width,
            mask_token_id=mask_token_id,
            width=width,
            has_swa=False,
            spec_info=spec_info,
        )

    assert swa_rows is None and lattice is None
    assert rows.positions.shape == (padded_bs, width)
    assert hidden.shape == (padded_bs, width, 4)
    assert drafts.shape == (padded_bs, width - 1)
    assert verify_toks.shape == (padded_bs * width,)
    assert recorded_init_batches == [(bs, bs * width, bs, padded_bs, None)]


def test_launch1_dynamo_zero_recompile_across_varying_bs_and_host_fields():
    compile_count = 0

    def counting_backend(gm, example_inputs):
        nonlocal compile_count
        compile_count += 1
        return gm.forward

    padded_bs, width = 4, 4
    cand_cols = 2 * width
    policy = _make_dummy_policy()
    policy.backend = counting_backend
    policy.fullgraph = True

    def fake_draft_forward(input_ids, positions, forward_batch):
        del positions
        assert forward_batch.spec_info.rpa_draft_kv_lens is not None
        next_draft = torch.zeros((padded_bs, width - 1), dtype=torch.int64)
        return LogitsProcessorOutput(
            next_token_logits=None,
            next_draft_tokens=next_draft,
            next_draft_lattice=(
                None,
                verify_tokens_of(input_ids.view(padded_bs, width)[:, 0], next_draft),
            ),
            hidden_states=torch.zeros((padded_bs * width, 4), dtype=torch.float32),
        )

    policy._eager_model_forward = fake_draft_forward
    attn_backend = SimpleNamespace(forward_metadata=None)

    from sglang.srt.environ import envs

    with (
        patch("sglang_tpu.experimental.compilation.block_compile_policy.is_tpu", return_value=True),
        patch.object(envs, "SGLANG_ENABLE_TORCH_COMPILE", SimpleNamespace(get=lambda: True)),
        patch.object(
            tpu_envs,
            "SGLANG_TPU_VERIFY_SKIP_GUARD_EVAL",
            SimpleNamespace(get=lambda: False),
        ),
    ):
        for bs_live, sum_val in ((3, 60), (2, 24), (4, 120)):
            packed = torch.randint(0, 100, (padded_bs, cand_cols + 1), dtype=torch.int64)
            bonus = torch.arange(padded_bs, dtype=torch.int64)
            seq_lens = torch.tensor([10, 20, 30, 40], dtype=torch.int64)
            live_mask = torch.arange(padded_bs) < bs_live
            # Distinct per-step host fields & distinct spec_info instance on ForwardBatch
            # must NOT trigger any Dynamo guard failure / recompilation.
            fb = ForwardBatch(
                forward_mode=ForwardMode.TARGET_VERIFY,
                out_cache_loc_is_physical=True,
                batch_size=padded_bs,
                input_ids=None,
                req_pool_indices=torch.arange(padded_bs, dtype=torch.int64),
                seq_lens=seq_lens,
                seq_lens_sum=sum_val,
                out_cache_loc=None,
                positions=None,
                extend_prefix_lens=None,
                extend_seq_lens=None,
                extend_prefix_lens_cpu=[bs_live * 5] * padded_bs,
                extend_seq_lens_cpu=[width] * bs_live + [0] * (padded_bs - bs_live),
                extend_num_tokens=bs_live * width,
                global_num_token_non_padded_cpu=bs_live * width,
                spec_info=make_draft_block_spec_info(draft_token_num=width, device="cpu"),
            )
            with forward_context(
                ForwardContext(attn_backend=attn_backend), forward_batch=fb
            ):
                policy.run_prep_and_draft(
                    forward_batch=fb,
                    packed_addressing=packed,
                    bonus_tokens=bonus,
                    state_seq_lens=seq_lens,
                    live_mask=live_mask,
                    num_verify_rows=width,
                    mask_token_id=999,
                    draft_width=width,
                    has_swa=False,
                )

    assert compile_count == 1
