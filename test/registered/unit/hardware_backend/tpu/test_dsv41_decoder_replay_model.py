# SPDX-License-Identifier: Apache-2.0
"""Decoder SWA bounded replay (CED prefill) end to end on CPU: the real core
``DeepseekV4Model.forward`` and its layer loop, the real DeepSeek-V4.1 TPU
attention backend and KV pool, with torch reference CSA2 / indexer ops and small
deterministic layer bodies (attention projections and a row-wise FFN for MoE).

Three runs over the same chunked prefill:

- REPLAY: decoder layers on each request's decoder rows only (padded buckets);
- FLOORED: decoder layers on every row, with the decoder SWA window floored at
  the same replay starts (the exact reference of the approximation);
- FULL: no replay.

REPLAY equals FLOORED bit for bit on the decoder rows (outputs, DSpark aux rows
and the decoder layers' SWA state); every encoder-layer state (SWA, compressed
main KV, index keys, ratio-2 pair state) equals FULL bit for bit. REPLAY vs FULL
on the decoder rows is the approximation itself and is reported, not asserted.

Layer bodies reduce row by row (no GEMM), so a row's numbers do not depend on
how many rows share the call; the TPU kernels' row independence is the TPU
correctness run's business.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=480, suite="base-a-test-cpu")

import contextlib
import functools
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.dsv4.dsv41_sparse import (
    DeepseekV41Compressor,
    RMSNorm,
)
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.model_executor.forward_mode import CaptureHiddenMode, ForwardMode
from sglang.srt.models.deepseek_v4 import DeepseekV4Model
from sglang.srt.models.deepseek_v4_mhc import HcState
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.models.deepseek_v41 import decoder_replay
from sglang_tpu.models.deepseek_v41.attention_backend import (
    DSV41TPUAttnBackend,
    rope_reference,
)
from sglang_tpu.models.deepseek_v41.decoder_replay import DecoderRowPlan
from sglang_tpu.models.deepseek_v41.hooks import tpu_dsv41_hc_combine
from sglang_tpu.models.deepseek_v41.memory_pool import DSV41TPUTokenToKVPool

# A tiny V4.1: ratio-0 layer, a ratio-2 kv_source, a ratio-1 kv_source that is
# also the candidate source, then four stateless decoder layers (one index
# source consuming candidates) and DSpark aux layers inside the decoder.
RATIOS = [0, 2, 2, 1, 1, 1, 1, 1]
KV_SOURCES = [1, 3]
INDEX_SOURCES = [1, 3, 5]
CANDIDATE_SOURCE = 3
LATE = max(KV_SOURCES) + 1
AUX_LAYERS = [2, 6, 7]  # one encoder layer: its tail rows scatter back too
H, HC, HEADS, QLORA, IDX_HEADS = 64, 2, 8, 32, 4
TOPK, CAND_BLOCKS, CAND_BLOCK = 16, 4, 8
MAX_POS, CTX, SLOT_TOKENS = 2048, 1024, 1024
EPS = 1e-6


# =============================================================================
# Reference ops (self-consistent stand-ins for ops/csa2.py and ops/indexer.py)
# =============================================================================


def _fp8_rows(x: torch.Tensor):
    scale = x.float().abs().amax(-1).clamp(min=1e-4) / 448.0
    q = (x.float() / scale[:, None]).to(torch.float8_e4m3fn)
    return q, scale


def _token_requests(qsl: torch.Tensor, num_tokens: int) -> torch.Tensor:
    req = torch.full((num_tokens,), -1, dtype=torch.int64)
    counts = (qsl[1:] - qsl[:-1]).long()
    real = int(qsl[-1])
    req[:real] = torch.repeat_interleave(torch.arange(counts.shape[0]), counts)
    return req


class RefCsa2:
    MAX_SELECTED = 512
    # (rows, extend_tiling, has main) of every attention call.
    tilings: list = []

    @staticmethod
    def swa_metadata(table, qsl, positions, minimum, non_causal=False):
        req = _token_requests(qsl, positions.shape[0])
        pos = positions.long()
        page = table.long()[req.clamp(min=0), pos // 64]
        rows = torch.where((req >= 0) & (page > 0), page * 64 + pos % 64, -1)
        meta = torch.cat([minimum.view(-1, 1).int(), table.int()], 1)
        return meta, torch.zeros(table.shape[0], 2, dtype=torch.int32), rows.int()

    @staticmethod
    def publish_swa(pages, records, kv):
        out = pages.clone()
        flat = out.view(-1, out.shape[-1])
        ok = records >= 64
        flat[records[ok].long()] = kv.reshape(-1, out.shape[-1])[ok].to(out.dtype)
        return out

    @staticmethod
    def publish_main(words, scales, records, latent):
        q, scale = _fp8_rows(latent)
        # Byte g of lane l is feature 128 g + l.
        packed = (
            (q.view(torch.uint8).view(-1, 4, 128).permute(0, 2, 1).contiguous())
            .view(torch.int32)
            .view(-1, 128)
        )
        ok = records >= 0
        words, scales = words.clone(), scales.clone()
        words[records[ok].long()] = packed[ok]
        scales[records[ok].long()] = scale[ok]
        return words, scales

    @staticmethod
    def decode_main(words, scales):
        q = words.contiguous().view(torch.uint8).view(-1, 128, 4).permute(0, 2, 1)
        q = q.reshape(-1, 512).contiguous().view(torch.float8_e4m3fn)
        return q.float() * scales[:, None]

    @staticmethod
    def write_rows(rows, slots, values, after):
        out = rows.clone()
        ok = slots >= 0
        out[slots[ok].long()] = values[ok].to(out.dtype)
        return out

    @classmethod
    def attention(
        cls,
        q,
        swa_pages,
        meta,
        bounds,
        qsl,
        positions,
        sink,
        cos_sin,
        compress_ratio,
        scale,
        main=None,
        keep_gathered=False,
        extend_tiling=False,
    ):
        # The fixture's backend has no gather groups: every layer gathers.
        assert not keep_gathered
        cls.tilings.append((q.shape[0], bool(extend_tiling), main is not None))
        minimum, table = meta[:, 0].long(), meta[:, 1:].long()
        flat = swa_pages.view(-1, swa_pages.shape[-1]).float()
        req = _token_requests(qsl, q.shape[0])
        out = torch.zeros(q.shape, dtype=torch.float32)
        for t in range(int(qsl[-1])):
            b, p = int(req[t]), int(positions[t])
            low = max(p - 127, int(minimum[b]), 0)
            keys = []
            if main is not None:
                words, scales, selected, main_pages, per_page = main
                ids = selected[t][selected[t] >= 0].long()
                rec = main_pages.long()[b, ids // per_page] * per_page + ids % per_page
                keys.append(cls.decode_main(words[rec], scales[rec]))
            if low <= p:
                pos = torch.arange(low, p + 1)
                keys.append(flat[table[b, pos // 64] * 64 + pos % 64])
            if not keys:
                continue
            k = torch.cat(keys)
            logits = (q[t].float().unsqueeze(1) * k.unsqueeze(0)).sum(-1) * scale
            m = torch.maximum(logits.max(-1).values, sink)
            w = torch.exp(logits - m[:, None])
            denom = w.sum(-1) + torch.exp(sink - m)
            out[t] = (w.unsqueeze(-1) * k.unsqueeze(0)).sum(1) / denom[:, None]
        out = rope_reference(out, positions.long(), cos_sin, inverse=True)
        return out.to(q.dtype)


class RefIndexer:
    @staticmethod
    def _blocks(index_k):
        b = index_k.view(-1, 33, 128)
        data = b[:, :32].contiguous().view(torch.uint8).view(b.shape[0], 128, 128)
        return data, b[:, 32]

    @classmethod
    def write_index_k(cls, index_k, records, keys, layout):
        out = index_k.clone()
        data, scales = cls._blocks(out)
        q, scale = _fp8_rows(keys.reshape(-1, 128))
        ok = records >= 0
        r = records[ok].long()
        data[r // 128, r % 128] = q[ok].view(torch.uint8)
        scales.view(torch.float32)[r // 128, r % 128] = scale[ok]
        return out

    @classmethod
    def _keys(cls, index_k, pages_row, ids):
        data, scales = cls._blocks(index_k)
        page = pages_row.long()[ids // 128]
        q = data[page, ids % 128].contiguous().view(torch.float8_e4m3fn).float()
        return q * scales.view(torch.float32)[page, ids % 128][:, None]

    @classmethod
    def select(
        cls,
        q,
        weights,
        index_k,
        index_pages,
        requests,
        lengths,
        candidates,
        layout,
        *,
        topk,
        block_size,
        publish,
        consume,
    ):
        width = index_pages.shape[1] * 128
        selected = torch.full((q.shape[0], min(topk, width)), -1, dtype=torch.int32)
        blocks = candidates.clone()
        for t in range(q.shape[0]):
            row, length = int(requests[t]), int(lengths[t])
            if row < 0 or length <= 0:
                continue
            ids = torch.arange(length)
            keys = cls._keys(index_k, index_pages[row], ids)
            s = (q[t].float().unsqueeze(1) * keys.unsqueeze(0)).sum(-1).relu()
            s = (s * weights[t].float()[:, None]).sum(0)
            if publish:
                nb = -(-length // block_size)
                bmax = torch.full((nb,), float("-inf"))
                bmax = bmax.scatter_reduce(0, ids // block_size, s, "amax")
                order = sorted(range(nb), key=lambda i: (-float(bmax[i]), i))
                take = order[: blocks.shape[1]]
                blocks[t] = -1
                blocks[t, : len(take)] = torch.tensor(take, dtype=torch.int32)
            if consume:
                allowed = torch.isin(
                    ids // block_size, candidates[t][candidates[t] >= 0].long()
                )
                s = torch.where(allowed, s, float("-inf"))
            if length <= topk and not consume:
                pick = ids
            else:
                order = sorted(range(length), key=lambda i: (-float(s[i]), i))
                pick = torch.tensor(
                    [i for i in order[:topk] if s[i] != float("-inf")],
                    dtype=torch.int64,
                )
            pick = pick.sort().values
            selected[t, : pick.shape[0]] = pick.int()
        return selected, blocks

    @staticmethod
    def uses_prepared_candidates(**_):
        return False


# =============================================================================
# Tiny model: real core loop, stub layer bodies on the real backend
# =============================================================================


def _rowwise(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """x [T, K] @ w [K, N] reduced per row, independent of T."""
    return (x.float().unsqueeze(-1) * w.float().unsqueeze(0)).sum(-2)


def _rope_table() -> torch.Tensor:
    inv = 1.0 / (10000 ** (torch.arange(0, 64, 2).float() / 64))
    angles = torch.arange(MAX_POS).float()[:, None] * inv[None]
    return torch.cat([angles.cos(), angles.sin()], -1)


class _StubIndexer(nn.Module):
    def __init__(self, layer_id: int, g: torch.Generator):
        super().__init__()
        self.n_heads = self.n_local_heads = IDX_HEADS
        self.index_head_dim = 128
        self.owns_k = layer_id in KV_SOURCES
        self.is_candidate_source = layer_id == CANDIDATE_SOURCE
        self.uses_candidates = 0 <= CANDIDATE_SOURCE < layer_id
        self.candidate_topk_blocks = CAND_BLOCKS
        self.candidate_block_size = CAND_BLOCK
        self.head_weight_scale = 128**-0.5 * IDX_HEADS**-0.5
        self.wq = torch.randn(QLORA, IDX_HEADS * 128, generator=g) * QLORA**-0.5
        self.wh = torch.randn(H, IDX_HEADS, generator=g) * H**-0.5
        if self.owns_k:
            self.wk = nn.Linear(512, 128, bias=False, dtype=torch.bfloat16)
            self.k_norm = RMSNorm(128, EPS)

    def wq_b(self, q_lora):
        return _rowwise(q_lora, self.wq).to(torch.bfloat16), None

    def head_weights_raw(self, x):
        return _rowwise(x, self.wh).to(torch.bfloat16)


class _StubAttn(nn.Module):
    def __init__(self, layer_id: int, g: torch.Generator):
        super().__init__()
        self.layer_id = layer_id
        self.compress_ratio = RATIOS[layer_id]
        self.compressor = None
        self.indexer = None
        if self.compress_ratio and layer_id in KV_SOURCES:
            self.compressor = DeepseekV41Compressor(
                hidden_size=H,
                head_dim=512,
                compress_ratio=self.compress_ratio,
                eps=EPS,
                fused_compress=False,
            )
        if self.compress_ratio and layer_id in INDEX_SOURCES:
            self.indexer = _StubIndexer(layer_id, g)
        self.register_buffer("freqs_cis", _rope_table(), persistent=False)
        self.wqa = torch.randn(H, QLORA, generator=g) * H**-0.5
        self.wqb = torch.randn(QLORA, HEADS * 512, generator=g) * QLORA**-0.5
        self.wkv = torch.randn(H, 512, generator=g) * H**-0.5
        self.wo = torch.randn(HEADS * 512, H, generator=g) * (HEADS * 512) ** -0.5
        self.sink = torch.randn(HEADS, generator=g) * 0.1
        self.radix = SimpleNamespace(
            layer_id=layer_id, scaling=512**-0.5, v_head_dim=512
        )


class _StubLayer(nn.Module):
    engram = None
    hc_boundary_fused = False

    def __init__(self, layer_id: int, g: torch.Generator, backend_ref):
        super().__init__()
        self.self_attn = _StubAttn(layer_id, g)
        self.norm = RMSNorm(H, EPS)
        self.input_layernorm = self.norm
        self.w1 = torch.randn(H, 2 * H, generator=g) * H**-0.5
        self.w2 = torch.randn(2 * H, H, generator=g) * (2 * H) ** -0.5
        self.mix = torch.linspace(0.5, 1.5, HC)
        self._backend = backend_ref
        # Read when the model materializes its last state; the stub's is plain.
        self.hc_cfg = None

    def forward_hc_pre_from_prev(self, *, positions, state, forward_batch, **_):
        hidden_states = state.residual
        backend = self._backend()
        attn = self.self_attn
        x = self.norm(hidden_states.float().mean(1)).to(torch.bfloat16)
        q_lora = _rowwise(x, attn.wqa).to(torch.bfloat16)
        if attn.compressor is not None or attn.indexer is not None:
            backend.forward_low_ratio_sources(
                layer=attn,
                x=x,
                q_lora=q_lora,
                positions=positions,
                forward_batch=forward_batch,
            )
        table = attn.freqs_cis
        q = _rowwise(q_lora, attn.wqb).view(-1, HEADS, 512).to(torch.bfloat16)
        q = rope_reference(q, positions.long(), table)
        kv = RMSNorm(512, EPS)(_rowwise(x, attn.wkv)).to(torch.bfloat16)
        kv = rope_reference(kv, positions.long(), table)
        o = backend.forward(
            q,
            kv,
            None,
            attn.radix,
            forward_batch,
            compress_ratio=attn.compress_ratio,
            attn_sink=attn.sink,
        )
        h = (
            hidden_states.float()
            + _rowwise(o.reshape(o.shape[0], -1), attn.wo)[:, None, :]
            * self.mix[None, :, None]
        )
        y = _rowwise(
            torch.nn.functional.silu(_rowwise(self.norm(h.mean(1)), self.w1)), self.w2
        )
        h = (h + y[:, None, :]).to(torch.bfloat16)
        pre = torch.softmax(h.float().mean(-1), -1)
        return HcState(h, pre)


def _make_pool(testcase, window: bool):
    from sglang.srt.runtime_context import get_context

    extra = (
        dict(enable_encoder_swa_bounded_replay=True, chunked_prefill_size=1024)
        if window
        else {}
    )
    override = get_context().override_server_args(page_size=256, **extra)
    override.install()
    testcase.addCleanup(override.restore)
    pool = DSV41TPUTokenToKVPool(
        max_num_reqs=4,
        num_req_slots=5,
        swa_size=8192,
        c4_size=0,
        c128_size=0,
        c4_state_pool_size=0,
        c128_state_pool_size=0,
        page_size=256,
        swa_page_size=256,
        sliding_window=128,
        dtype=torch.bfloat16,
        c4_state_dtype=torch.float32,
        c128_state_dtype=torch.float32,
        qk_nope_head_dim=448,
        qk_rope_head_dim=64,
        indexer_head_dim=128,
        layer_num=len(RATIOS),
        device="cpu",
        enable_memory_saver=False,
        compression_ratios=RATIOS,
        start_layer=0,
        end_layer=len(RATIOS),
        kv_source_layers=KV_SOURCES,
        full_size=8192,
    )
    return pool


def _loc(slot: int, pos):
    return slot * SLOT_TOKENS + pos


class _Run:
    """One model + backend + pool, fed the same chunked prefill."""

    def __init__(self, testcase, *, mode: str, window: bool, seed: int = 0):
        self.mode = mode
        torch.manual_seed(seed)
        g = torch.Generator().manual_seed(seed)
        self.pool = _make_pool(testcase, window)
        assert (self.pool.request_window is not None) == window
        self.pool.full_to_swa_index_mapping = None
        req_to_token = torch.zeros(8, CTX, dtype=torch.int64)
        for slot in range(1, 5):
            req_to_token[slot] = _loc(slot, torch.arange(CTX))
        cfg = SimpleNamespace(
            index_source_layer_ids=INDEX_SOURCES,
            compress_ratios=RATIOS,
            candidate_source_layer_id=CANDIDATE_SOURCE,
        )
        be = object.__new__(DSV41TPUAttnBackend)
        AttentionBackend.__init__(be)
        be.model_runner = SimpleNamespace(model=None)
        be.device = torch.device("cpu")
        be.req_to_token = req_to_token
        be.token_to_kv_pool = self.pool
        be.request_window = self.pool.request_window
        be.max_context_len = CTX
        be.page_size = 256
        be.index_topk = TOPK
        be.forward_metadata = None
        be._slot_tables = None
        be.is_dspark_draft = False
        be.low_ratios = (1, 2)
        be.last_candidate_consumer = be._last_candidate_consumers(cfg)
        be.rope_owners = {}
        be.csa2_ops, be.indexer_ops = RefCsa2, RefIndexer
        be.selection_width = RefCsa2.MAX_SELECTED
        be.decoder_replay = mode != "full"
        be.decoder_low_ratios = (1,)
        be.gather_groups = {}
        self.backend = be

        model = object.__new__(DeepseekV4Model)
        nn.Module.__init__(model)
        model.pp_group = SimpleNamespace(
            world_size=1, is_first_rank=True, is_last_rank=True
        )
        model.hc_mult, model.hidden_size = HC, H
        model.engram_hasher = None
        model.start_layer, model.end_layer = 0, len(RATIOS)
        model.layers = nn.ModuleList(
            _StubLayer(i, g, lambda: self.backend) for i in range(len(RATIOS))
        )
        for layer in model.layers:
            be.rope_owners[layer.self_attn.layer_id] = layer.self_attn
        model.late_layer_start = None if mode == "full" else LATE
        # As start_tail_after_last_source leaves them on TPU, without vision.
        model.tail_writes_source_kv = False
        model.late_layer_vl_routing = False
        model.use_fused_mhc_post_pre = False
        model.config = SimpleNamespace(model_type="deepseek_v41", vision_n_layers=0)
        model.dspark_layers_to_capture = AUX_LAYERS
        model.hc_pre_from_prev_sublayer = True
        model.norm = RMSNorm(H, EPS)
        model._can_run_tbo = lambda fb: False
        self.model = model
        self.embed = torch.randn(4096, H, generator=g).to(torch.bfloat16)

    def forward(self, chunk, *, bucket=None, head=False):
        """``chunk``: (slot, start, count, fill, token ids) per request. ``head``
        runs the real DeepseekV4ForCausalLM.forward around the model (stub
        logits processor) and returns what the head handed it."""
        positions = torch.cat([torch.arange(s, s + n) for _, s, n, _, _ in chunk])
        ids = torch.cat([t for *_, t in chunk])
        locs = torch.cat(
            [_loc(slot, torch.arange(s, s + n)) for slot, s, n, _, _ in chunk]
        )
        pad = 3
        fb = SimpleNamespace(
            batch_size=len(chunk),
            positions=torch.cat([positions, torch.zeros(pad, dtype=torch.int64)]),
            input_ids=torch.cat([ids, torch.zeros(pad, dtype=torch.int64)]),
            out_cache_loc=torch.cat([locs, torch.zeros(pad, dtype=torch.int64)]),
            forward_mode=ForwardMode.EXTEND,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            return_logprob=False,
            extend_seq_lens_cpu=[n for _, _, n, _, _ in chunk],
            extend_seq_lens=torch.tensor([n for _, _, n, _, _ in chunk]),
            extend_prefix_lens_cpu=[s for _, s, _, _, _ in chunk],
            req_pool_indices=torch.tensor([slot for slot, *_ in chunk]),
            encoder_swa_row_floor=None,
            encoder_swa_compress_rows=None,
            attn_cp_metadata=None,
            decoder_fill_lens=[f for *_, f, _ in chunk],
            decoder_rows_bucket=bucket,
        )
        plan = contextlib.nullcontext()
        if self.mode == "floored":
            # Every row runs the decoder; the window floors stay the replay starts.
            real = decoder_replay.plan_decoder_rows

            def keep_all(**kw):
                p = real(**kw)
                return DecoderRowPlan(p.replay_starts, list(kw["extend_lens"]))

            plan = patch.object(decoder_replay, "plan_decoder_rows", keep_all)
        with plan:
            self.backend.init_forward_metadata(fb)
        md = self.backend.forward_metadata
        # The real decoder rows (None: no replay).
        if md.tail is None:
            rows = None
        else:
            rows = md.tail_gather[: int(md.tail.valid.sum())]
        from sglang.srt.models import deepseek_v4 as core_model
        from sglang.srt.plugins.hook_registry import HookType, _wrap_fn
        from sglang_tpu.models.deepseek_v41.decoder_replay import (
            clear_full_layout_head,
            scatter_decoder_rows,
            scatter_dspark_capture,
            skip_empty_decoder,
        )

        # The TPU hooks on core (decoder_replay_sync.hook_specs).
        scatter = _wrap_fn(
            core_model._scatter_tail_rows, scatter_decoder_rows, HookType.AROUND
        )
        layers = _wrap_fn(
            core_model.DeepseekV4Model._forward_layers_hc_pre_from_prev,
            skip_empty_decoder,
            HookType.AROUND,
        )
        model_forward = _wrap_fn(
            core_model.DeepseekV4Model.forward, scatter_dspark_capture, HookType.AROUND
        )
        with (
            patch.object(core_model.DeepseekV4Model, "forward", model_forward),
            patch.object(core_model, "_scatter_tail_rows", scatter),
            patch.object(
                core_model.DeepseekV4Model, "_forward_layers_hc_pre_from_prev", layers
            ),
            forward_context(ForwardContext(attn_backend=self.backend)),
            patch(
                "sglang.kernels.ops.layernorm.mhc.mhc_combine",
                lambda x, pre, hc, dtype: tpu_dsv41_hc_combine(None, x, pre, hc, dtype),
            ),
            torch.no_grad(),
        ):
            if head:
                return self._head(fb, core_model, clear_full_layout_head)
            (hidden, pre_hc), aux = self.model.forward(
                fb.input_ids, fb.positions, fb, self.embed[fb.input_ids]
            )
            aux = _captures(aux)
        # The causal-LM head of this forward would read the identity tail.
        self.head_full_layout = self.backend.head_full_layout is fb
        self.backend.head_full_layout = None
        self.backend.forward_metadata = None
        return hidden, pre_hc, aux, rows, positions


def _captures(aux):
    """The per-layer DSpark captures; core packs them into one [rows, K * H]."""
    return list(aux.split(H, dim=-1)) if isinstance(aux, torch.Tensor) else aux


def _head_forward(run, fb, core_model, clear_hook):
    from sglang.srt.layers.logits_processor import LogitsMetadata
    from sglang.srt.plugins.hook_registry import HookType, _wrap_fn

    seen = {}

    def logits_processor(
        input_ids, hidden, lm_head, metadata, aux, hidden_states_before_norm=None
    ):
        seen.update(
            input_ids=input_ids, hidden=hidden, metadata=metadata, aux=_captures(aux)
        )
        return SimpleNamespace(hidden_states_token_indices=None)

    lm = SimpleNamespace(
        vision=None,
        model=run.model,
        pp_group=SimpleNamespace(is_last_rank=True),
        capture_aux_hidden_states=True,
        logits_processor=logits_processor,
        lm_head=None,
    )
    lm.config = run.model.config
    # The real head steps DeepseekV4ForCausalLM.forward runs around the model.
    for name in (
        "prepare_language_model_inputs",
        "_postprocess_output",
        "_get_dspark_prefill_tail",
    ):
        method = getattr(core_model.DeepseekV4ForCausalLM, name)
        setattr(lm, name, functools.partial(method, lm))
    head_forward = _wrap_fn(
        core_model.DeepseekV4ForCausalLM.forward, clear_hook, HookType.AROUND
    )
    try:
        with patch.object(
            LogitsMetadata,
            "from_forward_batch",
            staticmethod(lambda fb: SimpleNamespace()),
        ):
            out = head_forward(
                lm, fb.input_ids, fb.positions, fb, run.embed[fb.input_ids]
            )
    finally:
        run.backend.forward_metadata = None
    seen["out"] = out
    seen["head_full_layout_after"] = run.backend.head_full_layout
    return seen


_Run._head = lambda self, fb, core_model, clear_hook: _head_forward(
    self, fb, core_model, clear_hook
)


# Requests: (slot, fill length). Chunk 1 and chunk 2 of a chunked prefill.
_TOKENS = {
    slot: torch.randint(0, 4096, (fill,), generator=torch.Generator().manual_seed(slot))
    for slot, fill in ((1, 600), (2, 90), (3, 400))
}


def _chunk(slot, start, count):
    fill = _TOKENS[slot].shape[0]
    return (slot, start, count, fill, _TOKENS[slot][start : start + count])


# Prompt rule: slot 1's first chunk is wholly before its window (no decoder
# rows); slot 3's window (272..399) spans both chunks (28 + 100 rows). Chunk rule:
# each chunk keeps its own last <= 128 rows.
CHUNK1 = [_chunk(1, 0, 300), _chunk(2, 0, 90), _chunk(3, 0, 300)]
CHUNK2 = [_chunk(1, 300, 300), _chunk(3, 300, 100)]


class TestDecoderReplayModel(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.runs = {}

    window = False
    rule = "chunk"

    def setUp(self):
        from sglang_tpu.envs import envs as tpu_envs

        ctx = tpu_envs.SGLANG_TPU_DSV41_DECODER_REPLAY_ROWS.override(self.rule)
        ctx.__enter__()
        self.addCleanup(ctx.__exit__, None, None, None)

    def _run(self, mode, buckets=(None, None)):
        run = _Run(self, mode=mode, window=self.window)
        outs = [
            run.forward(CHUNK1, bucket=buckets[0]),
            run.forward(CHUNK2, bucket=buckets[1]),
        ]
        return run, outs

    @staticmethod
    def _kept(chunk):
        """[(slot, kept positions)] of a chunk under the active rule (no folds)."""
        out = []
        for slot, s, n, fill, _ in chunk:
            plan = decoder_replay.plan_decoder_rows(
                starts=[s], extend_lens=[n], fill_lens=[fill], floors=[0]
            )
            out.append((slot, torch.arange(s + n - plan.kept[0], s + n)))
        return out

    def _decoder_row_mask(self, positions, chunk):
        mask = torch.zeros(positions.shape[0] + 3, dtype=torch.bool)
        off = 0
        for (slot, s, n, fill, _), (_, kept) in zip(chunk, self._kept(chunk)):
            mask[off : off + n] = torch.isin(torch.arange(s, s + n), kept)
            off += n
        return mask

    def _decoder_positions(self):
        """(slot, positions) whose decoder SWA rows both runs keep: every kept
        row with paged SWA; under the request window, the last chunk's kept rows
        and the kept rows within each prompt's last window."""
        parts, last = {}, dict(self._kept(CHUNK2))
        for chunk in (CHUNK1, CHUNK2):
            for slot, kept in self._kept(chunk):
                parts.setdefault(slot, []).append(kept)
        result = {}
        for slot, kept in parts.items():
            pos = torch.cat(kept)
            if self.window:
                keep = pos >= _TOKENS[slot].shape[0] - 128
                if slot in last:
                    keep |= torch.isin(pos, last[slot])
                pos = pos[keep]
            result[slot] = pos
        return result

    def test_replay_equals_the_floored_reference_on_decoder_rows(self):
        replay, r_out = self._run("replay")
        padded, p_out = self._run("replay", buckets=(512, 256))
        floored, f_out = self._run("floored")
        full, u_out = self._run("full")
        for chunk, r, p, f, u in zip((CHUNK1, CHUNK2), r_out, p_out, f_out, u_out):
            mask = self._decoder_row_mask(r[4], chunk)
            # FULL plans no tail and fills every real row.
            self.assertIsNone(u[3])
            real = r[4].shape[0]
            self.assertGreater(float(u[0][:real].float().abs().amax(-1).min()), 0)
            # Exactly the planned rows ran the decoder (prompt rule).
            self.assertEqual(r[3].shape[0], int(mask.sum()))
            for a, b in ((r, f), (r, p)):
                self.assertTrue(torch.equal(a[0][mask], b[0][mask]))
                self.assertTrue(torch.equal(a[1][mask], b[1][mask]))
                for la, lb in zip(a[2], b[2]):
                    self.assertTrue(torch.equal(la[mask], lb[mask]))
            # Real rows the decoder skipped read as zeros (the logits processor
            # may read a chunk's last row even when it kept none).
            skipped = ~mask[:real]
            self.assertEqual(int(r[0][:real][skipped].abs().sum()), 0)
            for la in r[2]:
                self.assertEqual(int(la[:real][skipped].abs().sum()), 0)
            # The approximation itself, reported.
            if mask.any():
                delta = (r[0][mask].float() - u[0][mask].float()).abs().max().item()
                ref = u[0][mask].float().abs().max().item()
                print(
                    f"decoder rows vs FULL: max |delta| {delta:.4g} (max |x| {ref:.4g})"
                )

        # Decoder layers' SWA state on the decoder rows: REPLAY == FLOORED.
        for layer in range(LATE, len(RATIOS)):
            for slot, pos in self._decoder_positions().items():
                rows = self._swa_rows(slot, pos)
                a = replay.pool.get_swa_pages(layer).view(-1, 512)[rows]
                b = floored.pool.get_swa_pages(layer).view(-1, 512)[rows]
                c = padded.pool.get_swa_pages(layer).view(-1, 512)[rows]
                self.assertTrue(torch.equal(a, b), msg=f"layer {layer} slot {slot}")
                self.assertTrue(torch.equal(a, c), msg=f"layer {layer} slot {slot}")

        # Encoder-layer state: REPLAY == FULL bit for bit.
        written = torch.cat(
            [
                self._swa_rows(
                    slot,
                    torch.arange(t.shape[0])[-128:]
                    if self.window
                    else torch.arange(t.shape[0]),
                )
                for slot, t in _TOKENS.items()
            ]
        )
        for layer in range(LATE):
            a = replay.pool.get_swa_pages(layer).view(-1, 512)[written]
            b = full.pool.get_swa_pages(layer).view(-1, 512)[written]
            self.assertTrue(torch.equal(a, b), msg=f"encoder SWA layer {layer}")
        for source in KV_SOURCES:
            for x, y in zip(
                replay.pool.get_main_kv(source), full.pool.get_main_kv(source)
            ):
                self.assertTrue(torch.equal(x, y), msg=f"main KV {source}")
            self.assertTrue(
                torch.equal(
                    replay.pool.get_index_k(source), full.pool.get_index_k(source)
                )
            )
        state_r = replay.pool.get_attention_compress_states(1).kv_score_buffer.kv_score
        state_u = full.pool.get_attention_compress_states(1).kv_score_buffer.kv_score
        self.assertTrue(torch.equal(state_r, state_u))

    def test_no_kept_rows_skip_the_decoder_layers(self):
        """A batch wholly before every window (slot 1's first chunk) keeps no
        decoder rows: bucket 0 skips the decoder layers (no tail, no decoder
        attention), the full layout keeps its shape and finite values, the
        skipped DSpark capture layers give zeros, and the encoder state is still
        written. Prompt rule only: under the chunk rule every extend keeps its
        last rows."""
        from sglang_tpu.envs import envs as tpu_envs

        RefCsa2.tilings.clear()
        run = _Run(self, mode="replay", window=self.window)
        with tpu_envs.SGLANG_TPU_DSV41_DECODER_REPLAY_ROWS.override("prompt"):
            hidden, pre_hc, aux, rows, positions = run.forward(
                [_chunk(1, 0, 300)], bucket=0
            )
        self.assertIsNone(rows)
        # No tail: nothing to scatter, but the head still reads the identity.
        self.assertTrue(run.head_full_layout)
        self.assertEqual(len(RefCsa2.tilings), LATE)  # encoder attention only
        self.assertEqual(tuple(hidden.shape), (303, H))
        self.assertEqual(tuple(pre_hc.shape), (303, HC * H))
        self.assertTrue(bool(torch.isfinite(hidden).all()))
        self.assertEqual(len(aux), len(AUX_LAYERS))
        for layer, t in zip(AUX_LAYERS, aux):
            self.assertEqual(tuple(t.shape), (303, H))
            if layer >= LATE:
                self.assertEqual(int(t.abs().sum()), 0)
        full = _Run(self, mode="full", window=self.window)
        full.forward([_chunk(1, 0, 300)])
        for layer in range(LATE):
            rows_ = self._swa_rows(1, torch.arange(300)[-128:])
            self.assertTrue(
                torch.equal(
                    run.pool.get_swa_pages(layer).view(-1, 512)[rows_],
                    full.pool.get_swa_pages(layer).view(-1, 512)[rows_],
                )
            )

    def test_causal_lm_head_sees_the_full_layout(self):
        """The real DeepseekV4ForCausalLM.forward (model + compact-tail branch)
        through: a replay forward with a tail, bucket 0 (the dec0 warmup shape,
        decoder skipped), and an extend without replay. The head gets full-layout
        rows and the batch's ids each time, and the identity does not outlive
        the forward."""
        from sglang_tpu.envs import envs as tpu_envs

        cases = (
            ("replay", CHUNK1, 512),
            ("replay", [_chunk(1, 0, 300)], 0),
            ("full", CHUNK1, None),
        )
        for mode, chunk, bucket in cases:
            with (
                self.subTest(mode=mode, bucket=bucket),
                tpu_envs.SGLANG_TPU_DSV41_DECODER_REPLAY_ROWS.override("prompt"),
            ):
                run = _Run(self, mode=mode, window=self.window)
                seen = run.forward(chunk, bucket=bucket, head=True)
                rows = sum(n for _, _, n, _, _ in chunk) + 3
                self.assertEqual(seen["input_ids"].shape[0], rows)
                self.assertEqual(tuple(seen["hidden"].shape), (rows, H))
                self.assertEqual(len(seen["aux"]), len(AUX_LAYERS))
                for t in seen["aux"]:
                    self.assertEqual(t.shape[0], rows)
                self.assertIsNone(seen["head_full_layout_after"])

    def test_decoder_rows_take_the_extend_tiling(self):
        """Ratio-1 decoder calls on the tail rows ask for the extend CSA2 tiling;
        full-row calls keep the default."""
        RefCsa2.tilings.clear()
        run = _Run(self, mode="replay", window=self.window)
        _, _, _, _, positions = run.forward(CHUNK1, bucket=512)
        # The rows came back in the full layout: the head reads the identity.
        self.assertTrue(run.head_full_layout)
        full_rows = positions.shape[0] + 3  # token-bucket padding
        by_rows = {}
        for rows, ext, main in RefCsa2.tilings:
            by_rows.setdefault((rows, main), set()).add(ext)
        self.assertEqual(by_rows.get((512, True)), {True})
        self.assertEqual(
            {
                ext
                for (rows, _), exts in by_rows.items()
                if rows == full_rows
                for ext in exts
            },
            {False},
        )

    def test_last_rows_of_chunks_without_decoder_rows_are_defined(self):
        """Two requests mid-prompt (prompt rule) keep no decoder rows; the
        logits processor still reads each chunk's last row (rows 299 and 499),
        which must be defined."""
        from sglang_tpu.envs import envs as tpu_envs

        run = _Run(self, mode="replay", window=self.window)
        with tpu_envs.SGLANG_TPU_DSV41_DECODER_REPLAY_ROWS.override("prompt"):
            hidden, pre_hc, *_ = run.forward(
                [_chunk(1, 0, 300), _chunk(3, 0, 200)], bucket=128
            )
        # A padding-only tail (no real row) brings back zeros.
        self.assertTrue(bool(torch.isfinite(hidden).all()))
        for t in (hidden, pre_hc):
            self.assertEqual(int(t[[299, 499]].abs().sum()), 0)

    def _swa_rows(self, slot, pos):
        """SWA buffer rows of (slot, positions): the request ring under the
        request window (its last window), else the paged rows at the full loc."""
        if self.window:
            from sglang_tpu.models.deepseek_v41.request_window import ring_rows

            return ring_rows(torch.full_like(pos, slot), pos)
        return _loc(slot, pos)


class TestDecoderReplayModelRequestWindow(TestDecoderReplayModel):
    """The same under encoder SWA replay: SWA lives in each request's window
    ring, and a chunk copies its visible history in and commits its last window."""

    window = True


class TestDecoderReplayModelPromptRule(TestDecoderReplayModel):
    """The prompt-tail rule (each prompt's last window; vLLM-TorchTPU CED)."""

    rule = "prompt"


class TestDecoderReplayModelPromptRuleRequestWindow(TestDecoderReplayModel):
    window = True
    rule = "prompt"


if __name__ == "__main__":
    unittest.main()
