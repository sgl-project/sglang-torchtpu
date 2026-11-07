# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1 text model in plain torch, ported from the DeepSeek-V4.1-Flash reference model.py.

Module tree, parameter names and math follow the reference (see sglang_tpu/dsv41_native). What
changes is how state crosses forwards. The reference keeps one cache row per batch index and a
scalar `start_pos`; here every token carries its own position and its request's ReqToTokenPool
slot, so one graph serves any mix of prefill chunks and decode steps:

- The model owns its caches, one row per request slot (row 0 is the pool's padding row) and one
  column per position: raw window KV, compressor inputs, compressed KV, index keys and engram
  token ids. Each forward writes its tokens first, then gathers what each token may attend to.
- The sliding window is positions pos-127..pos read from the raw KV cache instead of a ring.
- A compressed group is pooled by the token that completes it, from the cached compressor inputs;
  a query sees groups below (pos + 1) // ratio, which is what the reference prefill mask and its
  decode slice both give.
- Routed experts run as the reference's static grouped MoE with capacity = tokens, so no token
  is dropped. TP over heads and vocab and EP over experts use the TP group (tp == ep).
Vision, the DSpark draft head (mtp.*) and the VL routing bias are not built.
"""

from __future__ import annotations

import json
import math
import os
import re
import struct
import time
from functools import lru_cache
from typing import Iterable, NamedTuple, Optional, Tuple

import msgspec
import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from torch import nn

from sglang.srt.distributed import tensor_model_parallel_all_reduce
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.runtime_context import (
    get_model,
    get_parallel,
    get_schedule,
    get_serving,
    process_model_config,
)
from sglang.srt.utils.hf_transformers.common import get_rope_config
from sglang.srt.utils.hf_transformers.tokenizer import get_tokenizer
from sglang_tpu._logging import get_logger
from sglang_tpu.dsv41_native.engram_hash32 import (
    MAX_TOKEN,
    engram_hash32,
    limb_weights,
    multiplier_limbs,
)
from sglang_tpu.dsv41_native.engram_ref import (
    EngramLayout,
    build_compressed_token_map,
    compute_hash_multipliers,
)
from sglang_tpu.dsv41_native.kernel_ref import (
    act_quant,
    fp4_act_quant,
    fp8_gemm,
    grouped_gemm,
    hc_split_sinkhorn,
    scale_f32,
    sparse_attn_gathered,
)
from sglang_tpu.ep_local_expert_reader import _SAFETENSORS_DTYPES

logger = get_logger(__name__)

FP8_BLOCK = 32  # one fp8 scale per 32x32 weight block / 32 activations
FP4_BLOCK = 32  # one fp4 scale per 32 elements along K
SCALE_FMT = "ue8m0"
# E8M0 scales are kept as raw bytes; kernel_ref.scale_f32 decodes them.
SCALE_STORAGE = torch.uint8
# Bound on one readinto() while reading an engram row range (not tuned).
_READ_CHUNK_BYTES = 64 * 1024 * 1024
_EXPERT_RE = re.compile(
    r"^layers\.(\d+)\.ffn\.experts\.(\d+)\.(w[123])\.(weight|scale)$"
)
# convert.py: checkpoint key -> dim sharded over TP ranks.
_SHARD_DIM = {
    "embed": 0,
    "wq_b": 0,
    "wo_a": 0,
    "wo_b": 1,
    "head": 0,
    "attn_sink": 0,
    "weights_proj": 0,
}
_SKIP_PREFIXES = ("vision.", "aligner.", "mtp.", "image_")


class _Args(msgspec.Struct, frozen=True):
    """The reference ModelArgs fields this model reads, plus serving limits and TP placement."""

    max_slots: int  # ReqToTokenPool rows: max_running_requests + the padding row 0
    max_seq_len: int
    tp_size: int
    tp_rank: int
    vocab_size: int
    dim: int
    moe_inter_dim: int
    n_layers: int
    n_heads: int
    n_routed_experts: int
    n_activated_experts: int
    score_func: str
    route_scale: float
    norm_topk_prob: bool
    swiglu_limit: float
    q_lora_rank: int
    head_dim: int
    rope_head_dim: int
    norm_eps: float
    o_groups: int
    o_lora_rank: int
    window_size: int
    compress_ratios: tuple[int, ...]
    kv_source_layers: tuple[int, ...]
    index_source_layers: tuple[int, ...]
    compress_rope_theta: float
    original_seq_len: int
    rope_theta: float
    rope_factor: float
    beta_fast: int
    beta_slow: int
    index_n_heads: int
    index_head_dim: int
    index_topk: int
    candidate_source_layer: int
    candidate_topk_blocks: int
    candidate_block_size: int
    hc_mult: int
    hc_sinkhorn_iters: int
    hc_eps: float
    engram_pad_id: int
    engram_compressed_vocab_size: int
    gate_temp: float = 1.0

    @classmethod
    def from_hf_config(cls, config, *, max_slots: int, max_seq_len: int) -> _Args:
        rope_theta, rope = get_rope_config(config)
        return cls(
            max_slots=max_slots,
            max_seq_len=max_seq_len,
            tp_size=get_parallel().tp_size,
            tp_rank=get_parallel().tp_rank,
            vocab_size=config.vocab_size,
            dim=config.hidden_size,
            moe_inter_dim=config.moe_intermediate_size,
            n_layers=config.num_hidden_layers,
            n_heads=config.num_attention_heads,
            n_routed_experts=config.n_routed_experts,
            n_activated_experts=config.num_experts_per_tok,
            score_func=config.scoring_func,
            route_scale=config.routed_scaling_factor,
            norm_topk_prob=config.norm_topk_prob,
            swiglu_limit=config.swiglu_limit,
            q_lora_rank=config.q_lora_rank,
            head_dim=config.head_dim,
            rope_head_dim=config.qk_rope_head_dim,
            norm_eps=config.rms_norm_eps,
            o_groups=config.o_groups,
            o_lora_rank=config.o_lora_rank,
            window_size=config.sliding_window,
            compress_ratios=tuple(config.compress_ratios),
            kv_source_layers=tuple(config.kv_source_layer_ids),
            index_source_layers=tuple(config.index_source_layer_ids),
            compress_rope_theta=config.compress_rope_theta,
            original_seq_len=rope["original_max_position_embeddings"],
            rope_theta=rope_theta,
            rope_factor=rope["factor"],
            beta_fast=rope["beta_fast"],
            beta_slow=rope["beta_slow"],
            index_n_heads=config.index_n_heads,
            index_head_dim=config.index_head_dim,
            index_topk=config.index_topk,
            candidate_source_layer=config.candidate_source_layer_id,
            candidate_topk_blocks=config.candidate_topk_blocks,
            candidate_block_size=config.candidate_block_size,
            hc_mult=config.hc_mult,
            hc_sinkhorn_iters=config.hc_sinkhorn_iters,
            hc_eps=config.hc_eps,
            engram_pad_id=config.engram_pad_token_id,
            engram_compressed_vocab_size=config.engram_compressed_vocab_size,
        )


class _TokenCtx(NamedTuple):
    """Per-token cache addressing, built inside the compiled forward. [T] int64 each."""

    positions: torch.Tensor
    slots: (
        torch.Tensor
    )  # ReqToTokenPool row of the token's request; padding tokens use row 0


def _token_ctx(
    *, positions: torch.Tensor, forward_batch: ForwardBatch
) -> tuple[_TokenCtx, torch.Tensor]:
    """Token addressing plus each request's last-token row ([B]), all int32: positions and slots
    feed modulo/divide index math, and XLA's 64-bit emulation aborted the TPU compiler (r6)."""
    req = forward_batch.req_pool_indices.to(torch.int32)
    positions = positions.to(torch.int32)
    num_tokens = positions.shape[0]
    tok = torch.arange(num_tokens, device=positions.device, dtype=torch.int32)
    if forward_batch.forward_mode.is_decode():
        return _TokenCtx(positions=positions, slots=req), tok
    # Padded requests have extend_seq_lens 0, so tokens past the real ones land on index B.
    # cumsum/sum of int32 or bool promote to int64 in torch unless dtype is given.
    ends = torch.cumsum(forward_batch.extend_seq_lens, dim=0, dtype=torch.int32)
    req_idx = (tok.unsqueeze(1) >= ends.unsqueeze(0)).sum(1, dtype=torch.int32)
    batch = req.shape[0]
    slots = torch.where(req_idx < batch, req[req_idx.clamp_max(batch - 1)], 0)
    return _TokenCtx(positions=positions, slots=slots), (ends - 1).clamp_min(0)


def _param(shape: tuple[int, ...], dtype: torch.dtype) -> nn.Parameter:
    return nn.Parameter(torch.empty(shape, dtype=dtype), requires_grad=False)


def _after(x: torch.Tensor, *tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """(x, *tensors) through an XLA optimization barrier: ops on the returned tensors run after
    x is computed. Weight dequantization does not depend on activations, so without this XLA
    may hoist every layer's dequantized weights to the program start and keep them live at once
    (r5: 192 GB of temporaries against 94.7 GB of HBM)."""
    if x.device.type != "tpu":
        return (x, *tensors)
    return tuple(torch.ops.tpu._optimization_barrier([x, *tensors]))


def linear(
    x: torch.Tensor, weight: torch.Tensor, scale: Optional[torch.Tensor]
) -> torch.Tensor:
    """reference linear(): fp8 weights take an fp8 activation, anything else is a plain matmul."""
    if weight.dtype == torch.float8_e4m3fn:
        q, s = act_quant(x, FP8_BLOCK, SCALE_FMT, torch.float32)
        q, s, weight, scale = _after(q, s, weight, scale)
        return fp8_gemm(q, s, weight, scale, torch.float32, block_size=FP8_BLOCK)
    return F.linear(x, weight)


class Linear(nn.Module):
    """bf16, fp32 or fp8 weights; fp8 ones carry one E8M0 `scale` per 32x32 block."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        dtype: torch.dtype = torch.float8_e4m3fn,
    ):
        super().__init__()
        self.weight = _param((out_features, in_features), dtype)
        if dtype == torch.float8_e4m3fn:
            blocks = (
                math.ceil(out_features / FP8_BLOCK),
                math.ceil(in_features / FP8_BLOCK),
            )
            self.scale = _param(blocks, SCALE_STORAGE)
        else:
            self.register_parameter("scale", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return linear(x, self.weight, self.scale)


class ColumnParallelLinear(Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        tp_size: int,
        dtype=torch.float8_e4m3fn,
    ):
        assert out_features % tp_size == 0
        super().__init__(in_features, out_features // tp_size, dtype)


class RowParallelLinear(Linear):
    """Partial sums per rank, reduced in fp32 as the reference does."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        tp_size: int,
        dtype=torch.float8_e4m3fn,
    ):
        assert in_features % tp_size == 0
        super().__init__(in_features // tp_size, out_features, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = linear(x, self.weight, self.scale)
        return tensor_model_parallel_all_reduce(y.float()).type_as(x)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = _param((dim,), torch.bfloat16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)
        return (self.weight * x).to(dtype)


class ParallelEmbedding(nn.Module):
    """Vocab-sharded embedding; off-rank ids read row 0 and are zeroed before the all_reduce."""

    def __init__(self, args: _Args):
        super().__init__()
        assert args.vocab_size % args.tp_size == 0
        part = args.vocab_size // args.tp_size
        self.start = args.tp_rank * part
        self.end = self.start + part
        self.weight = _param((part, args.dim), torch.bfloat16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.int32)  # input_ids arrive as int64
        mask = (x < self.start) | (x >= self.end)
        y = F.embedding(torch.where(mask, 0, x - self.start), self.weight)
        y = y.masked_fill(mask.unsqueeze(-1), 0)
        return tensor_model_parallel_all_reduce(y)


class ParallelHead(nn.Module):
    """fp32 logits, vocab gathered by summing zero-padded blocks over TP.

    gather-via-all_reduce works around the cross-chip all_gather defect,
    sgl-project/sglang-torchtpu#1700; drop it once that is fixed.
    """

    def __init__(self, args: _Args):
        super().__init__()
        self.tp_size, self.tp_rank = args.tp_size, args.tp_rank
        self.weight = _param((args.vocab_size // args.tp_size, args.dim), torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = F.linear(x.float(), self.weight)  # [B, vocab / tp]
        blocks = logits.new_zeros((self.tp_size, *logits.shape))
        blocks[self.tp_rank] = logits
        blocks = tensor_model_parallel_all_reduce(blocks)
        return blocks.permute(1, 0, 2).reshape(logits.shape[0], -1)


@lru_cache(4)
def _rope_cos_sin(dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow):
    """reference precompute_freqs_cis on CPU, split into its real and imaginary parts."""
    with torch.device("cpu"):
        return _rope_cos_sin_cpu(
            dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow
        )


def _rope_cos_sin_cpu(
    dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow
):
    # Built on CPU: the model is constructed under the TPU default device, and polar is complex.
    freqs = 1.0 / (
        base ** (torch.arange(0, dim, 2, dtype=torch.float32, device="cpu") / dim)
    )
    if original_seq_len > 0:

        def corrected_dim(rotations):
            return (
                dim
                * math.log(original_seq_len / (rotations * 2 * math.pi))
                / (2 * math.log(base))
            )

        low = max(math.floor(corrected_dim(beta_fast)), 0)
        high = min(math.ceil(corrected_dim(beta_slow)), dim - 1)
        ramp = (
            (torch.arange(dim // 2, dtype=torch.float32, device="cpu") - low)
            / max(high - low, 1e-3)
        ).clamp(0, 1)
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    freqs = torch.outer(torch.arange(seqlen, dtype=torch.float32, device="cpu"), freqs)
    cis = torch.polar(torch.ones_like(freqs), freqs)
    return cis.real.contiguous(), cis.imag.contiguous()


def _rope_tail(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rd: int
) -> torch.Tensor:
    """reference apply_rotary_emb on the last `rd` dims, as real arithmetic. x is [1, T, d] or
    [1, T, h, d]; cos/sin are [T, rd // 2] (negate sin for the inverse rotation)."""
    tail = x[..., -rd:].float().unflatten(-1, (-1, 2))
    if x.ndim == 4:
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    a, b = tail[..., 0], tail[..., 1]
    rotated = torch.stack([a * cos - b * sin, a * sin + b * cos], dim=-1).flatten(-2)
    return torch.cat([x[..., :-rd], rotated.to(x.dtype)], dim=-1)


class _EngramHasher(nn.Module):
    """reference NgramHashState with a per-slot token history indexed by position."""

    def __init__(self, args: _Args, layout: EngramLayout):
        super().__init__()
        self.max_ngram_size = layout.max_ngram_size
        # The token map is built with the HF normalizers, whatever the serving tokenizer backend.
        tokenizer = get_tokenizer(
            get_serving().tokenizer_path,
            tokenizer_mode=get_serving().tokenizer_mode,
            trust_remote_code=get_model().trust_remote_code,
            revision=get_model().revision,
            tokenizer_backend="huggingface",
        )
        token_map, vocab_size = build_compressed_token_map(tokenizer)
        assert vocab_size == args.engram_compressed_vocab_size, (
            vocab_size,
            args.engram_compressed_vocab_size,
        )
        self.pad_id = token_map[args.engram_pad_id]
        flat = [
            [p for per_ngram in layer for p in per_ngram] for layer in layout.primes
        ]
        offsets = [np.cumsum([0, *sizes[:-1]]) for sizes in flat]
        multipliers = compute_hash_multipliers(
            layout.layer_ids, layout.max_ngram_size, vocab_size
        )
        # Everything the compiled graph sees is int32: XLA's 64-bit integer emulation aborted the
        # TPU compiler (r6). engram_hash32 keeps the 63-bit hash exact with 7-bit limbs.
        assert vocab_size < MAX_TOKEN, vocab_size
        primes = torch.tensor(layout.primes, dtype=torch.int64)
        offsets = torch.tensor(np.array(offsets), dtype=torch.int64)
        assert int((offsets + primes.flatten(1)).max()) < 2**31, (
            "table offsets need int64"
        )
        device = torch.get_default_device()

        def buf(name, value):
            self.register_buffer(
                name, value.to(torch.int32).to(device), persistent=False
            )

        buf("primes", primes)
        buf("limb_weights", limb_weights(primes))
        buf("m_limbs", multiplier_limbs(multipliers))
        buf("offsets", offsets)
        buf("token_map", torch.tensor(token_map))
        buf("cache", torch.zeros(args.max_slots, args.max_seq_len))

    def forward(self, input_ids: torch.Tensor, ctx: _TokenCtx) -> torch.Tensor:
        """[T] ids -> [1, T, n_engram_layers, n_hash_cols] hash ids (int32)."""
        pos = ctx.positions
        self.cache[ctx.slots, pos] = self.token_map[input_ids]
        tokens, blocked = [], torch.zeros_like(pos, dtype=torch.bool)
        for shift in range(self.max_ngram_size):
            source = self.cache[ctx.slots, (pos - shift).clamp_min(0)]
            blocked = blocked | (pos < shift)
            tokens.append(torch.where(blocked, self.pad_id, source))
        tokens = torch.stack(tokens, dim=-1).unsqueeze(0)  # [1, T, max_ngram_size]
        hashes = engram_hash32(tokens, self.m_limbs, self.primes, self.limb_weights)
        return hashes + self.offsets


class ParallelEngramEmbedding(nn.Module):
    """Row-sharded fp8 n-gram table, dequantized on lookup."""

    def __init__(self, num_embeddings: int, dim: int, args: _Args):
        super().__init__()
        self.part = (num_embeddings + args.tp_size - 1) // args.tp_size
        self.start = args.tp_rank * self.part
        self.weight = _param((self.part, dim), torch.float8_e4m3fn)
        self.scale = _param((self.part, dim // FP8_BLOCK), SCALE_STORAGE)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        mask = (indices < self.start) | (indices >= self.start + self.part)
        local = (indices - self.start).masked_fill(mask, 0)
        values = F.embedding(local, self.weight).float().unflatten(-1, (-1, FP8_BLOCK))
        scales = scale_f32(F.embedding(local, self.scale))
        values = (values * scales.unsqueeze(-1)).flatten(-2).to(torch.bfloat16)
        return tensor_model_parallel_all_reduce(
            values.masked_fill(mask.unsqueeze(-1), 0)
        )


class Engram(nn.Module):
    """reference Engram: an n-gram lookup gated into each hc copy of the residual stream."""

    def __init__(self, args: _Args, layer_id: int, layout: EngramLayout):
        super().__init__()
        self.layer_hash_index = layout.layer_ids.index(layer_id)
        self.dim, self.hc_mult, self.eps = args.dim, args.hc_mult, args.norm_eps
        self.clamp_value = 1e-6
        self.embed = ParallelEngramEmbedding(
            layout.num_embeddings[self.layer_hash_index], layout.head_dim, args
        )
        n_hash_cols = (layout.max_ngram_size - 1) * layout.n_heads
        self.wkv = Linear(n_hash_cols * layout.head_dim, args.dim * (args.hc_mult + 1))
        self.q_weight = _param((args.hc_mult, args.dim), torch.bfloat16)
        self.k_weight = _param((args.hc_mult, args.dim), torch.bfloat16)

    def forward(self, x: torch.Tensor, hash_ids: torch.Tensor) -> torch.Tensor:
        kv = self.wkv(self.embed(hash_ids).flatten(-2))
        key, value = kv.split([self.hc_mult * self.dim, self.dim], dim=-1)
        key = key.float().unflatten(-1, (self.hc_mult, self.dim))
        weight = self.q_weight.float() * self.k_weight.float()
        h, eps = x.float(), self.eps
        rstd = torch.rsqrt(h.square().mean(-1) + eps) * torch.rsqrt(
            key.square().mean(-1) + eps
        )
        dot = (h * weight * key).sum(-1) * rstd * self.dim**-0.5
        mag = dot.abs().clamp_min(self.clamp_value).sqrt()
        gate = torch.sigmoid(torch.where(dot < 0, -mag, mag))
        return (h + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(x.dtype)


class Compressor(nn.Module):
    """Pools `ratio` consecutive tokens into one pre-RoPE latent with a softmax gate. The inputs of
    every position are cached, so the token that completes a group pools it from the cache."""

    def __init__(self, args: _Args, layer_id: int):
        super().__init__()
        self.ratio = args.compress_ratios[layer_id]
        self.norm = RMSNorm(args.head_dim, args.norm_eps)
        # ratio 1 is a plain projection in bf16; pooling above ratio 1 runs in fp32
        dtype = torch.float32 if self.ratio > 1 else torch.bfloat16
        self.wkv = Linear(args.dim, args.head_dim, dtype=dtype)
        if self.ratio > 1:
            self.wgate = Linear(args.dim, args.head_dim, dtype=torch.float32)
            shape = (args.max_slots, args.max_seq_len, args.head_dim)
            self.register_buffer(
                "kv_state", torch.zeros(shape, dtype=torch.float32), persistent=False
            )
            self.register_buffer(
                "score_state", torch.zeros(shape, dtype=torch.float32), persistent=False
            )

    def forward(self, x: torch.Tensor, ctx: _TokenCtx) -> torch.Tensor:
        """x [1, T, dim] -> [1, T, head_dim]: the latent of the group ending at each token."""
        if self.ratio == 1:
            return self.norm(self.wkv(x))
        dtype, xf = x.dtype, x.float()
        self.kv_state[ctx.slots, ctx.positions] = self.wkv(xf)[0]
        self.score_state[ctx.slots, ctx.positions] = self.wgate(xf)[0]
        group = (
            ctx.positions.unsqueeze(1)
            - (self.ratio - 1)
            + torch.arange(self.ratio, device=x.device, dtype=torch.int32)
        )
        rows, cols = ctx.slots.unsqueeze(1), group.clamp_min(0)
        kv, score = (
            self.kv_state[rows, cols],
            self.score_state[rows, cols],
        )  # [T, ratio, d]
        pooled = (kv * score.softmax(dim=1)).sum(dim=1)
        return self.norm(pooled.to(dtype)).unsqueeze(0)


def select_candidate_blocks(
    logits, compress_lens, topk_blocks: int, block_size: int
) -> torch.Tensor:
    """reference level one of the two-level top-k, unchanged."""
    width = logits.size(-1)
    scores = F.pad(logits, (0, -width % block_size), value=-torch.inf)
    scores = scores.unflatten(-1, (-1, block_size)).amax(dim=-1)
    num_blocks = scores.size(-1)
    last = (compress_lens - 1) // block_size
    scores = scores.masked_fill(
        torch.arange(num_blocks, device=logits.device, dtype=torch.int32) == last,
        torch.inf,
    )
    top = scores.topk(min(topk_blocks, num_blocks), dim=-1)
    keep = torch.zeros_like(scores, dtype=torch.bool).scatter_(
        -1, top.indices, top.values > -torch.inf
    )
    return keep.repeat_interleave(block_size, dim=-1)[..., :width]


class Indexer(nn.Module):
    """Keeps the `index_topk` best compressed positions per query (reference Indexer)."""

    def __init__(self, args: _Args, layer_id: int):
        super().__init__()
        self.owns_k = layer_id in args.kv_source_layers
        self.ratio = args.compress_ratios[layer_id]
        self.is_candidate_source = layer_id == args.candidate_source_layer
        self.uses_candidates = 0 <= args.candidate_source_layer < layer_id
        self.candidate_topk_blocks = args.candidate_topk_blocks
        self.candidate_block_size = args.candidate_block_size
        self.n_heads = args.index_n_heads
        self.n_local_heads = args.index_n_heads // args.tp_size
        self.head_dim = args.index_head_dim
        self.rope_head_dim = args.rope_head_dim
        self.num_compressed = args.max_seq_len // self.ratio
        self.topk = min(args.index_topk, self.num_compressed)
        self.softmax_scale = self.head_dim**-0.5
        self.wq_b = ColumnParallelLinear(
            args.q_lora_rank, self.n_heads * self.head_dim, tp_size=args.tp_size
        )
        self.weights_proj = ColumnParallelLinear(
            args.dim, self.n_heads, tp_size=args.tp_size, dtype=torch.bfloat16
        )
        if self.owns_k:
            self.wk = Linear(args.head_dim, self.head_dim, dtype=torch.bfloat16)
            self.k_norm = RMSNorm(self.head_dim, args.norm_eps)
            shape = (args.max_slots, self.num_compressed, self.head_dim)
            self.register_buffer(
                "k_cache", torch.zeros(shape, dtype=torch.bfloat16), persistent=False
            )

    def _write_index_k(self, latent, rope_c, write_slots, group, shared) -> None:
        """Index keys of the groups completed in this forward; latent is pre-RoPE."""
        k = self.k_norm(self.wk(latent))
        k = _rope_tail(k, *rope_c, self.rope_head_dim)
        k = fp4_act_quant(k, FP4_BLOCK, True)
        self.k_cache[write_slots, group] = k[0]
        shared["index_k"] = self.k_cache

    def forward(self, x, qr, rope, ctx: _TokenCtx, shared: dict) -> torch.Tensor:
        """[T, topk] compressed positions per query, -1 where unreachable."""
        q = self.wq_b(qr).unflatten(-1, (self.n_local_heads, self.head_dim))
        q = fp4_act_quant(_rope_tail(q, *rope, self.rope_head_dim), FP4_BLOCK, True)
        index_k = shared["index_k"][ctx.slots]  # [T, num_compressed, d]
        weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads**-0.5)
        score = torch.einsum("thd,tcd->thc", q[0], index_k)
        score = tensor_model_parallel_all_reduce(
            (score.relu() * weights[0].unsqueeze(-1)).sum(dim=1)
        )
        # a group becomes visible once the query has passed its last token
        compress_lens = ((ctx.positions + 1) // self.ratio).unsqueeze(-1)
        positions = torch.arange(
            self.num_compressed, device=x.device, dtype=torch.int32
        )
        score = score.masked_fill(positions >= compress_lens, -torch.inf)
        if self.is_candidate_source:
            shared["candidates"] = select_candidate_blocks(
                score,
                compress_lens,
                self.candidate_topk_blocks,
                self.candidate_block_size,
            )
        elif self.uses_candidates:
            score = score.masked_fill(~shared["candidates"], -torch.inf)
        idxs = score.topk(self.topk, dim=-1, sorted=False).indices.sort(dim=-1).values
        idxs = idxs.to(torch.int32)
        return torch.where(idxs < compress_lens, idxs, -1)


class Attention(nn.Module):
    """reference Attention: a sliding window of raw KV plus, when compress_ratio > 0, top-k
    compressed positions, in one sparse attention; low-rank Q and grouped low-rank output."""

    def __init__(self, layer_id: int, args: _Args):
        super().__init__()
        tp = args.tp_size
        self.n_local_heads = args.n_heads // tp
        self.n_local_groups = args.o_groups // tp
        self.o_lora_rank = args.o_lora_rank
        self.head_dim = args.head_dim
        self.rope_head_dim = args.rope_head_dim
        self.window_size = args.window_size
        self.ratio = args.compress_ratios[layer_id]
        self.softmax_scale = self.head_dim**-0.5
        self.attn_sink = _param((self.n_local_heads,), torch.float32)
        self.wq_a = Linear(args.dim, args.q_lora_rank)
        self.q_norm = RMSNorm(args.q_lora_rank, args.norm_eps)
        self.wq_b = ColumnParallelLinear(
            args.q_lora_rank, args.n_heads * self.head_dim, tp_size=tp
        )
        self.wkv = Linear(args.dim, self.head_dim)
        self.kv_norm = RMSNorm(self.head_dim, args.norm_eps)
        # convert.py dequantizes wo_a to bf16
        self.wo_a = ColumnParallelLinear(
            args.n_heads * self.head_dim // args.o_groups,
            args.o_groups * args.o_lora_rank,
            tp_size=tp,
            dtype=torch.bfloat16,
        )
        self.wo_b = RowParallelLinear(
            args.o_groups * args.o_lora_rank, args.dim, tp_size=tp
        )
        self.is_kv_source = layer_id in args.kv_source_layers
        self.is_index_source = layer_id in args.index_source_layers
        self.compressor = Compressor(args, layer_id) if self.is_kv_source else None
        self.indexer = Indexer(args, layer_id) if self.is_index_source else None
        shape = (args.max_slots, args.max_seq_len, self.head_dim)
        self.register_buffer(
            "kv_cache", torch.zeros(shape, dtype=torch.bfloat16), persistent=False
        )
        if self.is_kv_source:
            shape = (args.max_slots, args.max_seq_len // self.ratio, self.head_dim)
            self.register_buffer(
                "compress_kv_cache",
                torch.zeros(shape, dtype=torch.bfloat16),
                persistent=False,
            )
        if self.ratio:
            original_seq_len, theta = args.original_seq_len, args.compress_rope_theta
        else:  # pure sliding-window layers use the base theta without YaRN
            original_seq_len, theta = 0, args.rope_theta
        cos, sin = _rope_cos_sin(
            self.rope_head_dim,
            args.max_seq_len,
            original_seq_len,
            theta,
            args.rope_factor,
            args.beta_fast,
            args.beta_slow,
        )
        device = torch.get_default_device()
        self.register_buffer("rope_cos", cos.to(device), persistent=False)
        self.register_buffer("rope_sin", sin.to(device), persistent=False)

    def _window_kv(self, x, rope, ctx: _TokenCtx):
        """[T, window, d] raw KV at positions pos-window+1..pos, and which of them exist. The KV
        stays fp8-rounded over the whole post-RoPE vector, as the reference stores it."""
        kv = _rope_tail(self.kv_norm(self.wkv(x)), *rope, self.rope_head_dim)
        kv = act_quant(kv, FP8_BLOCK, SCALE_FMT, torch.float32, True)
        self.kv_cache[ctx.slots, ctx.positions] = kv[0]
        win = self.window_size
        idx = (
            ctx.positions.unsqueeze(1)
            - (win - 1)
            + torch.arange(win, device=x.device, dtype=torch.int32)
        )
        return self.kv_cache[ctx.slots.unsqueeze(1), idx.clamp_min(0)], idx >= 0

    def _write_compressed(self, x, qr, rope, ctx: _TokenCtx, shared: dict):
        """KV-source step: pool completed groups, hand the pre-RoPE latent to the indexer, then
        store the RoPE'd, fp4-rounded latent."""
        latent = self.compressor(x, ctx)
        done = (ctx.positions + 1) % self.ratio == 0
        group = ((ctx.positions + 1) // self.ratio - 1).clamp_min(0)
        # tokens that complete nothing write the padding row
        write_slots = torch.where(done, ctx.slots, 0)
        group_pos = (
            group * self.ratio
        )  # a latent stands for the first token of its group
        rope_c = (self.rope_cos[group_pos], self.rope_sin[group_pos])
        if self.indexer is not None and self.indexer.owns_k:
            self.indexer._write_index_k(latent, rope_c, write_slots, group, shared)
        latent = _rope_tail(latent, *rope_c, self.rope_head_dim)
        # compressed KV uses groups of 16 with E4M3 scales; the indexer uses 32 with E8M0
        latent = fp4_act_quant(latent, 16, True, scale_dtype=torch.float8_e4m3fn)
        self.compress_kv_cache[write_slots, group] = latent[0]
        shared["compress_kv"] = self.compress_kv_cache

    def _compress_kv(self, x, qr, rope, ctx: _TokenCtx, shared: dict):
        """[T, topk, d] compressed KV each query attends to, and which entries are real."""
        if self.is_kv_source:
            self._write_compressed(x, qr, rope, ctx, shared)
        if self.is_index_source:
            shared["topk_idxs"] = self.indexer(x, qr, rope, ctx, shared)
        idxs = shared["topk_idxs"]
        cache = shared["compress_kv"]
        return cache[ctx.slots.unsqueeze(1), idxs.clamp_min(0)], idxs >= 0

    def forward(self, x: torch.Tensor, ctx: _TokenCtx, shared: dict) -> torch.Tensor:
        rd = self.rope_head_dim
        rope = (self.rope_cos[ctx.positions], self.rope_sin[ctx.positions])
        qr = self.q_norm(self.wq_a(x))
        q = self.wq_b(qr).unflatten(-1, (self.n_local_heads, self.head_dim))
        q = _rope_tail(q, *rope, rd)
        kv, valid = self._window_kv(x, rope, ctx)
        if self.ratio:
            compress_kv, compress_valid = self._compress_kv(x, qr, rope, ctx, shared)
            kv = torch.cat([kv, compress_kv], dim=1)
            valid = torch.cat([valid, compress_valid], dim=1)
        o = sparse_attn_gathered(
            q, kv.unsqueeze(0), valid.unsqueeze(0), self.attn_sink, self.softmax_scale
        )
        o = _rope_tail(o, rope[0], -rope[1], rd)
        # wo_a is block-diagonal over groups, hence einsum; o is fp8-rounded first
        o = o.view(1, o.size(1), self.n_local_groups, -1)
        o = act_quant(o, FP8_BLOCK, SCALE_FMT, torch.float32, True)
        wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
        o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
        return self.wo_b(o.flatten(2))


class Gate(nn.Module):
    """MoE gate: the correction bias picks experts, the routing weights come from raw scores."""

    def __init__(self, args: _Args):
        super().__init__()
        self.topk = args.n_activated_experts
        self.score_func = args.score_func
        self.gate_temp = args.gate_temp
        self.norm_topk_prob = args.norm_topk_prob
        self.route_scale = args.route_scale
        self.weight = _param((args.n_routed_experts, args.dim), torch.bfloat16)
        self.bias = _param((args.n_routed_experts,), torch.float32)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = F.linear(x.float(), self.weight.float()) / self.gate_temp
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1)
        elif self.score_func == "sigmoid":
            scores = scores.sigmoid()
        else:
            scores = F.softplus(scores).sqrt()
        indices = (scores + self.bias).topk(self.topk, dim=-1)[1].to(torch.int32)
        weights = scores.gather(1, indices)
        if self.norm_topk_prob and self.topk > 1:
            weights = weights / (
                weights.sum(dim=-1, keepdim=True) + 1e-20
            )  # matches training
        return weights * self.route_scale, indices


class Expert(nn.Module):
    """SwiGLU FFN with the training clamps (up on both sides, gate from above)."""

    def __init__(self, dim: int, inter_dim: int, swiglu_limit: float):
        super().__init__()
        self.w1 = Linear(dim, inter_dim)
        self.w2 = Linear(inter_dim, dim)
        self.w3 = Linear(dim, inter_dim)
        self.swiglu_limit = swiglu_limit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.w1(x).float(), self.w3(x).float()
        if self.swiglu_limit > 0:
            up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
            gate = torch.clamp(gate, max=self.swiglu_limit)
        return self.w2((F.silu(gate) * up).to(x.dtype))


class MoE(nn.Module):
    """Routed experts split over ranks (stacked fp4, one grouped GEMM) plus one shared expert."""

    def __init__(self, args: _Args):
        super().__init__()
        assert args.n_routed_experts % args.tp_size == 0
        self.dim = args.dim
        self.n_local_experts = args.n_routed_experts // args.tp_size
        self.experts_start_idx = args.tp_rank * self.n_local_experts
        self.swiglu_limit = args.swiglu_limit
        self.gate = Gate(args)
        E, dim, inter = self.n_local_experts, args.dim, args.moe_inter_dim
        # fp4 pairs packed in bytes: [E, out, in // 2]; E8M0 scale per 32 along in
        self.experts_w1 = _param((E, inter, dim // 2), torch.uint8)
        self.experts_w1_scale = _param((E, inter, dim // FP4_BLOCK), SCALE_STORAGE)
        self.experts_w3 = _param((E, inter, dim // 2), torch.uint8)
        self.experts_w3_scale = _param((E, inter, dim // FP4_BLOCK), SCALE_STORAGE)
        self.experts_w2 = _param((E, dim, inter // 2), torch.uint8)
        self.experts_w2_scale = _param((E, dim, inter // FP4_BLOCK), SCALE_STORAGE)
        self.shared_experts = Expert(args.dim, args.moe_inter_dim, args.swiglu_limit)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.size()
        x = x.view(-1, self.dim)
        weights, indices = self.gate(x)
        y = tensor_model_parallel_all_reduce(self._forward_static(x, weights, indices))
        y = y + self.shared_experts(x)
        return y.type_as(x).view(shape)

    def _forward_static(self, x, weights, indices) -> torch.Tensor:
        """reference MoE._forward_static with capacity = tokens: each local expert takes every
        token routed to it, padded with weight-0 rows that contribute exactly 0."""
        n_tokens = x.size(0)
        experts = torch.arange(
            self.experts_start_idx,
            self.experts_start_idx + self.n_local_experts,
            device=x.device,
            dtype=torch.int32,
        )
        hit = indices.unsqueeze(-1) == experts  # [n, topk, E]
        routed = hit.any(1)
        token_weight = (weights.unsqueeze(-1) * hit).sum(1)
        priority = torch.arange(
            n_tokens, 0, -1, device=x.device, dtype=torch.int32
        ).unsqueeze(-1)
        idx = (
            torch.where(routed, priority, 0)
            .topk(n_tokens, dim=0, sorted=False)
            .indices.to(torch.int32)
        )  # [C, E]
        w = torch.where(routed.gather(0, idx), token_weight.gather(0, idx), 0.0).t()
        idx = idx.t()
        xs = x[idx]  # [E, C, dim]
        # Dequantize this layer's expert stacks only once its tokens are ready (see _after);
        # w2 waits for the activations, so at most two stacks are live.
        xs, w1, w1_s, w3, w3_s = _after(
            xs,
            self.experts_w1,
            self.experts_w1_scale,
            self.experts_w3,
            self.experts_w3_scale,
        )
        gate = grouped_gemm(xs, w1, w1_s, FP8_BLOCK).float()
        up = grouped_gemm(xs, w3, w3_s, FP8_BLOCK).float()
        if self.swiglu_limit > 0:
            up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
            gate = torch.clamp(gate, max=self.swiglu_limit)
        h = (w.unsqueeze(-1) * (F.silu(gate) * up)).to(x.dtype)
        h, w2, w2_s = _after(h, self.experts_w2, self.experts_w2_scale)
        out = grouped_gemm(h, w2, w2_s, FP8_BLOCK)
        y = torch.zeros_like(x, dtype=torch.float32)
        return y.index_add(0, idx.flatten(), out.flatten(0, 1).float())


class Block(nn.Module):
    """reference Block: the residual stream is hc_mult copies (Hyper-Connections). The mix one
    sublayer computes is used by the next one."""

    def __init__(
        self, layer_id: int, args: _Args, engram_layout: Optional[EngramLayout]
    ):
        super().__init__()
        self.norm_eps = args.norm_eps
        self.attn = Attention(layer_id, args)
        self.ffn = MoE(args)
        self.engram = None
        if engram_layout is not None and layer_id in engram_layout.layer_ids:
            self.engram = Engram(args, layer_id, engram_layout)
        self.attn_norm = RMSNorm(args.dim, args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, args.norm_eps)
        self.hc_mult = args.hc_mult
        self.hc_sinkhorn_iters = args.hc_sinkhorn_iters
        self.hc_eps = args.hc_eps
        mix_hc, hc_dim = (2 + args.hc_mult) * args.hc_mult, args.hc_mult * args.dim
        self.hc_attn_fn = _param((mix_hc, hc_dim), torch.float32)
        self.hc_ffn_fn = _param((mix_hc, hc_dim), torch.float32)
        self.hc_attn_base = _param((mix_hc,), torch.float32)
        self.hc_ffn_base = _param((mix_hc,), torch.float32)
        self.hc_attn_scale = _param((3,), torch.float32)
        self.hc_ffn_scale = _param((3,), torch.float32)

    def hc_mixes(self, x, hc_fn, hc_scale, hc_base):
        x = x.flatten(2).float()
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(x, hc_fn) * rsqrt
        return hc_split_sinkhorn(
            mixes, hc_scale, hc_base, self.hc_mult, self.hc_sinkhorn_iters, self.hc_eps
        )

    def hc_pre(self, x, pre_mix):
        return torch.sum(pre_mix.unsqueeze(-1) * x.float(), dim=2).to(x.dtype)

    def hc_post(self, x, residual, post, comb):
        y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(
            comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2
        )
        return y.type_as(x)

    def forward(self, x, pre_mix, ctx: _TokenCtx, shared: dict):
        residual = x
        attn_pre, attn_post, attn_comb = self.hc_mixes(
            x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base
        )
        x = self.attn(self.attn_norm(self.hc_pre(x, pre_mix)), ctx, shared)
        x = self.hc_post(x, residual, attn_post, attn_comb)
        residual = x
        ffn_pre, ffn_post, ffn_comb = self.hc_mixes(
            x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base
        )
        x = self.ffn(self.ffn_norm(self.hc_pre(x, attn_pre)))
        return self.hc_post(x, residual, ffn_post, ffn_comb), ffn_pre


class DeepseekV41NativeForCausalLM(nn.Module):
    """embed -> hc_mult copies -> blocks -> collapse -> norm -> logits of each request's last token."""

    def __init__(self, config, quant_config=None, prefix: str = ""):
        super().__init__()
        max_running_requests = get_schedule().max_running_requests
        assert max_running_requests is not None, (
            "set --max-running-requests: it sizes the model caches"
        )
        args = _Args.from_hf_config(
            config,
            max_slots=max_running_requests + 1,
            max_seq_len=process_model_config().context_len,
        )
        self.config = config
        self.args = args
        self.hc_mult = args.hc_mult
        engram_layout = EngramLayout.from_args(config)
        self.engram_hash = _EngramHasher(args, engram_layout)
        self.embed = ParallelEmbedding(args)
        self.layers = nn.ModuleList(
            Block(i, args, engram_layout) for i in range(args.n_layers)
        )
        self.norm = RMSNorm(args.dim, args.norm_eps)
        self.head = ParallelHead(args)
        logger.info(
            "DeepSeek-V4.1 native: %d layers, tp=%d rank=%d, %d local experts, %d cache slots x %d positions",
            args.n_layers,
            args.tp_size,
            args.tp_rank,
            self.layers[0].ffn.n_local_experts,
            args.max_slots,
            args.max_seq_len,
        )

    def get_input_embeddings(self) -> nn.Module:
        return self.embed

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        pp_proxy_tensors=None,
        **kwargs,
    ) -> LogitsProcessorOutput:
        assert input_embeds is None and pp_proxy_tensors is None
        input_ids = input_ids.to(torch.int32)  # the scheduler passes int64 ids
        ctx, last = _token_ctx(positions=positions, forward_batch=forward_batch)
        hashes = self.engram_hash(input_ids, ctx)
        h = self.embed(input_ids).unsqueeze(0)  # [1, T, dim]
        h = h.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)
        pre_mix = torch.zeros(*h.shape[:3], dtype=torch.float32, device=h.device)
        pre_mix[..., 0] = 1.0
        shared: dict = {}
        for layer in self.layers:
            if layer.engram is not None:
                h = layer.engram(h, hashes[:, :, layer.engram.layer_hash_index])
            h, pre_mix = layer(h, pre_mix, ctx, shared)
        h = self.norm(self.layers[-1].hc_pre(h, pre_mix))
        return LogitsProcessorOutput(next_token_logits=self.head(h[0, last]))

    # ---- weight loading (convert.py rules, applied per rank) ----

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> None:
        # The loader's iterator reads every tensor of every shard on every rank (8 x 510 GB over
        # gcsfuse); read the shards directly and fetch only what this rank keeps.
        del weights
        folder = process_model_config().model_path
        with open(os.path.join(folder, "model.safetensors.index.json")) as f:
            weight_map = json.load(f)["weight_map"]
        by_file: dict[str, list[str]] = {}
        for name, file in weight_map.items():
            if self._is_local(name):
                by_file.setdefault(file, []).append(name)
        params = dict(self.named_parameters())
        loaded: set[str] = set()
        pending_wo_a: dict[str, dict[str, torch.Tensor]] = {}
        expert_stacks: dict[str, tuple[torch.Tensor, list[int]]] = {}
        for i, file in enumerate(sorted(by_file)):
            with safe_open(
                os.path.join(folder, file), framework="pt", device="cpu"
            ) as f:
                for name in by_file[file]:
                    self._load_one(
                        name=name,
                        path=os.path.join(folder, file),
                        reader=f,
                        params=params,
                        pending_wo_a=pending_wo_a,
                        expert_stacks=expert_stacks,
                        loaded=loaded,
                    )
            # Drain queued uploads so the host staging of one shard is freed before the next.
            torch.accelerator.synchronize()
            logger.info(
                "DeepSeek-V4.1 native: loaded shard %d/%d (%s)",
                i + 1,
                len(by_file),
                file,
            )
        assert not pending_wo_a and not expert_stacks, (
            list(pending_wo_a),
            list(expert_stacks),
        )
        missing = sorted(set(params) - loaded)
        if missing:
            raise RuntimeError(
                f"DeepSeek-V4.1 checkpoint is missing {len(missing)} tensors, e.g. {missing[:5]}"
            )

    def _is_local(self, name: str) -> bool:
        """Whether this rank keeps (part of) checkpoint tensor `name`."""
        if name.startswith(_SKIP_PREFIXES) or name.endswith(".bias_vl"):
            return False
        match = _EXPERT_RE.match(name)
        if match is None:
            return True
        ffn = self.layers[int(match[1])].ffn
        return 0 <= int(match[2]) - ffn.experts_start_idx < ffn.n_local_experts

    def _load_one(
        self, *, name, path, reader, params, pending_wo_a, expert_stacks, loaded
    ) -> None:
        match = _EXPERT_RE.match(name)
        if ".engram.embed." in name:
            # convert.py pads scales with 1.0, which is E8M0 byte 127
            pad_value = 127 if name.endswith(".scale") else 0
            self._load_engram_rows(
                param=params[name], path=path, name=name, pad_byte=pad_value
            )
            loaded.add(name)
            return
        tensor = reader.get_tensor(name)
        if match is not None:
            self._load_expert(
                match=match,
                tensor=tensor,
                params=params,
                stacks=expert_stacks,
                loaded=loaded,
            )
        elif ".attn.wo_a." in name:
            self._load_wo_a(
                name=name,
                tensor=tensor,
                params=params,
                pending=pending_wo_a,
                loaded=loaded,
            )
        else:
            _copy_into(params[name], self._tp_shard(name, tensor))
            loaded.add(name)

    def _tp_shard(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        parts = name.split(".")
        key = parts[-1] if "hc" in name or "attn_sink" in name else parts[-2]
        dim = _SHARD_DIM.get(key)
        if dim is None:
            return tensor
        tp, rank = self.args.tp_size, self.args.tp_rank
        assert tensor.size(dim) % tp == 0, (name, tensor.shape)
        size = tensor.size(dim) // tp
        return tensor.narrow(dim, rank * size, size)

    def _load_expert(self, *, match, tensor, params, stacks, loaded) -> None:
        """Gather this rank's experts of one (layer, w, kind) on host, upload once complete."""
        layer, expert, w, kind = int(match[1]), int(match[2]), match[3], match[4]
        ffn = self.layers[layer].ffn
        local = expert - ffn.experts_start_idx
        if not 0 <= local < ffn.n_local_experts:
            return
        name = f"layers.{layer}.ffn.experts_{w}" + ("_scale" if kind == "scale" else "")
        param = params[name]
        if name not in stacks:
            stacks[name] = (
                torch.empty(param.shape, dtype=param.dtype, device="cpu"),
                [],
            )
        stack, seen = stacks[name]
        stack[local].copy_(_as_dtype(tensor, param.dtype))
        seen.append(local)
        if len(seen) == ffn.n_local_experts:
            assert sorted(seen) == list(range(ffn.n_local_experts)), (name, seen)
            _copy_into(param, stack)
            del stacks[name]
            loaded.add(name)

    def _load_wo_a(self, *, name, tensor, params, pending, loaded) -> None:
        """convert.py: dequantize this rank's wo_a rows to bf16 once weight and scale are both in."""
        prefix, kind = name.rsplit(".", 1)
        parts = pending.setdefault(prefix, {})
        parts[kind] = self._tp_shard(name, tensor)
        if len(parts) < 2:
            return
        weight, scale = parts["weight"], parts["scale"]
        ob, ib = weight.size(0) // scale.size(0), weight.size(1) // scale.size(1)
        assert (ob, ib) in ((32, 32), (128, 128)), (name, weight.shape, scale.shape)
        scale = scale_f32(_as_dtype(scale, torch.uint8))
        w = (
            weight.unflatten(0, (-1, ob)).unflatten(-1, (-1, ib)).float()
            * scale[:, None, :, None]
        )
        _copy_into(params[prefix + ".weight"], w.flatten(2, 3).flatten(0, 1).bfloat16())
        loaded.add(prefix + ".weight")
        del pending[prefix]

    def _load_engram_rows(
        self, *, param: nn.Parameter, path: str, name: str, pad_byte: int
    ) -> None:
        """This rank's ceil(rows / tp) rows, padded past the end as convert.py does, staged on
        host and uploaded in one copy. The rows are one contiguous byte range, read sequentially:
        mmap slices over gcsfuse fault in a page at a time and stalled r4 for over 30 minutes."""
        t0 = time.perf_counter()
        part = param.size(0)
        start = self.args.tp_rank * part
        rows = _read_rows(path, name, start, start + part)
        host = torch.empty(tuple(param.shape), dtype=param.dtype, device="cpu")
        host.view(torch.uint8).fill_(pad_byte)
        host[: rows.size(0)].copy_(_as_dtype(rows, param.dtype))
        _copy_into(param, host)
        logger.info(
            "DeepSeek-V4.1 native: %s rows [%d, %d) in %.1fs",
            name,
            start,
            start + rows.size(0),
            time.perf_counter() - t0,
        )


def _read_rows(path: str, name: str, start: int, stop: int) -> torch.Tensor:
    """Rows [start, stop) of 2-D safetensors tensor `name` (clipped to its row count), read with
    large sequential reads straight from the file."""
    with open(path, "rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        meta = json.loads(f.read(header_len))[name]
        num_rows, cols = meta["shape"]
        dtype = _SAFETENSORS_DTYPES[meta["dtype"]]
        row_bytes = cols * torch.empty((), dtype=dtype).element_size()
        stop = min(stop, num_rows)
        buf = torch.empty(max(stop - start, 0) * row_bytes, dtype=torch.uint8)
        f.seek(8 + header_len + meta["data_offsets"][0] + start * row_bytes)
        view, got = memoryview(buf.numpy()), 0
        while got < len(view):
            n = f.readinto(view[got : got + _READ_CHUNK_BYTES])
            if not n:
                raise EOFError(f"{path}: {name} ended after {got} of {len(view)} bytes")
            got += n
    return buf.view(dtype).view(-1, cols)


def _as_dtype(tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Checkpoint tensor in the parameter's storage dtype; E8M0 scales and packed fp4 stay bytes."""
    if dtype == torch.uint8:
        if tensor.element_size() == 1:
            return tensor.view(torch.uint8)
        # power-of-two float scales -> their E8M0 exponent byte
        return ((tensor.float().view(torch.int32) >> 23) & 0xFF).to(torch.uint8)
    return tensor.to(dtype)


def _copy_into(param: nn.Parameter, tensor: torch.Tensor) -> None:
    value = _as_dtype(tensor, param.dtype)
    assert value.shape == param.shape, (tuple(value.shape), tuple(param.shape))
    param.data.copy_(value.contiguous())


EntryClass = DeepseekV41NativeForCausalLM
