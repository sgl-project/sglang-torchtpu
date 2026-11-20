# SPDX-License-Identifier: Apache-2.0
"""HY-Nomas on TPU workers: segment and PixelUMM tile attention on the Pallas
flash kernel, host-side tables, compiled DiT and decoder regions.

Runtime patch, tracked debt:
- Why: the DiT's segment attention and PixelUMM's sliding-tile attention are
  FA3 / FlexAttention block masks on CUDA; elsewhere a dense SDPA reference
  that refuses masks above 16.7M elements (one 352x640 event: 7880 tokens).
  The Nomas stages bypass ``DenoisingStage``, so nothing is compiled. RoPE and
  MRoPE tables are built on the device, the DiT rotation reads stride-2 lane
  slices, and the Qwen2.5-VL text positions and ``action_metadata`` reduce the
  padded text mask in int64 on the device, which aborts torch_tpu.
- How: :func:`install`
  (1) runs ``NomasAttentionPlan`` for TPU tensors as
      :class:`TpuNomasAttentionPlan`: from the stock FA3 plan's host metadata,
      flash over the current ``[video | head text]`` prefix, resumed (LSE carry)
      over the visible history gathered into a prefix by a host-built index;
      prefix lengths are run-time inputs, so shapes never depend on the prompt.
      Timed-action rows add their action's text by a masked float32 pass;
  (2) orders PixelUMM's tile-major tokens by window centre; each centre's
      queries take one dense flash call over the 3 x 3 tiles around it, the
      key set of the block mask;
  (3) builds the tables on the host, rotates bf16 pairs as ``x @ R`` (exact),
      and computes text positions and action metadata from a host mask copy;
  (4) under ``--enable-torch-compile`` compiles the block halves around the
      segment attention (SPMD-safe: the Ulysses exchanges stay in one
      program), the camera modules, the text refiner and the decoder layers,
      fed offset-0 inputs. The segment attention runs eagerly in between.
- Upstream PR or issue: none.
- Remove when: the segment cache exposes a platform attention hook,
  FlexAttention lowers on TPU and the Nomas stages compile like
  ``DenoisingStage``.
"""

from __future__ import annotations

import math
from functools import lru_cache

import torch
from torch import nn
from torch.nn import functional as F

from sglang.multimodal_gen.runtime.layers.kvcache import nomas_segment_cache
from sglang.multimodal_gen.runtime.models.dits import nomas as dit
from sglang.multimodal_gen.runtime.models.dits import nomas_layers
from sglang.multimodal_gen.runtime.models.dits.nomas_layers import (
    get_nd_rotary_pos_embed,
    local_token_range,
)
from sglang.multimodal_gen.runtime.models.encoders.qwen_vl import QwenVLModelBase
from sglang.multimodal_gen.runtime.models.vaes import nomas as vae
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.nomas import (
    stages,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang_tpu.multimodal_gen import tpu_splash_attn as tsa
from sglang_tpu.multimodal_gen.rope import _pair_swap_matrix

logger = init_logger(__name__)

_DEVICE_TYPES = ("tpu",)
_LN2 = math.log(2.0)

_STOCK_PLAN = nomas_segment_cache.NomasAttentionPlan
_STOCK_PREPARE = dit.NomasTransformer3DModel.prepare_segment_context
_STOCK_LAYOUT = vae._cached_local_layout
_STOCK_FLEX = vae._flex_attention
_STOCK_MROPE = vae.mrope_tables
_STOCK_ACTION_METADATA = stages.action_metadata
_STOCK_ROTARY = nomas_layers.apply_rotary_emb_single
_STOCK_TEXT_ROPE = QwenVLModelBase._get_text_rope_index


def _on_tpu(device) -> bool:
    kind = device.type if isinstance(device, torch.device) else str(device)
    return kind.split(":")[0] in _DEVICE_TYPES


# ---- (1) DiT segment attention ----


def _flash_eligible(query, key, value) -> bool:
    if tsa._ring_kernel_fallback_reason(query, key, value) is not None:
        return False
    tsa._register_ops()
    return tsa._FLASH_RING_OP is not None


def _attend(query, key, value, kv_info, carry, scale, kernel):
    """Attention over the first ``kv_info[1]`` keys, resuming ``carry``.

    Returns the float32 output and the carry's log-sum-exp: base 2 with
    NUM_SUBLANES rows from the kernel, natural log from the reference.
    """
    if kernel:
        return tsa._FLASH_RING_OP(
            query, key, value, kv_info, *(carry or (None, None)), scale
        )
    out, lse = tsa._attention_with_lse_reference(
        query, key, value, scale, real_keys=kv_info[1]
    )
    if carry is not None:
        out, lse = tsa._merge_attention(out, lse, *carry)
    return out, lse


def _action_attention(query, key, value, scale, query_action, text_action):
    """float32 attention of each row over its action's text keys only.

    Rows outside every action see no key: output 0, log-sum-exp -inf.
    """
    key_t, value = key.float().transpose(-1, -2), value.float()
    visible = text_action >= 0
    chunk = tsa._reference_chunk_rows(query, key)
    outs, lses = [], []
    for start in range(0, query.shape[2], chunk):
        rows = query_action[start : start + chunk]
        mask = visible[None, :] & (rows[:, None] == text_action[None, :])
        scores = torch.matmul(query[:, :, start : start + chunk].float(), key_t)
        scores = (scores * scale).masked_fill(~mask, float("-inf"))
        lse = torch.logsumexp(scores, dim=-1)
        probs = torch.exp(scores - lse[..., None]).nan_to_num(0.0)
        outs.append(torch.matmul(probs, value))
        lses.append(lse)
    return torch.cat(outs, 2), torch.cat(lses, 2)


class TpuNomasAttentionPlan:
    """``NomasAttentionPlan`` executed by the TPU flash kernel.

    Same mask as the stock plan: every query sees the visible history, the
    current video and the head text; a timed action's rows also see that
    action's text; padded text queries return zeros. Built once per segment
    (and after an eviction), from host copies of the masks.
    """

    backend = "tpu"

    def __init__(self, text_mask, history_mask, *layout):
        """``layout``: frames, spatial tokens, text offsets, video windows[, backend]."""
        device = text_mask.device
        # The stock FA3 plan validates the layout and computes the visible
        # prefix and the action ranges from index metadata, without a mask.
        host = _STOCK_PLAN(text_mask.cpu(), history_mask.cpu(), *layout[:4], "fa3")
        self.video_tokens = host.video_tokens
        self.history_length = host.history_length
        visible = host.history_index.numel()
        int32 = dict(dtype=torch.int32)
        self.query_valid = host.query_valid.to(device)
        prefix = host.prefix_tokens - visible
        self.current_keys = torch.tensor([0, prefix], **int32).to(device)
        self.history_keys = torch.tensor([0, visible], **int32).to(device)
        self.history_index = None
        if self.history_length:
            index = torch.zeros(self.history_length, **int32)
            index[:visible] = host.history_index
            self.history_index = index.to(device)
        self.query_action = self.text_action = None
        if host.actions:
            text = text_mask.shape[1]
            offset = visible + self.video_tokens
            query_action = torch.full((self.video_tokens + text,), -1, **int32)
            text_action = torch.full((text,), -1, **int32)
            ids = {}
            for (q_start, q_end), (k_start, k_end), *_ in host.actions:
                span = (k_start - offset, k_end - offset)
                action = ids.setdefault(span, len(ids))
                query_action[q_start:q_end] = action
                text_action[span[0] : span[1]] = action
            self.query_action = query_action.to(device)
            self.text_action = text_action.to(device)

    def __call__(self, q, k, v, layer_cache=None, history_length=0):
        if history_length != self.history_length:
            raise ValueError("attention plan was built for a different history length")
        scale = q.shape[-1] ** -0.5
        q, k, v = (x.transpose(1, 2).contiguous() for x in (q, k, v))
        history = None
        if history_length:
            history = [
                layer_cache[key][0]
                .index_select(0, self.history_index)
                .permute(1, 0, 2)[None]
                .contiguous()
                for key in ("k", "v")
            ]
        kernel = _flash_eligible(q, k, v) and (
            history is None or _flash_eligible(q, *history)
        )
        out, lse = _attend(q, k, v, self.current_keys, None, scale, kernel)
        if history is not None:
            out, lse = _attend(
                q, *history, self.history_keys, (out, lse), scale, kernel
            )
        if self.query_action is not None:
            lse = lse[:, :, 0] * _LN2 if kernel else lse
            t = self.video_tokens
            actions = (self.query_action, self.text_action)
            extra = _action_attention(q, k[:, :, t:], v[:, :, t:], scale, *actions)
            out, lse = tsa._merge_attention(out, lse, *extra)
        out = out.to(q.dtype) * self.query_valid[:, None, :, None]
        return out.transpose(1, 2).contiguous()


def _attention_plan(text_mask, *args, **kwargs):
    if _on_tpu(text_mask.device):
        return TpuNomasAttentionPlan(text_mask, *args, **kwargs)
    return _STOCK_PLAN(text_mask, *args, **kwargs)


# ---- (2) PixelUMM sliding-tile attention ----


class TileGroups:
    """Query slices of a group-major token order and each group's key indices."""

    def __init__(self, slices, keys):
        self.slices = slices
        self.keys = keys


@lru_cache(maxsize=8)
def _cached_local_layout(grid, tile, device):
    """``vae._cached_local_layout`` with a group-major order on TPU devices.

    The stock tile-major order, stably regrouped by each tile's clamped window
    centre. The mask is the stock one, so each group's keys are the tokens of
    the 3 x 3 tiles around its centre.
    """
    if not _on_tpu(device):
        return _STOCK_LAYOUT(grid, tile, device)
    frames, height, width = grid
    tiles_h, tiles_w = -(-height // tile), -(-width // tile)
    t, h, w = torch.meshgrid(*(torch.arange(n) for n in grid), indexing="ij")
    tile_h, tile_w = (h // tile).flatten(), (w // tile).flatten()
    order = (
        ((tile_h * tiles_w + tile_w) * frames + t.flatten()) * tile
        + (h % tile).flatten()
    ) * tile
    perm = torch.argsort(order + (w % tile).flatten())
    center_h = tile_h[perm].clamp(1, max(tiles_h - 2, 1))
    center_w = tile_w[perm].clamp(1, max(tiles_w - 2, 1))
    perm = perm[torch.argsort(center_h * tiles_w + center_w, stable=True)]
    tile_h, tile_w = tile_h[perm], tile_w[perm]
    center_h = tile_h.clamp(1, max(tiles_h - 2, 1))
    center_w = tile_w.clamp(1, max(tiles_w - 2, 1))
    group = center_h * tiles_w + center_w
    slices, keys = [], []
    start = 0
    for value, count in zip(*torch.unique_consecutive(group, return_counts=True)):
        ch, cw = divmod(int(value), tiles_w)
        mask = ((ch - tile_h).abs() <= 1) & ((cw - tile_w).abs() <= 1)
        slices.append((start, start + int(count)))
        keys.append(mask.nonzero().squeeze(-1).to(device))
        start += int(count)
    return perm.to(device), torch.argsort(perm).to(device), TileGroups(slices, keys)


def _flex_attention(q, k, v, block_mask):
    """Grouped dense attention for a ``TileGroups`` layout, else the stock call."""
    if not isinstance(block_mask, TileGroups):
        return _STOCK_FLEX(q, k, v, block_mask)
    outs = []
    for (start, end), index in zip(block_mask.slices, block_mask.keys):
        outs.append(
            tsa.tpu_attention(
                q[:, :, start:end].contiguous(),
                k.index_select(2, index),
                v.index_select(2, index),
            )
        )
    return torch.cat(outs, 2)


# ---- (3) host-side tables and mask reads ----


def _mrope_tables(arch, grid, device):
    if not _on_tpu(device):
        return _STOCK_MROPE(arch, grid, device)
    return _host_mrope_tables(arch, tuple(grid), str(device))


@lru_cache(maxsize=8)
def _host_mrope_tables(arch, grid, device):
    return tuple(x.to(device) for x in _STOCK_MROPE(arch, grid, torch.device("cpu")))


def prepare_segment_context(self, latent, text_mask, cache, *args, rope=None, **kw):
    """The segment context, with RoPE tables built on the host for TPU latents
    (whose model's LayerNorms take autocast's float32 policy)."""
    if _on_tpu(latent.device) and not getattr(self, "_fp32_norms", False):
        for module in self.modules():
            if type(module) is nn.LayerNorm:
                module.__class__ = Fp32AutocastLayerNorm
        self._fp32_norms = True
    if rope is None and _on_tpu(latent.device):
        t, h, w = latent.shape[2:]
        c, offset = self.config, cache.frame_offset
        tables = get_nd_rotary_pos_embed(
            c.rope_dim_list, (offset, 0, 0), (offset + t, h, w), c.rope_theta
        )
        lo, hi = local_token_range(t * h * w, self.parallel)
        rope = tuple(x[lo:hi].clone().to(latent.device) for x in tables)
    return _STOCK_PREPARE(self, latent, text_mask, cache, *args, rope=rope, **kw)


def apply_rotary_emb_single(x, freqs_cis):
    """The DiT's pair rotation without stride-2 lane slices for bf16 TPU inputs.

    ``x @ R`` (R a signed pair swap) has one +-1 product per output, so it is
    exact in bf16 and equals the stock ``rotate_half``; the rotation then runs
    in float32 and rounds once, as the stock path does.
    """
    if x.dtype != torch.bfloat16 or not _on_tpu(x.device):
        return _STOCK_ROTARY(x, freqs_cis)
    cos, sin = nomas_layers._rope_tables(x, freqs_cis)
    rot = _pair_swap_matrix(x.shape[-1], x.dtype, x.device)
    return (x.float() * cos + torch.matmul(x, rot).float() * sin).type_as(x)


def _get_text_rope_index(self, input_ids, attention_mask):
    """Qwen2.5-VL text positions from a host copy of a TPU mask: the stock
    int64 cumsum and max abort torch_tpu ("conversion from vmreg/vreg to U64")."""
    if attention_mask is None or not _on_tpu(attention_mask.device):
        return _STOCK_TEXT_ROPE(self, input_ids, attention_mask)
    out = _STOCK_TEXT_ROPE(self, input_ids, attention_mask.cpu())
    return tuple(x.to(attention_mask.device) for x in out)


def action_metadata(caption, config, tokenizer, text_mask, frames, fps):
    """``action_metadata`` reading the valid text length from a host copy."""
    if _on_tpu(text_mask.device):
        text_mask = text_mask.cpu()
    return _STOCK_ACTION_METADATA(caption, config, tokenizer, text_mask, frames, fps)


# ---- (4) compiled regions ----


def _compile_enabled() -> bool:
    try:
        from sglang.multimodal_gen.runtime.server_args import get_global_server_args

        return bool(get_global_server_args().enable_torch_compile)
    except Exception:  # noqa: BLE001 - no server (unit tests)
        return False


def _compile_backend():
    from sglang.multimodal_gen.runtime.platforms import current_platform

    return current_platform.get_compile_backend()


def _compile(fn, spmd):
    """``fn`` compiled; ``spmd``: as one SPMD-safe program (identical on every rank)."""
    from sglang.multimodal_gen.runtime.platforms import current_platform

    wrap = current_platform.prepare_spmd_region_wrapper() if spmd else None
    return torch.compile(
        fn if wrap is None else wrap(fn), backend=_compile_backend(), dynamic=False
    )


def _offset0(x):
    # torch_tpu guards compiled inputs' storage offsets: block 0 gets this
    # rank's slice of the patch embedding, at a rank-dependent offset.
    if not isinstance(x, torch.Tensor) or (
        x.storage_offset() == 0 and x.is_contiguous()
    ):
        return x
    return x.clone(memory_format=torch.contiguous_format)


_REGIONS: dict[str, object] = {}


def _regional(name, stock, spmd=False, tpu=None):
    """``tpu`` (default ``stock``) for TPU tensors, compiled once under
    ``--enable-torch-compile``; module instances share the graph."""
    tpu = tpu or stock

    def method(self, x, *args, **kwargs):
        if not _on_tpu(x.device):
            return stock(self, x, *args, **kwargs)
        if name not in _REGIONS:
            _REGIONS[name] = _compile(tpu, spmd) if _compile_enabled() else None
            if _REGIONS[name] is not None:
                logger.info("HY-Nomas on TPU: compiled %s", name)
        if _REGIONS[name] is None:
            return tpu(self, x, *args, **kwargs)
        return _REGIONS[name](self, *map(_offset0, (x, *args)), **kwargs)

    return method


# ---- (5) CUDA autocast's float32 policy ----


class Fp32AutocastLayerNorm(nn.LayerNorm):
    """``nn.LayerNorm`` in float32 under autocast, as CUDA autocast runs it;
    torch_tpu's autocast keeps it in bf16."""

    def forward(self, x):
        if not torch.is_autocast_enabled(x.device.type):
            return super().forward(x)
        w, b = (None if p is None else p.float() for p in (self.weight, self.bias))
        return F.layer_norm(x.float(), self.normalized_shape, w, b, self.eps)


def _text_refiner_forward(self, x, t, mask):
    """``NomasTextRefiner.forward`` pooling with float32 sums under autocast."""
    weights = mask.to(x.dtype).unsqueeze(-1)
    if torch.is_autocast_enabled(x.device.type):
        pooled = (x * weights).float().sum(1) / weights.float().sum(1)
    else:
        pooled = (x * weights).sum(1) / weights.sum(1)
    c = self.t_embedder(t) + self.c_embedder(pooled)
    return self.individual_token_refiner(self.input_embedder(x), c, mask)


_REGIONAL = (
    # Ulysses exchanges inside: every rank runs the same program.
    (dit.NomasDoubleStreamBlock, "attention_inputs", True, None),
    (dit.NomasDoubleStreamBlock, "attention_outputs", True, None),
    (dit.NomasCameraAttention, "forward", False, None),
    (dit.NomasTextRefiner, "forward", False, _text_refiner_forward),
    (vae.GenLayer, "forward", False, None),
)
_STOCK_METHODS = {(cls, name): getattr(cls, name) for cls, name, *_ in _REGIONAL}


def patches():
    """``(owner, name, replacement)`` of every name :func:`install` binds."""
    yield dit, "NomasAttentionPlan", _attention_plan
    yield (
        dit.NomasTransformer3DModel,
        "prepare_segment_context",
        prepare_segment_context,
    )
    yield vae, "_cached_local_layout", _cached_local_layout
    yield vae, "_flex_attention", _flex_attention
    yield vae, "mrope_tables", _mrope_tables
    yield stages, "action_metadata", action_metadata
    yield nomas_layers, "apply_rotary_emb_single", apply_rotary_emb_single
    yield QwenVLModelBase, "_get_text_rope_index", _get_text_rope_index
    for cls, name, spmd, tpu in _REGIONAL:
        stock = _STOCK_METHODS[cls, name]
        yield cls, name, _regional(f"{cls.__name__}.{name}", stock, spmd, tpu)


def install() -> None:
    for owner, name, value in patches():
        setattr(owner, name, value)
