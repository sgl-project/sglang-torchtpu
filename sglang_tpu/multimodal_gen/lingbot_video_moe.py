# SPDX-License-Identifier: Apache-2.0
"""LingBot-Video-MoE on TPU workers: the Pallas fused MoE, one block program per length bucket.

Runtime patch, tracked debt:
- Why: the DiT's routed experts run srt's Triton ``fused_experts``, which has
  no TPU path. The DiT forward reads the text length from an int64 device
  reduction and rebuilds its RoPE table on the device from ``tolist()``
  positions every step. Its joint sequence has the prompt's length, so every
  prompt compiles the block region again, and the positive and negative
  prompts of one request need two programs (CFG-parallel ranks would run
  different ones). Stock sequence parallelism shards the latent frames, so
  requests round their frame count up to a multiple of ``--num-gpus``; each
  rank appends the whole text, not marked as replicated, and numbers its
  frames from 0. The Qwen3-VL text stage reads the real length and the MRoPE
  positions from int64 device reductions of the attention mask.
- How: :func:`install`, in the workers and the server process
  (1) runs the routed experts through sglang_tpu's Pallas ``fused_moe_call``,
      with expert weights moved once to its ``(experts, in, out)`` layout;
  (2) pads the joint sequence to a bucket of 512 tokens shared by both CFG
      branches. The stock RoPE table is built on the host, and the real key
      count goes in as device int32 that the splash ring step reads;
  (3) keeps the latents whole and the requested frame count, gives each
      Ulysses rank an equal share of the padded sequence, all-gathers the output;
  (4) computes the timestep modulation once instead of once per token;
  (5) runs the text encoder over prompt tokens right-padded to a multiple of
      256 (causal: real tokens never see the padding), with int32 positions
      built on the host, no KV cache and only the last token's logits.
- Upstream PR or issue: none.
- Remove when: the DiT takes a platform MoE runner, a host-built RoPE table and
  a length bucket with the key count as data, and the text stage builds its
  positions on the host.
"""

from __future__ import annotations

import functools
import math

import torch
import torch.nn.functional as F

from sglang.multimodal_gen.configs.pipeline_configs.lingbot_video_moe import (
    LingBotVideoMoEPipelineConfig,
)
from sglang.multimodal_gen.configs.sample.lingbot_video_moe import (
    LingBotVideoMoESamplingParams,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_ring_parallel_world_size,
    get_sp_parallel_rank,
    get_sp_world_size,
)
from sglang.multimodal_gen.runtime.distributed.sp_shard_utils import gather_seq
from sglang.multimodal_gen.runtime.layers.moe import (
    LingBotVideoGroupedExperts,
    LingBotVideoSparseMoeBlock,
)
from sglang.multimodal_gen.runtime.layers.usp import (
    _usp_input_all_to_all_qkv,
    _usp_output_all_to_all,
)
from sglang.multimodal_gen.runtime.managers.forward_context import (
    get_forward_context_or_none,
)
from sglang.multimodal_gen.runtime.models.dits import lingbot_video_moe as dit
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.lingbot_video_moe.text_encoding import (
    LingBotVideoTextEncodingStage,
)
from sglang.multimodal_gen.runtime.server_args import get_global_server_args
from sglang_tpu.multimodal_gen import tpu_splash_attn

# The splash kernel takes 512 or more keys; _bucket also rounds to the SP degree.
SEQUENCE_ALIGNMENT = 512
PROMPT_TOKEN_ALIGNMENT = 256
# Devices that take these paths.
_DEVICE_TYPES = ("tpu",)

_STOCK_RUN_EXPERTS = LingBotVideoSparseMoeBlock._run_sglang_triton_experts
_STOCK_ATTENTION_FORWARD = dit.LingBotVideoAttention.forward
_STOCK_DIT_FORWARD = dit.LingBotVideoTransformer3DModel.forward
_STOCK_ENCODE_INPUTS = LingBotVideoTextEncodingStage._encode_inputs
_STOCK_SHARD_LATENTS = LingBotVideoMoEPipelineConfig.shard_latents_for_sp
_STOCK_SAMPLING_POST_INIT = LingBotVideoMoESamplingParams.__post_init__


# --- MoE -------------------------------------------------------------------


def run_experts(self, tokens, top_scores, top_indices):
    """``_run_sglang_triton_experts`` through the Pallas fused MoE."""
    if tokens.device.type not in _DEVICE_TYPES:
        return _STOCK_RUN_EXPERTS(self, tokens, top_scores, top_indices)
    if not getattr(self.experts, "_tpu_layout", False):
        raise RuntimeError("LingBot-Video-MoE experts are not in the TPU layout")
    from sglang_tpu.moe.pallas.fused_moe_op import fused_moe_call

    # The stock casts. The router returns bf16 scores, so the bf16 weights
    # are the values the stock hands Triton as float32.
    return fused_moe_call(
        hidden_states=tokens.contiguous().bfloat16(),
        w1=self.experts.w13_weight.bfloat16(),
        w2=self.experts.w2.bfloat16(),
        topk_weights=top_scores.bfloat16(),
        topk_ids=top_indices.to(torch.int32),
        topk=self.top_k,
        # silu(gate) * up on the float32 accumulator, then rounded once.
        activation="silu",
    ).type_as(tokens)


def _to_tpu_layout(model) -> None:
    """Moves routed-expert weights once to the fused MoE's (experts, in, out) layout."""
    for module in model.modules():
        if isinstance(module, LingBotVideoGroupedExperts) and not getattr(
            module, "_tpu_layout", False
        ):
            module.w13_weight.data = module.w13_weight.data.transpose(1, 2).contiguous()
            module.w2.data = module.w2.data.transpose(1, 2).contiguous()
            module._tpu_layout = True
            if module.w2.device.type in _DEVICE_TYPES:
                # Drain so this module's old weights free before the next copy.
                torch.accelerator.synchronize()


def _tpu_metadata():
    from sglang_tpu.pallas_export import tpu_info_for_cpu_export

    return tpu_info_for_cpu_export()


# --- DiT -------------------------------------------------------------------


def attention_forward(self, x, freqs_cis, attention_mask=None, attn_mask_meta=None):
    """``LingBotVideoAttention.forward`` over the real keys of a padded sequence."""
    if not (isinstance(attn_mask_meta, dict) and "tpu_kv_info" in attn_mask_meta):
        return _STOCK_ATTENTION_FORWARD(
            self, x, freqs_cis, attention_mask, attn_mask_meta
        )
    # The stock projections, norms and RoPE.
    cos, sin = freqs_cis
    heads, head_dim = self.local_num_heads, self.head_dim
    q, _ = self.to_q(x)
    k, _ = self.to_k(x)
    v, _ = self.to_v(x)
    q = dit._apply_rotary_emb(
        self.norm_q(q.unflatten(2, (heads, head_dim))), cos, sin, is_neox_style=False
    )
    k = dit._apply_rotary_emb(
        self.norm_k(k.unflatten(2, (heads, head_dim))), cos, sin, is_neox_style=False
    )
    v = v.unflatten(2, (heads, head_dim))
    q, k, v = _usp_input_all_to_all_qkv(q, k, v)
    # One ring step over every key: the kernel, or the float32 reference for
    # shapes it does not take, counts the first kv_info[1] keys.
    out, _ = tpu_splash_attn.tpu_ring_attention_step(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        attn_mask_meta["tpu_kv_info"],
        None,
        self.attn.softmax_scale,
    )
    out = _usp_output_all_to_all(out.transpose(1, 2), head_dim=2)
    out, _ = self.to_out(out.flatten(2))
    return out


def _bucket(n_video: int, text_len: int, sp: int) -> int:
    """Joint-sequence length shared by the request's CFG branches."""
    lengths = [text_len]
    context = get_forward_context_or_none()
    batch = context.forward_batch if context is not None else None
    for embeds in (
        getattr(batch, "prompt_embeds", None),
        getattr(batch, "negative_prompt_embeds", None),
    ):
        if isinstance(embeds, (list, tuple)):
            embeds = embeds[0] if embeds else None
        if embeds is not None:
            lengths.append(embeds.shape[1])
    alignment = math.lcm(SEQUENCE_ALIGNMENT, sp)
    return -(-(n_video + max(lengths)) // alignment) * alignment


@functools.lru_cache(maxsize=8)
def _layout(rotary_emb, grid, text_len, bucket, sp, rank, device):
    """This rank's RoPE rows and the real key count of one branch, built on the host."""
    n_video = math.prod(grid)
    # The stock positions: padding rows after the real text get (0, 0, 0).
    positions = dit._joint_position_ids(
        torch.tensor([text_len]), *grid, bucket - n_video, "cpu"
    )
    rows = bucket // sp
    cos, sin = (
        table[rank * rows : (rank + 1) * rows].to(device)
        for table in rotary_emb.forward_uncached(positions)
    )
    kv_info = torch.tensor([0, n_video + text_len], dtype=torch.int32).to(device)
    return cos, sin, kv_info


def dit_forward(self, hidden_states, timestep, encoder_hidden_states, *rest, **kwargs):
    """``LingBotVideoTransformer3DModel.forward`` over a bucketed joint sequence."""
    if hidden_states.device.type not in _DEVICE_TYPES:
        return _STOCK_DIT_FORWARD(
            self, hidden_states, timestep, encoder_hidden_states, *rest, **kwargs
        )
    args = get_global_server_args()
    if getattr(args, "enable_torch_compile", False) and not args.regional_compile:
        # The forward builds host tables and re-lays out weights eagerly.
        raise ValueError(
            "LingBot-Video-MoE on TPU compiles per block: add --regional-compile"
        )
    batch_size, channels, frames, height, width = hidden_states.shape
    if batch_size != 1 or get_ring_parallel_world_size() > 1:
        raise NotImplementedError(
            "LingBot-Video-MoE on TPU runs one sample per CFG branch and shards "
            "with Ulysses only"
        )
    _to_tpu_layout(self)
    pt, ph, pw = self.patch_size
    grid = (frames // pt, height // ph, width // pw)
    n_video = math.prod(grid)
    # A single prompt reaches the DiT trimmed to its real tokens.
    text_len = encoder_hidden_states.shape[1]
    sp = get_sp_world_size()
    rank = get_sp_parallel_rank() if sp > 1 else 0
    bucket = _bucket(n_video, text_len, sp)
    cos, sin, kv_info = _layout(
        self.rotary_emb, grid, text_len, bucket, sp, rank, hidden_states.device
    )

    # The stock patchify and embeddings, with zero text rows up to the bucket.
    patches = hidden_states.reshape(1, channels, grid[0], pt, grid[1], ph, grid[2], pw)
    patches = patches.permute(0, 2, 4, 6, 3, 5, 7, 1).reshape(1, n_video, -1)
    text = F.pad(encoder_hidden_states, (0, 0, 0, bucket - n_video - text_len))
    joint = torch.cat((self.patch_embedder(patches), self.text_embedder(text)), 1)
    rows = bucket // sp
    if sp > 1:
        # A fresh shard: a view's storage offset would differ per rank, and
        # each rank must compile the same block program.
        joint = joint[:, rank * rows : (rank + 1) * rows].clone(
            memory_format=torch.contiguous_format
        )
    # Every token has the same timestep: modulate once, repeat the row.
    t_emb = self.time_embedder(self.time_proj(timestep.float()))
    temb6 = self.time_modulation(t_emb).expand(rows, -1).contiguous()
    meta = {"tpu_kv_info": kv_info}
    from sglang_tpu.moe.pallas.fused_moe_op import _build_fused_moe_op

    _build_fused_moe_op()  # register the op before the block regions trace
    # The diffusion launcher runs JAX on the CPU; the fused MoE reads the TPU's
    # metadata while it is exported.
    with _tpu_metadata():
        for block in self.blocks:
            joint = block(joint, temb6, (cos, sin), None, meta)
    shift, scale = self.norm_out_modulation(t_emb)[:, None].chunk(2, dim=-1)
    final_hidden = self.norm_out(joint) * (1.0 + scale) + shift
    projected = self.proj_out(final_hidden.to(self.proj_out.weight.dtype))
    x = gather_seq(projected, bucket)[:, :n_video]
    x = x.reshape(1, *grid, pt, ph, pw, self.out_channels)
    x = x.permute(0, 7, 1, 4, 2, 5, 3, 6)
    return x.reshape(1, self.out_channels, frames, height, width)


def shard_latents_for_sp(self, batch, latents):
    """Keeps the latents whole: the TPU DiT forward shards the joint sequence."""
    if latents.device.type not in _DEVICE_TYPES:
        return _STOCK_SHARD_LATENTS(self, batch, latents)
    return latents, False


def sampling_post_init(self) -> None:
    """``LingBotVideoMoESamplingParams.__post_init__`` that keeps the frame count."""
    _STOCK_SAMPLING_POST_INIT(self)
    # The stock rounds the frames up to --num-gpus for the latent sharding.
    self.adjust_frames = False


# --- Text encoder ----------------------------------------------------------


def encode_inputs(self, inputs, device, dtype):
    """``_encode_inputs`` over right-padded tokens with host-built positions."""
    if torch.device(device).type not in _DEVICE_TYPES:
        return _STOCK_ENCODE_INPUTS(self, inputs, device, dtype)
    self._begin_text_encoder_use(0)
    ids, mask = inputs["input_ids"], inputs["attention_mask"]
    length = ids.shape[1]
    padded = -(-length // PROMPT_TOKEN_ALIGNMENT) * PROMPT_TOKEN_ALIGNMENT
    padded_mask = F.pad(mask, (0, padded - length))
    # The positions Qwen3-VL derives from the mask (get_rope_index), in int32.
    positions = (padded_mask.long().cumsum(-1) - 1).masked_fill(padded_mask == 0, 1)
    positions = positions.to(torch.int32)[None].expand(3, -1, -1).contiguous()
    outputs = self._forward_text_encoder(
        self.text_encoders[0],
        {
            "input_ids": F.pad(ids, (0, padded - length)).to(torch.int32).to(device),
            "position_ids": positions.to(device),
            # Read only with a KV cache; given so no int64 arange runs on device.
            "cache_position": torch.arange(padded, dtype=torch.int32).to(device),
            # The attention is causal and ignores the mask.
            "attention_mask": None,
            "output_hidden_states": self.hidden_state_skip_layer is not None,
            "use_cache": False,
            "logits_to_keep": 1,
        },
    )
    if self.hidden_state_skip_layer is not None:
        prompt_embeds = outputs.hidden_states[-(self.hidden_state_skip_layer + 1)]
    else:
        prompt_embeds = outputs.last_hidden_state
    crop_start = self._compute_crop_start()
    prompt_embeds = prompt_embeds[:, crop_start:length]
    prompt_mask = mask[:, crop_start:]
    if prompt_embeds.shape[0] == 1:
        # The stock trim, with the real length read from the host mask.
        true_len = int(prompt_mask[0].sum())
        prompt_embeds = prompt_embeds[:, :true_len]
        prompt_mask = prompt_mask[:, :true_len]
    return prompt_embeds.to(dtype=dtype), prompt_mask.to(device)


def install() -> None:
    LingBotVideoSparseMoeBlock._run_sglang_triton_experts = run_experts
    dit.LingBotVideoAttention.forward = attention_forward
    dit.LingBotVideoTransformer3DModel.forward = dit_forward
    # Every rank runs each block on an equal fresh shard of the shared bucket,
    # with the key count as device data.
    dit.LingBotVideoTransformer3DModel._compile_regions_spmd_safe = True
    LingBotVideoMoEPipelineConfig.shard_latents_for_sp = shard_latents_for_sp
    LingBotVideoMoESamplingParams.__post_init__ = sampling_post_init
    LingBotVideoTextEncodingStage._encode_inputs = encode_inputs
