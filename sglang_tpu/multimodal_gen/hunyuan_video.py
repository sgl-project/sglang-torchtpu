# SPDX-License-Identifier: Apache-2.0
"""HunyuanVideo on TPU workers: a fixed text bucket with its real keys counted on device.

Runtime patch, tracked debt:
- Why: the DiT attends [video | prompt] at the prompt's exact token count and
  shards the prompt across SP ranks by that count. New prompt lengths compile
  again, and uneven shards (or a short prompt replicated on every rank) give
  the ranks different collective programs; on TPU the server hung in its
  warmup.
- How: :func:`install`
  (1) right-pads the Llama prompt tokens to its max length (the encoder is
      causal, so real tokens never see the padding);
  (2) pads the DiT's prompt rows to a multiple of 512 outside the compiled
      call, with the real prompt length as a device int32 count. The token
      refiner averages and attends only the real tokens; every SP rank takes
      an equal share of the prompt;
  (3) the joint attention reads each segment's real keys as resumable kernel
      steps (video keys, then prompt keys). These are the keys the stock DiT
      attends, in the same order, and padded keys are never read;
  (4) regional compile takes only the double- and single-stream blocks (the
      token refiner runs eager, once per step) and marks them spmd-safe, so
      each block keeps its collectives in one program. The bucketed forward
      gives every rank the same region inputs; the stock forward's uneven
      prompt shards would not, so TPU latents never take it;
  (5) the initial latents are drawn in float32 and cast to their dtype (fp16):
      a reduced-precision draw cuts the normal's tails (an fp16 draw on the
      CPU stops at |x| = 3.9).
- Upstream PR or issue: none.
- Remove when: the DiT takes a platform prompt bucket with its real length as
  data, and the latent stage draws reduced-precision noise in float32.
"""

from __future__ import annotations

import functools

import torch
import torch.nn.functional as F

from sglang.multimodal_gen.configs.models.fsdp import is_double_block, is_single_block
from sglang.multimodal_gen.configs.pipeline_configs import hunyuan as config
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_ring_parallel_world_size,
    get_sp_parallel_rank,
    get_sp_world_size,
)
from sglang.multimodal_gen.runtime.layers.attention.layer import USPAttention
from sglang.multimodal_gen.runtime.layers.rotary_embedding.factory import (
    get_rotary_pos_embed,
)
from sglang.multimodal_gen.runtime.layers.usp import (
    _usp_input_all_to_all,
    _usp_output_all_to_all,
)
from sglang.multimodal_gen.runtime.models.dits import hunyuanvideo as dit
from sglang.multimodal_gen.runtime.pipelines_core.stages import latent_preparation
from sglang.multimodal_gen.runtime.server_args import get_global_server_args
from sglang_tpu.multimodal_gen import tpu_splash_attn

# The kernel's prompt step needs >= 512 keys.
TEXT_ALIGNMENT = 512
# Devices that take these paths.
_DEVICE_TYPES = ("tpu",)

_STOCK_TOKENIZE = config.HunyuanConfig.tokenize_prompt
_STOCK_FORWARD = dit.HunyuanVideoTransformer3DModel.forward
_STOCK_USP_FORWARD = USPAttention.forward
_STOCK_RANDN = latent_preparation.randn_tensor


class Segments(list):
    """``(rows per rank, [0, real keys])`` per segment, in each rank's row order.

    Passed to the blocks in place of ``seq_lens``; every segment is padded at
    its tail.
    """


def _counts(real: int, device) -> torch.Tensor:
    return torch.tensor([0, real], dtype=torch.int32).to(device)


def _uses_kernel(impl, q, keys) -> bool:
    if not getattr(impl, "use_splash", False) or q.device.type != "tpu":
        return False
    # Every step takes the kernel or none does: their carries differ.
    return all(
        tpu_splash_attn._ring_kernel_fallback_reason(q, key, key) is None
        for key in keys
    )


def segment_attention(attn, q, k, v, segments, local):
    """``[B, S, H, D]`` attention over each segment's real keys."""
    sp = 1 if local else get_sp_world_size()
    heads = q.shape[2]
    if sp > 1:
        if get_ring_parallel_world_size() > 1 or attn.sp_attention_mode == "kv_gather":
            raise NotImplementedError("HunyuanVideo on TPU shards with Ulysses only")
        pad = (0, 0, 0, -heads % sp)
        q, k, v = (_usp_input_all_to_all(F.pad(t, pad), head_dim=2) for t in (q, k, v))
    rows = [n for n, _ in segments]

    # Ulysses gathers [rank 0 rows | rank 1 rows | ...]: a segment's rows from
    # every rank form one run of keys, real keys first.
    def runs(t):
        return [
            part.flatten(1, 2).transpose(1, 2)
            for part in t.unflatten(1, (sp, -1)).split(rows, 2)
        ]

    q, keys, values = q.transpose(1, 2), runs(k), runs(v)
    impl = attn.attn_impl
    if _uses_kernel(impl, q, keys):
        carry = None
        for key, value, (_, real) in zip(keys, values, segments, strict=True):
            out, carry = impl.forward_ring_resume(q, key, value, real, carry)
    else:
        valid = torch.cat(
            [
                torch.arange(key.shape[2], device=q.device) < real[1]
                for key, (_, real) in zip(keys, segments, strict=True)
            ]
        )
        # Zero padded keys and values: a zero probability times NaN is NaN.
        out = F.scaled_dot_product_attention(
            q,
            torch.where(valid[:, None], torch.cat(keys, 2), 0),
            torch.where(valid[:, None], torch.cat(values, 2), 0),
            attn_mask=valid[None, None, None],
            scale=impl.softmax_scale,
        )
    out = out.transpose(1, 2)
    if sp > 1:
        out = _usp_output_all_to_all(out, head_dim=2)[:, :, :heads]
    return out


def usp_forward(self, q, k, v, *args, **kwargs):
    segments = kwargs.get("seq_lens")
    if isinstance(segments, Segments):
        local = self.skip_sequence_parallel or kwargs.get(
            "skip_sequence_parallel_override", False
        )
        return segment_attention(self, q, k, v, segments, local)
    return _STOCK_USP_FORWARD(self, q, k, v, *args, **kwargs)


def _padded_tokenize(self, prompts, tokenizer, tok_kwargs):
    """The Llama prompt tokens right-padded to the tokenizer's max length."""
    inputs = _STOCK_TOKENIZE(self, prompts, tokenizer, tok_kwargs)
    length = tok_kwargs.get("max_length")
    if "llama" not in type(tokenizer).__name__.lower() or not length:
        return inputs
    extra = length - inputs["input_ids"].shape[1]
    if extra > 0:
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        inputs["input_ids"] = F.pad(inputs["input_ids"], (0, extra), value=pad_id)
        inputs["attention_mask"] = F.pad(inputs["attention_mask"], (0, extra))
    return inputs


def _refine(refiner, x, t, keys):
    """``SingleTokenRefiner.forward`` over the real tokens of a padded prompt."""
    real = (torch.arange(x.shape[1], device=x.device) < keys[1])[None, :, None]
    # The mean of the real rows, accumulated in float32 as torch.mean does.
    context = ((x.float() * real).sum(1) / keys[1].float()).to(x.dtype)
    c = refiner.t_embedder(t) + refiner.c_embedder(context)
    x, _ = refiner.input_embedder(x)
    segments = [(x.shape[1], keys)]
    for block in refiner.refiner_blocks:
        gate_msa, gate_mlp = block.adaLN_modulation(c).chunk(2, dim=-1)
        qkv, _ = block.self_attn_qkv(block.norm1(x))
        batch, length = qkv.shape[:2]
        q, k, v = qkv.view(
            batch, length, 3, block.local_num_attention_heads, -1
        ).unbind(2)
        out = segment_attention(block.attn, q, k, v, segments, local=True)
        out, _ = block.self_attn_proj(out.reshape(batch, length, -1))
        x = x + out * gate_msa.unsqueeze(1)
        x = x + block.mlp(block.norm2(x)) * gate_mlp.unsqueeze(1)
    return x


def _dit_call(self, *args, **kwargs):
    """Pads the prompt rows to the bucket before the compiled DiT runs."""
    latents = kwargs.get("hidden_states")
    text = kwargs.get("encoder_hidden_states")
    mask = kwargs.get("encoder_attention_mask")
    if latents is None or latents.device.type not in _DEVICE_TYPES:
        return torch.nn.Module.__call__(self, *args, **kwargs)
    # The blocks are compiled spmd-safe: the stock forward would hang them.
    if (
        args
        or not isinstance(text, torch.Tensor)
        or text.shape[0] != 1
        or kwargs.get("pooled_projections") is None
    ):
        raise NotImplementedError(
            "HunyuanVideo on TPU runs one prompt per request, with pooled projections"
        )
    real = text.shape[1] if mask is None else int(mask.cpu().sum())
    sp = get_sp_world_size()
    bucket = -(-max(real, 1) // TEXT_ALIGNMENT) * TEXT_ALIGNMENT
    bucket = -(-bucket // sp) * sp
    text = F.pad(text[:, :real], (0, 0, 0, bucket - real))
    kwargs.update(encoder_hidden_states=text, tpu_text_keys=_counts(real, text.device))
    return torch.nn.Module.__call__(self, **kwargs)


@functools.lru_cache(maxsize=8)
def _image_keys(count: int, device) -> torch.Tensor:
    return _counts(count, device)


def dit_forward(
    self,
    hidden_states,
    encoder_hidden_states,
    timestep,
    encoder_hidden_states_image=None,
    pooled_projections=None,
    guidance=None,
    **kwargs,
):
    """``HunyuanVideoTransformer3DModel.forward`` over a bucketed prompt."""
    text_keys = kwargs.pop("tpu_text_keys", None)
    if text_keys is None:
        return _STOCK_FORWARD(
            self,
            hidden_states,
            encoder_hidden_states,
            timestep,
            encoder_hidden_states_image,
            pooled_projections,
            guidance,
            **kwargs,
        )
    if guidance is None:
        guidance = torch.tensor(
            [6016.0], device=hidden_states.device, dtype=hidden_states.dtype
        )
    x, t = hidden_states, timestep
    _, _, ot, oh, ow = x.shape  # codespell:ignore
    tt, th, tw = (
        ot // self.patch_size[0],  # codespell:ignore
        oh // self.patch_size[1],
        ow // self.patch_size[2],
    )
    sp = get_sp_world_size()
    freqs_cos, freqs_sin = get_rotary_pos_embed(
        (tt * sp, th, tw),
        self.hidden_size,
        self.num_attention_heads,
        self.rope_dim_list,
        self.rope_theta,
    )
    freqs = (freqs_cos.to(x.device), freqs_sin.to(x.device))
    vec = self.time_in(t) + self.vector_in(pooled_projections)
    if self.guidance_in and guidance is not None:
        vec = vec + self.guidance_in(guidance)
    img = self.img_in(x)
    txt = _refine(self.txt_in, encoder_hidden_states, t, text_keys)
    text_rows = txt.shape[1] // sp
    if sp > 1:
        # A fresh tensor: every rank's region inputs then start at offset 0.
        txt = txt.narrow(1, get_sp_parallel_rank() * text_rows, text_rows).clone()
    image_rows = img.shape[1]
    segments = Segments(
        [(image_rows, _image_keys(image_rows * sp, x.device)), (text_rows, text_keys)]
    )
    for block in self.double_blocks:
        img, txt = block(img, txt, vec, freqs, True, segments)
    x = torch.cat((img, txt), 1)
    for block in self.single_blocks:
        x = block(x, vec, text_rows, freqs, True, segments)
    img = self.final_layer(x[:, :image_rows], vec)
    return dit.unpatchify(img, tt, th, tw, self.patch_size, self.out_channels)


def _randn(shape, generator=None, device=None, dtype=None, **kwargs):
    """HunyuanVideo's initial latents on TPU, drawn in float32 and then cast."""
    if (
        dtype in (None, torch.float32)
        or device is None
        or torch.device(device).type not in _DEVICE_TYPES
        or not isinstance(
            get_global_server_args().pipeline_config, config.HunyuanConfig
        )
    ):
        return _STOCK_RANDN(
            shape, generator=generator, device=device, dtype=dtype, **kwargs
        )
    noise = _STOCK_RANDN(
        shape, generator=generator, device=device, dtype=torch.float32, **kwargs
    )
    return noise.to(dtype)


def install() -> None:
    model = dit.HunyuanVideoTransformer3DModel
    model._compile_conditions = [is_double_block, is_single_block]
    model._compile_regions_spmd_safe = True
    latent_preparation.randn_tensor = _randn
    config.HunyuanConfig.tokenize_prompt = _padded_tokenize
    model.forward = dit_forward
    model.__call__ = _dit_call
    USPAttention.forward = usp_forward
