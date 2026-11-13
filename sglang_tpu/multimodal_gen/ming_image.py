# SPDX-License-Identifier: Apache-2.0
"""Ming-Image on TPU workers: a compiled MoE encoder and one compiled DiT per size.

Runtime patch, tracked debt:
- Why: the encoder's MoE runs a Triton kernel, and its routing, image-token
  mask and ``masked_scatter`` compute on int64 tensors on the device, which
  torch_tpu aborts on or lacks; run eagerly it would also compile every op per
  prompt length. The DiT's caption length is the prompt's, so the TPU default
  (the whole DiT compiles as one module) compiles it again for every prompt,
  and its masks and SP slices are Python ints of that length.
- How: :func:`install`
  (1) runs the encoder over prompt tokens right-padded to a multiple of 256
      (causal: real tokens never see the padding), with the image rows and
      their embeddings placed by host indices, and slices the real rows by
      host lengths;
  (2) compiles each encoder block whole, so routing stays in one XLA program,
      and runs the experts through sglang_tpu's Pallas ``fused_moe_call``
      (per-expert outputs, combined in float32 as the stock does), with expert
      weights moved once to its ``(experts, in, out)`` layout;
  (3) pads the DiT's caption to a multiple of 512 outside the compiled call.
      The stock RoPE positions are built on the host, and the real image and
      caption key counts go in as device int32. The joint and caption
      attentions read each segment's real keys as resumable kernel steps,
      with zero heads appended where Ulysses needs them (30 heads at 8).
- Upstream PR or issue: none.
- Remove when: the encoder has a platform MoE path and host-built masks, and
  the DiT takes a platform caption bucket with key counts as data.
"""

from __future__ import annotations

import functools

import torch
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_ring_parallel_world_size,
    get_sp_parallel_rank,
    get_sp_world_size,
)
from sglang.multimodal_gen.runtime.distributed.sp_shard_utils import gather_seq
from sglang.multimodal_gen.runtime.layers.attention.layer import USPAttention
from sglang.multimodal_gen.runtime.layers.usp import (
    _usp_input_all_to_all,
    _usp_output_all_to_all,
)
from sglang.multimodal_gen.runtime.models.dits import ming_image as dit
from sglang.multimodal_gen.runtime.models.dits.zimage import RopeEmbedder
from sglang.multimodal_gen.runtime.models.encoders import ming_image as encoder
from sglang.multimodal_gen.runtime.models.encoders.base import BaseEncoderOutput
from sglang.multimodal_gen.runtime.utils.torch_compile import (
    build_torch_compile_kwargs,
)
from sglang_tpu.multimodal_gen import tpu_splash_attn

PROMPT_TOKEN_ALIGNMENT = 256
# The kernel's caption step needs >= 512 keys.
CAPTION_ALIGNMENT = 512
# Devices that take these paths.
_DEVICE_TYPES = ("tpu",)

_STOCK_ENCODER_FORWARD = encoder.MingImageEncoder.forward
_STOCK_BLOCK_FORWARD = encoder.MingEncoderBlock.forward
_STOCK_FUSED_EXPERTS = encoder.fused_experts
_STOCK_DIT_FORWARD = dit.MingImageTransformer2DModel.forward
_STOCK_USP_FORWARD = USPAttention.forward


# --- Encoder ---------------------------------------------------------------


def _fused_experts(hidden_states, w13, w2, topk_output, config):
    """``fused_experts`` through the Pallas fused MoE: the same contract."""
    if hidden_states.device.type not in _DEVICE_TYPES:
        return _STOCK_FUSED_EXPERTS(hidden_states, w13, w2, topk_output, config)
    if w13.shape[1] != hidden_states.shape[-1]:
        raise RuntimeError("Ming-Image experts are not in the TPU layout")
    from sglang_tpu.moe.pallas.fused_moe_op import fused_moe_call

    return fused_moe_call(
        hidden_states=hidden_states,
        w1=w13,
        w2=w2,
        topk_weights=topk_output.topk_weights.to(hidden_states.dtype),
        topk_ids=topk_output.topk_ids,
        topk=config.top_k,
        # silu(gate) * up on the float32 accumulator, then rounded once.
        activation="silu",
        no_combine=config.no_combine,
    )


def _to_tpu_layout(model) -> None:
    """Moves expert weights once to the fused MoE's (experts, in, out) layout."""
    for module in model.modules():
        if isinstance(module, encoder.MingExperts) and not getattr(
            module, "_tpu_layout", False
        ):
            experts = module.experts
            experts.w13_weight.data = experts.w13_weight.data.transpose(
                1, 2
            ).contiguous()
            experts.w2.data = experts.w2.data.transpose(1, 2).contiguous()
            module._tpu_layout = True


@functools.lru_cache(maxsize=1)
def _compiled_block():
    from sglang.kernels.fused_op import get_fused_op_backend
    from sglang_tpu.moe.pallas.fused_moe_op import _build_fused_moe_op

    _build_fused_moe_op()  # register the op before tracing
    # Resolve the fused ops' lazy backend switch: a trace taken before it
    # resolves compiles the first block again.
    get_fused_op_backend()
    kwargs = build_torch_compile_kwargs(mode=None)
    kwargs.update(fullgraph=True, dynamic=False)
    return torch.compile(_STOCK_BLOCK_FORWARD, **kwargs)


def _tpu_metadata():
    from sglang_tpu.pallas_export import tpu_info_for_cpu_export

    return tpu_info_for_cpu_export()


def block_forward(self, x, positions, image_mask=None):
    if x.device.type not in _DEVICE_TYPES:
        return _STOCK_BLOCK_FORWARD(self, x, positions, image_mask)
    # The diffusion launcher runs JAX on the CPU; the fused MoE reads the TPU's
    # metadata while it is exported, as the splash kernels do.
    with _tpu_metadata():
        return _compiled_block()(self, x, positions, image_mask)


def encoder_forward(
    self, input_ids, position_ids, pixel_values=None, image_grid_thw=None, **kwargs
):
    """``MingImageEncoder.forward`` over right-padded tokens and host indices."""
    device = input_ids.device
    if device.type not in _DEVICE_TYPES:
        return _STOCK_ENCODER_FORWARD(
            self, input_ids, position_ids, pixel_values, image_grid_thw, **kwargs
        )
    _to_tpu_layout(self)
    ids = input_ids.cpu()
    length = ids.shape[1]
    padded = -(-length // PROMPT_TOKEN_ALIGNMENT) * PROMPT_TOKEN_ALIGNMENT
    image_rows = F.pad(ids[0] == self.image_token, (0, padded - length))
    # Row i of the image embeddings goes to the i-th image token, as masked_scatter.
    take = (image_rows.cumsum(0) - 1).clamp(min=0) * image_rows
    queries = torch.cat(list(self.query_tokens_dict.values()), 0)
    query_count = queries.shape[0]
    image_embeddings = queries
    if pixel_values is not None:
        vision_features = self.vision(pixel_values, image_grid_thw)
        with torch.autocast(device.type, enabled=False):
            image_embeddings = F.normalize(self.linear_proj(vision_features), dim=-1)
        image_embeddings = torch.cat((image_embeddings, queries), 0)
    tokens = F.pad(ids[0], (0, padded - length)).to(torch.int32).to(device)
    x = self.word_embeddings(tokens)[None]
    image_mask = image_rows.to(device)[None]
    rows = image_embeddings.to(x.dtype).index_select(0, take.to(torch.int32).to(device))
    x = torch.where(image_mask[..., None], rows[None], x)
    positions = F.pad(position_ids.cpu(), (0, padded - length)).to(torch.int32)
    positions = positions.to(device)
    selected = []
    for i, layer in enumerate(self.layers):
        if i in self.selected_layers:
            selected.append(x)
        x = layer(x, positions, image_mask)
    x = self.norm(x)
    if len(self.layers) in self.selected_layers:
        selected.append(x)
    prompt_length = length - query_count - 2
    direct = self.proj_directvlm(
        torch.cat([hidden[:, :prompt_length] for hidden in selected], -1)
    )
    queries = self.proj_in(x[:, length - query_count - 1 : length - 1])
    query_positions = torch.arange(query_count, dtype=torch.int32)[None].to(device)
    for layer in self.connector:
        queries = layer(queries, query_positions)
    queries = self.proj_out(self.connector_norm(queries))
    if self.config.projection_config["connector_norm"]:
        queries = F.normalize(queries, dim=-1)
    return BaseEncoderOutput(last_hidden_state=queries, hidden_states=(direct,))


# --- DiT -------------------------------------------------------------------


def _counts(real: int, device) -> torch.Tensor:
    return torch.tensor([0, real], dtype=torch.int32).to(device)


@functools.lru_cache(maxsize=16)
def _layout(rope, shape, device):
    """RoPE tables and real key counts of one request, built on the host.

    ``rope`` is ``(theta, axes_dims, axes_lens)``. Returns the image rows'
    and the caption rows' tables (the refiners'), this rank's
    [image share | caption share] tables (the joint layers'), and the counts.
    """
    frames, hp, wp, cap_length, bucket, sp, rank = shape
    rotary = RopeEmbedder(*rope)
    n_img = frames * hp * wp
    x_target = -(-n_img // 32) * 32
    # The stock ids: caption rows at (1.., 0, 0); the image after the stock
    # caption target, which is the caption rounded up to 32.
    cap_target = -(-cap_length // 32) * 32
    grid = dit.MingImageTransformer2DModel.create_coordinate_grid
    cap_ids = F.pad(
        grid((cap_target, 1, 1), (1, 0, 0)).flatten(0, 2),
        (0, 0, 0, bucket - cap_target),
    )
    x_ids = grid((frames, hp, wp), (cap_target + 1, 0, 0)).flatten(0, 2)
    x_freqs = rotary(F.pad(x_ids, (0, 0, 0, x_target - n_img)))
    cap_freqs = rotary(cap_ids)
    image_rows, cap_rows = -(-n_img // sp), bucket // sp

    def local(image, caption):
        image = F.pad(image[:n_img], (0, 0, 0, image_rows * sp - n_img))
        return torch.cat(
            (
                image.narrow(0, rank * image_rows, image_rows),
                caption.narrow(0, rank * cap_rows, cap_rows),
            )
        )

    joint = tuple(local(i, c) for i, c in zip(x_freqs, cap_freqs))
    to = functools.partial(torch.Tensor.to, device=device)
    return (
        tuple(map(to, x_freqs)),
        tuple(map(to, cap_freqs)),
        tuple(map(to, joint)),
        (_counts(n_img, device), _counts(cap_length, device)),
    )


def _dit_call(self, *args, **kwargs):
    """Pads the caption to the bucket before the compiled DiT runs."""
    latents = kwargs.get("hidden_states")
    text, direct = kwargs.get("encoder_hidden_states"), kwargs.get("direct_embeddings")
    if (
        args
        or latents is None
        or text is None
        or direct is None
        or latents.device.type not in _DEVICE_TYPES
        or self.learned_padding
    ):
        return torch.nn.Module.__call__(self, *args, **kwargs)
    cap_length = text.shape[1] + direct.shape[1]
    sp = get_sp_world_size()
    bucket = -(-cap_length // CAPTION_ALIGNMENT) * CAPTION_ALIGNMENT
    bucket = -(-bucket // sp) * sp
    frames = latents.shape[2]
    reference = kwargs.get("reference_latents")
    if reference is not None:
        frames += reference.shape[2]
    rank = get_sp_parallel_rank() if sp > 1 else 0
    rope = (self.rotary_emb.theta, tuple(self.axes_dims), tuple(self.axes_lens))
    shape = (
        frames,
        latents.shape[3] // 2,
        latents.shape[4] // 2,
        cap_length,
        bucket,
        sp,
        rank,
    )
    kwargs.update(
        # Zero rows, as the stock zero-masked padding.
        direct_embeddings=F.pad(direct, (0, 0, 0, bucket - cap_length)),
        tpu_layout=_layout(rope, shape, latents.device),
    )
    return torch.nn.Module.__call__(self, **kwargs)


def dit_forward(
    self,
    hidden_states,
    encoder_hidden_states,
    timestep,
    direct_embeddings,
    reference_latents=None,
    **kwargs,
):
    """``MingImageTransformer2DModel.forward`` over a bucketed caption."""
    layout = kwargs.pop("tpu_layout", None)
    if layout is None:
        return _STOCK_DIT_FORWARD(
            self,
            hidden_states,
            encoder_hidden_states,
            timestep,
            direct_embeddings,
            reference_latents,
            **kwargs,
        )
    x_freqs, cap_freqs, joint_freqs, (image_keys, cap_keys) = layout
    x = hidden_states
    batch_size, channels, output_frames, height, width = x.shape
    if reference_latents is not None:
        x = torch.cat((x, reference_latents), 2)
    frames = x.shape[2]
    # The stock patchify, embedding and image padding.
    x = x.reshape(batch_size, channels, frames, height // 2, 2, width // 2, 2)
    x = x.permute(0, 2, 3, 5, 4, 6, 1).flatten(1, 3).flatten(2)
    x_length = x.shape[1]
    x_target = -(-x_length // 32) * 32
    x = F.pad(x, (0, 0, 0, x_target - x_length))
    x, _ = self.all_x_embedder["2-1"](x)
    x = self._replace_padding_with_token_mask(
        x,
        (torch.arange(x_target, device=x.device) < x_length).unsqueeze(0),
        self.x_pad_token,
    )
    cap, _ = self.cap_embedder(encoder_hidden_states)
    cap = torch.cat((cap, direct_embeddings), 1)
    x_mask, x_meta = self._get_attn_mask_and_meta(
        "_ming_x_mask", [x_length] * batch_size, x_target, x.device
    )
    adaln = self.t_embedder(((1000.0 - timestep) / 1000.0) * self.t_scale).to(x.dtype)
    for layer in self.noise_refiner:
        x = layer(
            x,
            x_freqs,
            adaln,
            attn_mask=x_mask,
            attn_mask_meta=x_meta,
            skip_sequence_parallel_override=True,
        )
    cap_meta = {"tpu_segments": [(cap.shape[1], cap_keys)]}
    for layer in self.context_refiner:
        cap = layer(cap, cap_freqs, attn_mask_meta=cap_meta)
    sp = get_sp_world_size()
    image_rows, cap_rows = -(-x_length // sp), cap.shape[1] // sp
    x = F.pad(x[:, :x_length], (0, 0, 0, image_rows * sp - x_length))
    if sp > 1:
        # Each rank's share of the image (padded at its tail) and of the caption.
        rank = get_sp_parallel_rank()
        x = x.narrow(1, rank * image_rows, image_rows)
        cap = cap.narrow(1, rank * cap_rows, cap_rows)
    unified = torch.cat((x, cap), 1)
    meta = {"tpu_segments": [(image_rows, image_keys), (cap_rows, cap_keys)]}
    for layer in self.layers:
        unified = layer(unified, joint_freqs, adaln, attn_mask_meta=meta)
    output = self.all_final_layer["2-1"](unified[:, :image_rows], adaln)
    output = gather_seq(output, x_length)[:, :x_length]
    output = output.reshape(batch_size, frames, height // 2, width // 2, 2, 2, channels)
    output = output.permute(0, 6, 1, 2, 4, 3, 5).reshape(
        batch_size, channels, frames, height, width
    )
    return -output[:, :, :output_frames]


def _uses_kernel(impl, q, keys) -> bool:
    if not getattr(impl, "use_splash", False) or q.device.type != "tpu":
        return False
    # Every step takes the kernel or none does: their carries differ.
    return all(
        tpu_splash_attn._ring_kernel_fallback_reason(q, key, key) is None
        for key in keys
    )


def _segment_attention(usp, q, k, v, segments, local):
    """``[B, S, H, D]`` attention over each segment's real keys.

    ``segments`` lists ``(rows per rank, [0, real keys])`` in each rank's row
    order, every segment padded at its tail.
    """
    sp = 1 if local else get_sp_world_size()
    heads = q.shape[2]
    if sp > 1:
        if get_ring_parallel_world_size() > 1 or usp.sp_attention_mode == "kv_gather":
            raise NotImplementedError("Ming-Image on TPU shards with Ulysses only")
        # Zero heads attend uniformly to zero values and are cut again.
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
    impl = usp.attn_impl
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
    meta = kwargs.get("attn_mask_meta")
    if isinstance(meta, dict) and "tpu_segments" in meta:
        local = self.skip_sequence_parallel or kwargs.get(
            "skip_sequence_parallel_override", False
        )
        return _segment_attention(self, q, k, v, meta["tpu_segments"], local)
    return _STOCK_USP_FORWARD(self, q, k, v, *args, **kwargs)


def install() -> None:
    encoder.fused_experts = _fused_experts
    encoder.MingImageEncoder.forward = encoder_forward
    encoder.MingEncoderBlock.forward = block_forward
    dit.MingImageTransformer2DModel.forward = dit_forward
    dit.MingImageTransformer2DModel.__call__ = _dit_call
    USPAttention.forward = usp_forward
