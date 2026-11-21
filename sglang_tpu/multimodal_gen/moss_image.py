# SPDX-License-Identifier: Apache-2.0
"""MOSS-Image on TPU workers: one compiled DiT block per joint-row bucket.

Runtime patch, tracked debt:
- Why: torch_tpu compiles static shapes. The MOSS-Image DiT attends
  [image | prompt] at the prompt's exact token count (the image RoPE carries it
  too), and the negative prompt has its own, so every new prompt compiles the
  blocks again; the text refiner and the Qwen3.5 encoder run eagerly at that
  count, which compiles their ops per count. The DiT declares no compile
  regions.
- How: :func:`install`
  (1) right-pads the encoder's prompt tokens to a multiple of 256 on the host:
      the encoder is causal, so real tokens never see the padding;
  (2) pads the DiT's joint rows to a multiple of 256, prompt rows zero past the
      real tokens. The refiner and the blocks attend only the real keys, a count
      read on device (the live-length kernel, else a masked float32 SDPA), so
      prompts of one bucket, positive and negative, share one block program;
  (3) builds the RoPE (stock code) and timestep tables on the host per request;
      each Ulysses rank takes its share of the rows;
  (4) compiles the 42 joint blocks (``--regional-compile``), spmd-safe;
  (5) decodes a batch one image at a time: torch_tpu's group norm cannot view
      the FLUX.2 decoder's batched channels-last activations.
- Upstream PR or issue: none.
- Remove when: the DiT takes a platform-provided joint-row bucket and real-key
  count and builds its tables from host values, and torch_tpu's group norm
  takes channels-last inputs.
"""

from __future__ import annotations

import functools

import msgspec
import torch
import torch.nn.functional as F

from sglang.multimodal_gen.configs.models.fsdp import is_layer
from sglang.multimodal_gen.runtime.cache.conditioning import cached_encoder_call
from sglang.multimodal_gen.runtime.distributed.communication_op import (
    sequence_model_parallel_all_gather,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_ring_parallel_world_size,
    get_sp_parallel_rank,
    get_sp_world_size,
)
from sglang.multimodal_gen.runtime.layers.usp import (
    _usp_input_all_to_all,
    _usp_output_all_to_all,
)
from sglang.multimodal_gen.runtime.managers.forward_context import (
    get_forward_context,
    set_forward_context,
)
from sglang.multimodal_gen.runtime.models.dits import moss_image as dit
from sglang.multimodal_gen.runtime.pipelines import moss_image as pipeline
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages import (
    moss_image as stages,
)
from sglang_tpu.multimodal_gen import tpu_splash_attn

# Prompts up to 256 tokens share one 4352-row program at 1024x1024 (4096 image
# rows); 256 is a multiple of every Ulysses degree of a host.
JOINT_ROW_ALIGNMENT = 256
PROMPT_TOKEN_ALIGNMENT = 256
# Devices whose inputs take the padded paths (tests add "cpu").
_DEVICE_TYPES = ("tpu",)

_STOCK_ATTENTION = dit._attention
_STOCK_FORWARD = dit.MOSSImageTransformer.forward
_STOCK_PREPARE = dit.MOSSImageTransformer.prepare_conditioning
_STOCK_ENCODE = stages.MOSSImageBeforeDenoisingStage._encode_prompt
_STOCK_BEFORE_LOOP = stages.MOSSImageDenoisingStage._before_denoising_loop
_STOCK_CREATE_STAGES = pipeline.MOSSImagePipeline.create_pipeline_stages


def _round_up(value: int, multiple: int) -> int:
    return -(-value // multiple) * multiple


class TpuGroup(dit._ConditioningGroup, frozen=True):
    lengths: tuple[int, ...] = ()  # real prompt tokens per sample
    live: torch.Tensor | None = None  # int32 [B]: image rows + real prompt tokens


class TpuConditioning(dit.MOSSImageConditioning, frozen=True):
    rows: int = 0  # image rows + padded prompt rows
    time_embeddings: tuple[torch.Tensor, ...] = ()  # [1, freq_dim] per step


@functools.lru_cache(maxsize=64)
def _int32(values: tuple[int, ...], device: torch.device) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.int32).to(device)


def _live_attention(q, k, v, live, scale):
    """``[B, S, H, D]`` attention over keys ``[0, live[b])``; ``live`` int32 on
    device, so one program serves every count. Padded rows' output is dropped."""
    batch, seq, heads, head_dim = q.shape
    if (
        q.device.type == "tpu"
        and seq >= 512
        and tpu_splash_attn._kernel_input_fallback_reason(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        )
        is None
    ):
        tpu_splash_attn._register_ops()
        if tpu_splash_attn._FLASH_LIVE_LENGTH_OP is not None:
            outs = [
                tpu_splash_attn._FLASH_LIVE_LENGTH_OP(
                    *(t[b].reshape(seq, heads * head_dim) for t in (q, k, v)),
                    live[b : b + 1],
                    scale,
                    heads,
                ).view(seq, heads, head_dim)
                for b in range(batch)
            ]
            return torch.stack(outs) if batch > 1 else outs[0][None]
    valid = torch.arange(seq, device=q.device) < live[:, None]
    # Zero padded keys and values: a masked score still adds q . k, and a zero
    # probability times a NaN value is NaN. float32 as tpu_attention's SDPA.
    k, v = (torch.where(valid[:, :, None, None], t, 0) for t in (k, v))
    out = F.scaled_dot_product_attention(
        *(t.transpose(1, 2).float() for t in (q, k, v)),
        attn_mask=valid[:, None, None],
        scale=scale,
    )
    return out.to(q.dtype).transpose(1, 2)


def _attention(q, k, v, rope, heads, attention, attn_mask_meta=None):
    """``dit._attention``; with ``{"tpu_live", "tpu_sp"}`` metadata, over the
    real keys, the rows this rank's share (``tpu_sp``) or all of them."""
    if not (isinstance(attn_mask_meta, dict) and "tpu_live" in attn_mask_meta):
        return _STOCK_ATTENTION(q, k, v, rope, heads, attention, attn_mask_meta)
    shape = (*q.shape[:2], heads, q.shape[-1] // heads)
    q, k, v = q.reshape(shape), k.reshape(shape), v.reshape(shape)
    if rope is not None:
        q, k = dit._rotate(q, rope), dit._rotate(k, rope)
    q, k = q.to(v.dtype), k.to(v.dtype)
    sp = attn_mask_meta["tpu_sp"]
    if sp:
        q, k, v = (_usp_input_all_to_all(t, head_dim=2) for t in (q, k, v))
    out = _live_attention(q, k, v, attn_mask_meta["tpu_live"], attention.softmax_scale)
    if sp:
        out = _usp_output_all_to_all(out, head_dim=2)
    return out.flatten(2)


def _refine_text(projection, context, rope, live):
    """``_TextProjection.forward`` attending the first ``live`` prompt rows."""
    meta = {"tpu_live": live, "tpu_sp": False}
    x = projection.proj(context.to(projection.proj.weight.dtype))
    for block in projection.blocks:
        q, k, v = block.qkv(block.norm1(x)).chunk(3, -1)
        q = block.q_norm(q.unflatten(-1, (block.heads, -1))).flatten(2)
        k = block.k_norm(k.unflatten(-1, (block.heads, -1))).flatten(2)
        x = x + block.proj(
            _attention(q, k, v, rope, block.heads, block.attention, meta)
        )
        y = block.norm2(x)
        x = x + block.ffn_down(dit._situ(block.ffn_up(y), block.ffn_gate(y)))
    return x


@functools.lru_cache(maxsize=32)
def _rope_tables(model, height, width, length, rows, dtype, sp, rank, device):
    """The stock prepared RoPE of one prompt length, built on the host: this
    rank's (sin, cos) joint rows and the prompt rows, zero past the real ones."""
    image = height * width
    # A host copy: the model's frequency buffer lives on the device.
    rope = dit._RoPE(model.config.dim // model.config.num_heads, model.config.rope_base)
    sin, cos = dit._prepare_rope(
        rope(height, width, length, torch.device("cpu")).to(dtype)
    )
    joint = tuple(F.pad(t, (0, 0, 0, rows - image - length)) for t in (sin, cos))
    local = rows // sp
    shard = tuple(
        t[:, rank * local : (rank + 1) * local].contiguous().to(device) for t in joint
    )
    return shard, tuple(t[:, image:].contiguous().to(device) for t in joint)


def _prompt_lengths(states, mask):
    """Real tokens per sample of a binary prefix mask, read on the host."""
    batch, seq = states.shape[:2]
    if mask is None:
        return [seq] * batch
    mask = mask.cpu()
    if mask.shape != (batch, seq):
        return None
    lengths = (mask != 0).sum(-1)
    prefix = torch.arange(seq) < lengths[:, None]
    if not torch.equal(mask, prefix.to(mask.dtype)):
        return None
    return lengths.tolist()


def prepare_conditioning(
    self,
    encoder_hidden_states,
    encoder_attention_mask,
    height,
    width,
    dtype,
    *,
    batch_samples=False,
):
    """``prepare_conditioning`` on prompt rows padded to one joint bucket.

    The bucket fits the longest prompt: with ``batch_samples`` all samples run
    in one group (each attending its own real keys), else one per sample. Off
    TPU, with a ring, or for a non-prefix mask, the stock path runs.
    """
    device = encoder_hidden_states.device
    lengths = None
    if (
        device.type in _DEVICE_TYPES
        and encoder_hidden_states.ndim == 3
        and encoder_hidden_states.shape[-1] == self.config.text_dim
        and get_ring_parallel_world_size() == 1
    ):
        lengths = _prompt_lengths(encoder_hidden_states, encoder_attention_mask)
    if lengths is None:
        return _STOCK_PREPARE(
            self,
            encoder_hidden_states,
            encoder_attention_mask,
            height,
            width,
            dtype,
            batch_samples=batch_samples,
        )
    if min(lengths) == 0:
        raise ValueError("Each sample needs at least one valid text token.")
    sp = get_sp_world_size() if self._sequence_parallel else 1
    rank = get_sp_parallel_rank() if sp > 1 else 0
    image = height * width
    rows = _round_up(_round_up(image + max(lengths), JOINT_ROW_ALIGNMENT), sp)
    text_rows, seq = rows - image, encoder_hidden_states.shape[1]
    states = F.pad(encoder_hidden_states, (0, 0, 0, max(text_rows - seq, 0)))
    texts, ropes = [], []
    for index, length in enumerate(lengths):
        keep = (torch.arange(text_rows) < length)[None, :, None].to(device)
        shard, text_rope = _rope_tables(
            self, height, width, length, rows, dtype, sp, rank, device
        )
        # As stock, the refiner runs per sample.
        context = torch.where(keep, states[index : index + 1, :text_rows], 0)
        text = _refine_text(
            self.text_embedder, context, text_rope, _int32((length,), device)
        )
        texts.append(torch.where(keep, text, 0))
        ropes.append(shard)
    batch = len(lengths)
    members = [list(range(batch))] if batch_samples else [[i] for i in range(batch)]
    groups = tuple(
        TpuGroup(
            indices=torch.as_tensor(idx).to(device) if len(members) > 1 else None,
            text=torch.cat([texts[i] for i in idx]),
            rope=tuple(torch.cat([ropes[i][part] for i in idx]) for part in (0, 1)),
            lengths=tuple(lengths[i] for i in idx),
            live=_int32(tuple(image + lengths[i] for i in idx), device),
        )
        for idx in members
    )
    return TpuConditioning(spatial_shape=(height, width), groups=groups, rows=rows)


def _time_embeddings(timesteps, freq_dim, device):
    """The stock float64 sinusoid of each step's timestep, on the host; one
    tensor per step (a slice's storage offset would key a new executable)."""
    half = freq_dim // 2
    freq = 10000 ** (-torch.arange(half, dtype=torch.float64) / half)
    phase = timesteps.float().double()[:, None] * freq
    table = torch.cat((phase.cos(), phase.sin()), -1).float()
    return tuple(row[None].to(device) for row in table)


def forward(
    self,
    hidden_states,
    timestep,
    encoder_hidden_states,
    encoder_attention_mask=None,
    prepared_conditioning=None,
    skip_layers=(),
    fuse_qkv=False,
):
    """``MOSSImageTransformer.forward`` on padded joint rows (see :func:`prepare_conditioning`)."""
    if prepared_conditioning is None and hidden_states.device.type in _DEVICE_TYPES:
        prepared_conditioning = self.prepare_conditioning(
            encoder_hidden_states,
            encoder_attention_mask,
            *hidden_states.shape[-2:],
            self.video_embedder.weight.dtype,
        )
    if not isinstance(prepared_conditioning, TpuConditioning):
        return _STOCK_FORWARD(
            self,
            hidden_states,
            timestep,
            encoder_hidden_states,
            encoder_attention_mask,
            prepared_conditioning,
            skip_layers,
            fuse_qkv,
        )
    skip_layers = dit.validate_skip_layers(skip_layers, self.config.num_layers)
    batch, _, height, width = hidden_states.shape
    conditioning = prepared_conditioning
    if (conditioning.spatial_shape, conditioning.batch_size) != (
        (height, width),
        batch,
    ):
        raise ValueError("Prepared conditioning must match the latent batch and size.")
    step, table = get_forward_context().current_timestep, conditioning.time_embeddings
    if isinstance(step, int) and 0 <= step < len(table):
        embedding = table[step].expand(batch, -1)
    else:
        times = torch.as_tensor(timestep, dtype=torch.float32).reshape(-1).cpu()
        embedding = torch.cat(
            _time_embeddings(
                times.expand(batch), self.config.freq_dim, hidden_states.device
            )
        )
    time = self.time_embedding(
        embedding.to(self.time_embedding[0].weight.dtype)
    ).float()
    mod = self.shared_block_modulation(
        time.to(self.shared_block_modulation.weight.dtype)
    ).view(batch, 6, -1)
    head = self.video_head_modulation(
        time.to(self.video_head_modulation.weight.dtype)
    ).view(batch, 2, -1)
    sp = get_sp_world_size() if self._sequence_parallel else 1
    groups = conditioning.groups
    outputs = None
    if len(groups) > 1:
        outputs = torch.empty_like(hidden_states, dtype=torch.float32)
    for group in groups:
        tensors = (hidden_states, mod, head)
        if group.indices is not None:
            tensors = (t.index_select(0, group.indices) for t in tensors)
        group_video, group_mod, group_head = tensors
        group_video = self.video_embedder(
            group_video.flatten(2)
            .transpose(1, 2)
            .to(self.video_embedder.weight.dtype)
            .contiguous()
        )
        x = torch.cat((group_video, group.text), 1)
        if sp > 1:
            # A fresh shard: compiled programs guard storage offsets, and every
            # rank must run the same program.
            x = x.unflatten(1, (sp, -1))[:, get_sp_parallel_rank()].clone(
                memory_format=torch.contiguous_format
            )
        meta = {"tpu_live": group.live, "tpu_sp": sp > 1}
        for index, layer in enumerate(self.layers):
            if index not in skip_layers:
                x = layer(
                    x, group.rope, group_mod, attn_mask_meta=meta, fuse_qkv=fuse_qkv
                )
        if sp > 1:
            x = sequence_model_parallel_all_gather(x.contiguous(), dim=1)
        x = self.final_norm_video(
            x[:, : height * width].to(self.final_norm_video.weight.dtype)
        )
        x = x * (1 + group_head[:, 1, None]) + group_head[:, 0, None]
        x = self.final_linear_video(x.to(self.final_linear_video.weight.dtype))
        sample = (
            x.transpose(1, 2)
            .reshape(x.shape[0], -1, height, width)
            .to(dtype=torch.float32, memory_format=torch.contiguous_format)
        )
        if outputs is None:
            return sample
        outputs.index_copy_(0, group.indices, sample)
    return outputs


def _before_denoising_loop(self, ctx, batch, server_args):
    """The stock hook, plus the request's timestep table."""
    _STOCK_BEFORE_LOOP(self, ctx, batch, server_args)
    branches = ctx.cfg_policy.branches if ctx.cfg_policy is not None else ()
    table = None
    for branch in branches:
        conditioning = branch.kwargs.get("prepared_conditioning")
        if isinstance(conditioning, TpuConditioning):
            if table is None:
                table = _time_embeddings(
                    ctx.timesteps.cpu(),
                    self.transformer.config.freq_dim,
                    ctx.latents.device,
                )
            branch.kwargs["prepared_conditioning"] = msgspec.structs.replace(
                conditioning, time_embeddings=table
            )


@torch.no_grad()
def _encode_prompt(self, prompts, device, max_sequence_length=4096):
    """The stock encoding of prompt tokens right-padded to a multiple of 256.

    Returns embeddings zero past each prompt's tokens and their prefix mask,
    both built from host lengths: a new prompt in a known bucket compiles
    nothing.
    """
    if torch.device(device).type not in _DEVICE_TYPES:
        return _STOCK_ENCODE(self, prompts, device, max_sequence_length)
    # Any id: the real tokens never attend the padding.
    pad_id = getattr(self.tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(self.tokenizer, "eos_token_id", None) or 0
    rows, lengths = [], []
    for prompt in prompts:
        formatted = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        inputs = self.tokenizer(
            formatted,
            return_tensors="pt",
            truncation=True,
            max_length=max_sequence_length,
        )
        length = inputs["input_ids"].shape[1]
        bucket = _round_up(length, PROMPT_TOKEN_ALIGNMENT)

        def encode_row(inputs=inputs, length=length, bucket=bucket):
            self.begin_declared_component_use(
                component_name="text_encoder", module=self.text_encoder
            )
            ids = F.pad(inputs["input_ids"], (0, bucket - length), value=pad_id)
            # No attention mask: every token attends causally.
            with set_forward_context(0, None):
                output = self.text_encoder(
                    input_ids=ids.to(device), use_cache=False, return_dict=True
                )
            keep = (torch.arange(bucket) < length)[:, None].to(device)
            return torch.where(keep, output.last_hidden_state[0], 0)

        rows.append(
            cached_encoder_call(
                self.text_encoder,
                (dict(inputs),),
                {"device": str(device), "tpu_token_bucket": bucket},
                encode_row,
                namespace=self,
                nested=False,
                share_in_group=True,
            )
        )
        lengths.append(length)
    self.finish_unused_declared_component(
        component_name="text_encoder", module=self.text_encoder
    )
    width = max(row.shape[0] for row in rows)
    embeds = torch.stack([F.pad(row, (0, 0, 0, width - row.shape[0])) for row in rows])
    mask = torch.arange(width)[None] < torch.tensor(lengths)[:, None]
    return embeds, mask.to(device)


def create_pipeline_stages(self, server_args):
    """The stock stages, the VAE decoding a batch one image at a time."""
    _STOCK_CREATE_STAGES(self, server_args)
    vae = self.get_module("vae")
    if hasattr(vae, "use_slicing"):
        vae.use_slicing = True


def install() -> None:
    dit._attention = _attention
    dit.MOSSImageTransformer.prepare_conditioning = prepare_conditioning
    dit.MOSSImageTransformer.forward = forward
    # Only the joint blocks: the refiner runs once per request, eagerly.
    dit.MOSSImageTransformer._compile_conditions = [is_layer]
    # Every rank runs the same block program on its share of the rows.
    dit.MOSSImageTransformer._compile_regions_spmd_safe = True
    stages.MOSSImageDenoisingStage._before_denoising_loop = _before_denoising_loop
    stages.MOSSImageBeforeDenoisingStage._encode_prompt = _encode_prompt
    pipeline.MOSSImagePipeline.create_pipeline_stages = create_pipeline_stages
