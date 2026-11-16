# SPDX-License-Identifier: Apache-2.0
"""ERNIE-Image on TPU workers: one compiled program per image size and prompt bucket.

Runtime patch, tracked debt:
- Why: torch_tpu compiles static shapes. The single-stream DiT attends
  [image | prompt] jointly at the prompt's exact row count, and its RoPE ids
  place the image after the prompt, so every prompt length compiles the whole
  DiT again (the TPU default compiles the whole module), and it builds the
  RoPE in float64 on the device. The eager text encoder compiles once per
  prompt length too. Under Ulysses the stock pipeline splits the 4D latents as
  ``[B, S, D]`` tokens and fails, and the DiT skips SP.
- How: :func:`install`
  (1) wraps the DiT's ``__call__`` (batches of one prompt length), outside the
      compiled module: the prompt rows pad with zeros to a multiple of 512,
      then of the SP degree; the stock RoPE cos/sin are built on the host,
      padded rows taking the identity rotation; the real image and prompt key
      counts, and this rank's row indices, go in as device int32 tensors.
      Under Ulysses it gathers the image rows after the call;
  (2) replaces the DiT's ``forward`` on those inputs: the same layers, each
      Ulysses rank taking its share of the image rows (padded at the tail) and
      of the prompt rows;
  (3) replaces ``ErnieImageSelfAttention.forward``: queries attend the real
      image keys, then the real prompt keys, as resumable kernel steps (each
      key run padded at its tail only). These are the same keys, and padded
      keys are never read;
  (4) right-pads the prompt tokens to a multiple of 256. The encoder is causal,
      so real tokens never see the padding, and only real rows reach the DiT;
  (5) keeps the latents whole under SP (the DiT shards its own stream), and
      declares the blocks as spmd-safe regions for ``--regional-compile``;
  (6) runs requests one at a time (a batch takes one prompt length) and
      refuses CFG parallel at load.
- Upstream PR or issue: none.
- Remove when: the DiT takes a platform-provided prompt bucket and key counts
  and shards its own stream under SP.
"""

from __future__ import annotations

import functools

import torch
import torch.nn.functional as F

from sglang.multimodal_gen.configs.models.fsdp import is_layer
from sglang.multimodal_gen.configs.pipeline_configs import ernie_image as config
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_ring_parallel_world_size,
    get_sp_parallel_rank,
    get_sp_world_size,
)
from sglang.multimodal_gen.runtime.distributed.sp_shard_utils import gather_seq
from sglang.multimodal_gen.runtime.layers.usp import (
    _usp_input_all_to_all,
    _usp_output_all_to_all,
)
from sglang.multimodal_gen.runtime.models.dits import ernie_image as dit
from sglang.multimodal_gen.runtime.pipelines import ernie_image as pipeline
from sglang_tpu.multimodal_gen import tpu_splash_attn

# Prompts stay within one bucket; the kernel's prompt step needs >= 512 keys.
TEXT_ALIGNMENT = 512
PROMPT_TOKEN_ALIGNMENT = 256

_STOCK_LOAD_MODULES = pipeline.ErnieImagePipeline.load_modules
_STOCK_TOKENIZE = config.ErnieImagePipelineConfig.tokenize_prompt
_STOCK_FORWARD = dit.ErnieImageTransformer2DModel.forward
_STOCK_ATTENTION_FORWARD = dit.ErnieImageSelfAttention.forward


def _counts(real: int, device) -> torch.Tensor:
    return torch.tensor([0, real], dtype=torch.int32).to(device)


@functools.lru_cache(maxsize=16)
def _layout(pos_embed, shape, dtype, device):
    """This rank's rows, RoPE and the real key counts, built on the host.

    Rows run [image share | prompt share] (see :func:`forward`), indexing
    [image padded to ``sp`` shares | prompt bucket]; the RoPE tables repeat
    them per batch row as the stock tables do, padded rows taking the identity
    rotation.
    """
    hp, wp, text_len, bucket, batch, sp, rank = shape
    n_img = hp * wp
    grid = torch.stack(
        torch.meshgrid(
            torch.arange(hp, dtype=torch.float32),
            torch.arange(wp, dtype=torch.float32),
            indexing="ij",
        ),
        -1,
    ).reshape(-1, 2)
    # The stock ids: image rows at (prompt length, y, x), prompt rows at (t, 0, 0).
    ids = torch.cat(
        (
            torch.cat((torch.full((n_img, 1), float(text_len)), grid), -1),
            F.pad(torch.arange(text_len, dtype=torch.float32)[:, None], (0, 2)),
        )
    )
    tables = dit._precompute_rope_cos_sin(pos_embed(ids[None]), dtype)
    image_rows, text_rows = -(-n_img // sp), bucket // sp
    rows = torch.cat(
        (
            torch.arange(image_rows) + rank * image_rows,
            torch.arange(text_rows) + image_rows * sp + rank * text_rows,
        )
    )

    def local(table, identity):
        image, text = table.split((n_img, text_len))
        image = F.pad(image, (0, 0, 0, image_rows * sp - n_img), value=identity)
        text = F.pad(text, (0, 0, 0, bucket - text_len), value=identity)
        return torch.cat((image, text))[rows].repeat(batch, 1).to(device)

    rope = (local(tables[0], 1.0), local(tables[1], 0.0))
    keys = (_counts(n_img, device), _counts(text_len, device))
    return rope, keys, rows.to(torch.int32).to(device)


def _dit_call(self, *args, **kwargs):
    """Pads the prompt rows to the bucket before the compiled DiT runs.

    Padded rows are zeros with the identity rotation; the attention reads only
    the real keys (``tpu_keys``), and padded rows' outputs are dropped.
    """
    latents, states = kwargs.get("hidden_states"), kwargs.get("encoder_hidden_states")
    if args or latents is None or states is None:
        return torch.nn.Module.__call__(self, *args, **kwargs)
    text = states[0] if isinstance(states, (list, tuple)) else states
    batch, text_len = text.shape[:2]
    if kwargs.get("encoder_hidden_states_mask") is not None or any(
        n != text_len for n in kwargs.get("txt_seq_lens") or ()
    ):
        raise NotImplementedError(
            "ERNIE-Image on TPU batches only prompts of the same length"
        )
    sp = get_sp_world_size()
    bucket = -(-max(text_len, 1) // TEXT_ALIGNMENT) * TEXT_ALIGNMENT
    bucket = -(-bucket // sp) * sp
    _, _, height, width = latents.shape
    p = self.patch_size
    hp, wp = height // p, width // p
    rank = get_sp_parallel_rank() if sp > 1 else 0
    shape = (hp, wp, text_len, bucket, batch, sp, rank)
    rope, keys, rows = _layout(self.pos_embed, shape, latents.dtype, latents.device)
    kwargs.update(
        encoder_hidden_states=F.pad(text, (0, 0, 0, bucket - text_len)),
        encoder_hidden_states_mask=None,
        txt_seq_lens=[bucket] * batch,
        tpu_rope=rope,
        tpu_keys=keys,
        tpu_rows=rows,
    )
    patches = gather_seq(torch.nn.Module.__call__(self, **kwargs), hp * wp)
    # The stock unpatchify.
    out = patches.view(batch, hp, wp, p, p, self.out_channels)
    return out.permute(0, 5, 1, 3, 2, 4).reshape(
        batch, self.out_channels, height, width
    )


def forward(
    self,
    hidden_states,
    encoder_hidden_states,
    timestep,
    encoder_hidden_states_image=None,
    guidance=None,
    encoder_hidden_states_mask=None,
    **kwargs,
):
    """``ErnieImageTransformer2DModel.forward`` over a bucketed prompt (see :func:`_dit_call`).

    Returns this rank's image rows of patches, ``[B, rows, p * p * C]``.
    """
    rope = kwargs.get("tpu_rope")
    if rope is None:
        return _STOCK_FORWARD(
            self,
            hidden_states,
            encoder_hidden_states,
            timestep,
            encoder_hidden_states_image,
            guidance,
            encoder_hidden_states_mask,
            **kwargs,
        )
    dtype = hidden_states.dtype
    image = self.x_embedder["proj"](hidden_states).flatten(2).transpose(1, 2)
    text = encoder_hidden_states
    if isinstance(text, (list, tuple)):
        text = text[0]
    if self.text_proj is not None:
        text = self.text_proj(text)
    sp = get_sp_world_size()
    image_rows, text_rows = -(-image.shape[1] // sp), text.shape[1] // sp
    image = F.pad(image, (0, 0, 0, image_rows * sp - image.shape[1]))
    # This rank's rows by device index, so every rank runs the same program.
    x = torch.cat((image, text), 1).index_select(1, kwargs["tpu_rows"])
    c = self.time_embedding(self.time_proj(timestep.to(dtype)).to(dtype))
    # As the stock: the fused residual-gate kernel needs dense inputs.
    modulation = [
        t.unsqueeze(1).contiguous() for t in self.adaLN_modulation(c).chunk(6, -1)
    ]
    image_keys, text_keys = kwargs["tpu_keys"]
    segments = {"tpu_segments": [(image_rows, image_keys), (text_rows, text_keys)]}
    for layer in self.layers:
        x = layer(x, *rope, None, None, *modulation, attn_mask_meta=segments)
    scale, shift = self.final_norm["linear"](c).chunk(2, dim=-1)
    x = self.final_norm["norm"](x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    patches, _ = self.final_linear(x[:, :image_rows])
    return patches


def _uses_kernel(impl, q, keys) -> bool:
    if not isinstance(impl, tpu_splash_attn.TpuSplashAttentionImpl):
        return False
    if not impl.use_splash or q.device.type != "tpu":
        return False
    # Every step takes the kernel or none does: their carries differ.
    return all(
        tpu_splash_attn._ring_kernel_fallback_reason(q, key, key) is None
        for key in keys
    )


def _joint_attention(usp, q, k, v, segments):
    """``[B, S, H, D]`` attention over each segment's real keys.

    ``segments`` lists ``(rows per rank, [0, real keys])`` in each rank's row
    order [image | prompt], every segment padded at its tail.
    """
    sp = get_sp_world_size()
    if sp > 1:
        if get_ring_parallel_world_size() > 1 or usp.sp_attention_mode == "kv_gather":
            raise NotImplementedError(
                "ERNIE-Image on TPU shards the sequence with Ulysses only"
            )
        q, k, v = (_usp_input_all_to_all(t, head_dim=2) for t in (q, k, v))
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
        # Zero padded keys and values: a masked score still adds q . k, and a
        # zero probability times a NaN value is NaN.
        out = F.scaled_dot_product_attention(
            q,
            torch.where(valid[:, None], torch.cat(keys, 2), 0),
            torch.where(valid[:, None], torch.cat(values, 2), 0),
            attn_mask=valid[None, None, None],
            scale=impl.softmax_scale,
        )
    out = out.transpose(1, 2)
    return _usp_output_all_to_all(out, head_dim=2) if sp > 1 else out


def attention_forward(
    self,
    x,
    rope_cos,
    rope_sin,
    rope_cache,
    rope_positions,
    attn_mask=None,
    attn_mask_meta=None,
):
    """``ErnieImageSelfAttention.forward`` over a bucketed prompt (see :func:`forward`)."""
    segments = None
    if isinstance(attn_mask_meta, dict):
        segments = attn_mask_meta.get("tpu_segments")
    if segments is None:
        return _STOCK_ATTENTION_FORWARD(
            self,
            x,
            rope_cos,
            rope_sin,
            rope_cache,
            rope_positions,
            attn_mask=attn_mask,
            attn_mask_meta=attn_mask_meta,
        )
    q, k, v = (
        proj(x)[0].unflatten(-1, (self.num_local_heads, self.head_dim))
        for proj in (self.to_q, self.to_k, self.to_v)
    )
    if self.qk_layernorm:
        q, k = dit._ernie_qknorm_rope_reference(
            q, k, self.norm_q, self.norm_k, self.head_dim, rope_cos, rope_sin
        )
    else:
        q, k = (dit._ernie_rope(t, rope_cos, rope_sin) for t in (q, k))
    out = _joint_attention(self.attn, q, k, v, segments)
    out, _ = self.to_out[0](out.flatten(2))
    return out


def _pad_tokens(inputs, tokenizer):
    """``inputs`` with its token rows right-padded to a multiple of 256."""
    extra = -inputs["input_ids"].shape[1] % PROMPT_TOKEN_ALIGNMENT
    if extra:
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        inputs["input_ids"] = F.pad(inputs["input_ids"], (0, extra), value=pad_id)
        inputs["attention_mask"] = F.pad(inputs["attention_mask"], (0, extra))
    return inputs


def _tokenize(self, prompts, tokenizer, tok_kwargs):
    return _pad_tokens(_STOCK_TOKENIZE(self, prompts, tokenizer, tok_kwargs), tokenizer)


def _unbatched(self, batch) -> bool:
    return False


def _load_modules(self, server_args, loaded_modules=None):
    # Two CFG halves, each running Ulysses in its own process subgroup, gave
    # identical requests different images on TPU.
    if server_args.enable_cfg_parallel:
        raise ValueError(
            "ERNIE-Image on TPU runs classifier-free guidance serially: launch "
            "with --ulysses-degree N instead of CFG parallel"
        )
    return _STOCK_LOAD_MODULES(self, server_args, loaded_modules)


def _whole_latents(self, batch, latents):
    """Latents stay whole under SP: the DiT shards its own stream (:func:`forward`)."""
    return latents, False


def install() -> None:
    pipeline.ErnieImagePipeline.load_modules = _load_modules
    pipeline_config = config.ErnieImagePipelineConfig
    pipeline_config.tokenize_prompt = _tokenize
    pipeline_config.shard_latents_for_sp = _whole_latents
    pipeline_config.supports_dynamic_batching_for_request = _unbatched
    dit.ErnieImageSelfAttention.forward = attention_forward
    model = dit.ErnieImageTransformer2DModel
    model.forward = forward
    model.__call__ = _dit_call
    # Under --regional-compile each block is one spmd-safe program.
    model._compile_conditions = [is_layer]
    model._compile_regions_spmd_safe = True
