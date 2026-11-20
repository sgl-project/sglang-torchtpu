# SPDX-License-Identifier: Apache-2.0
"""HunyuanImage-3 on TPU workers: the srt fused MoE inside the diffusion
runtime, a bounded VAE working set, and bucketed sequence lengths.

Runtime patch, tracked debt:
- Why: the AR backbone's experts run sglang_tpu's Pallas fused MoE (srt
  FusedMoE on TPU). The diffusion launcher runs JAX on the CPU, and the kernel
  reads the TPU's metadata while it is exported. The float32 conv VAE runs
  eagerly: the host queues a whole decode (or encode) ahead of the device, and
  every queued block's activations stay allocated until the device has run it
  (a 1024x1024 decode peaked at 88-90 GiB of a core's 96 GiB). The eager ops
  compile once per shape, and the sequence has the prompt's length: every new
  prompt length compiled them again (+27 s).
- How: :func:`install`
  (1) runs each sparse MoE block's forward inside ``tpu_info_for_cpu_export``
      (as Ming-Image's encoder and the splash kernels do);
  (2) waits for the device after each VAE residual, attention and resampling
      block, so at most one block's activations are queued. The math is
      unchanged;
  (3) pads each sequence on the host before the stage uploads it: pad rows go
      right before the first denoised token, so the cached prefix ends at a
      multiple of PREFIX_ALIGNMENT, and the tokenizer's batch padding is
      dropped. The attention mask and the RoPE tables come from the unpadded
      layout; pad rows see only themselves and no row sees them. Prompts of one
      bucket share every per-request shape.
- Upstream PR or issue: none.
- Remove when: the fused-MoE op exports with TPU metadata on its own, the VAE
  runs compiled or torch_tpu bounds the eager queue, and the stage takes a
  platform sequence bucket.
"""

from __future__ import annotations

import dataclasses
import functools

import torch

# Devices that take these paths (tests add "cpu").
_DEVICE_TYPES = ("tpu",)
# The cached prefix of a bucketed sequence is a multiple of this many rows.
PREFIX_ALIGNMENT = 256
_SLICE_FIELDS = (
    "gen_image_slices",
    "joint_image_slices",
    "cond_vae_image_slices",
    "cond_vit_image_slices",
    "all_image_slices",
    "text_slices",
)
_INDEX_FIELDS = (
    "timestep_scatter_index",
    "gen_timestep_scatter_index",
    "cond_timestep_scatter_index",
)


def _synchronize(device) -> None:
    import torch

    torch.accelerator.synchronize(device)


def _patch_moe_forward() -> None:
    from sglang.multimodal_gen.runtime.models.dits import hunyuan_image3 as dit

    stock_moe_forward = dit.HunYuanSparseMoeBlock.forward
    if getattr(stock_moe_forward, "_tpu_export_metadata", False):
        return

    @functools.wraps(stock_moe_forward)
    def moe_forward(self, hidden_states):
        if hidden_states.device.type not in _DEVICE_TYPES:
            return stock_moe_forward(self, hidden_states)
        from sglang_tpu.pallas_export import tpu_info_for_cpu_export

        with tpu_info_for_cpu_export():
            return stock_moe_forward(self, hidden_states)

    moe_forward._tpu_export_metadata = True
    dit.HunYuanSparseMoeBlock.forward = moe_forward


def _patch_vae_blocks() -> None:
    from sglang.multimodal_gen.runtime.models.vaes import (
        autoencoder_kl_conv3d_hunyuan_image3 as vae,
    )

    for cls in (vae.ResnetBlock, vae.AttnBlock, vae.UpsampleDCAE, vae.DownsampleDCAE):
        stock = cls.forward
        if getattr(stock, "_tpu_block_sync", False):
            continue

        def forward(self, x, _stock=stock):
            out = _stock(self, x)
            if out.device.type in _DEVICE_TYPES:
                _synchronize(out.device)
            return out

        forward = functools.wraps(stock)(forward)
        forward._tpu_block_sync = True
        cls.forward = forward


def bucket_sequences(out, pad_id: int):
    """Bucket a batched TokenizerEncodeOutput in place; returns its unpadded
    copy and ``rows`` ([B, L_b], the source row of each position, L for pads).
    """
    batch, length = out.tokens.shape
    real = out.real_pos.reshape(batch).tolist()
    starts = [
        min(
            [s.start for s in out.gen_image_slices[b]]
            + out.gen_timestep_scatter_index[b].tolist()
        )
        for b in range(batch)
    ]
    padded_prefix = -(-max(starts) // PREFIX_ALIGNMENT) * PREFIX_ALIGNMENT
    padded_len = padded_prefix + max(r - p for r, p in zip(real, starts))
    rows = torch.full((batch, padded_len), length, dtype=torch.long)
    shift = torch.tensor([padded_prefix - p for p in starts])
    for b, (r, p) in enumerate(zip(real, starts)):
        rows[b, :p] = torch.arange(p)
        rows[b, padded_prefix : padded_prefix + r - p] = torch.arange(p, r)
    orig = dataclasses.replace(out)
    for field in dataclasses.fields(out):
        value = getattr(out, field.name)
        if field.name in _SLICE_FIELDS and value is not None:
            value = [
                [
                    slice(s.start + int(shift[b]), s.stop + int(shift[b]))
                    if s.start >= starts[b]
                    else s
                    for s in row
                ]
                for b, row in enumerate(value)
            ]
        elif field.name in _INDEX_FIELDS and value is not None:
            starts_col = torch.tensor(starts).reshape(-1, *[1] * (value.dim() - 1))
            value = torch.where(
                value >= starts_col, value + shift.reshape_as(starts_col), value
            )
        elif torch.is_tensor(value) and tuple(value.shape) == (batch, length):
            fill = pad_id if field.name == "tokens" else 0
            value = torch.cat([value, value.new_full((batch, 1), fill)], 1).gather(
                1, rows
            )
        setattr(out, field.name, value)
    out.real_pos = torch.tensor(
        [padded_prefix + r - p for r, p in zip(real, starts)]
    ).reshape(batch, 1)
    return orig, rows


def _bucketed_mask_and_rope(stage, orig, rows, sections, token_h, token_w, image_info):
    """The stock mask and RoPE tables of the unpadded layout, on the host,
    moved to the bucketed rows; pad rows see only themselves."""
    from sglang.multimodal_gen.runtime.models.dits.hunyuan_image3_utils import (
        build_batch_2d_rope,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.hunyuan_image3 import (
        ar_stage,
    )

    batch, length = orig.tokens.shape
    image_slices = [
        orig.joint_image_slices[b] + orig.gen_image_slices[b] for b in range(batch)
    ]
    mask = ar_stage._build_causal_attention_mask(batch, length, image_slices, "cpu")[
        :, 0
    ]
    mask = torch.nn.functional.pad(mask, (0, 1, 0, 1))
    mask = torch.stack(
        [mask[b][rows[b][:, None], rows[b][None, :]] for b in range(batch)]
    )
    pad = rows == length
    mask |= pad[:, :, None] & torch.eye(rows.shape[1], dtype=torch.bool)
    rope = stage.ar_model.cached_rope
    info = ar_stage._build_rope_image_info(
        orig, batch, token_h, token_w, image_info, sections
    )
    tables = build_batch_2d_rope(
        length, rope.head_dim, info, device="cpu", base=rope.rope_theta
    )
    index = rows[..., None].expand(-1, -1, rope.head_dim)
    cos, sin = (
        torch.cat([t, torch.full_like(t[:, :1], fill)], 1).gather(1, index)
        for t, fill in zip(tables, (1.0, 0.0))
    )
    return mask[:, None], cos, sin


def _patch_sequence_buckets() -> None:
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.hunyuan_image3 import (
        ar_stage,
    )

    stage_cls = ar_stage.HunyuanImage3AR
    stock_parse = stage_cls._parse_tokenizer_output
    stock_build = stage_cls._build_attention_and_rope
    if getattr(stock_parse, "_tpu_bucketed", False):
        return

    def parse(self, out, device):
        if (
            torch.device(device).type in _DEVICE_TYPES
            and out.gen_timestep_scatter_index is not None
        ):
            tok = self._custom_tokenizer
            pad_id = tok.special_token_map.get("<pad>", tok.pad_token_id)
            out._tpu_bucket = bucket_sequences(out, pad_id)
        return stock_parse(self, out, device)

    def build(
        self,
        out,
        sections,
        batch,
        seq_len,
        token_h,
        token_w,
        image_info,
        device,
        do_cfg=False,
    ):
        if getattr(out, "_tpu_bucket", None) is None:
            return stock_build(
                self,
                out,
                sections,
                batch,
                seq_len,
                token_h,
                token_w,
                image_info,
                device,
                do_cfg,
            )
        mask, cos, sin = _bucketed_mask_and_rope(
            self, *out._tpu_bucket, sections, token_h, token_w, image_info
        )
        half = batch // 2
        shared = do_cfg and batch % 2 == 0 and torch.equal(mask[:half], mask[half:])
        mask = mask[:half] if shared else mask
        return mask.to(device), cos.to(device), sin.to(device), shared

    parse._tpu_bucketed = True
    stage_cls._parse_tokenizer_output = parse
    stage_cls._build_attention_and_rope = build


def install() -> None:
    _patch_moe_forward()
    _patch_vae_blocks()
    _patch_sequence_buckets()
