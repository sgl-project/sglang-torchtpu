# SPDX-License-Identifier: Apache-2.0
"""Wan VAE decode on TPU workers: full-frame mid-block attention, and opt-in
decode options.

Runtime patch, tracked debt:
- Why: since sglang#46149 each spatial-parallel decode rank queries only its
  own rows in the mid-block attention (``WanAttentionBlock``), and torch_tpu's
  float32 attention kernel cannot compile that query block within scoped VMEM
  (CompileTimeScopedVmemOom at 8 TensorCores: queries [5, 1, 768, 384] against
  keys [5, 1, 7168, 384]). On tpu7x the decoder's last convolution (96 -> 3
  channels) took 19.0 ms a call against 2.1 ms for a 96-channel one at the
  same size, about a fifth of an 832x480x81 decode; and the relayout copies
  XLA places around every convolution are slower on the SparseCores than on
  the TensorCores.
- How: :func:`install` always replaces the mid-block attention's forward:
  under a spatial-parallel decode every rank attends the whole gathered frame
  and keeps its own rows, as the decode did before sglang#46149. With a Wan
  VAE variable set it also wraps ``AutoencoderKLWan.decode``:
  ``SGLANG_TPU_OPT_WAN_VAE_CONV_OUT_CHANNELS`` runs the decoder's last
  convolution with zero-padded weights at that many output channels and keeps
  the first three: each kept channel sums the same products.
  ``SGLANG_TPU_OPT_WAN_VAE_XLA_OPTIONS`` compiles the decode's programs, and
  nothing else, with extra XLA options. With
  ``xla_tpu_enable_offloading_copy_to_sparsecore=false`` the relayout copies
  run on the TensorCores instead of the SparseCores: data movement only, so
  the decode is bitwise the same, and an 832x480x81 decode takes 3.99 -> 3.49 s
  on one chip, 2.01 -> 1.86 s on two and 1.08 -> 1.04 s on four.
- Upstream PR or issue: none.
- Remove when: torch_tpu's float32 attention kernel compiles a rank's query
  block within scoped VMEM (the full-frame attention); XLA runs small-feature
  convolutions at full MXU width (the padded conv_out); XLA's TPU relayout
  copies are no slower on the SparseCores (the XLA options).
"""

from __future__ import annotations

import contextlib
import functools
import logging

import torch
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang_tpu.envs import envs

logger = logging.getLogger(__name__)

_INSTALLED = "_tpu_wan_vae_installed"
_FULL_FRAME_INSTALLED = "_tpu_wan_vae_full_frame_attention"


def conv_out_channels() -> int | None:
    """``SGLANG_TPU_OPT_WAN_VAE_CONV_OUT_CHANNELS``, or None to leave conv_out."""
    value = envs.SGLANG_TPU_OPT_WAN_VAE_CONV_OUT_CHANNELS.get()
    if value is None:
        return None
    if value < 1:
        raise ValueError(
            f"SGLANG_TPU_OPT_WAN_VAE_CONV_OUT_CHANNELS={value}: expected >= 1"
        )
    return value


def xla_options() -> dict[str, str] | None:
    """``SGLANG_TPU_OPT_WAN_VAE_XLA_OPTIONS`` as {name: value}, or None."""
    value = envs.SGLANG_TPU_OPT_WAN_VAE_XLA_OPTIONS.get()
    if value is None or not value.strip():
        return None
    options = {}
    for item in value.split(","):
        name, sep, setting = (part.strip() for part in item.partition("="))
        if not sep or not name or not setting or name in options:
            raise ValueError(
                f"SGLANG_TPU_OPT_WAN_VAE_XLA_OPTIONS item {item!r}: expected "
                "'<xla option>=<value>'"
            )
        options[name] = setting
    return options


def _custom_compiler_options(options: dict[str, str]):
    from torch_tpu._internal.compiler_options import custom_compiler_options

    return custom_compiler_options(options)


def _xla_scope(options: dict[str, str] | None):
    """Compile the programs the block materializes with ``options``."""
    return (
        contextlib.nullcontext() if not options else _custom_compiler_options(options)
    )


def pad_conv_out(decoder, channels: int | None) -> None:
    """Run ``decoder.conv_out`` at ``channels`` output channels (None: stock).

    The weights and bias are zero-padded to ``channels`` outputs (refreshed
    when the module's weight changes) and the output is cut back to the real
    channels behind a fusion barrier, so a graph compiler keeps the wide
    convolution. Each kept channel is the same sum of the same products.
    """
    conv = getattr(decoder, "conv_out", None)
    if conv is None or channels is None or conv.out_channels >= channels:
        return
    weight, bias = conv.weight, conv.bias
    key = (weight.device, weight.dtype, weight._version, channels)
    cached = conv.__dict__.get("_tpu_padded_conv_out_key")
    if cached is None or cached[0] is not weight or cached[1:] != key:
        with torch.no_grad():
            padded = weight.new_zeros((channels, *weight.shape[1:]))
            padded[: weight.shape[0]] = weight
            conv._tpu_padded_weight = padded
            conv._tpu_padded_bias = None
            if bias is not None:
                conv._tpu_padded_bias = bias.new_zeros(channels)
                conv._tpu_padded_bias[: bias.shape[0]] = bias
        conv._tpu_padded_conv_out_key = (weight, *key)
    if "_conv_forward" in conv.__dict__:
        return

    def conv_forward(input, weight, bias):
        # nn.Conv3d.forward's call (zeros padding_mode), at the padded width.
        out = F.conv3d(
            input,
            conv._tpu_padded_weight,
            # A caller that adds the bias itself passes None (skip_bias).
            conv._tpu_padded_bias if bias is not None else None,
            conv.stride,
            conv.padding,
            conv.dilation,
            conv.groups,
        )
        return current_platform.fusion_barrier(out)[:, : weight.shape[0]]

    conv._conv_forward = conv_forward


def full_frame_attention_forward(wanvae):
    """``WanAttentionBlock.forward`` attending the whole gathered frame.

    Each rank gathers the frame's height, attends all of it and keeps its own
    rows, as the decode did before sglang#46149. The rows a rank keeps are
    the same; only the query block handed to the attention kernel grows.
    """
    from sglang.multimodal_gen.runtime.layers import parallel_conv

    def forward(self, x):
        group = self.parallel_group
        if self.world_size > 1:
            x = parallel_conv.gather_height_for_global_op(
                x, parallel_group=group
            ).contiguous()
        x = wanvae.attention_block_forward(self, x)
        if self.world_size > 1:
            x = parallel_conv.chunk_height_for_parallel_decode(x, parallel_group=group)
        return x

    return forward


def install_full_frame_attention() -> None:
    from sglang.multimodal_gen.runtime.models.vaes import wanvae

    attention = wanvae.WanAttentionBlock
    if getattr(attention, _FULL_FRAME_INSTALLED, False):
        return
    attention.forward = full_frame_attention_forward(wanvae)
    setattr(attention, _FULL_FRAME_INSTALLED, True)


def install() -> None:
    """Wrap ``AutoencoderKLWan.decode`` when a Wan VAE variable is set.

    The full-frame mid-block attention is installed in every case.
    """
    install_full_frame_attention()
    conv_out = conv_out_channels()
    options = xla_options()
    if conv_out is None and not options:
        return
    from sglang.multimodal_gen.runtime.models.vaes import wanvae

    cls = wanvae.AutoencoderKLWan
    if getattr(cls, _INSTALLED, False):
        return
    torch_decode = cls.decode

    @functools.wraps(torch_decode)
    def decode(self, z, on_frames=None):
        if conv_out is not None and getattr(self, "decoder", None) is not None:
            pad_conv_out(self.decoder, conv_out)
        with _xla_scope(options):
            return torch_decode(self, z, on_frames=on_frames)

    cls.decode = decode
    setattr(cls, _INSTALLED, True)
    if conv_out is not None:
        logger.info(
            "Wan VAE decode: conv_out runs at %d output channels (zero-padded "
            "weights; the real channels are kept).",
            conv_out,
        )
    if options:
        logger.info("Wan VAE decode: XLA options %s for its programs.", options)
