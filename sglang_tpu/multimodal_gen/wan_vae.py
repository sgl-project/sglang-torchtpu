# SPDX-License-Identifier: Apache-2.0
"""Wan VAE decode on TPU workers: a decode-scoped float32 matmul precision.

Runtime patch, tracked debt:
- Why: the worker sets ``torch.set_float32_matmul_precision("highest")`` for
  every float32 matmul and convolution (worker.py), so Wan2.2-A14B's fp32 VAE
  decode (``vae_decode_precision="fp32"``) runs each convolution as 6 bf16 MXU
  passes. "high" runs 3 (bf16_3x): each product has a relative error of about
  3 * 2**-16, against about 2 * 2**-11 for TF32, the precision cuDNN uses for
  float32 convolutions under PyTorch's defaults
  (``torch.backends.cudnn.allow_tf32 = True``,
  ``torch.backends.cudnn.conv.fp32_precision = "tf32"``). Accumulation stays
  float32.
- How: with ``SGLANG_TPU_WAN_VAE_DECODE_FP32_MATMUL_PRECISION`` set,
  :func:`install` wraps ``AutoencoderKLWan.decode`` so that the decode, and
  nothing else, runs at that precision. The mid-block attention
  (``WanAttentionBlock``) keeps "highest": CUDA computes float32 matmuls in
  float32 by default (``torch.backends.cuda.matmul.allow_tf32 = False``), so
  only the convolutions, which cuDNN runs in TF32, change. torch_tpu keys its
  compiled eager ops on the precision, so each setting gets its own
  executables.
  ``SGLANG_TPU_DEBUG_WAN_VAE_AB`` instead decodes every call once per listed
  variant (chunking, precision, eager or compiled decoder calls), logs
  synchronized times and the differences from the first variant, and returns
  the first variant's decode.
  ``SGLANG_TPU_OPT_WAN_VAE_CONV_OUT_CHANNELS`` runs the decoder's last
  convolution (96 -> 3 channels) with zero-padded weights at that many output
  channels and keeps the first three: each kept channel sums the same products.
  On tpu7x the 3-channel convolution took 19.0 ms a call against 2.1 ms for a
  96-channel one at the same size (B1 decode trace), about a fifth of the
  decode.
  ``SGLANG_TPU_OPT_WAN_VAE_XLA_OPTIONS`` compiles the decode's programs, and
  nothing else, with extra XLA options. With
  ``xla_tpu_enable_offloading_copy_to_sparsecore=false`` the relayout copies
  XLA places around every convolution run on the TensorCores instead of the
  SparseCores: data movement only, so the decode is bitwise the same, and
  3.99 -> 3.49 s on one chip, 2.01 -> 1.86 s on two, 1.08 -> 1.04 s on four
  (A/B profile cells).
  Under a spatial-parallel decode, :func:`install` also runs the mid-block
  attention over the whole gathered frame on every rank and keeps the rank's
  rows, as the decode did before sglang#46149. Since #46149 each rank queries
  only its own rows, and torch_tpu's float32 attention kernel cannot compile
  that query block within scoped VMEM (CompileTimeScopedVmemOom at 8
  TensorCores: queries [5, 1, 768, 384] against keys [5, 1, 7168, 384]).
- Upstream PR or issue: none.
- Remove when: the platform interface scopes a precision to a pipeline stage,
  or the worker no longer sets a global float32 matmul precision; and XLA runs
  small-feature convolutions at full MXU width; and XLA's TPU relayout copies
  are no slower on the SparseCores.
"""

from __future__ import annotations

import contextlib
import functools
import logging
import math
import os
import time

import torch
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.utils.decode_stream import stream_decoded_chunks
from sglang_tpu.envs import envs

logger = logging.getLogger(__name__)

_INSTALLED = "_tpu_wan_vae_installed"
_FULL_FRAME_INSTALLED = "_tpu_wan_vae_full_frame_attention"
# torch.set_float32_matmul_precision names; torch_tpu maps "high" to StableHLO
# HIGH (3 bf16 passes on TPU) and "highest" to HIGHEST (6 passes).
PRECISIONS = ("high", "highest")
# Read lazily by AutoencoderKLWan.decode on every call.
_CHUNK_ENV = "SGLANG_DIFFUSION_WAN_VAE_DECODE_CHUNK_FRAMES"
_COMPILE_ENV = "SGLANG_DIFFUSION_ENABLE_WAN_VAE_DECODER_COMPILE"


def _parse_precision(value: str, source: str) -> str:
    mode = value.strip().lower()
    if mode not in PRECISIONS:
        raise ValueError(f"{source}={value!r}: expected one of {', '.join(PRECISIONS)}")
    return mode


def decode_precision() -> str | None:
    """The decode's float32 matmul precision, or None to keep the worker's."""
    value = envs.SGLANG_TPU_WAN_VAE_DECODE_FP32_MATMUL_PRECISION.get()
    if value is None or not value.strip():
        return None
    return _parse_precision(value, "SGLANG_TPU_WAN_VAE_DECODE_FP32_MATMUL_PRECISION")


def _parse_ab_options(fields: list[str]) -> tuple[bool, int | None, bool] | None:
    """(compiled, conv_out channels, xla) of an A/B item's options, or None."""
    compiled, conv_out, xla = False, None, False
    for field in fields:
        name = field.lower()
        if name == "compile" and not compiled:
            compiled = True
        elif name == "xla" and not xla:
            xla = True
        elif (
            name.startswith("convout")
            and name[len("convout") :].isdigit()
            and int(name[len("convout") :]) >= 1
            and conv_out is None
        ):
            conv_out = int(name[len("convout") :])
        else:
            return None
    if compiled and xla:  # a compiled graph takes no per-op compiler options
        return None
    return compiled, conv_out, xla


def ab_variants() -> list[tuple[int, str, bool, int | None, bool]] | None:
    """``SGLANG_TPU_DEBUG_WAN_VAE_AB`` as
    [(chunk_frames, precision, compiled, conv_out_channels, xla), ...].

    An item ``<chunk>:<precision>`` decodes eagerly with the stock conv_out;
    the options ``:compile`` (the decoder calls with full caches compiled,
    SGLANG_DIFFUSION_ENABLE_WAN_VAE_DECODER_COMPILE), ``:convout<N>``
    (conv_out at N output channels, as SGLANG_TPU_OPT_WAN_VAE_CONV_OUT_CHANNELS)
    and ``:xla`` (SGLANG_TPU_OPT_WAN_VAE_XLA_OPTIONS's options, not with
    ``:compile``) follow in any order.
    """
    value = envs.SGLANG_TPU_DEBUG_WAN_VAE_AB.get()
    if value is None or not value.strip():
        return None
    variants = []
    for item in value.split(","):
        fields = [field.strip() for field in item.split(":")]
        options = _parse_ab_options(fields[2:])
        if (
            len(fields) < 2
            or not fields[0].isdigit()
            or int(fields[0]) < 1
            or options is None
        ):
            raise ValueError(
                f"SGLANG_TPU_DEBUG_WAN_VAE_AB item {item!r}: expected "
                "'<latent frames per call>:<high|highest>[:compile|:xla][:convout<N>]'"
            )
        variants.append(
            (
                int(fields[0]),
                _parse_precision(fields[1], "SGLANG_TPU_DEBUG_WAN_VAE_AB"),
                *options,
            )
        )
    if any(variant[4] for variant in variants) and xla_options() is None:
        raise ValueError(
            "SGLANG_TPU_DEBUG_WAN_VAE_AB: ':xla' needs SGLANG_TPU_OPT_WAN_VAE_XLA_OPTIONS"
        )
    return variants


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


def requested() -> bool:
    return (
        decode_precision() is not None
        or ab_variants() is not None
        or conv_out_channels() is not None
        or xla_options() is not None
    )


def pad_conv_out(decoder, channels: int | None) -> None:
    """Run ``decoder.conv_out`` at ``channels`` output channels (None: stock).

    The weights and bias are zero-padded to ``channels`` outputs (refreshed
    when the module's weight changes) and the output is cut back to the real
    channels behind a fusion barrier, so a graph compiler keeps the wide
    convolution. Each kept channel is the same sum of the same products.
    """
    conv = getattr(decoder, "conv_out", None)
    if conv is None:
        return
    if channels is None or conv.out_channels >= channels:
        if "_conv_forward" in conv.__dict__:
            conv._tpu_conv_out_channels = None
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
    conv._tpu_conv_out_channels = channels
    if "_conv_forward" in conv.__dict__:
        return

    def conv_forward(input, weight, bias):
        # nn.Conv3d.forward's call (zeros padding_mode), at the padded width
        # unless an A/B variant runs the stock convolution.
        if conv._tpu_conv_out_channels is None:
            return F.conv3d(
                input,
                weight,
                bias,
                conv.stride,
                conv.padding,
                conv.dilation,
                conv.groups,
            )
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


@contextlib.contextmanager
def fp32_matmul_precision(mode: str | None):
    """Run the block at ``mode`` and restore the previous precision after it.

    Uses the global setting the worker itself sets (torch_tpu patches
    ``torch.set_float32_matmul_precision``), so it covers every float32 matmul
    and convolution the block issues.
    """
    if mode is None:
        yield
        return
    previous = torch.get_float32_matmul_precision()
    if previous == mode:
        yield
        return
    torch.set_float32_matmul_precision(mode)
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(previous)


@contextlib.contextmanager
def _env(name: str, value: str):
    previous = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


def _chunk_frames(chunk: int):
    return _env(_CHUNK_ENV, str(chunk))


def _variant_name(
    chunk: int,
    precision: str,
    compiled: bool,
    conv_out: int | None,
    xla: bool = False,
) -> str:
    return (
        f"{chunk}:{precision}"
        + (":compile" if compiled else "")
        + (f":convout{conv_out}" if conv_out is not None else "")
        + (":xla" if xla else "")
    )


@contextlib.contextmanager
def _conv_out_width(vae, channels: int | None):
    decoder = getattr(vae, "decoder", None)
    conv = getattr(decoder, "conv_out", None)
    previous = None if conv is None else conv.__dict__.get("_tpu_conv_out_channels")
    pad_conv_out(decoder, channels)
    try:
        yield
    finally:
        pad_conv_out(decoder, previous)


def _synchronize() -> None:
    module = torch.get_device_module()
    if module.is_available():
        module.synchronize()


def _to_uint8(x: torch.Tensor) -> torch.Tensor:
    # DecodingStage's (x / 2 + 0.5).clamp(0, 1), then the save path's
    # (x * 255).clamp(0, 255).to(uint8), which truncates.
    unit = (x.float() / 2 + 0.5).clamp(0, 1)
    return (unit * 255).clamp(0, 255).to(torch.uint8)


def compare(out: torch.Tensor, reference: torch.Tensor) -> dict:
    """Differences of two decodes in [-1, 1], measured as the served frames.

    ``max_abs`` and ``psnr_db`` are on the [0, 1] frame scale; ``u8_*`` on the
    uint8 frames the encoder receives.
    """
    diff = (out.float() - reference.float()).abs()
    max_abs = (diff.max() / 2).item()
    mse = (diff / 2).square().mean().item()
    diff8 = (_to_uint8(out).int() - _to_uint8(reference).int()).abs()
    mismatched = int((diff8 > 0).sum().item())
    return {
        "max_abs": max_abs,
        "psnr_db": 10 * math.log10(1.0 / mse) if mse > 0 else math.inf,
        "u8_mismatched": mismatched,
        "u8_mismatched_frac": mismatched / diff8.numel(),
        "u8_max_diff": int(diff8.max().item()),
    }


def _is_log_rank() -> bool:
    return (
        not torch.distributed.is_available()
        or not torch.distributed.is_initialized()
        or torch.distributed.get_rank() == 0
    )


_ab_calls = [0]


def decode_ab(
    vae, torch_decode, z: torch.Tensor, variants, on_frames=None
) -> torch.Tensor:
    """Decode ``z`` once per variant; return the first variant's decode.

    Every rank runs the same variants in the same order, so the decode
    group's collectives stay matched. Only the first variant's chunks reach a
    streaming video save or ``on_frames``.
    """
    _ab_calls[0] += 1
    reference = None
    reference_seconds = None
    for variant in variants:
        chunk, precision, compiled, conv_out, xla = (*variant, False)[:5]
        with (
            _chunk_frames(chunk),
            _env(_COMPILE_ENV, "1" if compiled else "0"),
            _conv_out_width(vae, conv_out),
            _xla_scope(xla_options() if xla else None),
            fp32_matmul_precision(precision),
            contextlib.nullcontext()
            if reference is None
            else stream_decoded_chunks(None),
        ):
            _synchronize()
            start = time.perf_counter()
            out = torch_decode(
                vae, z, on_frames=on_frames if reference is None else None
            )
            _synchronize()
            seconds = time.perf_counter() - start
        if reference is None:
            reference, reference_seconds = out, seconds
            stats = "reference"
        else:
            d = compare(out, reference)
            del out
            stats = (
                f"max_abs {d['max_abs']:.3e} psnr {d['psnr_db']:.2f} dB "
                f"u8 mismatched {d['u8_mismatched']} ({d['u8_mismatched_frac']:.3e}) "
                f"u8 max diff {d['u8_max_diff']}"
            )
        if _is_log_rank():
            logger.info(
                "Wan VAE A/B call %d: chunk %d %s%s%s%s: %.4f s (reference %.4f s), %s",
                _ab_calls[0],
                chunk,
                precision,
                " compiled" if compiled else "",
                f" conv_out {conv_out}" if conv_out is not None else "",
                " xla" if xla else "",
                seconds,
                reference_seconds,
                stats,
            )
    return reference


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
            x = parallel_conv.chunk_height_for_parallel_decode(
                x, parallel_group=group
            )
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
    precision = decode_precision()
    variants = ab_variants()
    conv_out = conv_out_channels()
    options = xla_options()
    if precision is None and variants is None and conv_out is None and not options:
        return
    from sglang.multimodal_gen.runtime.models.vaes import wanvae

    cls = wanvae.AutoencoderKLWan
    if getattr(cls, _INSTALLED, False):
        return
    torch_decode = cls.decode
    attention = wanvae.WanAttentionBlock
    attention_forward = attention.forward

    @functools.wraps(attention_forward)
    def attention_at_highest(self, x):
        # A no-op wherever the precision already is "highest". Dynamo does not
        # trace the precision setting: a compiled decoder call
        # (SGLANG_DIFFUSION_ENABLE_WAN_VAE_DECODER_COMPILE) runs its attention
        # at the precision the call compiled at, and recompiles when "highest"
        # and the lower settings alternate (its allow_tf32 guard).
        if torch.compiler.is_compiling():
            return attention_forward(self, x)
        with fp32_matmul_precision("highest"):
            return attention_forward(self, x)

    @functools.wraps(torch_decode)
    def decode(self, z, on_frames=None):
        if conv_out is not None and getattr(self, "decoder", None) is not None:
            pad_conv_out(self.decoder, conv_out)
        if variants is not None:
            return decode_ab(self, torch_decode, z, variants, on_frames)
        with fp32_matmul_precision(precision), _xla_scope(options):
            return torch_decode(self, z, on_frames=on_frames)

    cls.decode = decode
    if precision is not None or variants is not None:
        attention.forward = attention_at_highest
    setattr(cls, _INSTALLED, True)
    if conv_out is not None:
        logger.info(
            "Wan VAE decode: conv_out runs at %d output channels (zero-padded "
            "weights; the real channels are kept).",
            conv_out,
        )
    if options and variants is None:
        logger.info("Wan VAE decode: XLA options %s for its programs.", options)
    if variants is not None:
        logger.warning(
            "Wan VAE A/B (debug): every decode runs %s; the first is returned.",
            ", ".join(_variant_name(*variant) for variant in variants),
        )
    elif precision is not None:
        logger.info(
            "Wan VAE decode: float32 convolutions at %r for the decode only "
            "(its attention block stays at 'highest').",
            precision,
        )
