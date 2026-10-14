# SPDX-License-Identifier: Apache-2.0
"""Checkpoint -> per-rank GPT-OSS decode megakernel weights (torch, CPU).

The megakernel's own ``megakernel.pack_weights`` takes global bf16 weights in JAX and packs
all R ranks at once; that is too big to run per serving process. This module builds the
*same* per-rank layout (``megakernel.packed_specs`` without the leading R axis) straight
from the HF GPT-OSS MXFP4 checkpoint tensors as they stream through ``load_weights``:

- MXFP4 experts are pure byte reshuffles. A checkpoint row stores 32 e2m1 codes per 16-byte
  block, element 2j in the low nibble of byte j, so 4 consecutive bytes read as a
  little-endian uint32 hold 8 consecutive K codes with code k at bits 4k: exactly the
  kernel's word layout. ``[N, K/2] uint8 -> view int32 [N, K/8] -> transpose [K/8, N]``.
  e8m0 scales become bf16 powers of two by placing the exponent in the bf16 exponent field.
  No requantization, so the kernel sees bit-identical weights to the checkpoint.
- w13 rows are interleaved in the checkpoint (gate_i at row 2i, up_i at 2i+1); the kernel
  wants per-chunk ``[gate T | up T]`` columns.
- bf16 checkpoints (HF dequantized ``gate_up_proj [E, H, 2I]`` / ``down_proj [E, I, H]``) are
  requantized with ``reference.mxfp4_quantize``'s rule, which is lossless for weights that
  were dequantized from the MXFP4 release (every block re-derives an equivalent scale).
- TP MoE splits the intermediate dim on whole 32-wide scale blocks (``DecodeConfig.moe_cols``);
  EP MoE keeps whole experts ``[r*E/R, (r+1)*E/R)``.
- Attention / router / norms are sliced per rank exactly as ``pack_weights`` does.

Every checkpoint tensor maps to disjoint parts of the outputs, so tensors are consumed one
at a time as they stream (the RunAI streamer may reuse a tensor's buffer once the next one
arrives) and nothing but the preallocated per-rank outputs is kept.

Code words are stored as int32 (torch has limited uint32 support); the JAX side bitcasts.
"""

from __future__ import annotations

import re
from typing import Optional

import torch
import torch.nn.functional as F

from sglang.srt.hardware_backend.tpu.kernels.gptoss_decode_megakernel.config import (
    DecodeConfig,
)

NEG = -1e30
# rows of the per-layer `vec` operand (megakernel.V_LN1 ...)
V_LN1, V_LN2, V_BO, V_BQKV, V_BR = range(5)

AW_NAMES = ("wqkv", "wo", "wr", "vec", "sinks")
EW_NAMES_FP4 = ("w13q", "w13s", "b13", "w2q", "w2s", "b2")
EW_NAMES_FP8 = ("w13", "c13", "b13", "w2", "c2", "b2")
W_NAMES = AW_NAMES + EW_NAMES_FP4  # what the streaming builder produces


def w_names(c: DecodeConfig) -> tuple[str, ...]:
    """Operand names of ``c.w_dtype`` (``megakernel.w_names``)."""
    return AW_NAMES + {"mxfp4": EW_NAMES_FP4, "fp8": EW_NAMES_FP8}[c.w_dtype]


_LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")

# checkpoint key (after "model.layers.<l>.") -> handler name
_KEYS = {
    "self_attn.q_proj.weight": "q_w",
    "self_attn.q_proj.bias": "q_b",
    "self_attn.k_proj.weight": "k_w",
    "self_attn.k_proj.bias": "k_b",
    "self_attn.v_proj.weight": "v_w",
    "self_attn.v_proj.bias": "v_b",
    "self_attn.o_proj.weight": "o_w",
    "self_attn.o_proj.bias": "o_b",
    "self_attn.sinks": "sinks",
    "input_layernorm.weight": "ln1",
    "post_attention_layernorm.weight": "ln2",
    "mlp.router.weight": "r_w",
    "mlp.router.bias": "r_b",
    "mlp.experts.gate_up_proj_blocks": "gu_blocks",
    "mlp.experts.gate_up_proj_scales": "gu_scales",
    "mlp.experts.gate_up_proj_bias": "gu_bias",
    "mlp.experts.down_proj_blocks": "dn_blocks",
    "mlp.experts.down_proj_scales": "dn_scales",
    "mlp.experts.down_proj_bias": "dn_bias",
    # bf16 checkpoints (HF dequantized layout): requantized to MXFP4 on the fly
    "mlp.experts.gate_up_proj": "gu_w",
    "mlp.experts.down_proj": "dn_w",
}
# a bf16 expert tensor stands for both the codes and the scales of the MXFP4 layout
_COVERS = {"gu_w": ("gu_blocks", "gu_scales"), "dn_w": ("dn_blocks", "dn_scales")}
REQUIRED_KEYS = frozenset(v for v in _KEYS.values() if v not in _COVERS)


def per_rank_specs(c: DecodeConfig) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    """Shapes/dtypes of one rank's operands: ``megakernel.packed_specs`` minus the R axis
    (uint32 code words as int32 on the host)."""
    L, Hp, EL, NJ, T = c.L, c.Hp, c.EL, c.NJ, c.tne
    out = {
        "wqkv": ((L, Hp, c.QKVWL), torch.bfloat16),
        "wo": ((L, c.QWL, Hp), torch.bfloat16),
        "wr": ((L, Hp, 128), torch.bfloat16),
        "vec": ((L, 8, c.VW), torch.float32),
        "sinks": ((L, c.NQL, 128), torch.float32),
        "b13": ((L, EL, NJ, 1, 2 * T), torch.float32),
        "b2": ((L, EL, 1, Hp), torch.float32),
    }
    if c.w_dtype == "mxfp4":
        out.update(
            {
                "w13q": ((L, EL, NJ, Hp // 8, 2 * T), torch.int32),
                "w13s": ((L, EL, NJ, c.SR13, 2 * T), torch.bfloat16),
                "w2q": ((L, EL, NJ, T // 8, Hp), torch.int32),
                "w2s": ((L, EL, NJ, c.SR2, Hp), torch.bfloat16),
            }
        )
    elif c.w_dtype == "fp8":
        out.update(
            {
                "w13": ((L, EL, NJ, Hp, 2 * T), torch.float8_e4m3fn),
                "c13": ((L, EL, NJ, 1, 2 * T), torch.float32),
                "w2": ((L, EL, NJ, T, Hp), torch.float8_e4m3fn),
                "c2": ((L, EL, NJ, 1, Hp), torch.float32),
            }
        )
    else:
        raise ValueError(
            f"serving integration supports mxfp4/fp8 experts, not {c.w_dtype}"
        )
    return {k: out[k] for k in w_names(c)}


def per_rank_bytes(c: DecodeConfig) -> int:
    n = 0
    for shape, dt in per_rank_specs(c).values():
        numel = 1
        for s in shape:
            numel *= s
        n += numel * torch.empty((), dtype=dt).element_size()
    return n


# signed e2m1 value of each 4-bit code (sign bit 8)
_E2M1 = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ]
)


def mxfp4_dequant_kn(words: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Kernel MXFP4 layout -> f32 values: words ``[..., K/8, N]`` (int32/uint32 bits, code
    k of word r = row 8r+k) and bf16 scales ``[..., >=K/32, N]`` -> ``[..., K, N]``."""
    w = words.to(torch.int64) & 0xFFFFFFFF
    codes = torch.stack(
        [(w >> (4 * k)) & 15 for k in range(8)], dim=-2
    )  # [..., K/8, 8, N]
    codes = codes.reshape(*w.shape[:-2], w.shape[-2] * 8, w.shape[-1])
    K = codes.shape[-2]
    s = scales[..., : K // 32, :].float().repeat_interleave(32, dim=-2)
    return _E2M1.to(codes.device)[codes] * s


def fp8_pack_kn(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
    """``[..., K, N]`` f32 -> (e4m3 ``[..., K, N]``, f32 ``[..., 1, N]``, inexact count) with
    ``megakernel.fp8_pack``'s rule: per-column scale 2^(floor(log2 colmax) - 7)."""
    colmax = w.abs().amax(dim=-2, keepdim=True)
    g = torch.where(
        colmax > 0,
        torch.floor(torch.log2(colmax.clamp_min(1e-38))) - 7,
        torch.zeros_like(colmax),
    )
    f = torch.exp2(g)
    w8 = (w / f).to(torch.float8_e4m3fn)
    bad = int((w8.float() * f != w).sum())
    return w8, f, bad


def mxfp4_to_fp8(c: DecodeConfig, mx: dict[str, torch.Tensor], expert_block: int = 16):
    """Per-rank MXFP4 operands -> the fp8 expert operands (attention/router/biases shared).
    Returns (operands in ``w_names(fp8 config)``, number of inexact expert elements).
    Converts in (layer, expert-block) pieces into preallocated outputs so the host
    temporaries stay small (8 ranks convert at once on one host)."""
    import dataclasses

    c8 = dataclasses.replace(c, w_dtype="fp8")
    out = {k: mx[k] for k in AW_NAMES + ("b13", "b2")}
    for k, (shape, dt) in per_rank_specs(c8).items():
        if k not in out:
            out[k] = torch.empty(shape, dtype=dt)
    bad = 0
    for l in range(c.L):
        for e0 in range(0, c.EL, expert_block):
            e = slice(e0, min(e0 + expert_block, c.EL))
            for q, s, w8n, fn in (
                ("w13q", "w13s", "w13", "c13"),
                ("w2q", "w2s", "w2", "c2"),
            ):
                w8, f, b = fp8_pack_kn(mxfp4_dequant_kn(mx[q][l, e], mx[s][l, e]))
                out[w8n][l, e], out[fn][l, e] = w8, f
                bad += b
    return {k: out[k] for k in w_names(c8)}, bad


# ----------------------------------------------------------------------------- primitives


def e8m0_to_bf16(s: torch.Tensor) -> torch.Tensor:
    """uint8 e8m0 (2^(e-127)) -> exactly equal bf16. e=0 -> bf16 subnormal 2^-127;
    e=255 (NaN in the spec, never emitted for weights) is clamped to 2^127."""
    e = s.to(torch.int32).clamp(0, 254)
    bits = torch.where(e == 0, torch.full_like(e, 0x0040), e << 7)
    return bits.to(torch.int16).view(torch.bfloat16)


def bytes_to_words_t(rows: torch.Tensor) -> torch.Tensor:
    """``[..., N, K/2]`` uint8 (2 codes per byte along K) -> ``[..., K/8, N]`` int32 words."""
    assert rows.dtype == torch.uint8 and rows.shape[-1] % 4 == 0
    return rows.contiguous().view(torch.int32).transpose(-1, -2)


def _pad(x: torch.Tensor, shape, value=0.0) -> torch.Tensor:
    pad = []
    for want, have in zip(reversed(shape), reversed(x.shape)):
        assert want >= have, (shape, tuple(x.shape))
        pad += [0, want - have]
    return F.pad(x, pad, value=value) if any(pad) else x


def _chunks13(c: DecodeConfig, g: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """gate/up ``[EL, rows, IL]`` -> ``[EL, NJ, rows, 2T]`` (chunk j = [gate_j | up_j])."""
    EL, rows = g.shape[0], g.shape[1]
    NJ, T = c.NJ, c.tne
    g = g.reshape(EL, rows, NJ, T)
    u = u.reshape(EL, rows, NJ, T)
    return torch.cat([g, u], -1).permute(0, 2, 1, 3).contiguous()


def expert_slice(c: DecodeConfig, rank: int) -> tuple[slice, int, int]:
    """(experts, col_start, col_stop) of the intermediate dim this rank holds."""
    if c.moe_mode == "tp":
        a, b = c.moe_cols(rank)
        return slice(0, c.E), a, b
    el = c.E // c.R
    return slice(rank * el, (rank + 1) * el), 0, c.I


# ----------------------------------------------------------------------------- experts


def pack_gate_up_blocks(
    c: DecodeConfig, rank: int, blocks: torch.Tensor
) -> torch.Tensor:
    """``gate_up_proj_blocks [E, 2I, H/32, 16]`` -> w13q ``[EL, NJ, Hp/8, 2T]`` int32."""
    ex, a, b = expert_slice(c, rank)
    E, I, H, Hp = c.E, c.I, c.H, c.Hp
    x = blocks.reshape(E, 2 * I, H // 2)[ex]
    EL = x.shape[0]
    g = bytes_to_words_t(x[:, 2 * a : 2 * b : 2])  # [EL, H/8, n]
    u = bytes_to_words_t(x[:, 2 * a + 1 : 2 * b : 2])
    g, u = _pad(g, (EL, Hp // 8, c.IL)), _pad(u, (EL, Hp // 8, c.IL))
    return _chunks13(c, g, u)


def pack_gate_up_scales(
    c: DecodeConfig, rank: int, scales: torch.Tensor
) -> torch.Tensor:
    """``gate_up_proj_scales [E, 2I, H/32]`` -> w13s ``[EL, NJ, SR13, 2T]`` bf16."""
    ex, a, b = expert_slice(c, rank)
    x = scales.reshape(c.E, 2 * c.I, c.H // 32)[ex]
    EL = x.shape[0]
    g = e8m0_to_bf16(x[:, 2 * a : 2 * b : 2]).transpose(1, 2)  # [EL, H/32, n]
    u = e8m0_to_bf16(x[:, 2 * a + 1 : 2 * b : 2]).transpose(1, 2)
    # padding scales multiply zero codes; 1.0 like megakernel.mxfp4_pack's row padding
    g, u = _pad(g, (EL, c.SR13, c.IL), 1.0), _pad(u, (EL, c.SR13, c.IL), 1.0)
    return _chunks13(c, g, u)


def pack_gate_up_bias(c: DecodeConfig, rank: int, bias: torch.Tensor) -> torch.Tensor:
    """``gate_up_proj_bias [E, 2I]`` -> b13 ``[EL, NJ, 1, 2T]`` f32."""
    ex, a, b = expert_slice(c, rank)
    x = bias.reshape(c.E, 2 * c.I)[ex].float()
    EL = x.shape[0]
    g = _pad(x[:, 2 * a : 2 * b : 2][:, None], (EL, 1, c.IL))
    u = _pad(x[:, 2 * a + 1 : 2 * b : 2][:, None], (EL, 1, c.IL))
    return _chunks13(c, g, u)


def pack_down_blocks(c: DecodeConfig, rank: int, blocks: torch.Tensor) -> torch.Tensor:
    """``down_proj_blocks [E, H, I/32, 16]`` -> w2q ``[EL, NJ, T/8, Hp]`` int32."""
    ex, a, b = expert_slice(c, rank)
    x = blocks.reshape(c.E, c.H, c.I // 2)[ex][:, :, a // 2 : b // 2]
    EL = x.shape[0]
    w = _pad(bytes_to_words_t(x), (EL, c.IL // 8, c.Hp))  # [EL, IL/8, Hp]
    return w.reshape(EL, c.NJ, c.tne // 8, c.Hp).contiguous()


def pack_down_scales(c: DecodeConfig, rank: int, scales: torch.Tensor) -> torch.Tensor:
    """``down_proj_scales [E, H, I/32]`` -> w2s ``[EL, NJ, SR2, Hp]`` bf16."""
    ex, a, b = expert_slice(c, rank)
    x = scales.reshape(c.E, c.H, c.I // 32)[ex][:, :, a // 32 : b // 32]
    EL = x.shape[0]
    s = _pad(e8m0_to_bf16(x).transpose(1, 2), (EL, c.IL // 32, c.Hp), 1.0)
    s = s.reshape(EL, c.NJ, c.tne // 32, c.Hp)
    return _pad(s, (EL, c.NJ, c.SR2, c.Hp), 1.0).contiguous()


def pack_down_bias(c: DecodeConfig, rank: int, bias: torch.Tensor) -> torch.Tensor:
    """``down_proj_bias [E, H]`` -> b2 ``[EL, 1, Hp]`` f32 (the kernel adds it on rank 0 only
    in TP mode, so every rank stores it)."""
    ex, _, _ = expert_slice(c, rank)
    x = bias.reshape(c.E, c.H)[ex].float()
    return _pad(x, (x.shape[0], c.Hp))[:, None].contiguous()


# ----------------------------------------------------------------------------- bf16 experts

# midpoints between the e2m1 magnitudes 0, .5, 1, 1.5, 2, 3, 4, 6 (ties -> lower code, as
# reference.mxfp4_quantize's argmin)
_FP4_MID = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])


def quantize_mxfp4_kn(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``[..., K, N]`` float (K % 32 == 0) -> (int32 words ``[..., K/8, N]``, bf16 scales
    ``[..., K/32, N]``) with ``reference.mxfp4_quantize``'s rule: scale
    2^(floor(log2 amax) - 2) per 32 K elements, nearest e2m1 code. Lossless on weights that
    are MXFP4-representable (e.g. a bf16 dequantization of the MXFP4 release)."""
    *lead, K, N = w.shape
    assert K % 32 == 0, K
    blk = w.float().reshape(*lead, K // 32, 32, N)
    amax = blk.abs().amax(-2)
    e = torch.where(
        amax > 0,
        torch.floor(torch.log2(amax.clamp_min(1e-38))) - 2,
        torch.full_like(amax, -126.0),
    ).clamp(-126, 127)
    scale = torch.exp2(e)
    x = (blk / scale[..., None, :]).clamp(-6.0, 6.0)
    code = torch.bucketize(x.abs(), _FP4_MID.to(x.device)).to(torch.int32)
    code = code | torch.where((x < 0) & (code > 0), 8, 0).to(torch.int32)
    code = code.reshape(*lead, K // 8, 8, N)
    words = torch.zeros((*lead, K // 8, N), dtype=torch.int64, device=w.device)
    for k in range(8):
        words |= code[..., k, :].to(torch.int64) << (4 * k)
    words = torch.where(words >= 2**31, words - 2**32, words).to(torch.int32)
    return words, scale.to(torch.bfloat16)


def pack_gate_up_bf16(c: DecodeConfig, rank: int, w: torch.Tensor):
    """HF bf16 ``gate_up_proj [E, H, 2I]`` (gate/up interleaved along the last dim) ->
    (w13q ``[EL, NJ, Hp/8, 2T]``, w13s ``[EL, NJ, SR13, 2T]``)."""
    ex, a, b = expert_slice(c, rank)
    x = w.reshape(c.E, c.H, 2 * c.I)[ex]
    EL = x.shape[0]
    gq, gs = quantize_mxfp4_kn(x[:, :, 2 * a : 2 * b : 2])
    uq, us = quantize_mxfp4_kn(x[:, :, 2 * a + 1 : 2 * b : 2])
    q = _chunks13(c, _pad(gq, (EL, c.Hp // 8, c.IL)), _pad(uq, (EL, c.Hp // 8, c.IL)))
    s = _chunks13(
        c, _pad(gs, (EL, c.SR13, c.IL), 1.0), _pad(us, (EL, c.SR13, c.IL), 1.0)
    )
    return q, s


def pack_down_bf16(c: DecodeConfig, rank: int, w: torch.Tensor):
    """HF bf16 ``down_proj [E, I, H]`` -> (w2q ``[EL, NJ, T/8, Hp]``, w2s
    ``[EL, NJ, SR2, Hp]``)."""
    ex, a, b = expert_slice(c, rank)
    x = w.reshape(c.E, c.I, c.H)[ex][:, a:b]
    EL = x.shape[0]
    q, s = quantize_mxfp4_kn(x)  # [EL, n/8, H], [EL, n/32, H]
    q = _pad(q, (EL, c.IL // 8, c.Hp)).reshape(EL, c.NJ, c.tne // 8, c.Hp)
    s = _pad(s, (EL, c.IL // 32, c.Hp), 1.0).reshape(EL, c.NJ, c.tne // 32, c.Hp)
    return q.contiguous(), _pad(s, (EL, c.NJ, c.SR2, c.Hp), 1.0).contiguous()


# ----------------------------------------------------------------------------- builder


class MkWeightBuilder:
    """Consume HF GPT-OSS checkpoint tensors one at a time into one rank's kernel weights.

    ``consume(name, tensor)`` returns True when the tensor was used (it is never modified).
    ``missing()`` lists (layer, key) pairs not seen; ``finalize()`` returns the outputs.
    """

    def __init__(self, c: DecodeConfig, rank: int, device: str | torch.device = "cpu"):
        import dataclasses

        c = dataclasses.replace(c, w_dtype="mxfp4")  # streams MXFP4; see mxfp4_to_fp8
        c.check()
        self.c, self.rank = c, rank
        self.out = {
            k: torch.zeros(shape, dtype=dt, device=device)
            for k, (shape, dt) in per_rank_specs(c).items()
        }
        self.out["vec"][:, V_BR, :128] = NEG  # router padding lanes never win top-k
        self.seen: set[tuple[int, str]] = set()

    def missing(self) -> list[tuple[int, str]]:
        return sorted(
            (l, k)
            for l in range(self.c.L)
            for k in REQUIRED_KEYS
            if (l, k) not in self.seen
        )

    def consume(self, name: str, tensor: torch.Tensor) -> bool:
        m = _LAYER_RE.match(name)
        if m is None:
            return False
        layer, key = int(m.group(1)), _KEYS.get(m.group(2))
        if key is None or layer >= self.c.L:
            return False
        t = tensor.detach()
        if t.device.type != "cpu":
            t = t.to("cpu")
        getattr(self, "_" + key)(layer, t)
        for k in _COVERS.get(key, (key,)):
            self.seen.add((layer, k))
        return True

    def finalize(self) -> dict[str, torch.Tensor]:
        miss = self.missing()
        if miss:
            raise ValueError(f"megakernel weights incomplete, missing e.g. {miss[:4]}")
        return self.out

    # ---- attention / router / norms (bf16 checkpoint tensors)
    def _q_w(self, l, w):  # [NQ*D, H]
        c, r = self.c, self.rank
        self.out["wqkv"][l, : c.H, : c.QWL] = (
            w[r * c.QWL : (r + 1) * c.QWL].t().to(torch.bfloat16)
        )

    def _k_w(self, l, w):  # [NKV*D, H]
        c, r = self.c, self.rank
        o = c.QWL
        self.out["wqkv"][l, : c.H, o : o + c.D] = (
            w[r * c.D : (r + 1) * c.D].t().to(torch.bfloat16)
        )

    def _v_w(self, l, w):
        c, r = self.c, self.rank
        o = c.QWL + 128
        self.out["wqkv"][l, : c.H, o : o + c.D] = (
            w[r * c.D : (r + 1) * c.D].t().to(torch.bfloat16)
        )

    def _q_b(self, l, b):
        c, r = self.c, self.rank
        self.out["vec"][l, V_BQKV, : c.QWL] = b[r * c.QWL : (r + 1) * c.QWL].float()

    def _k_b(self, l, b):
        c, r = self.c, self.rank
        o = c.QWL
        self.out["vec"][l, V_BQKV, o : o + c.D] = b[r * c.D : (r + 1) * c.D].float()

    def _v_b(self, l, b):
        c, r = self.c, self.rank
        o = c.QWL + 128
        self.out["vec"][l, V_BQKV, o : o + c.D] = b[r * c.D : (r + 1) * c.D].float()

    def _o_w(self, l, w):  # [H, NQ*D]
        c, r = self.c, self.rank
        self.out["wo"][l, :, : c.H] = (
            w[:, r * c.QWL : (r + 1) * c.QWL].t().to(torch.bfloat16)
        )

    def _o_b(self, l, b):
        self.out["vec"][l, V_BO, : self.c.H] = b.float()

    def _sinks(self, l, s):  # [NQ]
        c, r = self.c, self.rank
        sk = s[r * c.NQL : (r + 1) * c.NQL].float()
        self.out["sinks"][l] = sk[:, None].expand(c.NQL, 128)

    def _ln1(self, l, w):
        self.out["vec"][l, V_LN1, : self.c.H] = w.float()

    def _ln2(self, l, w):
        self.out["vec"][l, V_LN2, : self.c.H] = w.float()

    def _r_w(self, l, w):  # [E, H]
        self.out["wr"][l, : self.c.H, : self.c.E] = w.t().to(torch.bfloat16)

    def _r_b(self, l, b):
        self.out["vec"][l, V_BR, : self.c.E] = b.float()

    # ---- MXFP4 experts
    def _gu_blocks(self, l, x):
        self.out["w13q"][l] = pack_gate_up_blocks(self.c, self.rank, x)

    def _gu_scales(self, l, x):
        self.out["w13s"][l] = pack_gate_up_scales(self.c, self.rank, x)

    def _gu_bias(self, l, x):
        self.out["b13"][l] = pack_gate_up_bias(self.c, self.rank, x)

    def _dn_blocks(self, l, x):
        self.out["w2q"][l] = pack_down_blocks(self.c, self.rank, x)

    def _dn_scales(self, l, x):
        self.out["w2s"][l] = pack_down_scales(self.c, self.rank, x)

    def _dn_bias(self, l, x):
        self.out["b2"][l] = pack_down_bias(self.c, self.rank, x)

    def _gu_w(self, l, x):
        self.out["w13q"][l], self.out["w13s"][l] = pack_gate_up_bf16(
            self.c, self.rank, x
        )

    def _dn_w(self, l, x):
        self.out["w2q"][l], self.out["w2s"][l] = pack_down_bf16(self.c, self.rank, x)


def config_from_hf(
    hf_config,
    *,
    tp_size: int,
    moe_mode: str = "tp",
    w_dtype: str = "fp8",
    B: int = 1,
    **kw,
) -> Optional[DecodeConfig]:
    """Kernel geometry from a GPT-OSS HF config; pools/page tables are filled in at trace
    time (``adapter.trace_config``)."""
    head_dim = getattr(hf_config, "head_dim", None) or (
        hf_config.hidden_size // hf_config.num_attention_heads
    )
    sw = getattr(hf_config, "sliding_window", None) or 128
    c = DecodeConfig(
        B=B,
        L=hf_config.num_hidden_layers,
        H=hf_config.hidden_size,
        NQ=hf_config.num_attention_heads,
        NKV=hf_config.num_key_value_heads,
        D=head_dim,
        E=hf_config.num_local_experts,
        TOPK=hf_config.num_experts_per_tok,
        I=hf_config.intermediate_size,
        # sglang window_left = sliding_window - 1; the kernel counts the current token
        window=sw,
        eps=hf_config.rms_norm_eps,
        alpha=float(getattr(hf_config, "hidden_act_alpha", 1.702)),
        limit=float(getattr(hf_config, "swiglu_limit", 7.0)),
        R=tp_size,
        moe_mode=moe_mode,
        w_dtype=w_dtype,
        split_pools=True,
        **kw,
    )
    return c
