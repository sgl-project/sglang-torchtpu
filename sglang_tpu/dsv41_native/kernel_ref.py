"""Pure-torch versions of the DeepSeek-V4.1 tilelang kernels, same signatures and numerics.

Copied from the DeepSeek-V4.1-Flash reference port (third_party/dsv41-reference/kernel_ref.py in
tpu-infer-loop). Changes for serving:

- GEMM outputs are cast to OUT_DTYPE (bf16) instead of torch.get_default_dtype(). The reference
  sets bf16 as the process default; a served model runs outside the loader's dtype context.
- Weight scales may be stored as raw E8M0 bytes (uint8); `scale_f32` turns them into the same
  power-of-two fp32 values `.float()` gives for float8_e8m0fnu.
- The dequantized GEMM operands are bf16 instead of fp32. fp8/fp4 values times a power-of-two
  scale are exact in bf16, and the product is accumulated in fp32 and rounded to bf16 once, as
  the reference's fp32 matmul followed by the bf16 cast does. It halves the dequantized expert
  stacks, which are the largest temporaries of a forward.
- grouped_gemm quantizes its activation with fp32 scales (same power-of-two values).

Each function follows the matching kernel step by step: same amax floors, same power-of-2 scale
rounding, fp32 quantization math.
"""

from typing import Optional

import torch

OUT_DTYPE = torch.bfloat16
FP8_MAX = 448.0
FP4_MAX = 6.0
FP4_TABLE = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)


def _pow2_ceil(x: torch.Tensor) -> torch.Tensor:
    """2 ** ceil(log2(x)) for positive fp32 x, by the same bit manipulation as fast_round_scale."""
    bits = x.float().view(torch.int32)
    exp = (bits >> 23) & 0xFF
    man = bits & ((1 << 23) - 1)
    e = exp - 127 + (man != 0).to(torch.int32)
    return ((e + 127) << 23).view(torch.float32)


def scale_f32(s: torch.Tensor) -> torch.Tensor:
    """Scales as fp32. uint8 holds E8M0 bytes: byte b is 2 ** (b - 127), built from the exponent
    bits (byte 0, 2 ** -127, comes out as 0; no checkpoint scale is that small)."""
    if s.dtype == torch.uint8:
        return (s.to(torch.int32) << 23).view(torch.float32)
    return s.float()


def _blocks(x: torch.Tensor, block_size: int) -> torch.Tensor:
    return x.float().unflatten(-1, (-1, block_size))


def _fp4_round(x: torch.Tensor) -> torch.Tensor:
    """Round to the nearest e2m1 value, ties to even, as cvt.rn does. Input already clamped to +-6."""
    a = x.abs()
    q = torch.where(
        a <= 0.25,
        0.0,
        torch.where(
            a < 0.75,
            0.5,
            torch.where(
                a <= 1.25,
                1.0,
                torch.where(
                    a < 1.75,
                    1.5,
                    torch.where(
                        a <= 2.5,
                        2.0,
                        torch.where(a < 3.5, 3.0, torch.where(a <= 5.0, 4.0, 6.0)),
                    ),
                ),
            ),
        ),
    )
    return torch.where(
        x < 0, -q, q
    )  # copysign without aten::copysign; only -0 vs +0 differs


def act_quant(
    x: torch.Tensor,
    block_size: int = 128,
    scale_fmt: Optional[str] = None,
    scale_dtype: torch.dtype = torch.float32,
    inplace: bool = False,
) -> torch.Tensor:
    """Block-wise FP8 quantization. inplace=True does fused quant+dequant back to x's dtype."""
    N = x.size(-1)
    assert N % block_size == 0
    xb = _blocks(x, block_size)
    amax = xb.abs().amax(-1, keepdim=True).clamp_min(1e-4)
    s = _pow2_ceil(amax / FP8_MAX) if scale_fmt is not None else amax * (1 / FP8_MAX)
    q = (xb / s).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    if inplace:
        x.copy_((q.float() * s).flatten(-2).to(x.dtype))
        return x
    return q.flatten(-2), s.squeeze(-1).to(scale_dtype)


def fp4_act_quant(
    x: torch.Tensor,
    block_size: int = 32,
    inplace: bool = False,
    scale_dtype: torch.dtype = torch.float8_e8m0fnu,
) -> torch.Tensor:
    """FP4 with E8M0 scales for the indexer or E4M3 scales for compressed KV. Only the inplace
    quant+dequant form is used by the model, so that is the only one implemented."""
    assert scale_dtype in (torch.float8_e8m0fnu, torch.float8_e4m3fn)
    assert inplace, "kernel_ref.fp4_act_quant only implements inplace=True"
    N = x.size(-1)
    assert N % block_size == 0
    xb = _blocks(x, block_size)
    amax = xb.abs().amax(-1, keepdim=True)
    if scale_dtype == torch.float8_e4m3fn:
        amax = amax.clamp_min(FP4_MAX * 2**-9)
        s = (amax / FP4_MAX).to(torch.float8_e4m3fn).float()
    else:
        amax = amax.clamp_min(FP4_MAX * 2**-126)
        s = _pow2_ceil(amax / FP4_MAX)
    q = _fp4_round((xb / s).clamp(-FP4_MAX, FP4_MAX))
    x.copy_((q * s).flatten(-2).to(x.dtype))
    return x


def _dequant_rows(a: torch.Tensor, a_s: torch.Tensor, block_size: int) -> torch.Tensor:
    """[M, K] fp8 activation with one scale per row and K block -> bf16 (exact)."""
    K = a.size(-1)
    a = a.reshape(-1, K).float().unflatten(-1, (-1, block_size))
    return (a * scale_f32(a_s).reshape(a.size(0), -1, 1)).flatten(-2).to(OUT_DTYPE)


def fp8_gemm(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
    scale_dtype: torch.dtype = torch.float32,
    block_size: int = 128,
) -> torch.Tensor:
    """C[M,N] = A[M,K] @ B[N,K]^T with per-block FP8 scaling, fp32 accumulation."""
    assert block_size in (32, 128)
    K = a.size(-1)
    N = b.size(0)
    assert K % block_size == 0
    a_f = _dequant_rows(a, a_s, block_size)
    sb = (
        scale_f32(b_s)
        .repeat_interleave(block_size, 0)[:N]
        .repeat_interleave(block_size, 1)
    )
    b_f = (b.float() * sb).to(OUT_DTYPE)
    c = a_f @ b_f.t()
    return c.to(OUT_DTYPE).view(*a.size()[:-1], N)


def _decode_e2m1(n: torch.Tensor) -> torch.Tensor:
    """4-bit e2m1 code -> fp32, the values of FP4_TABLE computed from the bits, so the graph needs no
    host-side lookup table. Code 8 gives -0.0 where the table has 0.0."""
    sign, e, m = (n >> 3) & 1, (n >> 1) & 3, (n & 1).float()
    scale = torch.where(e == 1, 1.0, torch.where(e == 2, 2.0, 4.0)).float()
    mag = torch.where(e == 0, 0.5 * m, (1 + 0.5 * m) * scale)
    return torch.where(sign == 1, -mag, mag)


def _unpack_fp4(b: torch.Tensor) -> torch.Tensor:
    """[..., N, K//2] packed e2m1 pairs (float4_e2m1fn_x2 or uint8) -> [..., N, K] fp32. Low
    nibble first, as in convert.py."""
    raw = b.view(torch.uint8).to(torch.int32)
    vals = torch.stack(
        [_decode_e2m1(raw & 0x0F), _decode_e2m1((raw >> 4) & 0x0F)], dim=-1
    )
    return vals.flatten(-2)


def fp4_gemm(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
    scale_dtype: torch.dtype = torch.float32,
    act_block_size: int = 128,
) -> torch.Tensor:
    """C[M,N] = A_fp8[M,K] @ B_fp4[N,K]^T. B has one E8M0 scale per 32 along K."""
    assert act_block_size in (32, 128)
    K = a.size(-1)
    N = b.size(0)
    assert K % act_block_size == 0
    assert b_s.shape == (N, K // 32)
    a_f = _dequant_rows(a, a_s, act_block_size)
    b_f = (_unpack_fp4(b) * scale_f32(b_s).repeat_interleave(32, 1)).to(OUT_DTYPE)
    c = a_f @ b_f.t()
    return c.to(OUT_DTYPE).view(*a.size()[:-1], N)


def grouped_gemm(
    x: torch.Tensor,
    w: torch.Tensor,
    w_s: torch.Tensor,
    act_block_size: int = 128,
    scale_fmt: Optional[str] = "ue8m0",
    scale_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """C[E,M,N] = A[E,M,K] @ B_fp4[E,N,K]^T, one fp4_gemm per expert done as a single bmm. A is
    quantized to fp8 the same way linear() does before fp4_gemm."""
    q, s = act_quant(x, act_block_size, scale_fmt, scale_dtype)
    a_f = (
        q.float().unflatten(-1, (-1, act_block_size)) * scale_f32(s).unsqueeze(-1)
    ).flatten(-2)
    b_f = _unpack_fp4(w) * scale_f32(w_s).repeat_interleave(32, -1)
    return torch.bmm(a_f.to(OUT_DTYPE), b_f.to(OUT_DTYPE).transpose(1, 2)).to(OUT_DTYPE)


def sparse_attn(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Attention over gathered KV positions; index -1 is masked out. attn_sink joins the softmax
    denominator only. Same finite floor of -1e30 as the kernel, so a row with no valid index is 0."""
    b, s, h, d = q.size()
    topk = topk_idxs.size(-1)
    valid = topk_idxs != -1
    idx = topk_idxs.clamp_min(0).long()
    gathered = kv.gather(1, idx.view(b, s * topk, 1).expand(-1, -1, d)).view(
        b, s, topk, d
    )
    return sparse_attn_gathered(q, gathered, valid, attn_sink, softmax_scale)


def sparse_attn_gathered(
    q: torch.Tensor,
    gathered: torch.Tensor,
    valid: torch.Tensor,
    attn_sink: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """sparse_attn after the gather: q [b, s, h, d], gathered [b, s, topk, d], valid [b, s, topk]."""
    h = q.size(2)
    scores = (
        torch.einsum("bshd,bstd->bsht", q.float(), gathered.float()) * softmax_scale
    )
    scores = scores.masked_fill(~valid.unsqueeze(2), -torch.inf)
    smax = scores.amax(-1, keepdim=True).clamp_min(-1e30)
    p = torch.exp(scores - smax)
    denom = p.sum(-1, keepdim=True) + torch.exp(
        attn_sink.float().view(1, 1, h, 1) - smax
    )
    o = torch.einsum("bsht,bstd->bshd", p, gathered.float()) / denom
    return o.to(q.dtype)


def hc_split_sinkhorn(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
):
    hc = hc_mult
    m = mixes.float()
    pre = torch.sigmoid(m[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2 * torch.sigmoid(m[..., hc : 2 * hc] * hc_scale[1] + hc_base[hc : 2 * hc])
    comb = (m[..., 2 * hc :] * hc_scale[2] + hc_base[2 * hc :]).unflatten(-1, (hc, hc))
    comb = comb.softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb
