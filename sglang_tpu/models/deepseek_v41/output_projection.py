"""DeepSeek-V4.1 attention output projection, wo_a then wo_b, as one Pallas call.

``SGLANG_TPU_DSV41_FUSED_OPROJ=1`` routes the target model's ``MQALayer`` output
projection through ``sglang_tpu.kernels.dsv41_output_projection`` (see its header). The kernel keeps the wo_a result in VMEM and starts streaming
wo_b's weights while the last wo_a groups compute.

Numerics against the default path:
- wo_b is unchanged in substance: the default TPU FP8 linear already quantizes
  each input row to FP8 with an FP32 ``amax / 448`` scale against the
  per-channel FP8 weight (``apply_tpu_fp8_linear``).
- wo_a changes. The default runs a BF16 einsum against the dequantized
  checkpoint weight. The kernel quantizes each token's group row (8 heads x 512)
  to FP8 with a BF16 scale, against wo_a requantized per output channel from that
  BF16 weight (``quantize_wo_a``). ``reference_output_projection`` spells out the
  kernel's math in torch.

The BF16 wo_a weight is released after the requantization. Every token count
runs the kernel (``fused_any_tokens``: the vendored tiling, including the
``MAX_TILE < T < TWO_TILE_MIN`` counts the vendored code leaves to a two-kernel chain, e.g.
the 384-row bs64 verify). There is no per-call dequantization.

``SGLANG_TPU_DSV41_FUSED_OPROJ_MIN_TOKENS`` gates the kernel to forwards of at
least that many rows (e.g. 416: extend chunks only). Smaller forwards run the
untouched default path, so the BF16 wo_a is then kept (+32 MiB FP8 per layer).
"""

from __future__ import annotations

from typing import Callable, Optional

import torch

from sglang_tpu._logging import get_logger

logger = get_logger(__name__)

HEADS = 64
HEAD_DIM = 512
GROUPS = 8
RANK = 1024
HIDDEN = 5120
FP8_MAX = 448.0

_OP: Optional[Callable] = None
_ENABLED_ATTR = "_dsv41_fused_oproj"
_LOGGED: set = set()


def _log_engagement(rows: int, used: bool, path: str) -> None:
    """Once per traced row count and path: whether the fused kernel engaged.

    Runs in Python while the direct tracer builds a bucket; skipped under Dynamo,
    where a logger call breaks a fullgraph capture.
    """
    is_dynamo_compiling = getattr(torch.compiler, "is_dynamo_compiling", None)
    if is_dynamo_compiling is not None and is_dynamo_compiling():
        return
    key = (rows, path)
    if key in _LOGGED:
        return
    _LOGGED.add(key)
    logger.info(
        "SGLANG_TPU_DSV41_FUSED_OPROJ engagement: site=MQALayer.o_proj rows=%d "
        "used=%s path=%s",
        rows,
        used,
        path,
    )


def is_enabled() -> bool:
    from sglang_tpu.envs import envs as tpu_envs

    return bool(tpu_envs.SGLANG_TPU_DSV41_FUSED_OPROJ.get())


def layer_filter() -> Optional[set[int]]:
    """``SGLANG_TPU_DSV41_FUSED_OPROJ_LAYERS``: layer ids that use the kernel,
    as ``"3"``, ``"0,5,7"`` or ``"0-3,10"``. Empty means every eligible layer."""
    from sglang_tpu.envs import envs as tpu_envs

    raw = (tpu_envs.SGLANG_TPU_DSV41_FUSED_OPROJ_LAYERS.get() or "").strip()
    if not raw:
        return None
    ids: set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = (int(x) for x in part.split("-"))
            ids.update(range(lo, hi + 1))
        elif part:
            ids.add(int(part))
    return ids


def min_tokens() -> int:
    """``SGLANG_TPU_DSV41_FUSED_OPROJ_MIN_TOKENS``: forwards with fewer rows keep the
    default wo_a/wo_b path (and the BF16 wo_a is kept for them). 0: every forward."""
    from sglang_tpu.envs import envs as tpu_envs

    return max(int(tpu_envs.SGLANG_TPU_DSV41_FUSED_OPROJ_MIN_TOKENS.get()), 0)


def token_count_supported(tokens: int) -> bool:
    """Token counts routed to the kernel: every positive count.

    The vendored ``is_supported`` also excludes ``MAX_TILE < T < TWO_TILE_MIN``,
    only because a two-kernel chain is about as fast there (T384 fused
    64.3 us vs chain 63.3 us on v7). Our alternative there is a per-call FP8
    dequantization, so ``fused_any_tokens`` runs those counts as two tiles.
    """
    return tokens > 0


def fused_any_tokens(x, wa, sa, wb, sb):
    """``fused_output_projection`` for any token count: the vendored kernel's own
    tiling (``token_tiles`` + ``_call``) without its token-range exclusion.
    The weight/dtype contract is checked as in ``is_supported``."""
    from sglang_tpu.kernels import dsv41_output_projection as k

    if not (
        x.ndim == 3
        and x.shape[0] > 0
        and x.shape[1:] == (HEADS, HEAD_DIM)
        and wa.shape == (HEADS * HEAD_DIM // GROUPS, GROUPS * RANK)
        and sa.shape == (GROUPS * RANK,)
        and wb.shape == (GROUPS * RANK, HIDDEN)
        and sb.shape == (HIDDEN,)
        and str(x.dtype) == "bfloat16"
        and str(wa.dtype) == "float8_e4m3fn"
        and str(wb.dtype) == "float8_e4m3fn"
    ):
        raise ValueError("Fused output projection requires the DSv4.1 v7 contract")
    total = x.shape[0]
    sa = sa.reshape(1, wa.shape[1])
    count, tokens, rest = k.token_tiles(total)
    out = None
    if count:
        out = k._call(x, wa, sa, wb, sb, None, tokens=tokens, count=count, start=None)
    if rest:
        start = count * tokens
        out = k._call(x, wa, sa, wb, sb, out, tokens=rest, count=1, start=start)
    return out[:total] if out.shape[0] != total else out


def register_op() -> Callable:
    """Build the torch op once; call before the model is traced."""
    global _OP
    if _OP is not None:
        return _OP

    import jax

    from sglang_tpu.pallas_jax_op import cpu_export_jax_op

    def _op(x, wa, sa, wb, sb):
        return fused_any_tokens(x, wa, sa, wb, sb)

    # torch_tpu's signature check needs annotation objects; this module's
    # `from __future__ import annotations` would leave strings.
    _op.__annotations__ = {
        **{name: jax.Array for name in ("x", "wa", "sa", "wb", "sb")},
        "return": jax.Array,
    }

    op = cpu_export_jax_op("sglang_tpu_pallas::dsv41_fused_oproj", _op)

    def _fake(x, wa, sa, wb, sb):
        del wa, sa, sb
        return x.new_empty((x.shape[0], wb.shape[1]))

    op.register_fake(_fake)
    _OP = op
    return op


# --------------------------------------------------------------------------- #
# Weights.
# --------------------------------------------------------------------------- #
def quantize_wo_a(weight: torch.Tensor, groups: int = GROUPS):
    """BF16 wo_a ``[G * R, D]`` (core's grouped layout, viewed ``[G, R, D]``) to
    the kernel's FP8 ``[D, G * R]`` with an FP32 per-output-channel scale ``[G * R]``.

    Column ``g * R + r`` of the result is group ``g``'s output channel ``r``, so
    group ``g``'s columns are the contiguous slice the kernel reads for it.
    """
    gr, d = weight.shape
    w = weight.view(groups, gr // groups, d).permute(2, 0, 1).reshape(d, gr)
    w = w.float()
    scale = torch.clamp(w.abs().amax(dim=0) / FP8_MAX, min=1e-12)
    # The permute+reshape is a transposed view, and the elementwise ops keep its
    # strides: materialize [D, G * R] row-major for the kernel.
    wa = torch.clamp(w / scale, -FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).contiguous()
    return wa, scale.contiguous()


def dequantize_wo_a(wa: torch.Tensor, sa: torch.Tensor, groups: int = GROUPS):
    """Inverse layout of ``quantize_wo_a``: BF16 ``[G, R, D]`` for the einsum."""
    d, gr = wa.shape
    w = (wa.float() * sa.reshape(1, -1)).to(torch.bfloat16)
    return w.reshape(d, groups, gr // groups).permute(1, 2, 0)


def reference_output_projection(
    x: torch.Tensor,
    wa: torch.Tensor,
    sa: torch.Tensor,
    wb: torch.Tensor,
    sb: torch.Tensor,
) -> torch.Tensor:
    """The fused kernel's arithmetic in torch (CPU tests).

    wo_a: per token and group, the 8 x 512 row quantizes to FP8 with a BF16
    inverse scale ``448 / max(amax, 1e-30)``; the FP32 product divides by that
    inverse, takes the channel scale and rounds to BF16. wo_b: each BF16 row of
    8192 quantizes with the FP32 scale ``amax / 448`` (1 for an all-zero row),
    clipped to +-448; the FP32 product takes the row and channel scales.
    """
    t, heads, head_dim = x.shape
    groups = heads // (wa.shape[0] // head_dim)
    width = wa.shape[1] // groups
    xg = x.reshape(t, groups, -1)
    ys = []
    for g in range(groups):
        v = xg[:, g]
        amax = v.abs().amax(dim=1, keepdim=True)
        inv = (FP8_MAX / torch.clamp(amax, min=1e-30)).to(torch.bfloat16)
        q = (v * inv).to(torch.float8_e4m3fn)
        part = q.float() @ wa[:, g * width : (g + 1) * width].float()
        part = part * (1.0 / inv.float())
        ys.append(
            (part * sa[g * width : (g + 1) * width].reshape(1, -1)).to(torch.bfloat16)
        )
    y = torch.cat(ys, dim=1)
    s = y.float().abs().amax(dim=1, keepdim=True) * (1.0 / FP8_MAX)
    xs = torch.where(s == 0, torch.ones_like(s), s)
    qb = torch.clamp(y.float() / xs, -FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    z = qb.float() @ wb.float()
    return ((z * xs) * sb.reshape(1, -1).float()).to(x.dtype)


# --------------------------------------------------------------------------- #
# Preparation (after the quant methods' weight processing).
# --------------------------------------------------------------------------- #
def _is_target_model(model: torch.nn.Module) -> bool:
    return type(model).__name__ == "DeepseekV4ForCausalLM"


def _is_mqa_layer(module: torch.nn.Module) -> bool:
    return type(module).__name__ == "MQALayer" and hasattr(module, "_project_wo_b")


def unmet_layer_requirements(attn: torch.nn.Module) -> list[str]:
    """Why ``attn`` cannot take the kernel; empty when it can."""
    unmet = []
    if not getattr(attn, "is_dsv41", False):
        unmet.append("not a V4.1 layer")
    if getattr(attn, "use_mega_attention", False):
        # _forward_mega computes wo_a itself and hands _project_wo_b a tensor.
        unmet.append("mega attention path")
    if getattr(attn, "attn_tp_size", None) != 1:
        unmet.append(f"attn_tp_size {getattr(attn, 'attn_tp_size', None)} != 1")
    if (attn.n_local_heads, attn.n_local_groups, attn.o_lora_rank) != (
        HEADS,
        GROUPS,
        RANK,
    ):
        unmet.append(
            "heads/groups/rank "
            f"{(attn.n_local_heads, attn.n_local_groups, attn.o_lora_rank)} != "
            f"{(HEADS, GROUPS, RANK)}"
        )
    wo_a = getattr(getattr(attn, "wo_a", None), "weight", None)
    if (
        wo_a is None
        or wo_a.dtype != torch.bfloat16
        or tuple(wo_a.shape)
        != (
            GROUPS * RANK,
            HEADS * HEAD_DIM // GROUPS,
        )
    ):
        unmet.append(
            "wo_a must be BF16 [8192, 4096], got "
            f"{None if wo_a is None else (wo_a.dtype, tuple(wo_a.shape))}"
        )
    wo_b = getattr(attn, "wo_b", None)
    wb = getattr(wo_b, "weight", None)
    sb = getattr(wo_b, "weight_scale", None)
    if (
        wb is None
        or wb.dtype != torch.float8_e4m3fn
        or tuple(wb.shape)
        != (
            GROUPS * RANK,
            HIDDEN,
        )
    ):
        unmet.append(
            "wo_b must be FP8 [K=8192, N=5120], got "
            f"{None if wb is None else (wb.dtype, tuple(wb.shape))}"
        )
    if sb is None or sb.dtype != torch.float32 or sb.numel() != HIDDEN:
        unmet.append(
            "wo_b scale must be FP32 per channel [5120], got "
            f"{None if sb is None else (sb.dtype, tuple(sb.shape))}"
        )
    if getattr(wo_b, "bias", None) is not None:
        unmet.append("wo_b has a bias")
    return unmet


def _release(param: torch.nn.Parameter) -> None:
    """Point ``param`` at an empty tensor and free its old device buffer."""
    old = param.data
    param.data = torch.empty(0, dtype=old.dtype, device=old.device)
    storage = old.untyped_storage()
    if storage.data_ptr() != param.data.untyped_storage().data_ptr():
        try:
            storage.resize_(0)
        except RuntimeError:
            pass
    del old


def prepare_layer(attn: torch.nn.Module, *, keep_bf16: bool = False) -> None:
    """Requantize ``attn``'s wo_a for the kernel and release the BF16 copy, unless
    ``keep_bf16`` (forwards below ``min_tokens`` still run the default wo_a)."""
    wa, sa = quantize_wo_a(attn.wo_a.weight.data)
    attn.dsv41_oproj_wa = torch.nn.Parameter(wa, requires_grad=False)
    attn.dsv41_oproj_sa = torch.nn.Parameter(sa.contiguous(), requires_grad=False)
    # A plain [N] buffer built from a host copy of wo_b's [N, 1] scale, with nothing
    # derived on device. A view argument to a compiled program costs ~19 us per call,
    # and under DEFER_AND_FUSE a device-side reshape + clone stays a deferred
    # as_strided over the [N, 1] scale that re-runs as its own eager program before
    # every verify (~18 per c4 cycle).
    scale = attn.wo_b.weight_scale.data
    host = scale.to("cpu", torch.float32, copy=True).reshape(-1)
    attn.dsv41_oproj_sb = torch.nn.Parameter(host.to(scale.device), requires_grad=False)
    if not keep_bf16:
        _release(attn.wo_a.weight)
    # The FP8 wo_a and the copied wo_b scale are derived at load; an in-place
    # weight update would not refresh them (and a released wo_a cannot take one).
    attn._derived_weight_cache_error = (
        "Online weight updates are not supported with SGLANG_TPU_DSV41_FUSED_OPROJ=1: "
        "the fused output projection keeps load-time FP8 copies of wo_a and the wo_b "
        "scale. Restart without it to update weights."
    )
    setattr(attn, _ENABLED_ATTR, True)


def prepare_model(model: torch.nn.Module) -> int:
    """Prepare every eligible layer of the target model; returns the count."""
    if not is_enabled() or not _is_target_model(model):
        return 0
    from sglang_tpu.attention.pallas import util as pallas_util

    # The vendored kernel's contract (and its tiling/VMEM budget) is v7's.
    version = pallas_util.get_tpu_version()
    if version != 7:
        logger.info(
            "SGLANG_TPU_DSV41_FUSED_OPROJ: TPU v%s is not the kernel's v7; keeping "
            "the default output projection.",
            version,
        )
        return 0
    wanted = layer_filter()
    layers = [m for m in model.modules() if _is_mqa_layer(m)]
    prepared, skipped = 0, {}
    for attn in layers:
        if wanted is not None and attn.layer_id not in wanted:
            continue
        unmet = unmet_layer_requirements(attn)
        if unmet:
            skipped[attn.layer_id] = unmet
            continue
        prepare_layer(attn, keep_bf16=min_tokens() > 0)
        prepared += 1
    if prepared:
        register_op()
    logger.info(
        "SGLANG_TPU_DSV41_FUSED_OPROJ: %d of %d attention layers use the fused "
        "wo_a+wo_b kernel%s%s.",
        prepared,
        len(layers),
        "" if wanted is None else f" (layer filter {sorted(wanted)})",
        f" for forwards of >= {min_tokens()} rows (BF16 wo_a kept)"
        if min_tokens()
        else "",
    )
    if skipped:
        first = next(iter(skipped.items()))
        logger.info(
            "SGLANG_TPU_DSV41_FUSED_OPROJ: %d layers keep the default path, "
            "e.g. layer %s: %s",
            len(skipped),
            first[0],
            "; ".join(first[1]),
        )
    return prepared


# --------------------------------------------------------------------------- #
# Forward.
# --------------------------------------------------------------------------- #
class PreWoA:
    """The inverse-RoPE'd attention output ``[T, 64, 512]`` that
    ``MQALayer._project_wo_a`` hands to ``_project_wo_b`` for a prepared layer."""

    __slots__ = ("o",)

    def __init__(self, o: torch.Tensor) -> None:
        self.o = o


def project_wo_a(
    original, attn, o, *, positions, forward_batch, inv_rope, wo_a_quant_freqs
):
    """AROUND MQALayer._project_wo_a: for a prepared layer, apply the inverse RoPE
    as the default TPU path does and defer wo_a to ``project_wo_b``."""
    if getattr(attn, _ENABLED_ATTR, False) and o.shape[0] < min_tokens():
        _log_engagement(int(o.shape[0]), False, "below_min_tokens")
    if not getattr(attn, _ENABLED_ATTR, False) or o.shape[0] < min_tokens():
        return original(
            attn,
            o,
            positions=positions,
            forward_batch=forward_batch,
            inv_rope=inv_rope,
            wo_a_quant_freqs=wo_a_quant_freqs,
        )
    if inv_rope is not None or wo_a_quant_freqs is not None:
        raise RuntimeError(
            "SGLANG_TPU_DSV41_FUSED_OPROJ: unexpected inv_rope / wo_a_quant_freqs "
            "for a prepared layer (HIP-only arguments)"
        )
    from sglang.srt.model_executor.forward_context import get_attn_backend

    backend = get_attn_backend()
    if not hasattr(backend, "apply_mqa_inv_rope"):
        raise RuntimeError(
            "SGLANG_TPU_DSV41_FUSED_OPROJ needs an attention backend with "
            f"apply_mqa_inv_rope; got {type(backend).__name__}"
        )
    return PreWoA(backend.apply_mqa_inv_rope(attn, o, positions))


def project_wo_b(original, attn, o, defer_all_reduce: bool = False):
    """AROUND MQALayer._project_wo_b: run wo_a + wo_b for a ``PreWoA``."""
    if not isinstance(o, PreWoA):
        return original(attn, o, defer_all_reduce)
    x = o.o
    tokens = x.shape[0]
    if tokens == 0:
        return x.new_zeros((0, HIDDEN))
    if x.dtype != torch.bfloat16:
        # The kernel's contract is BF16 activations (the attention output already is).
        _log_engagement(int(tokens), True, "kernel_cast_bf16")
        x = x.to(torch.bfloat16)
    else:
        _log_engagement(int(tokens), True, "kernel")
    return register_op()(
        x.contiguous(),
        attn.dsv41_oproj_wa,
        attn.dsv41_oproj_sa,
        attn.wo_b.weight,
        attn.dsv41_oproj_sb,
    )
