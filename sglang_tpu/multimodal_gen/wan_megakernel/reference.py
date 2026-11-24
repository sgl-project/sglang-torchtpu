# SPDX-License-Identifier: Apache-2.0
"""Wan2.2-A14B's DiT blocks in plain JAX: the megakernel's specification.

One block, as diffusers' ``WanTransformerBlock`` runs it with bf16 weights
(``mode="bf16"``): norms, modulation, residuals, RoPE and softmax in f32,
every value that diffusers keeps in bf16 rounded at the same point:

    n1 = bf16(LN(x) * (1 + scale) + shift)
    q, k, v = bf16(n1 W + b); q, k = bf16(rms(q) w), bf16(rms(k) w)   # rms over all heads
    q, k = bf16(rope(q)), bf16(rope(k));  a = bf16(attention(q, k, v))
    x = bf16(x + bf16(a Wo + bo) * gate)
    n2 = bf16(LN(x) w2 + b2);  q2 = bf16(rms(bf16(n2 Wq2 + bq2)) w)
    x = x + bf16(bf16(attention(q2, k2, v2)) Wo2 + bo2)                # a bf16 add
    n3 = bf16(LN(x) * (1 + c_scale) + c_shift)
    x = bf16(x + bf16(bf16(gelu(bf16(n3 W1 + b1))) W2 + b2) * c_gate)

``k2, v2`` (``cross_kv``) depend only on the text and the layer, so the
caller computes them once per request. ``mode="f32"`` drops the
intermediate roundings (an error yardstick).

The kernel's weight layout (``pack_layer``) differs from the checkpoint's
in two exact ways: projections are ``[in, out]``, and within every head the
q and k columns (bias, norm weight) are reordered even dims first, then odd
dims, so RoPE's adjacent pairs ``(2i, 2i+1)`` become ``(i, i + 64)`` (a lane
roll instead of a stride-2 shuffle). q.k and the rms over all heads are
unchanged by the reorder; v and the output projection are untouched.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping

import jax
import jax.numpy as jnp
import numpy as np

F32, BF16 = jnp.float32, jnp.bfloat16

# The six modulation rows of a block (``scale_shift_table + temb`` chunks).
SHIFT, SCALE, GATE, C_SHIFT, C_SCALE, C_GATE = range(6)


@dataclasses.dataclass(frozen=True)
class Config:
    layers: int = 40
    hidden: int = 5120
    heads: int = 40
    head_dim: int = 128
    ffn: int = 13824
    text_len: int = 512
    eps: float = 1e-6
    rope_theta: float = 10000.0

    @property
    def rope_axes(self) -> tuple[int, int, int]:
        """Head dims rotated by frame, height and width (Wan: 44, 42, 42)."""
        d = self.head_dim
        return (d - 4 * (d // 6), 2 * (d // 6), 2 * (d // 6))


WAN22_A14B = Config()


def small_config(layers=2, heads=4, ffn=1024, text_len=64) -> Config:
    """A test-sized stack (head_dim stays 128)."""
    return Config(
        layers=layers, hidden=heads * 128, heads=heads, ffn=ffn, text_len=text_len
    )


# ----------------------------------------------------------------- RoPE


def rope_angles(
    frames: int, height: int, width: int, cfg: Config = WAN22_A14B
) -> np.ndarray:
    """``[F*H*W, head_dim/2]`` float64 rotation angles, row-major (f, h, w) tokens.

    Pair ``i`` of a head rotates by ``pos_axis * theta^(-2j/d_axis)``: the
    frame pairs first, then height, then width, as Wan's ``rope_params``.
    """
    cols = []
    for size, d in zip((frames, height, width), cfg.rope_axes):
        inv = 1.0 / np.power(cfg.rope_theta, np.arange(0, d, 2, dtype=np.float64) / d)
        cols.append(np.outer(np.arange(size, dtype=np.float64), inv))
    f, h, w = np.meshgrid(
        np.arange(frames), np.arange(height), np.arange(width), indexing="ij"
    )
    f, h, w = f.reshape(-1), h.reshape(-1), w.reshape(-1)
    return np.concatenate([cols[0][f], cols[1][h], cols[2][w]], axis=1)


def rope_tables(
    angles: np.ndarray, rows: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """``(cos, sin)`` ``[rows, head_dim]`` f32 in the kernel's split layout.

    With q and k reordered (``qk_order``), ``rope(x) = x * cos + roll(x, D/2) * sin``
    where ``cos = [c, c]`` and ``sin = [-s, s]``. Rows past the angles (padding)
    rotate by nothing.
    """
    tokens, half = angles.shape
    rows = tokens if rows is None else rows
    cos = np.ones((rows, 2 * half), np.float64)
    sin = np.zeros((rows, 2 * half), np.float64)
    c, s = np.cos(angles), np.sin(angles)
    cos[:tokens] = np.concatenate([c, c], axis=1)
    sin[:tokens] = np.concatenate([-s, s], axis=1)
    return cos.astype(np.float32), sin.astype(np.float32)


def qk_order(cfg: Config = WAN22_A14B) -> np.ndarray:
    """Column order of q and k: within each head, even dims then odd dims."""
    d = cfg.head_dim
    one = np.concatenate([np.arange(0, d, 2), np.arange(1, d, 2)])
    return np.concatenate([h * d + one for h in range(cfg.heads)])


def rope_split(x: jax.Array, cos: jax.Array, sin: jax.Array) -> jax.Array:
    """``x [S, heads, D]`` f32 in the reordered layout, rotated."""
    half = x.shape[-1] // 2
    rolled = jnp.concatenate([x[..., half:], x[..., :half]], axis=-1)
    return x * cos[:, None, :] + rolled * sin[:, None, :]


# -------------------------------------------------------------- weights

# Per-layer checkpoint tensors (diffusers names under ``blocks.{i}.``).
CHECKPOINT_KEYS = (
    "scale_shift_table",
    "attn1.to_q.weight",
    "attn1.to_q.bias",
    "attn1.to_k.weight",
    "attn1.to_k.bias",
    "attn1.to_v.weight",
    "attn1.to_v.bias",
    "attn1.to_out.0.weight",
    "attn1.to_out.0.bias",
    "attn1.norm_q.weight",
    "attn1.norm_k.weight",
    "norm2.weight",
    "norm2.bias",
    "attn2.to_q.weight",
    "attn2.to_q.bias",
    "attn2.to_k.weight",
    "attn2.to_k.bias",
    "attn2.to_v.weight",
    "attn2.to_v.bias",
    "attn2.to_out.0.weight",
    "attn2.to_out.0.bias",
    "attn2.norm_q.weight",
    "attn2.norm_k.weight",
    "ffn.net.0.proj.weight",
    "ffn.net.0.proj.bias",
    "ffn.net.2.weight",
    "ffn.net.2.bias",
)

# The kernel's per-layer arrays (``pack_layer``), stacked on a leading layer axis.
WEIGHT_NAMES = (
    "wqkv",
    "bqkv",
    "qn",
    "kn",
    "wo",
    "bo",
    "n2w",
    "n2b",
    "wq2",
    "bq2",
    "qn2",
    "wo2",
    "bo2",
    "w1",
    "b1",
    "w2",
    "b2",
)
# Cross-attention K/V projections, used only by ``cross_kv``.
CROSS_KV_NAMES = ("wk2", "bk2", "kn2", "wv2", "bv2")


def pack_layer(
    raw: Mapping[str, np.ndarray], cfg: Config = WAN22_A14B, dtype=np.float32
) -> dict[str, np.ndarray]:
    """One block's checkpoint tensors (``CHECKPOINT_KEYS``, any float dtype) in
    the kernel's layout: matrices ``[in, out]`` in ``dtype`` (cast to bf16 by
    the caller), vectors ``[1, n]`` f32. Also returns the block's
    ``scale_shift_table`` as ``"table"`` ``[6, H]`` f32 and ``CROSS_KV_NAMES``."""
    order = qk_order(cfg)

    def mat(name):
        return np.asarray(raw[name], np.float32).T  # [out, in] -> [in, out]

    def vec(name):
        return np.asarray(raw[name], np.float32).reshape(1, -1)

    wq, wk, wv = (
        mat("attn1.to_q.weight")[:, order],
        mat("attn1.to_k.weight")[:, order],
        mat("attn1.to_v.weight"),
    )
    bq, bk, bv = (
        vec("attn1.to_q.bias")[:, order],
        vec("attn1.to_k.bias")[:, order],
        vec("attn1.to_v.bias"),
    )
    out = {
        "wqkv": np.concatenate([wq, wk, wv], axis=1).astype(dtype),
        "bqkv": np.concatenate([bq, bk, bv], axis=1),
        "qn": vec("attn1.norm_q.weight")[:, order],
        "kn": vec("attn1.norm_k.weight")[:, order],
        "wo": mat("attn1.to_out.0.weight").astype(dtype),
        "bo": vec("attn1.to_out.0.bias"),
        "n2w": vec("norm2.weight"),
        "n2b": vec("norm2.bias"),
        "wq2": mat("attn2.to_q.weight").astype(dtype),
        "bq2": vec("attn2.to_q.bias"),
        "qn2": vec("attn2.norm_q.weight"),
        "wo2": mat("attn2.to_out.0.weight").astype(dtype),
        "bo2": vec("attn2.to_out.0.bias"),
        "w1": mat("ffn.net.0.proj.weight").astype(dtype),
        "b1": vec("ffn.net.0.proj.bias"),
        "w2": mat("ffn.net.2.weight").astype(dtype),
        "b2": vec("ffn.net.2.bias"),
        "wk2": mat("attn2.to_k.weight").astype(dtype),
        "bk2": vec("attn2.to_k.bias"),
        "kn2": vec("attn2.norm_k.weight"),
        "wv2": mat("attn2.to_v.weight").astype(dtype),
        "bv2": vec("attn2.to_v.bias"),
        "table": np.asarray(raw["scale_shift_table"], np.float32).reshape(
            6, cfg.hidden
        ),
    }
    return out


def random_layer(cfg: Config, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Checkpoint-shaped random tensors with Wan-like scales (tests, benchmarks)."""
    h, f = cfg.hidden, cfg.ffn

    def w(o, i):
        return (rng.standard_normal((o, i)) / math.sqrt(i)).astype(np.float32)

    def b(n, s=0.02):
        return (rng.standard_normal(n) * s).astype(np.float32)

    def g(n):
        return (1.0 + 0.2 * rng.standard_normal(n)).astype(np.float32)

    raw = {
        "scale_shift_table": (rng.standard_normal((1, 6, h)) / math.sqrt(h)).astype(
            np.float32
        )
    }
    for a in ("attn1", "attn2"):
        for p in ("to_q", "to_k", "to_v", "to_out.0"):
            raw[f"{a}.{p}.weight"], raw[f"{a}.{p}.bias"] = w(h, h), b(h)
        raw[f"{a}.norm_q.weight"], raw[f"{a}.norm_k.weight"] = g(h), g(h)
    raw["norm2.weight"], raw["norm2.bias"] = g(h), b(h)
    raw["ffn.net.0.proj.weight"], raw["ffn.net.0.proj.bias"] = w(f, h), b(f)
    raw["ffn.net.2.weight"], raw["ffn.net.2.bias"] = w(h, f), b(h)
    return raw


def stack_layers(layers: list[Mapping[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {k: np.stack([one[k] for one in layers]) for k in layers[0]}


# ----------------------------------------------------------------- math


def _round(x: jax.Array, mode: str) -> jax.Array:
    return x.astype(BF16).astype(F32) if mode == "bf16" else x


def layer_norm(x: jax.Array, eps: float) -> jax.Array:
    mean = jnp.mean(x, axis=-1, keepdims=True)
    centered = x - mean
    var = jnp.mean(centered * centered, axis=-1, keepdims=True)
    return centered * jax.lax.rsqrt(var + eps)


def rms_norm(x: jax.Array, eps: float) -> jax.Array:
    return x * jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + eps)


def gelu_tanh(x: jax.Array) -> jax.Array:
    return (
        0.5
        * x
        * (1.0 + jnp.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x * x * x)))
    )


def attention(
    q: jax.Array, k: jax.Array, v: jax.Array, keys: int | None = None
) -> jax.Array:
    """Exact softmax attention in f32: ``[S, heads, D]`` -> ``[S, heads, D]``;
    keys from ``keys`` on are masked out."""
    scale = 1.0 / math.sqrt(q.shape[-1])
    s = (
        jnp.einsum("qhd,khd->hqk", q.astype(F32), k.astype(F32), precision="highest")
        * scale
    )
    if keys is not None and keys < k.shape[0]:
        s = jnp.where(jnp.arange(k.shape[0])[None, None, :] < keys, s, -jnp.inf)
    p = jax.nn.softmax(s, axis=-1)
    return jnp.einsum("hqk,khd->qhd", p, v.astype(F32), precision="highest")


def _mm(a: jax.Array, w: jax.Array) -> jax.Array:
    return jnp.dot(a, w.astype(F32), precision="highest")


def cross_kv(
    ctx: jax.Array,
    w: Mapping[str, jax.Array],
    cfg: Config = WAN22_A14B,
    mode: str = "bf16",
):
    """``(k2, v2)`` ``[T_ctx, H]`` of one layer: the text side of cross-attention."""
    c = _round(ctx.astype(F32), mode)
    k = _round(_mm(c, w["wk2"]) + w["bk2"], mode)
    k = _round(rms_norm(k, cfg.eps) * w["kn2"], mode)
    v = _round(_mm(c, w["wv2"]) + w["bv2"], mode)
    return k, v


def block(
    x,
    e,
    k2,
    v2,
    cos,
    sin,
    w,
    cfg: Config = WAN22_A14B,
    *,
    keys: int | None = None,
    mode: str = "bf16",
):
    """One block on ``x [S, H]``; ``e [6, H]`` f32 modulation rows, ``k2, v2
    [T_ctx, H]``, ``cos, sin [S, D]`` (``rope_tables``), ``w`` one layer of
    ``pack_layer``. Rows from ``keys`` on are padding: masked as keys."""
    h, heads, d = cfg.hidden, cfg.heads, cfg.head_dim
    r = lambda v: _round(v, mode)  # noqa: E731
    x = r(x.astype(F32))
    e = e.astype(F32)
    n1 = r(layer_norm(x, cfg.eps) * (1.0 + e[SCALE]) + e[SHIFT])
    qkv = r(_mm(n1, w["wqkv"]) + w["bqkv"])
    q, k, v = qkv[:, :h], qkv[:, h : 2 * h], qkv[:, 2 * h :]
    q = r(rms_norm(q, cfg.eps) * w["qn"])
    k = r(rms_norm(k, cfg.eps) * w["kn"])
    q = r(rope_split(q.reshape(-1, heads, d), cos, sin))
    k = r(rope_split(k.reshape(-1, heads, d), cos, sin))
    a = r(attention(q, k, v.reshape(-1, heads, d), keys).reshape(-1, h))
    o = r(_mm(a, w["wo"]) + w["bo"])
    x = r(x + o * e[GATE])
    n2 = r(layer_norm(x, cfg.eps) * w["n2w"] + w["n2b"])
    q2 = r(_mm(n2, w["wq2"]) + w["bq2"])
    q2 = r(rms_norm(q2, cfg.eps) * w["qn2"])
    a2 = r(
        attention(
            q2.reshape(-1, heads, d), k2.reshape(-1, heads, d), v2.reshape(-1, heads, d)
        ).reshape(-1, h)
    )
    o2 = r(_mm(a2, w["wo2"]) + w["bo2"])
    x = r(x + o2)
    n3 = r(layer_norm(x, cfg.eps) * (1.0 + e[C_SCALE]) + e[C_SHIFT])
    f = r(_mm(n3, w["w1"]) + w["b1"])
    f = r(gelu_tanh(f))
    f = r(_mm(f, w["w2"]) + w["b2"])
    return r(x + f * e[C_GATE])


def stack(
    x,
    e,
    k2,
    v2,
    cos,
    sin,
    weights,
    cfg: Config = WAN22_A14B,
    *,
    keys=None,
    mode="bf16",
    layers=None,
):
    """``block`` over every layer: ``e [L, 6, H]``, ``k2, v2 [L, T_ctx, H]``,
    ``weights`` with a leading layer axis."""
    n = cfg.layers if layers is None else layers
    for i in range(n):
        w = {k: v[i] for k, v in weights.items()}
        x = block(x, e[i], k2[i], v2[i], cos, sin, w, cfg, keys=keys, mode=mode)
    return x.astype(BF16) if mode == "bf16" else x
