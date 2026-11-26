# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MiniMax H3's DiT block stack in XLA: the megakernel's specification.

H3 denoises one packed sequence, ``[text | keyframes | audio | video |
pad]``, through 50 identical blocks. A block is RMSNorm, AdaLN shift and
scale, fused QKV, per-head RMSNorm on q and k, rotate-half RoPE on the
first 96 of 128 head dims, joint attention over the real rows, the output
projection with a gated residual, then the same norm and modulation
around a SwiGLU MLP with a second gated residual. Each token reads its own
AdaLN row: its timestep's, for its modality (video, text, audio).

``reference_stack`` computes it rounding where ``minimax_h3_megakernel``
rounds; ``pack_weights``, ``adaln_tables``, ``table_onehot`` and
``rope_tables`` turn the checkpoint and a request into both's operands.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax

_F32 = jnp.float32
_BF16 = jnp.bfloat16
# A token's AdaLN row is a one-hot of this many lanes times a table, so
# the lookup is one exact bf16 matmul; more rows than this cannot occur
# (3 modalities x the unique timesteps of one forward).
TABLE_LANES = 128
# Table rows are stored padded to a bf16 sublane tile.
_TABLE_ROW_TILE = 16
NUM_TABLES = 6


@dataclasses.dataclass(frozen=True)
class Config:
    """The block stack's hyper-parameters, as ``transformer/config.json``."""

    hidden: int = 5376
    heads: int = 56
    head_dim: int = 128
    ffn: int = 14336
    layers: int = 50
    rope_dims: int = 96
    eps: float = 1e-5
    qk_eps: float = 1e-5

    @classmethod
    def from_transformer_config(cls, raw: Mapping[str, Any]) -> Config:
        """Reads the H3 transformer's ``config.json`` fields."""
        return cls(
            hidden=raw["hidden_size"],
            heads=raw["num_attention_heads"],
            head_dim=raw["attention_head_dim"],
            ffn=raw["ffn_hidden_size"],
            layers=raw["num_layers"],
            # cos and sin of t, h and w frequencies, each rope_inv_freq_len.
            rope_dims=6 * raw["rope_inv_freq_len"],
            eps=raw["norm_eps"],
            qk_eps=raw["qk_norm_eps"],
        )

    @property
    def width(self) -> int:
        """Attention width: heads times head_dim (7168, wider than hidden)."""
        return self.heads * self.head_dim


def pack_weights(
    blocks: Sequence[Mapping[str, Any]],
    config: Config,
    *,
    qkv_heads: int = 2,
    out_cols: int = 768,
    ffn_cols: int = 256,
    down_cols: int = 256,
) -> dict[str, jax.Array]:
    """Stacks the checkpoint's blocks into the kernel's tiled layout.

    The defaults keep every streamed tile's output a multiple of the
    256-wide MXU (2 heads x [q | k | v] = 768 columns).

    Args:
      blocks: Per block, the checkpoint tensors named relative to
        ``blocks.<i>.``: ``attn.qkv_proj.weight`` ``[3W, H]`` with each
        head's q, k, v rows together, ``attn.out_proj.weight`` ``[H, W]``,
        ``mlp.fc1.weight`` ``[2F, H]`` (gate rows, then up rows),
        ``mlp.fc2.weight`` ``[H, F]``, ``norm1.weight`` and
        ``norm2.weight`` ``[H]``, ``attn.q_norm.weight`` and
        ``attn.k_norm.weight`` ``[D]``.
      config: The stack's hyper-parameters.
      qkv_heads: Heads per QKV tile.
      out_cols: Output-projection columns per tile.
      ffn_cols: FFN channels per (gate | up) tile.
      down_cols: Down-projection output columns per tile.

    Returns:
      ``wqkv`` ``[L, heads/qkv_heads, H, qkv_heads 3D]``, ``wo`` ``[L,
      H/out_cols, W, out_cols]``, ``w13`` ``[L, F/ffn_cols, H, 2 ffn_cols]``,
      ``w2`` ``[L, H/down_cols, F, down_cols]``, all bf16, and the norms
      ``[L, 1, H]`` / ``[L, 1, D]``. Every tile a DMA moves is contiguous.

    Raises:
      ValueError: If a tile width does not divide what it tiles or is not a
        multiple of 128 lanes.
    """
    hidden, ffn, dim = config.hidden, config.ffn, config.head_dim
    if (
        config.heads % qkv_heads
        or hidden % out_cols
        or ffn % ffn_cols
        or hidden % down_cols
        or any(cols % 128 for cols in (out_cols, ffn_cols, down_cols))
    ):
        raise ValueError(
            f"pack_weights cannot tile {config} with {qkv_heads} heads, "
            f"{out_cols}, {ffn_cols} and {down_cols} columns a tile (each a"
            " multiple of 128)"
        )

    def layer(block: Mapping[str, Any]) -> dict[str, jax.Array]:
        qkv = jnp.asarray(block["attn.qkv_proj.weight"], _BF16)
        out = jnp.asarray(block["attn.out_proj.weight"], _BF16)
        fc1 = jnp.asarray(block["mlp.fc1.weight"], _BF16)
        fc2 = jnp.asarray(block["mlp.fc2.weight"], _BF16)
        # [3W, H] rows grouped per head -> [heads/g, H, g 3D]: tile i holds
        # heads g i .. g i + g - 1, each one's [q | k | v] columns together.
        wqkv = qkv.T.reshape(hidden, -1, qkv_heads * 3 * dim).transpose(1, 0, 2)
        # [H, W] -> [W, H] -> [H/out_cols, W, out_cols]: column tiles.
        wo = out.T.reshape(config.width, -1, out_cols).transpose(1, 0, 2)
        # Gate and up channels of one tile sit side by side: [H, 2 ffn_cols].
        gate = fc1[:ffn].T.reshape(hidden, -1, ffn_cols)
        up = fc1[ffn:].T.reshape(hidden, -1, ffn_cols)
        w13 = jnp.concatenate((gate, up), axis=-1).transpose(1, 0, 2)
        # [H, F] -> [F, H] -> [H/down_cols, F, down_cols]: column tiles.
        w2 = fc2.T.reshape(ffn, -1, down_cols).transpose(1, 0, 2)

        def row(name: str) -> jax.Array:
            return jnp.asarray(block[name], _BF16)[None]

        return {
            "wqkv": wqkv,
            "wo": wo,
            "w13": w13,
            "w2": w2,
            "norm1": row("norm1.weight"),
            "norm2": row("norm2.weight"),
            "q_norm": row("attn.q_norm.weight"),
            "k_norm": row("attn.k_norm.weight"),
        }

    layers = [layer(block) for block in blocks]
    return {name: jnp.stack([w[name] for w in layers]) for name in layers[0]}


def adaln_tables(
    adaln_input: jax.Array, weights: jax.Array, biases: jax.Array
) -> jax.Array:
    """Every block's six AdaLN tables for one forward's timesteps.

    Args:
      adaln_input: ``[M, Dt]`` SiLU of the time embedding, one row per
        unique timestep.
      weights: ``[L, 18 H, Dt]`` the blocks' ``adaln_proj.linear.weight``.
      biases: ``[L, 18 H]`` their biases.

    Returns:
      ``[L, 6, R, H]`` bf16, ``R`` = ``3 M`` rounded up to 16: table ``k``
      row ``3 m + modality`` is the projection's ``(m, modality, k)``
      chunk, as the reference's ``view(3 M, 6 H).chunk(6)`` reads it.
    """
    steps = adaln_input.shape[0]
    layers, features, _ = weights.shape
    hidden = features // 18
    out = (
        jnp.einsum(
            "md,lfd->lmf",
            adaln_input.astype(_BF16),
            weights.astype(_BF16),
            preferred_element_type=_F32,
        )
        + biases.astype(_F32)[:, None]
    )
    out = out.astype(_BF16).reshape(layers, steps, 3, NUM_TABLES, hidden)
    # [L, M, 3, 6, H] -> [L, 6, M, 3, H] -> [L, 6, 3M, H]: row 3m + modality.
    out = out.transpose(0, 3, 1, 2, 4).reshape(layers, NUM_TABLES, -1, hidden)
    pad = -out.shape[2] % _TABLE_ROW_TILE
    return jnp.pad(out, ((0, 0), (0, 0), (0, pad), (0, 0)))


def table_onehot(rows: jax.Array) -> jax.Array:
    """``[T]`` AdaLN row per token -> ``[T, 128]`` bf16 one-hot."""
    lanes = jnp.arange(TABLE_LANES, dtype=jnp.int32)
    return (rows.astype(jnp.int32)[:, None] == lanes).astype(_BF16)


def rope_tables(
    position_ids: jax.Array, inv_freq: jax.Array, config: Config
) -> tuple[jax.Array, jax.Array]:
    """Lane-wide RoPE tables for ``[T, 3]`` (t, h, w) positions.

    The reference rotates the first ``rope_dims`` head dims as two halves
    of ``rope_dims / 2`` (t, h, w frequencies, each ``len(inv_freq)``) and
    passes the rest through. Over the whole head width that is ``x * cos +
    rotate(x) * sin`` with ``cos`` 1 and ``sin`` 0 on the passed dims.

    Args:
      position_ids: ``[T, 3]`` positions.
      inv_freq: ``[rope_dims / 6]`` the checkpoint's ``rope.inv_freq``.
      config: The stack's hyper-parameters.

    Returns:
      ``(cos, sin)``, each ``[T, head_dim]`` bf16.
    """
    tokens = position_ids.shape[0]
    angles = position_ids.astype(_F32)[:, :, None] * inv_freq.astype(_F32)
    half = angles.reshape(tokens, config.rope_dims // 2)  # t, then h, then w
    rest = config.head_dim - config.rope_dims
    cos = jnp.cos(half).astype(_BF16)
    sin = jnp.sin(half).astype(_BF16)
    ones = jnp.ones((tokens, rest), _BF16)
    return (
        jnp.concatenate((cos, cos, ones), axis=-1),
        jnp.concatenate((sin, sin, jnp.zeros_like(ones)), axis=-1),
    )


def round_bf16(x: jax.Array) -> jax.Array:
    return x.astype(_BF16).astype(_F32)


def rms(x: jax.Array, eps: float) -> jax.Array:
    return x * lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + eps)


def _rotate(x: jax.Array, rope_dims: int) -> jax.Array:
    """The rotate-half partner of every lane: ``(-x2, x1, 0)``."""
    half = rope_dims // 2
    lane = lax.broadcasted_iota(jnp.int32, x.shape, x.ndim - 1)
    # Rolling by -half brings x2 under x1's lanes; by +half, x1 under x2's.
    ahead = jnp.roll(x, x.shape[-1] - half, axis=-1)
    behind = jnp.roll(x, half, axis=-1)
    return jnp.where(lane < half, -ahead, jnp.where(lane < rope_dims, behind, 0.0))


def dense_attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    used: int | jax.Array,
    visible: jax.Array | None = None,
) -> jax.Array:
    """``[T, heads, D]`` bf16 attention: real rows see real rows, padding pads.

    The probabilities round to bf16 before ``P V`` and the sum stays f32, as
    a flash kernel computes them. ``visible``: ``[T, T]`` or per head
    ``[heads, T, T]`` bool, which keys each row may see besides.
    """
    scores = jnp.einsum("qgd,kgd->gqk", q, k, preferred_element_type=_F32)
    scores *= q.shape[-1] ** -0.5
    padding = jnp.arange(q.shape[0]) >= used
    same = (padding[:, None] == padding[None, :])[None]
    if visible is not None:
        same = same & (visible if visible.ndim == 3 else visible[None])
    scores = jnp.where(same, scores, -jnp.inf)
    probs = jnp.exp(scores - scores.max(axis=-1, keepdims=True))
    out = jnp.einsum(
        "gqk,kgd->qgd", probs.astype(_BF16), v, preferred_element_type=_F32
    )
    return (out / jnp.moveaxis(probs.sum(axis=-1), 0, 1)[..., None]).astype(_BF16)


def reference_layer(
    x: jax.Array,
    layer: Mapping[str, jax.Array],
    tables: jax.Array,
    onehot: jax.Array,
    cos: jax.Array,
    sin: jax.Array,
    used: int | jax.Array,
    config: Config,
    attend: Callable[..., jax.Array] | None = None,
) -> jax.Array:
    """One block on every row, in XLA, rounding where the kernel rounds.

    What the kernel rounds to bf16 is kept in bf16, so this is also the
    plain XLA implementation the kernel is measured against.

    Args:
      x: ``[T, H]`` bf16 packed rows.
      layer: One layer of ``pack_weights``'s output.
      tables: ``[6, R, H]`` this layer's AdaLN tables.
      onehot: ``[T, 128]`` from ``table_onehot``.
      cos: ``[T, D]`` from ``rope_tables``.
      sin: ``[T, D]``.
      used: Rows before the padding; real rows attend real rows, padding
        rows attend padding rows.
      config: The stack's hyper-parameters.
      attend: ``(q, k, v) -> out``, each ``[T, heads, D]`` bf16; None is
        ``dense_attention`` over ``used``.

    Returns:
      ``[T, H]`` bf16.
    """
    tokens = x.shape[0]
    dim, heads = config.head_dim, config.heads
    # Each token's AdaLN row: its one-hot's lane.
    row = jnp.argmax(onehot, axis=-1)
    params = [tables[k][row].astype(_F32) for k in range(NUM_TABLES)]
    shift1, scale1, gate1, shift2, scale2, gate2 = params

    def modulate(value, weight, shift, scale):
        normed = rms(value.astype(_F32), config.eps) * weight.astype(_F32)
        return (normed * (1.0 + scale) + shift).astype(_BF16)

    def project(lhs, name, spec):
        """``lhs`` x weight ``name``."""
        return jnp.einsum(spec, lhs, layer[name], preferred_element_type=_F32)

    h = modulate(x, layer["norm1"], shift1, scale1)
    # [T, H] x [heads/g, H, g 3D] -> [T, heads/g, g 3D] -> [T, heads, 3D]:
    # each head's q | k | v.
    qkv = project(h, "wqkv", "th,ghc->tgc").astype(_BF16)
    qkv = qkv.reshape(tokens, heads, 3 * dim)
    q, k, v = qkv[..., :dim], qkv[..., dim : 2 * dim], qkv[..., 2 * dim :]

    def norm_rope(value, weight):
        normed = rms(value.astype(_F32), config.qk_eps) * weight.astype(_F32)
        normed = round_bf16(normed)
        rope = (
            normed * cos.astype(_F32)[:, None]
            + _rotate(normed, config.rope_dims) * sin.astype(_F32)[:, None]
        )
        return rope.astype(_BF16)

    q = norm_rope(q, layer["q_norm"])
    k = norm_rope(k, layer["k_norm"])
    if attend is None:
        attn = dense_attention(q, k, v, used)
    else:
        attn = attend(q, k, v)
    attn = attn.reshape(tokens, heads * dim)
    # [T, W] x [H/c, W, c] -> [T, H/c, c] -> [T, H].
    out = project(attn, "wo", "tw,jwc->tjc")
    out = out.reshape(tokens, -1).astype(_BF16)
    x1 = (x.astype(_F32) + gate1 * out.astype(_F32)).astype(_BF16)
    h2 = modulate(x1, layer["norm2"], shift2, scale2)
    # [T, H] x [F/c, H, 2c] -> [T, F/c, 2c]: each tile's gate | up.
    gate_up = project(h2, "w13", "th,fhc->tfc").astype(_BF16)
    half = gate_up.shape[-1] // 2
    gate = round_bf16(jax.nn.silu(gate_up[..., :half].astype(_F32)))
    act = (gate * gate_up[..., half:].astype(_F32)).astype(_BF16)
    # [T, F/c, c] -> [T, F] x [H/n, F, n] -> [T, H/n, n] -> [T, H].
    mlp = jnp.einsum(
        "tf,jfn->tjn",
        act.reshape(tokens, -1),
        layer["w2"],
        preferred_element_type=_F32,
    )
    mlp = mlp.reshape(tokens, -1).astype(_BF16).astype(_F32)
    return (x1.astype(_F32) + gate2 * mlp).astype(_BF16)


def reference_stack(
    x: jax.Array,
    weights: Mapping[str, jax.Array],
    tables: jax.Array,
    onehot: jax.Array,
    cos: jax.Array,
    sin: jax.Array,
    used: int | jax.Array,
    config: Config,
    attend: Callable[..., jax.Array] | None = None,
) -> jax.Array:
    """Every block of ``weights`` in turn; ``stack``'s math in XLA."""
    for index in range(weights["wqkv"].shape[0]):
        layer = {name: value[index] for name, value in weights.items()}
        x = reference_layer(
            x, layer, tables[index], onehot, cos, sin, used, config, attend
        )
    return x
