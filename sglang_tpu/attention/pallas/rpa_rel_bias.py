"""Learned relative-position bias and per-token sinks for RPA v3.

Score for query position i, key position j, head h:

    s = q.k * sm_scale + r[i, h] . proj[:, i - j]     for 0 <= i - j < extent

``r`` is pre-multiplied by ``sm_scale`` (and any per-token query scale) by the
caller, so the bias is added after ``sm_scale`` and is untouched by fp8 KV
scales. A per-token sink logit acts as one extra zero-value key.

Both ride in ``REL_LANES`` extra lanes appended to every query row, so they
reach VMEM through the existing query DMA:

    lanes [head_dim, head_dim + rel_dim)   r
    lane  head_dim + SINK_LANE              sink logit
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

REL_LANES = 128
SINK_LANE = REL_LANES - 1
# Largest KV tile the bias path handles at once; bounds the window table. A
# wider compute tile is cut into tiles of this size when it needs the bias.
REL_MAX_BKV_CSZ = 512
# Rows per strided roll: Mosaic rejects strided shifts past one vreg's 128 lanes.
_SHEAR_CHUNK = 64


def _align(x: int, a: int) -> int:
    return (x + a - 1) // a * a


def rel_sub_tile(bkv_csz: int) -> int:
    """Keys per bias tile for a compute tile of ``bkv_csz`` keys."""
    if bkv_csz <= REL_MAX_BKV_CSZ:
        return bkv_csz
    if bkv_csz % REL_MAX_BKV_CSZ:
        raise ValueError(
            f"{bkv_csz=} (bkv_csz) above {REL_MAX_BKV_CSZ} must be a multiple of it"
            " with rel bias."
        )
    return REL_MAX_BKV_CSZ


def rel_window_width(*, bkv_csz: int, bq_csz: int) -> int:
    """Lanes of distance window one (bq_csz x bkv_csz) tile needs.

    Covers bkv_csz + bq_csz - 1 distances plus up to 127 lanes of
    misalignment from the 128-aligned window start.
    """
    return _align(bkv_csz + bq_csz + 127, 128)


def build_rel_window_table(rel_proj: jax.Array, *, window: int) -> jax.Array:
    """[rel_dim, extent] -> [REL_LANES, window + extent + window] bf16.

    Column ``window + extent - 1 - d`` holds ``proj[:, d]``; the zero margins
    make every clamped window outside [0, extent) read zeros.
    """
    rel_dim, extent = rel_proj.shape
    assert rel_dim <= SINK_LANE, f"{rel_dim=} must leave the sink lane free"
    assert extent % 128 == 0 and window % 128 == 0
    table = jnp.zeros((REL_LANES, window + extent + window), jnp.bfloat16)
    return table.at[:rel_dim, window : window + extent].set(
        rel_proj[:, ::-1].astype(jnp.bfloat16)
    )


def pack_rel_lanes(
    queries: jax.Array,  # [T, Hq, head_dim]
    rel_r: jax.Array | None,  # [T, Hq, rel_dim], pre-scaled
    token_sink: jax.Array | None,  # [T, Hq]
) -> jax.Array:
    """Append the REL_LANES block to every query row: [T, Hq, head_dim + 128]."""
    num_tokens, num_heads, _ = queries.shape
    block = jnp.zeros((num_tokens, num_heads, REL_LANES), queries.dtype)
    if rel_r is not None:
        block = block.at[..., : rel_r.shape[-1]].set(rel_r.astype(queries.dtype))
    if token_sink is not None:
        block = block.at[..., SINK_LANE].set(token_sink.astype(queries.dtype))
    return jnp.concatenate([queries, block], axis=-1)


# Scratches are lane-fold-major [folds, rows, 128]: Mosaic strides rows only
# on 128-lane memrefs and cannot reshape a wider f32 ref into one.


def _store_folded(ref, val, *, start=0, stride=None):
    """ref rows start, start + stride, ... (all rows if stride is None) = val."""
    rows = (
        pl.ds(start, val.shape[0])
        if stride is None
        else pl.ds(start, val.shape[0], stride=stride)
    )
    for f in range(val.shape[1] // 128):
        ref[f, rows, :] = val[:, f * 128 : (f + 1) * 128]


def _load_folded(ref, *, num, start=0, stride=None):
    rows = pl.ds(start, num) if stride is None else pl.ds(start, num, stride=stride)
    return jnp.concatenate([ref[f, rows, :] for f in range(ref.shape[0])], axis=1)


def _shear_rows(x: jax.Array) -> jax.Array:
    """Roll row a of x right by a lanes."""
    rows = x.shape[0]
    if rows == 1:
        return x
    chunks = []
    for c0 in range(0, rows, _SHEAR_CHUNK):
        chunk = pltpu.roll(x[c0 : c0 + _SHEAR_CHUNK], 0, 1, stride=1, stride_axis=0)
        if c0:
            chunk = pltpu.roll(chunk, c0 % x.shape[1], 1)
        chunks.append(chunk)
    return chunks[0] if len(chunks) == 1 else jnp.concatenate(chunks, axis=0)


def add_rel_bias(
    s: jax.Array,  # [abq * qpkv, bkv_csz] f32, already scaled
    r: jax.Array,  # [abq * qpkv, REL_LANES]
    *,
    table_ref,  # [REL_LANES, table_width] VMEM
    g_ref,  # [window // 128, >= abq * qpkv, 128] f32 VMEM scratch
    bias_ref,  # [ceil(bkv_csz / 128), >= abq * qpkv, 128] f32 VMEM scratch
    q_pos0,  # absolute position of the tile's first query row
    k_pos0,  # absolute position of the tile's first key
    abq: int,
    qpkv: int,
    extent: int,
    window: int,
) -> jax.Array:
    rows, bkv_csz = s.shape
    assert rows == abq * qpkv
    table_width = table_ref.shape[1]
    # Table column of the tile's largest distance, d_max = q_pos0 + abq - 1 - k_pos0.
    d_max = q_pos0 + abq - 1 - k_pos0
    start = jnp.clip(window + extent - 1 - d_max, 0, table_width - window)
    start_al = pl.multiple_of((start // 128) * 128, 128)
    delta = start - start_al

    win = table_ref[:, pl.ds(start_al, window)]
    assert g_ref.shape[0] * 128 == window
    g = jnp.matmul(r.astype(win.dtype), win, preferred_element_type=jnp.float32)
    # Row a (query position q_pos0 + a) needs window column c + delta + abq - 1 - a
    # for key c. Mosaic has no strided roll with a dynamic shift, so the shared
    # part goes in one dynamic roll; then each head's rows are sheared by a.
    g = pltpu.roll(g, (2 * window - delta - abq + 1) % window, 1)
    _store_folded(g_ref, g)
    bias_lanes = bias_ref.shape[0] * 128
    assert bkv_csz <= bias_lanes <= window
    for h in range(qpkv):
        gh = _load_folded(g_ref, num=abq, start=h, stride=qpkv)
        _store_folded(bias_ref, _shear_rows(gh)[:, :bias_lanes], start=h, stride=qpkv)
    return s + _load_folded(bias_ref, num=rows)[:, :bkv_csz]


def token_sink_rows(r_full: jax.Array) -> jax.Array:
    """[rows, REL_LANES] query lanes -> [rows, 128] sink logits for m init."""
    sink = r_full[:, SINK_LANE : SINK_LANE + 1].astype(jnp.float32)
    return jnp.broadcast_to(sink, (r_full.shape[0], 128))


def ref_rel_bias(
    rel_r: jax.Array,  # [q_len, Hq, rel_dim]
    rel_proj: jax.Array,  # [rel_dim, extent]
    *,
    q_pos: jax.Array,  # [q_len] absolute positions
    kv_len: int,
) -> jax.Array:
    """Reference bias [Hq, q_len, kv_len] f32."""
    extent = rel_proj.shape[1]
    rel = jnp.einsum(
        "qhf,fe->hqe", rel_r.astype(jnp.float32), rel_proj.astype(jnp.float32)
    )
    dist = q_pos[:, None] - jnp.arange(kv_len)[None, :]
    in_extent = (dist >= 0) & (dist < extent)
    idx = jnp.clip(dist, 0, extent - 1)
    bias = jnp.take_along_axis(
        rel, jnp.broadcast_to(idx[None], (rel.shape[0], *idx.shape)), axis=-1
    )
    return jnp.where(in_extent[None], bias, 0.0)
