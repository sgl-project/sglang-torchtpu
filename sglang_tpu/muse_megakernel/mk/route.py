# SPDX-License-Identifier: Apache-2.0
"""Expert routing tables: distinct experts and their (row, slot) groups."""

import jax
import jax.numpy as jnp

from sglang_tpu.muse_megakernel.mk.consts import LANE


def _route_table(ids_list, n_live, K, Ep, NB, pair_key, n_keys, split=0):
    """vec_route table ``[route_rows, LANE]`` int32 from per-slot expert ids ``K x [R, 1]``.

    Distinct experts are ordered by (the smallest ``pair_key`` [P, 1] of their
    pairs, id), pair
    q = k*R + r. Rows ``[f*NB, (f+1)*NB)`` hold field f: 0 experts, 1 their live
    (row, slot)
    counts, 2 offsets into field 3, 3 pairs as ``row * K + slot`` grouped by
    expert; row 4*NB
    the distinct count; row 4*NB+1 lane b the experts whose key is <= b. With
    ``split``, experts of more than ``split`` pairs sort after the others of their
    key, and row 4*NB+2 lane b counts the experts before key b's large ones.
    Prefix sums run on the MXU (0/1 and counts <= R*K are exact in bf16, summed in
    f32).
    """
    R = ids_list[0].shape[0]
    P = R * K
    f32 = jnp.float32
    col_e = jnp.concatenate(ids_list, axis=0)  # [P, 1]
    q_iota = jax.lax.broadcasted_iota(jnp.int32, (P, 1), 0)
    r_of, k_of = q_iota % R, q_iota // R
    enc = r_of * K + k_of
    live = r_of < n_live
    e_lane = jax.lax.broadcasted_iota(jnp.int32, (P, Ep), 1)
    hit = (col_e == e_lane) & live  # [P, Ep]
    onehot = jnp.where(hit, 1.0, 0.0)
    count = jnp.sum(onehot, axis=0, keepdims=True)  # [1, Ep]
    present = count > 0
    big = float(n_keys + 1)
    key = jnp.min(
        jnp.where(hit, pair_key.astype(f32), big), axis=0, keepdims=True
    )  # [1, Ep]
    mult = 2 if split else 1
    if split:
        key = 2 * key + jnp.where(count > split, 1.0, 0.0)
    e_row = jax.lax.broadcasted_iota(jnp.int32, (1, Ep), 1).astype(f32)
    # per-expert values as columns, padded to a lane tile so the transpose is tile-aligned
    cols = jnp.transpose(
        jnp.concatenate(
            [
                key,
                e_row,
                count,
                jnp.where(present, 1.0, 0.0),
                jnp.zeros((LANE - 4, Ep), f32),
            ],
            axis=0,
        )
    )  # [Ep, LANE]
    key_c, id_c, cnt_c, pres_c = (
        cols[:, 0:1],
        cols[:, 1:2],
        cols[:, 2:3],
        cols[:, 3:4],
    )
    # earlier[e, e']: present e' sorts before e
    earlier = present & ((key < key_c) | ((key == key_c) & (e_row < id_c)))  # [Ep, Ep]
    pos_c = jnp.sum(jnp.where(earlier, 1.0, 0.0), axis=1, keepdims=True).astype(
        jnp.int32
    )
    off_c = jnp.sum(jnp.where(earlier, count, 0.0), axis=1, keepdims=True)  # [Ep, 1]
    off_e = jnp.transpose(
        jnp.concatenate([off_c, jnp.zeros((Ep, LANE - 1), f32)], axis=1)
    )[0:1]  # [1, Ep]
    # each pair's rank among the earlier live pairs of its expert
    same = jax.lax.dot_general(
        onehot.astype(jnp.bfloat16),
        onehot.astype(jnp.bfloat16),
        (((1,), (1,)), ((), ())),
        preferred_element_type=f32,
    )  # [P, P]
    q_row = jax.lax.broadcasted_iota(jnp.int32, (P, P), 0)
    q_col = jax.lax.broadcasted_iota(jnp.int32, (P, P), 1)
    live_col = (q_col % R) < n_live
    rank = jnp.sum(
        jnp.where((q_col < q_row) & live_col, same, 0.0), axis=1, keepdims=True
    )
    dest = (jnp.sum(onehot * off_e, axis=1, keepdims=True) + rank).astype(jnp.int32)
    rows = [None] * (4 * NB)
    for b in range(NB):
        u_lane = jax.lax.broadcasted_iota(jnp.int32, (Ep, LANE), 1) + b * LANE
        m = (pres_c > 0) & (pos_c == u_lane)  # [Ep, LANE]
        for f, v in enumerate((id_c, cnt_c, off_c)):
            rows[f * NB + b] = jnp.sum(
                jnp.where(m, v, 0.0), axis=0, keepdims=True
            ).astype(jnp.int32)
        t_lane = jax.lax.broadcasted_iota(jnp.int32, (P, LANE), 1) + b * LANE
        rows[3 * NB + b] = jnp.sum(
            jnp.where((dest == t_lane) & live, enc, 0), axis=0, keepdims=True
        )
    n_u = jnp.sum(jnp.where(present, 1, 0), axis=1, keepdims=True)
    b_lane = jax.lax.broadcasted_iota(jnp.int32, (Ep, LANE), 1).astype(f32)
    blk_end = jnp.sum(
        jnp.where((pres_c > 0) & (key_c <= mult * b_lane + mult - 1), 1, 0),
        axis=0,
        keepdims=True,
    )
    tail = [jnp.broadcast_to(n_u, (1, LANE)), blk_end]
    if split:
        tail.append(
            jnp.sum(
                jnp.where((pres_c > 0) & (key_c <= 2 * b_lane), 1, 0),
                axis=0,
                keepdims=True,
            )
        )
    n_rows = -(-(4 * NB + len(tail)) // 8) * 8
    return jnp.concatenate(
        rows + tail + [jnp.zeros((n_rows - 4 * NB - len(tail), LANE), jnp.int32)],
        axis=0,
    )
