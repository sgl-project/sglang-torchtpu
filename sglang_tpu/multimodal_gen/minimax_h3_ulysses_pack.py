# SPDX-License-Identifier: Apache-2.0
"""MiniMax-H3 QK RMSNorm + RoPE written straight into Ulysses exchange buffers.

One Pallas pass over the fused qkv projection ``[rows, 3 * heads * 128]``:
per head, the arithmetic of ``minimax_h3_qk_norm_rope`` (RMSNorm in float32,
rounded to bf16, then the half-split rotation as a signed-permutation matmul)
on q and k, and v copied, written destination-major into one buffer per head
group: group ``g``'s buffer is ``[ranks, lead + 3 * heads_g, rows, 128]``,
rank ``w``'s heads of the group as q, k, v parts after ``lead`` empty parts
(where the caller writes another group's attention output before the
exchange). Every part is a contiguous ``[rows, 128]`` run, which the
attention kernel streams from as fast as from its own arrays. Replaces the
strided head gathers and relayout copies XLA emits for the same pack.
"""

import functools
import threading

HEAD_DIM = 128
ROTARY_DIM = 96


@functools.cache
def _rotation_matrix(head_dim: int, rot_dim: int):
    """``P`` with ``x @ P = [-x[h:2h], x[:h], 0]`` for ``h = rot_dim // 2``."""
    import numpy as np

    half = rot_dim // 2
    p = np.zeros((head_dim, head_dim), dtype=np.float32)
    for j in range(half):
        p[half + j, j] = -1.0
        p[j, half + j] = 1.0
    return p


def _kernel(*refs, eps, groups, lead):
    import jax
    import jax.numpy as jnp

    f32 = jnp.float32
    heads = sum(groups)
    q_refs, k_refs, v_refs = (
        refs[:heads],
        refs[heads : 2 * heads],
        refs[2 * heads : 3 * heads],
    )
    qw_ref, kw_ref, cos_ref, sin_ref, rot_ref = refs[3 * heads : 3 * heads + 5]
    out_refs = refs[3 * heads + 5 :]
    cos = cos_ref[...]
    sin = sin_ref[...]
    rot = rot_ref[...]
    qw = qw_ref[...][:1, :]
    kw = kw_ref[...][:1, :]

    def norm_rope(x, w):
        # minimax_h3_qk_norm_rope._kernel's arithmetic on one head
        x = x.astype(f32)
        normed = (
            (x * jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + eps) * w)
            .astype(jnp.bfloat16)
            .astype(f32)
        )
        rotated = jax.lax.dot_general(
            normed,
            rot,
            (((1,), (0,)), ((), ())),
            preferred_element_type=f32,
            precision=jax.lax.Precision.HIGHEST,
        )
        return (normed * cos + rotated * sin).astype(jnp.bfloat16)

    head = 0
    for g, size in enumerate(groups):
        out_ref = out_refs[g]
        for j in range(size):
            at = lead[g] + 3 * j
            out_ref[at] = norm_rope(q_refs[head][...], qw)
            out_ref[at + 1] = norm_rope(k_refs[head][...], kw)
            out_ref[at + 2] = v_refs[head][...]
            head += 1


def qknorm_rope_pack_jax(
    qkv,
    q_weight,
    k_weight,
    cos_sin,
    eps,
    *,
    ranks,
    groups,
    lead,
    block_rows=256,
    heads_per_rank=None,
    first_head=0,
):
    """``qkv [rows, 3 * ranks * hl * 128]`` bf16 -> one buffer per group.

    ``groups``: heads per group, covering local heads ``first_head`` on (all
    ``hl`` by default); ``lead``: empty 128-lane blocks in front of each
    group's q|k|v. ``cos_sin [rows, 96]`` = 48 cos | 48 sin.
    """
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    f32 = jnp.float32
    d = HEAD_DIM
    rows = qkv.shape[0]
    hl = heads_per_rank or sum(groups)
    assert qkv.shape[1] == 3 * ranks * hl * d
    assert first_head + sum(groups) <= hl
    # A ragged last row block is fine: every row is computed on its own, and
    # rows past the array are never written.
    assert block_rows % 8 == 0 or block_rows == rows

    half = ROTARY_DIM // 2
    table = cos_sin.astype(f32)
    cos_h, sin_h = table[:, :half], table[:, half:ROTARY_DIM]
    pass_dims = d - ROTARY_DIM
    # Pass-through dims rotate by angle 0: cos 1, sin 0.
    cos = jnp.concatenate([cos_h, cos_h, jnp.ones((rows, pass_dims), f32)], axis=-1)
    sin = jnp.concatenate([sin_h, sin_h, jnp.zeros((rows, pass_dims), f32)], axis=-1)
    q_w = jnp.broadcast_to(q_weight.astype(f32)[None, :], (8, d))
    k_w = jnp.broadcast_to(k_weight.astype(f32)[None, :], (8, d))
    rot = jnp.asarray(_rotation_matrix(d, ROTARY_DIM))

    def head_spec(p, h):
        # local head h of rank w in part p (q, k, v): one 128-wide column block
        return pl.BlockSpec(
            (block_rows, d), lambda i, w: (i, (p * ranks + w) * hl + first_head + h)
        )

    heads = sum(groups)
    in_specs = [head_spec(p, h) for p in range(3) for h in range(heads)]

    row_spec = pl.BlockSpec((block_rows, d), lambda i, w: (i, 0))
    w_spec = pl.BlockSpec((8, d), lambda i, w: (0, 0))
    rot_spec = pl.BlockSpec((d, d), lambda i, w: (0, 0))
    parts = [lead[g] + 3 * size for g, size in enumerate(groups)]
    out_specs = [
        pl.BlockSpec((None, n, block_rows, d), lambda i, w: (w, 0, i, 0)) for n in parts
    ]
    out_shape = [jax.ShapeDtypeStruct((ranks, n, rows, d), qkv.dtype) for n in parts]
    return pl.pallas_call(
        functools.partial(_kernel, eps=eps, groups=tuple(groups), lead=tuple(lead)),
        grid=(pl.cdiv(rows, block_rows), ranks),
        in_specs=in_specs + [w_spec, w_spec, row_spec, row_spec, rot_spec],
        out_specs=out_specs,
        out_shape=out_shape,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel"),
            vmem_limit_bytes=VMEM_LIMIT_BYTES,
        ),
        name="sglang_tpu_minimax_h3_qknorm_rope_ulysses_pack",
    )(*([qkv] * (3 * heads)), q_w, k_w, cos, sin, rot)


def group_layout(heads: int, heads_per_group: int, lead: int):
    """``(sizes, lead_parts)``: heads per group, and the empty parts in front
    of each group's q|k|v (group ``g >= lead`` carries group ``g - lead``'s
    attention output)."""
    sizes = [
        min(heads_per_group, heads - start)
        for start in range(0, heads, heads_per_group)
    ]
    blocks = [0 if g < lead else sizes[g - lead] for g in range(len(sizes))]
    return sizes, blocks


# Mosaic's VMEM limit for the pack kernel, and the share of it the
# double-buffered blocks may take (the rest: f32 temporaries of the norm and
# rotation, internal scratch).
VMEM_LIMIT_BYTES = 48 * 1024 * 1024
_VMEM_BLOCK_BUDGET_BYTES = 40 * 1024 * 1024


def pack_row_blocks(
    heads: int, heads_per_group: int, lead: int, first_group: int, groups: int
) -> int:
    """128-wide bf16 tiles per row block of one pack call: the q, k, v
    inputs of its heads plus the parts of its output buffers."""
    sizes, blocks = group_layout(heads, heads_per_group, lead)
    sel = slice(first_group, first_group + groups)
    return 3 * sum(sizes[sel]) + sum(b + 3 * n for n, b in zip(sizes[sel], blocks[sel]))


def _vmem_bytes(block_rows: int, row_blocks: int) -> int:
    d = HEAD_DIM
    per_buffer = (
        row_blocks * block_rows * d * 2  # bf16 head inputs and output parts
        + 2 * block_rows * d * 4  # float32 cos, sin
        + d * d * 4  # rotation
        + 2 * 8 * d * 4  # norm weights
    )
    return 2 * per_buffer  # double-buffered


def block_rows_for(rows: int, row_blocks: int = 0) -> int | None:
    """Row block of the pack kernel: the largest of 512, 256, .. rows (the last
    block may be ragged; all rows of a shorter shard) whose double-buffered
    blocks of ``row_blocks`` tiles (:func:`pack_row_blocks`) fit the VMEM
    budget, or None. Every row is computed on its own, so the block size
    does not change the result."""
    if rows % 8:
        return None
    for block in (512, 256, 128, 64, 32, 16, 8):
        block = min(block, rows)
        if _vmem_bytes(block, row_blocks) <= _VMEM_BLOCK_BUDGET_BYTES:
            return block
    return None


_OPS: dict[int, object] = {}
_OPS_LOCK = threading.Lock()


def registered_op(groups: int):
    """The op :func:`ensure_op` registered for ``groups``, or None."""
    return _OPS.get(groups)


def ensure_op(groups: int):
    """The torch op producing ``groups`` consecutive groups' buffers; register
    it outside any compiled region."""
    op = _OPS.get(groups)
    if op is not None:
        return op
    with _OPS_LOCK:
        if groups in _OPS:
            return _OPS[groups]
        import jax

        from sglang_tpu.pallas_jax_op import cpu_export_jax_op

        def pack(
            qkv: jax.Array,
            q_weight: jax.Array,
            k_weight: jax.Array,
            cos_sin: jax.Array,
            eps: float,
            ranks: int,
            heads_per_group: int,
            lead: int,
            first_group: int,
        ):
            heads = qkv.shape[1] // (3 * ranks * HEAD_DIM)
            sizes, blocks = group_layout(heads, heads_per_group, lead)
            sel = slice(first_group, first_group + groups)
            return tuple(
                qknorm_rope_pack_jax(
                    qkv,
                    q_weight,
                    k_weight,
                    cos_sin,
                    eps,
                    ranks=ranks,
                    groups=sizes[sel],
                    lead=blocks[sel],
                    block_rows=block_rows_for(
                        qkv.shape[0],
                        pack_row_blocks(
                            heads, heads_per_group, lead, first_group, groups
                        ),
                    ),
                    heads_per_rank=heads,
                    first_head=sum(sizes[:first_group]),
                )
            )

        pack.__annotations__["return"] = tuple[(jax.Array,) * groups]

        def pack_fake(
            qkv,
            q_weight,
            k_weight,
            cos_sin,
            eps,
            ranks,
            heads_per_group,
            lead,
            first_group,
        ):
            heads = qkv.shape[1] // (3 * ranks * HEAD_DIM)
            sizes, blocks = group_layout(heads, heads_per_group, lead)
            sel = slice(first_group, first_group + groups)
            return tuple(
                qkv.new_empty((ranks, b + 3 * n, qkv.shape[0], HEAD_DIM))
                for n, b in zip(sizes[sel], blocks[sel])
            )

        op = cpu_export_jax_op(
            f"sglang_tpu_pallas::minimax_h3_qknorm_rope_ulysses_pack_{groups}", pack
        )
        op.register_fake(pack_fake)
        _OPS[groups] = op
        return op
