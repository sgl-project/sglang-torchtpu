# SPDX-License-Identifier: Apache-2.0
# Adapted from sgl-project/sglang-jax (Apache-2.0)
# python/sgl_jax/srt/kernels/dsa/sparse_mla_prefill_qblock.py at b1b18af8a9.
"""Query-blocked sparse MLA attention for long prefills (pooled DSA selections).

The per-query kernel (``mla_sparse.kernel``) gathers every query's selected
tokens on its own: at an 8192-token prefill with 2176-wide rows that is ~11 GB
of SparseCore gathers per layer, and the same cache rows are fetched by
thousands of queries. This kernel does what sglang-jax's blocked prefill kernel
does instead: ``query_block`` queries of one request share one program, the
program DMAs the union of the pages they selected once per page, and each
query's own selection is restored with a ``-inf`` bias inside a flash softmax.

What differs from the source kernel:

* Selection format. The pooled indexer (``dsa_kpool_topk``) selects pools of
  ``kpool`` tokens and lists their tokens, then the open pool's tail. Column
  ``c * kpool`` of a row is the first token of one pool (selected or tail), so
  a row reduces to one pool id per entry. With ``page_size // kpool == 32``
  a query's selection inside one page is a single 32-bit word, bit ``i`` for
  tokens ``[i * kpool, (i + 1) * kpool)``. The tail pool is open, so its bit
  also covers tokens after the query; the causal bound removes them, which
  makes bit AND causal exactly the indexer's token list. ``-1`` entries set no
  bit.
* Blocks never straddle requests. The page union is taken over one request's
  pages, so the bitmap is ``[blocks, pages_per_seq, query_block]`` and
  per-block bounds are scalars. A request's last block is padded.
* Cache. Keys come from the split FP8 cache (``TpuSplitMLATokenToKVPool``):
  each union page is two contiguous DMAs, 128 NoPE rows of 512 bytes and 128
  RoPE rows of 128 bytes (64 RoPE bytes, zero padding), dequantized with
  ``k_scale`` like the per-query kernel.
* Layout. Score rows are head-major (``h * query_block + q``), so the
  per-query bias row is tiled ``num_heads`` times along lanes instead of
  expanded per element, and the accumulator is kept transposed
  (``[512, rows]``) so the flash rescale broadcasts along sublanes.

The DMA ring, the slot-free key-major score (keys on sublanes, query rows on
lanes) and the aligned-window bitmap read are the source kernel's.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

NOPE_DIM = 512
ROPE_LANES = 128  # RoPE row in the cache: 64 FP8 values, then zero padding
_NBUF = 4  # DMA ring depth (prefetch distance NBUF - 1 pages)
_WORD_DTYPE = jnp.int32  # a page's selection word: one bit per pool
_WORD_BITS = jnp.iinfo(_WORD_DTYPE).bits  # pools per word, i.e. per page
_POSITION_DTYPE = jnp.int32  # token positions
# The kernel computes scores, and keeps the softmax's running max and sum and
# the output accumulator, in this dtype.
_ACCUMULATOR_DTYPE = jnp.float32
_VMEM_LIMIT_BYTES = 64 * 1024 * 1024
_LANES = 128  # width of a TPU vector register
_SUBLANES = 8  # 32-bit rows of a vector register; a [1, n] buffer takes 8
# Queries per block: the default is the one this kernel came with from
# sglang-jax. Query positions sit on lanes, so a block is a whole number of
# lanes and the smallest is one lane width.
QUERY_BLOCK = 256
MIN_QUERY_BLOCK = _LANES


def _cdiv(a, b):
    return (a + b - 1) // b


def _align(x: int, m: int) -> int:
    return _cdiv(x, m) * m


def vmem_required_by_sparse_mla(
    num_heads: int,
    query_block: int,
    *,
    q_width: int,
    page_size: int,
    pages_per_seq: int,
    dtype: jnp.dtype,
) -> int:
    """VMEM the kernel needs, from the buffers its pallas_call declares.

    Pallas keeps two copies of each blocked input and output, and one of each
    scratch buffer. Most of the total is per score row (num_heads *
    query_block): the query window, the output, the output accumulator and
    one page of scores. Where the kernel does not fit, the estimate is up to
    0.6 MiB above what the tpu7x compiler reports.
    """
    itemsize = jnp.dtype(dtype).itemsize
    acc_bytes = jnp.dtype(_ACCUMULATOR_DTYPE).itemsize
    per_row = (
        2 * _align(q_width, _LANES) * itemsize  # query window
        + 2 * NOPE_DIM * itemsize  # output
        + NOPE_DIM * acc_bytes  # output accumulator
        + 2 * _SUBLANES * acc_bytes  # running max and sum, [1, rows] each
        + page_size * acc_bytes  # scores of one page, [page_size, rows]
    )
    per_block = (
        # Selection words, [pages, query_block].
        2
        * _align(pages_per_seq, _SUBLANES)
        * query_block
        * jnp.dtype(_WORD_DTYPE).itemsize
        # Query positions, [1, query_block].
        + 2 * _SUBLANES * query_block * jnp.dtype(_POSITION_DTYPE).itemsize
        # Key ring: one FP8 byte per NoPE value and per RoPE lane of a token.
        + _NBUF * page_size * (NOPE_DIM + ROPE_LANES)
    )
    return num_heads * query_block * per_row + per_block


def vmem_limit_bytes() -> int:
    """VMEM the kernel may use: its compile limit, or the chip's VMEM if less."""
    return min(_VMEM_LIMIT_BYTES, pltpu.get_tpu_info().vmem_capacity_bytes)


def _qblock_kernel(
    # Scalar prefetch
    num_live_ref,  # i32[1] blocks holding real queries; the rest are skipped
    counts_ref,  # i32[num_blocks] pages in each block's union
    kv_lens_ref,  # i32[num_blocks] kv length of each block's request
    # Inputs
    q_ref,  # bf16[1, rows, 640] rows = h * query_block + q; 576 real lanes
    qpos_ref,  # i32[1, 1, query_block] query positions, -1 on padded rows
    member_ref,  # i32[1, pages_pad, query_block] selection word per page
    units_ref,  # SMEM i32[1, 1, pages_pad] union, request-relative pages
    phys_ref,  # SMEM i32[1, 1, pages_pad] union, physical pages
    cache_nope_hbm,  # u8[num_pages, page_size, 4, 128]
    cache_rope_hbm,  # u8[num_pages, page_size // 4, 4, 128]
    # Output
    o_ref,  # bf16[1, 512, rows] transposed
    # Scratch
    nope_ring,  # u8[NBUF, page_size * 4, 128]
    rope_ring,  # u8[NBUF, page_size, 128]
    m_ref,  # f32[1, rows]
    l_ref,  # f32[1, rows]
    acc_ref,  # f32[512, rows]
    sems,  # DMA [2, NBUF]
    *,
    scale: float,
    k_scale: float,
    num_heads: int,
    page_size: int,
    kpool: int,
):
    b = pl.program_id(0)
    num_pages = cache_nope_hbm.shape[0]
    # Page views of the two caches. HBM refs are DMA-addressed, so the view
    # is free; it keeps a page one contiguous copy per cache.
    nope_pages = cache_nope_hbm.reshape(num_pages, page_size * 4, 128)
    rope_pages = cache_rope_hbm.reshape(num_pages, page_size, ROPE_LANES)

    @pl.when(b < num_live_ref[0])
    def _block():
        cnt = counts_ref[b]
        kv_len = kv_lens_ref[b]
        qpos = qpos_ref[0]  # [1, QB]
        rows = lax.broadcasted_iota(jnp.int32, (page_size, 1), 0)
        pool_of_row = rows // kpool

        def copies(j, slot):
            page = phys_ref[0, 0, j]
            return (
                pltpu.make_async_copy(
                    nope_pages.at[page], nope_ring.at[slot], sems.at[0, slot]
                ),
                pltpu.make_async_copy(
                    rope_pages.at[page], rope_ring.at[slot], sems.at[1, slot]
                ),
            )

        # Prologue: fill the ring (d=d binds the loop variable per iteration).
        for d in range(_NBUF - 1):

            @pl.when(d < cnt)
            def _(d=d):
                for cp in copies(d, d):
                    cp.start()

        m_ref[...] = jnp.full(m_ref.shape, -jnp.inf, _ACCUMULATOR_DTYPE)
        l_ref[...] = jnp.zeros(l_ref.shape, _ACCUMULATOR_DTYPE)
        acc_ref[...] = jnp.zeros(acc_ref.shape, _ACCUMULATOR_DTYPE)

        def unit(j, carry):
            g = units_ref[0, 0, j]  # request-relative page
            slot = lax.rem(j, _NBUF)

            # Keep the ring full: fetch the page NBUF - 1 ahead.
            nxt = j + _NBUF - 1

            @pl.when(nxt < cnt)
            def _():
                for cp in copies(nxt, lax.rem(nxt, _NBUF)):
                    cp.start()

            # Selection word of page g per query: an aligned 8-row window and
            # an iota select (a dynamic sublane index must be provably % 8).
            g8 = pl.multiple_of((g // 8) * 8, 8)
            window = member_ref[0, pl.ds(g8, 8), :]  # [8, QB]
            pick = lax.broadcasted_iota(jnp.int32, (8, 1), 0) == (g - g8)
            word = jnp.sum(jnp.where(pick, window, 0), axis=0, keepdims=True)

            # Key row r of page g is token g * page_size + r. Attend where the
            # query selected r's pool and r is not after the query.
            bits = lax.shift_right_logical(
                jnp.broadcast_to(word, (page_size, word.shape[1])),
                jnp.broadcast_to(pool_of_row, (page_size, word.shape[1])),
            )
            key_pos = g * page_size + rows
            attend = ((bits & 1) != 0) & (key_pos <= qpos)  # [page_size, QB]
            bias = jnp.where(attend, 0.0, -jnp.inf).astype(_ACCUMULATOR_DTYPE)
            bias = jnp.concatenate([bias] * num_heads, axis=1)  # [page, rows]

            for cp in copies(j, slot):
                cp.wait()
            nope = nope_ring[slot].reshape(page_size, NOPE_DIM)
            nope = pltpu.bitcast(nope, jnp.float8_e4m3fn).astype(jnp.bfloat16)
            rope = pltpu.bitcast(rope_ring[slot], jnp.float8_e4m3fn)
            rope = rope.astype(jnp.bfloat16)
            # Rows past the request's kv length were never written and can
            # decode to FP8 NaN; 0 * NaN would poison the PV sum.
            live = rows < kv_len - g * page_size
            nope = jnp.where(live, nope, 0)
            rope = jnp.where(live, rope, 0)

            # Key-major score: [page_size keys, rows].
            q = q_ref[0]
            nt = (((1,), (1,)), ((), ()))
            s = lax.dot_general(
                nope,
                q[:, :NOPE_DIM],
                nt,
                preferred_element_type=_ACCUMULATOR_DTYPE,
            )
            s += lax.dot_general(
                rope,
                q[:, NOPE_DIM : NOPE_DIM + ROPE_LANES],
                nt,
                preferred_element_type=_ACCUMULATOR_DTYPE,
            )
            s = s * scale + bias

            # Flash softmax over keys (axis 0, the sublane axis).
            m_prev = m_ref[...]
            m_new = jnp.maximum(m_prev, jnp.max(s, axis=0, keepdims=True))
            shift = jnp.where(m_new == -jnp.inf, 0.0, m_new)
            p = jnp.exp(s - shift)
            alpha = jnp.exp(m_prev - shift)
            l_ref[...] = alpha * l_ref[...] + jnp.sum(p, axis=0, keepdims=True)
            # acc[d, row] += sum_r v[r, d] * p[r, row] (contract the key axis).
            acc_ref[...] = alpha * acc_ref[...] + lax.dot_general(
                nope,
                p.astype(jnp.bfloat16),
                (((0,), (0,)), ((), ())),
                preferred_element_type=_ACCUMULATOR_DTYPE,
            )
            m_ref[...] = m_new
            return carry

        lax.fori_loop(0, cnt, unit, 0)

        # k_scale dequantizes V; a row that attended nothing (padding) is 0.
        l = l_ref[...]
        inv = jnp.where(l == 0.0, 0.0, k_scale / l)
        o_ref[0] = (acc_ref[...] * inv).astype(o_ref.dtype)


def _query_blocks(
    cu_q_lens, kv_lens, distribution, *, query_block, num_blocks, num_tokens
):
    """Lay requests' queries out in blocks of ``query_block``, one request each.

    Returns per block ``(request, first token, real queries, first position)``,
    the number of blocks holding real queries, and per token its row in the
    block layout (``-1`` for padding tokens).
    """
    # The page table has one row per cu_q_lens slot, as in the per-query kernel.
    max_num_seqs = cu_q_lens.shape[0] - 1
    real = jnp.arange(max_num_seqs) < distribution[2]
    q_lens = jnp.where(real, cu_q_lens[1:] - cu_q_lens[:-1], 0)
    seq_blocks = _cdiv(q_lens, query_block)
    ends = jnp.cumsum(seq_blocks)
    first_block = ends - seq_blocks

    blk = jnp.arange(num_blocks, dtype=jnp.int32)
    req = jnp.minimum(jnp.searchsorted(ends, blk, side="right"), max_num_seqs - 1)
    j = blk - first_block[req]
    count = jnp.clip(q_lens[req] - j * query_block, 0, query_block)
    count = jnp.where(blk < ends[-1], count, 0)
    start = cu_q_lens[req] + j * query_block
    first_pos = kv_lens[req] - q_lens[req] + j * query_block

    t = jnp.arange(num_tokens, dtype=jnp.int32)
    t_req = jnp.minimum(
        jnp.searchsorted(cu_q_lens[1:], t, side="right"), max_num_seqs - 1
    )
    t_off = t - cu_q_lens[t_req]
    t_row = first_block[t_req] * query_block + t_off
    t_row = jnp.where(t < cu_q_lens[distribution[2]], t_row, -1)
    return req, start, count, first_pos, ends[-1].astype(jnp.int32), t_row


@functools.partial(
    jax.jit,
    static_argnames=("sm_scale", "k_scale", "kpool", "query_block", "interpret"),
)
def sparse_mla_prefill_qblock(
    ql_nope: jax.Array,  # bf16[num_tokens, num_q_heads, 512]
    q_pe: jax.Array,  # bf16[num_tokens, num_q_heads, 64]
    cache_nope: jax.Array,  # u8[num_pages, page_size, 4, 128]
    cache_rope: jax.Array,  # u8[num_pages, page_size // 4, 4, 128]
    topk_indices: jax.Array,  # i32[num_tokens, width] from kpool_indexer_step
    kv_lens: jax.Array,  # i32[max_num_seqs]; counts this step's tokens
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]; [2] is the number of real requests
    *,
    sm_scale: float,
    k_scale: float,
    kpool: int,
    query_block: int = QUERY_BLOCK,
    interpret: bool = False,
) -> jax.Array:
    """Sparse MLA attention over pooled DSA selections, ``query_block`` at a time.

    Same result as ``sparse_ragged_paged_attention`` on the same rows (up to
    bf16 rounding of the softmax weights), for rows in the pooled indexer's
    format: selected pools' tokens, then the open pool's tail, then ``-1``.
    The cache must already hold this step's tokens.

    Returns ``bf16[num_tokens, num_q_heads, 512]``; rows of padding tokens are 0.
    """
    num_tokens, num_heads, _ = ql_nope.shape
    page_size = cache_nope.shape[1]
    max_num_seqs = kv_lens.shape[0]
    pages_per_seq = page_indices.shape[0] // max_num_seqs
    qb = query_block
    if page_size // kpool != _WORD_BITS or topk_indices.shape[1] % kpool:
        raise ValueError(
            f"q-block sparse MLA needs page_size // kpool == {_WORD_BITS} and a "
            f"row width divisible by kpool (page_size={page_size}, kpool={kpool}, "
            f"width={topk_indices.shape[1]})"
        )
    if qb % MIN_QUERY_BLOCK:
        raise ValueError(
            f"query_block must be a multiple of {MIN_QUERY_BLOCK}, got {qb}"
        )

    # Every block holds one request's queries, so a batch needs at most
    # sum(ceil(q_len / qb)) <= (num_tokens + seqs * (qb - 1)) / qb of them.
    num_blocks = (num_tokens + min(max_num_seqs, num_tokens) * (qb - 1)) // qb
    req, start, count, first_pos, num_live, t_row = _query_blocks(
        cu_q_lens,
        kv_lens,
        distribution,
        query_block=qb,
        num_blocks=num_blocks,
        num_tokens=num_tokens,
    )
    r = jnp.arange(qb, dtype=_POSITION_DTYPE)
    is_row = r[None, :] < count[:, None]  # [num_blocks, qb]
    tok = jnp.where(is_row, start[:, None] + r[None, :], 0)
    qpos = jnp.where(is_row, first_pos[:, None] + r[None, :], -1)

    # One pool id per entry: column c * kpool is the first token of a pool.
    first_tok = topk_indices[:, ::kpool][tok]  # [num_blocks, qb, width / kpool]
    sel = jnp.where(is_row[..., None] & (first_tok >= 0), first_tok // kpool, -1)
    page = jnp.where(sel >= 0, sel // _WORD_BITS, -1)
    bit = jnp.where(sel >= 0, jnp.left_shift(1, sel % _WORD_BITS), 0)
    # member[b, g, q]: query q's selection word for page g. A compare-OR, not a
    # scatter (scatters with this many updates are slow on the TPU).
    pages_pad = _align(pages_per_seq, _SUBLANES)
    g = jnp.arange(pages_pad, dtype=jnp.int32)
    member = lax.reduce(
        jnp.where(page[:, None] == g[None, :, None, None], bit[:, None], 0),
        np.dtype(_WORD_DTYPE).type(0),
        lax.bitwise_or,
        (3,),
    )  # [num_blocks, pages_pad, qb]

    # Each block's page union in ascending order, compacted to the front.
    used = jnp.any(member != 0, axis=2)  # [num_blocks, pages_pad]
    counts = jnp.sum(used, axis=1, dtype=jnp.int32)
    rank = jnp.where(used, jnp.cumsum(used, axis=1) - 1, pages_pad)
    blk = jnp.broadcast_to(jnp.arange(num_blocks)[:, None], rank.shape)
    units = jnp.zeros((num_blocks, pages_pad), jnp.int32)
    units = units.at[blk, rank].set(
        jnp.broadcast_to(g[None, :], rank.shape), mode="drop"
    )
    phys = page_indices[
        req[:, None] * pages_per_seq + jnp.minimum(units, pages_per_seq - 1)
    ]

    # Queries in block layout, head-major rows, 128-aligned head dim.
    q = jnp.concatenate([ql_nope, q_pe], axis=-1)
    head_dim = _align(q.shape[-1], _LANES)
    q = jnp.pad(q, ((0, 0), (0, 0), (0, head_dim - q.shape[-1])))
    q = jnp.where(is_row[..., None, None], q[tok], 0)  # [nb, qb, H, D]
    q = q.transpose(0, 2, 1, 3).reshape(num_blocks, num_heads * qb, head_dim)

    rows = num_heads * qb
    kernel = functools.partial(
        _qblock_kernel,
        scale=sm_scale * k_scale,
        k_scale=k_scale,
        num_heads=num_heads,
        page_size=page_size,
        kpool=kpool,
    )

    def live(b, num_live_ref, *_):
        # Blocks past the live ones revisit the last live block: no new DMA
        # in or out, and the kernel leaves its output alone.
        return jnp.maximum(jnp.minimum(b, num_live_ref[0] - 1), 0)

    smem_units = pl.BlockSpec(
        (1, 1, pages_pad), lambda b, *s: (live(b, *s), 0, 0), memory_space=pltpu.SMEM
    )
    out = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=3,
            grid=(num_blocks,),
            in_specs=[
                pl.BlockSpec((1, rows, head_dim), lambda b, *s: (live(b, *s), 0, 0)),
                pl.BlockSpec((1, 1, qb), lambda b, *s: (live(b, *s), 0, 0)),
                pl.BlockSpec((1, pages_pad, qb), lambda b, *s: (live(b, *s), 0, 0)),
                smem_units,
                smem_units,
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM),
            ],
            out_specs=pl.BlockSpec(
                (1, NOPE_DIM, rows), lambda b, *s: (live(b, *s), 0, 0)
            ),
            scratch_shapes=[
                pltpu.VMEM((_NBUF, page_size * 4, 128), cache_nope.dtype),
                pltpu.VMEM((_NBUF, page_size, ROPE_LANES), cache_rope.dtype),
                pltpu.VMEM((1, rows), _ACCUMULATOR_DTYPE),
                pltpu.VMEM((1, rows), _ACCUMULATOR_DTYPE),
                pltpu.VMEM((NOPE_DIM, rows), _ACCUMULATOR_DTYPE),
                pltpu.SemaphoreType.DMA((2, _NBUF)),
            ],
        ),
        out_shape=jax.ShapeDtypeStruct((num_blocks, NOPE_DIM, rows), ql_nope.dtype),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            vmem_limit_bytes=_VMEM_LIMIT_BYTES,
        ),
        interpret=interpret,
        name="sparse_mla_prefill_qblock",
    )(
        num_live.reshape(1),
        counts,
        kv_lens[req],
        q,
        qpos.reshape(num_blocks, 1, qb),
        member,
        units.reshape(num_blocks, 1, pages_pad),
        phys.reshape(num_blocks, 1, pages_pad),
        cache_nope,
        cache_rope,
    )

    # Back to token order: token t is row t_row of every head.
    out = out.reshape(num_blocks, NOPE_DIM, num_heads, qb).transpose(0, 3, 2, 1)
    out = out.reshape(num_blocks * qb, num_heads, NOPE_DIM)
    return jnp.where(t_row[:, None, None] >= 0, out[jnp.maximum(t_row, 0)], 0)
