# SPDX-License-Identifier: Apache-2.0
"""Query-blocked sparse MLA prefill for plain DSA (token top-k selections).

The per-query kernel (``mla_sparse.kernel``) gathers every query's own
``topk`` cache rows on the SparseCore. For a 1024-token prefill chunk with
2048 selections that is 2M row gathers per layer (~1.9 ms on TPU7x), and it
does not shrink with fewer heads per chip. Neighbouring queries select mostly
the same pages, so this kernel does what ``mla_sparse_qblock`` does for pooled
selections: a block of queries reads its request's pages once with plain DMAs
and restores each query's own selection with a mask inside a flash softmax.
The attention work then scales with the heads, like dense MLA, and with the
context (a block reads every page up to its last query: with ~2048 picks per
query, a 256-query block's selections cover nearly all of them anyway).

Selections. The indexer's rows are exact token top-k lists
(``i32[num_tokens, topk]``, ``-1`` padded). ``dsa_select_bits`` turns them into
per-query bitmaps on the SparseCore. In its strided layout a page of the
request is one 128-lane column window of a query's words at one bit index,
so a block's mask for a page is a single aligned ``[queries, page_size]``
load and a shift.

Layout (differs from ``mla_sparse_qblock``):

* Blocks are aligned to ``query_block``-token tiles of the batch, not to
  requests. A tile spanning several requests is visited once per request;
  consecutive visits keep the tile's output resident and each writes only its
  request's rows. Tiles without real tokens get one visit that writes zeros.
  The kernel therefore reads queries, bitmaps and outputs in place, with no
  gather into a block layout.
* Queries and outputs are head-major, ``[heads, tokens, 512]``: MLA's absorbed
  query is a per-head batched matmul, so the caller passes it without the
  transpose to token-major, and returns the output the same way.
* The softmax is query-major (query rows on sublanes, keys on lanes), over
  ``pages_per_step`` pages per flash step, with every head's rows in one
  matmul and NoPE + RoPE in one score matmul.
"""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sglang_tpu.attention.pallas.dsa_select_bits import (
    select_bits,
    select_bits_num_words,
)

NOPE_DIM = 512
ROPE_DIM = 64
ROPE_LANES = 128  # RoPE row in the cache: 64 FP8 values, then zero padding
# Block shape: the largest query tile with heads * tile <= _MAX_ROWS query
# rows, and as many pages per flash step as keep the f32 score tile at
# _SCORE_ELEMS (12 MiB). Bigger steps amortize the per-step work (accumulator
# rescale, mask build, DMA waits): on TPU7x at 12 heads, 2/4/8 pages per step
# took 1097/691/575 us per 1024-token chunk; at 3 heads, 512-query tiles with
# 16 pages per step 206 us vs 243 us for 256 x 8.
_MAX_ROWS = 3072
_SCORE_ELEMS = 3 * 1024 * 1024
_MAX_PAGES_PER_STEP = 16
_RING_STEPS = 2  # DMA ring depth in steps (prefetch one step ahead)
_VMEM_LIMIT_BYTES = 64 * 1024 * 1024


def _cdiv(a, b):
    return (a + b - 1) // b


def _align(x: int, m: int) -> int:
    return _cdiv(x, m) * m


def _kernel(
    # Scalar prefetch, i32[num_blocks] each (num_live: i32[1])
    num_live_ref,  # blocks that are real visits; later ones are skipped
    tile_ref,  # query tile of the block
    first_ref,  # 1 on the first visit of a tile: zero its output
    tok_lo_ref,  # the block's request owns tokens [tok_lo, tok_hi) of the batch
    tok_hi_ref,
    pos_lo_ref,  # position of token tok_lo in its request
    kv_len_ref,  # kv length of the request
    count_ref,  # pages the block visits: the request's pages up to its last query
    # Inputs
    q_ref,  # bf16[H, QB, 512]
    qpe_ref,  # bf16[H, QB, 128], RoPE lanes 64.. are zero
    member_ref,  # i32[QB, num_words] token bitmaps of the tile's queries
    phys_ref,  # SMEM i32[1, 1, pages_pad] the request's physical pages
    cache_nope_hbm,  # u8[num_pages, page_size, 4, 128]
    cache_rope_hbm,  # u8[num_pages, page_size // 4, 4, 128]
    # Output
    o_ref,  # bf16[H, QB, 512]
    # Scratch
    nope_ring,  # u8[RING_STEPS * S, page_size * 4, 128]
    rope_ring,  # u8[RING_STEPS * S, page_size, 128]
    m_ref,  # f32[H * QB, 1]
    l_ref,  # f32[H * QB, 1]
    acc_ref,  # f32[H * QB, 512]
    qk_ref,  # bf16[H * QB, 640] queries, NoPE then RoPE lanes
    sems,  # DMA [2, RING_STEPS * S]
    *,
    scale: float,
    k_scale: float,
    page_size: int,
    num_words: int,
    pages_per_step: int,
):
    b = pl.program_id(0)
    num_heads, qb, _ = q_ref.shape
    S = pages_per_step
    sk = S * page_size
    num_pages = cache_nope_hbm.shape[0]
    nope_pages = cache_nope_hbm.reshape(num_pages, page_size * 4, 128)
    rope_pages = cache_rope_hbm.reshape(num_pages, page_size, ROPE_LANES)
    word_shift = num_words.bit_length() - 1

    @pl.when(b < num_live_ref[0])
    def _block():
        @pl.when(first_ref[b] != 0)
        def _():
            o_ref[...] = jnp.zeros(o_ref.shape, o_ref.dtype)

        cnt = count_ref[b]
        kv_len = kv_len_ref[b]
        tok_lo = tok_lo_ref[b]
        tok = tile_ref[b] * qb + lax.broadcasted_iota(jnp.int32, (qb, 1), 0)
        mine = (tok >= tok_lo) & (tok < tok_hi_ref[b])  # [QB, 1]
        qpos = jnp.where(mine, pos_lo_ref[b] + tok - tok_lo, -1)
        key_lane = lax.broadcasted_iota(jnp.int32, (1, page_size), 1)
        key_row = lax.broadcasted_iota(jnp.int32, (page_size, 1), 0)
        num_steps = (cnt + S - 1) // S

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

        def for_step_pages(i, fn):
            # Step i holds union entries [i * S, (i + 1) * S) in ring slots
            # [(i % RING_STEPS) * S, + S). Entries past the union are skipped
            # here and masked out below.
            base = lax.rem(i, _RING_STEPS) * S
            for jj in range(S):

                @pl.when(i * S + jj < cnt)
                def _(jj=jj):
                    for cp in copies(i * S + jj, base + jj):
                        fn(cp)

        for d in range(_RING_STEPS - 1):

            @pl.when(d < num_steps)
            def _(d=d):
                for_step_pages(d, lambda cp: cp.start())

        rows = num_heads * qb
        # One score matmul over NoPE and RoPE together: K = 640 takes the
        # MXU passes of 512 + 64 anyway and saves a pass over the scores.
        qk_ref[:, :NOPE_DIM] = q_ref[...].reshape(rows, NOPE_DIM)
        qk_ref[:, NOPE_DIM:] = qpe_ref[...].reshape(rows, ROPE_LANES)
        m_ref[...] = jnp.full(m_ref.shape, -jnp.inf, jnp.float32)
        l_ref[...] = jnp.zeros(l_ref.shape, jnp.float32)
        acc_ref[...] = jnp.zeros(acc_ref.shape, jnp.float32)

        def step(i, carry):
            base = lax.rem(i, _RING_STEPS) * S
            nxt = i + _RING_STEPS - 1

            @pl.when(nxt < num_steps)
            def _():
                for_step_pages(nxt, lambda cp: cp.start())

            # Which keys each query attends, [QB, S * page_size], and which
            # keys hold written cache rows, [S * page_size, 1]. Rows past the
            # kv length were never written and can decode to FP8 NaN (0 * NaN
            # would poison the PV sum); entries past the union are stale slots.
            attend, live = [], []
            for jj in range(S):
                j = i * S + jj
                g = j  # request-relative page
                first = g * page_size
                off = pl.multiple_of(lax.bitwise_and(first, num_words - 1), 128)
                words = member_ref[:, pl.ds(off, page_size)]  # [QB, page]
                bits = lax.shift_right_logical(
                    words, jnp.broadcast_to(first >> word_shift, words.shape)
                )
                n_live = jnp.where(j < cnt, kv_len - first, 0)
                attend.append(
                    ((bits & 1) != 0) & (first + key_lane <= qpos) & (key_lane < n_live)
                )
                live.append(key_row < n_live)
            bias = jnp.where(jnp.concatenate(attend, axis=1), 0.0, -jnp.inf)
            live = jnp.concatenate(live, axis=0)

            for_step_pages(i, lambda cp: cp.wait())
            nope = nope_ring[pl.ds(base, S)].reshape(sk, NOPE_DIM)
            nope = pltpu.bitcast(nope, jnp.float8_e4m3fn).astype(jnp.bfloat16)
            nope = jnp.where(live, nope, 0)
            rope = rope_ring[pl.ds(base, S)].reshape(sk, ROPE_LANES)
            rope = pltpu.bitcast(rope, jnp.float8_e4m3fn).astype(jnp.bfloat16)
            rope = jnp.where(live, rope, 0)

            # All heads in one matmul each way (rows = h * QB + q): the MXU
            # loads the step's keys once and streams H * QB query rows, where
            # one matmul per head would reload them per head.
            s = lax.dot_general(
                qk_ref[...],
                jnp.concatenate([nope, rope], axis=1),
                (((1,), (1,)), ((), ())),
                preferred_element_type=jnp.float32,
            )
            s = (s.reshape(num_heads, qb, sk) * scale + bias[None]).reshape(rows, sk)
            m_prev = m_ref[...]
            m_new = jnp.maximum(m_prev, jnp.max(s, axis=1, keepdims=True))
            shift = jnp.where(m_new == -jnp.inf, 0.0, m_new)
            p = jnp.exp(s - shift)
            alpha = jnp.exp(m_prev - shift)
            l_ref[...] = alpha * l_ref[...] + jnp.sum(p, axis=1, keepdims=True)
            acc_ref[...] = alpha * acc_ref[...] + jnp.dot(
                p.astype(jnp.bfloat16), nope, preferred_element_type=jnp.float32
            )
            m_ref[...] = m_new
            return carry

        lax.fori_loop(0, num_steps, step, 0)

        # k_scale dequantizes V; this block's rows only (other requests'
        # rows of the tile are written by their own visits).
        l = l_ref[...]
        inv = jnp.where(l == 0.0, 0.0, k_scale / l)
        out = (acc_ref[...] * inv).astype(o_ref.dtype)
        o_ref[...] = jnp.where(
            mine[None], out.reshape(num_heads, qb, NOPE_DIM), o_ref[...]
        )


def block_shape(num_heads: int, num_tokens: int, page_size: int) -> tuple[int, int]:
    """Default ``(query_block, pages_per_step)`` (see _MAX_ROWS)."""
    qb = 128
    for cand in (512, 256):
        if num_tokens % cand == 0 and num_heads * cand <= _MAX_ROWS:
            qb = cand
            break
    pps = 1
    while (
        pps * 2 <= _MAX_PAGES_PER_STEP
        and num_heads * qb * page_size * pps * 2 <= _SCORE_ELEMS
    ):
        pps *= 2
    return qb, pps


def _blocks(cu_q_lens, kv_lens, distribution, *, query_block, num_tiles):
    """Visits of (tile, request): every overlap of a tile with a request, in
    tile order, plus one visit per tile without real tokens.

    Returns per visit ``(tile, first, tok_lo, tok_hi, pos_lo, kv_len, req)``
    padded to the static bound ``num_tiles + max_num_seqs``, and the number
    of real visits.
    """
    max_num_seqs = cu_q_lens.shape[0] - 1
    num_blocks = num_tiles + max_num_seqs
    real = jnp.arange(max_num_seqs) < distribution[2]
    lo = cu_q_lens[:-1]
    hi = jnp.where(real, cu_q_lens[1:], lo)
    t_lo = jnp.arange(num_tiles, dtype=jnp.int32)[:, None] * query_block
    overlap = (lo[None, :] < t_lo + query_block) & (hi[None, :] > t_lo)
    overlap &= (hi > lo)[None, :]  # [tiles, seqs]
    empty = ~jnp.any(overlap, axis=1)
    # An empty tile takes request slot 0 with an empty token range.
    visit = overlap.at[:, 0].set(overlap[:, 0] | empty)
    flat = visit.reshape(-1)
    slot = jnp.cumsum(flat) - 1
    dst = jnp.where(flat, slot, num_blocks)
    tile_of = jnp.repeat(jnp.arange(num_tiles, dtype=jnp.int32), max_num_seqs)
    req_of = jnp.tile(jnp.arange(max_num_seqs, dtype=jnp.int32), num_tiles)
    empty_of = jnp.repeat(empty, max_num_seqs)

    def compact(x, fill):
        out = jnp.full((num_blocks,), fill, jnp.int32)
        return out.at[dst].set(x.astype(jnp.int32), mode="drop")

    num_live = jnp.sum(flat).astype(jnp.int32)
    tile = compact(tile_of, num_tiles - 1)
    req = compact(req_of, 0)
    is_empty = compact(empty_of, 1) != 0
    first = jnp.concatenate(
        [jnp.ones((1,), jnp.int32), (tile[1:] != tile[:-1]).astype(jnp.int32)]
    )
    q_lens = hi - lo
    tok_lo = jnp.where(is_empty, 0, lo[req])
    tok_hi = jnp.where(is_empty, 0, hi[req])
    kv_len = jnp.where(is_empty, 0, kv_lens[req])
    pos_lo = kv_len - q_lens[req]
    return tile, first, tok_lo, tok_hi, pos_lo, kv_len, req, num_live


@functools.partial(
    jax.jit,
    static_argnames=("sm_scale", "k_scale", "query_block", "pages_per_step"),
)
def sparse_mla_prefill_qblock_tokens(
    ql_nope: jax.Array,  # bf16[num_q_heads, num_tokens, 512], head-major
    q_pe: jax.Array,  # bf16[num_q_heads, num_tokens, 64], head-major
    cache_nope: jax.Array,  # u8[num_pages, page_size, 4, 128]
    cache_rope: jax.Array,  # u8[num_pages, page_size // 4, 4, 128]
    topk_indices: jax.Array,  # i32[num_tokens, topk], -1 padded
    kv_lens: jax.Array,  # i32[max_num_seqs]; counts this step's tokens
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]; [2] is the number of real requests
    *,
    sm_scale: float,
    k_scale: float,
    query_block: int | None = None,
    pages_per_step: int | None = None,
) -> jax.Array:
    """Sparse MLA attention over exact token top-k rows, a tile at a time.

    Same result as ``sparse_ragged_paged_attention`` on the same rows up to
    summation order and bf16 rounding of the softmax weights. The cache must
    already hold this step's tokens. ``num_tokens`` must be a multiple of
    128 (of ``query_block`` when given).

    Returns ``bf16[num_q_heads, num_tokens, 512]``; padding tokens are 0.
    """
    num_heads, num_tokens, _ = ql_nope.shape
    page_size = cache_nope.shape[1]
    max_num_seqs = kv_lens.shape[0]
    pages_per_seq = page_indices.shape[0] // max_num_seqs
    qb = query_block or block_shape(num_heads, num_tokens, page_size)[0]
    if pages_per_step is None:
        pages_per_step = block_shape(num_heads, num_tokens, page_size)[1]
    if num_tokens % qb or qb % 8:
        raise ValueError(f"{num_tokens=} must be a multiple of {query_block=} (% 8)")
    if page_size % 128 or page_size & (page_size - 1):
        raise ValueError(
            f"page_size must be a power-of-two multiple of 128: {page_size}"
        )
    num_tiles = num_tokens // qb

    # Token bitmaps of every query, [num_tokens, num_words], strided layout.
    num_words = select_bits_num_words(pages_per_seq * page_size, page_size)
    member = select_bits(topk_indices, num_words=num_words)

    tile, first, tok_lo, tok_hi, pos_lo, kv_len, req, num_live = _blocks(
        cu_q_lens, kv_lens, distribution, query_block=qb, num_tiles=num_tiles
    )
    num_blocks = tile.shape[0]

    # A visit reads its request's pages up to its last query's position. A
    # block of queries with ~2048 picks each selects nearly every one of them,
    # so this is the union of their selections in practice; pages nobody
    # picked are masked out.
    last = jnp.minimum(tok_hi, (tile + 1) * qb) - 1
    reach = jnp.minimum(kv_len, pos_lo + last - tok_lo + 1)
    counts = jnp.where(tok_hi > tok_lo, _cdiv(reach, page_size), 0).astype(jnp.int32)
    pages_pad = _align(pages_per_seq, pages_per_step)
    phys = page_indices.reshape(max_num_seqs, pages_per_seq)[req]
    phys = jnp.pad(phys, ((0, 0), (0, pages_pad - pages_per_seq)))
    q_pe = jnp.pad(q_pe, ((0, 0), (0, 0), (0, ROPE_LANES - q_pe.shape[-1])))

    kernel = functools.partial(
        _kernel,
        scale=sm_scale * k_scale,
        k_scale=k_scale,
        page_size=page_size,
        num_words=num_words,
        pages_per_step=pages_per_step,
    )

    def live(b, num_live_ref):
        # Visits past the real ones repeat the last real one: no new DMA in or
        # out, and the kernel skips them.
        return jnp.maximum(jnp.minimum(b, num_live_ref[0] - 1), 0)

    def tile_map(b, num_live_ref, tile_ref, *_):
        return tile_ref[live(b, num_live_ref)]

    ring = _RING_STEPS * pages_per_step
    out = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=8,
            grid=(num_blocks,),
            in_specs=[
                pl.BlockSpec(
                    (num_heads, qb, NOPE_DIM), lambda b, *s: (0, tile_map(b, *s), 0)
                ),
                pl.BlockSpec(
                    (num_heads, qb, ROPE_LANES), lambda b, *s: (0, tile_map(b, *s), 0)
                ),
                pl.BlockSpec((qb, num_words), lambda b, *s: (tile_map(b, *s), 0)),
                pl.BlockSpec(
                    (1, 1, pages_pad),
                    lambda b, nl, *_: (live(b, nl), 0, 0),
                    memory_space=pltpu.SMEM,
                ),
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM),
            ],
            out_specs=pl.BlockSpec(
                (num_heads, qb, NOPE_DIM), lambda b, *s: (0, tile_map(b, *s), 0)
            ),
            scratch_shapes=[
                pltpu.VMEM((ring, page_size * 4, 128), cache_nope.dtype),
                pltpu.VMEM((ring, page_size, ROPE_LANES), cache_rope.dtype),
                pltpu.VMEM((num_heads * qb, 1), jnp.float32),
                pltpu.VMEM((num_heads * qb, 1), jnp.float32),
                pltpu.VMEM((num_heads * qb, NOPE_DIM), jnp.float32),
                pltpu.VMEM((num_heads * qb, NOPE_DIM + ROPE_LANES), ql_nope.dtype),
                pltpu.SemaphoreType.DMA((2, ring)),
            ],
        ),
        out_shape=jax.ShapeDtypeStruct(ql_nope.shape, ql_nope.dtype),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            vmem_limit_bytes=_VMEM_LIMIT_BYTES,
        ),
        name="sparse_mla_prefill_qblock_tokens",
    )(
        num_live.reshape(1),
        tile,
        first,
        tok_lo,
        tok_hi,
        pos_lo,
        kv_len,
        counts,
        ql_nope,
        q_pe,
        member,
        phys.reshape(num_blocks, 1, pages_pad),
        cache_nope,
        cache_rope,
    )
    return out
