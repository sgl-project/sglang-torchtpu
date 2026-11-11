# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3.1's DSA indexer on the K3 programs' decode context parallelism.

Index keys live in one uint8 pool per MLA layer beside the unified pools
(``kimi_k3/layout.index_block_shape``): pool row ``r`` holds the index keys of
the tokens whose latents are in block ``r`` (``layout.tokens_per_block`` of
them, the rank's DCP chunk), in 128-token sub-pages of 33 rows x 4 x 128
bytes: rows 0..31 hold four tokens' 128 fp8 key bytes each (token ``t`` of the
sub-page: row ``t // 4``, sublane ``t % 4``), row 32 sublane 0 lane ``t``
holds its UE8M0 scale byte (``exponent + 127``). The selection itself is in
``kimi_k3/dsa.py`` (reference) and the decode kernel.

This module holds the layout helpers and the XLA-level commit of the new
tokens' index keys (after a decode step, as ``paged.commit_latent_rows`` for
the latents).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from sglang_tpu.k3_programs.kimi_k3.layout import INDEX_SUBPAGE_ROWS, INDEX_SUBPAGE_TOKENS

HEAD_DIM = 128
RECORD = HEAD_DIM + 1  # key bytes, then the scale byte


def record_place(token):
    """``(row, sublane)`` of token ``token``'s key bytes in a pool row, and the
    ``(row, lane)`` of its scale byte (sublane 0); ``token`` is the offset in
    the row (traced or not)."""
    subpage, t = token // INDEX_SUBPAGE_TOKENS, token % INDEX_SUBPAGE_TOKENS
    base = subpage * INDEX_SUBPAGE_ROWS
    return (base + t // 4, t % 4), (base + INDEX_SUBPAGE_ROWS - 1, t)


def key_records(keys: jax.Array) -> jax.Array:
    """Normed index keys f32/bf16 ``[..., 128]`` -> uint8 records ``[..., 129]``
    (fp8 E4M3 bytes at the UE8M0 scale, then ``exponent + 127``), as
    ``kimi_k3/dsa.quantize_fp8``."""
    x = keys.astype(jnp.float32)
    bits = jax.lax.bitcast_convert_type(jnp.max(jnp.abs(x), axis=-1, keepdims=True), jnp.int32)
    biased = bits >> 23
    exp = biased - (127 + 8) + ((bits & 0x7FFFFF) > 0x600000).astype(jnp.int32)
    exp = jnp.where(biased == 0, jnp.where(bits == 0, 0, -126), exp)
    exp = jnp.clip(exp, -126, 127)
    inv = jax.lax.bitcast_convert_type((127 - exp) << 23, jnp.float32)
    values = jnp.clip(x * inv, -448.0, 448.0).astype(jnp.float8_e4m3fn)
    key_bytes = jax.lax.bitcast_convert_type(values, jnp.uint8)
    return jnp.concatenate([key_bytes, (exp + 127).astype(jnp.uint8)], axis=-1)


def commit_index_rows(index_pools, records, block_tables, positions, *, group=32,
                      owner_table=None, page_tokens=None, global_pages=False):
    """Write each row's new index-key records into the DSA index pools on the
    rank that owns its position (XLA level, inside ``shard_map`` over ``tp``):
    MLA layer ``m``'s record ``records[b, m]`` (uint8 ``[129]``) of the token
    at ``positions[b]`` lands in ``index_pools[m]`` in the block the row's
    latent goes to (``paged.commit_latent_rows``'s rule: global page ``p //
    page_tokens`` is owned by member ``p % group``, local tables hold one entry
    per ``group`` pages). Rows this rank does not own are dropped."""
    from sglang_tpu.k3_programs.kimi_k3.kernels import mla_dcp

    member = mla_dcp.owner_index(owner_table) % group
    pools = list(index_pools)
    if page_tokens is None:
        subpages = pools[0].shape[1] // INDEX_SUBPAGE_ROWS
        page_tokens = subpages * INDEX_SUBPAGE_TOKENS
    rows = records.shape[0]
    pos = positions[:rows]
    pages = pos // page_tokens
    owned = pages % group == member
    blocks = block_tables[jnp.arange(rows), pages if global_pages else pages // group]
    (key_row, sublane), (scale_row, lane) = record_place(pos % page_tokens)
    for m, pool in enumerate(pools):
        target = jnp.where(owned, blocks, pool.shape[0])
        pool = pool.at[target, key_row, sublane, :].set(records[:, m, :HEAD_DIM], mode="drop")
        pools[m] = pool.at[target, scale_row, 0, lane].set(records[:, m, HEAD_DIM], mode="drop")
    return tuple(pools)


def index_rows(keys: jax.Array, pages: int) -> jax.Array:
    """A prefill window's normed index keys ``[W, 128]`` -> index-pool rows
    ``[pages, R, 4, 128]`` uint8, ``W / pages`` tokens a row (the layout
    above; what ``commit_index_rows`` writes token by token)."""
    records = key_records(keys)  # [W, 129]
    subpages = keys.shape[0] // pages // INDEX_SUBPAGE_TOKENS
    key_rows = records[:, :HEAD_DIM].reshape(pages, subpages, 32, 4, HEAD_DIM)
    scale_rows = jnp.zeros((pages, subpages, 1, 4, HEAD_DIM), jnp.uint8).at[:, :, 0, 0, :].set(
        records[:, HEAD_DIM].reshape(pages, subpages, INDEX_SUBPAGE_TOKENS))
    return jnp.concatenate((key_rows, scale_rows), axis=2).reshape(
        pages, subpages * INDEX_SUBPAGE_ROWS, 4, HEAD_DIM)


def read_records(pool_row: jax.Array, tokens: int) -> tuple[jax.Array, jax.Array]:
    """One index pool row -> (fp8-valued key f32 ``[tokens, 128]``, exponent
    i32 ``[tokens, 1]``): the inverse of the layout (host checks and tests)."""
    subpages = tokens // INDEX_SUBPAGE_TOKENS
    rows = pool_row.reshape(subpages, INDEX_SUBPAGE_ROWS, 4, HEAD_DIM)
    keys = rows[:, :32].reshape(subpages * INDEX_SUBPAGE_TOKENS, HEAD_DIM)
    exps = rows[:, 32, 0].reshape(subpages * INDEX_SUBPAGE_TOKENS, 1).astype(jnp.int32) - 127
    values = jax.lax.bitcast_convert_type(keys, jnp.float8_e4m3fn).astype(jnp.float32)
    return values, exps


# ------------------------------------------------------------------ selection kernel
INT_MIN = -(2**31)
INT_MAX = 2**31 - 1
LANES = 128
DIGIT_BITS = 4
CANDIDATES = (1 << DIGIT_BITS) - 1  # thresholds counted per round
POSITION_BITS = 24
KEY_SHIFTS = tuple(range(32 - DIGIT_BITS, -1, -DIGIT_BITS))  # 8 rounds
POSITION_SHIFTS = tuple(range(POSITION_BITS - DIGIT_BITS, -1, -DIGIT_BITS))  # 6 rounds (ties only)
CHUNK_ROWS = 64  # key rows (of 128 tokens) a counting step reads
# DMA ring depths: a page or chunk step computes little, so several copies stay
# in flight (one HBM -> VMEM copy is ~0.9 us; a ring of two waits on each).
PAGE_SLOTS = 4  # index pages (scoring)
KEY_SLOTS = 8  # key chunks (counting, step activity)
WIDE_BATCH = 64  # from this many rows: rings of 3 and 4 (the decode's VMEM plan)


def ring_slots(batch: int) -> tuple[int, int]:
    """(page ring, key ring) depths of a ``batch``-row selection."""
    return (PAGE_SLOTS, KEY_SLOTS) if batch < WIDE_BATCH else (min(PAGE_SLOTS, 3), min(KEY_SLOTS, 4))
COUNT_ROWS = 16  # candidate sublanes of the count exchange
_PREFIX, _UPPER, _LOWER, _POSITION, _LIMIT = range(5)  # state rows (lanes: batch rows)


def key_rows_for(pages: int, page_tokens: int, span: int = 1) -> int:
    """Rows of a row's keys in the keys scratch: ``pages`` local pages
    (rounded up to whole ``span``-page attention steps), 128 keys a row,
    in whole vregs."""
    rows = -(-pages // span) * span * page_tokens // LANES
    return -(-rows // 8) * 8


def _i32(value: int) -> int:
    """A Python int as the int32 of the same low 32 bits."""
    value &= 0xFFFFFFFF
    return value - (1 << 32) if value >= 1 << 31 else value


def selected(keys, positions, tau, p_tie):
    """The selection mask of keys at positions given a row's threshold."""
    return (keys > tau) | ((keys == tau) & (positions <= p_tie))


def _monotone(scores):
    """f32 -> int32 keys of the same order (``kimi_k3/dsa.monotone_key``)."""
    bits = jax.lax.bitcast_convert_type(scores, jnp.int32)
    return bits ^ ((bits >> 31) & jnp.int32(0x7FFFFFFF))


def _pow2(biased):
    """``2^(biased - 127)`` for an int32 biased exponent (UE8M0 byte)."""
    return jax.lax.bitcast_convert_type(biased << 23, jnp.float32)


def record_key(q32, k, w_col, biased_exp):
    """The key of one record (``k`` f32 ``[1, 128]`` fp8 values, its UE8M0
    byte ``biased_exp``) for a query (``q32`` f32 ``[32, 128]``, head weights
    ``w_col`` ``[32, 1]``): ``[1, LANES]`` (every lane)."""
    logits = jnp.sum(q32 * k, axis=1, keepdims=True)  # [32, 1]
    score = jnp.sum(jnp.maximum(logits, 0.0) * w_col, axis=0, keepdims=True)  # [1, 1]
    exp = jnp.full((1, LANES), biased_exp, jnp.int32)
    return _monotone(jnp.broadcast_to(score, (1, LANES)) * _pow2(exp))


def _row_scalar(vector, row):
    """Lane ``row`` of a ``[1, LANES]`` int32 vector as a scalar."""
    lane = jax.lax.broadcasted_iota(jnp.int32, vector.shape, 1)
    return jnp.sum(jnp.where(lane == row, vector, 0))


def _locate(starts_ref, step, batch):
    """``(row, step - starts[row])`` with ``row`` the last with ``starts[row] <= step``."""
    row = jnp.int32(0)
    for bit in range(max(batch - 1, 1).bit_length() - 1, -1, -1):
        candidate = jnp.minimum(row + (1 << bit), batch - 1)
        row = jnp.where(starts_ref[candidate] <= step, candidate, row)
    return row, step - starts_ref[row]


class IndexSelect:
    """The exact top-``topk`` index selection of ``batch`` decode rows over a
    DCP group, inside a Pallas kernel (every rank of the group runs it).

    Row ``b`` (``position_ref[b]`` its token's position) selects among
    positions ``0 .. position_ref[b]``: the pool tokens below its sequence's
    first position (``sequence_rows`` consecutive rows are one sequence's
    in-flight tokens) and the sequence's in-flight tokens up to its own, whose
    records (``new_k8_ref`` f32 ``[B, 128]`` fp8 values, ``new_exp_ref`` SMEM
    ``[B]`` UE8M0 bytes) are not in the pool yet.

    1. ``score``: each rank scores its own pages (one index page per step, a
       DMA ring) into keys, the monotone int32 of ``sum_h relu(q_h . k) w_h
       2^e``, ``INT_MIN`` past the row's tokens; in-flight tokens on its pages
       are scored from their records. Keys go to the HBM scratch ``keys``
       ``[B, pages * T / 128, 128]`` (row ``b``'s local page ``j`` at rows
       ``j T / 128 ..``).
    2. ``select``: the threshold ``tau``, the ``topk``-th largest key, by
       rounds over its bits, ``DIGIT_BITS`` a round: every rank counts its
       keys at or above ``CANDIDATES`` thresholds per row (full vregs, one
       compare and add each), the counts are summed over the group (remote
       copies into every peer's slot), and the largest threshold still
       reaching ``topk`` extends the prefix. When more keys tie at ``tau``
       than remain to take, rounds over the tied keys' positions find
       ``p_tie``, keeping the lowest positions. Rows with at most ``topk``
       tokens select everything (``tau = INT_MIN``, ``p_tie = -1``).

    ``tau`` / ``p_tie`` land in SMEM (``refs["tau"]``, ``refs["p_tie"]``); a
    local token is selected when ``selected(key, position, tau, p_tie)``.
    ``refs["own"][b]`` is 1 when row ``b``'s own token is selected (every
    rank computes the in-flight keys alike: ``in_flight``).
    """

    def __init__(self, *, pools, layer, table_ref, pages, page_tokens, group, owner, member, base,
                 position_ref, new_exp_ref, batch, topk, refs, sequence_rows=1, global_pages=False,
                 table_row=None):
        self.pools, self.layer, self.table_ref, self.pages = tuple(pools), layer, table_ref, pages
        self.T, self.group, self.owner, self.member, self.base = page_tokens, group, owner, member, base
        self.position_ref, self.new_exp_ref, self.batch, self.topk = position_ref, new_exp_ref, batch, topk
        self.refs, self.S, self.global_pages, self.table_row = refs, sequence_rows, global_pages, table_row
        self.subpages = page_tokens // INDEX_SUBPAGE_TOKENS
        self.key_rows = refs["keys"].shape[1]  # >= pages * subpages, whole vregs
        self.page_slots, self.key_slots = refs["page_buf"].shape[0], refs["key_buf"].shape[0]
        self.chunk = min(CHUNK_ROWS, self.key_rows)
        if batch > LANES or self.key_rows % 8 or batch % sequence_rows:
            raise ValueError("index selection: batch <= 128, key rows in whole vregs, whole sequences")

    # ------------------------------------------------------------------ plan
    def plan(self):
        """Per-row local page counts and the flat step starts (SMEM)."""
        refs, T, group = self.refs, self.T, self.group
        page_start, chunk_start = jnp.int32(0), jnp.int32(0)
        state = jnp.zeros((8, LANES), jnp.int32)
        lane = jax.lax.broadcasted_iota(jnp.int32, (8, LANES), 1)
        sub = jax.lax.broadcasted_iota(jnp.int32, (8, LANES), 0)
        seq_count = jnp.int32(0)
        for b in range(self.batch):
            limit = self.position_ref[b] + 1
            needed = (limit + T - 1) // T
            count = jnp.maximum(needed - self.owner + group - 1, 0) // group
            refs["counts"][b] = count
            refs["chunk_starts"][b] = chunk_start
            chunk_start = chunk_start + (count * self.subpages + self.chunk - 1) // self.chunk
            state = jnp.where((lane == b) & (sub == _LIMIT), limit, state)
            # Scoring steps go by sequence (its rows share their pages): its
            # longest row's page count.
            seq_count = count if b % self.S == 0 else jnp.maximum(seq_count, count)
            if b % self.S == self.S - 1:
                refs["page_starts"][b // self.S] = page_start
                page_start = page_start + seq_count
        refs["page_starts"][self.batch // self.S] = page_start
        refs["chunk_starts"][self.batch] = chunk_start
        refs["state"][...] = state

    # ------------------------------------------------------------------ scoring
    def _w_col(self, w_t_ref, row):
        lane = jax.lax.broadcasted_iota(jnp.int32, w_t_ref.shape, 1)
        return jnp.sum(jnp.where(lane == row, w_t_ref[...], 0.0), axis=1, keepdims=True)  # [32, 1]

    def in_flight(self, query, w_t_ref, new_k8_ref):
        """``refs["in_flight"][b, n]``: the key of row ``b``'s sequence's
        in-flight token ``n`` (rows ``b - b % S + n``, up to ``b``) under row
        ``b``'s query, ``INT_MIN`` past its own."""
        lane = jax.lax.broadcasted_iota(jnp.int32, (1, LANES), 1)
        for b in range(self.batch):
            first, r = b - b % self.S, b % self.S
            q32 = query(b).astype(jnp.float32)
            w_col = self._w_col(w_t_ref, b)
            keys = jnp.full((1, LANES), INT_MIN, jnp.int32)
            for n in range(r + 1):
                key = record_key(q32, new_k8_ref[first + n:first + n + 1, :], w_col,
                                 self.new_exp_ref[first + n])
                keys = jnp.where(lane == n, key, keys)
            self.refs["in_flight"][b:b + 1, :] = keys

    def _block(self, row, local):
        entry = local * self.group + self.member if self.global_pages else local
        if self.table_row is not None:
            row = self.table_row(row)
        return self.table_ref[row * self.pages + entry]

    def _page_copy(self, step):
        from jax.experimental.pallas import tpu as pltpu

        seq, local = _locate(self.refs["page_starts"], step, self.batch // self.S)
        slot = step % self.page_slots
        block = self._block(seq * self.S, local)
        for index, pool in enumerate(self.pools):
            @pl_when(self.layer == index if len(self.pools) > 1 else True)
            def start(pool=pool):
                pltpu.make_async_copy(pool.at[block], self.refs["page_buf"].at[slot],
                                      self.refs["page_sems"].at[slot]).start()

    def _wait_page(self, step):
        from jax.experimental.pallas import tpu as pltpu

        buf = self.refs["page_buf"].at[step % self.page_slots]
        pltpu.make_async_copy(buf, buf, self.refs["page_sems"].at[step % self.page_slots]).wait()

    def _stage_copy(self, slot, row=0, local=0):
        from jax.experimental.pallas import tpu as pltpu

        return pltpu.make_async_copy(
            self.refs["stage"].at[slot],
            self.refs["keys"].at[pl_ds(row, self.S), pl_ds(local * self.subpages, self.subpages)],
            self.refs["stage_sems"].at[slot])

    def score(self, query, w_t_ref, query_rows=None):
        """``query(row)``: the row's fp8 ``[32, 128]`` index query heads
        (``row`` traced or not); ``query_rows(first)``: the ``S`` rows of the
        sequence starting at row ``first`` as one ``[S * 32, 128]`` block
        (default: ``query``'s, concatenated); ``w_t_ref`` f32 ``[32, 128]``
        head weights (column ``b``: row ``b``); after ``in_flight``. A
        sequence's rows share its pages: each page is copied and scored once
        for all of them (one ``[S * 32, 128] x [128, 128]`` product a subpage)."""
        refs, T, S = self.refs, self.T, self.S
        sequences = self.batch // S
        if query_rows is None:
            query_rows = lambda first: (jnp.concatenate([query(first + r) for r in range(S)], axis=0)
                                        if S > 1 else query(first))
        total = refs["page_starts"][sequences]
        for step in range(self.page_slots - 1):
            pl_when(step < total)(lambda step=step: self._page_copy(step))
        lane_row = jax.lax.broadcasted_iota(jnp.int32, (1, LANES), 1)

        def page_step(step, carry):
            seq, local = _locate(refs["page_starts"], step, sequences)
            first_row = seq * S
            slot, stage = step % self.page_slots, step % 2
            self._wait_page(step)
            pl_when(step + self.page_slots - 1 < total)(lambda: self._page_copy(step + self.page_slots - 1))
            pl_when(step >= 2)(lambda: self._stage_copy(stage).wait())
            q = query_rows(first_row)  # [S * 32, 128] fp8
            w_cols = [self._w_col(w_t_ref, first_row + r) for r in range(S)]
            limits = [self.position_ref[first_row + r] + 1 for r in range(S)]
            first = (local * self.group + self.owner) * T  # the page's first position

            def subpage_keys(sp):
                raw = refs["page_buf"][slot, pl_ds(sp * INDEX_SUBPAGE_ROWS, 32)]  # [32, 4, 128] u8
                return jax.lax.bitcast_convert_type(raw.reshape(INDEX_SUBPAGE_TOKENS, 128), jnp.float8_e4m3fn)

            def subpage_scale(sp):
                return _pow2(refs["page_buf"][slot, sp * INDEX_SUBPAGE_ROWS + 32].astype(jnp.int32)[0:1])

            # Up to two rows a sequence: one product over the whole page (a
            # [32, 128] x [128, 128] product per subpage is all MXU latency);
            # from four, a subpage at a time (the page's logits outgrow the vregs).
            parts = [tuple(range(self.subpages))] if S <= 2 else [(sp,) for sp in range(self.subpages)]
            for part in parts:
                k8 = jnp.concatenate([subpage_keys(sp) for sp in part], axis=0) if len(part) > 1 \
                    else subpage_keys(part[0])
                logits = jax.lax.dot_general(q, k8, (((1,), (1,)), ((), ())),
                                             preferred_element_type=jnp.float32)  # [S * 32 heads, tokens]
                scales = [subpage_scale(sp) for sp in part]
                for r in range(S):
                    heads = logits[r * 32:(r + 1) * 32]
                    score = jnp.sum(jnp.maximum(heads, 0.0) * w_cols[r], axis=0, keepdims=True)  # [1, tokens]
                    for i, sp in enumerate(part):
                        position = first + sp * INDEX_SUBPAGE_TOKENS + lane_row
                        piece = score[:, i * INDEX_SUBPAGE_TOKENS:(i + 1) * INDEX_SUBPAGE_TOKENS]
                        key = jnp.where(position < limits[r], _monotone(piece * scales[i]), INT_MIN)
                        refs["stage"][stage, r, pl_ds(sp, 1), :] = key
            for r in range(S):
                self._patch_in_flight(first_row + r, stage, first, r)
            self._stage_copy(stage, first_row, local).start()
            return carry

        jax.lax.fori_loop(0, total, page_step, 0)
        for back in (1, 2):
            pl_when(total >= back)(lambda back=back: self._stage_copy((total - back) % 2).wait())

    def _patch_in_flight(self, row, stage, first, stage_row=0):
        """Keys of the row's sequence's in-flight tokens (up to its own) that
        fall on this page, from ``refs["in_flight"]`` (into the stage's
        ``stage_row``)."""
        refs, S = self.refs, self.S
        seq_first = row - row % S if S > 1 else row
        lane_row = jax.lax.broadcasted_iota(jnp.int32, (1, LANES), 1)
        lo, hi = first, first + self.T

        @pl_when((self.position_ref[seq_first] < hi) & (self.position_ref[row] >= lo))
        def patch():
            table = refs["in_flight"][pl_ds(row, 1), :]  # [1, LANES]
            for n in range(S):
                token_row = seq_first + n
                position = self.position_ref[token_row]

                @pl_when((token_row <= row) & (position >= lo) & (position < hi))
                def one(n=n, position=position):
                    key = jnp.full((1, LANES), _row_scalar(table, n), jnp.int32)
                    offset = position - lo
                    at = pl_ds(offset // INDEX_SUBPAGE_TOKENS, 1)
                    refs["stage"][stage, stage_row, at, :] = jnp.where(
                        lane_row == offset % INDEX_SUBPAGE_TOKENS, key, refs["stage"][stage, stage_row, at, :])

    # ------------------------------------------------------------------ selection
    def _chunk_copy(self, step):
        from jax.experimental.pallas import tpu as pltpu

        row, local = _locate(self.refs["chunk_starts"], step, self.batch)
        start = jnp.minimum(local * self.chunk, self.key_rows - self.chunk)
        pltpu.make_async_copy(self.refs["keys"].at[row, pl_ds(start, self.chunk)],
                              self.refs["key_buf"].at[step % self.key_slots],
                              self.refs["key_sems"].at[step % self.key_slots]).start()

    def _wait_chunk(self, step):
        from jax.experimental.pallas import tpu as pltpu

        buf = self.refs["key_buf"].at[step % self.key_slots]
        pltpu.make_async_copy(buf, buf, self.refs["key_sems"].at[step % self.key_slots]).wait()

    def _count(self, thresholds_of):
        """This rank's counts ``[COUNT_ROWS, LANES]`` (sublane ``j``: candidate
        ``j + 1``, lane ``b``: row ``b``) of its keys passing
        ``thresholds_of(row)``'s tests: a callable ``(row) -> test(key, position,
        j)`` for ``j < CANDIDATES``."""
        refs, chunk = self.refs, self.chunk
        total = refs["chunk_starts"][self.batch]
        refs["local"][...] = jnp.zeros((COUNT_ROWS, LANES), jnp.int32)
        for step in range(self.key_slots - 1):
            pl_when(step < total)(lambda step=step: self._chunk_copy(step))
        sub8 = jax.lax.broadcasted_iota(jnp.int32, (8, LANES), 0)
        lane8 = jax.lax.broadcasted_iota(jnp.int32, (8, LANES), 1)
        cand = jax.lax.broadcasted_iota(jnp.int32, (COUNT_ROWS, LANES), 0)
        lane16 = jax.lax.broadcasted_iota(jnp.int32, (COUNT_ROWS, LANES), 1)
        zeros = tuple(jnp.zeros((8, LANES), jnp.int32) for _ in range(CANDIDATES))

        def step_fn(step, accs):
            row, local = _locate(refs["chunk_starts"], step, self.batch)
            self._wait_chunk(step)
            pl_when(step + self.key_slots - 1 < total)(lambda: self._chunk_copy(step + self.key_slots - 1))
            start = jnp.minimum(local * chunk, self.key_rows - chunk)
            rows_valid = refs["counts"][row] * self.subpages
            accs = tuple(jnp.where(local == 0, 0, a) for a in accs)
            test = thresholds_of(row)
            accs = list(accs)
            for v in range(chunk // 8):
                key = refs["key_buf"][step % self.key_slots, pl_ds(v * 8, 8), :]
                key_row = start + v * 8 + sub8
                valid = (key_row >= local * chunk) & (key_row < rows_valid)
                page = key_row // self.subpages
                position = ((page * self.group + self.owner) * self.T
                            + (key_row % self.subpages) * INDEX_SUBPAGE_TOKENS + lane8)
                for j in range(CANDIDATES):
                    accs[j] = accs[j] + (valid & test(key, position, j)).astype(jnp.int32)
            last = local == (rows_valid + chunk - 1) // chunk - 1

            @pl_when(last)
            def column():
                # Sublane sums stacked into one [16, LANES] block, then a single
                # lane reduction (not a full reduction per candidate).
                partial = jnp.concatenate(
                    [jnp.sum(a, axis=0, keepdims=True) for a in accs]
                    + [jnp.zeros((COUNT_ROWS - CANDIDATES, LANES), jnp.int32)], axis=0)  # [16, LANES]
                total = jnp.sum(partial, axis=1, keepdims=True)  # [16, 1]
                refs["local"][...] = jnp.where(lane16 == row, total, refs["local"][...])

            return tuple(accs)

        jax.lax.fori_loop(0, total, step_fn, zeros)
        return refs["local"][...]

    def _all_reduce(self, local, parity):
        from jax.experimental import pallas as pl
        from jax.experimental.pallas import tpu as pltpu

        refs, group, member = self.refs, self.group, self.member
        refs["exchange"][parity, member] = local
        for peer in range(group):
            @pl.when(peer != member)
            def send(peer=peer):
                pltpu.make_async_remote_copy(
                    refs["exchange"].at[parity, member], refs["exchange"].at[parity, member],
                    refs["exchange_sems"].at[0, parity], refs["exchange_sems"].at[1, parity],
                    device_id=(self.base + peer,), device_id_type=pl.DeviceIdType.MESH,
                ).start()
        others = refs["exchange"].at[parity, pl.ds(0, group - 1)]
        pltpu.make_async_copy(others, others, refs["exchange_sems"].at[0, parity]).wait()
        pltpu.make_async_copy(others, others, refs["exchange_sems"].at[1, parity]).wait()
        return jnp.sum(refs["exchange"][parity], axis=0)  # [COUNT_ROWS, LANES]

    def select(self):
        """The rounds; ``tau`` / ``p_tie`` per row into SMEM."""
        refs, topk = self.refs, self.topk
        cand = jax.lax.broadcasted_iota(jnp.int32, (COUNT_ROWS, LANES), 0)
        parity = 0
        for shift in KEY_SHIFTS:
            def key_test(row, shift=shift):
                prefix = _row_scalar(refs["state"][_PREFIX:_PREFIX + 1], row)
                return lambda key, position, j: key >= ((prefix + _i32((j + 1) << shift)) ^ INT_MIN)

            total = self._all_reduce(self._count(key_test), parity)
            parity ^= 1
            ge = ((total >= topk) & (cand < CANDIDATES)).astype(jnp.int32)
            digit = jnp.sum(ge, axis=0, keepdims=True)  # [1, LANES]
            upper = jnp.sum(jnp.where(cand == digit, total, 0), axis=0, keepdims=True)
            lower = jnp.sum(jnp.where(cand == digit - 1, total, 0), axis=0, keepdims=True)
            state = refs["state"]
            state[_PREFIX:_PREFIX + 1] = state[_PREFIX:_PREFIX + 1] + (digit << shift)  # wraps: unsigned bits
            state[_UPPER:_UPPER + 1] = jnp.where(digit < CANDIDATES, upper, state[_UPPER:_UPPER + 1])
            state[_LOWER:_LOWER + 1] = jnp.where(digit >= 1, lower, state[_LOWER:_LOWER + 1])
        state = refs["state"]
        lane = jax.lax.broadcasted_iota(jnp.int32, (1, LANES), 1)
        active = (state[_LIMIT:_LIMIT + 1] > topk) & (lane < self.batch)
        tie = active & (state[_LOWER:_LOWER + 1] > topk)  # more tied keys than the rest of the top-k
        state[_POSITION:_POSITION + 1] = jnp.zeros((1, LANES), jnp.int32)

        @pl_when(jnp.sum(tie.astype(jnp.int32)) > 0)
        def ties():
            nonlocal_parity = parity
            for shift in POSITION_SHIFTS:
                def position_test(row, shift=shift):
                    tau = _row_scalar(refs["state"][_PREFIX:_PREFIX + 1], row) ^ INT_MIN
                    prefix = _row_scalar(refs["state"][_POSITION:_POSITION + 1], row)
                    return lambda key, position, j: (key == tau) & (position < prefix + ((j + 1) << shift))

                total = self._all_reduce(self._count(position_test), nonlocal_parity)
                nonlocal_parity ^= 1
                need = topk - refs["state"][_UPPER:_UPPER + 1]
                below = ((total < need) & (cand < CANDIDATES)).astype(jnp.int32)
                digit = jnp.sum(below, axis=0, keepdims=True)
                refs["state"][_POSITION:_POSITION + 1] = refs["state"][_POSITION:_POSITION + 1] + (digit << shift)

        state = refs["state"]
        tau = jnp.where(active, state[_PREFIX:_PREFIX + 1] ^ INT_MIN, INT_MIN)
        p_tie = jnp.where(active, jnp.where(tie, state[_POSITION:_POSITION + 1], INT_MAX), -1)
        for b in range(self.batch):
            tau_b, p_tie_b = _row_scalar(tau, b), _row_scalar(p_tie, b)
            refs["tau"][b] = tau_b
            refs["p_tie"][b] = p_tie_b
            own_key = _row_scalar(refs["in_flight"][b:b + 1, :], b % self.S)
            refs["own"][b] = selected(own_key, self.position_ref[b], tau_b, p_tie_b).astype(jnp.int32)


def _step_activity(self, span_rows: int, active_ref, steps: int):
    """``active_ref[b * steps + s]`` = 1 where attention step ``s`` (key rows
    ``[s span_rows, (s + 1) span_rows)``) of row ``b``'s sequence (``b`` its
    first row) holds a selected token of any of its rows; after ``select``."""
    refs, chunk = self.refs, self.chunk
    if chunk % span_rows or self.key_rows % span_rows:
        raise ValueError("attention steps must tile the key chunks")

    def clear(i, carry):
        active_ref[i] = jnp.int32(0)
        return carry

    jax.lax.fori_loop(0, self.batch * steps, clear, 0)
    total = refs["chunk_starts"][self.batch]
    for step in range(self.key_slots - 1):
        pl_when(step < total)(lambda step=step: self._chunk_copy(step))
    sub = jax.lax.broadcasted_iota(jnp.int32, (span_rows, LANES), 0)
    lane = jax.lax.broadcasted_iota(jnp.int32, (span_rows, LANES), 1)

    def step_fn(step, carry):
        row, local = _locate(refs["chunk_starts"], step, self.batch)
        self._wait_chunk(step)
        pl_when(step + self.key_slots - 1 < total)(lambda: self._chunk_copy(step + self.key_slots - 1))
        start = jnp.minimum(local * chunk, self.key_rows - chunk)
        rows_valid = refs["counts"][row] * self.subpages
        tau, p_tie = refs["tau"][row], refs["p_tie"][row]
        target = row - row % self.S if self.S > 1 else row
        for v in range(chunk // span_rows):
            key = refs["key_buf"][step % self.key_slots, pl_ds(v * span_rows, span_rows), :]
            key_row = start + v * span_rows + sub
            valid = (key_row >= local * chunk) & (key_row < rows_valid)
            page = key_row // self.subpages
            position = ((page * self.group + self.owner) * self.T
                        + (key_row % self.subpages) * INDEX_SUBPAGE_TOKENS + lane)
            hit = jnp.sum((valid & selected(key, position, tau, p_tie)).astype(jnp.int32))
            index = target * steps + (start + v * span_rows) // span_rows

            @pl_when(hit > 0)
            def mark(index=index):
                active_ref[index] = jnp.int32(1)
        return carry

    jax.lax.fori_loop(0, total, step_fn, 0)


IndexSelect.step_activity = _step_activity


def pl_when(condition):
    from jax.experimental import pallas as pl

    if condition is True:
        return lambda f: f()
    return pl.when(condition)


def pl_ds(start, size):
    from jax.experimental import pallas as pl

    return pl.ds(start, size)


def select_scratch(batch, pool_block_shape, key_rows, group=32, sequence_rows=1):
    """Scratch of an ``IndexSelect`` (names -> shapes), in ``refs`` order."""
    from jax.experimental.pallas import tpu as pltpu

    vmem = pltpu.VMEM
    subpages = pool_block_shape[0] // INDEX_SUBPAGE_ROWS
    page_slots, key_slots = ring_slots(batch)
    return {
        "page_buf": vmem((page_slots, *pool_block_shape), jnp.uint8),
        "page_sems": pltpu.SemaphoreType.DMA((page_slots,)),
        "stage": vmem((2, sequence_rows, subpages, LANES), jnp.int32),
        "stage_sems": pltpu.SemaphoreType.DMA((2,)),
        "key_buf": vmem((key_slots, min(CHUNK_ROWS, key_rows), LANES), jnp.int32),
        "key_sems": pltpu.SemaphoreType.DMA((key_slots,)),
        "exchange": vmem((2, group, COUNT_ROWS, LANES), jnp.int32),
        "exchange_sems": pltpu.SemaphoreType.DMA((2, 2)),
        "local": vmem((COUNT_ROWS, LANES), jnp.int32),
        "state": vmem((8, LANES), jnp.int32),
        "in_flight": vmem((batch, LANES), jnp.int32),
        "counts": pltpu.SMEM((batch,), jnp.int32),
        "page_starts": pltpu.SMEM((batch + 1,), jnp.int32),
        "chunk_starts": pltpu.SMEM((batch + 1,), jnp.int32),
    }


def dsa_select(q8, w, pool, tables, positions, new_k8, new_exp, *, topk: int = 2048, group: int = 32,
               max_local_pages: int, sequence_rows: int = 1, owner_table=None, interpret=None,
               axis: str = "tp"):
    """``IndexSelect`` as its own kernel (per rank, inside ``shard_map`` over
    ``axis``): ``(tau i32 [B], p_tie i32 [B], own i32 [B], keys i32 [B,
    key_rows_for(max_local_pages, T), 128])``.

    ``q8`` ``[B, 32, 128]`` fp8-valued index queries, ``w`` f32 ``[B, 32]``
    head weights (incl. the queries' scales, 32^-0.5 and 128^-0.5), ``pool``
    this layer's index-key pool ``[N, R, 4, 128]`` uint8, ``tables`` i32
    ``[B, P]`` local block tables, ``positions`` i32 ``[B]`` the rows' token
    positions, ``new_k8`` ``[B, 128]`` / ``new_exp`` i32 ``[B]`` the rows'
    own tokens' key records (fp8 values, UE8M0 bytes)."""
    batch = q8.shape[0]
    w_t = jnp.zeros((w.shape[1], LANES), jnp.float32).at[:, :batch].set(w.astype(jnp.float32).T)
    return select_call(q8.astype(jnp.float8_e4m3fn), w_t, new_k8.astype(jnp.float32), pool, tables.reshape(-1),
                       positions, new_exp, pages=tables.shape[1], topk=topk, group=group,
                       max_local_pages=max_local_pages, sequence_rows=sequence_rows, owner_table=owner_table,
                       interpret=interpret, axis=axis)


def select_call(q8, w_t, new_k8, pool, tables, positions, new_exp, *, pages: int, topk: int, group: int,
                max_local_pages: int, sequence_rows: int = 1, owner_table=None, interpret=None, axis: str = "tp"):
    """``dsa_select``'s kernel on prepared inputs: ``q8`` fp8, ``w_t`` f32
    ``[32, 128]`` (column ``b``: row ``b``'s head weights), ``new_k8`` f32,
    ``tables`` flat ``[B * pages]``."""
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    from sglang_tpu.k3_programs.kimi_k3.kernels import mla_dcp

    batch = q8.shape[0]
    block_shape = pool.shape[1:]
    subpages = block_shape[0] // INDEX_SUBPAGE_ROWS
    page_tokens = subpages * INDEX_SUBPAGE_TOKENS
    key_rows = key_rows_for(max_local_pages, page_tokens)
    scratch = select_scratch(batch, block_shape, key_rows, group, sequence_rows)
    names = list(scratch)

    def kernel(positions_ref, exp_ref, tables_ref, q8_ref, w_t_ref, new_k8_ref, pool_ref,
               tau_ref, ptie_ref, own_ref, keys_ref, *scratch_refs):
        rank = jax.lax.axis_index(axis)
        member = rank % group
        owner = member if owner_table is None else mla_dcp.owner_index(owner_table) % group
        refs = dict(zip(names, scratch_refs), keys=keys_ref, tau=tau_ref, p_tie=ptie_ref, own=own_ref)
        select = IndexSelect(
            pools=(pool_ref,), layer=0, table_ref=tables_ref, pages=pages, page_tokens=page_tokens,
            group=group, owner=owner, member=member, base=rank - member, position_ref=positions_ref,
            new_exp_ref=exp_ref, batch=batch, topk=topk, refs=refs, sequence_rows=sequence_rows)
        select.plan()
        select.in_flight(lambda row: q8_ref[row], w_t_ref, new_k8_ref)
        select.score(lambda row: q8_ref[row], w_t_ref,
                     lambda first: q8_ref[pl.ds(first, sequence_rows)].reshape(sequence_rows * 32, 128))
        select.select()

    vmem = pltpu.VMEM if hasattr(pltpu, "VMEM") else pltpu.MemorySpace.VMEM
    return pl.pallas_call(
        kernel,
        out_shape=(
            jax.ShapeDtypeStruct((batch,), jnp.int32),
            jax.ShapeDtypeStruct((batch,), jnp.int32),
            jax.ShapeDtypeStruct((batch,), jnp.int32),
            jax.ShapeDtypeStruct((batch, key_rows, LANES), jnp.int32),
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=3,
            in_specs=[
                pl.BlockSpec(memory_space=vmem),
                pl.BlockSpec(memory_space=vmem),
                pl.BlockSpec(memory_space=vmem),
                pl.BlockSpec(memory_space=pl.ANY),
            ],
            out_specs=[
                pl.BlockSpec(memory_space=pltpu.SMEM),
                pl.BlockSpec(memory_space=pltpu.SMEM),
                pl.BlockSpec(memory_space=pltpu.SMEM),
                pl.BlockSpec(memory_space=pl.ANY),
            ],
            scratch_shapes=[scratch[name] for name in names],
        ),
        compiler_params=pltpu.CompilerParams(has_side_effects=True),
        interpret=interpret,
    )(positions, new_exp, tables, q8, w_t, new_k8, pool)
