# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3 MLA (absorbed, NoPE, output-gated) decode step as an in-kernel stage.

Reads and writes sglang's paged latent cache in place
(TpuFusedMLATokenToKVPool: ``[pages, 64, 2, 640]`` bf16, two tokens per
32-bit word; row = kv_c 512 | k_pe 64 | zeros 64). Rounding follows sglang's
TPU path (forward_mla absorbed + mla_ragged_paged_attention_v2 decode):
bf16 stage outputs, scores ``bf16((q_nope.kv_c + q_pe.k_pe) * 192**-0.5)``,
f32 online softmax over 2048-token blocks with ``bf16(p) @ kv_c``. The o_proj
partials are all-reduced in f32 and rounded once.
"""

import jax
import jax.numpy as jnp
import msgspec
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sglang_tpu.k3_megakernel.config import (
    K3MK_COLLECTIVE_ID,
    Relax,
    Topology,
)
from sglang_tpu.k3_megakernel.kernel.collectives import entry_barrier
from sglang_tpu.k3_megakernel.kernel.stages.attn_res import rmsnorm
from sglang_tpu.k3_megakernel.kernel.stages.common import (
    DN,
    OutProjSpec,
    bf16,
    bind_stream,
    flat,
    phase,
    stream_scratch,
    tree_when,
)

PAGE = 128  # tokens per page (64 rows x 2 packed tokens)
BLOCK_PAGES = 16  # 2048-token softmax blocks, as sglang's tuned decode table
LATENT = 512
ROW = 640
SM_SCALE = 192**-0.5
_NT = (((1,), (1,)), ((), ()))  # A @ B.T
_MASK = float(jnp.finfo(jnp.bfloat16).min)


REPLICATED = 2176  # w1 columns every rank needs: q_a 1536 | kv_c 512 | k_pe 64 | 0 64


class W1Split(msgspec.Struct, frozen=True):
    """The replicated w1 columns split over the lanes of a host: lane ``j`` projects the
    128-column blocks ``[start(j), start(j) + per)`` and the host all-gathers them (one in-host
    round trip, hidden behind the g_proj stream) instead of every rank reading all 2176 columns.
    Each column's dot and K-chunking are unchanged, so the values are bit-identical."""

    lanes: int

    @property
    def blocks(self) -> int:
        return REPLICATED // 128

    @property
    def per(self) -> int:
        return -(-self.blocks // self.lanes)

    @property
    def width(self) -> int:
        return self.per * 128

    def start_block(self, lane):
        """First block of ``lane`` (traced or static); the last lanes overlap so all widths match."""
        return jnp.minimum(self.per * lane, self.blocks - self.per)

    def owners(self) -> list[tuple[int, int]]:
        """(lane, row in that lane's gathered block) of every replicated 128-column block."""
        out = []
        for b in range(self.blocks):
            for j in range(self.lanes):
                s = min(self.per * j, self.blocks - self.per)
                if s <= b < s + self.per:
                    out.append((j, b - s))
                    break
        return out


class SplitW1Projection:
    """``x @ w1`` (all ``2176 + hd`` columns) from this lane's replicated slice ``rep``, the
    in-host gather of the other lanes' slices and this rank's g_proj columns ``gate``."""

    def __init__(self, split: W1Split, rep, gate, gather):
        self.split, self.rep, self.gate, self.gather = split, rep, gate, gather

    def matmul(self, x, *, prefetched) -> jax.Array:
        self.gather.start(flat(self.rep.matmul(x, prefetched=prefetched)))
        g = self.gate.matmul(x, prefetched=prefetched)
        got = self.gather.finish()[
            0
        ]  # [lanes, flat rows, 128]; row r's blocks at r * per + c
        per = self.split.per
        rows = [
            jnp.concatenate(
                [
                    got[j, r * per + c : r * per + c + 1, :]
                    for j, c in self.split.owners()
                ],
                axis=1,
            )
            for r in range(x.shape[0])
        ]
        return jnp.concatenate([jnp.concatenate(rows, axis=0), g], axis=1)


class MlaStage:
    """MLA layer for one decode row on ``topo`` (heads = 96 / world per rank)."""

    def __init__(
        self,
        topo: Topology,
        *,
        hidden: int = 7168,
        heads: int = 96,
        eps: float = 1e-5,
        relax: Relax = Relax(),
        collective_id: int = K3MK_COLLECTIVE_ID,
        repeat: int = 1,
        early_kv: bool = False,
        async_write: bool = False,
        kv_window: bool = False,
        block_pages: int = BLOCK_PAGES,
        rows: int = 1,
    ):
        """``early_kv`` starts the first KV block's page DMAs before the projections;
        ``async_write`` leaves the new token's cache write in flight (``wait_new`` drains it).
        ``kv_window``: the KV buffer is zeroed once per call (``clear_kv``) and only the last
        fetched page is masked / patched with the new token, in place; the cache write then
        DMAs the patched word row straight from the buffer (see ``attend``)."""
        self.early_kv, self.async_write, self.kv_window = (
            early_kv,
            async_write,
            kv_window,
        )
        # Pages per softmax block (sglang: 16); fewer pages shrink the KV double buffer and change
        # only the online-softmax blocking (f32 reordering).
        assert 16 % block_pages == 0, block_pages
        self.bp = block_pages
        self.skip_core = (
            False  # benchmark-only: no attention over the cache, no cache write
        )
        self.topo, self.hidden, self.eps, self.collective_id, self.repeat = (
            topo,
            hidden,
            eps,
            collective_id,
            repeat,
        )
        self.relax = relax
        self.rows = rows
        self.heads = heads // topo.world
        self.hd = self.heads * 128
        self.out = OutProjSpec(topo, self.hd, hidden, relax, rows=rows)
        self.streams = (
            (hidden, 2176 + self.hd),
            (1536, 2 * self.hd),
            self.out.weight_shape,
        )

    def scratch(self):
        shapes = []
        for k, n in self.streams:
            shapes += stream_scratch(k, n, self.relax.dense_dtype)
        shapes += [
            pltpu.VMEM(
                (2, self.bp * PAGE // 2, 2, ROW), jnp.bfloat16
            ),  # KV block double buffer
            pltpu.SemaphoreType.DMA((2, self.bp)),
            pltpu.VMEM(
                (self.word_rows, 2, ROW), jnp.bfloat16
            ),  # word rows holding the new token(s)
            pltpu.SemaphoreType.DMA((1,)),
            pltpu.VMEM(
                (self.heads * self.rows, LATENT), jnp.float32
            ),  # o accumulator, head-major rows
        ]
        return shapes + self.out.scratch_shapes()

    # ----- paged KV -----
    # ``caches`` is either a list of per-layer pools ``[pages, 64, 2, 640]`` (sglang allocates
    # one tensor per layer; every DMA is emitted once per candidate pool under ``pl.when``)
    # or a one-element list holding a stacked ``[n_mla, pages, 64, 2, 640]`` pool; ``which``
    # is the traced index of this layer's pool.
    @staticmethod
    def _pools(caches, which):
        """[(predicate or None, page-indexable ref [pages, 64, 2, 640])]."""
        if len(caches) == 1 and len(caches[0].shape) == 5:
            return [(None, caches[0].at[which])]
        return [(which == j, cache) for j, cache in enumerate(caches)]

    @staticmethod
    def _each_pool(caches, which, active, fn):
        """``fn(page-indexable pool)`` for this layer's pool under ``active`` (per-layer pools: a
        binary ``pl.when`` tree over the candidates instead of one test per pool)."""

        def run():
            if len(caches) == 1 and len(caches[0].shape) == 5:
                fn(caches[0].at[which])
            else:
                tree_when(which, len(caches), lambda j: fn(caches[j]))

        if active is True:
            run()
        else:
            pl.when(active)(run)

    @staticmethod
    def _any_pool(caches):
        """A pool of the right shape for DMA waits (a wait only needs the semaphore and byte count)."""
        if len(caches) == 1 and len(caches[0].shape) == 5:
            return caches[0].at[0]
        return caches[0]

    @staticmethod
    def _when(pred, active, fn):
        cond = active if pred is None else (pred if active is True else active & pred)
        if cond is True:
            fn()
        else:
            pl.when(cond)(fn)

    def _page_copy(self, cache, kv_buf, sems, pages_ref, block, i):
        half = PAGE // 2
        slot = block % 2
        return pltpu.make_async_copy(
            cache.at[pages_ref[block * self.bp + i]],
            kv_buf.at[slot, pl.ds(i * half, half)],
            sems.at[slot, i],
        )

    def _page_dma(self, caches, which, kv_buf, sems, pages_ref, block, i, active, op):
        if op == "wait":  # any pool's descriptor has this page's byte count
            cp = self._page_copy(
                self._any_pool(caches), kv_buf, sems, pages_ref, block, i
            )
            self._when(None, active, cp.wait)
            return
        self._each_pool(
            caches,
            which,
            active,
            lambda cache: self._page_copy(
                cache, kv_buf, sems, pages_ref, block, i
            ).start(),
        )

    def _start_block(self, caches, which, kv_buf, sems, pages_ref, block, n_pages):
        def starts(cache):  # one pool selection for the whole block
            for i in range(self.bp):
                cp = self._page_copy(cache, kv_buf, sems, pages_ref, block, i)
                self._when(None, block * self.bp + i < n_pages, cp.start)

        self._each_pool(caches, which, True, starts)

    def _wait_block(self, caches, which, kv_buf, sems, pages_ref, block, n_pages):
        for i in range(self.bp):
            self._page_dma(
                caches,
                which,
                kv_buf,
                sems,
                pages_ref,
                block,
                i,
                block * self.bp + i < n_pages,
                "wait",
            )

    @staticmethod
    def _tokens(kv_buf, slot):
        """[2048, 640] bf16 tokens of the block in ``slot`` (sglang's load_bkv bitcast)."""
        words = kv_buf.bitcast(jnp.uint32).at[slot]
        words = words.reshape(words.shape[0] * words.shape[1], ROW)
        return pltpu.bitcast(words[...], jnp.bfloat16)

    @staticmethod
    def _words(kv_buf, slot):
        """[1024, 640] u32 word rows of ``slot`` (two tokens per word)."""
        words = kv_buf.bitcast(jnp.uint32).at[slot]
        return words.reshape(words.shape[0] * words.shape[1], ROW)

    def clear_kv(self, kv_buf):
        """Zeros the KV buffer (rows a request never fetches must not hold NaN bit patterns)."""
        for slot in range(kv_buf.shape[0]):
            w = self._words(kv_buf, slot)
            w[...] = jnp.zeros(w.shape, jnp.uint32)

    def _patch_last_page(self, kv_buf, slot, b, new_row, kv_len):
        """In block ``b``: zero the rows at or past kv_len of the last fetched page and write the
        new token (index kv_len - 1, always on that page) into it."""
        last = (kv_len + PAGE - 1) // PAGE - 1 - b * self.bp

        @pl.when((last >= 0) & (last < self.bp))
        def _():
            half = PAGE // 2
            win = self._words(kv_buf, slot).at[
                pl.ds(pl.multiple_of(last * half, half), half)
            ]
            tok = pltpu.bitcast(win[...], jnp.bfloat16)  # [128, 640]
            pos = (b * self.bp + last) * PAGE + jax.lax.broadcasted_iota(
                jnp.int32, (PAGE, 1), 0
            )
            tok = jnp.where(
                pos == kv_len - 1,
                new_row,
                jnp.where(pos < kv_len, tok, jnp.zeros_like(tok)),
            )
            win[...] = pltpu.bitcast(tok, jnp.uint32)

    def start_kv(self, caches, which, pages_ref, kv_len, kv_buf, sems):
        """Starts the first KV block's page DMAs (``attend`` with ``started=True`` consumes them)."""
        self._start_block(
            caches, which, kv_buf, sems, pages_ref, 0, (kv_len + PAGE - 1) // PAGE
        )

    def attend(
        self,
        ql,
        qp,
        new_row,
        caches,
        which,
        pages_ref,
        kv_len,
        kv_buf,
        sems,
        acc_ref,
        started=False,
    ):
        """``ql`` [h, 512], ``qp`` [h, 128] bf16; the new token (index kv_len-1) is patched
        into its block before use. Returns bf16-valued o_lat [h, 512]."""
        n_tok = self.bp * PAGE
        n_pages = (kv_len + PAGE - 1) // PAGE
        n_blocks = (kv_len + n_tok - 1) // n_tok
        if not started:
            self._start_block(caches, which, kv_buf, sems, pages_ref, 0, n_pages)
        acc_ref[...] = jnp.zeros(acc_ref.shape, jnp.float32)

        def block_body(b, carry):
            m, l = carry

            @pl.when(b + 1 < n_blocks)
            def _():
                self._start_block(
                    caches, which, kv_buf, sems, pages_ref, b + 1, n_pages
                )

            self._wait_block(caches, which, kv_buf, sems, pages_ref, b, n_pages)
            if self.kv_window:
                self._patch_last_page(kv_buf, b % 2, b, new_row, kv_len)
                tok = self._tokens(kv_buf, b % 2)
            else:
                tok = self._tokens(kv_buf, b % 2)
                pos = b * n_tok + jax.lax.broadcasted_iota(jnp.int32, (n_tok, 1), 0)
                # Rows at or past kv_len were never DMA'd into this block (stale VMEM); a NaN there
                # would survive p = 0 in the p @ kv product, so they are zeroed.
                tok = jnp.where(
                    pos == kv_len - 1,
                    new_row,
                    jnp.where(pos < kv_len, tok, jnp.zeros_like(tok)),
                )
            s = jax.lax.dot_general(
                ql, tok[:, :LATENT], _NT, preferred_element_type=jnp.float32
            )
            s = s + jax.lax.dot_general(
                qp, tok[:, LATENT:], _NT, preferred_element_type=jnp.float32
            )
            s = bf16(s * SM_SCALE)
            col = b * n_tok + jax.lax.broadcasted_iota(jnp.int32, s.shape, 1)
            s = jnp.where(col < kv_len, s, _MASK)
            m_new = jnp.maximum(m, jnp.max(s, axis=1, keepdims=True))
            p = jnp.exp(s - m_new)
            alpha = jnp.exp(m - m_new)
            l_new = alpha * l + jnp.sum(p, axis=1, keepdims=True)
            pv = jax.lax.dot_general(
                p.astype(jnp.bfloat16),
                tok[:, :LATENT],
                DN,
                preferred_element_type=jnp.float32,
            )
            acc_ref[...] = alpha * acc_ref[...] + pv
            return m_new, l_new

        h = ql.shape[0]
        m0 = jnp.full((h, 1), -jnp.inf, jnp.float32)
        l0 = jnp.zeros((h, 1), jnp.float32)
        _, l = jax.lax.fori_loop(0, n_blocks, block_body, (m0, l0))
        return bf16(acc_ref[...] / l)

    def write_new(
        self, new_row, caches, which, pages_ref, kv_len, kv_buf, word_ref, wsem
    ):
        """Store the new token's row at index kv_len-1 (read-modify-write of its word row)."""
        t = kv_len - 1
        n_tok = self.bp * PAGE
        block = t // n_tok
        if self.kv_window:
            # attend patched the word row in the buffer (its other token is the cached one, or 0 past kv_len).
            src = kv_buf.at[block % 2, pl.ds((t - block * n_tok) // 2, 1)]
            self._store_word(src, caches, which, pages_ref, t, wsem)
            return
        tok = self._tokens(kv_buf, block % 2)
        local = t - block * n_tok
        pair = local // 2 * 2
        idx = jax.lax.broadcasted_iota(jnp.int32, (n_tok, 1), 0)
        even = jnp.sum(
            jnp.where(idx == pair, tok.astype(jnp.float32), 0.0), axis=0, keepdims=True
        )
        odd = jnp.sum(
            jnp.where(idx == pair + 1, tok.astype(jnp.float32), 0.0),
            axis=0,
            keepdims=True,
        )
        new = new_row.astype(jnp.float32)
        even = jnp.where(t % 2 == 0, new, even)
        odd = jnp.where(t % 2 == 1, new, odd)
        word_ref[0] = jnp.concatenate([even, odd], axis=0).astype(jnp.bfloat16)
        self._store_word(word_ref, caches, which, pages_ref, t, wsem)

    def _store_word(self, src, caches, which, pages_ref, t, wsem):
        """DMA ``src`` [1, 2, 640] (the word row holding token ``t``) into the cache."""
        page = pages_ref[t // PAGE]
        row = (t % PAGE) // 2
        self._each_pool(
            caches,
            which,
            True,
            lambda cache: pltpu.make_async_copy(
                src, cache.at[page, pl.ds(row, 1)], wsem.at[0]
            ).start(),
        )
        if not self.async_write:
            pltpu.make_async_copy(
                src, self._any_pool(caches).at[page, pl.ds(row, 1)], wsem.at[0]
            ).wait()

    def wait_new(self, caches, word_ref, wsem):
        """Drains an ``async_write`` cache write (any pool's descriptor has the same byte count)."""
        _, cache = self._pools(caches, 0)[0]
        pltpu.make_async_copy(word_ref, cache.at[0, pl.ds(0, 1)], wsem.at[0]).wait()

    # ----- speculative verify: R rows at positions kv0 - 1 .. kv0 + R - 2 -----
    def attend_rows(
        self, ql, qp, new_rows, caches, which, pages_ref, kv0, kv_buf, sems, acc_ref
    ):
        """``attend`` for R query rows (``ql`` [h * R, 512] / ``qp`` [h * R, 128] head-major); row r sees
        the cache up to its own token at ``kv0 - 1 + r``. Per row the arithmetic is ``attend``'s at
        kv_len ``kv0 + r`` (masked columns and blocks add exact zeros)."""
        rows = new_rows.shape[0]
        kv_end = kv0 + rows - 1
        n_tok = self.bp * PAGE
        n_pages = (kv_end + PAGE - 1) // PAGE
        n_blocks = (kv_end + n_tok - 1) // n_tok
        self._start_block(caches, which, kv_buf, sems, pages_ref, 0, n_pages)
        acc_ref[...] = jnp.zeros(acc_ref.shape, jnp.float32)
        lim = kv0 + jax.lax.broadcasted_iota(jnp.int32, (ql.shape[0], 1), 0) % rows

        def block_body(b, carry):
            m, l = carry

            @pl.when(b + 1 < n_blocks)
            def _():
                self._start_block(
                    caches, which, kv_buf, sems, pages_ref, b + 1, n_pages
                )

            self._wait_block(caches, which, kv_buf, sems, pages_ref, b, n_pages)
            tok = self._tokens(kv_buf, b % 2)
            pos = b * n_tok + jax.lax.broadcasted_iota(jnp.int32, (n_tok, 1), 0)
            tok = jnp.where(pos < kv0 - 1, tok, jnp.zeros_like(tok))
            for r in range(rows):
                tok = jnp.where(pos == kv0 - 1 + r, new_rows[r : r + 1], tok)
            s = jax.lax.dot_general(
                ql, tok[:, :LATENT], _NT, preferred_element_type=jnp.float32
            )
            s = s + jax.lax.dot_general(
                qp, tok[:, LATENT:], _NT, preferred_element_type=jnp.float32
            )
            s = bf16(s * SM_SCALE)
            col = b * n_tok + jax.lax.broadcasted_iota(jnp.int32, s.shape, 1)
            s = jnp.where(col < lim, s, _MASK)
            m_new = jnp.maximum(m, jnp.max(s, axis=1, keepdims=True))
            p = jnp.exp(s - m_new)
            alpha = jnp.exp(m - m_new)
            l_new = alpha * l + jnp.sum(p, axis=1, keepdims=True)
            pv = jax.lax.dot_general(
                p.astype(jnp.bfloat16),
                tok[:, :LATENT],
                DN,
                preferred_element_type=jnp.float32,
            )
            acc_ref[...] = alpha * acc_ref[...] + pv
            return m_new, l_new

        n_q = ql.shape[0]
        m0 = jnp.full((n_q, 1), -jnp.inf, jnp.float32)
        l0 = jnp.zeros((n_q, 1), jnp.float32)
        _, l = jax.lax.fori_loop(0, n_blocks, block_body, (m0, l0))
        return bf16(acc_ref[...] / l)

    @property
    def word_rows(self) -> int:
        """Word rows (two tokens each) a call writes: R tokens from any parity touch R / 2 + 1."""
        return 1 if self.rows == 1 else self.rows // 2 + 1

    def _word_dmas(
        self, caches, which, pages_ref, t, src, wsem, *, active, op, to_cache=True
    ):
        """DMA of the word row holding token ``t`` and ``src`` (``[1, 2, 640]``), under ``active``."""
        page = pages_ref[t // PAGE]
        row = (t % PAGE) // 2

        def copy(cache):
            hbm = cache.at[page, pl.ds(row, 1)]
            return pltpu.make_async_copy(
                *((src, hbm) if to_cache else (hbm, src)), wsem.at[0]
            )

        if op == "wait":
            self._when(None, active, copy(self._any_pool(caches)).wait)
        else:
            self._each_pool(caches, which, active, lambda cache: copy(cache).start())

    def write_rows(self, new_rows, caches, which, pages_ref, kv0, word_ref, wsem):
        """Stores the R new tokens at ``kv0 - 1 ..``: the R / 2 (+ 1) word rows are built in VMEM and written
        with concurrent DMAs. Only a word shared with an older token (odd start) is read first; the
        token slot after the last new one is written as 0 (past kv length, rewritten by the next call)."""
        rows = new_rows.shape[0]
        t0 = kv0 - 1
        odd = t0 % 2 == 1
        first = word_ref.at[pl.ds(0, 1)]
        for op in ("start", "wait"):
            self._word_dmas(
                caches,
                which,
                pages_ref,
                t0 - 1,
                first,
                wsem,
                active=odd,
                op=op,
                to_cache=False,
            )
        prev = jnp.where(
            odd, word_ref[0].astype(jnp.float32)[0:1], 0.0
        )  # token t0 - 1 (even) when odd
        zero = jnp.zeros((1, ROW), jnp.float32)
        ext = jnp.concatenate(
            [prev, new_rows.astype(jnp.float32), zero], axis=0
        )  # ext[j + 1] = token t0 + j
        for i in range(self.word_rows):
            even = jnp.where(odd, ext[2 * i : 2 * i + 1], ext[2 * i + 1 : 2 * i + 2])
            odd_tok = jnp.where(
                odd,
                ext[2 * i + 1 : 2 * i + 2],
                ext[2 * i + 2 : 2 * i + 3] if 2 * i + 2 < rows + 2 else zero,
            )
            word_ref[i] = jnp.concatenate([even, odd_tok], axis=0).astype(jnp.bfloat16)
        w0 = t0 - t0 % 2  # first token of the first word
        for op in ("start", "wait"):
            for i in range(self.word_rows):
                active = True if i < rows // 2 else odd
                self._word_dmas(
                    caches,
                    which,
                    pages_ref,
                    w0 + 2 * i,
                    word_ref.at[pl.ds(i, 1)],
                    wsem,
                    active=active,
                    op=op,
                )

    def body_rows(
        self,
        x,
        w1,
        qnorm,
        kvnorm,
        qb,
        wkc_ref,
        wvc_ref,
        oproj,
        caches,
        which,
        pages_ref,
        kv0,
        kv_buf,
        sems,
        word_ref,
        wsem,
        acc_ref,
        out_proj,
    ):
        """``body`` for ``x`` bf16 [R, H] at kv lengths ``kv0 .. kv0 + R - 1`` (speculative verify)."""
        hd, h = self.hd, self.heads
        rows = x.shape[0]
        y = w1.matmul(x, prefetched=False)
        q = rmsnorm(bf16(y[:, :1536]), qnorm, self.eps)
        kvc = rmsnorm(bf16(y[:, 1536:2048]), kvnorm, self.eps)
        kpe = bf16(y[:, 2048:2176])
        gate = bf16(y[:, 2176 : 2176 + hd])
        new_rows = jnp.concatenate([kvc, kpe], axis=1).astype(jnp.bfloat16)  # [R, 640]
        qq = bf16(qb.matmul(q.astype(jnp.bfloat16), prefetched=False))
        qn = qq[:, :hd].astype(jnp.bfloat16)
        qp = qq[:, hd:].astype(jnp.bfloat16)
        ql = jnp.concatenate(
            [
                jax.lax.dot_general(
                    qn[:, i * 128 : (i + 1) * 128],
                    wkc_ref[i],
                    DN,
                    preferred_element_type=jnp.float32,
                )
                for i in range(h)
            ],
            axis=0,
        ).astype(jnp.bfloat16)  # [h * R, 512], head-major
        qp = jnp.concatenate([qp[:, i * 128 : (i + 1) * 128] for i in range(h)], axis=0)
        if self.skip_core:  # benchmark-only
            o_lat = jnp.zeros((h * rows, LATENT), jnp.float32)
        else:
            with phase("mla_attend"):
                o_lat = self.attend_rows(
                    ql,
                    qp,
                    new_rows,
                    caches,
                    which,
                    pages_ref,
                    kv0,
                    kv_buf,
                    sems,
                    acc_ref,
                )
                self.write_rows(new_rows, caches, which, pages_ref, kv0, word_ref, wsem)
        o_lat = o_lat.astype(jnp.bfloat16)
        attn = jnp.concatenate(
            [
                jax.lax.dot_general(
                    o_lat[i * rows : (i + 1) * rows],
                    wvc_ref[i],
                    DN,
                    preferred_element_type=jnp.float32,
                )
                for i in range(h)
            ],
            axis=1,
        )
        gated = bf16(bf16(attn) * bf16(jax.nn.sigmoid(gate)))
        with phase("attn_ar"):
            return out_proj(gated.astype(jnp.bfloat16), oproj)

    def body(
        self,
        x,
        w1,
        qnorm,
        kvnorm,
        qb,
        wkc_ref,
        wvc_ref,
        oproj,
        caches,
        which,
        pages_ref,
        kv_len,
        kv_buf,
        sems,
        word_ref,
        wsem,
        acc_ref,
        out_proj,
    ):
        hd, h = self.hd, self.heads
        if self.early_kv:
            self.start_kv(caches, which, pages_ref, kv_len, kv_buf, sems)
        y = w1.matmul(x, prefetched=False)
        q = rmsnorm(bf16(y[:, :1536]), qnorm, self.eps)
        kvc = rmsnorm(bf16(y[:, 1536:2048]), kvnorm, self.eps)
        kpe = bf16(y[:, 2048:2176])  # 64 k_pe + 64 zero columns
        gate = bf16(y[:, 2176 : 2176 + hd])
        new_row = jnp.concatenate([kvc, kpe], axis=1).astype(jnp.bfloat16)  # [1, 640]
        qq = bf16(qb.matmul(q.astype(jnp.bfloat16), prefetched=False))
        qn, qp = qq[:, :hd], qq[:, hd:]
        qn = qn.reshape(h, 128).astype(jnp.bfloat16)
        ql = jnp.concatenate(
            [
                jax.lax.dot_general(
                    qn[i : i + 1], wkc_ref[i], DN, preferred_element_type=jnp.float32
                )
                for i in range(h)
            ],
            axis=0,
        ).astype(jnp.bfloat16)
        if self.skip_core:
            if self.early_kv:
                self._wait_block(
                    caches,
                    which,
                    kv_buf,
                    sems,
                    pages_ref,
                    0,
                    (kv_len + PAGE - 1) // PAGE,
                )
            o_lat = jnp.zeros((h, LATENT), jnp.float32)
        else:
            with phase("mla_attend"):
                o_lat = self.attend(
                    ql,
                    qp.reshape(h, 128).astype(jnp.bfloat16),
                    new_row,
                    caches,
                    which,
                    pages_ref,
                    kv_len,
                    kv_buf,
                    sems,
                    acc_ref,
                    started=self.early_kv,
                )
                self.write_new(
                    new_row, caches, which, pages_ref, kv_len, kv_buf, word_ref, wsem
                )
        o_lat = o_lat.astype(jnp.bfloat16)
        attn = jnp.concatenate(
            [
                jax.lax.dot_general(
                    o_lat[i : i + 1], wvc_ref[i], DN, preferred_element_type=jnp.float32
                )
                for i in range(h)
            ],
            axis=1,
        )
        gated = bf16(bf16(attn) * bf16(jax.nn.sigmoid(gate)))
        with phase("attn_ar"):  # o_proj stream + its reduction
            return out_proj(gated.astype(jnp.bfloat16), oproj)

    def kernel(self, *refs):
        """Inputs: meta, pages, x, q_norm, kv_norm, w_kc, w_vc, w1, q_b, o_proj[, w1_s, q_b_s, o_proj_s],
        cache; outputs: out, cache (aliased); then scratch."""
        fp8 = self.relax.dense_fp8
        n_in = 14 if fp8 else 11
        (
            meta_ref,
            pages_ref,
            x_ref,
            qnorm_ref,
            kvnorm_ref,
            wkc_ref,
            wvc_ref,
            w1_hbm,
            qb_hbm,
            oproj_hbm,
        ) = refs[:10]
        scales = refs[10:13] if fp8 else (None,) * 3
        out_ref, cache_hbm = refs[
            n_in : n_in + 2
        ]  # refs[n_in - 1] is the aliased cache input
        scratch = refs[n_in + 2 :]
        me = jax.lax.axis_index("x")
        per = len(stream_scratch(1, 128, self.relax.dense_dtype))
        w1 = bind_stream(w1_hbm, scratch[0:per], scales[0])
        qb = bind_stream(qb_hbm, scratch[per : 2 * per], scales[1])
        oproj = bind_stream(oproj_hbm, scratch[2 * per : 3 * per], scales[2])
        kv_buf, sems, word_ref, wsem, acc_ref = scratch[3 * per : 3 * per + 5]
        ar = self.out.bind(me, scratch[3 * per + 5 :])
        ar.init()
        if self.kv_window:
            self.clear_kv(kv_buf)
        entry_barrier(me, self.topo)

        def run():
            refs = (
                w1,
                qnorm_ref[...],
                kvnorm_ref[...],
                qb,
                wkc_ref,
                wvc_ref,
                oproj,
                [cache_hbm],
                0,
                pages_ref,
            )
            tail = (kv_buf, sems, word_ref, wsem, acc_ref, ar)
            if self.rows > 1:
                out_ref[...] = self.body_rows(
                    x_ref[...], *refs, meta_ref[0], *tail
                ).astype(jnp.bfloat16)
                return
            # Rows one by one through the B1 body at kv lengths kv0 + t: the oracle of ``body_rows``.
            for t in range(x_ref.shape[0]):
                out_ref[t : t + 1] = self.body(
                    x_ref[t : t + 1], *refs, meta_ref[0] + t, *tail
                ).astype(jnp.bfloat16)

        if self.repeat == 1:
            run()
        else:

            @pl.loop(0, self.repeat)
            def _(i):
                run()

    def build(self, mesh, cache_shape, n_rows: int = 1):
        """Per-rank inputs in ``kernel`` order: meta [1] (kv_len), pages [n], x, norms, w_kc, w_vc, w1, q_b,
        o_proj[, their column scales], cache."""
        from jax.sharding import PartitionSpec as P

        spaces = (
            ["smem", "smem"]
            + ["vmem"] * 5
            + ["hbm"] * (7 if self.relax.dense_fp8 else 4)
        )
        call = pl.pallas_call(
            self.kernel,
            out_shape=(
                jax.ShapeDtypeStruct(
                    (max(n_rows, self.rows), self.hidden), jnp.bfloat16
                ),
                jax.ShapeDtypeStruct(cache_shape, jnp.bfloat16),
            ),
            in_specs=[
                pl.BlockSpec(
                    memory_space={
                        "smem": pltpu.SMEM,
                        "vmem": pltpu.VMEM,
                        "hbm": pltpu.HBM,
                    }[s]
                )
                for s in spaces
            ],
            out_specs=(
                pl.BlockSpec(memory_space=pltpu.VMEM),
                pl.BlockSpec(memory_space=pltpu.HBM),
            ),
            scratch_shapes=self.scratch(),
            input_output_aliases={len(spaces) - 1: 1},
            compiler_params=pltpu.CompilerParams(
                collective_id=self.collective_id, vmem_limit_bytes=64 * 1024 * 1024
            ),
        )

        def local(*args):
            out, cache = call(*[a[0] for a in args])
            return out[None], cache[None]

        spec = P("x")
        return jax.jit(
            jax.shard_map(
                local,
                mesh=mesh,
                in_specs=(spec,) * len(spaces),
                out_specs=(spec, spec),
                check_vma=False,
            )
        )
