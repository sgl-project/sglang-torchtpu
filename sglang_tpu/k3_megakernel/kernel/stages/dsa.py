# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3.1 DSA ("lightning") indexer as an in-kernel megakernel stage.

One DSA layer per call site, for one decode row (``rows=1``) or R rows at the
consecutive positions ``kv0 - 1 .. kv0 + R - 2`` (prefill / verify rows mode,
each row with its own causal top-k).  The selection is not a gather: it is a
per-token mask the MK's dense MLA block loop applies (``MlaStage.attend`` /
``attend_rows`` take ``dsa_mask=``).

Native reference:
  * ``attention/pallas/dsa_index_topk.dsa_indexer_step``: quantize q / key
    (``quantize_fp8_ue8m0``), store the key record, ``streamindex_topk``.
  * ``attention/pallas/dsa_key_store``: the decode key writer (record bytes,
    UE8M0 exponent from the abs-max bits, torch_tpu-style LayerNorm).
  * ``dsv4/streamindex_topk``: scores ``relu(q8 . k8) * w`` summed over heads
    (decode: ``.sum(axis=1)`` of ``[1, 32, N]``; mixed / rows: head-sequential
    ``bf16_reduce``), times the key's UE8M0 scale, ``-inf`` past the causal bound.
  * ``dsv4/sparsecore_topk``: exact top-k over the monotone int32 keys of the
    f32 score bits (unsorted; ties at the threshold arbitrary).

Index-key cache: per DSA layer ``u8[pages, 32, 4, 256]`` (native
``index_key_cache_shape``); token t is page ``t // 128``, row ``(t % 128) // 4``,
slot ``t % 4``; a record is 128 fp8 key bytes | UE8M0 scale byte | zeros.

Selection (TensorCore, exact): the scores of a row are mapped to the monotone
keys and a digit search (``sel_bits`` = 2 bits, i.e. 3 candidate counts, per pass
over the scores; 16 passes) finds the largest key ``tau`` with
``count(key >= tau) >= topk``.  When more than ``topk`` keys are ``>= tau`` the
tied tokens with the lowest positions are kept (a second bisection over the
position, only when such a tie exists).  The mask is
``key > tau | (key == tau & pos <= p_tie)``; rows with ``lim <= topk`` keys
(``kv_len <= 2048``) select everything and skip scoring and selection.

``index_skip_topk_offset`` / ``index_topk_freq``: K3.1 has freq 1, offset 1, so
every DSA layer computes its own top-k (``model_config`` skip rule
``max(l - offset + 1, 0) % freq != 0`` is never true); ``check_config`` asserts
that.
"""

import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sglang_tpu.k3_megakernel.kernel.stages.common import bf16, tree_when

PAGE = 128
TOKENS_PER_ROW = 4
ROWS_PER_PAGE = PAGE // TOKENS_PER_ROW  # 32
RECORD = 256
HEADS = 32
HEAD_DIM = 128
TOPK = 2048
SCORE_BLOCK_PAGES = 16  # 2048 index keys per scoring block (native decode block)
_INT_MIN = -(2**31)
_MONO = 0x7FFFFFFF
_FP8_MAX_MANTISSA = 0x600000  # 448 = 1.75 * 2**8
_FP8_MAX = 448.0

# Compact index-key layout (``DsaStage(compact=True)``): u8[pages, 33, 4, 128] per layer. Rows 0..31 hold
# the 128 fp8 key bytes of 4 tokens each (token t: row (t % 128) // 4, slot t % 4); row 32, sublane 0,
# lane t % 128 holds token t's UE8M0 scale byte. Half the bytes of the native record (128 B key + 1 B
# scale per token instead of 256 B), so scoring DMAs and VMEM blocks are halved.
COMPACT_PAGE_SHAPE = (ROWS_PER_PAGE + 1, TOKENS_PER_ROW, HEAD_DIM)
NATIVE_PAGE_SHAPE = (ROWS_PER_PAGE, TOKENS_PER_ROW, RECORD)


def compact_to_native(cache):
    """numpy: compact pages [P, 33, 4, 128] -> native records [P, 32, 4, 256] (for byte tests)."""
    import numpy as np

    c = np.asarray(cache)
    out = np.zeros(c.shape[:1] + NATIVE_PAGE_SHAPE, np.uint8)
    out[..., :HEAD_DIM] = c[:, :ROWS_PER_PAGE]
    out[..., HEAD_DIM] = c[:, ROWS_PER_PAGE, 0, :].reshape(
        -1, ROWS_PER_PAGE, TOKENS_PER_ROW
    )
    return out


def native_to_compact(cache):
    import numpy as np

    c = np.asarray(cache)
    out = np.zeros(c.shape[:1] + COMPACT_PAGE_SHAPE, np.uint8)
    out[:, :ROWS_PER_PAGE] = c[..., :HEAD_DIM]
    out[:, ROWS_PER_PAGE, 0, :] = c[..., HEAD_DIM].reshape(-1, PAGE)
    return out


def check_config(text_config: dict, full_attn_layers) -> None:
    """Asserts the DSA config the stage implements: every full-attention (MLA) layer is a DSA layer
    that selects its own top-k (32 x 128 heads, no rope, LayerNorm key norm)."""
    dsa = text_config.get("dsa_config", text_config)
    assert dsa.get("index_topk", TOPK) == TOPK, dsa
    assert dsa.get("index_num_heads", HEADS) == HEADS, dsa
    assert dsa.get("index_head_dim", HEAD_DIM) == HEAD_DIM, dsa
    assert not dsa.get("index_rope_dim", 0), dsa
    assert dsa.get("index_k_norm_type", "layernorm") in ("layernorm", "layer"), dsa
    freq = dsa.get("index_topk_freq", 1) or 1
    offset = dsa.get("index_skip_topk_offset", 1)
    layers = {l - 1 for l in dsa["dsa_layers"]}  # 1-based in the config
    assert set(full_attn_layers) <= layers, (sorted(full_attn_layers), sorted(layers))
    for l in full_attn_layers:  # model_config skip rule
        assert max(l - offset + 1, 0) % freq == 0, f"layer {l} would reuse a top-k"


def monotone_key(scores):
    """f32 scores -> int32 keys with the same order (``sparsecore_topk``'s ``monotone_key``)."""
    bits = lax.bitcast_convert_type(scores, jnp.int32)
    return bits ^ ((bits >> 31) & _MONO)


def _ue8m0_exp(abs_max):
    """``_ue8m0_quantize``'s scale exponent from an f32 abs max (any shape)."""
    bits = lax.bitcast_convert_type(abs_max, jnp.int32)
    biased = bits >> 23
    exp = biased - (127 + 8) + ((bits & 0x7FFFFF) > _FP8_MAX_MANTISSA).astype(jnp.int32)
    exp = jnp.where(biased == 0, jnp.where(bits == 0, 0, -126), exp)
    return jnp.clip(exp, -126, 127)


def _ue8m0_quantize(x):
    """``quantize_fp8_ue8m0`` for f32 ``x[T, D]``: (fp8 values, exponent i32[T, 1]), as the native
    key writer (``attention/pallas/dsa_key_store``)."""
    exp = _ue8m0_exp(jnp.max(jnp.abs(x), axis=-1, keepdims=True))
    inv_scale = lax.bitcast_convert_type((127 - exp) << 23, jnp.float32)
    values = jnp.clip(x * inv_scale, -_FP8_MAX, _FP8_MAX).astype(jnp.float8_e4m3fn)
    return values, exp


def _halving_sum(v):
    """Sum over the last axis by halving, XLA's lane-reduction order."""
    n = v.shape[-1]
    while n > 1:
        n //= 2
        v = v[:, :n] + v[:, n : 2 * n]
    return v


def _pow2(exp):
    return lax.bitcast_convert_type((exp + 127) << 23, jnp.float32)


def layer_norm(x, w, b, eps):
    """torch_tpu F.layer_norm in f32: ``var = E[x^2] - mean^2`` with halving lane sums. ``x`` f32 [R, 128], ``w``/``b`` f32 [1, 128]."""
    inv_n = 1.0 / x.shape[-1]
    mean = _halving_sum(x) * inv_n
    var = _halving_sum(x * x) * inv_n - mean * mean
    return (x - mean) * lax.rsqrt(var + eps) * w + b


def key_records(key):
    """Index-key records of the normed key ``key`` (f32 [R, 128], bf16-valued): u32 [R, 256],
    one record byte per lane (128 fp8 bytes | UE8M0 byte | zeros), as ``dsa_key_store``."""
    values, exp = _ue8m0_quantize(key)
    key_bytes = lax.bitcast_convert_type(values, jnp.uint8).astype(jnp.uint32)
    lane = lax.broadcasted_iota(jnp.int32, (key.shape[0], RECORD - HEAD_DIM), 1)
    tail = jnp.where(lane == 0, (exp + 127).astype(jnp.uint32), jnp.uint32(0))
    return jnp.concatenate([key_bytes, tail], axis=-1)


def quantize_query(q, head_weights, weight_scale):
    """``q`` bf16-valued [R, 32 * 128] (wq_b output), ``head_weights`` f32 [R, 32] (the bf16 head gate).
    Returns (q8 fp8 [32 * R, 128] head-major (row ``h * R + r``), w f32 [R, 32]) with
    ``w = head_weights * q_scale * weight_scale`` as ``dsa_indexer_step``."""
    q = q.astype(jnp.float32)
    heads = [q[:, h * HEAD_DIM : (h + 1) * HEAD_DIM] for h in range(HEADS)]
    q8, _ = _ue8m0_quantize(jnp.concatenate(heads, axis=0))
    abs_max = jnp.concatenate(
        [jnp.max(jnp.abs(x), axis=-1, keepdims=True) for x in heads], axis=1
    )  # [R, 32]
    w = head_weights * _pow2(_ue8m0_exp(abs_max)) * weight_scale
    return q8, w


class DsaStage:
    """DSA indexer for ``rows`` query rows (1: decode at kv_len; R: rows ``kv0 - 1 + r``).

    Scratch (``scratch()``, then ``bind(refs)`` per layer):
      * index-key block ring u8[slots, bp, 32, 4, 256] + DMA sems [slots, bp]
      * record word rows u8[nw, 1, 4, 256] + DMA sems [2, nw] (read, write)
      * score keys i32: decode [capacity / 128, 128]; rows [R, capacity]
    """

    def __init__(
        self,
        *,
        rows: int = 1,
        capacity: int = 16384,
        topk: int = TOPK,
        block_pages: int = SCORE_BLOCK_PAGES,
        slots: int = 4,
        eps: float = 1e-5,
        softmax_scale: float = HEAD_DIM**-0.5,
        head_gate_scale: float = HEADS**-0.5,
        contiguous: bool = False,
        count_group: int | None = None,
        sel_bits: int = 2,
        compact: bool = False,
    ):
        """``slots``: index-key blocks in flight + 1 (``prefetch`` starts ``slots - 1`` of them).
        ``contiguous``: the page table maps each scoring block to consecutive physical pages
        (the standalone engine's identity table), so a block is one DMA instead of ``block_pages``.
        ``count_group``: scoring blocks per selection-pass iteration (default: 8 vregs of keys).
        ``sel_bits``: threshold bits found per pass over the scores (1, 2, 4 or 8)."""
        assert slots >= 2 and sel_bits in (1, 2, 4, 8)
        self.compact = compact
        self.page_shape = COMPACT_PAGE_SHAPE if compact else NATIVE_PAGE_SHAPE
        self.sel_bits = sel_bits
        self.R, self.topk, self.bp, self.eps, self.slots = (
            rows,
            topk,
            block_pages,
            eps,
            slots,
        )
        self.n_tok = block_pages * PAGE
        self.contiguous = contiguous
        vregs = self.n_tok // 1024 if rows == 1 else self.n_tok // 128
        self.group = count_group or max(1, 8 // vregs)
        span = self.n_tok * self.group
        self.capacity = -(-capacity // span) * span
        self.weight_scale = softmax_scale
        # torch_tpu (eager and torch.compile) multiplies a bf16 tensor by a Python float as bf16: the
        # native head gate is bf16(gate * bf16(32**-0.5)) (checked bit-exact against torch_tpu).
        self.head_gate_scale = float(
            jnp.asarray(head_gate_scale, jnp.bfloat16).astype(jnp.float32)
        )
        self.nw = 1 if rows == 1 else -(-(rows + 3) // TOKENS_PER_ROW)
        self.ns = (
            (1 if rows == 1 else 2) if compact else 0
        )  # scale rows (pages) a write touches
        self.pos_bits = max(1, (self.capacity - 1).bit_length())
        # MK call-site state (``attach`` / ``begin`` / ``ready``).
        self._mk = None
        self._pending = None
        self.last_run = None

    def scratch(self) -> list:
        keys = (self.capacity // 128, 128) if self.R == 1 else (self.R, self.capacity)
        return [
            pltpu.VMEM((self.slots, self.bp) + self.page_shape, jnp.uint8),
            pltpu.SemaphoreType.DMA((self.slots, self.bp)),
            pltpu.VMEM((self.nw + self.ns, 1) + self.page_shape[1:], jnp.uint8),
            pltpu.SemaphoreType.DMA((2, self.nw + self.ns)),
            pltpu.VMEM(keys, jnp.int32),
        ]

    def vmem_bytes(self) -> int:
        page = 1
        for d in self.page_shape:
            page *= d
        b = (self.slots * self.bp + self.nw + self.ns) * page
        return b + 4 * self.capacity * (1 if self.R == 1 else self.R)

    def bind(self, refs, caches, which=0, pages_ref=None) -> "DsaRun":
        """``caches``: this layer's index-key pool ``u8[pages, 32, 4, 256]`` (a ref), or the MK pool
        convention: a list of per-layer pools / a one-element list holding a stacked
        ``[n, pages, 32, 4, 256]`` pool, with ``which`` the (traced) layer index."""
        return DsaRun(self, refs, caches, which, pages_ref)

    # ----- MK call-site adapter (``MlaStage._dsa_select``) -----
    def attach(self, refs, *, qb_refs=None, gather=None, idx_k=2176, idx_w=2304):
        """Binds the scratch for the MK hook: ``qb_refs`` the ``stream_scratch`` refs of the
        ``mla_idx_qb`` stream, ``gather`` an in-host ``AllGather`` (row mode, ``cols=512``,
        ``by_cols`` for R > 1) of the lane's 4 index heads, or None when ``idx_qb`` holds all 32
        heads (TP1 tests)."""
        self._mk = dict(
            refs=refs, qb_refs=qb_refs, gather=gather, idx_k=idx_k, idx_w=idx_w
        )
        return self

    # The MK hook runs in three phases so the DMAs and the in-host gather overlap the MLA work:
    #   begin (before w1): index-key blocks, record-row reads and the idx_qb stream start;
    #   decode / rows (after w1): index query part + gather start, key record + write;
    #   ready (after the MLA q path, before attend): gather finish, q quantize, score + select.
    # ``early=False``: begin and ready run inside decode / rows (all after w1).
    early = True

    def begin(self, dsa, *, which, pages_ref, kv0):
        from sglang_tpu.k3_megakernel.kernel.stages.common import bind_stream

        mk = self._mk
        run = self.bind(mk["refs"], dsa.idx_caches, which, pages_ref)
        sel = kv0 + self.R - 1 > self.topk
        run.prefetch(kv0)
        run.start_row_reads(kv0 - 1)
        qb = bind_stream(
            dsa.idx_qb.at[which],
            mk["qb_refs"],
            None if dsa.idx_qb_s is None else dsa.idx_qb_s.at[which],
        )
        pl.when(sel)(lambda: qb.prefetch(qb.slots))
        self._pending = dict(
            run=run, qb=qb, sel=sel, kv0=kv0, q=None, gate=None, ready=False
        )

    def _mk_run(self, dsa, *, x, y, q_lora, which, pages_ref, kv0):
        mk = self._mk
        del x
        if self._pending is None:
            self.begin(dsa, which=which, pages_ref=pages_ref, kv0=kv0)
        p = self._pending
        run, qb, sel = p["run"], p["qb"], p["sel"]
        gather = mk["gather"]
        # The index query (wq_b stream + in-host gather) is only needed when some row selects
        # (more than topk keys); short contexts only write their keys. Same branch on every rank.
        if gather is not None:
            pl.when(sel)(
                lambda: gather.start(
                    bf16(qb.matmul(q_lora.astype(jnp.bfloat16), prefetched=False))
                )
            )
        else:
            p["q"] = lax.cond(
                sel,
                lambda _: bf16(
                    qb.matmul(q_lora.astype(jnp.bfloat16), prefetched=False)
                ),
                lambda _: jnp.zeros((q_lora.shape[0], HEADS * HEAD_DIM), jnp.float32),
                None,
            )
        run.prep_key(
            y[:, mk["idx_k"] : mk["idx_k"] + HEAD_DIM], dsa.knorm_w, dsa.knorm_b
        )
        p["gate"] = y[:, mk["idx_w"] : mk["idx_w"] + HEADS]
        run.write_keys(kv0 - 1)
        self.last_run = run
        if not self.early:
            self.ready()
        return run

    def ready(self):
        """Finishes the index query and selects (call before the attend that uses the mask)."""
        p = self._pending
        if p is None or p["ready"]:
            return
        run, sel, gather = p["run"], p["sel"], self._mk["gather"]
        q = p["q"]
        if q is None:
            rows = p["gate"].shape[0]
            q = lax.cond(
                sel,
                lambda _: gather.finish(),  # [R, 32 * 128] in lane (= head) order
                lambda _: jnp.zeros((rows, HEADS * HEAD_DIM), jnp.float32),
                None,
            )
        run.prep_q(q, p["gate"])
        run.select(p["kv0"])
        p["ready"] = True
        self._pending = None

    def decode(self, dsa, *, x, y, q_lora, which, pages_ref, kv_len):
        """Returns ``select(b, col)`` for ``MlaStage.attend`` (bool broadcastable to [h, n_tok]).
        The caller drains the key write with ``stage.last_run.finish()`` after the attend."""
        run = self._mk_run(
            dsa, x=x, y=y, q_lora=q_lora, which=which, pages_ref=pages_ref, kv0=kv_len
        )
        return lambda b, col: run.mask(b, col.shape[-1])

    def rows(self, dsa, *, x, y, q_lora, which, pages_ref, kv0):
        """R-rows hook: ``select(b, col)`` bool [h * R, n_tok] (head-major rows) for ``attend_rows``."""
        run = self._mk_run(
            dsa, x=x, y=y, q_lora=q_lora, which=which, pages_ref=pages_ref, kv0=kv0
        )

        def select(b, col):
            m = run.mask(b, col.shape[-1])
            return jnp.concatenate([m] * (col.shape[0] // self.R), axis=0)

        return select


class DsaRun:
    """One DSA layer's in-kernel work on bound scratch. Decode::

        run = stage.bind(scratch, idx_cache, which, pages_ref)
        run.prefetch(kv_len)                              # first index-key blocks (kv_len > topk)
        run.prep(q, key_raw, gate_raw, ln_w, ln_b)        # quantize, LayerNorm, records
        run.write_keys(kv_len - 1)                        # async RMW of the record row(s)
        run.select(kv_len)                                # scores + threshold (kv_len > topk only)
        o = mla.attend(..., dsa_mask=run.mask)            # dense MLA with the selection mask
        run.finish()                                      # drains the key write

    Rows mode: the same with ``kv0`` (rows at ``kv0 - 1 ..``) and ``write_keys(kv0 - 1)``.
    """

    def __init__(self, stage: DsaStage, refs, caches, which, pages_ref):
        self.s = stage
        self.buf, self.sems, self.word_ref, self.wsems, self.keys_ref = refs[:5]
        if not isinstance(caches, (list, tuple)):
            caches = [caches]
        self.caches, self.which, self.pages_ref = list(caches), which, pages_ref
        self.q8 = self.w = self.rec = None
        self._t0 = None  # first written token (write_keys)
        self._pref = None  # kv0 of prefetch
        self._reads = None  # t0 of start_row_reads
        self.selecting = None

    # ----- pools (MlaStage._pools convention) -----
    def _start(self, make):
        """``make(pool).start()`` on this layer's pool."""
        c = self.caches
        if len(c) == 1 and len(c[0].shape) == 5:
            make(c[0].at[self.which]).start()
        elif len(c) == 1:
            make(c[0]).start()
        else:
            tree_when(self.which, len(c), lambda j: make(c[j]).start())

    def _any(self):
        c = self.caches[0]
        return c.at[0] if len(c.shape) == 5 else c

    # ----- 1. per-token inputs -----
    def prep(self, q, key_raw, gate_raw, ln_w, ln_b):
        """``q`` [R, 4096] wq_b output (rounded to bf16 here), ``key_raw`` [R, 128] wk output,
        ``gate_raw`` [R, 32] weights_proj output, ``ln_w``/``ln_b`` f32 [1, 128] k_norm.
        Rounding as native: bf16 linear outputs, ``bf16(LN(key))``, ``bf16(gate * 32**-0.5)``."""
        self.prep_key(key_raw, ln_w, ln_b)
        return self.prep_q(q, gate_raw)

    def prep_key(self, key_raw, ln_w, ln_b):
        key = bf16(layer_norm(bf16(key_raw), ln_w, ln_b, self.s.eps))
        self.rec = key_records(key)
        return self

    def prep_q(self, q, gate_raw):
        s = self.s
        gate = bf16(bf16(gate_raw) * s.head_gate_scale)
        self.q8, self.w = quantize_query(bf16(q), gate, s.weight_scale)
        return self

    def prep_normed(self, q, key, gate):
        """``prep`` from the native op's own inputs: ``key`` the normed bf16 key, ``gate`` the bf16
        head gate (already times 32**-0.5)."""
        self.rec = key_records(key.astype(jnp.float32))
        self.q8, self.w = quantize_query(
            q, gate.astype(jnp.float32), self.s.weight_scale
        )
        return self

    # ----- 2. index-key write (read-modify-write of the 4-token record rows) -----
    def _rows(self, t0):
        return t0 // TOKENS_PER_ROW, (t0 + self.s.R - 1) // TOKENS_PER_ROW

    def _row_list(self, t0):
        """[(scratch row i, page index (logical), row in page, active)] of the rows a write touches."""
        r0, r_last = self._rows(t0)
        out = [
            (i, (r0 + i) // ROWS_PER_PAGE, (r0 + i) % ROWS_PER_PAGE, r0 + i <= r_last)
            for i in range(self.s.nw)
        ]
        p0, p_last = t0 // PAGE, (t0 + self.s.R - 1) // PAGE
        for k in range(self.s.ns):  # compact: the pages' scale rows
            out.append((self.s.nw + k, p0 + k, ROWS_PER_PAGE, p0 + k <= p_last))
        return out

    def _row_copy(self, cache, page, row, i, read):
        hbm = cache.at[pl.ds(self.pages_ref[page], 1), pl.ds(row, 1)]
        vm = self.word_ref.at[pl.ds(i, 1)]
        sem = self.wsems.at[0 if read else 1, i]
        return (
            pltpu.make_async_copy(hbm, vm, sem)
            if read
            else pltpu.make_async_copy(vm, hbm, sem)
        )

    def start_row_reads(self, t0):
        """Starts reading the record rows ``write_keys(t0)`` merges into (early, off the critical path)."""
        for i, pg, row, live in self._row_list(t0):
            pl.when(live)(
                lambda i=i, pg=pg, row=row: self._start(
                    lambda c: self._row_copy(c, pg, row, i, True)
                )
            )
        self._reads = t0

    def write_keys(self, t0):
        """Stores the R records at token positions ``t0 ..`` (decode: ``t0 = kv_len - 1``). The row
        writes stay in flight until ``finish``; scoring patches the rows into its VMEM blocks."""
        r0, _ = self._rows(t0)
        rows = self._row_list(t0)
        if self._reads is None:
            self.start_row_reads(t0)
        for i, pg, row, live in rows:
            pl.when(live)(self._row_copy(self._any(), pg, row, i, True).wait)
        width = HEAD_DIM if self.s.compact else RECORD
        for j in range(self.s.R):
            t = t0 + j
            i = t // TOKENS_PER_ROW - r0
            shift = ((t % TOKENS_PER_ROW) * 8).astype(jnp.uint32)
            words = pltpu.bitcast(self.word_ref[i, 0], jnp.uint32)  # [1, width]
            rec = self.rec[j : j + 1, :width]
            words = (words & ~(jnp.uint32(0xFF) << shift)) | (rec << shift)
            self.word_ref[i, 0] = pltpu.bitcast(words, jnp.uint8)
            if (
                self.s.compact
            ):  # scale byte: sublane 0 (byte 0 of the word), lane t % 128
                k = self.s.nw + t // PAGE - t0 // PAGE
                words = pltpu.bitcast(self.word_ref[k, 0], jnp.uint32)  # [1, 128]
                lane = lax.broadcasted_iota(jnp.int32, words.shape, 1)
                sc = self.rec[j : j + 1, HEAD_DIM : HEAD_DIM + 1]
                words = jnp.where(
                    lane == t % PAGE, (words & ~jnp.uint32(0xFF)) | sc, words
                )
                self.word_ref[k, 0] = pltpu.bitcast(words, jnp.uint8)
        for i, pg, row, live in rows:
            pl.when(live)(
                lambda i=i, pg=pg, row=row: self._start(
                    lambda c: self._row_copy(c, pg, row, i, False)
                )
            )
        self._t0 = t0

    def finish(self):
        """Drains the key write (call after the last use of the scores / mask is issued)."""
        if self._t0 is None:
            return
        for i, pg, row, live in self._row_list(self._t0):
            pl.when(live)(self._row_copy(self._any(), pg, row, i, False).wait)

    # ----- 3. scores -----
    def _page_copy(self, cache, block, i, whole=False):
        s = self.s
        slot = block % s.slots
        if (
            whole
        ):  # one DMA of the whole block from its first page (contiguous page table)
            return pltpu.make_async_copy(
                cache.at[pl.ds(self.pages_ref[block * s.bp], s.bp)],
                self.buf.at[slot],
                self.sems.at[slot, 0],
            )
        return pltpu.make_async_copy(
            cache.at[self.pages_ref[block * s.bp + i]],
            self.buf.at[slot, i],
            self.sems.at[slot, i],
        )

    def _block_dmas(self, block, n_pages, op):
        """Page DMAs of scoring block ``block`` (pages < ``n_pages``). ``contiguous``: one DMA for a block
        that lies inside the pool, page DMAs for the pool's last, partial block."""
        s = self.s

        def go(whole, i, live):
            if op == "start":
                pl.when(live)(
                    lambda: self._start(lambda c: self._page_copy(c, block, i, whole))
                )
            else:
                pl.when(live)(self._page_copy(self._any(), block, i, whole).wait)

        pages = self._any().shape[0]
        whole = None
        if (
            s.contiguous and pages % s.bp == 0
        ):  # pool padded to whole blocks: one DMA, no fallback
            go(True, 0, block * s.bp < n_pages)
            return
        if s.contiguous:
            whole = (block + 1) * s.bp <= pages
            go(True, 0, whole & (block * s.bp < n_pages))
        for i in range(s.bp):
            live = block * s.bp + i < n_pages
            go(False, i, live if whole is None else live & jnp.logical_not(whole))

    def _bounds(self, kv0):
        s = self.s
        lim_max = kv0 + s.R - 1
        return lim_max, (lim_max + PAGE - 1) // PAGE, (lim_max + s.n_tok - 1) // s.n_tok

    def prefetch(self, kv0):
        """Starts the first ``slots - 1`` index-key blocks when this call will select (any row with
        more than ``topk`` keys). Call it early (before the projections) to hide the DMA latency."""
        s = self.s
        lim_max, n_pages, n_blocks = self._bounds(kv0)

        @pl.when(lim_max > s.topk)
        def _():
            for j in range(s.slots - 1):
                pl.when(j < n_blocks)(lambda j=j: self._block_dmas(j, n_pages, "start"))

        self._pref = kv0

    def _patch_new_rows(self, slot, b):
        """Copies the merged record rows (``word_ref``) into block ``b``'s VMEM copy: the block DMA
        may have read the rows before (or while) ``write_keys`` stored them."""
        s = self.s
        for i, pg, row, live in self._row_list(self._t0):
            page = pg - b * s.bp

            @pl.when(live & (page >= 0) & (page < s.bp))
            def _():
                self.buf[slot, page, row] = self.word_ref[i, 0]

    def _block_scores(self, slot, b, lim):
        """f32 scores [R, n_tok] of block ``b`` (``-inf`` at positions >= ``lim`` [R, 1])."""
        s, n = self.s, self.s.n_tok
        if s.compact:
            raw = self.buf[slot]  # u8 [bp, 33, 4, 128]
            k8 = pltpu.bitcast(
                raw[:, :ROWS_PER_PAGE].reshape(n, HEAD_DIM), jnp.float8_e4m3fn
            )
            sb = jnp.concatenate(
                [raw[p, ROWS_PER_PAGE, 0:1, :] for p in range(s.bp)], axis=1
            )
            scale = pltpu.bitcast(sb, jnp.float8_e8m0fnu).astype(jnp.bfloat16)  # [1, n]
        else:
            raw = self.buf[slot]  # u8 [bp, 32, 4, 256]
            flat = raw.reshape(n, RECORD)
            k8 = pltpu.bitcast(flat[:, :HEAD_DIM], jnp.float8_e4m3fn)
            scale = pltpu.bitcast(
                flat[:, HEAD_DIM : HEAD_DIM + 1].T, jnp.float8_e8m0fnu
            ).astype(jnp.bfloat16)  # [1, n]
        st = jnp.einsum("nd,md->nm", self.q8, k8, preferred_element_type=jnp.float32)
        if s.R == 1:  # streamindex_topk reduce_heads, n_tok == 1
            st = st.reshape(1, HEADS, -1)
            st = jnp.maximum(st, 0.0) * self.w[:, :, None]
            acc = st.sum(axis=1)
        else:  # head-major, sequential over heads (the mixed pass; bf16_reduce)
            st = st.reshape(HEADS, s.R, -1)
            w = self.w.astype(jnp.bfloat16).astype(jnp.float32)
            acc = (
                jnp.maximum(st[0].astype(jnp.bfloat16), 0).astype(jnp.float32)
                * w[:, 0:1]
            )
            for h in range(1, HEADS):
                r = jnp.maximum(st[h].astype(jnp.bfloat16), 0)
                acc = acc + r.astype(jnp.float32) * w[:, h : h + 1]
        acc = acc * scale
        pos = b * n + lax.broadcasted_iota(jnp.int32, acc.shape, 1)
        return jnp.where(pos < lim, acc, -jnp.inf)

    def _store_keys(self, b, keys):
        n = self.s.n_tok
        if self.s.R == 1:
            nr = n // 128
            for i in range(nr):
                self.keys_ref[pl.ds(b * nr + i, 1), :] = keys[
                    :, i * 128 : (i + 1) * 128
                ]
        else:
            self.keys_ref[:, pl.ds(pl.multiple_of(b * n, n), n)] = keys

    def score(self, kv0, lim):
        """Scores every position < ``kv0 + R - 1`` into the key scratch (``lim`` [R, 1] per-row
        bound). Consumes ``prefetch``'s blocks (starts them itself when there was none)."""
        s = self.s
        lim_max, n_pages, n_blocks = self._bounds(kv0)
        if self._pref is None:
            for j in range(s.slots - 1):
                pl.when(j < n_blocks)(lambda j=j: self._block_dmas(j, n_pages, "start"))

        def body(b, c):
            nxt = b + s.slots - 1

            @pl.when(nxt < n_blocks)
            def _():
                self._block_dmas(nxt, n_pages, "start")

            self._block_dmas(b, n_pages, "wait")
            slot = b % s.slots
            if self._t0 is not None:
                self._patch_new_rows(slot, b)
            self._store_keys(b, monotone_key(self._block_scores(slot, b, lim)))
            return c

        lax.fori_loop(0, n_blocks, body, 0)
        n_groups = (n_blocks + s.group - 1) // s.group
        if (
            s.group > 1
        ):  # blocks past the last scored one up to a whole group: never selected
            lowest = jnp.full((s.R, s.n_tok), _INT_MIN, jnp.int32)

            def pad(b, c):
                self._store_keys(b, lowest)
                return c

            lax.fori_loop(n_blocks, n_groups * s.group, pad, 0)
        return n_groups

    # ----- 4. threshold selection -----
    def _counts(self, n_groups, preds):
        """``[sum(pred(keys, pos)) for pred in preds]`` per row over the scored groups, each [1, 1]
        (decode) or [R, 1]; the predicates share one pass over the keys."""
        s = self.s
        n = s.n_tok * s.group
        np_ = len(preds)
        if s.R == 1:
            nr = n // 128

            def body(g, accs):
                k = self.keys_ref[pl.ds(pl.multiple_of(g * nr, nr), nr), :]
                pos = (
                    g * n
                    + lax.broadcasted_iota(jnp.int32, k.shape, 0) * 128
                    + lax.broadcasted_iota(jnp.int32, k.shape, 1)
                )
                out = []
                for pred, acc in zip(preds, accs):
                    v = pred(k, pos).astype(jnp.int32)
                    for i in range(nr // 8):
                        acc = acc + v[i * 8 : (i + 1) * 8]
                    out.append(acc)
                return tuple(out)

            z = tuple(jnp.zeros((8, 128), jnp.int32) for _ in range(np_))
            accs = lax.fori_loop(0, n_groups, body, z)
            return [
                jnp.sum(jnp.sum(a, axis=0, keepdims=True), axis=1, keepdims=True)
                for a in accs
            ]

        def body(g, accs):
            k = self.keys_ref[:, pl.ds(pl.multiple_of(g * n, n), n)]
            pos = g * n + lax.broadcasted_iota(jnp.int32, k.shape, 1)
            out = []
            for pred, acc in zip(preds, accs):
                v = pred(k, pos).astype(jnp.int32)
                for i in range(n // 128):
                    acc = acc + v[:, i * 128 : (i + 1) * 128]
                out.append(acc)
            return tuple(out)

        z = tuple(jnp.zeros((s.R, 128), jnp.int32) for _ in range(np_))
        accs = lax.fori_loop(0, n_groups, body, z)
        return [jnp.sum(a, axis=1, keepdims=True) for a in accs]

    def _threshold(self, n_groups):
        """(tau, p_tie) [R or 1, 1]: the top-k keys are ``key > tau | (key == tau & pos <= p_tie)``.

        ``tau`` (the k-th largest key) is found digit by digit from the top, ``sel_bits`` bits per
        pass over the keys (``2**sel_bits - 1`` candidate counts per pass)."""
        s = self.s
        k_top, bits = s.topk, s.sel_bits
        shape = (1, 1) if s.R == 1 else (s.R, 1)

        def digit(i, u):  # u: the threshold's offset-binary bits found so far
            shift = 32 - bits * (i + 1)
            cands = [u | lax.shift_left(jnp.int32(j), shift) for j in range(1, 2**bits)]
            taus = [c ^ jnp.int32(_INT_MIN) for c in cands]
            counts = self._counts(n_groups, [lambda k, p, t=t: k >= t for t in taus])
            j = sum(
                (c >= k_top).astype(jnp.int32) for c in counts
            )  # counts fall with j
            return u | lax.shift_left(j, shift)

        u = lax.fori_loop(0, 32 // bits, digit, jnp.zeros(shape, jnp.int32))
        tau = u ^ jnp.int32(_INT_MIN)
        n_ge, n_gt = self._counts(
            n_groups, [lambda k, p: k >= tau, lambda k, p: k > tau]
        )
        excess = n_ge > k_top
        need = k_top - n_gt

        def pbit(i, p):  # largest p with count(key == tau & pos < p) < need
            cand = p | lax.shift_left(jnp.int32(1), s.pos_bits - 1 - i)
            (c,) = self._counts(n_groups, [lambda k, pos: (k == tau) & (pos < cand)])
            return jnp.where(c < need, cand, p)

        any_excess = jnp.max(excess.astype(jnp.int32)) > 0
        p = lax.fori_loop(
            0, jnp.where(any_excess, s.pos_bits, 0), pbit, jnp.zeros(shape, jnp.int32)
        )
        p_tie = jnp.where(excess, p, jnp.int32(2**31 - 1))
        return tau, p_tie

    def select(self, kv0):
        """Scores and selects for the R rows (row r: positions < ``kv0 + r``; decode: ``kv0 = kv_len``).
        Rows with at most ``topk`` positions select all of them; when every row does, scoring and
        selection are skipped (the kv_len <= 2048 fast path)."""
        s = self.s
        shape = (1, 1) if s.R == 1 else (s.R, 1)
        lim = kv0 + lax.broadcasted_iota(jnp.int32, shape, 0)
        lim_max = kv0 + s.R - 1
        self.all_sel = lim <= s.topk
        self.selecting = lim_max > s.topk

        def run(_):
            return self._threshold(self.score(kv0, lim))

        def skip(_):
            z = jnp.zeros(shape, jnp.int32)
            return z, z

        self.tau, self.p_tie = lax.cond(self.selecting, run, skip, None)
        return self

    # ----- 5. mask for the MLA block loop -----
    def mask(self, b, n_tok):
        """bool [R, n_tok] (decode [1, n_tok]): selected tokens of MLA attend block ``b``."""
        s = self.s
        if s.R == 1:
            nr = n_tok // 128
            k = self.keys_ref[pl.ds(pl.multiple_of(b * nr, nr), nr), :]
            k = jnp.concatenate([k[i : i + 1] for i in range(nr)], axis=1)
        else:
            k = self.keys_ref[:, pl.ds(pl.multiple_of(b * n_tok, n_tok), n_tok)]
        pos = b * n_tok + lax.broadcasted_iota(jnp.int32, k.shape, 1)
        m = (k > self.tau) | ((k == self.tau) & (pos <= self.p_tie))
        return m | self.all_sel
