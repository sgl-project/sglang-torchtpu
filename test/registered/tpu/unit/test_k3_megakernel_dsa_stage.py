# SPDX-License-Identifier: Apache-2.0
"""The K3 megakernel's in-kernel DSA indexer stage against the native DSA path.

``stages.dsa.DsaStage`` (Kimi-K3.1) writes each token's index-key record,
scores the cached keys and selects the top-2048 as a mask on the megakernel's
dense MLA. Checked on one TPU core, against the native ops of the sglang path:

* records: the stage's quantized key records are byte-identical to the Pallas
  key writer's (``store_keys_pallas``);
* selection: after one step the index-key cache is byte-identical to
  ``dsa_indexer_step``'s, and the selected set equals the native top-k up to
  keys tied at the threshold (decode and 8-row verify; native and compact
  layouts);
* attention: the masked dense MLA attend matches native sparse MLA over the
  native top-k within bf16 noise (decode and 8 rows).

Usage:
    python test/registered/tpu/unit/test_k3_megakernel_dsa_stage.py
"""

import functools
import os
import sys

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=150, suite="stage-a-test-tpu-unit", chips=1)

# Standalone raw-JAX kernel test: select the physical backend before JAX loads.
assert "jax" not in sys.modules
os.environ["JAX_PLATFORMS"] = "tpu"

import unittest

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sglang_tpu.attention.pallas import dsa_index_topk as dit
from sglang_tpu.attention.pallas import mla_sparse_fused as mfs
from sglang_tpu.attention.pallas.dsa_key_store import store_keys_pallas
from sglang_tpu.attention.pallas.mla_sparse import kv_cache_utils as kcu
from sglang_tpu.attention.pallas.mla_sparse.kernel import (
    TENSORCORE_CACHE_LAYOUT,
    sparse_ragged_paged_attention,
)
from sglang_tpu.attention.pallas.util import quantize_fp8_ue8m0
from sglang_tpu.k3_megakernel.config import Topology
from sglang_tpu.k3_megakernel.kernel.stages import dsa as D
from sglang_tpu.k3_megakernel.kernel.stages.mla import SM_SCALE, MlaStage

H, DH, K = D.HEADS, D.HEAD_DIM, D.TOPK
WEIGHT_SCALE = DH**-0.5
CAP = 16384  # tokens of the score scratch / the pools
N_PAGES = CAP // 128
MLA_BLOCK_PAGES = 4
MLA_HEADS = 12  # one lane's heads at 8 lanes
E4 = jnp.float8_e4m3fn
VMEM_LIMIT = 100 * 1024 * 1024


def _make_cache(rng, n_pages, ctx, dup_every=0):
    """Native index-key records [n_pages, 32, 4, 256], valid at positions < ctx."""
    keys = rng.standard_normal((n_pages * 128, DH)).astype(np.float32)
    if dup_every:  # many keys with the same score: ties at the threshold
        keys[::dup_every] = np.abs(keys[0]) * 3
    k8, _, sb = quantize_fp8_ue8m0(jnp.asarray(keys, jnp.bfloat16))
    rec = np.zeros((n_pages * 128, 256), np.uint8)
    rec[:, :128] = np.asarray(jax.lax.bitcast_convert_type(k8, jnp.uint8))
    rec[:, 128] = np.asarray(sb)[:, 0]
    rec[ctx:] = rng.integers(0, 256, rec[ctx:].shape, dtype=np.uint8)  # stale garbage
    return rec.reshape(n_pages, 32, 4, 256)


def _native_step(q, key, gate, cache, kv_len, pages, n_q):
    """dsa_indexer_step for one request of n_q query rows ending at kv_len: (top-k [n_q, K], cache)."""
    args = (
        jnp.asarray(q, jnp.bfloat16).reshape(n_q, H, DH),
        jnp.asarray(key, jnp.bfloat16),
        jnp.asarray(gate, jnp.bfloat16),
        jnp.asarray(cache),
        jnp.array([kv_len], jnp.int32),
        jnp.asarray(pages, jnp.int32),
        jnp.array([0, n_q], jnp.int32),
        jnp.array([1, 1, 1] if n_q == 1 else [0, 0, 1], jnp.int32),
    )
    step = jax.jit(
        lambda *a: dit.dsa_indexer_step(
            *a,
            weight_scale=WEIGHT_SCALE,
            topk=K,
            key_store="pallas",
            segments="decode" if n_q == 1 else "mixed",
        )
    )
    idx, out = step(*args)
    return np.asarray(idx), np.asarray(out)


@functools.lru_cache(maxsize=None)  # one compile per layout, not per case
def _build_indexer(rows, compact, contiguous, cache_shape):
    """Single-core call: prep (native-normed inputs) + write_keys + select; returns
    (cache, score keys, [tau | p_tie] rows)."""
    stage = D.DsaStage(rows=rows, capacity=CAP, compact=compact, contiguous=contiguous)
    keys_shape = (
        (stage.capacity // 128, 128) if stage.R == 1 else (stage.R, stage.capacity)
    )

    def kernel(
        meta, pages, q_ref, k_ref, g_ref, cache_in, cache, keys_out, thr_out, *scratch
    ):
        del cache_in
        run = stage.bind(scratch, cache, 0, pages)
        kv0 = meta[0]
        run.prefetch(kv0)
        run.prep_normed(
            q_ref[...], k_ref[...].astype(jnp.float32), g_ref[...].astype(jnp.float32)
        )
        run.write_keys(kv0 - 1)
        run.select(kv0)
        run.finish()
        keys_out[...] = run.keys_ref[...]
        rows = run.tau.shape[0]
        thr_out[...] = jnp.zeros(thr_out.shape, jnp.int32)
        thr_out[0:rows, 0:1] = run.tau
        thr_out[0:rows, 1:2] = run.p_tie

    call = pl.pallas_call(
        kernel,
        out_shape=(
            jax.ShapeDtypeStruct(cache_shape, jnp.uint8),
            jax.ShapeDtypeStruct(keys_shape, jnp.int32),
            jax.ShapeDtypeStruct((8, 128), jnp.int32),
        ),
        in_specs=[pl.BlockSpec(memory_space=pltpu.SMEM)] * 2
        + [pl.BlockSpec(memory_space=pltpu.VMEM)] * 3
        + [pl.BlockSpec(memory_space=pltpu.HBM)],
        out_specs=(
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pltpu.VMEM),
        ),
        scratch_shapes=stage.scratch(),
        input_output_aliases={5: 0},
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=VMEM_LIMIT),
    )
    return jax.jit(call)


def _selected(keys, tau, p_tie, lim, rows):
    """Bool [rows, capacity]: the stage's selection mask (``DsaRun.mask``) over the score keys."""
    k = np.asarray(keys).reshape(rows, -1)
    pos = np.arange(k.shape[1])[None]
    tau = np.asarray(tau).reshape(rows, 1)
    p = np.asarray(p_tie).reshape(rows, 1)
    lim = np.asarray(lim).reshape(rows, 1)
    m = (k > tau) | ((k == tau) & (pos <= p))
    m = np.where(lim <= K, True, m)
    return m & (pos < lim)


def _inputs(rng, rows, *, ties=False):
    q = (rng.standard_normal((rows, H * DH)) * 2).astype(np.float32)
    key = rng.standard_normal((rows, DH)).astype(np.float32)
    gate = (rng.standard_normal((rows, H)) * 0.2).astype(np.float32)
    if ties:
        key[:] = 0
        key[:, 0] = 1.0
        q, gate = np.abs(q), np.abs(gate)
    return tuple(jnp.asarray(a, jnp.bfloat16) for a in (q, key, gate))


def _rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


class TestDsaStageRecords(unittest.TestCase):
    def test_records_match_pallas_writer(self):
        """``key_records`` == the bytes store_keys_pallas writes for the same normed key."""
        rng = np.random.default_rng(5)
        tokens = 1024

        def mine(x):
            def kernel(x_ref, o_ref):
                o_ref[...] = D.key_records(x_ref[...].astype(jnp.float32))

            return pl.pallas_call(
                kernel, out_shape=jax.ShapeDtypeStruct((x.shape[0], 256), jnp.uint32)
            )(x)

        n_pages = tokens // 128
        pages = jnp.arange(n_pages, dtype=jnp.int32)
        store = jax.jit(
            lambda c, k, kv: store_keys_pallas(
                c,
                k,
                kv,
                pages,
                jnp.array([0, 16], jnp.int32),
                jnp.array([0, 0, 1], jnp.int32),
            )
        )
        for scale, off in ((0.05, 0.0), (1.0, 0.0), (3.0, 0.5)):
            x = jnp.asarray(
                rng.standard_normal((tokens, DH)) * scale + off, jnp.bfloat16
            )
            cache = jnp.zeros(dit.index_key_cache_shape(n_pages, 128, DH), jnp.uint8)
            for s in range(0, tokens, 16):  # 16 tokens per call, like decode batches
                cache = store(cache, x[s : s + 16], jnp.array([s + 16], jnp.int32))
            want = np.asarray(cache).reshape(tokens, 256)
            got = np.asarray(mine(x)).astype(np.uint8)
            with self.subTest(scale=scale, off=off):
                np.testing.assert_array_equal(got, want)


class TestDsaStageSelect(unittest.TestCase):
    def _check(self, *, rows, ctx, compact, contiguous, ties=False, seed=0):
        rng = np.random.default_rng(seed)
        kv0 = ctx - rows + 1  # rows at positions kv0 - 1 .. ctx - 1
        if contiguous:  # identity page table, pool padded to whole scoring blocks
            pages = np.arange(N_PAGES, dtype=np.int32)
            pool_pages = N_PAGES + D.SCORE_BLOCK_PAGES
        else:
            pages = rng.permutation(N_PAGES + 4)[:N_PAGES].astype(np.int32)
            pool_pages = N_PAGES + 4
        cache = np.zeros((pool_pages, 32, 4, 256), np.uint8)
        cache[pages] = _make_cache(rng, N_PAGES, kv0 - 1, dup_every=3 if ties else 0)
        q, key, gate = _inputs(rng, rows, ties=ties)

        idx_native, cache_native = _native_step(q, key, gate, cache, ctx, pages, rows)

        cache_in = D.native_to_compact(cache) if compact else cache
        f = _build_indexer(rows, compact, contiguous, cache_in.shape)
        cache_mk, keys, thr = (
            np.asarray(a)
            for a in f(
                jnp.array([kv0], jnp.int32),
                jnp.asarray(pages),
                q,
                key,
                gate,
                jnp.asarray(cache_in),
            )
        )
        if compact:
            cache_native = D.native_to_compact(cache_native)
        np.testing.assert_array_equal(cache_mk, cache_native, err_msg="index-key cache")

        lims = kv0 + np.arange(rows)
        tau, p_tie = thr[:rows, 0], thr[:rows, 1]
        mask = _selected(keys, tau, p_tie, lims, rows)
        flat_keys = np.asarray(keys).reshape(rows, -1)
        for r in range(rows):
            mine = set(np.nonzero(mask[r])[0].tolist())
            native = {int(i) for i in idx_native[r] if i >= 0}
            self.assertEqual(len(mine), min(lims[r], K), f"row {r}")
            # Ties at the threshold may resolve differently (the native top-k is unordered).
            for i in mine ^ native:
                self.assertEqual(flat_keys[r, i], tau[r], f"row {r} token {i}")

    def test_decode(self):
        for ctx in (2048, 2049, 3001, 8192):
            for compact, contiguous in ((False, False), (True, True)):
                with self.subTest(ctx=ctx, compact=compact):
                    self._check(rows=1, ctx=ctx, compact=compact, contiguous=contiguous)

    def test_decode_ties(self):
        self._check(rows=1, ctx=8000, compact=True, contiguous=True, ties=True)

    def test_rows(self):
        for ctx in (2050, 8192):
            for compact, contiguous in ((False, False), (True, True)):
                with self.subTest(ctx=ctx, compact=compact):
                    self._check(rows=8, ctx=ctx, compact=compact, contiguous=contiguous)


def _latent(rng, ctx):
    """fp8-representable latent rows (bf16 [CAP, 640]) and their fp8 kv_c / k_pe for native."""
    kvc8 = jnp.asarray(rng.standard_normal((CAP, 512)), E4)
    kpe8 = jnp.asarray(rng.standard_normal((CAP, 64)), E4)
    rows = np.zeros((CAP, 640), np.float32)
    rows[:, :512] = np.asarray(kvc8.astype(jnp.float32))
    rows[:, 512:576] = np.asarray(kpe8.astype(jnp.float32))
    rows[ctx:] = 0
    return rows.astype(ml_dtypes.bfloat16), kvc8, kpe8


def _native_attend(ql, qpe, kvc8, kpe8, ctx, topk_idx, n_q):
    """Native sparse MLA over the native top-k: [n_q, h, 512] f32."""
    pages = jnp.arange(N_PAGES, dtype=jnp.int32)
    nope_spec, rope_spec = mfs._split_specs(N_PAGES, 128)
    cn, cr = kcu.update_sparse_mla_kv_cache_jax(
        jnp.zeros(nope_spec.shape, jnp.uint8),
        jnp.zeros(rope_spec.shape, jnp.uint8),
        kvc8[:ctx],
        kpe8[:ctx],
        jnp.array([ctx], jnp.int32),
        pages,
        jnp.array([0, ctx], jnp.int32),
        nope_spec=nope_spec,
        rope_spec=rope_spec,
    )
    out = sparse_ragged_paged_attention(
        jnp.concatenate([ql, qpe], axis=-1),
        cn,
        cr,
        jnp.asarray(topk_idx, jnp.int32),
        pages,
        jnp.array([0, n_q], jnp.int32),
        jnp.array([1, 1, 1] if n_q == 1 else [0, 0, 1], jnp.int32),
        sm_scale=SM_SCALE,
        k_scale=1.0,
        cache_layout=TENSORCORE_CACHE_LAYOUT,
    )
    return np.asarray(out[:, :, :512]).astype(np.float32)


@functools.lru_cache(maxsize=None)  # one compile per row count, not per case
def _build_attend(rows, idx_shape):
    """Single-core call: DSA select, then the MK's masked dense MLA attend (f32 scores)."""
    h = MLA_HEADS
    mla = MlaStage(Topology(1, 8), block_pages=MLA_BLOCK_PAGES, rows=rows)
    mla.score_f32 = True
    stage = D.DsaStage(rows=rows, capacity=CAP)

    def kernel(
        meta, pages, ql, qp, new, q, k, g, lat_hbm, idx_in, out_ref, idx_hbm, *scr
    ):
        del idx_in
        kv_buf, sems, acc = scr[:3]
        run = stage.bind(scr[3:], idx_hbm, 0, pages)
        kv0 = meta[0]
        run.prefetch(kv0)
        run.prep_normed(q[...], k[...].astype(jnp.float32), g[...].astype(jnp.float32))
        run.write_keys(kv0 - 1)
        run.select(kv0)

        def select(b, col):
            m = run.mask(b, col.shape[-1])
            # Decode: one mask row broadcasts over the heads; concatenating i1
            # rows fails to lower in Mosaic ("Invalid vector register cast").
            if m.shape[0] == 1:
                return m
            return jnp.concatenate([m] * (col.shape[0] // m.shape[0]), axis=0)

        args = (ql[...], qp[...], new[...], [lat_hbm], 0, pages, kv0, kv_buf, sems, acc)
        if rows == 1:
            out_ref[...] = mla.attend(*args, select=select)
        else:
            out_ref[...] = mla.attend_rows(*args, select=select)
        run.finish()

    call = pl.pallas_call(
        kernel,
        out_shape=(
            jax.ShapeDtypeStruct((h * rows, 512), jnp.float32),
            jax.ShapeDtypeStruct(idx_shape, jnp.uint8),
        ),
        in_specs=[pl.BlockSpec(memory_space=pltpu.SMEM)] * 2
        + [pl.BlockSpec(memory_space=pltpu.VMEM)] * 6
        + [pl.BlockSpec(memory_space=pltpu.HBM)] * 2,
        out_specs=(
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pltpu.HBM),
        ),
        scratch_shapes=[
            pltpu.VMEM((2, MLA_BLOCK_PAGES * 64, 2, 640), jnp.bfloat16),
            pltpu.SemaphoreType.DMA((2, MLA_BLOCK_PAGES)),
            pltpu.VMEM((h * rows, 512), jnp.float32),
        ]
        + stage.scratch(),
        input_output_aliases={9: 1},
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=VMEM_LIMIT),
    )
    return jax.jit(call)


class TestDsaStageAttend(unittest.TestCase):
    # Same keys and the same selection; the dense masked softmax and the sparse
    # kernel differ only in bf16 rounding / f32 summation order (measured 1.4-1.9e-3).
    REL_TOL = 1e-2

    def _check(self, *, rows, ctx, seed):
        rng = np.random.default_rng(seed)
        h = MLA_HEADS
        kv0 = ctx - rows + 1
        pages = np.arange(N_PAGES, dtype=np.int32)
        idx_cache = _make_cache(rng, N_PAGES, kv0 - 1)
        q, key, gate = _inputs(rng, rows)
        idx_native, _ = _native_step(q, key, gate, idx_cache, ctx, pages, rows)

        ql = jnp.asarray(rng.standard_normal((rows, h, 512)), jnp.bfloat16)
        qpe = jnp.asarray(rng.standard_normal((rows, h, 64)), jnp.bfloat16)
        lat, kvc8, kpe8 = _latent(rng, ctx)
        want = _native_attend(ql, qpe, kvc8, kpe8, ctx, idx_native, rows)

        # MK rows are head-major (row h * R + r); qp carries 64 zero columns.
        qlh = jnp.transpose(ql, (1, 0, 2)).reshape(h * rows, 512)
        qph = jnp.concatenate(
            [jnp.transpose(qpe, (1, 0, 2)), jnp.zeros((h, rows, 64), jnp.bfloat16)], -1
        ).reshape(h * rows, 128)
        f = _build_attend(rows, idx_cache.shape)
        got, _ = f(
            jnp.array([kv0], jnp.int32),
            jnp.asarray(pages),
            qlh,
            qph,
            jnp.asarray(lat[kv0 - 1 : kv0 - 1 + rows]),
            q,
            key,
            gate,
            jnp.asarray(lat.reshape(N_PAGES, 64, 2, 640)),
            jnp.asarray(idx_cache),
        )
        got = np.asarray(got).reshape(h, rows, 512).transpose(1, 0, 2)
        for r in range(rows):
            self.assertLess(_rel(got[r], want[r]), self.REL_TOL, f"row {r}")

    def test_decode(self):
        for ctx in (1500, 3001, 8192):
            with self.subTest(ctx=ctx):
                self._check(rows=1, ctx=ctx, seed=3)

    def test_rows(self):
        for ctx in (3001, 8192):
            with self.subTest(ctx=ctx):
                self._check(rows=8, ctx=ctx, seed=7)


if __name__ == "__main__":
    unittest.main(verbosity=2)
