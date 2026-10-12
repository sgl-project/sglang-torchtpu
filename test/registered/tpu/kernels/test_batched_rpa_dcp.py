"""Pure-JAX TPU tests for the context-parallel modes of the batched RPA kernel.

The geometries mirror gpt-oss-120b decode context parallelism on one host: each
rank keeps all of its KV heads over 1/cp of the positions and attends the
group's gathered query heads. DCP2 is 16 query heads over 2 KV heads, DCP8 is 64
over 8; head_dim 64, bf16, page size 256, sliding window 128.

Cases:

* ``cache_only`` + ``return_lse`` on every rank's position-interleaved shard,
  merged by LSE, equals a dense reference over the whole sequence -- with and
  without the window, with and without the sink fold -- on a decode batch with
  a sequence whose shard is empty on most ranks, and with k blocks small enough
  that the global window skips whole blocks.
* The same on an extend prefix pass in the MIXED range, including q blocks
  past the window that see no prefix key and a sequence with no prefix at all.
* The extend two-pass merge: a cache-only prefix pass and a chunk-local pass
  (kv_len == q_len, keys only from the k/v operands, no KV write) merged by LSE
  equal full causal attention, with and without sinks.
* ``skip_kv_write`` leaves the KV cache untouched while the output is unchanged.
* With every new flag at its default the kernel still reproduces the v3
  reference over decode, extend, windowed and soft-capped shapes.

Requires JAX on a TPU device; skipped otherwise. No torch_tpu, no server.
"""

from __future__ import annotations

import os
import sys
import unittest
from typing import NamedTuple

import numpy as np

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=360, suite="stage-a-test-tpu-unit")

# Standalone raw-JAX kernel test, not a TorchTPU runtime process. SGLang
# platform discovery selects CPU-only JAX for export, so select the physical
# backend before the first JAX import.
assert "jax" not in sys.modules
os.environ["JAX_PLATFORMS"] = "tpu"


def _has_jax_tpu() -> bool:
    try:
        import jax
    except Exception:
        return False
    try:
        return any("TPU" in d.device_kind for d in jax.devices())
    except Exception:
        return False


_SKIP_REASON = "Requires JAX with a TPU device"

HEAD_DIM = 64
PAGE_SIZE = 256
SLIDING_WINDOW = 128
SM_SCALE = 1.0 / float(np.sqrt(HEAD_DIM))
TOLERANCE = 0.05


class Geometry(NamedTuple):
    name: str
    num_q_heads: int
    num_kv_heads: int
    cp: int


DCP2 = Geometry("dcp2", num_q_heads=16, num_kv_heads=2, cp=2)
DCP8 = Geometry("dcp8", num_q_heads=64, num_kv_heads=8, cp=8)


def _small_decode_blocks():
    """One page per k block, so long shards span several k blocks."""
    from sglang_tpu.attention.pallas.batched_rpa.configs import BlockSizes

    return BlockSizes(bq_sz=1, bq_c_sz=1, bkv_sz=PAGE_SIZE, batch_size=8, n_buffer=3)


def _small_prefill_blocks():
    """Several q blocks per chunk, two compute tiles per q block, two lanes."""
    from sglang_tpu.attention.pallas.batched_rpa.configs import BlockSizes

    return BlockSizes(bq_sz=128, bq_c_sz=64, bkv_sz=PAGE_SIZE, batch_size=2, n_buffer=3)


def _round_to_bf16(array: np.ndarray) -> np.ndarray:
    """bf16-round a float32 array and return it as float32.

    Kernel and reference then consume identical numbers, so the comparison
    measures the kernel, not the cast.
    """
    import jax.numpy as jnp

    return np.array(jnp.asarray(array, jnp.bfloat16).astype(jnp.float32))


def _to_bf16(array: np.ndarray):
    import jax.numpy as jnp

    return jnp.asarray(array, jnp.bfloat16)


def _random(rng, shape, scale=1.0) -> np.ndarray:
    return _round_to_bf16(scale * rng.standard_normal(shape, np.float32))


def _fused_cache(slot_k: np.ndarray, slot_v: np.ndarray):
    """Pack per-slot K/V [slots, num_kv_heads, HEAD_DIM] into the kernel's
    token-major paged cache: K and V of a head interleaved along the head axis."""
    import jax.numpy as jnp

    from sglang_tpu.attention.pallas.batched_rpa.wrapper import (
        get_kv_cache_shape,
    )

    slots, num_kv_heads, _ = slot_k.shape
    shape = get_kv_cache_shape(
        slots // PAGE_SIZE, PAGE_SIZE, num_kv_heads, HEAD_DIM, jnp.bfloat16
    )
    _, _, kv_words, packing, aligned_head_dim = shape
    kv = np.stack([slot_k, slot_v], axis=2).reshape(slots, 2 * num_kv_heads, HEAD_DIM)
    kv = np.pad(
        kv,
        (
            (0, 0),
            (0, kv_words * packing - 2 * num_kv_heads),
            (0, aligned_head_dim - HEAD_DIM),
        ),
    )
    return _to_bf16(kv).reshape(shape)


class _PagedShard(NamedTuple):
    cache: object
    page_indices: np.ndarray
    kv_lens: list


def _paged_shard(seq_keys, seq_values, pages_per_seq, rng, num_kv_heads):
    """Page table and cache holding seq_keys[s] / seq_values[s] as sequence s.

    Physical pages are shuffled so the page table is followed, and every slot
    past a sequence's length holds large stale values that must never be read.
    """
    num_seqs = len(seq_keys)
    total_pages = num_seqs * pages_per_seq
    slots = total_pages * PAGE_SIZE
    slot_k = _random(rng, (slots, num_kv_heads, HEAD_DIM), scale=50.0)
    slot_v = _random(rng, (slots, num_kv_heads, HEAD_DIM), scale=50.0)
    page_indices = rng.permutation(total_pages).astype(np.int32)
    for seq, (keys, values) in enumerate(zip(seq_keys, seq_values)):
        assert len(keys) <= pages_per_seq * PAGE_SIZE
        for j in range(len(keys)):
            page = page_indices[seq * pages_per_seq + j // PAGE_SIZE]
            slot = page * PAGE_SIZE + j % PAGE_SIZE
            slot_k[slot] = keys[j]
            slot_v[slot] = values[j]
    return _PagedShard(
        cache=_fused_cache(slot_k, slot_v),
        page_indices=page_indices,
        kv_lens=[len(keys) for keys in seq_keys],
    )


def _run_kernel(queries, keys, values, shard, cu_q_lens, distribution, **kwargs):
    """Call the kernel and return every output as a float32 numpy array."""
    import jax.numpy as jnp

    from sglang_tpu.attention.pallas.batched_rpa.wrapper import (
        ragged_paged_attention,
    )

    result = ragged_paged_attention(
        _to_bf16(queries),
        _to_bf16(keys),
        _to_bf16(values),
        shard.cache,
        jnp.asarray(shard.kv_lens, jnp.int32),
        jnp.asarray(shard.page_indices, jnp.int32),
        jnp.asarray(cu_q_lens, jnp.int32),
        jnp.asarray(distribution, jnp.int32),
        sm_scale=SM_SCALE,
        **kwargs,
    )
    return [np.asarray(x.astype(jnp.float32)) for x in result]


def _cp_kwargs(geo: Geometry, rank: int, kv_lens_global):
    import jax.numpy as jnp

    return dict(
        cache_only=True,
        skip_kv_write=True,
        return_lse=True,
        cp_group_size=geo.cp,
        cp_rank=jnp.asarray([rank], jnp.int32),
        kv_lens_global=jnp.asarray(kv_lens_global, jnp.int32),
    )


def _local_positions(length: int, rank: int, cp: int) -> np.ndarray:
    """Global positions rank `rank` stores, in local key order."""
    return np.arange(rank, length, cp)


def _pages_per_seq(lengths, cp: int) -> int:
    """Pages per sequence on the rank with the longest shard (rank 0)."""
    longest = max(-(-length // cp) for length in lengths)
    return max(1, -(-longest // PAGE_SIZE))


def _dense_attention(
    queries: np.ndarray,
    keys: np.ndarray,
    values: np.ndarray,
    query_positions: np.ndarray,
    window: int | None,
    causal: bool = False,
    key_positions: np.ndarray | None = None,
):
    """Reference softmax attention; (out, lse) in float32.

    queries: [q_len, num_q_heads, HEAD_DIM]; keys/values: [kv_len, num_kv_heads,
    HEAD_DIM] at ``key_positions`` (default 0..kv_len-1). ``window`` keeps key
    ``p`` for the query at position ``Q`` when ``p > Q - window``, ``causal``
    when ``p <= Q``. A row that keeps no key has output 0 and LSE -inf.
    """
    q_len, num_q_heads, _ = queries.shape
    kv_len, num_kv_heads, _ = keys.shape
    q_per_kv = num_q_heads // num_kv_heads
    out = np.zeros(queries.shape, np.float32)
    lse = np.full((q_len, num_q_heads), -np.inf, np.float32)
    if key_positions is None:
        key_positions = np.arange(kv_len)
    for i in range(q_len):
        keep = np.ones(kv_len, dtype=bool)
        if causal:
            keep &= key_positions <= int(query_positions[i])
        if window is not None:
            keep &= key_positions > int(query_positions[i]) - window
        if not keep.any():
            continue
        for kv_head in range(num_kv_heads):
            heads = slice(kv_head * q_per_kv, (kv_head + 1) * q_per_kv)
            scores = queries[i, heads] @ keys[keep, kv_head].T * SM_SCALE
            top = scores.max(axis=1, keepdims=True)
            weights = np.exp(scores - top)
            total = weights.sum(axis=1, keepdims=True)
            lse[i, heads] = (top + np.log(total))[:, 0]
            out[i, heads] = (weights / total) @ values[keep, kv_head]
    return out, lse


def _first_window_key(q_global: int, rank: int, cp: int) -> int:
    """First local key j of `rank` inside the window of global query q_global."""
    j = 0
    while j * cp + rank <= q_global - SLIDING_WINDOW:
        j += 1
    return j


def _merge_by_lse(partials):
    """Combine per-rank (out, lse) pairs into the full-sequence (out, lse)."""
    lses = np.stack([lse for _, lse in partials])
    top = lses.max(axis=0)
    finite = np.isfinite(top)
    shifted = np.where(np.isfinite(lses), lses - np.where(finite, top, 0.0), -np.inf)
    weights = np.exp(shifted)
    total = weights.sum(axis=0)
    merged_lse = np.where(
        finite, top + np.log(np.where(total > 0, total, 1.0)), -np.inf
    )
    out = np.zeros_like(partials[0][0])
    for rank, (partial_out, _) in enumerate(partials):
        scale = np.where(total > 0, weights[rank] / total, 0.0)
        out += partial_out * scale[..., None]
    return out, merged_lse


def _fold_sink(out: np.ndarray, lse: np.ndarray, sinks: np.ndarray) -> np.ndarray:
    """The sink is a virtual key with logit ``sink`` and an all-zero value."""
    scale = 1.0 / (1.0 + np.exp(np.where(np.isfinite(lse), sinks[None, :] - lse, 0.0)))
    scale = np.where(np.isfinite(lse), scale, 0.0)
    return out * scale[..., None]


class _CloseAssertions(unittest.TestCase):
    def assert_out_close(self, got, want, label):
        diff = float(np.abs(got - want).max())
        self.assertLess(diff, TOLERANCE, f"{label}: max abs output diff {diff}")

    def assert_lse_close(self, got, want, label):
        np.testing.assert_array_equal(
            np.isneginf(got), np.isneginf(want), f"{label}: -inf rows differ"
        )
        finite = np.isfinite(want)
        diff = float(np.abs(got[finite] - want[finite]).max(initial=0.0))
        self.assertLess(diff, TOLERANCE, f"{label}: max abs lse diff {diff}")


@unittest.skipIf(not _has_jax_tpu(), _SKIP_REASON)
class TestBatchedRpaDcpDecode(_CloseAssertions):
    """Interleaved shards of a decode batch merge back into the dense result."""

    # Global KV lengths including the decoded token. Length 1 leaves every rank
    # but rank 0 empty; 2600 spans several one-page k blocks on every rank.
    SEQ_LENS = (2600, 1, 173, 300)
    MAX_NUM_TOKENS = 8

    @classmethod
    def setUpClass(cls):
        cls._partials_cache = {}

    def _data(self, geo: Geometry):
        rng = np.random.default_rng(1234 + geo.cp)
        kv_shape = (geo.num_kv_heads, HEAD_DIM)
        full_k = [_random(rng, (n, *kv_shape)) for n in self.SEQ_LENS]
        full_v = [_random(rng, (n, *kv_shape)) for n in self.SEQ_LENS]
        queries = _random(rng, (self.MAX_NUM_TOKENS, geo.num_q_heads, HEAD_DIM))
        sinks = _random(rng, (geo.num_q_heads,))
        return full_k, full_v, queries, sinks

    def _run_shard(self, geo: Geometry, rank: int, window, blocks, sinks=None):
        import jax.numpy as jnp

        full_k, full_v, queries, _ = self._data(geo)
        rng = np.random.default_rng(rank)
        positions = [_local_positions(n, rank, geo.cp) for n in self.SEQ_LENS]
        shard = _paged_shard(
            [k[p] for k, p in zip(full_k, positions)],
            [v[p] for v, p in zip(full_v, positions)],
            _pages_per_seq(self.SEQ_LENS, geo.cp),
            rng,
            geo.num_kv_heads,
        )
        num_seqs = len(self.SEQ_LENS)
        unused_kv = np.zeros((self.MAX_NUM_TOKENS, geo.num_kv_heads, HEAD_DIM))
        out, _, lse = _run_kernel(
            queries,
            unused_kv,
            unused_kv,
            shard,
            np.arange(num_seqs + 1),
            [num_seqs] * 3,
            attention_sink=None if sinks is None else jnp.asarray(sinks),
            sliding_window=window,
            decode_block_sizes=blocks,
            **_cp_kwargs(geo, rank, self.SEQ_LENS),
        )
        return out[:num_seqs], lse[:num_seqs]

    def _partials(self, geo: Geometry, window, blocks):
        key = (geo, window, blocks)
        if key not in self._partials_cache:
            self._partials_cache[key] = [
                self._run_shard(geo, rank, window, blocks) for rank in range(geo.cp)
            ]
        return self._partials_cache[key]

    def _reference(self, geo: Geometry, window, sinks=None):
        full_k, full_v, queries, _ = self._data(geo)
        num_seqs = len(self.SEQ_LENS)
        out = np.zeros((num_seqs, geo.num_q_heads, HEAD_DIM), np.float32)
        lse = np.zeros((num_seqs, geo.num_q_heads), np.float32)
        for seq, length in enumerate(self.SEQ_LENS):
            seq_out, seq_lse = _dense_attention(
                queries[seq : seq + 1],
                full_k[seq],
                full_v[seq],
                np.asarray([length - 1]),
                window,
            )
            if sinks is not None:
                seq_out = _fold_sink(seq_out, seq_lse, sinks)
            out[seq], lse[seq] = seq_out[0], seq_lse[0]
        return out, lse

    def test_merged_shards_match_the_dense_reference(self):
        for geo in (DCP2, DCP8):
            for window in (None, SLIDING_WINDOW):
                for blocks in (None, _small_decode_blocks()):
                    label = f"{geo.name} window={window} blocks={blocks}"
                    with self.subTest(label):
                        out, lse = _merge_by_lse(self._partials(geo, window, blocks))
                        ref_out, ref_lse = self._reference(geo, window)
                        self.assert_out_close(out, ref_out, label)
                        self.assert_lse_close(lse, ref_lse, label)

    def test_sink_fold_after_the_merge_matches_the_dense_reference(self):
        for geo in (DCP2, DCP8):
            sinks = self._data(geo)[3]
            for window in (None, SLIDING_WINDOW):
                label = f"{geo.name} window={window}"
                with self.subTest(label):
                    out, lse = _merge_by_lse(self._partials(geo, window, None))
                    ref_out, _ = self._reference(geo, window, sinks)
                    self.assert_out_close(_fold_sink(out, lse, sinks), ref_out, label)

    def test_empty_shard_row_returns_zero_output_and_minus_inf_lse(self):
        # Sequence 1 has one global token, which rank 0 owns: on the last rank
        # its shard is empty and its q_len (1) exceeds its kv_len (0).
        for geo in (DCP2, DCP8):
            with self.subTest(geo.name):
                out, lse = self._partials(geo, SLIDING_WINDOW, None)[geo.cp - 1]
                self.assertTrue(np.all(out[1] == 0.0), out[1])
                self.assertTrue(np.all(np.isneginf(lse[1])), lse[1])
                # The sequence after the empty one must still be attended.
                self.assertTrue(np.all(np.isfinite(lse[2])))
                self.assertGreater(float(np.abs(out[2]).max()), 0.0)

    def test_sink_in_a_cache_only_pass_joins_every_row(self):
        # A sink is one more key with a zero value: lse = logaddexp(sink, lse)
        # and out is scaled by its softmax share; the empty row (sequence 1 on
        # the last rank) outputs 0 with lse == sink rather than inf or NaN.
        for geo in (DCP2, DCP8):
            with self.subTest(geo.name):
                rank = geo.cp - 1
                sinks = self._data(geo)[3]
                out, lse = self._run_shard(geo, rank, SLIDING_WINDOW, None, sinks)
                bare_out, bare_lse = self._partials(geo, SLIDING_WINDOW, None)[rank]
                self.assert_out_close(
                    out, _fold_sink(bare_out, bare_lse, sinks), geo.name
                )
                self.assert_lse_close(
                    lse, np.logaddexp(sinks[None, :], bare_lse), geo.name
                )
                self.assertTrue(np.all(out[1] == 0.0), out[1])


@unittest.skipIf(not _has_jax_tpu(), _SKIP_REASON)
class TestBatchedRpaDcpExtendPrefixPass(_CloseAssertions):
    """cache_only with q_len > 1 in the MIXED range: every cached key precedes
    every query, and the window runs in global positions."""

    # (prefix, chunk) per sequence. The 300-token chunk outruns the window, so
    # its later q blocks see no prefix key; the last sequence has no prefix.
    SEQS = ((700, 300), (37, 5), (0, 3))
    MAX_NUM_TOKENS = 320

    def _data(self, geo: Geometry):
        rng = np.random.default_rng(99 + geo.cp)
        kv_shape = (geo.num_kv_heads, HEAD_DIM)
        prefix_k = [_random(rng, (p, *kv_shape)) for p, _ in self.SEQS]
        prefix_v = [_random(rng, (p, *kv_shape)) for p, _ in self.SEQS]
        queries = _random(rng, (self.MAX_NUM_TOKENS, geo.num_q_heads, HEAD_DIM))
        return prefix_k, prefix_v, queries

    def _run_shard(self, geo: Geometry, rank: int, window, blocks):
        prefix_k, prefix_v, queries = self._data(geo)
        prefix_lens = [p for p, _ in self.SEQS]
        positions = [_local_positions(p, rank, geo.cp) for p in prefix_lens]
        shard = _paged_shard(
            [k[p] for k, p in zip(prefix_k, positions)],
            [v[p] for v, p in zip(prefix_v, positions)],
            _pages_per_seq(prefix_lens, geo.cp),
            np.random.default_rng(rank),
            geo.num_kv_heads,
        )
        cu_q_lens = np.concatenate([[0], np.cumsum([c for _, c in self.SEQS])])
        unused_kv = np.zeros((self.MAX_NUM_TOKENS, geo.num_kv_heads, HEAD_DIM))
        out, _, lse = _run_kernel(
            queries,
            unused_kv,
            unused_kv,
            shard,
            cu_q_lens,
            [0, 0, len(self.SEQS)],
            sliding_window=window,
            prefill_block_sizes=blocks,
            **_cp_kwargs(geo, rank, [p + c for p, c in self.SEQS]),
        )
        return out[: cu_q_lens[-1]], lse[: cu_q_lens[-1]]

    def _reference(self, geo: Geometry, window):
        prefix_k, prefix_v, queries = self._data(geo)
        outs, lses = [], []
        start = 0
        for seq, (prefix, chunk) in enumerate(self.SEQS):
            out, lse = _dense_attention(
                queries[start : start + chunk],
                prefix_k[seq],
                prefix_v[seq],
                prefix + np.arange(chunk),
                window,
            )
            outs.append(out)
            lses.append(lse)
            start += chunk
        return np.concatenate(outs), np.concatenate(lses)

    def test_merged_prefix_pass_matches_the_dense_prefix_reference(self):
        for geo in (DCP2, DCP8):
            for window in (None, SLIDING_WINDOW):
                for blocks in (None, _small_prefill_blocks()):
                    label = f"{geo.name} window={window} blocks={blocks}"
                    with self.subTest(label):
                        partials = [
                            self._run_shard(geo, rank, window, blocks)
                            for rank in range(geo.cp)
                        ]
                        out, lse = _merge_by_lse(partials)
                        ref_out, ref_lse = self._reference(geo, window)
                        self.assert_out_close(out, ref_out, label)
                        self.assert_lse_close(lse, ref_lse, label)


@unittest.skipIf(not _has_jax_tpu(), _SKIP_REASON)
class TestBatchedRpaDcpWindowStart(_CloseAssertions):
    """The schedule's global-window start on a rank > 0, with one-page k blocks.

    The first local key j in the window of query q (j * cp + rank > q - window)
    lands mid-block, exactly on a block boundary, or past the end of a short
    shard. Each rank's partial output and LSE must equal dense attention over
    that rank's own keys at their global positions.
    """

    # Decode global lengths per geometry and the checked ranks; their first
    # window keys fall exactly on a block boundary and inside a block.
    DECODE = {DCP2: ((1152, 2600), (1,)), DCP8: ((2172, 2600), (3, 5))}
    # One extend: the queries of later q blocks are past the window of the
    # whole 37-token prefix.
    PREFIX, CHUNK = 37, 700

    def _check_rank(self, geo, rank, partial, seq_k, seq_v, q, q_positions, label):
        positions = _local_positions(len(seq_k), rank, geo.cp)
        want_out, want_lse = _dense_attention(
            q,
            seq_k[positions],
            seq_v[positions],
            q_positions,
            SLIDING_WINDOW,
            key_positions=positions,
        )
        self.assert_out_close(partial[0], want_out, label)
        self.assert_lse_close(partial[1], want_lse, label)

    def test_decode_window_start_on_and_off_block_boundaries(self):
        for geo, (seq_lens, ranks) in self.DECODE.items():
            starts = [
                _first_window_key(n - 1, r, geo.cp) for r in ranks for n in seq_lens
            ]
            # The cases cover a start exactly on a block boundary and one inside.
            self.assertIn(0, [j % PAGE_SIZE for j in starts if j > 0])
            self.assertTrue(any(j % PAGE_SIZE for j in starts))

            rng = np.random.default_rng(31 + geo.cp)
            kv_shape = (geo.num_kv_heads, HEAD_DIM)
            full_k = [_random(rng, (n, *kv_shape)) for n in seq_lens]
            full_v = [_random(rng, (n, *kv_shape)) for n in seq_lens]
            num_seqs = len(seq_lens)
            queries = _random(rng, (8, geo.num_q_heads, HEAD_DIM))
            unused_kv = np.zeros((8, geo.num_kv_heads, HEAD_DIM))
            partials = []
            for rank in range(geo.cp):
                positions = [_local_positions(n, rank, geo.cp) for n in seq_lens]
                shard = _paged_shard(
                    [k[p] for k, p in zip(full_k, positions)],
                    [v[p] for v, p in zip(full_v, positions)],
                    _pages_per_seq(seq_lens, geo.cp),
                    np.random.default_rng(rank),
                    geo.num_kv_heads,
                )
                out, _, lse = _run_kernel(
                    queries,
                    unused_kv,
                    unused_kv,
                    shard,
                    np.arange(num_seqs + 1),
                    [num_seqs] * 3,
                    sliding_window=SLIDING_WINDOW,
                    decode_block_sizes=_small_decode_blocks(),
                    **_cp_kwargs(geo, rank, seq_lens),
                )
                partials.append((out[:num_seqs], lse[:num_seqs]))
            for rank in ranks:
                for seq, length in enumerate(seq_lens):
                    label = f"{geo.name} rank {rank} length {length}"
                    with self.subTest(label):
                        self._check_rank(
                            geo,
                            rank,
                            (
                                partials[rank][0][seq : seq + 1],
                                partials[rank][1][seq : seq + 1],
                            ),
                            full_k[seq],
                            full_v[seq],
                            queries[seq : seq + 1],
                            np.asarray([length - 1]),
                            label,
                        )
            with self.subTest(f"{geo.name} merged"):
                out, lse = _merge_by_lse(partials)
                for seq, length in enumerate(seq_lens):
                    want_out, want_lse = _dense_attention(
                        queries[seq : seq + 1],
                        full_k[seq],
                        full_v[seq],
                        np.asarray([length - 1]),
                        SLIDING_WINDOW,
                    )
                    self.assert_out_close(out[seq : seq + 1], want_out, geo.name)
                    self.assert_lse_close(lse[seq : seq + 1], want_lse, geo.name)

    def test_extend_window_start_past_a_short_shard(self):
        blocks = _small_prefill_blocks()
        for geo, rank in ((DCP2, 1), (DCP8, 5)):
            local_len = len(_local_positions(self.PREFIX, rank, geo.cp))
            first_keys = [
                _first_window_key(self.PREFIX + q_start, rank, geo.cp)
                for q_start in range(0, self.CHUNK, blocks.bq_sz)
            ]
            # Some q block starts its window past the last key of the shard.
            self.assertTrue(any(j >= local_len for j in first_keys))
            if geo is DCP2:
                # ... and past the shard's last k block, so it runs a guard block.
                num_k = -(-local_len // blocks.bkv_sz)
                self.assertTrue(any(j // blocks.bkv_sz >= num_k for j in first_keys))

            rng = np.random.default_rng(77 + geo.cp)
            kv_shape = (geo.num_kv_heads, HEAD_DIM)
            prefix_k = _random(rng, (self.PREFIX, *kv_shape))
            prefix_v = _random(rng, (self.PREFIX, *kv_shape))
            max_tokens = -(-self.CHUNK // 8) * 8
            queries = _random(rng, (max_tokens, geo.num_q_heads, HEAD_DIM))
            unused_kv = np.zeros((max_tokens, geo.num_kv_heads, HEAD_DIM))
            q_positions = self.PREFIX + np.arange(self.CHUNK)
            partials = []
            for r in range(geo.cp):
                positions = _local_positions(self.PREFIX, r, geo.cp)
                shard = _paged_shard(
                    [prefix_k[positions]],
                    [prefix_v[positions]],
                    1,
                    np.random.default_rng(r),
                    geo.num_kv_heads,
                )
                out, _, lse = _run_kernel(
                    queries,
                    unused_kv,
                    unused_kv,
                    shard,
                    [0, self.CHUNK],
                    [0, 0, 1],
                    sliding_window=SLIDING_WINDOW,
                    prefill_block_sizes=blocks,
                    **_cp_kwargs(geo, r, [self.PREFIX + self.CHUNK]),
                )
                partials.append((out[: self.CHUNK], lse[: self.CHUNK]))
            label = f"{geo.name} rank {rank}"
            with self.subTest(label):
                self._check_rank(
                    geo,
                    rank,
                    partials[rank],
                    prefix_k,
                    prefix_v,
                    queries[: self.CHUNK],
                    q_positions,
                    label,
                )
            with self.subTest(f"{geo.name} merged"):
                out, lse = _merge_by_lse(partials)
                want_out, want_lse = _dense_attention(
                    queries[: self.CHUNK],
                    prefix_k,
                    prefix_v,
                    q_positions,
                    SLIDING_WINDOW,
                )
                self.assert_out_close(out, want_out, geo.name)
                self.assert_lse_close(lse, want_lse, geo.name)


@unittest.skipIf(not _has_jax_tpu(), _SKIP_REASON)
class TestBatchedRpaTwoPassExtend(_CloseAssertions):
    """The extend path's prefix and chunk passes, merged by LSE.

    The prefix pass reads the cached prefix with ``cache_only``; the chunk pass
    has kv_len == q_len and a zero page table, so every key comes from the k/v
    operands. Both return their LSE and write nothing. Four query heads per KV
    head, as one rank's own heads in the gpt-oss DCP2 chunk pass, and the
    sequence metadata padded past the active sequences as the backend pads it.
    """

    NUM_Q_HEADS = 8
    NUM_KV_HEADS = 2
    SEQS = ((192, 8), (300, 37))
    NUM_PADDED_SEQS = 2
    MAX_NUM_TOKENS = 48

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(2026)
        kv_shape = (cls.NUM_KV_HEADS, HEAD_DIM)
        cls.prefix_k = [_random(rng, (p, *kv_shape)) for p, _ in cls.SEQS]
        cls.prefix_v = [_random(rng, (p, *kv_shape)) for p, _ in cls.SEQS]
        cls.chunk_k = _random(rng, (cls.MAX_NUM_TOKENS, *kv_shape))
        cls.chunk_v = _random(rng, (cls.MAX_NUM_TOKENS, *kv_shape))
        cls.queries = _random(rng, (cls.MAX_NUM_TOKENS, cls.NUM_Q_HEADS, HEAD_DIM))
        cls.sinks = _random(rng, (cls.NUM_Q_HEADS,))
        cu_q_lens = np.cumsum([0] + [c for _, c in cls.SEQS])
        cls.cu_q_lens = np.pad(cu_q_lens, (0, cls.NUM_PADDED_SEQS), mode="edge")
        cls.no_keys = [np.zeros((0, *kv_shape), np.float32)] * cls.NUM_PADDED_SEQS

    def _prefix_pass(self, window):
        prefix_lens = [p for p, _ in self.SEQS]
        shard = _paged_shard(
            self.prefix_k + self.no_keys,
            self.prefix_v + self.no_keys,
            _pages_per_seq(prefix_lens, 1),
            np.random.default_rng(0),
            self.NUM_KV_HEADS,
        )
        out, _, lse = _run_kernel(
            self.queries,
            np.zeros_like(self.chunk_k),
            np.zeros_like(self.chunk_v),
            shard,
            self.cu_q_lens,
            [0, 0, len(self.SEQS)],
            sliding_window=window,
            cache_only=True,
            skip_kv_write=True,
            return_lse=True,
        )
        return out[: self.cu_q_lens[-1]], lse[: self.cu_q_lens[-1]]

    def _chunk_pass(self, window, sinks=None):
        import jax.numpy as jnp

        no_keys = self.no_keys[:1] * (len(self.SEQS) + self.NUM_PADDED_SEQS)
        shard = _paged_shard(
            no_keys, no_keys, 1, np.random.default_rng(1), self.NUM_KV_HEADS
        )
        # The backend's chunk-pass call: kv_len == q_len, an all-zero page table.
        shard = shard._replace(
            page_indices=np.zeros_like(shard.page_indices),
            kv_lens=list(np.diff(self.cu_q_lens)),
        )
        kwargs = {}
        if sinks is not None:
            kwargs["attention_sink"] = jnp.asarray(sinks, jnp.float32)
        out, _, lse = _run_kernel(
            self.queries,
            self.chunk_k,
            self.chunk_v,
            shard,
            self.cu_q_lens,
            [0, 0, len(self.SEQS)],
            sliding_window=window,
            skip_kv_write=True,
            return_lse=True,
            **kwargs,
        )
        return out[: self.cu_q_lens[-1]], lse[: self.cu_q_lens[-1]]

    def _reference(self, window, *, with_prefix: bool):
        outs, lses = [], []
        for seq, (prefix, chunk) in enumerate(self.SEQS):
            start = self.cu_q_lens[seq]
            chunk_k = self.chunk_k[start : start + chunk]
            chunk_v = self.chunk_v[start : start + chunk]
            if with_prefix:
                keys = np.concatenate([self.prefix_k[seq], chunk_k])
                values = np.concatenate([self.prefix_v[seq], chunk_v])
                positions = prefix + np.arange(chunk)
            else:
                keys, values, positions = chunk_k, chunk_v, np.arange(chunk)
            out, lse = _dense_attention(
                self.queries[start : start + chunk],
                keys,
                values,
                positions,
                window,
                causal=True,
            )
            outs.append(out)
            lses.append(lse)
        return np.concatenate(outs), np.concatenate(lses)

    def test_prefix_pass_reports_the_bare_lse(self):
        out, lse = self._prefix_pass(None)
        want_out, want_lse = [], []
        for seq, (prefix, chunk) in enumerate(self.SEQS):
            start = self.cu_q_lens[seq]
            seq_out, seq_lse = _dense_attention(
                self.queries[start : start + chunk],
                self.prefix_k[seq],
                self.prefix_v[seq],
                np.full(chunk, prefix),
                None,
            )
            want_out.append(seq_out)
            want_lse.append(seq_lse)
        self.assert_out_close(out, np.concatenate(want_out), "prefix pass")
        self.assert_lse_close(lse, np.concatenate(want_lse), "prefix pass")

    def test_two_pass_merge_reproduces_the_full_causal_reference(self):
        for window in (None, SLIDING_WINDOW):
            with self.subTest(window=window):
                out, lse = _merge_by_lse(
                    [self._prefix_pass(window), self._chunk_pass(window)]
                )
                ref_out, ref_lse = self._reference(window, with_prefix=True)
                self.assert_out_close(out, ref_out, f"window={window}")
                self.assert_lse_close(lse, ref_lse, f"window={window}")

    def test_chunk_pass_output_and_lse_both_carry_the_sink(self):
        out, lse = self._chunk_pass(None, self.sinks)
        ref_out, ref_lse = self._reference(None, with_prefix=False)
        self.assert_out_close(out, _fold_sink(ref_out, ref_lse, self.sinks), "out")
        # log(e^sink + sum e^s): the sink enters m/l, so it is inside the LSE.
        want_lse = np.logaddexp(self.sinks[None, :], ref_lse)
        self.assert_lse_close(lse, want_lse, "lse")
        # A sink-free LSE would be the reference one; the two must differ or the
        # assertion above proves nothing.
        self.assertGreater(float(np.abs(want_lse - ref_lse).max()), TOLERANCE)

    def test_two_pass_merge_reproduces_the_full_causal_sink_reference(self):
        for window in (None, SLIDING_WINDOW):
            with self.subTest(window=window):
                out, lse = _merge_by_lse(
                    [self._prefix_pass(window), self._chunk_pass(window, self.sinks)]
                )
                ref_out, ref_lse = self._reference(window, with_prefix=True)
                self.assert_out_close(
                    out, _fold_sink(ref_out, ref_lse, self.sinks), f"window={window}"
                )
                self.assert_lse_close(
                    lse,
                    np.logaddexp(self.sinks[None, :], ref_lse),
                    f"window={window}",
                )


@unittest.skipIf(not _has_jax_tpu(), _SKIP_REASON)
class TestBatchedRpaDcpServingShapesFitVmem(unittest.TestCase):
    """The DCP passes compile within VMEM at gpt-oss serving shapes.

    A tile picked by the block-size estimate for the DCP8 chunk pass (8 q heads
    over 8 KV heads with sinks, LSE and the window) overflowed VMEM at warmup.
    Shapes: 16384 tokens, 64 sequences of 17 pages (32K context per rank at
    DCP8); one call compiles both the decode and the MIXED kernel.
    """

    NUM_TOKENS = 16384
    NUM_SEQS = 64
    PAGES_PER_SEQ = 17

    def _compile(self, num_q_heads, num_kv_heads, window, sinks, **kwargs):
        import jax
        import jax.numpy as jnp

        from sglang_tpu.attention.pallas.batched_rpa.wrapper import (
            get_kv_cache_shape,
            ragged_paged_attention,
        )

        num_pages = self.NUM_SEQS * self.PAGES_PER_SEQ
        q = jax.ShapeDtypeStruct((self.NUM_TOKENS, num_q_heads, HEAD_DIM), jnp.bfloat16)
        kv = jax.ShapeDtypeStruct(
            (self.NUM_TOKENS, num_kv_heads, HEAD_DIM), jnp.bfloat16
        )
        cache_shape = get_kv_cache_shape(
            num_pages, PAGE_SIZE, num_kv_heads, HEAD_DIM, jnp.bfloat16
        )
        if "cp_group_size" in kwargs:
            kwargs["cp_rank"] = jax.ShapeDtypeStruct((1,), jnp.int32)
            kwargs["kv_lens_global"] = jax.ShapeDtypeStruct((self.NUM_SEQS,), jnp.int32)
        if sinks:
            kwargs["attention_sink"] = jax.ShapeDtypeStruct((num_q_heads,), jnp.float32)
        ragged_paged_attention.lower(
            q,
            kv,
            kv,
            jax.ShapeDtypeStruct(cache_shape, jnp.bfloat16),
            jax.ShapeDtypeStruct((self.NUM_SEQS,), jnp.int32),
            jax.ShapeDtypeStruct((num_pages,), jnp.int32),
            jax.ShapeDtypeStruct((self.NUM_SEQS + 1,), jnp.int32),
            jax.ShapeDtypeStruct((3,), jnp.int32),
            sm_scale=SM_SCALE,
            sliding_window=window,
            skip_kv_write=True,
            return_lse=True,
            **kwargs,
        ).compile()

    def test_chunk_and_prefix_passes_compile(self):
        # (name, q heads, KV heads, sinks, extra flags): the chunk pass attends
        # one rank's own heads; the prefix pass the group's gathered heads.
        passes = (
            ("dcp8 chunk", 8, 8, True, {}),
            ("dcp8 prefix", 64, 8, False, dict(cache_only=True, cp_group_size=8)),
            ("dcp2 chunk", 8, 2, True, {}),
            ("dcp2 prefix", 16, 2, False, dict(cache_only=True, cp_group_size=2)),
        )
        for name, num_q_heads, num_kv_heads, sinks, flags in passes:
            for window in (SLIDING_WINDOW, None):
                with self.subTest(f"{name} window={window}"):
                    self._compile(num_q_heads, num_kv_heads, window, sinks, **flags)


@unittest.skipIf(not _has_jax_tpu(), _SKIP_REASON)
class TestBatchedRpaFlagsOff(_CloseAssertions):
    """skip_kv_write on the plain path, and every new flag at its default
    against the v3 reference -- the arm TP and CI run through the shared
    schedule, mask and output code the context-parallel modes touch."""

    NUM_Q_HEADS = 16
    PAGES_PER_SEQ = 9

    def _inputs(self, seqs, seed, num_kv_heads):
        """Kernel and reference inputs for ``seqs`` = [(kv_len, q_len), ...].

        ``kv_len`` counts the ``q_len`` new tokens the kernel appends, so the
        reference cache carries them while the kernel's cache is pre-write.
        """
        rng = np.random.default_rng(seed)
        num_seqs = len(seqs)
        total_q = sum(q for _, q in seqs)
        max_num_tokens = max(8, -(-total_q // 8) * 8)
        kv_shape = (num_kv_heads, HEAD_DIM)
        cached = [_random(rng, (kv - q, *kv_shape)) for kv, q in seqs]
        cached_v = [_random(rng, (kv - q, *kv_shape)) for kv, q in seqs]
        new_k = _random(rng, (max_num_tokens, *kv_shape))
        new_v = _random(rng, (max_num_tokens, *kv_shape))
        queries = _random(rng, (max_num_tokens, self.NUM_Q_HEADS, HEAD_DIM))
        cu_q_lens = np.concatenate([[0], np.cumsum([q for _, q in seqs])])
        full_k = [
            np.concatenate([cached[s], new_k[cu_q_lens[s] : cu_q_lens[s + 1]]])
            for s in range(num_seqs)
        ]
        full_v = [
            np.concatenate([cached_v[s], new_v[cu_q_lens[s] : cu_q_lens[s + 1]]])
            for s in range(num_seqs)
        ]
        shard = _paged_shard(
            cached,
            cached_v,
            self.PAGES_PER_SEQ,
            np.random.default_rng(seed),
            num_kv_heads,
        )
        shard = shard._replace(kv_lens=[kv for kv, _ in seqs])
        return queries, new_k, new_v, shard, cu_q_lens, full_k, full_v

    def _compare(self, seqs, seed, num_kv_heads=2, sliding_window=None, soft_cap=None):
        queries, new_k, new_v, shard, cu_q_lens, full_k, full_v = self._inputs(
            seqs, seed, num_kv_heads
        )
        is_decode = all(q == 1 for _, q in seqs)
        num_seqs = len(seqs)
        distribution = [num_seqs] * 3 if is_decode else [0, 0, num_seqs]
        out, _ = _run_kernel(
            queries,
            new_k,
            new_v,
            shard,
            cu_q_lens,
            distribution,
            sliding_window=sliding_window,
            soft_cap=soft_cap,
        )
        for seq in range(num_seqs):
            rows = slice(cu_q_lens[seq], cu_q_lens[seq + 1])
            want = self._v3_reference(
                queries[rows],
                full_k[seq],
                full_v[seq],
                sliding_window=sliding_window,
                soft_cap=soft_cap,
            )
            self.assert_out_close(out[rows], want, f"seq {seq}")

    def _v3_reference(self, queries, keys, values, **kwargs):
        """The in-repo v3 reference on one sequence held in contiguous pages."""
        import jax.numpy as jnp

        from sglang_tpu.attention.pallas.ragged_paged_attention_v3 import (
            ref_ragged_paged_attention,
        )

        kv_len, num_kv_heads, _ = keys.shape
        pages = -(-kv_len // PAGE_SIZE)
        padded = pages * PAGE_SIZE

        def paged(array):
            array = np.pad(array, ((0, padded - kv_len), (0, 0), (0, 0)))
            return _to_bf16(array.reshape(pages, PAGE_SIZE, num_kv_heads, HEAD_DIM))

        out = ref_ragged_paged_attention(
            _to_bf16(queries),
            paged(keys),
            paged(values),
            jnp.asarray([kv_len], jnp.int32),
            jnp.arange(pages, dtype=jnp.int32).reshape(1, pages),
            jnp.asarray([0, len(queries)], jnp.int32),
            jnp.asarray([1], jnp.int32),
            sm_scale=SM_SCALE,
            **kwargs,
        )
        return np.asarray(out.astype(jnp.float32))

    def test_decode(self):
        self._compare([(37, 1), (256, 1), (1, 1), (2049, 1)], seed=1)

    def test_extend(self):
        self._compare([(200, 8), (64, 5), (600, 300)], seed=2)

    def test_multi_head_grouping(self):
        # One KV head for all 16 query heads, and eight KV heads (2 each).
        self._compare([(129, 4)], seed=3, num_kv_heads=1)
        self._compare([(129, 4)], seed=3, num_kv_heads=8)

    def test_sliding_window(self):
        self._compare([(700, 300), (130, 1)], seed=4, sliding_window=SLIDING_WINDOW)

    def test_soft_cap(self):
        self._compare([(200, 8)], seed=5, soft_cap=30.0)

    def test_skip_kv_write_leaves_the_cache_unchanged(self):
        seqs = [(200, 8)]

        def call(**kwargs):
            queries, new_k, new_v, shard, cu_q_lens, _, _ = self._inputs(
                seqs, seed=7, num_kv_heads=2
            )
            original = np.asarray(shard.cache.astype(np.float32))
            out, cache = _run_kernel(
                queries, new_k, new_v, shard, cu_q_lens, [0, 0, 1], **kwargs
            )
            return out, cache, original

        written_out, written_cache, original = call()
        skipped_out, skipped_cache, _ = call(skip_kv_write=True)
        np.testing.assert_array_equal(skipped_cache, original)
        self.assertFalse(np.array_equal(written_cache, original))
        # Fetch semantics are unchanged: the tail still comes from keys/values.
        np.testing.assert_array_equal(written_out, skipped_out)


if __name__ == "__main__":
    unittest.main()
