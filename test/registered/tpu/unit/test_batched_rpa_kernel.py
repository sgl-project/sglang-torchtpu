# SPDX-License-Identifier: Apache-2.0
"""Direct TPU correctness coverage for the batched-RPA kernel."""

from __future__ import annotations

import math
import os
import sys
import unittest
import unittest.mock

import numpy as np

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=480, suite="stage-a-test-tpu-unit", chips=1)

# This is a standalone raw-JAX kernel test, not a TorchTPU runtime process.
# SGLang platform discovery selects CPU-only JAX for export, so select the
# physical backend before the first JAX import.
assert "jax" not in sys.modules
os.environ["JAX_PLATFORMS"] = "tpu"


def _has_jax_tpu() -> bool:
    try:
        import jax

        return any("TPU" in device.device_kind for device in jax.devices())
    except Exception:
        return False


@unittest.skipIf(not _has_jax_tpu(), "Requires JAX with a TPU device")
class TestBatchedRpaKernel(unittest.TestCase):
    """Compare the kernel against pure-JAX attention on the token-major layout.

    16 q heads over 8 kv heads with kv_len 2049 -- a partial last page and an
    idle lane -- plus a check that the new K/V landed in the right cache slot.
    """

    PAGE_SIZE = 256
    PAGES_PER_SEQ = 9
    NUM_Q_HEADS = 16
    NUM_KV_HEADS = 8
    HEAD_DIM = 128
    TOLERANCE = 0.05

    def test_idle_lane_multihead_decode_and_writeback_match_reference(self):
        import jax
        import jax.numpy as jnp

        from sglang_tpu.attention.pallas.batched_rpa.utils import (
            get_dtype_packing,
        )
        from sglang_tpu.attention.pallas.batched_rpa.wrapper import (
            ragged_paged_attention,
        )
        from sglang_tpu.attention.pallas.ragged_paged_attention_v3 import (
            ref_ragged_paged_attention,
        )

        kv_dtype = jnp.bfloat16
        kv_len = 2049
        total_pages = self.PAGES_PER_SEQ
        padded_q_tokens = 8

        key = jax.random.PRNGKey(17)
        q_key, new_k_key, new_v_key, cache_k_key, cache_v_key = jax.random.split(key, 5)
        q = jax.random.normal(
            q_key,
            (padded_q_tokens, self.NUM_Q_HEADS, self.HEAD_DIM),
            dtype=jnp.bfloat16,
        )
        new_k = jax.random.normal(
            new_k_key,
            (padded_q_tokens, self.NUM_KV_HEADS, self.HEAD_DIM),
            dtype=kv_dtype,
        )
        new_v = jax.random.normal(
            new_v_key,
            (padded_q_tokens, self.NUM_KV_HEADS, self.HEAD_DIM),
            dtype=kv_dtype,
        )
        cache_k = jax.random.normal(
            cache_k_key,
            (total_pages, self.PAGE_SIZE, self.NUM_KV_HEADS, self.HEAD_DIM),
            dtype=kv_dtype,
        )
        cache_v = jax.random.normal(
            cache_v_key,
            (total_pages, self.PAGE_SIZE, self.NUM_KV_HEADS, self.HEAD_DIM),
            dtype=kv_dtype,
        )

        page_indices_host = np.arange(total_pages, dtype=np.int32).reshape(1, -1)
        new_token_pos = kv_len - 1
        rel_page, slot = divmod(new_token_pos, self.PAGE_SIZE)
        abs_page = int(page_indices_host[0, rel_page])
        ref_k = cache_k.at[abs_page, slot].set(new_k[0])
        ref_v = cache_v.at[abs_page, slot].set(new_v[0])

        packing = get_dtype_packing(kv_dtype)
        unpacked_kv_heads = self.NUM_KV_HEADS * 2
        aligned_kv_heads = ((unpacked_kv_heads + packing - 1) // packing) * packing
        kv_cache_unpacked = jnp.stack([cache_k, cache_v], axis=3).reshape(
            total_pages,
            self.PAGE_SIZE,
            unpacked_kv_heads,
            self.HEAD_DIM,
        )
        kv_cache_unpacked = jnp.pad(
            kv_cache_unpacked,
            ((0, 0), (0, 0), (0, aligned_kv_heads - unpacked_kv_heads), (0, 0)),
        )
        kv_cache = kv_cache_unpacked.reshape(
            total_pages,
            self.PAGE_SIZE,
            aligned_kv_heads // packing,
            packing,
            self.HEAD_DIM,
        )

        kv_lens = jnp.asarray([kv_len], dtype=jnp.int32)
        cu_q_lens = jnp.asarray([0, 1], dtype=jnp.int32)
        page_indices = jnp.asarray(page_indices_host.reshape(-1))
        distribution = jnp.ones((3,), dtype=jnp.int32)
        sm_scale = 1.0 / math.sqrt(self.HEAD_DIM)

        expected = ref_ragged_paged_attention(
            q[:1],
            ref_k,
            ref_v,
            kv_lens,
            jnp.asarray(page_indices_host),
            cu_q_lens,
            jnp.asarray([1], dtype=jnp.int32),
            sm_scale=sm_scale,
        )
        actual, updated_cache = ragged_paged_attention(
            q,
            new_k,
            new_v,
            kv_cache,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
            sm_scale=sm_scale,
        )

        max_diff = jnp.max(
            jnp.abs(actual[:1].astype(jnp.float32) - expected.astype(jnp.float32))
        )
        self.assertLess(float(max_diff), self.TOLERANCE)

        unpacked_cache = updated_cache.reshape(
            total_pages, self.PAGE_SIZE, aligned_kv_heads, self.HEAD_DIM
        )
        np.testing.assert_array_equal(
            np.asarray(unpacked_cache[abs_page, slot, 0:unpacked_kv_heads:2]),
            np.asarray(new_k[0]),
        )
        np.testing.assert_array_equal(
            np.asarray(unpacked_cache[abs_page, slot, 1:unpacked_kv_heads:2]),
            np.asarray(new_v[0]),
        )


def _round_through(x: np.ndarray, dtype) -> np.ndarray:
    """Round host values through ``dtype``; the f32 result casts back exactly."""
    import jax.numpy as jnp

    return np.array(jnp.asarray(x, dtype=dtype).astype(jnp.float32))


def _kv_cache_to_kernel(fused: np.ndarray, kv_dtype, kv_layout):
    """[pages, page, 2 * kv_heads, head_dim] interleaved K/V as the kernel's cache."""
    import jax.numpy as jnp

    from sglang_tpu.attention.pallas.batched_rpa import configs, wrapper

    pages, page_size, heads_x2, head_dim = fused.shape
    shape = wrapper.get_kv_cache_shape(
        pages, page_size, heads_x2 // 2, head_dim, kv_dtype, kv_layout=kv_layout
    )
    if kv_layout == configs.KVLayout.SEQ_ALONG_LANE:
        dim = shape[2] * shape[3]
        fused = np.pad(fused, ((0, 0), (0, 0), (0, 0), (0, dim - head_dim)))
        fused = fused.transpose(0, 2, 3, 1)
    else:
        fused = np.pad(
            fused,
            (
                (0, 0),
                (0, 0),
                (0, shape[2] * shape[3] - heads_x2),
                (0, shape[4] - head_dim),
            ),
        )
    return jnp.asarray(fused.reshape(shape), dtype=kv_dtype)


def _kv_cache_from_kernel(cache, heads_x2: int, head_dim: int, kv_layout):
    """Inverse of ``_kv_cache_to_kernel``, as f32 on the host."""
    import jax.numpy as jnp

    from sglang_tpu.attention.pallas.batched_rpa import configs

    cache = np.asarray(cache.astype(jnp.float32))
    pages = cache.shape[0]
    if kv_layout == configs.KVLayout.SEQ_ALONG_LANE:
        page_size = cache.shape[4]
        cache = cache.reshape(pages, heads_x2, -1, page_size).transpose(0, 3, 1, 2)
    else:
        cache = cache.reshape(pages, cache.shape[1], -1, cache.shape[4])
    return cache[:, :, :heads_x2, :head_dim]


@unittest.skipIf(not _has_jax_tpu(), "Requires JAX with a TPU device")
class TestBatchedRpaKernelPackedBatches(unittest.TestCase):
    """Compare every sequence of packed batches against ``ref_ragged_paged_attention``.

    Each case also checks the cache after the call: every new token's K/V is in
    its slot and no other live position changed. Sinks span logits -1..9 so
    that some heads change materially.
    """

    PAGE_SIZE = 256
    # Logits here have std 2 and the kernel rounds them to bf16 before the
    # softmax, which alone moves outputs by up to about 0.06.
    TOLERANCE = 0.1
    # Prefix 0, unaligned, and past several default KV blocks; query lengths off
    # every tile size, one spanning two q blocks; tokens padded past the last.
    EXTEND_SEQS = ((0, 300), (437, 77), (3000, 515), (1100, 1500))
    EXTEND_TOKENS = 2560

    def _check(
        self,
        *,
        seqs,
        padded_tokens,
        num_q_heads=8,
        num_kv_heads=1,
        head_dim=64,
        decode=False,
        sliding_window=None,
        use_sink=False,
        kv_dtype_name="bfloat16",
        k_scale=None,
        v_scale=None,
        seq_along_lane=False,
        share_pages=False,
        prefill_block_sizes=None,
        seed=0,
    ):
        """Runs one batch of ``(prefix_len, q_len)`` sequences and checks it.

        ``share_pages`` points every row at one page row whose cache already
        holds each row's new K/V, the layout context-parallel prefill uses.
        """
        import jax.numpy as jnp

        from sglang_tpu.attention.pallas.batched_rpa import configs
        from sglang_tpu.attention.pallas.batched_rpa.wrapper import (
            ragged_paged_attention,
        )
        from sglang_tpu.attention.pallas.ragged_paged_attention_v3 import (
            ref_ragged_paged_attention,
        )

        kv_dtype = jnp.dtype(kv_dtype_name)
        kv_layout = (
            configs.KVLayout.SEQ_ALONG_LANE
            if seq_along_lane
            else configs.KVLayout.HEAD_ALONG_SUBLANE
        )
        page_size = self.PAGE_SIZE
        rng = np.random.default_rng(seed)
        num_seqs = len(seqs)
        # One idle slot past the distribution, as the backend pads the batch.
        max_num_seqs = num_seqs + 1
        prefix_lens = np.asarray([p for p, _ in seqs], dtype=np.int32)
        q_lens = np.asarray([q for _, q in seqs], dtype=np.int32)
        kv_lens = prefix_lens + q_lens
        cu_q_lens = np.concatenate([[0], np.cumsum(q_lens)]).astype(np.int32)
        total_tokens = int(cu_q_lens[-1])
        self.assertLess(total_tokens, padded_tokens)

        pages_per_seq = -(-int(kv_lens.max()) // page_size)
        num_pages = max_num_seqs * pages_per_seq
        page_indices = rng.permutation(num_pages).astype(np.int32)
        page_indices = page_indices.reshape(max_num_seqs, pages_per_seq)
        if share_pages:
            page_indices[:] = page_indices[0]

        def _kv(shape, scale):
            values = rng.standard_normal(shape, dtype=np.float32)
            return _round_through(values / (scale or 1.0), kv_dtype)

        cache_shape = (num_pages, page_size, num_kv_heads, head_dim)
        cache_k = _kv(cache_shape, k_scale)
        cache_v = _kv(cache_shape, v_scale)
        q = _round_through(
            2.0
            * rng.standard_normal((padded_tokens, num_q_heads, head_dim), np.float32),
            jnp.bfloat16,
        )
        new_k = _kv((padded_tokens, num_kv_heads, head_dim), k_scale)
        new_v = _kv((padded_tokens, num_kv_heads, head_dim), v_scale)

        ref_k, ref_v = cache_k.copy(), cache_v.copy()
        # Slots no live position of any sequence covers; the kernel may fill them.
        free_slots = np.zeros((num_pages, page_size), dtype=bool)
        free_slots[page_indices[:num_seqs]] = True
        for i in range(num_seqs):
            positions = np.arange(kv_lens[i])
            pages = page_indices[i, positions // page_size]
            slots = positions % page_size
            free_slots[pages, slots] = False
            new = slice(prefix_lens[i], kv_lens[i])
            rows = slice(cu_q_lens[i], cu_q_lens[i + 1])
            if share_pages:
                new_k[rows] = cache_k[pages[new], slots[new]]
                new_v[rows] = cache_v[pages[new], slots[new]]
            ref_k[pages[new], slots[new]] = new_k[rows]
            ref_v[pages[new], slots[new]] = new_v[rows]

        def _interleave(k, v):
            return np.stack([k, v], axis=3).reshape(*k.shape[:2], -1, head_dim)

        sinks = None
        if use_sink:
            sinks = rng.permutation(np.linspace(-1.0, 9.0, num_q_heads))
            sinks = jnp.asarray(sinks, dtype=jnp.float32)

        kv_lens_padded = np.zeros(max_num_seqs, dtype=np.int32)
        kv_lens_padded[:num_seqs] = kv_lens
        cu_q_lens_padded = np.full(max_num_seqs + 1, total_tokens, dtype=np.int32)
        cu_q_lens_padded[: num_seqs + 1] = cu_q_lens
        if decode:
            distribution = [num_seqs, num_seqs, num_seqs]
        else:
            distribution = [0, 0, num_seqs]
        sm_scale = 1.0 / math.sqrt(head_dim)

        def _reference(attention_sink):
            out = ref_ragged_paged_attention(
                jnp.asarray(q[:total_tokens], dtype=jnp.bfloat16),
                jnp.asarray(ref_k, dtype=kv_dtype),
                jnp.asarray(ref_v, dtype=kv_dtype),
                jnp.asarray(kv_lens),
                jnp.asarray(page_indices[:num_seqs]),
                jnp.asarray(cu_q_lens),
                jnp.asarray([num_seqs], dtype=jnp.int32),
                sm_scale=sm_scale,
                sliding_window=sliding_window,
                k_scale=k_scale,
                v_scale=v_scale,
                attention_sink=attention_sink,
            )
            return np.asarray(out.astype(jnp.float32))

        expected = _reference(sinks)
        sink_effect = 0.0
        if use_sink:
            sink_effect = float(np.max(np.abs(expected - _reference(None))))
            self.assertGreater(sink_effect, 3 * self.TOLERANCE)

        actual, updated_cache = ragged_paged_attention(
            jnp.asarray(q, dtype=jnp.bfloat16),
            jnp.asarray(new_k, dtype=kv_dtype),
            jnp.asarray(new_v, dtype=kv_dtype),
            _kv_cache_to_kernel(_interleave(cache_k, cache_v), kv_dtype, kv_layout),
            jnp.asarray(kv_lens_padded),
            jnp.asarray(page_indices.reshape(-1)),
            jnp.asarray(cu_q_lens_padded),
            jnp.asarray(distribution, dtype=jnp.int32),
            sinks,
            sm_scale=sm_scale,
            sliding_window=sliding_window,
            k_scale=k_scale,
            v_scale=v_scale,
            prefill_block_sizes=prefill_block_sizes,
            kv_layout=kv_layout,
        )
        actual = np.asarray(actual[:total_tokens].astype(jnp.float32))

        diffs = []
        rel_errors = []
        for i in range(num_seqs):
            rows = slice(cu_q_lens[i], cu_q_lens[i + 1])
            error = actual[rows] - expected[rows]
            diffs.append(float(np.max(np.abs(error))))
            rel_errors.append(
                float(np.linalg.norm(error) / np.linalg.norm(expected[rows]))
            )
        print(
            f"{self.id()} max_diff_per_seq={[round(d, 4) for d in diffs]}"
            f" rel_rms_per_seq={[round(r, 4) for r in rel_errors]}"
            f" sink_effect={sink_effect:.3f}"
        )
        for i, diff in enumerate(diffs):
            self.assertLess(diff, self.TOLERANCE, f"sequence {i} {seqs[i]}")

        updated = _kv_cache_from_kernel(
            updated_cache, 2 * num_kv_heads, head_dim, kv_layout
        )
        expected_cache = _interleave(ref_k, ref_v)
        if kv_layout == configs.KVLayout.HEAD_ALONG_SUBLANE:
            # Token-major writeback moves only the new tokens' rows.
            free_slots[:] = False
        np.testing.assert_array_equal(updated[~free_slots], expected_cache[~free_slots])

    def test_packed_extend_every_sequence(self):
        for use_sink in (False, True):
            for sliding_window in (None, 128):
                with self.subTest(use_sink=use_sink, sliding_window=sliding_window):
                    self._check(
                        seqs=self.EXTEND_SEQS,
                        padded_tokens=self.EXTEND_TOKENS,
                        use_sink=use_sink,
                        sliding_window=sliding_window,
                    )

    def test_packed_extend_sink_uses_final_max_across_lanes(self):
        """Small blocks over two lanes: a q block's running max moves between
        k blocks and lanes, and the sink must be scaled by the final one."""
        from sglang_tpu.attention.pallas.batched_rpa.configs import (
            BlockSizes,
        )

        self._check(
            seqs=self.EXTEND_SEQS,
            padded_tokens=self.EXTEND_TOKENS,
            use_sink=True,
            prefill_block_sizes=BlockSizes(
                bq_sz=256, bq_c_sz=128, bkv_sz=512, batch_size=2, n_buffer=3
            ),
        )

    def test_packed_extend_sinks_over_several_kv_heads(self):
        self._check(
            seqs=self.EXTEND_SEQS,
            padded_tokens=self.EXTEND_TOKENS,
            num_q_heads=16,
            num_kv_heads=8,
            head_dim=128,
            use_sink=True,
            sliding_window=128,
        )

    def test_single_kv_head_geometries(self):
        """Per-rank shapes with one KV head and bf16 KV: 4 q heads at head_dim
        128 (a 4-KV-head model at TP8) and 8 q heads at head_dim 64 (gpt-oss)."""
        decode_seqs = tuple((n - 1, 1) for n in (6, 7, 300, 2049))
        cases = (
            ("extend_5", ((0, 5),), 8, False, False),
            ("extend_1000", ((0, 1000),), 1024, False, False),
            ("decode", decode_seqs, 8, True, False),
            # 4 q heads fill only part of a sink row period.
            ("decode_sinks", decode_seqs, 8, True, True),
        )
        for num_q_heads, head_dim in ((4, 128), (8, 64)):
            for name, seqs, padded_tokens, decode, use_sink in cases:
                with self.subTest(q_heads=num_q_heads, head_dim=head_dim, case=name):
                    self._check(
                        seqs=seqs,
                        padded_tokens=padded_tokens,
                        num_q_heads=num_q_heads,
                        num_kv_heads=1,
                        head_dim=head_dim,
                        decode=decode,
                        use_sink=use_sink,
                    )

    def test_decode_batch_with_sinks_and_window(self):
        kv_lens = (1, 2, 127, 128, 129, 255, 256, 257, 511, 600, 1024, 1300, 2049)
        kv_lens += (2500, 3071, 3333)
        for seq_along_lane in (False, True):
            with self.subTest(seq_along_lane=seq_along_lane):
                self._check(
                    seqs=tuple((kv_len - 1, 1) for kv_len in kv_lens),
                    padded_tokens=24,
                    decode=True,
                    use_sink=True,
                    sliding_window=128,
                    seq_along_lane=seq_along_lane,
                )

    def test_decode_sinks_over_several_kv_blocks(self):
        """Eight KV heads shrink the decode k block to one page, so a long
        sequence's running max moves across k blocks before the sink."""
        kv_lens = (1, 255, 257, 600, 1300, 2049, 3333)
        for sliding_window in (None, 128):
            with self.subTest(sliding_window=sliding_window):
                self._check(
                    seqs=tuple((kv_len - 1, 1) for kv_len in kv_lens),
                    padded_tokens=16,
                    num_q_heads=16,
                    num_kv_heads=8,
                    head_dim=128,
                    decode=True,
                    use_sink=True,
                    sliding_window=sliding_window,
                )

    def test_seq_along_lane_packed_extend(self):
        for use_sink, sliding_window in ((False, None), (True, 128)):
            with self.subTest(use_sink=use_sink, sliding_window=sliding_window):
                self._check(
                    seqs=self.EXTEND_SEQS,
                    padded_tokens=self.EXTEND_TOKENS,
                    use_sink=use_sink,
                    sliding_window=sliding_window,
                    seq_along_lane=True,
                )

    def test_context_parallel_rows_share_pages(self):
        """Two rows of one sequence on the same page row; the later row reads the
        earlier row's positions from the cache in the same call."""
        self._check(
            seqs=((300, 400), (1500, 400)),
            padded_tokens=1024,
            use_sink=True,
            share_pages=True,
        )

    def test_fp8_packed_extend_with_sinks(self):
        for sliding_window, seq_along_lane in (
            (None, False),
            (128, False),
            (128, True),
        ):
            with self.subTest(
                sliding_window=sliding_window, seq_along_lane=seq_along_lane
            ):
                self._check(
                    seqs=self.EXTEND_SEQS,
                    padded_tokens=self.EXTEND_TOKENS,
                    kv_dtype_name="float8_e4m3fn",
                    k_scale=0.5,
                    v_scale=0.25,
                    use_sink=True,
                    sliding_window=sliding_window,
                    seq_along_lane=seq_along_lane,
                )

    def test_seq_along_lane_softmax_variants(self):
        """exp2 softmax and ones-in-V row sums match the reference on the fp8
        SEQ_ALONG_LANE prefill path they are meant for, with and without a window."""
        import jax

        from sglang_tpu.envs import envs as tpu_envs

        variants = (
            {"SGLANG_TPU_RPA_EXP2_SOFTMAX": "1"},
            {"SGLANG_TPU_RPA_ONES_V": "1"},
            {"SGLANG_TPU_RPA_EXP2_SOFTMAX": "1", "SGLANG_TPU_RPA_ONES_V": "1"},
            {"SGLANG_TPU_RPA_ONES_V": "1", "SGLANG_TPU_RPA_SKIP_FULL_TILE_MASK": "1"},
        )
        for env in variants:
            for sliding_window in (None, 128):
                with (
                    self.subTest(env=env, sliding_window=sliding_window),
                    unittest.mock.patch.dict(os.environ, env),
                ):
                    # The wrapper reads the flags while tracing; drop cached traces.
                    jax.clear_caches()
                    self.assertTrue(all(getattr(tpu_envs, name).get() for name in env))
                    self._check(
                        seqs=self.EXTEND_SEQS,
                        padded_tokens=self.EXTEND_TOKENS,
                        kv_dtype_name="float8_e4m3fn",
                        k_scale=1.0,
                        v_scale=1.0,
                        sliding_window=sliding_window,
                        seq_along_lane=True,
                    )
        jax.clear_caches()

    def test_seq_along_lane_transposed(self):
        """The transposed prefill tile (scores^T, fp8 q) tracks the reference on
        the fp8 SEQ_ALONG_LANE path, with and without a window and the mask skip."""
        import jax

        from sglang_tpu.envs import envs as tpu_envs

        variants = (
            {"SGLANG_TPU_RPA_TRANSPOSED": "1"},
            {
                "SGLANG_TPU_RPA_TRANSPOSED": "1",
                "SGLANG_TPU_RPA_SKIP_FULL_TILE_MASK": "1",
            },
        )
        # fp8 q: a few percent of relative error on the scores, see transposed.py.
        self.TOLERANCE = 0.3
        for env in variants:
            for sliding_window in (None, 128):
                with (
                    self.subTest(env=env, sliding_window=sliding_window),
                    unittest.mock.patch.dict(os.environ, env),
                ):
                    jax.clear_caches()
                    self.assertTrue(all(getattr(tpu_envs, name).get() for name in env))
                    self._check(
                        seqs=self.EXTEND_SEQS,
                        padded_tokens=self.EXTEND_TOKENS,
                        kv_dtype_name="float8_e4m3fn",
                        k_scale=1.0,
                        v_scale=1.0,
                        sliding_window=sliding_window,
                        seq_along_lane=True,
                    )
        jax.clear_caches()
        del self.TOLERANCE


if __name__ == "__main__":
    unittest.main()
