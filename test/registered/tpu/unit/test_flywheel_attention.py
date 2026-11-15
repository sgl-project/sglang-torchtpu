"""Correctness of --attention-backend flywheel's kernel call on the fused KV pool.

Runs ``fused_cache_flywheel_attention`` on the buffer layout
``TpuFusedMHATokenToKVPool`` allocates, with steps planned by
``plan_flywheel_step`` the way the backend plans them (padded batch rows,
page 0 reserved), and checks every live output row and the whole cache
against numpy. The cache starts filled with random values, which serve as the
cached prefix and catch a write to the wrong slot.
"""

import functools
import os
import sys
import unittest

import numpy as np

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=180, suite="stage-a-test-tpu-unit", chips=1)

# A raw-JAX kernel test, not a TorchTPU runtime process: select the physical
# backend before the first JAX import.
assert "jax" not in sys.modules
os.environ["JAX_PLATFORMS"] = "tpu"


def _has_jax_tpu() -> bool:
    try:
        import jax

        return any("TPU" in d.device_kind for d in jax.devices())
    except Exception:
        return False


PAGE_SIZE = 128
PAGES_PER_SEQ = 5
NUM_BLOCKS = 32
NUM_Q_HEADS = 8
NUM_KV_HEADS = 2
TOLERANCE = 3e-2
# Per-tensor fp8 scales that keep the standard-normal test K/V in e4m3 range.
K_SCALE = 0.05
V_SCALE = 0.04


@unittest.skipIf(not _has_jax_tpu(), "Requires JAX with a TPU device")
class TestFlywheelFusedCacheAttention(unittest.TestCase):
    def _run_step(
        self,
        *,
        is_decode,
        q_lens,
        kv_lens,
        num_tokens,
        bs,
        head_dim=128,
        fp8=False,
        num_kv_heads=NUM_KV_HEADS,
    ):
        """Plan and run one step; return (kernel name, output, cache before/after)."""
        import jax
        import jax.numpy as jnp

        from sglang_tpu.attention import flywheel_attention as fw
        from sglang_tpu.attention_flywheel import (
            fused_cache_flywheel_attention,
            plan_flywheel_step,
        )

        rng = np.random.default_rng(len(q_lens) * 1000 + sum(kv_lens))
        live = len(q_lens)
        q_lens = np.asarray(q_lens, np.int32)
        kv_lens = np.asarray(kv_lens, np.int32)
        # Distinct pages per request, page 0 (the pool's padding page) unused.
        pages = rng.permutation(np.arange(1, NUM_BLOCKS))[: live * PAGES_PER_SEQ]
        block_table = np.zeros((bs, PAGES_PER_SEQ), np.int32)
        block_table[:live] = pages.reshape(live, PAGES_PER_SEQ)

        q = rng.standard_normal((num_tokens, NUM_Q_HEADS, head_dim), np.float32)
        k = rng.standard_normal((num_tokens, num_kv_heads, head_dim), np.float32)
        v = rng.standard_normal((num_tokens, num_kv_heads, head_dim), np.float32)
        q = q.astype(jnp.bfloat16)
        if fp8:
            # The pool's fp8 buffer, [blocks, page, ceil(2H / 4), 4, head_dim],
            # rows past 2H padding; the backend's caller quantizes K/V before
            # the kernel call.
            cache = (
                rng.standard_normal(
                    (NUM_BLOCKS, PAGE_SIZE, -(-num_kv_heads * 2 // 4), 4, head_dim),
                    np.float32,
                )
                / K_SCALE
            ).astype(jnp.float8_e4m3fn)
            k = (k / K_SCALE).astype(jnp.float8_e4m3fn)
            v = (v / V_SCALE).astype(jnp.float8_e4m3fn)
            scales = {"k_scale": K_SCALE, "v_scale": V_SCALE}
        else:
            cache = rng.standard_normal(
                (NUM_BLOCKS, PAGE_SIZE, num_kv_heads, 2, head_dim), np.float32
            ).astype(jnp.bfloat16)
            k, v = (x.astype(jnp.bfloat16) for x in (k, v))
            scales = {}

        slices, num_slices, kernel_index = plan_flywheel_step(
            is_decode=is_decode,
            q_lens=q_lens,
            kv_lens=kv_lens,
            block_table=block_table[:live],
            num_tokens=num_tokens,
            max_num_seqs=bs,
            page_size=PAGE_SIZE,
        )
        padded_q_lens = np.zeros(bs, np.int32)
        padded_q_lens[:live] = q_lens
        cu_q_lens = np.concatenate([[0], np.cumsum(padded_q_lens)]).astype(np.int32)
        padded_kv_lens = np.zeros(bs, np.int32)
        padded_kv_lens[:live] = kv_lens
        # RPA v3's distribution: decode rows first.
        distribution = np.array(
            [live, live, live] if is_decode else [0, live, live], np.int32
        )
        sm_scale = head_dim**-0.5
        run = jax.jit(
            functools.partial(
                fused_cache_flywheel_attention, sm_scale=sm_scale, **scales
            )
        )
        out, new_cache = run(
            jnp.asarray(q),
            jnp.asarray(k),
            jnp.asarray(v),
            jnp.asarray(cache),
            jnp.asarray(padded_kv_lens),
            jnp.asarray(block_table.reshape(-1)),
            jnp.asarray(cu_q_lens),
            jnp.asarray(distribution),
            jnp.asarray(slices.reshape(-1)),
            jnp.asarray([num_slices], np.int32),
            jnp.asarray([kernel_index], np.int32),
        )
        kernel = fw.step_kernels(num_tokens)[kernel_index]
        expected_out, expected_cache = _reference(
            q=np.asarray(q, np.float32),
            k=np.asarray(k, np.float32),
            v=np.asarray(v, np.float32),
            cache=np.asarray(cache, np.float32),
            q_lens=q_lens,
            kv_lens=kv_lens,
            block_table=block_table,
            sm_scale=sm_scale,
            fp8=fp8,
            num_kv_heads=num_kv_heads,
        )
        real_rows = int(q_lens.sum())
        np.testing.assert_allclose(
            np.asarray(out, np.float32)[:real_rows],
            expected_out[:real_rows],
            atol=TOLERANCE,
            rtol=TOLERANCE,
        )
        # Only the K/V rows: the writer may fill an fp8 lane group's padding.
        for got, want in zip(
            _kv_views(np.asarray(new_cache, np.float32), num_kv_heads),
            _kv_views(expected_cache, num_kv_heads),
        ):
            np.testing.assert_array_equal(got, want)
        return kernel

    def test_decode_with_padding_rows(self):
        kernel = self._run_step(
            is_decode=True,
            q_lens=[1, 1, 1, 1, 1],
            kv_lens=[1, 129, 300, 128, 640],
            num_tokens=8,
            bs=8,
        )
        self.assertEqual(kernel, "decode")

    def test_dense_prefill_one_request(self):
        kernel = self._run_step(
            is_decode=False, q_lens=[256], kv_lens=[256], num_tokens=256, bs=4
        )
        self.assertEqual(kernel, "dense")

    def test_ragged_prefill(self):
        kernel = self._run_step(
            is_decode=False,
            q_lens=[100, 300, 37],
            kv_lens=[100, 300, 37],
            num_tokens=512,
            bs=4,
        )
        self.assertIsInstance(kernel, int)

    def test_extend_onto_cached_prefix(self):
        kernel = self._run_step(
            is_decode=False,
            q_lens=[50, 3, 128],
            kv_lens=[300, 131, 384],
            num_tokens=256,
            bs=4,
        )
        self.assertEqual(kernel, "extend")

    def test_head_dim_256(self):
        self._run_step(
            is_decode=False,
            q_lens=[70, 1],
            kv_lens=[200, 129],
            num_tokens=128,
            bs=4,
            head_dim=256,
        )
        self._run_step(
            is_decode=True,
            q_lens=[1, 1],
            kv_lens=[200, 129],
            num_tokens=4,
            bs=4,
            head_dim=256,
        )

    def test_fp8_decode(self):
        self._run_step(
            is_decode=True,
            q_lens=[1, 1, 1],
            kv_lens=[1, 300, 640],
            num_tokens=4,
            bs=4,
            fp8=True,
        )

    def test_fp8_prefill_and_extend(self):
        self._run_step(
            is_decode=False,
            q_lens=[256],
            kv_lens=[256],
            num_tokens=256,
            bs=4,
            fp8=True,
        )
        self._run_step(
            is_decode=False,
            q_lens=[100, 300, 37],
            kv_lens=[100, 300, 37],
            num_tokens=512,
            bs=4,
            fp8=True,
        )
        kernel = self._run_step(
            is_decode=False,
            q_lens=[50, 3, 128],
            kv_lens=[300, 131, 384],
            num_tokens=256,
            bs=4,
            fp8=True,
        )
        self.assertEqual(kernel, "extend")

    def test_fp8_one_kv_head(self):
        # One KV head per rank pads the fp8 buffer's lane group with two rows.
        self._run_step(
            is_decode=True,
            q_lens=[1, 1, 1],
            kv_lens=[1, 300, 640],
            num_tokens=4,
            bs=4,
            fp8=True,
            num_kv_heads=1,
        )
        kernel = self._run_step(
            is_decode=False,
            q_lens=[50, 3, 128],
            kv_lens=[300, 131, 384],
            num_tokens=256,
            bs=4,
            fp8=True,
            num_kv_heads=1,
        )
        self.assertEqual(kernel, "extend")


@unittest.skipIf(not _has_jax_tpu(), "Requires JAX with a TPU device")
class TestFlywheelStepPlan(unittest.TestCase):
    def test_write_pieces_cover_exactly_the_new_rows(self):
        from sglang_tpu.attention_flywheel import plan_flywheel_step

        rng = np.random.default_rng(0)
        for _ in range(200):
            live = int(rng.integers(1, 6))
            kv_lens = rng.integers(1, PAGES_PER_SEQ * PAGE_SIZE, live).astype(np.int32)
            q_lens = np.minimum(rng.integers(1, 300, live), kv_lens).astype(np.int32)
            block_table = rng.integers(1, NUM_BLOCKS, (live, PAGES_PER_SEQ)).astype(
                np.int32
            )
            num_tokens = int(1 << int(np.ceil(np.log2(max(64, q_lens.sum())))))
            slices, count, _ = plan_flywheel_step(
                is_decode=False,
                q_lens=q_lens,
                kv_lens=kv_lens,
                block_table=block_table,
                num_tokens=num_tokens,
                max_num_seqs=8,
                page_size=PAGE_SIZE,
            )
            self.assertTrue(np.all(slices[:, count:] == 0))
            written = {}
            for cache_row, new_row, length in slices[:, :count].T:
                self.assertGreater(length, 0)
                # A piece never crosses a page.
                self.assertEqual(
                    cache_row // PAGE_SIZE, (cache_row + length - 1) // PAGE_SIZE
                )
                for j in range(length):
                    written[int(new_row + j)] = int(cache_row + j)
            expected = {}
            row = 0
            for r in range(live):
                for pos in range(kv_lens[r] - q_lens[r], kv_lens[r]):
                    page = block_table[r, pos // PAGE_SIZE]
                    expected[row] = int(page * PAGE_SIZE + pos % PAGE_SIZE)
                    row += 1
            self.assertEqual(written, expected)


def _kv_views(cache, num_kv_heads):
    """The pool buffer's K and V as [blocks, page, H, D] views.

    Both dtypes interleave each head's K and V row, [k0, v0, k1, v1, ...]; an
    fp8 buffer only groups them four to a 32-bit lane, padding the last group.
    """
    rows = cache.reshape(cache.shape[:2] + (-1, 2, cache.shape[-1]))
    return rows[:, :, :num_kv_heads, 0], rows[:, :, :num_kv_heads, 1]


def _write_kv(cache, page, slot, k_row, v_row):
    rows = cache[page, slot].reshape(-1, 2, cache.shape[-1])
    rows[: len(k_row), 0] = k_row
    rows[: len(v_row), 1] = v_row
    cache[page, slot] = rows.reshape(cache.shape[2:])


def _reference(
    *,
    q,
    k,
    v,
    cache,
    q_lens,
    kv_lens,
    block_table,
    sm_scale,
    fp8=False,
    num_kv_heads=NUM_KV_HEADS,
):
    """Causal GQA attention over each request's cached prefix plus its new rows,
    and the cache with the new rows written. An fp8 cache holds K/V divided by
    K_SCALE / V_SCALE."""
    cache = cache.copy()
    out = np.zeros_like(q)
    group = NUM_Q_HEADS // num_kv_heads
    row = 0
    for r, (q_len, kv_len) in enumerate(zip(q_lens, kv_lens)):
        prefix = kv_len - q_len
        for i in range(q_len):
            pos = prefix + i
            page, slot = block_table[r, pos // PAGE_SIZE], pos % PAGE_SIZE
            _write_kv(cache, page, slot, k[row + i], v[row + i])
        positions = np.arange(kv_len)
        pages = block_table[r, positions // PAGE_SIZE]
        cache_k, cache_v = _kv_views(cache, num_kv_heads)
        keys = cache_k[pages, positions % PAGE_SIZE]  # [kv_len, H, D]
        values = cache_v[pages, positions % PAGE_SIZE]
        if fp8:
            keys, values = keys * K_SCALE, values * V_SCALE
        for i in range(q_len):
            visible = prefix + i + 1
            for h in range(NUM_Q_HEADS):
                logits = keys[:visible, h // group] @ q[row + i, h] * sm_scale
                p = np.exp(logits - logits.max())
                out[row + i, h] = (p / p.sum()) @ values[:visible, h // group]
        row += q_len
    return out, cache


if __name__ == "__main__":
    unittest.main()
