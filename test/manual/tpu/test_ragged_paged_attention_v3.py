"""Correctness tests for the vendored Pallas ragged-paged-attention v3 kernel.

The kernel and its pure-JAX ``ref_ragged_paged_attention`` reference both
live in ``sglang.srt.layers.attention.pallas.ragged_paged_attention_v3``.
These tests build small inputs in both formats (fused/packed for the
kernel, unpacked pages for the reference) and assert the outputs match
within bfloat16 tolerance.

Cases covered:

* decode-only, single sequence (q_len == 1)
* extend, single sequence (q_len > 1)
* decode-only batch with multiple sequences (exercises the per-seq
  page-indices offset math)
* extend with grouped-query attention (num_q_heads > num_kv_heads)
* decode with sliding-window masking
* decode with logit soft-cap
* mixed batch: one decode sequence + one extend sequence in the same
  call (exercises the MIXED case path with heterogeneous q_lens)

The tests require JAX on a TPU device and are skipped otherwise. They
do **not** depend on torch_tpu or the SGLang server runtime.
"""

from __future__ import annotations

import unittest
from typing import Optional

import numpy as np


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


@unittest.skipIf(not _has_jax_tpu(), _SKIP_REASON)
class TestRaggedPagedAttentionV3(unittest.TestCase):
    """Compare the Pallas kernel against the in-file reference impl."""

    NUM_Q_HEADS = 4
    NUM_KV_HEADS = 1
    HEAD_DIM = 128
    PAGE_SIZE = 16
    PAGES_PER_SEQ = 4
    # Per-seq pages are assigned contiguous absolute indices, so the table
    # has to be large enough for ``max_num_seqs * PAGES_PER_SEQ``. The
    # multi-seq tests use up to 3 sequences -> 12 pages; rounded up to 16.
    TOTAL_NUM_PAGES = 16
    TOLERANCE = 0.05

    def _build_inputs(
        self,
        seqs,
        *,
        seed: int = 0,
        num_q_heads: Optional[int] = None,
        num_kv_heads: Optional[int] = None,
    ):
        """Build matching kernel/reference inputs for one or more sequences.

        Args:
          seqs: list of ``(kv_len, q_len)`` pairs. ``kv_len`` includes the
            new ``q_len`` tokens the kernel will append to the cache.
          num_q_heads / num_kv_heads: overrides for GQA testing. Default
            to the class-level ``NUM_Q_HEADS`` / ``NUM_KV_HEADS``.

        Returns a dict carrying both the kernel-format and reference-format
        tensors plus the shared metadata. Pages for sequence ``i`` are
        assigned contiguous absolute indices ``[i * PAGES_PER_SEQ,
        (i + 1) * PAGES_PER_SEQ)``, and the kernel's ``cu_kv_lens`` is the
        rectangular ``i * PAGES_PER_SEQ * PAGE_SIZE`` layout described in
        the backend.
        """
        import jax
        import jax.numpy as jnp

        from sglang.srt.layers.attention.pallas.util import get_dtype_packing

        num_q_heads = num_q_heads if num_q_heads is not None else self.NUM_Q_HEADS
        num_kv_heads = (
            num_kv_heads if num_kv_heads is not None else self.NUM_KV_HEADS
        )
        assert num_q_heads % num_kv_heads == 0, (
            "num_q_heads must be a multiple of num_kv_heads for GQA tests"
        )

        for kv_len, q_len in seqs:
            assert kv_len >= q_len, "kv_len must include the new q_len tokens"
            assert kv_len <= self.PAGES_PER_SEQ * self.PAGE_SIZE, (
                f"kv_len={kv_len} exceeds per-seq capacity "
                f"({self.PAGES_PER_SEQ * self.PAGE_SIZE})"
            )
        num_seqs = len(seqs)
        assert num_seqs * self.PAGES_PER_SEQ <= self.TOTAL_NUM_PAGES, (
            "too many sequences for the configured page table"
        )

        dtype = jnp.bfloat16
        total_q_tokens = sum(q for _, q in seqs)
        # Pad max_num_tokens to a multiple of 8 so the kernel's tile shapes
        # are happy; the real q region is total_q_tokens.
        max_num_tokens = max(8, ((total_q_tokens + 7) // 8) * 8)

        key = jax.random.PRNGKey(seed)
        k_q, k_new_k, k_new_v, k_kpages, k_vpages = jax.random.split(key, 5)

        q = jax.random.normal(
            k_q, (max_num_tokens, num_q_heads, self.HEAD_DIM), dtype=dtype
        )
        new_k = jax.random.normal(
            k_new_k, (max_num_tokens, num_kv_heads, self.HEAD_DIM), dtype=dtype
        )
        new_v = jax.random.normal(
            k_new_v, (max_num_tokens, num_kv_heads, self.HEAD_DIM), dtype=dtype
        )

        # Unpacked KV pages for the reference. The kernel will write its new
        # K / V into the cache itself; we pre-populate the reference's pages
        # at the same slots so both impls attend over the same KV state.
        k_pages = jax.random.normal(
            k_kpages,
            (self.TOTAL_NUM_PAGES, self.PAGE_SIZE, num_kv_heads, self.HEAD_DIM),
            dtype=dtype,
        )
        v_pages = jax.random.normal(
            k_vpages,
            (self.TOTAL_NUM_PAGES, self.PAGE_SIZE, num_kv_heads, self.HEAD_DIM),
            dtype=dtype,
        )

        # 2D ref page table [num_seqs, pages_per_seq] with disjoint, contiguous
        # absolute pages per seq.
        page_indices_2d = np.arange(
            num_seqs * self.PAGES_PER_SEQ, dtype=np.int32
        ).reshape(num_seqs, self.PAGES_PER_SEQ)

        ref_kpages = k_pages
        ref_vpages = v_pages
        q_offset = 0
        for seq_idx, (kv_len, q_len) in enumerate(seqs):
            prefix_len = kv_len - q_len
            for tok in range(q_len):
                abs_pos = prefix_len + tok
                rel_page = abs_pos // self.PAGE_SIZE
                slot = abs_pos % self.PAGE_SIZE
                abs_page = int(page_indices_2d[seq_idx, rel_page])
                ref_kpages = ref_kpages.at[abs_page, slot].set(
                    new_k[q_offset + tok]
                )
                ref_vpages = ref_vpages.at[abs_page, slot].set(
                    new_v[q_offset + tok]
                )
            q_offset += q_len

        # Fused/packed KV cache for the kernel, built from the *pre-write*
        # pages so the kernel must perform the writes itself. The kernel's
        # ``merge_kv`` lays out heads as interleaved [K0, V0, K1, V1, ...]
        # (not [K0, K1, ..., V0, V1, ...]), so build the same order here —
        # otherwise GQA cases with num_kv_heads > 1 silently mis-feed the
        # kernel even though they'd accidentally match for num_kv_heads == 1.
        kv_packing = get_dtype_packing(dtype)
        kv_interleaved = jnp.stack([k_pages, v_pages], axis=3).reshape(
            self.TOTAL_NUM_PAGES,
            self.PAGE_SIZE,
            num_kv_heads * 2,
            self.HEAD_DIM,
        )
        kv_fused = kv_interleaved.reshape(
            self.TOTAL_NUM_PAGES,
            self.PAGE_SIZE,
            (num_kv_heads * 2) // kv_packing,
            kv_packing,
            self.HEAD_DIM,
        )

        kv_lens = jnp.asarray([kv_len for kv_len, _ in seqs], dtype=jnp.int32)
        cu_q_lens = jnp.asarray(
            [0] + list(np.cumsum([q for _, q in seqs], dtype=np.int32)),
            dtype=jnp.int32,
        )
        # Rectangular cu_kv_lens layout — the kernel reads cu_kv_lens[i] only
        # as a page-indices offset (start_kv_page_idx = cdiv(cu_kv_lens[i],
        # page_size)). Per the backend convention this is i * pages_per_seq *
        # page_size, regardless of each seq's actual KV length.
        cu_kv_lens = jnp.asarray(
            [i * self.PAGES_PER_SEQ * self.PAGE_SIZE for i in range(num_seqs + 1)],
            dtype=jnp.int32,
        )
        page_indices_flat = jnp.asarray(
            page_indices_2d.reshape(-1), dtype=jnp.int32
        )
        ref_page_indices = jnp.asarray(page_indices_2d, dtype=jnp.int32)

        return {
            "q": q,
            "new_k": new_k,
            "new_v": new_v,
            "kv_fused": kv_fused,
            "ref_kpages": ref_kpages,
            "ref_vpages": ref_vpages,
            "kv_lens": kv_lens,
            "page_indices_flat": page_indices_flat,
            "ref_page_indices": ref_page_indices,
            "cu_q_lens": cu_q_lens,
            "cu_kv_lens": cu_kv_lens,
            "max_num_tokens": max_num_tokens,
            "total_q_tokens": total_q_tokens,
            "num_seqs": num_seqs,
            "num_q_heads": num_q_heads,
            "num_kv_heads": num_kv_heads,
        }

    def _run_and_compare(self, inputs, *, distribution, **kernel_kwargs):
        """Run kernel + reference with matching kwargs and assert close."""
        import jax.numpy as jnp

        from sglang.srt.layers.attention.pallas.ragged_paged_attention_v3 import (
            ragged_paged_attention,
            ref_ragged_paged_attention,
        )

        sm_scale = 1.0 / float(np.sqrt(self.HEAD_DIM))
        # ``ref_ragged_paged_attention`` accepts a subset of the kernel
        # kwargs; forward only those it understands.
        ref_kwargs = {
            k: v
            for k, v in kernel_kwargs.items()
            if k in ("sliding_window", "soft_cap", "xai_temperature_len")
        }

        ref_out = ref_ragged_paged_attention(
            inputs["q"][: inputs["total_q_tokens"]],
            inputs["ref_kpages"],
            inputs["ref_vpages"],
            inputs["kv_lens"],
            inputs["ref_page_indices"],
            inputs["cu_q_lens"],
            jnp.array([inputs["num_seqs"]], dtype=jnp.int32),
            sm_scale=sm_scale,
            **ref_kwargs,
        )

        kernel_out, _ = ragged_paged_attention(
            inputs["q"],
            inputs["new_k"],
            inputs["new_v"],
            inputs["kv_fused"],
            inputs["kv_lens"],
            inputs["page_indices_flat"],
            inputs["cu_q_lens"],
            inputs["cu_kv_lens"],
            jnp.asarray(distribution, dtype=jnp.int32),
            None,  # custom_mask
            None,  # attention_sink
            sm_scale=sm_scale,
            **kernel_kwargs,
        )
        kernel_out = kernel_out[: inputs["total_q_tokens"]]

        diff = jnp.abs(
            ref_out.astype(jnp.float32) - kernel_out.astype(jnp.float32)
        ).max()
        # bf16 has ~3 decimal digits of precision; outputs are O(1).
        self.assertLess(
            float(diff), self.TOLERANCE, f"max abs diff {float(diff)} >= tolerance"
        )

    # ------------------------------------------------------------------ #
    # Original two cases — kept (single seq), now via the multi-seq helper.
    # ------------------------------------------------------------------ #

    def test_decode_single_seq(self):
        """q_len = 1, distribution = (1, 1, 1) -> DECODE-only path."""
        inputs = self._build_inputs([(17, 1)])
        self._run_and_compare(inputs, distribution=(1, 1, 1))

    def test_extend_single_seq(self):
        """q_len > 1, distribution = (0, 0, 1) -> MIXED path."""
        inputs = self._build_inputs([(33, 8)])
        self._run_and_compare(inputs, distribution=(0, 0, 1))

    # ------------------------------------------------------------------ #
    # Additional cases.
    # ------------------------------------------------------------------ #

    def test_decode_multi_seq(self):
        """3 decode seqs with different kv_lens -> exercises per-seq page
        offset math and the DECODE-path tile loop across multiple
        sequences in one call."""
        inputs = self._build_inputs([(17, 1), (33, 1), (49, 1)])
        # All 3 sequences are decode-only.
        self._run_and_compare(inputs, distribution=(3, 3, 3))

    def test_extend_gqa(self):
        """GQA: num_q_heads=8, num_kv_heads=2 (group=4). Verifies the
        kernel's Q-heads-per-KV-head replication matches the reference's
        ``jnp.repeat(..., num_query_per_kv, axis=1)`` path."""
        inputs = self._build_inputs(
            [(33, 8)], num_q_heads=8, num_kv_heads=2
        )
        self._run_and_compare(inputs, distribution=(0, 0, 1))

    def test_decode_sliding_window(self):
        """Sliding window of 8 over a kv_len of 33 -> only the last 8
        prefix tokens (plus the new decode token) are attended to.
        Hits the sliding-window precise-skipping path in the kernel."""
        inputs = self._build_inputs([(33, 1)])
        self._run_and_compare(
            inputs, distribution=(1, 1, 1), sliding_window=8
        )

    def test_decode_soft_cap(self):
        """soft_cap rescales logits with tanh before softmax. Verifies the
        kernel's logit-cap path matches the reference's
        ``soft_cap * jnp.tanh(attn / soft_cap)``."""
        inputs = self._build_inputs([(33, 1)])
        self._run_and_compare(
            inputs, distribution=(1, 1, 1), soft_cap=30.0
        )

    def test_mixed_decode_and_extend(self):
        """One decode seq (q=1) + one extend seq (q=8) in the same batch.
        Both seqs end up on the MIXED path (chunk_prefill_size is None,
        which remaps PREFILL -> MIXED), so distribution=(1, 1, 2):
        index 0 is decode, index 1 is mixed."""
        inputs = self._build_inputs([(17, 1), (33, 8)])
        self._run_and_compare(inputs, distribution=(1, 1, 2))


if __name__ == "__main__":
    unittest.main()
