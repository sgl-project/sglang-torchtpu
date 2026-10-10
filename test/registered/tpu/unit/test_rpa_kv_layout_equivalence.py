# SPDX-License-Identifier: Apache-2.0
"""TPU numerics coverage for the SEQ_ALONG_LANE KV layout.

The two layouts compute the same attention, so the same logical inputs must
give the same output. The seq-along-lane cache has no independent reference in
the tree, since nothing else reads a buffer with the token index on lanes, so
the long-standing token-major path stands in as one.

Both sides start from a zeroed cache and let the kernel write K/V itself, which
keeps the test from having to know how either layout packs head_dim. It caught
a KV-DMA change that left decode 37% off while prefill still matched.
"""

from __future__ import annotations

import math
import os
import sys
import unittest

import numpy as np

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=240, suite="stage-a-test-tpu-unit")

# Standalone raw-JAX kernel test, not a TorchTPU runtime process: SGLang
# platform discovery selects CPU-only JAX for export, so pick the physical
# backend before the first JAX import.
assert "jax" not in sys.modules
os.environ["JAX_PLATFORMS"] = "tpu"


def _has_jax_tpu() -> bool:
    try:
        import jax

        return any("TPU" in device.device_kind for device in jax.devices())
    except Exception:
        return False


@unittest.skipIf(not _has_jax_tpu(), "Requires JAX with a TPU device")
class TestKvLayoutEquivalence(unittest.TestCase):
    HEAD_DIM = 128
    PAGE_SIZE = 256
    # Prefill and decode may differ in bf16 accumulation order between the two
    # layouts, so require closeness rather than bit-equality. The regression
    # this guards against was 37% off, far outside this.
    TOLERANCE = 2e-2

    def _run(self, layout, *, num_q_heads, num_kv_heads, num_seqs, chunk, steps):
        import jax.numpy as jnp

        from sglang_tpu.attention.pallas.batched_rpa import wrapper

        rng = np.random.default_rng(0)
        pages_per_seq = -(-(chunk + steps + self.PAGE_SIZE) // self.PAGE_SIZE)
        total_pages = pages_per_seq * num_seqs
        shape = wrapper.get_kv_cache_shape(
            total_num_pages=total_pages,
            page_size=self.PAGE_SIZE,
            actual_num_kv_heads=num_kv_heads,
            actual_head_dim=self.HEAD_DIM,
            kv_dtype=jnp.bfloat16,
            kv_layout=layout,
        )
        cache = jnp.zeros(shape, jnp.bfloat16)
        # Shuffled, so a write to the slot index cannot pass for the mapped page.
        page_indices = jnp.asarray(rng.permutation(total_pages).astype(np.int32))
        sm_scale = 1.0 / math.sqrt(self.HEAD_DIM)
        outs = []

        def call(q, k, v, c, kv_lens, cu_q_lens, distribution):
            return wrapper.ragged_paged_attention(
                q,
                k,
                v,
                c,
                kv_lens,
                page_indices,
                cu_q_lens,
                distribution,
                sm_scale=sm_scale,
                skip_kv_update=False,
                kv_layout=layout,
            )

        def randn(shape_):
            return jnp.asarray(rng.standard_normal(shape_) * 0.5, jnp.bfloat16)

        # One MIXED step: every sequence extends by `chunk` tokens.
        total_q = chunk * num_seqs
        out, cache = call(
            randn((total_q, num_q_heads, self.HEAD_DIM)),
            randn((total_q, num_kv_heads, self.HEAD_DIM)),
            randn((total_q, num_kv_heads, self.HEAD_DIM)),
            cache,
            jnp.full((num_seqs,), chunk, jnp.int32),
            jnp.asarray(np.arange(num_seqs + 1, dtype=np.int32) * chunk),
            jnp.asarray([0, 0, num_seqs], jnp.int32),
        )
        outs.append(np.asarray(out.astype(jnp.float32)))

        # Decode steps: one new token per sequence, kv_len growing each step.
        cu_q_decode = jnp.asarray(np.arange(num_seqs + 1, dtype=np.int32))
        distribution = jnp.full((3,), num_seqs, jnp.int32)
        for step in range(steps):
            out, cache = call(
                randn((num_seqs, num_q_heads, self.HEAD_DIM)),
                randn((num_seqs, num_kv_heads, self.HEAD_DIM)),
                randn((num_seqs, num_kv_heads, self.HEAD_DIM)),
                cache,
                jnp.full((num_seqs,), chunk + step + 1, jnp.int32),
                cu_q_decode,
                distribution,
            )
            outs.append(np.asarray(out.astype(jnp.float32)))
        return outs

    def _assert_layouts_agree(self, **kw):
        from sglang_tpu.attention.pallas.batched_rpa import configs

        ref = self._run(configs.KVLayout.HEAD_ALONG_SUBLANE, **kw)
        got = self._run(configs.KVLayout.SEQ_ALONG_LANE, **kw)
        self.assertEqual(len(ref), len(got))
        for i, (a, b) in enumerate(zip(ref, got)):
            rel = float(np.abs(a - b).max()) / max(float(np.abs(a).max()), 1e-6)
            phase = "mixed" if i == 0 else f"decode step {i}"
            self.assertLess(rel, self.TOLERANCE, f"{phase}: relative diff {rel}")

    def test_partial_last_page(self):
        """A chunk that is not a page multiple leaves a partial last page.

        Sizing a KV transfer by a whole page instead of by what it moves
        only shows up when the last page is partial.
        """
        self._assert_layouts_agree(
            num_q_heads=4, num_kv_heads=1, num_seqs=8, chunk=600, steps=4
        )

    def test_page_aligned_chunk(self):
        self._assert_layouts_agree(
            num_q_heads=4, num_kv_heads=1, num_seqs=4, chunk=512, steps=2
        )

    def test_multiple_kv_heads(self):
        self._assert_layouts_agree(
            num_q_heads=8, num_kv_heads=2, num_seqs=4, chunk=300, steps=2
        )


if __name__ == "__main__":
    unittest.main()
