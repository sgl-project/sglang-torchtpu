# SPDX-License-Identifier: Apache-2.0
"""MLA v2 kernel vs the NumPy reference with an fp8 e4m3 latent cache.

``pallas_tpu_mla`` quantizes the new latent and rope rows with the layer's
``k_scale`` and passes that one scale as both ``k_scale`` and ``v_scale``,
since MLA's V is the K latent. The unit file covers bf16 only; here the same
reference, inputs and cache-write check run with an fp8 cache, a bf16 Q and a
non-unit scale, through the production call shape (tuned block sizes,
``decode_batch_size`` 4, page 128).

fp8 packs four tokens per 32-bit word, so the kernel's sub-word merge of new
rows into a partly filled word is a different code path from bf16's two.
"""

from __future__ import annotations

import os
import sys
import unittest

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(
    est_time=60,
    suite="stage-a-test-tpu-kernel",
    labels=["kernel", "attention", "mla"],
)

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


PAGE_SIZE = 128
# Mirrors attention_pallas_mla._MLA_DECODE_BATCH_SIZE.
DECODE_BATCH_SIZE = 4
# Power of two, so the dequantized reference latents are exact.
KV_SCALE = 0.5
# The kernel rounds the unnormalized probabilities to the fp8 V dtype before
# P @ V (p_same_dtype_as_v); the reference keeps them in bf16.
RTOL, ATOL = 5e-2, 1e-1

# Nine single-token decodes straddling pages: two batched groups of four
# plus one tail row.
DECODE_LENS = [
    (1, 127),
    (1, 128),
    (1, 129),
    (1, 133),
    (1, 255),
    (1, 513),
    (1, 1023),
    (1, 1025),
    (1, 61),
]
# Prefix and total lengths cover every residue mod 4, so new rows start and
# end inside a partly filled fp8 word.
EXTEND_LENS = [(1, 128), (125, 125), (123, 522), (3, 515), (6, 263), (2, 7)]


@unittest.skipIf(not _has_jax_tpu(), "Requires JAX with a TPU device")
class TestMlaRpaV2ReferenceModes(unittest.TestCase):
    def _run(self, lens, *, distribution):
        from types import SimpleNamespace

        import jax
        import jax.numpy as jnp
        import numpy as np

        from sglang_tpu.attention.pallas.mla_ragged_paged_attention_v2 import (
            mla_ragged_paged_attention,
        )
        from sglang_tpu.testing.mla_rpa_v2 import (
            MASK_VALUE,
            NUM_Q_HEADS,
            SM_SCALE,
            assert_cache_write,
            build_case,
            ref_attention,
        )

        case = build_case(
            lens,
            page_size=PAGE_SIZE,
            num_heads=NUM_Q_HEADS,
            dtype=jnp.float8_e4m3fn,
            q_dtype=jnp.bfloat16,
            seed=0,
        )
        out, updated_cache = mla_ragged_paged_attention(
            jnp.asarray(case.ql_nope),
            jnp.asarray(case.q_pe),
            jnp.asarray(case.new_kv_c),
            jnp.asarray(case.new_k_pe),
            jnp.asarray(case.cache_before),
            jnp.asarray(case.kv_lens),
            jnp.asarray(case.page_indices),
            jnp.asarray(case.cu_q_lens),
            jnp.asarray(case.cu_kv_lens),
            jnp.asarray(np.asarray(distribution, dtype=np.int32)),
            sm_scale=SM_SCALE,
            mask_value=MASK_VALUE,
            k_scale=KV_SCALE,
            v_scale=KV_SCALE,
            decode_batch_size=DECODE_BATCH_SIZE,
        )
        jax.block_until_ready((out, updated_cache))

        # Reference over the dequantized latents, in bf16 (the Q/output dtype).
        dequantized = SimpleNamespace(
            lens=case.lens,
            np_dtype=jnp.dtype(jnp.bfloat16),
            per_seq=[
                SimpleNamespace(
                    ql_nope=s.ql_nope,
                    q_pe=s.q_pe,
                    kv_c=s.kv_c.astype(np.float32) * KV_SCALE,
                    k_pe=s.k_pe.astype(np.float32) * KV_SCALE,
                )
                for s in case.per_seq
            ],
        )
        expected = ref_attention(
            dequantized,
            sm_scale=SM_SCALE,
            sliding_window=None,
            soft_cap=None,
            mask_value=MASK_VALUE,
        ).astype(np.float32)
        got = np.asarray(out).astype(np.float32)
        self.assertEqual(got.shape, expected.shape)
        np.testing.assert_allclose(
            got,
            expected,
            rtol=RTOL,
            atol=ATOL,
            err_msg=f"max_diff={float(np.max(np.abs(got - expected)))}",
        )
        assert_cache_write(case, updated_cache)

    def test_fp8_decode(self):
        """Batched and tail DECODE passes read fp8 latents and dequantize
        both the score and the output with the shared scale."""
        n = len(DECODE_LENS)
        self._run(DECODE_LENS, distribution=(n, n, n))

    def test_fp8_extend(self):
        """Extend batches route through MIXED, as the MLA backend sends them;
        new fp8 rows merge into partly filled words without clobbering the
        prefix."""
        self._run(EXTEND_LENS, distribution=(0, 0, len(EXTEND_LENS)))


if __name__ == "__main__":
    unittest.main()
