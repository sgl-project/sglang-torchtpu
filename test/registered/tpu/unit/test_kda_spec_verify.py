# SPDX-License-Identifier: Apache-2.0
"""KDA speculative verify on the GDN v3 kernel must match token-by-token decode.

A verify window reads its initial state from `base + read_offset` and writes the
state after window position t to `base + t`. Committing k accepted tokens then
only has to point the next read at `base + k - 1`, so every checkpoint must
equal the state decode reaches after the same tokens.
"""

import unittest

import numpy as np

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase

try:
    import jax
    import jax.numpy as jnp

    _HAS_TPU = any(d.platform == "tpu" for d in jax.devices())
except Exception:  # pragma: no cover - non-TPU host
    _HAS_TPU = False

register_tpu_ci(est_time=120, suite="stage-a-test-tpu-unit", chips=1)

# GLM-5.3-Flash KDA: heads of 128, short conv 4, bounded gate. A TP8 rank runs
# 8 of the 64 heads; an attention-DP rank runs all 64.
D, K, LOWER = 128, 4, -5.0
TP8_HEADS, DP_HEADS = 8, 64


@unittest.skipUnless(_HAS_TPU, "requires a JAX TPU backend")
class TestKdaSpecVerify(CustomTestCase):
    def _kernel(self, n_v, num_spec_tokens, batched_only=True):
        from sglang_tpu.attention.pallas.gdn import wrapper
        from sglang_tpu.attention.pallas.gdn.config import AttentionMode

        dim = 3 * n_v * D

        def run(qkv, b, a, conv, rec, packed, qsl, idx, dist, seq_lens, offsets):
            return wrapper.fused_conv1d_gdn(
                qkv,
                b,
                a,
                conv,
                rec,
                jnp.zeros((dim, 1, K), jnp.float32),
                None,
                jnp.zeros((n_v,), jnp.float32),
                jnp.zeros((n_v, D), jnp.float32),
                qsl,
                idx,
                dist,
                seq_lens,
                offsets,
                n_kq=n_v,
                n_v=n_v,
                d_k=D,
                d_v=D,
                kernel_size=K,
                attention_mode=AttentionMode.KDA,
                gate_lower_bound=LOWER,
                num_spec_tokens=num_spec_tokens,
                batched_only=batched_only,
                packed_weights=packed,
            )

        return jax.jit(run)

    def test_window_checkpoints_match_sequential_decode(self):
        self._check_window_against_decode(n_v=TP8_HEADS, window=8, read_offsets=[3, 0])

    def test_dp_rank_verify_window_fits_vmem(self):
        """A 64-head verify window must compile and match sequential decode.

        An attention-DP rank runs every KDA head, so each window checkpoint is a
        4 MiB f32 state; holding all of them at once exceeded the scoped VMEM
        limit at compile time (sgl-project/sglang-torchtpu#2053).
        """
        self._check_window_against_decode(n_v=DP_HEADS, window=4, read_offsets=[2, 0])

    def _check_window_against_decode(self, *, n_v, window, read_offsets):
        from sglang_tpu.kda_layout import (
            kda_packed_weights_shape,
        )

        dim = 3 * n_v * D
        context = 20
        bs = len(read_offsets)
        rng = np.random.default_rng(7)
        # Group g owns slots [g * window, (g + 1) * window); group 0 is null.
        slots = (bs + 1) * window
        bases = [(s + 1) * window for s in range(bs)]
        conv = rng.normal(size=(slots, K - 1, dim)).astype(np.float32)
        rec = (0.1 * rng.normal(size=(slots, n_v, D, D))).astype(np.float32)
        rows, width = kda_packed_weights_shape(K, dim, n_v, D)
        packed = (0.3 * rng.normal(size=(rows, width))).astype(np.float32)
        tokens = bs * window
        qkv = (0.5 * rng.normal(size=(tokens, dim))).astype(np.float32)
        b = rng.normal(size=(tokens, n_v)).astype(np.float32)
        a = rng.normal(size=(tokens, n_v * D)).astype(np.float32)
        bf = lambda x: jnp.asarray(x).astype(jnp.bfloat16)  # noqa: E731

        verify = self._kernel(n_v, window - 1)
        (new_conv, new_rec), out = verify(
            bf(qkv),
            bf(b),
            bf(a),
            jnp.asarray(conv),
            jnp.asarray(rec),
            jnp.asarray(packed),
            jnp.arange(bs + 1, dtype=jnp.int32) * window,
            jnp.asarray(bases, jnp.int32),
            jnp.full((3,), bs, jnp.int32),
            jnp.full((bs,), context + window, jnp.int32),
            jnp.asarray(read_offsets, jnp.int32),
        )
        new_conv, new_rec, out = map(np.asarray, (new_conv, new_rec, out))

        decode = self._kernel(n_v, 0)
        touched = set()
        for s, (base, off) in enumerate(zip(bases, read_offsets)):
            # A two-slot pool (null + this sequence) seeded from the read slot.
            c = np.stack([conv[0], conv[base + off]])
            r = np.stack([rec[0], rec[base + off]])
            for t in range(window):
                row = s * window + t
                (c, r), o = decode(
                    bf(qkv[row : row + 1]),
                    bf(b[row : row + 1]),
                    bf(a[row : row + 1]),
                    jnp.asarray(c),
                    jnp.asarray(r),
                    jnp.asarray(packed),
                    jnp.array([0, 1], jnp.int32),
                    jnp.array([1], jnp.int32),
                    jnp.array([1, 1, 1], jnp.int32),
                    jnp.array([context + t + 1], jnp.int32),
                    jnp.zeros((1,), jnp.int32),
                )
                c, r, o = np.asarray(c), np.asarray(r), np.asarray(o)
                np.testing.assert_allclose(
                    out[row].astype(np.float32),
                    o[0].astype(np.float32),
                    rtol=2e-2,
                    atol=2e-2,
                    err_msg=f"seq {s} token {t} output",
                )
                # Verify upcasts its inputs to f32 up front while decode loads
                # them compactly, so states agree to precision, not bitwise; a
                # wrong slot or offset is O(1) off.
                for name, got, want in (
                    ("recurrent", new_rec[base + t], r[1]),
                    ("conv", new_conv[base + t], c[1]),
                ):
                    rel = np.linalg.norm(got - want) / np.linalg.norm(want)
                    self.assertLess(rel, 1e-2, f"seq {s} {name} checkpoint {t}")
                touched.add(base + t)
        for slot in range(slots):
            if slot not in touched:
                np.testing.assert_array_equal(new_rec[slot], rec[slot])
                np.testing.assert_array_equal(new_conv[slot], conv[slot])


if __name__ == "__main__":
    unittest.main()
