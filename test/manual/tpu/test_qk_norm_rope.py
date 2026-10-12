"""Fused QK RMSNorm + RoPE Pallas op vs a numpy reference."""

from __future__ import annotations

import unittest
import unittest.mock

import numpy as np
import torch

try:
    import jax
    import jax.numpy as jnp

    _HAS_JAX = True
except Exception:  # pragma: no cover
    _HAS_JAX = False


def _has_tpu_via_jax() -> bool:
    if not _HAS_JAX:
        return False
    try:
        return any(getattr(d, "platform", "") == "tpu" for d in jax.devices())
    except Exception:
        return False


_SKIP_REASON = "Requires JAX with a TPU backend visible to jax.devices()"

# Qwen3-Coder-480B at TP8: 96 / 8 q heads, 8 / 8 kv heads, 128-wide heads.
NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM = 12, 1, 128
QKV_WIDTH = (NUM_Q_HEADS + 2 * NUM_KV_HEADS) * HEAD_DIM
EPS = 1e-6


def _make_inputs(num_tokens: int, seed: int):
    """bf16 inputs with magnitudes that stay inside bf16's accurate range."""
    rng = np.random.default_rng(seed)
    qkv = rng.standard_normal((num_tokens, QKV_WIDTH), dtype=np.float32)
    cos_sin = rng.standard_normal((num_tokens, HEAD_DIM), dtype=np.float32)
    # Norm weights sit near 1.0, as trained weights do.
    q_weight = 1.0 + 0.1 * rng.standard_normal(HEAD_DIM, dtype=np.float32)
    k_weight = 1.0 + 0.1 * rng.standard_normal(HEAD_DIM, dtype=np.float32)
    to_bf16 = lambda x: jnp.asarray(x, dtype=jnp.bfloat16)
    return tuple(map(to_bf16, (qkv, cos_sin, q_weight, k_weight)))


def _numpy_reference(qkv, cos_sin, q_weight, k_weight):
    """RMSNorm per (token, head), then neox rotate-half. v is untouched."""
    num_tokens = qkv.shape[0]
    q_size = NUM_Q_HEADS * HEAD_DIM
    kv_size = NUM_KV_HEADS * HEAD_DIM
    half = HEAD_DIM // 2

    q = qkv[:, :q_size].reshape(num_tokens, NUM_Q_HEADS, HEAD_DIM)
    k = qkv[:, q_size : q_size + kv_size].reshape(num_tokens, NUM_KV_HEADS, HEAD_DIM)
    v = qkv[:, q_size + kv_size :]

    # One angle per channel pair, broadcast across the heads of a token.
    cos = cos_sin[:, None, :half]
    sin = cos_sin[:, None, half:]

    def rms_norm(x, weight):
        return x / np.sqrt((x * x).mean(-1, keepdims=True) + EPS) * weight

    def rope_neox(x):
        x1, x2 = x[..., :half], x[..., half:]
        return np.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)

    q = rope_neox(rms_norm(q, q_weight)).reshape(num_tokens, q_size)
    k = rope_neox(rms_norm(k, k_weight)).reshape(num_tokens, kv_size)
    return q, k, v


@unittest.skipUnless(_has_tpu_via_jax(), _SKIP_REASON)
class TestQKNormRopeKernel(unittest.TestCase):
    """The kernel blocks over token rows, so the block geometry is what varies:
    an exact multiple of the block, a single short block, and a ragged final
    block whose out-of-range writes must be masked.
    """

    # bf16 rounding through an f32 norm and rotate. The op computes in f32 and
    # casts the result to bf16, so per-element error is one bf16 ulp (~2^-8 of
    # the magnitude). Elementwise allclose catches localized corruption (a
    # wrong channel pairing shows up as an error of order 1) that a
    # mean-relative metric would average away.
    RTOL = 0.03
    ATOL = 0.02

    def _check(self, num_tokens: int, seed: int = 0):
        from sglang_tpu.qk_norm_rope import _qk_norm_rope_jax

        inputs = _make_inputs(num_tokens, seed)
        got = jax.block_until_ready(
            _qk_norm_rope_jax(*inputs, EPS, NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM)
        )
        got_q, got_k = (np.asarray(t, dtype=np.float32) for t in got)
        want_q, want_k, _ = _numpy_reference(
            *(np.asarray(t, dtype=np.float32) for t in inputs)
        )

        for name, a, b in (("q", got_q, want_q), ("k", got_k, want_k)):
            self.assertEqual(a.shape, b.shape, f"{name} shape (m={num_tokens})")
            self.assertTrue(np.isfinite(a).all(), f"{name} not finite")
            np.testing.assert_allclose(
                a,
                b,
                rtol=self.RTOL,
                atol=self.ATOL,
                err_msg=f"{name} mismatch (m={num_tokens})",
            )

    def test_exact_multiple_of_block(self):
        from sglang_tpu.qk_norm_rope import _MAX_BLOCK_TOKENS

        self._check(_MAX_BLOCK_TOKENS * 4)

    def test_prefill_chunk(self):
        self._check(8192)

    def test_single_short_block(self):
        self._check(64)

    def test_ragged_final_block(self):
        from sglang_tpu.qk_norm_rope import _MAX_BLOCK_TOKENS

        # One full block plus a partial one: the padded rows are computed but
        # their writes fall outside the array, so valid rows must be unaffected.
        self._check(_MAX_BLOCK_TOKENS + 188)

    def test_block_size_does_not_change_results(self):
        """The kernel is per-row, so row blocking must not alter the output."""
        from sglang_tpu import qk_norm_rope

        inputs = _make_inputs(1024, seed=1)
        outputs = []
        for block_tokens in (128, 512):
            with unittest.mock.patch.object(
                qk_norm_rope, "_MAX_BLOCK_TOKENS", block_tokens
            ):
                out = jax.block_until_ready(
                    qk_norm_rope._qk_norm_rope_jax(
                        *inputs, EPS, NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM
                    )
                )
            outputs.append([np.asarray(t, dtype=np.float32) for t in out])
        for name, a, b in zip("qk", outputs[0], outputs[1]):
            np.testing.assert_array_equal(
                a, b, err_msg=f"{name} changed with the block size"
            )


class TestQKNormRopeInputContract(unittest.TestCase):
    """The wrapper's checks read shapes and dtypes only, so CPU tensors suffice.

    These guard callers that bypass the model's eligibility gate.
    """

    def _args(self, **overrides):
        num_tokens = 8
        args = dict(
            qkv=torch.zeros((num_tokens, QKV_WIDTH), dtype=torch.bfloat16),
            positions=torch.zeros(num_tokens, dtype=torch.int32),
            cos_sin_cache=torch.zeros((16, HEAD_DIM), dtype=torch.float32),
            q_weight=torch.zeros(HEAD_DIM, dtype=torch.bfloat16),
            k_weight=torch.zeros(HEAD_DIM, dtype=torch.bfloat16),
            eps=EPS,
            num_q_heads=NUM_Q_HEADS,
            num_kv_heads=NUM_KV_HEADS,
            head_dim=HEAD_DIM,
        )
        args.update(overrides)
        return args

    def _assert_rejected(self, pattern: str, **overrides):
        from sglang_tpu.qk_norm_rope import (
            tpu_fused_qk_norm_rope,
        )

        with self.assertRaisesRegex(ValueError, pattern):
            tpu_fused_qk_norm_rope(**self._args(**overrides))

    def test_rejects_non_bfloat16(self):
        # fp8 would be re-cast on output with no scale, and the cos/sin rows
        # would be cast down to it as well.
        self._assert_rejected(
            "bfloat16",
            qkv=torch.zeros((8, QKV_WIDTH), dtype=torch.float8_e4m3fn),
        )

    def test_rejects_non_2d_qkv(self):
        self._assert_rejected(
            "2-D", qkv=torch.zeros((2, 4, QKV_WIDTH), dtype=torch.bfloat16)
        )

    def test_rejects_positions_token_mismatch(self):
        self._assert_rejected("positions", positions=torch.zeros(7, dtype=torch.int32))

    def test_rejects_unsupported_head_dim(self):
        self._assert_rejected("head_dim", head_dim=96)

    def test_rejects_nonpositive_head_counts(self):
        self._assert_rejected("must be positive", num_kv_heads=0)

    def test_rejects_1d_cache(self):
        # A 1-D cache slips past a last-dim-only check but breaks the
        # per-position gather the kernel relies on.
        self._assert_rejected(
            "2-D", cos_sin_cache=torch.zeros(HEAD_DIM, dtype=torch.float32)
        )

    def test_rejects_3d_cache(self):
        # A per-head cache, as FoPE builds. A caller that cached the
        # eligibility decision and then swapped rotary_emb lands here, so the
        # error names the flag rather than failing inside Pallas.
        self._assert_rejected(
            "enable-fused-qk-norm-rope",
            cos_sin_cache=torch.zeros((16, 8, 2 * HEAD_DIM), dtype=torch.float32),
        )

    def test_rejects_qkv_width_mismatch(self):
        self._assert_rejected(
            "qkv width", qkv=torch.zeros((8, QKV_WIDTH + 128), dtype=torch.bfloat16)
        )

    def test_rejects_wrong_norm_weight_shape(self):
        self._assert_rejected(
            "norm weights", q_weight=torch.zeros(64, dtype=torch.bfloat16)
        )

    def test_rejects_partial_rotary_cache(self):
        # rotary_dim < head_dim: the kernel rotates the whole head, so the
        # narrower cache row means the model uses partial rotary.
        self._assert_rejected(
            "Partial rotary",
            cos_sin_cache=torch.zeros((16, HEAD_DIM // 2), dtype=torch.float32),
        )

    def test_rejects_zero_tokens(self):
        # An empty batch would divide by zero when the Pallas grid is sized.
        # The scheduler never dispatches one, so it means the caller miscounted.
        self._assert_rejected(
            "at least one token row",
            qkv=torch.zeros((0, QKV_WIDTH), dtype=torch.bfloat16),
            positions=torch.zeros(0, dtype=torch.int32),
        )


class TestQKNormRopeRegistration(unittest.TestCase):
    def test_registration_is_idempotent(self):
        from sglang_tpu import qk_norm_rope

        first = qk_norm_rope._get_qk_norm_rope_op()
        qk_norm_rope.register_tpu_qk_norm_rope_op()
        second = qk_norm_rope._get_qk_norm_rope_op()
        self.assertIs(first, second)


if __name__ == "__main__":
    unittest.main(verbosity=2)
