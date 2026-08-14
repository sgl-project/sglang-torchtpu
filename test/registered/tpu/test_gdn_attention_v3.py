"""Correctness tests for the vendored Pallas fused Conv1D + GDN kernel.

The kernel lives in ``sglang.srt.layers.attention.pallas.gdn`` (public entry
``fused_conv1d_gdn``). These tests build small ragged batches and compare the
kernel output and updated conv/recurrent state against an in-file pure-JAX
eager reference that runs the gated-delta-rule sequence-by-sequence.

Cases covered (kernel vs. reference, bf16-ish tolerance):

* single long prefill
* mixed multi-sequence prefill
* decode-only batch
* mixed prefill + decode in one call
* padded batches (trailing dummy sequences)

Plus two state-plumbing invariants:

* ``has_initial_state=False`` masks a stale (non-zero) slot so a fresh
  prefill matches a fresh-zero slot bit-for-bit.
* ``has_initial_state=True`` continues a chunked prefill from the slot's
  state, matching the equivalent single-shot prefill.

The kernel is pure JAX/Pallas and requires a TPU device; tests are skipped
otherwise. They do **not** depend on torch_tpu or the SGLang server runtime.

Adapted from torchtpu-vllm ``tests/kernels/test_gdn_attention_v3.py``.

Usage:
    python test/registered/tpu/test_gdn_attention_v3.py
"""

from __future__ import annotations

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=180, suite="stage-a-test-tpu-unit")

import unittest

from sglang.test.test_utils import CustomTestCase


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


if _has_jax_tpu():
    import jax
    import jax.numpy as jnp
    import numpy as np


def _l2_normalize(x, eps: float = 1e-6):
    x_f32 = x.astype(jnp.float32)
    norm = jnp.sqrt(jnp.sum(x_f32 * x_f32, axis=-1, keepdims=True) + eps)
    return (x_f32 / norm).astype(x.dtype)


def _gdn_attention_ref(
    qkv,
    b,
    a,
    conv_state,
    recurrent_state,
    conv_weight,
    conv_bias,
    a_log,
    dt_bias,
    query_start_loc,
    state_indices,
    distribution,
    seq_lens,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    kernel_size: int,
):
    """Eager GDN attention, sequence-by-sequence, in float32.

    Bypasses the ragged-masking complexity of the kernel by slicing each
    sequence out of the packed batch using ``query_start_loc``.
    """
    num_tokens = qkv.shape[0]
    num_valid_seqs = int(distribution[2])

    out_mixed_qkv = jnp.zeros_like(qkv)
    new_conv_state = jnp.array(conv_state)
    new_recurrent_state = jnp.array(recurrent_state)
    output = jnp.zeros((num_tokens, n_v * d_v), dtype=qkv.dtype)

    # Part 1: causal depthwise conv1d per sequence.
    for req_idx in range(num_valid_seqs):
        s = int(state_indices[req_idx])
        start = int(query_start_loc[req_idx])
        end = int(query_start_loc[req_idx + 1])
        query_len = end - start
        if query_len <= 0:
            continue

        has_init = bool((seq_lens[req_idx] - query_len) > 0)
        if has_init:
            c_state = new_conv_state[s]
        else:
            c_state = jnp.zeros_like(new_conv_state[s])
            new_recurrent_state = new_recurrent_state.at[s].set(
                jnp.zeros_like(new_recurrent_state[s])
            )

        X = qkv[start:end]
        x_full = jnp.concatenate([c_state, X], axis=0)
        acc = jnp.zeros((query_len, qkv.shape[-1]), dtype=jnp.float32)
        for k in range(kernel_size):
            acc += (
                x_full[k : k + query_len].astype(jnp.float32)
                * conv_weight[:, 0, k].astype(jnp.float32)[None, :]
            )
        if conv_bias is not None:
            acc += conv_bias.astype(jnp.float32)[None, :]
        conv_out = acc.astype(qkv.dtype)
        new_conv_state = new_conv_state.at[s].set(x_full[-(kernel_size - 1) :])
        out_mixed_qkv = out_mixed_qkv.at[start:end].set(conv_out)

    out_mixed_qkv = jax.nn.silu(out_mixed_qkv)

    # Part 2: gated-delta-rule recurrence per sequence.
    for req_idx in range(num_valid_seqs):
        s = int(state_indices[req_idx])
        start = int(query_start_loc[req_idx])
        end = int(query_start_loc[req_idx + 1])
        query_len = end - start
        if query_len <= 0:
            continue

        r_state = new_recurrent_state[s]
        qkv_seq = out_mixed_qkv[start:end]
        key_dim = n_kq * d_k
        q_seq = qkv_seq[:, :key_dim].reshape(query_len, n_kq, d_k)
        k_seq = qkv_seq[:, key_dim : key_dim * 2].reshape(query_len, n_kq, d_k)
        v_seq = qkv_seq[:, key_dim * 2 :].reshape(query_len, n_v, d_v)

        repeat_factor = n_v // n_kq
        if repeat_factor > 1:
            q_seq = jnp.repeat(q_seq, repeat_factor, axis=1)
            k_seq = jnp.repeat(k_seq, repeat_factor, axis=1)

        q_seq = _l2_normalize(q_seq)
        k_seq = _l2_normalize(k_seq)
        v_seq = v_seq.astype(jnp.float32)

        b_seq = b[start:end]
        a_seq = a[start:end]

        beta_seq = jax.nn.sigmoid(b_seq.astype(jnp.float32))
        g_seq = -jnp.exp(a_log.astype(jnp.float32))[None, :] * jax.nn.softplus(
            a_seq.astype(jnp.float32) + dt_bias.astype(jnp.float32)[None, :]
        )

        def step_fn(carry_state, xs):
            q_t, k_t, v_t, beta_t, g_t = xs
            q_t = q_t * (d_k**-0.5)
            exp_g = jnp.exp(g_t)

            k_state = jnp.einsum("hd, hdm -> hm", k_t, carry_state)
            v_diff = v_t - exp_g[:, None] * k_state
            v_new = beta_t[:, None] * v_diff

            q_state = jnp.einsum("hd, hdm -> hm", q_t, carry_state)
            q_k = jnp.sum(q_t * k_t, axis=-1, keepdims=True)
            out_step = exp_g[:, None] * q_state + q_k * v_new

            k_v_new = jnp.einsum("hd, hm -> hdm", k_t, v_new)
            new_state = carry_state * exp_g[:, None, None] + k_v_new

            return (new_state.astype(recurrent_state.dtype), out_step.astype(qkv.dtype))

        final_r_state, out_seq = jax.lax.scan(
            step_fn, r_state, (q_seq, k_seq, v_seq, beta_seq, g_seq)
        )
        new_recurrent_state = new_recurrent_state.at[s].set(final_r_state)
        output = output.at[start:end].set(out_seq.reshape(query_len, n_v * d_v))

    return (new_conv_state, new_recurrent_state), output


@unittest.skipIf(not _has_jax_tpu(), _SKIP_REASON)
class TestGDNAttentionV3(CustomTestCase):
    """Compare the Pallas fused Conv1D+GDN kernel against an eager reference."""

    KQ_HEAD_DIM = 128
    V_HEAD_DIM = 128
    N_KQ = 2
    N_V = 8
    KERNEL_SIZE = 4

    # (name, max_reqs, lengths, q_loc, distribution)
    CASES = [
        ("prefill", 1, [8192], [0, 8192], [0, 0, 3]),
        ("mixed", 3, [256, 128, 128], [0, 256, 384, 512], [0, 3, 3]),
        ("decode_only", 64, [1] * 64, list(range(65)), [64, 64, 64]),
        (
            "mixed_prefill_decode",
            11,
            [1] * 8 + [128, 128, 256],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 136, 264, 520],
            [8, 11, 11],
        ),
        (
            "padded_mixed_prefill",
            16,
            [128, 64, 32, 16, 8],
            [0, 128, 192, 224, 240, 248] + [1] * 11,
            [0, 5, 5],
        ),
        (
            "padded_decode_only",
            512,
            [1] * 64,
            list(range(65)) + [1] * 448,
            [64, 64, 64],
        ),
        # Regression: batches whose token count is NOT a multiple of the
        # invert-triangular block (16) previously crashed chunked_gdn
        # ("array split does not result in an equal division"). The wrapper now
        # pads the token axis so the chunk stays a valid block multiple.
        ("prefill_nonmult16_40", 1, [40], [0, 40], [0, 1, 1]),
        ("prefill_below_block_10", 1, [10], [0, 10], [0, 1, 1]),
        ("mixed_nonmult16_total40", 3, [1, 1, 38], [0, 1, 2, 40], [2, 3, 3]),
    ]

    def _kernel(self):
        from sglang.srt.layers.attention.pallas.gdn import fused_conv1d_gdn

        return jax.jit(
            fused_conv1d_gdn,
            static_argnames=["n_kq", "n_v", "d_k", "d_v", "kernel_size"],
        )

    def test_kernel_matches_reference(self):
        d_k, d_v = self.KQ_HEAD_DIM, self.V_HEAD_DIM
        n_kq, n_v, K = self.N_KQ, self.N_V, self.KERNEL_SIZE
        kernel = self._kernel()

        for name, max_reqs, lengths, q_loc, distribution in self.CASES:
            with self.subTest(case=name):
                num_tokens = sum(lengths)
                q_loc = jnp.array(q_loc)
                distribution = jnp.array(distribution, dtype=jnp.int32)
                # slot 0 is the reserved null block; real slots start at 1.
                state_indices = jnp.arange(1, max_reqs + 1)
                num_blocks = max_reqs + 1

                rngs = iter(jax.random.split(jax.random.key(0), 12))
                query = jax.random.normal(next(rngs), (num_tokens, n_kq * d_k))
                key = jax.random.normal(next(rngs), (num_tokens, n_kq * d_k))
                value = jax.random.normal(next(rngs), (num_tokens, n_v * d_v))
                b = jax.random.normal(next(rngs), (num_tokens, n_v))
                a = jax.random.normal(next(rngs), (num_tokens, n_v))

                conv_dim = n_kq * d_k * 2 + n_v * d_v
                conv_state = jnp.zeros((num_blocks, K - 1, conv_dim))
                recurrent_state = jnp.zeros((num_blocks, n_v, d_k, d_v))
                conv_weight = jax.random.normal(next(rngs), (conv_dim, 1, K))
                conv_bias = jax.random.normal(next(rngs), (conv_dim,))
                a_log = jax.random.normal(next(rngs), (n_v,))
                dt_bias = jax.random.normal(jax.random.key(0), (n_v,))

                mixed_qkv = jnp.concatenate([query, key, value], axis=-1)
                # Fresh slots: seq_lens == query_lens (context_len == 0).
                seq_lens = jnp.asarray(
                    q_loc[1 : max_reqs + 1] - q_loc[:max_reqs], dtype=jnp.int32
                )

                kwargs = dict(
                    qkv=mixed_qkv,
                    b=b,
                    a=a,
                    conv_state=conv_state,
                    recurrent_state=recurrent_state,
                    conv_weight=conv_weight,
                    conv_bias=conv_bias,
                    a_log=a_log,
                    dt_bias=dt_bias,
                    query_start_loc=q_loc,
                    state_indices=state_indices,
                    distribution=distribution,
                    seq_lens=seq_lens,
                    n_kq=n_kq,
                    n_v=n_v,
                    d_k=d_k,
                    d_v=d_v,
                    kernel_size=K,
                )

                ref_states, ref_out = _gdn_attention_ref(**kwargs)
                (conv_k, rec_k), out_k = kernel(**kwargs)

                np.testing.assert_allclose(out_k, ref_out, rtol=2e-2, atol=2e-2)
                np.testing.assert_allclose(conv_k, ref_states[0], rtol=2e-2, atol=2e-2)
                np.testing.assert_allclose(rec_k, ref_states[1], rtol=2e-2, atol=2e-2)

    def test_has_initial_state_zeros_stale_slot(self):
        """A fresh prefill on a stale (non-zero) slot must match a fresh-zero slot."""
        d_k, d_v = self.KQ_HEAD_DIM, self.V_HEAD_DIM
        n_kq, n_v, K = self.N_KQ, self.N_V, self.KERNEL_SIZE
        kernel = self._kernel()

        max_reqs = 2
        lengths = [64, 64]
        q_loc = jnp.array([0, 64, 128])
        distribution = jnp.array([0, 2, 2], dtype=jnp.int32)
        num_tokens = sum(lengths)
        state_indices = jnp.arange(1, max_reqs + 1)
        num_blocks = max_reqs + 1

        rngs = iter(jax.random.split(jax.random.key(7), 12))
        query = jax.random.normal(next(rngs), (num_tokens, n_kq * d_k))
        key = jax.random.normal(next(rngs), (num_tokens, n_kq * d_k))
        value = jax.random.normal(next(rngs), (num_tokens, n_v * d_v))
        b = jax.random.normal(next(rngs), (num_tokens, n_v))
        a = jax.random.normal(next(rngs), (num_tokens, n_v))

        conv_dim = n_kq * d_k * 2 + n_v * d_v
        conv_zero = jnp.zeros((num_blocks, K - 1, conv_dim))
        rec_zero = jnp.zeros((num_blocks, n_v, d_k, d_v))
        stale_conv = jax.random.normal(next(rngs), (num_blocks, K - 1, conv_dim))
        stale_rec = jax.random.normal(next(rngs), (num_blocks, n_v, d_k, d_v))
        # Slot 0 (null block) stays zero.
        conv_stale = conv_zero.at[1:].set(stale_conv[1:])
        rec_stale = rec_zero.at[1:].set(stale_rec[1:])

        conv_weight = jax.random.normal(next(rngs), (conv_dim, 1, K))
        conv_bias = jax.random.normal(next(rngs), (conv_dim,))
        a_log = jax.random.normal(next(rngs), (n_v,))
        dt_bias = jax.random.normal(next(rngs), (n_v,))
        mixed_qkv = jnp.concatenate([query, key, value], axis=-1)
        seq_lens = jnp.asarray(lengths, dtype=jnp.int32)  # context_len == 0

        common = dict(
            qkv=mixed_qkv,
            b=b,
            a=a,
            conv_weight=conv_weight,
            conv_bias=conv_bias,
            a_log=a_log,
            dt_bias=dt_bias,
            query_start_loc=q_loc,
            state_indices=state_indices,
            distribution=distribution,
            seq_lens=seq_lens,
            n_kq=n_kq,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
            kernel_size=K,
        )

        (conv_f, rec_f), out_f = kernel(
            conv_state=conv_zero, recurrent_state=rec_zero, **common
        )
        (conv_s, rec_s), out_s = kernel(
            conv_state=conv_stale, recurrent_state=rec_stale, **common
        )

        np.testing.assert_allclose(out_f, out_s, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(
            conv_f[1 : max_reqs + 1], conv_s[1 : max_reqs + 1], rtol=1e-5, atol=1e-5
        )
        np.testing.assert_allclose(
            rec_f[1 : max_reqs + 1], rec_s[1 : max_reqs + 1], rtol=1e-5, atol=1e-5
        )

    def test_has_initial_state_preserves_continuation(self):
        """Chunked prefill continued from a slot must match the single-shot prefill."""
        d_k, d_v = self.KQ_HEAD_DIM, self.V_HEAD_DIM
        n_kq, n_v, K = self.N_KQ, self.N_V, self.KERNEL_SIZE
        kernel = self._kernel()

        half, full = 32, 64
        state_indices = jnp.array([1])
        num_blocks = 2

        rngs = iter(jax.random.split(jax.random.key(11), 12))
        query = jax.random.normal(next(rngs), (full, n_kq * d_k))
        key = jax.random.normal(next(rngs), (full, n_kq * d_k))
        value = jax.random.normal(next(rngs), (full, n_v * d_v))
        b = jax.random.normal(next(rngs), (full, n_v))
        a = jax.random.normal(next(rngs), (full, n_v))

        conv_dim = n_kq * d_k * 2 + n_v * d_v
        conv_weight = jax.random.normal(next(rngs), (conv_dim, 1, K))
        conv_bias = jax.random.normal(next(rngs), (conv_dim,))
        a_log = jax.random.normal(next(rngs), (n_v,))
        dt_bias = jax.random.normal(next(rngs), (n_v,))

        mixed = jnp.concatenate([query, key, value], axis=-1)
        conv_zero = jnp.zeros((num_blocks, K - 1, conv_dim))
        rec_zero = jnp.zeros((num_blocks, n_v, d_k, d_v))

        static = dict(
            conv_weight=conv_weight,
            conv_bias=conv_bias,
            a_log=a_log,
            dt_bias=dt_bias,
            state_indices=state_indices,
            n_kq=n_kq,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
            kernel_size=K,
        )

        (_, _), out_ref = kernel(
            qkv=mixed,
            b=b,
            a=a,
            conv_state=conv_zero,
            recurrent_state=rec_zero,
            query_start_loc=jnp.array([0, full]),
            distribution=jnp.array([0, 1, 1], dtype=jnp.int32),
            seq_lens=jnp.array([full], dtype=jnp.int32),
            **static,
        )

        (conv_a, rec_a), out_a = kernel(
            qkv=mixed[:half],
            b=b[:half],
            a=a[:half],
            conv_state=conv_zero,
            recurrent_state=rec_zero,
            query_start_loc=jnp.array([0, half]),
            distribution=jnp.array([0, 1, 1], dtype=jnp.int32),
            seq_lens=jnp.array([half], dtype=jnp.int32),
            **static,
        )

        # context_len = full - half > 0 => has_initial_state = True.
        (_, _), out_b = kernel(
            qkv=mixed[half:],
            b=b[half:],
            a=a[half:],
            conv_state=conv_a,
            recurrent_state=rec_a,
            query_start_loc=jnp.array([0, half]),
            distribution=jnp.array([0, 1, 1], dtype=jnp.int32),
            seq_lens=jnp.array([full], dtype=jnp.int32),
            **static,
        )

        np.testing.assert_allclose(out_a, out_ref[:half], rtol=2e-2, atol=2e-2)
        np.testing.assert_allclose(out_b, out_ref[half:], rtol=2e-2, atol=2e-2)


if __name__ == "__main__":
    unittest.main()
