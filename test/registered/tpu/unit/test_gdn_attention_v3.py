"""Correctness tests for the vendored Pallas fused Conv1D + GDN kernel.

The kernel lives in ``sglang_tpu.attention.pallas.gdn`` (public entry
``fused_conv1d_gdn``). These tests build small ragged batches and compare the
kernel output and updated conv/recurrent state against an in-file pure-JAX
eager reference that runs the gated-delta-rule sequence-by-sequence.

Cases covered (kernel vs. reference, bf16-ish tolerance):

* single long prefill
* mixed multi-sequence prefill
* decode-only batch
* mixed prefill + decode in one call
* padded batches (trailing dummy sequences)

Multi-tile state pipeline regressions, ported with vllm-project/vllm-torchtpu#1927
(a later tile's state load must not overwrite an in-flight checkpoint store):

* CPU interpret-mode DMA race detection (decode + prefill, and a W=8 verify)
* multi-tile batched decode keeps every sequence's state (fp32 / bf16)
* multi-tile W=8 verify windows write the right checkpoints

Plus two state-plumbing invariants:

* ``has_initial_state=False`` masks a stale (non-zero) slot so a fresh
  prefill matches a fresh-zero slot bit-for-bit.
* ``has_initial_state=True`` continues a chunked prefill from the slot's
  state, matching the equivalent single-shot prefill.

The kernel is pure JAX/Pallas and requires a TPU device; tests are skipped
otherwise, except the race test, which runs in CPU interpret mode. They do **not** depend on torch_tpu or the SGLang server runtime.

Adapted from torchtpu-vllm ``tests/kernels/test_gdn_attention_v3.py``.

Usage:
    python test/registered/tpu/unit/test_gdn_attention_v3.py
"""

from __future__ import annotations

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=100, suite="stage-a-test-tpu-unit", chips=1)

import unittest

from sglang.test.test_utils import CustomTestCase


def _has_jax() -> bool:
    try:
        import jax  # noqa: F401
    except Exception:
        return False
    return True


def _has_jax_tpu() -> bool:
    if not _has_jax():
        return False
    import jax

    try:
        return any("TPU" in d.device_kind for d in jax.devices())
    except Exception:
        return False


_SKIP_REASON = "Requires JAX with a TPU device"


if _has_jax():
    import sys

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


def gdn_attention_spec_ref(
    qkv: jnp.ndarray,
    b: jnp.ndarray,
    a: jnp.ndarray,
    conv_state: jnp.ndarray,
    recurrent_state: jnp.ndarray,
    conv_weight: jnp.ndarray,
    conv_bias: jnp.ndarray | None,
    a_log: jnp.ndarray,
    dt_bias: jnp.ndarray,
    query_start_loc: jnp.ndarray,
    state_indices: jnp.ndarray,
    read_offsets: jnp.ndarray,
    num_spec_seqs: int,
    seq_lens: jnp.ndarray,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    kernel_size: int,
) -> tuple[tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """Reference for SPEC mode: per-token state checkpoints with read offsets.

    Sequence s reads its initial state from slot
    `state_indices[s] + read_offsets[s]` and writes the state after window
    position t to slot `state_indices[s] + t`.
    """
    num_tokens = qkv.shape[0]
    new_conv_state = jnp.array(conv_state)
    new_recurrent_state = jnp.array(recurrent_state)
    output = jnp.zeros((num_tokens, n_v * d_v), dtype=qkv.dtype)

    for req_idx in range(num_spec_seqs):
        base = int(state_indices[req_idx])
        read_slot = base + int(read_offsets[req_idx])
        start = int(query_start_loc[req_idx])
        end = int(query_start_loc[req_idx + 1])
        query_len = end - start
        if query_len <= 0:
            continue

        has_init = bool((seq_lens[req_idx] - query_len) > 0)
        c_state = (
            conv_state[read_slot] if has_init else jnp.zeros_like(conv_state[read_slot])
        )
        r_state = (
            recurrent_state[read_slot]
            if has_init
            else jnp.zeros_like(recurrent_state[read_slot])
        )

        # Conv1D with per-token windows.
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
        conv_out = jax.nn.silu(acc).astype(qkv.dtype)
        for t in range(query_len):
            new_conv_state = new_conv_state.at[base + t].set(
                x_full[t + 1 : t + kernel_size]
            )

        # Recurrent GDN with per-token states.
        key_dim = n_kq * d_k
        q_seq = conv_out[:, :key_dim].reshape(query_len, n_kq, d_k)
        k_seq = conv_out[:, key_dim : key_dim * 2].reshape(query_len, n_kq, d_k)
        v_seq = conv_out[:, key_dim * 2 :].reshape(query_len, n_v, d_v)

        repeat_factor = n_v // n_kq
        if repeat_factor > 1:
            q_seq = jnp.repeat(q_seq, repeat_factor, axis=1)
            k_seq = jnp.repeat(k_seq, repeat_factor, axis=1)

        q_seq = _l2_normalize(q_seq)
        k_seq = _l2_normalize(k_seq)
        v_seq = v_seq.astype(jnp.float32)

        beta_seq = jax.nn.sigmoid(b[start:end].astype(jnp.float32))
        g_seq = -jnp.exp(a_log.astype(jnp.float32))[None, :] * jax.nn.softplus(
            a[start:end].astype(jnp.float32) + dt_bias.astype(jnp.float32)[None, :]
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
            new_state = new_state.astype(recurrent_state.dtype)

            return new_state, (out_step.astype(qkv.dtype), new_state)

        _, (out_seq, states_seq) = jax.lax.scan(
            step_fn, r_state, (q_seq, k_seq, v_seq, beta_seq, g_seq)
        )
        for t in range(query_len):
            new_recurrent_state = new_recurrent_state.at[base + t].set(states_seq[t])
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
        from sglang_tpu.attention.pallas.gdn import fused_conv1d_gdn

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

    def test_spec_mode_checkpoints(self):
        for lengths, offsets, prefills in [
            ([5, 3, 1], [2, 0, 4], []),
            ([5, 1, 3, 2, 4], [0, 1, 2, 3, 4], []),
            ([5, 2], [3, 1], [64]),
        ]:
            with self.subTest(lengths=lengths, offsets=offsets, prefills=prefills):
                self._check_spec_mode_checkpoints(lengths, offsets, prefills)

    def _check_spec_mode_checkpoints(self, spec_lengths, read_offsets, prefill_lengths):
        """SPEC mode: initial state comes from `base + read_offset` and one
        state checkpoint per window position lands at `base + t`, matching a
        token-by-token eager reference. Prefill sequences in the same batch
        keep the original PER_SEQ behavior (read/write the base slot)."""
        from sglang_tpu.attention.pallas.gdn import wrapper

        kq_head_dim = 128
        v_head_dim = 128
        n_kq = 2
        n_v = 8
        kernel_size = 4
        num_spec_tokens = 4
        window = num_spec_tokens + 1

        lengths = list(spec_lengths) + list(prefill_lengths)
        num_seqs = len(lengths)
        num_spec_seqs = len(spec_lengths)
        num_tokens = sum(lengths)
        q_loc = jnp.array(np.concatenate([[0], np.cumsum(lengths)]), dtype=jnp.int32)
        distribution = jnp.array(
            [num_spec_seqs, num_spec_seqs, num_seqs], dtype=jnp.int32
        )

        # Slot groups of `window` consecutive slots per sequence; slot 0 is
        # the null block.
        state_indices = jnp.array(
            [1 + i * window for i in range(num_seqs)], dtype=jnp.int32
        )
        num_blocks = 1 + num_seqs * window
        read_offsets_arr = jnp.array(
            list(read_offsets) + [0] * len(prefill_lengths), dtype=jnp.int32
        )

        rngs = iter(jax.random.split(jax.random.key(3), 12))
        conv_dim = (n_kq * kq_head_dim) * 2 + n_v * v_head_dim

        mixed_qkv = jax.random.normal(next(rngs), (num_tokens, conv_dim))
        b = jax.random.normal(next(rngs), (num_tokens, n_v))
        a = jax.random.normal(next(rngs), (num_tokens, n_v))

        # Every slot holds a distinct random state so a read from the wrong
        # slot (or a write to the wrong slot) is caught.
        conv_state = jax.random.normal(
            next(rngs), (num_blocks, kernel_size - 1, conv_dim)
        )
        recurrent_state = jax.random.normal(
            next(rngs), (num_blocks, n_v, kq_head_dim, v_head_dim)
        )

        conv_weight = jax.random.normal(next(rngs), (conv_dim, 1, kernel_size))
        conv_bias = jax.random.normal(next(rngs), (conv_dim,))
        A_log = jax.random.normal(next(rngs), (n_v,))
        dt_bias = jax.random.normal(next(rngs), (n_v,))

        # All sequences continue an existing context (has_initial=True) for
        # the spec windows; prefills start fresh (context_len = 0).
        seq_lens = jnp.array(
            [16 + sl for sl in spec_lengths] + list(prefill_lengths), dtype=jnp.int32
        )

        run_jitted = jax.jit(
            wrapper.fused_conv1d_gdn,
            static_argnames=[
                "n_kq",
                "n_v",
                "d_k",
                "d_v",
                "kernel_size",
                "num_spec_tokens",
            ],
        )

        (new_conv, new_rec), output = run_jitted(
            mixed_qkv,
            b,
            a,
            conv_state,
            recurrent_state,
            conv_weight,
            conv_bias,
            A_log,
            dt_bias,
            q_loc,
            state_indices,
            distribution,
            seq_lens,
            read_offsets_arr,
            n_kq=n_kq,
            n_v=n_v,
            d_k=kq_head_dim,
            d_v=v_head_dim,
            kernel_size=kernel_size,
            num_spec_tokens=num_spec_tokens,
        )

        # Reference: spec windows token-by-token with checkpoints.
        (ref_conv, ref_rec), ref_output = gdn_attention_spec_ref(
            qkv=mixed_qkv,
            b=b,
            a=a,
            conv_state=conv_state,
            recurrent_state=recurrent_state,
            conv_weight=conv_weight,
            conv_bias=conv_bias,
            a_log=A_log,
            dt_bias=dt_bias,
            query_start_loc=q_loc,
            state_indices=state_indices,
            read_offsets=read_offsets_arr,
            num_spec_seqs=num_spec_seqs,
            seq_lens=seq_lens,
            n_kq=n_kq,
            n_v=n_v,
            d_k=kq_head_dim,
            d_v=v_head_dim,
            kernel_size=kernel_size,
        )
        # Reference for the prefill tail: original eager reference over the
        # same buffers (spec sequences excluded via the distribution).
        if prefill_lengths:
            spec_token_end = int(q_loc[num_spec_seqs])
            (ref_conv, ref_rec), prefill_output = _gdn_attention_ref(
                qkv=mixed_qkv[spec_token_end:],
                b=b[spec_token_end:],
                a=a[spec_token_end:],
                conv_state=ref_conv,
                recurrent_state=ref_rec,
                conv_weight=conv_weight,
                conv_bias=conv_bias,
                a_log=A_log,
                dt_bias=dt_bias,
                query_start_loc=q_loc[num_spec_seqs:] - q_loc[num_spec_seqs],
                state_indices=state_indices[num_spec_seqs:],
                distribution=jnp.array(
                    [0, 0, num_seqs - num_spec_seqs], dtype=jnp.int32
                ),
                seq_lens=seq_lens[num_spec_seqs:],
                n_kq=n_kq,
                n_v=n_v,
                d_k=kq_head_dim,
                d_v=v_head_dim,
                kernel_size=kernel_size,
            )
            ref_output = ref_output.at[spec_token_end:].set(prefill_output)

        np.testing.assert_allclose(output, ref_output, rtol=2e-2, atol=2e-2)

        # Verify every written checkpoint slot; untouched slots must keep
        # their original contents.
        touched = set()
        for i, sl in enumerate(spec_lengths):
            base = int(state_indices[i])
            touched.update(range(base, base + sl))
        for i in range(len(prefill_lengths)):
            touched.add(int(state_indices[num_spec_seqs + i]))
        for slot in range(num_blocks):
            expected_conv = ref_conv[slot] if slot in touched else conv_state[slot]
            expected_rec = ref_rec[slot] if slot in touched else (recurrent_state[slot])
            np.testing.assert_allclose(
                new_conv[slot],
                expected_conv,
                rtol=2e-2,
                atol=2e-2,
                err_msg=f"conv checkpoint mismatch at slot {slot}",
            )
            np.testing.assert_allclose(
                new_rec[slot],
                expected_rec,
                rtol=2e-2,
                atol=2e-2,
                err_msg=f"recurrent checkpoint mismatch at slot {slot}",
            )

    def test_batched_decode_initial_state_parity(self):
        """Multi-tile batched decode keeps every sequence's own state."""
        from sglang_tpu.attention.pallas.gdn import wrapper

        n_kq, n_v, d_k, d_v, kernel_size = 2, 8, 128, 128, 4
        dim = 2 * n_kq * d_k + n_v * d_v
        num_seqs = 256
        decode_tile = 4

        # A distinct value per slot so a cross-slot copy is identifiable; the
        # gates below keep the state unchanged, so output state == input state.
        block_ids = jnp.arange(1, num_seqs + 1, dtype=jnp.float32)
        init_conv = jnp.broadcast_to(
            block_ids[:, None, None] * jnp.float32(1e-2),
            (num_seqs, kernel_size - 1, dim),
        ).astype(jnp.bfloat16)
        qkv = jnp.zeros((num_seqs, dim), dtype=jnp.bfloat16)
        b = jnp.full((num_seqs, n_v), -20.0, dtype=jnp.bfloat16)
        a = jnp.full((num_seqs, n_v), -20.0, dtype=jnp.bfloat16)
        conv_weight = jnp.zeros((dim, 1, kernel_size), dtype=jnp.bfloat16)
        conv_bias = jnp.zeros((dim,), dtype=jnp.bfloat16)
        a_log = jnp.full((n_v,), -20.0, dtype=jnp.float32)
        dt_bias = jnp.full((n_v,), -20.0, dtype=jnp.float32)
        kwargs = dict(
            qkv=qkv,
            b=b,
            a=a,
            conv_weight=conv_weight,
            conv_bias=conv_bias,
            a_log=a_log,
            dt_bias=dt_bias,
            query_start_loc=jnp.arange(num_seqs + 1, dtype=jnp.int32),
            state_indices=jnp.arange(num_seqs, dtype=jnp.int32),
            distribution=jnp.array([num_seqs] * 3, dtype=jnp.int32),
            seq_lens=jnp.full((num_seqs,), 11, dtype=jnp.int32),
            n_kq=n_kq,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
            kernel_size=kernel_size,
        )

        for state_dtype in (jnp.float32, jnp.bfloat16):
            init_rec = jnp.broadcast_to(
                block_ids[:, None, None, None] * jnp.float32(1e-2),
                (num_seqs, n_v, d_k, d_v),
            ).astype(state_dtype)
            (conv_ref, rec_ref), out_ref = _gdn_attention_ref(
                conv_state=init_conv, recurrent_state=init_rec, **kwargs
            )
            for rep in range(3):
                with self.subTest(state_dtype=jnp.dtype(state_dtype).name, rep=rep):
                    (conv_out, rec_out), out = wrapper.fused_conv1d_gdn(
                        conv_state=init_conv.copy(),
                        recurrent_state=init_rec.copy(),
                        decode_tile_size=decode_tile,
                        **kwargs,
                    )
                    rec_np = np.asarray(rec_out, dtype=np.float32)
                    ref_np = np.asarray(rec_ref, dtype=np.float32)
                    bad = np.where(
                        np.max(np.abs(rec_np - ref_np), axis=(1, 2, 3)) > 1e-5
                    )[0]
                    self.assertEqual(
                        bad.size, 0, f"corrupted recurrent slots: {bad.tolist()}"
                    )
                    np.testing.assert_allclose(conv_out, conv_ref, rtol=1e-5, atol=1e-5)
                    np.testing.assert_allclose(out, out_ref, rtol=1e-5, atol=1e-5)

    def test_spec_window_checkpoints_multi_tile(self):
        """W=8 verify over several multi-sequence tiles writes each sequence's
        own checkpoints, including checkpoint 0."""
        from sglang_tpu.attention.pallas.gdn import wrapper

        n_kq, n_v, d_k, d_v, kernel_size = 2, 4, 128, 128, 4
        dim = 2 * n_kq * d_k + n_v * d_v
        window = 8
        num_seqs = 16
        batch_size = num_seqs * window
        num_blocks = num_seqs * window + 8

        rngs = iter(jax.random.split(jax.random.key(314), 10))
        qkv = jax.random.normal(next(rngs), (batch_size, dim), dtype=jnp.bfloat16)
        b = jax.random.normal(next(rngs), (batch_size, n_v), dtype=jnp.bfloat16)
        a = jax.random.normal(next(rngs), (batch_size, n_v), dtype=jnp.bfloat16)
        conv_w = jax.random.normal(
            next(rngs), (dim, 1, kernel_size), dtype=jnp.bfloat16
        )
        conv_b = jax.random.normal(next(rngs), (dim,), dtype=jnp.bfloat16)
        a_log = jax.random.normal(next(rngs), (n_v,), dtype=jnp.float32)
        dt_bias = jax.random.normal(next(rngs), (n_v,), dtype=jnp.float32)
        init_conv = jax.random.normal(
            next(rngs), (num_blocks, kernel_size - 1, dim), dtype=jnp.bfloat16
        )
        init_rec = jax.random.normal(
            next(rngs), (num_blocks, n_v, d_k, d_v), dtype=jnp.float32
        )

        q_loc = jnp.arange(0, batch_size + 1, window, dtype=jnp.int32)
        state_indices = jnp.arange(num_seqs, dtype=jnp.int32) * window
        read_offsets = jnp.array([s % window for s in range(num_seqs)], jnp.int32)
        seq_lens = jnp.full((num_seqs,), window + 10, dtype=jnp.int32)

        (conv_ref, rec_ref), out_ref = gdn_attention_spec_ref(
            qkv=qkv,
            b=b,
            a=a,
            conv_state=init_conv,
            recurrent_state=init_rec,
            conv_weight=conv_w,
            conv_bias=conv_b,
            a_log=a_log,
            dt_bias=dt_bias,
            query_start_loc=q_loc,
            state_indices=state_indices,
            read_offsets=read_offsets,
            num_spec_seqs=num_seqs,
            seq_lens=seq_lens,
            n_kq=n_kq,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
            kernel_size=kernel_size,
        )

        for rep in range(3):
            (conv_out, rec_out), out = wrapper.fused_conv1d_gdn(
                qkv=qkv,
                b=b,
                a=a,
                conv_state=init_conv.copy(),
                recurrent_state=init_rec.copy(),
                conv_weight=conv_w,
                conv_bias=conv_b,
                a_log=a_log,
                dt_bias=dt_bias,
                query_start_loc=q_loc,
                state_indices=state_indices,
                distribution=jnp.array([num_seqs] * 3, dtype=jnp.int32),
                seq_lens=seq_lens,
                read_offsets=read_offsets,
                n_kq=n_kq,
                n_v=n_v,
                d_k=d_k,
                d_v=d_v,
                kernel_size=kernel_size,
                num_spec_tokens=window - 1,
                decode_tile_size=4,
            )
            np.testing.assert_allclose(out, out_ref, rtol=1e-2, atol=1e-2)
            for s in range(num_seqs):
                base = int(state_indices[s])
                for name, got, want in (
                    ("conv", conv_out, conv_ref),
                    ("recurrent", rec_out, rec_ref),
                ):
                    np.testing.assert_allclose(
                        got[base : base + window],
                        want[base : base + window],
                        rtol=1e-2,
                        atol=1e-2,
                        err_msg=f"rep={rep} seq={s} {name} checkpoints",
                    )


@unittest.skipIf(not _has_jax(), "Requires JAX")
class TestGDNStatePipelineRace(CustomTestCase):
    """CPU interpret-mode vector-clock race detection on the state pipeline."""

    def _races_per_call(self, num_spec_tokens: int) -> list[bool]:
        from jax.experimental.pallas import tpu as pltpu
        from jax.sharding import AbstractDevice, AbstractMesh, use_abstract_mesh

        from sglang_tpu.attention.pallas.gdn import wrapper

        n_kq, n_v, d_k, d_v, kernel_size = 1, 2, 128, 128, 4
        dim = 2 * n_kq * d_k + n_v * d_v
        window = num_spec_tokens + 1
        # 6 decode/verify sequences at tile 2 (step 2 reuses step 0's slot),
        # then 3 one-token PER_SEQ tiles that do the same.
        num_decode, num_seqs = 6, 9
        lengths = [window] * num_decode + [1] * (num_seqs - num_decode)
        num_tokens = sum(lengths)
        num_blocks = num_seqs * window

        cpu = jax.devices("cpu")[0]
        with jax.default_device(cpu):
            rngs = iter(jax.random.split(jax.random.key(0), 9))
            qkv = jax.random.normal(next(rngs), (num_tokens, dim), jnp.bfloat16)
            b = jax.random.normal(next(rngs), (num_tokens, n_v), jnp.bfloat16)
            a = jax.random.normal(next(rngs), (num_tokens, n_v), jnp.bfloat16)
            conv_w = jax.random.normal(next(rngs), (dim, 1, kernel_size), jnp.bfloat16)
            conv_b = jax.random.normal(next(rngs), (dim,), jnp.bfloat16)
            a_log = jax.random.normal(next(rngs), (n_v,), jnp.float32)
            dt_bias = jax.random.normal(next(rngs), (n_v,), jnp.float32)
            init_conv = jax.random.normal(
                next(rngs), (num_blocks, kernel_size - 1, dim), jnp.bfloat16
            )
            init_rec = jax.random.normal(
                next(rngs), (num_blocks, n_v, d_k, d_v), jnp.float32
            )
            q_loc = jnp.array(np.concatenate([[0], np.cumsum(lengths)]), jnp.int32)
            read_offsets = None
            if num_spec_tokens:
                read_offsets = jnp.array(
                    [s % window for s in range(num_seqs)], jnp.int32
                )

        # jax has no InterpretParams.on_race; read each pallas_call's verdict
        # before its clean-up barrier drops the shared state.
        mosaic_interpret = sys.modules[pltpu.reset_tpu_interpret_mode_state.__module__]
        races_per_call: list[bool] = []
        orig_clear = mosaic_interpret._clear_shared_memory

        def _record_and_clear():
            races = mosaic_interpret.races
            if races is not None:
                races_per_call.append(bool(races.races_found))
            orig_clear()

        mesh = AbstractMesh(
            (), (), (), abstract_device=AbstractDevice("TPU7x", 1, "tpu")
        )
        pltpu.reset_tpu_interpret_mode_state()
        mosaic_interpret._clear_shared_memory = _record_and_clear
        try:
            with (
                jax.default_device(cpu),
                use_abstract_mesh(mesh),
                pltpu.force_tpu_interpret_mode(
                    pltpu.InterpretParams(
                        detect_races=True,
                        dma_execution_mode="eager",
                        skip_floating_point_ops=False,
                        vector_clock_size=128,
                    )
                ),
            ):
                (conv_out, rec_out), out = wrapper.fused_conv1d_gdn(
                    qkv=qkv,
                    b=b,
                    a=a,
                    conv_state=init_conv,
                    recurrent_state=init_rec,
                    conv_weight=conv_w,
                    conv_bias=conv_b,
                    a_log=a_log,
                    dt_bias=dt_bias,
                    query_start_loc=q_loc,
                    state_indices=jnp.arange(num_seqs, dtype=jnp.int32) * window,
                    distribution=jnp.array(
                        [num_decode, num_seqs, num_seqs], dtype=jnp.int32
                    ),
                    seq_lens=jnp.full((num_seqs,), 11, dtype=jnp.int32),
                    read_offsets=read_offsets,
                    n_kq=n_kq,
                    n_v=n_v,
                    d_k=d_k,
                    d_v=d_v,
                    kernel_size=kernel_size,
                    num_spec_tokens=num_spec_tokens,
                    decode_tile_size=2,
                    # Interpret mode cannot reshape refs on the C > 1 chunked
                    # path; one-token prefills run at C = 1.
                    mixed_tile_size=1,
                )
                jax.block_until_ready((conv_out, rec_out, out))
        finally:
            mosaic_interpret._clear_shared_memory = orig_clear
            pltpu.reset_tpu_interpret_mode_state()
        return races_per_call

    def test_state_pipeline_has_no_dma_race(self):
        for num_spec_tokens in (0, 7):
            with self.subTest(num_spec_tokens=num_spec_tokens):
                races = self._races_per_call(num_spec_tokens)
                # One verdict per pallas_call: the batched and PER_SEQ kernels.
                self.assertEqual(len(races), 2, races)
                self.assertFalse(any(races), f"DMA race per pallas_call: {races}")


if __name__ == "__main__":
    unittest.main()
