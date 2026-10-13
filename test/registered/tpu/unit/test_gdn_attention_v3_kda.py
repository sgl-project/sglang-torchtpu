"""Correctness tests for the KDA mode of the fused Conv1D + GDN kernel.

The kernel lives in ``sglang_tpu.attention.pallas.gdn`` (public entry
``fused_conv1d_gdn`` with ``attention_mode=AttentionMode.KDA``). The KDA gate is
per-channel, ``kq_head_dim`` times wider than GDN's per-head scalar, so it takes
its own BlockSpec, VMEM load and gate activation.

The oracle is a float64 token-at-a-time recurrence, preceded by the conv1d and
silu the fused kernel folds in. Both gate forms are covered:
``gate_lower_bound=None`` gives ``-exp(A_log) * softplus(g + dt_bias)``
(Kimi-Linear); a float gives ``gate_lower_bound * sigmoid(exp(A_log) *
(g + dt_bias))`` (Kimi-K3, at -5.0). Decode runs the token recurrence and
prefill the chunked solve; cases cover fresh slots, slots continuing a
context (stale state must be ignored when the context is empty), padded
buckets (trailing empty sequence rows and tokens, as TPU serving pads them),
``prefill_only``, and speculative verify windows (per-token checkpoints,
resumed from a checkpoint offset).

Requires a TPU device; skipped otherwise. Adapted from vllm-torchtpu
``tests/kernels/test_gdn_attention_v3_kda.py``.
"""

from __future__ import annotations

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=180, suite="stage-a-test-tpu-unit", chips=1)

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


def _l2_normalize(x):
    """Mirror the kernel's ``_l2_norm_f32``, which accumulates in float32."""
    x_f32 = x.astype(np.float32)
    inv_norm = 1.0 / np.sqrt(np.sum(x_f32 * x_f32, axis=-1, keepdims=True) + 1e-6)
    return (x_f32 * inv_norm).astype(np.float64)


def _activate_gate(a_seq, a_log_np, dt_bias_np, gate_lower_bound):
    if gate_lower_bound is None:
        return -np.exp(a_log_np) * np.logaddexp(0.0, a_seq + dt_bias_np)
    # sigmoid(x) written as exp(-logaddexp(0, -x)) to avoid overflow in the
    # exponential for large negative x.
    return gate_lower_bound * np.exp(
        -np.logaddexp(0.0, -(np.exp(a_log_np) * (a_seq + dt_bias_np)))
    )


def _recurrence(q, k, v, gate, beta, state, scale):
    """One token at a time, in float64, in the kernel's token-major layout.

    q/k/gate are [T, H, K], v is [T, H, V], beta is [T, H], state is [H, K, V].
    """
    running = state.astype(np.float64).copy()
    output = np.zeros((q.shape[0], q.shape[1], v.shape[-1]), np.float64)
    for token in range(q.shape[0]):
        # The per-channel gate decays along K, hence the trailing axis.
        running *= np.exp(gate[token])[:, :, None]
        prediction = np.einsum("hk,hkv->hv", k[token], running)
        delta = beta[token][:, None] * (v[token] - prediction)
        running += k[token][:, :, None] * delta[:, None, :]
        output[token] = scale * np.einsum("hk,hkv->hv", q[token], running)
    return output, running


def _conv1d(x_full, conv_weight, conv_bias, query_len, kernel_size):
    acc = jnp.zeros((query_len, x_full.shape[-1]), dtype=jnp.float32)
    for k_idx in range(kernel_size):
        acc += (
            x_full[k_idx : k_idx + query_len].astype(jnp.float32)
            * conv_weight[:, 0, k_idx].astype(jnp.float32)[None, :]
        )
    if conv_bias is not None:
        acc += conv_bias.astype(jnp.float32)[None, :]
    return acc


def _split_qkv(qkv_seq, query_len, n_kq, n_v, d_k, d_v):
    key_dim = n_kq * d_k
    q_seq = qkv_seq[:, :key_dim].reshape(query_len, n_kq, d_k)
    k_seq = qkv_seq[:, key_dim : 2 * key_dim].reshape(query_len, n_kq, d_k)
    v_seq = qkv_seq[:, 2 * key_dim :].reshape(query_len, n_v, d_v)
    repeat_factor = n_v // n_kq
    if repeat_factor > 1:
        q_seq = np.repeat(q_seq, repeat_factor, axis=1)
        k_seq = np.repeat(k_seq, repeat_factor, axis=1)
    return q_seq, k_seq, v_seq


def _kda_attention_ref(
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
    gate_lower_bound: float | None = None,
):
    """Reference conv1d + silu + KDA, sequence by sequence in eager mode."""
    num_tokens = qkv.shape[0]
    num_valid_seqs = int(distribution[2])

    out_mixed_qkv = jnp.zeros_like(qkv)
    new_conv_state = jnp.array(conv_state)
    new_recurrent_state = jnp.array(recurrent_state)
    output = np.zeros((num_tokens, n_v * d_v), np.float64)

    # Part 1: Conv1D, over the raw activations.
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

        x_full = jnp.concatenate([c_state, qkv[start:end]], axis=0)
        acc = _conv1d(x_full, conv_weight, conv_bias, query_len, kernel_size)
        new_conv_state = new_conv_state.at[s].set(x_full[-(kernel_size - 1) :])
        out_mixed_qkv = out_mixed_qkv.at[start:end].set(acc.astype(qkv.dtype))

    out_mixed_qkv = jax.nn.silu(out_mixed_qkv)

    # Part 2: KDA, in float64.
    a_log_np = np.asarray(a_log, np.float64).reshape(n_v, 1)
    dt_bias_np = np.asarray(dt_bias, np.float64).reshape(n_v, d_k)
    scale = float(d_k**-0.5)

    for req_idx in range(num_valid_seqs):
        s = int(state_indices[req_idx])
        start = int(query_start_loc[req_idx])
        end = int(query_start_loc[req_idx + 1])
        query_len = end - start
        if query_len <= 0:
            continue

        qkv_seq = np.asarray(out_mixed_qkv[start:end], np.float64)
        q_seq, k_seq, v_seq = _split_qkv(qkv_seq, query_len, n_kq, n_v, d_k, d_v)
        q_seq = _l2_normalize(q_seq)
        k_seq = _l2_normalize(k_seq)

        beta_seq = 1.0 / (1.0 + np.exp(-np.asarray(b[start:end], np.float64)))
        # The gate is per (head, channel): [T, H, K].
        a_seq = np.asarray(a[start:end], np.float64).reshape(query_len, n_v, d_k)
        gate_seq = _activate_gate(a_seq, a_log_np, dt_bias_np, gate_lower_bound)

        out_seq, final_state = _recurrence(
            q_seq,
            k_seq,
            v_seq,
            gate_seq,
            beta_seq,
            np.asarray(new_recurrent_state[s], np.float64),
            scale,
        )
        new_recurrent_state = new_recurrent_state.at[s].set(
            jnp.asarray(final_state, new_recurrent_state.dtype)
        )
        output[start:end] = out_seq.reshape(query_len, n_v * d_v)

    return (new_conv_state, new_recurrent_state), output


def _kda_attention_spec_ref(
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
    read_offsets,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    kernel_size: int,
    gate_lower_bound: float | None,
):
    """Token reference for verify windows: read from ``base + read_offset``,
    checkpoint conv and recurrent state after token t at ``base + t``."""
    new_conv_state = jnp.array(conv_state)
    new_recurrent_state = jnp.array(recurrent_state)
    output = np.zeros((qkv.shape[0], n_v * d_v), np.float64)
    a_log_np = np.asarray(a_log, np.float64).reshape(n_v, 1)
    dt_bias_np = np.asarray(dt_bias, np.float64).reshape(n_v, d_k)
    scale = float(d_k**-0.5)

    for req_idx in range(int(distribution[0])):
        base = int(state_indices[req_idx])
        read_slot = base + int(read_offsets[req_idx])
        start = int(query_start_loc[req_idx])
        end = int(query_start_loc[req_idx + 1])
        query_len = end - start
        if query_len <= 0:
            continue

        has_init = bool((seq_lens[req_idx] - query_len) > 0)
        c_state = conv_state[read_slot]
        running = np.asarray(recurrent_state[read_slot], np.float64)
        if not has_init:
            c_state = jnp.zeros_like(c_state)
            running = np.zeros_like(running)

        x_full = jnp.concatenate([c_state, qkv[start:end]], axis=0)
        acc = _conv1d(x_full, conv_weight, conv_bias, query_len, kernel_size)
        # silu and beta run in float32 and are stored in the activation dtype.
        conv_out = np.asarray(jax.nn.silu(acc).astype(qkv.dtype), np.float64)
        q_seq, k_seq, v_seq = _split_qkv(conv_out, query_len, n_kq, n_v, d_k, d_v)
        q_seq = _l2_normalize(q_seq)
        k_seq = _l2_normalize(k_seq)
        beta_seq = np.asarray(
            jax.nn.sigmoid(b[start:end].astype(jnp.float32)).astype(b.dtype),
            np.float64,
        )
        a_seq = np.asarray(a[start:end], np.float64).reshape(query_len, n_v, d_k)
        gate_seq = _activate_gate(a_seq, a_log_np, dt_bias_np, gate_lower_bound)

        for t in range(query_len):
            step = slice(t, t + 1)
            out_t, running = _recurrence(
                q_seq[step],
                k_seq[step],
                v_seq[step],
                gate_seq[step],
                beta_seq[step],
                running,
                scale,
            )
            output[start + t] = out_t.reshape(-1)
            new_conv_state = new_conv_state.at[base + t].set(
                x_full[t + 1 : t + kernel_size]
            )
            new_recurrent_state = new_recurrent_state.at[base + t].set(
                jnp.asarray(running, recurrent_state.dtype)
            )

    return (new_conv_state, new_recurrent_state), jnp.asarray(output, qkv.dtype)


# (name, query lens, context lens, distribution, extra); context 0 is a fresh
# slot. extra: n_v, and pad_seqs / pad_tokens for empty rows past the real
# sequences, which carry the null slot 0 and zero length.
_SHAPES = [
    ("decode_only", [1] * 8, [0] * 8, [8, 8, 8], {}),
    ("decode_single", [1], [5], [1, 1, 1], {}),
    ("decode_with_context", [1] * 8, [3, 17, 64, 1, 250, 7, 33, 2], [8, 8, 8], {}),
    # 3 real sequences in a 4-row decode bucket.
    ("padded_decode", [1] * 3, [5, 9, 2], [3, 3, 3], dict(pad_seqs=1, pad_tokens=1)),
    # 40 is not a multiple of the 16-row triangular block.
    ("prefill_nonmult16_40", [40], [0], [0, 1, 1], {}),
    ("mixed_prefill", [192, 128, 64], [0, 0, 0], [0, 3, 3], {}),
    # 65 crosses a 64-token tile; the 1-token row is a prefill, not a decode.
    ("ragged_prefill", [65, 65, 1, 64], [0] * 4, [0, 4, 4], {}),
    ("prefill_long", [1000], [0], [0, 1, 1], {}),
    ("prefill_with_context", [100], [50], [0, 1, 1], {}),
    ("decode_and_prefill", [1, 1, 38], [5, 9, 0], [2, 3, 3], {}),
    # 2 real prefills in a 4-row, 128-token bucket.
    (
        "padded_prefill",
        [40, 24],
        [0, 0],
        [0, 2, 2],
        dict(pad_seqs=2, pad_tokens=64),
    ),
    # Kimi-Linear-48B's 32 heads use a different triangular sub-block (8).
    ("kimi_linear_heads", [192, 64], [0, 0], [0, 2, 2], dict(n_v=32)),
]

# Kimi-Linear sets no bound; Kimi-K3 sets -5.0.
_GATES = [("softplus", None), ("bounded", -5.0)]


@unittest.skipIf(not _has_jax_tpu(), _SKIP_REASON)
class TestGDNAttentionV3KDA(CustomTestCase):
    """Compare the KDA mode of the fused kernel against a float64 recurrence."""

    # K3 and Kimi-Linear both run KDA without GQA, so n_kq == n_v.
    D_K = 128
    D_V = 128
    KERNEL_SIZE = 4

    def _kernel(self):
        from sglang_tpu.attention.pallas.gdn import fused_conv1d_gdn

        return fused_conv1d_gdn

    def _inputs(
        self,
        query_lens,
        context_lens,
        distribution,
        gate_lower_bound,
        *,
        n_v=4,
        pad_seqs=0,
        pad_tokens=0,
    ):
        n_kq = n_v
        d_k, d_v, kernel_size = self.D_K, self.D_V, self.KERNEL_SIZE
        num_seqs = len(query_lens)
        ends = np.cumsum([0] + query_lens)
        q_loc = jnp.array(list(ends) + [ends[-1]] * pad_seqs, dtype=jnp.int32)
        num_tokens = int(ends[-1]) + pad_tokens
        dim_size = 2 * n_kq * d_k + n_v * d_v
        # Slot 0 is the reserved null block for invalid / padded tokens.
        num_blocks = num_seqs + 1

        rngs = iter(jax.random.split(jax.random.key(0), 10))
        # Every slot starts non-zero: fresh sequences must ignore it.
        conv_state = jax.random.normal(
            next(rngs), (num_blocks, kernel_size - 1, dim_size)
        )
        recurrent_state = 0.1 * jax.random.normal(
            next(rngs), (num_blocks, n_v, d_k, d_v)
        )
        return dict(
            qkv=jax.random.normal(next(rngs), (num_tokens, dim_size)),
            b=jax.random.normal(next(rngs), (num_tokens, n_v)),
            # The KDA gate is per-channel: n_v * d_k wide, not n_v.
            a=jax.random.normal(next(rngs), (num_tokens, n_v * d_k)),
            conv_state=conv_state,
            recurrent_state=recurrent_state,
            conv_weight=jax.random.normal(next(rngs), (dim_size, 1, kernel_size)),
            conv_bias=jax.random.normal(next(rngs), (dim_size,)),
            a_log=jax.random.normal(next(rngs), (n_v,)),
            # Per-channel and kept 2D: Mosaic will not broadcast a flat vector
            # over (head, channel).
            dt_bias=jax.random.normal(next(rngs), (n_v, d_k)),
            query_start_loc=q_loc,
            state_indices=jnp.array(
                list(range(1, num_seqs + 1)) + [0] * pad_seqs, dtype=jnp.int32
            ),
            distribution=jnp.array(distribution, dtype=jnp.int32),
            seq_lens=jnp.array(
                [q + c for q, c in zip(query_lens, context_lens)] + [0] * pad_seqs,
                dtype=jnp.int32,
            ),
            n_kq=n_kq,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
            kernel_size=kernel_size,
            gate_lower_bound=gate_lower_bound,
        )

    def test_kda_matches_the_recurrence(self):
        from sglang_tpu.attention.pallas.gdn import config

        kernel = self._kernel()
        for name, query_lens, context_lens, distribution, extra in _SHAPES:
            # Without decode rows, prefill_only must give the same result.
            prefill_only_modes = (False, True) if distribution[0] == 0 else (False,)
            for gate_name, gate_lower_bound in _GATES:
                args = (query_lens, context_lens, distribution, gate_lower_bound)
                (ref_conv, ref_state), ref_out = _kda_attention_ref(
                    **self._inputs(*args, **extra)
                )
                for prefill_only in prefill_only_modes:
                    with self.subTest(
                        case=f"{name}_{gate_name}", prefill_only=prefill_only
                    ):
                        # The kernel donates both state inputs; each call gets its own.
                        kwargs = self._inputs(*args, **extra)
                        (out_conv, out_state), out = kernel(
                            **kwargs,
                            attention_mode=config.AttentionMode.KDA,
                            prefill_only=prefill_only,
                        )

                        # |out| peaks near 0.3 (L2 norm and d_k**-0.5 scale); a 2e-2
                        # atol would hide gate-layout errors.
                        np.testing.assert_allclose(
                            np.asarray(out, np.float64), ref_out, rtol=2e-2, atol=5e-3
                        )
                        # Looser than the output: the state accumulates the delta rule
                        # over the whole sequence without the output's norm and scale.
                        slots = np.asarray(kwargs["state_indices"])[: len(query_lens)]
                        np.testing.assert_allclose(
                            np.asarray(out_state, np.float64)[slots],
                            np.asarray(ref_state, np.float64)[slots],
                            rtol=2e-2,
                            atol=1e-2,
                        )
                        np.testing.assert_allclose(
                            np.asarray(out_conv, np.float64)[slots],
                            np.asarray(ref_conv, np.float64)[slots],
                            rtol=1e-5,
                            atol=1e-5,
                        )

    def test_speculative_window_checkpoints(self):
        """8-token verify windows: reads resume at ``base + read_offset``, each
        real token checkpoints to ``base + t``, unused window slots stay intact."""
        from sglang_tpu.attention.pallas.gdn import config

        num_spec_tokens = 7
        window = num_spec_tokens + 1
        d_k, d_v, kernel_size = self.D_K, self.D_V, self.KERNEL_SIZE
        # (name, heads, window lengths, read offsets): Kimi-K3's local heads at
        # TP32 (32 windows span several decode tiles) and TP8 (large fp32 state).
        cases = [
            (
                "tp32_multi_tile_recurrent_window",
                3,
                [1, 6, 3, 8, 5, 2, 7, 4] * 4,
                [0, 2, 4, 6, 0, 2, 4, 6] * 4,
            ),
            (
                "tp8_large_recurrent_window",
                12,
                [1, 6, 3, 8, 5, 2, 7, 4],
                [0, 2, 4, 6, 0, 2, 4, 6],
            ),
        ]
        for name, n_v, spec_lengths, read_offsets in cases:
            with self.subTest(case=name):
                n_kq = n_v
                num_seqs = len(spec_lengths)
                num_tokens = sum(spec_lengths)
                # One group of `window` slots per sequence after the null slot 0.
                num_blocks = 1 + num_seqs * window
                dim_size = 2 * n_kq * d_k + n_v * d_v
                rngs = iter(jax.random.split(jax.random.key(11), 8))
                kwargs = dict(
                    qkv=jax.random.normal(
                        next(rngs), (num_tokens, dim_size), jnp.bfloat16
                    ),
                    b=jax.random.normal(next(rngs), (num_tokens, n_v), jnp.bfloat16),
                    a=jax.random.normal(
                        next(rngs), (num_tokens, n_v * d_k), jnp.bfloat16
                    ),
                    conv_state=jax.random.normal(
                        next(rngs),
                        (num_blocks, kernel_size - 1, dim_size),
                        jnp.bfloat16,
                    ),
                    recurrent_state=jax.random.normal(
                        next(rngs), (num_blocks, n_v, d_k, d_v)
                    ),
                    conv_weight=jax.random.normal(
                        next(rngs), (dim_size, 1, kernel_size), jnp.bfloat16
                    ),
                    conv_bias=None,
                    a_log=jax.random.normal(next(rngs), (n_v,)),
                    dt_bias=jax.random.normal(next(rngs), (n_v, d_k)),
                    query_start_loc=jnp.array(
                        np.cumsum([0] + spec_lengths), dtype=jnp.int32
                    ),
                    state_indices=jnp.array(
                        [1 + i * window for i in range(num_seqs)], dtype=jnp.int32
                    ),
                    distribution=jnp.array([num_seqs] * 3, dtype=jnp.int32),
                    # Every window continues a 32-token context.
                    seq_lens=jnp.array(
                        [32 + length for length in spec_lengths], dtype=jnp.int32
                    ),
                    read_offsets=jnp.array(read_offsets, dtype=jnp.int32),
                    n_kq=n_kq,
                    n_v=n_v,
                    d_k=d_k,
                    d_v=d_v,
                    kernel_size=kernel_size,
                    gate_lower_bound=-5.0,
                )
                # The kernel donates both state inputs; build the reference first.
                (ref_conv, ref_rec), ref_out = _kda_attention_spec_ref(**kwargs)
                (out_conv, out_rec), out = self._kernel()(
                    **kwargs,
                    num_spec_tokens=num_spec_tokens,
                    batched_only=True,
                    attention_mode=config.AttentionMode.KDA,
                )

                np.testing.assert_allclose(
                    np.asarray(out, np.float64),
                    np.asarray(ref_out, np.float64),
                    rtol=2e-2,
                    atol=5e-3,
                )
                # Every slot, including the null slot and unused window slots.
                np.testing.assert_allclose(
                    np.asarray(out_conv, np.float64),
                    np.asarray(ref_conv, np.float64),
                    rtol=2e-2,
                    atol=5e-3,
                )
                np.testing.assert_allclose(
                    np.asarray(out_rec, np.float64),
                    np.asarray(ref_rec, np.float64),
                    rtol=2e-2,
                    atol=1e-2,
                )

    def test_rejects_invalid_gate_bound(self):
        """The entry point's gate guards, which all fail silently if dropped."""
        from sglang_tpu.attention.pallas.gdn import config

        n_kq = n_v = 4
        d_k, d_v, kernel_size = self.D_K, self.D_V, self.KERNEL_SIZE
        dim_size = 2 * n_kq * d_k + n_v * d_v
        kernel = self._kernel()

        cases = [
            (
                "bound_under_gdn",
                dict(attention_mode=config.AttentionMode.GDN, gate_lower_bound=-5.0),
            ),
            ("non_negative_bound", dict(gate_lower_bound=0.0)),
        ]
        for name, overrides in cases:
            with self.subTest(case=name):
                kwargs = dict(
                    qkv=jnp.zeros((8, dim_size)),
                    b=jnp.zeros((8, n_v)),
                    a=jnp.zeros((8, n_v * d_k)),
                    conv_state=jnp.zeros((9, kernel_size - 1, dim_size)),
                    recurrent_state=jnp.zeros((9, n_v, d_k, d_v)),
                    conv_weight=jnp.zeros((dim_size, 1, kernel_size)),
                    conv_bias=None,
                    a_log=jnp.zeros((n_v,)),
                    dt_bias=jnp.zeros((n_v, d_k)),
                    query_start_loc=jnp.arange(9, dtype=jnp.int32),
                    state_indices=jnp.arange(1, 9),
                    distribution=jnp.array([8, 8, 8], dtype=jnp.int32),
                    seq_lens=jnp.ones((8,), dtype=jnp.int32),
                    n_kq=n_kq,
                    n_v=n_v,
                    d_k=d_k,
                    d_v=d_v,
                    kernel_size=kernel_size,
                    attention_mode=config.AttentionMode.KDA,
                )
                kwargs.update(overrides)
                with self.assertRaises(ValueError):
                    kernel(**kwargs)


if __name__ == "__main__":
    unittest.main()
