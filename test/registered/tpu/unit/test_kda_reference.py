# SPDX-License-Identifier: Apache-2.0
"""Numerical contracts for the TPU reference KDA (CPU JAX, no TPU required)."""

import unittest

from sglang.test.ci.ci_register import register_cpu_ci, register_tpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")
register_tpu_ci(est_time=10, suite="stage-a-test-tpu-unit")

import numpy as np

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    jax = None

if jax is not None:
    from sglang_tpu.kernels.kda import (
        ragged_kda,
        recurrent_kda_step,
    )


@unittest.skipIf(jax is None, "Requires JAX")
class TestKDAReference(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(53)
        self.h, self.d, self.width = 2, 4, 4
        self.channels = 3 * self.h * self.d
        self.weight = (
            self.rng.normal(size=(self.channels, self.width)).astype("float32") * 0.2
        )
        self.alog = np.array([-0.5, 0.2], dtype="float32")
        self.bias = self.rng.normal(size=self.h * self.d).astype("float32")

    def inputs(self, n):
        return [
            self.rng.normal(size=s).astype("float32")
            for s in [(n, self.channels), (n, self.h * self.d), (n, self.h)]
        ]

    def empty(self):
        return (
            np.zeros((4, self.width - 1, self.channels), dtype="float32"),
            np.zeros((4, self.h, self.d, self.d), dtype="float32"),
        )

    def call(self, inputs, states, starts, slots, lens, lower_bound=-5.0):
        args = [
            *inputs,
            *states,
            self.weight,
            self.alog,
            self.bias,
            np.array(starts, dtype="int32"),
            np.array(slots, dtype="int32"),
            np.array(lens, dtype="int32"),
        ]
        return jax.jit(ragged_kda, static_argnums=(11, 12, 13))(
            *map(jnp.asarray, args), self.h, self.d, lower_bound
        )

    def reference(self, inputs, conv=None, state=None, lower_bound=-5.0):
        # Independent timestep implementation with explicit head-wise matrices.
        conv = (
            np.zeros((self.width - 1, self.channels), dtype="float32")
            if conv is None
            else conv.copy()
        )
        state = (
            np.zeros((self.h, self.d, self.d), dtype="float32")
            if state is None
            else state.copy()
        )
        outputs = []
        sigmoid = lambda x: 1 / (1 + np.exp(-x))
        for raw, gate, beta in zip(*inputs):
            window = np.concatenate((conv, raw[None]), axis=0)
            x = np.sum(window * self.weight.T, axis=0)
            q, k, v = (x * sigmoid(x)).reshape(3, self.h, self.d)
            q /= np.sqrt(np.sum(q * q, axis=-1, keepdims=True) + 1e-6)
            k /= np.sqrt(np.sum(k * k, axis=-1, keepdims=True) + 1e-6)
            gate = (gate + self.bias).reshape(self.h, self.d)
            decay = np.exp(self.alog)[:, None]
            g = (
                lower_bound * sigmoid(decay * gate)
                if lower_bound < 0
                else -decay * np.logaddexp(0, gate)
            )
            output = []
            for head in range(self.h):
                state[head] = np.diag(np.exp(g[head])) @ state[head]
                error = v[head] - k[head] @ state[head]
                state[head] += sigmoid(beta[head]) * np.outer(k[head], error)
                output.append(q[head] @ state[head] / np.sqrt(self.d))
            outputs.append(output)
            conv = window[1:]
        return np.asarray(outputs), conv, state

    def test_single_step_decay_and_delta_update(self):
        q, k, v = self.inputs(1)[0].reshape(3, self.h, self.d)
        g = -np.abs(self.rng.normal(size=q.shape).astype("float32"))
        state = self.rng.normal(size=(self.h, self.d, self.d)).astype("float32")
        # beta=0 must only decay; beta=1 must fully apply the delta correction.
        for beta_value in (0.0, 1.0):
            with self.subTest(beta=beta_value):
                beta = np.full(self.h, beta_value, dtype="float32")
                out, updated = recurrent_kda_step(
                    *map(jnp.asarray, (q, k, v, g, beta, state))
                )
                expected = np.stack(
                    [np.diag(np.exp(g[h])) @ state[h] for h in range(self.h)]
                )
                for h in range(self.h):
                    expected[h] += beta_value * np.outer(
                        k[h], v[h] - k[h] @ expected[h]
                    )
                np.testing.assert_allclose(updated, expected, atol=2e-6, rtol=1e-5)
                np.testing.assert_allclose(
                    out,
                    np.stack(
                        [q[h] @ expected[h] / np.sqrt(self.d) for h in range(self.h)]
                    ),
                    atol=2e-6,
                    rtol=1e-5,
                )

    def test_unbounded_gate_matches_numpy(self):
        inputs = self.inputs(7)
        conv, state, out = self.call(
            inputs, self.empty(), [0, 7], [1], [7], lower_bound=0.0
        )
        expected, ec, es = self.reference(inputs, lower_bound=0.0)
        np.testing.assert_allclose(out, expected, atol=1e-6, rtol=1e-5)
        np.testing.assert_allclose(conv[1], ec, atol=1e-6)
        np.testing.assert_allclose(state[1], es, atol=1e-6)

    def test_mixed_decode_prefill_preserves_other_slots(self):
        decode, prefill = self.inputs(1), self.inputs(4)
        initial = tuple(
            self.rng.normal(size=x.shape).astype("float32") for x in self.empty()
        )
        packed = [
            np.concatenate((x, y, np.full_like(x, np.nan)))
            for x, y in zip(decode, prefill)
        ]
        conv, state, out = self.call(
            packed, initial, [0, 1, 5, 5], [2, 1, 0], [9, 4, 0]
        )
        ed, cd, sd = self.reference(decode, initial[0][2], initial[1][2])
        # The prefill slot contains stale data and must reset independently.
        ep, cp, sp = self.reference(prefill)
        np.testing.assert_allclose(out[:5], np.concatenate((ed, ep)), atol=1e-6)
        np.testing.assert_array_equal(out[5:], 0)
        for slot, ec, es in [(1, cp, sp), (2, cd, sd)]:
            np.testing.assert_allclose(conv[slot], ec, atol=1e-6)
            np.testing.assert_allclose(state[slot], es, atol=1e-6)
        for slot in (0, 3):
            np.testing.assert_array_equal(conv[slot], initial[0][slot])
            np.testing.assert_array_equal(state[slot], initial[1][slot])

    def test_numpy_recurrence_and_final_state(self):
        inputs = self.inputs(9)
        conv, state, out = self.call(inputs, self.empty(), [0, 9], [1], [9])
        expected, ec, es = self.reference(inputs)
        np.testing.assert_allclose(out, expected, atol=1e-6, rtol=1e-5)
        np.testing.assert_allclose(conv[1], ec, atol=1e-6)
        np.testing.assert_allclose(state[1], es, atol=1e-6, rtol=1e-5)

    def test_all_null_requests_leave_nonzero_state_untouched(self):
        initial = tuple(
            self.rng.normal(size=x.shape).astype("float32") for x in self.empty()
        )
        conv, state, output = self.call(
            self.inputs(3), initial, [0, 1, 3], [0, 0], [7, 2]
        )
        np.testing.assert_array_equal(conv, initial[0])
        np.testing.assert_array_equal(state, initial[1])
        np.testing.assert_array_equal(output, 0)

    def test_chunked_prefill_decode_and_slot_reuse(self):
        inputs = self.inputs(9)
        full = self.call(inputs, self.empty(), [0, 9], [1], [9])
        first = self.call([x[:4] for x in inputs], self.empty(), [0, 4], [1], [4])
        second = self.call([x[4:8] for x in inputs], first[:2], [0, 4], [1], [8])
        last = self.call([x[8:] for x in inputs], second[:2], [0, 1], [1], [9])
        np.testing.assert_allclose(
            jnp.concatenate([first[2], second[2], last[2]]), full[2], atol=1e-6
        )
        np.testing.assert_allclose(last[1], full[1], atol=1e-6)
        reused = self.call(inputs, last[:2], [0, 9], [1], [9])
        np.testing.assert_allclose(reused[2], full[2], atol=1e-6)

    def test_ragged_requests_and_padding(self):
        a, b = self.inputs(3), self.inputs(2)
        packed = [
            np.concatenate([x, y, np.full((2, *x.shape[1:]), 99, dtype=x.dtype)])
            for x, y in zip(a, b)
        ]
        conv, state, output = self.call(
            packed, self.empty(), [0, 3, 5, 5], [2, 1, 0], [3, 2, 0]
        )
        ea, ca, sa = self.reference(a)
        eb, cb, sb = self.reference(b)
        np.testing.assert_allclose(output[:5], np.concatenate([ea, eb]), atol=1e-6)
        np.testing.assert_array_equal(output[5:], 0)
        np.testing.assert_array_equal(state[0], 0)
        np.testing.assert_array_equal(conv[0], 0)
        np.testing.assert_allclose(state[2], sa, atol=1e-6)
        np.testing.assert_allclose(state[1], sb, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
