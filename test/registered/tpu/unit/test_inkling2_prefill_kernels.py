"""Inkling2 prefill Pallas kernels vs fp32 JAX transcriptions of the XLA paths."""

import os
import sys
import unittest

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=120, suite="stage-a-test-tpu-unit", chips=1)

# Standalone raw-JAX kernel test: select the physical backend before the first JAX import.
assert "jax" not in sys.modules
os.environ["JAX_PLATFORMS"] = "tpu"

import jax
import jax.numpy as jnp
import numpy as np

from sglang_tpu.inkling2_attn_res import attn_res_norm_jax
from sglang_tpu.inkling2_sconv_prefill import sconv_taps_jax

H = 4608
T = 512
EPS_RES = 1e-6
EPS_NORM = 1e-6


def _reference(sources, addend, queries, weights, normalize):
    """inkling2._block_attn_res followed by Inkling2RMSNorm (no pre-affine), in fp32."""
    srcs = [s.astype(jnp.float32) for s in sources]
    if addend is not None:
        srcs[-1] = srcs[-1] + addend.astype(jnp.float32)
    inv = [jax.lax.rsqrt(jnp.mean(s * s, axis=-1) + EPS_RES) for s in srcs]
    hp = jax.lax.Precision.HIGHEST
    logits = jnp.stack(
        [jnp.dot(s, queries.T, precision=hp) * r[:, None] for s, r in zip(srcs, inv)],
        axis=-1,
    )
    probs = jax.nn.softmax(logits, axis=-1)
    out = 0
    for i, (s, r) in enumerate(zip(srcs, inv)):
        value = s * r[:, None] if normalize else s
        out = out + probs[:, :, i, None] * value[:, None, :]
    out = jnp.moveaxis(out, 1, 0)
    y = out * jax.lax.rsqrt(jnp.mean(out * out, axis=-1, keepdims=True) + EPS_NORM)
    return (y * weights[:, None, :]).astype(jnp.bfloat16)


def _inputs(n, nq, has_addend, seed):
    keys = jax.random.split(jax.random.PRNGKey(seed), n + 3)
    # Sources of different scales so both the logits and the RMS normalization matter.
    sources = [
        (jax.random.normal(keys[i], (T, H)) * (0.5 + i)).astype(jnp.bfloat16)
        for i in range(n)
    ]
    addend = (
        jax.random.normal(keys[n], (T, H)).astype(jnp.bfloat16) if has_addend else None
    )
    queries = jax.random.normal(keys[n + 1], (nq, H)) * 2.0 * H**-0.5
    weights = 1.0 + 0.1 * jax.random.normal(keys[n + 2], (nq, H))
    return sources, addend, queries, weights


@unittest.skipUnless(
    any("TPU" in d.device_kind for d in jax.devices()), "Requires JAX with a TPU device"
)
class TestInkling2AttnResKernel(unittest.TestCase):
    def test_matches_reference(self):
        # (sources, query rows, addend): SWA attention input, full-layer 3-query input with the
        # most blocks, non-boundary MLP input (addend on the last source), boundary MLP input.
        for n, nq, has_addend in (
            (2, 1, False),
            (9, 3, False),
            (5, 1, True),
            (10, 1, False),
        ):
            for normalize in (True, False):
                with self.subTest(n=n, nq=nq, addend=has_addend, normalize=normalize):
                    sources, addend, queries, weights = _inputs(
                        n, nq, has_addend, seed=n
                    )
                    got = attn_res_norm_jax(
                        sources,
                        addend,
                        queries,
                        weights,
                        normalize=normalize,
                        eps_res=EPS_RES,
                        eps_norm=EPS_NORM,
                    )
                    want = _reference(sources, addend, queries, weights, normalize)
                    np.testing.assert_allclose(
                        np.asarray(got, np.float32),
                        np.asarray(want, np.float32),
                        rtol=2e-2,
                        atol=2e-2,
                    )
                    # bf16 output: at most a 1-ulp disagreement on a small fraction of elements.
                    diff = np.abs(
                        np.asarray(got, np.float32) - np.asarray(want, np.float32)
                    )
                    self.assertLess(
                        float(
                            np.mean(
                                diff > 1e-2 * (1 + np.abs(np.asarray(want, np.float32)))
                            )
                        ),
                        1e-3,
                    )


def _taps_reference(x, pos, w_lag, use_residual, scale):
    """attention_inkling_sconv_pallas._xla_sconv's in-forward taps (state taps zero)."""
    xf = x.astype(jnp.float32)
    acc = xf if use_residual else jnp.zeros_like(xf)
    for lag in range(w_lag.shape[0]):
        past = xf
        if lag:
            shifted = jnp.concatenate([jnp.zeros((lag, xf.shape[1])), xf])[
                : xf.shape[0]
            ]
            past = jnp.where(pos >= lag, shifted, 0.0)
        acc = acc + w_lag[lag][None] * past
    return (acc * scale).astype(x.dtype)


@unittest.skipUnless(
    any("TPU" in d.device_kind for d in jax.devices()), "Requires JAX with a TPU device"
)
class TestInkling2SconvTapsKernel(unittest.TestCase):
    def test_matches_reference_across_tile_and_sequence_boundaries(self):
        # Sequence starts inside a tile, on a tile boundary (256) and one row past it, so the
        # halo rows and the pos mask are both exercised.
        T, D = 1024, 4608
        starts = [0, 5, 256, 257, 300, 700]
        rows = np.arange(T)
        seq_start = np.array(starts)[np.searchsorted(starts, rows, side="right") - 1]
        pos = jnp.asarray((rows - seq_start)[:, None], jnp.int32)
        k = jax.random.split(jax.random.PRNGKey(3), 2)
        x = jax.random.normal(k[0], (T, D)).astype(jnp.bfloat16)
        w_lag = jax.random.normal(k[1], (4, D))
        for use_residual in (True, False):
            scale = (4 + int(use_residual)) ** -0.5
            got = sconv_taps_jax(x, pos, w_lag, use_residual=use_residual, scale=scale)
            want = _taps_reference(x, pos, w_lag, use_residual, scale)
            np.testing.assert_allclose(
                np.asarray(got, np.float32),
                np.asarray(want, np.float32),
                rtol=1e-2,
                atol=1e-2,
            )


if __name__ == "__main__":
    unittest.main()
