"""kv_span_write: every span lands on its flat slots, nothing else changes.

Compares the Pallas span write against a plain copy on a small fused buffer
with random rows, including a zero-length padding span.
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=20, suite="stage-a-test-tpu-unit", chips=1)

# Force the TPU backend before the first JAX import.
assert "jax" not in sys.modules
os.environ.setdefault("JAX_PLATFORMS", "tpu")


def _has_jax_tpu() -> bool:
    try:
        import jax

        return any(d.platform == "tpu" for d in jax.devices())
    except Exception:  # noqa: BLE001
        return False


@unittest.skipIf(not _has_jax_tpu(), "needs a TPU-backed JAX")
class TestKvSpanWrite(unittest.TestCase):
    def test_spans_land_on_their_slots_and_nothing_else_changes(self):
        import jax.numpy as jnp

        from sglang_tpu.kv_span_write import kv_span_write_jax

        rng = np.random.default_rng(0)
        pages, page_size, tail = 6, 8, (2, 2, 128)
        buf = rng.standard_normal((pages, page_size, *tail)).astype(np.float32)
        rows = rng.standard_normal((16, *tail)).astype(np.float32)
        dst = np.array([9, 30, 0, 40], np.int32)
        src = np.array([0, 5, 0, 12], np.int32)
        lens = np.array([5, 3, 0, 4], np.int32)

        expected = buf.reshape(pages * page_size, *tail).copy()
        for d, s, n in zip(dst, src, lens):
            expected[d : d + n] = rows[s : s + n]
        expected = expected.reshape(buf.shape)

        out = kv_span_write_jax(
            jnp.asarray(dst),
            jnp.asarray(src),
            jnp.asarray(lens),
            jnp.asarray(rows, jnp.bfloat16),
            jnp.asarray(buf, jnp.bfloat16),
        )
        got = np.asarray(out.astype(jnp.float32))
        want = np.asarray(jnp.asarray(expected, jnp.bfloat16).astype(jnp.float32))
        np.testing.assert_array_equal(got, want)


if __name__ == "__main__":
    unittest.main()
