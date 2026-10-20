"""K3 megakernel speculative-verify attention stages: ``body_rows`` (R rows at once) vs the
B1 ``body`` run row by row, on one host's 8 cores with random weights. KDA: outputs and the
state after every row; MLA: outputs and the cache rows written at kv0 - 1 .. kv0 + R - 2,
for start positions of both parities, across a page and a softmax-block boundary."""

import os
import unittest

import ml_dtypes
import numpy as np

from sglang.test.test_utils import CustomTestCase

try:
    import jax

    _DEVICES = [d for d in jax.devices() if d.platform == "tpu"]
except Exception:  # pragma: no cover - non-TPU host
    _DEVICES = []

_ROWS = 8
_BF16 = ml_dtypes.bfloat16


def _w(rng, shape, scale):
    return (rng.standard_normal(shape) * scale).astype(_BF16)


def _report(name, a, b):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    rel = float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-30))
    print(f"{name}: bit-equal={np.array_equal(a, b)} max rel {rel:.2e}", flush=True)
    return rel


@unittest.skipUnless(len(_DEVICES) >= 8, "requires 8 TPU cores (one tpu7x host)")
class TestVerifyAttention(CustomTestCase):
    def setUp(self):
        from jax.sharding import Mesh, NamedSharding
        from jax.sharding import PartitionSpec as P

        from sglang_tpu.k3_megakernel.config import Topology

        self.topo = Topology(1, 8)
        self.mesh = Mesh(np.array(_DEVICES[:8]), ("x",))
        sh = NamedSharding(self.mesh, P("x"))
        self.put = lambda a: jax.device_put(np.ascontiguousarray(a), sh)
        self.exact = os.environ.get("K3MK_VERIFY_EXACT", "") == "1"

    def _check(self, name, got, want, tol):
        rel = _report(name, got, want)
        if self.exact:
            np.testing.assert_array_equal(
                np.asarray(got, np.float32), np.asarray(want, np.float32), err_msg=name
            )
        else:
            self.assertLess(rel, tol, name)

    def test_kda(self):
        from sglang_tpu.k3_megakernel.kernel.stages.kda import KdaStage

        world, hidden = self.topo.world, 7168
        rows_stage = KdaStage(self.topo, rows=_ROWS)
        b1 = KdaStage(self.topo)
        h, hd = b1.heads, b1.hd
        rng = np.random.default_rng(0)
        ins = [
            (rng.standard_normal((world, _ROWS, hidden)) * 0.5).astype(_BF16),
            _w(rng, (world, 128, hd), 0.1),
            (rng.standard_normal((world, 4, 3 * hd)) * 0.3).astype(np.float32),
            (rng.standard_normal((world, 1, hd)) * 0.1).astype(np.float32),
            (rng.standard_normal((world, 1, hd)) * 0.1).astype(np.float32),
            (1 + rng.standard_normal((world, 1, 128)) * 0.1).astype(np.float32),
            (rng.standard_normal((world, 3, 3 * hd)) * 0.5).astype(np.float32),
            (rng.standard_normal((world, h, 128, 128)) * 0.05).astype(np.float32),
            _w(rng, (world, hidden, 4 * hd + 256), hidden**-0.5),
            _w(rng, (world, hd, hidden), hd**-0.5),
        ]
        args = [self.put(a) for a in ins]
        got = rows_stage.build(self.mesh)(*args)
        want = b1.build(self.mesh, n_rows=_ROWS)(*args)
        self._check("kda out", got[0][0], want[0][0], 2e-2)
        self._check("kda conv checkpoints", got[3][0], want[3][0], 1e-6)
        self._check("kda rec checkpoints", got[4][0], want[4][0], 1e-4)

    def test_mla(self):
        from sglang_tpu.k3_megakernel.kernel.stages.mla import PAGE, MlaStage

        world, hidden = self.topo.world, 7168
        rows_stage = MlaStage(self.topo, rows=_ROWS, block_pages=4)
        # The B1 path as the stack runs it by default (KV window, early KV fetch).
        b1 = MlaStage(self.topo, block_pages=4, kv_window=True, early_kv=True)
        h, hd = b1.heads, b1.hd
        n_pages = 16
        pages = np.arange(1, n_pages + 1, dtype=np.int32)
        for kv0 in (1, 4, 5, 100, 101, 126, 127, 509, 512):
            rng = np.random.default_rng(kv0)
            pool = np.zeros((world, n_pages + 2, 64, 2, 640), _BF16)
            # Cached tokens before kv0 - 1; the rest of the pool stays 0 (as freshly allocated).
            tok = (rng.standard_normal((kv0 - 1, 640)) * 0.5).astype(_BF16)
            tok[:, 576:] = 0
            for t in range(kv0 - 1):
                pool[:, pages[t // PAGE], (t % PAGE) // 2, t % 2] = tok[t]
            ins = [
                np.full((world, 1), kv0, np.int32),
                np.tile(pages, (world, 1)),
                (rng.standard_normal((world, _ROWS, hidden)) * 0.5).astype(_BF16),
                (1 + rng.standard_normal((world, 1, 1536)) * 0.1).astype(np.float32),
                (1 + rng.standard_normal((world, 1, 512)) * 0.1).astype(np.float32),
                _w(rng, (world, h, 128, 512), 128**-0.5),
                _w(rng, (world, h, 512, 128), 512**-0.5),
                _w(rng, (world, hidden, 2176 + hd), hidden**-0.5),
                _w(rng, (world, 1536, 2 * hd), 1536**-0.5),
                _w(rng, (world, hd, hidden), hd**-0.5),
            ]
            args = [self.put(a) for a in ins]
            got_out, got_pool = rows_stage.build(self.mesh, pool.shape[1:])(
                *args, self.put(pool)
            )
            want_out, want_pool = b1.build(self.mesh, pool.shape[1:], n_rows=_ROWS)(
                *args, self.put(pool)
            )
            self._check(f"mla kv0={kv0} out", got_out[0], want_out[0], 2e-2)
            gp, wp = (
                np.asarray(got_pool[0], np.float32),
                np.asarray(want_pool[0], np.float32),
            )
            for t in range(kv0 - 2, kv0 - 1 + _ROWS):
                if t < 0:
                    continue
                p, w, half = pages[t // PAGE], (t % PAGE) // 2, t % 2
                self._check(
                    f"mla kv0={kv0} cache t={t}", gp[p, w, half], wp[p, w, half], 2e-2
                )


if __name__ == "__main__":
    unittest.main()
