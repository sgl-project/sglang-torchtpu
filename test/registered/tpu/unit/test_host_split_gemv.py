"""Host-split single-token GEMV (sglang_tpu/kernels/host_split_gemv.py) on one host's
devices.

The ranks-per-host split is a kernel parameter, so one host's TensorCores
stand in for one or two hosts. The reference runs the replicated-path static
fp8 GEMV on every rank's own slice and gathers the columns on the host side:
the gathered GEMV columns must match it bit for bit, the router logits to
rounding.
"""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=90, suite="stage-a-test-tpu-unit", chips=4)

import functools
import unittest

from sglang.test.test_utils import CustomTestCase

try:
    import jax
    import jax.numpy as jnp
    import numpy as np

    _DEVICES = [d for d in jax.devices() if d.platform == "tpu"]
except Exception:  # pragma: no cover - non-TPU host
    _DEVICES = []


def _smap(fn, n_in, n_out):
    from jax.sharding import Mesh
    from jax.sharding import PartitionSpec as P

    mesh = Mesh(np.array(_DEVICES), ("model",))
    return jax.jit(
        jax.shard_map(
            fn,
            mesh=mesh,
            in_specs=(P("model"),) * n_in,
            out_specs=P("model") if n_out == 1 else (P("model"),) * n_out,
            check_vma=False,
        )
    )


def _gather(per_rank, local, used, cols):
    """Rank r gets columns [:cols] of the first ``used`` ranks of its host."""
    world = per_rank.shape[0]
    return np.stack(
        [
            np.concatenate(
                [per_rank[r // local * local + j, :cols] for j in range(used)]
            )
            for r in range(world)
        ]
    )


@unittest.skipUnless(len(_DEVICES) >= 4, "requires >= 4 TPU devices")
class TestHostSplitGemv(CustomTestCase):
    def _inputs(self, k, n, seed):
        world = len(_DEVICES)
        rng = np.random.default_rng(seed)
        x = jnp.asarray(rng.standard_normal((world, k)), jnp.bfloat16)
        w = jnp.asarray(rng.standard_normal((world * k, n)) * 30, jnp.float8_e4m3fn)
        scale = jnp.asarray(rng.random(world * n) * 1e-3, jnp.float32)
        inv = jnp.full((world,), 50.0, jnp.float32)
        router = jnp.asarray(rng.standard_normal((world * k, 128)) * 0.02, jnp.bfloat16)
        return x, w, scale, inv, router

    def _reference(self, k, n, x, w, scale, inv, router=None):
        from sglang_tpu.kernels.static_fp8_gemv import block_k, static_fp8_gemv_jax

        def per_rank(x, w, s, inv):
            return static_fp8_gemv_jax(x, w, s, inv, block_k(k, n))[:1]

        y = np.asarray(_smap(per_rank, 4, 1)(x, w, scale, inv))
        if router is None:
            return y, None

        def logits(x, r):
            return jnp.dot(x, r, preferred_element_type=jnp.float32).astype(x.dtype)

        return y, np.asarray(_smap(logits, 2, 1)(x, router), np.float32)

    def test_front_gathers_latent_and_router_columns(self):
        """Kimi-K3 front: 512 latent columns and 128 router experts from all but
        one rank of each host, plus this rank's own shared gate_up columns."""
        from sglang_tpu.kernels.host_split_gemv import host_split_gemv_local

        world = len(_DEVICES)
        k, n, cols = 7168, 896, 512
        for local in sorted({world, world // 2}):
            used = local - 1
            x, w, s, inv, r = self._inputs(k, n, seed=local)
            fn = functools.partial(
                host_split_gemv_local, local=local, cols=cols, used=used, used_e=used
            )
            lat, logits, rest = _smap(fn, 5, 3)(x, w, s, inv, r)
            y, ref_logits = self._reference(k, n, x, w, s, inv, r)
            np.testing.assert_array_equal(
                np.asarray(lat), _gather(y, local, used, cols), f"local={local}"
            )
            np.testing.assert_array_equal(np.asarray(rest), y[:, cols:])
            want = _gather(ref_logits, local, used, 128)
            got = np.asarray(logits, np.float32)
            self.assertLess(np.abs(got - want).max() / np.abs(want).max(), 1e-2)

    def test_up_gathers_all_ranks(self):
        from sglang_tpu.kernels.host_split_gemv import host_split_gemv_local

        world = len(_DEVICES)
        k, n = 3584, 896
        for local in sorted({world, world // 2}):
            x, w, s, inv, _ = self._inputs(k, n, seed=10 + local)

            def fn(x, w, s, inv):
                return host_split_gemv_local(
                    x, w, s, inv, None, local=local, cols=n, used=local, used_e=0
                )[0]

            y, _ = self._reference(k, n, x, w, s, inv)
            np.testing.assert_array_equal(
                np.asarray(_smap(fn, 4, 1)(x, w, s, inv)),
                _gather(y, local, local, n),
                f"local={local}",
            )


if __name__ == "__main__":
    unittest.main()
