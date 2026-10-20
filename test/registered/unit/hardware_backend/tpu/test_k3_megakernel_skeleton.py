"""Skeleton pieces that run on any JAX backend: the packed attention-residual mixture
(stages/attn_res.aggregate_packed) equals the reference ``aggregate`` bit for bit, and the
expert-replica route assignment (stages/moe.route_owners) sends every route to a host
holding its expert without raising the slowest host's route count."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

import unittest

import numpy as np

from sglang.test.test_utils import CustomTestCase


class TestSkeleton(CustomTestCase):
    def test_aggregate_packed(self):
        import jax
        import jax.numpy as jnp

        from sglang_tpu.k3_megakernel.kernel.stages.attn_res import (
            aggregate,
            aggregate_packed,
            bf16,
            inv_rms,
        )

        rng = np.random.default_rng(0)
        h, nb, eps = 7168, 8, 1e-5
        ref_fn = jax.jit(aggregate, static_argnums=5)
        packed_fn = jax.jit(aggregate_packed, static_argnums=6)
        for t in range(18):
            scale = rng.uniform(0.1, 30, (nb, 1))
            bank = bf16(jnp.asarray(rng.standard_normal((nb, h)) * scale, jnp.float32))
            prefix = bf16(jnp.asarray(rng.standard_normal((1, h)) * 3, jnp.float32))
            pw = bf16(jnp.asarray(rng.standard_normal((1, h)) * 0.05, jnp.float32))
            nw = bf16(jnp.asarray(rng.uniform(0.5, 1.5, (1, h)), jnp.float32))
            nvb = jnp.int32(t % (nb + 1))
            inv = jnp.concatenate(
                [inv_rms(bank[j : j + 1], eps) for j in range(nb)], axis=0
            )
            ref = ref_fn(prefix, bank[:, None, :], nvb, pw, nw, eps)
            got, _ = packed_fn(prefix, bank, inv, nvb, pw, nw, eps)
            np.testing.assert_array_equal(
                np.asarray(got), np.asarray(ref), err_msg=f"nvb={int(nvb)}"
            )

    def test_route_owners(self):
        import jax
        import jax.numpy as jnp

        from sglang_tpu.k3_megakernel.config import (
            K3Dims,
            MoeLayout,
            Topology,
        )
        from sglang_tpu.k3_megakernel.kernel.stages.moe import route_owners

        rng = np.random.default_rng(1)
        for hosts, reps in ((4, 0), (4, 48), (2, 48), (1, 48)):
            lay = MoeLayout(K3Dims(), Topology(hosts), replicas=reps)
            fn = jax.jit(
                lambda ids, lay=lay: [
                    jnp.stack(x) for x in route_owners(list(ids), lay)
                ]
            )
            before, after = [], []
            for _ in range(300):
                ids = rng.choice(896, 16, replace=False).astype(np.int32)
                servers, slots = (np.asarray(x) for x in fn(jnp.asarray(ids)))
                for e, s, j in zip(ids, servers, slots):
                    self.assertEqual(lay.expert_slot(int(s), int(e)), int(j))
                    self.assertEqual(lay.slot_experts(int(s))[int(j)], int(e))
                b = np.bincount(ids // lay.experts_per_host, minlength=hosts).max()
                a = np.bincount(servers, minlength=hosts).max()
                self.assertLessEqual(a, b)
                before.append(b)
                after.append(a)
            if lay.replica_count:
                self.assertLess(np.mean(after), np.mean(before))
            else:
                self.assertEqual(np.mean(after), np.mean(before))


if __name__ == "__main__":
    unittest.main()
