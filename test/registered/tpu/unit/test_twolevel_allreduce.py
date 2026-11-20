"""Two-level one-shot Pallas all-reduce vs a plain sum on one host's devices.

The kernel's host split is a parameter, so splitting one host's 8 TensorCores
into 2 or 4 "hosts" exercises both the host-local and the cross-host phase.
"""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=60, suite="stage-a-test-tpu-unit", chips=4)

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


@unittest.skipUnless(len(_DEVICES) >= 4, "requires >= 4 TPU devices")
class TestTwoLevelAllReduce(CustomTestCase):
    def _run(self, widths, local, seed, split_barrier=False):
        from jax.sharding import Mesh
        from jax.sharding import PartitionSpec as P

        from sglang_tpu.twolevel_allreduce import twolevel_all_reduce_local

        world = len(_DEVICES)
        mesh = Mesh(np.array(_DEVICES), ("model",))
        rng = np.random.default_rng(seed)
        # Each rank's [1, H] input is one row of a [world, H] array.
        xs = [
            jnp.asarray(rng.standard_normal((world, h)), jnp.bfloat16) for h in widths
        ]
        n = len(xs)
        fn = jax.jit(
            jax.shard_map(
                functools.partial(
                    twolevel_all_reduce_local,
                    world=world,
                    local=local,
                    split_barrier=split_barrier,
                ),
                mesh=mesh,
                in_specs=(P("model"),) * n,
                out_specs=P("model") if n == 1 else (P("model"),) * n,
                check_vma=False,
            )
        )
        outs = fn(*xs)
        outs = outs if n > 1 else (outs,)
        for x, out in zip(xs, outs):
            want = np.asarray(x, np.float32).sum(axis=0)
            got = np.asarray(out, np.float32)
            self.assertEqual(got.shape, (world, x.shape[1]))
            # Every rank holds the same sum; the cross-host partial is bf16.
            for r in range(world):
                err = np.abs(got[r] - want).max() / np.abs(want).max()
                self.assertLess(err, 2e-2, f"local={local} rank={r} H={x.shape[1]}")
            np.testing.assert_array_equal(got, np.broadcast_to(got[0], got.shape))

    def test_single_tensor(self):
        world = len(_DEVICES)
        # local=1: one rank per "host", so only the cross-host phase runs.
        for local in sorted({world, world // 2, 2, 1}):
            self._run((7168,), local, seed=local)

    def test_packed_pair(self):
        # Kimi-K3 TP32: routed latent (3584) + shared-expert output (7168).
        world = len(_DEVICES)
        for local in sorted({world, world // 2}):
            self._run((3584, 7168), local, seed=100 + local)

    def test_fused_latent_norm(self):
        """Pair reduce with the first output RMSNorm'd in the kernel, against
        the unfused kernel followed by RMSNorm.forward_native in f32."""
        from jax.sharding import Mesh
        from jax.sharding import PartitionSpec as P

        from sglang_tpu.twolevel_allreduce import twolevel_all_reduce_local

        world = len(_DEVICES)
        mesh = Mesh(np.array(_DEVICES), ("model",))
        rng = np.random.default_rng(11)
        a = jnp.asarray(rng.standard_normal((world, 3584)), jnp.bfloat16)
        b = jnp.asarray(rng.standard_normal((world, 7168)), jnp.bfloat16)
        w = jnp.asarray(
            np.broadcast_to(rng.standard_normal((1, 3584)) * 0.5 + 1, (world, 3584)),
            jnp.float32,
        )
        for local in sorted({world, world // 2}):
            base = functools.partial(
                twolevel_all_reduce_local,
                world=world,
                local=local,
                split_barrier=True,
            )
            fused = jax.jit(
                jax.shard_map(
                    lambda x, y, z: base(x, y, norm_weight=z, norm_eps=1e-5),
                    mesh=mesh,
                    in_specs=(P("model"),) * 3,
                    out_specs=(P("model"),) * 2,
                    check_vma=False,
                )
            )
            plain = jax.jit(
                jax.shard_map(
                    base,
                    mesh=mesh,
                    in_specs=(P("model"),) * 2,
                    out_specs=(P("model"),) * 2,
                    check_vma=False,
                )
            )
            got_a, got_b = fused(a, b, w)
            ref_a, ref_b = plain(a, b)
            np.testing.assert_array_equal(np.asarray(got_b), np.asarray(ref_b))
            # The kernel reduces the [1, H] row like XLA does: bit-exact
            # against the XLA RMSNorm of the unfused result.
            xr = ref_a.astype(jnp.float32)
            xla = jax.jit(
                lambda v, ww: (
                    v
                    * jax.lax.rsqrt(jnp.mean(v * v, axis=-1, keepdims=True) + 1e-5)
                    * ww
                ).astype(jnp.bfloat16)
            )(xr, w)
            np.testing.assert_array_equal(np.asarray(got_a), np.asarray(xla))

    def test_fused_attn_res(self):
        """All-reduce + Kimi-K3 attention-residual decode aggregation in one
        kernel is bit-identical to the all-reduce followed by the standalone
        kimi_attn_res_decode kernel (with its pending add)."""
        from jax.sharding import Mesh
        from jax.sharding import PartitionSpec as P

        from sglang_tpu.kimi_attn_residual import attention_residual_decode_jax
        from sglang_tpu.twolevel_allreduce import (
            twolevel_all_reduce_attn_res_local,
            twolevel_all_reduce_local,
        )

        world = len(_DEVICES)
        mesh = Mesh(np.array(_DEVICES), ("model",))
        rng = np.random.default_rng(5)
        h, nb, eps = 7168, 9, 1e-6

        def rep(v):
            return np.broadcast_to(v, (world,) + v.shape[1:])

        x = jnp.asarray(rng.standard_normal((world, h)), jnp.bfloat16)
        a = jnp.asarray(rep(rng.standard_normal((1, h)) * 3), jnp.bfloat16)
        bank = jnp.asarray(
            rep(rng.standard_normal((1, nb, h)) * rng.uniform(0.3, 3, (1, nb, 1))),
            jnp.bfloat16,
        )
        cw = jnp.asarray(rep(rng.standard_normal((1, h)) * 0.03), jnp.float32)
        w = jnp.asarray(rep(1 + 0.1 * rng.standard_normal((1, h))), jnp.float32)
        for local in sorted({world, world // 2}):
            for nvb in (1, 8):

                def ref(x, a, bank, cw, w):
                    y = twolevel_all_reduce_local(x, world=world, local=local)
                    return attention_residual_decode_jax(
                        a, y, bank, cw[0], w[0], nvb, eps, eps
                    )

                fused = functools.partial(
                    twolevel_all_reduce_attn_res_local,
                    world=world,
                    local=local,
                    split_barrier=local < world,
                    nvb=nvb,
                    eps=eps,
                    out_eps=eps,
                )
                outs = [
                    jax.jit(
                        jax.shard_map(
                            f,
                            mesh=mesh,
                            in_specs=(P("model"),) * 5,
                            out_specs=(P("model"),) * 2,
                            check_vma=False,
                        )
                    )(x, a, bank, cw, w)
                    for f in (ref, fused)
                ]
                for want, got in zip(*outs):
                    np.testing.assert_array_equal(np.asarray(got), np.asarray(want))

    def test_alternating_single_and_pair_calls(self):
        world = len(_DEVICES)
        cases = [(False, world // 2), (True, world // 2)]
        if world >= 8:
            # Split barrier with more than two "hosts".
            cases.append((True, 2))
        for split, local in cases:
            with self.subTest(split_barrier=split, local=local):
                self._alternating(split, local)

    def _alternating(self, split_barrier, local):
        """n=1 and n=2 exchanges share collective id 1, so the barrier carries
        over between consecutive calls, inside one program and across two.
        Round i reduces (previous result / world + input), which sums to i * the
        plain sum on every rank."""
        from jax.sharding import Mesh
        from jax.sharding import PartitionSpec as P

        from sglang_tpu.twolevel_allreduce import twolevel_all_reduce_local

        world = len(_DEVICES)
        rounds = 3 if not split_barrier else 8
        mesh = Mesh(np.array(_DEVICES), ("model",))
        ar = functools.partial(
            twolevel_all_reduce_local,
            world=world,
            local=local,
            split_barrier=split_barrier,
        )
        rng = np.random.default_rng(7)
        x, a, b = (
            jnp.asarray(rng.standard_normal((world, h)), jnp.bfloat16)
            for h in (7168, 3584, 7168)
        )

        def skew(v, r):
            # With the split barrier, delay ranks by different amounts each
            # round so peers enter calls early / late (next-call signals
            # arrive while others are still in the previous call). The loop
            # adds zero.
            if not split_barrier:
                return v
            me = jax.lax.axis_index("model")
            trips = jax.lax.rem(me * 7 + r * 3, 5) * 400
            z = jax.lax.fori_loop(
                0, trips, lambda i, c: c * 0.5, jnp.zeros_like(v, jnp.float32)
            )
            return v + z.astype(v.dtype)

        def body(x, a, b):
            y, p, q = jnp.zeros_like(x), jnp.zeros_like(a), jnp.zeros_like(b)
            for r in range(rounds):
                y = ar(skew(y, r) / world + x)
                p, q = ar(skew(p, r + 1) / world + a, q / world + b)
            return y, p, q

        fn = jax.jit(
            jax.shard_map(
                body,
                mesh=mesh,
                in_specs=(P("model"),) * 3,
                out_specs=(P("model"),) * 3,
                check_vma=False,
            )
        )
        for call in range(2):
            for inp, out in zip((x, a, b), fn(x, a, b)):
                want = rounds * np.asarray(inp, np.float32).sum(axis=0)
                got = np.asarray(out, np.float32)
                err = np.abs(got - want[None]).max() / np.abs(want).max()
                self.assertLess(
                    err, 3e-2, f"local={local} call={call} H={inp.shape[1]}"
                )
                np.testing.assert_array_equal(got, np.broadcast_to(got[0], got.shape))


if __name__ == "__main__":
    unittest.main()
