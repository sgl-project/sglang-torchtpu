"""Host-split single-token GEMV (sglang_tpu/kernels/ops/gemm/host_split_gemv.py) on one host's
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


def _assert_fp8_bytes_equal(got, want):
    """fp8 arrays compared byte for byte (prefetch copies are raw copies)."""
    np.testing.assert_array_equal(
        np.asarray(got).view(np.uint8), np.asarray(want).view(np.uint8)
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
        from sglang_tpu.kernels.ops.gemm.static_fp8_gemv import (
            block_k,
            static_fp8_gemv_jax,
        )

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
        from sglang_tpu.kernels.ops.gemm.host_split_gemv import host_split_gemv_local

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
        from sglang_tpu.kernels.ops.gemm.host_split_gemv import host_split_gemv_local

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

    def test_prefetch_copies_weight(self):
        """prefetch: the up-proj output is unchanged and the second output is
        a bit-exact copy of the prefetched array (KDA qkvg shape, TP32)."""
        from sglang_tpu.kernels.ops.gemm.host_split_gemv import host_split_gemv_local

        world = len(_DEVICES)
        k, n = 3584, 896
        x, w, s, inv, _ = self._inputs(k, n, seed=30)
        pf = jnp.asarray(
            np.random.default_rng(31).standard_normal((world * 7168, 1536)) * 30,
            jnp.float8_e4m3fn,
        )
        kw = dict(local=world, cols=n, used=world, used_e=0)

        def plain(x, w, s, inv):
            return host_split_gemv_local(x, w, s, inv, None, **kw)[0]

        def with_pf(x, w, s, inv, pf):
            return host_split_gemv_local(x, w, s, inv, None, prefetch=pf, **kw)

        want = _smap(plain, 4, 1)(x, w, s, inv)
        got, copy = _smap(with_pf, 5, 2)(x, w, s, inv, pf)
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
        _assert_fp8_bytes_equal(copy, pf)

    def test_tail_add_and_attn_res(self):
        """tail: the gathered up-proj becomes (up + shared) [+ prefix] with
        bf16 rounding after each add, and the second output equals the
        attention-residual decode kernel on that sum, bit for bit (with and
        without the prefix add and the qkvg prefetch)."""
        from sglang_tpu.kernels.ops.gemm.host_split_gemv import host_split_gemv_local
        from sglang_tpu.kimi_attn_residual import attention_residual_decode_jax

        world = len(_DEVICES)
        k, n = 3584, 896
        hidden = world * n
        rng = np.random.default_rng(40)
        x, w, s, inv, _ = self._inputs(k, n, seed=41)
        shared = jnp.asarray(rng.standard_normal((world, hidden)), jnp.bfloat16)
        prefix = jnp.asarray(rng.standard_normal((world, hidden)) * 4, jnp.bfloat16)
        nb = 9
        bank = jnp.asarray(rng.standard_normal((world, nb, hidden)) * 4, jnp.bfloat16)
        cw = jnp.asarray(rng.standard_normal((world * hidden,)) * 0.05, jnp.float32)
        nw = jnp.asarray(1 + 0.1 * rng.standard_normal((world * hidden,)), jnp.float32)
        pf = jnp.asarray(
            rng.standard_normal((world * 1024, 1536)) * 30, jnp.float8_e4m3fn
        )
        kw = dict(local=world, cols=n, used=world, used_e=0)

        def plain(x, w, s, inv):
            return host_split_gemv_local(x, w, s, inv, None, **kw)[0]

        up = np.asarray(_smap(plain, 4, 1)(x, w, s, inv)).astype(np.float32)
        bf = lambda a: np.asarray(jnp.asarray(a, jnp.bfloat16)).astype(np.float32)
        # (nvb, prefix add, qkvg prefetch): no extras, everything, prefix only.
        for nvb, has_prefix, use_pf in (
            (1, False, False),
            (8, True, True),
            (4, True, False),
        ):
            o = bf(up + np.asarray(shared).astype(np.float32))
            if has_prefix:
                o = bf(o + np.asarray(prefix).astype(np.float32))
            o16 = jnp.asarray(o, jnp.bfloat16)

            def ref(o, bank, cw, nw, nvb=nvb):
                return attention_residual_decode_jax(
                    o, None, bank, cw, nw, nvb, 1e-6, 1e-5
                )[0]

            want = np.asarray(_smap(ref, 4, 1)(o16, bank, cw, nw))

            # Traced right below, inside this iteration: the loop variables are
            # read before they change.
            def fused(x, w, s, inv, sh, pre, bank, cw, nw, pf):
                tail = (sh, pre if has_prefix else None, bank, cw, nw, nvb, 1e-6, 1e-5)
                prefetch = pf if use_pf else None
                return host_split_gemv_local(
                    x, w, s, inv, None, prefetch=prefetch, tail=tail, **kw
                )

            outs = _smap(fused, 10, 2 + use_pf)(
                x, w, s, inv, shared, prefix, bank, cw, nw, pf
            )
            tag = f"{nvb=} {has_prefix=} {use_pf=}"
            np.testing.assert_array_equal(
                np.asarray(outs[0]).astype(np.float32), o, tag
            )
            np.testing.assert_array_equal(np.asarray(outs[1]), want, tag)
            if use_pf:
                _assert_fp8_bytes_equal(outs[2], pf)


if __name__ == "__main__":
    unittest.main()
