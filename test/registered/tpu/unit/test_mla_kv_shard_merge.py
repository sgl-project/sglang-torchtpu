"""The split latent cache's merge on one host's 8 TensorCores (mesh in torch rank
order, shard groups of 2, 4 and 8 consecutive ranks, a 4096-token chunk of bf16
partials): merge_kv_shards' steps with XLA's reduce-scatter in FP32 round once,
within half a bf16 step (plus FP32 slack) of the f64 merge; with bf16_sum only
within 2 * shards further roundings of at most 2^-9 of sum|terms| each.
"""

from sglang.test.ci.ci_register import register_tpu_ci

# A four-chip test holds the whole host: nightly, outside the stage-a budget.
register_tpu_ci(
    est_time=150,
    suite="nightly-tpu-4-v7",
    nightly=True,
    chips=4,
    labels=["singlehost", "infra", "collectives"],
)

import functools
import unittest

from sglang.test.test_utils import CustomTestCase

try:
    import jax
    import jax.numpy as jnp
    import ml_dtypes
    import numpy as np
    from jax import lax

    _DEVICES = {d.id: d for d in jax.devices() if d.platform == "tpu"}
except Exception:  # pragma: no cover - non-TPU host
    _DEVICES = {}

W, TOKENS, HEADS, DIM = 8, 4096, 8, 512  # per rank: one 4096-token chunk, 8 own heads
CHECKED = 256  # tokens compared against the f64 merge (host time)
# Torch rank -> physical device id on a 2x2x1 host (k3_megakernel/config.py).
TORCH_ORDER = [0, 1, 4, 5, 6, 7, 2, 3]


def _merge(out, lse, *, shards, bf16_sum):
    """merge_kv_shards' steps over the "shard" mesh axis."""
    me = lax.axis_index("shard")
    la = lax.all_gather(lse, "shard")
    m = la.max(0)
    m = jnp.where(jnp.isfinite(m), m, 0.0)
    w = jnp.exp(la - m)
    mine = lax.dynamic_index_in_dim(w, me, 0, keepdims=False)
    mine = mine / jnp.maximum(w.sum(0), 1e-30)
    part = jnp.where(jnp.isfinite(lse)[..., None], out.astype(jnp.float32), 0.0)
    part = part * mine[..., None]
    if bf16_sum:
        part = part.astype(jnp.bfloat16)
    part = part.reshape(TOKENS, shards, HEADS, DIM).transpose(1, 0, 2, 3)
    red = lax.psum_scatter(part, "shard", scatter_dimension=0, tiled=True)
    return red[0].astype(jnp.bfloat16)


def _inputs(shards, seed):
    """Every rank's bf16 partial over its group's heads and lse: spread over ~30
    nats, ranks with no candidate (NaN partials), and padding tokens."""
    rng = np.random.default_rng(seed)
    out = rng.standard_normal((W, TOKENS, shards * HEADS, DIM), dtype=np.float32)
    lse = rng.normal(0.0, 4.0, out.shape[:3]) + rng.normal(0.0, 8.0, (1, TOKENS, 1))
    lse = lse.astype(np.float32)
    lse[rng.random(lse.shape) < 0.2] = -np.inf
    lse[:, -3:] = -np.inf
    out = out.astype(ml_dtypes.bfloat16)
    out[~np.isfinite(lse)] = np.nan
    return out, lse


def _f64_merge(out, lse, shards):
    """[W, CHECKED, HEADS, DIM]: each rank's own heads merged in f64 over its
    group, and the per-element sum of |terms|."""
    groups = W // shards
    o = out[:, -CHECKED:].astype(np.float64)
    o = o.reshape(groups, shards, CHECKED, shards, HEADS, DIM)
    lv = lse[:, -CHECKED:].astype(np.float64)
    lv = lv.reshape(groups, shards, CHECKED, shards, HEADS)
    held = np.isfinite(lv)
    m = np.where(held.any(1), np.where(held, lv, -np.inf).max(1), 0.0)[:, None]
    w = np.where(held, np.exp(lv - m), 0.0)
    w = w / np.maximum(w.sum(1, keepdims=True), 1e-300)
    terms = np.where(held[..., None], o, 0.0) * w[..., None]  # [g, src, T, dst, h, D]
    want, scale = terms.sum(1), np.abs(terms).sum(1)  # [g, T, dst, h, D]
    shape = (W, CHECKED, HEADS, DIM)
    return [a.transpose(0, 2, 1, 3, 4).reshape(shape) for a in (want, scale)]


def _half_step(x):  # half the bf16 spacing at |x|
    mag = np.maximum(np.abs(x), np.finfo(np.float32).tiny)
    return np.exp2(np.floor(np.log2(mag)) - 8)


@unittest.skipUnless(len(_DEVICES) == W, "requires one host's 8 TensorCores")
class TestKvShardMerge(CustomTestCase):
    def _run(self, fn, mesh, out, lse):
        from jax.sharding import NamedSharding
        from jax.sharding import PartitionSpec as P

        spec = P(("group", "shard"))
        sharded = jax.jit(
            jax.shard_map(
                fn, mesh=mesh, in_specs=(spec, spec), out_specs=spec, check_vma=False
            )
        )
        put = (
            lambda a: jax.device_put(  # noqa: E731
                a.reshape((-1,) + a.shape[2:]), NamedSharding(mesh, spec)
            )
        )
        got = np.asarray(jax.device_get(sharded(put(out), put(lse))), np.float32)
        return got.reshape(W, TOKENS, HEADS, DIM)[:, -CHECKED:]

    def test_merge_rounding(self):
        from jax.sharding import Mesh

        devices = np.array([_DEVICES[i] for i in TORCH_ORDER])
        for shards in (2, 4, 8):
            mesh = Mesh(devices.reshape(W // shards, shards), ("group", "shard"))
            out, lse = _inputs(shards, seed=shards)
            want, scale = _f64_merge(out, lse, shards)
            tight = _half_step(want) + 1e-6 * scale
            for bf16_sum in (False, True):
                with self.subTest(shards=shards, bf16_sum=bf16_sum):
                    fn = functools.partial(_merge, shards=shards, bf16_sum=bf16_sum)
                    got = self._run(fn, mesh, out, lse)
                    bound = tight + (2.0**-8 * shards * scale if bf16_sum else 0.0)
                    self.assertTrue(np.all(np.abs(got - want) <= bound))
                    np.testing.assert_array_equal(got[:, -3:], 0.0)  # padding


if __name__ == "__main__":
    unittest.main()
