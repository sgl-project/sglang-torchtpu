"""EP run-write metadata: every owner row is written exactly once, by the right source.

Derived property of ``_ep_run_write_metadata``: a tile's rows are token-sorted, so each
owner's rows form one run; the run's destination is the owner's row count before the tile
in the expert-sorted order. Simulating every rank's runs on CPU must cover each owner's
``chunk * top_k`` output rows exactly once with the matching global row, and the receive
count must equal the power-of-two pieces the other ranks send. Pins the offset / rank /
piece bookkeeping that a "looks equivalent" rewrite can silently break.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

import unittest

import numpy as np

from sglang.test.test_utils import CustomTestCase

try:
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P

    from sglang_tpu.moe.pallas.fused_rs.gmm_fused_rs_nodedup import (
        _ep_run_write_metadata,
        _precompute_fused_rs_metadata,
    )

    _HAS_JAX = jax.device_count() >= 4
except Exception:  # pragma: no cover - depends on JAX availability
    _HAS_JAX = False

EP, TOP_K, TOKENS, EXPERTS = 4, 2, 16, 12
CHUNK = TOKENS // EP
LOCAL_E = EXPERTS // EP
TILE_M, SUBLANE = 4, 2
PIECES = tuple(b for b in (512, 256, 128, 64, 32, 16, 8, 4, 2, 1) if b <= TILE_M)


def _routing(seed):
    rng = np.random.default_rng(seed)
    topk_ids = rng.integers(0, EXPERTS, size=(TOKENS, TOP_K)).astype(np.int32)
    flat = topk_ids.reshape(-1)
    order = np.argsort(flat, kind="stable")
    lhs_indices = (order // TOP_K).astype(np.int32)  # token of each expert-sorted row
    group_sizes = np.bincount(flat, minlength=EXPERTS).astype(np.int32)
    return lhs_indices, group_sizes


@unittest.skipUnless(_HAS_JAX, "Requires JAX with 4 CPU devices")
class TestEpRunWriteMetadata(CustomTestCase):
    def _metadata(self, lhs_indices, group_sizes):
        num_slots = LOCAL_E + (TOKENS * TOP_K + TILE_M - 1) // TILE_M
        mesh = Mesh(np.array(jax.devices()[:EP]), ("ep",))

        def body(group_offset):
            group_offset = group_offset.reshape(-1)
            sizes = jnp.asarray(group_sizes)
            local_start = jnp.sum(
                jnp.where(
                    jnp.arange(EXPERTS, dtype=jnp.int32) < group_offset[0], sizes, 0
                )
            )
            gs = _precompute_fused_rs_metadata(
                sizes,
                group_offset,
                jnp.array([local_start], jnp.int32),
                jnp.asarray(lhs_indices),
                size_group=LOCAL_E,
                tile_m=TILE_M,
                sublane=SUBLANE,
                chunk_size=CHUNK,
                num_slots=num_slots,
            )
            meta, base, stride = _ep_run_write_metadata(
                gs,
                jnp.asarray(lhs_indices),
                group_offset=group_offset,
                size_group=LOCAL_E,
                pack_indices=False,
                top_k=TOP_K,
                chunk_size=CHUNK,
                ep_size=EP,
                ep_axis_name="ep",
                num_slots=num_slots,
                tile_m=TILE_M,
            )
            return meta[None]

        offsets = jnp.arange(EP, dtype=jnp.int32)[:, None] * LOCAL_E
        fn = jax.jit(
            jax.shard_map(
                body, mesh=mesh, in_specs=P("ep"), out_specs=P("ep"), check_vma=False
            )
        )
        meta = np.asarray(fn(jax.device_put(offsets, NamedSharding(mesh, P("ep")))))
        return meta, num_slots

    def _simulate(self, lhs_indices, group_sizes):
        """Replay every rank's runs; return the receive-piece matrix [source, owner]."""
        meta, num_slots = self._metadata(lhs_indices, group_sizes)
        base = 4 + 2 * num_slots + 1
        rows_per_owner = CHUNK * TOP_K
        written = {d: np.full(rows_per_owner, -1, np.int64) for d in range(EP)}
        pieces_to = np.zeros((EP, EP), np.int64)
        for src in range(EP):
            m = meta[src]
            num_gm = int(m[3])
            m_offset = m[4 + num_slots : 4 + 2 * num_slots + 1]
            run_len = m[base : base + num_slots * EP].reshape(num_slots, EP)
            run_rank = m[base + num_slots * EP : base + 2 * num_slots * EP].reshape(
                num_slots, EP
            )
            for g in range(num_gm):
                tile_rows = np.arange(m_offset[g], m_offset[g + 1])
                owners = lhs_indices[tile_rows] // CHUNK
                self.assertTrue(np.all(np.diff(owners) >= 0), "tile rows owner-sorted")
                off = 0
                for d in range(EP):
                    n = int(run_len[g, d])
                    self.assertEqual(n, int(np.sum(owners == d)))
                    for i in range(n):
                        dst = int(run_rank[g, d]) + i
                        self.assertEqual(written[d][dst], -1, "owner row written twice")
                        written[d][dst] = tile_rows[off + i]
                    off += n
                    pieces_to[src, d] += sum(1 for b in PIECES if n & b)
        for d in range(EP):
            # every owner row written once, in the expert-sorted order of the owner's rows
            np.testing.assert_array_equal(
                written[d], np.flatnonzero(lhs_indices // CHUNK == d)
            )
            self.assertEqual(
                int(meta[d][2]), int(pieces_to[:, d].sum() - pieces_to[d, d])
            )
        return pieces_to

    def test_random_routings(self):
        for seed in range(4):
            with self.subTest(seed=seed):
                self._simulate(*_routing(seed))

    def test_single_hot_expert(self):
        # every row on expert 0: one source, one long multi-piece run per (tile, owner)
        lhs_indices = np.array(sorted(list(range(TOKENS)) * TOP_K), np.int32)
        group_sizes = np.zeros(EXPERTS, np.int32)
        group_sizes[0] = TOKENS * TOP_K
        pieces_to = self._simulate(lhs_indices, group_sizes)
        self.assertTrue(np.all(pieces_to[1:] == 0), "only rank 0 sends")
        self.assertTrue(np.all(pieces_to[0] > 0))


if __name__ == "__main__":
    unittest.main()
