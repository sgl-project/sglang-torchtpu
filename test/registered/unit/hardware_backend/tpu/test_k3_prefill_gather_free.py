"""K3 prefill: gather-free forms of the reference MoE's top-k scores
(prefill_megakernel.selected_scores), the prefix page order
(prefill_megakernel.prefix_page_order) and the DSpark ring scatters
(dspark.scatter_rings) give the same bits as the gathers and scatters they
replace."""

import unittest

import numpy as np

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase

# Needs jax, which the CPU runners do not install; the TPU unit stage's time
# budget is spent, so test_k3_programs_nightly.py runs it in the TPU nightly.
register_tpu_ci(
    est_time=60,
    suite="nightly-tpu-4-v7",
    nightly=True,
    disabled="runs in the TPU nightly through test_k3_programs_nightly.py",
)


def _page_order_gathers(every, *, count, page_tokens, slots, capacity):
    """prefix_page_order before: four gathers by each arena page's run."""
    import jax.numpy as jnp

    ranks = every.shape[0]
    starts = (every[:, :count] // page_tokens).T.reshape(-1)
    sizes = (every[:, count:] // page_tokens).T.reshape(-1)
    run_rank = jnp.tile(jnp.arange(ranks, dtype=jnp.int32), count)
    ends = jnp.cumsum(sizes)
    arena = jnp.arange(capacity // page_tokens, dtype=jnp.int32)
    run = jnp.minimum(
        jnp.sum(arena[:, None] >= ends[None, :], axis=1), sizes.shape[0] - 1
    )
    within = arena - (ends[run] - sizes[run])
    source = run_rank[run] * slots + 1 + starts[run] + within
    total = ends[-1]
    order = jnp.where(arena < total, source, 0).astype(jnp.int32)
    return jnp.concatenate((total[None].astype(jnp.int32), order))


def _scatter_rings_before(rings, rows, positions, sequences, ends, stops, window):
    """dspark.prefill_ring_update's scatters before: every slot, both rings."""
    import jax.numpy as jnp

    slot = jnp.mod(positions, window)

    def scatter(ring, keep):
        index = jnp.where(keep, slot, window)
        return ring.at[:, index].set(rows, mode="drop")

    after, at = [], []
    for s in range(rings.shape[0]):
        member = sequences == s
        after.append(scatter(rings[s], member & (positions >= ends[s] - window)))
        at.append(
            scatter(
                rings[s],
                member & (positions < stops[s]) & (positions >= stops[s] - window),
            )
        )
    return jnp.stack(after), jnp.stack(at)


class TestGatherFree(CustomTestCase):
    def test_selected_scores(self):
        import jax
        import jax.numpy as jnp

        from sglang_tpu.k3_programs.kimi_k3.kernels import prefill_megakernel as k

        rng = np.random.default_rng(0)
        for rows in (16, 256):
            logits = rng.normal(0, 4, (rows, 896)).astype(np.float32)
            logits[0, :8] = -200.0  # sigmoids that underflow to zero
            raw = jax.nn.sigmoid(jnp.asarray(logits))
            bias = jnp.asarray(rng.normal(0, 1, (1, 896)).astype(np.float32))
            _, ids = jax.lax.top_k(raw + bias, 16)
            got = np.asarray(k.selected_scores(raw, ids))
            want = np.asarray(jnp.take_along_axis(raw, ids, axis=1))
            np.testing.assert_array_equal(got.view(np.uint32), want.view(np.uint32))

    def test_prefix_page_order(self):
        import jax.numpy as jnp

        from sglang_tpu.k3_programs.kimi_k3.kernels import prefill_megakernel as k

        rng = np.random.default_rng(1)
        page = 512
        for count, capacity in ((1, 1 << 20), (4, 1 << 20), (16, 311296), (4, 4096)):
            limit = capacity // page
            for trial in range(4):
                pages = rng.integers(0, 4, (32, count))
                if trial == 0:
                    pages[:] = 0  # no prefix
                while pages.sum() > limit:
                    pages = pages // 2
                bases = np.cumsum(pages, axis=1) - pages
                every = np.concatenate((bases * page, pages * page), axis=1)
                slots = 1 + int(pages.sum(axis=1).max()) + 3
                args = dict(
                    count=count, page_tokens=page, slots=slots, capacity=capacity
                )
                got = np.asarray(
                    k.prefix_page_order(jnp.asarray(every, jnp.int32), **args)
                )
                want = np.asarray(
                    _page_order_gathers(jnp.asarray(every, jnp.int32), **args)
                )
                np.testing.assert_array_equal(got, want)

    def test_scatter_rings(self):
        import jax.numpy as jnp

        from sglang_tpu.k3_programs.kimi_k3.kernels import dspark

        rng = np.random.default_rng(2)
        window, layers, width, slots, rows = 64, 2, 16, 4, 256
        rings = jnp.asarray(
            rng.normal(size=(slots, layers, window, width)), jnp.bfloat16
        )
        values = jnp.asarray(rng.normal(size=(layers, rows, width)), jnp.bfloat16)
        # Two sequences in slots 0 and 2 (slots 1 and 3 unused), padding rows.
        sequences = np.full(rows, -1, np.int32)
        positions = np.full(rows, -1, np.int32)
        sequences[:100], positions[:100] = 0, np.arange(300, 400)
        sequences[128:200], positions[128:200] = 2, np.arange(0, 72)
        order = rng.permutation(rows)  # rank-major striping scrambles rows
        sequences, positions = sequences[order], positions[order]
        values = values[:, order]
        for stops in ([-1, -1, -1, -1], [350, -1, 40, -1], [-1, -1, 72, 9]):
            ends = jnp.asarray([400, 0, 72, 0], jnp.int32)
            args = (
                rings,
                values,
                jnp.asarray(positions),
                jnp.asarray(sequences),
                ends,
                jnp.asarray(stops, jnp.int32),
                window,
            )
            for got, want in zip(
                dspark.scatter_rings(*args), _scatter_rings_before(*args)
            ):
                np.testing.assert_array_equal(
                    np.asarray(got).view(np.uint16), np.asarray(want).view(np.uint16)
                )


if __name__ == "__main__":
    unittest.main()
