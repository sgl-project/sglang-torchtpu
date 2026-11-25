"""K3 fused prefill MoE expert layouts (prefill_moe.fused.Layout): the route
metadata (local counts, the expert-major sorted routes, their token ids and
probabilities, the 8K group maximum) at EP8/TP4, with and without the
sibling-steal group order, and at EP16/TP2, against NumPy, in Pallas TPU
interpret mode."""

import unittest

import numpy as np

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase

# Needs jax, which the CPU runners do not install; the TPU unit stage's time
# budget is spent, so test_k3_programs_nightly.py runs it in the TPU nightly.
register_tpu_ci(
    est_time=300,
    suite="nightly-tpu-4-v7",
    nightly=True,
    disabled="runs in the TPU nightly through test_k3_programs_nightly.py",
)

EXPERTS, TOPK = 896, 16


def _routes(rng, tokens):
    """Per token (global, in rank order) 16 distinct experts and FP32
    probabilities; some tokens are padding (all probabilities zero)."""
    ids = np.stack([rng.permutation(EXPERTS)[:TOPK] for _ in range(tokens)])
    probs = rng.random((tokens, TOPK), np.float32) + np.float32(0.01)
    probs[rng.random(tokens) < 0.1] = 0
    # Skew: a third of the tokens route mostly into the first 112 experts.
    skewed = rng.random(tokens) < 0.33
    ids[skewed] = np.stack(
        [rng.permutation(112)[:TOPK] for _ in range(int(skewed.sum()))]
    )
    return ids.astype(np.int32), probs


def _wire(ids, probs):
    """The all-gathered route wire as the fused prologue lays it out: each
    rank's rows, ``[rows / 4, 128]`` int32 a rank, rank-major."""
    tokens = ids.shape[0]
    rows = tokens // 32
    bits = probs.view(np.int32)
    parts = []
    for rank in range(32):
        own = slice(rank * rows, (rank + 1) * rows)
        if rows < 128:
            payload = np.concatenate((ids[own], bits[own]), axis=1)
        else:
            payload = np.concatenate(
                (ids[own].reshape(rows // 8, 128), bits[own].reshape(rows // 8, 128))
            )
        parts.append(payload.reshape(rows // 4, 128))
    return np.concatenate(parts)


def _reference(ids, probs, layout, rank):
    tokens = ids.shape[0]
    bits = (tokens * TOPK - 1).bit_length()
    first = int(layout.first_expert(rank))
    local = ids - first
    valid = (probs != 0) & (local >= 0) & (local < layout.experts)
    local = np.where(valid, local, layout.experts)
    position = np.arange(tokens * TOPK, dtype=np.int64).reshape(tokens, TOPK)
    keys = (local.astype(np.int64) << bits) | position
    order = np.argsort(keys.reshape(-1), kind="stable")
    sorted_keys = keys.reshape(-1)[order]
    counts = np.bincount(local[valid].reshape(-1), minlength=layout.experts)
    live = probs != 0
    groups = np.bincount(
        (ids[live] // layout.experts).reshape(-1), minlength=layout.groups
    )
    return dict(
        keys=sorted_keys.astype(np.int32),
        probabilities=probs.reshape(-1)[order],
        token_ids=((sorted_keys & ((1 << bits) - 1)) // TOPK).astype(np.int32),
        counts=counts[: layout.experts].astype(np.int32),
        maximum=np.int32(groups.max()),
    )


class _Ref:
    """A NumPy-backed stand-in for a VMEM ref: ``ref[...]``, ``ref[pl.ds(..)
    , :]`` reads and writes and ``ref.at[...]`` views, enough to run the
    fused kernel's metadata generator eagerly (each array op on the host)."""

    def __init__(self, array, index=()):
        self.array, self.index = array, index

    @staticmethod
    def _key(index):
        from jax._src.indexing import Slice

        if not isinstance(index, tuple):
            index = (index,)
        out = []
        for part in index:
            if isinstance(part, Slice):
                start = int(part.start)
                out.append(slice(start, start + int(part.size)))
            elif part is Ellipsis:
                out.append(slice(None))
            else:
                out.append(part)
        return tuple(out)

    def _view(self):
        return self.array[self.index] if self.index else self.array

    @property
    def at(self):
        ref = self

        class _At:
            def __getitem__(self, index):
                return _Ref(ref._view(), _Ref._key(index))

        return _At()

    @property
    def shape(self):
        return self._view().shape

    def __getitem__(self, index):
        import jax.numpy as jnp

        return jnp.asarray(self._view()[self._key(index)])

    def __setitem__(self, index, value):
        self._view()[self._key(index)] = np.asarray(value).astype(self.array.dtype)


def _run_eager(wire, tokens, layout, rank):
    """``fused._metadata_steps`` on NumPy refs, ``pl.loop`` as a host loop."""
    from unittest import mock

    from sglang_tpu.k3_programs.kimi_k3.kernels.prefill_moe import fused

    shape = (tokens * TOPK // 128, 128)
    refs = dict(
        keys=_Ref(np.zeros(shape, np.int32)),
        probabilities=_Ref(np.zeros(shape, np.float32)),
        token_ids=_Ref(np.zeros(shape, np.int32)),
        counts=_Ref(np.zeros((layout.experts,), np.int32)),
        maximum=_Ref(np.zeros((1,), np.int32)),
    )

    def host_loop(lower, upper):
        def run(body):
            for i in range(int(lower), int(upper)):
                body(i)

        return run

    with mock.patch.object(fused.pl, "loop", host_loop):
        for _ in fused._metadata_steps(
            _Ref(np.asarray(wire)),
            refs["keys"],
            refs["probabilities"],
            refs["token_ids"],
            refs["counts"],
            refs["maximum"] if tokens == 8192 else None,
            tokens=tokens,
            rank=rank,
            counts_ready=lambda: None,
            layout=layout,
        ):
            pass
    return {name: ref.array for name, ref in refs.items()}


def _run(wire, tokens, layout, rank):
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    from sglang_tpu.k3_programs.kimi_k3.kernels.prefill_moe import fused

    shape = (tokens * TOPK // 128, 128)
    with_maximum = tokens == 8192

    def kernel(rank_ref, wire_ref, keys, probabilities, token_ids, counts, *rest):
        maximum = rest[0] if with_maximum else None
        for _ in fused._metadata_steps(
            wire_ref,
            keys,
            probabilities,
            token_ids,
            counts,
            maximum,
            tokens=tokens,
            rank=rank_ref[0],
            counts_ready=lambda: None,
            layout=layout,
        ):
            pass

    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    out_shape = [
        jax.ShapeDtypeStruct(shape, jnp.int32),
        jax.ShapeDtypeStruct(shape, jnp.float32),
        jax.ShapeDtypeStruct(shape, jnp.int32),
        jax.ShapeDtypeStruct((layout.experts,), jnp.int32),
    ]
    if with_maximum:
        out_shape.append(jax.ShapeDtypeStruct((1,), jnp.int32))
    outputs = pl.pallas_call(
        kernel,
        out_shape=tuple(out_shape),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=(vmem,),
            out_specs=tuple(vmem for _ in out_shape),
        ),
        interpret=pltpu.InterpretParams(),
    )(jnp.asarray([rank], jnp.int32), jnp.asarray(wire))
    names = ("keys", "probabilities", "token_ids", "counts", "maximum")
    return {n: np.asarray(o) for n, o in zip(names, outputs)}


class TestFusedMoeLayouts(CustomTestCase):
    def test_layouts(self):
        from sglang_tpu.k3_programs.kimi_k3.kernels.prefill_moe import fused

        self.assertIs(fused.layout_for(16), fused.EP16)
        ep8 = fused.layout_for(8)
        steal = fused.layout_for(8, steal=True)
        self.assertEqual((ep8.experts, ep8.width, ep8.groups), (112, 768, 8))
        # Stock EP8: rank // 4; steal: 2 * (rank // 8) + rank % 2 (expert_steal).
        self.assertEqual(
            [ep8.first_expert(r) // 112 for r in range(32)],
            [r // 4 for r in range(32)],
        )
        self.assertEqual(
            [steal.first_expert(r) // 112 for r in range(32)],
            [2 * (r // 8) + r % 2 for r in range(32)],
        )
        self.assertEqual(
            [fused.EP16.first_expert(r) for r in range(32)],
            [r // 2 * 56 for r in range(32)],
        )

    def _check(self, tokens, layout, rank, seed, run=_run):
        rng = np.random.default_rng(seed)
        ids, probs = _routes(rng, tokens)
        got = run(_wire(ids, probs), tokens, layout, rank)
        want = _reference(ids, probs, layout, rank)
        for name in ("keys", "probabilities", "token_ids"):
            np.testing.assert_array_equal(got[name].reshape(-1), want[name], name)
        np.testing.assert_array_equal(got["counts"], want["counts"], "counts")
        self.assertGreater(int(want["counts"].sum()), 0)
        if tokens == 8192:
            self.assertEqual(int(got["maximum"][0]), int(want["maximum"]))

    def test_ep8_metadata_interpret(self):
        """The Pallas kernel (interpret mode) at a small window."""
        from sglang_tpu.k3_programs.kimi_k3.kernels.prefill_moe import fused

        self._check(512, fused.layout_for(8, steal=True), 5, 1)

    def test_ep8_metadata(self):
        """The same generator on host refs: every window format, both EP8
        group orders, and the 8K group maximum."""
        from sglang_tpu.k3_programs.kimi_k3.kernels.prefill_moe import fused

        for tokens, steal, rank in (
            (512, False, 30),
            (1024, True, 6),
            (4096, False, 13),
            (8192, True, 9),
            (8192, False, 22),
        ):
            with self.subTest(tokens=tokens, steal=steal, rank=rank):
                self._check(
                    tokens,
                    fused.layout_for(8, steal=steal),
                    rank,
                    tokens + rank,
                    run=_run_eager,
                )

    def test_ep16_metadata(self):
        from sglang_tpu.k3_programs.kimi_k3.kernels.prefill_moe import fused

        self._check(1024, fused.EP16, 7, 3, run=_run_eager)
        self._check(8192, fused.EP16, 20, 4, run=_run_eager)


if __name__ == "__main__":
    unittest.main()
