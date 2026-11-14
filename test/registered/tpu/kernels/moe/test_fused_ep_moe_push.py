# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Adapted for SGLang-TorchTPU from vllm-project/vllm-torchtpu (Apache-2.0).
# Source: tests/kernels/experimental/push_fused_moe/test_fused_ep_moe_push.py
"""Push dispatch and static-placement tests on an eight-device EP mesh."""

import os
import shlex
import sys

if __name__ == "__main__":
    # The DP8 topology-preprocessing path uses remote DMA between SparseCores.
    # Set its two startup requirements before creating the JAX client, as the
    # upstream DP8 serving recipe does. Keep any explicit caller overrides.
    _libtpu_args = shlex.split(os.environ.get("LIBTPU_INIT_ARGS", ""))
    for _flag in (
        "--xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes=0",
        "--xla_mosaic_unsafe_allow_multicore_remote_dma=true",
    ):
        if not any(a.split("=", 1)[0] == _flag.split("=", 1)[0] for a in _libtpu_args):
            _libtpu_args.append(_flag)
    os.environ["LIBTPU_INIT_ARGS"] = " ".join(_libtpu_args)

from sglang.test.ci.ci_register import register_tpu_ci

# Unmeasured estimate for the eight-device sweep at the served dims; the
# weekly est-time job refreshes it. Nightly, like the v2 numerical sweep.
register_tpu_ci(
    est_time=1800,
    suite="nightly-tpu-4-v7",
    nightly=True,
    chips=4,
    labels=["singlehost", "infra"],
)


import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from sglang_tpu.moe.pallas.fused_moe_v2 import AXIS, WeightFormat, fused_ep_moe_v2
from sglang_tpu.moe.pallas.fused_moe_v2.push_fused_moe import fused_ep_moe_push

pytestmark = pytest.mark.multichip


@pytest.fixture(scope="module", autouse=True)
def _require_eight_devices():
    """Skip the module unless this host has the eight-way slice. A fixture,
    not a `skipif`, so collection never builds the PJRT client (see the
    same fixture in test_fused_ep_moe_v2.py)."""
    if jax.device_count() < 8:
        pytest.skip("the push fused EP MoE is an eight-way collective here")


EP = 8
CAPACITY = 128
FP8 = jnp.float8_e4m3fn
TOPK = 4
TOKENS = 256

# Push and v2 run the same plan, FFN and combine arithmetic on the same
# quantized rows, so the outputs must be bit-identical (measured on tpu7x
# for every case here); the rotated control must sit far from them.
CONTROL_BOUND = 0.1


def _mesh():
    return Mesh(np.asarray(jax.devices()[:EP]), axis_names=(AXIS,))


def _fp8_weights(mesh, seed, *, hidden, inter, e_total):
    """(w1, w2, w1_scale, w2_scale), fp8 with per-output-channel scales."""
    rng = np.random.default_rng(seed)
    shard = NamedSharding(mesh, P(AXIS))
    out = []
    scales = []
    for shape in ((e_total, hidden, 2 * inter), (e_total, inter, hidden)):
        w = rng.normal(0, 0.1, shape).astype(np.float32)
        amax = np.max(np.abs(w), axis=1, keepdims=True)
        scale = np.where(amax == 0, 1.0, amax / float(jnp.finfo(FP8).max))
        out.append(jax.device_put(w / scale, shard).astype(FP8))
        scales.append(jax.device_put(scale.reshape(shape[0], shape[2]), shard))
    return out[0], out[1], scales[0], scales[1]


def _inputs(mesh, seed, *, hidden, e_logical, tokens=TOKENS):
    rng = np.random.default_rng(seed)
    shard = NamedSharding(mesh, P(AXIS))
    x = jax.device_put(rng.normal(0, 0.1, (tokens, hidden)).astype(np.float32), shard)
    gating = rng.normal(0, 1, (tokens, e_logical)).astype(np.float32)
    return x.astype(jnp.bfloat16), jax.device_put(gating, shard)


def _rel(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def _roll(weights):
    w1, w2, s1, s2 = weights
    return (
        jnp.roll(w1, 1, axis=0),
        jnp.roll(w2, 1, axis=0),
        jnp.roll(s1, 1, axis=0),
        jnp.roll(s2, 1, axis=0),
    )


SMALL = dict(hidden=512, inter=256, e_total=32)


def _check_parity(weights, common, *, sc_prepush, tc_exchange, ragged_stride):
    mesh = common["mesh"]
    hidden = weights[0].shape[1]
    x, gating = _inputs(mesh, 7, hidden=hidden, e_logical=weights[0].shape[0])
    kw = dict(common, ragged_stride=ragged_stride)
    want = fused_ep_moe_v2(x, *weights, gating, **kw)
    got = fused_ep_moe_push(
        x, *weights, gating, sc_prepush=sc_prepush, tc_exchange=tc_exchange, **kw
    )
    np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
    wrong = fused_ep_moe_v2(x, *_roll(weights), gating, **kw)
    assert _rel(got, wrong) > CONTROL_BOUND


@pytest.mark.parametrize("sc_prepush,tc_exchange,ragged_stride", [(1, True, None)])
def test_fp8_push_matches_v2(sc_prepush, tc_exchange, ragged_stride):
    mesh = _mesh()
    weights = _fp8_weights(mesh, 11, **SMALL)
    common = dict(
        topk=TOPK,
        renormalize=True,
        mesh=mesh,
        capacity=CAPACITY,
        weight_format=WeightFormat.FP8,
        sharded_plan=True,
    )
    _check_parity(
        weights,
        common,
        sc_prepush=sc_prepush,
        tc_exchange=tc_exchange,
        ragged_stride=ragged_stride,
    )


def _contiguous_plus_one(experts, ep):
    """Rank r holds its contiguous block plus the next rank's first expert:
    per-slot expert ids [ep * (experts / ep + 1)] and the [experts, 2] slot
    table (slots holding each expert in slot order, -1 padded)."""
    per = experts // ep
    held = [
        [r * per + j for j in range(per)] + [((r + 1) * per) % experts]
        for r in range(ep)
    ]
    slot_expert = [e for row in held for e in row]
    slots = [[] for _ in range(experts)]
    for s, e in enumerate(slot_expert):
        slots[e].append(s)
    table = [h + [-1] * (2 - len(h)) for h in slots]
    return np.asarray(slot_expert, np.int32), np.asarray(table, np.int32)


# The served DSv4.1-Flash expert dims (hidden 5120, inter 2560 holding a
# logical 2304, fp4 block-512 weights with row-padded flat scale tables) at
# the DP16 C768 per-shard shapes, on eight shards: the target layers' 25
# slots, 288 rows and top-6, and the drafter's 9 slots, 240 rows and top-3.
SERVED_HIDDEN, SERVED_INTER, SERVED_LOGICAL_INTER = 5120, 2560, 2304


def _served_fp4_weights(mesh, seed, *, e_total):
    """(w1, w2, w1_scale, w2_scale) in the served operand forms, generated on
    device (the host copy of the f32 masters would be tens of GB)."""
    from sglang_tpu.moe.pallas.fused_moe_v2.host import fp4_scale_rows

    shard = NamedSharding(mesh, P(AXIS))
    hidden, inter, logical = SERVED_HIDDEN, SERVED_INTER, SERVED_LOGICAL_INTER

    def one(key, k, n):
        raw = jax.random.normal(key, (e_total, k, n), jnp.float32) * 0.1
        if k == inter:
            raw = jnp.where(jnp.arange(k)[None, :, None] < logical, raw, 0)
        else:
            col = jnp.arange(n) % inter
            raw = jnp.where(col[None, None, :] < logical, raw, 0)
        grouped = raw.reshape(e_total, k // 512, 512, n)
        scale = jnp.maximum(jnp.max(jnp.abs(grouped), axis=2) / 6, 1e-12)
        w = (grouped / scale[:, :, None, :]).reshape(e_total, k, n)
        rows = fp4_scale_rows(k, 512, padded=True)
        padded = jnp.zeros((e_total, rows, n), jnp.float32).at[:, : k // 512].set(scale)
        return w.astype(jnp.float4_e2m1fn), padded.reshape(e_total * rows, n)

    k1, k2 = jax.random.split(jax.random.key(seed))
    w1, s1 = jax.jit(lambda k: one(k, hidden, 2 * inter), out_shardings=(shard,) * 2)(
        k1
    )
    w2, s2 = jax.jit(lambda k: one(k, inter, hidden), out_shardings=(shard,) * 2)(k2)
    return w1, w2, s1, s2


def _shared_expert_weights(mesh, seed):
    """A per-channel FP8 shared expert at the served dims, replicated."""
    from sglang_tpu.fp8_swiglu_mlp import _FP8_MAX

    rng = np.random.default_rng(seed)
    repl = NamedSharding(mesh, P())
    k, n = SERVED_HIDDEN, SERVED_LOGICAL_INTER

    def w(shape):
        raw = np.clip(rng.normal(0, 64, shape), -_FP8_MAX, _FP8_MAX)
        return jax.device_put(jnp.asarray(raw, jnp.float32).astype(FP8), repl)

    def s(size):
        return jax.device_put(
            rng.uniform(0.5, 1.5, size).astype(np.float32) * 1e-3, repl
        )

    return w((k, 2 * n)), s(2 * n), w((n, k)), s(k)


# The served program is compiled through torch_tpu, which keeps the bf16
# roundings between fused elementwise ops; compile these the same way.
_AS_SERVED = {"xla_allow_excess_precision": False}


def test_dispatch_push_through_v2_is_the_push_layer():
    """`fused_ep_moe_v2(dispatch="push")` is the push layer with its
    push_options forwarded, and gives the all-gather dispatch's bits."""
    mesh = _mesh()
    weights = _fp8_weights(mesh, 91, **SMALL)
    x, gating = _inputs(mesh, 92, hidden=SMALL["hidden"], e_logical=SMALL["e_total"])
    common = dict(
        topk=TOPK,
        renormalize=True,
        mesh=mesh,
        capacity=CAPACITY,
        weight_format=WeightFormat.FP8,
        sharded_plan=True,
    )
    options = dict(sc_prepush=2, tc_exchange=True)
    want = fused_ep_moe_v2(x, *weights, gating, **common)
    got = fused_ep_moe_v2(
        x, *weights, gating, dispatch="push", push_options=options, **common
    )
    direct = fused_ep_moe_push(x, *weights, gating, **options, **common)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
    np.testing.assert_array_equal(np.asarray(got), np.asarray(direct))


@pytest.fixture(scope="module")
def four_chip_weights():
    """Full EP8 expert sets, cached across the model's token buckets."""
    mesh = _mesh()
    cache = {}

    def build(experts):
        if experts not in cache:
            logical = _served_fp4_weights(mesh, 101, e_total=experts)
            slot_expert, table = _contiguous_plus_one(experts, EP)
            shard = NamedSharding(mesh, P(AXIS))
            physical = []
            for w in logical:
                # Flat scales are reordered by expert, not by scale row.
                grouped = w.reshape(experts, -1, w.shape[-1])
                placed = jnp.take(grouped, jnp.asarray(slot_expert), axis=0)
                if w.ndim == 2:
                    placed = placed.reshape(-1, w.shape[-1])
                physical.append(jax.device_put(placed, shard))
            cache[experts] = mesh, logical, tuple(physical), table
        return cache[experts]

    return build


def _served_dense_samples(mesh, weights, x, ids, coefficients):
    """FP32 sampled reference using explicit dequantization and local expert sums."""

    def local(x, ids, coefficients, w1, w2, s1, s2):
        count = w1.shape[0]
        s1 = s1.reshape(count, -1, w1.shape[-1])
        s2 = s2.reshape(count, -1, w2.shape[-1])
        first = lax.axis_index(AXIS) * count

        def expert(e, acc):
            def decode(w, scale):
                k, n = w.shape[1:]
                raw = w[e].astype(jnp.float32).reshape(k // 512, 512, n)
                return (raw * scale[e, : k // 512, None, :]).reshape(k, n)

            proj = jnp.matmul(
                x.astype(jnp.float32), decode(w1, s1), precision=lax.Precision.HIGHEST
            )
            gate, up = jnp.split(proj, 2, axis=-1)
            gate = jnp.minimum(gate, 10.0)
            up = jnp.clip(up, -10.0, 10.0)
            hidden = (gate * jax.nn.sigmoid(gate)) * up
            out = jnp.matmul(hidden, decode(w2, s2), precision=lax.Precision.HIGHEST)
            coef = jnp.where(ids == first + e, coefficients, 0).sum(-1)
            return acc + out * coef[:, None]

        mine = lax.fori_loop(0, count, expert, jnp.zeros(x.shape, jnp.float32))
        return lax.psum(mine, AXIS)

    return jax.jit(
        jax.shard_map(
            local,
            mesh=mesh,
            in_specs=(P(),) * 3 + (P(AXIS),) * 4,
            out_specs=P(),
            check_vma=False,
        )
    )(x, ids, coefficients, *weights)


@pytest.mark.parametrize(
    "experts,rows,topk",
    [(384, 288, 6), (128, 240, 3), (384, 2048, 6)],
    ids=["verify", "draft", "prefill"],
)
def test_four_chip_full_experts_and_static_eplb(four_chip_weights, experts, rows, topk):
    """V4.1 EP8: full expert sets, replicas, padding, load counts and shared output."""
    mesh, logical, physical, table = four_chip_weights(experts)
    physical_count = experts + EP
    shard, repl = NamedSharding(mesh, P(AXIS)), NamedSharding(mesh, P())
    x, gating = _inputs(
        mesh, 102, hidden=SERVED_HIDDEN, e_logical=experts, tokens=EP * rows
    )
    x = x * 8  # Exercise the model's activation clamp as well as its padding.
    gate_host = np.asarray(gating).copy()
    active = np.ones((EP, rows), bool)
    for rank in range(EP):
        active[rank, -(rank + 1) :] = False
    active[-1] = False
    active = active.reshape(-1)
    gate_host[~active] = -np.inf
    gating = jax.device_put(gate_host, shard)
    bias = np.zeros(experts, np.float32)
    bias[:: experts // EP] = 0.75
    common = dict(
        topk=topk,
        renormalize=True,
        mesh=mesh,
        capacity=CAPACITY,
        weight_format=WeightFormat.FP4,
        rhs_qb=512,
        sharded_plan=True,
        scoring_fn="sqrtsoftplus",
        score_bias=jax.device_put(bias, repl),
        routed_scaling_factor=2.5,
        act_fn="silu_and_mul_with_clamp",
        logical_intermediate_size=SERVED_LOGICAL_INTER,
    )
    plain = np.asarray(fused_ep_moe_v2(x, *logical, gating, **common))
    assert np.isfinite(plain.astype(np.float32)).all()
    np.testing.assert_array_equal(plain[~active], 0)

    scores = np.sqrt(np.logaddexp(0, gate_host))
    ids = np.argsort(-(scores + bias), axis=-1, kind="stable")[:, :topk]
    selected = np.take_along_axis(scores, ids, axis=-1)
    sample = np.arange(EP - 1) * rows
    coefficients = selected[sample] / selected[sample].sum(-1, keepdims=True) * 2.5
    reference = _served_dense_samples(
        mesh,
        logical,
        jax.device_put(x[sample], repl),
        jax.device_put(ids[sample].astype(np.int32), repl),
        jax.device_put(coefficients, repl),
    )

    def check_reference(result, dispatch):
        error = _rel(result[sample], reference)
        per_row = np.linalg.norm(
            result[sample].astype(np.float32) - np.asarray(reference), axis=-1
        )
        per_row /= np.linalg.norm(np.asarray(reference), axis=-1)
        print(
            f"EP8 {experts=} {rows=} {topk=} {dispatch=} FP32 reference "
            f"relative_l2={error} worst_row={per_row.max()}"
        )
        assert error < 0.06
        assert per_row.max() < 0.075

    check_reference(plain, "all_gather")

    copies = (table >= 0).sum(-1)
    slots = table[ids, np.arange(EP * rows)[:, None] % copies[ids]]
    counts = np.bincount(slots[active].ravel(), minlength=physical_count)
    assert counts.sum() == active.sum() * topk
    initial = jax.device_put(np.arange(physical_count, dtype=np.int32), repl)
    placement = dict(expert_slot=jax.device_put(table, repl), expert_load=initial)
    placed, load = fused_ep_moe_v2(x, *physical, gating, **placement, **common)
    np.testing.assert_array_equal(np.asarray(placed), plain)
    np.testing.assert_array_equal(np.asarray(load), np.arange(physical_count) + counts)

    if rows == 288:
        from sglang_tpu.moe.pallas.fused_moe_v2.preprocessing import (
            topology_for_coordinates,
        )

        topology = topology_for_coordinates(
            tuple((*d.coords, d.core_on_chip) for d in mesh.devices.flat)
        )
        assert topology is not None
        placed, load = fused_ep_moe_v2(
            x, *physical, gating, preprocess_topology=topology, **placement, **common
        )
        np.testing.assert_array_equal(np.asarray(placed), plain)
        np.testing.assert_array_equal(
            np.asarray(load), np.arange(physical_count) + counts
        )

    # A conservative push plus the combined options the model can opt into.
    push_plain = None
    for options in (
        dict(sc_prepush=2, tc_exchange=True),
        dict(
            sc_prepush=2,
            tc_exchange=True,
            onehot_slots=True,
            tail_barrier=False,
            hbm_staging=True,
            dispatch_ahead=physical_count // EP,
            matmul_scale_repeat=True,
            exchange_row_gather=True,
            bf16_sort_counts=True,
            slim_prepush_barrier=True,
            fused_dispatch_meta=True,
            table_kernel=True,
            exchange_tables=True,
            kernel_plan_tables=True,
            light_prepush=2,
            single_barrier=True,
            early_scale_scatter=2,
        ),
    ):
        got, load = fused_ep_moe_v2(
            x,
            *physical,
            gating,
            dispatch="push",
            push_options=options,
            **placement,
            **common,
        )
        got = np.asarray(got)
        if push_plain is None:
            assert np.isfinite(got.astype(np.float32)).all()
            np.testing.assert_array_equal(got[~active], 0)
            check_reference(got, "push")
            # Upstream's measured all-gather router and generic push router
            # use different scoring/normalization lowerings at these shapes.
            # Cross-dispatch accuracy is numerical; push scheduling options
            # below must still preserve every output bit.
            delta = np.linalg.norm(
                got.astype(np.float32) - plain.astype(np.float32), axis=-1
            )
            norm = np.linalg.norm(plain.astype(np.float32), axis=-1)
            relative = np.linalg.norm(delta) / np.linalg.norm(norm)
            worst = np.max(delta / np.maximum(norm, 1e-30))
            print(f"push/all_gather relative_l2={relative} worst_row={worst}")
            assert relative < 1e-4
            assert worst < 1e-3
            push_plain = got
        else:
            np.testing.assert_array_equal(got, push_plain, err_msg=str(options))
        np.testing.assert_array_equal(
            np.asarray(load), np.arange(physical_count) + counts
        )
    # The returned count tensor is an accumulator, while the input stays intact.
    got, twice = fused_ep_moe_v2(
        x,
        *physical,
        gating,
        dispatch="push",
        push_options=options,
        expert_slot=placement["expert_slot"],
        expert_load=load,
        **common,
    )
    np.testing.assert_array_equal(np.asarray(got), push_plain)
    np.testing.assert_array_equal(
        np.asarray(twice), np.arange(physical_count) + 2 * counts
    )
    np.testing.assert_array_equal(np.asarray(initial), np.arange(physical_count))

    if rows in (288, 240):
        from sglang_tpu.fp8_swiglu_mlp import fp8_swiglu_mlp

        shared = _shared_expert_weights(mesh, 103)

        def with_shared(x, gating, load, *weights):
            return fused_ep_moe_v2(
                x,
                *weights[:4],
                gating,
                dispatch="push",
                push_options=options,
                expert_slot=placement["expert_slot"],
                expert_load=load,
                shared_expert=weights[4:],
                shared_limit=10.0,
                **common,
            )

        routed, counted, got_shared = jax.jit(
            with_shared,
            compiler_options=_AS_SERVED,
        )(x, gating, initial, *physical, *shared)
        want_shared = jax.jit(
            jax.shard_map(
                lambda xl, *w: fp8_swiglu_mlp(xl, *w, limit=10.0),
                mesh=mesh,
                in_specs=(P(AXIS),) + (P(),) * 4,
                out_specs=P(AXIS),
                check_vma=False,
            ),
            compiler_options=_AS_SERVED,
        )(x, *shared)
        # This outer JIT uses serving's excess-precision setting; the standalone
        # calls above use JAX defaults. Check basic accuracy across that boundary.
        assert np.isfinite(np.asarray(routed, np.float32)).all()
        assert _rel(routed, push_plain) < 0.01
        np.testing.assert_array_equal(
            np.asarray(counted), np.arange(physical_count) + counts
        )
        assert np.isfinite(np.asarray(got_shared, np.float32)).all()
        assert _rel(got_shared, want_shared) < 0.01


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__]))
