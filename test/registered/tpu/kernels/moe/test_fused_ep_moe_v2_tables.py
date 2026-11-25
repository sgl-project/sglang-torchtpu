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
# Source: tests/kernels/fused_moe/test_fused_ep_moe_v2_tables.py
# Source: tests/kernels/fused_moe/test_fused_ep_moe_v2_exchange.py
"""Routing-table identities, payload round trips and fused metadata parity."""

import sys
from functools import partial
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(
    est_time=60,
    suite="stage-a-test-tpu-unit",
    chips=1,
    labels=["singlehost", "infra"],
)


import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy

from sglang_tpu.moe.pallas.fused_moe_v2 import host, layer  # noqa: E402
from sglang_tpu.moe.pallas.fused_moe_v2.push_fused_moe import fusion  # noqa: E402
from sglang_tpu.moe.pallas.fused_moe_v2.push_fused_moe import routing  # noqa: E402
from sglang_tpu.moe.pallas.fused_moe_v2.push_fused_moe import (  # noqa: E402
    host as push_host,
)


@pytest.fixture(autouse=True)
def _cpu_backend():
    """Keep every array in this file off the accelerator."""
    with jax.default_device(jax.devices("cpu")[0]):
        yield


# One decode-sized and one prefill-sized geometry, both at the 512-expert,
# top-10 layout of the 397B model vLLM-TorchTPU serves (8 shards).
GEOMETRIES = [
    pytest.param(8, 64, 64, 10, id="decode_t64_ep8_g64"),
    pytest.param(8, 512, 64, 10, id="prefill_t512_ep8_g64"),
    pytest.param(4, 128, 32, 8, id="ep4_g32_topk8"),
]


def _routing_inputs(ep, t_local, g_local, topk, *, skew, seed):
    """Routing indices and builder arguments for one step.

    `skew` reproduces low concurrency, where only a few experts are selected at
    all and most runs are empty -- the case where the alignment padding and the
    empty-region predicates actually differ from the uniform case.
    """
    e_total = g_local * ep
    T = t_local * ep
    rng = np.random.default_rng(seed)
    if skew:
        hot = rng.choice(e_total, size=max(4, e_total // 12), replace=False)
        idx = rng.choice(hot, size=(T, topk))
    else:
        idx = rng.integers(0, e_total, (T, topk))
    capacity = 128
    stride = host.ragged_stride_bound(T, topk, e_total, capacity)
    block = host.routing_block(t_local, topk)
    return jnp.asarray(idx.astype(np.int32)), dict(
        e_total=e_total,
        ep=ep,
        t_local=t_local,
        block=block,
        tile_m=capacity,
        shard_stride=stride,
    )


def _routing(ep, t_local, g_local, topk, *, skew, seed):
    idx, kwargs = _routing_inputs(ep, t_local, g_local, topk, skew=skew, seed=seed)
    return host.build_routing_tables(idx, **kwargs), kwargs["e_total"]


def _assert_arrays_equal(left, right, *, shard):
    assert len(left) == len(right)
    for i, (actual, expected) in enumerate(zip(left, right)):
        np.testing.assert_array_equal(
            np.asarray(actual),
            np.asarray(expected),
            err_msg=f"shard {shard}, table {i}",
        )


@pytest.mark.parametrize("ep,t_local,g_local,topk", GEOMETRIES)
@pytest.mark.parametrize("skew", [False, True], ids=["uniform", "skewed"])
@pytest.mark.parametrize("seed", [0, 1])
def test_sharded_plan_produces_the_same_kernel_operands(
    ep, t_local, g_local, topk, skew, seed
):
    """The decomposed plan changes plan arithmetic, not kernel inputs."""
    idx, kwargs = _routing_inputs(ep, t_local, g_local, topk, skew=skew, seed=seed)
    whole = host.build_routing_tables(idx, **kwargs)
    e_total = kwargs["e_total"]
    stride = kwargs["shard_stride"]
    flat = np.asarray(idx)
    counts = np.stack(
        [
            np.bincount(
                flat[d * t_local : (d + 1) * t_local].reshape(-1), minlength=e_total
            )
            for d in range(ep)
        ]
    ).astype(np.int32)

    for me in range(ep):
        seen = []

        def all_gather_rows(row, seen=seen):
            seen.append(np.asarray(row))
            return jnp.asarray(counts)

        shard = host.build_routing_tables_sharded(
            idx, jnp.int32(me), all_gather_rows=all_gather_rows, **kwargs
        )
        assert len(seen) == 1
        np.testing.assert_array_equal(seen[0].reshape(-1), counts[me])
        np.testing.assert_array_equal(
            np.asarray(shard.pos),
            np.asarray(whole.arrival_row[me * t_local : (me + 1) * t_local]),
        )

        whole_rows = np.asarray(host.local_slab_rows(whole, me, shard_stride=stride))
        shard_rows = np.asarray(host.local_slab_rows(shard, me, shard_stride=stride))
        whole_live = whole_rows < stride
        shard_live = shard_rows < stride
        np.testing.assert_array_equal(whole_live, shard_live)
        np.testing.assert_array_equal(whole_rows[whole_live], shard_rows[shard_live])

        for builder in (
            host.shard_transport_tables_in_blocks,
            host.shard_push_tables_in_rows,
        ):
            _assert_arrays_equal(
                builder(whole, me, e_total=e_total, ep=ep),
                builder(shard, me, e_total=e_total, ep=ep),
                shard=me,
            )
        whole_experts = host.shard_expert_slabs(whole, me, e_total=e_total, ep=ep)
        shard_experts = host.shard_expert_slabs(shard, me, e_total=e_total, ep=ep)
        _assert_arrays_equal(whole_experts, shard_experts, shard=me)
        np.testing.assert_array_equal(
            np.asarray(host.shard_token_gather(whole, me, shard_stride=stride)),
            np.asarray(host.shard_token_gather(shard, me, shard_stride=stride)),
        )

        whole_visit = host.expert_visit_list(whole_experts[0], g_local)
        shard_visit = host.expert_visit_list(shard_experts[0], g_local)
        _assert_arrays_equal(whole_visit, shard_visit, shard=me)
        np.testing.assert_array_equal(
            np.asarray(
                host.shard_count_vector(
                    whole, whole_experts[0], me, e_total=e_total, ep=ep
                )
            ),
            np.asarray(
                host.shard_count_vector(
                    shard, shard_experts[0], me, e_total=e_total, ep=ep
                )
            ),
        )


@pytest.mark.parametrize("ep,t_local,g_local,topk", GEOMETRIES)
@pytest.mark.parametrize("skew", [False, True], ids=["uniform", "skewed"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_transport_tables_the_kernel_derives(ep, t_local, g_local, topk, skew, seed):
    """The three tables the kernel no longer prefetches equal the ones it does.

    Checked on every shard, not just rank 0: the builders slice by `me`, so an
    identity can hold at one shard and fail at another.
    """
    routing, e_total = _routing(ep, t_local, g_local, topk, skew=skew, seed=seed)
    for me in range(ep):
        (
            commit_start,
            commit_len,
            contrib_off,
            push_src,
            push_len,
            push_dst,
            _totals,
        ) = host.shard_transport_tables_in_blocks(
            routing, jnp.int32(me), e_total=e_total, ep=ep
        )
        true_rows, recv_row_off, _t = host.shard_push_tables_in_rows(
            routing, jnp.int32(me), e_total=e_total, ep=ep
        )
        del commit_start, true_rows

        np.testing.assert_array_equal(
            np.asarray(push_src),
            np.asarray(contrib_off),
            err_msg=f"push_src != contrib_off on shard {me}",
        )
        np.testing.assert_array_equal(
            np.asarray(push_len),
            np.asarray(commit_len),
            err_msg=f"push_len != commit_len on shard {me}; region_rows is no "
            "longer run_rows_aligned reshaped",
        )
        np.testing.assert_array_equal(
            np.asarray(push_dst) * host.ROWBLK,
            np.asarray(recv_row_off),
            err_msg=f"recv_row_off != push_dst * ROWBLK on shard {me}; "
            "recv_base no longer accumulates ALIGNED run lengths, so the "
            "kernel would push remote rows to the wrong offset",
        )


@pytest.mark.parametrize("ep,t_local,g_local,topk", GEOMETRIES)
@pytest.mark.parametrize("skew", [False, True], ids=["uniform", "skewed"])
def test_receive_offsets_are_block_aligned(ep, t_local, g_local, topk, skew):
    """The premise of the derivation, stated on its own.

    `push_dst * ROWBLK` can only reconstruct a row offset while every receive
    offset is a whole number of blocks. If that stops being true the identity
    above fails too, but this says why in one line.
    """
    routing, e_total = _routing(ep, t_local, g_local, topk, skew=skew, seed=7)
    for me in range(ep):
        _tr, recv_row_off, _t = host.shard_push_tables_in_rows(
            routing, jnp.int32(me), e_total=e_total, ep=ep
        )
        rem = np.asarray(recv_row_off) % host.ROWBLK
        assert not rem.any(), (
            f"receive row offsets are not {host.ROWBLK}-row aligned on shard "
            f"{me}: {np.unique(rem)}"
        )


@pytest.mark.parametrize("ep,t_local,g_local,topk", GEOMETRIES)
@pytest.mark.parametrize("all_padded", [False, True], ids=["partial", "all_padded"])
def test_padded_tokens_are_dropped_from_routing_tables(
    ep, t_local, g_local, topk, all_padded
):
    """Tokens with sentinel expert `e_total` add zero work and match across plans."""
    idx_jnp, kwargs = _routing_inputs(ep, t_local, g_local, topk, skew=False, seed=11)
    e_total = kwargs["e_total"]
    stride = kwargs["shard_stride"]
    idx = np.asarray(idx_jnp).copy()

    if all_padded:
        # Warmup (`_dummy_run`) marks every token on every shard as padded.
        idx[:, :] = e_total
    else:
        # Mark the second half of each shard's tokens as padded (`topk_idx = e_total`),
        # and mark shard 1 as completely padded (`0` active tokens).
        for d in range(ep):
            start = d * t_local if d == 1 else d * t_local + t_local // 2
            idx[start : (d + 1) * t_local, :] = e_total
    idx_padded = jnp.asarray(idx)

    whole = host.build_routing_tables(idx_padded, **kwargs)
    off_shard = host.off_shard_slab_row(stride, ep)

    # Every padded pair's slab_row must be the off-shard sentinel.
    padded_pair_mask = idx.reshape(-1) == e_total
    np.testing.assert_array_equal(
        np.asarray(whole.slab_row)[padded_pair_mask],
        np.full(padded_pair_mask.sum(), off_shard, dtype=np.int32),
    )
    if all_padded:
        np.testing.assert_array_equal(
            np.asarray(whole.run_rows), np.zeros((e_total, ep), dtype=np.int32)
        )
        np.testing.assert_array_equal(
            np.asarray(whole.expert_rows_aligned), np.zeros(e_total, dtype=np.int32)
        )
        np.testing.assert_array_equal(
            np.asarray(whole.recv_rows), np.zeros(ep, dtype=np.int32)
        )
    else:
        # Shard 1 has 0 valid tokens, so it sends 0 rows to every expert.
        np.testing.assert_array_equal(
            np.asarray(whole.run_rows)[:, 1], np.zeros(e_total, dtype=np.int32)
        )

    counts = np.stack(
        [
            np.bincount(
                idx[d * t_local : (d + 1) * t_local][
                    idx[d * t_local : (d + 1) * t_local] < e_total
                ],
                minlength=e_total,
            )
            for d in range(ep)
        ]
    ).astype(np.int32)

    for me in range(ep):
        shard = host.build_routing_tables_sharded(
            idx_padded,
            jnp.int32(me),
            all_gather_rows=lambda _row: jnp.asarray(counts),
            **kwargs,
        )
        np.testing.assert_array_equal(
            np.asarray(shard.pos),
            np.asarray(whole.arrival_row[me * t_local : (me + 1) * t_local]),
        )
        np.testing.assert_array_equal(
            np.asarray(shard.mirror_pos),
            np.asarray(whole.mirror_row[me * t_local : (me + 1) * t_local]),
        )
        whole_rows = np.asarray(host.local_slab_rows(whole, me, shard_stride=stride))
        shard_rows = np.asarray(host.local_slab_rows(shard, me, shard_stride=stride))
        whole_live = whole_rows < stride
        shard_live = shard_rows < stride
        np.testing.assert_array_equal(whole_live, shard_live)
        np.testing.assert_array_equal(whole_rows[whole_live], shard_rows[shard_live])
        for builder in (
            host.shard_transport_tables_in_blocks,
            host.shard_push_tables_in_rows,
            host.shard_expert_slabs,
        ):
            _assert_arrays_equal(
                builder(whole, me, e_total=e_total, ep=ep),
                builder(shard, me, e_total=e_total, ep=ep),
                shard=me,
            )
        np.testing.assert_array_equal(
            np.asarray(host.shard_token_gather(whole, me, shard_stride=stride)),
            np.asarray(host.shard_token_gather(shard, me, shard_stride=stride)),
        )
        whole_experts = host.shard_expert_slabs(whole, me, e_total=e_total, ep=ep)
        shard_experts = host.shard_expert_slabs(shard, me, e_total=e_total, ep=ep)
        whole_visit = host.expert_visit_list(whole_experts[0], g_local)
        shard_visit = host.expert_visit_list(shard_experts[0], g_local)
        _assert_arrays_equal(whole_visit, shard_visit, shard=me)
        if all_padded:
            assert int(np.asarray(whole_visit[1]).item()) == 0


def test_combine_arrivals_isolates_nan_on_unrouted_padded_tokens():
    """NaN arrival rows stay isolated to padded rows and zero out via row_routes."""
    from sglang_tpu.moe.pallas.fused_moe_v2.layer import _combine_arrivals

    # Use t_local=8 so combine_step_tokens(t_local) is 0 and the XLA epilogue runs
    # on CPU without invoking the Pallas TPU combine kernel.
    t_local, topk, lane_blocks, lanes, tile_m = 8, 4, 2, 128, 128
    recv_rows = 32
    arrivals = np.ones((recv_rows, lane_blocks, lanes), dtype=np.float32)
    arrival_scales = np.ones((4, tile_m), dtype=np.float32)

    # Sentinel pairs receive `pos = mirror_pos =` their rank among sentinel pairs
    # within the routing block (`[0, block)`). Poison the entire range `[0, 16)`
    # that the 4 padded tokens (`4 * topk = 16` pairs) index into.
    sentinel_ranks = np.arange(4 * topk, dtype=np.int32).reshape(4, topk)
    arrivals[: 4 * topk] = np.nan
    arrival_scales[0, : 4 * topk] = np.nan

    pos = np.full((t_local, topk), 16, dtype=np.int32)
    mirror_pos = np.full((t_local, topk), 16, dtype=np.int32)
    topk_weights = np.full((t_local, topk), 0.25, dtype=np.float32)

    # Mark rows 4..7 as padded: pos and mirror_pos span [0, 16), topk_weights=0.0.
    pos[4:] = sentinel_ranks
    mirror_pos[4:] = sentinel_ranks
    topk_weights[4:] = 0.0
    row_routes = jnp.asarray([[True]] * 4 + [[False]] * 4)

    combined = _combine_arrivals(
        jnp.asarray(arrivals),
        jnp.asarray(arrival_scales),
        jnp.asarray(pos),
        jnp.asarray(mirror_pos),
        jnp.asarray(topk_weights),
        jnp.bfloat16,
    )
    out = np.asarray(jnp.where(row_routes, combined, jnp.zeros_like(combined)))
    assert np.isfinite(out).all()
    np.testing.assert_allclose(out[:4].astype(np.float32), 1.0)
    np.testing.assert_array_equal(
        out[4:].astype(np.float32), np.zeros((4, lane_blocks * lanes), np.float32)
    )


@pytest.mark.parametrize("scoring_fn", ["softmax", "sigmoid", "sqrtsoftplus"])
def test_row_routes_rejects_nan_and_neg_inf_across_all_scoring_fns(scoring_fn):
    """Both NaN-masked rows and -inf replica-padded rows must return False."""
    from sglang_tpu.moe.pallas.fused_moe_v2.layer import _apply_scoring_fn, _row_routes

    gate_l = jnp.asarray(
        [
            [0.5, -0.2, 1.1, 0.0],
            [np.nan, np.nan, np.nan, np.nan],
            [-np.inf, -np.inf, -np.inf, -np.inf],
            [-np.inf, 0.0, -np.inf, -np.inf],
        ],
        dtype=jnp.float32,
    )
    scores = _apply_scoring_fn(gate_l, scoring_fn)
    routes = np.asarray(_row_routes(scores, gate_l))
    np.testing.assert_array_equal(routes, np.array([[True], [False], [False], [True]]))


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "t_local,topk,e_total,quantized,width",
    [
        (288, 6, 384, True, 2560),  # DP8/DP16 C384-per-host: unchanged
        (144, 6, 384, True, 1536),  # DP32 at 24 per rank: unchanged
        (288, 6, 392, True, 2688),  # DP8 with one spare slot per rank
        (144, 6, 416, True, 1664),  # DP32 with one spare slot per rank
        (240, 3, 128, True, 1152),  # the drafter at DP8
        (120, 3, 160, True, 768),  # the drafter at DP32 with 5 slots
        (256, 4, 32, False, 1152),  # the kernel tests' bf16 shape
    ],
)
def test_payload_width_is_lane_aligned_and_old_widths_are_unchanged(
    t_local, topk, e_total, quantized, width
):
    idx, count_at, got = layer._exchange_layout(t_local, topk, e_total, quantized)
    assert got == width and got % layer.HIDDEN_LANE_BLOCK == 0
    assert (
        idx % layer.HIDDEN_LANE_BLOCK == 0 and count_at % layer.HIDDEN_LANE_BLOCK == 0
    )
    assert count_at + e_total <= got


@pytest.mark.cpu_test
@pytest.mark.parametrize("e_total", [384, 392, 416])
def test_pack_and_unpack_round_trip_with_a_padded_tail(e_total):
    t_local, topk, ep = 144, 6, 2
    lay = layer._exchange_layout(t_local, topk, e_total, True)
    rng = np.random.default_rng(e_total)
    pairs = [
        rng.integers(0, e_total, size=t_local * topk).astype(np.int32)
        for _ in range(ep)
    ]
    scales = [rng.integers(0, 2**30, size=t_local).astype(np.int32) for _ in range(ep)]
    counts = [np.bincount(p, minlength=e_total).astype(np.int32) for p in pairs]
    packed = [
        layer._pack_routing_exchange(
            jnp.asarray(p), jnp.asarray(s), jnp.asarray(c), lay
        )
        for p, s, c in zip(pairs, scales, counts)
    ]
    assert all(p.shape == (lay[-1],) for p in packed)
    exchange = jnp.stack(packed)
    got_pairs, got_bits, got_counts = layer._unpack_routing_exchange(
        exchange, lay, t_local, topk, e_total
    )
    np.testing.assert_array_equal(np.asarray(got_pairs), np.stack(pairs))
    np.testing.assert_array_equal(np.asarray(got_bits), np.concatenate(scales))
    np.testing.assert_array_equal(np.asarray(got_counts), np.stack(counts))


@pytest.mark.cpu_test
def test_unpermute_load_to_ep_rank_order_inverts_mesh_relabeling():
    """Slot counts gathered in mesh order map back to EP-rank slot order."""
    ep, g_local = 4, 6
    e_total = ep * g_local
    mesh_ep_ranks = (2, 0, 3, 1)
    rng = np.random.default_rng(19)
    ep_slots = rng.integers(0, e_total, size=(32, 4), dtype=np.int32)
    mesh_slots = np.asarray(
        layer._relabel_expert_ids_to_mesh_order(
            jnp.asarray(ep_slots), g_local=g_local, mesh_ep_ranks=mesh_ep_ranks
        )
    )
    mesh_counts = np.bincount(mesh_slots.ravel(), minlength=e_total).astype(np.int32)
    ep_counts = np.bincount(ep_slots.ravel(), minlength=e_total).astype(np.int32)
    unpermuted = np.asarray(
        layer._unpermute_load_to_ep_rank_order(
            jnp.asarray(mesh_counts), g_local=g_local, mesh_ep_ranks=mesh_ep_ranks
        )
    )
    np.testing.assert_array_equal(unpermuted, ep_counts)


def test_push_segments_preserve_row_counts_and_destinations():
    """Every non-padding routing pair is sent once to its destination slot."""
    ep, tokens, local_experts, topk = 4, 64, 2, 2
    experts = ep * local_experts
    ids = (np.arange(ep * tokens)[:, None] + np.arange(topk)) % experts
    ids[::7] = experts
    counts = np.stack(
        [
            np.bincount(rows.ravel(), minlength=experts + 1)[:experts]
            for rows in ids.reshape(ep, tokens, topk)
        ]
    ).astype(np.int32)
    by_dest = jnp.asarray(counts)
    runs = push_host.run_tables(by_dest)
    for rank in (0, ep - 1):
        starts = np.cumsum(counts[rank]) - counts[rank]
        segments, metadata, _ = push_host.push_dispatch_segments(
            runs[3],
            runs[2],
            by_dest,
            jnp.int32(rank),
            jnp.asarray(starts),
            e_total=experts,
            ep=ep,
        )
        segments = np.asarray(segments).reshape(3, -1)
        seen = np.zeros(experts, np.int32)
        for packed, start in zip(segments[0], segments[1]):
            size = int(packed) & 65535
            if size:
                dest, local = int(packed) >> 24, (int(packed) >> 16) & 255
                expert = dest * local_experts + local
                assert start == starts[expert]
                seen[expert] += size
        np.testing.assert_array_equal(seen, counts[rank])
        received = counts[:, rank * local_experts : (rank + 1) * local_experts].sum(0)
        np.testing.assert_array_equal(
            np.asarray(metadata)[local_experts + 1 : 2 * local_experts + 1], received
        )


@pytest.mark.parametrize(
    "options,match",
    [
        ({"dispatch": "scatter"}, "dispatch must be one of"),
        ({"push_options": {"sc_prepush": 1}}, 'need dispatch="push"'),
        ({"shared_limit": 10.0}, 'need dispatch="push"'),
        ({"dispatch": "push", "stage": "route"}, "fused stage only"),
        (
            {"dispatch": "push", "push_options": {"sc_pre_push": 1}},
            "unknown push_options",
        ),
        (
            {"dispatch": "push", "push_options": {"early_scale_scatter": 3}},
            "early_scale_scatter is 0, 1 or 2",
        ),
        (
            {"dispatch": "push", "push_options": {"dispatch_ahead": 0}},
            "dispatch_ahead must be >= 1",
        ),
        (
            {"dispatch": "push", "push_options": {"dispatch_ahead": -1}},
            "dispatch_ahead must be >= 1",
        ),
        ({"dispatch": "push", "shared_limit": 10.0}, "requires shared_expert"),
    ],
)
def test_public_dispatch_rejects_invalid_options(options, match):
    with pytest.raises(ValueError, match=match):
        layer.fused_ep_moe_v2(
            None,
            None,
            None,
            None,
            None,
            None,
            topk=6,
            renormalize=True,
            mesh=None,
            capacity=128,
            **options,
        )


def test_push_option_inventory_matches_the_public_wrapper():
    """A new push keyword must be forwarded or exposed as an explicit option."""
    from inspect import signature

    from sglang_tpu.moe.pallas.fused_moe_v2.push_fused_moe import (
        K512_ONLY_OPTIONS,
        PUSH_OPTIONS,
    )
    from sglang_tpu.moe.pallas.fused_moe_v2.push_fused_moe import layer as k32z
    from sglang_tpu.moe.pallas.fused_moe_v2.push_fused_moe import layer_k512 as k512

    forwarded = signature(layer.fused_ep_moe_v2).parameters.keys()
    k512_options = signature(k512.fused_ep_moe_push).parameters.keys() - forwarded
    k32z_options = signature(k32z.fused_ep_moe_push).parameters.keys() - forwarded
    assert set(PUSH_OPTIONS) == k512_options
    assert set(PUSH_OPTIONS) - set(K512_ONLY_OPTIONS) == k32z_options


@pytest.mark.parametrize(
    "tokens,ep,generation,expected",
    [
        (288, 16, 7, True),
        (289, 16, 7, False),
        (288, 8, 7, False),
        (288, 16, 6, False),
        (288, 16, -1, False),
    ],
)
def test_push_metadata_admission_checks_geometry_and_target(
    monkeypatch, tokens, ep, generation, expected
):
    monkeypatch.setattr(fusion, "get_tpu_generation", lambda: generation)
    i32 = jax.ShapeDtypeStruct((1,), jnp.int32)
    gathered = SimpleNamespace(
        topk=6, n_tokens=tokens * ep, local_blocks=i32, block_hist=i32
    )
    problem = routing.routing_problem(
        gathered,
        (i32,) * 4,
        i32,
        e_total=384,
        ep=ep,
        t_local=tokens,
        block=host.routing_block(tokens, 6),
        tile_m=128,
        shard_stride=host.ragged_stride_bound(tokens * ep, 6, 384, 128),
    )
    assert (
        fusion.admit("routing", problem, tokens=tokens, buckets=fusion.ROUTING_TOKENS)
        is expected
    )


@pytest.mark.parametrize("interpret", [True, False], ids=["interpret", "tpu"])
@pytest.mark.parametrize("scenario", ["uniform", "padded", "all_padded"])
def test_ep16_fused_metadata_matches_native_tables(monkeypatch, interpret, scenario):
    """Compare all consumer tables at the admitted EP16 geometry on one chip."""
    if interpret:
        device = jax.devices("cpu")[0]
        monkeypatch.setattr(fusion, "get_tpu_generation", lambda: 7)
    else:
        devices = jax.devices()
        if devices[0].platform != "tpu":
            pytest.skip("requires a v7 TPU for routing metadata lowering")
        device = devices[0]
        with jax.default_device(device):
            if fusion.get_tpu_generation() != 7:
                pytest.skip("requires a v7 TPU for routing metadata lowering")
    ep, tokens, experts, topk = 16, 288, 384, 6
    block = host.routing_block(tokens, topk)
    kwargs = dict(
        e_total=experts,
        ep=ep,
        t_local=tokens,
        block=block,
        tile_m=128,
        shard_stride=host.ragged_stride_bound(ep * tokens, topk, experts, 128),
    )
    rng = np.random.default_rng(42)
    choices = experts if scenario == "uniform" else 24
    ids = np.stack(
        [rng.choice(choices, topk, replace=False) for _ in range(ep * tokens)]
    )
    if scenario == "padded":
        ids[::3] = experts
        ids[7 * tokens : 8 * tokens] = experts
    elif scenario == "all_padded":
        ids[:] = experts
    counts = np.stack(
        [
            np.bincount(rows.ravel(), minlength=experts + 1)[:experts]
            for rows in ids.reshape(ep, tokens, topk)
        ]
    ).astype(np.int32)
    fused_calls = []
    fuse = partial(routing.fuse_function, interpret=interpret)

    def record_fusion(fn, args, *, name):
        fused_calls.append(name)
        return fuse(fn, args, name=name)

    monkeypatch.setattr(routing, "fuse_function", record_fusion)
    with jax.default_device(device):
        blocks = jnp.asarray(ids, jnp.int32).reshape(-1, block)
        rows_by_dest = jnp.asarray(counts)
        runs = push_host.run_tables(rows_by_dest)
        for rank in (0, 7, 15):
            local = blocks[
                rank * tokens * topk // block : (rank + 1) * tokens * topk // block
            ]
            gathered = host.GatheredPairs(
                blocks,
                local,
                host.pair_block_hist(local, experts),
                rows_by_dest,
                ep * tokens,
                topk,
            )
            me = jnp.int32(rank)
            fused_calls.clear()
            actual = routing.prepare_routing(me, gathered, runs, **kwargs)
            assert fused_calls == ["moe_push_routing_position_metadata"]
            fused_calls.clear()
            expected = routing.prepare_routing(
                me, gathered, runs, fuse_metadata=False, **kwargs
            )
            assert not fused_calls
            _assert_arrays_equal(
                jax.tree.leaves(actual), jax.tree.leaves(expected), shard=rank
            )


# --------------------------------------------------------------------------- #
# Push plan kernels (interpret mode) and the TensorCore scale slab of the K512
# build.
# --------------------------------------------------------------------------- #
def _push_routing(ep, t_local, g_local, topk, *, skew, seed):
    """Global top-k ids [T, topk] (some rows routed nowhere) and plan kwargs;
    a token selects an expert at most once, as top-k does."""
    e_total = g_local * ep
    T = t_local * ep
    rng = np.random.default_rng(seed)
    if skew:
        hot = rng.choice(e_total, size=max(topk, e_total // 12), replace=False)
        idx = np.stack([rng.choice(hot, size=topk, replace=False) for _ in range(T)])
    else:
        idx = np.stack([rng.permutation(e_total)[:topk] for _ in range(T)])
    idx[rng.random(T) < 0.05] = e_total  # rows that route nowhere
    capacity = 128
    kwargs = dict(
        e_total=e_total,
        ep=ep,
        t_local=t_local,
        block=host.routing_block(t_local, topk),
        tile_m=capacity,
        shard_stride=host.ragged_stride_bound(T, topk, e_total, capacity),
    )
    return idx.astype(np.int32), kwargs


def _push_counts(idx, ep, t_local, e_total):
    """rows_by_dest [ep, e_total]: each source shard's pairs per expert."""
    return np.stack(
        [
            np.bincount(
                idx[s * t_local : (s + 1) * t_local].reshape(-1),
                minlength=e_total + 1,
            )[:e_total]
            for s in range(ep)
        ]
    ).astype(np.int32)


def _push_gathered(idx, me, kw):
    ep, t_local, block, e_total = kw["ep"], kw["t_local"], kw["block"], kw["e_total"]
    topk = idx.shape[1]
    blocks = jnp.asarray(idx.reshape(-1, block))
    per_dest = t_local * topk // block
    local = blocks[me * per_dest : (me + 1) * per_dest]
    return host.GatheredPairs(
        expert_blocks=blocks,
        local_blocks=local,
        block_hist=host.pair_block_hist(local, e_total),
        rows_by_dest=jnp.asarray(_push_counts(idx, ep, t_local, e_total)),
        n_tokens=ep * t_local,
        topk=topk,
    )


@pytest.mark.parametrize("n,scale_rows", [(768, 11), (3072, 11), (4096, 21)])
def test_push_scale_slab_onehot_equals_the_scatter(n, scale_rows):
    """The one-hot scale slab is the drop-mode scatter-add bit for bit when
    every routed pair has a slab row of its own: any bit pattern, rows left
    unwritten, and off-slab sentinels dropped."""
    from sglang_tpu.moe.pallas.fused_moe_v2.push_fused_moe.layer_k512 import (
        _scale_slab_onehot,
    )

    rng = np.random.default_rng(n)
    cells = scale_rows * 128
    rows = rng.permutation(cells)[: min(n, cells - 7)]
    slab_row = np.full(n, cells + 5, np.int32)  # off-slab: dropped
    slab_row[rng.permutation(n)[: rows.size]] = rows
    bits = rng.integers(-(2**31), 2**31, n, dtype=np.int64).astype(np.int32)
    bits[:4] = np.array([np.nan, np.inf, -np.inf, -0.0], np.float32).view(np.int32)
    want = (
        jnp.zeros((cells,), jnp.int32)
        .at[jnp.asarray(slab_row)]
        .add(jnp.asarray(bits), mode="drop")
    )
    got = _scale_slab_onehot(jnp.asarray(slab_row), jnp.asarray(bits), scale_rows)
    np.testing.assert_array_equal(
        np.asarray(got).view(np.int32).reshape(-1), np.asarray(want)
    )


@pytest.mark.parametrize("skew", [False, True], ids=["uniform", "skewed"])
@pytest.mark.parametrize("t_local,topk", [(16, 6), (16, 8), (8, 6)])
def test_push_pair_tables_kernel_matches_the_xla_plan(t_local, topk, skew):
    """plan_kernel.build_pair_tables (interpret mode), fed the exchanged
    payload and the table kernel's plan tables, gives routing_pairs_from_tables'
    pos and mirror_pos and the drop-mode scale-slab scatter exactly, at the
    served decode geometry (ep 8, 48 slots), rows routed nowhere included."""
    from sglang_tpu.moe.pallas.fused_moe_v2.push_fused_moe.layer_k512 import (
        _exchange_layout,
        _pack_routing_exchange,
    )
    from sglang_tpu.moe.pallas.fused_moe_v2.push_fused_moe.plan_kernel_k512 import (
        build_pair_tables,
        build_push_tables,
        pair_tables_apply,
    )

    ep, g_local = 8, 48
    idx, kw = _push_routing(ep, t_local, g_local, topk, skew=skew, seed=17 + topk)
    e_total = kw["e_total"]
    layout = _exchange_layout(t_local, topk, e_total, True)
    scale_rows = host.act_scale_slab_rows(kw["shard_stride"])
    assert pair_tables_apply(t_local, topk, layout[0], scale_rows)
    rng = np.random.default_rng(t_local + topk)
    bits = rng.integers(-(2**31), 2**31, (ep, t_local), dtype=np.int64)
    bits = jnp.asarray(bits.astype(np.int32))
    blocks = idx.reshape(ep, t_local * topk // kw["block"], kw["block"])
    counts = _push_counts(idx, ep, t_local, e_total)
    payload = jnp.stack(
        [
            _pack_routing_exchange(
                jnp.asarray(blocks[s].reshape(-1)), bits[s], counts[s], layout
            )
            for s in range(ep)
        ]
    )
    rows = -(-layout[2] // 128)
    out = jnp.pad(payload, ((0, 0), (0, rows * 128 - layout[2]))).reshape(ep, rows, 128)
    for me in (0, 5, ep - 1):
        gathered = _push_gathered(idx, me, kw)
        seg_start = jnp.asarray(np.cumsum(counts[me]) - counts[me])
        tables = build_push_tables(
            gathered.rows_by_dest, seg_start, jnp.int32(me), ep=ep, n_pre=2,
            tile_m=kw["tile_m"], interpret=True,
        )[8]  # fmt: skip
        want_pos, want_mir, slab_row = push_host.routing_pairs_from_tables(
            jnp.int32(me),
            gathered,
            *tables,
            e_total=e_total,
            ep=ep,
            t_local=t_local,
            block=kw["block"],
            shard_stride=kw["shard_stride"],
        )
        want_slab = (
            jnp.zeros((scale_rows * 128,), jnp.int32)
            .at[slab_row]
            .add(jnp.repeat(bits.reshape(-1), topk), mode="drop")
        )
        pos, mir, slab = build_pair_tables(
            out, *tables, jnp.int32(me), ep=ep, t_local=t_local, topk=topk,
            block=kw["block"], e_total=e_total, scale_rows=scale_rows,
            interpret=True,
        )  # fmt: skip
        np.testing.assert_array_equal(np.asarray(pos), np.asarray(want_pos))
        np.testing.assert_array_equal(np.asarray(mir), np.asarray(want_mir))
        np.testing.assert_array_equal(
            np.asarray(slab).view(np.int32).reshape(-1), np.asarray(want_slab)
        )


# Prefill-sized steps: the served mixed step (t_local 1024), a step whose
# routing block is one 128-pair chunk, and an unaligned expert count.
PUSH_WIDE_GEOMETRIES = [
    pytest.param(8, 1024, 48, 6, 256, id="ep8_t1024_g48"),
    pytest.param(8, 512, 48, 4, 128, id="ep8_t512_g48_topk4_block128"),
    pytest.param(4, 128, 50, 8, 256, id="ep4_t128_g50_topk8"),
]
PUSH_WIDE_MODES = ["uniform", "skewed", "one_expert", "mostly_nowhere"]


def _push_wide_routing(ep, t_local, g_local, topk, block, mode, seed):
    """_push_routing's ids and kwargs (with routing block `block`), plus
    degenerate routings: every pair on one expert, or most rows routed
    nowhere."""
    idx, kw = _push_routing(
        ep, t_local, g_local, topk, skew=mode == "skewed", seed=seed
    )
    kw["block"] = block
    rng = np.random.default_rng(seed + 1)
    if mode == "one_expert":
        idx[:] = rng.integers(0, kw["e_total"])
        idx[rng.random(idx.shape[0]) < 0.05] = kw["e_total"]
    elif mode == "mostly_nowhere":
        idx[rng.random(idx.shape[0]) < 0.8] = kw["e_total"]
    return idx, kw


@pytest.mark.parametrize("ep,t_local,g_local,topk,block", PUSH_WIDE_GEOMETRIES)
@pytest.mark.parametrize("mode", PUSH_WIDE_MODES)
def test_push_local_ranks_kernel_matches_the_local_sort(
    ep, t_local, g_local, topk, block, mode
):
    """plan_kernel.build_local_ranks (interpret mode): its counts are
    pair_block_hist's, its segment starts and row map give exactly
    local_sort_tokens' (tok, seg_start), rows routed nowhere included, and
    the argsort path's tokens when every pair is on one expert."""
    from sglang_tpu.moe.pallas.fused_moe_v2.push_fused_moe.plan_kernel_k512 import (
        build_local_ranks,
        wide_pair_tables_apply,
    )

    idx, kw = _push_wide_routing(
        ep, t_local, g_local, topk, block, mode, seed=t_local + topk
    )
    e_total = kw["e_total"]
    assert wide_pair_tables_apply(t_local, topk, block)
    n = t_local * topk
    for me in (0, ep - 1):
        local = jnp.asarray(idx[me * t_local : (me + 1) * t_local].reshape(-1))
        want_counts = host.pair_block_hist(local.reshape(-1, block), e_total).sum(0)
        want_tok, want_seg = push_host.local_sort_tokens(
            local, want_counts, e_total=e_total, topk=topk
        )
        if mode == "one_expert":
            # A token selecting one expert topk times breaks the count path's
            # premise (top-k ids are distinct); the argsort path takes it.
            want_tok = push_host._sorted_tokens_by_argsort(
                local, jnp.cumsum(want_counts), t_local=t_local, topk=topk
            )
        counts, seg, row_map, words = build_local_ranks(
            local, e_total=e_total, topk=topk, block=block, interpret=True
        )
        np.testing.assert_array_equal(np.asarray(counts), np.asarray(want_counts))
        np.testing.assert_array_equal(np.asarray(seg), np.asarray(want_seg))
        np.testing.assert_array_equal(np.asarray(words) & 0xFFFF, np.asarray(local))
        row_map = np.asarray(row_map)
        dest, src = row_map & 0xFFFF, row_map >> 16
        np.testing.assert_array_equal(np.sort(dest), np.arange(n))
        tok = np.empty(n, np.int32)
        tok[dest] = src
        np.testing.assert_array_equal(tok, np.asarray(want_tok))


@pytest.mark.parametrize("ep,t_local,g_local,topk,block", PUSH_WIDE_GEOMETRIES)
@pytest.mark.parametrize("mode", PUSH_WIDE_MODES)
def test_push_pair_rows_kernel_matches_the_xla_plan(
    ep, t_local, g_local, topk, block, mode
):
    """plan_kernel.build_pair_rows (interpret mode), fed a payload carrying
    every source's build_local_ranks words and the table kernel's plan
    tables, gives routing_pairs_from_tables' pos, mirror_pos and slab_row
    exactly, rows routed nowhere included."""
    from sglang_tpu.moe.pallas.fused_moe_v2.push_fused_moe.layer_k512 import (
        _exchange_layout,
        _pack_routing_exchange,
    )
    from sglang_tpu.moe.pallas.fused_moe_v2.push_fused_moe.plan_kernel_k512 import (
        build_local_ranks,
        build_pair_rows,
        build_push_tables,
    )

    idx, kw = _push_wide_routing(
        ep, t_local, g_local, topk, block, mode, seed=3 * t_local
    )
    e_total = kw["e_total"]
    layout = _exchange_layout(t_local, topk, e_total, True)
    blocks = idx.reshape(ep, -1)
    bits = jnp.zeros((t_local,), jnp.int32)
    payload = []
    for s in range(ep):
        counts, _, _, words = build_local_ranks(
            jnp.asarray(blocks[s]), e_total=e_total, topk=topk, block=block,
            interpret=True,
        )  # fmt: skip
        payload.append(_pack_routing_exchange(words, bits, counts, layout))
    rows = -(-layout[2] // 1024) * 8
    out = jnp.pad(jnp.stack(payload), ((0, 0), (0, rows * 128 - layout[2])))
    out = out.reshape(ep, rows, 128)
    counts = _push_counts(idx, ep, t_local, e_total)
    for me in (0, ep - 1):
        gathered = _push_gathered(idx, me, kw)
        seg_start = jnp.asarray(np.cumsum(counts[me]) - counts[me])
        tables = build_push_tables(
            gathered.rows_by_dest, seg_start, jnp.int32(me), ep=ep, n_pre=2,
            tile_m=kw["tile_m"], interpret=True,
        )[8]  # fmt: skip
        want = push_host.routing_pairs_from_tables(
            jnp.int32(me),
            gathered,
            *tables,
            e_total=e_total,
            ep=ep,
            t_local=t_local,
            block=block,
            shard_stride=kw["shard_stride"],
        )
        got = build_pair_rows(
            out, *tables, jnp.int32(me), ep=ep, t_local=t_local, topk=topk,
            e_total=e_total, shard_stride=kw["shard_stride"], interpret=True,
        )  # fmt: skip
        for g_arr, w_arr in zip(got, want):
            np.testing.assert_array_equal(np.asarray(g_arr), np.asarray(w_arr))


def test_push_pair_kernels_admit_only_their_sizes():
    """The decode pair kernel takes a step whose routed pairs fit one payload
    row (DSV4.1 top-6: t_local <= 21, so the c4 verify buckets 6 and 12 but not
    24 or 48); the wide kernels take prefill-sized steps whose sorted rows,
    ranks and source tokens fit their 16-bit word fields."""
    from sglang_tpu.moe.pallas.fused_moe_v2.push_fused_moe.layer_k512 import (
        _exchange_layout,
    )
    from sglang_tpu.moe.pallas.fused_moe_v2.push_fused_moe.plan_kernel_k512 import (
        WIDE_PAIRS_MAX,
        pair_tables_apply,
        wide_pair_tables_apply,
    )

    def decode(t_local):
        return pair_tables_apply(
            t_local, 6, _exchange_layout(t_local, 6, 384, True)[0], 11
        )

    assert [decode(t) for t in (6, 12, 21, 24, 48)] == [1, 1, 1, 0, 0]
    assert wide_pair_tables_apply(1024, 6, 256)
    assert not wide_pair_tables_apply(16, 6, 32)  # decode: build_pair_tables
    assert not wide_pair_tables_apply(288, 6, 64)  # not whole [8, 128] tiles
    assert not wide_pair_tables_apply(16384, 6, 256)  # rows overflow 16 bits
    assert WIDE_PAIRS_MAX < 1 << 15


def test_push_combine_paths_by_token_count():
    """Which push combine a DSV4.1 step takes: the one-pass decode kernel at
    whole bf16 tiles up to COMBINE_TOKENS tokens a shard (48, 64), the
    pipelined fs kernel from COMBINE_MIN_TOKENS up, v2's XLA epilogue between
    and below (the c4 verify buckets 6, 12 and 24)."""
    from sglang_tpu.moe.pallas.fused_moe_v2.kernel import combine_step_tokens
    from sglang_tpu.moe.pallas.fused_moe_v2.push_fused_moe.kernel import (
        decode_combine_applies,
        fs_combine_chunk,
    )

    decode = [t for t in (6, 12, 16, 24, 48, 64, 96, 128) if decode_combine_applies(t)]
    assert decode == [16, 48, 64]
    assert fs_combine_chunk(256, 6) == 32 and fs_combine_chunk(1024, 6) == 32
    # Below COMBINE_MIN_TOKENS v2's step rule refuses, whatever the fs chunk.
    assert not combine_step_tokens(128) and not combine_step_tokens(96)


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__]))


def test_push_routes_k32z_to_upstream_layer_and_the_rest_to_k512(monkeypatch):
    """K32Z keeps layer.py (its program unchanged); every other format runs the
    K512 build in layer_k512.py, whose extra options K32Z refuses."""
    from sglang_tpu.moe.pallas.fused_moe_v2 import push_fused_moe as push
    from sglang_tpu.moe.pallas.fused_moe_v2.host import WeightFormat

    calls = []
    monkeypatch.setattr(
        push._layer, "fused_ep_moe_push", lambda *a, **k: calls.append(("k32z", k))
    )
    monkeypatch.setattr(
        push._layer_k512, "fused_ep_moe_push", lambda *a, **k: calls.append(("k512", k))
    )
    push.fused_ep_moe_push(1, weight_format=WeightFormat.K32Z, sc_prepush=2)
    push.fused_ep_moe_push(1, weight_format="k32z")
    for fmt in (WeightFormat.FP4, WeightFormat.FP8):
        push.fused_ep_moe_push(1, weight_format=fmt, decode_combine=True)
    push.fused_ep_moe_push(1)
    assert [c[0] for c in calls] == ["k32z", "k32z", "k512", "k512", "k512"]
    for name in push.K512_ONLY_OPTIONS:
        with pytest.raises(ValueError, match="not available for K32Z"):
            push.fused_ep_moe_push(1, weight_format=WeightFormat.K32Z, **{name: True})
