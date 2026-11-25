# SPDX-License-Identifier: Apache-2.0
"""CPU checks for the DeepSeek-V4.1 K32Z expert format of the fused EP MoE.

K32Z keeps the checkpoint's block-32 MXFP4 experts lossless: e2m1 codes plus
per-column shift tables (sglang_tpu/dsv4_mxfp4_k32z.py builds them, the
fused_moe_v2 kernels decode them). These tests hold it to:

* the weight conversion: converting a checkpoint-format expert and converting
  it back gives the checkpoint's codes (-0 as +0) and exponents, and every
  weight decoded from the operands as the kernel addresses them equals the
  checkpoint weight;
* the kernel's decode, bit for bit, in the Pallas TPU interpreter: the table
  build, the column-scale fill and the block readers against an independent
  numpy decode (code -> e5m2 byte + 4d, column scale 2^(b - 127));
* the per-bucket choices: which rows per rank push or all-gather, and which
  schedule each bucket takes.

Nothing here needs a TPU. Run it in its own process:

    python -m pytest -q <this file>
"""

import os
import sys
from types import SimpleNamespace

# CPU JAX; no effect once JAX is up.
if "jax" not in sys.modules:
    os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np
import pytest

try:
    from sglang.test.ci.ci_register import register_tpu_ci
except ImportError:  # minimal env without SGLang's runtime dependencies

    def register_tpu_ci(*_args, **_kwargs):
        return None


register_tpu_ci(est_time=60, suite="stage-a-test-tpu-unit", chips=1)

jax = pytest.importorskip("jax")
ml_dtypes = pytest.importorskip("ml_dtypes")
torch = pytest.importorskip("torch")

import dsv41_fused_ep_reference as ref  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax.experimental import pallas as pl  # noqa: E402
from jax.experimental.pallas import tpu as pltpu  # noqa: E402

from sglang_tpu import dsv4_mxfp4_k32z as k32z_loader  # noqa: E402
from sglang_tpu.models.deepseek_v41 import fused_ep as fe  # noqa: E402

# Reduced shape: three gate/up K blocks, a down K tail of 256, six chunks each.
HIDDEN, INTER = 1536, 768
E2M1 = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    + [-0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
)


@pytest.fixture(scope="module")
def v2():
    try:
        package = ref.import_fused_moe_v2(isolated=True)
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"fused_moe_v2 does not import here: {exc!r}")
    from sglang_tpu.moe.pallas.fused_moe_v2 import host, kernel
    from sglang_tpu.moe.pallas.fused_moe_v2.push_fused_moe import layer as push_layer

    return SimpleNamespace(
        package=package, host=host, kernel=kernel, push_layer=push_layer
    )


# --------------------------------------------------------------------------- #
# Checkpoint experts and an independent numpy decode.
# --------------------------------------------------------------------------- #
def _experts(num_experts, seed, *, top):
    """Checkpoint-format experts: codes w13 [E, 2I, H/2], w2 [E, H, I/2] (all
    16 codes) and E8M0 exponents e13 [E, 2I, H/32], e2 [E, H, I/32]: a random
    base per column plus a shift of at most top[0] (gate/up) / top[1] (down),
    each extreme present."""
    rng = np.random.default_rng(seed)
    w13 = rng.integers(0, 256, (num_experts, 2 * INTER, HIDDEN // 2), dtype=np.uint8)
    w2 = rng.integers(0, 256, (num_experts, HIDDEN, INTER // 2), dtype=np.uint8)

    def exps(n, g, t):
        base = rng.integers(110, 124, (num_experts, n, 1))
        d = rng.integers(0, t + 1, (num_experts, n, g))
        d[:, :, 0] = 0  # the column minimum is the base
        d[:, :, -1] = np.where(rng.random((num_experts, n)) < 0.5, t, d[:, :, -1])
        return (base + d).astype(np.uint8)

    return (
        w13,
        exps(2 * INTER, HIDDEN // 32, top[0]),
        w2,
        exps(HIDDEN, INTER // 32, top[1]),
    )


def _convert(w13, e13, w2, e2, bits):
    return k32z_loader.convert_mxfp4_experts_to_k32z(
        torch.from_numpy(w13).view(torch.int8),
        torch.from_numpy(e13.copy()),
        torch.from_numpy(w2).view(torch.int8),
        torch.from_numpy(e2.copy()),
        gate_up_bits=bits,
    )


def _canon(q):
    """Codes with -0 (0x8) rewritten to +0 in both nibbles."""
    lo, hi = q & 0x0F, q & 0xF0
    return (np.where(lo == 0x08, 0, lo) | np.where(hi == 0x80, 0, hi)).astype(np.uint8)


def _codes_kn(w):
    """Checkpoint codes [N, K/2] -> [K, N] (low nibble = even K)."""
    return np.stack([w & 15, w >> 4], -1).reshape(w.shape[0], -1).T


def _shift_base(e):
    """Exponents [N, K/32] -> (d [K/32, N], b [N]) from the checkpoint alone."""
    e = e.astype(np.int64)
    b = e.min(-1)
    return (e - b[:, None]).T, b


def _checkpoint_values(w, e):
    """Exact checkpoint weights, f64 [K, N]."""
    return E2M1[_codes_kn(w)] * np.exp2(
        np.repeat(e.astype(np.float64).T, 32, axis=0) - 127.0
    )


def _kernel_view(ops, e, which, bits):
    """Matmul `which` of expert `e` as the kernel addresses the operands:
    (codes [K, N], shifts [K/32, N], bases [N]), numpy int64."""
    s2 = ops.w2_shift[e].numpy().astype(np.int64)
    g2 = INTER // 32
    if which == 1:
        q, tab, nbits, n, g = (
            ops.w13[e].numpy(),
            ops.w13_shift[e].numpy(),
            bits,
            2 * INTER,
            HIDDEN // 32,
        )
        base_rows = s2[g2 + 2 : g2 + 4]
    else:
        q, tab, nbits, n, g = ops.w2[e].numpy(), ops.w2_shift[e].numpy(), 4, HIDDEN, g2
        base_rows = s2[g2 : g2 + 2]
    codes = np.stack([q & 15, q >> 4], 1).reshape(2 * q.shape[0], n).astype(np.int64)
    per, cols = 8 // nbits, np.arange(n)
    tiles, lanes = cols // 128, cols % 128
    byte_col = (tiles // per) * 128 + lanes
    d = (tab[:g, byte_col].astype(np.int64) >> (nbits * (tiles % per))) & (
        (1 << nbits) - 1
    )
    b = base_rows[tiles % 2, (tiles // 2) * 128 + lanes]
    return codes, d, b


# 2-bit gate/up fields hold shifts up to 3; the serving width is 4.
@pytest.mark.parametrize("bits,top", [(2, (3, 13)), (4, (6, 13))])
def test_weight_conversion_is_lossless(bits, top):
    w13, e13, w2, e2 = _experts(2, seed=5 + bits, top=top)
    ops = _convert(w13, e13, w2, e2, bits)

    back = [t.numpy() for t in k32z_loader.k32z_to_mxfp4(ops, gate_up_bits=bits)]
    np.testing.assert_array_equal(back[0], _canon(w13))
    np.testing.assert_array_equal(back[1], e13)
    np.testing.assert_array_equal(back[2], _canon(w2))
    np.testing.assert_array_equal(back[3], e2)

    # Every weight, decoded from the operands as the kernel addresses them
    # (codes x 2^(b + d - 127)), equals the checkpoint weight.
    for e in range(2):
        for which, w, ex in ((1, w13, e13), (2, w2, e2)):
            codes, d, b = _kernel_view(ops, e, which, bits)
            got = E2M1[codes] * np.exp2(np.repeat(d, 32, axis=0) + b[None, :] - 127.0)
            np.testing.assert_array_equal(got, _checkpoint_values(w[e], ex[e]))


def test_conversion_refuses_shifts_the_fields_cannot_hold():
    w13, e13, w2, e2 = _experts(1, seed=3, top=(6, 13))
    with pytest.raises(ValueError, match="shift"):
        _convert(w13, e13, w2, e2, 2)


# --------------------------------------------------------------------------- #
# The kernel's decode, in the Pallas TPU interpreter.
# --------------------------------------------------------------------------- #
def _pack_u32(q):
    """u8 [K/2, N] K-packed codes -> u32 [K/8, N], word r = bytes of rows
    4r..4r+3 little endian (the kernel's packed-weight view)."""
    r, n = q.shape
    return (
        np.ascontiguousarray(np.swapaxes(q.reshape(r // 4, 4, n), 1, 2))
        .view("<u4")
        .reshape(r // 4, n)
    )


def _k32z_e5m2_bytes(w, e):
    """The K32Z decode in numpy: e2m1 code -> e5m2 byte (-0 as +0) + 4d.
    Returns uint8 [K, N]."""
    lut = np.asarray(E2M1, np.float32).astype(ml_dtypes.float8_e5m2).view(np.uint8)
    lut[8] = 0  # -0 -> +0, as the loader canonicalizes
    d, _ = _shift_base(e)
    return (lut[_codes_kn(w)] + 4 * np.repeat(d, 32, axis=0)).astype(np.uint8)


def _decode_call(v2, bits):
    """One expert's whole decode through the kernel's block readers."""
    hidden, inter = HIDDEN, INTER
    geo = v2.host.k32z_table_geometry(hidden, inter, bits)
    kernel = v2.kernel

    def body(w1, w2, s1, s2, d1, d2, col, tab_vm, col_vm):
        slot = jnp.int32(0)
        kernel.k32z_fill_colscale(s2, slot, col_vm, hidden=hidden, inter=inter)
        col[...] = col_vm[...]
        readers = kernel.k32z_block_readers(
            w1,
            w2,
            s1,
            s2,
            tab_vm,
            slot,
            hidden=hidden,
            inter=inter,
            gate_up_bits=bits,
            interpret=True,  # the interpreter has no stride-0 load
        )
        for read, out, k, n in (
            (readers[0], d1, hidden, 2 * inter),
            (readers[1], d2, inter, hidden),
        ):
            for n0 in range(0, n, 256):
                for b, (k0, rows) in enumerate(kernel.k32z_blocks(k)):
                    out[pl.ds(k0, rows), pl.ds(n0, 256)] = read(b, n0, n0 + 256)

    return pl.pallas_call(
        body,
        out_shape=(
            jax.ShapeDtypeStruct((hidden, 2 * inter), jnp.float8_e5m2),
            jax.ShapeDtypeStruct((inter, hidden), jnp.float8_e5m2),
            jax.ShapeDtypeStruct((8, max(2 * inter, hidden)), jnp.float32),
        ),
        scratch_shapes=[
            pltpu.VMEM((2, 2, geo.tab_rows, 128), jnp.int32),
            pltpu.VMEM((8, max(2 * inter, hidden)), jnp.float32),
        ],
        interpret=pltpu.InterpretParams(),
    )


@pytest.mark.parametrize("bits,top", [(2, (3, 13)), (4, (6, 13))])
def test_kernel_decode_is_k32z_bit_for_bit(v2, bits, top):
    """The kernel's table build, column-scale fill and block readers decode
    every block to code -> e5m2 + 4d, and the column scales to 2^(b - 127)."""
    w13, e13, w2, e2 = _experts(2, seed=11, top=top)
    ops = _convert(w13, e13, w2, e2, bits)
    call = jax.jit(_decode_call(v2, bits))
    for e in range(2):
        d1, d2, col = call(
            jnp.asarray(_pack_u32(ops.w13[e].numpy())[None]),
            jnp.asarray(_pack_u32(ops.w2[e].numpy())[None]),
            jnp.asarray(ops.w13_shift[e].numpy()[None]),
            jnp.asarray(ops.w2_shift[e].numpy()[None]),
        )
        np.testing.assert_array_equal(
            np.asarray(d1).view(np.uint8), _k32z_e5m2_bytes(w13[e], e13[e])
        )
        np.testing.assert_array_equal(
            np.asarray(d2).view(np.uint8), _k32z_e5m2_bytes(w2[e], e2[e])
        )
        col = np.asarray(col)
        np.testing.assert_array_equal(
            col[0, : 2 * INTER],
            np.exp2(_shift_base(e13[e])[1] - 127.0).astype(np.float32),
        )
        np.testing.assert_array_equal(
            col[1, :HIDDEN], np.exp2(_shift_base(e2[e])[1] - 127.0).astype(np.float32)
        )


# --------------------------------------------------------------------------- #
# The per-bucket choices.
# --------------------------------------------------------------------------- #
def test_dispatch_is_chosen_by_rows_per_rank():
    pick = lambda rows: fe.resolve_dispatch(  # noqa: E731
        fe.DISPATCH_AUTO, rows_per_rank=rows, push_max_rows=fe.PUSH_MAX_ROWS
    )
    assert [pick(r) for r in (8, 64, 512, 1024)] == [fe.DISPATCH_PUSH] * 4
    assert pick(2048) == fe.DISPATCH
    # Without "auto" the dispatch is what was asked for.
    for name in (fe.DISPATCH, fe.DISPATCH_PUSH):
        assert fe.resolve_dispatch(name, rows_per_rank=2048, push_max_rows=1) == name


def test_push_recipes(v2):
    recipes = v2.push_layer.K32Z_PUSH_RECIPES
    assert sorted(recipes) == [512, 1024]
    assert recipes[512] == {
        "x_tables": True,
        "fold_select": True,
        "hot_visit": 16,
        "run_sems": 8,
        "self_direct": True,
    }
    assert recipes[1024] == {
        "x_dedup": True,
        "defer_push": True,
        "fold_select": True,
        "early_ship": 2,
        "run_sems": 8,
        "self_direct": True,
        "x_dup_unroll": 8,
    }
    # Decode buckets take the plan's slab rows from the table kernel instead.
    for rows in (8, 16, 32, 64):
        assert v2.host.push_slab_rows_mode(k32z=True, t_local=rows)
        assert not v2.host.push_slab_rows_mode(k32z=False, t_local=rows)
    for rows in (512, 1024, 2048):
        assert not v2.host.push_slab_rows_mode(k32z=True, t_local=rows)


def test_all_gather_recipe_applies_at_2048_only(v2):
    host = v2.host
    for rows, on in ((8, False), (512, False), (1024, False), (2048, True)):
        assert host.combine_fuse_scales(k32z=True, t_local=rows) == on
        assert host.sc_side_core(k32z=True, t_local=rows) == (2 if on else 0)
        assert host.gather_unroll_mode(k32z=True, t_local=rows) == (8 if on else 0)
        assert (
            host.early_ship_select(
                k32z=True, t_local=rows, ep=8, shard_xy=_shard_xy_of(range(8))
            )
            == on
        )
    # K512 builds keep the plain program everywhere.
    assert not host.combine_fuse_scales(k32z=False, t_local=2048)
    assert host.sc_side_core(k32z=False, t_local=2048) == 0
    assert host.gather_unroll_mode(k32z=False, t_local=2048) == 0
    # The selector needs the EP8 chip grid of the plan model.
    assert not host.early_ship_select(k32z=True, t_local=2048, ep=4, shard_xy=None)
    assert host.early_ship_shard_xy(_mesh_of(range(4))) is None


# Physical device d sits on chip d // 2 of the 2 x 2 grid, this (x, y) each.
_CHIP_XY = ((0, 0), (1, 0), (0, 1), (1, 1))
_SERVING_ORDER = (0, 1, 4, 5, 6, 7, 2, 3)  # torch rank -> physical device id


def _mesh_of(device_ids):
    devices = np.empty(len(device_ids), dtype=object)
    for i, d in enumerate(device_ids):
        x, y = _CHIP_XY[d // 2]
        devices[i] = SimpleNamespace(id=d, coords=(x, y, 0), core_on_chip=d % 2)
    return SimpleNamespace(devices=devices)


def _shard_xy_of(device_ids):
    return tuple(d.coords[:2] for d in _mesh_of(device_ids).devices)


# Directed links of the 2 x 2 grid in the table's order (chips (0,0), (1,0),
# (0,1), (1,1); by source chip, then destination chip).
_LINKS = (
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1)),
    ((1, 0), (0, 0)),
    ((1, 0), (1, 1)),
    ((0, 1), (0, 0)),
    ((0, 1), (1, 1)),
    ((1, 1), (1, 0)),
    ((1, 1), (0, 1)),
)
# Links each chip-to-chip push crosses, routed x first, worked out by hand.
_ROUTES = {
    ((0, 0), (1, 0)): (0,),
    ((0, 0), (0, 1)): (1,),
    ((0, 0), (1, 1)): (0, 3),
    ((1, 0), (0, 0)): (2,),
    ((1, 0), (1, 1)): (3,),
    ((1, 0), (0, 1)): (2, 1),
    ((0, 1), (0, 0)): (4,),
    ((0, 1), (1, 1)): (5,),
    ((0, 1), (1, 0)): (5, 6),
    ((1, 1), (1, 0)): (6,),
    ((1, 1), (0, 1)): (7,),
    ((1, 1), (0, 0)): (7, 4),
}


def _expected_incidence(shard_xy):
    inc = [[[0] * 8 for _ in range(8)] for _ in _LINKS]
    for s in range(8):
        for d in range(8):
            for link in _ROUTES.get((shard_xy[s], shard_xy[d]), ()):
                inc[link][s][d] = 1
    return inc


@pytest.mark.parametrize("order", [tuple(range(8)), _SERVING_ORDER])
def test_early_ship_link_model_follows_the_mesh_placement(v2, order):
    host = v2.host
    shard_xy = host.early_ship_shard_xy(_mesh_of(order))
    assert shard_xy == tuple(_CHIP_XY[d // 2] for d in order)
    assert host.early_ship_link_incidence(shard_xy) == _expected_incidence(shard_xy)


def test_early_ship_identity_order_keeps_the_fitted_table(v2):
    # Shard d on chip d // 2 is the placement the model was fitted on.
    host = v2.host
    inc = host.early_ship_link_incidence(host.early_ship_shard_xy(_mesh_of(range(8))))
    assert inc[0][0][2] == 1 and inc[0][0][6] == 1  # shard 0 -> chips (1,0), (1,1)
    # The serving order differs: shards 2, 3 are on chip (0,1), shards 6, 7 on (1,0).
    serving = host.early_ship_link_incidence(
        host.early_ship_shard_xy(_mesh_of(_SERVING_ORDER))
    )
    assert serving != inc


def test_early_ship_needs_a_two_by_two_grid_of_two_shard_chips(v2):
    host = v2.host
    bad = _mesh_of(range(8))
    bad.devices[3] = SimpleNamespace(id=3, coords=(0, 0, 0), core_on_chip=1)
    assert host.early_ship_shard_xy(bad) is None  # chip (0,0) holds three shards
    no_coords = SimpleNamespace(devices=np.array([object()] * 8, dtype=object))
    assert host.early_ship_shard_xy(no_coords) is None


def test_decode_buckets_loop_their_tall_bodies(v2):
    host, ep, g_local = v2.host, 8, 48
    # DeepSeek-V4.1-Flash's expert shape (hidden 5120, intermediate 2304),
    # whose chunk counts the (3, 4) step divides.
    for rows in (8, 16, 32, 64):
        stride = host.ragged_stride_bound(ep * rows, 6, ep * g_local, 256)
        roll, roll_from = host.k32z_roll_plan(stride, ep, g_local, 5120, 2304)
        assert roll == (3, 4) and roll_from == 128, rows
        assert host.k32z_build_plan(stride, ep, g_local, 5120, 2304)[
            "k32z_ffn_roll"
        ] == (3, 4)
    # A shape the (3, 4) step does not divide takes the plain count.
    assert host.k32z_roll_plan(1, ep, g_local, 1536, 512) == (2, 128)
    # Large builds drop rungs and defer the push; they do not loop.
    for rows, rungs in ((1024, (64, 128)), (2048, (32, 128))):
        stride = host.ragged_stride_bound(ep * rows, 6, ep * g_local, 256)
        plan = host.k32z_build_plan(stride, ep, g_local, 5120, 2304)
        assert plan["k32z_ffn_roll"] == 0
        assert plan["k32z_rungs"] == rungs and plan["defer_push"]
