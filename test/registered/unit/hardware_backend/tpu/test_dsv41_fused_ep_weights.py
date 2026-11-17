# SPDX-License-Identifier: Apache-2.0
"""CPU checks for DeepSeek-V4.1 Flash MoE weights on fused_ep_moe_v2.

The gmm preparation (sglang_tpu/dsv4_mxfp4_moe.py) turns V4.1's block-32
MXFP4 experts into block-512 packed FP4 for gmm_v2, the gmm layout.
dsv41_fused_ep_reference.gmm_to_fused_ep_weights relayouts those prepared
tensors for the fused EP kernel on the host (the scales through
sglang_tpu/models/deepseek_v41/fused_ep.py). These tests run the gmm
preparation itself on checkpoint-format random experts and hold the relayout
to it bit for bit, check the FP32 reference against a naive per-token loop,
and check the production kernel arguments. Nothing here needs a TPU.

The gmm preparation runs from this tree's sources with only its SGLang runtime
imports stubbed (dsv41_fused_ep_reference.load_gmm_mxfp4_prep), so a minimal
CPU env (jax, ml_dtypes, torch, numpy, pytest) is enough:

    PYTHONPATH=<tree>/python:<tree> python -m pytest -q <this file>
"""

import inspect
import os
import sys
import zlib

# CPU JAX (bit-exact checks as CPU XLA computes them) with four virtual devices
# for the sharded checks, also on a TPU host; no effect once JAX is initialized.
if "jax" not in sys.modules:
    os.environ["JAX_PLATFORMS"] = "cpu"
    if "device_count" not in os.environ.get("XLA_FLAGS", ""):
        os.environ["XLA_FLAGS"] = (
            os.environ.get("XLA_FLAGS", "")
            + " --xla_force_host_platform_device_count=4"
        ).strip()

import numpy as np
import pytest

try:
    from sglang.test.ci.ci_register import register_tpu_ci
except ImportError:  # minimal env without SGLang's runtime dependencies

    def register_tpu_ci(*_args, **_kwargs):
        return None


# The TPU unit lane has jax and ml_dtypes; the CPU lane does not.
register_tpu_ci(est_time=45, suite="stage-a-test-tpu-unit", chips=1)

jax = pytest.importorskip("jax")
pytest.importorskip("ml_dtypes")
torch = pytest.importorskip("torch")

import dsv41_fused_ep_reference as ref  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax import lax  # noqa: E402
from jax.sharding import Mesh  # noqa: E402

from sglang_tpu.models.deepseek_v41 import fused_ep as fe  # noqa: E402

# (experts, hidden, logical intermediate): the served geometry with few
# experts, and a reduced one with the same structure (intermediate pads by
# less than one 512 block inside each gate/up half).
REAL = (4, 5120, 2304)
REDUCED = (4, 1024, 768)


def _padded(inter):
    return -(-inter // 512) * 512


def _checkpoint_experts(num_experts, hidden, inter, seed, *, zero_up=(), zero_gate=()):
    """Checkpoint-format block-32 MXFP4 experts as SGLang's loader leaves them:
    w13 int8 [E, 2I, H/2] (w1 rows, then w3 rows), w2 int8 [E, H, I/2], and f32
    scales 2^(e8m0 - 127) with exponents varying per 32-block."""
    rng = np.random.default_rng(seed)
    w13 = rng.integers(0, 256, (num_experts, 2 * inter, hidden // 2), dtype=np.uint8)
    w2 = rng.integers(0, 256, (num_experts, hidden, inter // 2), dtype=np.uint8)
    e13 = rng.integers(117, 122, (num_experts, 2 * inter, hidden // 32), dtype=np.uint8)
    e2 = rng.integers(117, 122, (num_experts, hidden, inter // 32), dtype=np.uint8)
    for e in zero_up:
        w13[e, inter:] = 0
    for e in zero_gate:
        w13[e, :inter] = 0
    return (
        torch.from_numpy(w13),
        ref.e8m0_to_float32(e13),
        torch.from_numpy(w2),
        ref.e8m0_to_float32(e2),
    )


@pytest.fixture(scope="module")
def prepared():
    """geometry -> (checkpoint tensors, gmm-layout tensors as numpy)."""
    cache = {}

    def get(geometry, **zero):
        key = (geometry, tuple(sorted((k, tuple(v)) for k, v in zero.items())))
        if key not in cache:
            ckpt = _checkpoint_experts(
                *geometry, seed=zlib.crc32(repr(key).encode()) % 1000, **zero
            )
            # The preparation frees what it replaces; keep the checkpoint copies.
            gmm = ref.run_gmm_prep(*(t.clone() for t in ckpt))
            cache[key] = (ckpt, {k: v.numpy() for k, v in gmm.items()})
        return cache[key]

    return get


def _gmm_layout_dequant(packed_kn, scale):
    """gmm-layout (E, K/2, N) bytes + (E, K/512, 1, N) scale -> f32 [E, K, N],
    decoded the way test_dsv4_mxfp4_moe.py:_unpack_gmm_v2_fp4_to_nk does,
    with the preparation's own code table: independent of the relayout under
    test."""
    lut = torch.tensor(
        ref.load_gmm_mxfp4_prep().module._FP4_E2M1_LUT, dtype=torch.float32
    )
    w_nk_u8 = torch.from_numpy(packed_kn).permute(0, 2, 1).contiguous()
    low = (w_nk_u8 & 0x0F).to(torch.long)
    high = ((w_nk_u8 >> 4) & 0x0F).to(torch.long)
    w_fp32 = torch.stack((lut[low], lut[high]), dim=-1).flatten(-2)  # [E, N, K]
    scale_nk = torch.from_numpy(scale).squeeze(2).permute(0, 2, 1).contiguous()
    deq = (w_fp32.unflatten(-1, (-1, 512)) * scale_nk.unsqueeze(-1)).flatten(-2)
    return deq.permute(0, 2, 1).contiguous().numpy()


def _bits(a):
    return np.asarray(a, dtype=np.float32).view(np.uint32)


def test_fp4_code_table_is_ml_dtypes_and_jax():
    lut = np.asarray(ref.load_gmm_mxfp4_prep().module._FP4_E2M1_LUT, np.float32)
    codes = np.arange(16, dtype=np.uint8)
    np.testing.assert_array_equal(_bits(lut), _bits(ref.FP4_E2M1_VALUES))
    np.testing.assert_array_equal(
        _bits(ref.codes_to_float4(codes).astype(np.float32)), _bits(lut)
    )
    on_jax = jnp.asarray(ref.codes_to_float4(codes)).astype(jnp.float32)
    np.testing.assert_array_equal(_bits(on_jax), _bits(lut))


@pytest.mark.parametrize("geometry", [REDUCED, REAL], ids=["reduced", "real"])
@pytest.mark.parametrize("scale_layout", ["flat", "grouped"])
def test_relayout_is_bit_exact(prepared, geometry, scale_layout):
    num_experts, hidden, inter = geometry
    ip = _padded(inter)
    _, gmm = prepared(geometry)
    w13, s13, w2, s2 = (
        gmm["w13_weight"],
        gmm["w13_weight_scale"],
        gmm["w2_weight"],
        gmm["w2_weight_scale"],
    )
    # The gmm layout (see fused_ep.py).
    assert w13.dtype == np.uint8 and w13.shape == (num_experts, hidden // 2, 2 * ip)
    assert s13.dtype == np.float32 and s13.shape == (
        num_experts,
        hidden // 512,
        1,
        2 * ip,
    )
    assert w2.dtype == np.uint8 and w2.shape == (num_experts, ip // 2, hidden)
    assert s2.dtype == np.float32 and s2.shape == (num_experts, ip // 512, 1, hidden)

    kw = ref.gmm_to_fused_ep_weights(w13, s13, w2, s2, scale_layout=scale_layout)
    f4 = np.dtype(jnp.float4_e2m1fn)
    assert kw.w1.dtype == f4 and kw.w1.shape == (num_experts, hidden, 2 * ip)
    assert kw.w2.dtype == f4 and kw.w2.shape == (num_experts, ip, hidden)

    # Nibble order: K row 2j is the low nibble of gmm-layout byte row j.
    for w, packed in ((kw.w1, w13), (kw.w2, w2)):
        codes = w.view(np.uint8)
        np.testing.assert_array_equal(codes[:, 0::2], packed & 0x0F)
        np.testing.assert_array_equal(codes[:, 1::2], packed >> 4)

    # Scale layouts.
    for got, src, k, n in (
        (kw.w1_scale, s13, hidden, 2 * ip),
        (kw.w2_scale, s2, ip, hidden),
    ):
        blocks = k // 512
        if scale_layout == "grouped":
            assert got.shape == (num_experts, blocks, n)
            per_expert = got
        else:
            rows = fe.fp4_scale_rows(k, 512, padded=True)
            assert rows == -(-blocks // 8) * 8
            assert got.shape == (num_experts * rows, n)
            per_expert = got.reshape(num_experts, rows, n)
            assert not per_expert[:, blocks:].any(), "flat scale pad rows must be 0"
            per_expert = per_expert[:, :blocks]
        np.testing.assert_array_equal(_bits(per_expert), _bits(src[:, :, 0, :]))
    if geometry == REAL:
        assert kw.w1.shape == (num_experts, 5120, 5120) and kw.w2.shape == (
            num_experts,
            2560,
            5120,
        )
        if scale_layout == "flat":
            assert kw.w1_scale.shape == (num_experts * 16, 5120)
            assert kw.w2_scale.shape == (num_experts * 8, 5120)

    # Dequantized values: kernel layout == gmm layout, bit for bit, on the
    # host (ml_dtypes) and through JAX.
    for w, s, packed, src in (
        (kw.w1, kw.w1_scale, w13, s13),
        (kw.w2, kw.w2_scale, w2, s2),
    ):
        want = _gmm_layout_dequant(packed, src)
        got_np = ref.dequantize_kernel_layout(w, s)
        np.testing.assert_array_equal(_bits(got_np), _bits(want))
        got_jax = ref.dequantize_kernel_layout(jnp.asarray(w), jnp.asarray(s))
        np.testing.assert_array_equal(_bits(got_jax), _bits(want))
        del want, got_np, got_jax

    # The padded intermediate is exactly zero: w1's columns [I, Ip) of each
    # half (codes and scales), w2's rows [I, Ip) (codes; their block scale is
    # shared with real rows).
    codes1 = kw.w1.view(np.uint8)
    for lo, hi in ((inter, ip), (ip + inter, 2 * ip)):
        assert not codes1[:, :, lo:hi].any()
        assert not kw.w1_scale[..., lo:hi].any()
    assert codes1[:, :, :inter].any() and codes1[:, :, ip : ip + inter].any()
    codes2 = kw.w2.view(np.uint8)
    assert not codes2[:, inter:ip].any()
    assert codes2[:, :inter].any()


def test_gate_up_order(prepared):
    """w1 (gate) lands in w13 columns [0, Ip), w3 (up) in [Ip, 2Ip): expert 1
    has an all-zero up projection, expert 2 an all-zero gate."""
    num_experts, hidden, inter = REDUCED
    ip = _padded(inter)
    ckpt, gmm = prepared(REDUCED, zero_up=(1,), zero_gate=(2,))
    kw = ref.gmm_to_fused_ep_weights(
        gmm["w13_weight"],
        gmm["w13_weight_scale"],
        gmm["w2_weight"],
        gmm["w2_weight_scale"],
    )
    deq = ref.dequantize_kernel_layout(kw.w1, kw.w1_scale)  # [E, H, 2Ip]
    assert not deq[1, :, ip:].any() and deq[1, :, :inter].any()
    assert not deq[2, :, :ip].any() and deq[2, :, ip : ip + inter].any()

    # And the content follows: expert 0's kernel gate block approximates the
    # checkpoint's w1 (the requant to block 512 is lossy), not its w3.
    prep_mod = ref.load_gmm_mxfp4_prep().module
    block32 = prep_mod._dequantize_dsv4_mxfp4_block32(ckpt[0][:1], ckpt[1][:1])[
        0
    ].numpy()
    gate_ckpt, up_ckpt = block32[:inter].T, block32[inter:].T  # [H, I]

    def rel(a, b):
        return float(np.linalg.norm(a - b) / np.linalg.norm(b))

    gate_kernel = deq[0, :, :inter]
    up_kernel = deq[0, :, ip : ip + inter]
    assert rel(gate_kernel, gate_ckpt) < 0.5 < 1.0 < rel(gate_kernel, up_ckpt)
    assert rel(up_kernel, up_ckpt) < 0.5 < 1.0 < rel(up_kernel, gate_ckpt)


def test_device_bridges_match_host_bridge(prepared):
    num_experts, hidden, inter = REDUCED
    _, gmm = prepared(REDUCED)
    w13 = gmm["w13_weight"]
    host = ref.unpack_kpacked_fp4_codes(w13)
    on_jax = np.asarray(ref.kpacked_bytes_to_float4_jax(jnp.asarray(w13)))
    np.testing.assert_array_equal(on_jax.view(np.uint8), host)

    # Torch-native float4_e2m1fn_x2 storage packs N, low nibble = even column.
    x2 = ref.gmm_to_torch_fp4_x2_bytes(w13)
    assert x2.shape == (num_experts, hidden, w13.shape[2] // 2)
    unpacked = np.stack((x2 & 0x0F, x2 >> 4), axis=-1).reshape(host.shape)
    np.testing.assert_array_equal(unpacked, host)
    via_bitcast = lax.bitcast_convert_type(jnp.asarray(x2), jnp.float4_e2m1fn)
    np.testing.assert_array_equal(
        np.asarray(via_bitcast).reshape(host.shape).view(np.uint8), host
    )


def _isolated_kernel_package():
    try:
        return ref.import_fused_moe_v2(isolated=True)
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"fused_moe_v2 does not import here: {exc!r}")


def test_constants_and_activation_match_kernel_package():
    package = _isolated_kernel_package()
    from sglang_tpu.moe.pallas.fused_moe_v2 import activations

    for k in (5120, 2560, 1024, 512):
        for padded in (False, True):
            assert fe.fp4_scale_rows(
                k, 512, padded=padded
            ) == package.host.fp4_scale_rows(k, 512, padded=padded)
    assert package.weight_format_of_dtype(jnp.float4_e2m1fn) == package.WeightFormat.FP4
    assert fe.ACT_FN in package.ACT_FNS
    assert package.host.U32_SUBLANE_TILE == fe._U32_SUBLANE_TILE

    rng = np.random.default_rng(3)
    gate = jnp.asarray(rng.normal(0, 12, (64, 256)).astype(np.float32))
    up = jnp.asarray(rng.normal(0, 12, (64, 256)).astype(np.float32))
    want = activations.silu_and_mul_with_clamp(gate, up)
    np.testing.assert_array_equal(_bits(ref.clamped_swiglu(gate, up)), _bits(want))
    assert float(jnp.max(jnp.abs(gate))) > 10 and float(jnp.max(jnp.abs(up))) > 10


def _naive_moe(x, gating, w1, w2, bias, *, topk, scale, limit, logical):
    """Per-token float64 loop following the official Gate/Expert code."""
    inter = w1.shape[2] // 2
    out = np.zeros(x.shape, np.float64)
    clamped = 0
    for t in range(x.shape[0]):
        g = gating[t].astype(np.float64)
        if not np.isfinite(g).any():
            continue
        scores = np.sqrt(np.logaddexp(0.0, g))
        ids = np.argsort(-(scores + bias), kind="stable")[:topk]
        weights = scores[ids] / (scores[ids].sum() + 1e-20) * scale
        for wt, e in zip(weights, ids):
            h = x[t].astype(np.float64) @ w1[e].astype(np.float64)
            gate, up = h[:logical], h[inter : inter + logical]
            clamped += int((gate > limit).sum() + (np.abs(up) > limit).sum())
            gate = np.minimum(gate, limit)
            up = np.clip(up, -limit, limit)
            act = gate / (1.0 + np.exp(-gate)) * up
            out[t] += wt * (act @ w2[e][:logical].astype(np.float64))
    return out, clamped


def _tiny_case(seed, *, tokens=12, experts=16, hidden=64, inter=48, ip=64, pad=2):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, (tokens, hidden)).astype(np.float32)
    gating = rng.normal(0, 1.5, (tokens, experts)).astype(np.float32)
    gating[-pad:] = -np.inf
    bias = rng.normal(0, 0.3, (experts,)).astype(np.float32)
    w1 = np.zeros((experts, hidden, 2 * ip), np.float32)
    w1[:, :, :inter] = rng.normal(0, 0.6, (experts, hidden, inter))
    w1[:, :, ip : ip + inter] = rng.normal(0, 0.6, (experts, hidden, inter))
    w2 = np.zeros((experts, ip, hidden), np.float32)
    w2[:, :inter] = rng.normal(0, 0.1, (experts, inter, hidden))
    return x, gating, bias, w1, w2


def test_reference_moe_matches_naive_loop():
    x, gating, bias, w1, w2 = _tiny_case(7)
    want, clamped = _naive_moe(
        x,
        gating,
        w1,
        w2,
        bias.astype(np.float64),
        topk=6,
        scale=1.5,
        limit=10.0,
        logical=48,
    )
    assert clamped > 0, "the case must exercise the swiglu clamps"
    for logical in (48, None):
        got = np.asarray(
            ref.reference_moe(
                x,
                gating,
                w1,
                w2,
                bias,
                topk=6,
                routed_scaling_factor=1.5,
                logical_intermediate_size=logical,
            )
        )
        assert not got[-2:].any(), "padding rows (all -inf logits) must be 0"
        real = slice(0, x.shape[0] - 2)
        per_row = np.linalg.norm(got[real] - want[real], axis=-1) / np.linalg.norm(
            want[real], axis=-1
        )
        assert per_row.max() < 1e-5, per_row
    # The wrong-expert control moves the output far away.
    rotated = np.asarray(ref.reference_moe(x, gating, w1, w2, bias, expert_id_shift=1))
    assert np.linalg.norm(rotated[:-2] - want[:-2]) > 0.5 * np.linalg.norm(want[:-2])


def _cpu_mesh(experts):
    devices = jax.devices()
    n = max(d for d in (4, 2, 1) if d <= len(devices) and experts % d == 0)
    return Mesh(np.asarray(devices[:n]), ("ep",))


@pytest.mark.parametrize("scale_layout", ["flat", "grouped"])
def test_sharded_reference_on_kernel_operands(prepared, scale_layout):
    """make_reference_moe_sharded over float4 operands == reference_moe over
    the dequantized weights (and both scale layouts give the same output)."""
    num_experts, hidden, inter = REDUCED
    _, gmm = prepared(REDUCED)
    kw = ref.gmm_to_fused_ep_weights(
        gmm["w13_weight"],
        gmm["w13_weight_scale"],
        gmm["w2_weight"],
        gmm["w2_weight_scale"],
        scale_layout=scale_layout,
    )
    mesh = _cpu_mesh(num_experts)
    rng = np.random.default_rng(11)
    tokens = 4 * mesh.shape["ep"]
    x = (rng.normal(0, 1, (tokens, hidden))).astype(jnp.bfloat16)
    gating = rng.normal(0, 1, (tokens, num_experts)).astype(np.float32)
    gating[-1] = -np.inf
    bias = rng.normal(0, 0.1, (num_experts,)).astype(np.float32)
    common = dict(topk=2, logical_intermediate_size=inter)
    sharded = ref.make_reference_moe_sharded(mesh, **common)
    got = np.asarray(sharded(x, gating, kw.w1, kw.w2, kw.w1_scale, kw.w2_scale, bias))
    want = np.asarray(
        ref.reference_moe(
            x,
            gating,
            ref.dequantize_kernel_layout(kw.w1, kw.w1_scale),
            ref.dequantize_kernel_layout(kw.w2, kw.w2_scale),
            bias,
            **common,
        )
    )
    assert not got[-1].any()
    np.testing.assert_allclose(got, want, rtol=2e-5, atol=1e-6 * np.abs(want).max())


def test_make_fused_ep_moe_passes_production_settings():
    package = _isolated_kernel_package()
    calls = []

    def fake_kernel(x, w1, w2, w1_scale, w2_scale, gating, **kwargs):
        calls.append(
            dict(
                kwargs,
                gating_dtype=gating.dtype,
                positional=(w1, w2, w1_scale, w2_scale),
            )
        )
        return jnp.zeros_like(x)

    mesh = _cpu_mesh(8)
    ep = mesh.shape["ep"]
    moe = fe.make_fused_ep_moe(mesh, kernel_fn=fake_kernel)
    tokens = 8 * ep
    out = moe(
        jnp.zeros((tokens, 512), jnp.bfloat16),
        jnp.zeros((tokens, 8), jnp.bfloat16),
        jnp.zeros((8, 512, 1024), jnp.float4_e2m1fn),
        jnp.zeros((8, 512, 512), jnp.float4_e2m1fn),
        jnp.zeros((8 * 8, 1024), jnp.float32),
        jnp.zeros((8 * 8, 512), jnp.float32),
        jnp.zeros((8,), jnp.bfloat16),
    )
    assert out.shape == (tokens, 512) and out.dtype == jnp.bfloat16
    (kwargs,) = calls
    assert kwargs.pop("gating_dtype") == jnp.float32
    assert kwargs.pop("score_bias").dtype == jnp.float32
    kwargs.pop("positional")
    assert kwargs.pop("mesh") is mesh
    assert kwargs == dict(
        topk=6,
        renormalize=True,
        capacity=128,
        weight_format=package.WeightFormat.FP4,
        rhs_qb=512,
        act_fn="silu_and_mul_with_clamp",
        rank=None,
        mesh_ep_ranks=tuple(range(ep)),
        sharded_plan=True,
        scoring_fn="sqrtsoftplus",
        routed_scaling_factor=1.5,
        logical_intermediate_size=2304,
        preprocess_topology=None,
        dispatch="all_gather",
    )
    # Every keyword exists on the real entry point.
    inspect.signature(package.fused_ep_moe_v2).bind(
        None, None, None, None, None, None, mesh=mesh, score_bias=None, **kwargs
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
