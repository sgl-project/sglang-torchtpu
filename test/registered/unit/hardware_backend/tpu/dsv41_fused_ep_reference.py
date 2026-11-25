# SPDX-License-Identifier: Apache-2.0
"""Test support for DeepSeek-V4.1 fused EP (``fused_ep`` / ``fused_ep_op`` in
``sglang_tpu/models/deepseek_v41/``), imported by the tests next to it.

Holds:

* the host relayout of the gmm layout into the kernel's operands
  (:func:`gmm_to_fused_ep_weights`) and the host, JAX and torch-x2 bridges of
  the K-packed bytes (layout facts in ``fused_ep.py``'s docstring);
* an FP32 reference for V4.1's routed experts (sqrt(softplus) scores,
  bias-corrected top-k, unbiased weights, renormalize, x1.5, clamped SwiGLU)
  over dequantized kernel-layout weights, on one device or spread over an EP
  mesh;
* loaders that run without SGLang's runtime or torch_tpu: the
  ``fused_moe_v2`` package without ``sglang_tpu.moe.pallas.__init__``, and
  the real gmm preparation of the MXFP4 experts with its runtime imports
  stubbed.
"""

from __future__ import annotations

import contextlib
import functools
import importlib
import importlib.machinery
import importlib.util
import os
import sys
import types
from pathlib import Path
from typing import NamedTuple

import numpy as np

from sglang_tpu.models.deepseek_v41.fused_ep import (
    LOGICAL_INTER,
    RENORMALIZE,
    RHS_QB,
    ROUTED_SCALING_FACTOR,
    SWIGLU_LIMIT,
    TOPK,
    gmm_scale_to_kernel,
)

try:
    from sglang.test.ci.ci_register import register_tpu_ci
except ImportError:  # minimal env without SGLang's runtime dependencies

    def register_tpu_ci(*_args, **_kwargs):
        return None


# Every .py under test/registered/ must carry a registration; the tests that
# import this module register the real rows.
register_tpu_ci(
    est_time=0,
    suite="stage-a-test-tpu-unit",
    disabled="helper module, no tests",
)

# E2M1 code -> value. Identical to dsv4_mxfp4_moe._FP4_E2M1_LUT and to ml_dtypes'.
FP4_E2M1_VALUES = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)


# --------------------------------------------------------------------------- #
# gmm layout -> kernel layout (host).
# --------------------------------------------------------------------------- #
class FusedEPWeights(NamedTuple):
    """Kernel operands for one rank (or the global stack): numpy on the host."""

    w1: np.ndarray  # float4_e2m1fn [E, H, 2*Ip]
    w2: np.ndarray  # float4_e2m1fn [E, Ip, H]
    w1_scale: np.ndarray  # f32 [E, H/qb, 2*Ip] or [E*rows(H), 2*Ip]
    w2_scale: np.ndarray  # f32 [E, Ip/qb, H] or [E*rows(Ip), H]


def _float4_dtype():
    import ml_dtypes

    return np.dtype(ml_dtypes.float4_e2m1fn)


def _as_uint8_numpy(t) -> np.ndarray:
    """A torch (u)int8 tensor or numpy array viewed as uint8, without copying."""
    if hasattr(t, "detach"):  # torch.Tensor
        import torch

        t = t.detach().cpu()
        if t.dtype == torch.int8:
            t = t.view(torch.uint8)
        if t.dtype != torch.uint8:
            raise TypeError(f"expected packed (u)int8 FP4 bytes, got {t.dtype}")
        return t.contiguous().numpy()
    a = np.asarray(t)
    if a.dtype == np.int8:
        a = a.view(np.uint8)
    if a.dtype != np.uint8:
        raise TypeError(f"expected packed (u)int8 FP4 bytes, got {a.dtype}")
    return a


def unpack_kpacked_fp4_codes(packed: np.ndarray) -> np.ndarray:
    """gmm-layout bytes ``[E, K/2, N]`` -> E2M1 codes ``[E, K, N]`` (uint8, 0..15).

    Row ``2j`` of K is the low nibble of byte row ``j``, row ``2j+1`` the high
    nibble.
    """
    packed = np.asarray(packed, dtype=np.uint8)
    e, kh, n = packed.shape
    codes = np.empty((e, kh, 2, n), dtype=np.uint8)
    np.bitwise_and(packed, 0x0F, out=codes[:, :, 0, :])
    np.right_shift(packed, 4, out=codes[:, :, 1, :])
    return codes.reshape(e, 2 * kh, n)


def codes_to_float4(codes: np.ndarray) -> np.ndarray:
    """uint8 E2M1 codes -> ``ml_dtypes.float4_e2m1fn`` (same bytes, a view).

    ml_dtypes stores one float4 per byte in the low nibble, with the same code
    table as the gmm layout.
    """
    codes = np.asarray(codes)
    if codes.dtype != np.uint8:
        raise TypeError(f"codes must be uint8, got {codes.dtype}")
    return codes.view(_float4_dtype())


def _check_gmm_shapes(w13, s13, w2, s2, rhs_qb):
    e, h_half, n1 = w13.shape
    hidden = 2 * h_half
    e2, ip_half, h2 = w2.shape
    inter = 2 * ip_half
    if e2 != e or h2 != hidden or n1 != 2 * inter:
        raise ValueError(
            f"gmm-layout w13 {w13.shape} / w2 {w2.shape} are not [E, H/2, 2Ip] / "
            "[E, Ip/2, H] of one expert set"
        )
    if hidden % rhs_qb or inter % rhs_qb:
        raise ValueError(f"block {rhs_qb} must divide H={hidden} and Ip={inter}")
    for name, s, k, n in (("w13", s13, hidden, n1), ("w2", s2, inter, hidden)):
        want = (e, k // rhs_qb, 1, n)
        if tuple(s.shape) not in (want, (e, k // rhs_qb, n)):
            raise ValueError(f"{name} scale {tuple(s.shape)} is not {want}")
    return e, hidden, inter


def gmm_to_fused_ep_weights(
    w13_weight,
    w13_weight_scale,
    w2_weight,
    w2_weight_scale,
    *,
    rhs_qb: int = RHS_QB,
    scale_layout: str = "flat",
) -> FusedEPWeights:
    """Relayout the gmm layout's tensors (torch or numpy) for ``fused_ep_moe_v2``.

    No arithmetic touches a value: weights are a nibble split along K plus a
    dtype view, scales a squeeze (plus zero row padding for ``"flat"``). The
    gate/up order and the zero padding are the gmm layout's and already the
    kernel's.
    """
    w13 = _as_uint8_numpy(w13_weight)
    w2 = _as_uint8_numpy(w2_weight)
    _, hidden, inter = _check_gmm_shapes(
        w13, w13_weight_scale, w2, w2_weight_scale, rhs_qb
    )
    return FusedEPWeights(
        w1=codes_to_float4(unpack_kpacked_fp4_codes(w13)),
        w2=codes_to_float4(unpack_kpacked_fp4_codes(w2)),
        w1_scale=gmm_scale_to_kernel(
            w13_weight_scale, k=hidden, rhs_qb=rhs_qb, layout=scale_layout
        ),
        w2_scale=gmm_scale_to_kernel(
            w2_weight_scale, k=inter, rhs_qb=rhs_qb, layout=scale_layout
        ),
    )


def kpacked_bytes_to_float4_jax(packed):
    """Device-side bridge: gmm-layout bytes ``[E, K/2, N]`` (jax uint8) -> float4
    ``[E, K, N]``. ``bitcast_convert_type`` to a narrower type appends a minor
    axis holding the low nibble at index 0; that axis is moved next to K."""
    import jax.numpy as jnp
    from jax import lax

    e, kh, n = packed.shape
    pairs = lax.bitcast_convert_type(packed.astype(jnp.uint8), jnp.float4_e2m1fn)
    return jnp.moveaxis(pairs, -1, 2).reshape(e, 2 * kh, n)


def gmm_to_torch_fp4_x2_bytes(w_packed_k) -> np.ndarray:
    """gmm-layout bytes ``[E, K/2, N]`` (K-packed) -> ``[E, K, N/2]`` bytes
    packed on N, low nibble = even column: the storage of a torch
    ``float4_e2m1fn_x2`` ``[E, K, N/2]``, which torch_tpu presents to JAX as
    float4 ``[E, K, N]``."""
    codes = unpack_kpacked_fp4_codes(_as_uint8_numpy(w_packed_k))
    return codes[..., 0::2] | (codes[..., 1::2] << 4)


# --------------------------------------------------------------------------- #
# FP32 reference.
# --------------------------------------------------------------------------- #
def dequantize_kernel_layout(w, scale, *, rhs_qb: int = RHS_QB):
    """Kernel-layout FP4 ``[E, K, N]`` + scale (grouped, flat, or gmm-layout 4-D) ->
    f32 ``[E, K, N]`` = value * block scale. numpy in -> numpy out (exact, via
    ml_dtypes); otherwise jax.numpy."""
    e, k, n = w.shape
    blocks = k // rhs_qb
    if isinstance(w, np.ndarray) and isinstance(scale, np.ndarray):
        xp = np
        w32 = w.astype(np.float32)
        s = scale.astype(np.float32, copy=False)
    else:
        import jax.numpy as xp

        w32 = xp.asarray(w).astype(xp.float32)
        s = xp.asarray(scale, dtype=xp.float32)
    if s.ndim == 4:  # gmm layout [E, nb, 1, N]
        s = s.reshape(e, blocks, n)
    elif s.ndim == 2:  # row-padded flat [E * rows, N]
        s = s.reshape(e, -1, n)[:, :blocks]
    if tuple(s.shape) != (e, blocks, n):
        raise ValueError(f"scale {tuple(s.shape)} does not match weight {w.shape}")
    out = w32.reshape(e, blocks, rhs_qb, n) * s[:, :, None, :]
    return out.reshape(e, k, n)


def sqrtsoftplus(logits):
    """Scores as the kernel's generic router computes them.

    The kernel's fused normalized router uses an algebraically equal softplus;
    this reference follows the generic path.
    """
    import jax
    import jax.numpy as jnp

    return jnp.sqrt(jax.nn.softplus(logits.astype(jnp.float32)))


def clamped_swiglu(gate, up, limit: float = SWIGLU_LIMIT):
    """Same ops as fused_moe_v2/activations.py:silu_and_mul_with_clamp (bit
    equality is checked in tests): gate clipped from above, up on both sides."""
    import jax
    import jax.numpy as jnp

    gate = jnp.clip(gate, max=limit)
    up = jnp.clip(up, min=-limit, max=limit)
    return jax.nn.silu(gate) * up


def route(
    gating,
    score_bias,
    *,
    topk: int = TOPK,
    renormalize: bool = RENORMALIZE,
    routed_scaling_factor: float = ROUTED_SCALING_FACTOR,
):
    """V4.1 Gate: top-k on score + bias, weights from the unbiased scores,
    renormalized, times route_scale. Rows whose logits are all non-finite
    (padding) get zero weight.

    Returns (weights f32 [T, k], ids i32 [T, k], valid bool [T, 1]).
    """
    import jax.numpy as jnp
    from jax import lax

    g = gating.astype(jnp.float32)
    scores = sqrtsoftplus(g)
    biased = scores if score_bias is None else scores + score_bias.astype(jnp.float32)
    _, ids = lax.top_k(biased, topk)
    weights = jnp.take_along_axis(scores, ids, axis=-1)
    if renormalize:
        weights = weights / (weights.sum(axis=-1, keepdims=True) + 1e-20)
    weights = weights * jnp.float32(routed_scaling_factor)
    valid = jnp.any(jnp.isfinite(g), axis=-1, keepdims=True)
    return jnp.where(valid, weights, 0.0), ids.astype(jnp.int32), valid


def _expert_sum(
    x32,
    weights,
    ids,
    w1_at,
    w2_at,
    num_experts,
    first_expert,
    *,
    inter,
    logical,
    limit,
    precision,
):
    """sum_e coef[t, e] * FFN_e(x[t]) over experts [first, first + n), dense."""
    import jax.numpy as jnp
    from jax import lax

    def body(e, acc):
        w1e = w1_at(e)  # [H, 2 * inter] f32
        w2e = w2_at(e)  # [inter, H] f32
        coef = jnp.sum(
            jnp.where(ids == first_expert + e, weights, 0.0), axis=-1, keepdims=True
        )
        h = jnp.dot(x32, w1e, precision=precision)
        act = clamped_swiglu(h[:, :logical], h[:, inter : inter + logical], limit)
        y = jnp.dot(act, w2e[:logical], precision=precision)
        return acc + coef * y

    return lax.fori_loop(0, num_experts, body, jnp.zeros_like(x32))


def reference_moe(
    x,
    gating,
    deq_w1,
    deq_w2,
    score_bias,
    *,
    topk: int = TOPK,
    renormalize: bool = RENORMALIZE,
    routed_scaling_factor: float = ROUTED_SCALING_FACTOR,
    swiglu_limit: float = SWIGLU_LIMIT,
    logical_intermediate_size: int | None = None,
    expert_id_shift: int = 0,
):
    """FP32 V4.1 routed-expert output on one device.

    x [T, H], gating [T, E] raw logits, deq_w1 [E, H, 2I] (gate | up),
    deq_w2 [E, I, H], score_bias [E] or None. Every expert is applied densely;
    nothing is quantized. ``expert_id_shift`` routes each selected id to
    ``(id + shift) % E`` (a wrong-expert control). Returns f32 [T, H].
    """
    import jax.numpy as jnp
    from jax import lax

    w1 = jnp.asarray(deq_w1, jnp.float32)
    w2 = jnp.asarray(deq_w2, jnp.float32)
    e_total = w1.shape[0]
    inter = w1.shape[2] // 2
    logical = inter if logical_intermediate_size is None else logical_intermediate_size
    weights, ids, valid = route(
        jnp.asarray(gating),
        None if score_bias is None else jnp.asarray(score_bias),
        topk=topk,
        renormalize=renormalize,
        routed_scaling_factor=routed_scaling_factor,
    )
    if expert_id_shift:
        ids = (ids + expert_id_shift) % e_total
    out = _expert_sum(
        jnp.asarray(x).astype(jnp.float32),
        weights,
        ids,
        lambda e: lax.dynamic_index_in_dim(w1, e, 0, keepdims=False),
        lambda e: lax.dynamic_index_in_dim(w2, e, 0, keepdims=False),
        e_total,
        0,
        inter=inter,
        logical=logical,
        limit=swiglu_limit,
        precision=lax.Precision.HIGHEST,
    )
    return jnp.where(valid, out, 0.0)


def make_reference_moe_sharded(
    mesh,
    *,
    topk: int = TOPK,
    renormalize: bool = RENORMALIZE,
    routed_scaling_factor: float = ROUTED_SCALING_FACTOR,
    swiglu_limit: float = SWIGLU_LIMIT,
    logical_intermediate_size: int | None = LOGICAL_INTER,
    rhs_qb: int = RHS_QB,
    expert_id_shift: int = 0,
):
    """:func:`reference_moe` spread over the EP mesh, on the KERNEL operands.

    Each shard dequantizes its own float4 experts one at a time (value * block
    scale, exact in f32) and applies them to the whole replicated batch; the
    shards' sums are psum'd. Returns jitted
    ``(x, gating, w1, w2, w1_scale, w2_scale, score_bias) -> f32 [T, H]``
    (replicated).
    """
    import jax
    import jax.numpy as jnp
    from jax import lax
    from jax.sharding import PartitionSpec as P

    (axis,) = mesh.axis_names
    ep = mesh.shape[axis]

    def local(x_g, gating_g, w1_l, w2_l, s1_l, s2_l, bias):
        g_local, hidden, two_inter = w1_l.shape
        inter = two_inter // 2
        e_total = g_local * ep
        logical = (
            inter if logical_intermediate_size is None else logical_intermediate_size
        )
        weights, ids, valid = route(
            gating_g,
            bias,
            topk=topk,
            renormalize=renormalize,
            routed_scaling_factor=routed_scaling_factor,
        )
        if expert_id_shift:
            ids = (ids + expert_id_shift) % e_total

        def grouped(s, k, n):
            nb = k // rhs_qb
            return s.reshape(g_local, -1, n)[:, :nb] if s.ndim == 2 else s

        s1 = grouped(s1_l, hidden, two_inter)
        s2 = grouped(s2_l, inter, hidden)

        def deq(w_l, s, e):
            w = lax.dynamic_index_in_dim(w_l, e, 0, keepdims=False)
            se = lax.dynamic_index_in_dim(s, e, 0, keepdims=False)
            k, n = w.shape
            w32 = w.astype(jnp.float32).reshape(k // rhs_qb, rhs_qb, n)
            return (w32 * se[:, None, :]).reshape(k, n)

        out = _expert_sum(
            x_g.astype(jnp.float32),
            weights,
            ids,
            lambda e: deq(w1_l, s1, e),
            lambda e: deq(w2_l, s2, e),
            g_local,
            lax.axis_index(axis) * g_local,
            inter=inter,
            logical=logical,
            limit=swiglu_limit,
            precision=lax.Precision.HIGHEST,
        )
        out = lax.psum(out, axis)
        return jnp.where(valid, out, 0.0)

    sm = jax.shard_map(
        local,
        mesh=mesh,
        in_specs=(P(), P(), P(axis), P(axis), P(axis), P(axis), P()),
        out_specs=P(),
        check_vma=False,
    )
    return jax.jit(sm)


# --------------------------------------------------------------------------- #
# Loaders that run without SGLang's runtime or torch_tpu.
# --------------------------------------------------------------------------- #
def import_fused_moe_v2(*, isolated: bool = False):
    """The ``fused_moe_v2`` package.

    ``sglang_tpu/moe/pallas/__init__.py`` imports ``fused_moe_op`` and through
    it ``torch_tpu``, which registers PJRT and takes the libtpu lock. A
    standalone JAX process must not import it, so ``isolated=True`` registers
    a bare ``sglang_tpu.moe.pallas`` package (no ``__init__`` run) when that
    package is not imported yet. Do not use ``isolated=True`` inside an SGLang
    server process: later ``from sglang_tpu.moe.pallas import fused_moe_call``
    would then fail.
    """
    name = "sglang_tpu.moe.pallas"
    if isolated and name not in sys.modules:
        import sglang_tpu.moe as moe_pkg

        spec = importlib.machinery.ModuleSpec(name, None, is_package=True)
        spec.submodule_search_locations = [
            os.path.join(os.path.dirname(moe_pkg.__file__), "pallas")
        ]
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        moe_pkg.pallas = module
    return importlib.import_module(name + ".fused_moe_v2")


_TREE = Path(__file__).resolve().parents[5]
_GMM_PREP_FILES = {
    "fp8_moe": _TREE / "sglang_tpu" / "fp8_moe.py",
    "dsv4_mxfp4_moe": _TREE / "sglang_tpu" / "dsv4_mxfp4_moe.py",
    "mxfp4_tpu_util": _TREE / "python/sglang/srt/layers/quantization/mxfp4_tpu_util.py",
    "mxfp4_tensor": _TREE / "python/sglang/srt/layers/quantization/mxfp4_tensor.py",
    "layers_utils_common": _TREE / "python/sglang/srt/layers/utils/common.py",
}


@contextlib.contextmanager
def _modules_installed(modules: dict):
    """Temporarily put ``modules`` in sys.modules; restore what was there."""
    saved = {name: sys.modules.get(name) for name in modules}
    sys.modules.update(modules)
    try:
        yield
    finally:
        for name, old in saved.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old


def _load_file(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _stub(name: str, **attrs):
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    return module


class GmmPrep(NamedTuple):
    prepare: object  # callable(layer) -> None, the gmm preparation under stubs
    module: types.ModuleType  # the loaded dsv4_mxfp4_moe module


@functools.lru_cache(maxsize=1)
def load_gmm_mxfp4_prep() -> GmmPrep:
    """The real gmm preparation (``prepare_tpu_dsv4_mxfp4_moe_weights``) without
    SGLang's runtime.

    ``dsv4_mxfp4_moe.py`` imports ``fp8_moe.py``, whose module-level imports
    pull in the SGLang runtime (and on a TPU host possibly torch_tpu). The
    function under test needs none of it: it rebinds parameters with
    ``copy_or_rebind_param`` and lazily imports two padding helpers. So
    dsv4_mxfp4_moe.py, fp8_moe.py, mxfp4_tpu_util.py, mxfp4_tensor.py and
    layers/utils/common.py are executed from this tree's sources, and only
    these names are stubbed while they load/run: StandardCombineInput,
    block_quant_dequant, current_platform, get_exec/get_parallel (tp_size=1),
    print_warning_once, sglang_tpu.envs. None of the stubs is on the MXFP4
    path.
    """
    common = _load_file(
        "_dsv41_gmm_prep_layers_utils_common", _GMM_PREP_FILES["layers_utils_common"]
    )
    parallel = types.SimpleNamespace(tp_size=1, pp_size=1, moe_ep_size=1, moe_ep_rank=0)

    def _unavailable(*_args, **_kwargs):
        raise NotImplementedError("stubbed: not on the DSV4 MXFP4 prep path")

    stubs = {
        "sglang.srt.layers.moe.token_dispatcher": _stub(
            "sglang.srt.layers.moe.token_dispatcher", StandardCombineInput=object
        ),
        "sglang.srt.layers.quantization.fp8_utils": _stub(
            "sglang.srt.layers.quantization.fp8_utils",
            block_quant_dequant=_unavailable,
        ),
        "sglang.srt.layers.utils": _stub(
            "sglang.srt.layers.utils",
            copy_or_rebind_param=common.copy_or_rebind_param,
        ),
        "sglang.srt.platforms": _stub(
            "sglang.srt.platforms", current_platform=types.SimpleNamespace()
        ),
        "sglang.srt.runtime_context": _stub(
            "sglang.srt.runtime_context",
            get_parallel=lambda: parallel,
            get_exec=_unavailable,
        ),
        "sglang.srt.utils": _stub("sglang.srt.utils", print_warning_once=print),
        "sglang_tpu.envs": _stub("sglang_tpu.envs", envs=types.SimpleNamespace()),
    }
    with _modules_installed(stubs):
        tensor_mod = _load_file(
            "sglang.srt.layers.quantization.mxfp4_tensor",
            _GMM_PREP_FILES["mxfp4_tensor"],
        )
        stubs["sglang.srt.layers.quantization.mxfp4_tensor"] = tensor_mod
    with _modules_installed(stubs):
        util_mod = _load_file(
            "sglang.srt.layers.quantization.mxfp4_tpu_util",
            _GMM_PREP_FILES["mxfp4_tpu_util"],
        )
        stubs["sglang.srt.layers.quantization.mxfp4_tpu_util"] = util_mod
        stubs["sglang_tpu.fp8_moe"] = _load_file(
            "_dsv41_gmm_prep_fp8_moe", _GMM_PREP_FILES["fp8_moe"]
        )
    with _modules_installed(stubs):
        module = _load_file(
            "_dsv41_gmm_prep_dsv4_mxfp4_moe", _GMM_PREP_FILES["dsv4_mxfp4_moe"]
        )
    prep_fn = module.prepare_tpu_dsv4_mxfp4_moe_weights

    def prepare(layer) -> None:
        # The lazy import inside the function resolves while these are in place.
        with _modules_installed(stubs):
            prep_fn(layer)

    return GmmPrep(prepare=prepare, module=module)


def run_gmm_prep(w13_packed, w13_scale, w2_packed, w2_scale) -> dict:
    """Run the gmm preparation on checkpoint-format CPU tensors of one expert set.

    Inputs are what SGLang's loader leaves in the FusedMoE parameters
    (fp8.py create_weights, ``is_fp4_expert``): ``w13_weight`` int8
    ``[E, 2I, H/2]`` (w1 rows then w3 rows), ``w13_weight_scale_inv`` f32
    ``[E, 2I, H/32]`` (= 2^(e8m0 - 127)), ``w2_weight`` int8 ``[E, H, I/2]``,
    ``w2_weight_scale_inv`` f32 ``[E, H, I/32]``. Returns the gmm-layout
    ``w13_weight``, ``w13_weight_scale``, ``w2_weight``, ``w2_weight_scale``.
    The preparation frees the buffers it replaces, so resizable inputs are
    consumed (their storage is emptied); pass clones to keep them.
    """
    import torch
    from torch.nn import Module, Parameter

    def owned(t):
        # The preparation frees the replaced buffers, which needs a resizable
        # storage; a tensor made by torch.from_numpy is not one.
        return t if t.untyped_storage().resizable() else t.clone()

    layer = Module()
    layer.w13_weight = Parameter(
        owned(w13_packed).view(torch.int8), requires_grad=False
    )
    layer.w2_weight = Parameter(owned(w2_packed).view(torch.int8), requires_grad=False)
    layer.w13_weight_scale_inv = Parameter(owned(w13_scale), requires_grad=False)
    layer.w2_weight_scale_inv = Parameter(owned(w2_scale), requires_grad=False)
    load_gmm_mxfp4_prep().prepare(layer)
    return {
        name: getattr(layer, name).data
        for name in ("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale")
    }


def e8m0_to_float32(exponents):
    """E8M0 bytes -> f32 the way SGLang's loader fills an f32 scale parameter
    (``param.copy_(float8_e8m0fnu tensor)``): 2^(e - 127), 0xFF -> NaN."""
    import torch

    if isinstance(exponents, np.ndarray):
        exponents = np.ascontiguousarray(exponents)
        if not exponents.flags.writeable:
            exponents = exponents.copy()
        exponents = torch.from_numpy(exponents)
    return exponents.view(torch.uint8).view(torch.float8_e8m0fnu).to(torch.float32)
