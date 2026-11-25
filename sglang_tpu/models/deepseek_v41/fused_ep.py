# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1 Flash routed experts on ``fused_ep_moe_v2``: settings and layout.

By default V4.1's routed experts run through gmm_v2: at load, the gmm
preparation (``sglang_tpu/dsv4_mxfp4_moe.py:prepare_tpu_dsv4_mxfp4_moe_weights``)
dequantizes the checkpoint's block-32 MXFP4 experts, pads each gate/up half
2304 -> 2560 and requantizes to E2M1 with block-512 FP32 scales (the gmm
layout below). The fused EP kernel reads those prepared values without
requantizing:

* :func:`make_fused_ep_moe` -- ``(x, gating, w1, w2, s1, s2, score_bias[,
  rank]) -> out`` calling ``fused_ep_moe_v2`` with the V4.1 settings below.
* :func:`gmm_scale_to_kernel` -- host relayout of the gmm layout's scales
  into the kernel's operands. Every value is bit-identical before and after.

Layout facts:

* gmm layout, per rank with ``E`` local experts, ``H = 5120``,
  ``Ip = 2560``: ``w13_weight`` uint8 ``[E, H/2, 2*Ip]``,
  ``w13_weight_scale`` f32 ``[E, H/512, 1, 2*Ip]``, ``w2_weight`` uint8
  ``[E, Ip/2, H]``, ``w2_weight_scale`` f32 ``[E, Ip/512, 1, H]``. The packing
  axis is K (axis 1): byte ``[e, j, n]`` holds K row ``2j`` in its low nibble
  and row ``2j+1`` in its high nibble
  (``dsv4_mxfp4_moe._quantize_tpu_fp4_block512_kmajor``). Codes are E2M1 with
  bit 3 the sign, the same table as ``ml_dtypes.float4_e2m1fn``.
* Gate/up order along w13's N axis: ``[gate(2304) | 0(256) | up(2304) |
  0(256)]`` (checkpoint ``w1`` -> gate_proj, ``w3`` -> up_proj; each half
  padded by ``mxfp4_tpu_util._pad_gate_up_halves_axis1``). w2's K axis is
  zero-padded at the end. The kernel reads gate as columns ``[0, inter)`` and
  up as ``[inter, 2*inter)``, so no column permutation is needed; vLLM
  epic/dsv41 builds the same order.
* Kernel FP4 operands: ``w1 [E, H, 2*Ip]``, ``w2 [E, Ip, H]`` as
  ``float4_e2m1fn``, one f32 scale per (512-row K block, output column),
  grouped ``[E, K/512, N]`` or row-padded flat ``[E * fp4_scale_rows(K, 512,
  padded=True), N]`` (K/512 rounded up to 8 rows, padding rows zero). vLLM
  serves the flat form. ``logical_intermediate_size=2304`` makes the kernel
  skip the padded gate/up columns and w2 rows.
* K32Z (``expert_format="k32z"``, ``sglang_tpu/dsv4_mxfp4_k32z.py``): the
  checkpoint's block-32 experts losslessly, no padding (``inter = 2304``):
  ``w1 [E, H, 2*inter]``, ``w2 [E, inter, H]`` float4 codes and uint8 shift
  tables flat ``[E * rows, cols]`` (``fused_moe_v2.host.k32z_table_geometry``).
  The kernel runs 256-row tiles (:data:`K32Z_CAPACITY`).
* Bridge: a uint8 gmm-layout byte becomes two ``float4_e2m1fn`` elements on
  K. On the host this is a nibble split; in JAX,
  ``lax.bitcast_convert_type(u8, float4_e2m1fn)`` appends a minor axis with
  the low nibble first, moved next to K. Torch's ``float4_e2m1fn_x2`` packs N
  instead; ``fused_ep_op.kpacked_to_npacked_fp4_bytes`` gives that byte form,
  which is what ``fused_ep_op`` serves.

Nothing here imports torch_tpu; the kernel package is imported lazily. The
host relayout of the weights, the FP32 reference and the CPU-test loaders live
in ``test/registered/unit/hardware_backend/tpu/dsv41_fused_ep_reference.py``.
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------- #
# V4.1 Flash routed-expert geometry and routing policy (official config.json:
# moe_inter_dim, n_activated_experts, score_func, route_scale, swiglu_limit;
# ModelArgs.norm_topk_prob=True, gate_temp=1.0).
# --------------------------------------------------------------------------- #
LOGICAL_INTER = 2304
TOPK = 6
RENORMALIZE = True
ROUTED_SCALING_FACTOR = 1.5
SWIGLU_LIMIT = 10.0
SCORING_FN = "sqrtsoftplus"
# Kernel call settings (as vLLM epic/dsv41 serves V4.1).
RHS_QB = 512
CAPACITY = 128
ACT_FN = "silu_and_mul_with_clamp"
DISPATCH = "all_gather"
# fused_moe_v2's point-to-point dispatch (push_fused_moe), and "auto", which
# picks one of the two per compiled shape (resolve_dispatch).
DISPATCH_PUSH = "push"
DISPATCH_AUTO = "auto"
DISPATCHES = (DISPATCH, DISPATCH_PUSH, DISPATCH_AUTO)
# auto's default bound in rows per EP rank. In the layer-3 benchmark at EP8
# (v7x, K512), push beat all-gather at 8-1024 rows per rank and lost at 2048.
PUSH_MAX_ROWS = 1024
# Expert formats (sglang_tpu/dsv4_mxfp4_k32z.py's names): the block-512 gmm
# preparation, or K32Z, whose kernel runs 256-row tiles.
EXPERT_FORMAT_K512 = "k512"
EXPERT_FORMAT_K32Z = "k32z"
K32Z_CAPACITY = 256

# host.U32_SUBLANE_TILE; repeated so the relayout needs no JAX import.
_U32_SUBLANE_TILE = 8


def fp4_scale_rows(k: int, rhs_qb: int = RHS_QB, *, padded: bool = False) -> int:
    """Mirror of ``fused_moe_v2.host.fp4_scale_rows`` (checked equal in tests)."""
    blocks = k // rhs_qb
    if not padded:
        return blocks
    return -(-blocks // _U32_SUBLANE_TILE) * _U32_SUBLANE_TILE


# --------------------------------------------------------------------------- #
# gmm layout -> kernel layout.
# --------------------------------------------------------------------------- #
def _as_float32_numpy(t) -> np.ndarray:
    if hasattr(t, "detach"):
        import torch

        t = t.detach().cpu()
        if t.dtype != torch.float32:
            raise TypeError(f"expected f32 block scales, got {t.dtype}")
        return t.contiguous().numpy()
    a = np.asarray(t)
    if a.dtype != np.float32:
        raise TypeError(f"expected f32 block scales, got {a.dtype}")
    return a


def gmm_scale_to_kernel(
    scale, *, k: int, rhs_qb: int = RHS_QB, layout: str = "flat"
) -> np.ndarray:
    """gmm-layout scale ``[E, K/qb, 1, N]`` f32 -> grouped ``[E, K/qb, N]`` or the
    row-padded flat ``[E * fp4_scale_rows(K, qb, padded=True), N]``."""
    s = _as_float32_numpy(scale)
    if s.ndim == 4:
        if s.shape[2] != 1:
            raise ValueError(f"gmm-layout scale must be [E, K/qb, 1, N], got {s.shape}")
        s = s[:, :, 0, :]
    blocks = fp4_scale_rows(k, rhs_qb)
    if s.ndim != 3 or s.shape[1] != blocks:
        raise ValueError(f"scale {s.shape} is not [E, {blocks}, N] for K={k}")
    if layout == "grouped":
        return np.ascontiguousarray(s)
    if layout != "flat":
        raise ValueError(f"layout must be 'flat' or 'grouped', got {layout!r}")
    e, _, n = s.shape
    rows = fp4_scale_rows(k, rhs_qb, padded=True)
    flat = np.zeros((e, rows, n), dtype=np.float32)
    flat[:, :blocks] = s
    return flat.reshape(e * rows, n)


# --------------------------------------------------------------------------- #
# The kernel call.
# --------------------------------------------------------------------------- #
def tuned_push_options(g_local: int) -> dict:
    """The tuned push options (the combined options of
    test_fused_ep_moe_push.py), with dispatch_ahead = the local experts per
    rank. K32Z builds also get their per-bucket schedule from the push layer."""
    return dict(
        sc_prepush=2,
        tc_exchange=True,
        onehot_slots=True,
        tail_barrier=False,
        hbm_staging=True,
        dispatch_ahead=g_local,
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
    )


def resolve_dispatch(dispatch: str, *, rows_per_rank: int, push_max_rows: int) -> str:
    """The dispatch one compiled shape runs: "auto" pushes while the bucket has
    at most ``push_max_rows`` rows per EP rank and all-gathers above."""
    if dispatch != DISPATCH_AUTO:
        return dispatch
    return DISPATCH_PUSH if rows_per_rank <= push_max_rows else DISPATCH


def make_fused_ep_moe(
    mesh,
    *,
    ep: int | None = None,
    topk: int = TOPK,
    renormalize: bool = RENORMALIZE,
    routed_scaling_factor: float = ROUTED_SCALING_FACTOR,
    logical_intermediate_size: int | None = LOGICAL_INTER,
    capacity: int | None = None,
    rhs_qb: int = RHS_QB,
    act_fn: str = ACT_FN,
    scoring_fn: str = SCORING_FN,
    sharded_plan: bool = True,
    kernel_fn=None,
    jit: bool = True,
    expert_format: str = EXPERT_FORMAT_K512,
    dispatch: str = DISPATCH,
    push_max_rows: int = PUSH_MAX_ROWS,
):
    """Return ``(x, gating, w1, w2, w1_scale, w2_scale, score_bias[, rank]) -> out``.

    x bf16 [T, H] and gating f32 [T, E] raw router logits, both sharded on the
    1-D EP mesh with T % ep == 0; padding rows carry all -inf logits and come
    back exactly zero. w1/w2 float4 and their scales sharded on the experts
    (the kernel FP4 operands in the module docstring). score_bias f32 [E]
    replicated (the gate's e_score_correction_bias; selection only). Output
    [T, H] in x's dtype, sharded like x.

    Settings mirror vLLM epic/dsv41's V4.1 layer except
    ``preprocess_topology`` (None here; vLLM passes
    ``prepare_moe_preprocessing(mesh)``). Tokens are exchanged with
    ``dispatch`` (below). ``rank`` is optional: left out (None), the kernel layer
    builds the rank operand itself under a plain ``jax.jit``; a torch_tpu
    export must pass it as an int32 ``[ep, 1]`` operand sharded on the mesh
    axis (see fused_ep_op.py). ``kernel_fn`` replaces ``fused_ep_moe_v2``
    (tests).

    ``expert_format`` "k512" takes the FP4 block-512 operands at ``capacity``
    (default :data:`CAPACITY`); "k32z" the K32Z operands at ``capacity``
    (default :data:`K32Z_CAPACITY`), where ``logical_intermediate_size`` can
    only be the operands' own (they carry no padding).

    ``dispatch`` exchanges the tokens with :data:`DISPATCH` (all-gather, the
    default), "push" or "auto" (:func:`resolve_dispatch` on each traced
    shape's rows per EP rank against ``push_max_rows``). A push call uses
    :func:`tuned_push_options` for the operands' local experts. It runs the
    kernel's fused stage without the push layer's optional fused shared
    expert, so the output stays the routed experts alone, as on all-gather.
    """
    import jax
    import jax.numpy as jnp

    from sglang_tpu.moe.pallas import fused_moe_v2 as package

    if dispatch not in DISPATCHES:
        raise ValueError(f"dispatch must be one of {DISPATCHES}, got {dispatch!r}")
    if dispatch == DISPATCH_AUTO and push_max_rows < 1:
        raise ValueError(f"push_max_rows must be >= 1, got {push_max_rows}")
    (axis,) = mesh.axis_names
    mesh_ep = mesh.shape[axis]
    if ep is None:
        ep = mesh_ep
    if ep != mesh_ep:
        raise ValueError(f"ep={ep} but the mesh axis {axis!r} has {mesh_ep} devices")
    if kernel_fn is None:
        kernel_fn = package.fused_ep_moe_v2
    if expert_format == EXPERT_FORMAT_K512:
        weight_format = package.WeightFormat.FP4
        capacity = CAPACITY if capacity is None else capacity
    elif expert_format == EXPERT_FORMAT_K32Z:
        weight_format = package.WeightFormat.K32Z
        capacity = K32Z_CAPACITY if capacity is None else capacity
    else:
        raise ValueError(
            f"expert_format must be {EXPERT_FORMAT_K512!r} or "
            f"{EXPERT_FORMAT_K32Z!r}, got {expert_format!r}"
        )
    mesh_ep_ranks = tuple(range(ep))

    def moe(x, gating, w1, w2, w1_scale, w2_scale, score_bias, rank=None):
        # Static per traced shape: x holds every rank's rows.
        chosen = resolve_dispatch(
            dispatch, rows_per_rank=x.shape[0] // ep, push_max_rows=push_max_rows
        )
        push = {}
        if chosen == DISPATCH_PUSH:
            push["push_options"] = tuned_push_options(w1.shape[0] // ep)
        return kernel_fn(
            x,
            w1,
            w2,
            w1_scale,
            w2_scale,
            gating.astype(jnp.float32),
            topk=topk,
            renormalize=renormalize,
            mesh=mesh,
            capacity=capacity,
            weight_format=weight_format,
            rhs_qb=rhs_qb,
            act_fn=act_fn,
            rank=rank,
            mesh_ep_ranks=mesh_ep_ranks,
            sharded_plan=sharded_plan,
            scoring_fn=scoring_fn,
            score_bias=score_bias.astype(jnp.float32),
            routed_scaling_factor=routed_scaling_factor,
            logical_intermediate_size=logical_intermediate_size,
            preprocess_topology=None,
            dispatch=chosen,
            **push,
        )

    return jax.jit(moe) if jit else moe
