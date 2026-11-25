# SPDX-License-Identifier: Apache-2.0
# NOTE: do *not* add `from __future__ import annotations` here: the op body
# built in `_make_jax_fn` is registered through
# `sglang_tpu.pallas_jax_op.cpu_export_jax_op`, which inspects
# `param.annotation is jax.Array`.
"""DeepSeek-V4.1 Flash routed experts on ``fused_ep_moe_v2`` inside SGLang.

Opt-in with ``SGLANG_TPU_DSV41_FUSED_EP=1`` (``sglang_tpu.envs``). Off, nothing in
this module runs: the plugin hook that calls it
(``sglang_tpu/models/deepseek_v41/hooks.py``) is not registered.

With the switch off, every V4.1 MoE layer under attention DP8 + EP8 gathers
all ranks' hidden states (``dp_gather_replicate``), runs gate, top-k and
gmm_v2 on the gathered rows, all-reduces the expert sums and scatters each
rank its rows back (``NonCudaDecoderMixin._run_moe_ffn_dp_sync`` in
``sglang.srt.models.deepseek_v4_platform_stage``). With it
on, preparation sets ``mlp.dp_local_forward`` on every eligible layer;
``_run_moe_ffn_dp_sync`` calls it with this rank's own rows, and it:

1. computes the router logits of this rank's own rows with the layer's own
   ``MoEGate`` (same matmul and dtype as today), upcast to FP32;
2. sets the logits of padding rows to -inf (rows at or past
   ``forward_batch.num_token_non_padded``, this rank's real row count), so the
   kernel routes them nowhere and returns exact zeros for them;
3. calls one custom op that runs ``fused_ep_moe_v2`` over the EP mesh: the
   kernel exchanges FP8 tokens itself, applies sqrt(softplus) scoring with the
   correction bias, renormalized top-6, the routed scale 1.5 (once, inside the
   kernel) and the clamped SwiGLU, and returns this rank's rows;
4. adds the replicated (TP1, ``SGLANG_SHARED_EXPERT_TP1=1``) shared expert
   computed on the same local rows.

No DP gather, scatter or post-experts all-reduce runs for such a layer.

Every rank calls the op with the same row count. Attention DP on TPU only runs
``DpPaddingMode.MAX_LEN``: ``prepare_mlp_sync_batch`` sets every rank's count
to the cross-rank maximum, the compile shim pads every rank to the same token
bucket (``_dp_attn_compute_padding`` / ``_dp_attn_apply_bucket_metadata`` set
``global_num_tokens_cpu = [bucket] * dp``), and idle ranks are materialized as
dummy decode rows (``_match_busiest_dp_decode_rows``) or a dummy extend
(``materialize_dp_idle_extend``) before that padding.
:func:`check_equal_dp_rows` asserts it at trace time.

Weights. The gmm preparation
(``sglang_tpu/dsv4_mxfp4_moe.py:prepare_tpu_dsv4_mxfp4_moe_weights``) leaves
gmm_v2 operands, the gmm layout: ``w13_weight`` uint8 ``[E, H/2, 2*Ip]``
packed along K (low nibble = even K row), ``w13_weight_scale`` f32
``[E, H/512, 1, 2*Ip]``, ``w2_weight`` uint8 ``[E, Ip/2, H]``,
``w2_weight_scale`` f32 ``[E, Ip/512, 1, H]`` (H=5120, Ip=2560, E=48 per
rank: ~0.94 GB per layer per rank). Both layouts cannot be kept, so
:func:`prepare_moe_layer_for_fused_ep` replaces them, one tensor at a time,
by the kernel's operands:

* weights as ``torch.float4_e2m1fn_x2`` ``[E, K, N/2]``: the same 4 bits per
  value, packed along N with the low nibble = even column. TorchTPU's JAX
  bridge presents a ``float4_e2m1fn_x2 [..., N/2]`` tensor to JAX as
  ``float4_e2m1fn [..., N]`` (``pallas_jax_op._jax_placeholder_with_sharding``
  doubles the last axis), which is the dtype and ``[E, K, N]`` shape the kernel
  takes, so nothing is relaid out per call. Keeping the gmm layout's K-packed
  uint8 would need a nibble transpose of the whole weight inside every call.
  It is the same storage the NVFP4 fused-RS path serves
  (``fp4_moe._transpose_packed_fp4`` + ``_materialize_packed_fp4``), and costs
  the same bytes as the gmm layout's uint8. The relayout is bit-exact (no
  requantization).
* scales as f32 row-padded flat ``[E * fp4_scale_rows(K, 512, padded=True),
  N]`` (``[E*16, 2*Ip]`` and ``[E*8, H]``): the form vLLM serves; a 2-D
  buffer has one device layout, so the kernel never relays it out. Built on
  the host and moved to the device (a host-moved buffer is not
  re-materialized before every compiled call, see ``fp8_moe._own_storage``).

The weight relayout runs on the parameter's device, a few experts at a time
into a new buffer; the old parameter is freed before the float4 copy is made,
so the load holds at most one extra copy of the tensor being converted (0.63 GB
for w13 at V4.1 Flash size) plus one chunk's temporaries. Steady state: the
same weight bytes as the gmm layout, plus ~9 MB per layer for the padded scale
rows.

With ``SGLANG_TPU_DSV41_EXPERT_FORMAT=k32z`` the gmm preparation is deferred to
this step (``sglang_tpu/dsv4_mxfp4_k32z.py``): eligible layers take the K32Z
operands, converted losslessly from the checkpoint experts, and every other
deferred layer gets the gmm layout here. The K32Z codes are relaid out to
``float4_e2m1fn_x2 [E, K, N/2]`` like the gmm layout's (no padding: N = 2 *
2304 for w1), and the uint8 shift tables go flat, ``[E * rows, cols]``, built
on the host and moved to the device like the scales above. Their op is built
with ``weight_format=K32Z`` at 256-row tiles and 4-bit gate/up shift fields
(``k32z.SERVING_GATE_UP_BITS``); both are in its name.

``SGLANG_TPU_DSV41_FUSED_EP_DISPATCH`` sets how the op exchanges tokens:
``all_gather`` (the default), ``push`` (fused_moe_v2's push dispatch with
``fused_ep.tuned_push_options``) or ``auto``, decided statically per compiled
shape: push while the token bucket has at most ``fused_ep.PUSH_MAX_ROWS`` rows
per EP rank, all-gather above.
Either expert format takes either dispatch. Push and auto are in the op's key
and name (an all-gather op's are the format's plain ones), so their compiled
programs never mix. The op calls the kernel's fused stage only, and the shared
expert stays outside the op as above.
"""

import contextlib
import functools
import logging
from threading import Lock
from typing import Callable, Optional

import torch

from sglang_tpu import dsv4_mxfp4_k32z as k32z
from sglang_tpu.models.deepseek_v41 import fused_ep as fe

logger = logging.getLogger(__name__)

# Set on a DeepseekV2MoE once its experts carry the fused-EP operands, together
# with its ``dp_local_forward`` (the seam in _run_moe_ffn_dp_sync).
READY_ATTR = "_tpu_dsv41_fused_ep_ready"
_KEY_ATTR = "_tpu_dsv41_fused_ep_key"
_RANK_ATTR = "_tpu_dsv41_fused_ep_ep_rank"

# Buffers registered on the FusedMoE (``moe.experts``).
W1, W2, W1_SCALE, W2_SCALE, RANK = (
    "_fused_ep_w1",
    "_fused_ep_w2",
    "_fused_ep_w1_scale",
    "_fused_ep_w2_scale",
    "_fused_ep_rank",
)

_DEFAULT_MESH_AXIS = "ep"
_EXPERTS_PER_CHUNK = 8

_OPS: dict[tuple, Callable] = {}
_OPS_LOCK = Lock()
# Test hook: replaces fused_ep_moe_v2 in ops built while set (kernel_override).
_KERNEL_FN: Optional[Callable] = None
# sglang_tpu.rank_local.traced_rank, resolved at preparation (needs torch_tpu).
_TRACED_RANK: Optional[Callable] = None


def is_dsv41_fused_ep_enabled() -> bool:
    from sglang_tpu.envs import envs as tpu_envs

    return bool(tpu_envs.SGLANG_TPU_DSV41_FUSED_EP.get())


def check_dsv41_fused_ep_env() -> None:
    """Refuse the switches without their prerequisites at startup; the per-layer
    check would refuse every V4.1 MoE layer only once the weights have loaded."""
    from sglang.srt.environ import envs
    from sglang_tpu.dsv4_mxfp4_k32z import check_expert_format_env

    if is_dsv41_fused_ep_enabled() and not envs.SGLANG_SHARED_EXPERT_TP1.get():
        raise ValueError(
            "SGLANG_TPU_DSV41_FUSED_EP=1 needs SGLANG_SHARED_EXPERT_TP1=1 (the "
            "fused EP layers add a replicated shared expert on each rank's own "
            "rows). Set SGLANG_SHARED_EXPERT_TP1=1."
        )
    check_expert_format_env()
    dispatch = selected_dispatch()
    if dispatch != fe.DISPATCH and not is_dsv41_fused_ep_enabled():
        raise ValueError(
            f"SGLANG_TPU_DSV41_FUSED_EP_DISPATCH={dispatch} needs "
            "SGLANG_TPU_DSV41_FUSED_EP=1 (gmm_v2 has no dispatch to choose)."
        )


def selected_dispatch() -> str:
    """``SGLANG_TPU_DSV41_FUSED_EP_DISPATCH``, validated. Every rank reads the
    same environment, so every rank builds the same op (a collective)."""
    from sglang_tpu.envs import envs as tpu_envs

    dispatch = str(tpu_envs.SGLANG_TPU_DSV41_FUSED_EP_DISPATCH.get()).strip().lower()
    if dispatch not in fe.DISPATCHES:
        raise ValueError(
            f"SGLANG_TPU_DSV41_FUSED_EP_DISPATCH must be one of {fe.DISPATCHES}, "
            f"got {dispatch!r}"
        )
    return dispatch


# --------------------------------------------------------------------------- #
# The custom op.
# --------------------------------------------------------------------------- #
@functools.cache
def _get_mesh(world_size: int, axis_name: str = _DEFAULT_MESH_AXIS):
    from torch_tpu._internal.pallas import get_pallas_mesh

    return get_pallas_mesh((axis_name,), (world_size,))


def _export_op(name, fn, *, mesh, input_partition_specs, output_partition_specs):
    from sglang_tpu.pallas_jax_op import cpu_export_jax_op

    return cpu_export_jax_op(
        name,
        fn,
        mesh=mesh,
        input_partition_specs=input_partition_specs,
        output_partition_specs=output_partition_specs,
    )


def _tag(value: float) -> str:
    return f"{value:.6g}".replace(".", "_").replace("-", "m").replace("+", "p")


def _dispatch_tag(dispatch: str, push_max_rows: int) -> str:
    # All-gather adds no tag: its op names are the plain format names.
    if dispatch == fe.DISPATCH:
        return ""
    if dispatch == fe.DISPATCH_PUSH:
        return "_push"
    if dispatch == fe.DISPATCH_AUTO:
        return f"_auto{int(push_max_rows)}"
    raise ValueError(f"dispatch must be one of {fe.DISPATCHES}, got {dispatch!r}")


def _op_name(
    ep: int,
    topk: int,
    renormalize: bool,
    routed_scaling_factor: float,
    logical_intermediate_size: int,
    axis_name: str = _DEFAULT_MESH_AXIS,
    expert_format: str = fe.EXPERT_FORMAT_K512,
    gate_up_bits: int = k32z.GATE_UP_SHIFT_BITS,
    dispatch: str = fe.DISPATCH,
    push_max_rows: int = fe.PUSH_MAX_ROWS,
) -> str:
    """One torch custom-op name per closure configuration (the kernel's other
    settings are module constants of fused_ep and part of the name too)."""
    head = (
        f"sglang_tpu_pallas::dsv41_fused_ep_ep{ep}_k{topk}_r{int(renormalize)}"
        f"_s{_tag(routed_scaling_factor)}_i{logical_intermediate_size}"
    )
    if expert_format == fe.EXPERT_FORMAT_K32Z:
        head = f"{head}_k32z_g{gate_up_bits}_c{fe.K32Z_CAPACITY}"
    else:
        head = f"{head}_c{fe.CAPACITY}_q{fe.RHS_QB}"
    return f"{head}{_dispatch_tag(dispatch, push_max_rows)}_{axis_name}"


def _op_key(
    ep: int,
    topk: int,
    renormalize: bool,
    routed_scaling_factor: float,
    logical_intermediate_size: int,
    axis_name: str = _DEFAULT_MESH_AXIS,
    expert_format: str = fe.EXPERT_FORMAT_K512,
    gate_up_bits: int = k32z.GATE_UP_SHIFT_BITS,
    dispatch: str = fe.DISPATCH,
    push_max_rows: int = fe.PUSH_MAX_ROWS,
) -> tuple:
    """``_build_op``'s arguments, normalized: one key per op (and op name)."""
    if dispatch not in fe.DISPATCHES:
        raise ValueError(f"dispatch must be one of {fe.DISPATCHES}, got {dispatch!r}")
    key = (
        int(ep),
        int(topk),
        bool(renormalize),
        float(routed_scaling_factor),
        int(logical_intermediate_size),
        str(axis_name),
    )
    # A K512 all-gather key is these six fields and a K32Z one adds its format
    # and width; a push or auto key adds both (width 0 for K512) and its
    # dispatch with auto's bound (0 for push, which has none).
    if expert_format == fe.EXPERT_FORMAT_K512 and dispatch == fe.DISPATCH:
        return key
    if expert_format == fe.EXPERT_FORMAT_K512:
        gate_up_bits = 0
    key += (str(expert_format), int(gate_up_bits))
    if dispatch == fe.DISPATCH:
        return key
    bound = int(push_max_rows) if dispatch == fe.DISPATCH_AUTO else 0
    return key + (str(dispatch), bound)


def _make_jax_fn(
    mesh,
    ep: int,
    topk: int,
    renormalize: bool,
    routed_scaling_factor: float,
    logical_intermediate_size: int,
    axis_name: str,
    expert_format: str = fe.EXPERT_FORMAT_K512,
    dispatch: str = fe.DISPATCH,
    push_max_rows: int = fe.PUSH_MAX_ROWS,
):
    """The exported JAX body, its input and its output partition specs.

    Arguments arrive as global arrays: rows of x/gating, experts of the
    weights and scales, and the rank vector are sharded on the EP axis (each
    rank's torch tensor is its shard); the correction bias is replicated, as
    it is indexed by global expert id.
    """
    import jax
    from jax.sharding import PartitionSpec as P

    moe = fe.make_fused_ep_moe(
        mesh,
        ep=ep,
        topk=topk,
        renormalize=renormalize,
        routed_scaling_factor=routed_scaling_factor,
        logical_intermediate_size=logical_intermediate_size,
        kernel_fn=_KERNEL_FN,
        jit=False,
        expert_format=expert_format,
        dispatch=dispatch,
        push_max_rows=push_max_rows,
    )

    def dsv41_fused_ep(
        x: jax.Array,
        gating: jax.Array,
        w1: jax.Array,
        w2: jax.Array,
        w1_scale: jax.Array,
        w2_scale: jax.Array,
        score_bias: jax.Array,
        rank: jax.Array,
    ) -> jax.Array:
        return moe(x, gating, w1, w2, w1_scale, w2_scale, score_bias, rank)

    rows = P(axis_name)
    in_specs = (rows,) * 6 + (P(), rows)
    return dsv41_fused_ep, in_specs, (rows,)


def _fake_like_x(x, *_args):
    # The output is sharded like x, so the local result has x's shape and
    # dtype. Registered over cpu_export_jax_op's export-based fake so that a
    # Dynamo trace does not lower the Pallas kernel once per MoE layer.
    return torch.empty_like(x)


def _build_op(
    ep: int,
    topk: int,
    renormalize: bool,
    routed_scaling_factor: float,
    logical_intermediate_size: int,
    axis_name: str = _DEFAULT_MESH_AXIS,
    expert_format: str = fe.EXPERT_FORMAT_K512,
    gate_up_bits: int = k32z.GATE_UP_SHIFT_BITS,
    dispatch: str = fe.DISPATCH,
    push_max_rows: int = fe.PUSH_MAX_ROWS,
) -> Callable:
    mesh = _get_mesh(ep, axis_name)
    fn, in_specs, out_specs = _make_jax_fn(
        mesh,
        ep,
        topk,
        renormalize,
        routed_scaling_factor,
        logical_intermediate_size,
        axis_name,
        expert_format,
        dispatch,
        push_max_rows,
    )
    op = _export_op(
        _op_name(
            ep,
            topk,
            renormalize,
            routed_scaling_factor,
            logical_intermediate_size,
            axis_name,
            expert_format,
            gate_up_bits,
            dispatch,
            push_max_rows,
        ),
        fn,
        mesh=mesh,
        input_partition_specs=in_specs,
        output_partition_specs=out_specs,
    )
    if hasattr(op, "register_fake"):
        op.register_fake(_fake_like_x)
    return op


def _get_op(
    ep: int,
    topk: int,
    renormalize: bool,
    routed_scaling_factor: float,
    logical_intermediate_size: int,
    axis_name: str = _DEFAULT_MESH_AXIS,
    expert_format: str = fe.EXPERT_FORMAT_K512,
    gate_up_bits: int = k32z.GATE_UP_SHIFT_BITS,
    dispatch: str = fe.DISPATCH,
    push_max_rows: int = fe.PUSH_MAX_ROWS,
) -> Callable:
    key = _op_key(
        ep,
        topk,
        renormalize,
        routed_scaling_factor,
        logical_intermediate_size,
        axis_name,
        expert_format,
        gate_up_bits,
        dispatch,
        push_max_rows,
    )
    op = _OPS.get(key)
    if op is not None:
        return op
    with _OPS_LOCK:
        op = _OPS.get(key)
        if op is None:
            op = _build_op(*key)
            _OPS[key] = op
    return op


@contextlib.contextmanager
def kernel_override(kernel_fn: Callable):
    """Test hook: ops built inside the block call ``kernel_fn`` (same signature
    as ``fused_ep_moe_v2``) instead of the kernel. Clears the op cache on entry
    and restores it on exit."""
    global _KERNEL_FN
    with _OPS_LOCK:
        saved_fn, saved_ops = _KERNEL_FN, dict(_OPS)
        _KERNEL_FN = kernel_fn
        _OPS.clear()
    try:
        yield
    finally:
        with _OPS_LOCK:
            _KERNEL_FN = saved_fn
            _OPS.clear()
            _OPS.update(saved_ops)


# --------------------------------------------------------------------------- #
# gmm layout -> kernel operands (torch; any device).
# --------------------------------------------------------------------------- #
def kpacked_to_npacked_fp4_bytes(packed: torch.Tensor) -> torch.Tensor:
    """gmm-layout bytes ``[E, K/2, N]`` (packed along K, low nibble = even K row) ->
    ``[E, K, N/2]`` bytes packed along N (low nibble = even column): the
    storage of a ``float4_e2m1fn_x2`` ``[E, K, N/2]``. Bit moves only; equal
    to the tests' ``dsv41_fused_ep_reference.gmm_to_torch_fp4_x2_bytes``
    (checked on CPU)."""
    if packed.dtype == torch.int8:
        packed = packed.view(torch.uint8)
    if packed.dtype != torch.uint8 or packed.dim() != 3 or packed.shape[2] % 2:
        raise ValueError(
            f"expected uint8 [E, K/2, N] with even N, got {packed.dtype} "
            f"{tuple(packed.shape)}"
        )
    e, k_half, n = packed.shape
    even_col = packed[:, :, 0::2]  # codes (2j, 2m) low, (2j+1, 2m) high
    odd_col = packed[:, :, 1::2]  # codes (2j, 2m+1) low, (2j+1, 2m+1) high
    even_row = (even_col & 0x0F) | ((odd_col & 0x0F) << 4)  # K row 2j
    odd_row = (even_col >> 4) | (odd_col & 0xF0)  # K row 2j+1
    return torch.stack((even_row, odd_row), dim=2).reshape(e, 2 * k_half, n // 2)


def _drain(device: torch.device) -> None:
    # Bound the chunk transients on the accelerator (as fp8_moe does).
    if device.type != "cpu":
        torch.accelerator.synchronize()


def relayout_gmm_weight(
    weight: torch.Tensor, experts_per_chunk: int = _EXPERTS_PER_CHUNK
) -> torch.Tensor:
    """:func:`kpacked_to_npacked_fp4_bytes` a few experts at a time, on the
    weight's device, into one new uint8 buffer."""
    w = weight.view(torch.uint8) if weight.dtype == torch.int8 else weight
    e, k_half, n = w.shape
    out = torch.empty((e, 2 * k_half, n // 2), dtype=torch.uint8, device=w.device)
    for c0 in range(0, e, experts_per_chunk):
        c1 = min(c0 + experts_per_chunk, e)
        out[c0:c1].copy_(kpacked_to_npacked_fp4_bytes(w[c0:c1]))
        _drain(w.device)
    return out


def materialize_float4_x2(packed_n: torch.Tensor) -> torch.Tensor:
    """N-packed uint8 -> its own ``torch.float4_e2m1fn_x2`` buffer (same bytes),
    as ``fp4_moe._materialize_packed_fp4`` builds the fused-RS weights."""
    fp4 = getattr(torch, "float4_e2m1fn_x2", None)
    if fp4 is None:
        raise RuntimeError("DeepSeek-V4.1 fused EP needs torch.float4_e2m1fn_x2")
    out = torch.empty_like(packed_n, dtype=fp4)
    out.copy_(packed_n.view(fp4))
    return out


def gmm_scale_to_flat(scale: torch.Tensor, *, k: int) -> torch.Tensor:
    """gmm-layout f32 ``[E, K/512, 1, N]`` -> host f32 row-padded flat
    ``[E * fp4_scale_rows(K, 512, padded=True), N]`` (pad rows zero)."""
    flat = fe.gmm_scale_to_kernel(
        scale.detach().to("cpu"), k=k, rhs_qb=fe.RHS_QB, layout="flat"
    )
    return torch.from_numpy(flat)


def _release(param: torch.nn.Parameter) -> None:
    """Point ``param`` at an empty tensor and free its old device buffer."""
    old = param.data
    param.data = torch.empty(0, dtype=old.dtype, device=old.device)
    storage = old.untyped_storage()
    if storage.data_ptr() != param.data.untyped_storage().data_ptr():
        try:
            storage.resize_(0)
        except RuntimeError:  # not resizable (e.g. numpy-backed); dropped below
            pass
    del old


def _convert_weight(experts: torch.nn.Module, name: str) -> torch.Tensor:
    param = getattr(experts, name)
    device = param.data.device
    packed_n = relayout_gmm_weight(param.data)  # held: old + new
    _release(param)  # held: new
    weight = materialize_float4_x2(packed_n)  # held: new + float4
    packed_n.untyped_storage().resize_(0)
    del packed_n
    _drain(device)
    return weight


def _convert_scale(experts: torch.nn.Module, name: str, *, k: int) -> torch.Tensor:
    param = getattr(experts, name)
    device = param.data.device
    flat = gmm_scale_to_flat(param.data, k=k)
    _release(param)
    return flat.to(device)


def _flat_shift_table(experts: torch.nn.Module, name: str) -> torch.Tensor:
    """A K32Z shift table ``[E, rows, cols]`` uint8 -> flat ``[E * rows, cols]``,
    built on the host and moved to the device as the scales above are."""
    param = getattr(experts, name)
    device = param.data.device
    e, rows, cols = param.data.shape
    # copy=True: on a CPU parameter .to("cpu") would alias the storage that
    # _release frees below.
    flat = param.data.detach().to("cpu", copy=True).reshape(e * rows, cols)
    _release(param)
    return flat.to(device)


def _convert_k32z_layer(experts: torch.nn.Module, *, gate_up_bits: int):
    """Checkpoint FP4 experts -> the K32Z kernel operands, one tensor at a time:
    the lossless K32Z conversion (K-packed codes and shift tables), then the
    codes relaid out as the gmm layout's are and the tables flattened."""
    k32z.prepare_tpu_dsv4_mxfp4_k32z_weights(experts, gate_up_bits=gate_up_bits)
    w1 = _convert_weight(experts, "w13_weight")
    w2 = _convert_weight(experts, "w2_weight")
    s1 = _flat_shift_table(experts, "w13_weight_shift")
    s2 = _flat_shift_table(experts, "w2_weight_shift")
    return w1, w2, s1, s2


# --------------------------------------------------------------------------- #
# Eligibility.
# --------------------------------------------------------------------------- #
def is_dsv41_moe(module: torch.nn.Module) -> bool:
    """A DeepseekV2MoE built for DeepSeek-V4.1 (duck-typed: importing
    deepseek_v2 here would pull the whole model zoo)."""
    config = getattr(module, "config", None)
    return (
        bool(getattr(module, "is_deepseek_v4", False))
        and getattr(config, "model_type", None) == "deepseek_v41"
        and getattr(module, "experts", None) is not None
        and getattr(module, "gate", None) is not None
    )


def layer_skip_reason(moe: torch.nn.Module) -> Optional[str]:
    """Why a V4.1 MoE layer keeps the gmm_v2 path although the switch is on.
    The decision depends only on the model config, so all ranks agree."""
    if moe.is_nextn:
        return "NextN/MTP draft layer"
    if moe.is_hash:
        return "hash-routed layer (token-id top-k)"
    return None


def unmet_global_requirements(parallel, exec_ctx, a2a_backend) -> list[str]:
    """Process-wide conditions the fused path relies on (empty when all hold)."""
    world = int(parallel.world_group.world_size)
    ep = int(parallel.moe_ep_size)
    moe_args = exec_ctx.moe
    features = exec_ctx.features
    requirements = (
        ("attention TP size 1", int(parallel.attn_tp_size) == 1),
        (f"MoE EP size == world size ({ep} vs {world})", ep == world),
        (
            f"attention DP size == EP size ({parallel.attn_dp_size} vs {ep})",
            int(parallel.attn_dp_size) == ep,
        ),
        ("pipeline parallel size 1", int(parallel.pp_size) == 1),
        ("attention context parallel size 1", int(parallel.attn_cp_size) == 1),
        ("no MoE all-to-all backend", a2a_backend.is_none()),
        (
            "EPLB and redundant experts off",
            not moe_args.enable_eplb and not moe_args.ep_num_redundant_experts,
        ),
        # A fused layer never runs the layer's TopK, so its routing reaches
        # neither the routed-experts capture nor the expert distribution
        # recorder: refuse both rather than return data without these layers.
        (
            "--enable-return-routed-experts off (fused layers capture no routing)",
            not features.enable_return_routed_experts,
        ),
        (
            "--expert-distribution-recorder-mode unset (fused layers record no "
            "routing)",
            moe_args.expert_distribution_recorder_mode is None,
        ),
        # The relayout and the in-kernel gating take physical expert e as
        # logical expert e.
        (
            "--init-expert-location trivial (experts in logical order)",
            moe_args.init_expert_location == "trivial",
        ),
        # Decoder SWA bounded replay runs layers 21..39 on the decoder tail:
        # enter_late_layer_tail packs the kept rows first (plan_tail_rows), so
        # the tail's real rows are a contiguous prefix, and it swaps
        # num_token_non_padded to that count with the DP row counts. Allowed.
    )
    return [name for name, met in requirements if not met]


def _gmm_fp4_layout(experts, hidden: int, logical: int) -> bool:
    """The gmm layout's FP4-expert tensors with the model's geometry."""
    names = ("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale")
    w13, s13, w2, s2 = (getattr(experts, n, None) for n in names)
    if any(t is None for t in (w13, s13, w2, s2)):
        return False
    packed = (torch.uint8, torch.int8)
    if w13.dtype not in packed or w2.dtype not in packed or w13.dim() != 3:
        return False
    if s13.dtype != torch.float32 or s2.dtype != torch.float32:
        return False
    e, h_half, two_ip = w13.shape
    ip = two_ip // 2
    qb = fe.RHS_QB
    return (
        2 * h_half == hidden
        and hidden % qb == 0
        and ip % qb == 0
        and logical <= ip
        and tuple(w2.shape) == (e, ip // 2, hidden)
        and tuple(s13.shape) == (e, hidden // qb, 1, two_ip)
        and tuple(s2.shape) == (e, ip // qb, 1, hidden)
    )


def _expert_layout_ok(experts, hidden: int, logical: int) -> bool:
    """The gmm layout, or (SGLANG_TPU_DSV41_EXPERT_FORMAT=k32z) the checkpoint
    layout the gmm preparation left for this preparation to convert."""
    if k32z.is_deferred(experts):
        return k32z.checkpoint_layout_ok(experts, hidden, logical)
    return _gmm_fp4_layout(experts, hidden, logical)


def unmet_layer_requirements(
    moe: torch.nn.Module, *, ep: int, multimodal_enabled: Optional[bool]
) -> list[str]:
    """Per-layer conditions (empty when all hold)."""
    config = moe.config
    experts = moe.experts
    gate = moe.gate
    n_routed = int(config.n_routed_experts)
    bias = getattr(gate, "e_score_correction_bias", None)
    vision_routing = gate.e_score_correction_bias_vl is not None
    requirements = (
        (
            "text-only serving (vision top-k is active unless the server runs "
            "--no-enable-multimodal)",
            not vision_routing or multimodal_enabled is False,
        ),
        (
            "FP4 experts prepared for gmm_v2 (SGLANG_DSV4_FP4_EXPERTS=1, "
            "SGLANG_DSV4_FP4_DEQUANT=0)",
            bool(getattr(getattr(experts, "quant_method", None), "is_fp4_expert", 0))
            and _expert_layout_ok(
                experts, int(config.hidden_size), int(config.moe_intermediate_size)
            ),
        ),
        (
            "replicated shared expert (SGLANG_SHARED_EXPERT_TP1=1)",
            getattr(moe, "shared_experts", None) is not None
            and bool(getattr(moe, "_shared_expert_tp1", False)),
        ),
        ("no fused shared experts", not moe.num_fused_shared_experts),
        ("sqrtsoftplus scoring", config.scoring_func == fe.SCORING_FN),
        (
            "swiglu_limit 10 (fixed in the kernel's activation)",
            getattr(config, "swiglu_limit", None) is not None
            and float(config.swiglu_limit) == fe.SWIGLU_LIMIT,
        ),
        (
            "fp32 correction bias over every routed expert",
            bias is not None
            and bias.dtype == torch.float32
            and tuple(bias.shape) == (n_routed,),
        ),
        (
            "routed experts split evenly over EP",
            int(getattr(experts, "num_local_experts", -1)) * ep == n_routed,
        ),
        (
            "layer not prepared for fused_moe_decode",
            not getattr(moe, "_tpu_fused_moe_decode_ready", False),
        ),
    )
    return [name for name, met in requirements if not met]


def _multimodal_enabled() -> Optional[bool]:
    from sglang.srt.runtime_context import get_mm

    try:
        return get_mm().enable_multimodal
    except Exception:  # unpublished config: treat as unknown (not False)
        return None


def _resolve_traced_rank() -> Optional[Callable]:
    try:
        from sglang_tpu.rank_local import traced_rank
    except ImportError:  # no torch_tpu (CPU tests)
        return None
    return traced_rank


# --------------------------------------------------------------------------- #
# Preparation (after weight loading, before precompile).
# --------------------------------------------------------------------------- #
def prepare_moe_layer_for_fused_ep(
    moe: torch.nn.Module,
    *,
    ep: int,
    ep_rank: int,
    multimodal_enabled: Optional[bool],
) -> bool:
    """Replace one layer's gmm-layout expert tensors by the kernel operands,
    build its op and set ``moe.dp_local_forward``. Returns False for a layer
    that keeps the gmm_v2 path; raises when the switch cannot be honored."""
    if getattr(moe, READY_ATTR, False):
        return True
    reason = layer_skip_reason(moe)
    if reason is not None:
        logger.info(
            "SGLANG_TPU_DSV41_FUSED_EP: layer %s keeps gmm_v2 (%s).",
            getattr(moe, "layer_id", "?"),
            reason,
        )
        # K32Z selected: gmm_v2 needs the block-512 layout the gmm preparation
        # left to this step.
        k32z.finish_deferred_as_k512(moe.experts)
        return False
    unmet = unmet_layer_requirements(moe, ep=ep, multimodal_enabled=multimodal_enabled)
    if unmet:
        raise ValueError(
            "SGLANG_TPU_DSV41_FUSED_EP=1, but MoE layer "
            f"{getattr(moe, 'layer_id', '?')} cannot run fused EP: needs "
            + "; ".join(unmet)
        )
    config = moe.config
    experts = moe.experts
    # SGLANG_TPU_DSV41_EXPERT_FORMAT=k32z left the checkpoint experts for this
    # step: the layer takes the K32Z operands and the K32Z op.
    to_k32z = k32z.is_deferred(experts)
    gate_up_bits = k32z.SERVING_GATE_UP_BITS if to_k32z else 0
    dispatch, push_max_rows = selected_dispatch(), fe.PUSH_MAX_ROWS
    key = _op_key(
        ep,
        config.num_experts_per_tok,
        config.norm_topk_prob,
        moe.routed_scaling_factor,
        config.moe_intermediate_size,
        _DEFAULT_MESH_AXIS,
        fe.EXPERT_FORMAT_K32Z if to_k32z else fe.EXPERT_FORMAT_K512,
        gate_up_bits,
        dispatch,
        push_max_rows,
    )
    # Build the op first (outside Dynamo): a failure leaves the weights as loaded.
    _get_op(*key)

    device = experts.w13_weight.data.device
    if to_k32z:
        w1, w2, s1, s2 = _convert_k32z_layer(experts, gate_up_bits=gate_up_bits)
    else:
        hidden = int(experts.w13_weight.shape[1]) * 2
        padded_inter = int(experts.w2_weight.shape[1]) * 2
        w1 = _convert_weight(experts, "w13_weight")
        w2 = _convert_weight(experts, "w2_weight")
        s1 = _convert_scale(experts, "w13_weight_scale", k=hidden)
        s2 = _convert_scale(experts, "w2_weight_scale", k=padded_inter)
    rank = torch.tensor([[ep_rank]], dtype=torch.int32).to(device)
    for name, tensor in ((W1, w1), (W2, w2), (W1_SCALE, s1), (W2_SCALE, s2)):
        experts.register_buffer(name, tensor, persistent=False)
    experts.register_buffer(RANK, rank, persistent=False)

    setattr(moe, _KEY_ATTR, key)
    setattr(moe, _RANK_ATTR, int(ep_rank))
    # The seam in NonCudaDecoderMixin._run_moe_ffn_dp_sync: from now on this
    # layer runs on local rows only (the gmm_v2 tensors are gone).
    moe.dp_local_forward = functools.partial(forward_fused_ep_moe, moe)
    setattr(moe, READY_ATTR, True)
    return True


def prepare_model_for_dsv41_fused_ep(model: torch.nn.Module) -> int:
    """Prepare every eligible V4.1 MoE layer of ``model``; returns the count."""
    global _TRACED_RANK
    if not is_dsv41_fused_ep_enabled():
        return 0
    layers = [m for m in model.modules() if is_dsv41_moe(m)]
    if not layers:
        logger.info(
            "SGLANG_TPU_DSV41_FUSED_EP: %s has no DeepSeek-V4.1 MoE layer.",
            type(model).__name__,
        )
        # K32Z selected: FP4 experts outside V4.1 MoE layers keep gmm_v2.
        k32z.finish_deferred_as_k512(model)
        return 0

    from sglang.srt.layers.moe import get_moe_a2a_backend
    from sglang.srt.runtime_context import get_exec, get_parallel

    parallel = get_parallel()
    unmet = unmet_global_requirements(parallel, get_exec(), get_moe_a2a_backend())
    if unmet:
        raise ValueError("SGLANG_TPU_DSV41_FUSED_EP=1 needs " + "; ".join(unmet))
    ep = int(parallel.moe_ep_size)
    ep_rank = int(parallel.moe_ep_rank)
    _TRACED_RANK = _resolve_traced_rank()
    multimodal = _multimodal_enabled()

    prepared = [
        m
        for m in layers
        if prepare_moe_layer_for_fused_ep(
            m, ep=ep, ep_rank=ep_rank, multimodal_enabled=multimodal
        )
    ]
    if prepared:
        key = getattr(prepared[0], _KEY_ATTR)
        logger.info(
            "SGLANG_TPU_DSV41_FUSED_EP: %d of %d MoE layers run fused_ep_moe_v2 "
            "(op %s, ep rank %d); %d keep gmm_v2.",
            len(prepared),
            len(layers),
            _op_name(*key),
            ep_rank,
            len(layers) - len(prepared),
        )
    # K32Z selected: whatever else still holds deferred checkpoint experts
    # keeps gmm_v2 and needs its block-512 layout.
    k32z.finish_deferred_as_k512(model)
    return len(prepared)


# --------------------------------------------------------------------------- #
# Forward.
# --------------------------------------------------------------------------- #
def check_equal_dp_rows(t_local: int, forward_batch, ep: int) -> None:
    """Trace-time check that every rank runs this many rows (MAX_LEN padding):
    the op is a collective over the EP ranks, so a rank with another row count
    would hang or corrupt the exchange."""
    mode = forward_batch.dp_padding_mode
    if mode is not None and not mode.is_max_len():
        raise RuntimeError(
            f"DeepSeek-V4.1 fused EP needs DpPaddingMode.MAX_LEN, got {mode}"
        )
    counts = forward_batch.global_num_tokens_cpu
    if counts is None:
        if ep > 1:
            raise RuntimeError(
                "DeepSeek-V4.1 fused EP over several ranks needs the DP "
                "per-rank token counts (global_num_tokens_cpu)"
            )
        return
    if len(counts) != ep:
        raise RuntimeError(f"expected {ep} per-rank token counts, got {list(counts)}")
    for count in counts:
        if int(count) != t_local:
            raise RuntimeError(
                f"DeepSeek-V4.1 fused EP needs equal rows on every rank: this "
                f"rank has {t_local}, the padded counts are {list(counts)}"
            )


def real_row_count(forward_batch) -> Optional[torch.Tensor]:
    """This rank's real rows as a device scalar, or None (treat all as real).

    ``num_token_non_padded`` is the LOCAL count derived from the batch's own
    token count (``ForwardBatch.set_local_num_token_non_padded``; set whenever
    ``moe_ep_size > 1``). With attention TP 1 it is the number of real rows at
    the head of this rank's hidden states: decode rows past the real batch,
    extend tokens past the real prompt chunk and every row of an idle rank
    (count 0) are padding. It is a tensor, so the compiled graph does not
    specialize on it.
    """
    return forward_batch.num_token_non_padded


def set_live_verify_rows(forward_batch, live_tokens: int) -> None:
    """TARGET_VERIFY: the real rows are the live requests' verify rows.

    The verify batch is built at the padded batch bucket, so init_new counts every
    row as real; fused EP then routes the padding rows (and every row of an idle DP
    rank) to experts. Live requests come first, so their rows are the prefix
    [0, live_tokens). The DP row counts stay physical: every rank still runs the
    same bucket, only the routing mask changes. A no-op unless fused EP is on and
    the batch carries the count (attention TP 1, so local == global).
    """
    if not is_dsv41_fused_ep_enabled():
        return
    if getattr(forward_batch, "num_token_non_padded", None) is None:
        return
    from sglang_tpu.host_upload import device_scalar

    count = device_scalar(
        int(live_tokens),
        forward_batch.num_token_non_padded.dtype,
        forward_batch.num_token_non_padded.device,
    ).reshape(forward_batch.num_token_non_padded.shape)
    forward_batch.num_token_non_padded = count
    if getattr(forward_batch, "global_num_token_non_padded", None) is not None:
        forward_batch.global_num_token_non_padded = count
        forward_batch.global_num_token_non_padded_cpu = int(live_tokens)


def local_gating_logits(
    moe: torch.nn.Module,
    hidden_states: torch.Tensor,
    num_real: Optional[torch.Tensor],
) -> torch.Tensor:
    """FP32 router logits of this rank's rows; padding rows all -inf.

    The logits come from the layer's own MoEGate exactly as the gmm_v2 path
    computes them (on TPU with a BF16 gate weight: a BF16 x BF16 matmul with
    a BF16 result), then are upcast; the kernel applies sqrt(softplus) in FP32
    as vision_topk's torch fallback does.
    """
    logits = moe.gate(hidden_states).float()
    if num_real is None:
        return logits
    rows = torch.arange(logits.shape[0], device=logits.device, dtype=num_real.dtype)
    padding = (rows >= num_real).unsqueeze(-1)
    return logits.masked_fill(padding, float("-inf"))


def _rank_operand(experts: torch.nn.Module, ep_rank: int, ep: int) -> torch.Tensor:
    if ep > 1 and _TRACED_RANK is not None:
        traced = _TRACED_RANK(ep_rank, group_size=ep)
        if traced is not None:
            return traced.to(dtype=torch.int32).reshape(1, 1)
    return getattr(experts, RANK)


def apply_fused_ep(
    moe: torch.nn.Module, hidden_states: torch.Tensor, gating: torch.Tensor
) -> torch.Tensor:
    """The routed experts of this rank's rows, routed scale included."""
    key = getattr(moe, _KEY_ATTR)
    experts = moe.experts
    op = _get_op(*key)
    return op(
        hidden_states,
        gating,
        getattr(experts, W1),
        getattr(experts, W2),
        getattr(experts, W1_SCALE),
        getattr(experts, W2_SCALE),
        moe.gate.e_score_correction_bias,
        _rank_operand(experts, getattr(moe, _RANK_ATTR), key[0]),
    )


def forward_fused_ep_moe(
    moe: torch.nn.Module,
    hidden_states: torch.Tensor,
    forward_batch,
    *,
    input_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """A prepared layer's ``dp_local_forward``: local rows in, this rank's
    complete MoE output out (routed + shared experts), with no DP
    gather/scatter and no post-experts all-reduce.

    ``input_ids`` is unused: hash-routed layers are never prepared, and vision
    routing is excluded by the text-only requirement."""
    key = getattr(moe, _KEY_ATTR)
    t_local = int(hidden_states.shape[0])
    check_equal_dp_rows(t_local, forward_batch, key[0])
    if t_local == 0:
        # Every rank has 0 rows (counts are equal), so no rank enters the op.
        return hidden_states.new_zeros(hidden_states.shape)
    gating = local_gating_logits(moe, hidden_states, real_row_count(forward_batch))
    out = apply_fused_ep(moe, hidden_states, gating)
    shared = moe._forward_shared_experts(hidden_states)
    if shared is not None:
        out = out + shared
    return out
