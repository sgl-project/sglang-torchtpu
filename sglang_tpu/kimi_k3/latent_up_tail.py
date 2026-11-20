# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3 MoE tail fused into the host-split latent up-proj (opt-in,
SGLANG_TPU_K3_LATENT_UP_ATTN_RES).

A single-token decode layer ends with the latent up-proj (host-split GEMV
kernel, whose tail is a latency-bound host exchange), the residual add
``(up + shared) + prefix`` and, at the start of the next layer, the
attention-side attention-residual aggregation of that sum (score, softmax,
mix, input RMSNorm: the kimi_attn_res_decode kernel). Here the latent up-proj
kernel also does the add and the aggregation in its epilogue
(host_split_gemv_local(tail=...)), with the same ops in the same order.

Wiring (all Python, resolved while Dynamo traces the decode forward):
- HostSplitUpTail replaces layer i's latent up-proj callable. For a
  single-token call it defers: it returns the latent unchanged and leaves
  itself pending.
- The AROUND hook on kimi_k3._add3 (the MoE tail add, which always follows the
  latent up-proj when there is a shared-expert output) runs the fused kernel
  on (latent, shared, prefix) and returns the sum; the normed aggregation
  waits for layer i + 1.
- The attention-residual aggregate hook records each call's (bank, nvb) (the
  MLP-side aggregation of layer i is the last one before its MoE, and layer
  i + 1's attention side reads the same bank rows) and hands the pending
  normed tensor to layer i + 1's attention-side aggregation.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import torch


class _State:
    def __init__(self):
        # (bank, nvb) of the last attention-residual aggregation.
        self.agg: list = []
        # (HostSplitUpTail, latent) of a deferred latent up-proj.
        self.pending: list = []
        # (normed, tail id, nvb) for the next attention-side aggregation.
        self.normed: list = []


_STATE = _State()

# Plain bool (Dynamo traces the check): set once a layer was bound.
_ACTIVE = False


def active() -> bool:
    return _ACTIVE


def record_aggregation(bank: torch.Tensor, nvb: int) -> None:
    _STATE.agg.clear()
    _STATE.agg.append((bank, nvb))


def take_normed(prefix_a, prefix_b, nvb, out_norm) -> Optional[torch.Tensor]:
    """The fused aggregation for this attention-side call, if the previous
    layer's latent up-proj computed it for exactly this call."""
    if not _STATE.normed:
        return None
    normed, tail_id, nv = _STATE.normed.pop()
    if (
        prefix_b is None
        and nv == nvb
        and getattr(out_norm, "_latent_up_tail_id", -1) == tail_id
        and prefix_a.shape == normed.shape
    ):
        return normed
    return None


class HostSplitUpTail(NamedTuple):
    """Layer i's single-token latent up-proj with the MoE tail add and layer
    i + 1's attention-side aggregation in its kernel."""

    plain: object  # the HostSplitUp / HostSplitUpPrefetch it replaces
    weight: object  # StaticFp8Weight
    ops: tuple  # (op without the prefix add, op with it)
    next_weight: Optional[torch.Tensor]  # the next layer's qkvg prefetch, if any
    slot: object  # its PrefetchSlot, or None
    cw: torch.Tensor  # layer i + 1 score weight (get_cw), f32 [H]
    norm_w: torch.Tensor  # layer i + 1 input_layernorm weight, f32 [H]
    eps: float
    out_eps: float
    tail_id: int

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] == 1 and _STATE.agg:
            _STATE.pending.clear()
            _STATE.pending.append((self, x))
            return x
        return self.plain(x)

    def finish(self, latent, shared, prefix):
        bank, nvb = _STATE.agg[-1]
        w = self.weight
        args = [latent, w.weight, w.out_scale, w.inv_x]
        if self.next_weight is not None:
            args.append(self.next_weight)
        args.append(shared)
        if prefix is not None:
            args.append(prefix)
        args += [bank, self.cw, self.norm_w]
        outs = self.ops[0 if prefix is None else 1](
            *args, int(nvb), float(self.eps), float(self.out_eps)
        )
        if self.next_weight is not None:
            self.slot.value.clear()
            self.slot.value.append(outs[2])
        _STATE.normed.clear()
        _STATE.normed.append((outs[1], self.tail_id, nvb))
        return outs[0]


def _fusable(up: HostSplitUpTail, latent, shared, prefix) -> bool:
    if not _STATE.agg:
        return False
    bank, nvb = _STATE.agg[-1]
    hidden = up.cw.shape[0]
    return (
        nvb >= 1
        and bank.dim() == 3
        and bank.shape[0] == 1
        and bank.shape[2] == hidden
        and bank.dtype == torch.bfloat16
        and shared.shape == (1, hidden)
        and shared.dtype == torch.bfloat16
        and (
            prefix is None
            or (prefix.shape == (1, hidden) and prefix.dtype == torch.bfloat16)
        )
    )


def add3_around(original, a, b, c, **kwargs):
    """AROUND hook for kimi_k3._add3 (the MoE tail add bf16(a + b) [+ c])."""
    if not _STATE.pending:
        return original(a, b, c, **kwargs)
    up, latent = _STATE.pending.pop()
    if a.shape != latent.shape:
        raise RuntimeError(
            "deferred Kimi-K3 latent up-proj: the next tail add got an "
            "unexpected operand"
        )
    if _fusable(up, latent, b, c):
        return up.finish(latent, b, c)
    return original(up.plain(latent), b, c, **kwargs)


def check_consumed() -> None:
    """A deferred latent up-proj that no tail add consumed is a wiring bug."""
    if _STATE.pending:
        _STATE.pending.clear()
        raise RuntimeError("deferred Kimi-K3 latent up-proj was never consumed")


def bind(layers, layout) -> int:
    """Wrap each host-split latent up-proj whose next layer has an
    attention-side aggregation the kernel can take. Returns the count."""
    global _ACTIVE
    from sglang.srt.layers.attn_residual import get_cw
    from sglang.srt.models.kimi_k3 import KimiK3MoE
    from sglang_tpu.hooks import attn_residual as attn_res_hook
    from sglang_tpu.kimi_k3.host_split_op import host_split_op
    from sglang_tpu.kimi_k3.static_fp8 import HostSplitUp, HostSplitUpPrefetch

    kernel = attn_res_hook._kernel
    if layout is None or kernel is None:
        return 0
    world, local, _ = layout
    count = 0
    for idx, (prev, nxt) in enumerate(zip(layers, layers[1:])):
        moe = getattr(prev, "mlp", None)
        if not isinstance(moe, KimiK3MoE):
            continue
        up = moe._host_split_latent_up
        if isinstance(up, HostSplitUpPrefetch):
            next_weight, slot = up.next_weight, up.slot
        elif isinstance(up, HostSplitUp):
            next_weight, slot = None, None
        else:
            continue
        used = up.op_used
        if not used or not getattr(nxt, "use_attn_residuals", False):
            continue
        score_proj = getattr(nxt, "self_attention_res_proj", None)
        score_norm = getattr(nxt, "self_attention_res_norm", None)
        out_norm = getattr(nxt, "input_layernorm", None)
        if score_proj is None or score_norm is None or out_norm is None:
            continue
        if not (
            kernel._default_rms_norm(score_norm) and kernel._default_rms_norm(out_norm)
        ):
            continue
        w = up.weight
        cols = w.weight.shape[1]
        ops = tuple(
            host_split_op(
                world, local, cols, used, prefetch=next_weight is not None, tail=t
            )
            for t in (1, 2)
        )
        tail_id = idx + 1
        out_norm._latent_up_tail_id = tail_id
        moe._host_split_latent_up = HostSplitUpTail(
            plain=up,
            weight=w,
            ops=ops,
            next_weight=next_weight,
            slot=slot,
            cw=get_cw(score_proj, score_norm),
            norm_w=kernel.get_norm_weight_f32(out_norm),
            eps=float(score_norm.variance_epsilon),
            out_eps=float(out_norm.variance_epsilon),
            tail_id=tail_id,
        )
        count += 1
    _ACTIVE = count > 0
    return count
