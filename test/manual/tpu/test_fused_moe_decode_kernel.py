# SPDX-License-Identifier: Apache-2.0
"""Numerical TP8/EP8 test for the compiled FP8 fused_moe_decode Pallas kernel."""

import os
from types import SimpleNamespace

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault(
    "XLA_FLAGS",
    f"--xla_force_host_platform_device_count={os.environ.get('WORLD_SIZE', '8')}",
)
os.environ.setdefault("SGLANG_TPU_FUSED_MOE_DECODE_MAX_TOKENS", "256")

import torch
import torch.distributed as dist
import torch.nn as nn
from torch_tpu._internal.distributed import tpu_distributed  # noqa: F401

import sglang_tpu.moe.pallas.fused_moe_decode.fused_moe_decode_op as fmd_op
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.platforms import current_platform
from sglang_tpu.moe.pallas.fused_moe_decode.fused_moe_decode_op import (
    apply_fused_moe_decode,
    prepare_moe_layer_for_fused_decode,
    should_use_fused_moe_decode,
)


def _quantize_colwise(w: torch.Tensor):
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    scale = w.abs().amax(dim=-2).clamp_min(1e-6) / fp8_max
    q = (w / scale.unsqueeze(-2)).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
    return q, scale.to(torch.bfloat16)


class DummyExperts(nn.Module):
    def __init__(self, e_local: int, h: int, i_e: int, device: torch.device):
        super().__init__()
        torch.manual_seed(42)
        w13_f = torch.randn(e_local, h, 2 * i_e, dtype=torch.float32) * 0.05
        w2_f = torch.randn(e_local, i_e, h, dtype=torch.float32) * 0.05
        w13_q, w13_s = _quantize_colwise(w13_f)
        w2_q, w2_s = _quantize_colwise(w2_f)
        self.register_buffer("w13_weight", w13_q.to(device))
        self.register_buffer("w13_weight_scale", w13_s[:, None, None, :].to(device))
        self.register_buffer("w2_weight", w2_q.to(device))
        self.register_buffer("w2_weight_scale", w2_s[:, None, None, :].to(device))
        self.w13_deq = (w13_q.float() * w13_s.float().unsqueeze(1)).to(device)
        self.w2_deq = (w2_q.float() * w2_s.float().unsqueeze(1)).to(device)


class DummyLinear(nn.Module):
    def __init__(self, k_dim: int, n_dim: int, seed: int, device: torch.device) -> None:
        super().__init__()
        torch.manual_seed(seed)
        w_f = torch.randn(k_dim, n_dim, dtype=torch.float32) * 0.05
        w_q, w_s = _quantize_colwise(w_f)
        self.register_buffer("weight", w_q.to(device))
        self.register_buffer("weight_scale", w_s[None, None, :].to(device))
        self.w_deq = (w_q.float() * w_s.float().unsqueeze(0)).to(device)


class DummySharedExperts(nn.Module):
    def __init__(self, h: int, i_s: int, rank: int, device: torch.device):
        super().__init__()
        self.gate_up_proj = DummyLinear(h, 2 * i_s, 100 + rank, device)
        self.down_proj = DummyLinear(i_s, h, 200 + rank, device)


class DummyMoE(nn.Module):
    # Stands in for a DeepseekV2MoE under a decoder layer that passes the
    # predicate to its FFN boundary as summed_by_compute.
    allow_tpu_fused_moe_decode = True

    def __init__(
        self,
        rank: int,
        world_size: int,
        e_local: int,
        h: int,
        i_e: int,
        i_s: int,
        routed_scale: float,
        swiglu_limit,
        device: torch.device,
    ):
        super().__init__()
        self.tp_size = world_size
        self.routed_scaling_factor = routed_scale
        self.swiglu_limit = swiglu_limit
        self.experts = DummyExperts(e_local, h, i_e, device)
        self.shared_experts = DummySharedExperts(h, i_s, rank, device)

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_w: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        if should_use_fused_moe_decode(self, hidden_states, _DECODE_BATCH):
            return apply_fused_moe_decode(self, hidden_states, (topk_w, topk_ids, None))
        raise RuntimeError("Expected should_use_fused_moe_decode=True")


_DECODE_BATCH = SimpleNamespace(forward_mode=ForwardMode.DECODE)
_EXTEND_BATCH = SimpleNamespace(forward_mode=ForwardMode.EXTEND)
_VERIFY_BATCH = SimpleNamespace(forward_mode=ForwardMode.TARGET_VERIFY)


def _swiglu(gate_up: torch.Tensor, limit) -> torch.Tensor:
    gate, up = gate_up.chunk(2, dim=-1)
    if limit is not None:
        gate = gate.clamp(max=limit)
        up = up.clamp(min=-limit, max=limit)
    return torch.nn.functional.silu(gate) * up


def _check_case(
    name: str,
    *,
    rank: int,
    world_size: int,
    device: torch.device,
    e_local: int,
    h: int,
    i_e: int,
    i_s: int,
    routed_scale: float,
    swiglu_limit,
    token_sizes,
) -> None:
    top_k = 8
    moe = DummyMoE(
        rank, world_size, e_local, h, i_e, i_s, routed_scale, swiglu_limit, device
    )
    assert prepare_moe_layer_for_fused_decode(moe), name
    assert moe._tpu_fused_moe_decode_swiglu_limit == swiglu_limit, name

    x_small = torch.empty((16, h), dtype=torch.bfloat16, device=device)
    x_257 = torch.empty((257, h), dtype=torch.bfloat16, device=device)
    assert not should_use_fused_moe_decode(moe, x_257, _DECODE_BATCH), name
    # Small non-decode batches keep the gmm_v2 path.
    assert not should_use_fused_moe_decode(moe, x_small, _EXTEND_BATCH), name
    assert not should_use_fused_moe_decode(moe, x_small, _VERIFY_BATCH), name
    assert not should_use_fused_moe_decode(moe, x_small, None), name

    compiled_moe = torch.compile(
        moe, backend=current_platform.get_compile_backend(), fullgraph=True
    )

    for num_tokens in token_sizes:
        torch.manual_seed(999 + num_tokens)
        x_cpu = torch.randn(num_tokens, h, dtype=torch.bfloat16)
        topk_ids_cpu = torch.randint(
            0, world_size * e_local, (num_tokens, top_k), dtype=torch.int32
        )
        topk_w_cpu = torch.softmax(
            torch.randn(num_tokens, top_k, dtype=torch.float32), dim=-1
        ).to(torch.bfloat16)

        x = x_cpu.to(device)
        topk_ids = topk_ids_cpu.to(device)
        topk_w = topk_w_cpu.to(device)

        assert should_use_fused_moe_decode(moe, x, _DECODE_BATCH), name
        out = compiled_moe(x, topk_w, topk_ids)

        # Reference on the local rank followed by an all-reduce across ranks.
        xf = x.float()
        local_ids = topk_ids.long() - rank * e_local
        local_routed = torch.zeros(num_tokens, h, dtype=torch.float32, device=device)
        for ei in range(e_local):
            coeff = ((local_ids == ei).float() * topk_w.float()).sum(dim=-1)
            act = _swiglu(xf @ moe.experts.w13_deq[ei], swiglu_limit)
            y = act @ moe.experts.w2_deq[ei]
            local_routed += (coeff * routed_scale).unsqueeze(-1) * y

        s_act = _swiglu(xf @ moe.shared_experts.gate_up_proj.w_deq, swiglu_limit)
        local_shared = s_act @ moe.shared_experts.down_proj.w_deq
        local_total = (local_routed + local_shared).to(torch.bfloat16)
        dist.all_reduce(local_total, op=dist.ReduceOp.SUM)

        cos = torch.nn.functional.cosine_similarity(
            out.float().flatten(), local_total.float().flatten(), dim=0
        ).item()
        rel_l2 = (
            (out.float() - local_total.float()).norm()
            / local_total.float().norm().clamp_min(1e-6)
        ).item()
        assert cos > 0.99, f"[{name}] Low cosine similarity {cos} at T={num_tokens}"
        assert rel_l2 < 0.08, f"[{name}] High relative L2 {rel_l2} at T={num_tokens}"
        if rank == 0:
            print(
                f"[{name}] T={num_tokens} cos={cos:.5f} rel_l2={rel_l2:.4f}",
                flush=True,
            )


def main():
    dist.init_process_group(backend="tpu_dist")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == 8
    device = torch.device("tpu")

    fake_parallel = SimpleNamespace(
        tp_size=world_size,
        moe_ep_size=world_size,
        moe_ep_rank=rank,
        pp_size=1,
    )

    tokens_env = os.environ.get(
        "SGLANG_TPU_FUSED_MOE_DECODE_TEST_TOKENS", "1,16,64,256"
    )
    token_sizes = [int(x) for x in tokens_env.split(",") if x.strip()]

    orig_get_parallel = fmd_op.get_parallel
    orig_enabled = fmd_op.is_fused_moe_decode_enabled
    fmd_op.get_parallel = lambda: fake_parallel
    # Stands in for `--moe-runner-backend pallas_tpu_fused_decode`.
    fmd_op.is_fused_moe_decode_enabled = lambda: True
    try:
        # GLM-5.3-Flash at TP8+EP8: h = 4096, 288 routed experts -> e_local = 36,
        # moe_intermediate_size = 2048, one shared expert -> i_s = 2048 / 8 = 256,
        # routed_scaling_factor = 2.5, swiglu_limit = 10.0.
        _check_case(
            "glm53_flash",
            rank=rank,
            world_size=world_size,
            device=device,
            e_local=36,
            h=4096,
            i_e=2048,
            i_s=256,
            routed_scale=2.5,
            swiglu_limit=10.0,
            token_sizes=token_sizes,
        )
        # Unclamped SwiGLU with a shared-expert width that is not a multiple of
        # 128 (i_s = 192 is zero-padded to 256 at preparation time).
        _check_case(
            "unclamped_padded_shared",
            rank=rank,
            world_size=world_size,
            device=device,
            e_local=16,
            h=2048,
            i_e=1536,
            i_s=192,
            routed_scale=2.5,
            swiglu_limit=None,
            token_sizes=[t for t in token_sizes if t <= 64] or token_sizes[:1],
        )
    finally:
        fmd_op.get_parallel = orig_get_parallel
        fmd_op.is_fused_moe_decode_enabled = orig_enabled

    print(f"FUSED_MOE_DECODE_KERNEL_PASS rank={rank}", flush=True)


if __name__ == "__main__":
    main()
