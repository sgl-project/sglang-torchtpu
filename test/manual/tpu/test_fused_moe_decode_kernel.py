# SPDX-License-Identifier: Apache-2.0
"""Numerical TP8/EP8 test for the compiled FP8 fused_moe_decode Pallas kernel."""

import os
from types import SimpleNamespace

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault(
    "XLA_FLAGS",
    f"--xla_force_host_platform_device_count={os.environ.get('WORLD_SIZE', '8')}",
)
os.environ.setdefault("SGLANG_TPU_ENABLE_FUSED_MOE_DECODE_KERNEL", "1")
os.environ.setdefault("SGLANG_TPU_MOE_IMPLEMENTATION", "fused_moe_decode")
os.environ.setdefault("SGLANG_TPU_FUSED_MOE_DECODE_MAX_TOKENS", "256")

import torch
import torch.distributed as dist
import torch.nn as nn
from torch_tpu._internal.distributed import tpu_distributed  # noqa: F401

import sglang_tpu.moe.pallas.fused_moe_decode.fused_moe_decode_op as fmd_op
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
    def __init__(
        self,
        rank: int,
        world_size: int,
        e_local: int,
        h: int,
        i_e: int,
        i_s: int,
        routed_scale: float,
        device: torch.device,
    ):
        super().__init__()
        self.tp_size = world_size
        self.routed_scaling_factor = routed_scale
        self.experts = DummyExperts(e_local, h, i_e, device)
        self.shared_experts = DummySharedExperts(h, i_s, rank, device)

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_w: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        if should_use_fused_moe_decode(self, hidden_states):
            return apply_fused_moe_decode(self, hidden_states, (topk_w, topk_ids, None))
        raise RuntimeError("Expected should_use_fused_moe_decode=True")


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

    # GLM-5.3-Flash TP8+EP8 dimensions:
    # h = 2048, e_local = 128 // 8 = 16, i_e = 1536, i_s = 1536 // 8 = 192 (padded to 256)
    e_local, h, i_e, i_s = 16, 2048, 1536, 192
    routed_scale = 2.5

    orig_get_parallel = fmd_op.get_parallel
    fmd_op.get_parallel = lambda: fake_parallel
    try:
        moe = DummyMoE(rank, world_size, e_local, h, i_e, i_s, routed_scale, device)
        assert prepare_moe_layer_for_fused_decode(moe)
        x_257 = torch.empty((257, h), dtype=torch.bfloat16, device=device)
        assert not should_use_fused_moe_decode(moe, x_257)

        compiled_moe = torch.compile(
            moe, backend=current_platform.get_compile_backend(), fullgraph=True
        )

        tokens_env = os.environ.get(
            "SGLANG_TPU_FUSED_MOE_DECODE_TEST_TOKENS", "1,16,64,256"
        )
        token_sizes = [int(x) for x in tokens_env.split(",") if x.strip()]

        for num_tokens in token_sizes:
            torch.manual_seed(999 + num_tokens)
            x_cpu = torch.randn(num_tokens, h, dtype=torch.bfloat16)
            topk_ids_cpu = torch.randint(
                0, world_size * e_local, (num_tokens, 8), dtype=torch.int32
            )
            topk_w_cpu = torch.softmax(
                torch.randn(num_tokens, 8, dtype=torch.float32), dim=-1
            ).to(torch.bfloat16)

            x = x_cpu.to(device)
            topk_ids = topk_ids_cpu.to(device)
            topk_w = topk_w_cpu.to(device)

            assert should_use_fused_moe_decode(moe, x)
            out = compiled_moe(x, topk_w, topk_ids)

            # Reference on local rank followed by all-reduce across 8 ranks
            xf = x.float()
            local_routed = torch.zeros(
                num_tokens, h, dtype=torch.float32, device=device
            )
            exp_start = rank * e_local
            for slot in range(8):
                e_idx = topk_ids[:, slot] - exp_start
                valid = (e_idx >= 0) & (e_idx < e_local)
                if valid.any():
                    for t in range(num_tokens):
                        if bool(valid[t].item()):
                            ei = int(e_idx[t].item())
                            gu = xf[t : t + 1] @ moe.experts.w13_deq[ei]
                            g, u = gu.chunk(2, dim=-1)
                            act = torch.nn.functional.silu(g) * u
                            y = act @ moe.experts.w2_deq[ei]
                            local_routed[t : t + 1] += (
                                topk_w[t, slot].float() * routed_scale * y
                            )

            s_gu = xf @ moe.shared_experts.gate_up_proj.w_deq
            sg, su = s_gu.chunk(2, dim=-1)
            s_act = torch.nn.functional.silu(sg) * su
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
            assert cos > 0.99, f"Low cosine similarity {cos} at T={num_tokens}"
            assert rel_l2 < 0.08, f"High relative L2 {rel_l2} at T={num_tokens}"
    finally:
        fmd_op.get_parallel = orig_get_parallel

    print(f"FUSED_MOE_DECODE_KERNEL_PASS rank={rank}", flush=True)


if __name__ == "__main__":
    main()
