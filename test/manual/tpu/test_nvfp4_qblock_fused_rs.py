# SPDX-License-Identifier: Apache-2.0
"""TP8 expert-TP numerics of SGLANG_TPU_MUSE_MK_QBLOCK's served experts on the native fused-RS MoE.

Each rank loads checkpoint-layout NVFP4 experts at the Muse Spark geometry (256 experts, hidden
4096, an intermediate slice of 512, e4m3 block-16 scales with per-expert globals) through the
serving path: prepare_tpu_nvfp4_moe_weights (fused-RS layout, bf16 scales), then
requant_tpu_nvfp4_experts (block 256). Checks:

  1. the device requant equals an independent CPU requant of the checkpoint tensors, bit for
     bit (codes and bf16 scales), which pins the fused-RS byte layout;
  2. fused-RS (fp4_w4a16, Muse's post-expert norm) against the stock bf16
     gmm_v2 MoE on the requantized weights dequantized, all-reduced and normed in f32;
  3. the requant's own error: the same bf16 reference on the block-16 weights.

    eval $(python3 -m torch.tpu.distributed)
    export TORCH_TPU_TOPOLOGY TORCH_TPU_SLICEBUILDER_ADDRESSES WORLD_SIZE
    torchrun --nproc_per_node="$WORLD_SIZE" test/manual/tpu/test_nvfp4_qblock_fused_rs.py

Rank 0 prints QBLOCK_FUSED_RS lines and QBLOCK_FUSED_RS_PASS when every check holds.
"""

import argparse
import gc
import time

import torch
import torch.distributed as dist

N_EXPERTS, TOP_K, HIDDEN, INTER_LOCAL = 256, 8, 4096, 512
QBLOCK = 256
EXACT_EXPERTS = 4  # experts per rank whose requant is checked bit for bit on CPU
# fused-RS on the requantized weights vs bf16 dequantized ones: same codes and scales, so only
# rounding differs (bf16 weights code x scale, the bf16 SiLU product, output rounding).
TOL_REL_MEAN = 1e-2
TOL_COS = 0.9999
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def _codes_k_last(packed: torch.Tensor) -> torch.Tensor:
    """[..., X/2] bytes packed along the last dim (low nibble first) -> [..., X] codes."""
    return torch.stack((packed & 0xF, packed >> 4), dim=-1).reshape(
        *packed.shape[:-1], -1
    )


def _values(codes: torch.Tensor) -> torch.Tensor:
    return E2M1[(codes & 7).long()] * torch.where(codes >= 8, -1.0, 1.0)


def _checkpoint_layer(rank: int) -> torch.nn.Module:
    """One rank's expert-TP slice in checkpoint layout: (E, N, K/2) e2m1 packed along K,
    (E, N, K/16) e4m3 block scales; block scales vary within a 256 block like real ones."""
    import types

    g = torch.Generator().manual_seed(4242 + rank)
    layer = torch.nn.Module()
    layer.moe_runner_config = types.SimpleNamespace(is_gated=True, activation="silu")

    def param(t):
        return torch.nn.Parameter(t, requires_grad=False)

    def scales(shape, gain):
        s = torch.exp(torch.randn(shape, generator=g) * 0.35) * gain
        return s.clamp(max=448.0).to(torch.float8_e4m3fn)  # e4m3fn overflows to NaN

    he, il = HIDDEN, INTER_LOCAL
    layer.w13_weight = param(
        torch.randint(
            0, 256, (N_EXPERTS, 2 * il, he // 2), dtype=torch.uint8, generator=g
        )
    )
    layer.w2_weight = param(
        torch.randint(0, 256, (N_EXPERTS, he, il // 2), dtype=torch.uint8, generator=g)
    )
    layer.w13_weight_scale = param(scales((N_EXPERTS, 2 * il, he // 16), 64.0))
    layer.w2_weight_scale = param(scales((N_EXPERTS, he, il // 16), 64.0))
    layer.w13_weight_scale_2 = param(
        (torch.rand(N_EXPERTS, 2, generator=g) * 0.5 + 0.75) / (64.0 * 2.0 * he**0.5)
    )
    layer.w2_weight_scale_2 = param(
        (torch.rand(N_EXPERTS, generator=g) * 0.5 + 0.75)
        / (64.0 * 2.0 * (8 * il) ** 0.5)
    )
    return layer


def _cpu_requant(packed: torch.Tensor, scale: torch.Tensor, gs: torch.Tensor):
    """Checkpoint-layout reference: block-16 values (scales folded with the globals and rounded
    to bf16, as the served layout stores them) -> codes [E, N, K] and bf16 scales [E, N, K / 256]
    of one scale per 256 K, nearest e2m1 by argmin with ties to even."""
    s16 = (scale.float() * gs).to(torch.bfloat16).float()
    w = _values(_codes_k_last(packed)) * s16.repeat_interleave(16, dim=-1)
    e, n, k = w.shape
    wb = w.reshape(e, n, k // QBLOCK, QBLOCK)
    sb = (wb.abs().amax(dim=-1) / 6.0).to(torch.bfloat16)
    sbf = sb.float()
    q = (wb / torch.where(sbf > 0, sbf, 1.0)[..., None]).clamp(-6, 6)
    d = (q.abs()[..., None] - E2M1).abs()
    near = d == d.amin(dim=-1, keepdim=True)
    idx = torch.where(near, torch.arange(8) % 2, 2).argmin(dim=-1)
    return (idx | torch.where(q < 0, 8, 0)).reshape(e, n, k), sb


def _served_codes(w: torch.Tensor) -> torch.Tensor:
    """Served (E, K, N/2) fp4 pairs -> codes [E, N, K]."""
    return _codes_k_last(w.view(torch.uint8).cpu()).transpose(1, 2)


def _dequant_served(w: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Served (E, K, N/2) fp4 + (E, K/G, N) scales -> bf16 (E, K, N), per expert on device."""
    from sglang_tpu.moe.nvfp4_requant import _e2m1_values

    out = []
    u = w.view(torch.uint8)
    for e0 in range(0, u.shape[0], 16):
        c = _codes_k_last(u[e0 : e0 + 16])
        sc = s[e0 : e0 + 16].float()
        v = _e2m1_values(c) * sc.repeat_interleave(c.shape[1] // sc.shape[1], dim=1)
        out.append(v.to(torch.bfloat16))
    return torch.cat(out)


def _norm_combine(y, gain, tw):
    """fused-RS post-expert norm + top-k combine on the summed slot outputs [T, K, H], f32."""
    y = y.to(torch.bfloat16).float() * (gain.float() + 1.0)
    z = y * torch.rsqrt((y * y).mean(dim=-1, keepdim=True) + 1e-8)
    return (z.to(torch.bfloat16).float() * tw.float()[..., None]).sum(dim=1)


def _stats(a: torch.Tensor, b: torch.Tensor) -> tuple:
    rel = (a - b).abs().mean().item() / b.abs().mean().item()
    cos = torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
    return rel, cos


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", default="16,256,2048")
    args = ap.parse_args()
    dist.init_process_group(backend="tpu_dist")
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world == 8, f"TP8 expert TP, got {world} ranks"

    from sglang_tpu.fp4_moe import prepare_tpu_nvfp4_moe_weights, qblock_layout
    from sglang_tpu.moe.pallas import fused_moe_rs_op
    from sglang_tpu.moe.pallas.fused_moe_op import fused_moe_call
    from sglang_tpu.muse_megakernel.expert_requant import requant_tpu_nvfp4_experts

    device = torch.device("tpu")
    layer = _checkpoint_layer(rank)
    ck = {
        n: getattr(layer, n).data[:EXACT_EXPERTS].clone()
        for n in ("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale")
    }
    gs13 = layer.w13_weight_scale_2.data[:EXACT_EXPERTS]
    gs2 = layer.w2_weight_scale_2.data[:EXACT_EXPERTS]
    layer = layer.to(device)
    original_prepare = fused_moe_rs_op.prepare_fused_moe_rs
    fused_moe_rs_op.prepare_fused_moe_rs = lambda _: None
    try:
        prepare_tpu_nvfp4_moe_weights(
            layer, group_size=16, scale_dtype=torch.bfloat16, use_fused_rs=True
        )
    finally:
        fused_moe_rs_op.prepare_fused_moe_rs = original_prepare
    layer.__dict__["_tpu_fused_rs_expert_tp"] = True
    w13_b16 = _dequant_served(layer.w13_weight, layer.w13_weight_scale)
    w2_b16 = _dequant_served(layer.w2_weight, layer.w2_weight_scale)
    t0 = time.time()
    requant_tpu_nvfp4_experts(layer, QBLOCK)
    torch.accelerator.synchronize()
    if rank == 0:
        print(f"QBLOCK_FUSED_RS requant one layer: {time.time() - t0:.1f}s", flush=True)
    assert qblock_layout(layer) == QBLOCK
    assert tuple(layer.w13_weight_scale.shape) == (
        N_EXPERTS,
        HIDDEN // QBLOCK,
        2 * INTER_LOCAL,
    )
    assert tuple(layer.w2_weight_scale.shape) == (
        N_EXPERTS,
        INTER_LOCAL // QBLOCK,
        HIDDEN,
    )
    gc.collect()

    ok = True
    # 1. bit-exact requant against the checkpoint-layout CPU reference
    il = INTER_LOCAL
    gate_up = torch.cat([gs13[:, :1].expand(-1, il), gs13[:, 1:].expand(-1, il)], dim=1)
    for name, sname, gs in (
        ("w13", "w13_weight_scale", gate_up[..., None]),
        ("w2", "w2_weight_scale", gs2[:, None, None]),
    ):
        ref_c, ref_s = _cpu_requant(ck[name + "_weight"], ck[sname], gs)
        dev_w = (layer.w13_weight if name == "w13" else layer.w2_weight)[:EXACT_EXPERTS]
        dev_s = (layer.w13_weight_scale if name == "w13" else layer.w2_weight_scale)[
            :EXACT_EXPERTS
        ]
        codes_eq = torch.equal(_served_codes(dev_w), ref_c.to(torch.uint8))
        scales_eq = torch.equal(dev_s.cpu().transpose(1, 2), ref_s)
        ok &= codes_eq and scales_eq
        if rank == 0:
            print(
                f"QBLOCK_FUSED_RS requant {name}: codes_exact={codes_eq} scales_exact={scales_eq}",
                flush=True,
            )

    w13_q = _dequant_served(layer.w13_weight, layer.w13_weight_scale)
    w2_q = _dequant_served(layer.w2_weight, layer.w2_weight_scale)
    gain = (torch.randn(HIDDEN, generator=torch.Generator().manual_seed(7)) * 0.1).to(
        torch.bfloat16
    )
    gain_d = gain.to(device)
    group_offset = torch.tensor([rank * N_EXPERTS], dtype=torch.int32, device=device)

    def kernel(x, tw, ids):
        return fused_moe_rs_op.fused_moe_rs_call(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            topk_weights=tw,
            topk_ids=ids,
            post_expert_norm_weight=gain_d,
            group_offset=group_offset,
            topk=TOP_K,
            activation="silu",
            has_post_expert_norm=True,
            fp8_direct_write=False,
            expert_tp=True,
            fp4_w4a16=True,
        )

    fused_moe_rs_op._build_fused_moe_rs_op()  # the mesh lookup does not trace
    compiled = torch.compile(kernel, fullgraph=True, dynamic=False)
    for tokens in (int(t) for t in args.tokens.split(",")):
        g = torch.Generator().manual_seed(9000 + tokens)
        x = (torch.randn(tokens, HIDDEN, generator=g)).to(torch.bfloat16)
        ids = torch.stack(
            [torch.randperm(N_EXPERTS, generator=g)[:TOP_K] for _ in range(tokens)]
        ).to(torch.int32)
        tw = torch.softmax(torch.randn(tokens, TOP_K, generator=g), dim=-1).to(
            torch.bfloat16
        )
        xd, idd, twd = x.to(device), ids.to(device), tw.to(device)
        refs = []
        for w1, w2 in ((w13_q, w2_q), (w13_b16, w2_b16)):
            y = fused_moe_call(
                hidden_states=xd,
                w1=w1,
                w2=w2,
                topk_weights=torch.ones_like(twd),
                topk_ids=idd,
                topk=TOP_K,
                activation="silu",
                no_combine=True,
                global_num_experts=N_EXPERTS,
            ).float()
            dist.all_reduce(y)
            refs.append(_norm_combine(y.cpu(), gain, tw))
        padded = fused_moe_rs_op.fused_moe_rs_padded_tokens(tokens, TOP_K, world)
        shard = padded // world
        out = compiled(xd, twd, idd).float().cpu()
        assert out.shape == (shard, HIDDEN), out.shape
        lo, hi = rank * shard, min((rank + 1) * shard, tokens)
        # every rank joins the reductions; a rank with only padding rows reports no error
        rel_q = rel_16 = rel_rq = 0.0
        cos_q = cos_16 = cos_rq = 1.0
        if hi > lo:
            rel_q, cos_q = _stats(out[: hi - lo], refs[0][lo:hi])
            rel_16, cos_16 = _stats(out[: hi - lo], refs[1][lo:hi])
            rel_rq, cos_rq = _stats(refs[0][lo:hi], refs[1][lo:hi])
        passed = rel_q < TOL_REL_MEAN and cos_q > TOL_COS
        flag = torch.tensor([int(passed)], dtype=torch.int32, device=device)
        # max over ranks of the errors and of -cos
        stats = torch.tensor(
            [rel_q, -cos_q, rel_16, -cos_16, rel_rq, -cos_rq], device=device
        )
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        dist.all_reduce(stats, op=dist.ReduceOp.MAX)
        stats = stats.cpu()
        ok &= bool(flag.item())
        if rank == 0:
            print(
                f"QBLOCK_FUSED_RS tokens={tokens} (max over ranks) "
                f"vs_block256_bf16: rel_mean={stats[0]:.2e} cos={-stats[1]:.6f}; "
                f"vs_block16_bf16: rel_mean={stats[2]:.2e} cos={-stats[3]:.6f}; "
                f"requant_error(bf16 refs): rel_mean={stats[4]:.2e} cos={-stats[5]:.6f} "
                f"pass={bool(flag.item())}",
                flush=True,
            )
    okt = torch.tensor([int(ok)], dtype=torch.int32, device=device)
    dist.all_reduce(okt, op=dist.ReduceOp.MIN)
    if rank == 0 and okt.item():
        print("QBLOCK_FUSED_RS_PASS", flush=True)
    print(f"QBLOCK_FUSED_RS_DONE rank={rank} ok={bool(okt.item())}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
