# SPDX-License-Identifier: Apache-2.0
"""TP8 expert-TP numerics of the NVFP4 fused-RS MoE at the Muse Spark geometry.

Each rank holds an intermediate slice of 512 of all 256 experts (hidden 4096, top-8,
post-expert norm), prepared through the serving path. The fused-RS output, chunked as
the serving wrapper chunks expert-TP calls, is checked against the stock gmm_v2 MoE on
the same weights dequantized, all-reduced, normed and combined in f32, with and without
the post-expert norm. ``--weights bf16`` runs fused-RS on those dequantized weights
instead. Set SGLANG_TPU_FUSED_RS_ETP_PAIR_EXCHANGE=1 to cover pair exchange (4096 tokens
reach it).

    eval $(python3 -m torch.tpu.distributed)
    export TORCH_TPU_TOPOLOGY TORCH_TPU_SLICEBUILDER_ADDRESSES WORLD_SIZE
    torchrun --nproc_per_node="$WORLD_SIZE" test/manual/tpu/test_nvfp4_expert_tp_fused_rs.py

Every rank prints ETP_FUSED_RS_PASS when its token shard is within tolerance.
"""

import argparse
import types

import torch
import torch.distributed as dist

N_EXPERTS, TOP_K, HIDDEN, INTER_LOCAL = 256, 8, 4096, 512
# Same fp4 codes and scales on both sides, so only rounding differs (bf16 dequantized
# weights, fp8-quantized activations in fused-RS, bf16 partial sums).
TOL_REL_MEAN = 3e-2
TOL_COS = 0.999


def _checkpoint_layer(rank: int) -> torch.nn.Module:
    """One rank's expert-TP slice in checkpoint layout: e2m1 packed along K, e4m3
    block-16 scales and per-expert global scales."""
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


def _codes_k_last(packed: torch.Tensor) -> torch.Tensor:
    """[..., X/2] bytes packed along the last dim (low nibble first) -> [..., X] codes."""
    lo, hi = packed & 0xF, packed >> 4
    return torch.stack([lo, hi], dim=-1).reshape(*packed.shape[:-1], -1)


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


def _combine(y, gain, tw):
    """Post-expert norm (when ``gain`` is set) + top-k combine of the summed slot
    outputs [T, K, H], f32."""
    y = y.to(torch.bfloat16).float()
    if gain is not None:
        y = y * (gain.float() + 1.0)
        y = y * torch.rsqrt((y * y).mean(dim=-1, keepdim=True) + 1e-8)
    return (y.to(torch.bfloat16).float() * tw.float()[..., None]).sum(dim=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", default="64,4096")
    ap.add_argument("--weights", choices=("fp4", "bf16"), default="fp4")
    args = ap.parse_args()
    dist.init_process_group(backend="tpu_dist")
    from sglang_tpu.compile_cache import add_tpu_kernel_inputs_to_compile_cache_key

    # The fused-RS env flags change the kernels inside the custom op; key the
    # torch.compile cache on them, as serving does.
    add_tpu_kernel_inputs_to_compile_cache_key()
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world == 8, f"TP8 expert TP, got {world} ranks"

    from sglang_tpu.fp4_moe import prepare_tpu_nvfp4_moe_weights
    from sglang_tpu.moe.pallas import fused_moe_rs_op
    from sglang_tpu.moe.pallas.fused_moe_op import fused_moe_call

    device = torch.device("tpu")
    layer = _checkpoint_layer(rank).to(device)
    original_prepare = fused_moe_rs_op.prepare_fused_moe_rs
    fused_moe_rs_op.prepare_fused_moe_rs = lambda _: None
    try:
        prepare_tpu_nvfp4_moe_weights(
            layer, group_size=16, scale_dtype=torch.bfloat16, use_fused_rs=True
        )
    finally:
        fused_moe_rs_op.prepare_fused_moe_rs = original_prepare
    w13 = _dequant_served(layer.w13_weight, layer.w13_weight_scale)
    w2 = _dequant_served(layer.w2_weight, layer.w2_weight_scale)
    gain = (torch.randn(HIDDEN, generator=torch.Generator().manual_seed(7)) * 0.1).to(
        torch.bfloat16
    )
    gain_d = gain.to(device)
    group_offset = torch.tensor([rank * N_EXPERTS], dtype=torch.int32, device=device)
    if args.weights == "fp4":
        weights = dict(
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
        )
    else:
        weights = dict(w1=w13, w2=w2, w1_scale=None, w2_scale=None)
    max_tokens = fused_moe_rs_op.fused_moe_rs_expert_tp_max_tokens(TOP_K)

    def chunks(tokens):
        return [(s, min(max_tokens, tokens - s)) for s in range(0, tokens, max_tokens)]

    def kernel(x, tw, ids, post_norm: bool):
        # Chunked as fused_moe_rs_gather chunks expert-TP calls; this rank's token
        # shard of each chunk.
        return tuple(
            fused_moe_rs_op.fused_moe_rs_call(
                hidden_states=x[s : s + n],
                **weights,
                topk_weights=tw[s : s + n],
                topk_ids=ids[s : s + n],
                post_expert_norm_weight=gain_d,
                group_offset=group_offset,
                topk=TOP_K,
                activation="silu",
                has_post_expert_norm=post_norm,
                fp8_direct_write=False,
                expert_tp=True,
            )
            for s, n in chunks(x.shape[0])
        )

    fused_moe_rs_op._build_fused_moe_rs_op()  # the mesh lookup does not trace
    compiled = torch.compile(kernel, fullgraph=True, dynamic=False)
    ok = True
    cases = [
        (int(t), post_norm)
        for t in args.tokens.split(",")
        for post_norm in (True, False)
    ]
    for tokens, post_norm in cases:
        g = torch.Generator().manual_seed(9000 + tokens)
        x = torch.randn(tokens, HIDDEN, generator=g).to(torch.bfloat16)
        ids = torch.stack(
            [torch.randperm(N_EXPERTS, generator=g)[:TOP_K] for _ in range(tokens)]
        ).to(torch.int32)
        tw = torch.softmax(torch.randn(tokens, TOP_K, generator=g), dim=-1).to(
            torch.bfloat16
        )
        xd, idd, twd = x.to(device), ids.to(device), tw.to(device)
        y = fused_moe_call(
            hidden_states=xd,
            w1=w13,
            w2=w2,
            topk_weights=torch.ones_like(twd),
            topk_ids=idd,
            topk=TOP_K,
            activation="silu",
            no_combine=True,
            global_num_experts=N_EXPERTS,
        ).float()
        dist.all_reduce(y)
        ref = _combine(y.cpu(), gain if post_norm else None, tw)
        shards = compiled(xd, twd, idd, post_norm)
        got, want = [], []
        for (s, n), shard in zip(chunks(tokens), shards):
            per = shard.shape[0]
            lo, hi = s + rank * per, min(s + (rank + 1) * per, s + n)
            if hi > lo:  # else this rank holds only padding rows of the chunk
                got.append(shard[: hi - lo].float().cpu())
                want.append(ref[lo:hi])
        rel = cos = None
        if got:
            a, b = torch.cat(got).flatten(), torch.cat(want).flatten()
            rel = ((a - b).abs().mean() / b.abs().mean()).item()
            cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
            ok &= rel < TOL_REL_MEAN and cos > TOL_COS
        print(
            f"ETP_FUSED_RS rank={rank} tokens={tokens} post_norm={post_norm} "
            f"rel_mean={rel} cos={cos}",
            flush=True,
        )
    print(f"ETP_FUSED_RS_{'PASS' if ok else 'FAIL'} rank={rank}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
