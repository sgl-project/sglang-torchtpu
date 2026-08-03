# SPDX-License-Identifier: Apache-2.0
"""Numerics test for the TPU W4A16 (int4 weight-only) fused-MoE path.

Builds synthetic compressed-tensors pack-quantized weights (offset-binary
nibbles, LSB-first along K, exactly the checkpoint format), applies the same
load-time transform prepare_tpu_w4a16_moe_weights performs (XOR nibble sign
flip + scale reshape), runs the Pallas fused_moe_call with w_quant_bits=4 on
one TPU chip, and compares against a float32 torch reference MoE.

Run on a TPU host (single chip):
  source /cache/venv/bin/activate
  python test/manual/tpu/test_w4a16_moe_tpu.py
"""

import os

# Pin to a single local chip and scrub every multihost hint the GKE pod env
# injects (2-host slice) — libtpu otherwise expects both slice-builder
# workers and aborts in single-process use.
os.environ["TPU_VISIBLE_CHIPS"] = "0"
os.environ["TPU_CHIPS_PER_HOST_BOUNDS"] = "1,1,1"
os.environ["TPU_HOST_BOUNDS"] = "1,1,1"
os.environ["HOST_BOUNDS"] = "1,1,1"
os.environ["CHIPS_PER_HOST_BOUNDS"] = "1,1,1"
os.environ["TPU_WORKER_ID"] = "0"
os.environ["WRAP"] = "false,false,false"
for _v in (
    "TPU_WORKER_HOSTNAMES",
    "TPU_PROCESS_ADDRESSES",
    "TPU_PROCESS_PORT",
    "TORCH_TPU_SLICEBUILDER_ADDRESSES",
    "TORCH_TPU_TOPOLOGY",
):
    os.environ.pop(_v, None)

import torch  # noqa: E402

from sglang.srt.platforms import current_platform  # noqa: E402

assert current_platform.device_type == "tpu", current_platform.device_type
current_platform.init_backend()

from sglang.srt.layers.moe.pallas.fused_moe_op import fused_moe_call  # noqa: E402

INT4_SIGN_XOR = -2004318072  # 0x88888888 as int32


def pack_ct_int4(q: torch.Tensor) -> torch.Tensor:
    """Pack signed int4 values [..., K, N] the compressed-tensors way:
    offset-binary (q+8), 8 consecutive K-values per int32 word, LSB-first.
    Returns [..., K//8, N] int32."""
    assert q.shape[-2] % 8 == 0
    stored = (q + 8).to(torch.int64)  # [0, 15]
    *lead, K, N = stored.shape
    stored = stored.reshape(*lead, K // 8, 8, N)
    word = torch.zeros(*lead, K // 8, N, dtype=torch.int64)
    for i in range(8):
        word |= stored[..., i, :] << (4 * i)
    # keep low 32 bits, reinterpret as signed int32
    word = (word & 0xFFFFFFFF).to(torch.uint32).view(torch.int32)
    return word


def main():
    torch.manual_seed(0)
    E, K, N = 4, 256, 128  # hidden K, per-expert intermediate N
    GS = 32  # group size
    M, TOPK = 64, 2

    # ---- synthetic checkpoint (float32 reference + packed int4) ----
    q13 = torch.randint(-8, 8, (E, K, 2 * N), dtype=torch.int32)
    s13 = (torch.rand(E, K // GS, 2 * N) * 0.05 + 0.01).to(torch.bfloat16)
    q2 = torch.randint(-8, 8, (E, N, K), dtype=torch.int32)
    s2 = (torch.rand(E, N // GS, K) * 0.05 + 0.01).to(torch.bfloat16)

    # dequant reference [E, K, 2N] / [E, N, K] float32
    w13_ref = q13.float() * s13.float().repeat_interleave(GS, dim=1)
    w2_ref = q2.float() * s2.float().repeat_interleave(GS, dim=1)

    x = (torch.randn(M, K) * 0.5).to(torch.bfloat16)
    topk_ids = torch.randint(0, E, (M, TOPK), dtype=torch.int32)
    topk_weights = torch.rand(M, TOPK)
    topk_weights = (topk_weights / topk_weights.sum(-1, keepdim=True)).to(
        torch.bfloat16
    )

    # ---- torch reference MoE (float32) ----
    xf = x.float()
    ref = torch.zeros(M, K)
    for m in range(M):
        for t in range(TOPK):
            e = int(topk_ids[m, t])
            gate_up = xf[m] @ w13_ref[e]  # [2N]
            act = torch.nn.functional.silu(gate_up[:N]) * gate_up[N:]
            ref[m] += float(topk_weights[m, t]) * (act @ w2_ref[e])

    # ---- kernel path: pack -> XOR flip -> fused_moe_call ----
    w13_packed = pack_ct_int4(q13).bitwise_xor(INT4_SIGN_XOR)
    w2_packed = pack_ct_int4(q2).bitwise_xor(INT4_SIGN_XOR)
    # scales -> [E, B, 1, out] (prepare_tpu_w4a16_moe_weights layout)
    s13_k = s13.unsqueeze(2).contiguous()
    s2_k = s2.unsqueeze(2).contiguous()

    dev = "tpu"
    out = fused_moe_call(
        hidden_states=x.to(dev),
        w1=w13_packed.to(dev),
        w2=w2_packed.to(dev),
        topk_weights=topk_weights.to(dev),
        topk_ids=topk_ids.to(dev),
        topk=TOPK,
        activation="silu",
        w1_scale=s13_k.to(dev),
        w2_scale=s2_k.to(dev),
        w_quant_bits=4,
    )
    got = out.float().cpu()

    # ---- compare ----
    # Pointwise relative error is meaningless near silu zero-crossings in
    # bf16; check the p99 absolute error against the reference RMS instead.
    err_p99 = (got - ref).abs().flatten().quantile(0.99).item()
    ref_rms = ref.pow(2).mean().sqrt().item()
    cos = torch.nn.functional.cosine_similarity(
        got.flatten(), ref.flatten(), dim=0
    ).item()
    print(f"p99_abs_err={err_p99:.5f}  ref_rms={ref_rms:.5f}  cosine={cos:.6f}")
    assert cos > 0.999, f"cosine {cos} too low — nibble order/sign mismatch?"
    assert err_p99 < 0.05 * ref_rms, f"p99 err {err_p99} vs rms {ref_rms}"
    print("W4A16 MoE numerics: PASS")


if __name__ == "__main__":
    main()
