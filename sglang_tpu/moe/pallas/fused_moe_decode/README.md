# Fused MoE Decode Kernel (`sglang_tpu/moe/pallas/fused_moe_decode`)

Port of `vllm_torchtpu/kernels/experimental/fused_moe_decode` to `sglang-torchtpu` for TPU v7x.

Fuses the entire decode MoE block (`num_tokens <= 256`) into a single Pallas call:
1. TopK routing mask and per-expert token counts
2. Sparse expert weight gather via an `NB`-buffered VMEM ring buffer (full-expert mode when `2 * slot <= 24 MiB`, or blocked-intermediate `ib = 256` mode for large experts such as GLM-5.3-Flash / DeepSeek-V3)
3. Routed gate/up projection + FP8 dequantization + optional SwiGLU clamping (`swiglu_limit`) + SiLU(gate) * up + routed down projection
4. Shared expert MLP (`gate_up_proj` + SwiGLU + `down_proj`) overlapped with routed expert DMA
5. In-kernel cross-device All-Reduce (Reduce-Scatter + All-Gather via remote VMEM DMA)

## Configuration

Enable via either server config (`--moe-runner-backend` / `SERVER_ARGS_JSON`) or environment variables (`RUNTIME_ENV_JSON`):

- `--moe-runner-backend pallas_tpu_fused_decode` (or `"moe_runner_backend": "pallas_tpu_fused_decode"` in `SERVER_ARGS_JSON`)
- `SGLANG_TPU_MOE_IMPLEMENTATION=fused_moe_decode` (or `SGLANG_TPU_ENABLE_FUSED_MOE_DECODE_KERNEL=1`)
- `SGLANG_TPU_FUSED_MOE_DECODE_MAX_TOKENS=256` (default: `256`; tokens `> 256` automatically dispatch to `gmm_v2`)
