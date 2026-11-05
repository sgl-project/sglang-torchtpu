# Fused MoE Decode Kernel (`sglang_tpu/kernels/ops/moe/fused_moe_decode`)

Port of `vllm_torchtpu/kernels/experimental/fused_moe_decode` to `sglang-torchtpu` for TPU v7x.

For `ForwardMode.DECODE` batches with `num_tokens <= SGLANG_TPU_FUSED_MOE_DECODE_MAX_TOKENS` (configurable, `256` by default), the kernel fuses the whole MoE block into one Pallas call:
1. TopK routing mask and per-expert token counts
2. Sparse expert weight gather through a VMEM ring buffer of `nb` slots (`3 <= nb <= 6`), chosen by `plan_weight_ring`:
   - whole-expert slots when a 3-deep ring of whole experts fits in VMEM (`3 * _unit_bytes(h, i) <= vmem_capacity`), or when no block width divides `i`;
   - otherwise blocked-intermediate slots of width `ib`: the widest of `(256, 128)` that divides `i` and still leaves room for 3 slots next to the token-side buffers and an 8 MiB reserve (e.g. GLM-5.3-Flash at EP8, `h = 4096`, `i = 2048`).
3. Routed gate/up projection + FP8 dequantization + optional SwiGLU clamping (`swiglu_limit`, `10.0` for GLM-5.3-Flash) + SiLU(gate) * up + routed down projection
4. Shared expert MLP (`gate_up_proj` + SwiGLU + `down_proj`) overlapped with routed expert DMA
5. In-kernel cross-device All-Reduce (Reduce-Scatter + All-Gather via remote VMEM DMA)

All other batches (extend/prefill, mixed chunked prefill, speculative target-verify and draft-extend, and decode batches above the threshold) use the regular `gmm_v2` path.

## Configuration

- `--moe-runner-backend pallas_tpu_fused_decode` (or `"moe_runner_backend": "pallas_tpu_fused_decode"` in `SERVER_ARGS_JSON`) enables the kernel.
- `SGLANG_TPU_FUSED_MOE_DECODE_MAX_TOKENS` (default `256`) sets the largest decode batch that uses the kernel.

## Supported models

The kernel all-reduces in-kernel, so the decoder layer must pass `DeepseekV2MoE.use_tpu_fused_moe_decode(...)` to its FFN boundary as `summed_by_compute`. Decoder layers that do so set `mlp.allow_tpu_fused_moe_decode = True`: `DeepseekV2DecoderLayer` (DeepSeek-V2/V3 and their NextN layers) and `Glm5NextDecoderLayer` (GLM-5.3-Flash). MoE layers that use hash routing, vision top-k routing, all-to-all dispatch, or a replicated (TP1) shared expert are skipped at preparation time.
