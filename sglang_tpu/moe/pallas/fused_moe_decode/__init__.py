# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""Fused MoE + Shared Expert + All-Reduce decode kernel for TPU v7x.

Use Case & Overview:
  When serving sparse Mixture-of-Experts (MoE) models on TPU v7x under TP+EP
  (e.g., GLM-5.3-Flash, DeepSeek-V2/V3, Qwen3-MoE), decode and low-token-count
  batches (`num_tokens <= SGLANG_TPU_FUSED_MOE_DECODE_MAX_TOKENS`, `256` by
  default) are bottlenecked by HBM weight-read bandwidth, kernel launch
  overhead, and cross-device collective latency rather than MXU compute.

  In the standard `gmm_v2` MoE path, each layer executes multiple separate
  operations: token permutation/sorting, grouped GEMM1 (`w13`), SwiGLU
  activation, grouped GEMM2 (`w2`), token un-permutation, a separate shared
  expert MLP (`gate_up_proj` + SwiGLU + `down_proj`), and a standalone
  cross-device All-Reduce.

  `fused_moe_decode` fuses the entire decode MoE block into a single Pallas
  kernel call per layer:
    1. Builds the TopK routing mask and per-expert token counts in-kernel,
       skipping HBM DMA reads for any local expert with zero routed tokens.
    2. Streams active routed expert FP8 weights and scales through an
       `NB`-buffered VMEM ring buffer (using full-expert slots when
       `2 * slot <= 24 MiB`, or blocked-intermediate `ib = 256` tiles when
       `e_local * i_e > 16384`).
    3. Computes routed gate/up projections, FP8 dequantization, optional SwiGLU
       clamping (`swiglu_limit`), `SiLU(gate) * up`, and routed down projections
       directly in VMEM without permuting token activations.
    4. Overlaps the shared expert MLP (`gate_up_proj` + SwiGLU + `down_proj`)
       with the initial routed expert weight DMAs.
    5. Performs an in-kernel cross-device All-Reduce (Reduce-Scatter +
       All-Gather via remote VMEM DMA) across the TP/EP group.

  Batches exceeding `SGLANG_TPU_FUSED_MOE_DECODE_MAX_TOKENS` (such as prefill
  or large chunked-prefill batches) automatically dispatch to `gmm_v2`.

Modules:
  - `fused_moe_decode_kernel`: Pallas kernel implementation (`fused_moe_decode`).
  - `fused_moe_decode_op`: TorchTPU custom-op registration, model/layer weight
    preparation, and runtime dispatch helpers.
"""
