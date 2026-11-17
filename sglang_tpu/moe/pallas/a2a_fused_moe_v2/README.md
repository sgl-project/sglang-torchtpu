# A2A fused MoE V2

The kernel and tuning table originate from `sgl-project/sglang-jax` revision
`c948bdda95208e68a474c8088937347d08fe8937`, under
`python/sgl_jax/srt/kernels/fused_moe/v2/`, Apache-2.0.
The `get_op(ep_size, has_shared)` bridge accepts selected expert IDs and
coefficients, channelwise FP8 expert weights, and an explicit logical EP rank.
Gate/up weights use `[2,E_local,H,I]` and scales use `[2,E_local,I]`.

The adapter splits the replicated FFN rows into physical-mesh slices, maps
logical expert ownership, and restores the complete output row order. Shared
fusion takes replicated FP8 gate/up/down weights and channel scales. These
shared weights require approximately `3*H*I` additional bytes per MoE layer
compared with a fully TP-sharded shared expert.

Load-time block-to-channel conversion gives each TP down-projection shard its
own output-channel scales. Shared fusion takes the maximum scale across the
TP shards for each output channel, re-encodes each local FP8 down shard at
that common scale, then gathers FP8 weights. This adds one load-time FP8
rounding step for differing shard scales and keeps transient storage local.
Already common scales produce an exact scale ratio of one.

TorchTPU CPU-export adaptations retain the valid-route compute pipeline.
The custom call is pure, mutable scratch buffers are explicit outputs, and
individual invalid routing slots have guarded DMA starts, waits and combines.
Mesh identity derives from a rank tensor because TorchTPU repartitions the
exported module when embedding it in the enclosing compiled graph.

The source quantization packs a float32 activation scale into four FP8 lane
positions before FFN1, clearing those positions for computation. Numerical
checks therefore use a finite-output requirement and an explicit relaxed
FP32-reference error bound.

`SGLJAX_SKIP_PREQUANT`, `FUSED_MOE_V2_PAD_TOPK_TO_128` and
`FUSED_MOE_V2_ROUTE_SMEM_TOPK_ONLY` are rejected when present. The supported
configuration uses prequantization, 128-wide TopK padding and ordinary routing
metadata.

The registered TorchTPU numerical test is
`test/registered/kernels/ops/moe/test_a2a_fused_moe_v2.py`. It covers selected
routes, invalid slots, padding, drain/refill, shared-expert gathering and full
GLM dimensions. Runtime dependencies are pinned in `python/pyproject_tpu.toml`.

## Model integration

Set `SGLANG_TPU_USE_A2A_FUSED_MOE_V2=1` before model load, with
`--moe-runner-backend pallas_tpu --tpu-fp8-runtime channelwise`.
The existing loader finalization entry
`prepare_model_for_fused_moe_decode` binds eligible target GLM MoE blocks to
`GlmA2AFusedMoEV2` after quantization preparation. The native MTP draft keeps
its attention-TP path. The shared expert is fused by default, controlled by
`SGLANG_TPU_A2A_FUSED_MOE_V2_SHARED`.

The binding requires PP1, EP=TP, attention-DP=TP, unsharded routed experts,
channelwise FP8, SiLU, ordinary selected routes, and separate shared MLP
weights. SBO/TBO and EPLB are excluded by eligibility checks. The target model
logs `A2A_FUSED_MOE_V2_PREPARED` and the exported row shapes log
`A2A_FUSED_MOE_V2_EXPORT`. The FFN `summed_by_compute` contract consumes the
already-combined output once.

PR #2238's block-to-channel conversion stores dense shared-expert channel
scales in `weight_scale`. Shared packing reads those canonical post-load
scales and gathers gate/up TP shards separately. Expert weights are packed
once per load. Initialize a fresh model when changing this implementation
switch or reloading its weights.
