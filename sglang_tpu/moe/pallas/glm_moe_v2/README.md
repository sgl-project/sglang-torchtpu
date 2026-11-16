# GLM selected-route fused EP MoE v2

This opt-in candidate ports `sgl-project/sglang-jax` revision
`c948bdda95208e68a474c8088937347d08fe8937`,
`python/sgl_jax/srt/kernels/fused_moe/v2/{kernel,tuned_block_configs}.py`
(Apache-2.0). `kernel.py` preserves the source computation and all RDMA/waits.
The sole compiler adaptation sets the Pallas call's `has_side_effects=False`:
TorchTPU re-partitions CPU-exported StableHLO and rejects the source's unsharded
side-effect custom call. The MoE is pure input-to-output; communication buffers
and semaphores are internal scratch, and output dependencies retain ordering.
This follows the existing fused EP port's compiler contract. The tuning file
changes only device-name discovery to the CPU-export TPU-info context. This is
separate from the existing vLLM-derived `fused_moe_v2`, whose input is router
logits and packed gate/up rather than GLM's existing selected ids/coefficients.

Set `SGLANG_TPU_USE_GLM_MOE_V2=1` before model load. The default remains the
existing backend. Eligible target GLM DSA MoE blocks use this candidate;
`is_nextn` draft blocks keep their ordinary attention-TP backend. Unsupported
layouts retain that backend with a diagnostic. Required layout: PP1, EP=TP,
expert TP1, attention-DP=TP, selected-route SiLU, channelwise FP8, no EPLB or
external all-to-all. MTP remains enabled.

`SGLANG_TPU_GLM_MOE_V2_SHARED=0` leaves the shared expert external. Otherwise
eligible FP8 channelwise shared experts run inside the kernel. Loading gathers
the shared projection shards once, preserving their checkpoint FP8 values and
FP32 channel scales. Packed gate and up are gathered independently; the down
projection is gathered along K, while its output-channel scales are replicated.
The additional replicated shared weights cost about `3*H*I` bytes per layer
plus scales. Unsupported shared formats retain their external MLP.

The platform's existing `process_model_after_loading` seam binds a plugin
subclass of `DeepseekV2MoE`. Its original gate/TopK computes the selected routes,
including bias correction, grouping, renormalization and the original selected
ids. Routed scaling applies only to routed coefficients. The subclass's
existing `use_tpu_fused_moe_decode` contract tells the FFN boundary that the
result is already summed; the kernel owns dispatch, EP exchange, FFN, route
weighting and combine. No external StandardDispatcher or expert all-reduce is
executed. With external shared MLP, only its TP partial is all-reduced once.
With fusion it contributes once to each originating row inside the kernel.

The FFN boundary presents replicated, globally gathered DP rows. The bridge
splits that replica into contiguous physical-mesh slices, masks padding with
`-1` ids and zero coefficients, and all-gathers the completed rows back into
original order. A tensor operand carries logical EP rank so selected ids are
mapped to the actual mesh's expert ownership. Pod rank is never an EP rank.
Gate/up runtime weights are packed once as `[2,E_local,H,I]` and scales as
`[2,E_local,I]`; these replace canonical Parameter data while preserving loader
attributes and hold no duplicate routed copy. The candidate currently targets
initial checkpoint load; changing its toggle on a loaded/compiled model is
unsupported.

Numerical differences from ordinary `pallas_tpu`: the source kernel uses FP8
activation quantization per token/quantization group, FP32 dot accumulation, and
its fused FP8 shared-expert path. The existing GMM uses K-block activation
quantization and its default BF16 accumulator. Scales still multiply weights;
no online weight requantization, checkpoint change, W8A16 conversion or router
replacement is performed. Basic function/performance takes precedence over tiny
floating-point differences for this candidate.

Run the device bridge/reference test with the target absolute interpreter:

```sh
PYTHONPATH="$PWD/python:$PWD" /absolute/candidate/venv/bin/python \
  test/registered/kernels/ops/moe/test_glm_ep_moe_v2.py
```

The launcher uses that interpreter for `torch.tpu.distributed` and every
TorchTPU worker. It covers local 6/12/18/24 rows, a 65-row drain tail,
random/hot/zero routing, invalid ids, partial DP rows and shared on/off. It
records actual worker imports, package versions, compiler flags, mapped libtpu,
FP32-reference relative L2 and five post-warmup host-observed kernel latencies.
This test is separate from model E2E and does not establish E2E speedup.

JAX source declares `jax[tpu]==0.11.1`; the Torch candidate declares JAX/JAXLIB
0.11.2, Torch 2.11.0, torch_tpu 0.2.1.dev20261028201649 and libtpu 0.0.50.
Use the candidate-declared stack for source/port testing and record the actual
worker runtime. E2E uses the frozen historical Falcon baseline
`exp-6mtkc6t2qi`, as explicitly requested, without rerunning baseline A/B.
Cross-run comparisons are descriptive and cannot isolate kernel causality.
