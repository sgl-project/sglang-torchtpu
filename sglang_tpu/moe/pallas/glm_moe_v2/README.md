# GLM selected-route fused EP MoE v2

This opt-in candidate ports `sgl-project/sglang-jax` revision
`c948bdda95208e68a474c8088937347d08fe8937`,
`python/sgl_jax/srt/kernels/fused_moe/v2/{kernel,tuned_block_configs}.py`
(Apache-2.0). The valid-route computation and EP pipeline are retained.
Compiler adaptation uses `has_side_effects=False` for the returned-output
call: TorchTPU repartitions CPU-exported StableHLO and rejects the source's
unsharded side-effect custom call. Original and port reproduced that failure
under the same target runtime; an explicit manual-sharding annotation also
failed. The tuning adapter only changes device-name discovery to the existing
CPU-export TPU-info context. This is
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


The bridge uses the explicit logical-rank operand and its gathered table for
its mesh index. It does not emit a host-HLO `partition-id`, which is ambiguous
when TorchTPU partitions the locally embedded module a second time. Pallas's
internal device-id/RDMA semantics are unchanged.

The pure-call dependency contract requires the supported TPU MAX_LEN FFN
caller: while any DP peer is active, every rank consumes a nonempty physical
bucket. The compiled-forward shim publishes that bucket, idle decode creates
physical dummy rows, speculative idle keeps outputs padded, and the FFN
scatter consumes a nonempty slot. Global-empty calls return empty on every
rank without entering the kernel. Directly cropping a participating rank's
kernel result to an unused/empty value is outside this adapter's contract.
The registered test checks a compiled 64-row slot through DP drain/refill.

Source-only debug variables `SGLJAX_SKIP_PREQUANT`,
`FUSED_MOE_V2_PAD_TOPK_TO_128` and `FUSED_MOE_V2_ROUTE_SMEM_TOPK_ONLY` are rejected
when present, before model load and again when building the op. In particular,
the source treats even `SGLJAX_SKIP_PREQUANT=0` as truthy. This candidate always
uses prequantization, 128-wide TopK padding and the ordinary route metadata.
These values are frozen for the process; no source debug API is exposed.

For an eight-chip allocation, run the independent preflight launcher on both
hosts after the harness peer exchange, before checkpoint preparation or model
startup (the same frozen production code is tested, with no temporary kernel):

```sh
/absolute/candidate/venv/bin/python \
  test/manual/kernels/glm_moe_v2_multihost_preflight.py \
  --hosts "$TPU_WORKER_HOSTNAMES" --host-index "$FALCON_RANK" \
  --output-dir "$GLM52_ART/moe-preflight" --seconds 1500
```

It reuses `TopologyConfig`/`discover_host_order` and native environment setup,
launches two hosts x eight workers, and records exact source/runtime identity.
Real binder preflight first packs/replaces small TPU Parameters and checks
storage pointers, release and preserved attributes. It also invokes the real
GroupCoordinator FP8 shared-weight all-gather and verifies the reconstructed
gate/up/down weights and FP32 scales. The tests then exercise EP16 numerical
cases, ownership permutations, padded idle/drain, and full H6144/I2048/E256/K8
at local 6/24/64/2048 rows. Only after both hosts report success may the original
model harness start; failures propagate nonzero and the preflight cannot renew
its deadline. This stage counts inside the allocation's overall deadline.


The source combine assumes the first route determines validity for the entire
row, while its metadata/scatter already handle individual invalid slots. The
port completes that contract: guard each gather by its slot id, wait for the
actual number of started DMA copies, and mask each invalid scratch value and
coefficient before multiplication. A valid slot in a partially invalid row is
still computed; no invalid id is remapped to a real expert and no whole-row
post-mask discards valid contributions. Tests cover an invalid first slot and
invalid later slots, shared on/off, and dump any nonfinite row with its valid
route count. All valid DMA completions and end-of-pipeline drain remain intact.
The committed `glm_moe_v2_source_to_port.patch` reverses the adaptations to the
exact original source bytes; the source diagnostic verifies its pinned SHA256.

The source W8A8 transport also embeds a float32 activation scale into four FP8
lane positions (one at the end of each hidden packing segment), then clears
those positions before FFN1. This loses four activation coordinates; it is
visible in small-H dense-reference error and is part of the source numerical
strategy, not a claim of bitwise equivalence with the old GMM. Full H6144 cases
and model smoke/E2E remain necessary; a relaxed error bound never permits NaN.

Mutable HBM scatter, expert-accumulate, gather and prequantization workspaces
are explicit outputs of the Pallas call. They have the same shapes and memory
spaces as the source, and only completed rows leave the enclosing JAX function.
They are not writable `pl.empty` inputs: XLA sees their writes and simultaneous
buffer lifetimes. No input/output alias is needed because no caller-owned input
is mutated. The compute, DMA, semaphore and drain body is unchanged by this
buffer-interface adaptation. This follows the existing fused EP kernel's
output-workspace pattern; sequential-shape, idle and full-geometry device
validation remain necessary.
