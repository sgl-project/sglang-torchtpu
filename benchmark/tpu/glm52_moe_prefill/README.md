# GLM5.2 eight-chip prefill MoE comparison

Eight physical v7x chips, sixteen logical workers, EP16 and attention DP16.
H6144/I2048/E256/TopK8, channelwise FP8. Prefill buckets are 2048 rows per DP
(32768 global rows) and a 64-row tail (1024 global rows).

The input is the replicated FFN tensor plus common selected expert IDs and
coefficients. Each arm returns its DP-local output. Timing includes activation
quantization, route preparation, expert computation, EP transport/combine,
shared-expert computation, and required reduction/scatter. The input DP gather,
gate projection, and expert selection precede this boundary.

- GMM baseline calls the production `_fused_moe_jax` and the production dense
  FP8 linear helper for its TP-sharded shared MLP, followed by reduce-scatter.
- JAX GLM V2 uses the unchanged production `get_op(ep=16, has_shared=True)`,
  including its split/remap/restore adapter and fused replicated shared expert.
- Existing vLLM-derived V2 uses its original `stage='compute'` entry. The
  benchmark adapter reuses its row quantizer and payload builder to consume the
  caller's selected IDs/weights. Its original routing-table construction, fused
  FFN/transport/combine and replica restoration execute unchanged. An external
  TP-sharded shared MLP uses the same dense FP8 helper as the baseline.

The third arm adds a benchmark integration for the existing kernel. Its tile
capacity is 128, matching the registered device test and existing caller
convention. It uses the default routing-plan construction. This first run does
not tune kernel parameters.

Every arm is warmed through `torch.compile(backend='tpu', fullgraph=True)`.
Twelve samples alternate execution order. Barriers precede each timer and
accelerator synchronization ends it. Every worker records actual runtime,
imports, mapped libtpu, correctness, and samples. The report aggregates the
maximum worker latency per iteration for the complete EP operation.

Inputs are synthetic uniform routing and expert-specific constant FP8 slabs
with nonuniform channel scales. Weights are runtime tensor operands with full
production dimensions. Their analytic FP32 reference covers all DP-local
outputs across the sixteen workers. The relative L2 acceptance is the existing
GLM bridge test's 0.15 bound, and all outputs must be finite.
