# Fused EP MoE v2: model integration

This package ports the common MoE kernels from vllm-torchtpu `epic/dsv41`
at `ef2599d3d4342f41d2ad223b266e126299862b53`, including the shared-expert
kernel and related tests. File headers retain their upstream source paths.
Models, including the separate DeepSeek V4.1 package, call one entry point:

```python
from sglang_tpu.moe.pallas.fused_moe_v2 import WeightFormat, fused_ep_moe_v2

routed, next_load = fused_ep_moe_v2(
    x, w1, w2, w1_scale, w2_scale, gating,
    topk=6, renormalize=True, mesh=ep_mesh, capacity=128,
    weight_format=WeightFormat.FP4, rhs_qb=512,
    scoring_fn="sqrtsoftplus", score_bias=score_bias,
    hash_topk_ids=hash_topk_ids,  # None on ordinary top-k layers
    expert_slot=expert_slot, expert_load=load,
    dispatch="all_gather",  # Or "push", with explicit push_options.
)
```

This is a kernel contract, not a serving-backend registration or a model
weight loader. The caller owns the EP mesh, weight conversion, layout and
accumulator lifetime. TPU execution and model integration must be validated
separately from CPU table tests and TPU-target export.

## Static placement and load accounting

- `gating` and `score_bias` use **logical expert** order. `hash_topk_ids`, when
  present, also contains logical IDs; it changes selection, not score weights.
  Do not remap those IDs to physical slots before calling this kernel.
- Weights, weight scales and expert biases use **physical slot** order, with
  the same expert axis on every operand. Every replica must hold the same
  logical expert's weights. The physical slot count is divisible by EP size.
- `expert_slot` is replicated `int32[E_logical, R]`. Each row contains at least
  one valid physical slot followed by `-1` padding. The model's loader must
  validate slot bounds, ownership and replica contents before device upload;
  the hot path does not copy this table back to the host to validate it.
- Replica selection is `(global token row) % replica_count`. It preserves the
  selected logical experts, top-k ordering and weights. `mesh_ep_ranks` maps
  mesh order to EP-rank order when those differ; both slot IDs and returned
  load counts use EP-rank order.
- `expert_load` is replicated `int32[E_physical]`. Passing it returns
  `(routed, old_load + step_counts)`; it does not mutate the input. Retain the
  returned accumulator for the next call and reset it at the model's chosen
  collection boundary. Omitting it returns only `routed`.
- Mask padding with non-finite gate rows (normally all `-inf`); zero logits
  describe a live row and will count as routed traffic. Internal token-replica
  padding is already masked. Input rows must divide evenly over the EP mesh.
- Static placement works in both dispatch modes. `expert_load` requires
  `stage="fused"`. The package does not implement a layout optimizer, dynamic
  weight movement or SGLang's EPLB recorder/control plane.

## Dispatch and optional fusion

`all_gather` remains the default. `push` sends routed activations directly to
their destination slots and accepts only the fused stage. The names in
`push_fused_moe.PUSH_OPTIONS` are passed through `push_options`; defaults are the kernel
function's defaults, not vLLM's environment defaults. In particular, enabling
`exchange_row_gather` requires `tc_exchange`. Model configuration should choose
the options explicitly after measuring its shapes. No environment variables
are read by this package.

`shared_expert=(w13, s13, w2, s2)` with `shared_limit` is push-only and requires
replicated per-channel FP8 weights. It returns the shared result separately:
`(routed, shared)` or `(routed, next_load, shared)`. The model combines the
shared and routed outputs; the kernel does not add them together.

The normalized Top6 router and fused shared expert use measured shape tables.
Push's fused routing metadata additionally requires
`(topk, E_physical, EP, tile_m) == (6, 384, 16, 128)` and a measured token
bucket. Adding replica slots changes `E_physical`, so that metadata fusion
falls back while static placement and push dispatch remain available.
`preprocess_topology` is opt-in, applies only to all-gather, and describes the
validated single-host eight-rank topology.

Hardware gates read the **Pallas lowering target**, including
`tpu_info_for_cpu_export()`, rather than the JAX tracing host. Untargeted CPU
runs and unmeasured targets/shapes retain the general implementations.

## Module boundaries

`push_fused_moe/host.py` builds dispatch segments and sorted rows; `plan_kernel.py`
optionally fuses table construction; `sc_dispatch.py` performs SparseCore
pre-push; `tc_exchange.py` performs the optional TensorCore routing exchange.
These implement different communication steps and are not copies of v2's
activation all-gather. `push_fused_moe/kernel.py` and `push_fused_moe/layer.py` retain push's
distinct buffer lifetime and scheduling logic, while reusing v2's FFN math,
wire epilogue, payload format, slot mapping and combine builder.

`push_fused_moe/routing.py` and its small `fusion.py` helpers isolate the measured
metadata fusion without importing a V4.1 model module. The upstream unused
`presort.py` is intentionally not ported. The dense shared-expert kernel lives
in `sglang_tpu/fp8_swiglu_mlp.py` so it can also be used outside MoE.

## Four-chip kernel validation

The registered `test_fused_ep_moe_push.py` includes full expert sets on one
v7x 2x2x1 host (four physical chips, eight EP ranks):

| Bucket | Rows per rank | Logical / physical experts | Top-k |
|---|---:|---:|---:|
| Verify | 288 | 384 / 392 | 6 |
| Draft | 240 | 128 / 136 | 3 |
| Prefill | 2048 | 384 / 392 | 6 |

These cases use hidden size 5120, logical intermediate size 2304 padded to
2560, and FP4 block-512 weights. They cover static replica placement and flat
scale layout, exact load accumulation, uneven padding including an idle rank,
all-gather, topology preprocessing, push scheduling options and an optional
FP8 shared expert. An independent FP32 reference checks sampled output rows.
Cross-dispatch accuracy is numerical because the measured all-gather router
and generic push router use different lowerings; scheduling-only push options
retain bitwise checks.

Run the nightly kernel suite in the checkout's declared dependency environment:

```bash
python test/registered/tpu/kernels/moe/test_fused_ep_moe_push.py
```

Verify the actual Python executable, imported JAX/JAXLIB versions and loaded
libtpu before testing. The script supplies the upstream DP8 remote-DMA startup
flags before creating a JAX client; a launcher that initializes JAX earlier
must supply those flags itself. This coverage uses synthetic weights and does
not establish checkpoint loading, model E2E accuracy or multi-host support.

## Test cadence

Kernel checks live under `test/registered/tpu/kernels/moe/`; the standalone
shared-expert check lives under `test/registered/tpu/kernels/gemm/`. These
checks run in `nightly-tpu-4-v7`, with explicit chip requirements and `infra`
labels. They use synthetic weights and do not claim model accuracy coverage.
The push suite keeps one FP8 parity case, the public dispatch entry point,
and the full-expert V4.1 verify/draft/prefill cases; broader tuning sweeps are
outside this basic-function suite.
