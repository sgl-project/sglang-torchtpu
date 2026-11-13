# GLM5.2 epic benchmark evidence

These files are experimental scaffolding, not a declaration of model/topology support.
Do not use the candidate JSON as a known-good performance baseline until the gates below pass.
No Flash recipe is imported or modified.

## JAX historical reference

`jax_reference_8chip.yaml` was exported from Falcon `exp-s31bemg2me`, retaining
source SHA, credential name, runtime pins, input and model mounts. It restores two
non-secret numeric envs redacted by export from the pinned chunk2k environment
(max prefill tokens and compile token bucket both 32768). The runner is bounded to
6600 seconds, below the selected cluster's two-hour running limit.

Submit only once and retain the returned ID; don't submit another copy while one
is queued. Current reproduction: `exp-266i34jcm1`.

## Metadata preflight

```sh
python3 benchmark/tpu/glm52_epic/inspect_checkpoint.py \
  /models/GLM5.2-fp8-channel-wise --output /tmp/glm52-checkpoint.json
```

Reads config, index and all shard headers only; outputs per-channel/scalar/other
scale classification and hashes. `UNVERIFIED` in the output is intentional: a
valid safetensors schema does not prove the Torch loader can consume it. The
private converter writes `weight_scale_inv[out,1]` with block size null; the main
non-block FP8 loader allocates scalar/shard scales. Do not silently substitute a
block checkpoint and call it the same numerical model.

## Torch candidate: NOT VALIDATED

`torch_prefill_candidate.json` proposes TP16 / attention-DP16 / EP16 on two hosts,
matching private's intended attention tensor axis1. `dp_size=1` is one complete
model, not a translation of JAX DP16. Native communication, loader ABI, group
constraints and per-rank compile buckets still require validation. In particular,
JAX global32768/local2048 compile semantics may not match Torch's request/global
buckets; inspect actual scheduler/compiler shapes before performance comparison.

The private manifest uses global max_prefill32768, page64, context135168, C32,
chunk2048 and FP8 cache; this JSON records those candidate settings without
claiming equivalent memory use or dispatch. Current Torch attention arithmetic
is not private FP8 Q/P. No private HLO pass or Flash-specific optimization flag
is enabled here.

After checkpoint compatibility and native startup are resolved, run on each host
with explicit `NODE_RANK=0/1`, `RANK0_HOST`, and `DIST_INIT_ADDR`, using the current
native launcher's JSON config support:

```sh
python3 -m benchmark.tpu.launch_server \
  --config @benchmark/tpu/glm52_epic/torch_prefill_candidate.json --dry-run
```

Dry-run is the first check, NOT device execution or a support proof. Pin actual
Torch/torch_tpu/JAX/libtpu and image digest before execution. Capture startup,
rank mesh, loaded scale shapes, cache budget, compile shapes and active kernels.
Use identical fixed token inputs from the reference, not independently tokenized
random text. Isolate compile/warmup, steady prefill device time and client TTFT.

## Flash safety

Only GLM5.2-specific files added so far; no production/shared path has changed.
Any later shared scale loader, pool, attention, MoE or compilation changes need
separate commits plus relevant Flash plain/MTP/megakernel regression evidence.
