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

`torch_prefill_candidate.json` explicitly opts into the experimental plugin loader via
`json_model_override_args.quantization_config.tpu_checkpoint_layout`; it leaves
the mounted model metadata unchanged. The loader implementation is not yet CPU
or TPU execution-validated. Ordinary FP8 and Flash omit this field and should
return the upstream configuration unchanged (CPU regression pending).

The JSON proposes TP16 / attention-DP16 / EP16 on two hosts,
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

The checked-in `torch_8chip_UNSUBMITTED.yaml` contains an older inline base64
runner payload and older source SHA; its header explicitly says it must not be
submitted unchanged. Before any future allocation, regenerate that payload from
the current reviewed `torch_8chip_runner.sh` and pin the current commit. Do not
treat the YAML's embedded runner hash as current evidence.

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

## Explicit unsupported loader/speculative modes

The private layout currently supports the target `GlmMoeDsaForCausalLM` only.
Same-checkpoint NEXTN/MTP rewrites the architecture to its draft class and is
intentionally rejected; do not advertise decode MTP acceptance before that
separate gate. `npcache` is rejected because its NumPy conversion cannot
serialize FP8 tensors. Use the default `auto` / `safetensors` loader for the
mounted checkpoint; `pt` requires an actual matching serialized PT checkpoint.
CPU offload and parameter-replacing alternate loaders remain excluded.

## Global vs local chunk contract

With attention-DP16, native argument resolution divides the supplied
`chunked_prefill_size` by 16. The candidate therefore supplies **32768 global**
for **2048 per rank**. Compile token buckets are per-rank literals: `[2048]`.
`max_prefill_tokens=32768` is separate and is not divided by that hook.
The earlier candidate supplied 2048 globally, which resolved to 128 locally but
padded compute to bucket2048; do not use that setup as the matched W-P baseline.
On-device effective buckets and workload equivalence still require verification.

## CPU test runtime selection

`SGLANG_PLATFORM=cpu` is NOT a valid platform plugin in this checkout. For CPU
contract tests, unset `SGLANG_PLATFORM` and set `SGLANG_USE_CPU_ENGINE=1`.
The public release image starts successfully on CPU, but differs from the
current pinned TPU dependency stack; record versions and do not use it for
on-device performance claims. The private CI base image was inaccessible to
this cluster (ImagePullBackOff), not a test result.

## Current TorchTPU failure diagnosis

The first short-input eager smoke reached model execution setup but failed in the
TorchTPU/XLA lowering of the sparse MLA path, before `/health` became ready. The
observed fatal signatures were `window_bounds` rank mismatch and an unsupported
`vmreg/vreg` to `U64` conversion; the later Gloo reset is secondary. Disabling
`enable_torch_compile` does not bypass this lowering because
`pallas_tpu_mla_sparse` still exports a Pallas custom JAX kernel.

The peer Kimi-K3.1 and GLM-5.3-Flash recipes enable the query-blocked sparse MLA
prefill path and set the libtpu SparseCore/fast-merge flags. The short GLM52
candidates now carry the same narrowly relevant settings. They also use
`page_size=128`, matching the peer DSA recipes and satisfying the q-block
plain-DSA kernel's hardware contract (`page_size % 128 == 0`); the earlier
`page_size=64` candidate could never enter that q-block path.
`SGLANG_TPU_ENABLE_QBLOCK_SPARSE_MLA_PREFILL=1`,
`SGLANG_TPU_ASYNC_COPY_WORKER=0`,
`xla_tpu_buffer_assignment_enable_fast_merge_fallback=false`, and
`xla_tpu_offload_gather_to_sparsecore=true`. This is a diagnostic candidate,
not a confirmed fix: q-block only changes extend prefill and decode still uses
the original per-query kernel. The runner's smoke request is now 512 tokens,
which is the q-block activation threshold; a four-token request would silently
exercise the old per-query fallback and would not test the intended candidate.
It must be validated on TPU before any claim of a runnable baseline.

The one-shape compile smoke is a separate failure. It exited 137 after host
memory peaked near 505 GB, without cgroup OOM events or the sparse-lowering
fatal. Falcon did not identify the first killed native child, so this remains a
high-confidence host-compiler-pressure hypothesis rather than a proven HBM OOM.

## Current exact-source smoke template

`torch_8chip_current_UNSUBMITTED.yaml` is the only current 8-chip smoke
starting point. It pins the current epic branch source commit, runs the checked-in runner
from that checkout instead of embedding a base64 copy, and fails before TPU
initialization when placeholders remain. It is intentionally not submitted:
replace the cluster, immutable image digest, and allocation hostnames only after
Falcon admission/capacity is revalidated. The pinned TorchTPU wheel filename/SHA
is now taken from the successful peer artifact provenance, but the image remains
intentionally unresolved because the peer only recorded a mutable tag.
The older
`torch_8chip_UNSUBMITTED.yaml` is historical evidence only because its inline
runner and source SHA are stale.

## Flash safety

Only GLM5.2-specific files added so far; no production/shared path has changed.
Any later shared scale loader, pool, attention, MoE or compilation changes need
separate commits plus relevant Flash plain/MTP/megakernel regression evidence.
