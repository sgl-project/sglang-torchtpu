# GLM-5.2 native Torch MTP baseline

This harness targets 8 physical v7x chips, two hosts and 16 visible devices.
It preserves the existing Torch W8A8 kernels, ordinary `pallas_tpu` MoE,
FP8 KV, native NEXTN (5 steps / 6 draft tokens), TP16 / attention-DP16 / EP16.
NEXTN resolves internally to EAGLE; that alias is checked before model loading.

The acceptance workloads are:

| Case | Concurrency | Cached prefix | Extend | Output |
| --- | ---: | ---: | ---: | ---: |
| Prefill | 16 | 49,152 per DP | 4,096 | 16 |
| Decode | 64 | 89,984 | 16 | 1,024 |

The global chunk budget is 32,768 and resolves to 2,048 per attention-DP
worker. Local compile BS buckets are 1, 2, 3, 4; speculative verification
expands each request to six rows. The context ceiling is 92,160. KV allocation
is automatic with `mem_fraction_static=0.85`; its actual capacity is recorded.

`prepare.py` reads checkpoint metadata and representative safetensors headers,
preserves quantization exclusions, and adds the channel scale declaration.
It verifies FP8 channel-scale shapes and saves `checkpoint-audit.json` plus
`server.json`. `draft_view.py` builds a filtered index and symlinks to the five
original shards containing layer 78. The native draft loader consumes the same
1,569 MTP tensors and shares the target embedding/head; no weights are converted.
This avoids a second scan of all 282 target shards while retaining native MTP. `validate.py` checks native argument resolution before weights
are loaded. `session.py` resolves host addresses, uses the weekend reference's
physical-host preparation, and supervises the native launchers and client.

The client uses the checkpoint's chat template for routed short requests.
Natural source-code input IDs and file hashes are frozen before timing.
Decode chooses independent prefixes when the observed per-DP KV capacity
covers four 90K requests, otherwise shared prefixes under the user's explicit
authorization. `prefix-decision.json` identifies the choice. Shared-prefix
results are a separate workload and cannot establish independent-prefix
capacity. Both cases have a warmup and three measured rounds with disjoint
suffixes. Per-request streams, cache hits, routing receipts, acceptance counts,
and load snapshots are retained. C64 decode requires a common full-batch window.
HTTP TTFT and first-to-last-output TPOT are client timings, not TPU step times.

The paused native Torch scheduler does not publish fresh load snapshots.
The client therefore waits for HTTP SSE ingress headers before resuming and
checks actual batch size separately during execution. Ingress acknowledgements
alone do not prove scheduler admission or concurrency.

Use `benchmark/tpu/launch_server.py` through this harness on both hosts.
Falcon manifests must pin the exact Git commit and provide the model, private
TorchTPU wheel and prepared wheelhouse as Falcon mounts. The executable is
`bash benchmark/tpu/glm52_baseline/falcon_runner.sh`. Use one bounded allocation,
retain failures, and set the parent experiment exit code from the actual
application result. A holding command does not establish an E2E pass.

Put the venv, pip temporary files and compiler caches on appropriately sized
storage. `/tmp/tpu_logs` was only a 1 GiB tmpfs in the November 16 allocation;
placing the venv there failed before model loading. The bootstrap uses
`/tmp/glm52-baseline-runtime/venv` and an absolute Python executable throughout.
`runtime_probe/sitecustomize.py` records serving-process imports and libraries;
a separate successful probe is not a substitute.

The `prepare_host_checkpoint.py`, `plan_expert_reads.py`, and
`verify_expert_shard.py` helpers reuse the successful weekend implementation.
Their CPU shard oracle establishes exact selected weight/scale bytes, not
model correctness, numerical equivalence, or device performance.

See the task execution report for verified results and incomplete validation.
No model performance number is claimed by committing this harness.
