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
checks actual batch size separately during execution. Tokenizer imports, loading,
and long-input encoding run in a thread so the HTTP event loop can process
keep-alive closure while CPU work is in progress. Ingress acknowledgements
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

The pinned Falcon manifest also mounts the previous run's compilation cache.
`GLM52_COMPILE_CACHE_ROOT` selects that mount; omit it to keep a new cache under
the current artifact. The cache is an optimization and does not replace warmup
or serving-time compile checks.

## Complete allocation and recovery

The Falcon entry is `allocation_runner.sh`, which starts the standard-library
allocation supervisor before installing the runtime. Its 6,900-second bound
covers setup, all model work, client recovery and cleanup. The application runs
under `application-attempts/1/`; both ranks must report an actual successful
application result. A zero process exit without that result retains an
`ATTENTION` state, rather than establishing a pass.

The native Git entry hashes the actual benchmark `.py` and `.sh` source tree;
there is no dependency on a manually uploaded tar. Both hosts record and compare
that identity before model loading. Serving processes and clients record the
same source hash, their exact interpreter and the pinned Git revision.

A client failure leaves the already compiled server running. Its log,
measurements and `session-*/client-attention.json` explain the failure. After a
confirmed client fix, write `session-*/retry-client.json` with a new `id` to run
the full client again in `measurements/attempt-2/`. This repeats all workload
checks without restarting the model. Success requires six measured round
summaries and the 16 routed smoke receipts, not just exit code zero.

Setup or model failures retain the allocation for explicit recovery. After
fixing the recorded cause, place `allocation/retry.json` containing a larger
`generation` and the exact checked-out `source_commit`; both hosts then run the
application again on those same machines. There are no automatic retries.
The deadline remains fixed across recovery attempts. A deadline exits 124.

`inspect_run.py` is a read-only, standard-library inspector for use through
`falcon exp exec`. It reads the current allocation state, bounded log tails,
client attention state and measurement summaries, and also writes
`/tmp/glm52-inspection-rank-<rank>.json` for `falcon exp cp`.

## CPU preflight before requesting TPU resources

Use a local Python with `aiohttp` and `numpy`:

```sh
python -m benchmark.tpu.glm52_baseline.preflight_allocation
python -m benchmark.tpu.glm52_baseline.preflight --output /tmp/glm52-preflight
```

The first checks retained setup failures, explicit recovery, bounded expiry and
false-success prevention. The second exercises the real two-rank session and
full client against a local HTTP fixture: source-only startup, metadata and
host reads, one injected client failure, recovery without an engine restart,
16 routed smoke requests, both warmups, and all six measured rounds (368 total
requests). The fixture includes native plain-text cache-control responses and
short keep-alive expiry. File writes, tokenization and summary work run outside
the client event loop; pooled connections expire after one second, below the
native server's five-second timeout. Active batches are sampled every 20 ms.

These are control-flow and HTTP-contract checks. They provide no evidence of
TPU compilation, model accuracy, cache residency or measured device performance.

## GLM-5.2 port candidate runs (2026-11-16)

This copy comes from frozen baseline `811865ca48ca23aa14dc74c9dbebaf5b380279dc`.
The MoE and TopK ports are independent children of PR #2198 head
`ffd83db8f6168e0752e0769f8947cffb308e724d`. Each candidate runs the same
native MTP workloads once (one warmup plus three formal rounds per workload).
The user requested historical comparison with `exp-6mtkc6t2qi`, without a new
baseline A/B allocation. Cross-run deltas are descriptive and cannot establish
isolated kernel causality.

Set the candidate's plugin environment switch before starting the allocation;
it is inherited by both hosts, argument validation, and actual model processes.
The runtime probe records these switches. Checkpoint config and index hashes
are asserted during setup. Compilation caches use a namespace derived from the
exact source, plugin switches, runtime versions, and compiler environment;
candidates do not reuse the old baseline compiled programs.

The main agent owns one prod1 8-chip slot. Submit a second E2E only after the
first experiment is terminal; kernel development proceeds on prod2 4-chip
allocations. Keep the 6,900-second allocation bound below prod1's two-hour
runtime policy, inspect failures promptly, and release this task's failed
allocation if no concrete recovery is underway. Never stop unrelated runs.
