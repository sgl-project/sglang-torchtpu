# GLM52 startup-path audit

## Confirmed source facts

The 512-token runner probe is not necessarily the first model forward.
`python/sglang/srt/entrypoints/http_server.py:2523-2525` runs
`execute_warmup_func` unless `skip_server_warmup` is set. Its generation request
uses `max_new_tokens=8` (line 2376), and either three token IDs per DP rank
(lines 2383-2389) or the short text `The capital city of France is`
(lines 2431-2437). It sends that request before marking the server Up
(lines 2529-2533).

Consequently, enabling q-block and changing the runner request to 512 tokens
does not by itself route startup through q-block. The short startup prefill can
fall back to per-query sparse MLA; subsequent startup decode uses per-query
MLA regardless of q-block. This is a reachable pre-probe path, not proof that it
caused the archived lowering fatal's exact operation. Wider archived logs now
confirm that the fatal occurred during the pre-ready startup request window
(see timeline below), not after the runner's health-gated 512-token probe.

Do not call `/health` success alone a working baseline. Disabling server warmup
would only isolate the later probe, not fix per-query lowering or establish
decode correctness. No default or shared runtime behavior is changed here.

## Archived eager fatal timeline (Falcon exp-30eq5xauez)

Retrieved via `falcon exp logs exp-30eq5xauez --tail 2000 --output json`
(source=gcs, 1492 lines; default tail200 omitted the startup context).

- 05:37:53: schedulers report async result-copy worker enabled.
- 05:37:57: engine startup completes, Uvicorn starts.
- 05:37:58: `/model_info` returns 200; `/health` returns 503. Several DP ranks
  report prefill inputs uploaded at live length, then logits processing begins.
- 05:37:59–05:38:03: `/health` remains 503.
- 05:38:03: scheduler creates copy_done Event, followed by vmreg/vreg-to-U64
  and window_bounds rank fatal; result-copy worker event.record warnings also
  occur near the crash.

This narrows failure to startup request processing before readiness. It does
not attribute the fault to sparse gather, first prefill, decode, or event/copy
lowering: dispatch is asynchronous and neighboring log messages do not identify
the failing compiled operation. The corrected current candidate disables async
copy; a gather-only reproduction cannot test that alternative fault boundary.

## Async-copy toggle does not remove events

Traced `scheduler.py:4843-4874` and `result_copy_worker.py:77-121`.
The scheduler creates `copy_done = device_module.Event()` before selecting
worker versus inline copy. Disabling the async-copy worker only moves
`GenerationBatchResult.copy_to_cpu` back to the scheduler/copy-stream context;
that method still records copy_done (`managers/utils.py:221`). Therefore
`SGLANG_TPU_ASYNC_COPY_WORKER=0` does not remove event lowering or make the
request synchronous. It isolates worker/thread/stream scheduling only.
Do not describe this setting as an event-free diagnostic or treat another
fatal with it as proof against an event-related hypothesis. No synchronization
or event ownership changes are justified without an actual minimal failure.

## Peer topology correction

Falcon `exp get exp-bsvzoblh0d` reports one worker, device_count=8,
device_topo=2x2x1, cluster `gke-tpu-train-us-central1-2-prod`. Its successful
requests validate that peer workload/runtime only. Earlier narration calling
it a successful two-host GLM52-compatible topology check was incorrect.

## Existing kernel coverage is not the production signature

`test/registered/tpu/unit/test_mla_sparse_fused.py` already compares fused
sparse attention against dense MLA and checks cache bytes. It uses 8 heads,
top-k 256, page128, a 128-token extend bucket, and a 5-token decode bucket
(lines 60-68, 309-338). Those are useful correctness checks, but do not cover
the full production top-k 2048 or the short-smoke single-request decode shape.
The checkpoint config records 64 query heads and top-k 2048. Derive local heads
from the actual attention TP group, not global TP16: with attention-DP16,
attention TP is 1, so an 8-head kernel test cannot be assumed representative.

The backend passes `layer.tp_q_head_num` into the query reshape
(`sglang_tpu/attention_pallas_mla.py:328-336`). A minimal reproduction should
capture that value and actual padded T at the bridge, then compare:

| Stage | Required distinctions |
| --- | --- |
| Short extend | actual T below 512; top-k 2048; local attention heads |
| Q-block extend | T=512 and T=2048; page128; top-k 2048 |
| Decode | actual local batch/padding; top-k 2048; per-query path |
| Bridge | raw JAX vs TorchTPU bridge with identical shapes/dtypes |

Raw-JAX numerical success alone cannot establish that TorchTPU's bridge lowers
the same operation successfully. No hardware tests were executed locally;
JAX, Torch and pytest are absent from the local interpreter.

## Offline shape-selector result

Executed the actual `block_shape` function and its three constant assignments
via Python AST extraction, without importing JAX or stubbing kernel libraries.
For page128, both T=512 and T=2048 give:

| Heads | Query block | Pages/step | Head-query rows |
| --- | --- | --- | --- |
| 4 | 512 | 8 | 2048 |
| 8 | 256 | 8 | 2048 |
| 64 | 128 | 2 | 8192 |

The selector's documented `_MAX_ROWS=3072` only constrains its larger candidates
(512 and 256). Its unconditional fallback 128 exceeds that stated row budget
at 64 heads. This is a confirmed selector-budget discrepancy, **not a confirmed
TPU compilation failure**: `_MAX_ROWS` is a tuning budget, and the total VMEM
must be assessed separately. Do not blindly enable or disable the shared kernel
based on this calculation. It makes the 64-head production-signature compile
probe important before a full-checkpoint retry. Also corrects the earlier
unexecuted assumption that 4 heads would select `(512, 2)`; actual code returns
`(512, 8)`.

## VMEM estimate, not an allocation report

Source buffer shapes (`mla_sparse_qblock_tokens.py:393-419`) at 64 heads,
QB128, page128, pages_per_step2 give 44.375 MiB of logical bytes for query,
rope query, output, accumulator, qk, m/l and the two cache rings. If each
`[heads*QB,1]` m/l buffer occupies 128 lanes, this subtotal becomes 52.3125 MiB.
This excludes bitmap inputs, compiler temporaries, score/probability liveness,
and possible double buffering/aliasing. The kernel requests 64 MiB VMEM
(lines 423-425). Therefore the row-budget discrepancy alone does not prove
VMEM OOM, but 64 heads has substantially less headroom than the 8-head case
(12.265625 MiB logical / 14.25 MiB with that m/l padding assumption).
Only compiler allocation evidence can settle the actual peak. Do not compare
this on-chip estimate to the earlier 96 GiB HBM allocation failure: those are
different memory spaces and different candidates.

## Smoke acceptance and exit137 observability corrections

Short smoke now requests two generated tokens with ignore_eos, checks two
completion tokens and two integer output IDs. One generated token can come
entirely from prefill logits and is not evidence of a decode forward. Default
long-candidate probing remains unchanged. This is an acceptance fix, not a
kernel fix or evidence that the corrected probe has run on TPU.

Archived cgroup sampling peaks were 468.99 GiB on host0 and 470.63 GiB on host1;
listed process RSS sums at those samples were only 53.76 and 55.41 GiB.
The sampler lists processes above 100000 KiB and RSS can double-count shared
pages, so those sums are not complete unique process memory. Nevertheless,
cgroup usage must not be equated with compiler RSS. Both sampled OOM counters
were zero. The sampler already saved memory.stat and host meminfo locally but
excluded them from stdout; it now mirrors both to durable logs to distinguish
anon/file/shmem and host MemAvailable after a kill. Historical cause of exit137
remains unknown.

## Lowering boundary traced through the actual bridge

`attention_pallas_mla_sparse.py:116-178` wraps the JAX step with
`torch_tpu._internal.pallas.custom_jax_kernel`, donates both cache inputs and
aliases outputs 1/2. The plain-DSA q-block op calls
`sparse_mla_prefill_fused_tokens`; its fallback transposes queries to token-major
and calls the same per-query fused function (fused source lines 338-353).

In the default tensorcore cache layout the per-query attention reaches
`mla_sparse/kernel.py:1334-1346`, which calls `dsa_gather.dsa_gather` and then
runs MLA on gathered rows. That gather builds a SparseCore `pl.kernel` with
`VectorSubcoreMesh`, `use_tc_tiling_on_sc=True`, and
`needs_layout_passes=True` (`dsa_gather.py:361-396`). This is a concrete
SC/layout compilation boundary to isolate for the archived vmreg/vreg-to-U64
fatal; it is not evidence identifying which operation actually crashed.
The q-block path bypasses this gather for attention, but still performs cache
writes and `select_bits`, and does not change decode's per-query gather.

The default per-query gather chunk is 64 tokens. The main attention loop is a
Python `for` over ceil(T/64) (`kernel.py:898-922`), so T=2048 emits 32 chunk
bodies per layer, versus one for a tiny decode. This is a source-level reason
to compare compile time/memory across shapes, not proof that it caused exit137.
The 64-head q-block VMEM concern is a separate candidate risk and cannot explain
the earlier run before q-block was enabled.

## Prepared standalone gather reproduction

`repro_sparse_gather.py` isolates the existing SC gather without checkpoint
loading. It checks JAX/JAXLIB 0.11.2 and libtpu 0.0.50 before importing JAX,
requires TPU/SparseCore, emits lower/compile/execute markers, and checks raw
bytes against NumPy indexing. Use a fresh process per case:

```sh
python -m benchmark.tpu.glm52_epic.repro_sparse_gather --queries 1 --compile-only
python -m benchmark.tpu.glm52_epic.repro_sparse_gather --queries 1
python -m benchmark.tpu.glm52_epic.repro_sparse_gather --queries 64
python -m benchmark.tpu.glm52_epic.repro_sparse_gather --queries 4 --live-queries 1
```

Repeat page64 versus page128 only when the first case yields useful evidence.
Local CLI, invalid-argument and runtime-mismatch checks passed; independent
source review found no actionable shape/oracle defect. Subsequent hardware
results are recorded below; passing raw JAX alone does not validate the bridge
or serving baseline.

The optional `--backend torch` now isolates the bridge using the same inputs
and byte oracle. It selects CPU JAX before import, allocates inputs through
TorchTPU and wraps the first lazy `custom_jax_kernel` invocation in
`tpu_info_for_cpu_export`. Compile and execution cannot be separated by this
bridge API, so `--compile-only` is rejected in torch mode. Run in a separate
fresh process, after the matching raw-JAX case:

```sh
python -m benchmark.tpu.glm52_epic.repro_sparse_gather --queries 1 --backend torch
```

Independent source review found no actionable initialization, concrete JAX
annotation, shape or oracle defect in this bridge adapter. Experiment
`exp-8vb32ntf9t` (source `edcf9b47b998b37d4bf3d5dcdf12e80a2eb07a99`)
then passed bridge byte checks on both hosts for query1/page128/topk2048.
Raw JAX passed byte checks on rank1; rank0 compiled successfully but selected
a non-addressable global device and failed at NumPy host read. The diagnostic
now selects `jax.local_devices()[0]`. Logs are retained as
`gather_pair_r49_rank0.log` and `gather_pair_r49_rank1.log`.
The experiment's overall FAILED status is not a gather lowering failure.

Experiment `exp-ts3syd5ifc`, pinned to
`0ed40cc6912a173cc10eaf56724f2549d401150a`, tests query1 and query64 in each
backend with that selector fix. It SUCCEEDED: all eight host/backend/query
cases returned exit0 and byte-exact PASS (logs `gather_shapes_r54_rank0.txt`
and `gather_shapes_r54_rank1.txt`). This also hardware-validates local device
selection. Neither experiment exercises
cache donation, whole attention, model decode or repeated-call compile-cache
reuse. The next useful scope is the actual production custom op, whose two
cache inputs are donated/aliased and whose public return is only attention
output (the updated cache tensors must be checked through their input objects).

## Production sparse step isolation

`repro_sparse_step.py` calls the actual production factory with qblock=0,
H64/topk2048/page128. Each query selects only itself, so the output oracle is
its newly written FP8 latent, broadcast across heads. Both original whole cache
objects are checked byte-for-byte after each of two same-shape calls that
change the values. This covers production cache scatter, per-query attention
and retained Torch cache ownership; it does not test 2048 live keys, scoring
accuracy across keys, q-block prefill, indexer, model layers or serving.
Independent static review found no actionable metadata/layout/oracle issue.

Submitted `exp-boqyjlht0k` / `art-iyn4i6c37i`, source
`1ba0d458d5f4aee3169cf01afe58a2035b997fbf`, using the same 8-chip peer bootstrap
identity and no model mount. Query1 and query64 run in fresh processes, each
bounded to 600s. It SUCCEEDED: both hosts passed query1 and query64, each
with two value-changing calls, including output and whole-cache oracles. Logs
are `sparse_step_rank0.txt` and `sparse_step_rank1.txt`. This is single-live-key
evidence, not proof of multi-key scores or the full serving graph.

Full-checkpoint eager experiment `exp-2h7w8l2bga` / `art-vo3x7r5w0m` is now
submitted with source `ea78ecf2d621fb7f45c66c27b90cb8d49f620ddd`, the existing
hashed bootstrap identity, and `GLM52_BASELINE_MODE=eager`. It restores the model
mount, retains 8 physical chips / two hosts and requires 512 input tokens plus
2 generated IDs. Compile is disabled. Result pending; wait is bounded and no
duplicate allocation submitted. Do not confuse the diagnostic step passes with
this full-model acceptance criterion.

## Padding has two distinct contracts

The gather-only reproduction intentionally supplies nonnegative **physical**
indices. Production does not pass the indexer's `-1` entries directly to
`dsa_gather`: `prepare_gather_indices` (`kernel.py:972-1021`) counts valid
selections, remaps logical positions through the page table, and replaces `-1`
with scattered in-range cache reads. Attention masks those extra gathered keys.
Thus adding raw `-1` gather inputs would test an invalid boundary, not reproduce
production padding. A separate prepare-indices test is needed to isolate page
mapping if gather itself succeeds.

Also, `num_valid_indices` is a valid **query-row prefix rounded to the attention
kernel batch size**, multiplied by top-k (`kernel.py:1234-1247`), not the number
of nonpadding picks within a query. `--live-queries` in the reproduction models
that query-prefix boundary only. Its random physical indices do not model the
logical remap or prove that startup's page metadata is correct.

## Compile-smoke reconciliation check

Executed the real dependency-free `reconcile_tpu_batch_limits` function via
AST extraction on `torch_compile_smoke.json`: max_bs1, bs[1], global MRR16,
attention-DP16 and attention-TP1 resolve to `(1, [1], 16)`. Thus this concurrency
reconciliation does not silently grow the smoke decode grid back to bs32.
This is a local pure-function result, not evidence that a remote compile used
these settings. Runtime compile-policy EXTEND/MIXED and DECODE graph-variant
logs must still establish effective shapes and modes; one bucket does not
necessarily mean one executable.

## Validation boundary

`falcon workflow validate` validates the workflow DSL (`schema_version`,
`workflow`, `steps`), not a bare experiment manifest. Running it on the checked-in
bare smoke template returned input error / exit2 for those missing DSL fields;
this is not experiment admission or evidence the job is launchable. The submit
wrapper help exposes no dry-run flag. Do not submit placeholders just to test
schema validation. Local shell/JSON checks remain distinct from Falcon admission.

## Next diagnostic boundary

Before another full-checkpoint run, isolate sparse MLA at the kernel interface:
plain DSA kpool=1, prefill below/at q-block threshold, and decode separately.
Keep the original per-query path as a required baseline criterion. If an
explicit skip-server-warmup diagnostic is used, label it as isolation only and
require an actual generated output plus a subsequent decode request; do not
report readiness as model success.
