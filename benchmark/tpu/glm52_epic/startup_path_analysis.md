# GLM52 startup-path audit

## Latest full-checkpoint eager outcome: exp-2h7w8l2bga

FAILED (peer task exit1; schedulers SIGABRT -6), not exit137. Both hosts
finished loading; peer DP8-15 ended at 12:01:02-03 UTC after ~3026s, versus
primary DP0-7 ~2600s. Peer allocated tokens8192/KV0.43GB at 12:01:09.
Crucially, its global compile warmup explicitly submitted **0 calls** at
12:01:10. At 12:01:20 it emitted the same window_bounds rank0-vs1/base(2)
and vmreg/vreg-to-U64 native lowering fatals as the archived run. Thus this
reproduction does not require torch.compile graph warmup, large KV capacity,
or the asynchronous copy worker. It still requires attribution to an actual
operation; asynchronous Python broadcast/Event stacks are not that attribution.
Peer evidence: `eager_r59_peer_failure.txt`. Primary durable stdout currently
contains bootstrap provenance only, despite copied live log proving load end;
do not invent primary HTTP/prefill timing from the peer log.

Next bounded experiment: exp-pjqqwzqvwk / art-87lllasmfe, source
da842fd3a6d1b492c2a12a6c1ed1865be8866ac0, query4 head-major prefill fallback,
no checkpoint. SUCCEEDED: both hosts ran two output/cache-checked calls,
all four passed (prefill_fallback_rank0/1.txt). Each host used one local device;
this does not validate distributed idle/collectives or query-score numerics.

Follow-up exp-0vl9iuolg7 / art-sfn90qoppa, source e4ab0d1903: indexer query1
and4 x XLA/Pallas writer with ordinary serving segments also SUCCEEDED on both
hosts (eight membership/i32/padding passes; indexer_bridge_rank0/1.txt).
Thus neither narrow bridge reproduction triggers the full-model fatal. Scoring,
combined graphs, larger cache shapes, model ops and distributed scheduling remain
uncovered. No production kernel change is justified from these passing probes.

## Request-table writer candidate rejected as current-path reproduction

Independent review of the unsubmitted request-table draft found it directly
calls the device vectorized scatter with (5,4096), whereas current serving
capacity16 / attentionDP16 resolves to one request per worker, plus padding
row and extra context4 => (2,4100). More importantly the TPU request table is
host-resident by default. Actual peer fatal evidence goes beyond that default:
`eager_r59_peer_failure.txt:871` places a worker in allocation.py:188,
`out_cache_loc.cpu()`, guarded by req_to_token being CPU and locations non-CPU.
At least that worker demonstrably took the host-table path, not the device
repeat_interleave/index_put_ helper. This synchronous read can surface an earlier
asynchronous compiler failure; it does NOT identify D2H as its cause. Do not
launch the device-table draft as a current fatal gate; no hardware was spent.

## Instrumented compile-smoke live memory curve

The corrected model-mounted exp-9320106a77 (source 1e9935c863) is still
running. The runner redirects native stdout/stderr to `server.log`; absence
of compile markers in Falcon workload stdout alone does not establish its stage. The durable
sampler has now observed alternating host peaks: roughly 274 GiB on one host
and 250 GiB on the other, with anon only ~47.6/46.3 GiB, file cache
~225.2/202.9 GiB, kernel ~1.3 GiB, and cgroup `oom=0`, `oom_kill=0`.
A live `falcon exp exec --rank 0 -- tail` of the native log confirmed DP8–15
entered `Load weight begin` at 14:05:04Z; the tail ends in initialization
warnings with no load-complete/compile marker. This supports weight-loading
as the last logged stage, not proof of every rank's current stack. Cgroup
anon is not per-process/compiler RSS. These samples cannot explain historical
exit137 alone. Earlier progress replies mislabeled raw decimal billions of
bytes as GiB (e.g. 284309639168 bytes is 264.78 GiB, not 284.3 GiB);
the 274/250 GiB values above were explicitly divided by 2**30.

Host attribution correction: live `falcon exp cp --rank 0` of
`rank-0/host-memory.jsonl` identifies observer PID 605 / launcher 1480 as
Falcon host rank 0 (286.38 GiB at timestamp 1794666810423603038, events
all zero). Prior stdout contains interleaved observers 604 and 605; the
274.18 GiB sample belongs to observer 604, not confirmed rank 0. Prior
progress replies reversed this host label. At the earlier 274/250 GiB
snapshots, each observer retained 12–13 launcher descendants, with largest
surviving scheduler RSS ~6.9–7.0 GiB and HWM ~9 GiB. Host MemAvailable was
~857–861 GiB. This is positive process evidence, but neither a compiler
peak nor evidence excluding a future ancestor-cgroup/system OOM.

Live native rank-1 `server_args` was parsed independently: compile is enabled,
`torch_compile_max_bs=1`, `tpu_compile_bs=[1]`, `tpu_compile_tokens=[2048]`,
TP/attentionDP=16. Effective local `chunked_prefill_size=256` (requested
4096 across attentionDP16); do not confuse this with historical EXTEND bs32
or its six decode buckets. At 14:37:00Z, rank-1 current=351.47 GiB,
file=302.24 GiB, anon=47.67 GiB; largest tracked survivor RSS=7.58 GiB.
The earlier rank-0 snapshot at 14:33:30Z was current=286.38 GiB,
file=238.68 GiB, anon=46.30 GiB, largest survivor RSS=6.93 GiB. These
snapshots are not simultaneous. Both copied native logs still end after
`Load weight begin` with no completion/compile marker.
The experiment deadline is 16:01Z; do not interpret its running state as a
successful baseline until the server log and smoke result are present.

## Compile-concurrency scope audit

The live native args also confirm `tpu_compile_tracer=dynamo` and
`use_spmd_compile=False`. DecoderCompilePolicy's async warmup submission
requires `tracer == direct` (decoder_compile_policy.py:1765–1778); that branch
is not reachable for this run. Do not attribute a future peak here to the
plugin's direct-compile async queue or change its queue depth as a fix.
This does not prove serialized native XLA compilation: TorchTPU/Dynamo may
have internal threads/processes, and separate scheduler ranks can compile
concurrently. Attribution still requires sampled process identity/stacks and
compile-stage evidence. Current source alone does not establish historical
exit137's compiler mode.

## Loading-policy audit (no runtime change)

The effective native arguments have `load_format=auto`, extra config `{}`,
`weight_loader_disable_mmap=False`, `weight_loader_prefetch_checkpoints=False`,
and `weight_loader_drop_cache_after_load=False`. DefaultModelLoader selects
multi-threaded safetensors loading (loader.py:633–733), with default 8 workers
(loader.py:412). The iterator's sliding window bounds in-flight shard objects,
but `safe_open`/`get_tensor` is mmap-backed and no DONTNEED follows consumed
shards (weight_utils.py:1265–1339). Its documented RAM bound is not a bound
on accumulated cgroup page cache. With eight schedulers per host, reader
concurrency can also multiply; cache pages are not necessarily duplicated
per rank. The EP-selective read path requires disable_mmap, so the effective
mmap path does not use that optimization. These are code-path explanations
consistent with observed file-cache growth, not a demonstrated loading
bottleneck or a reason to flip mmap/drop-cache during this ongoing run.

Live rank-1 native progress is CR-delimited tqdm, not newline-delimited:
229/282 at 36:44 advanced to 239/282 at 38:18. At 14:43:30Z, its sampler
recorded current/peak=413.58 GiB, file=364.18 GiB, anon=47.69 GiB,
zero memory.events, and max tracked surviving HWM=9.19 GiB. This confirms
reader progress rather than a completely stuck loader; it does not prove
all ranks have the same shard position. A bounded log-tail helper must
recognize CR as well as LF or it can discard the entire tqdm tail.

## Instrumented compile-smoke outcome

exp-sx62ld4v1e (source 7830981b0e) failed before model loading: the bootstrap
had no `/models/GLM5.2-fp8-channel-wise` mount, so HuggingFace raised
`HFValidationError` followed by `OSError` for the local model path. It is not
an exit137 or compiler-memory sample. One host-memory sample did capture
cgroup `memory.current=5.64GiB`, `memory.peak=8.10GiB`, `oom_kill=0`; the
sample's launcher root list was empty because sampling starts before the native
launcher is spawned. Do not use this run to infer compile memory. The new
sampler is still valid for a run that reaches launcher startup, but a future
runner should record the launcher PID only after Popen and sample descendants
from the next interval.

## Local TPU distributed probe outcome

The corrected standalone local probe exp-p2624l9hla (source
3c5b0ee805) did not reach the Python boundary probe. Discovery validated one
host, eight localhost slice-builder addresses, WORLD_SIZE=8 and topology
2,2,1,2. However, all eight torchrun children blocked during
`torch_tpu::ProcessGroupTpu -> PjrtBackend::GetClient -> PJRT_Client_Create`.
The per-case 240s timeout sent SIGTERM; there were no BOUNDARY/STAGE/pass
markers. This is a TPU distributed/PJRT initialization hang under this
launcher setup, not evidence about all_reduce, sparse MLA, allocator transfer,
or the historical XLA lowering fatal. The earlier exp-a77tly0f0f and
exp-vdlzc92azo outer exit0 logs lacked child markers and are not acceptance
runs. Do not retry this same local launcher unchanged.

## Distributed stack interpretation: CPU control broadcast is not TPU HLO

The repeated request_receiver.py:200 -> broadcast_pyobj:2816 frames are the
control-request broadcast on tp_group.cpu_group (request_receiver.py:198-205).
The helper defaults force_cpu_device=True and creates a two-element int64
`sizes` tensor on CPU (common.py:2777,2786-2792,2815-2816). Its length2 is NOT
an identified match for the XLA descriptor base_bounds(2). A thread blocked in
this CPU broadcast can simply be waiting for a peer that has aborted; no device
collective attribution follows from this stack. A meaningful distributed probe
must explicitly distinguish Gloo coordination from actual tpu_dist operations.
Existing test_spmd_safe.py:50-60 supplies the supported local multi-device
launcher pattern via torch.tpu.distributed + torch.distributed.run; no replacement
multi-host bootstrap or arbitrary world-size assumptions should be invented.

## Copy controls and compiler-descriptor size interpretation

Corrected copy/Event experiment exp-rkxqx7sxep succeeded: both hosts reported
initial and effective DEFER_NEVER; all12 cases per host x2 iterations passed,
including int64 upload/argmax and copy-only/Event-only/copy+Event. Evidence is
result_copy_defer_never_rank0/1.txt. No target native fatal reproduced.

The fatal's `element_size_in_bytes:4096` belongs to an internal XLA window
descriptor (peer log723), not an established total user-tensor byte size.
512 int64 elements in copy controls are a size variation only; do NOT describe
them as a matched failing shape or use that field to identify an operand.
Matching the operation/producer boundary is stronger evidence than this guess.

Current allocator-transfer isolation exp-h3780e83wv / art-oejsgmh18d, source
39d65f571db9b9341ee124d3aacce1824113dcea, tests actual CPU page expansion and
int32 upload/blocking readback with page128, no model/distributed groups.
One15-minute bounded wait; hardware result pending.

## Allocator producer at the host-table sync boundary

TPU platform get_paged_allocator_cls returns TpuPagedTokenToKVPoolAllocator.
Its alloc_extend computes page-expanded indices on CPU using
alloc_extend_vectorized_cpu and returns one async H2D of **int32**, unless
return_host is requested (allocator_tpu.py:230-257). The source is not the
int64/device repeat_interleave scatter previously considered. In ordinary
alloc_for_extend with host table, last_loc is already CPU. The subsequent
out_cache_loc.cpu() can therefore be a plain int32 H2D->D2H roundtrip, or a
synchronization exposing unrelated in-flight work. No evidence yet identifies
a faulty page-expansion kernel on TPU: this producer does not launch one.
The copy/Event probe used int64, so it is not an exact allocator dtype match.
Do not launch an allocator kernel rewrite based on the native U64 diagnostic.

## Result-copy isolation execution-mode caveat

Full-model peer logs explicitly show `TorchTPU eager mode set to DEFER_NEVER`
on all eight workers. `patches._configure_eager_mode` sets that internal runtime
state during backend setup; importing TorchTPU alone is not evidence that the
same state is selected. The currently submitted result-copy experiment
exp-kxinhtzlc0 / art-mhc4o29set (source8b1aa30567) does not explicitly configure
or print execution_mode. Its results must be scoped to the installed runtime's
initial mode unless the logs establish otherwise. Do not silently interpret
passing copy/Event controls as matching serving DEFER_NEVER. A follow-up should
explicitly record/pin that mode before dispatch if this distinction matters.

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

## Attention-DP divides the short probe's chunk ceiling

`arg_groups/parallel_hook.py:312-327` divides global chunked_prefill_size by
attention-DP size. Both short configs specify 4096 / DP16, hence a per-worker
chunk ceiling of 256. The 512-token external probe therefore does NOT guarantee
a >=512-token q-block invocation; an eager scheduler can split it into two
256-token chunks, each taking the head-major wrapper's per-query fallback.
The runner's expected_attention_path field records intent only, not proof.
The active eager run remains useful for full-checkpoint generation/decode
acceptance, but must not be called q-block coverage without actual shape logs.
A later q-block-specific run needs a local chunk >=512 (global >=8192 at DP16)
and evidence of its actual invocation. Do not mutate the in-flight experiment.

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
duplicate allocation submitted. A read-only Falcon exec inspection confirmed
rank0 weight loading progressing to 56/282 shards after 9m25s (load began
11:10:37 UTC). This is loading progress, not ready/generation evidence. An
initial Python `-c` inspection command failed with exit2 through exec's command
quoting; the simple `tail` inspection succeeded without modifying the workload. Do not confuse the diagnostic step passes with
this full-model acceptance criterion.

## Production cache ownership matches the isolated step

Traced `_kv_cache_buffers` to `TpuSplitMLATokenToKVPool` allocations in
`kv_pool.py:1400-1417` and getters `1463-1470`. Each layer's NoPE, RoPE and
index cache is separately allocated, not a slice of one layer-stacked tensor.
`_zeros_kv_buffer` creates a genuine uint8 base allocation and asserts `_base
is None` / matching dtype (`58-70`). The direct getters return those objects.
Thus the isolated step's whole uint8 cache setup matches this production
boundary; source inspection does not support a current layer-view donation
bug. This does not establish runtime absence of buffer copies or reuse bugs.

## Loading progress is rank-gated, not a cross-host skew metric

`weight_utils.py:1286-1288` enables the multi-thread shard progress bar only
before distributed init or on global distributed rank0. Rank1 host logs contain
DP8-15 Load weight begin but no shard counter; this is expected logging policy,
not evidence its loader is stuck. The bar increments after yielding each shard's
tensors (`1322-1339`), not after all hosts finish that shard. Thus rank0's
147/282 snapshot is local consumer progress only and cannot establish global
checkpoint readiness or relative host speed.

## Indexer remains a separate untested startup boundary

The serving backend builds `build_dsa_index_topk_op` for plain DSA. This op
quantizes query/key, writes a third (index-key) cache, then invokes
`streamindex_topk`; the passing attention-step reproduction injects selections
and does not execute any of it. Existing `test_dsa_index_topk.py` is raw JAX,
H32/D128/page128 but topk256, not the production topk2048 bridge.

`repro_sparse_indexer.py` now prepares that bridge isolation using the actual factory, H32/D128,
query4, topk2048, one sequence with kvlen4 and page1. Its public contract gives
a cheap selection oracle: row j must contain each of 0..j exactly once in an
unspecified order, then only -1 (because all causal keys fit within topk).
The index cache has uint8[pages,32,4,256] at page128/D128; it is distinct from
the attention caches. This tests selection membership and lowering, not score
ordering or the quantization byte oracle. Do not attribute an indexer failure
to attention gather merely because both are Pallas custom ops. No new remote
allocation is warranted while the full-model run is progressing.
The explicit `--key-store xla|pallas` and `--segments all|serving` switches
separate writer and specialization boundaries. Ordinary serving mode here means
query1 decode / query4,64 mixed, matching MLA metadata (not speculative verify).
Independent review found no actionable metadata/shape/oracle issue; dtype i32
is also asserted. Imports still require the installed SGLang dependency stack
through the DSV4 package initializer; this is checkpoint-free, not dependency-free.
No hardware result is claimed for this diagnostic.

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

Executed `build_bucketers`' real body with the DP-adjusted max_t256, explicit
smoke token list[2048], bs[1], and dynamic-chunk expansion disabled: it retains
TokenBucketer[2048] and BatchSizeBucketer[1], so token next_bucket(256)=2048.
The function keeps the smallest configured bucket covering max_t rather than
clipping it to max_t. Consequently compile-smoke can still warm a T2048 q-block
shape despite the eager local chunk ceiling256. This is not a config error by
itself and does not establish actual graph routing; it explains why eager
success would not establish compile-smoke coverage or memory requirements.

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
