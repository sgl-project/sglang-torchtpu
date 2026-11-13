# Benchmark-only rank-local prefill observer

**Status: implemented and stdlib-selftested, NOT accelerator-validated and NOT a
matched JAX metric.** This observer is not installed in either existing smoke
runner or submitted smoke configuration. No production files, registered tests,
SGLang environment registry, or collective operations are changed.

Read the [measurement protocol](<../MEASUREMENT_PROTOCOL.md>) before interpreting
any output. In particular, this does not establish full-DP completion cadence,
compilation-free execution, identical workload admission/cache residency, or a
Torch/JAX speedup.

## Local selftest (no dependencies/server/device)

From the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 benchmark/tpu/glm52_epic/observer/selftest.py
```

The [selftest](<selftest.py>) calls the **real wrapper factories** against fake
external events and owners. It checks immutable pre-launch CPU metadata and
runtime attention-DP identity, expected forward ordinal, no launch-time wait,
query/attribute delegation, existing synchronize argument/return preservation,
one timestamp immediately after the first successful existing wait, restoration
on success/wait failure/host failure, missing wait/event unavailability, invalid
metadata, path exclusions, JSONL output, and active/inactive startup import hooks
in clean subprocesses. This standalone stdlib benchmark test intentionally does
not use the registered-test infrastructure or import SGLang/torch.

## Opt in only in a future authorized diagnostic launch

Do **not** add this to the submitted smoke yet. For an independently approved
future launch, set these benchmark-local variables on **every serving host**, in
the environment inherited by every scheduler worker:

```bash
# cwd: the pinned benchmark checkout, on each host
export GLM52_EPIC_OBSERVER_DIR="/absolute/artifact/path/torch-prefill-observer"
export PYTHONPATH="$PWD/benchmark/tpu/glm52_epic/observer${PYTHONPATH:+:$PYTHONPATH}"
# Invoke the separately approved, pinned server command here; none is started
# by this README or observer, and the existing smoke remains unchanged.
```

The output directory must be absolute and writable. Unset or empty
`GLM52_EPIC_OBSERVER_DIR` disables installation. Python must load `site`; `-S` or
an environment that strips `PYTHONPATH` prevents startup installation. Check for
an existing `sitecustomize` in the launch environment: this explicit path takes
precedence and does not chain an unrelated startup module. Do not accidentally
replace required deployment startup customization. The benchmark directory
contains a top-level `observer` module; avoid conflicting `observer` modules.

[sitecustomize](<sitecustomize.py>) installs a narrow import hook for the existing
`Scheduler.run_batch` and
`SchedulerBatchResultProcessor.process_batch_result_prefill`. It does not eagerly
import SGLang/torch. Installation is idempotent; already loaded target modules
are patched too. It makes no edits to their source files.

## Boundary and scope

The [observer](<observer.py>) supports only **PP=1, no speculative decoding/MTP,
normal generation, ordinary overlap**, disaggregation disabled, and exact
`EXTEND` mode. It intentionally excludes `MIXED`, decode, idle, prebuilt,
embedding, split-prefill, dLLM, MLX overlap and speculative paths. An excluded
path is passed through without observations; this is not a claim that every
possible runtime feature combination has been validated.

Before calling original `run_batch`, the wrapper creates frozen dataclass/tuple
metadata from existing Python-only `batch.prefix_lens`, `batch.extend_lens` and
request RIDs. These CPU lists are the actual prepared prefix/extension lengths;
it does not read tensor scalars or derive lengths from potentially mutated
requests later. Each request's `dp` is **runtime-context
`get_parallel().attn_dp_rank`**, not generic DP rank. It snapshots local batch
identity (`id(batch)`, not a globally unique ID), ranks and
`expected_forward_step = scheduler.forward_ct + 1`. After original returns, it
checks `batch.forward_iter` against that expected step and attaches the immutable
snapshot to the **result**, not to a mutable batch or global timestamp table.
Invalid metadata/ordinal is explicitly unavailable rather than silently repaired.

At prefill processor **entry**, a temporary `copy_done` proxy is installed. It
delegates existing `query`/other attributes and every `synchronize` call to the
original event. Immediately after the first successful existing synchronize
returns, it captures one `time.perf_counter_ns()` value. There are no new waits,
events, device reads, event queries, barriers or collectives. `copy_done` is
restored in `finally`, including when the original processor throws. JSON
serialization/file I/O occurs afterward, not at the measurement boundary. The
observer still adds CPU allocation/wrapper/I/O overhead; hardware validation is
required before even rank-local performance conclusions.

If `copy_done` is missing, the processor never waits, or synchronization fails,
the observer emits `status="unavailable"`, a reason, and `completed_ns=null`.
It does **not** substitute processor entry/exit or enqueue time. A later host
processing failure does not invalidate a successfully observed wait boundary;
the original exception continues to propagate. Sink write failures warn rather
than replace the original processor exception. Missing/failed records must be
treated as incomplete evidence, never as zero latency.

## Output contract

Every participating process/rank writes independently to
`<directory>/<hostname>.pid<PID>.jsonl`. There is no rank-zero gate, shared
cross-process file handle, rank aggregation or global timestamp store. Files
append; choose a fresh artifact directory per run and retain every rank file.
Do not assume an empty file set means a healthy measured run (it may indicate
unsupported paths, disabled startup injection or startup failure).

Each JSON line carries:

- `marker="TORCH_RANK_LOCAL_PREFILL_COMPLETION"`, `schema_version=1`;
- `scope="rank_local_diagnostic_not_matched_jax"`;
- `boundary="after_existing_copy_done_synchronize"`;
- `clock="process_local_perf_counter_ns"`, `hostname`, `pid`, `tp_rank`,
  `pp_rank`, `attn_dp_rank`, `bid`, `forward_step`;
- `requests=[{"rid":...,"dp":...,"prefix":...,"tokens":...}]`;
- `status`, `unavailable_reason`, and `completed_ns` (null when unavailable).

Never subtract monotonic timestamps across files/processes/hosts, merge rank
logs into one timeline, relabel this marker as JAX `PREFILL_STEP_COMPLETED`, or
feed it to the matched-JAX estimator as if one local event represented all 16
attention-DP shards. No matching/aggregation estimator is supplied here.
Archive all ranks and separately establish the measurement protocol's workload,
cache-hit, compilation, numerical and boundary-equivalence gates before any
later matched claim. CPU fake-event selftests establish wrapper behavior only;
TPU asynchronous event identity/delegation, overhead and completion semantics
remain **unvalidated**.
