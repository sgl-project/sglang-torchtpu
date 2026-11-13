# GLM-5.2 W-P measurement protocol (source-grounded draft)

## Status and scope

This is an executable-method plan, **not a performance result or a claim that
Torch already implements the matched measurement**. No accelerator/remote run,
server launch, dependency installation, production modification, or commit was
performed to prepare this document. Missing benchmark instrumentation is marked
below; do not substitute HTTP TTFT for an unavailable internal measurement.

Source audit:

- JAX reference SHA: `3695225808f0293cae422ea7adf55eddacadd43d`.
  The local JAX checkout was at `b3e3e16b2d3f0281a9e23f547cd4382f4a166eb6`, so
  the JAX evidence below was extracted with `git show <reference>:<path>`, not
  inferred from its working tree.
- Torch checkout inspected at `4854af7ab1d77f0b65e67a78190b734abc006e29`.
- Actual external [pinned runner](</tmp/glm52-night-pinned-runner.sh>) SHA256:
  `df43d50604405881e641b844dd54df2dd0f2a925fdf1982ff1fb7738ad56f49b`.
- Actual external [pinned environment](</tmp/glm52-night-pinned-env.sh>) SHA256:
  `a0bf23f164161fc7198c0b2547b8b8cdd4133f3846d907f0d34d2cacb3bb114a`.

## 1. Define the quantity before comparing numbers

For this chunked W-P workload the actual JAX primary JSON field is
`steady_chunk_extend_ms`. It is the **median interval between consecutive
host-observed prefill-result completions**, filtered for full-DP interior chunks.
It is neither a bare model-forward duration nor device-kernel elapsed time.
The workload is driven over HTTP; the primary timing observations are made
inside the engine and parsed from its log. Calling it an entirely non-HTTP
private microbenchmark would therefore also be inaccurate.

Exact command chain at the pinned SHA:

1. The external runner selects `benchmark/glm52/delivery/serve/channelwise_8chip.sh`
   and, in benchmark mode, `benchmark/glm52/delivery/benchmark/run_8chip.sh`.
2. [run_8chip.sh](<https://github.com/sgl-project/sglang-jax/blob/3695225808f0293cae422ea7adf55eddacadd43d/benchmark/glm52/delivery/benchmark/run_8chip.sh>)
   executes [benchmark/common.sh](<https://github.com/sgl-project/sglang-jax/blob/3695225808f0293cae422ea7adf55eddacadd43d/benchmark/glm52/delivery/benchmark/common.sh#L68-L133>).
3. That invokes [bench_dsa_cache_hit.py](<https://github.com/sgl-project/sglang-jax/blob/3695225808f0293cae422ea7adf55eddacadd43d/benchmark/glm52/bench_dsa_cache_hit.py>).
   `_steady_chunk_extend_metrics`, lines 1063–1119, defines the statistic;
   lines 1836–1853 select it as primary when `--chunked-extend` is set.

| Quantity | Actual boundary | Comparison rule |
| --- | --- | --- |
| `steady_chunk_extend_ms` | Consecutive qualified `completed_ns` differences / 1e6, median across those intervals | Compare only with identical completion boundary and filtering on Torch |
| `ttft_p50_s` | By default, client timestamp immediately before scheduler resume control request through first token SSE line | Includes all extension chunks and resume/control/output overhead; **not step latency** |
| `http_ttft_p50_s` | HTTP submission through first token SSE line | Includes admission waiting; not interchangeable with resume-origin TTFT |
| `wall_s`, `http_wall_s`, throughput, TPOT | Whole workload/client or output-window quantities, as separately emitted | Preserve their definitions; none is the W-P step median |
| Device/profile span | XPlane or another separately defined synchronized device measurement | Report separately, not as a renamed host completion interval |

Never divide TTFT by six, C32, DP16, or token count and label the result measured
step latency. Never compare a Torch client TTFT/TPOT or static forward time with
JAX `steady_chunk_extend_ms` as a speedup.

## 2. Fixed workload and data contract

The external environment overrides the 8-chip wrapper's default shared prefix:

- Eight physical chips, two serving hosts, 16 logical JAX devices; TP16/DP16/EP16.
- C32, two requests per attention-DP shard; **unique** prefixes.
- Prefix **49152**, extension **12288**, total input **61440** tokens/request.
- Local per-DP chunk **2048**, global prefill budget **32768** (= 16 × 2048).
  Precompile token padding 32768, batch padding 32, context 135168, page size 64.
- Output **16** tokens, no MTP (`GLM52_ENABLE_MTP=0`, no NEXTN arguments).
- Channelwise checkpoint, BF16 model dtype, FP8 E4M3 KV cache; memory fraction
  default in the external environment is 0.93, HLO overlap default is enabled.
- Sampling is native token-ID generation, temperature zero and ignore-EOS (see
  `_run_native_batch` in the pinned benchmark); seed 3 in wrapper and server.

The frozen [dataset manifest](<https://github.com/sgl-project/sglang-jax/blob/3695225808f0293cae422ea7adf55eddacadd43d/benchmark/glm52/datasets/longbench-v2/longbench-code-c64-p49152-e12288.manifest.json#L1-L31>)
records LongBench-v2, domain `Code Repository Understanding`, 64 windows of
61440 tokens, derived as leading contiguous slices of a larger frozen matrix.
Its NPY SHA256 is:

```text
9a393227fb7104feef12440dba1d104dc1c2c6a5a54fc956d9225000b1e25c53
```

`_make_longbench_inputs` (lines 297–395) checks matrix shape, manifest lengths,
and file hash; selects **rows 0–31 in stored order**, not randomly sampled rows;
uses each row's first 49152 tokens as its prefix and remainder as extension.
It does not tokenize live text or apply a chat template. Use those same integer
IDs on Torch. Preserve the adjacent manifest and hashes; do not regenerate
from current upstream text/tokenizers and call it identical data.

Warm path: flush cache and require idle; warm all 32 unique prefixes in stored
order with a deliberately different branch token appended and one generated
output token. This establishes reusable prefix pages without warming the
measured extension branch. With unique mode there is one warm batch of 32;
round-robin order is essential to matching warm/measured cache placement.
Decode warmup is **off by default** (`--decode-warmup-output-len=0`).
Warm prefix work is outside measured log offset, but this is **not proof every
measured shape is already compiled**. Verify no compilation/cache miss distorts
qualified intervals; report cold/compile evidence separately rather than quietly
changing the reference warmup protocol.

## 3. Exact JAX completion event and estimator

[Scheduler.run_batch](<https://github.com/sgl-project/sglang-jax/blob/3695225808f0293cae422ea7adf55eddacadd43d/python/sgl_jax/srt/managers/scheduler.py#L2515-L2527>)
increments `forward_ct` and snapshots request metadata **before overlap can
mutate request prefix lengths**. For each request it records `rid`, `dp`,
`prefix = len(req.fill_ids) - int(req.extend_input_len)`, and
`tokens = int(req.extend_input_len)`. The result carries the batch ID,
`forward_step=self.forward_ct`, and this snapshot.

[process_batch_result_prefill](<https://github.com/sgl-project/sglang-jax/blob/3695225808f0293cae422ea7adf55eddacadd43d/python/sgl_jax/srt/managers/scheduler_output_processor_mixin.py#L146-L201>)
resolves the prior result using `tp_worker.resolve_last_batch_result(launch_done)`
in ordinary overlap mode, then immediately takes `time.perf_counter_ns()` before
subsequent logprob/request processing. The source explicitly forbids adding
`block_until_ready` at this point because it changes overlap scheduling.
It emits an ordinary logger prefix followed by this exact marker and compact JSON:

```text
PREFILL_STEP_COMPLETED {"bid":123,"forward_step":456,"completed_ns":789000000,"requests":[{"rid":"<namespace>-cache-hit-extend-decode-0","dp":0,"prefix":51200,"tokens":2048}]}
```

The example shows **one metadata item only for format**; a qualifying actual
event must contain 16 distinct RIDs, one on every DP shard 0–15. Parse the JSON
after the marker, not the wall-clock logger prefix. `completed_ns` is a
process-local monotonic clock; do not subtract clocks from different hosts or
ranks, or merge duplicate rank logs into a single timeline.

The pinned estimator scans only bytes after the measured log offset:

1. A full event has exactly 16 unique requests, DP set `{0,...,15}`, matching
   measured namespace prefix, and every request has 2048 new tokens.
2. The current event is interior iff every request has `49152 < prefix`,
   `prefix + 2048 < 61440`, and `(prefix - 49152) % 2048 == 0`.
3. There must be a preceding full event with identical RID set, same DP mapping,
   `forward_step` exactly one smaller, and every prefix exactly 2048 smaller.
4. Append `(current.completed_ns - previous.completed_ns)/1e6`; reject nonpositive
   values. A nonfull event resets the predecessor. Every full event, including
   the first/final chunk, can become a predecessor.
5. Return `statistics.median(durations)`, sample list, count, p95 and status.
   No samples means `null`/`no_qualifying_completion_intervals`, not zero.

**Important literal-code nuance:** the helper's docstring says discard first and
final chunks. Only the **current** event must be interior, so an interval from
completion of chunk 1 to completion of chunk 2 is included. For a completely
regular six-chunk request group the candidates are 1→2, 2→3, 3→4, 4→5, not 5→6.
C32 has two requests per shard, but do not hard-code a sample count: mixed,
interleaved or partial batches are filtered by the actual rules.

The chunked log validator additionally accounts for extension tokens, cached
prefixes and full-concurrency decode. Save its evidence and require successful
16-token completions and cache hit ≥49088 for every request. A median without
those workload gates is not an accepted matched run.

## 4. Reproduce JAX using existing entry points (future accelerator execution)

Do not run the external runner from its copied `/tmp` location: it sources a
relative parent helper and hardcodes `/workspace/sglang-jax`. On the intended
runtime preserve the original delivery layout, source exact files at the pin,
and separately pin the image/runtime and model/data mounts. The following is
an existing-command recipe, not an instruction to submit a remote run now.

On **both** hosts, using the runtime's own ordered rank/address mapping:

```bash
# cwd: a clean, installed JAX checkout at the exact reference SHA
 test "$(git rev-parse HEAD)" = 3695225808f0293cae422ea7adf55eddacadd43d
 source benchmark/glm52/delivery/chunk2k_8chip_env.sh
 export GLM52_LONGBENCH_PREFIX_FILE="$PWD/benchmark/glm52/datasets/longbench-v2/longbench-code-c64-p49152-e12288.npy"
 # WORLD=2, RANK=0 or 1 and MASTER_ADDR must already be supplied explicitly.
 WORLD=2 RANK="$RANK" MASTER_ADDR="$MASTER_ADDR" \
   benchmark/glm52/delivery/serve/channelwise_8chip.sh
```

First compare that checked-in environment with the supplied external snapshot;
explicitly retain its fixed values in section 2 and record all default overrides.
Alternatively reproduce the original runner in its expected container layout
with `SOURCE_COMMIT` pinned and `RUN_MODE=benchmark_profile`. It starts the same
server, waits for readiness, executes three benchmark repeats by default and
then a separate prefill profile. `RUN_MODE=benchmark` executes **one** benchmark
invocation, irrespective of `GLM52_BENCHMARK_REPEATS`.

On rank 0, in another shell with the same sourced environment:

```bash
# SERVER_LOG must name the actual readable rank-0 server log.
for run_id in 1 2 3; do
  QUANTIZATION=channelwise SERVER_LOG="$SERVER_LOG" \
    OUTPUT="/tmp/glm52-benchmark-run${run_id}.jsonl" \
    benchmark/glm52/delivery/benchmark/run_8chip.sh
done
```

This expands to the pinned Python benchmark with `--concurrency 32 --dp-size 16
--expected-requests-per-dp 2 --prefix-mode unique --prefix-len 49152
--extend-len 12288 --output-len 16 --random-seed 3 --cache-hit-tolerance 64
--kv-cache-dtype fp8_e4m3 --chunked-extend --input-mode longbench-code`, plus the
base URL, log, dataset, output and variant arguments. Admission barrier is on:
pause, queue one ordered native batch, verify exact waiting RIDs and no running
requests, then timestamp immediately before resume. No unrelated traffic.

Keep each run's median and all raw qualified intervals; report a median of the
three run medians only as a separate, explicitly named aggregation. The wrapper
does not compute that cross-run aggregation. Do not pool profile-enabled and
unprofiled samples. Profiles are diagnostics, not steady benchmark replacements.

## 5. Current Torch facilities and missing matched implementation

The [candidate config](<torch_prefill_candidate.json>) proposes attention-DP16
with TP16/EP16, `dp_size=1`. This is not JAX's identically named DP setting.
The [existing contract notes](<README.md#L68-L76>) document global chunk32768
resolving to local2048 and local compile bucket `[2048]`. Require actual scheduler
and compiled tensor shapes, not just matching argument names. The
[existing runner](<torch_8chip_runner.sh#L1-L3>) is a startup/one-token smoke,
**not a benchmark**.

Available existing helpers/endpoints:

- [HTTP server](<../../../python/sglang/srt/entrypoints/http_server.py>): native
  generation, server info, cache flush, pause/continue generation, start/stop
  profile. Presence of routes does not establish compatibility with JAX's
  server-info/admission/cache-report JSON schema.
- [one_batch_server](<../../../python/sglang/benchmark/one_batch_server.py#L1-L12>)
  and [serving benchmark](<../../../python/sglang/benchmark/serving.py>) provide
  HTTP-oriented measurements. They are not a ready-made port of the frozen
  unique-prefix warm/extend workload or its completion-interval filter.
- [one_batch latency timing](<../../../python/sglang/benchmark/one_batch.py#L802-L806>)
  synchronizes before/after `model_runner.extend(reqs)` for a static synthetic
  batch. It is an available internal timing harness, **not** the matched
  cache-hit overlapping W-P estimator.
- [recorded step times](<../../../python/sglang/srt/managers/scheduler_components/metrics_reporter.py#L909-L922>)
  (`SGLANG_RECORD_STEP_TIME`) are decode-log gap divided by decode log interval,
  grouped by running request count, exposed in internal state. They do not
  measure interior prefill chunk completions.
- [DeviceTimer](<../../../python/sglang/srt/utils/device_timer.py#L118-L133>)
  explicitly creates CUDA events. Do not claim this as validated TPU timing.
- [benchmark batch gate](<../../../sglang_tpu/benchmark_batch_gate.py#L130-L154>)
  measures synchronized decode and rejects attention-DP !=1; it cannot be used
  unchanged for this attention-DP16 prefill workload.

### Benchmark-only internal completion observation plan — NOT IMPLEMENTED HERE

The closest existing Torch completion boundary is
[BatchResultProcessor.process_batch_result_prefill](<../../../python/sglang/srt/managers/scheduler_components/batch_result_processor.py#L261-L299>):
`result.copy_done.synchronize()` at line 272, followed by host token materialization
at line 297. [Scheduler.process_batch_result](<../../../python/sglang/srt/managers/scheduler.py#L5196-L5227>)
invokes that processor and finishes with a completed-inference health observation;
that later boundary includes more host result processing than the JAX timestamp.

For an opt-in **benchmark-side** harness, snapshot request RID/attention-DP,
logical prefix length, actual extension token count, forward ordinal and batch
identity before launch; observe the return of the **existing** result-completion
wait without introducing a new wait, then write the event schema in section 3.
Do not read device scalars to build metadata; use existing CPU batch metadata.
Do not time only entry/exit of an asynchronous forward enqueue. Do not timestamp
processor entry (before completion). A wrapper of processor exit is possible
without copying its body, but is a **different, later boundary** and must be named
as such until the added host work is reconciled on both frameworks.

A benchmark-only wrapper or plugin hook can avoid a new core TPU branch; exact
per-rank metadata and completion-wait observation still need implementation and
validation. No production hook is added by this document. In particular:

1. Determine how Torch's 16 per-attention-DP scheduler views map to one global
   event. Save rank-local logs; never subtract cross-process monotonic clocks.
   A rank-local2048 median is not automatically a full-DP32768 median.
2. Reproduce the frozen warm branch, cache residency, ordered C32 admission and
   16 output completions using current native endpoints. Audit actual cache
   hits and RIDs; adapt the client-side schema rather than changing production.
3. Emit the same immutable per-forward metadata and equivalent completion event.
   A parser can be tested independently on historical JAX logs first. Accept
   logger prefixes, reject malformed/duplicate metadata, retain namespace and
   rank identity, enforce consecutive steps and monotonic timestamps.
4. Apply the exact filter above, retain all samples and rejection reasons,
   and fail closed on empty samples, missing ranks, retractions or cache misses.
5. Validate on-device that instrumentation introduces no extra synchronization,
   compilation variants or material host overhead. Until then mark the matched
   Torch metric **UNIMPLEMENTED/UNVALIDATED**, not measured.

## 6. Acceptance evidence for any later comparison

Archive exact SHAs + dirty diff, script/config/data hashes, immutable image digest,
package versions, checkpoint/config/scale-layout evidence, physical topology and
logical mesh, all effective arguments, rank logs, admission/cache invariants,
compiled local/global shape evidence, warmup/compile logs, raw completion events,
selected interval list, per-run JSON and any separate profile artifacts.

Report four separate columns: **JAX host-completion step median**, **Torch
matched host-completion step median (or unavailable)**, client TTFT/E2E, and
separately defined device profile timing. Equal token shapes do not prove equal
FP8 attention arithmetic, quantization, compiler scheduling or memory behavior;
record those differences, and do not infer numerical or performance parity
from a dry run or CPU tests.
