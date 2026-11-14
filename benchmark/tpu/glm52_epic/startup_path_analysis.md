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
caused the archived lowering fatal. Actual fatal-stage logs are still needed.

Do not call `/health` success alone a working baseline. Disabling server warmup
would only isolate the later probe, not fix per-query lowering or establish
decode correctness. No default or shared runtime behavior is changed here.

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

## Next diagnostic boundary

Before another full-checkpoint run, isolate sparse MLA at the kernel interface:
plain DSA kpool=1, prefill below/at q-block threshold, and decode separately.
Keep the original per-query path as a required baseline criterion. If an
explicit skip-server-warmup diagnostic is used, label it as isolation only and
require an actual generated output plus a subsequent decode request; do not
report readiness as model success.
