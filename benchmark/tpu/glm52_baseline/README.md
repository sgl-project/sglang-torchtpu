# GLM-5.2 block-FP8 startup and serving recipe

This recipe loads the official block-FP8 checkpoint, converts it to channelwise
FP8 on TPU, and runs native NEXTN 5/1/6 on 8 physical v7x chips (two hosts,
TP/attention-DP/EP16). It does not produce an offline quantized checkpoint.

The startup settings are `SGLANG_PLATFORM=tpu`,
`--tpu-fp8-runtime channelwise`, `--weight-loader-disable-mmap`, and
`--tpu-compile-tracer direct`. The recipe also enables
`SGLANG_TPU_ENABLE_EXPERT_HOST_STAGING=1` to reuse the existing host assembly
and chunked expert upload path outside PP. This flag defaults off elsewhere.
`sglang_tpu.glm` registers exact GLM DSA classes
with the existing EP-local reader. Shared experts, EPLB and incompatible expert
layouts retain the reader's fallback checks. The current attention-DP worker
replicates its draft at EP1, so NEXTN correctly reads all experts.

Target warmups use the existing native asynchronous direct-compile path.
Draft compilation retains its existing synchronous path. SPMD sharing is not
enabled; this recipe has not verified cross-host GLM graph/layout equivalence.

`prepare.py` writes the full server recipe; `draft_view.py` creates an index and
symlinks for the original native MTP shards without modifying tensor bytes.
`session.py` performs explicit once-per-host pre-reading and runs the serving
client. Pre-reading must be counted separately from model loading. The reader's
`readinto_bytes` reports application payload reads (including merged gaps), not
physical GCS traffic. Filesystem caching and read-ahead can change actual I/O.

`bootstrap.sh` pins the interpreter and dependencies from `pyproject_tpu.toml`.
`runtime_probe/sitecustomize.py` validates the actual serving processes' versions
and records imported paths and compiler settings. Falcon manifests must pin the
source SHA and provide the model and wheel mounts. Do not set the source to an
unpinned moving branch for a reported run.

## Initial validated run (without host staging): 2026-11-17

- Main base: `e1c3e4369b5928de0ea4a073a152c375ec9a112b`.
- Tested source: `d7605d6d315c74b9abbf786b5d5efdc6e3a9299b`.
- Experiment: `exp-tl5o6jxury`, artifact `art-gc0urm2wq8`, **SUCCEEDED**.
- Analysis: `an-5ebgn4k01d`, **SUCCEEDED**.
- Successful application attempt: 2; client attempt: 1.
- Checkpoint: `gs://inference-model-storage-poc-tpu-hns/GLM-5.2-FP8`.
- Independent compile-cache namespace; no previous experiment cache mounted.

| Phase | Observed time |
| --- | --- |
| Host target pre-read payload | 60.84–66.26 s per host |
| Host draft pre-read payload | 2.61–5.23 s per host |
| Target model load, including conversion | 60.56–78.47 s across 16 ranks |
| NEXTN model load, including conversion | 221.42–223.98 s across 16 ranks |
| Global compile and warmup | 312.48 s wall time |
| Launcher first timestamped runtime log to ready | Approximately 11 min 36 s |

The last interval is 12:33:38–12:45:14 UTC (20:33:38–20:45:14 Asia/Shanghai).
It excludes environment installation and checkpoint pre-reading. It is a log
interval, not an instrumented process-launch timer. Initial environment setup
was reused after correcting a test-platform selector; do not present this run
as a clean measurement of dependency-install time or add retry time to normal
startup. The global warmup timer includes asynchronous overlap and execution;
summing individual compile durations would double-count concurrent work.

Each target rank retained 9,189 tensors and skipped 109,440 routed-expert
tensors. Retained payload was 66,912,095,936 bytes per rank, compared with
755,617,140,416 indexed checkpoint payload bytes. This includes common weights;
it is not simply total checkpoint size divided by EP size.

Both hosts passed all 19 reader unit tests. Six measured serving rounds passed:
C16 uses 49,152 cached + 4,096 extend tokens, output16; C64 uses 90,000 input
with shared prefix, output1024, sustained four active requests on each of 16 DP
ranks. All 16 DP smoke checks and MTP verify checks passed.

- C16 median TTFT: 6,117.21 ms.
- C64 median client TPOT: 16.306 ms/token.
- C64 median steady throughput: 4,085.57 token/s, 510.70 token/s/chip.

No new direct model specialization was logged after ready. Some small eager
XLA programs still compiled or loaded from the persistent cache on first use;
warning events reported 104–311 ms of aggregate compilation per event. This
run does not establish zero serving-time compilation. It is a new-scenario
functional/startup measurement, not a strict A/B or a model-quality evaluation.

```sh
falcon workflow analysis outputs an-5ebgn4k01d --output json
falcon workflow analysis cat an-5ebgn4k01d summary.json --output json
falcon workflow analysis cat an-5ebgn4k01d rank-1-startup.txt --output json
falcon workflow analysis cat an-5ebgn4k01d rank-1-serving-compile.txt --output json
```

## Host-staging follow-up: 2026-11-17

- Tested source: `310b2eca55dcd043f83a785ccb730752d3909752`.
- Experiment: `exp-lhz0s984yj`, artifact `art-0nqi25r7ym`, **SUCCEEDED**.
- Application attempt 1; same block checkpoint, topology, NEXTN and serving recipe.
- `SGLANG_TPU_ENABLE_EXPERT_HOST_STAGING=1` enables the existing PP staging
  mechanism for TP/EP too, assembling expert slices on CPU before chunked uploads.
  The production default remains off; this adds no GLM cases to shared loading.
- Both hosts passed 12 staging tests, including the non-PP activation/flush test.
- The 16-DP smoke summary reports PASS.

| Phase | Observed time |
| --- | --- |
| Target model load, including conversion | 69.74–82.24 s across 16 ranks |
| NEXTN model load, including conversion | 20.84–22.70 s across 16 ranks |
| Global compile and warmup | 304.55–304.56 s wall time |
| Launcher first timestamped runtime log to ready | Approximately 8 min 26 s |

The last interval is 13:09:33–13:17:59 UTC and excludes environment installation
and checkpoint pre-reading, as above. NEXTN loading is about 200 seconds shorter
than the previous run; target loading did not improve. These are successive
iteration results, not a controlled A/B. No quantization math or expert mapping
was changed. The direct-compile and SPMD settings are unchanged.

Analysis `an-dtjygnuy2w` succeeded. Both host results and all six measured
serving rounds passed (240 requests); all 22 actual-process runtime records
match the pinned source/dependencies and show host staging enabled.

- C16 median TTFT: 6,185.14 ms.
- C64 median client TPOT: 16.862 ms/token.
- C64 median steady throughput: 4,007.84 token/s, 500.98 token/s/chip
  (rounds: 4,070.11 / 4,007.84 / 3,952.72 token/s).

Steady throughput was about 1.9% below the preceding run; this single iteration
does not establish a serving-performance improvement or regression. The
physical rank-0 log records no new direct model specialization after ready,
while small eager XLA compilation/cache loads remain.

```sh
falcon workflow analysis cat an-dtjygnuy2w summary.json --output json
falcon workflow analysis cat an-dtjygnuy2w rank-1-startup.txt --output json
```

Remaining work is draft asynchronous compilation (including deferred KV/DSA
output ownership) and GLM target SPMD graph/layout verification. The current
recipe does not enable either unvalidated change.
