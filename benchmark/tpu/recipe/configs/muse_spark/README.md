# Muse Spark AgentX on one TPU v7 host

These files reproduce the Muse Spark AgentX sweep at concurrency 1, 7, 9, 10, 16
and 22 from this repo alone. That covers the server at each point, the trace
replay, and the metrics. They reproduce the InferenceX AgentX submission made
with the image built from this branch's engine: the configs render to the
submission's launch flags and environment (see [Equivalence](#equivalence)).

| File | What it is |
| --- | --- |
| `agentx.py` | The profile table and `config_for(conc)`, on top of `agentic.py` and `server.py`. |
| `agentx_c<N>.py` | The server config of one swept point (`CONFIG = config_for(N)`). |
| `windowed_draft.py` | Writes the draft directory the configs serve from (below). |
| `run_agentx.sh` | The sweep driver: per point, boot, replay with AIPerf, summarize, stop. |
| `agentx_summary.py` | The headline metrics from one AIPerf run. |

`agentic.py` stays the single-session (cc=1, 256K) recipe. `agentic_1m.py`, an
earlier 1M profile, is retired: `agentx.py` replaces it.

## What you need

- **Host**: one Cloud TPU v7 (tpu7x) host, 2x2x1: 4 chips, 8 TensorCores, served
  as tp8. The host KV tier is pinned host RAM: 56 GB per rank (448 GB) from conc 16,
  and 2.0x the device KV pool per rank below that. The submission's hosts had
  960 GiB of RAM.
- **Image**: build it from this branch. `docker/tpu-base.Dockerfile` installs the
  pinned TPU stack from `python/pyproject_tpu.toml` (torch-tpu, libtpu, jax).
  `docker/tpu.Dockerfile` adds this source tree on top. `docker/cloudbuild.tpu-base.yaml`
  and `docker/cloudbuild.tpu.yaml` are the Cloud Build recipes. This branch changes
  no engine code, so its image serves the same engine as the submitted one.
- **Weights**: `meta-models/Muse-Spark-Open-816B-A42B-NVFP4` is gated on the Hub
  (access is approved by hand). Request access, then download it once:
  `hf download meta-models/Muse-Spark-Open-816B-A42B-NVFP4`. The draft,
  `RadixArk/Muse-Spark-Open-DFlash2`, is public; `windowed_draft.py` downloads it.
- **Client**: [AIPerf](https://github.com/SemiAnalysisAI/aiperf) at commit
  `754356e9a39acc6cc6afb242d123bb57c3fb6f75`, with its `inferencex-agentx-mvp`
  scenario, and the trace corpus
  [`semianalysisai/cc-traces-weka-062126`](https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126)
  (the 1M corpus). Both are public; `run_agentx.sh` installs AIPerf into its own
  venv with [uv](https://docs.astral.sh/uv/) and fetches the corpus.

Still private:
- torch-tpu wheels: the base image installs them from Google's torch-tpu release
  registry, which needs an access token.
- The submitted image's registry.
- The InferenceX harness that drove the submission. This directory mirrors its
  replay command, its client environment and its metric definitions, but not its
  power accounting or result upload.

## Running the sweep

From the repo root, inside the image:

```bash
benchmark/tpu/recipe/configs/muse_spark/run_agentx.sh
```

For each concurrency the driver does six things:

1. It renders `agentx.py --conc N` to `server_config.json`, and the launcher's
   dry run to `launch_plan.json`.
2. It boots `python3 -m benchmark.tpu.launch_server --config @server_config.json`
   offline (`HF_HUB_OFFLINE=1`, from the local HF cache, as in the submission).
3. It waits for `/health`.
4. It replays the traces for 60 minutes after AIPerf's warmup.
5. It stops the server.
6. It writes `summary.json`.

Each point boots its own server, because the profiles differ. A cold boot
compiles every graph first; later boots reuse the compile cache.

| Variable | Default | |
| --- | --- | --- |
| `CONCS` | `1 7 9 10 16 22` | any concurrency in `agentx.py`'s table, swept or not |
| `DURATION` | `3600` | profiling seconds; the scenario requires 900 or more |
| `FAST` | `0` | `1`: AIPerf's short mode (1200 s, one warmup request per lane) |
| `MODEL` | the Hub id | or a local snapshot directory; also the tokenizer |
| `PORT` | `30000` | |
| `RUN_DIR` | `agentx_runs/<UTC time>` | |
| `MUSE_WINDOWED_DRAFT_DIR` | `~/.cache/sglang/muse-spark-open-dflash2-windowed` | |
| `CHIPS` | `4` | the per-chip denominator |
| `PRINT_ONLY` | `0` | `1`: render each point's launch plan and AIPerf command, run nothing |

### One point by hand

```bash
python3 -m benchmark.tpu.recipe.configs.muse_spark.windowed_draft
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    python3 -m benchmark.tpu.launch_server --config recipe/configs/muse_spark/agentx_c22.py
```

Then, from AIPerf's venv, once `/health` answers:

```bash
AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES=0 \
AIPERF_DATASET_CONFIGURATION_TIMEOUT=1800 \
AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT=1800 \
AIPERF_UI_REALTIME_METRICS_ENABLED=true \
AIPERF_HTTP_TCP_USER_TIMEOUT=900000 \
aiperf profile --scenario inferencex-agentx-mvp \
    --url http://localhost:30000 --endpoint /v1/chat/completions --endpoint-type chat \
    --streaming --model meta-models/Muse-Spark-Open-816B-A42B-NVFP4 \
    --tokenizer meta-models/Muse-Spark-Open-816B-A42B-NVFP4 \
    --concurrency 22 --benchmark-duration 3600 --stats-interval 30 --random-seed 42 \
    --failed-request-threshold 0.10 \
    --trajectory-start-min-ratio 0.25 --trajectory-start-max-ratio 0.75 \
    --warmup-requests-per-lane 10 --trace-idle-gap-cap-seconds 300 \
    --warmup-grace-period 3600 --use-server-token-count --no-gpu-telemetry \
    --tokenizer-trust-remote-code --num-dataset-entries 393 --slice-duration 1.0 \
    --output-artifact-dir c22/aiperf_artifacts \
    --public-dataset semianalysis_cc_traces_weka_062126 \
    --server-metrics http://localhost:30000/metrics
```

What the flags and env do:
- Recorded assistant turns build the later prompts, and live responses are discarded.
- Each trajectory starts between 25% and 75% of its turns, after ten warmup requests
  per lane.
- The scenario caps whole-system idle at 10 s. The trace idle cap is 300 s per
  trajectory.
- Token counts come from the server's usage fields.

To see what any profile renders to, run
`python3 -m benchmark.tpu.recipe.configs.muse_spark.agentx --conc N`. Pipe that
into `launch_server --config @file --dry-run` for the full command.

### Evals

Throughput runs pin DFlash acceptance to the AgentX golden acceptance length:
`SGLANG_SIMULATE_ACC_*` commit drafts regardless of the target's logits. An
accuracy check must verify for real. `agentx.py --conc 22 --eval` renders the same
server without simulated acceptance. The submission's GSM8K check ran InferenceX's
lm-eval wrapper against that server, and that wrapper is private. This repo's
`benchmark/tpu/run_lm_eval.sh` runs lm-eval against a live server from a
`test/lm_eval_configs/` YAML, but there is no Muse Spark GSM8K config yet.

## The profiles

All points share the same base:
- fp8 KV in the SEQ_ALONG_LANE layout, with 256-token pages, batched-RPA prefill
  and megakernel decode;
- a 0.0625 sliding-window pool and `mem_fraction_static` 0.95;
- FCFS admission;
- HiCache over raiden, `page_first_fused`;
- DFlash verifying 8 tokens, narrowing to 4 once *min* or more requests decode;
- the model's native context.

| Conc | Running | Interval | Narrow from | Batch buckets | Prefill graphs | Host KV pool | Sessions |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2 | 128 | 4 | 1 2 | 1024 2048 4096 | ratio 2.0, write-back | no |
| 7 | 3 | 128 | 4 | 1-3 | 1024 2048 4096 | ratio 2.0, write-back | no |
| 9 | 4 | 128 | 4 | 1-4 | 1024 2048 4096 | ratio 2.0, write-back | no |
| 10 | 6 | 64 | 4 | 1-6 | 1024 2048 4096 | ratio 2.0, write-back | no |
| 16 | 16 | 64 | 3 | 1-12, 16 | 4096 | 56 GB/rank, write-through, 30% SWA | yes |
| 22 | 16 | 64, adaptive 6:8 | 3 | 1-12, 16 | 4096 | 56 GB/rank, write-through, 30% SWA | yes |

- *Interval* is the number of decode steps between prefill batches. At conc 22, while
  fewer than 6 requests decode and prefill waits, the interval ends after 8 steps.
- *Sessions* means the X-Correlation-ID session radix cache.
- From conc 16, radix-tree checkpoints are taken every 16K prefill tokens
  (128K below), and the megakernel takes 64-row steps.

The other rows of `agentx.py`'s table were not swept; they are kept for reruns.

**The windowed draft.** Every layer of the DFlash2 draft is 2048-token
sliding-window, but its config does not say so, so SGLang would give it
full-context KV. `windowed_draft.py` writes a directory that symlinks the
snapshot's files and holds a `config.json` with `is_hybrid_swa` and an all-SWA
`hybrid_layer_pattern`. The draft's KV then lives in the target's sliding-window
slots. Its output is byte-identical to the submission's rewrite.

## Reading the numbers

Each point writes `RUN_DIR/c<N>/`:
- `server_config.json`, `launch_plan.json`, `server.log`;
- `benchmark_command.txt`, `benchmark.log` and `aiperf_artifacts/`, where
  `profile_export.jsonl` holds one record per request;
- `summary.json`, with its one-line form in `summary.txt`.

`agentx_summary.py` reads the profiling-phase requests that completed. Warmup and
failed requests are excluded, as in the InferenceX AgentX aggregation:

- **Throughput per chip**: prompt plus completion tokens (the server's usage
  counts), divided by the window from the first request's start to the last
  one's end, divided by 4 chips. Prompt tokens include cached ones.
- **E2E latency**: from request start to the last token.
- **TTFT**: from request start to the first token.
- **Interactivity**: 1 / ITL, where ITL = (E2E − TTFT) / (completion tokens − 1)
  per request. *p90 interactivity* is 1 / (p90 ITL), the slow tail.
- **Validity**: AIPerf's `submission_valid`, failed requests at most 10% of the
  profiled ones, and `sglang:` metrics captured from `/metrics`.

Percentiles interpolate linearly between closest ranks.

## Equivalence

`test/registered/unit/test_muse_spark_agentx_recipe.py` pins each swept
point's rendered flags and environment to the submission's launch.

The public launch differs from the submitted one only in where things live and
in how `launch_server.py` treats every single-host recipe:
- `--host 127.0.0.1` instead of `0.0.0.0`; set `HOST=0.0.0.0` to listen on every
  interface.
- The port, and the model and draft paths.
- `--nnodes 1`, which is the default.
- On a GKE pod, the launcher clears `TPU_VISIBLE_CHIPS`,
  `TPU_CHIPS_PER_HOST_BOUNDS` and `TPU_HOST_BOUNDS`. On a whole host, those
  describe the same four chips.
- The launcher sets the compile-cache variables (`TORCH_TPU_INTERNAL_TIER*`).
  `ENABLE_COMPILE_CACHE=0` leaves them unset. The tier-2 cache lives in
  `/dev/shm` and counts against host RAM.
- The launcher preloads `libtpu_telemetry_stub.so`. The diagnostics and telemetry
  it stubs are off through the `TPU_*` variables either way.
