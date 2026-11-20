# GLM-5.3-Flash AgentX on TPU: submission recipes

GLM-5.3-Flash on one tpu7x host (2x2x1, 4 chips = 8 TensorCores, tp8), 1M context,
MTP verify at InferenceX's golden acceptance. One recipe per point of the chart
"GLM Flash AgentX Curves" v26. `_base.py` builds every recipe.

## Points

All points: SemiAnalysis cc traces, 1 h window, dummy weights, 0 failed requests,
0 recompiles after ready.

| cc | recipe | tok/s/chip | interactivity p50 / p90 | TTFT p50 / p90 s | e2e p50 / p90 s | prefix hit % (ideal) | requests | run JSON (commit) | tree | host, image | job |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `agentx_cc1.py` | 6,220 | 464.1 / 421.5 | 0.812 / 1.321 | 3.01 / 11.87 | 98.12 (98.25) | 298 | `runs/R1b_c1.json` (`7ea944ccba`) | `6841608503` | tpuq slice, `1affcdb` | 2460 |
| 4 | `agentx_cc4.py` | 8,958 | 446.5 / 337.7 | 0.505 / 1.374 | 1.69 / 10.02 | 97.71 (97.92) | 675 | `runs/C4u_c4.json` (`62f382f7cf`) | `6841608503` | dev pod, `5448af1b` | 6076 |
| 8 | `agentx_cc8.py` | 14,705 | 320.9 / 217.7 | 0.418 / 1.159 | 2.11 / 10.27 | 97.51 (97.81) | 1,373 | `runs/K2ts_c8.json` (`f75a8f7c7a`) | `cc68b88a76` | dev pod, `5448af1b` | f366 / 8c26 |
| 16 | `agentx_cc16.py` | 26,037 | 209.0 / 110.9 | 0.493 / 1.616 | 2.61 / 14.01 | 97.07 (97.57) | 2,506 | `runs/C16_S2ts.json` (`5132ae1f7c`) | `b720ee3df8` | dev pod, `5448af1b` | 8c26 |
| 16 (alt, not on the chart) | `agentx_cc16_kda1400.py` | 26,359 | 211.9 / 112.6 | 0.492 / 1.598 | 2.64 / 13.40 | 97.23 (97.59) | 2,510 | `runs/C16_S2ts_kda1400.json` (`ba8a3ebde9`) | `b720ee3df8` | dev pod, `5448af1b` | 8c26 |
| 32 | `agentx_cc32.py` | 38,358 | 96.2 / 54.4 | 0.934 / 13.643 | 8.36 / 35.79 | 96.77 (97.57) | 3,604 | `runs/R5_c32_ts.json` (`c130b2d432`) | `310855e651` | dev pod, `5448af1b` | fee5 |
| 32, max running 32 | pending: no recipe | | | | | | | | | | 175a (split 8), 3ca4 (split 4) |
| 48, max running 48 | pending: no recipe | | | | | | | | | | ccd1 (queued) |

- Every measured tree is an ancestor of this branch. Code added after a point's tree is the
  token split on the split latent cache (`310855e651`, `cc68b88a76`): c1/c4 neither split nor
  token-split, and at c32's 8 shards the gather width is unchanged. c8/c16 differ in tests at most.
- `runs/C4u_c4.json` is copied from `qinghan-glm-agentx-mk` `62f382f7cf`; its parent
  `7ea944ccba` is an ancestor of this branch.
- The c16 alternative moves HBM from the KV pool to KDA slots. Throughput and interactivity
  are within the ~3% run-to-run noise of c16; its prefix-hit gap to ideal is 0.36 pt, not 0.50.
- Pending: c32 at max running 32 (decode buckets 1, 2, 4, 8, 12, 16, 20, 24, 28, 32), split 8
  with host KV 2.0x (175a, branch `qinghan-glm-agentx-c32-mr32`) and split 4 with host KV
  1.33x (3ca4, branch `qinghan-glm-agentx-c32-s4`); c48 at max running 48, split 8 (ccd1).
- Images: `sglang-tpu@sha256:1affcdb…` (jax 0.11.2, libtpu 0.0.50, torch_tpu
  0.2.1.dev20261028); dev pods `sha256:5448af1b…` (same jax/libtpu, torch_tpu 0.2.0; deleted
  from the registry by its cleanup policy).

### Config per point

| cc | MTP K / AL | max running | KV pool at boot (cap) | KDA slots | latent split | token split | host KV / KDA states per rank | compile | pod memory peak | boot s |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 7 / 3.93 | 2 | 3,407,872 | 74 | none | off | 6.82M tok, 47.1 GB / 149, 2.9 GB | SPMD | 521 GB | 185 |
| 4 | 7 / 3.93 | 4 | 3,407,872 | 384 | none | off | 6.82M, 47.1 GB / 769, 15.1 GB | per rank | 670 GB | 286 |
| 8 | 3 / 3.00 | 8 | 8,798,720 (9,437,184) | 800 | 2 | on | 13.2M, 50.7 GB / 1,201, 23.6 GB | per rank | 858 GB | 545 |
| 16 | 3 / 3.00 | 16 | 8,639,744 (9,437,184) | 800 | 2 | on | 9.24M, 35.5 GB / 2,001, 39.3 GB | per rank | 926 GB | 766 |
| 16 alt | 3 / 3.00 | 16 | 5,658,624 (9,437,184) | 1,400 | 2 | on | 9.62M, 36.9 GB / 2,001, 39.3 GB | per rank | 943 GB | 766 |
| 32 | 3 / 3.00 | 16 | 9,437,184 | 800 | 8 | on | 23.6M, 36.2 GB / 2,001, 39.3 GB | per rank | 983 GB | 816 |

- K = `speculative_num_steps`; K + 1 draft tokens, topk 1. AL is the golden value for K.
- Boot = launch to ready with dummy weights and a warm compile cache.
- c4 compiles per rank because its dev-pod image (torch_tpu 0.2.0) has no SPMD compile; c1
  needs torch_tpu 0.2.1.dev or later for it. The split cache always compiles per rank.
- Same at every point: context 1,048,576, page 128, fp8 weights, fp8_e4m3 KV, ep 1,
  `mem_fraction_static` 0.9, decode programs 1..max running, prefill programs 1k/2k/3k/4k,
  4096-token chunks, direct tracer, megakernel off, host tier raiden / `page_first_fused` /
  `write_through`, raw-ring pooled indexer, page table in HBM, served name
  `zai-org/GLM-5.3-Flash`, parsers glm45 / glm47, `127.0.0.1:30000`.

### Submission vs measured

`check_recipes.py` renders each recipe both ways and diffs it against its run JSON:
with `AGENTX_DUMMY=1` every recipe equals its run JSON exactly. The default (submission)
render loads real weights from `/weights/GLM-5.3-Flash` and differs only by:

| key | run JSON | submission | why |
|---|---|---|---|
| `load_format` | `dummy` | absent | real weights |
| `weight_loader_prefetch_checkpoints` | absent | true | NFS load 454 s, ~19 min without |
| `weight_loader_drop_cache_after_load` | absent | true | load page cache plus host tier evict the pod |
| env `SGLANG_UNBALANCED_MODEL_LOADING_TIMEOUT_S` | absent | 2400 | skewed NFS loads pass the 480 s default |

These are the settings of the real-weight GSM8K boots of these configs (GSM8K-200 0.920
with the c8 config and with the c1 config plus token split). No AgentX point ran on real
weights.

Golden acceptance stays in every recipe: `SGLANG_SIMULATE_ACC_LEN` = golden AL for K,
`SGLANG_SIMULATE_ACC_METHOD=match-expected`, `SGLANG_SIMULATE_ACC_TOKEN_MODE=fixed`. Open
questions against InferenceX:
- Token mode: InferenceX's connector sets `real-draft-token` for SGLang; the points ran
  `fixed`. The TPU spec worker supports both; the switch is unmeasured.
- InferenceX says not to hard-code acceptance in srt-slurm recipes or launchers: its
  connector injects it. This TPU launch has no connector. Through InferenceX's srt driver,
  drop the three variables.
- Client: the points ran `agentx-harness==1.0.12` (`aiperf_version` 1.0.12 in the five
  kept exports; c4's is not kept); InferenceX's standalone guide now pins 1.0.14.
- Weights: InferenceX's GB300 FP8 recipe pins `zai-org/GLM-5.3-Flash` revision
  `46a050bec13002ad29ee795c16e61cf4fa3c4a9c`; the revision at `/weights/GLM-5.3-Flash` is
  not recorded.
- InferenceX: "do not flush caches or reuse a live server across concurrency points".
  Each point had its own fresh server; the harness flushed once, before the window, to drop
  the warm-up prompts.

## Launch a point

### Requirements
- One tpu7x 2x2x1 host (tpu7x-8). On one host of a 2x2x2 slice run alone (the dev pods
  of c4-c32), set `AGENTX_HALF_SLICE=1`.
- `libibverbs.so.1` (raiden host-tier transfers; `apt-get install libibverbs1`).
- Weights at `/weights/GLM-5.3-Flash` (target and MTP draft).
- Caches on a PVC, never the container overlay: `CACHE_ROOT` (default `/cache/agentx`)
  holds `TORCH_TPU_TIER3_ROOT`, `TORCHINDUCTOR_CACHE_DIR`, `TMPDIR`, `SGLANG_CACHE_DIR`
  (`~/.cache/sglang` links to it) and `PROMETHEUS_MULTIPROC_DIR`, one directory per point.
- `ulimit -c 0`: a core dump of one rank fills the disk.
- Host memory: the pod peaks at 521-983 GB of ~987 GB allocatable (table above). Keep
  `/dev/shm` and other processes off the node; one c16 arm next to 47 GB of foreign
  `/dev/shm` peaked at 975 GB.
- Boot: 3-14 min with dummy weights and a warm compile cache (table), 15 min cold (c32,
  926 s). Real weights add the NFS load: real-weight boots of these configs took 15-26 min.

### Server

From the repo root:

```bash
benchmark/tpu/recipe/configs/glm_5_3_flash/experimental/agentx/launch_agentx.sh 16
```

- Points: `1`, `4`, `8`, `16`, `16-kda1400`, `32`. One fresh server per point; stop it
  (SIGTERM to the launcher) before the next.
- It exports the branch's `PYTHONPATH`, the cache dirs and `ulimit -c 0`, then runs
  `python3 -u -m benchmark.tpu.launch_server --config recipe/configs/glm_5_3_flash/experimental/agentx/agentx_cc16.py`.
- `AGENTX_DUMMY=1`: the measured config exactly (random weights).
- `--dry-run`: print the resolved command and server args, start nothing.
- Ready when the log prints `The server is fired up and ready to roll`.

### Client

On the same host, as the harness ran every point (`CC` = the point's concurrency, 16 for
`16-kda1400`):

```bash
URL=http://127.0.0.1:30000 CC=16 OUT=$PWD/agentx-c16
mkdir -p $OUT
# Harness warm-up: four random-id prompts of 100-6000 tokens, 8 new tokens each.
python3 - <<'EOF'
import json, random, urllib.request
rng = random.Random(7)
for n in (100, 1500, 3000, 6000):
    body = {"input_ids": [rng.randrange(1000, 150000) for _ in range(n)],
            "sampling_params": {"max_new_tokens": 8, "temperature": 0.0, "ignore_eos": True}}
    urllib.request.urlopen(urllib.request.Request(
        "http://127.0.0.1:30000/generate", json.dumps(body).encode(),
        {"Content-Type": "application/json"}), timeout=3600).read()
EOF
curl -s -m 600 -X POST $URL/flush_cache; sleep 20  # empty prefix cache
# HF_HOME holds the pre-fetched trace dataset; drop HF_DATASETS_OFFLINE to download it.
HF_DATASETS_OFFLINE=1 AIPERF_HTTP_TCP_USER_TIMEOUT=900000 TMPDIR=/tmp \
  env -u HF_HUB_OFFLINE -u PYTHONPATH agentx profile \
  --url $URL --endpoint /v1/chat/completions --model zai-org/GLM-5.3-Flash \
  --tokenizer /weights/GLM-5.3-Flash --tokenizer-trust-remote-code --concurrency $CC \
  --public-dataset semianalysis_cc_traces_weka_062126 --benchmark-duration 3600 \
  --server-metrics $URL/metrics --output-artifact-dir $OUT/aiperf_artifacts \
  > $OUT/aiperf.log 2>&1
python3 benchmark/tpu/recipe/configs/glm_5_3_flash/experimental/agentx/summarize.py --md $OUT
```

- `agentx profile` = `agentx-harness==1.0.12` in its own venv, seed 42 (default).
- The `agentx` preset replays sessions (`agentic_replay`) with `ignore_eos`, a first-turn
  cache bust and a 10 s idle-gap cap. Its warm-up runs before the window: snapshot
  primers for the sessions in flight, then 10 requests per lane at `max_tokens=1` (~6 min at
  c16). Only the profiling phase counts.
- Trace: `semianalysis_cc_traces_weka_062126` (HF `semianalysisai/cc-traces-weka-062126`,
  393 entries), the full uncapped corpus InferenceX uses for GLM.
- Per point: boot, then ~70 min of client (warm-up, 1 h window, drain).
- c8 and c16 also ran a prefill probe (random-id prompts, 4k chunks on a 128k prefix) before the flush.

### Metrics

`summarize.py` (stdlib only) turns an output dir into a table row:
- tok/s/chip = (prompt + completion tokens) / window / 4 chips; prompt tokens include
  cached ones. Window = first request start to last request end, profiling phase only.
- Interactivity = 1000 / inter-token latency (ms) at p50 / p90; its p90 is the chart's x.
- Prefix hit = (device + host hits) / all prefill tokens on TP0 (`--server-metrics` CSV);
  ideal = aiperf `theoretical_prefix_cache_hit`.
- The chart's interactivity at c4/c8/c16 is the percentile over `profile_export.jsonl`
  records; summarize.py's aiperf aggregate gives up to 0.6% more (c8: 321.7 / 217.9).
  Everything else matches the table exactly (checked on the c1/c8/c16/c32 outputs).

## Files

- `_base.py`: `agentx_config`, shared by every recipe; `GOLDEN_AL`; the real-weight settings.
- `agentx_cc<N>.py`: one recipe per point.
- `check_recipes.py`: recipe vs run JSON (also `benchmark/tpu/test_launch_server.py`).
- `launch_agentx.sh`: server for one point.
- `summarize.py`: metrics of one client output dir.
- `runs/*.json`: launch JSONs of measured runs, the measurement record. The six above are
  the points; the rest are earlier arms (jobs 2460, 6076, fee5, f366, 8c26), not chart points.
