# GLM-5.3-Flash AgentX on TPU

GLM-5.3-Flash (fp8) on one tpu7x host: 2x2x1, 4 chips = 8 TensorCores, tp8, 1M context.
Benchmark: AgentX (`agentx-harness==1.0.12`, SemiAnalysis cc traces, 1 h window per
concurrency). MTP verify at InferenceX's golden acceptance: K=3 steps -> AL 3.00, K=7 ->
AL 3.93. One recipe per point of the chart "GLM Flash AgentX Curves"; `_base.py` builds
them all.

## Results

Chart v26 (artifact version `1795126667-18de`, read 2026-11-20); the rows below are its
TPU points.

| cc | recipe | run JSON (commit) | tok/s/chip | M tokens/$ | ix p50 / p90 | TTFT p50 / p90 s | prefix hit % (ideal) | tree | job |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `agentx_cc1.py` | `runs/R1b_c1.json` (`7ea944ccba`) | 6,220 | 21.7 | 464.1 / 421.5 | 0.81 / 1.32 | 98.12 (98.25) | `6841608503` | 2460 |
| 4 | `agentx_cc4.py` | `runs/C4u_c4.json` (`6e4989b2f8`, copy of `62f382f7cf`) | 8,958 | 31.3 | 446.5 / 337.7 | 0.51 / 1.37 | 97.71 (97.92) | `6841608503` | 6076 |
| 8 | `agentx_cc8.py` | `runs/K2ts_c8.json` (`f75a8f7c7a`) | 14,705 | 51.4 | 320.9 / 217.7 | 0.42 / 1.16 | 97.51 (97.81) | `cc68b88a76` | f366 |
| 16 | `agentx_cc16.py` | `runs/C16_S2ts.json` (`5132ae1f7c`) | 26,037 | 91.0 | 209.0 / 110.9 | 0.49 / 1.62 | 97.07 (97.57) | `b720ee3df8` | 8c26 |
| 16 alt, off chart | `agentx_cc16_kda1400.py` | `runs/C16_S2ts_kda1400.json` (`ba8a3ebde9`) | 26,359 | 92.1 | 211.9 / 112.6 | 0.49 / 1.60 | 97.23 (97.59) | `b720ee3df8` | 8c26 |
| 32 | `agentx_cc32.py` | `runs/R5_c32_ts.json` (`c130b2d432`) | 38,358 | 134.1 | 96.2 / 54.4 | 0.93 / 13.64 | 96.77 (97.57) | `310855e651` | fee5 |

- ix = interactivity, output tok/s per user = 1000 / inter-token latency (ms). The chart
  plots tokens/$ against ix p90.
- Tokens/$ = tok/s/chip x 3600 / $1.03 per chip-hour (InferenceX's TPU7x rate), in millions.
- tok/s/chip counts prompt and completion tokens, cached prompt tokens included.
- Every point ran with dummy weights, 0 failed requests and 0 recompiles after ready.
- Ideal = aiperf's theoretical prefix-cache hit, the figure InferenceX publishes.
- The c16 alternative moves HBM from the KV pool to KDA slots. Throughput and ix are within
  the ~3% run-to-run noise of c16. Its hit gap to ideal is 0.36 pt instead of 0.50.
- Hosts:
  - c1: tpuq slice, image `sglang-tpu@sha256:1affcdb…` (jax 0.11.2, libtpu 0.0.50,
    torch_tpu 0.2.1.dev20261028).
  - c4-c32: one host of a 2x2x2 dev pod, image `sha256:5448af1b…` (same jax and libtpu,
    torch_tpu 0.2.0). The registry's cleanup policy has since deleted that image.
- Every tree is an ancestor of this branch. The only code added after a point's tree is
  the token split on the split latent cache (`310855e651`, `cc68b88a76`):
  - c1 and c4 do not split the cache.
  - At c32's 8 shards the gather width is unchanged.
  - The c8 tree differs from c16's (`b720ee3df8`) by a kernel test only.

## Replicate

### Requirements

- Host: one tpu7x 2x2x1 host (tpu7x-8). The image above, plus `libibverbs.so.1`
  (`apt-get install libibverbs1`) for the raiden host-tier transfers. On one host of a
  2x2x2 slice run alone, set `AGENTX_HALF_SLICE=1`; the c4-c32 points ran that way.
- Weights: target and MTP draft at `/weights/GLM-5.3-Flash`.
- Caches: put them on a PVC, never on the container overlay. `CACHE_ROOT` (default
  `/cache/agentx`) gets one directory per point, holding:
  - the compile caches (`TORCH_TPU_TIER3_ROOT`, `TORCHINDUCTOR_CACHE_DIR`);
  - `TMPDIR`;
  - `SGLANG_CACHE_DIR` (`~/.cache/sglang` links to it);
  - `PROMETHEUS_MULTIPROC_DIR`.
- Core dumps: `ulimit -c 0`, because one rank's core dump fills the disk. The launcher sets it.
- Host memory: the pod peaks at 521-983 GB of the node's ~987 GB allocatable (table
  below). Keep `/dev/shm` and other processes off the node. One c16 arm running next to
  47 GB of foreign `/dev/shm` peaked at 975 GB.
- Client: `agentx-harness==1.0.12` in its own venv, run on the same host. The trace
  `semianalysis_cc_traces_weka_062126` (HF `semianalysisai/cc-traces-weka-062126`, 393
  entries) must be pre-fetched into `HF_HOME`. This is the full uncapped corpus InferenceX
  uses for GLM.

### Per point

Run one fresh server per point. Stop it (SIGTERM to the launcher) before the next point.

| point | server | client `CC` | MTP K / AL | max running | KV pool at boot (cap) | KDA slots | latent split / token split | host tier: KV tokens / KDA states | compile | pod memory peak | boot s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| c1 | `$A/launch_agentx.sh 1` | 1 | 7 / 3.93 | 2 | 3,407,872 | 74 | none / off | 6.82M / 149 | SPMD | 521 GB | 185 |
| c4 | `$A/launch_agentx.sh 4` | 4 | 7 / 3.93 | 4 | 3,407,872 | 384 | none / off | 6.82M / 769 | per rank | 670 GB | 286 |
| c8 | `$A/launch_agentx.sh 8` | 8 | 3 / 3.00 | 8 | 8,798,720 (9,437,184) | 800 | 2 / on | 13.2M / 1,201 | per rank | 858 GB | 545 |
| c16 | `$A/launch_agentx.sh 16` | 16 | 3 / 3.00 | 16 | 8,639,744 (9,437,184) | 800 | 2 / on | 9.24M / 2,001 | per rank | 926 GB | 766 |
| c16 alt | `$A/launch_agentx.sh 16-kda1400` | 16 | 3 / 3.00 | 16 | 5,658,624 (9,437,184) | 1,400 | 2 / on | 9.62M / 2,001 | per rank | 943 GB | 766 |
| c32 | `$A/launch_agentx.sh 32` | 32 | 3 / 3.00 | 16 | 9,437,184 | 800 | 8 / on | 23.6M / 2,001 | per rank | 983 GB | 816 |

- `A=benchmark/tpu/recipe/configs/glm_5_3_flash/experimental/agentx`, run from the repo root.
- K = `speculative_num_steps`, giving K + 1 draft tokens and topk 1. Acceptance is
  simulated on the server, and each recipe sets it:
  - `SGLANG_SIMULATE_ACC_LEN` = the golden AL for K;
  - `SGLANG_SIMULATE_ACC_METHOD=match-expected`;
  - `SGLANG_SIMULATE_ACC_TOKEN_MODE=fixed`.

  The client sets nothing for acceptance.
- Boot time = launch to ready, with dummy weights and a warm compile cache. A cold c32
  boot took 926 s. Real weights add the NFS load: 15-26 min in total.
- c4 compiles per rank because its image (torch_tpu 0.2.0) has no SPMD compile. c1's SPMD
  compile needs torch_tpu 0.2.1.dev or later. The split cache always compiles per rank.
- Same at every point:
  - context 1,048,576, page 128;
  - fp8 weights and fp8_e4m3 KV;
  - `mem_fraction_static` 0.9, ep 1;
  - decode programs 1..max running; prefill programs 1k/2k/3k/4k with 4096-token chunks;
  - direct tracer, megakernel off;
  - host tier raiden / `page_first_fused` / `write_through`;
  - raw-ring pooled indexer, page table in HBM;
  - served name `zai-org/GLM-5.3-Flash`, parsers glm45 / glm47, `127.0.0.1:30000`.

### Server

```bash
A=benchmark/tpu/recipe/configs/glm_5_3_flash/experimental/agentx
CACHE_ROOT=/cache/agentx $A/launch_agentx.sh 16    # real weights (submission)
AGENTX_DUMMY=1 $A/launch_agentx.sh 16              # dummy weights: the measured config exactly
$A/launch_agentx.sh 16 --dry-run                   # print the resolved command; start nothing
```

- What the launcher does:
  - exports the branch's `PYTHONPATH`, the cache dirs and `ulimit -c 0`;
  - runs `python3 -u -m benchmark.tpu.launch_server --config recipe/configs/glm_5_3_flash/experimental/agentx/<recipe>.py`.
- The server is ready when the log prints `The server is fired up and ready to roll`.

### Client

On the same host, with `CC` from the table:

```bash
A=benchmark/tpu/recipe/configs/glm_5_3_flash/experimental/agentx
URL=http://127.0.0.1:30000 CC=16 OUT=$PWD/agentx-c16
mkdir -p $OUT
# 1. Warm-up: four random-id prompts of 100-6000 tokens, 8 new tokens each.
python3 - "$URL" <<'EOF'
import json, random, sys, urllib.request
rng = random.Random(7)
for n in (100, 1500, 3000, 6000):
    body = {"input_ids": [rng.randrange(1000, 150000) for _ in range(n)],
            "sampling_params": {"max_new_tokens": 8, "temperature": 0.0, "ignore_eos": True}}
    urllib.request.urlopen(urllib.request.Request(
        sys.argv[1] + "/generate", json.dumps(body).encode(),
        {"Content-Type": "application/json"}), timeout=3600).read()
EOF
# 2. Empty the prefix cache.
curl -s -m 600 -X POST $URL/flush_cache; sleep 20
# 3. The 1 h window. Drop HF_DATASETS_OFFLINE to download the trace instead.
HF_DATASETS_OFFLINE=1 AIPERF_HTTP_TCP_USER_TIMEOUT=900000 TMPDIR=/tmp \
  env -u HF_HUB_OFFLINE -u PYTHONPATH agentx profile \
  --url $URL --endpoint /v1/chat/completions --model zai-org/GLM-5.3-Flash \
  --tokenizer /weights/GLM-5.3-Flash --tokenizer-trust-remote-code --concurrency $CC \
  --public-dataset semianalysis_cc_traces_weka_062126 --benchmark-duration 3600 \
  --server-metrics $URL/metrics --output-artifact-dir $OUT/aiperf_artifacts \
  > $OUT/aiperf.log 2>&1
# 4. The table row.
python3 $A/summarize.py --md $OUT
```

- The `agentx` preset (seed 42):
  - replays sessions (`agentic_replay`) with `ignore_eos`;
  - busts the cache on each first turn;
  - caps idle gaps at 10 s.
- The harness warm-up runs before the window: snapshot primers, then 10 requests per lane
  at `max_tokens=1` (~6 min at c16). Only the profiling phase counts.
- Per point: boot, then ~70 min of client time (warm-up, the 1 h window, drain).
- At c8 and c16 the measured runs also sent a prefill probe before step 2: random-id
  prompts, 4k chunks on a 128k prefix.

### Metrics

`summarize.py` (stdlib only) turns one output dir into a table row:
- tok/s/chip = (prompt + completion tokens) / window / 4 chips. The window runs from the
  first request start to the last request end, profiling phase only.
- ix p50 / p90 = 1000 / inter-token latency (ms).
- Prefix hit = (device + host hits) / all prefill tokens on TP0, from the
  `--server-metrics` CSV.
- Ideal = aiperf `theoretical_prefix_cache_hit`.
- Interactivity sources: the chart's c4/c8/c16 values are percentiles over the
  `profile_export.jsonl` records, and summarize.py's aiperf aggregate reads up to 0.6% higher
  (c8: 321.7 / 217.9). For c1 and c32 the chart uses the aiperf aggregate.

### Submission vs measured

`check_recipes.py` renders each recipe and diffs it against its run JSON:
- With `AGENTX_DUMMY=1`, every recipe renders its run JSON exactly.
- The default render, which is the submission, loads real weights and differs only in these keys:

| key | run JSON | submission | why |
|---|---|---|---|
| `load_format` | `dummy` | absent | real weights |
| `weight_loader_prefetch_checkpoints` | absent | true | NFS load takes 454 s; ~19 min without |
| `weight_loader_drop_cache_after_load` | absent | true | the load's page cache plus the host tier evict the pod |
| env `SGLANG_UNBALANCED_MODEL_LOADING_TIMEOUT_S` | absent | 2400 | skewed NFS loads pass the 480 s default |

These settings come from the real-weight GSM8K boots of these configs: GSM8K-200 scored
0.920 with the c8 config, and 0.920 with the c1 config plus token split. No AgentX point
ran on real weights.

## Other measured runs

Each of these files in `runs/` is a run's launch JSON. All are 1 h AgentX windows with
dummy weights unless stated otherwise.

c32 follow-ups, none of which beats the c32 point (R5_c32_ts: 38,358 tok/s/chip, ix p90 54.4):

| run JSON | job | change vs `R5_c32_ts` | outcome |
|---|---|---|---|
| `R6_c32_ts_mr32.json` | 175a | max running 32, decode buckets 1..32 | no window: boot ran out of host RAM allocating the KDA host pool |
| `R6b_c32_ts_mr32_sparse.json` | 175a | as above, buckets 1..16, 20, 24, 28, 32 | OOM-killed 35 s into the window (host RAM) |
| `R6c_c32_ts_mr32_b10_h20.json` | 175a | max running 32, 10 buckets, host KV 2.0x | 38,149, ix p50/p90 85.0 / 39.9, TTFT p90 3.03 s; dominated by R5_c32_ts |
| `R7_c32_s4ts_mr32_b10.json` | 3ca4 | as R6c with split 4, host KV 1.33x | 38,807, ix 87.9 / 41.2, TTFT p90 2.80 s; dominated on the chart axes |
| `R5k_c32_ts_kda1600.json` | 98c5 | 1,600 KDA slots | 37,900, ix p90 53.2, hit gap 0.83 vs 0.80 pt: no gain |

Earlier arms, superseded by the points above (tok/s/chip, ix p90):

| run JSON | job | config | outcome |
|---|---|---|---|
| `E1_c1.json` | 2460 | c1, K=8 / AL 3.99 | 5,557, 352.5 |
| `E2_c4.json` | 2460 | c4, split 8, K=8 | 7,996, 236.4 |
| `E3_c8.json` | 2460 | c8, split 8, K=8 | 13,693, 148.8 |
| `E4_c16.json` | 2460 | c16, split 8, K=3 | 23,207, 83.7 |
| `R2_c4.json` | 2460 | E2 with K=3 / AL 3.00 | 8,359, 282.5 |
| `R3_c8.json` | 2460 | E3 with K=3 | 14,337, 175.0 |
| `R5_c32.json` | 2460 | c32, split 8, K=3, max running 16 (ran as R5b after a warm-boot hang) | 35,755, 50.2 |
| `R1b_c1_ts.json` | fee5 | c1 point + token split | 6,269, 416.8 (neutral) |
| `C4u_c4_ts.json` | fee5 | c4 point + token split | 8,939, 336.4 (neutral) |
| `R3_c8_ts.json` | fee5 | R3 + token split | 14,466, 188.1 |
| `E4_c16_ts.json` | fee5 | E4 + token split | 24,407, 93.5 |
| `K2_c8.json` | f366 | c8, split 2, no token split (decode probe) | 14,796, 198.2 |
| `C16_S4ts.json` | 8c26 | c16 point with split 4, host KV 1.65x | 25,139, 98.1 |
| `C16_S2ts_kda1100.json` | 8c26 | c16 point with 1,100 KDA slots | 26,004, 111.6 |
| `C16_S4ts_real.json` | 8c26 | C16_S4ts, real weights | GSM8K-200 0.905 (no AgentX window) |

## Pending

These are not folded into the results above:
- Prefill patches on branch `qinghan-glm-prefill-combo` (`c0a255c163`, job 2507):
  - SparseCore top-k width buckets, which are exact;
  - an opt-in bf16 split-cache merge.

  Measured at split 8 + token split, a 4k chunk drops from 0.2432 to 0.2204 s at context 0
  and from 0.2795 to 0.2393 s at 256k. GSM8K-1319 is unchanged (0.9060 on vs 0.9037 off).
- c32 AgentX on that tree with the R5_c32_ts config (job 15c4).
- c48 at max running 48 with split 8 + token split, on the same tree (job ccd1).
- `prefill_decode_interval` arms at c32 on the same tree (job 754a).

## Tunable configs

Values as the run JSONs set them, c1 / c4 / c8 / c16 / c32 (c16 alt in parentheses where
it differs); unset = server default. Evidence is a job ID; ix numbers are p90.

| knob | where set | c1 / c4 / c8 / c16 / c32 | what it trades | evidence |
|---|---|---|---|---|
| prefill-decode interval | `prefill_decode_interval` | unset (0, off) at every point | N decode steps between prefill batches: higher ix, later TTFT; a fixed N starves admission at high concurrency | c16 N=16, split 8 before token split: ix 84.9 -> 91.3, tok/s -0.5%, TTFT p90 2.06 -> 2.95 s (ff06). Muse Spark c10, N=64 vs its shipped profile: ix 169 vs 116 (Haolin's perf KB, F-0069..71). c32 N=16/64, c48 N=16: running (754a, ccd1) |
| latent-cache split | env `SGLANG_TPU_DSA_MLA_KV_SHARDS` | unset (1) / unset / 2 / 2 / 8 | fewer shards: cheaper prefill merge, more KV bytes per rank (host KV 30,720 / 18,432 / 12,288 B/token/pod at 2 / 4 / 8; the split-2 pool is HBM-bound below the 9.4M cap) | c16 split 2 vs 8: ix +18.9%, tok/s +6.7%; split 4: ix 98.1 (8c26). c32 at max running 32, split 4 vs 8: 38,807 vs 38,149 tok/s, ix 41.2 vs 39.9 (3ca4, 175a). c4 unsplit vs split 8: ix 337.7 vs 287.5 (6076) |
| token split | env `SGLANG_TPU_ENABLE_DSA_ATTENTION_TOKEN_SPLIT` | unset / unset / 1 / 1 / 1 | each selected cache row gathered by one rank, plus an all-to-all of queries and outputs | c16 split 8: ix 83.7 -> 93.5, tok/s 23,207 -> 24,407; c8: ix 175.0 -> 188.1; neutral on the unsplit c1/c4 (fee5) |
| device KV pool | `max_total_tokens` (cap; HBM may bound it lower), `mem_fraction_static` 0.9 | 3,407,872 / 3,407,872 / 9,437,184 / 9,437,184 / 9,437,184 (boot sizes in the per-point table) | HBM for KV tokens vs device KDA slots | c16 alt (8.64M -> 5.66M tokens, 800 -> 1,400 slots): state evictions 6,399 -> 2,192 per h, tok/s and ix within noise (8c26) |
| device KDA slots | `max_mamba_cache_size` | 74 / 384 / 800 / 800 (1,400) / 800 | HBM for KDA checkpoints vs KV pool; sized for the conversations AgentX keeps cached, not from max running | c16 1,100: 26,004 / ix 111.6; 1,400: hit gap 0.36 vs 0.50 pt (8c26). c32 1,600: no gain (98c5) |
| host KV tier | `hicache_ratio` (x device pool) | 2 / 2 / 1.5 / 1.07 (1.7) / 2.5 | host-hit reach vs host RAM | c32 at max running 32 fit with 2.0 (R6c, 175a) and 1.33 at split 4 (R7, 3ca4) |
| host KDA states | env `SGLANG_TPU_HICACHE_STATE_RATIO` (x device KDA slots; unset = `hicache_ratio`) | unset / unset / unset / 2.5 (1.429) / unset | host states (~19.6 MB/rank each) give the host-tier hits after idle gaps vs host RAM | c16 2,001 states reach ~15 min; ratio 1.5 would cut them to 1,201, ~9 min (8c26) |
| max running | `max_running_requests`, `torch_compile_max_bs` | 2 / 4 / 8 / 16 / 16 | decode batch vs step time; c32's other 16 sessions queue | c32 at 32: tok/s flat, ix -24 to -27% (175a, 3ca4). c48 at 48: running (ccd1) |
| decode buckets | `tpu_compile_bs` | 1..max running at every point | unpadded decode steps vs host RAM per program | c32 at max running 32: buckets 1..32 ran out of host RAM at boot, 1..16 + 20..32 was OOM-killed in the window, 10 buckets ran (175a) |
| MTP draft length | `speculative_num_steps` = K (`speculative_num_draft_tokens` K + 1, `speculative_eagle_topk` 1) | 7 / 7 / 3 / 3 / 3 | more tokens per step at low batch vs more verify work at high batch | at the golden AL: c4 K=3 vs 8 ix 282.5 vs 236.4; c8 175.0 vs 148.8; c1 K=7 vs 8 421.5 vs 352.5 (2460) |
| simulated acceptance | env `SGLANG_SIMULATE_ACC_LEN`, `SGLANG_SIMULATE_ACC_METHOD`, `SGLANG_SIMULATE_ACC_TOKEN_MODE` | 3.93 / 3.93 / 3.0 / 3.0 / 3.0; `match-expected`; `fixed` | not a perf knob: AL = `GOLDEN_AL[K]`, never 3.5. Token mode `fixed` vs `real-draft-token`: same accepted count, only the emitted token ids differ | token mode untested (question 1 below) |
| prefill chunk | `chunked_prefill_size`, `max_prefill_tokens`, `tpu_compile_tokens` | 4096, 4096, [1024, 2048, 3072, 4096] at every point | bigger chunks: less fixed cost per token vs longer decode stalls, more programs | 8192: -2.2% per token, stalls ~2x, +18 programs/rank (dcd8). c16 at 2048: ix 73.4 vs 83.7 (6076) |
| expert parallel | `ep_size` (`tp_size` 8) | 1 at every point | expert-TP vs expert-parallel MoE | untested; judged not worth it for c32+ |
| top-k width buckets | env `SGLANG_TPU_SCTOPK_LONG_ROWS` | not on this branch; on by default on `qinghan-glm-prefill-combo` (0 disables) | exact; faster prefill top-k | with the bf16 merge at split 8 + token split: 4k chunk 0.2432 -> 0.2204 s at ctx 0, 0.2795 -> 0.2393 s at 256k (2507). c32 AgentX running (15c4) |
| bf16 split-cache merge | env `SGLANG_TPU_DSA_MLA_KV_SHARD_BF16_MERGE=1` | not on this branch; opt-in on `qinghan-glm-prefill-combo` | bf16 instead of fp32 reduce-scatter for the prefill merge | as above; GSM8K-1319 0.9060 on vs 0.9037 off (2507) |

- Host RAM: resident host memory grows ~34-39 MB per compiled program per rank (3ca4), so
  decode buckets cost host RAM. When a config does not fit, shrink the host tier first and
  keep the buckets: host-tier hits are fine as long as nothing is evicted or thrashing.

## Open InferenceX questions

1. **Token mode.** The points ran `SGLANG_SIMULATE_ACC_TOKEN_MODE=fixed`. InferenceX's
   connector sets `real-draft-token` for SGLang. The TPU spec worker supports both, but
   the switch is unmeasured.
2. **Acceptance in the recipe.** InferenceX says not to hard-code acceptance in srt-slurm
   recipes or launchers, because its connector injects it. This TPU launch has no
   connector. If the submission goes through InferenceX's srt driver, drop the three
   `SGLANG_SIMULATE_ACC_*` variables.
3. **Client version.** The points ran `agentx-harness==1.0.12`: `aiperf_version` is 1.0.12
   in the five kept exports, and c4's export is not kept. InferenceX's standalone guide now
   pins 1.0.14.
4. **Cache flush.** InferenceX says "do not flush caches or reuse a live server across
   concurrency points". Each point had its own fresh server, but the harness flushed once
   before the window to drop the warm-up prompts.

## Files

- `_base.py`: `agentx_config`, shared by every recipe; `GOLDEN_AL`; the real-weight settings.
- `agentx_cc<N>.py`: one recipe per point.
- `check_recipes.py`: diffs each recipe against its run JSON. `benchmark/tpu/test_launch_server.py`
  also runs it.
- `launch_agentx.sh`: starts the server for one point.
- `summarize.py`: computes the metrics of one client output dir.
- `runs/*.json`: the launch JSONs of measured runs, which are the measurement record.

Chart: [GLM Flash AgentX Curves](https://claude.ai/artifact/JnGfziw2GBSqDx5LCZArVf)
