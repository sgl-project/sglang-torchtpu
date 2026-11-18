# Muse Spark InferenceX AgentX: launch configs and results

One tpu7x 2x2x1 host (4 chips, tp8), native 1M context, AIPerf `inferencex-agentx-mvp`
replay, 3600 s window. README.md covers the image, weights, client and metrics; this
page lists what each point launches with and what has been measured.

## Launch

Every point goes through `run_agentx.sh`, which renders `agentx.py --conc N` and boots
`benchmark.tpu.launch_server` on it:

```bash
MODEL=/models/meta-models/Muse-Spark-Open-816B-A42B-NVFP4 \
CONCS=22 MK_PREFILL=1 MK_QBLOCK=256 DURATION=3600 \
    bash benchmark/tpu/recipe/configs/muse_spark/run_agentx.sh
```

- `MK_PREFILL=1`: extend steps run the megakernel prefill program (`agentx.py --mk-prefill`).
- `MK_QBLOCK=256`: block-256 requantized fp4 experts (`--mk-qblock 256`).
- `MOE_MODE=ep` (default `tp`): EP8 experts on `muse_megakernel_v2` decode/spep and
  prefill/spep; `EP_ARM` gather | a2a | mixed.
- Print a point's config without booting: `python3 -m benchmark.tpu.recipe.configs.muse_spark.agentx --conc N --mk-prefill --mk-qblock 256`.

## Per-concurrency config

All points: tp8, ep1 (TP mode), fp8 KV in SEQ_ALONG_LANE, 256-token pages,
`chunked_prefill_size` 4096, `mem_fraction_static` 0.95, DFlash2 draft (windowed) on the
experimental spec worker with acceptance pinned to the AgentX golden length.

| Conc | Running | Verify tokens | Simulated AL | Interval | `tpu_compile_bs` | `MK_MAX_ROWS` | Host KV (HiCache) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2 | 8 | 4.05 | 128 | 1-2 | - | ratio 2.0, write-back |
| 7 | 3 | 8 | 4.05 | 128 | 1-3 | - | ratio 2.0, write-back |
| 9 | 4 | 8 | 4.05 | 128 | 1-4 | - | ratio 2.0, write-back |
| 10 | 6 | 8 | 4.05 | 64 | 1-6 | 48 | ratio 2.0, write-back |
| 16 | 16 | 4 | 3.06 | 64 | 1-12, 16 | 64 | 72 GB/rank, write-through |
| 22 | 16 | 4 | 3.06 | 64 | 1-12, 16 | 64 | 72 GB/rank, write-through |
| 26 | 19 | 4 | 3.06 | 64 | 1-12, 16, 19 | 80 | 72 GB/rank, write-through |

*Interval*: decode steps between prefill batches. c26 is c22 with ~20% more running
requests and clients.

## Results

All rows: 3600 s window, 0 failed requests, `submission_valid` true unless noted.

### This branch's engine (refkernel MK_PREFILL + QBLOCK 256, TP)

TP serves on `muse_megakernel_v2` (the same stack as EP); the v1 rows below ran the
retired `muse_megakernel`.

| Conc | Megakernel | tok/s/chip | p90 interactivity | p50 interactivity | p90 TTFT | p90 E2E | Requests | Run |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 22 | v2 | 31,039 | 115.8 | 163.0 | 12.7 s | 29.2 s | 2979 | j-955acea7, 2026-11-18 |
| 22 | v1 | 31,055 | 117.1 | 161.0 | 13.1 s | 30.1 s | 2982 | j-9dae2f55, 2026-11-13 |
| 22 | v1 | 31,253 | 119.2 | - | 12.7 s | 28.9 s | 2985 | j-955acea7, 2026-11-18 (rerun) |
| 26 | v1 | 31,447 | 111.7 | 151.6 | 38.1 s | 50.7 s | 3155 | j-9dae2f55, 2026-11-13 |

### EP (`MOE_MODE=ep`, EP_ARM mixed, q8), 1200 s window

| Conc | tok/s/chip | p90 interactivity | p50 interactivity | p90 TTFT | Note |
| --- | --- | --- | --- | --- | --- |
| 22 | 22,509 | 104.5 | 141.6 | 30.7 s | before the fp8 w13 as_strided fix |
| 22 | 23,985 | 116.3 | 150.5 | 25.3 s | with the fix (f665a1a659) |

### Low concurrency (not yet rerun on this engine)

Launch-cut sweep, image `muse-q4mk-launchcut-80345b8a94` (recipe 80345b8a94, frozen
draft/verify compile), 2026-11-10/11:

| Conc | tok/s/chip | p90 interactivity | p90 TTFT | p90 E2E | Requests | Job |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 6,449 | 487.3 | 1.9 s | 11.3 s | 303 | j-24fcba7c |
| 7 | 12,118 | 340.2 | 4.0 s | 10.7 s | 1091 | j-500bf362 |
| 9 | 15,775 | 286.3 | 6.7 s | 13.6 s | 1403 | j-7519d90e |
| 10 | 17,139 | 228.8 | 3.4 s | 12.7 s | 1547 | j-72a31b3f |
| 16 | 23,314 | 166.4 | 8.6 s | 21.8 s | 2318 | j-e7259334 |
| 22 | 29,213 | 109.0 | 15.3 s | 33.7 s | 2859 | j-6bbec2a7 |

### Reference: InferenceX v11 submission

Run 41384211546, branch `haolin/ms-v11` @ 8c00ae865a.

| Conc | tok/s/chip | p90 interactivity | p90 E2E |
| --- | --- | --- | --- |
| 1 | 6,101 | 433 | 12.5 s |
| 7 | 11,462 | 302 | 11.1 s |
| 9 | 15,052 | 249 | 15.9 s |
| 10 | 16,555 | 199 | 16.5 s |
| 16 | 20,949 | 139 | 26.7 s |
| 22 | 25,906 | 95 | 46.4 s |
