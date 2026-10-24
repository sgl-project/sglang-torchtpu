# Reflection Beam-1.0-501B-A23B-FP8 on TPU

Single-host serving recipe for **Beam-1.0-501B-A23B-FP8** on `tpu7x` (one 2x2x1
host = 8 devices x ~101.7 GB): tp8 / ep8, pure TP attention, compressed-tensors
FP8 (per-channel weights, dynamic per-token activations), `pallas_tpu_rpa_v3`
with hybrid SWA (43 sliding layers, window 512) / full (9 NoPE layers) KV pools.

## Files
| file | what |
|---|---|
| `launch_beam_fp8.sh` | single host, tp8 / ep8, bf16 KV, 8K context, max_running 32 |
| `launch_beam_bf16_fp8_online.sh` | BF16 checkpoint quantized to FP8 at load, then exactly `launch_beam_fp8.sh` |
| `launch_beam_bf16_2host.sh` | BF16 served as BF16 on a 2x2x2 slice (2 hosts), tp16 / ep8 x MoE-TP2 |
| `fp8_dynamic_quantization_config.json` | the official FP8_DYNAMIC `quantization_config`, injected by the online recipe |
| `check_online_fp8_parity.py` | CPU check: online-quantized BF16 == the FP8 checkpoint, byte for byte |
| `launch_beam_nvfp4.sh` | NVFP4 checkpoint, single host, tp8 / ep8, fused-RS W4A16 experts, 128K context, max_running 64 |

## Bring-up checklist

1. **Weights on local disk.** ~503 GB in 188 shards. Serving straight from the
   gcsfuse mount works (`--weight-loader-disable-mmap` is set) but loads slowly;
   copy first:
   ```bash
   gcloud storage cp -r gs://tpu-pod-gcs-shared/models/Beam-1.0-501B-A23B-FP8 /mnt/disks/local/
   ls /mnt/disks/local/Beam-1.0-501B-A23B-FP8/config.json
   ```
2. **Launch** (wait for `The server is fired up and ready to roll!`):
   ```bash
   BEAM_LOCAL_MODEL=/mnt/disks/local/Beam-1.0-501B-A23B-FP8 ./launch_beam_fp8.sh
   # or from the gcsfuse mount: ./launch_beam_fp8.sh   (MODEL=/gcs/Beam-1.0-501B-A23B-FP8)
   ```
   Check in the log: `quant_method` resolves to compressed-tensors (no
   `--quantization fp8`), the KV pool is an `SWAKVPool` (full + swa), and the
   EP-local expert reader reads 32 of 256 experts per rank.
3. **Smoke curl:**
   ```bash
   curl -s localhost:30000/v1/chat/completions -H 'Content-Type: application/json' -d '{
     "max_tokens": 256, "temperature": 0,
     "reasoning_effort": "low",
     "messages": [{"role": "user", "content": "What is 17 * 23? Answer with the number."}]}'
   ```
   Expect `391` in the final channel. Then a >512-token prompt to exercise the
   sliding window, and a >2048-token one so the SSMax scale is well above 1.
4. **GSM8K** (200 questions):
   ```bash
   MODEL=/mnt/disks/local/Beam-1.0-501B-A23B-FP8 EVAL_NAME=gsm8k NUM_EXAMPLES=200 NUM_THREADS=16 \
     EXTRA_ARGS="--max-tokens 4096 --reasoning-effort low" ../../run_eval.sh
   # MODEL must match the served path (the --model-path the server got).
   ```
   Record the score in
   `test/registered/tpu/e2e/test_tpu_beam_501b_singlehost_gsm8k.py` and enable it.
5. **A/B the Pallas kernels** (sandwich residual norm, q/k norm + RoPE/SSMax,
   grouped top-k router): `SGLANG_TPU_BEAM_KERNELS=0 ./launch_beam_fp8.sh` runs
   every Beam op on its torch reference; compare GSM8K and greedy outputs.

## Notes / gotchas
- **`QUANTIZATION` must stay empty.** `launch_server.sh` defaults to
  `--quantization fp8`, which would bypass the checkpoint's compressed-tensors
  config.
- **No DP attention.** `BeamForCausalLM` rejects it (the dense MLP and MoE are
  sharded over the full TP group), and DFlash speculative decoding (phase 2)
  rejects it too.
- **Window.** The model passes `sliding_window_size=511`; RPA adds one, so each
  query attends to 512 keys including itself, as in the GPU reference.
- **KV dtype.** bf16 by default for bring-up. At tp8 there is one KV head per
  rank and fp8 KV pads it 2x; the checkpoint ships no KV scales. Try
  `KV_CACHE_DTYPE=fp8_e4m3` once bf16 accuracy is recorded.
- **Precompile buckets.** Tokens `64..4096` (chunked prefill 4096), batch
  `1..32`. A prefill shape outside the buckets takes the recompile path, which
  fails after warmup; widen `TPU_COMPILE_TOKENS` together with
  `CHUNKED_PREFILL_SIZE`.
- **Fused reduce-scatter MoE** (`EXTRA_ARGS="--moe-runner-backend pallas_tpu_fused_rs"`)
  is wired (EP == TP == 8, no DP attention, 256 experts) but untested.

## BF16 checkpoint (`reflection/Beam-1.0-501B-A23B`, 1002.66 GB)

Two ways to serve it; pick by purpose.

| | FP8 at load, 1 host (`launch_beam_bf16_fp8_online.sh`) | BF16, 2 hosts (`launch_beam_bf16_2host.sh`) |
|---|---|---|
| chips | 4 (one 2x2x1) | 8 (one 2x2x2) |
| weights / device | ~63 GB, identical to the FP8 checkpoint | ~63 GB BF16 |
| numerics | byte-identical to Beam-1.0-501B-A23B-FP8 | BF16 reference (no weight or activation quantization) |
| decode / prefill | the FP8 baseline exactly | modeled: ITL +~1-2 ms/step (16-way cross-host reductions; same HBM bytes per device as FP8 at ep8), prefill ~0.75-1.0x per slice, i.e. about half per chip |
| load | streams ~1.1 TB from gcsfuse and quantizes on the host | `STAGE=1`: ~506 GB sparse slice per host into /dev/shm, then a /dev/shm load |
| spec decode | DFlash possible (tp8) | no DFlash (40 draft heads over 16 ranks) |

**FP8 at load.** The recipe injects the official `quantization_config` with
`--json-model-override-args`; `BeamForCausalLM.load_weights` then quantizes each
BF16 linear on the host as it streams (per-output-channel scale = amax/448,
quotient rounded in BF16 before the FP8 cast, as llm-compressor does) and hands
the loader exactly the FP8 checkpoint's names and shards. `lm_head`,
`embed_tokens` and the router `mlp.gate` stay BF16/FP32 through the same ignore
list. BF16 never reaches HBM. Before relying on it, confirm parity on a CPU host
that mounts both checkpoints:
```bash
python benchmark/tpu/recipe/beam/check_online_fp8_parity.py \
    --bf16 /gcs/Beam-1.0-501B-A23B --fp8 /gcs/Beam-1.0-501B-A23B-FP8 --sample 400
```
For the stock model this proves the official FP8 checkpoint is the cheaper
artifact to stage (503 GB fits /dev/shm, 40 s loads); the online path is for
BF16-only derivatives.

**BF16 on 2 hosts.** Run the recipe on both hosts (`kubectl exec` into each
tpuq pod, or the CI JobSet); ranks and the rendezvous come from
`TPU_WORKER_ID` / `TPU_WORKER_HOSTNAMES`. 1.0 TB does not fit one host's
/dev/shm, but EP-local reads need only that host's 128 experts, so
`STAGE=1` runs `python -m sglang_tpu.stage_ep_local_checkpoint` first: shards are
recreated sparse (holes for the other host's experts). Do not point any other
loader at the staged copy. With 48 q / 8 kv heads over 16 ranks every KV head
lives on 2 ranks: the attention uses `QKVParallelLinear` plus a separate gate
GEMM instead of the packed q|k|v|gate GEMM.

## NVFP4 checkpoint (`reflection/Beam-1.0-501B-A23B-NVFP4`, ~294 GB)

ModelOpt NVFP4 on the routed experts only (packed E2M1 weights, e4m3 block-16
scales, fp32 `weight_scale_2`, static `input_scale`); attention, router, dense
MLPs (layers 0-2) and `lm_head` stay BF16 through the checkpoint's ignore list.
`TpuSRTPlatform` maps `modelopt_fp4` to `sglang_tpu.modelopt_fp4`.

| | FP8 checkpoint (`launch_beam_fp8.sh`) | NVFP4 (`launch_beam_nvfp4.sh`) |
|---|---|---|
| weights / device | ~63 GB | ~41 GB (experts 38.5 GB, BF16 dense 1.6 GB) |
| experts | fp8 x fp8 MXU, gmm_v2 + all-reduce | W4A16: packed FP4 in HBM, bf16 in VMEM, fused reduce-scatter kernel |
| decode (HBM roofline, per step) | 1.0x | ~1.25x faster at bs 1, ~1.5x at bs >= 32 |
| prefill (roofline, per chunk) | 1.0x | ~1.0x at 4096-token chunks, ~1.7x slower at 8192 |
| KV pool / device | ~17 GB | ~40 GB |
| spec decode | DFlash possible | not yet (fused-RS has no spec-decoding support, and group-16 FP4 has no other TPU MoE kernel) |

Bring-up deltas from the FP8 checklist:
- Expect `TpuModelOptFp4MoEMethod` / `Using TPU Pallas NVFP4 fused-RS MoE backend`
  in the log, and the EP-local reader reading 32 of 256 experts per rank (~51 GB
  per rank instead of 294 GB).
- `[GMM v2 fused_rs] using default block sizes ... rhs_bs-16` is expected until
  Beam's shapes get a tuned entry in
  `sglang_tpu/moe/pallas/fused_rs/v2_fused_rs_tuned_block_sizes.py`; sweep with
  `torchrun ... test/manual/tpu/test_beam_nvfp4_moe.py --bench`.
- Activations stay BF16 (the GPU NVFP4 path quantizes them to FP4), so expect
  GSM8K at or above the GPU NVFP4 reference.
- Keep `CHUNKED_PREFILL_SIZE` at or below 4096 (the W4A16 expert GEMMs are
  compute-bound above ~3K tokens per chunk).
