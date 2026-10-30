# Benchmarking SGLang on TPU (torch_tpu backend)

Launch + serving-benchmark scripts for the **torch_tpu** SGLang backend
(`sglang.launch_server --device tpu`). Parameterized by `MODEL` (and `TP_SIZE`) —
no per-model files. (For the separate sglang-jax engine, see
[`docs/platforms/tpu.md`](../../docs/platforms/tpu.md).)

```
launch_server.sh     # start the server
bench_serving.sh     # run sglang.bench_serving (single run or sweep)
run_eval.sh          # run sglang.test.run_eval (e.g. MMLU, GSM8K)
run_lm_eval.sh       # run lm-eval against a live server and check baselines
warmup.sh            # optional compile-bucket pre-warmup (stopgap)
warmup_buckets.py    # request-sender used by warmup.sh
```

## Usage

The Python launcher (`python -m benchmark.tpu.launch_server --config ...`)
runs a per-model recipe; [`recipe/README.md`](recipe/README.md) is the index of
them and the config format. It preserves hardware metadata such as
`TPU_TOPOLOGY`. For one-host-per-engine P/D, it removes cross-host rendezvous
settings so each engine initializes locally.
The shared serving recipes include the shell launcher's diagnostic, telemetry,
recompilation logging, and online Hugging Face defaults. Override them with
`--runtime-envs`, for example `'{"HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1"}'`.

When `enable_torch_compile` is enabled, the Python launcher defaults the tier-2
cache to `torchtpu_<model-slug>` and tier-3 to
`$HOME/.cache/torch_tpu_compile_cache`, with local backup enabled. Explicit
`TORCH_TPU_INTERNAL_TIER*` settings take precedence; `runtime_envs` entries set
to `null` remove those variables. `ENABLE_COMPILE_CACHE=0` suppresses the cache
defaults and sets `TORCH_TPU_TIER2_COMPILATION_CACHE=disabled` (unless already
set): with no tier-2 name, torch_tpu otherwise falls back to an implicit
`/dev/shm/torch_tpu_cache/default` tier-2 cache whenever `WORLD_SIZE > 1`.
Both launchers behave the same way. Cache directories are created only for a
real launch, not a dry-run.

Run launch and bench in separate shells; set `MODEL` for both.

```bash
# 1. Launch — blocks while kernels compile; /health -> 200 when ready
MODEL=Qwen/Qwen3-32B TP_SIZE=8 ./launch_server.sh

# 2. Bench — defaults to bs=64, isl=1024, osl=1024
MODEL=Qwen/Qwen3-32B ./bench_serving.sh

# Concurrency/shape sweep ("concurrency:isl:osl" per leg)
MODEL=Qwen/Qwen3-32B SWEEP="16:1024:1024 64:1024:1024" ./bench_serving.sh

# 3. Eval — defaults to EVAL_NAME=mmlu, API=chat, NUM_THREADS=4
MODEL=Qwen/Qwen3-4B ./run_eval.sh
MODEL=Qwen/Qwen3-4B EVAL_NAME=gsm8k ./run_eval.sh
```

Qwen3-Coder-30B FP8 TP8/EP8 quality eval:

```bash
MODEL=Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 \
TP_SIZE=8 EP_SIZE=8 \
PAGE_SIZE=128 CONTEXT_LEN=8192 \
CHUNKED_PREFILL_SIZE=4096 MAX_PREFILL_TOKENS=4096 \
MAX_TOTAL_TOKENS=262144 MEM_FRACTION_STATIC=0.30 \
TORCH_COMPILE_MAX_BS=64 \
./launch_server.sh

./run_lm_eval.sh
```

Qwen3-Coder-480B FP8 TP8/EP8 quality eval (weights dominate HBM, so use a
low mem-fraction and fewer running requests):

```bash
MODEL=Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8 \
TP_SIZE=8 EP_SIZE=8 \
PAGE_SIZE=128 CONTEXT_LEN=8192 \
CHUNKED_PREFILL_SIZE=4096 MAX_PREFILL_TOKENS=4096 \
MAX_TOTAL_TOKENS=131072 MEM_FRACTION_STATIC=0.05 \
MAX_RUNNING=32 TORCH_COMPILE_MAX_BS=32 \
./launch_server.sh

EVAL_CONFIG=../../test/lm_eval_configs/Qwen3-Coder-480B-A35B-Instruct-FP8-TPU.yaml \
./run_lm_eval.sh
```

Fast vs full: `run_lm_eval.sh` runs whatever tasks the `EVAL_CONFIG` lists, so
each model has two configs:
- `<Model>-TPU.yaml` — **full** quality eval (`mmlu_llama` + `mmlu_pro`); run for
  release sign-off. Needs `CONTEXT_LEN=8192` for mmlu_pro's 2048-token CoT.
- `<Model>-TPU-fast.yaml` — **fast** gate (`mmlu_llama` only, 10-token answers);
  lighter (skips the heavy CoT) and runs at `CONTEXT_LEN=4096`. Use for per-PR gating.

```bash
# fast gate (mmlu_llama only) against the same server:
EVAL_CONFIG=../../test/lm_eval_configs/Qwen3-Coder-30B-A3B-Instruct-FP8-TPU-fast.yaml \
./run_lm_eval.sh
```

(gsm8k would be the cheaper sglang-style gate, but it currently crashes the
torch_tpu server on a serve-time recompile — `InductorError: KeyError 'tpu'` for
request shapes outside the precompiled buckets — so the fast gate uses mmlu_llama
for now.)

Single-chip smoke (tp=1 auto-pins chiplet 0):

```bash
MODEL=Qwen/Qwen3-0.6B TP_SIZE=1 PAGE_SIZE=32 ./launch_server.sh
MODEL=Qwen/Qwen3-0.6B TP_SIZE=1 ./bench_serving.sh
```

TP8/EP8 MoE throughput run:

```bash
MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 \
TP_SIZE=8 EP_SIZE=8 \
PAGE_SIZE=128 CONTEXT_LEN=2048 \
CHUNKED_PREFILL_SIZE=1024 MAX_PREFILL_TOKENS=1024 \
MAX_TOTAL_TOKENS=262144 MEM_FRACTION_STATIC=0.30 \
TORCH_COMPILE_MAX_BS=64 \
NUM_CONTINUOUS_DECODE_STEPS=8 \
./launch_server.sh

MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 \
NUM_PROMPTS=640 \
./bench_serving.sh
```

TP8/EP8 480B FP8 1k/1k c=64 run:

```bash
MODEL=Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8 \
TP_SIZE=8 EP_SIZE=8 \
PAGE_SIZE=128 CONTEXT_LEN=2048 \
CHUNKED_PREFILL_SIZE=1024 MAX_PREFILL_TOKENS=1024 \
MAX_TOTAL_TOKENS=131072 MEM_FRACTION_STATIC=0.05 \
TORCH_COMPILE_MAX_BS=64 \
NUM_CONTINUOUS_DECODE_STEPS=4 \
./launch_server.sh

MODEL=Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8 \
DATASET_NAME=random-ids TOKENIZE_PROMPT=1 \
BATCH_SIZE=64 NUM_PROMPTS=64 ISL=1024 OSL=1024 \
./bench_serving.sh
```

Qwen3-Coder-480B FP8 **standard agentic benchmark** (the team's fixed contract:
multi-turn 4k-per-round workload accumulating to 16k / 256k, tp=16 dp=4 colocated
and tp=8 dp=2 P/D, prefix caching + KV offload, one results JSON, wired to a
nightly workflow) — see
[`recipe/qwen3_coder_480b/nightly/`](recipe/qwen3_coder_480b/nightly/README.md):

```bash
cd recipe/qwen3_coder_480b/nightly
PROFILE=smoke MODE=colocated ./run_nightly.sh     # plumbing check
SGLANG_USE_RAY=1 NNODES=2 MODE=colocated ./run_nightly.sh
```

## Pipeline parallelism (PP8 x TP1)

`--pp-size N --tp-size 1` runs N pipeline stages, one chiplet per stage, on
one host. Each stage owns `num_hidden_layers / N` layers and its own KV pool
for those layers; the stages hand their activations (hidden states and
residual summed into one buffer, padded to the largest token bucket under
`--enable-torch-compile`) to the next stage over ICI, as one `jax.lax.ppermute`
collective-permute ring per iteration (`pp_wave.py`), and the last stage
samples. At startup each device logs
`pipeline wave stage r/N column c/K: hand-off cycles (...)`; the gloo
`cpu_group` carries only the last stage's output dict and control traffic.
Set `--pp-max-micro-batch-size` so that `max_running_requests /
pp_max_micro_batch_size >= pp_size`: the pipeline is full only when at least
one microbatch per stage is in flight (the default is `max_running_requests /
pp_size`, i.e. half the stages idle at 4-request microbatches). Long prompts
are still split by `--chunked-prefill-size`.

```bash
MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 \
TP_SIZE=1 PP_SIZE=8 PP_MAX_MICRO_BATCH_SIZE=4 \
PAGE_SIZE=128 CONTEXT_LEN=32768 CONC=32 \
CHUNKED_PREFILL_SIZE=16384 MAX_PREFILL_TOKENS=16384 \
MAX_TOTAL_TOKENS=1048576 MEM_FRACTION_STATIC=0.30 \
TORCH_COMPILE_MAX_BS=32 \
EXTRA_ARGS="--sampling-backend pytorch --watchdog-timeout 3600" \
./launch_server.sh
```

Supported: chunked prefill, `--kv-cache-dtype fp8_e4m3`, `--max-total-tokens`
(every stage sizes its pool to the smallest stage; a hybrid model's SWA pool
holds one prefill chunk per pipeline slot on top of `--swa-full-tokens-ratio`).
Rejected at launch with `--pp-size > 1` on TPU: eager mode (without
`--enable-torch-compile`),
`--tp-size > 1` unless a stage is a tensor-parallel pair or one
context-parallel group (as below), speculative decoding, `--dcp-size > 1`,
and `--enable-dynamic-chunking`.
Input (prompt) logprobs are rejected per request.
`--dp-size` has not been run with pipeline parallelism on
TPU. Overlap scheduling is switched off by pipeline parallelism itself. A
stage waiting for the previous stage looks idle to the watchdog, so pass
`--watchdog-timeout` at least `pp_size` times the longest microbatch.

Optional warmup — run between launch and bench *only* while the in-server
torch.compile warmup doesn't yet cover all compile-shape buckets:

```bash
MODEL=Qwen/Qwen3-32B ./warmup.sh
```

### Context-parallel stages (PP4 x PCP2)

Requires `--tp-size` equal to `--attn-cp-size`, one host, a bf16 KV cache, and a
torch_tpu with standalone process subgroups, so that a stage's collectives run
on the stage's devices only.

```bash
MODEL=openai/gpt-oss-120b QUANTIZATION=unquant KV_CACHE_DTYPE=bfloat16 \
TP_SIZE=2 PP_SIZE=4 PP_MAX_MICRO_BATCH_SIZE=4 \
ATTENTION_BACKEND=pallas_tpu_rpa_batched PAGE_SIZE=256 CONTEXT_LEN=33000 CONC=64 \
CHUNKED_PREFILL_SIZE=16384 MAX_PREFILL_TOKENS=16384 \
MAX_TOTAL_TOKENS=524288 MEM_FRACTION_STATIC=0.8 \
EXTRA_ARGS="--attn-cp-size 2 --enable-prefill-cp --cp-strategy zigzag --moe-dp-size 2 --watchdog-timeout 3600" \
./launch_server.sh
```

### Tensor-parallel stages (PP4 x TP2)

`--tp-size 2` with `--attn-cp-size 1` makes each stage a tensor-parallel pair
that shards the stage's heads and expert weights (no DP attention or
`--ep-size`); it needs the same torch_tpu as the context-parallel stages.

```bash
MODEL=openai/gpt-oss-120b QUANTIZATION=unquant SGLANG_TPU_FUSED_MOE_SC_GATHER=0 \
TP_SIZE=2 PP_SIZE=4 PP_MAX_MICRO_BATCH_SIZE=4 KV_CACHE_DTYPE=bfloat16 \
ATTENTION_BACKEND=pallas_tpu_rpa_batched PAGE_SIZE=256 CONTEXT_LEN=33000 CONC=64 \
CHUNKED_PREFILL_SIZE=16384 MAX_PREFILL_TOKENS=16384 MAX_TOTAL_TOKENS=524288 \
TPU_COMPILE_TOKENS="2048 4096 8192 16384" \
EXTRA_ARGS="--sampling-backend pytorch --watchdog-timeout 3600" \
./launch_server.sh
```

### Prefill chunk sizing

With a positive `--chunked-prefill-size`, the stages profile their forward
costs at startup and size each prefill step from the slowest stage: the step
is bounded by the pending prompt tokens divided by the stage count, by a
fixed compute-time budget, and by the batched attention kernel's schedule
capacity, so few requests get small steps and later chunks of a long prompt
shrink with their attention cost. `SGLANG_TPU_PP_DYNAMIC_CHUNKS=0` keeps
fixed token budgets; `SGLANG_TPU_PP_CHUNK_SLACK` (default `0.1`) is the slack
above the profiled full-step time.

## Notes

- Needs `torch_tpu` + `sglang` importable by `PYTHON` (default `python3`). For a
  Bazel-built `torch_tpu`, set `JAX_BAZEL=<bazel-output-user-root>/<hash>/external`.
  Run offline against a local cache with `HF_HOME=... HF_HUB_OFFLINE=1`.
- Every knob is an env var — see each script's header comment for the full list.
- `bench_serving.sh` writes JSONL summaries under `./results/<model-slug>-tpu/`
  (gitignored).

## Reproducibility metadata (for the CI dashboard)

Both scripts emit a versioned JSON record (`dump_meta.py`) so every benchmark
data point carries enough to reproduce it — no more "what commit / libtpu /
flags was that run on?". Enabled by default; set `DUMP_META=0` to skip.

- `launch_server.sh` → `$RUN_DIR/launch_meta.json` (`kind: "launch"`): resolved
  server knobs, the exact `server_argv`, the repro-critical env
  (`LIBTPU_INIT_ARGS`, `XLA_FLAGS`, topology, `SGLANG_*`, `TORCH_TPU_*`), git
  commit/dirty of the repo + `torch_tpu`, and package versions.
- `bench_serving.sh` → `$OUTPUT_DIR/<tag>.meta.json` (`kind: "bench"`), one per
  sweep leg beside its metrics `<tag>.jsonl`: workload/bench args, git +
  versions, and a **live pull of the server's `/server_info`** (the authoritative
  running config, so launch and bench can't silently disagree).

Every record has a top-level `schema_version` and a stable shape
(`git` / `versions` / `config` / `env` [/ `server_info`]). A CI dashboard
ingests the `*.meta.json` files plus the metrics `<tag>.jsonl` and joins them by
`(hostname, model, git commit, timestamp)`.

**Secrets are redacted.** The `env` capture is a blanket prefix match, and
`/server_info` echoes the full server args, so a credential set in the
environment (e.g. `SGLANG_BEARER_TOKEN`, `SGLANG_S3_SECRET_ACCESS_KEY`) or an
`api_key` would otherwise land in the record and flow to the shared dashboard.
Any field whose name reads as a credential (`*_token`, `*secret*`, `*password*`,
`*_access_key`, `api_key`, …) has its value replaced with `***redacted***`
before the record is written.

## Kimi INT4-resident FP8 MoE

For a group-32 INT4 Kimi checkpoint, set the mode before launching the server:

```bash
export SGLANG_TPU_INT4_MOE_MODE=vmem-fp8
```

This mode keeps packed INT4 expert weights in HBM. At load time it prepares
normalized group scales and output-channel scales. The existing
`sglang/srt/layers/moe/pallas/gmm_v2.py` unpacks and scales each weight tile in
VMEM, converts it to FP8, and multiplies it by activations quantized in K=512
groups. It restores output-channel scales before the optional fused activation.
This is W4A8 compute with BF16 partial accumulation; no second GMM implementation
or full FP8 copy of the expert weights is required.

Routing, padding and tiling are the same as in the other modes; measured at one
token the general path matched the earlier Kimi-specific fast path, so there is
none.

The other modes retain their behavior: unset uses W4A16 group products;
`vmem-bf16` dequantizes each INT4 group in VMEM to BF16 and leaves activations
in BF16.
