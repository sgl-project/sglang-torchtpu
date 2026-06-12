# SGLang TorchTPU Working Notes

## TPU PR Validation

For TPU FP8, MoE, compile, scheduler, and benchmark-script PRs, use the 30B
TP8/EP8 run as the standard validation target before trying 480B. Do not create
extra SGLang repo checkouts on the pod; edit on devgpu and sync changed files
into `/mnt/hyperdisk/sglang-torchtpu`.

Launch the 30B FP8 server:

```bash
cd /mnt/hyperdisk/sglang-torchtpu/benchmark/tpu
LD_PRELOAD=/mnt/hyperdisk/_v211/abi_shim.so \
HF_HOME=/mnt/hyperdisk/.cache/huggingface \
HUGGINGFACE_HUB_CACHE=/mnt/hyperdisk/.cache/huggingface/hub \
TRANSFORMERS_CACHE=/mnt/hyperdisk/.cache/huggingface/transformers \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS=false \
MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 \
TP_SIZE=8 EP_SIZE=8 \
PAGE_SIZE=128 CONTEXT_LEN=2048 \
CHUNKED_PREFILL_SIZE=1024 MAX_PREFILL_TOKENS=1024 \
MAX_TOTAL_TOKENS=262144 MEM_FRACTION_STATIC=0.30 \
MAX_RUNNING=64 TORCH_COMPILE_MAX_BS=64 \
ENABLE_PROFILER_ENDPOINT=0 \
PYTHON=python3 \
./launch_server.sh
```

Throughput validation:

```bash
cd /mnt/hyperdisk/sglang-torchtpu/benchmark/tpu
LD_PRELOAD=/mnt/hyperdisk/_v211/abi_shim.so \
HF_HOME=/mnt/hyperdisk/.cache/huggingface \
HUGGINGFACE_HUB_CACHE=/mnt/hyperdisk/.cache/huggingface/hub \
TRANSFORMERS_CACHE=/mnt/hyperdisk/.cache/huggingface/transformers \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 \
BATCH_SIZE=64 NUM_PROMPTS=640 ISL=1024 OSL=1024 \
WARMUP_REQUESTS=0 \
PYTHON=python3 \
./bench_serving.sh
```

Eval validation:

```bash
cd /mnt/hyperdisk/sglang-torchtpu/benchmark/tpu
LD_PRELOAD=/mnt/hyperdisk/_v211/abi_shim.so \
HF_HOME=/mnt/hyperdisk/.cache/huggingface \
HUGGINGFACE_HUB_CACHE=/mnt/hyperdisk/.cache/huggingface/hub \
TRANSFORMERS_CACHE=/mnt/hyperdisk/.cache/huggingface/transformers \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 \
EVAL_NAME=mmlu NUM_EXAMPLES=100 NUM_THREADS=16 \
EXTRA_ARGS="--max-tokens 512" \
PYTHON=python3 \
./run_eval.sh
```

The default simple MMLU prompt asks for step-by-step reasoning. Do not run the
full 14k-example eval as routine PR validation; with `--max-tokens 512` it is
hours-long. `--max-tokens 128` is too short and usually truncates before the
final `Answer: X`, producing a false zero score.

For PR descriptions, include the model, TP/EP, benchmark shape, success count,
throughput, and eval score. Do not list routine lint or pre-commit commands.
