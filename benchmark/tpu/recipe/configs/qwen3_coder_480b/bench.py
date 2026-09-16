"""Settings for the 480B 8k/1k serving benchmark client."""

INPUT_TOKENS = 8192
OUTPUT_TOKENS = 1024
CONCURRENCY = 64

CONFIG = {
    "bench_args": {
        "backend": "sglang",
        "model": "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8",
        "dataset_name": "random-ids",
        "tokenize_prompt": True,
        "random_input_len": INPUT_TOKENS,
        "random_output_len": OUTPUT_TOKENS,
        "max_concurrency": CONCURRENCY,
        "num_prompts": CONCURRENCY * 10,
    },
    "runtime_env": {},
}
