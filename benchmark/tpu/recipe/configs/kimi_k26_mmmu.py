"""Kimi MMMU-Pro: independent TP16/EP16 replicas, no DP attention."""

ENV = {
    "MODEL": "/weights/Kimi-K2.6",
    "NNODES": "2",
    "TP_SIZE": "16",
    "EP_SIZE": "16",
    # The two CI jobs split the dataset; each server has one attention group.
    "DP_SIZE": "1",
    "QUANTIZATION": "",
    "KV_CACHE_DTYPE": "fp8_e4m3",
    "ATTENTION_BACKEND": "pallas_tpu_mla",
    "PAGE_SIZE": "128",
    "MEM_FRACTION_STATIC": "0.88",
    "MAX_RUNNING": "128",
    "CONTEXT_LEN": "106496",
    "CHUNKED_PREFILL_SIZE": "4096",
    "MAX_PREFILL_TOKENS": "4096",
    "TPU_COMPILE_TOKENS": "4096",
    "TPU_COMPILE_BS": "1 16 32 64 128",
    "TORCH_COMPILE_MAX_BS": "128",
    "DISABLE_RADIX_CACHE": "1",
    "ENABLE_PROFILER_ENDPOINT": "0",
    "SGLANG_TPU_MM_EMBED_ON_DEVICE": "1",
    "SGLANG_ENABLE_EMBED_REPLICATION": "1",
    "SGLANG_TPU_VIT_PRECOMPILE_SIZES": "256,400,576,784,1024,1600,2304,3136,4096,6400,9216,12544,16900",
    "EXTRA_ARGS": "--trust-remote-code --skip-server-warmup --scheduler-recv-interval 8 --weight-loader-drop-cache-after-load",
}
