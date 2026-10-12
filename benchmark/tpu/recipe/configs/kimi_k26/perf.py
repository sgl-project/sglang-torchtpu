"""Kimi-K2.6 throughput config: tp8/ep8 on ONE tpu7x host, int4 experts.

    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/kimi_k26/perf.py

NOT YET VALIDATED AT tp8. The int4 experts measured 40.9 GB per rank at tp16, so
~82 GB of a rank's 101.7 GB here, and the KV pool plus the 8192-token extend
graph share what is left. If bring-up dies in the precompile, lower CONCURRENCY
(that is the pool) before mem_fraction_static -- the weights still have to fit.

Text only: MoonViT is in the checkpoint, but the sweep sends no images, so the
vision bucket ladder is switched off below rather than warmed at startup.
"""

from copy import deepcopy

from benchmark.tpu.recipe.configs.base import _FP8_BASE_CONFIG

INPUT_TOKENS = 8192
OUTPUT_TOKENS = 1024
CONCURRENCY = 16
CONTEXT_LENGTH = INPUT_TOKENS + OUTPUT_TOKENS + 64

CONFIG = deepcopy(_FP8_BASE_CONFIG)
CONFIG["server_args"].update(
    {
        # No public HF id for this checkpoint, so the mount is also the served
        # name; the dashboard joins the series on the test's `id` regardless.
        "model_path": "/weights/Kimi-K2.6",
        "tp_size": 8,
        # ep=tp. The tp16 recipe's warning against ep8 is about
        # tp-within-expert=2, whose per-rank intermediate has no 128-aligned
        # divisor for the gmm tile; at tp8/ep8 that factor is 1.
        "ep_size": 8,
        # Native int4 (compressed-tensors): "unquant" asks for no online
        # quantization, and the checkpoint's own quant_method still wins. fp8
        # here would silently mis-load the experts.
        "quantization": "unquant",
        "attention_backend": "pallas_tpu_mla",
        "page_size": 128,
        "max_running_requests": CONCURRENCY,
        "torch_compile_max_bs": CONCURRENCY,
        "context_length": CONTEXT_LENGTH,
        "chunked_prefill_size": INPUT_TOKENS,
        "max_prefill_tokens": INPUT_TOKENS,
        "tpu_compile_tokens": [INPUT_TOKENS],
        "tpu_compile_bs": [CONCURRENCY],
        "mem_fraction_static": 0.8,
        # Custom config/tokenizer/processor code ships in the checkpoint.
        "trust_remote_code": True,
        # The VL warmup exercises the multimodal embed-merge path.
        "skip_server_warmup": True,
        "weight_loader_drop_cache_after_load": True,
    }
)
# Empty skips the MoonViT patch ladder (57 warmups, ~14 min); the sweep sends no images.
CONFIG["runtime_env"].update({"SGLANG_TPU_VIT_PRECOMPILE_SIZES": ""})
