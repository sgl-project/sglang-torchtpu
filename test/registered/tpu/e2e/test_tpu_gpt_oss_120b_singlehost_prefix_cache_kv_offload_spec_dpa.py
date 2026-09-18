"""GPT-OSS-120B full DPA + prefix cache + KV offload + EAGLE3 gate."""

import os
import sys
from copy import deepcopy
from pathlib import Path

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu import model_config_main
from sglang.test.tpu.kv_offload_probe import kv_offload_runner

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from benchmark.tpu.recipe.configs.gpt_oss_120b.server import CONFIG

register_tpu_ci(
    est_time=5400,
    suite="nightly-tpu-4-v7",
    nightly=True,
    labels=["singlehost", "gpt-oss-120b", "prefix-cache", "kv-offload", "spec", "dpa"],
)

SERVED_MODEL_NAME = "gpt-oss-120b"
BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:30000")
HEALTH_TIMEOUT = float(os.environ.get("HEALTH_TIMEOUT", "1800"))

DRAFT_MODEL_PATH = "lmsys/EAGLE3-gpt-oss-120b-bf16"
GSM8K_ACCURACY = 0.70
# Measured 2.7971 over 200 examples; same floor as the non-offload DP8 spec row.
SPEC_ACCEPT_LENGTH_THRESHOLD = float(
    os.environ.get("SPEC_ACCEPT_LENGTH_THRESHOLD", "1.90")
)
DP_SIZE = 8
DEVICE_POOL_TOKENS = 8192
HICACHE_RATIO = 3.0
PROBE_TOKENS = 2048
PAGE_SIZE = 64


LAUNCH_CONFIG = deepcopy(CONFIG)
LAUNCH_CONFIG["server_args"].update(
    {
        "context_length": 8192,
        "max_running_requests": 16,
        "torch_compile_max_bs": 16,
        "chunked_prefill_size": min(max(4096, 2048), 8192),
        "max_prefill_tokens": min(max(4096, 2048), 8192),
        "max_total_tokens": int(DEVICE_POOL_TOKENS),
        "model_path": "/weights/gpt-oss-120b",
        "page_size": int(PAGE_SIZE),
        "tpu_compile_tokens": [512, 2048, 4096],
        "tpu_compile_bs": [1, 2],
        "dp_size": int(DP_SIZE),
        "attention_backend": "pallas_tpu_rpa_v3",
        "hicache_mem_layout": "page_first_fused",
        "disable_radix_cache": False,
        "enable_hierarchical_cache": True,
        "hicache_io_backend": "raiden",
        "hicache_ratio": float(HICACHE_RATIO),
        "hicache_write_policy": "write_through",
        "enable_cache_report": True,
        "reasoning_parser": "gpt-oss",
        "served_model_name": SERVED_MODEL_NAME,
        "speculative_algorithm": "EAGLE3",
        "speculative_draft_model_path": DRAFT_MODEL_PATH,
        "speculative_num_steps": 3,
        "speculative_eagle_topk": 1,
        "speculative_num_draft_tokens": 4,
        "tpu_compile_tracer": "dynamo",
        "enable_dp_attention": True,
        "enable_dp_lm_head": CONFIG["server_args"]["tp_size"] // (int(DP_SIZE)) > 1,
    }
)

MODEL_CONFIG = {
    "model": SERVED_MODEL_NAME,
    "test_name": "gpt-oss-120b-prefix-cache-kv-offload-spec-dpa",
    "variant": "TP8+EP8+DP8(attn-tp1)+HiCache-raiden+EAGLE3",
    "launch": {
        **LAUNCH_CONFIG,
        "health_timeout": HEALTH_TIMEOUT,
        "tag": "gpt-oss-prefix-cache-kv-offload-spec-dpa-e2e",
    },
    "gates": [
        {
            "dataset": "prefix_cache_kv_offload",
            "label": "prefix-cache-kv-offload-spec-dpa",
            "runner": kv_offload_runner(
                test_name="gpt-oss-120b-prefix-cache-kv-offload-spec-dpa",
                device_pool_tokens=DEVICE_POOL_TOKENS,
                page_size=PAGE_SIZE,
                probe_tokens=PROBE_TOKENS,
                hicache_ratio=HICACHE_RATIO,
                spec_decoding=True,
                dp_size=DP_SIZE,
            ),
        },
        {
            "dataset": "gsm8k",
            "label": "gsm8k",
            "baseline": GSM8K_ACCURACY,
            "spec_accept_length_threshold": SPEC_ACCEPT_LENGTH_THRESHOLD,
            "num_examples": 200,
            "num_threads": 16,
            "max_tokens": 512,
            "temperature": 0.0,
        },
    ],
}


if __name__ == "__main__":
    model_config_main(MODEL_CONFIG, base_url=BASE_URL)
