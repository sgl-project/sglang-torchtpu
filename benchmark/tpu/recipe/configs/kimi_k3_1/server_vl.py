"""Kimi-K3.1 with image input on four tpu7x hosts, tp32/ep8.

kimi_k3_1/server.py serves text only and answers a request with image input
with a 400. This recipe serves text and images: the same native path and
sizing, plus

* the input_embeds EXTEND graphs (SGLANG_TPU_VL_TEXT_ONLY_EXTEND=0), which
  merge the ViT output into the prompt;
* the splash vision attention (SGLANG_TPU_ENABLE_VISION_SPLASH_ATTENTION=1).
  Kimi-K3's processor admits up to 65536 patches per image, and the flash op
  fails to compile ViT buckets above ~16k patches (XLA scoped-VMEM OOM at
  33856). With splash the default ViT ladder runs up to 66564 patches;
* up to 32 running requests (decode buckets 1..32), the size the image runs
  measured; the extra EXTEND and ViT programs leave less host RAM to spare
  (see below).

Every host runs

    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/kimi_k3_1/server_vl.py
    # node_rank / dist_init_addr come from TPU_WORKER_ID / TPU_WORKER_HOSTNAMES.

Measured on 16 chips (2026-11-04), reasoning effort max: OCRBench 0.874 (all
1000), MMMU-Pro standard 10-option smoke 0.860 (50 questions), no request
errors. Startup took ~37 min, most of it compiling the two EXTEND variants and
the six decode buckets. With the 444 GB/host expert-parallel weight shard in
/dev/shm, host MemAvailable bottomed out at ~160 GB per host during startup
(end of the decode compile), against a 140 GB guard; it recovers to ~280 GB
once the server is up.
"""

from copy import deepcopy

from benchmark.tpu.recipe.configs.kimi_k3_1.server import CONFIG as K31_CONFIG

CONCURRENCY = 32
# Powers of two up to CONCURRENCY, as in kimi_k3_1/server.py.
DECODE_BS = [1, 2, 4, 8, 16, CONCURRENCY]

CONFIG = deepcopy(K31_CONFIG)
CONFIG["server_args"].update(
    {
        "max_running_requests": CONCURRENCY,
        "torch_compile_max_bs": CONCURRENCY,
        "tpu_compile_bs": DECODE_BS,
        # The TPU backend compiles and warms every bucket, ViT included, at
        # startup; the HTTP warmup (an image request) would add nothing.
        "skip_server_warmup": True,
    }
)
CONFIG["runtime_env"].update(
    {
        # Compile the input_embeds EXTEND graphs as well, and admit image requests.
        "SGLANG_TPU_VL_TEXT_ONLY_EXTEND": "0",
        # Block-skipping splash for the ViT; it also lifts the default ViT
        # precompile ladder (SGLANG_TPU_VIT_PRECOMPILE_SIZES unset) to
        # 1024, 2304, 4096, 9216, 16384, 33856, 66564 patches.
        "SGLANG_TPU_ENABLE_VISION_SPLASH_ATTENTION": "1",
    }
)
