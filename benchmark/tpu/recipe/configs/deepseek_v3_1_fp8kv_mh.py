"""V3.1 thinking: the same TP16/EP16 FP8-KV settings as R1-0528."""

from benchmark.tpu.recipe.configs.deepseek_r1_0528_fp8kv_mh import ENV as R1_ENV

ENV = {
    **R1_ENV,
    "MODEL": "/weights/DeepSeek-V3.1",
    "EXTRA_ARGS": "--reasoning-parser deepseek-v3 --weight-loader-drop-cache-after-load",
}
