"""Colocated and P/D server settings for the agentic nightly."""

from copy import deepcopy

from benchmark.tpu.recipe.configs.qwen3_coder_480b.nightly.config import (
    CONFIG as BASE_CONFIG,
)

CONFIG = deepcopy(BASE_CONFIG)
PREFILL_CONFIG = deepcopy(CONFIG)
PREFILL_CONFIG["server_args"].update(
    nnodes=1,
    tp_size=8,
    ep_size=8,
    port=30000,
    disaggregation_mode="prefill",
    disaggregation_transfer_backend="tpu",
    disaggregation_bootstrap_port=8998,
)
DECODE_CONFIG = deepcopy(CONFIG)
DECODE_CONFIG["server_args"].update(
    nnodes=1,
    tp_size=8,
    ep_size=8,
    port=30001,
    disaggregation_mode="decode",
    disaggregation_transfer_backend="tpu",
)
