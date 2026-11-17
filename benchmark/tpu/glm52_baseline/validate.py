"""Resolve native server arguments before reading model tensor payloads."""

import argparse
import json
import os
from pathlib import Path

from benchmark.tpu.launch_server import LaunchConfig, resolve_server
from benchmark.tpu.recipe.configs.config_eval import server_args_to_cli


def main():
    p = argparse.ArgumentParser()
    p.add_argument("config", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    cfg, env = resolve_server(
        LaunchConfig.from_dict(json.loads(a.config.read_text())), os.environ
    )
    os.environ.update(env)
    from sglang.srt.plugins import load_plugins
    from sglang.srt.server_args import prepare_server_args

    load_plugins()
    args = prepare_server_args(server_args_to_cli(cfg.server_args))
    args.resolve_once()
    resolved = args.resolved_dict()
    a.output.write_text(json.dumps(resolved, indent=2, default=str) + "\n")
    expected = dict(
        tpu_fp8_runtime="channelwise",
        chunked_prefill_size=2048,
        max_running_requests=64,
        tp_size=16,
        attn_dp_size=16,
        ep_size=16,
        nnodes=2,
        speculative_algorithm="EAGLE",
        speculative_num_steps=5,
        speculative_num_draft_tokens=6,
        speculative_draft_model_quantization="fp8",
        enable_dp_lm_head=True,
        kv_cache_dtype="fp8_e4m3",
        page_size=64,
        disable_radix_cache=False,
        moe_runner_backend="pallas_tpu",
        attention_backend="pallas_tpu_mla_sparse",
    )
    assert not json.loads(resolved.get("json_model_override_args") or "{}"), (
        resolved.get("json_model_override_args")
    )
    for k, v in expected.items():
        assert resolved.get(k) == v, (k, resolved.get(k), v)
    print("RESOLVED_CLI_PASS", json.dumps({k: resolved.get(k) for k in expected}))


if __name__ == "__main__":
    main()
