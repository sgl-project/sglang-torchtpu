"""Inspect checkpoint metadata and freeze a native Torch GLM-5.2 MTP recipe."""

import argparse
import hashlib
import json
import struct
from pathlib import Path

from benchmark.tpu.recipe.configs.base import LIBTPU_FLAGS
from benchmark.tpu.glm52_baseline.draft_view import prepare_draft_view


def prepare(model: Path, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads((model / "config.json").read_text())
    index = json.loads((model / "model.safetensors.index.json").read_text())
    quant = dict(config.get("quantization_config") or {})
    # get_config replaces this object; preserve exclusion lists from the checkpoint.
    quant.update(
        quant_method="fp8",
        activation_scheme="dynamic",
        fmt="e4m3",
        weight_block_size=None,
        weight_scale_granularity="channel",
    )
    layers = config["num_hidden_layers"]
    mapping = index["weight_map"]
    sample_layers = {0, config.get("first_k_dense_replace", 3), layers - 1, layers}
    interesting = []
    for name in mapping:
        if name in ("lm_head.weight", "model.embed_tokens.weight"):
            interesting.append(name)
            continue
        parts = name.split(".")
        if len(parts) < 4 or parts[:2] != ["model", "layers"] or not parts[2].isdigit():
            continue
        if int(parts[2]) not in sample_layers:
            continue
        if ".experts." not in name or ".experts.0." in name:
            interesting.append(name)
    files = sorted({mapping[name] for name in interesting})
    headers = {}
    for name in files:
        with (model / name).open("rb") as f:
            length = struct.unpack("<Q", f.read(8))[0]
            assert 0 < length < 64 * 1024 * 1024, (name, length)
            headers[name] = json.loads(f.read(length))
    selected = {name: headers[mapping[name]][name] for name in interesting}
    for name, header in selected.items():
        if not name.endswith(".weight") or header["dtype"] != "F8_E4M3":
            continue
        scale_name = name.removesuffix(".weight") + ".weight_scale_inv"
        assert scale_name in selected, (name, "missing channel scales")
        assert selected[scale_name]["shape"] == [header["shape"][0]], (
            name,
            header,
            selected[scale_name],
        )
    assert any(f"layers.{layers}." in name for name in mapping), (
        "missing native MTP layer"
    )
    evidence = {
        "model_path": str(model),
        "config": config,
        "metadata_sha256": {
            name: hashlib.sha256((model / name).read_bytes()).hexdigest()
            for name in (
                "config.json",
                "model.safetensors.index.json",
                "tokenizer_config.json",
            )
        },
        "selected_tensor_headers": selected,
        "header_files": files,
        "override_quantization_config": quant,
    }
    (output / "checkpoint-audit.json").write_text(json.dumps(evidence, indent=2) + "\n")
    draft = prepare_draft_view(model, output / "native-mtp-shards")
    args = dict(
        model_path=str(model),
        device="tpu",
        dtype="bfloat16",
        quantization="fp8",
        load_format="safetensors",
        trust_remote_code=True,
        host="0.0.0.0",
        port=30000,
        tp_size=16,
        attn_dp_size=16,
        ep_size=16,
        nnodes=2,
        attention_backend="pallas_tpu_mla_sparse",
        moe_runner_backend="pallas_tpu",
        kv_cache_dtype="fp8_e4m3",
        page_size=64,
        context_length=92160,
        chunked_prefill_size=32768,
        max_prefill_tokens=32768,
        max_running_requests=64,
        torch_compile_max_bs=4,
        tpu_compile_bs=[1, 2, 3, 4],
        tpu_compile_tokens=[64, 2048],
        enable_torch_compile=True,
        disable_radix_cache=False,
        mem_fraction_static=0.85,
        disable_shared_experts_fusion=True,
        speculative_algorithm="NEXTN",
        speculative_draft_model_path=str(draft),
        speculative_draft_model_quantization="fp8",
        speculative_num_steps=5,
        speculative_eagle_topk=1,
        speculative_num_draft_tokens=6,
        enable_dp_lm_head=True,
        grammar_backend="none",
        watchdog_timeout=1800,
        weight_loader_disable_mmap=True,
        weight_loader_prefetch_checkpoints=False,
        enable_eplb=False,
        model_loader_extra_config=json.dumps(
            {"enable_multithread_load": True, "num_threads": 1}
        ),
        enable_metrics=True,
        log_level="info",
        decode_log_interval=1,
        json_model_override_args=json.dumps({"quantization_config": quant}),
    )
    env = dict(
        SGLANG_PLATFORM="tpu",
        PYTHONUNBUFFERED="1",
        SGLANG_TPU_CHECK_RECOMPILATION="1",
        SGLANG_TPU_ENABLE_EXPERIMENTAL_SPEC_WORKER="1",
        SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION="0",
        SGLANG_TPU_PRELOAD_TELEMETRY_STUB="1",
        TPU_ENABLE_DIAGNOSTIC_COORDINATOR="0",
        TPU_DISABLE_DIAGNOSTIC_COORDINATOR="1",
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
        TORCH_LOGS="+recompiles",
        LIBTPU_INIT_ARGS=" ".join(
            (
                *LIBTPU_FLAGS,
                "--xla_tpu_buffer_assignment_enable_fast_merge_fallback=false",
            )
        ),
    )
    (output / "server.json").write_text(
        json.dumps(dict(server_args=args, runtime_envs=env), indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "layers": layers,
                "audited_headers": len(files),
                "quant": quant,
                "tensor_examples": dict(list(selected.items())[:12]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=Path, default=Path("/models/GLM5.2-fp8-channel-wise")
    )
    parser.add_argument("--output", type=Path, required=True)
    a = parser.parse_args()
    prepare(a.model, a.output)
