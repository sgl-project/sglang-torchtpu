#!/usr/bin/env python3
"""Inspect GLM checkpoint metadata without loading weights or requiring torch.

This is an evidence collector, not a declaration of loader compatibility.
Only config/index JSON and safetensors headers are read. No model is modified.
"""
import argparse
import collections
import hashlib
import json
import struct
from pathlib import Path


def inspect(root: Path) -> dict:
    config_bytes = (root / "config.json").read_bytes()
    index_bytes = (root / "model.safetensors.index.json").read_bytes()
    config = json.loads(config_bytes)
    index = json.loads(index_bytes)
    tensors = {}
    headers = {}
    for shard in sorted(set(index["weight_map"].values())):
        path = (root / shard).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError(f"shard escapes checkpoint: {shard}")
        with path.open("rb") as stream:
            size_bytes = stream.read(8)
            if len(size_bytes) != 8:
                raise ValueError(f"truncated safetensors prefix: {shard}")
            size = struct.unpack("<Q", size_bytes)[0]
            if size > 256 * 1024 * 1024:
                raise ValueError(f"oversized safetensors header: {shard}: {size}")
            raw = stream.read(size)
            if len(raw) != size:
                raise ValueError(f"truncated safetensors header: {shard}")
        header = json.loads(raw)
        headers[shard] = hashlib.sha256(raw).hexdigest()
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            if name in tensors:
                raise ValueError(f"duplicate tensor: {name}")
            tensors[name] = {"shape": meta["shape"], "dtype": meta["dtype"], "shard": shard}
    missing = sorted(set(index["weight_map"]) - tensors.keys())
    wrong_shard = sorted(name for name, shard in index["weight_map"].items()
                         if name in tensors and tensors[name]["shard"] != shard)
    scales = []
    policy = collections.Counter()
    for name, meta in sorted(tensors.items()):
        if not name.endswith((".weight_scale_inv", ".weight_scale")):
            continue
        suffix = ".weight_scale_inv" if name.endswith(".weight_scale_inv") else ".weight_scale"
        weight_name = name[:-len(suffix)] + ".weight"
        weight = tensors.get(weight_name)
        shape = meta["shape"]
        if not weight:
            kind = "missing_weight"
        elif len(weight["shape"]) == 2 and shape in ([weight["shape"][0]], [weight["shape"][0], 1]):
            kind = "output_channel_2d_weight"
        elif shape in ([], [1]):
            kind = "scalar"
        else:
            kind = "other_or_block"
        policy[(suffix, kind)] += 1
        scales.append({"name": name, **meta, "weight_shape": weight["shape"] if weight else None,
                       "classification": kind})
    return {
        "schema": "glm52-checkpoint-inspection-v1", "checkpoint": str(root.resolve()),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "index_sha256": hashlib.sha256(index_bytes).hexdigest(),
        "header_sha256": headers, "architectures": config.get("architectures"),
        "quantization_config": config.get("quantization_config"),
        "model_dimensions": {key: config.get(key) for key in (
            "hidden_size", "num_hidden_layers", "n_routed_experts", "moe_intermediate_size",
            "qk_rope_head_dim", "index_head_dim", "index_topk", "index_kpool")},
        "tensor_count": len(tensors), "missing_index_tensors": missing,
        "wrong_index_shard": wrong_shard,
        "scale_summary": [{"suffix": key[0], "kind": key[1], "count": value}
                          for key, value in sorted(policy.items())],
        "scales": scales,
        "loader_compatibility": "UNVERIFIED: compare recorded scale ABI against consumer allocation/loading",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = inspect(args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ("scales", "header_sha256")}, indent=2))
    if result["missing_index_tensors"] or result["wrong_index_shard"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
