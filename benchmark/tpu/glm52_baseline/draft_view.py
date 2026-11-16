"""Expose the original native MTP shards without scanning target-only shards."""

import hashlib
import json
import struct
from pathlib import Path


def prepare_draft_view(model: Path, output: Path) -> Path:
    model = model.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads((model / "config.json").read_text())
    assert config["num_nextn_predict_layers"] == 1
    prefix = f"model.layers.{config['num_hidden_layers']}."
    original_index = model / "model.safetensors.index.json"
    index = json.loads(original_index.read_text())
    mapping = {k: v for k, v in index["weight_map"].items() if k.startswith(prefix)}
    assert mapping and any(k.endswith("eh_proj.weight") for k in mapping)
    shards = sorted(set(mapping.values()))
    size = 0
    for shard in shards:
        with (model / shard).open("rb") as f:
            length = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(length))
        for name in (k for k, v in mapping.items() if v == shard):
            begin, end = header[name]["data_offsets"]
            size += end - begin
    # The native NextN loader skips non-MTP keys and shares the target's
    # embedding/head. Symlinks retain the original shard bytes and names;
    # the filtered index restricts discovery to shards containing MTP keys.
    for path in model.iterdir():
        if path.is_file() and path.suffix != ".safetensors" and path != original_index:
            dest = output / path.name
            if not dest.exists():
                dest.symlink_to(path)
    for shard in shards:
        dest = output / shard
        if not dest.exists():
            dest.symlink_to(model / shard)
        assert dest.resolve() == (model / shard).resolve()
    filtered = {"metadata": {"total_size": size}, "weight_map": mapping}
    (output / original_index.name).write_text(json.dumps(filtered, indent=2) + "\n")
    audit = dict(
        source_model=str(model),
        source_index_sha256=hashlib.sha256(original_index.read_bytes()).hexdigest(),
        prefix=prefix,
        mtp_tensor_count=len(mapping),
        mtp_tensor_bytes=size,
        original_shard_count=len(set(index["weight_map"].values())),
        selected_shards=shards,
        shard_storage="symlinks to original checkpoint; no tensor conversion",
    )
    (output / "draft-view-audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    return output.resolve()
