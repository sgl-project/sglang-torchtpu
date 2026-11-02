"""Serve the DFlash2 draft through a config that declares its sliding windows.

Every layer of RadixArk/Muse-Spark-Open-DFlash2 is 2048-token sliding-window
attention (its config's layer_types), but the config does not declare a hybrid
sliding-window model, so SGLang would give the draft full-context KV. This
writes a directory whose config.json adds is_hybrid_swa and an all-SWA
hybrid_layer_pattern, with every other file a symlink to the snapshot's own:
SGLang then keeps the draft's KV in the target's sliding-window slots.

    python3 -m benchmark.tpu.recipe.configs.muse_spark.windowed_draft

agentx.py serves the draft from this directory: MUSE_WINDOWED_DRAFT_DIR, or
~/.cache/sglang/muse-spark-open-dflash2-windowed.
"""

import argparse
import json
import os
from pathlib import Path

DRAFT_MODEL = "RadixArk/Muse-Spark-Open-DFlash2"


def default_dir():
    return os.environ.get("MUSE_WINDOWED_DRAFT_DIR") or str(
        Path.home() / ".cache/sglang/muse-spark-open-dflash2-windowed"
    )


def windowed_config(config):
    """The draft's config with its all-sliding-window layers declared."""
    layers = config["num_hidden_layers"]
    if (
        config.get("layer_types", ["sliding_attention"] * layers)
        != ["sliding_attention"] * layers
    ):
        raise ValueError("the draft has full-attention layers; serve it as is")
    return {**config, "is_hybrid_swa": True, "hybrid_layer_pattern": [1] * layers}


def write_windowed_draft(snapshot, out):
    """Symlink the snapshot's files into out and write the windowed config.json."""
    snapshot, out = Path(snapshot), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for path in sorted(snapshot.iterdir()):
        if path.name == "config.json" or path.name.startswith("."):
            continue
        link = out / path.name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(path.resolve())
    config = json.loads((snapshot / "config.json").read_text())
    (out / "config.json").write_text(json.dumps(windowed_config(config), indent=2))
    return out


def resolve_snapshot(draft):
    """A local directory as is; a Hub repo id through the HF cache (downloaded
    unless HF_HUB_OFFLINE=1)."""
    if (Path(draft) / "config.json").is_file():
        return Path(draft)
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(draft))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--draft", default=DRAFT_MODEL, help="Hub repo id or directory")
    parser.add_argument("--out", default=default_dir())
    args = parser.parse_args(argv)
    print(write_windowed_draft(resolve_snapshot(args.draft), args.out))


if __name__ == "__main__":
    main()
