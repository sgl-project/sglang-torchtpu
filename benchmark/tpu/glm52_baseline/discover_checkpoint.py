"""Select a unique GLM-5.2 block-FP8 checkpoint from a read-only model mount."""

import argparse
import json
import re
from pathlib import Path


def discover(root: Path, output: Path):
    candidates = []
    observed = []
    for path in sorted(root.iterdir()):
        if not re.search(r"glm[-_.]?5[-_.]?2", path.name, re.IGNORECASE):
            continue
        config_path = path / "config.json"
        if not config_path.is_file():
            continue
        config = json.loads(config_path.read_text())
        quant = config.get("quantization_config") or {}
        observed.append(
            dict(
                path=str(path),
                quantization_config=quant,
                architectures=config.get("architectures"),
            )
        )
        if (
            quant.get("quant_method") == "fp8"
            and quant.get("weight_block_size")
            and config.get("architectures") == ["GlmMoeDsaForCausalLM"]
        ):
            candidates.append(path)
    output.mkdir(parents=True, exist_ok=True)
    (output / "model-discovery.json").write_text(json.dumps(observed, indent=2) + "\n")
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected one GLM-5.2 block-FP8 model, found {candidates}; inspect {output}"
        )
    (output / "model-path.txt").write_text(str(candidates[0]) + "\n")
    print("BLOCK_FP8_MODEL_SELECTED", candidates[0], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("/models"))
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    discover(args.root, args.output)
