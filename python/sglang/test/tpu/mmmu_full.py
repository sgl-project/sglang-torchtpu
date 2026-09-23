"""Full MMMU-Pro, split deterministically across independent serving instances.

Only dataset transport changes: retain numbered references and prepend each
image once, in column order. The pinned sgl-eval prompt and scorer are unchanged.
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def shard_rows(rows, shard, shards):
    if shards < 1 or not 0 <= shard < shards:
        raise ValueError("Invalid shard index/count")
    return list(enumerate(rows))[shard::shards]


def build_example(row, index):
    from sgl_eval.evals import _mmmu_pro as loader
    from sgl_eval.types import Example

    options = loader._parse_options(row)
    text = "\n".join([row["question"], *map(str, options)])
    for match in loader._IMAGE_REF_RE.finditer(text):
        if row.get(f"image_{int(match.group(1))}") is None:
            raise ValueError(f"Missing image in {row.get('id', index)}")
    images = [
        row[f"image_{n}"] for n in range(1, 8) if row.get(f"image_{n}") is not None
    ]
    if not images:
        raise ValueError(f"No images in {row.get('id', index)}")
    return Example(
        id=row.get("id") or f"mmmu_pro-{index}",
        inputs={"problem": loader.get_mcq_fields(row["question"], options)["problem"]},
        target=loader._normalize_answer(row["answer"]),
        media=loader._image_media(images),
        meta={"subject": row.get("subject")},
    )


def run_shard(base_url, *, shard, shards=2):
    output = Path(tempfile.mkdtemp(prefix=f"mmmu-pro-shard-{shard}-"))
    subprocess.run(
        [
            sys.executable,
            "-m",
            __name__,
            "--base-url",
            base_url,
            "--shard",
            str(shard),
            "--shards",
            str(shards),
            "--output",
            str(output),
        ],
        check=True,
    )
    files = list(output.glob("sgl_eval_mmmu_pro_*/metrics.json"))
    if len(files) != 1:
        raise RuntimeError(f"Expected one metrics file, found {files}")
    result = json.loads(files[0].read_text())
    expected = len(range(shard, 1730, shards))
    if result["num_examples"] != expected or result["aggregate"]["error_rate"] != 0:
        raise RuntimeError("Incomplete MMMU-Pro shard or request errors")
    from sglang.test.tpu.full_eval_artifacts import archive_results

    archive_results(files[0].parent)
    summary = {
        "shard": shard,
        "shards": shards,
        "questions": expected,
        "score": result["aggregate"]["score"],
        "correct": round(result["aggregate"]["score"] * expected),
        "ids": [
            json.loads(line)["id"]
            for line in (files[0].parent / "output-rs0.jsonl").read_text().splitlines()
        ],
    }
    print("MMMU_SHARD_RESULT " + json.dumps(summary), flush=True)
    return {
        **result["aggregate"],
        "latency": result["latency_seconds"],
        "output_throughput": result["output_throughput_tps"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--shards", type=int, default=2)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    from datasets import load_dataset
    from sgl_eval.cli import main as eval_main
    from sgl_eval.evals import _registry

    def load_examples(split="test", num_examples=None):
        if num_examples is not None:
            raise ValueError("Full-set evaluation cannot have a sample limit")
        rows = load_dataset("MMMU/MMMU_Pro", "standard (10 options)", split=split)
        if len(rows) != 1730:
            raise ValueError(f"Expected 1730 questions, got {len(rows)}")
        return [
            build_example(row, index)
            for index, row in shard_rows(rows, args.shard, args.shards)
        ]

    # Process-local adapter; no installed package or other benchmark is modified.
    _registry.load_mmmu_pro = load_examples
    return eval_main(
        [
            "run",
            "mmmu_pro",
            "--base-url",
            args.base_url.rstrip("/") + "/v1",
            "--model",
            "/weights/Kimi-K2.6",
            "--thinking",
            "--num-threads",
            "128",
            "--temperature",
            "1",
            "--top-p",
            "1",
            "--max-tokens",
            "98304",
            "--n-repeats",
            "1",
            "--out-dir",
            args.output,
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
