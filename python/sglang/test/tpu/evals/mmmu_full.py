"""MMMU-Pro's nightly question set, split deterministically across serving instances.

Only the row list changes: sgl-eval's own loader builds every example and its
own grader scores it, so a shard is comparable with an unsharded run of the
same benchmark. The nightly set is every SUBSET_STRIDE-th question, which keeps
every subject since rows are ordered by subject. The shards partition that set
by stride, and the collector rejects a missing or duplicated id before
reporting the combined score.

Sharding is for a model too slow to finish the set inside one CI job; the caller
supplies the model and its sampling.
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

SHARD_RESULT_PREFIX = "MMMU_SHARD_RESULT "
# 433 of the 1,730 questions, to fit the accuracy slots; 1 restores the full set.
SUBSET_STRIDE = 4


def shard_rows(rows, shard, shards):
    if shards < 1 or not 0 <= shard < shards:
        raise ValueError("Invalid shard index/count")
    return list(rows)[::SUBSET_STRIDE][shard::shards]


def eval_command(
    base_url: str, *, shard: int, shards: int, output: str, sampling: dict
):
    """argv scoring one shard of the nightly set into ``output``."""
    return [
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
        output,
        "--sampling",
        json.dumps(sampling),
    ]


def read_jsonl_ids(path):
    # sgl-eval writes with ensure_ascii=False, so a generation can hold raw
    # U+2028/U+2029/U+0085. str.splitlines() breaks on those; JSONL records
    # end only at "\n".
    text = Path(path).read_text(encoding="utf-8")
    return [json.loads(line)["id"] for line in text.split("\n") if line.strip()]


def run_shard(
    base_url: str,
    *,
    shard: int,
    shards: int,
    model: str,
    max_tokens: int,
    num_threads: int,
    temperature: float,
    top_p: float,
    chat_template_kwargs: dict | None = None,
) -> dict:
    """Score one shard of MMMU-Pro and print its summary for the collector."""
    output = Path(tempfile.mkdtemp(prefix=f"mmmu-pro-shard-{shard}-"))
    subprocess.run(
        eval_command(
            base_url,
            shard=shard,
            shards=shards,
            output=str(output),
            sampling={
                "model": model,
                "max_tokens": max_tokens,
                "num_threads": num_threads,
                "temperature": temperature,
                "top_p": top_p,
                "chat_template_kwargs": chat_template_kwargs or {},
            },
        ),
        check=True,
    )
    files = list(output.glob("sgl_eval_mmmu_pro_*/metrics.json"))
    if len(files) != 1:
        raise RuntimeError(f"Expected one metrics file, found {files}")
    result = json.loads(files[0].read_text())
    ids = read_jsonl_ids(files[0].parent / "output-rs0.jsonl")
    if result["aggregate"]["error_rate"] != 0:
        raise RuntimeError("MMMU-Pro shard contains request errors")
    if result["num_examples"] != len(ids):
        raise RuntimeError("MMMU-Pro shard is incomplete")

    from sglang.test.tpu.evals.full_eval_artifacts import archive_results

    archive_results(files[0].parent)
    summary = {
        "shard": shard,
        "shards": shards,
        "stride": SUBSET_STRIDE,
        "questions": result["num_examples"],
        "score": result["aggregate"]["score"],
        "correct": round(result["aggregate"]["score"] * result["num_examples"]),
        "ids": ids,
    }
    print(SHARD_RESULT_PREFIX + json.dumps(summary), flush=True)
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
    parser.add_argument("--sampling", type=json.loads, required=True)
    args = parser.parse_args()

    from sgl_eval.cli import main as eval_main
    from sgl_eval.evals import _registry
    from sgl_eval.evals._mmmu_pro import load_mmmu_pro

    from sglang.test.tpu.evals.eval_support import sgl_eval_args

    def load_shard(split="test", num_examples=None):
        if num_examples is not None:
            raise ValueError("The nightly set cannot take a sample limit")
        examples = load_mmmu_pro(split)
        if not examples:
            raise ValueError("MMMU-Pro loaded no examples")
        return shard_rows(examples, args.shard, args.shards)

    # Process-local: no installed package or other benchmark is modified.
    _registry.load_mmmu_pro = load_shard
    return eval_main(
        sgl_eval_args(
            "mmmu_pro",
            base_url=args.base_url,
            out_dir=args.output,
            n_repeats=1,
            **args.sampling,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
