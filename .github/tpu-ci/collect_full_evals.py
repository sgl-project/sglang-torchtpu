"""Collect full-evaluation archives and combine the two MMMU-Pro shards."""

import base64
import io
import json
import os
import subprocess
import tarfile
from pathlib import Path


def combine_shards(shards):
    if len(shards) != 2 or {s["shard"] for s in shards} != {0, 1}:
        raise ValueError("Expected both MMMU-Pro shards")
    ids = [qid for shard in shards for qid in shard["ids"]]
    if len(ids) != 1730 or len(set(ids)) != 1730:
        raise ValueError("MMMU-Pro coverage is incomplete or contains duplicates")
    if any(
        s["questions"] != 865 or len(s["ids"]) != 865 or s["shards"] != 2
        for s in shards
    ):
        raise ValueError("Unexpected shard sizes")
    correct = sum(s["correct"] for s in shards)
    return {"questions": 1730, "correct": correct, "score": correct / 1730}


def main():
    repo, run_id = os.environ["GH_REPO"], os.environ["RUN_ID"]
    data = subprocess.check_output(
        [
            "gh",
            "api",
            "--paginate",
            "--slurp",
            f"repos/{repo}/actions/runs/{run_id}/jobs?per_page=100",
        ],
        text=True,
    )
    jobs = [job for page in json.loads(data) for job in page["jobs"]]
    shards = {}
    kimi_jobs = [j for j in jobs if "test_tpu_kimi_k26_mmmu_shard" in j["name"]]
    for job in jobs:
        if not any(
            name in job["name"]
            for name in (
                "test_tpu_kimi_k26_mmmu_shard",
                "test_tpu_deepseek_r1_0528_tp16_dp1_accuracy",
            )
        ):
            continue
        log = subprocess.check_output(
            [
                "gh",
                "api",
                f"repos/{repo}/actions/jobs/{job['id']}/logs",
                "--allow-escape-sequences",
            ],
            text=True,
        )
        chunks = None
        for line in log.splitlines():
            if "MMMU_SHARD_RESULT " in line:
                row = json.loads(line.split("MMMU_SHARD_RESULT ", 1)[1])
                if row["shard"] in shards and shards[row["shard"]] != row:
                    raise ValueError("Conflicting shard summaries")
                shards[row["shard"]] = row
            if "FULL_EVAL_ARCHIVE_BEGIN" in line:
                chunks = []
            elif "FULL_EVAL_ARCHIVE_CHUNK " in line and chunks is not None:
                chunks.append(line.split("FULL_EVAL_ARCHIVE_CHUNK ", 1)[1].strip())
            elif "FULL_EVAL_ARCHIVE_END" in line and chunks:
                output = Path("full-eval-results") / str(job["id"])
                output.mkdir(parents=True, exist_ok=True)
                with tarfile.open(
                    fileobj=io.BytesIO(
                        base64.b64decode("".join(chunks), validate=True)
                    ),
                    mode="r:gz",
                ) as archive:
                    archive.extractall(output, filter="data")
                chunks = None
    if kimi_jobs:
        result = combine_shards(list(shards.values()))
        output = Path("full-eval-results")
        output.mkdir(exist_ok=True)
        (output / "mmmu-full-summary.json").write_text(json.dumps(result, indent=2))
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as summary:
            summary.write(
                f"MMMU-Pro full set: **{result['correct']}/1730 = {result['score']:.2%}**\n"
            )


if __name__ == "__main__":
    main()
