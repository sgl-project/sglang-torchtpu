"""Collect full-evaluation archives and combine the MMMU-Pro shards."""

import base64
import io
import json
import os
import subprocess
import tarfile
from pathlib import Path

# MMMU-Pro standard (10 options). A shard that silently loses rows must not be
# reported as the benchmark, so the count is asserted rather than inferred.
MMMU_PRO_QUESTIONS = 1730


def combine_shards(shards):
    if not shards:
        raise ValueError("No MMMU-Pro shard reported a result")
    count = shards[0]["shards"]
    if len(shards) != count or {s["shard"] for s in shards} != set(range(count)):
        raise ValueError(f"Expected all {count} MMMU-Pro shards, got {len(shards)}")
    if len({s["stride"] for s in shards}) != 1:
        raise ValueError("MMMU-Pro shards scored different question sets")
    # The shards split every stride-th question of the full set between them.
    total = len(range(0, MMMU_PRO_QUESTIONS, shards[0]["stride"]))
    ids = [qid for shard in shards for qid in shard["ids"]]
    if len(ids) != total or len(set(ids)) != total:
        raise ValueError("MMMU-Pro coverage is incomplete or contains duplicates")
    for shard in shards:
        expected = len(range(shard["shard"], total, count))
        if shard["questions"] != expected or len(shard["ids"]) != expected:
            raise ValueError(f"shard {shard['shard']} has {shard['questions']} rows")
    correct = sum(s["correct"] for s in shards)
    return {"questions": total, "correct": correct, "score": correct / total}


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
    shard_jobs = [j for j in jobs if "mmmu_shard" in j["name"]]
    for job in jobs:
        # Only sharded evals archive predictions; the rest report a score.
        if "mmmu_shard" not in job["name"]:
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
    if shard_jobs:
        result = combine_shards(list(shards.values()))
        output = Path("full-eval-results")
        output.mkdir(exist_ok=True)
        (output / "mmmu-full-summary.json").write_text(json.dumps(result, indent=2))
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as summary:
            summary.write(
                f"MMMU-Pro: **{result['correct']}/{result['questions']} "
                f"= {result['score']:.2%}**\n"
            )


if __name__ == "__main__":
    main()
