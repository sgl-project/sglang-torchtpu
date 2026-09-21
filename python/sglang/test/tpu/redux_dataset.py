import json
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow as pa

REPO = "edinburgh-dawg/mmlu-redux-2.0"
REVISION = "372ea425445d51e1ba1188c56e5e893f8138621f"


def prepare(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(
        f"https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}?recursive=true&limit=1000"
    ) as response:
        paths = [x["path"] for x in json.load(response) if x["path"].endswith(".arrow")]
    assert len(paths) == 57, len(paths)

    def fetch(path):
        with urllib.request.urlopen(
            f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{path}"
        ) as response:
            rows = pa.ipc.open_stream(response.read()).read_all().to_pylist()
        subject = path.split("/")[0]
        return [
            dict(row, subject=subject, id=f"{subject}/{i}")
            for i, row in enumerate(rows)
        ]

    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = [r for batch in pool.map(fetch, sorted(paths)) for r in batch]
    assert len(rows) == 5700, len(rows)
    for r in rows:
        r["scorable"] = r["error_type"] == "ok"
        r["scoring_answer"] = r["answer"]
        if r["error_type"] == "wrong_groundtruth":
            value = str(r.get("correct_answer", "")).strip()
            answer_map = {
                **{str(i): i for i in range(4)},
                **{f"{i}.0": i for i in range(4)},
                **{letter: i for i, letter in enumerate("ABCD")},
            }
            if value in answer_map:
                r["scoring_answer"] = answer_map[value]
                r["scorable"] = True
        assert len(r["choices"]) == 4 and r["scoring_answer"] in range(4)
    (output / "dataset.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    counts = {
        "repo": REPO,
        "revision": REVISION,
        "total": len(rows),
        "subjects": 57,
        "error_types": dict(Counter(r["error_type"] for r in rows)),
        "scorable": sum(r["scorable"] for r in rows),
        "corrected": sum(
            r["error_type"] == "wrong_groundtruth" and r["scorable"] for r in rows
        ),
    }
    (output / "dataset_manifest.json").write_text(json.dumps(counts, indent=2))
    print(json.dumps(counts), flush=True)
    return rows


if __name__ == "__main__":
    import sys

    prepare(sys.argv[1])
