"""Submit standard T2VA to replica0, optionally keeping both replicas active."""

import argparse
import json
import shutil
import time
import urllib.request
from pathlib import Path

REQUEST = {
    "prompt": "A fixed camera records a quiet seaside at sunrise. Gentle waves roll onto the sand with natural ocean sounds.",
    "task": "t2va",
    "conditions": [],
    "target": {"short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 4},
    "num_inference_steps": 50,
    "flow_shift": 12.0,
    "audio_flow_shift": 3.0,
    "seed": 42,
    "quality": "lossless",
    "n": 1,
    "enhance_prompt": False,
    "enable_teacache": False,
    "enable_spectrum": False,
    "enable_cache_dit": False,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--receipts", required=True)
    parser.add_argument("--timeout", type=int, default=10800)
    parser.add_argument("--paired-replicas", action="store_true")
    args = parser.parse_args()
    root = Path(args.receipts)
    root.mkdir(parents=True, exist_ok=True)
    url = args.url.rstrip("/")
    (root / "request.json").write_text(json.dumps(REQUEST, indent=2) + "\n")
    start = time.monotonic()
    request = urllib.request.Request(
        url + "/v1/videos",
        data=json.dumps(REQUEST).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        job = json.load(response)
    (root / "submit.json").write_text(json.dumps(job, indent=2) + "\n")
    job_id = job["id"]
    companion_id = None
    if args.paired_replicas:
        # Submit before waiting for generation. The fresh scheduler client
        # routes these two identical requests to replicas 0 and 1 in order.
        companion = urllib.request.Request(
            url + "/v1/videos",
            data=json.dumps(REQUEST).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(companion, timeout=120) as response:
            companion_job = json.load(response)
        companion_id = companion_job["id"]
        (root / "companion-submit.json").write_text(
            json.dumps(companion_job, indent=2) + "\n"
        )
    while True:
        with urllib.request.urlopen(
            url + "/v1/videos/" + job_id, timeout=60
        ) as response:
            job = json.load(response)
        (root / "job.json").write_text(json.dumps(job, indent=2) + "\n")
        if job.get("status") == "failed":
            raise RuntimeError(f"Standard request failed: {job}")
        if job.get("status") == "completed":
            break
        if time.monotonic() - start > args.timeout:
            raise TimeoutError("Request timeout is inconclusive; retain worker logs")
        time.sleep(5)
    output = root / "h3-standard-4s.mp4"
    with (
        urllib.request.urlopen(
            url + "/v1/videos/" + job_id + "/content",
            timeout=120,
        ) as response,
        output.open("wb") as target,
    ):
        shutil.copyfileobj(response, target)
    (root / "client-result.json").write_text(
        json.dumps(
            {
                "job_id": job_id,
                "companion_job_id": companion_id,
                "paired_replicas": args.paired_replicas,
                "output": str(output),
                "complete_video_wall_seconds": time.monotonic() - start,
                "media_validation": "NOT_RUN",
                "replica_validation": "CHECK_WORKER_LOGS",
            },
            indent=2,
        )
        + "\n"
    )
    print(str(output))


if __name__ == "__main__":
    main()
