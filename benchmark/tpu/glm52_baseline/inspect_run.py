"""Read bounded status/log tails inside a Falcon allocation without touching services."""

import argparse
import json
import os
from pathlib import Path


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def tail(path, length=3500):
    try:
        with path.open("rb") as f:
            f.seek(max(0, path.stat().st_size - length))
            return f.read().decode(errors="replace").replace("\r", "\n")
    except FileNotFoundError:
        return None


def inspect(root, rank):
    allocation = read_json(root / "allocation" / f"rank-{rank}.json")
    result = {"rank": rank, "allocation": allocation}
    if not allocation:
        return result
    attempt = Path(allocation["attempt_root"]) if "attempt_root" in allocation else None
    if attempt is None:
        return result
    art = attempt / "glm52-e2e" / f"rank-{rank}"
    result.update(
        application_log=tail(attempt / f"rank-{rank}-application.log"),
        application_result=read_json(art / "result.json"),
        source=read_json(art / "benchmark-source.json"),
        server_tail=tail(art / "server.log", 7000),
    )
    if result["source"]:
        result["source"] = {k: v for k, v in result["source"].items() if k != "files"}
    for name in [
        "host-preparation.log",
        "mtp-host-preparation.log",
        "cli-validation.log",
    ]:
        result[name] = tail(art / name, 1200)
    clients = sorted(
        art.glob("client-attempt-*.log"), key=lambda p: int(p.stem.rsplit("-", 1)[1])
    )
    if clients:
        result["client_tail"] = tail(clients[-1], 2500)
    shared = attempt / "glm52-e2e"
    result["client_attention"] = [
        read_json(p) for p in shared.glob("session-*/client-attention.json")
    ]
    result["measurements"] = {}
    for p in shared.glob("measurements/attempt-*"):
        result["measurements"][p.name] = {
            "records": len(list(p.glob("*.json"))),
            "smoke": read_json(p / "smoke-summary.json"),
            "prefix": read_json(p / "prefix-decision.json"),
            "summary": read_json(p / "summary.json"),
        }
        smoke = result["measurements"][p.name]["smoke"]
        if smoke:
            result["measurements"][p.name]["smoke"] = {
                "status": smoke["status"],
                "dp_receipts": len(smoke["checks"]),
            }
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path(os.environ["ARTIFACT_LOCAL_DIR"]))
    p.add_argument("--rank", type=int, default=int(os.environ["FALCON_RANK"]))
    a = p.parse_args()
    result = inspect(a.root, a.rank)
    text = json.dumps(result, indent=2)
    Path(f"/tmp/glm52-inspection-rank-{a.rank}.json").write_text(text + "\n")
    print(text, flush=True)
