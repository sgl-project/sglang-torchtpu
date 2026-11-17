"""Exchange the four host addresses and launch one bounded EP32 benchmark."""

import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time


def main():
    rank = int(os.environ["FALCON_RANK"])
    assert rank in range(4)
    root = Path(os.environ["ARTIFACT_LOCAL_DIR"])
    peers = root / "moe-decode-peers"
    peers.mkdir(exist_ok=True)
    source = os.environ["SOURCE_COMMIT"]
    record = dict(
        rank=rank, ip=socket.gethostbyname(socket.gethostname()), source=source
    )
    temporary = peers / f"host-{rank}.{os.getpid()}.tmp"
    temporary.write_text(json.dumps(record))
    temporary.replace(peers / f"host-{rank}.json")
    deadline = time.monotonic() + 300
    while True:
        try:
            hosts = [
                json.loads((peers / f"host-{r}.json").read_text()) for r in range(4)
            ]
            assert all(h["source"] == source for h in hosts)
            assert len({h["ip"] for h in hosts}) == 4
            break
        except (FileNotFoundError, json.JSONDecodeError):
            if time.monotonic() >= deadline:
                raise TimeoutError("peer address exchange exceeded 300 seconds")
            time.sleep(1)
    if rank == 0:
        (root / "manifest.json").write_text(
            json.dumps(
                dict(
                    schema_version=1,
                    workflow="operator-optimization",
                    operator_family="fused_moe",
                    operator_name="glm52_moe_decode_three_way",
                    source=source,
                    dimensions=dict(
                        physical_chips=16,
                        ep=32,
                        h=6144,
                        intermediate=2048,
                        experts=256,
                        topk=8,
                        shared=True,
                        local_rows=[24, 4, 6, 12, 18],
                    ),
                ),
                indent=2,
            )
        )
    command = [
        sys.executable,
        str(Path(__file__).with_name("launch.py")),
        "--hosts",
        ",".join(h["ip"] for h in hosts),
        "--host-index",
        str(rank),
        "--output-dir",
        str(root / f"host-{rank}" / "launcher"),
        "--seconds",
        "2400",
    ]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
