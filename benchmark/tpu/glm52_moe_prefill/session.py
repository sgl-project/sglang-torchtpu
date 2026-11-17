"""Exchange the two host addresses and launch one bounded EP16 benchmark."""

import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time


def main():
    rank = int(os.environ["FALCON_RANK"])
    assert rank in (0, 1)
    root = Path(os.environ["ARTIFACT_LOCAL_DIR"])
    peers = root / "moe-prefill-peers"
    peers.mkdir(exist_ok=True)
    source = os.environ["SOURCE_COMMIT"]
    record = dict(rank=rank, ip=socket.gethostbyname(socket.gethostname()), source=source)
    temporary = peers / f"host-{rank}.{os.getpid()}.tmp"
    temporary.write_text(json.dumps(record))
    temporary.replace(peers / f"host-{rank}.json")
    deadline = time.monotonic() + 300
    while True:
        try:
            hosts = [json.loads((peers / f"host-{r}.json").read_text()) for r in range(2)]
            assert all(h["source"] == source for h in hosts)
            assert hosts[0]["ip"] != hosts[1]["ip"]
            break
        except (FileNotFoundError, json.JSONDecodeError):
            if time.monotonic() >= deadline:
                raise TimeoutError("peer address exchange exceeded 300 seconds")
            time.sleep(1)
    if rank == 0:
        (root / "manifest.json").write_text(json.dumps(dict(
            schema_version=1, workflow="operator-optimization", operator_family="fused_moe",
            operator_name="glm52_moe_prefill_three_way", source=source,
            dimensions=dict(physical_chips=8, ep=16, h=6144, intermediate=2048,
                            experts=256, topk=8, shared=True, local_rows=[64, 2048]),
        ), indent=2))
    command = [sys.executable, str(Path(__file__).with_name("launch.py")),
               "--hosts", ",".join(h["ip"] for h in hosts), "--host-index", str(rank),
               "--output-dir", str(root / f"host-{rank}" / "launcher"), "--seconds", "1500"]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
