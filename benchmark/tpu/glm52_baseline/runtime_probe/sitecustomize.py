"""Record the actual serving process after its runtime imports complete."""

import importlib.metadata
import json
import os
import sys
import threading
import time
from pathlib import Path


def capture():
    names = ("torch", "torch_tpu", "jax", "jaxlib", "libtpu")
    for _ in range(1800):
        modules = [sys.modules.get(name) for name in names]
        if all(getattr(module, "__file__", None) for module in modules):
            versions = {
                name: importlib.metadata.version(name)
                for name in (*names, "tpu-sync-torch", "torchvision")
            }
            evidence = dict(
                pid=os.getpid(),
                executable=sys.executable,
                argv=sys.argv,
                versions=versions,
                imports={name: sys.modules[name].__file__ for name in names},
                source_commit=os.environ.get("SOURCE_COMMIT"),
                benchmark_source_sha256=os.environ.get("GLM52_BENCHMARK_SHA256"),
                compiler_env={
                    k: v
                    for k, v in os.environ.items()
                    if k.startswith(
                        ("TPU_", "TORCH_TPU_", "LIBTPU", "XLA_", "SGLANG_TPU_")
                    )
                },
            )
            evidence["mapped_libraries"] = sorted(
                {
                    line.split()[-1]
                    for line in Path("/proc/self/maps").read_text().splitlines()
                    if any(s in line for s in ("libtpu.so", "libtorch", "torch_tpu"))
                    and "/" in line
                }
            )
            dest = Path(os.environ["GLM52_RUNTIME_DIR"])
            dest.mkdir(parents=True, exist_ok=True)
            (dest / f"pid-{os.getpid()}.json").write_text(
                json.dumps(evidence, indent=2) + "\n"
            )
            print("GLM52_ACTUAL_PROCESS_RUNTIME " + json.dumps(evidence), flush=True)
            return
        time.sleep(1)


if os.environ.get("GLM52_RUNTIME_DIR"):
    threading.Thread(target=capture, daemon=True).start()
