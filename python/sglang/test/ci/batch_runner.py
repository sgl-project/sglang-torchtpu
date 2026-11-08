"""Run several registered test files in one interpreter, one after another.

Each file still runs as ``__main__`` with the suite's ``-f`` the way
``python3 <file> -f`` would, through ``skip_guard.run_test_file`` so the
no-test-executed check still applies. What is shared is the process: torch,
torch_tpu, jax and sglang stay imported between files, which is most of a
short kernel test's wall time. The batch stops at the first failing file;
the caller reruns that file alone, so a verdict never depends on a neighbour.

``python3 batch_runner.py --results <json> [--guard] <file>...``

Standard library only: the runner launches this module by path.
"""

import argparse
import json
import os
import runpy
import sys
import time
from typing import List, Sequence

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from skip_guard import _exit_code, run_test_file  # noqa: E402

_FILE_ARGS = ("-f",)


def _run_plain(path: str) -> int:
    path = os.path.abspath(path)
    sys.argv = [path, *_FILE_ARGS]
    sys.path[0] = os.path.dirname(path)
    try:
        runpy.run_path(path, run_name="__main__")
        return 0
    except SystemExit as exc:
        return _exit_code(exc.code)


def run_batch(files: Sequence[str], *, guard: bool) -> List[dict]:
    results = []
    for path in files:
        print(f"\n[batch] >>> {path}", flush=True)
        tic = time.perf_counter()
        code = run_test_file(path, _FILE_ARGS) if guard else _run_plain(path)
        elapsed = time.perf_counter() - tic
        print(f"[batch] <<< {path} exit {code} after {elapsed:.0f}s", flush=True)
        results.append({"file": path, "returncode": code, "elapsed": elapsed})
        if code != 0:
            break
    return results


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--guard", action="store_true")
    parser.add_argument("files", nargs="+")
    args = parser.parse_args(argv)
    results = run_batch(args.files, guard=args.guard)
    with open(args.results, "w") as handle:
        json.dump(results, handle)
    return 0 if all(r["returncode"] == 0 for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
