"""Run several registered test files from one preloaded interpreter.

The parent imports the expensive common stack once, then forks a child per
file; the child runs the file as ``__main__`` with the suite's ``-f`` the way
``python3 <file> -f`` would (through ``skip_guard.run_test_file`` when asked,
so the no-test-executed check still applies) and exits. Each file therefore
starts from the same post-import state and nothing a file does to the process
(env vars, jax config, a chosen JAX backend) reaches the next one.

``python3 batch_runner.py --results <json> [--guard] <file>...``

SGLANG_TPU_BATCH_PRELOAD lists the modules to import before forking; the
default stops short of anything that opens the accelerator, which must not
happen before a fork.

Standard library only: the runner launches this module by path.
"""

import argparse
import importlib
import json
import os
import runpy
import signal
import sys
import time
import traceback
from typing import List, Sequence

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from skip_guard import _exit_code, run_test_file  # noqa: E402

_FILE_ARGS = ("-f",)
_DEFAULT_PRELOAD = "numpy,torch,sglang.srt.environ,sglang.test.ci.ci_register"


def _preload() -> List[str]:
    names = os.environ.get("SGLANG_TPU_BATCH_PRELOAD", _DEFAULT_PRELOAD)
    loaded = []
    for name in filter(None, (n.strip() for n in names.split(","))):
        tic = time.perf_counter()
        importlib.import_module(name)
        loaded.append(f"{name}={time.perf_counter() - tic:.1f}s")
    return loaded


def _run_plain(path: str) -> int:
    path = os.path.abspath(path)
    sys.argv = [path, *_FILE_ARGS]
    sys.path[0] = os.path.dirname(path)
    try:
        runpy.run_path(path, run_name="__main__")
        return 0
    except SystemExit as exc:
        return _exit_code(exc.code)


def _status_to_returncode(status: int) -> int:
    if os.WIFEXITED(status):
        return os.WEXITSTATUS(status)
    if os.WIFSIGNALED(status):
        return 128 + os.WTERMSIG(status)
    return 1


def _run_in_fork(path: str, *, guard: bool) -> int:
    sys.stdout.flush()
    sys.stderr.flush()
    pid = os.fork()
    if pid == 0:
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        try:
            code = run_test_file(path, _FILE_ARGS) if guard else _run_plain(path)
        except BaseException:
            traceback.print_exc()
            code = 1
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(max(0, min(code, 255)))
    _, status = os.waitpid(pid, 0)
    return _status_to_returncode(status)


def run_batch(files: Sequence[str], *, guard: bool) -> List[dict]:
    print(f"[batch] preloaded {' '.join(_preload())}", flush=True)
    results = []
    for path in files:
        print(f"\n[batch] >>> {path}", flush=True)
        tic = time.perf_counter()
        code = _run_in_fork(path, guard=guard)
        elapsed = time.perf_counter() - tic
        print(f"[batch] <<< {path} exit {code} after {elapsed:.0f}s", flush=True)
        results.append({"file": path, "returncode": code, "elapsed": elapsed})
    return results


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--guard", action="store_true")
    parser.add_argument("files", nargs="+")
    args = parser.parse_args(argv)
    # The preloader ignores Ctrl-C; the active child keeps the default handler.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    results = run_batch(args.files, guard=args.guard)
    with open(args.results, "w") as handle:
        json.dump(results, handle)
    return 0 if all(r["returncode"] == 0 for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
