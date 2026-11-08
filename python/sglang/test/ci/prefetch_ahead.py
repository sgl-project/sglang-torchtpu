"""Warm the page cache with the next test file's checkpoint while the current
file runs.

A serving test's weight load is a cold read of the checkpoint from the
weights mount while the slice sits idle. The files in a suite run one after
another on one host, so the read for file i+1 can happen during file i: a
low-priority reader streams its safetensors through the page cache, and the
engine's loader then finds them in RAM. Enabled by SGLANG_TPU_PREFETCH_AHEAD=1.

The checkpoint is taken from the recipe config the test file names
(``recipe/configs/...py``); files that launch some other way are skipped.
"""

import glob
import logging
import os
import re
import subprocess
import sys
from typing import List, Optional

logger = logging.getLogger(__name__)

_RECIPE_RE = re.compile(r'"(recipe/configs/[\w/.-]+\.py)"')
# A checkpoint larger than this is not read ahead: it would evict the running
# server's own pages. Arbitrary; the gate's models are under 60 GB.
_MAX_PREFETCH_BYTES = 200 * 1024**3


def enabled() -> bool:
    return os.environ.get("SGLANG_TPU_PREFETCH_AHEAD") == "1"


def _resolve(model_id: str) -> Optional[str]:
    root = os.environ.get("SGLANG_TPU_WEIGHTS_ROOT", "/weights")
    for candidate in (
        os.path.join(root, model_id),
        os.path.join(root, model_id.rsplit("/", 1)[-1]),
    ):
        if os.path.isdir(candidate):
            return candidate
    return model_id if os.path.isdir(model_id) else None


def checkpoint_dirs(test_file: str, repo_root: str) -> List[str]:
    """The checkpoint directories the test file's recipes load."""
    try:
        source = open(test_file, errors="replace").read()
    except OSError:
        return []
    configs = sorted(set(_RECIPE_RE.findall(source)))
    if not configs:
        return []
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from benchmark.tpu.launch_server import load_recipe

    dirs = []
    for config in configs:
        try:
            args = load_recipe(config)["server_args"]
        except Exception as exc:  # a recipe that needs the topology at load
            logger.info(f"[prefetch-ahead] skip {config}: {exc}")
            continue
        for key in ("model_path", "speculative_draft_model_path"):
            path = args.get(key)
            resolved = _resolve(path) if path else None
            if resolved and resolved not in dirs:
                dirs.append(resolved)
    return dirs


def _weight_files(dirs: List[str]) -> List[str]:
    files = []
    for d in dirs:
        files.extend(sorted(glob.glob(os.path.join(d, "*.safetensors"))))
    return files


class PrefetchAhead:
    """One background reader at a time; the next start replaces the last."""

    def __init__(self, repo_root: str) -> None:
        self._repo_root = repo_root
        self._process: Optional[subprocess.Popen] = None

    def start_for(self, test_file: Optional[str]) -> None:
        self.close()
        if test_file is None:
            return
        files = _weight_files(checkpoint_dirs(test_file, self._repo_root))
        if not files:
            return
        total = sum(os.path.getsize(f) for f in files)
        if total > _MAX_PREFETCH_BYTES:
            logger.info(
                f"[prefetch-ahead] skip {os.path.basename(test_file)}: "
                f"{total / 1024**3:.0f} GB"
            )
            return
        logger.info(
            f"[prefetch-ahead] reading {len(files)} files, {total / 1024**3:.1f} GB "
            f"for {os.path.basename(test_file)}"
        )
        self._process = subprocess.Popen(
            ["nice", "-n", "19", "cat", *files],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def close(self) -> None:
        if self._process is not None and self._process.poll() is None:
            self._process.kill()
            self._process.wait()
        self._process = None
