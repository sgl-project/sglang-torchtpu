# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/model_executor/model_loader/runai_utils.py

import hashlib
import logging
import os
import time
from pathlib import Path

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

SUPPORTED_SCHEMES = ["s3://", "gs://", "az://"]

# The object-storage clients give up on a throttled download after about two
# minutes (google-cloud-storage: 120 s), which failed whole server starts when
# many servers pulled the same model at once. Metadata pulls retry up to this
# long in total.
_METADATA_PULL_DEADLINE_S = 600.0
_METADATA_PULL_MAX_BACKOFF_S = 60.0
_TRANSIENT_ERROR_NAMES = frozenset(
    {
        "RetryError",
        "TooManyRequests",
        "ServiceUnavailable",
        "InternalServerError",
        "GatewayTimeout",
        "BadGateway",
        "ConnectionError",
        "Timeout",
        "ReadTimeout",
        "ConnectTimeout",
    }
)
_TRANSIENT_STATUS = frozenset({408, 429, 500, 502, 503, 504})


def _is_transient_storage_error(exc: BaseException) -> bool:
    """A throttling, timeout or server-side failure anywhere in ``exc``'s chain
    (matched by name and HTTP status, so no storage SDK import is needed)."""
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if type(exc).__name__ in _TRANSIENT_ERROR_NAMES:
            return True
        if any(isinstance(a, int) and a in _TRANSIENT_STATUS for a in exc.args):
            return True
        exc = exc.__cause__ or exc.__context__
    return False


# Design Pattern: Single Metadata Download Before Process Launch

#   1. Engine entrypoint (engine.py) or server arguments post init  (server_args.py):
#     - Downloads config/tokenizer metadata ONCE before launching subprocesses
#     - This happens in the main process, avoiding multi-process coordination
#
#   2. ModelConfig/HF Utils (model_config.py, hf_transformers/):
#     - Use ObjectStorageModel.get_path() to retrieve the cached local path
#     - NO re-download - just path resolution
#
#   3. RunaiModelStreamerLoader (loader.py):
#     - Calls list_safetensors() which operates directly on the object storage URI
#     - Streams weights lazily during model loading

#   This avoids file locks, race conditions, and duplicate downloads


def list_safetensors(path: str = "") -> list[str]:
    """
    List full file names from object path and filter by allow pattern.

    Args:
        path: The object storage path to list from.

    Returns:
        list[str]: List of full object storage paths allowed by the pattern
    """
    from runai_model_streamer import list_safetensors as runai_list_safetensors

    return runai_list_safetensors(path)


def is_runai_obj_uri(model_or_path: str | Path) -> bool:
    # Cast to str to handle pathlib.Path inputs which lack string methods (like .lower)
    return str(model_or_path).lower().startswith(tuple(SUPPORTED_SCHEMES))


class ObjectStorageModel:
    """
    Model loader that uses Runai Model Streamer to load a model.

      Supports object storage (S3, GCS) with lazy weight streaming.

      Configuration (via load_config.model_loader_extra_config):
          - distributed (bool): Enable distributed streaming
          - concurrency (int): Number of concurrent downloads
          - memory_limit (int): Memory limit for streaming buffer

      Note: Metadata files must be pre-downloaded via
      ObjectStorageModel.download_and_get_path() before instantiation.

    Attributes:
        dir: The temporary created directory.
    """

    def __init__(self, url: str) -> None:
        self.dir = ObjectStorageModel.get_path(url)

        from runai_model_streamer import ObjectStorageModel as RunaiObjectStorageModel

        self._runai_obj = RunaiObjectStorageModel(model_path=url, dst=self.dir)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self._runai_obj.__exit__(exc_type, exc_val, exc_tb)

    def pull_files(
        self,
        allow_pattern: list[str] | None = None,
        ignore_pattern: list[str] | None = None,
    ) -> None:
        """Pull files from object storage into the local cache directory.

        Args:
            allow_pattern: File patterns to include (e.g. ["*.json"]).
            ignore_pattern: File patterns to exclude.
        """
        self._runai_obj.pull_files(allow_pattern, ignore_pattern)

    @classmethod
    def download_and_get_path(cls, model_path: str) -> str:
        """
        Downloads the model metadata (excluding heavy weights) and returns
        the local directory path. Safe for concurrent usage by multiple processes
        """
        deadline = time.monotonic() + _METADATA_PULL_DEADLINE_S
        backoff = 5.0
        while True:
            try:
                with cls(url=model_path) as downloader:
                    downloader.pull_files(
                        ignore_pattern=[
                            "*.pt",
                            "*.safetensors",
                            "*.bin",
                            "*.tensors",
                            "*.pth",
                        ],
                    )
                    cache_dir = downloader.dir
                break
            except Exception as exc:
                remaining = deadline - time.monotonic()
                if not _is_transient_storage_error(exc) or remaining <= backoff:
                    raise
                logger.warning(
                    "Metadata pull for %s failed (%s: %s); retrying in %.0f s",
                    model_path,
                    type(exc).__name__,
                    str(exc)[:200],
                    backoff,
                )
                time.sleep(backoff)
                backoff = min(backoff * 2, _METADATA_PULL_MAX_BACKOFF_S)
        logger.info(f"Runai Model : {cache_dir}, metadata ready.")
        return cache_dir

    @classmethod
    def get_path(cls, model_path: str) -> str:
        """
        Returns the local directory path.
        """
        model_hash = hashlib.sha256(str(model_path).encode()).hexdigest()[:16]
        base_dir = envs.SGLANG_CACHE_DIR.get()

        # Ensure base cache dir exists
        os.makedirs(os.path.join(base_dir, "model_streamer"), exist_ok=True)

        return os.path.join(
            base_dir,
            "model_streamer",
            model_hash,
        )
