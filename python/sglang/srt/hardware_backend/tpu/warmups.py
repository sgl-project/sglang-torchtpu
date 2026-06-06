"""TPU-specific startup warmups.

These warmups are opt-in through ``--warmups``. They intentionally live under
the TPU backend so the generic server warmup registry stays mostly platform
agnostic while still letting TPU pre-hit torch.compile/Pallas serving buckets.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Iterable

import numpy as np
import requests

logger = logging.getLogger(__name__)


def _parse_int_list(raw: str | None, default: Iterable[int]) -> list[int]:
    if raw is None or raw.strip() == "":
        return list(default)
    values: list[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(int(item))
    return values


def _default_batch_sizes(max_bs: int) -> list[int]:
    """Small bucket set that covers common scheduler groupings.

    SGLang often admits the first request alone, then chunks prefill in small
    groups, then runs decode at the target concurrency. Warming only ``max_bs``
    misses the early shapes; warming every CUDA/CPU capture bucket is too slow
    on TPU. Keep this intentionally sparse and let users override with
    ``SGLANG_TPU_WARMUP_BATCH_SIZES`` for a specific benchmark.
    """

    batch_sizes = {
        1,
        min(3, max_bs),
        min(4, max_bs),
        max(1, max_bs - 2),
        max(1, max_bs - 1),
        max_bs,
    }
    for serving_bucket in (64, 128):
        if serving_bucket <= max_bs:
            batch_sizes.add(serving_bucket)
    return sorted(batch_sizes)


def _warmup_config(max_bs: int) -> tuple[list[int], int, int, int, int]:
    batch_sizes = _parse_int_list(
        os.getenv("SGLANG_TPU_WARMUP_BATCH_SIZES"),
        default=_default_batch_sizes(max_bs),
    )
    input_len = int(os.getenv("SGLANG_TPU_WARMUP_INPUT_LEN", "1024"))
    output_len = int(os.getenv("SGLANG_TPU_WARMUP_OUTPUT_LEN", "16"))
    repeats = int(os.getenv("SGLANG_TPU_WARMUP_REPEATS", "2"))
    seed = int(os.getenv("SGLANG_TPU_WARMUP_SEED", "20260606"))
    return batch_sizes, input_len, output_len, repeats, seed


def _token_id_range(vocab_size: int) -> tuple[int, int]:
    # Avoid special-token-looking IDs while staying valid for small test models.
    low = min(max(vocab_size - 1, 1), 100)
    high = max(low + 1, min(vocab_size, low + 8192))
    return low, high


def run_tpu_http_compilation_warmup(server_args) -> None:
    """Run TPU compile warmup through the same HTTP path as serving traffic."""

    max_bs = int(getattr(server_args, "torch_compile_max_bs", 32) or 32)
    batch_sizes, input_len, output_len, repeats, seed = _warmup_config(max_bs)
    rng = np.random.default_rng(seed)
    # ServerArgs does not expose vocab size here; keep IDs in the conservative
    # low range that works for common tokenizer vocabularies.
    low, high = _token_id_range(32000)

    headers = {}
    if server_args.api_key:
        headers["Authorization"] = f"Bearer {server_args.api_key}"
    url = server_args.url()
    ssl_verify = server_args.ssl_verify()
    timeout = int(os.getenv("SGLANG_TPU_WARMUP_TIMEOUT", "1800"))

    logger.info(
        "TPU HTTP compilation warmup begin: batch_sizes=%s input_len=%s output_len=%s repeats=%s",
        batch_sizes,
        input_len,
        output_len,
        repeats,
    )
    tic = time.perf_counter()

    def _post_batch(input_ids: list[list[int]]) -> None:
        res = requests.post(
            url + "/generate",
            json={
                "input_ids": input_ids,
                "stream": False,
                "lora_path": None,
                "return_logprob": False,
                "return_routed_experts": False,
                "logprob_start_len": -1,
                "sampling_params": {
                    "max_new_tokens": output_len,
                    "temperature": 0,
                    "ignore_eos": True,
                },
            },
            headers=headers,
            stream=True,
            timeout=timeout,
            verify=ssl_verify,
        )
        res.raise_for_status()

    for repeat in range(repeats):
        for batch_size in batch_sizes:
            input_ids = rng.integers(
                low,
                high,
                size=(batch_size, input_len),
                dtype=np.int32,
            ).tolist()
            _post_batch(input_ids)
            logger.info(
                "TPU HTTP compilation warmup completed repeat=%s/%s batch_size=%s",
                repeat + 1,
                repeats,
                batch_size,
            )

    logger.info("TPU HTTP compilation warmup ended in %.2fs", time.perf_counter() - tic)
