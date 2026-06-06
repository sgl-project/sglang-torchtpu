"""TPU-specific startup warmups."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Iterable

import numpy as np

from sglang.srt.entrypoints.warmup import warmup
from sglang.srt.managers.io_struct import GenerateReqInput

logger = logging.getLogger(__name__)


def _parse_int_list(raw: str | None, default: Iterable[int]) -> list[int]:
    if raw is None or raw.strip() == "":
        return list(default)

    values: list[int] = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    return values


def _default_batch_sizes(max_bs: int) -> list[int]:
    """Return sparse TPU compile buckets for common serving shapes."""

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
    low = min(max(vocab_size - 1, 1), 100)
    high = max(low + 1, min(vocab_size, low + 8192))
    return low, high


@warmup("tpu_compile_bucket_warmup")
async def tpu_compile_bucket_warmup(disaggregation_mode: str, tokenizer_manager):
    """Pre-hit TPU serving buckets before the server is marked ready."""

    server_args = tokenizer_manager.server_args
    max_bs = int(getattr(server_args, "torch_compile_max_bs", 32) or 32)
    batch_sizes, input_len, output_len, repeats, seed = _warmup_config(max_bs)

    vocab_size = int(getattr(tokenizer_manager.model_config, "vocab_size", 32000))
    low, high = _token_id_range(vocab_size)
    rng = np.random.default_rng(seed)

    logger.info(
        "TPU compile bucket warmup begin: batch_sizes=%s input_len=%s output_len=%s repeats=%s",
        batch_sizes,
        input_len,
        output_len,
        repeats,
    )
    tic = time.perf_counter()

    async def _drain_one(input_ids: list[int]) -> int:
        req = GenerateReqInput(
            input_ids=input_ids,
            sampling_params={
                "max_new_tokens": output_len,
                "temperature": 0,
                "ignore_eos": True,
            },
        )
        if disaggregation_mode != "null":
            from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST

            req.bootstrap_room = 0
            req.bootstrap_host = FAKE_BOOTSTRAP_HOST

        drained = 0
        async for _ in tokenizer_manager.generate_request(req, None):
            drained += 1
        return drained

    for repeat in range(repeats):
        for batch_size in batch_sizes:
            input_ids = rng.integers(
                low,
                high,
                size=(batch_size, input_len),
                dtype=np.int32,
            ).tolist()
            responses = await asyncio.gather(*(_drain_one(ids) for ids in input_ids))
            logger.info(
                "TPU compile bucket warmup completed repeat=%s/%s batch_size=%s responses=%s",
                repeat + 1,
                repeats,
                batch_size,
                sum(responses),
            )

    logger.info("TPU compile bucket warmup ended in %.2fs", time.perf_counter() - tic)
