"""Warm eager sampling and request-relay operations before serving admission."""

from __future__ import annotations

import logging
import time
from functools import singledispatch
from typing import TYPE_CHECKING

import torch

from sglang.kernels.ops.attention.clamp_position import clamp_position
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers import overlap_utils
from sglang.srt.runtime_context import get_exec, get_spec
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.sampling.sampling_params import TOP_K_ALL
from sglang_tpu.output_logprobs import (
    precompile_output_top_logprobs,
    warmup_output_top_logprobs,
)
from sglang_tpu.row_ops import precompile_row_ops, warmup_row_ops

if TYPE_CHECKING:
    from sglang.srt.managers.overlap_utils import FutureMap
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


def _max_batch_size(model_runner: ModelRunner) -> int:
    return min(
        model_runner.max_running_requests,
        get_exec().graph.torch_compile_max_bs,
    )


def _is_target_generation_runner(model_runner: ModelRunner) -> bool:
    return (
        not model_runner.is_draft_worker
        and model_runner.pp_group.is_last_rank
        and model_runner.model_config.is_generation
    )


def _uses_generic_sampling(model_runner: ModelRunner) -> bool:
    return (
        _is_target_generation_runner(model_runner)
        and get_spec().speculative_algorithm is None
    )


def _batch_row_templates(device: torch.device | str) -> list[torch.Tensor]:
    return [
        torch.zeros(shape, dtype=dtype, device="cpu").to(device)
        for shape, dtype in (
            ((1,), torch.int64),
            ((1,), torch.int32),
            ((1,), torch.float32),
            ((1, 1), torch.float32),
        )
    ]


def precompile_sampling(model_runner: ModelRunner) -> None:
    """Queue batch metadata and output sorting alongside model compilation."""
    if not (
        _is_target_generation_runner(model_runner)
        and get_exec().graph.enable_torch_compile
    ):
        return
    with torch.no_grad():
        precompile_row_ops(
            _batch_row_templates(model_runner.device), _max_batch_size(model_runner)
        )
        if not _uses_generic_sampling(model_runner):
            return
        for bs in range(1, _max_batch_size(model_runner) + 1):
            logprobs = torch.zeros(
                bs, model_runner.model_config.vocab_size, dtype=torch.float32
            ).to(model_runner.device)
            precompile_output_top_logprobs(logprobs)


def warmup_sampling(model_runner: ModelRunner) -> None:
    """Cover the full-vocabulary sampling tail outside the compiled decoder."""
    if (
        _is_target_generation_runner(model_runner)
        and get_exec().graph.enable_torch_compile
    ):
        started = time.perf_counter()
        max_bs = _max_batch_size(model_runner)
        with torch.no_grad():
            warmup_row_ops(_batch_row_templates(model_runner.device), max_bs)
        logger.info(
            "TPU batch row warmup: rows=1..%d, %.3fs",
            max_bs,
            time.perf_counter() - started,
        )
    if not _uses_generic_sampling(model_runner):
        return
    sampler = model_runner.sampler
    device = torch.device(model_runner.device)
    device_module = (
        torch.random if device.type == "cpu" else torch.get_device_module(device)
    )
    state = device_module.get_rng_state()
    vocab_size = model_runner.model_config.vocab_size
    max_bs = _max_batch_size(model_runner)
    started = time.perf_counter()
    try:
        with torch.no_grad():
            for bs in range(1, max_bs + 1):
                seq_lens = torch.ones(bs, dtype=torch.int64).to(device)
                positions = seq_lens - 1
                info = SamplingBatchInfo(
                    temperatures=torch.ones(bs, 1, device=device),
                    top_ps=torch.ones(bs, device=device),
                    top_ks=torch.full(
                        (bs,), TOP_K_ALL, dtype=torch.int32, device=device
                    ),
                    min_ps=torch.zeros(bs, device=device),
                    is_all_greedy=False,
                    is_any_greedy=False,
                    need_top_k_sampling=False,
                    need_top_p_sampling=False,
                    need_min_p_sampling=False,
                    vocab_size=vocab_size,
                    device=str(device),
                    uniform_top_k=TOP_K_ALL,
                )
                if sampler.enable_deterministic:
                    info.sampling_seed = torch.zeros(
                        bs, dtype=torch.int64, device=device
                    )
                for greedy in (True, False):
                    info.is_all_greedy = info.is_any_greedy = greedy
                    logits = torch.zeros(
                        bs, vocab_size, dtype=torch.float32, device="cpu"
                    ).to(device)
                    output = LogitsProcessorOutput(next_token_logits=logits)
                    tokens = sampler(
                        output, info, True, [0] * bs, [None] * bs, positions
                    )
                    tokens.cpu()
                    output.next_token_logprobs.cpu()
                warmup_output_top_logprobs(torch.log_softmax(logits, dim=-1))
    finally:
        device_module.set_rng_state(state)
    logger.info(
        "TPU full-vocabulary sampler warmup: rows=1..%d, %.3fs",
        max_bs,
        time.perf_counter() - started,
    )


@singledispatch
def warmup_request_relay(future_map: FutureMap, max_batch_size: int) -> None:
    """Exercise AR bookkeeping on scratch storage with the real relay layout."""
    if not future_map.spec_algo.is_none():
        return
    device = torch.device(future_map.device)
    started = time.perf_counter()
    # No live slots exist yet, but scratch storage also preserves debug poison
    # and future changes to the relay's initial-state contract.
    output_tokens = torch.empty_like(future_map.output_tokens_buf)
    new_seq_lens = torch.empty_like(future_map.new_seq_lens_buf)
    with torch.no_grad():
        for bs in range(1, min(max_batch_size, future_map.req_pool_size) + 1):
            indices = torch.arange(bs, dtype=torch.int64).to(device)
            seq_lens = torch.ones(bs, dtype=torch.int64).to(device)
            next_lens = seq_lens + 1
            tokens = torch.zeros(bs, dtype=torch.int64).to(device)
            # Exercise initial publication and the next decode's relay update.
            for _ in range(2):
                future_map._store(new_seq_lens, indices, next_lens)
                new_seq_lens.cpu()
                future_map._store(output_tokens, indices, tokens)
                values = output_tokens[indices]
                if overlap_utils._DEBUG_ASSERT:
                    overlap_utils._assert_nonneg_and_invalidate(
                        values, output_tokens, indices
                    )
                    output_tokens.cpu()
                values.cpu()
            # The next decode increments both int32 and int64 length fields,
            # then computes positions from the relay's gathered int64 lengths.
            (torch.ones(bs, dtype=torch.int32).to(device) + 1).cpu()
            clamp_position(next_lens).cpu()
    logger.info(
        "TPU request relay warmup: rows=1..%d, %.3fs",
        max_batch_size,
        time.perf_counter() - started,
    )
