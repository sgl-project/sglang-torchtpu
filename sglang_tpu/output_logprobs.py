# SPDX-License-Identifier: Apache-2.0
"""Bucket TPU output TopK copies before trimming request lengths on the CPU."""

from __future__ import annotations

import functools

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.logprob_processor import (
    BatchedTopLogprobs,
    LogprobResult,
    OutputLogprobProcessor,
)
from sglang.srt.runtime_context import get_exec
from sglang.utils import LazyImport
from sglang_tpu.compilation.warmup_queue import current_warmup_queue

direct_compile = LazyImport("sglang_tpu.compilation.direct_compile", "direct_compile")


def _sort_logprobs_impl(logprobs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    values, indices = torch.sort(logprobs, dim=-1, descending=True, stable=True)
    return values, indices


@functools.cache
def _compiled_sort():
    # This standalone helper needs native async compilation even when the model
    # uses Dynamo, whose warmup executes synchronously.
    return direct_compile(
        _sort_logprobs_impl,
        specialization_key=lambda logprobs: (
            tuple(logprobs.shape),
            logprobs.dtype,
            logprobs.device,
        ),
    )


def _sort_logprobs(logprobs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if (
        logprobs.device.type != "tpu"
        or torch.compiler.is_compiling()
        or not get_exec().graph.enable_torch_compile
    ):
        return _sort_logprobs_impl(logprobs)
    return _compiled_sort()(logprobs)


def precompile_output_top_logprobs(logprobs: torch.Tensor) -> None:
    """Submit the serving sort while the model compilation queue is open."""
    pending = _compiled_sort().precompile(
        logprobs, async_compile=envs.SGLANG_DIRECT_COMPILE_ASYNC.get()
    )
    queue = current_warmup_queue()
    if queue is None:
        pending.wait()
    else:
        queue.calls.append(pending)


def _batched_top_logprobs(
    values: torch.Tensor,
    indices: torch.Tensor,
    width: int,
    lengths: list[int],
) -> BatchedTopLogprobs:
    # Copy the complete batch before selecting rows: TPU view materialization
    # otherwise specializes on each row offset and exact requested K.
    return BatchedTopLogprobs(values[:, :width], indices[:, :width], lengths)


def _get_top_logprobs(
    logprobs: torch.Tensor, top_logprobs_nums: list[int]
) -> BatchedTopLogprobs:
    max_k = max(top_logprobs_nums)
    vocab_size = logprobs.shape[-1]
    if max_k > vocab_size:
        raise RuntimeError("selected index k out of range")
    width = min(1 << (max_k - 1).bit_length(), vocab_size)
    # Keep the stable sort used by the TPU TopK workaround, including ties.
    values, indices = _sort_logprobs(logprobs)
    return _batched_top_logprobs(values, indices, width, list(top_logprobs_nums))


def warmup_output_top_logprobs(logprobs: torch.Tensor) -> None:
    """Warm one sort and every possible output-copy bucket for this batch size."""
    values, indices = _sort_logprobs(logprobs)
    vocab_size = logprobs.shape[-1]
    width = 1
    # Warmup discards host outputs, so finish each copy before releasing its
    # destination. Serving retains its asynchronous copies on the result.
    while width < vocab_size:
        _batched_top_logprobs(values, indices, width, []).copy_to_cpu(torch.Tensor.cpu)
        width *= 2
    _batched_top_logprobs(values, indices, vocab_size, []).copy_to_cpu(torch.Tensor.cpu)


class TpuOutputLogprobProcessor(OutputLogprobProcessor):
    def _compute_top_logprobs(
        self,
        result: LogprobResult,
        logprobs: torch.Tensor,
        top_logprobs_nums: list[int],
    ) -> None:
        result.batched_top_logprobs = _get_top_logprobs(logprobs, top_logprobs_nums)
