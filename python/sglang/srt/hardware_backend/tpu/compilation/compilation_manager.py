"""In-process TPU precompilation to warm up the PJRT/HLO compilation cache.

Iterates over shape buckets to compile:
* EXTEND: configured token-count and batch-size bucket combinations.
* DECODE: configured batch-size buckets (tokens == batch_size).
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Callable

import torch

from sglang.srt.hardware_backend.tpu._logging import get_logger
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)

if TYPE_CHECKING:
    from sglang.srt.hardware_backend.tpu.compilation.bucketer import Bucketer
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = get_logger(__name__)


class CompilationManager:
    """Precompiles model forward paths across token and batch-size buckets."""

    def __init__(
        self,
        model_runner: ModelRunner,
        compiled_fns: dict[str, Callable],
        t_bucketer: Bucketer,
        b_bucketer: Bucketer,
    ):
        self.model_runner = model_runner
        self.compiled_fns = compiled_fns
        self.t_bucketer = t_bucketer
        self.b_bucketer = b_bucketer

        # TODO(wyzhang): support ``forward_split_prefill`` on TPU. Its
        # signature includes a ``(split_index, next_split_index)`` tuple
        # that Dynamo specializes per value, so precompile needs an
        # inner layer-range loop here. Until that exists, the
        # split-prefill entry point cold-compiles on first invocation
        # if the scheduler ever routes a request through it.
        if hasattr(model_runner.model, "forward_split_prefill"):
            logger.warning(
                "[compilation manager] %s exposes forward_split_prefill "
                "but this path is not precompiled yet. Requests routed "
                "through split-prefill will cold-compile on first use.",
                type(model_runner.model).__name__,
            )

    def precompile_all(self) -> None:
        """Run precompilation for all valid extend and decode bucket shapes."""
        overall = time.perf_counter()
        self._precompile_extend()
        self._precompile_decode()
        logger.info(
            "[compilation manager] precompile_all done in %.1fs.",
            time.perf_counter() - overall,
        )

    def _precompile_extend(self) -> None:
        """Compile every (token_bucket, bs_bucket) combination.

        Real extend batches can land at any bs bucket (the pad shim
        rounds bs up to ``b_bucketer.next_bucket(real_b)``), and each
        unique (token, bs) shape is a separate HLO cache entry. Skip
        pairs where the token bucket can't hold ``bs`` one-token reqs
        (the kernel would crash on a degenerate cu_q_lens layout)."""
        valid_shapes = [
            (t, b)
            for b in self.b_bucketer.buckets
            for t in self.t_bucketer.buckets
            if t >= b
        ]
        n = len(valid_shapes)
        logger.info(
            "[compilation manager] EXTEND: %d shapes (bs_buckets=%s, "
            "token_buckets=%s)",
            n,
            list(self.b_bucketer.buckets),
            list(self.t_bucketer.buckets),
        )
        for idx, (num_tokens, bs) in enumerate(valid_shapes, start=1):
            start = time.perf_counter()
            self._run_one(num_tokens=num_tokens, bs=bs, mode=ForwardMode.EXTEND)
            logger.info(
                "[compilation manager] EXTEND [%d/%d] (tokens=%d, bs=%d) in %.1fs",
                idx,
                n,
                num_tokens,
                bs,
                time.perf_counter() - start,
            )

    def _precompile_decode(self) -> None:
        """Compile decode mode: vary batch size (where tokens == batch_size)."""
        bs_buckets = list(self.b_bucketer.buckets)
        n = len(bs_buckets)
        logger.info(
            "[compilation manager] DECODE: %d shapes (bs=%s)",
            n,
            bs_buckets,
        )
        for idx, bs in enumerate(bs_buckets, start=1):
            start = time.perf_counter()
            self._run_one(num_tokens=bs, bs=bs, mode=ForwardMode.DECODE)
            logger.info(
                "[compilation manager] DECODE [%d/%d] (bs=%d) in %.1fs",
                idx,
                n,
                bs,
                time.perf_counter() - start,
            )

    def _run_one(self, num_tokens: int, bs: int, mode: ForwardMode) -> None:
        """Build and execute one dummy forward pass for compilation."""
        forward_batch = self._make_dummy_forward_batch(
            num_tokens=num_tokens, bs=bs, mode=mode
        )
        model = self.model_runner.model
        model.prepare_forward_batch(forward_batch)
        self.model_runner.attn_backend.init_forward_metadata(forward_batch)
        # Match TpuGraphRunner's no_grad wrapper for Dynamo cache reuse.
        with torch.no_grad():
            out = model.forward(
                forward_batch.input_ids,
                forward_batch.positions,
                forward_batch,
            )
            # Materialize lazy outputs and backend-marked side effects.
            _materialize_out(out, forward_batch)

    def _make_dummy_forward_batch(
        self, num_tokens: int, bs: int, mode: ForwardMode
    ) -> ForwardBatch:
        """Construct a dummy ForwardBatch at the target bucket shape."""
        device = self.model_runner.device
        is_extend = mode == ForwardMode.EXTEND

        # Token-level inputs
        input_ids = torch.zeros(num_tokens, dtype=torch.int64, device=device)
        input_ids[:bs] = 1
        positions = torch.zeros(num_tokens, dtype=torch.int64, device=device)
        out_cache_loc = torch.zeros(num_tokens, dtype=torch.int64, device=device)

        # Batch-level inputs
        seq_lens = torch.ones(bs, dtype=torch.int64, device=device)
        seq_lens_cpu = torch.ones(bs, dtype=torch.int64)
        req_pool_indices = torch.zeros(bs, dtype=torch.int64, device=device)

        extend_seq_lens = (
            torch.ones(bs, dtype=torch.int32, device=device) if is_extend else None
        )
        extend_prefix_lens = (
            torch.zeros(bs, dtype=torch.int32, device=device) if is_extend else None
        )
        extend_start_loc = (
            torch.arange(0, bs, dtype=torch.int32, device=device) if is_extend else None
        )
        extend_num_tokens = bs if is_extend else None
        extend_seq_lens_cpu = [1] * bs if is_extend else None
        extend_prefix_lens_cpu = [0] * bs if is_extend else None
        # Match runtime extend metadata; None vs list creates guard misses.
        extend_logprob_start_lens_cpu = [0] * bs if is_extend else None

        return ForwardBatch(
            forward_mode=mode,
            batch_size=bs,
            input_ids=input_ids,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            seq_lens_sum=bs,
            orig_seq_lens=seq_lens,
            out_cache_loc=out_cache_loc,
            seq_lens_cpu=seq_lens_cpu,
            positions=positions,
            return_logprob=False,
            extend_num_tokens=extend_num_tokens,
            extend_seq_lens=extend_seq_lens,
            extend_prefix_lens=extend_prefix_lens,
            extend_start_loc=extend_start_loc,
            extend_seq_lens_cpu=extend_seq_lens_cpu,
            extend_prefix_lens_cpu=extend_prefix_lens_cpu,
            extend_logprob_start_lens_cpu=extend_logprob_start_lens_cpu,
            req_to_token_pool=self.model_runner.req_to_token_pool,
            token_to_kv_pool=self.model_runner.token_to_kv_pool,
            attn_backend=self.model_runner.attn_backend,
            # Model accesses ``capture_hidden_mode.need_capture()``; default
            # of None would raise AttributeError inside the compiled region.
            capture_hidden_mode=CaptureHiddenMode.NULL,
        )


def _materialize_out(out, forward_batch) -> None:
    """Sync returned tensors and backend-marked KV side effects."""
    tensors: list[torch.Tensor] = []

    next_token_logits = getattr(out, "next_token_logits", None)
    if isinstance(next_token_logits, torch.Tensor):
        tensors.append(next_token_logits)

    if getattr(forward_batch.attn_backend, "requires_kv_sync_after_compile", False):
        pool = forward_batch.token_to_kv_pool
        for layer_id in range(pool.start_layer, pool.start_layer + pool.layer_num):
            tensors.append(pool.get_fused_kv_buffer(layer_id))

    if not tensors:
        return
    from torch_tpu._internal.sync.sync import synchronize as tpu_sync

    tpu_sync(tensors, wait=True)
