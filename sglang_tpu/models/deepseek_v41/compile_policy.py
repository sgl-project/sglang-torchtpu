# SPDX-License-Identifier: Apache-2.0
"""TPU compile policy of ``DeepseekV4ForCausalLM`` (attached in
``sglang_tpu.compilation.core_compile_annotations``).

After the batch is padded to its bucket and before attention planning, it
stages the graph inputs the compiled forward reads:

- Engram, when the model has Engram layers: this rank's tokens are hashed on
  the host and their rows of the shared host table copied to the device
  (``engram_host``); the graph takes the staged rows and their int32 row ids
  instead of hashing and looking up.
- Token embedding, under attention DP: looked up eagerly and passed in as
  ``input_embeds``. Attention DP builds the embedding unsharded
  (``enable_tp=not is_dp_attention_enabled()``), and inside the TP1/DP8
  compiled graph that BF16 gather returns wrong rows, with int64 ids too,
  while the same lookup outside the graph is exact (as vllm-torchtpu #1848
  does). Remove once torch_tpu compiles that gather correctly.

Warmup pads through the same ``prepare_forward_batch``, so precompiled and
serving graphs share one signature.

Under ``--tpu-compile-tracer direct`` (V4.1): the policy first registers the
model's plain tensor attributes and prunes unused inputs
(``direct_tracer.prepare_direct_compile``), and the staged Engram rows ride
on the attention metadata, because the tracer rebuilds ``ForwardBatch`` from
its dataclass fields only. V4.1's graphs are ``spmd_safe``.
"""

from __future__ import annotations

import functools
from typing import Any

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_context import get_forward_context
from sglang.srt.runtime_context import get_parallel
from sglang_tpu.compilation.decoder_compile_policy import DecoderTpuCompilePolicy
from sglang_tpu.models.deepseek_v41 import engram_host


def is_deepseek_v41(model_runner) -> bool:
    config = getattr(model_runner.model_config, "hf_config", None)
    return getattr(config, "model_type", None) == "deepseek_v41"


class DeepseekV4DecoderTpuCompilePolicy(DecoderTpuCompilePolicy):
    """Decoder policy that stages host Engram rows and DP token embeddings.

    For V4.1, ``spmd_safe`` declares the traced graphs identical across ranks,
    so ``--use-spmd-compile`` compiles each warmup once and broadcasts it.
    Rank-dependent values must reach the graph as rank inputs: attention DP and
    MoE EP offsets go through the lifted rank tensors, and every per-rank
    batch quantity (tokens, staged Engram rows, attention metadata) is a graph
    input. Verify with ``VERIFY_SPMD_SAFE=1`` before a recipe enables the
    flag: a rank-dependent constant would ship the owner rank's executable to
    every rank with no error. V4 was not verified and keeps the defaults.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.is_v41 = is_deepseek_v41(self.model_runner)
        self.spmd_safe = self.is_v41
        model = self.model_runner.model
        # No Engram layers (DeepSeek-V4) or a NextN draft, which has no hasher.
        self._engram_hasher = (
            model.model.engram_hasher
            if self.model_runner.model_config.use_engram
            and not self.model_runner.is_draft_worker
            else None
        )
        # Empty unless every Engram table is a shared host table.
        self._engram_tables = (
            engram_host.shared_host_tables(model)
            if self._engram_hasher is not None
            else ()
        )
        self._embeds_outside_graph = get_parallel().attn_dp_size > 1

    def prepare_forward_batch(self, forward_batch: ForwardBatch) -> None:
        super().prepare_forward_batch(forward_batch)
        if self._engram_tables:
            self._stage_engram_rows(forward_batch)
        if self._embeds_outside_graph and forward_batch.input_embeds is None:
            embed_tokens = self.model_runner.model.get_input_embeddings()
            with torch.no_grad():
                forward_batch.input_embeds = embed_tokens(forward_batch.input_ids)

    def _stage_engram_rows(self, forward_batch: ForwardBatch) -> None:
        hasher = self._engram_hasher
        input_ids = forward_batch.input_ids
        engram_host.move_hasher_to_host(hasher)
        with torch.no_grad():
            hash_ids = engram_host.host_hash_ids(hasher, input_ids, forward_batch)
        num_tokens, num_layers, num_cols = hash_ids.shape
        device = input_ids.device
        forward_batch.tpu_engram_staged = engram_host.gather_staged_rows(
            hash_ids, self._engram_tables, device
        )
        forward_batch.tpu_engram_local_ids = engram_host.local_row_ids(
            num_tokens, num_layers, num_cols, device
        )

    def _direct_compile(self, fn):
        if self.is_v41:
            from sglang_tpu.models.deepseek_v41.direct_tracer import (
                prepare_direct_compile,
            )

            prepare_direct_compile(self)
        call, warmup_async = super()._direct_compile(fn)
        if not self._engram_tables:
            return call, warmup_async
        return _carrying_engram(call), _carrying_engram(warmup_async)


def _carrying_engram(run):
    """Put the batch's staged Engram on the attention metadata, a pytree the
    direct tracer lifts as graph inputs (engram_host.staged_inputs), before
    ``run(input_ids, positions, forward_batch, ...)`` reads the metadata."""

    @functools.wraps(run)
    def carried(input_ids, positions, forward_batch, *args, **kwargs):
        metadata = get_forward_context().attn_backend.forward_metadata
        if metadata is None or not hasattr(metadata, "engram"):
            raise RuntimeError(
                "DeepSeek-V4.1 direct tracer: the attention metadata must be built "
                "before the compiled forward to carry the staged Engram rows"
            )
        metadata.engram = (
            forward_batch.tpu_engram_local_ids,
            tuple(tuple(pair) for pair in forward_batch.tpu_engram_staged),
        )
        return run(input_ids, positions, forward_batch, *args, **kwargs)

    return carried
