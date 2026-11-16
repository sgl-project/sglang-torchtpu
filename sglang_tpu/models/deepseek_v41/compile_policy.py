# SPDX-License-Identifier: Apache-2.0
"""TPU compile policy of ``DeepseekV4ForCausalLM`` (attached in
``sglang_tpu.compilation.core_compile_annotations``).

V4 and V4.1 share the model class; for V4 this is the plain decoder policy.
For V4.1, after the batch is padded to its bucket and before attention
planning, it stages the graph inputs the compiled forward reads:

- Engram: this rank's tokens are hashed on the host and their rows of the
  shared host table copied to the device (``engram_host``); the graph takes
  the staged rows and their int32 row ids instead of hashing and looking up.
- Token embedding, under attention DP: looked up eagerly and passed in as
  ``input_embeds``. Inside the TP1/DP8 compiled graph the unsharded BF16
  embedding gather returns wrong rows, while the same lookup outside the graph
  is exact (as vllm-torchtpu #1848 does).

Warmup pads through the same ``prepare_forward_batch``, so precompiled and
serving graphs share one signature.
"""

from __future__ import annotations

from typing import Any

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.runtime_context import get_parallel
from sglang_tpu.compilation.decoder_compile_policy import DecoderTpuCompilePolicy
from sglang_tpu.models.deepseek_v41 import engram_host


class DeepseekV4DecoderTpuCompilePolicy(DecoderTpuCompilePolicy):
    """Decoder policy that stages V4.1 Engram rows and DP token embeddings."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        model = self.model_runner.model
        self._is_dsv41 = (
            self.model_runner.model_config.hf_config.model_type == "deepseek_v41"
        )
        self._engram_hasher = model.model.engram_hasher if self._is_dsv41 else None
        # Empty unless every Engram table is a shared host table.
        self._engram_tables = (
            engram_host.shared_host_tables(model)
            if self._engram_hasher is not None
            else ()
        )
        self._embeds_outside_graph = self._is_dsv41 and get_parallel().attn_dp_size > 1

    def prepare_forward_batch(self, forward_batch: ForwardBatch) -> None:
        super().prepare_forward_batch(forward_batch)
        if not self._is_dsv41:
            return
        input_ids = forward_batch.input_ids
        if self._engram_tables:
            self._stage_engram_rows(forward_batch=forward_batch, input_ids=input_ids)
        if self._embeds_outside_graph and forward_batch.input_embeds is None:
            embed_tokens = self.model_runner.model.get_input_embeddings()
            with torch.no_grad():
                forward_batch.input_embeds = embed_tokens(input_ids)

    def _stage_engram_rows(
        self, *, forward_batch: ForwardBatch, input_ids: torch.Tensor
    ) -> None:
        hasher = self._engram_hasher
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
