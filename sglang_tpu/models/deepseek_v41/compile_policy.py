# SPDX-License-Identifier: Apache-2.0
"""TPU compile policy of ``DeepseekV4ForCausalLM`` (attached in
``sglang_tpu.compilation.core_compile_annotations``).

When the model has Engram layers, after the batch is padded to its bucket and
before attention planning, this rank's tokens are hashed on the host and their
rows of the shared host table copied to the device (``engram_host``); the
graph takes the staged rows and their int32 row ids instead of hashing and
looking up. Without Engram layers (DeepSeek-V4) it is the plain decoder policy.

Warmup pads through the same ``prepare_forward_batch``, so precompiled and
serving graphs share one signature.
"""

from __future__ import annotations

from typing import Any

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang_tpu.compilation.decoder_compile_policy import DecoderTpuCompilePolicy
from sglang_tpu.models.deepseek_v41 import engram_host


class DeepseekV4DecoderTpuCompilePolicy(DecoderTpuCompilePolicy):
    """Decoder policy that stages host Engram rows."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
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

    def prepare_forward_batch(self, forward_batch: ForwardBatch) -> None:
        super().prepare_forward_batch(forward_batch)
        if self._engram_tables:
            self._stage_engram_rows(forward_batch)

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
