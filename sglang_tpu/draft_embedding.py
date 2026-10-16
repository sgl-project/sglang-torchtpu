# SPDX-License-Identifier: Apache-2.0
"""Rank-uniform vocabulary masking for compiled TPU draft models."""

from types import MethodType

import torch

from sglang.srt.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
    get_masked_input_and_mask,
)


def _embed_local_shard(self: VocabParallelEmbedding, input_: torch.Tensor):
    bounds = self._tpu_vocab_bounds
    masked_input, input_mask = get_masked_input_and_mask(
        input_, bounds[0], bounds[1], bounds[2], bounds[3], bounds[4]
    )
    output = self.quant_method.embedding(self, masked_input.long())
    if self.output_dtype is not None:
        output = output.to(self.output_dtype)
    output.masked_fill_(input_mask.unsqueeze(-1), 0)
    return output


def install_tensor_vocab_bounds(embedding: VocabParallelEmbedding) -> None:
    """Install on the draft instance before tracing, retaining its shared weight.

    Rank-dependent Python constants produced different fused Qwen MTP graphs
    and shifted data in the second vocabulary all-gather. Tensor inputs keep
    the shard bounds dynamic without changing the embedding's collective.
    """
    if embedding.tp_size == 1 or "_tpu_vocab_bounds" in embedding._buffers:
        return
    indices = embedding.shard_indices
    embedding.register_buffer(
        "_tpu_vocab_bounds",
        torch.tensor(
            [
                indices.org_vocab_start_index,
                indices.org_vocab_end_index,
                indices.num_org_vocab_padding,
                indices.added_vocab_start_index,
                indices.added_vocab_end_index,
            ],
            dtype=torch.int64,
            device=embedding.weight.device,
        ),
        persistent=False,
    )
    embedding._embed_local_shard = MethodType(_embed_local_shard, embedding)
