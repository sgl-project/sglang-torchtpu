"""Logical embedding rows backed by padded encoder tensors.

TPU slicing/concatenation specializes on tensor shapes. Keep those operations
on CPU row indices until placement into a padded prefill tensor instead.
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class EmbeddingRows:
    data: torch.Tensor
    indices: torch.Tensor  # CPU int32, in logical prompt order


@dataclass(frozen=True)
class PaddedEmbedding:
    parts: tuple[EmbeddingRows, ...]
    hidden_size: int

    @classmethod
    def from_tensor(
        cls,
        data: torch.Tensor,
        num_tokens: int | None = None,
        row_indices: torch.Tensor | None = None,
    ) -> "PaddedEmbedding":
        if data.ndim != 2:
            raise ValueError("Padded embeddings must have shape [bucket, hidden]")
        if num_tokens is None:
            num_tokens = data.shape[0]
        if not 0 <= num_tokens <= data.shape[0]:
            raise ValueError("Valid embedding rows exceed the padded tensor")
        if row_indices is None:
            row_indices = torch.arange(num_tokens, dtype=torch.int32)
        else:
            row_indices = row_indices.to(device="cpu", dtype=torch.int32)
            if row_indices.ndim != 1 or (
                row_indices.numel()
                and (row_indices.min() < 0 or row_indices.max() >= num_tokens)
            ):
                raise ValueError("Embedding row indices exceed the valid rows")
        return cls((EmbeddingRows(data, row_indices),), data.shape[1])

    @property
    def num_tokens(self) -> int:
        return sum(part.indices.numel() for part in self.parts)

    @property
    def storage_sizes(self) -> dict[int, int]:
        # A logical slice pins the entire padded encoder output. Cache entries
        # sharing that output must charge it until the last entry is evicted.
        return {id(p.data): p.data.numel() * p.data.element_size() for p in self.parts}

    def to_cpu(self) -> torch.Tensor:
        """Materialize logical rows for consumers such as encoder disaggregation."""
        buffers = {}
        for part in self.parts:
            if id(part.data) not in buffers:
                buffers[id(part.data)] = part.data.to("cpu")
        if not self.parts:
            raise ValueError("Cannot materialize an empty embedding without a dtype")
        return torch.cat([buffers[id(p.data)][p.indices.long()] for p in self.parts])

    def slice_rows(self, start: int, end: int) -> "PaddedEmbedding":
        start, end, _ = slice(start, end).indices(self.num_tokens)
        parts = []
        offset = 0
        for part in self.parts:
            size = part.indices.numel()
            lo, hi = max(start - offset, 0), min(end - offset, size)
            if lo < hi:
                parts.append(EmbeddingRows(part.data, part.indices[lo:hi]))
            offset += size
        return PaddedEmbedding(tuple(parts), self.hidden_size)

    def split_columns(self, index: int) -> tuple["PaddedEmbedding", "PaddedEmbedding"]:
        # These slices retain the padded row dimension; only the model's fixed
        # deepstack widths specialize, independently of the prefill bucket.
        if not 0 < index < self.hidden_size:
            raise ValueError("Deepstack split must leave both column groups nonempty")
        buffers = {}
        for part in self.parts:
            if id(part.data) not in buffers:
                buffers[id(part.data)] = (part.data[:, :index], part.data[:, index:])
        return tuple(
            PaddedEmbedding(
                tuple(
                    EmbeddingRows(buffers[id(p.data)][side], p.indices)
                    for p in self.parts
                ),
                width,
            )
            for side, width in enumerate((index, self.hidden_size - index))
        )


Embedding = torch.Tensor | PaddedEmbedding


def flatten_embedding(embedding: Embedding) -> Embedding:
    if isinstance(embedding, PaddedEmbedding):
        return embedding
    embedding = embedding.reshape(-1, embedding.shape[-1])
    if embedding.device.type == "tpu":
        return PaddedEmbedding.from_tensor(embedding)
    return embedding


def embedding_num_tokens(embedding: Embedding) -> int:
    if isinstance(embedding, PaddedEmbedding):
        return embedding.num_tokens
    return embedding.numel() // embedding.shape[-1]


def slice_embedding(embedding: Embedding, start: int, end: int) -> Embedding:
    embedding = flatten_embedding(embedding)
    if isinstance(embedding, PaddedEmbedding):
        return embedding.slice_rows(start, end)
    return embedding[start:end]


def concat_embeddings(embeddings: list[Embedding]) -> Embedding:
    embeddings = [flatten_embedding(e) for e in embeddings]
    if not any(isinstance(e, PaddedEmbedding) for e in embeddings):
        return torch.cat(embeddings, dim=0)
    padded = [
        e if isinstance(e, PaddedEmbedding) else PaddedEmbedding.from_tensor(e)
        for e in embeddings
    ]
    hidden = padded[0].hidden_size
    if any(e.hidden_size != hidden for e in padded):
        raise ValueError("Cannot concatenate embeddings with different hidden sizes")
    return PaddedEmbedding(tuple(p for e in padded for p in e.parts), hidden)


def split_embedding(embedding: Embedding, sizes: list[int]) -> list[Embedding]:
    embedding = flatten_embedding(embedding)
    if not isinstance(embedding, PaddedEmbedding):
        return list(torch.split(embedding, sizes, dim=0))
    if sum(sizes) != embedding.num_tokens:
        raise ValueError("Per-item token counts do not match the encoder output")
    offset = 0
    result = []
    for size in sizes:
        result.append(embedding.slice_rows(offset, offset + size))
        offset += size
    return result
