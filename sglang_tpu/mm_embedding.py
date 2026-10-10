"""Represent logical multimodal rows without slicing their physical buffers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from sglang.srt.mem_cache.multimodal_cache import EmbeddingResult


@dataclass(frozen=True)
class EmbeddingSpan:
    buffer: torch.Tensor
    start: int
    length: int

    def __post_init__(self) -> None:
        if self.buffer.ndim != 2:
            raise ValueError("Padded vision embeddings must have two dimensions")
        if (
            self.start < 0
            or self.length < 0
            or self.start + self.length > self.buffer.shape[0]
        ):
            raise ValueError("Vision embedding span exceeds its padded buffer")


@dataclass(frozen=True)
class PaddedEmbedding:
    spans: tuple[EmbeddingSpan, ...]

    def __post_init__(self) -> None:
        if not self.spans:
            raise ValueError("A padded embedding needs at least one buffer")
        first = self.spans[0].buffer
        for span in self.spans[1:]:
            if (span.buffer.shape[1], span.buffer.dtype, span.buffer.device) != (
                first.shape[1],
                first.dtype,
                first.device,
            ):
                raise ValueError(
                    "Vision embedding buffers must share width, dtype and device"
                )

    @classmethod
    def from_buffer(cls, buffer: torch.Tensor, valid_rows: int) -> PaddedEmbedding:
        return cls((EmbeddingSpan(buffer, 0, valid_rows),))

    @property
    def shape(self) -> torch.Size:
        return torch.Size(
            (sum(span.length for span in self.spans), self.spans[0].buffer.shape[1])
        )

    def slice(self, start: int, stop: int) -> PaddedEmbedding:
        start, stop, _ = slice(start, stop).indices(self.shape[0])
        spans = []
        offset = 0
        for span in self.spans:
            begin = max(start - offset, 0)
            end = min(stop - offset, span.length)
            if begin < end:
                spans.append(
                    EmbeddingSpan(span.buffer, span.start + begin, end - begin)
                )
            offset += span.length
        if not spans:
            spans = [EmbeddingSpan(self.spans[0].buffer, 0, 0)]
        return PaddedEmbedding(tuple(spans))

    def storage_nbytes(self) -> int:
        buffers = {id(span.buffer): span.buffer for span in self.spans}
        return sum(buffer.untyped_storage().nbytes() for buffer in buffers.values())

    def to_host(self) -> torch.Tensor:
        buffers = {id(span.buffer): span.buffer for span in self.spans}
        buffers = {key: buffer.to("cpu") for key, buffer in buffers.items()}
        rows = [
            buffers[id(span.buffer)][span.start : span.start + span.length]
            for span in self.spans
        ]
        return rows[0] if len(rows) == 1 else torch.cat(rows)


MultimodalEmbedding = torch.Tensor | PaddedEmbedding


def flatten_embedding(embedding: MultimodalEmbedding) -> MultimodalEmbedding:
    if isinstance(embedding, PaddedEmbedding):
        return embedding
    return embedding.reshape(-1, embedding.shape[-1])


def slice_embedding(
    embedding: MultimodalEmbedding, start: int, stop: int
) -> MultimodalEmbedding:
    if isinstance(embedding, PaddedEmbedding):
        return embedding.slice(start, stop)
    return embedding[start:stop]


def concat_embeddings(embeddings: Sequence[MultimodalEmbedding]) -> MultimodalEmbedding:
    if any(isinstance(embedding, PaddedEmbedding) for embedding in embeddings):
        spans = []
        for embedding in embeddings:
            if isinstance(embedding, PaddedEmbedding):
                spans.extend(embedding.spans)
            else:
                buffer = flatten_embedding(embedding)
                spans.append(EmbeddingSpan(buffer, 0, buffer.shape[0]))
        return PaddedEmbedding(tuple(spans))
    return torch.cat(embeddings, dim=0)


def split_embedding(
    embedding: MultimodalEmbedding, row_counts: Sequence[int]
) -> Sequence[MultimodalEmbedding]:
    embedding = flatten_embedding(embedding)
    if not isinstance(embedding, PaddedEmbedding):
        return torch.split(embedding, row_counts, dim=0)
    if sum(row_counts) != embedding.shape[0]:
        raise ValueError("Vision embedding rows do not match item token counts")
    offset = 0
    parts = []
    for count in row_counts:
        parts.append(embedding.slice(offset, offset + count))
        offset += count
    return parts


@dataclass(kw_only=True)
class TpuEmbeddingResult(EmbeddingResult):
    embedding: MultimodalEmbedding

    def storage_size(self) -> int:
        if isinstance(self.embedding, PaddedEmbedding):
            return self.embedding.storage_nbytes()
        return super().storage_size()

    def cache_size(self) -> int:
        if isinstance(self.embedding, PaddedEmbedding):
            return self.storage_size()
        return super().cache_size()
