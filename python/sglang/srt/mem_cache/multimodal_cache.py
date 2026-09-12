import abc
from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Optional

import torch

from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.multimodal.embedding import Embedding, PaddedEmbedding


class MultimodalCache(abc.ABC):
    @abc.abstractmethod
    def __init__(
        self,
    ): ...

    @staticmethod
    def combine_hashes(mm_hashes: List[int]) -> Optional[int]:
        """
        Get a combined hash from individual mm item hashes
        """
        if not mm_hashes:
            return None
        return hash(tuple(mm_hashes))

    @abc.abstractmethod
    def get(
        self, mm_hashes: List[int], combined_hash: Optional[int] = None
    ) -> Optional[torch.Tensor]:
        """
        Extract the embedding with the hash-ids of the queried items. Try combined hash first, if missed, fallback to individual hashes
        The returned tensor may not be contiguous
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def set(
        self,
        mm_hash: int,
        embedding: torch.Tensor,
        mm_embedding_allocator: BaseTokenToKVPoolAllocator,
    ) -> bool:
        """
        Set the embedding to the pre-allocated locations with a hash id
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def has(self, mm_hash: int) -> bool:
        raise NotImplementedError()

    @abc.abstractmethod
    def free(
        self, mm_hash: int, mm_embedding_allocator: BaseTokenToKVPoolAllocator
    ) -> bool:
        raise NotImplementedError()

    @abc.abstractmethod
    def clear(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def available_size(self):
        raise NotImplementedError()


def _get_storage_sizes(embedding: Embedding) -> dict[int, int]:
    if isinstance(embedding, PaddedEmbedding):
        return embedding.storage_sizes
    return {id(embedding): embedding.element_size() * embedding.numel()}


@dataclass(kw_only=True)
class EmbeddingResult:
    embedding: Embedding


class MultiModalStaticCache(MultimodalCache):
    """
    A server-level cache for multimodal embedding.
    Embeddings are computed prior, and this cache does not really pre-alloc
    """

    def __init__(
        self,
        max_size: int,
    ):
        super().__init__()
        self.max_size = max_size
        self.mm_cache: OrderedDict[int, EmbeddingResult] = OrderedDict()
        self.current_size = 0
        self._storage_refcounts: dict[int, int] = {}

    def get(
        self, mm_hashes: List[int], combined_hash: Optional[int] = None
    ) -> Optional[EmbeddingResult]:
        combined_hash = self.combine_hashes(mm_hashes)
        # MultiModalStaticCache does not fallback to individual item lookup

        embedding = self.mm_cache.get(combined_hash)
        if embedding is not None:
            self.mm_cache.move_to_end(combined_hash)
        return embedding

    def set(
        self,
        mm_hash: int,
        embedding: EmbeddingResult,
        loc: Optional[torch.Tensor] = None,
    ) -> bool:
        assert isinstance(embedding, EmbeddingResult), embedding
        if mm_hash in self.mm_cache:
            self.mm_cache.move_to_end(mm_hash)
            return True
        sizes = _get_storage_sizes(embedding.embedding)
        while (
            self.current_size
            + sum(
                size
                for key, size in sizes.items()
                if key not in self._storage_refcounts
            )
            > self.max_size
        ):
            if not self.mm_cache:
                return False
            lru_hash, lru_embedding = self.mm_cache.popitem(last=False)
            self._release_storage(lru_embedding.embedding)

        self.mm_cache[mm_hash] = embedding
        for key, size in sizes.items():
            if key not in self._storage_refcounts:
                self.current_size += size
                self._storage_refcounts[key] = 0
            self._storage_refcounts[key] += 1
        return True

    def _release_storage(self, embedding: Embedding) -> None:
        for key, size in _get_storage_sizes(embedding).items():
            self._storage_refcounts[key] -= 1
            if not self._storage_refcounts[key]:
                del self._storage_refcounts[key]
                self.current_size -= size

    def get_single(self, mm_hash: int) -> Optional[EmbeddingResult]:
        """Get a single cached embedding by its hash (no combine_hashes)."""
        embedding = self.mm_cache.get(mm_hash)
        if embedding is not None:
            self.mm_cache.move_to_end(mm_hash)
        return embedding

    def has(self, mm_hash: int) -> bool:
        return mm_hash in self.mm_cache

    def free(
        self, mm_hash: int, mm_embedding_allocator: BaseTokenToKVPoolAllocator
    ) -> bool:
        if mm_hash not in self.mm_cache:
            return False
        old_embedding = self.mm_cache.pop(mm_hash)
        self._release_storage(old_embedding.embedding)
        return True

    def clear(self):
        self.mm_cache.clear()
        self.current_size = 0
        self._storage_refcounts.clear()

    def __len__(self):
        return len(self.mm_cache)

    def available_size(self):
        return self.__len__()
