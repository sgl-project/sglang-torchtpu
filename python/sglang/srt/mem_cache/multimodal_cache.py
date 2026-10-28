import abc
from collections import OrderedDict
from dataclasses import dataclass, replace
from typing import List, Optional

import torch

from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator


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


@dataclass(kw_only=True)
class EmbeddingResult:
    embedding: torch.Tensor

    def storage_size(self) -> int:
        """Bytes retained by this cache entry's backing storage."""
        return self.embedding.untyped_storage().nbytes()

    def cache_size(self) -> int:
        """Bytes needed after compacting an admitted entry."""
        return min(
            self.storage_size(), self.embedding.element_size() * self.embedding.numel()
        )

    def backing_buffers(self) -> dict[int, int]:
        """Bytes of each buffer this entry keeps alive, keyed by ``id()`` of the buffer tensor.

        The cache charges a buffer once however many entries hold the same tensor
        object. Different views of one storage count as different buffers, which
        over-counts but never under-counts.
        """
        return {id(self.embedding): self.storage_size()}


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
        # Buffer identity -> [bytes, entries holding it]. Entries keep their
        # buffers alive, so an identity is not reused while it is listed here.
        self._buffer_refs: dict[int, list[int]] = {}

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
        tensor = embedding.embedding
        compact = embedding.storage_size() > embedding.cache_size()
        total = (
            embedding.cache_size()
            if compact
            else sum(embedding.backing_buffers().values())
        )
        # An entry larger than the whole cache must not evict useful entries on
        # every chunk before it is rejected.
        if total > self.max_size:
            return False
        while True:
            # Recomputed after each eviction: evicting an entry that shares a
            # buffer with the new one leaves that buffer uncharged.
            needed = embedding.cache_size() if compact else self._uncharged(embedding)
            if self.current_size + needed <= self.max_size:
                break
            if not self.mm_cache:
                return False
            _, lru_embedding = self.mm_cache.popitem(last=False)
            self._release(lru_embedding)

        if compact:
            # Clone admitted slices so one cached item cannot retain a whole batch.
            embedding = replace(embedding, embedding=tensor.clone())
        self.mm_cache[mm_hash] = embedding
        self._acquire(embedding)
        return True

    def _uncharged(self, embedding: EmbeddingResult) -> int:
        return sum(
            nbytes
            for key, nbytes in embedding.backing_buffers().items()
            if key not in self._buffer_refs
        )

    def _acquire(self, embedding: EmbeddingResult) -> None:
        for key, nbytes in embedding.backing_buffers().items():
            ref = self._buffer_refs.get(key)
            if ref is None:
                self._buffer_refs[key] = [nbytes, 1]
                self.current_size += nbytes
            else:
                ref[1] += 1

    def _release(self, embedding: EmbeddingResult) -> None:
        for key in embedding.backing_buffers():
            ref = self._buffer_refs[key]
            ref[1] -= 1
            if ref[1] == 0:
                del self._buffer_refs[key]
                self.current_size -= ref[0]

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
        self._release(old_embedding)
        return True

    def clear(self):
        self.mm_cache.clear()
        self._buffer_refs.clear()
        self.current_size = 0

    def __len__(self):
        return len(self.mm_cache)

    def available_size(self):
        return self.__len__()
