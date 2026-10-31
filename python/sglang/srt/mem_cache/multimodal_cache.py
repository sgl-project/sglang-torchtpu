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

    def backing_buffers(self) -> tuple[torch.Tensor, ...]:
        """Buffer tensors this entry keeps alive.

        The cache charges a buffer once however many entries hold the same tensor
        object. Different views of one storage count as different buffers, which
        over-counts but never under-counts.
        """
        return (self.embedding,)


@dataclass
class _BufferCharge:
    # Holding the buffer keeps its id from being reused while it is charged.
    buffer: torch.Tensor
    nbytes: int
    holders: int


def _buffers(embedding: EmbeddingResult) -> dict[int, torch.Tensor]:
    return {id(buffer): buffer for buffer in embedding.backing_buffers()}


def _nbytes(buffer: torch.Tensor) -> int:
    return buffer.untyped_storage().nbytes()


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
        # Buffer id -> its charge, shared by every entry holding the buffer.
        self._buffer_charges: dict[int, _BufferCharge] = {}

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
        # A slice of a larger tensor is cloned on admission, so it is charged its
        # own bytes rather than the tensor it was sliced from.
        clone_on_admit = embedding.storage_size() > embedding.cache_size()
        total = (
            embedding.cache_size()
            if clone_on_admit
            else sum(_nbytes(buffer) for buffer in _buffers(embedding).values())
        )
        # An entry larger than the whole cache must not evict useful entries on
        # every chunk before it is rejected.
        if total > self.max_size:
            return False
        while True:
            # Recomputed after each eviction: evicting an entry that shares a
            # buffer with the new one leaves that buffer uncharged.
            needed = (
                embedding.cache_size()
                if clone_on_admit
                else self._bytes_to_charge(embedding)
            )
            if self.current_size + needed <= self.max_size:
                break
            if not self.mm_cache:
                return False
            _, lru_embedding = self.mm_cache.popitem(last=False)
            self._release(lru_embedding)

        if clone_on_admit:
            # Clone admitted slices so one cached item cannot retain a whole batch.
            embedding = replace(embedding, embedding=tensor.clone())
        self.mm_cache[mm_hash] = embedding
        self._acquire(embedding)
        return True

    def _bytes_to_charge(self, embedding: EmbeddingResult) -> int:
        """Bytes of the entry's buffers that no cached entry holds yet."""
        return sum(
            _nbytes(buffer)
            for buffer_id, buffer in _buffers(embedding).items()
            if buffer_id not in self._buffer_charges
        )

    def _acquire(self, embedding: EmbeddingResult) -> None:
        for buffer_id, buffer in _buffers(embedding).items():
            charge = self._buffer_charges.get(buffer_id)
            if charge is None:
                charge = _BufferCharge(buffer, _nbytes(buffer), holders=0)
                self._buffer_charges[buffer_id] = charge
                self.current_size += charge.nbytes
            charge.holders += 1

    def _release(self, embedding: EmbeddingResult) -> None:
        for buffer_id in _buffers(embedding):
            charge = self._buffer_charges[buffer_id]
            charge.holders -= 1
            if charge.holders == 0:
                del self._buffer_charges[buffer_id]
                self.current_size -= charge.nbytes

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
        self._buffer_charges.clear()
        self.current_size = 0

    def __len__(self):
        return len(self.mm_cache)

    def available_size(self):
        return self.__len__()
