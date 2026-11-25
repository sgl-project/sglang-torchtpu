from __future__ import annotations

from typing import TYPE_CHECKING, Callable, List, Optional

from sglang.srt.mem_cache.common import match_kv_cache
from sglang.srt.utils.common import ceil_div

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache


class PrefillChunkSharer:
    """Lets waiting requests share a prefill pass with a prompt prefilled in
    chunks. The chunk is capped to leave room for the requests whose uncached
    prefill fits, in queue order, passing over longer ones; the chunked prompt
    keeps at least `min_tokens` of each pass."""

    def __init__(
        self,
        *,
        min_tokens: int,
        tree_cache: BasePrefixCache,
        prefill_work: Callable[[Req], int],
        scan_limit: int,
    ):
        self._min_tokens = min_tokens
        self._tree_cache = tree_cache
        self._prefill_work = prefill_work
        self._scan_limit = scan_limit

    def chunk_limit(
        self,
        *,
        chunked_req: Req,
        waiting_queue: List[Req],
        budget: int,
        page_size: int,
        prefix_matched: bool,
    ) -> Optional[int]:
        """Cap for the continuing chunk this pass, or None to leave it uncapped."""
        candidates = list(waiting_queue)[: self._scan_limit]
        if not prefix_matched:
            # Fit is judged by uncached length. Head last, as in SchedulePolicy's
            # _touch_waiting_prefixes, so LRU evicts in reverse queue order.
            for req in reversed(candidates):
                match_kv_cache(req, self._tree_cache)
        keep = ceil_div(self._min_tokens, page_size) * page_size
        remaining = len(chunked_req.full_untruncated_fill_ids) - chunked_req.prefix_len
        reserved = 0
        # Queue order, passing over a request that is not shorter than the
        # chunked prompt's rest or does not fit; it waits for a later pass.
        for req in candidates:
            work = self._prefill_work(req)
            charge = ceil_div(work, page_size) * page_size
            if work < remaining and reserved + charge <= budget - keep:
                reserved += charge
        if not reserved:
            return None
        return (budget - reserved) // page_size * page_size

    def can_join(self, *, req: Req, rem_chunk_tokens: Optional[int]) -> bool:
        """Whether req may join a pass beside the continuing chunk: only when its
        uncached prefill fits what the pass has left, so a longer request keeps
        its queue place."""
        return rem_chunk_tokens is None or self._prefill_work(req) <= rem_chunk_tokens
