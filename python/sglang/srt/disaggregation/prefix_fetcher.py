"""Optional hook that fetches a request's prompt prefix before its prefill.

A disaggregated prefill may get part of a prompt's KV from elsewhere (for
example from the decode that cached it) instead of computing it. The fetch runs
between bootstrap and the waiting queue: PrefillBootstrapQueue keeps a request
whose fetch started in its bootstrap queue, and releases it once its sender
reports WaitingForInput again. A fetcher therefore holds its request's sender
out of WaitingForInput until the fetch is ready, so pipeline stages agree on
the release through the bootstrap consensus; a fetch that fails reports
Failed, and abandon() either fails the request or recovers it on every rank
alike. Unset, the queue behaves as before.
"""

from __future__ import annotations

from typing import Protocol

from sglang.srt.managers.schedule_batch import Req


class PrefixFetcher(Protocol):
    def start(self, req: Req) -> bool:
        """After the request's bootstrap finalized: True when a fetch started and
        the request stays in the bootstrap queue until it is ready."""
        ...

    def is_fetching(self, req: Req) -> bool:
        """Whether start() holds the request."""
        ...

    def finish(self, req: Req) -> None:
        """The fetch is ready on every rank: hand its KV to the prefix cache and
        leave the request ready for the waiting queue."""
        ...

    def abandon(self, req: Req) -> bool:
        """The held request's sender reported Failed: release what the fetch
        holds, never pages a transfer may still write. True recovers the request,
        which then prefills without the fetched prefix; False lets it fail."""
        ...
