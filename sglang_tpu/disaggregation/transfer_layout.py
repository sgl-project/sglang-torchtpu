"""How a KV pool lays its pages out across attention-TP ranks, for P/D transfer.

A pool declares it with a ``raiden_page_sharding`` class attribute holding one
of the ``PageSharding`` values as a plain string, so pool modules import
nothing from the transport. A pool that declares nothing is ``UNSUPPORTED``.
The Raiden transport reads the declaration to choose how a page moves between
unequal attention TPs.
"""

from __future__ import annotations

import enum


class PageSharding(str, enum.Enum):
    # Token-major fused MHA page; each attention-TP rank holds its kv-head slice.
    HEAD_SLICED = "head_sliced"
    # The same bytes on every attention-TP rank (an MLA latent); moves whole.
    REPLICATED = "replicated"
    # No reshard geometry, so equal attention TP only.
    UNSUPPORTED = "unsupported"
