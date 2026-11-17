# SPDX-License-Identifier: Apache-2.0
"""Per-request ring of draft KV slots for an all-sliding-window DFlash draft.

The draft only attends to its last sliding_window positions, so its KV needs no
slot per target token: request row r keeps position p at
trash_tokens + r * ring_tokens + p % ring_tokens. The pool then scales with the
request rows instead of the target's token capacity (8x under DCP8).

Slots [0, trash_tokens) absorb the extend rows a ring drops, so one write never
lands two kept positions on one ring slot; trash is never read, so dropped rows
may share trash slots. Ring contents are not content-stable: a
request that reuses a cached prefix drafts against its slot's earlier contents
until its own extend and verify rows refill the window. Verify keeps the output
exact; only acceptance drops while the window is stale.
"""

from __future__ import annotations

import math

import msgspec
import torch

# Arbitrary: a 4K prefill chunk's dropped rows land in distinct trash slots;
# longer chunks wrap around it, which is harmless since trash is never read.
_TRASH_TOKENS = 4096


class DraftKvRing(msgspec.Struct, frozen=True):
    page_size: int
    window: int
    ring_tokens: int
    trash_tokens: int

    @classmethod
    def create(cls, *, window: int, verify_width: int, page_size: int) -> DraftKvRing:
        # A verify reads [seq_len + width - window, seq_len + width) while it writes
        # width new rows; one extra page keeps ring pages aligned to KV pages.
        ring_tokens = (
            math.ceil((window + 2 * verify_width) / page_size) + 1
        ) * page_size
        trash_tokens = math.ceil(_TRASH_TOKENS / page_size) * page_size
        return cls(
            page_size=page_size,
            window=window,
            ring_tokens=ring_tokens,
            trash_tokens=trash_tokens,
        )

    def num_tokens(self, num_req_rows: int) -> int:
        return self.trash_tokens + num_req_rows * self.ring_tokens

    def slots(self, req_rows: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """int64 ring slots; req_rows broadcasts against positions."""
        return (
            self.trash_tokens
            + req_rows.to(torch.int64) * self.ring_tokens
            + positions.to(torch.int64) % self.ring_tokens
        )

    def req_to_token(self, *, num_req_rows: int, width: int) -> torch.Tensor:
        """Host [num_req_rows, width] int32 table the draft attention pages through."""
        rows = torch.arange(num_req_rows, dtype=torch.int64)[:, None]
        cols = torch.arange(width, dtype=torch.int64)[None, :]
        table = self.slots(rows, cols)
        assert int(table.max()) < 2**31, table.max()
        return table.to(torch.int32)

    def extend_slots(
        self,
        *,
        req_rows: torch.Tensor,
        prefix_lens: list[int],
        extend_lens: list[int],
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Slots for a prompt pass's flat rows; rows older than one ring go to trash."""
        extend = torch.tensor(extend_lens, dtype=torch.int64)
        seq_ends = torch.repeat_interleave(
            torch.tensor(prefix_lens, dtype=torch.int64) + extend, extend
        )
        token_rows = torch.repeat_interleave(req_rows.to(torch.int64), extend)
        kept = positions >= seq_ends - self.ring_tokens
        trash = torch.arange(positions.shape[0], dtype=torch.int64) % self.trash_tokens
        return torch.where(kept, self.slots(token_rows, positions), trash)

    def verify_candidates(
        self,
        *,
        req_rows: torch.Tensor,
        committed_lens: torch.Tensor,
        num_candidates: int,
    ) -> torch.Tensor:
        """[bs, num_candidates] slots of [base, base + num_candidates), as the target window."""
        columns = committed_lens.to(torch.int64)[:, None] + torch.arange(
            num_candidates, dtype=torch.int64
        )
        return self.slots(req_rows.to(torch.int64)[:, None], columns)
