# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Zigzag context parallel strategy shell.

For ``cp_size = 4``, each sequence is split into ``2 * cp_size`` blocks. Each
rank owns one early block and one late block:

    cp0: block0, block7
    cp1: block1, block6
    cp2: block2, block5
    cp3: block3, block4

After all-gather, the blocks are reranged back to their original order:

    block0 | block7 | block1 | block6 | block2 | block5 | block3 | block4
      -> block0 | block1 | block2 | block3 | block4 | block5 | block6 | block7
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from itertools import accumulate
from typing import Any, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.layers.cp.base import (
    BaseContextParallelMetadata,
    ContextParallelStrategy,
    ContextParallelStrategyKind,
    CPAttentionBackendKind,
)
from sglang.srt.layers.cp.padding import pad_local_rows
from sglang.srt.layers.dp_attention import (
    is_allocation_symmetric,
)
from sglang.srt.mem_cache.memory_pool import KVWriteLoc
from sglang.srt.model_executor.forward_context import get_token_to_kv_pool
from sglang.srt.runtime_context import get_device, get_parallel


@dataclass
class ZigzagContextParallelMetadata(BaseContextParallelMetadata):
    # Layout lists have length bs * cp_segment_num (= bs * 2 * cp_size).
    split_list: Optional[List[int]] = None
    zigzag_index: Optional[List[int]] = None
    cp_reverse_index: Optional[List[int]] = None
    reverse_split_len: Optional[List[int]] = None
    reorder_index_tensor: Optional[Any] = None
    perm: Optional[List[int]] = None
    direct_reorder_index: Optional[Any] = None
    gather_reorder_index: Optional[Any] = None
    # Host (numpy int32) owner rank / local row of each natural-order token;
    # `padding.gather_reorder_index_for` re-strides them for a gather whose
    # per-rank row count differs from `stride`.
    gather_rank: Optional[Any] = None
    gather_local: Optional[Any] = None
    # Page-aligned layout (TPU): per request, the owner rank and the row
    # inside that rank's local block of every page of its new KV, in global
    # token order; `new_kv_num_pages` pages are valid per request. The shim
    # turns these into `new_kv_page_indices` for the bucketed gather buffer.
    new_kv_page_rank: Optional[Any] = None
    new_kv_page_row: Optional[Any] = None
    new_kv_num_pages: Optional[Any] = None
    # Per-request layout lengths (page-padded on TPU; == extend lens otherwise).
    layout_seqs_len: Optional[List[int]] = None
    # Leading pad rows per request under `pool_align` (0 otherwise).
    layout_leads: Optional[List[int]] = None
    layout_page: int = 0
    local_rows: Optional[Any] = None
    cp_last_rows: Optional[Any] = None
    cp_last_owner_mask: Optional[Any] = None
    stride: int = 0

    # Per-rank aggregate lists have length cp_size.
    per_rank_actual_token: Optional[List[int]] = None
    max_rank_len: Optional[List[int]] = None
    per_rank_logical_token: Optional[List[int]] = None

    # Per-sequence FlashAttention tensors (shape [bs] or [bs + 1]).
    kv_len_prev_tensor: Optional[Any] = None
    kv_len_next_tensor: Optional[Any] = None
    cu_seqlens_kv_prev_tensor: Optional[Any] = None
    cu_seqlens_kv_next_tensor: Optional[Any] = None
    actual_seq_q_prev_tensor: Optional[Any] = None
    actual_seq_q_next_tensor: Optional[Any] = None
    cu_seqlens_q_prev_tensor: Optional[Any] = None
    cu_seqlens_q_next_tensor: Optional[Any] = None

    # Combined prev-then-next TRT-LLM geometry (shape [2 * bs] or [2 * bs + 1]).
    actual_seq_q_combined_tensor: Optional[Any] = None
    kv_len_combined_tensor: Optional[Any] = None
    cu_seqlens_q_combined_tensor: Optional[Any] = None
    cu_seqlens_kv_combined_tensor: Optional[Any] = None

    # Scalars derived from the per-sequence lists above.
    total_q_prev_tokens: int = 0
    total_q_next_tokens: int = 0
    max_seqlen_q_prev: int = 0
    max_seqlen_q_next: int = 0
    max_seqlen_q_combined: int = 0

    # Per-sequence CPU lists, useful for indexers and diagnostics.
    kv_len_prev_list: Optional[List[int]] = None
    kv_len_next_list: Optional[List[int]] = None
    actual_seq_q_prev_list: Optional[List[int]] = None
    actual_seq_q_next_list: Optional[List[int]] = None
    # Per request, the largest head / tail chunk over all CP ranks (equal to
    # the actual_* lists unless the TPU page-granular layout is on).
    max_seq_q_prev_list: Optional[List[int]] = None
    max_seq_q_next_list: Optional[List[int]] = None


ContextParallelMetadata = ZigzagContextParallelMetadata


# TPU prefill-CP layout knobs (off = the CUDA zigzag split, token granular).
#
# page_align > 0: every zigzag chunk is a whole number of KV pages, so each
# rank's chunks start on page boundaries in the rank-major all-gathered K/V
# and the attention kernel can address that buffer through a page table
# (`new_kv_page_indices`) instead of the host reordering 16K rows per layer.
# A request whose length is not a page multiple is laid out as if padded to
# one; the padding rows duplicate its last real token and the kernel ignores
# them through `kv_new_lens`.
#
# balance_pairs: instead of rank r always owning zigzag pair r, each request
# hands its largest pair to the least-loaded rank so far (same permutation on
# every rank). With page-granular chunks a request's pairs differ by up to a
# page, and the scheduler fills a chunk budget page-ceiled, so this keeps the
# per-rank row count at budget / cp (e.g. a 63-page request plus a 1-page
# head of the next one is exactly 8 pages on every rank of 8) instead of
# spilling one rank into the next token bucket.
# pool_align: lay each request out starting at the pool page boundary below
# its prefix, i.e. with `prefix % page` leading pad rows, so every layout page
# is exactly one pool page (replicated-pool prefill writes whole pages; the
# zero-copy new-KV table of the two-phase path is NOT valid with this on).
_TPU_LAYOUT = {"page_align": 0, "balance_pairs": False, "pool_align": False}


def set_zigzag_tpu_layout(
    page_align: int = 0, balance_pairs: bool = False, pool_align: bool = False
) -> None:
    _TPU_LAYOUT["page_align"] = int(page_align or 0)
    _TPU_LAYOUT["balance_pairs"] = bool(balance_pairs)
    _TPU_LAYOUT["pool_align"] = bool(pool_align) and _TPU_LAYOUT["page_align"] > 0


def _concat_ranges(starts: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Concatenate ``range(starts[i], starts[i] + lengths[i])`` for all i."""
    lengths = np.asarray(lengths, dtype=np.int64)
    total = int(lengths.sum())
    if total == 0:
        return np.zeros((0,), dtype=np.int64)
    ends = np.cumsum(lengths)
    begins = ends - lengths
    # offset of each output element to the start of its own range
    idx = np.arange(total, dtype=np.int64)
    which = np.searchsorted(ends, idx, side="right")
    return np.asarray(starts, dtype=np.int64)[which] + (idx - begins[which])


class ZigzagCPStrategy(ContextParallelStrategy):
    name = "zigzag"
    kind = ContextParallelStrategyKind.ZIGZAG

    def can_apply(self, num_tokens: int, forward_batch) -> bool:
        if self.cp_size <= 1 or num_tokens < self.cp_size * 2:
            return False
        forward_mode = getattr(forward_batch, "forward_mode", None)
        if forward_mode is not None and not forward_mode.is_context_parallel_extend():
            return False

        extend_lens = getattr(forward_batch, "extend_seq_lens_cpu", None)
        if extend_lens is None:
            return True
        return all(int(length) >= self.cp_size * 2 for length in extend_lens)

    def build_metadata(
        self,
        num_tokens: int,
        seqs_len: Optional[List[int]],
        extend_seqs_len: Optional[List[int]] = None,
    ) -> ZigzagContextParallelMetadata:
        if extend_seqs_len is None:
            extend_seqs_len = seqs_len or [num_tokens]
        extend_seqs_len = [int(x) for x in extend_seqs_len]

        pad_len = int(num_tokens) - sum(extend_seqs_len)
        if pad_len > 0:
            extend_seqs_len[-1] += pad_len
            if seqs_len is not None and len(seqs_len) == len(extend_seqs_len):
                seqs_len = list(seqs_len)
                seqs_len[-1] += pad_len

        bs = len(extend_seqs_len)
        cp_segment_num = self.cp_size * 2
        if seqs_len is not None and len(seqs_len) == bs:
            prefix_offsets = [
                max(int(seqs_len[i]) - extend_seqs_len[i], 0) for i in range(bs)
            ]
        else:
            prefix_offsets = [0] * bs

        page = _TPU_LAYOUT["page_align"]
        balance = _TPU_LAYOUT["balance_pairs"] and page > 0
        pool_align = _TPU_LAYOUT["pool_align"] and page > 0
        cp = self.cp_size
        # Leading pad rows per request (pool_align): the request starts at
        # the pool page boundary below its prefix.
        leads = [
            (prefix_offsets[i] % page) if (pool_align and extend_seqs_len[i] > 0) else 0
            for i in range(bs)
        ]
        # Layout lengths: page-padded on TPU, the real lengths otherwise.
        if page > 0:
            layout_lens = [
                -(-(leads[i] + length) // page) * page
                for i, length in enumerate(extend_seqs_len)
            ]
        else:
            layout_lens = list(extend_seqs_len)

        per_seq_block_sizes: List[List[int]] = []
        split_list: List[int] = []
        for length in layout_lens:
            if page > 0:
                units, unit = length // page, page
            else:
                units, unit = length, 1
            base = units // cp_segment_num
            rem = units % cp_segment_num
            block_sizes = [
                (base + 1 if block_id < rem else base) * unit
                for block_id in range(cp_segment_num)
            ]
            per_seq_block_sizes.append(block_sizes)
            split_list.extend(block_sizes)

        # Zigzag pair p = chunks (p, 2cp-1-p). Rank r of request b owns pair
        # pair_of[b][r]: p itself, or under `balance_pairs` the greedy
        # largest-pair-to-least-loaded-rank assignment (LPT), identical on
        # every rank since it only depends on the batch.
        pair_of: List[List[int]] = []
        loads = [0] * cp
        for b in range(bs):
            if not balance:
                pair_of.append(list(range(cp)))
                continue
            sizes = per_seq_block_sizes[b]
            pair_sz = [sizes[p] + sizes[cp_segment_num - 1 - p] for p in range(cp)]
            pairs_desc = sorted(range(cp), key=lambda p: (-pair_sz[p], p))
            ranks_asc = sorted(range(cp), key=lambda r: (loads[r], r))
            assign = [0] * cp
            for p, r in zip(pairs_desc, ranks_asc):
                assign[r] = p
                loads[r] += pair_sz[p]
            pair_of.append(assign)

        def head_chunk(rank: int, b: int) -> int:
            return pair_of[b][rank]

        def tail_chunk(rank: int, b: int) -> int:
            return cp_segment_num - 1 - head_chunk(rank, b)

        per_rank_actual_token = []
        for rank in range(cp):
            per_rank_actual_token.append(
                sum(
                    per_seq_block_sizes[b][head_chunk(rank, b)]
                    + per_seq_block_sizes[b][tail_chunk(rank, b)]
                    for b in range(bs)
                )
            )
        max_rank_len = [max(per_rank_actual_token)] * cp

        cp_rank = self.cp_rank
        zigzag_index: List[int] = []
        for batch_id in range(bs):
            zigzag_index.append(
                batch_id * cp_segment_num + head_chunk(cp_rank, batch_id)
            )
            zigzag_index.append(
                batch_id * cp_segment_num + tail_chunk(cp_rank, batch_id)
            )

        reverse_split_len: List[int] = []
        gathered_chunk_ids: List[int] = []
        for r in range(cp):
            for batch_id in range(bs):
                h, t = head_chunk(r, batch_id), tail_chunk(r, batch_id)
                reverse_split_len.append(per_seq_block_sizes[batch_id][h])
                gathered_chunk_ids.append(batch_id * cp_segment_num + h)
                reverse_split_len.append(per_seq_block_sizes[batch_id][t])
                gathered_chunk_ids.append(batch_id * cp_segment_num + t)

        gathered_pos = {c: i for i, c in enumerate(gathered_chunk_ids)}
        cp_reverse_index = [gathered_pos[c] for c in range(bs * cp_segment_num)]

        kv_len_prev_list: List[int] = []
        kv_len_next_list: List[int] = []
        actual_seq_q_prev_list: List[int] = []
        actual_seq_q_next_list: List[int] = []
        # Largest head / tail chunk of each request over the whole CP group.
        # With the page-granular split chunks of one request can differ by a
        # page, and the ring (cache phase) needs every rank to walk the same
        # number of Q blocks per virtual sequence.
        max_seq_q_prev_list: List[int] = []
        max_seq_q_next_list: List[int] = []
        for batch_id, block_sizes in enumerate(per_seq_block_sizes):
            h, t = head_chunk(cp_rank, batch_id), tail_chunk(cp_rank, batch_id)
            base_pos = prefix_offsets[batch_id] - leads[batch_id]
            kv_len_prev_list.append(base_pos + sum(block_sizes[: h + 1]))
            kv_len_next_list.append(base_pos + sum(block_sizes[: t + 1]))
            actual_seq_q_prev_list.append(block_sizes[h])
            actual_seq_q_next_list.append(block_sizes[t])
            max_seq_q_prev_list.append(max(block_sizes[:cp]))
            max_seq_q_next_list.append(max(block_sizes[cp:]))

        try:
            device = torch.device(get_device().device)
        except Exception:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cu_prev = [0] + list(accumulate(actual_seq_q_prev_list))
        cu_next = [0] + list(accumulate(actual_seq_q_next_list))
        cu_kv_prev = [0] + list(accumulate(kv_len_prev_list))
        cu_kv_next = [0] + list(accumulate(kv_len_next_list))
        actual_seq_q_combined_list = actual_seq_q_prev_list + actual_seq_q_next_list
        kv_len_combined_list = kv_len_prev_list + kv_len_next_list
        cu_q_combined = [0] + list(accumulate(actual_seq_q_combined_list))
        cu_kv_combined = [0] + list(accumulate(kv_len_combined_list))

        total_seq_lens = sum(layout_lens)
        assert len(split_list) == bs * cp_segment_num
        assert sum(split_list) == total_seq_lens
        assert len(zigzag_index) == 2 * bs
        assert len(cp_reverse_index) == bs * cp_segment_num
        assert sorted(cp_reverse_index) == list(range(bs * cp_segment_num))
        assert sum(per_rank_actual_token) == total_seq_lens

        # Host index construction, vectorized with numpy: every quantity below
        # is a concatenation of integer ranges, and at 16K tokens the Python
        # loops this replaces cost ~6 ms per prefill batch.
        rsl = np.asarray(reverse_split_len, dtype=np.int64)
        offsets = np.concatenate([[0], np.cumsum(rsl)])
        rev = np.asarray(cp_reverse_index, dtype=np.int64)
        perm = _concat_ranges(offsets[rev], rsl[rev])
        reorder_index_tensor = torch.from_numpy(perm.astype(np.int32)).to(device)

        stride = (
            max(per_rank_actual_token) if per_rank_actual_token else int(num_tokens)
        )
        rank_offsets = np.concatenate(
            [[0], np.cumsum(np.asarray(per_rank_actual_token, dtype=np.int64))]
        )
        gather_rank = np.searchsorted(rank_offsets, perm, side="right") - 1
        gather_local = perm - rank_offsets[gather_rank]
        slot = (cp_rank - gather_rank) % self.cp_size
        direct_reorder_index = torch.from_numpy(
            (slot * stride + gather_local).astype(np.int32)
        ).to(device)
        gather_reorder_index = torch.from_numpy(
            (gather_rank * stride + gather_local).astype(np.int32)
        ).to(device)

        sl = np.asarray(split_list, dtype=np.int64)
        chunk_offsets = np.concatenate([[0], np.cumsum(sl)])
        zz = np.asarray(zigzag_index, dtype=np.int64)
        # Local rows in *layout* token space (page-padded per request on TPU).
        local_layout = _concat_ranges(chunk_offsets[zz], sl[zz])
        n_local = len(local_layout)
        ext = np.asarray(extend_seqs_len, dtype=np.int64)
        lay = np.asarray(layout_lens, dtype=np.int64)
        real_start = np.concatenate([[0], np.cumsum(ext)])
        lay_start = np.concatenate([[0], np.cumsum(lay)])
        lead_arr = np.asarray(leads, dtype=np.int64)
        if page > 0 and n_local:
            # Map each layout row to a real global token: padding rows of a
            # request duplicate its first (leading pad) / last real token.
            req = np.searchsorted(lay_start, local_layout, side="right") - 1
            within = local_layout - lay_start[req] - lead_arr[req]
            within = np.clip(within, 0, np.maximum(ext[req] - 1, 0))
            local_indices = real_start[req] + within
        else:
            local_indices = local_layout
        local_indices_padded = np.zeros((max(stride, n_local),), dtype=np.int32)
        local_indices_padded[:n_local] = local_indices
        local_rows = torch.from_numpy(local_indices_padded).to(device)

        # Last real token of each request -> local row (owner rank only).
        # local rows are sorted ascending per zigzag chunk but not globally;
        # map layout token -> local row through an inverse table.
        last_tok = lay_start[:-1] + lead_arr + np.maximum(ext - 1, 0)
        inv = np.full((int(total_seq_lens) + 1,), -1, dtype=np.int64)
        inv[local_layout] = np.arange(n_local)
        loc = inv[np.where(ext > 0, last_tok, 0)]
        owner = (ext > 0) & (loc >= 0)
        cp_last_rows = torch.from_numpy(np.where(owner, loc, 0).astype(np.int32)).to(
            device
        )
        cp_last_owner_mask = torch.from_numpy(owner).to(device)

        # Page table of the rank-major gathered K/V (TPU page-aligned layout):
        # page p of request b -> (owner rank, row inside that rank's block).
        new_kv_page_rank = new_kv_page_row = new_kv_num_pages = None
        if page > 0:
            n_pages = lay // page
            max_pages = int(n_pages.max()) if bs else 0
            new_kv_page_rank = np.zeros((bs, max(max_pages, 1)), dtype=np.int32)
            new_kv_page_row = np.zeros((bs, max(max_pages, 1)), dtype=np.int32)
            # Row offset of (rank, b, head/tail) inside rank's local block:
            # local order on every rank is [b0 head, b0 tail, b1 head, b1 tail, ...].
            for r in range(cp):
                off = 0
                for b in range(bs):
                    sizes = per_seq_block_sizes[b]
                    h, t = head_chunk(r, b), tail_chunk(r, b)
                    chunk_page0 = np.cumsum([0] + sizes[:-1]) // page
                    for c, row0 in ((h, off), (t, off + sizes[h])):
                        n_c = sizes[c] // page
                        if n_c:
                            p0 = int(chunk_page0[c])
                            new_kv_page_rank[b, p0 : p0 + n_c] = r
                            new_kv_page_row[b, p0 : p0 + n_c] = (
                                row0 + np.arange(n_c) * page
                            )
                    off += sizes[h] + sizes[t]
            new_kv_num_pages = n_pages.astype(np.int32)

        return ZigzagContextParallelMetadata(
            split_list=split_list,
            zigzag_index=zigzag_index,
            cp_reverse_index=cp_reverse_index,
            reverse_split_len=reverse_split_len,
            reorder_index_tensor=reorder_index_tensor,
            perm=perm.tolist(),
            direct_reorder_index=direct_reorder_index,
            gather_reorder_index=gather_reorder_index,
            gather_rank=gather_rank.astype(np.int32),
            gather_local=gather_local.astype(np.int32),
            local_rows=local_rows,
            cp_last_rows=cp_last_rows,
            cp_last_owner_mask=cp_last_owner_mask,
            new_kv_page_rank=new_kv_page_rank,
            new_kv_page_row=new_kv_page_row,
            new_kv_num_pages=new_kv_num_pages,
            layout_seqs_len=layout_lens,
            layout_leads=leads,
            layout_page=page,
            stride=stride,
            max_seq_q_prev_list=max_seq_q_prev_list,
            max_seq_q_next_list=max_seq_q_next_list,
            per_rank_actual_token=per_rank_actual_token,
            max_rank_len=max_rank_len,
            kv_len_prev_tensor=torch.tensor(
                kv_len_prev_list, device=device, dtype=torch.int32
            ),
            kv_len_next_tensor=torch.tensor(
                kv_len_next_list, device=device, dtype=torch.int32
            ),
            cu_seqlens_kv_prev_tensor=torch.tensor(
                cu_kv_prev, device=device, dtype=torch.int32
            ),
            cu_seqlens_kv_next_tensor=torch.tensor(
                cu_kv_next, device=device, dtype=torch.int32
            ),
            actual_seq_q_prev_tensor=torch.tensor(
                actual_seq_q_prev_list, device=device, dtype=torch.int32
            ),
            actual_seq_q_next_tensor=torch.tensor(
                actual_seq_q_next_list, device=device, dtype=torch.int32
            ),
            cu_seqlens_q_prev_tensor=torch.tensor(
                cu_prev, device=device, dtype=torch.int32
            ),
            cu_seqlens_q_next_tensor=torch.tensor(
                cu_next, device=device, dtype=torch.int32
            ),
            actual_seq_q_combined_tensor=torch.tensor(
                actual_seq_q_combined_list, device=device, dtype=torch.int32
            ),
            kv_len_combined_tensor=torch.tensor(
                kv_len_combined_list, device=device, dtype=torch.int32
            ),
            cu_seqlens_q_combined_tensor=torch.tensor(
                cu_q_combined, device=device, dtype=torch.int32
            ),
            cu_seqlens_kv_combined_tensor=torch.tensor(
                cu_kv_combined, device=device, dtype=torch.int32
            ),
            total_q_prev_tokens=cu_prev[-1],
            total_q_next_tokens=cu_next[-1],
            max_seqlen_q_prev=(
                max(actual_seq_q_prev_list) if actual_seq_q_prev_list else 0
            ),
            max_seqlen_q_next=(
                max(actual_seq_q_next_list) if actual_seq_q_next_list else 0
            ),
            max_seqlen_q_combined=(
                max(actual_seq_q_combined_list) if actual_seq_q_combined_list else 0
            ),
            kv_len_prev_list=kv_len_prev_list,
            kv_len_next_list=kv_len_next_list,
            actual_seq_q_prev_list=actual_seq_q_prev_list,
            actual_seq_q_next_list=actual_seq_q_next_list,
            total_seq_lens=total_seq_lens,
            bs=bs,
        )

    def shard_hidden_states(self, x: Any, forward_batch) -> Any:
        metadata = forward_batch.attn_cp_metadata
        x = x[: metadata.total_seq_lens]
        chunks = torch.split(x, metadata.split_list, dim=0)
        local_x = torch.cat([chunks[i] for i in metadata.zigzag_index], dim=0)
        return pad_local_rows(local_x, metadata, dim=0)

    def shard_position_ids(self, positions: Any, forward_batch) -> Any:
        metadata = forward_batch.attn_cp_metadata
        positions = positions[..., : metadata.total_seq_lens]
        chunks = torch.split(positions, metadata.split_list, dim=-1)
        local_positions = torch.cat([chunks[i] for i in metadata.zigzag_index], dim=-1)
        return pad_local_rows(local_positions, metadata, dim=-1)

    def gather_hidden_states(
        self, x: Any, forward_batch, stream: Optional[Any] = None
    ) -> Any:
        meta = forward_batch.attn_cp_metadata
        if x.device.type == "tpu" and (
            getattr(meta, "gather_reorder_index", None) is not None
            or getattr(meta, "direct_reorder_index", None) is not None
        ):
            from sglang_tpu.batched_rpa_cp import (
                cp_all_gather_rows,
            )

            minor = tuple(x.shape[1:])
            x2 = x.reshape(x.shape[0], -1).contiguous()
            out = cp_all_gather_rows(x2, forward_batch)
            return out.reshape(-1, *minor)

        gathered = self._all_gather_reorganized(x, forward_batch)
        reorder_index = getattr(meta, "reorder_index_tensor", None)
        if reorder_index is not None:
            if reorder_index.device != gathered.device:
                reorder_index = reorder_index.to(gathered.device)
            return gathered.index_select(0, reorder_index)
        chunks = torch.split(gathered, meta.reverse_split_len, dim=0)
        return torch.cat([chunks[i] for i in meta.cp_reverse_index], dim=0)

    def gather_kv_cache(
        self, x: Any, forward_batch, stream: Optional[Any] = None
    ) -> Any:
        meta = forward_batch.attn_cp_metadata
        if x.device.type == "tpu" and (
            getattr(meta, "gather_reorder_index", None) is not None
            or getattr(meta, "direct_reorder_index", None) is not None
        ):
            from sglang_tpu.batched_rpa_cp import (
                cp_all_gather_rows,
            )

            minor = tuple(x.shape[1:])
            x2 = x.reshape(x.shape[0], -1).contiguous()
            out = cp_all_gather_rows(x2, forward_batch)
            return out.reshape(-1, *minor)

        gathered = self._all_gather_reorganized(x, forward_batch)
        reorder_index = getattr(meta, "reorder_index_tensor", None)
        if reorder_index is not None:
            if reorder_index.device != gathered.device:
                reorder_index = reorder_index.to(gathered.device)
            return gathered.index_select(0, reorder_index)
        chunks = torch.split(gathered, meta.reverse_split_len, dim=0)
        return torch.cat([chunks[i] for i in meta.cp_reverse_index], dim=0)

    def local_q_indices(self, num_tokens: int, forward_batch) -> Any:
        meta = forward_batch.attn_cp_metadata
        device = getattr(getattr(forward_batch, "input_ids", None), "device", None)
        if device is None and meta.cu_seqlens_q_prev_tensor is not None:
            device = meta.cu_seqlens_q_prev_tensor.device
        if device is None:
            device = torch.device("cpu")

        offsets = [0] + list(accumulate(meta.split_list))
        pieces = []
        for chunk_idx in meta.zigzag_index:
            start = offsets[chunk_idx]
            end = offsets[chunk_idx + 1]
            if end > start:
                pieces.append(torch.arange(start, end, device=device, dtype=torch.long))
        if not pieces:
            return torch.empty(0, device=device, dtype=torch.long)
        return torch.cat(pieces, dim=0)

    def get_supported_attention_backend(self):
        return [
            CPAttentionBackendKind.FLASH_ATTENTION,
            CPAttentionBackendKind.TRTLLM_MHA,
            CPAttentionBackendKind.PALLAS_RPA,
        ]

    def run_attention(
        self,
        q: Any,
        forward_batch,
        device: Any,
        attn_fn,
        attention_backend: CPAttentionBackendKind = CPAttentionBackendKind.FLASH_ATTENTION,
    ) -> Any:
        assert attention_backend in self.get_supported_attention_backend(), (
            f"{self.name} CP does not support {attention_backend=}"
        )

        meta = forward_batch.attn_cp_metadata
        q_prev = q[: meta.total_q_prev_tokens]
        logical_tokens = meta.total_q_prev_tokens + meta.total_q_next_tokens
        q_next = q[meta.total_q_prev_tokens : logical_tokens]

        prev_kwargs = {}
        next_kwargs = {}
        if attention_backend == CPAttentionBackendKind.TRTLLM_MHA:
            result = attn_fn(
                q[:logical_tokens],
                meta.cu_seqlens_q_combined_tensor,
                meta.kv_len_combined_tensor,
                meta.max_seqlen_q_combined,
                cu_seqlens_kv=meta.cu_seqlens_kv_combined_tensor,
                use_zigzag_page_table=True,
            )
        else:
            result_prev = attn_fn(
                q_prev,
                meta.cu_seqlens_q_prev_tensor,
                meta.kv_len_prev_tensor,
                meta.max_seqlen_q_prev,
                **prev_kwargs,
            )
            result_next = attn_fn(
                q_next,
                meta.cu_seqlens_q_next_tensor,
                meta.kv_len_next_tensor,
                meta.max_seqlen_q_next,
                **next_kwargs,
            )
            result = torch.cat([result_prev, result_next], dim=0)

        pad_size = q.shape[0] - logical_tokens
        assert pad_size >= 0
        if pad_size > 0:
            result = torch.cat(
                [result, result.new_zeros(pad_size, *result.shape[1:])], dim=0
            )
        return result

    def materialize_full_kv(
        self,
        forward_batch,
        layer: Any = None,
        k: Any = None,
        v: Any = None,
        swa_loc: Optional[Any] = None,
    ) -> Any:
        cache_loc = (
            forward_batch.out_cache_loc
            if not layer.is_cross_attention
            else forward_batch.encoder_out_cache_loc
        )
        if swa_loc is not None:
            swa_loc = swa_loc[: cache_loc.shape[0]]
        k_dim = k.shape[-1]
        v_dim = v.shape[-1]
        kv_cache = torch.cat([k, v], dim=-1).contiguous()
        key_cache_full, value_cache_full = self.gather_kv_cache(
            kv_cache, forward_batch
        ).split([k_dim, v_dim], dim=-1)
        key_cache_full = key_cache_full.contiguous()
        value_cache_full = value_cache_full.contiguous()
        get_token_to_kv_pool().set_kv_buffer(
            layer,
            KVWriteLoc.for_layer(forward_batch, layer, swa_loc=swa_loc),
            key_cache_full,
            value_cache_full,
            layer.k_scale,
            layer.v_scale,
        )

    def materialize_full_mla_kv(
        self, forward_batch, layer: Any, k_nope: Any, k_rope: Any
    ) -> None:
        kv_lora_rank = k_nope.shape[-1]
        latent = torch.cat([k_nope, k_rope], dim=-1).contiguous()
        latent_full = self.gather_kv_cache(latent, forward_batch)
        get_token_to_kv_pool().set_mla_kv_buffer(
            layer,
            KVWriteLoc.for_batch(forward_batch),
            latent_full[..., :kv_lora_rank],
            latent_full[..., kv_lora_rank:],
        )

    def _all_gather_reorganized(self, x: torch.Tensor, forward_batch):
        meta = forward_batch.attn_cp_metadata
        per_rank_token = meta.per_rank_logical_token or meta.per_rank_actual_token
        max_len = max(per_rank_token)
        local_len = per_rank_token[self.cp_rank]
        assert x.shape[0] >= local_len
        if x.device.type == "tpu":
            return self._all_gather_reorganized_tpu(
                x, per_rank_token, max_len, local_len
            )
        x = x[:local_len]
        pad_size = max_len - x.shape[0]
        if pad_size > 0:
            padding = [0, 0] * (x.ndim - 1) + [0, pad_size]
            x = F.pad(x, padding, mode="constant", value=0)

        group = get_parallel().attn_cp_group
        ctx = (
            use_symmetric_memory(group, disabled=not is_allocation_symmetric())
            if x.is_cuda
            else nullcontext()
        )
        with ctx:
            gathered = torch.empty(
                max_len * self.cp_size,
                *x.shape[1:],
                device=x.device,
                dtype=x.dtype,
            )
        group.all_gather_into_tensor(gathered, x)

        chunks = torch.split(gathered, [max_len] * self.cp_size, dim=0)
        return torch.cat(
            [
                chunks[rank][:per_rank_len]
                for rank, per_rank_len in enumerate(per_rank_token)
            ],
            dim=0,
        )

    def _all_gather_reorganized_tpu(self, x, per_rank_token, max_len, local_len):
        """TPU: gather through the exported XLA collective (see
        `batched_rpa_cp.build_cp_all_gather_op`). The eager torch collective
        returned wrong rows for the remote ranks whenever the per-rank shard
        lengths differ (uneven zigzag chunks)."""
        from sglang_tpu.batched_rpa_cp import cp_all_gather_padded

        minor = tuple(x.shape[1:])
        x2 = x.reshape(x.shape[0], -1)
        chunks = cp_all_gather_padded(x2, local_len, max_len, self.cp_size)
        return torch.cat(
            [chunks[r][:n] for r, n in enumerate(per_rank_token)], dim=0
        ).reshape(-1, *minor)
