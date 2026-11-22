# Copyright 2026 Google LLC
#
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
#
# Adapted for SGLang-TorchTPU from vllm-project/vllm-torchtpu (Apache-2.0).
# Source: src/vllm_torchtpu/kernels/deepseek_v4/streamindex_topk/bref_override.py
"""Custom BufferedRef overrides for StreamIndex Top-K kernel."""

import dataclasses
import functools
from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _step_offset(meta_ref: Any, chunk_idx: int):
    """Flat schedule row where chunk `chunk_idx`'s pipeline steps begin."""
    return sum(meta_ref.num_steps[m] for m in range(chunk_idx))


def _token_window(
    refs: tuple[Any, ...],
    p_id: int | jax.Array,
    chunk_tokens: int,
    *,
    bq_sz: int,
    seq_batch_size: int,
    chunk_idx: int = 0,
):
    """Flat-token rows one pipeline step covers, as (token_start, size)."""
    _, cu_q_lens_ref, meta_ref, start_end_seq_idx_ref = refs
    chunk_start = start_end_seq_idx_ref[2]
    p_id = p_id + _step_offset(meta_ref, chunk_idx)
    seq_idx = meta_ref.batch_tile_idx[p_id]
    lo = jnp.maximum(cu_q_lens_ref[seq_idx], chunk_start)

    if seq_batch_size > 1:
        return lo, seq_batch_size * bq_sz
    token_start = lo + meta_ref.bq_idx[p_id] * bq_sz
    if bq_sz == 1:
        return token_start, 1
    hi = jnp.minimum(cu_q_lens_ref[seq_idx + 1], chunk_start + chunk_tokens)
    return token_start, jnp.maximum(0, jnp.minimum(bq_sz, hi - token_start))


def _fetch_page_row(page_table, page_rows, *, seq_idx, batch_idx, bkv_idx):
    """With the page table in HBM, copy a sequence's row into SMEM with its first
    KV block. A prefetched table (``page_rows`` None) needs no copy."""
    if page_rows is None:
        return
    # [max_num_seqs, pages_per_seq]: a dynamic leading index needs no alignment proof.
    row = jnp.minimum(seq_idx, page_table.shape[0] - 1)

    @pl.when(bkv_idx == 0)
    def _():
        pltpu.sync_copy(page_table.at[row], page_rows.at[batch_idx])


def _page_id(page_table, page_rows, *, seq_idx, batch_idx, page_col, pages_per_seq):
    """Page id of column ``page_col`` of sequence ``seq_idx``, from the prefetched
    flat table or from the sequence's row fetched into SMEM."""
    if page_rows is None:
        flat = jnp.minimum(seq_idx * pages_per_seq + page_col, page_table.shape[0] - 1)
        return page_table[flat]
    return page_rows[batch_idx, jnp.minimum(page_col, pages_per_seq - 1)]


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class StreamIndexKVBufferedRef(pltpu.BufferedRef):
    """Handles fetching paged KV cache into VMEM buffers across sequence sublanes."""

    bkv_p: int = dataclasses.field(default=1, metadata=dict(static=True))
    page_size_per_kv_packing: int = dataclasses.field(
        default=64, metadata=dict(static=True)
    )
    kv_packing: int = dataclasses.field(default=4, metadata=dict(static=True))
    pages_per_seq: int = dataclasses.field(default=1, metadata=dict(static=True))
    seq_batch_size: int = dataclasses.field(default=1, metadata=dict(static=True))
    chunk_idx: int = dataclasses.field(default=0, metadata=dict(static=True))
    # > 0: page-table rows end with page_group_starts (one entry per group).
    page_group: int = dataclasses.field(default=0, metadata=dict(static=True))
    # > 0: then one entry per sub-group, tried when a group is not one run.
    sub_group: int = dataclasses.field(default=0, metadata=dict(static=True))

    @classmethod
    def create(
        cls,
        spec: pl.BlockSpec,
        dtype_or_type: Any,
        buffer_type: pltpu.BufferType,
        buffer_count: int,
        use_lookahead: bool,
        bkv_p: int,
        page_size_per_kv_packing: int,
        kv_packing: int,
        pages_per_seq: int,
        seq_batch_size: int,
        grid_rank: int = 1,
        chunk_idx: int = 0,
        page_group: int = 0,
        sub_group: int = 0,
    ):
        assert buffer_type == pltpu.BufferType.INPUT
        standard_ref = pltpu.BufferedRef.create(
            spec=spec,
            dtype_or_type=dtype_or_type,
            buffer_type=buffer_type,
            buffer_count=buffer_count,
            grid_rank=grid_rank,
            use_lookahead=use_lookahead,
        )
        return cls(
            bkv_p=bkv_p,
            page_size_per_kv_packing=page_size_per_kv_packing,
            kv_packing=kv_packing,
            pages_per_seq=pages_per_seq,
            seq_batch_size=seq_batch_size,
            chunk_idx=chunk_idx,
            page_group=page_group,
            sub_group=sub_group,
            **{
                f.name: getattr(standard_ref, f.name)
                for f in dataclasses.fields(pltpu.BufferedRef)
            },
        )

    @classmethod
    def input(
        cls,
        spec: pl.BlockSpec,
        dtype_or_type: Any,
        buffer_count: int,
        use_lookahead: bool,
        bkv_p: int,
        page_size_per_kv_packing: int,
        kv_packing: int,
        pages_per_seq: int,
        seq_batch_size: int,
        grid_rank: int = 1,
        chunk_idx: int = 0,
        page_group: int = 0,
        sub_group: int = 0,
        **kwargs: Any,
    ):
        return cls.create(
            spec=spec,
            dtype_or_type=dtype_or_type,
            buffer_type=pltpu.BufferType.INPUT,
            buffer_count=buffer_count,
            use_lookahead=use_lookahead,
            bkv_p=bkv_p,
            page_size_per_kv_packing=page_size_per_kv_packing,
            kv_packing=kv_packing,
            pages_per_seq=pages_per_seq,
            seq_batch_size=seq_batch_size,
            grid_rank=grid_rank,
            chunk_idx=chunk_idx,
            page_group=page_group,
            sub_group=sub_group,
        )

    def copy_in(
        self,
        src_ref: tuple[jax.Ref, jax.Ref, Any],
        grid_indices: tuple[int | jax.Array, ...],
    ):
        cache_kv_hbm, page_table, meta_ref, page_rows = src_ref
        p_id = grid_indices[0] + _step_offset(meta_ref, self.chunk_idx)
        batch_tile_idx = meta_ref.batch_tile_idx[p_id]
        bkv_idx = meta_ref.bkv_idx[p_id]

        assert self.sem_recvs is not None
        assert self.window_ref is not None
        assert isinstance(cache_kv_hbm, jax.Ref)
        slot = self.current_copy_in_slot
        sem = self.sem_recvs.at[slot]
        vmem_dst: Any = self.window_ref.at[slot]

        reshaped_cache_hbm_ref: Any = cache_kv_hbm.reshape(
            cache_kv_hbm.shape[0] * self.page_size_per_kv_packing,
            self.kv_packing,
            cache_kv_hbm.shape[-1],
        )
        max_hbm_pages = reshaped_cache_hbm_ref.shape[0]
        effective_seq_idx = batch_tile_idx

        for batch_idx in range(self.seq_batch_size):
            kv_p_start = bkv_idx * self.bkv_p
            _fetch_page_row(
                page_table,
                page_rows,
                seq_idx=effective_seq_idx + batch_idx,
                batch_idx=batch_idx,
                bkv_idx=bkv_idx,
            )
            group = self.page_group if page_rows is not None else 0
            for g in range(0, self.bkv_p, max(group, 1)):
                pages = range(g, g + group) if group else (g,)
                if group:
                    start = page_rows[
                        batch_idx, self.pages_per_seq + (kv_p_start + g) // group
                    ]
                    pl.when(start >= 0)(
                        functools.partial(
                            self._copy_pages, reshaped_cache_hbm_ref, vmem_dst, sem,
                            start * self.page_size_per_kv_packing, batch_idx, g, group,
                        )
                    )  # fmt: skip

                def per_page(pages=pages):
                    for i in pages:
                        page = _page_id(
                            page_table,
                            page_rows,
                            seq_idx=effective_seq_idx + batch_idx,
                            batch_idx=batch_idx,
                            page_col=kv_p_start + i,
                            pages_per_seq=self.pages_per_seq,
                        )
                        safe_page_offset = jnp.minimum(
                            page * self.page_size_per_kv_packing,
                            jnp.maximum(
                                0, max_hbm_pages - self.page_size_per_kv_packing
                            ),
                        )
                        self._copy_pages(
                            reshaped_cache_hbm_ref, vmem_dst, sem, safe_page_offset,
                            batch_idx, i, 1,
                        )  # fmt: skip

                if group and self.sub_group:
                    pl.when(start < 0)(
                        functools.partial(
                            self._copy_sub_groups, page_table, page_rows,
                            reshaped_cache_hbm_ref, vmem_dst, sem,
                            effective_seq_idx + batch_idx, batch_idx, kv_p_start, g,
                            max_hbm_pages,
                        )
                    )  # fmt: skip
                elif group:
                    pl.when(start < 0)(per_page)
                else:
                    per_page()

    def _copy_sub_groups(
        self, page_table, page_rows, src, vmem_dst, sem, seq_idx, batch_idx,
        kv_p_start, first, max_hbm_pages,
    ):  # fmt: skip
        """A group that is not one run: one DMA per sub-group run, else page by page."""
        sub = self.sub_group
        sub_base = self.pages_per_seq + self.pages_per_seq // self.page_group
        rows = self.page_size_per_kv_packing
        for j in range(first, first + self.page_group, sub):
            sub_start = page_rows[batch_idx, sub_base + (kv_p_start + j) // sub]
            pl.when(sub_start >= 0)(
                functools.partial(
                    self._copy_pages, src, vmem_dst, sem, sub_start * rows,
                    batch_idx, j, sub,
                )
            )  # fmt: skip

            @pl.when(sub_start < 0)
            def _(j=j):
                @pl.loop(0, sub)
                def _(i):
                    page = _page_id(
                        page_table,
                        page_rows,
                        seq_idx=seq_idx,
                        batch_idx=batch_idx,
                        page_col=kv_p_start + j + i,
                        pages_per_seq=self.pages_per_seq,
                    )
                    offset = jnp.minimum(
                        page * rows, jnp.maximum(0, max_hbm_pages - rows)
                    )
                    self._copy_pages(src, vmem_dst, sem, offset, batch_idx, j + i, 1)

    def _copy_pages(self, src, vmem_dst, sem, src_row, batch_idx, first, count):
        """Start one DMA of ``count`` pages from row ``src_row`` into slot pages
        ``first``.. of sequence ``batch_idx``."""
        rows = self.page_size_per_kv_packing
        pltpu.make_async_copy(
            src.at[pl.ds(src_row, count * rows)],
            vmem_dst.at[batch_idx, pl.ds(first * rows, count * rows)],
            sem,
        ).start()

    def wait_in(
        self,
        src_ref: tuple[jax.Ref, jax.Ref, Any],
        grid_indices: tuple[int | jax.Array, ...],
    ):
        assert self.is_input
        if not self.is_buffered:
            return
        assert self.sem_recvs is not None
        assert self.window_ref is not None
        slot = self.current_wait_in_slot
        sem = self.sem_recvs.at[slot]
        vmem_dst: Any = self.window_ref.at[slot]
        pltpu.make_async_copy(
            vmem_dst,
            vmem_dst,
            sem,
        ).wait()


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class StreamIndexKVSeqAlongLaneBufferedRef(pltpu.BufferedRef):
    """Fetches paged KV cache into VMEM for the SEQ_ALONG_LANE layout."""

    bkv_p: int = dataclasses.field(default=1, metadata=dict(static=True))
    page_size: int = dataclasses.field(default=128, metadata=dict(static=True))
    pages_per_seq: int = dataclasses.field(default=1, metadata=dict(static=True))
    seq_batch_size: int = dataclasses.field(default=1, metadata=dict(static=True))
    chunk_idx: int = dataclasses.field(default=0, metadata=dict(static=True))

    @classmethod
    def create(
        cls,
        spec: pl.BlockSpec,
        dtype_or_type: Any,
        buffer_type: pltpu.BufferType,
        buffer_count: int,
        use_lookahead: bool,
        bkv_p: int,
        page_size: int,
        pages_per_seq: int,
        seq_batch_size: int,
        grid_rank: int = 1,
        chunk_idx: int = 0,
    ):
        assert buffer_type == pltpu.BufferType.INPUT
        standard_ref = pltpu.BufferedRef.create(
            spec=spec,
            dtype_or_type=dtype_or_type,
            buffer_type=buffer_type,
            buffer_count=buffer_count,
            grid_rank=grid_rank,
            use_lookahead=use_lookahead,
        )
        return cls(
            bkv_p=bkv_p,
            page_size=page_size,
            pages_per_seq=pages_per_seq,
            seq_batch_size=seq_batch_size,
            chunk_idx=chunk_idx,
            **{
                f.name: getattr(standard_ref, f.name)
                for f in dataclasses.fields(pltpu.BufferedRef)
            },
        )

    @classmethod
    def input(
        cls,
        spec: pl.BlockSpec,
        dtype_or_type: Any,
        buffer_count: int,
        use_lookahead: bool,
        bkv_p: int,
        page_size: int,
        pages_per_seq: int,
        seq_batch_size: int,
        grid_rank: int = 1,
        chunk_idx: int = 0,
        **kwargs: Any,
    ):
        return cls.create(
            spec=spec,
            dtype_or_type=dtype_or_type,
            buffer_type=pltpu.BufferType.INPUT,
            buffer_count=buffer_count,
            use_lookahead=use_lookahead,
            bkv_p=bkv_p,
            page_size=page_size,
            pages_per_seq=pages_per_seq,
            seq_batch_size=seq_batch_size,
            grid_rank=grid_rank,
            chunk_idx=chunk_idx,
        )

    def copy_in(
        self,
        src_ref: tuple[jax.Ref, jax.Ref, Any],
        grid_indices: tuple[int | jax.Array, ...],
    ):
        cache_kv_hbm, page_table, meta_ref, page_rows = src_ref
        p_id = grid_indices[0] + _step_offset(meta_ref, self.chunk_idx)
        batch_tile_idx = meta_ref.batch_tile_idx[p_id]
        bkv_idx = meta_ref.bkv_idx[p_id]

        assert self.sem_recvs is not None
        assert self.window_ref is not None
        assert isinstance(cache_kv_hbm, jax.Ref)
        slot = self.current_copy_in_slot
        sem = self.sem_recvs.at[slot]
        vmem_dst: Any = self.window_ref.at[slot]

        max_hbm_pages = cache_kv_hbm.shape[0]
        effective_seq_idx = batch_tile_idx

        for batch_idx in range(self.seq_batch_size):
            kv_p_start = bkv_idx * self.bkv_p
            _fetch_page_row(
                page_table,
                page_rows,
                seq_idx=effective_seq_idx + batch_idx,
                batch_idx=batch_idx,
                bkv_idx=bkv_idx,
            )
            for i in range(self.bkv_p):
                page = _page_id(
                    page_table,
                    page_rows,
                    seq_idx=effective_seq_idx + batch_idx,
                    batch_idx=batch_idx,
                    page_col=kv_p_start + i,
                    pages_per_seq=self.pages_per_seq,
                )
                safe_page = jnp.minimum(page, max_hbm_pages - 1)
                dst_lane_start = i * self.page_size
                pltpu.make_async_copy(
                    cache_kv_hbm.at[safe_page],
                    vmem_dst.at[
                        batch_idx,
                        :,
                        :,
                        pl.ds(dst_lane_start, self.page_size),
                    ],
                    sem,
                ).start()

    def wait_in(
        self,
        src_ref: tuple[jax.Ref, jax.Ref, Any],
        grid_indices: tuple[int | jax.Array, ...],
    ):
        assert self.is_input
        if not self.is_buffered:
            return
        assert self.sem_recvs is not None
        assert self.window_ref is not None
        slot = self.current_wait_in_slot
        sem = self.sem_recvs.at[slot]
        vmem_dst: Any = self.window_ref.at[slot]
        pltpu.make_async_copy(
            vmem_dst,
            vmem_dst,
            sem,
        ).wait()


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class StreamIndexQBufferedRef(pltpu.BufferedRef):
    """Handles fetching query slices into VMEM buffers."""

    bq_sz: int = dataclasses.field(default=16, metadata=dict(static=True))
    seq_batch_size: int = dataclasses.field(default=1, metadata=dict(static=True))
    chunk_tokens: int = dataclasses.field(default=0, metadata=dict(static=True))
    chunk_idx: int = dataclasses.field(default=0, metadata=dict(static=True))

    @classmethod
    def create(
        cls,
        spec: pl.BlockSpec,
        dtype_or_type: Any,
        buffer_type: pltpu.BufferType,
        buffer_count: int,
        use_lookahead: bool,
        bq_sz: int,
        chunk_tokens: int,
        seq_batch_size: int = 1,
        grid_rank: int = 1,
        chunk_idx: int = 0,
    ):
        assert buffer_type == pltpu.BufferType.INPUT
        standard_ref = pltpu.BufferedRef.create(
            spec=spec,
            dtype_or_type=dtype_or_type,
            buffer_type=buffer_type,
            buffer_count=buffer_count,
            grid_rank=grid_rank,
            use_lookahead=use_lookahead,
        )
        return cls(
            bq_sz=bq_sz,
            seq_batch_size=seq_batch_size,
            chunk_tokens=chunk_tokens,
            chunk_idx=chunk_idx,
            **{
                f.name: getattr(standard_ref, f.name)
                for f in dataclasses.fields(pltpu.BufferedRef)
            },
        )

    @classmethod
    def input(
        cls,
        spec: pl.BlockSpec,
        dtype_or_type: Any,
        buffer_count: int,
        use_lookahead: bool,
        bq_sz: int,
        chunk_tokens: int,
        seq_batch_size: int = 1,
        grid_rank: int = 1,
        chunk_idx: int = 0,
        **kwargs: Any,
    ):
        return cls.create(
            spec=spec,
            dtype_or_type=dtype_or_type,
            buffer_type=pltpu.BufferType.INPUT,
            buffer_count=buffer_count,
            use_lookahead=use_lookahead,
            bq_sz=bq_sz,
            seq_batch_size=seq_batch_size,
            chunk_tokens=chunk_tokens,
            grid_rank=grid_rank,
            chunk_idx=chunk_idx,
        )

    def copy_in(
        self,
        src_ref: tuple[Any, ...],
        grid_indices: tuple[int | jax.Array, ...],
    ):
        q_hbm = src_ref[0]
        assert self.sem_recvs is not None
        assert self.window_ref is not None
        slot = self.current_copy_in_slot
        sem = self.sem_recvs.at[slot]
        vmem_dst: Any = self.window_ref.at[slot]

        q_len_start, total_sz = _token_window(
            src_ref,
            grid_indices[0],
            self.chunk_tokens,
            bq_sz=self.bq_sz,
            seq_batch_size=self.seq_batch_size,
            chunk_idx=self.chunk_idx,
        )

        pltpu.make_async_copy(
            q_hbm.at[pl.ds(q_len_start, total_sz)],
            vmem_dst.at[pl.ds(0, total_sz)],
            sem,
        ).start()

    def wait_in(
        self,
        src_ref: tuple[Any, ...],
        grid_indices: tuple[int | jax.Array, ...],
    ):
        assert self.is_input
        if not self.is_buffered:
            return
        assert self.sem_recvs is not None
        assert self.window_ref is not None
        slot = self.current_wait_in_slot
        sem = self.sem_recvs.at[slot]
        vmem_dst: Any = self.window_ref.at[slot]
        _, total_sz = _token_window(
            src_ref,
            grid_indices[0],
            self.chunk_tokens,
            bq_sz=self.bq_sz,
            seq_batch_size=self.seq_batch_size,
            chunk_idx=self.chunk_idx,
        )

        pltpu.make_async_copy(
            vmem_dst.at[pl.ds(0, total_sz)],
            vmem_dst.at[pl.ds(0, total_sz)],
            sem,
        ).wait()


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class StreamIndexOBufferedRef(pltpu.BufferedRef):
    """Handles scattering computed Top-K scores back to HBM."""

    bq_sz: int = dataclasses.field(default=16, metadata=dict(static=True))
    num_sublanes: int = dataclasses.field(default=1, metadata=dict(static=True))
    seq_batch_size: int = dataclasses.field(default=1, metadata=dict(static=True))
    chunk_idx: int = dataclasses.field(default=0, metadata=dict(static=True))

    @classmethod
    def create(
        cls,
        spec: pl.BlockSpec,
        dtype_or_type: Any,
        buffer_type: pltpu.BufferType,
        buffer_count: int,
        use_lookahead: bool,
        bq_sz: int,
        num_sublanes: int,
        seq_batch_size: int = 1,
        grid_rank: int = 1,
        chunk_idx: int = 0,
    ):
        assert buffer_type == pltpu.BufferType.OUTPUT
        standard_ref = pltpu.BufferedRef.create(
            spec=spec,
            dtype_or_type=dtype_or_type,
            buffer_type=buffer_type,
            buffer_count=buffer_count,
            grid_rank=grid_rank,
            use_lookahead=use_lookahead,
        )
        return cls(
            bq_sz=bq_sz,
            num_sublanes=num_sublanes,
            seq_batch_size=seq_batch_size,
            chunk_idx=chunk_idx,
            **{
                f.name: getattr(standard_ref, f.name)
                for f in dataclasses.fields(pltpu.BufferedRef)
            },
        )

    @classmethod
    def output(
        cls,
        spec: pl.BlockSpec,
        dtype_or_type: Any,
        buffer_count: int,
        use_lookahead: bool,
        bq_sz: int,
        num_sublanes: int,
        seq_batch_size: int = 1,
        grid_rank: int = 1,
        chunk_idx: int = 0,
        **kwargs: Any,
    ):
        return cls.create(
            spec=spec,
            dtype_or_type=dtype_or_type,
            buffer_type=pltpu.BufferType.OUTPUT,
            buffer_count=buffer_count,
            use_lookahead=use_lookahead,
            bq_sz=bq_sz,
            num_sublanes=num_sublanes,
            seq_batch_size=seq_batch_size,
            grid_rank=grid_rank,
            chunk_idx=chunk_idx,
        )

    def copy_out(
        self,
        dst_ref: tuple[Any, ...],
        grid_indices: tuple[int | jax.Array, ...],
    ):
        scores_hbm, _, meta_ref, start_end_seq_idx_ref = dst_ref
        bkv_idx = meta_ref.bkv_idx[
            grid_indices[0] + _step_offset(meta_ref, self.chunk_idx)
        ]

        assert self.sem_sends is not None
        assert self.window_ref is not None
        slot = self.current_copy_out_slot
        sem = self.sem_sends.at[slot]
        vmem_src: Any = self.window_ref.at[slot]

        token_start, total_sz = _token_window(
            dst_ref,
            grid_indices[0],
            scores_hbm.shape[0],
            bq_sz=self.bq_sz,
            seq_batch_size=self.seq_batch_size,
            chunk_idx=self.chunk_idx,
        )
        dst_start = token_start - start_end_seq_idx_ref[2]
        sublane_start = bkv_idx * self.num_sublanes

        pltpu.make_async_copy(
            self._unpadded_slice(vmem_src, total_sz),
            scores_hbm.at[
                pl.ds(dst_start, total_sz),
                pl.ds(sublane_start, self.num_sublanes),
            ],
            sem,
        ).start()

    def wait_out(
        self,
        dst_ref: tuple[Any, ...],
        grid_indices: tuple[int | jax.Array, ...],
    ):
        assert self.is_output
        if not self.is_buffered:
            return
        assert self.sem_sends is not None
        assert self.window_ref is not None
        slot = self.current_wait_out_slot
        sem = self.sem_sends.at[slot]
        vmem_src: Any = self.window_ref.at[slot]

        _, total_sz = _token_window(
            dst_ref,
            grid_indices[0],
            dst_ref[0].shape[0],
            bq_sz=self.bq_sz,
            seq_batch_size=self.seq_batch_size,
            chunk_idx=self.chunk_idx,
        )

        unpadded_src = self._unpadded_slice(vmem_src, total_sz)
        pltpu.make_async_copy(unpadded_src, unpadded_src, sem).wait()

    def _unpadded_slice(self, vmem_src: Any, total_sz: int | jax.Array):
        """Slices out valid tokens and unpadded sublanes (`num_sublanes`).

        The buffer may give each token more rows (`score_pitch`) than it copies out,
        so that the kernel's sublane-strided stores into it stay free of bank
        conflicts.

        Args:
          vmem_src: one [tokens, rows per token, 128] buffer slot.
          total_sz: number of tokens copied out.
        """
        return vmem_src.at[pl.ds(0, total_sz), pl.ds(0, self.num_sublanes)]
