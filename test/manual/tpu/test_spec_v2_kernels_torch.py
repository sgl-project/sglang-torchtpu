"""CPU differential coverage for TPU SpecV2 Torch dispatchers."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

import sglang.srt.speculative.spec_utils as kernels
import sglang.srt.speculative.triton_ops.gather_spec_extras as gather_kernels


class _ForceTpu:
    def __enter__(self):
        self.previous = kernels._is_tpu
        kernels._is_tpu = True

    def __exit__(self, *_):
        kernels._is_tpu = self.previous


def test_gather_spec_extras_matches_row_indexing():
    indices = torch.tensor([3, 0, 2], dtype=torch.int32)[::2]
    buffers = (
        torch.arange(20, dtype=torch.float32).reshape(4, 5),
        torch.arange(8, dtype=torch.int32).reshape(4, 2),
        torch.arange(4, dtype=torch.int64),
        torch.arange(24, dtype=torch.bfloat16).reshape(4, 2, 3),
    )
    previous = gather_kernels._is_tpu
    gather_kernels._is_tpu = True
    try:
        actual = gather_kernels.gather_spec_extras(indices, *buffers)
        empty = gather_kernels.gather_spec_extras(indices[:0], *buffers)
    finally:
        gather_kernels._is_tpu = previous

    expected_indices = indices.to(torch.int64)
    for result, source in zip(actual, buffers):
        assert torch.equal(result, source.index_select(0, expected_indices))
    assert [result.shape[0] for result in empty] == [0, 0, 0, 0]


def test_draft_cache_gather_randomized_and_empty():
    torch.manual_seed(7)
    table = torch.randint(0, 10000, (16, 128), dtype=torch.int32)
    for bs in (0, 1, 4, 8):
        rows = torch.arange(bs, dtype=torch.int32)
        starts = torch.arange(bs, dtype=torch.int32) + 3
        out = torch.empty(bs * 3, dtype=torch.int64)
        with _ForceTpu():
            kernels.assign_draft_cache_locs_contiguous_func(
                rows, table, starts, out, 1, 3
            )
        expected = (
            torch.cat(
                [
                    table[int(rows[i]), int(starts[i]) : int(starts[i]) + 3]
                    for i in range(bs)
                ]
            )
            if bs
            else out
        )
        assert torch.equal(out, expected.to(torch.int64))


def test_draft_cache_uses_host_shadows():
    table = torch.arange(4 * 32).reshape(4, 32)
    out = torch.zeros(4, dtype=torch.int32)
    with _ForceTpu():
        kernels.assign_draft_cache_locs_contiguous_func(
            torch.zeros(2, dtype=torch.int64),
            table,
            torch.zeros(2, dtype=torch.int64),
            out,
            1,
            2,
            host_req_pool_indices=lambda: torch.tensor([2, 3]),
            host_seq_lens=torch.tensor([5, 7]),
        )
    assert torch.equal(out, torch.tensor([69, 70, 103, 104], dtype=torch.int32))


def test_bonus_tokens_matches_flat_kernel_semantics():
    tokens = torch.tensor([[10, 11, 12, 13], [20, 21, 22, 23]])
    lengths = torch.tensor([2, 4], dtype=torch.int32)
    out = torch.empty(2, dtype=torch.int32)
    with _ForceTpu():
        kernels.fill_bonus_tokens_func(tokens, lengths, out, 4)
    assert torch.equal(out, torch.tensor([11, 23], dtype=torch.int32))


def test_accepted_cache_compaction_preserves_order_and_dtype():
    accept = torch.tensor([[2, -1, 0], [-1, 3, -1]], dtype=torch.int32)
    slots = torch.tensor([100, 101, 102, 103], dtype=torch.int32)
    out = torch.zeros(accept.numel(), dtype=torch.int64)
    with _ForceTpu():
        kernels.fill_accepted_out_cache_loc_func(accept, slots, out)
    assert torch.equal(out, torch.tensor([102, 100, 103, 0, 0, 0]))


def test_extend_cache_gather_variable_lengths():
    table = torch.arange(8 * 32, dtype=torch.int32).reshape(8, 32)
    rows = torch.tensor([3, 1, 6], dtype=torch.int32)
    starts = torch.tensor([2, 7, 4], dtype=torch.int32)
    ends = torch.tensor([4, 8, 7], dtype=torch.int64)
    out = torch.zeros(6, dtype=torch.int64)
    with _ForceTpu():
        kernels.assign_extend_cache_locs_kernel_func(rows, table, starts, ends, out)
    assert torch.equal(out, torch.tensor([98, 99, 39, 196, 197, 198]))


def test_extend_cache_router_uses_host_shadows():
    table = torch.arange(4 * 32).reshape(4, 32)
    with _ForceTpu():
        out = kernels.assign_extend_cache_locs_func(
            torch.zeros(2, dtype=torch.int64),
            table,
            torch.zeros(2, dtype=torch.int64),
            torch.zeros(2, dtype=torch.int64),
            2,
            2,
            torch.device("cpu"),
            host_req_pool_indices=torch.tensor([2, 3]),
            host_seq_lens=lambda: torch.tensor([5, 7]),
        )
    assert torch.equal(out, torch.tensor([69, 70, 103, 104]))


def test_allocator_reuses_host_allocation_shadow():
    table = torch.zeros((4, 32), dtype=torch.int32)
    host_out = torch.tensor([101, 102, 103, 104], dtype=torch.int32)
    device_out = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
    allocator = SimpleNamespace(page_size=4, last_alloc_extend_cpu=host_out)
    tree_cache = SimpleNamespace(token_to_kv_pool_allocator=allocator)
    pool = SimpleNamespace(req_to_token=table, host_resident=True)
    host_rows = Mock(return_value=torch.tensor([2, 3]))
    cur = torch.tensor([5, 7], dtype=torch.int32)
    nxt = torch.tensor([7, 9], dtype=torch.int32)
    with (
        patch.object(kernels, "get_last_loc", return_value=torch.tensor([1, 2])),
        patch.object(
            kernels, "alloc_paged_token_slots_extend", return_value=device_out
        ),
        patch.object(kernels, "assign_req_to_token_pool_func") as assign,
    ):
        result = kernels.allocate_and_assign_decode_cache(
            tree_cache=tree_cache,
            req_to_token_pool=pool,
            req_pool_indices=torch.zeros(2, dtype=torch.int64),
            cur_kv_lens_cpu=cur,
            nxt_kv_lens_cpu=nxt,
            num_needed_tokens=4,
            batch_size=2,
            device=torch.device("cpu"),
            host_req_pool_indices=host_rows,
        )
    assert result is device_out
    assert assign.call_args.args[4] is host_out
    host_rows.assert_called_once_with()
