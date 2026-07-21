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


def test_draft_cache_gather_at_pool_boundary():
    table = torch.arange(2 * 8, dtype=torch.int32).reshape(2, 8)
    rows = torch.tensor([0, 1], dtype=torch.int32)
    starts = torch.tensor([5, 5], dtype=torch.int32)
    out = torch.empty(6, dtype=torch.int64)

    with _ForceTpu():
        kernels.assign_draft_cache_locs_contiguous_func(
            rows, table, starts, out, topk=1, speculative_num_steps=3
        )

    assert torch.equal(out, torch.tensor([5, 6, 7, 13, 14, 15]))


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


def test_assign_req_to_token_pool_randomized_and_empty():
    torch.manual_seed(11)
    for dtype in (torch.int32, torch.int64):
        for bs in (0, 1, 4, 8):
            table = torch.full((16, 32), -7, dtype=dtype)
            rows = torch.arange(bs, dtype=torch.int32) + 2
            starts = torch.arange(bs, dtype=torch.int64) * 2 + 1
            lengths = torch.arange(bs, dtype=torch.int32) % 4
            ends = starts + lengths
            slots = torch.randint(0, 10000, (int(lengths.sum()),), dtype=dtype)
            expected = table.clone()
            cursor = 0
            for row, start, length in zip(rows, starts, lengths):
                length = int(length)
                expected[int(row), int(start) : int(start) + length] = slots[
                    cursor : cursor + length
                ]
                cursor += length

            with _ForceTpu():
                kernels.assign_req_to_token_pool_func(
                    rows, table, starts, ends, slots, bs
                )

            assert torch.equal(table, expected)


def test_assign_req_to_token_pool_ignores_padded_rows():
    table = torch.full((4, 16), -1, dtype=torch.int32)
    rows = torch.tensor([2, 3, 1], dtype=torch.int64)
    starts = torch.tensor([3, 7, 11], dtype=torch.int32)
    ends = torch.tensor([5, 9, 13], dtype=torch.int32)
    slots = torch.tensor([101, 102, 201, 202], dtype=torch.int64)
    with _ForceTpu():
        kernels.assign_req_to_token_pool_func(rows, table, starts, ends, slots, 2)

    assert torch.equal(table[2, 3:5], torch.tensor([101, 102]))
    assert torch.equal(table[3, 7:9], torch.tensor([201, 202]))
    assert torch.all(table[1] == -1)
    assert table[0, 0] == -1


def test_bonus_tokens_matches_flat_kernel_semantics():
    tokens = torch.tensor([[10, 11, 12, 13], [20, 21, 22, 23]])
    lengths = torch.tensor([2, 4], dtype=torch.int32)
    out = torch.empty(2, dtype=torch.int32)
    with _ForceTpu():
        kernels.fill_bonus_tokens_func(tokens, lengths, out, 4)
    assert torch.equal(out, torch.tensor([11, 23], dtype=torch.int32))


def test_bonus_tokens_boundaries_and_empty():
    tokens = torch.tensor([[10, 11, 12, 13], [20, 21, 22, 23]])
    out = torch.empty(2, dtype=torch.int64)
    with _ForceTpu():
        kernels.fill_bonus_tokens_func(
            tokens, torch.tensor([1, 4], dtype=torch.int32), out, 4
        )
        kernels.fill_bonus_tokens_func(
            tokens[:0],
            torch.empty(0, dtype=torch.int32),
            torch.empty(0, dtype=torch.int64),
            4,
        )
    assert torch.equal(out, torch.tensor([10, 23]))


def test_accepted_cache_compaction_preserves_order_and_dtype():
    accept = torch.tensor([[2, -1, 0], [-1, 3, -1]], dtype=torch.int32)
    slots = torch.tensor([100, 101, 102, 103], dtype=torch.int32)
    out = torch.zeros(accept.numel(), dtype=torch.int64)
    with _ForceTpu():
        kernels.fill_accept_out_cache_loc_func(accept, slots, out)
    assert torch.equal(out, torch.tensor([102, 100, 103, 0, 0, 0]))


def test_accepted_cache_compaction_all_or_none_and_empty():
    slots = torch.tensor([100, 101, 102, 103], dtype=torch.int64)
    cases = (
        (torch.tensor([3, 1, 0, 2]), torch.tensor([103, 101, 100, 102])),
        (torch.full((4,), -1), torch.zeros(4, dtype=torch.int64)),
        (torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int64)),
    )
    with _ForceTpu():
        for accept, expected in cases:
            out = torch.full((accept.numel(),), -9, dtype=torch.int64)
            kernels.fill_accept_out_cache_loc_func(accept, slots, out)
            assert torch.equal(out, expected)


def test_extend_cache_gather_variable_lengths():
    table = torch.arange(8 * 32, dtype=torch.int32).reshape(8, 32)
    rows = torch.tensor([3, 1, 6], dtype=torch.int32)
    starts = torch.tensor([2, 7, 4], dtype=torch.int32)
    ends = torch.tensor([4, 8, 7], dtype=torch.int64)
    out = torch.zeros(6, dtype=torch.int64)
    with _ForceTpu():
        kernels.assign_extend_cache_locs_kernel_func(rows, table, starts, ends, out)
    assert torch.equal(out, torch.tensor([98, 99, 39, 196, 197, 198]))


def test_extend_cache_gather_zero_lengths_trailing_capacity_and_empty():
    table = torch.arange(4 * 16, dtype=torch.int32).reshape(4, 16)
    out = torch.full((4,), -9, dtype=torch.int64)
    with _ForceTpu():
        kernels.assign_extend_cache_locs_kernel_func(
            torch.tensor([1, 2, 3]),
            table,
            torch.tensor([2, 5, 7]),
            torch.tensor([2, 7, 7]),
            out,
        )
        empty_out = torch.empty(0, dtype=torch.int64)
        kernels.assign_extend_cache_locs_kernel_func(
            torch.empty(0, dtype=torch.int64),
            table,
            torch.empty(0, dtype=torch.int64),
            torch.empty(0, dtype=torch.int64),
            empty_out,
        )
    assert torch.equal(out, torch.tensor([37, 38, -9, -9]))
    assert empty_out.numel() == 0


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


def test_allocator_page_size_one_assigns_allocated_slots():
    table = torch.zeros((4, 32), dtype=torch.int32)
    device_out = torch.tensor([11, 12, 13, 14], dtype=torch.int64)
    allocator = SimpleNamespace(page_size=1)
    tree_cache = SimpleNamespace(token_to_kv_pool_allocator=allocator)
    pool = SimpleNamespace(req_to_token=table, host_resident=False)
    cur = torch.tensor([5, 7], dtype=torch.int32)
    nxt = torch.tensor([7, 9], dtype=torch.int32)
    with (
        patch.object(kernels, "alloc_token_slots", return_value=device_out) as alloc,
        patch.object(kernels, "alloc_paged_token_slots_extend") as paged_alloc,
        patch.object(kernels, "assign_req_to_token_pool_func") as assign,
    ):
        result = kernels.allocate_and_assign_decode_cache(
            tree_cache=tree_cache,
            req_to_token_pool=pool,
            req_pool_indices=torch.tensor([2, 3]),
            cur_kv_lens_cpu=cur,
            nxt_kv_lens_cpu=nxt,
            num_needed_tokens=4,
            batch_size=2,
            device=torch.device("cpu"),
        )

    assert result is device_out
    alloc.assert_called_once_with(tree_cache, 4)
    paged_alloc.assert_not_called()
    assert assign.call_args.args[4] is device_out


def test_allocator_page_size_one_handles_empty_batch():
    table = torch.zeros((4, 32), dtype=torch.int32)
    device_out = torch.empty(0, dtype=torch.int64)
    allocator = SimpleNamespace(page_size=1)
    tree_cache = SimpleNamespace(token_to_kv_pool_allocator=allocator)
    pool = SimpleNamespace(req_to_token=table, host_resident=False)
    empty = torch.empty(0, dtype=torch.int32)
    with (
        patch.object(kernels, "alloc_token_slots", return_value=device_out) as alloc,
        patch.object(kernels, "assign_req_to_token_pool_func") as assign,
    ):
        result = kernels.allocate_and_assign_decode_cache(
            tree_cache=tree_cache,
            req_to_token_pool=pool,
            req_pool_indices=empty,
            cur_kv_lens_cpu=empty,
            nxt_kv_lens_cpu=empty,
            num_needed_tokens=0,
            batch_size=0,
            device=torch.device("cpu"),
        )

    assert result is device_out
    alloc.assert_called_once_with(tree_cache, 0)
    assert assign.call_args.args[-1] == 0


def test_allocator_device_resident_uses_paged_device_result():
    table = torch.zeros((4, 32), dtype=torch.int32)
    rows = torch.tensor([2, 3], dtype=torch.int64)
    device_out = torch.tensor([11, 12, 13, 14], dtype=torch.int64)
    allocator = SimpleNamespace(
        page_size=4,
        last_alloc_extend_cpu=torch.tensor([101, 102, 103, 104]),
    )
    tree_cache = SimpleNamespace(token_to_kv_pool_allocator=allocator)
    pool = SimpleNamespace(req_to_token=table, host_resident=False)
    cur = torch.tensor([5, 7], dtype=torch.int32)
    nxt = torch.tensor([7, 9], dtype=torch.int32)
    with (
        patch.object(
            kernels, "get_last_loc", return_value=torch.tensor([1, 2])
        ) as last,
        patch.object(
            kernels, "alloc_paged_token_slots_extend", return_value=device_out
        ) as alloc,
        patch.object(kernels, "assign_req_to_token_pool_func") as assign,
    ):
        result = kernels.allocate_and_assign_decode_cache(
            tree_cache=tree_cache,
            req_to_token_pool=pool,
            req_pool_indices=rows,
            cur_kv_lens_cpu=cur,
            nxt_kv_lens_cpu=nxt,
            num_needed_tokens=4,
            batch_size=2,
            device=torch.device("cpu"),
        )

    assert result is device_out
    last.assert_called_once_with(table, rows, cur)
    assert alloc.call_args.args[1] is cur
    assert alloc.call_args.args[3] is nxt
    assert assign.call_args.args[4] is device_out


def test_allocator_host_resident_falls_back_to_device_copy():
    table = torch.zeros((4, 32), dtype=torch.int32)
    host_copy = torch.tensor([101, 102, 103, 104], dtype=torch.int32)
    device_out = Mock()
    device_out.cpu.return_value = host_copy
    allocator = SimpleNamespace(page_size=4, last_alloc_extend_cpu=None)
    tree_cache = SimpleNamespace(token_to_kv_pool_allocator=allocator)
    pool = SimpleNamespace(req_to_token=table, host_resident=True)
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
            req_pool_indices=torch.tensor([2, 3]),
            cur_kv_lens_cpu=cur,
            nxt_kv_lens_cpu=nxt,
            num_needed_tokens=4,
            batch_size=2,
            device=torch.device("cpu"),
        )

    assert result is device_out
    device_out.cpu.assert_called_once_with()
    assert assign.call_args.args[4] is host_copy
