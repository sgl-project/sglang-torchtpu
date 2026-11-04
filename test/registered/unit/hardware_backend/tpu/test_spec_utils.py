"""CPU differential coverage for TPU SpecV2 Torch dispatchers."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

import sglang.srt.speculative.eagle_utils as eagle_utils
import sglang.srt.speculative.spec_utils as kernels
from sglang_tpu.speculative import eagle_utils_tpu


class _ForceTpu:
    def __enter__(self):
        self.previous = kernels._is_tpu, kernels._use_torch_native
        kernels._is_tpu = True
        kernels._use_torch_native = True

    def __exit__(self, *_):
        kernels._is_tpu, kernels._use_torch_native = self.previous


def test_gather_spec_extras_matches_row_indexing():
    indices = torch.tensor([3, 0, 2], dtype=torch.int32)[::2]
    buffers = (
        torch.arange(20, dtype=torch.float32).reshape(4, 5),
        torch.arange(8, dtype=torch.int32).reshape(4, 2),
        torch.arange(4, dtype=torch.int64),
        torch.arange(24, dtype=torch.bfloat16).reshape(4, 2, 3),
    )
    with _ForceTpu():
        actual = kernels.gather_spec_extras_func(indices, *buffers)
        empty = kernels.gather_spec_extras_func(indices[:0], *buffers)

    expected_indices = indices.to(torch.int64)
    for result, source in zip(actual, buffers):
        assert torch.equal(result, source.index_select(0, expected_indices))
    assert [result.shape[0] for result in empty] == [0, 0, 0, 0]


def test_draft_cache_gather_randomized_and_empty():
    # Seed only CPU tensors: torch.manual_seed also calls the TPU RNG hook
    # when torch_tpu is installed, needlessly initializing its runtime.
    torch.random.default_generator.manual_seed(7)
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
    torch.random.default_generator.manual_seed(11)
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
        verify_locs = eagle_utils_tpu.prepare_verify_cache_locs_tpu(
            SimpleNamespace(device_accept_cache_locs=None, draft_token_num=2),
            SimpleNamespace(req_to_token=table),
            SimpleNamespace(
                req_pool_indices=torch.zeros(2, dtype=torch.int64),
                seq_lens=torch.zeros(2, dtype=torch.int64),
                device=torch.device("cpu"),
                req_pool_indices_cpu=torch.tensor([2, 3]),
                seq_lens_cpu=torch.tensor([5, 7]),
            ),
        )
    assert torch.equal(out, torch.tensor([69, 70, 103, 104]))
    assert torch.equal(verify_locs, out)


def test_allocator_fills_caller_owned_host_allocation():
    table = torch.zeros((4, 32), dtype=torch.int32)
    host_out = torch.tensor([101, 102, 103, 104], dtype=torch.int32)
    device_out = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
    allocator = SimpleNamespace(page_size=4)
    tree_cache = SimpleNamespace(token_to_kv_pool_allocator=allocator)
    pool = SimpleNamespace(req_to_token=table, host_resident=True)
    host_rows = Mock(return_value=torch.tensor([2, 3]))
    cur = torch.tensor([5, 7], dtype=torch.int32)
    nxt = torch.tensor([7, 9], dtype=torch.int32)
    with (
        patch.object(kernels, "get_last_loc", return_value=torch.tensor([1, 2])),
        patch.object(kernels, "alloc_paged_token_slots_extend") as alloc,
        patch.object(kernels, "assign_req_to_token_pool_func") as assign,
    ):
        alloc.side_effect = lambda *args, **kwargs: (
            kwargs["out_cache_loc_host"].copy_(host_out),
            device_out,
        )[1]
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
    assert torch.equal(assign.call_args.args[4], host_out)
    assert alloc.call_args.kwargs["out_cache_loc_host"] is assign.call_args.args[4]
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
    allocator = SimpleNamespace(page_size=4)
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


def test_eagle_prepare_for_decode_uses_torch_native_allocator():
    # Both lengths live on Req.kv (ReqKvInfo).
    reqs = [
        SimpleNamespace(
            kv=SimpleNamespace(kv_allocated_len=2, kv_committed_len=3),
            decode_batch_idx=0,
        ),
        SimpleNamespace(
            kv=SimpleNamespace(kv_allocated_len=5, kv_committed_len=1),
            decode_batch_idx=0,
        ),
    ]
    req_pool_indices = torch.tensor([2, 3], dtype=torch.int64)
    req_pool_indices_cpu = torch.tensor([2, 3], dtype=torch.int64)
    batch = SimpleNamespace(
        maybe_evict_swa=Mock(),
        batch_size=Mock(return_value=2),
        sampling_info=SimpleNamespace(
            penalizer_orchestrator=SimpleNamespace(is_required=False)
        ),
        token_to_kv_pool_allocator=SimpleNamespace(page_size=1),
        reqs=reqs,
        tree_cache=object(),
        req_to_token_pool=SimpleNamespace(host_resident=True),
        req_pool_indices=req_pool_indices,
        req_pool_indices_cpu=req_pool_indices_cpu,
        device=torch.device("cpu"),
        enable_overlap=False,
        spec_info=None,
    )
    out = torch.arange(5)

    with (
        patch.object(eagle_utils, "_is_tpu", True),
        patch.object(
            eagle_utils,
            "prepare_device_accept_tpu",
            eagle_utils_tpu.prepare_device_accept_tpu,
            create=True,
        ),
        patch.object(eagle_utils, "get_alloc_reserve_per_decode", return_value=4),
        patch.object(
            kernels, "allocate_and_assign_decode_cache", return_value=out
        ) as allocate,
    ):
        eagle_utils.eagle_prepare_for_decode(batch)

    allocate.assert_called_once()
    kwargs = allocate.call_args.kwargs
    assert kwargs["tree_cache"] is batch.tree_cache
    assert kwargs["req_to_token_pool"] is batch.req_to_token_pool
    assert kwargs["req_pool_indices"] is req_pool_indices
    assert kwargs["host_req_pool_indices"] is req_pool_indices_cpu
    assert kwargs["num_needed_tokens"] == 5
    assert kwargs["batch_size"] == 2
    assert torch.equal(kwargs["cur_kv_lens_cpu"], torch.tensor([2, 5]))
    assert torch.equal(kwargs["nxt_kv_lens_cpu"], torch.tensor([7, 5]))


def test_dflash_prepare_for_decode_uses_torch_native_allocator():
    import sglang.srt.speculative.dflash_info_v2 as dflash_info_v2
    from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2

    # Both lengths live on Req.kv (ReqKvInfo).
    reqs = [
        SimpleNamespace(
            kv=SimpleNamespace(kv_allocated_len=2, kv_committed_len=3),
            sampling_params=SimpleNamespace(top_k=1),
            decode_batch_idx=0,
        ),
        SimpleNamespace(
            kv=SimpleNamespace(kv_allocated_len=5, kv_committed_len=1),
            sampling_params=SimpleNamespace(top_k=1),
            decode_batch_idx=0,
        ),
    ]
    req_pool_indices = torch.tensor([2, 3], dtype=torch.int64)
    req_pool_indices_cpu = torch.tensor([2, 3], dtype=torch.int64)
    batch = SimpleNamespace(
        batch_size=Mock(return_value=2),
        token_to_kv_pool_allocator=SimpleNamespace(page_size=1),
        reqs=reqs,
        tree_cache=object(),
        req_to_token_pool=SimpleNamespace(host_resident=True),
        req_pool_indices=req_pool_indices,
        req_pool_indices_cpu=req_pool_indices_cpu,
        device=torch.device("cpu"),
        maybe_evict_swa=Mock(),
        # Device accept routing is a separate concern; keep this case on the
        # synchronizing path so the allocator assertions below stay the subject.
        enable_overlap=False,
    )
    out = torch.arange(5)
    spec_mock = SimpleNamespace(
        speculative_num_draft_tokens=4, speculative_draft_window_size=None
    )

    info = DFlashDraftInputV2.create_idle_input(device=torch.device("cpu"))

    with (
        patch.object(dflash_info_v2, "_is_tpu", True),
        patch.object(dflash_info_v2, "get_spec", return_value=spec_mock),
        patch.object(
            dflash_info_v2, "allocate_and_assign_decode_cache", return_value=out
        ) as allocate,
    ):
        info.prepare_for_decode(batch)

    allocate.assert_called_once()
    kwargs = allocate.call_args.kwargs
    assert kwargs["tree_cache"] is batch.tree_cache
    assert kwargs["req_to_token_pool"] is batch.req_to_token_pool
    assert kwargs["req_pool_indices"] is req_pool_indices
    assert kwargs["host_req_pool_indices"] is req_pool_indices_cpu
    assert kwargs["num_needed_tokens"] == 13
    assert kwargs["batch_size"] == 2
    assert torch.equal(
        kwargs["cur_kv_lens_cpu"], torch.tensor([2, 5], dtype=torch.int32)
    )
    assert torch.equal(
        kwargs["nxt_kv_lens_cpu"], torch.tensor([11, 9], dtype=torch.int32)
    )
    assert reqs[0].kv.kv_allocated_len == 11
    assert reqs[1].kv.kv_allocated_len == 9


@pytest.mark.parametrize(
    "page_size,num_steps,num_draft_tokens,max_num_draft_tokens,candidate_width",
    [
        (1, 1, 2, 2, 3),
        (16, 2, 3, 3, 5),
        (64, 3, 4, 4, 7),
        # Adaptive spec: the window follows the reserved bound, not the
        # currently active draft-token count.
        (64, 3, 4, 5, 9),
        (64, 0, 1, 1, None),
    ],
)
def test_eagle_prepare_for_decode_builds_device_accept_window_for_paged_tpu(
    page_size, num_steps, num_draft_tokens, max_num_draft_tokens, candidate_width
):
    spec = SimpleNamespace(
        speculative_num_steps=num_steps,
        speculative_eagle_topk=1,
        speculative_num_draft_tokens=num_draft_tokens,
        speculative_use_rejection_sampling=False,
    )
    host_base = max(1, page_size - 2)
    req = SimpleNamespace(
        kv=SimpleNamespace(kv_allocated_len=0, kv_committed_len=host_base - 1),
        decode_batch_idx=0,
        seqlen=host_base,
        lora_id=None,
    )
    table = torch.arange(2 * 512, dtype=torch.int32).reshape(2, 512)
    draft_input = SimpleNamespace(
        accept_lens_device=torch.tensor([num_draft_tokens], dtype=torch.int32),
        device_accept_candidate_locs=None,
        device_accept_selected_cache_locs=torch.tensor([[999]]),
    )
    batch = SimpleNamespace(
        maybe_evict_swa=Mock(),
        batch_size=Mock(return_value=1),
        sampling_info=SimpleNamespace(
            penalizer_orchestrator=SimpleNamespace(is_required=False)
        ),
        token_to_kv_pool_allocator=SimpleNamespace(page_size=page_size),
        reqs=[req],
        tree_cache=object(),
        req_to_token_pool=SimpleNamespace(host_resident=True, req_to_token=table),
        req_pool_indices=torch.tensor([1], dtype=torch.int64),
        req_pool_indices_cpu=torch.tensor([1], dtype=torch.int64),
        device=torch.device("cpu"),
        enable_overlap=True,
        spec_info=draft_input,
        return_logprob=False,
        has_grammar=False,
    )

    with (
        patch.object(eagle_utils, "_is_tpu", True),
        patch.object(
            eagle_utils,
            "prepare_device_accept_tpu",
            eagle_utils_tpu.prepare_device_accept_tpu,
            create=True,
        ),
        patch.object(
            eagle_utils,
            "get_alloc_reserve_per_decode",
            return_value=2 * max_num_draft_tokens,
        ),
        patch.object(eagle_utils, "get_spec", return_value=spec),
        patch.object(eagle_utils_tpu, "get_spec", return_value=spec),
        patch.object(
            eagle_utils_tpu,
            "max_speculative_num_draft_tokens",
            return_value=max_num_draft_tokens,
        ),
        patch.object(kernels, "allocate_and_assign_decode_cache"),
    ):
        eagle_utils.eagle_prepare_for_decode(batch)

    if candidate_width is None:
        assert draft_input.device_accept_candidate_locs is None
        return

    assert draft_input.device_accept_selected_cache_locs is None
    assert torch.equal(batch.seq_lens_cpu, torch.tensor([host_base]))
    assert batch.seq_lens_sum == host_base
    # The candidate window covers minimum and maximum acceptance followed
    # by the next full-width verification window.
    assert torch.equal(
        draft_input.device_accept_candidate_locs,
        table[1, host_base + 1 : host_base + 1 + candidate_width]
        .to(torch.int64)
        .reshape(1, candidate_width),
    )
    expected_alloc = (
        (host_base + 2 * max_num_draft_tokens + page_size - 1) // page_size * page_size
    )
    assert req.kv.kv_allocated_len == expected_alloc


def test_device_accept_state_tracks_filtered_and_merged_future_rows():
    from sglang.srt.speculative import eagle_info

    candidates = torch.arange(15).reshape(3, 5)
    draft = eagle_info.EagleDraftInput(
        future_indices=torch.tensor([10, 11, 12]),
        accept_lens_device=torch.tensor([1, 2, 3]),
        device_accept_candidate_locs=candidates,
        device_accept_selected_cache_locs=candidates[:, :3],
    )
    other = eagle_info.EagleDraftInput(
        future_indices=torch.tensor([13]),
        accept_lens_device=torch.tensor([2]),
        device_accept_candidate_locs=torch.arange(20, 25).reshape(1, 5),
    )
    with (
        patch.object(eagle_info, "_is_tpu", True),
        patch.object(
            eagle_info,
            "filter_device_accept_tpu",
            eagle_utils_tpu.filter_device_accept_tpu,
            create=True,
        ),
        patch.object(
            eagle_info,
            "merge_device_accept_tpu",
            eagle_utils_tpu.merge_device_accept_tpu,
            create=True,
        ),
    ):
        draft.filter_batch(torch.tensor([2, 0]))
        torch.testing.assert_close(draft.future_indices, torch.tensor([12, 10]))
        torch.testing.assert_close(draft.accept_lens_device, torch.tensor([3, 1]))
        torch.testing.assert_close(
            draft.device_accept_selected_cache_locs, candidates[[2, 0], :3]
        )
        draft.merge_batch(other)
        torch.testing.assert_close(draft.future_indices, torch.tensor([12, 10, 13]))
        torch.testing.assert_close(draft.accept_lens_device, torch.tensor([3, 1, 2]))
        torch.testing.assert_close(
            draft.device_accept_candidate_locs,
            torch.cat([candidates[[2, 0]], other.device_accept_candidate_locs]),
        )
        assert draft.device_accept_selected_cache_locs is None
        draft.merge_batch(eagle_info.EagleDraftInput(future_indices=torch.tensor([14])))
        assert draft.accept_lens_device is None
        assert draft.device_accept_candidate_locs is None


@pytest.mark.parametrize(
    "candidate_shape,accept_shape",
    [((5,), (1,)), ((2, 5), (1,)), ((1, 5), (2,)), ((1, 4), (1,))],
)
def test_device_accept_rejects_invalid_future_window(candidate_shape, accept_shape):
    from sglang_tpu.managers.overlap_utils_tpu import (
        resolve_spec_extras_tpu,
    )

    batch = SimpleNamespace(
        spec_info=SimpleNamespace(
            future_indices=torch.tensor([0]),
            device_accept_candidate_locs=torch.zeros(candidate_shape),
            device_accept_selected_cache_locs=None,
            accept_lens_device=torch.ones(accept_shape),
            future_dsa_topk_indices_available=False,
        ),
    )
    future_map = SimpleNamespace(need_topk=True, draft_probs_buf=None)
    with pytest.raises(AssertionError, match="cache|acceptance"):
        resolve_spec_extras_tpu(future_map, batch)


@pytest.mark.parametrize("is_tpu", (False, True))
def test_device_accept_cache_window_follows_verify_input(is_tpu):
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang.srt.speculative import eagle_worker_common
    from sglang.srt.speculative.eagle_info import EagleDraftInput

    selected = torch.arange(8).reshape(2, 4)
    draft_input = EagleDraftInput(device_accept_selected_cache_locs=selected)
    batch = SimpleNamespace(
        forward_mode=ForwardMode.DECODE,
        seq_lens=torch.tensor([10, 20]),
        req_pool_indices=torch.arange(2),
        spec_info=draft_input,
    )
    prebuilt_tree = (torch.ones(1, dtype=torch.bool),) + tuple(
        torch.zeros((2, 4), dtype=torch.int64) for _ in range(5)
    )

    def build():
        return eagle_worker_common.build_eagle_verify_input(
            batch,
            batch.spec_info,
            None,
            None,
            None,
            None,
            target_worker=None,
            topk=1,
            num_steps=3,
            num_draft_tokens=4,
            tree_mask_mode=None,
            device="cpu",
            prebuilt_tree=prebuilt_tree,
        )

    with (
        patch.object(eagle_worker_common, "_is_tpu", is_tpu),
        patch.object(
            eagle_worker_common,
            "device_accept_verify_cache_locs_tpu",
            eagle_utils_tpu.device_accept_verify_cache_locs_tpu,
            create=True,
        ),
    ):
        verify_input = build()
        batch.spec_info = verify_input
        if is_tpu:
            torch.testing.assert_close(
                verify_input.device_accept_cache_locs, selected.flatten()
            )
        else:
            assert verify_input.device_accept_cache_locs is None
        # A fresh iteration must not inherit the previous verify window.
        batch.spec_info = EagleDraftInput()
        assert build().device_accept_cache_locs is None
        # Resetting draft routing does not mutate an in-flight verify input.
        draft_input.device_accept_selected_cache_locs = None
        if is_tpu:
            torch.testing.assert_close(
                verify_input.device_accept_cache_locs, selected.flatten()
            )

            # Verify consumes the device-selected window without reading the host pool.
            verify_locs = eagle_utils_tpu.prepare_verify_cache_locs_tpu(
                verify_input, None, batch
            )
            torch.testing.assert_close(verify_locs, selected.flatten())
            assert verify_input.device_accept_cache_locs is None


@pytest.mark.parametrize("with_hidden", (False, True))
def test_device_accept_gather_routes_seq_lens_and_reuses_state(with_hidden):
    from sglang.srt.managers import overlap_utils
    from sglang_tpu.managers import overlap_utils_tpu

    bs = 3
    indices = torch.arange(1, bs + 1, dtype=torch.int64)
    candidates = torch.arange(bs * 7).reshape(bs, 7)
    accept_lens = torch.arange(bs, dtype=torch.int32) % 4 + 1
    draft = SimpleNamespace(
        future_indices=indices,
        device_accept_candidate_locs=candidates,
        device_accept_selected_cache_locs=None,
        accept_lens_device=accept_lens,
        future_dsa_topk_indices_available=False,
    )
    future_map = SimpleNamespace(
        publish_ready=None,
        tpu_direct_handoff=None,
        new_seq_lens_buf=torch.arange(8) + 20,
        need_topk=True,
        need_hidden_states=with_hidden,
        draft_probs_buf=None,
        topk_p_buf=torch.arange(8, dtype=torch.float32)[:, None],
        topk_index_buf=torch.arange(8)[:, None],
        output_tokens_buf=torch.arange(8) + 10,
        hidden_states_buf=torch.arange(16).reshape(8, 2) if with_hidden else None,
    )
    batch = SimpleNamespace(spec_info=draft, seq_lens_cpu=torch.tensor([99]))

    def gather(rows, probs, topk, tokens, hidden):
        return (
            probs[rows],
            topk[rows],
            tokens[rows],
            hidden[rows] if hidden is not None else None,
        )

    def invalidate(values, buf, rows):
        assert (values >= 0).all()
        buf[rows] = -1

    with (
        patch.object(overlap_utils, "_is_tpu", True),
        patch.object(
            overlap_utils,
            "resolve_device_accept_seq_lens_tpu",
            overlap_utils_tpu.resolve_device_accept_seq_lens_tpu,
            create=True,
        ),
        patch.object(
            overlap_utils,
            "device_accept_resolved_tpu",
            overlap_utils_tpu.device_accept_resolved_tpu,
            create=True,
        ),
        patch.object(kernels, "gather_spec_extras_func", side_effect=gather),
        patch.object(overlap_utils, "_DEBUG_ASSERT", True),
        patch.object(
            overlap_utils, "_assert_nonneg_and_invalidate", side_effect=invalidate
        ) as debug,
    ):
        overlap_utils.FutureMap.resolve_seq_lens_cpu(future_map, batch)
        torch.testing.assert_close(batch.seq_lens, indices + 20)
        torch.testing.assert_close(batch.seq_lens_cpu, torch.tensor([99]))
        torch.testing.assert_close(debug.call_args.args[2], indices)
        # Reusing resolved state must neither gather nor invalidate again.
        overlap_utils.FutureMap._resolve_spec_extras(future_map, batch)
        assert debug.call_count == 1

    torch.testing.assert_close(draft.topk_p, future_map.topk_p_buf[indices])
    torch.testing.assert_close(draft.topk_index, future_map.topk_index_buf[indices])
    torch.testing.assert_close(draft.bonus_tokens, indices + 10)
    expected = torch.stack(
        [
            candidates[i, int(length) - 1 : int(length) + 3]
            for i, length in enumerate(accept_lens)
        ]
    )
    torch.testing.assert_close(draft.device_accept_selected_cache_locs, expected)
    if with_hidden:
        torch.testing.assert_close(
            draft.hidden_states, future_map.hidden_states_buf[indices]
        )
    torch.testing.assert_close(
        future_map.output_tokens_buf[bs + 1 :], torch.arange(bs + 1, 8) + 10
    )
    assert draft.future_indices is indices
    assert draft.accept_lens_device is accept_lens
    assert draft.device_accept_candidate_locs is candidates


def test_draft_extend_positions_follow_device_seq_lens():
    from sglang_tpu.speculative.eagle_utils_tpu import (
        draft_extend_positions_tpu,
    )

    positions = draft_extend_positions_tpu(torch.tensor([5, 9]), 3)
    assert torch.equal(positions, torch.tensor([5, 6, 7, 9, 10, 11]))


def test_prepare_for_draft_extend_tpu_ignores_stale_host_seq_lens():
    """Device accept routing leaves seq_lens_cpu one acceptance behind; the
    draft-extend positions must come from the device seq_lens."""
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang.srt.speculative import eagle_worker_common

    seq_lens = torch.tensor([7, 12])
    stale_seq_lens_cpu = torch.tensor([4, 10])
    batch = SimpleNamespace(
        seq_lens=seq_lens,
        seq_lens_cpu=stale_seq_lens_cpu,
        forward_mode=ForwardMode.DECODE,
        model_config=SimpleNamespace(vocab_size=100),
        spec_info=None,
    )
    draft_extend_input = SimpleNamespace(
        num_front_tokens=0, hidden_states=None, positions=None
    )
    forward_batch = SimpleNamespace(
        seq_lens=seq_lens.clone(),
        seq_lens_cpu=stale_seq_lens_cpu.clone(),
        forward_mode=ForwardMode.DRAFT_EXTEND_V2,
        mark_forward_metadata_ready=Mock(),
    )
    draft_model_runner = Mock()
    draft_model_runner.spec_algorithm.is_standalone.return_value = False
    with (
        patch.object(eagle_worker_common, "_is_tpu", True),
        patch.object(
            eagle_worker_common,
            "draft_extend_positions_tpu",
            eagle_utils_tpu.draft_extend_positions_tpu,
            create=True,
        ),
        patch.object(
            eagle_worker_common.ForwardBatch, "init_new", return_value=forward_batch
        ),
        patch.object(eagle_worker_common, "maybe_detect_oob"),
        patch.object(eagle_worker_common, "is_npu", return_value=False),
    ):
        eagle_worker_common.prepare_for_draft_extend(
            draft_extend_input,
            batch,
            torch.arange(6),
            3,
            draft_model_runner,
            None,
            return_hidden_states_before_norm=False,
        )

    assert torch.equal(
        draft_extend_input.positions, torch.tensor([7, 8, 9, 12, 13, 14])
    )
    # The host-derived prefix is still built for init_new but is not what
    # the positions come from.
    assert batch.prefix_lens == [4, 10]


@pytest.mark.parametrize("algorithm", ("EAGLE3", "DFLASH", "DSPARK"))
@pytest.mark.parametrize(
    "host_resident,overlap", ((True, True), (True, False), (False, True))
)
def test_decode_reservation_matches_allocator_at_page_boundary(
    algorithm, host_resident, overlap
):
    from sglang.srt.managers import schedule_batch
    from sglang.srt.mem_cache.allocation_sizing import page_aligned_decode_alloc_lens
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    reqs = [
        SimpleNamespace(kv=SimpleNamespace(kv_committed_len=96, kv_allocated_len=128))
    ]
    batch = SimpleNamespace(
        enable_overlap=overlap,
        spec_algorithm=SpeculativeAlgorithm[algorithm],
        req_to_token_pool=SimpleNamespace(host_resident=host_resident),
    )
    # A 16-token DFlash/DSpark block reserves 32 tokens. Only EAGLE device
    # accept routing (host-resident req_to_token + overlap) needs the extra
    # slot that crosses this page boundary.
    expected_tokens = 16 if algorithm == "EAGLE3" and host_resident and overlap else 0
    with patch.object(schedule_batch, "get_alloc_reserve_per_decode", return_value=32):
        estimate = (
            schedule_batch.ScheduleBatch._new_tokens_required_next_decode_spec_v2(
                batch, reqs, page_size=16
            )
        )
    assert estimate == expected_tokens
    if algorithm != "EAGLE3":
        _, _, actual = page_aligned_decode_alloc_lens(reqs, reserve=32, page_size=16)
        assert estimate == actual


def _ragged_reference(table, rows, starts, width):
    """The generic ragged gather this TPU window replaces, as the oracle."""
    out = torch.zeros(rows.numel() * width, dtype=torch.int64)
    with _ForceTpu():
        kernels.assign_extend_cache_locs_kernel_func(
            rows, table, starts, starts + width, out
        )
    return out


def test_verify_window_matches_ragged_reference_and_owns_output():
    torch.random.default_generator.manual_seed(23)
    table = torch.randint(0, 10000, (8, 129), dtype=torch.int32)
    for bs in (0, 1, 4, 8):
        for width in (1, 2, 4, 7):
            # Reversed rows catch a gather that assumes request order, and the
            # negative and past-the-row starts exercise the column clamp.
            rows = torch.arange(bs - 1, -1, -1, dtype=torch.int64)
            starts = torch.arange(bs, dtype=torch.int64) * 17 - 2
            if bs:
                starts[-1] = 128
            expected = _ragged_reference(table, rows, starts, width)
            actual = eagle_utils_tpu.prepare_verify_cache_locs_tpu(
                SimpleNamespace(device_accept_cache_locs=None, draft_token_num=width),
                SimpleNamespace(req_to_token=table),
                SimpleNamespace(
                    req_pool_indices=rows,
                    seq_lens=starts,
                    device=torch.device("cpu"),
                    req_pool_indices_cpu=rows,
                    seq_lens_cpu=starts,
                ),
            )
            assert actual.dtype == torch.int64
            assert actual.shape == (bs * width,)
            assert torch.equal(actual, expected)
            # The window must own its storage: the page table keeps changing
            # under the verify that consumes it.
            saved = actual.clone()
            table.add_(1)
            assert torch.equal(actual, saved)


def test_token_window_only_clamps_when_asked():
    table = torch.arange(4 * 8).reshape(4, 8)
    rows = torch.tensor([1, 2], dtype=torch.int64)
    starts = torch.tensor([6, 6], dtype=torch.int64)
    clamped = eagle_utils_tpu._gather_token_window(
        table, rows, starts, 4, clamp_columns=True
    )
    assert torch.equal(clamped, torch.tensor([[14, 15, 15, 15], [22, 23, 23, 23]]))
    # prepare_device_accept_tpu relies on an unclamped window raising instead of
    # silently folding a bad start onto the row edge.
    with pytest.raises(IndexError):
        eagle_utils_tpu._gather_token_window(table, rows, starts, 4)


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
