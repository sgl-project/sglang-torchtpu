"""CPU coverage for DFlash device-resident verify-block addressing."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

import sglang.srt.managers.overlap_utils as overlap_utils
import sglang.srt.speculative.dflash_info_v2 as dflash_info_v2
from sglang.srt.managers.overlap_utils import FutureMap
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang_tpu.dflash_adaptive_verify import (
    VerifyWidthRule,
    adaptive_verify_shapes,
    full_verify_buckets,
    parse_adaptive_verify,
    select_verify_width,
)
from sglang_tpu.dflash_utils_tpu import (
    DeviceVerifyBlockSelector,
    build_dflash_draft_block,
    gather_dflash_candidate_window,
    select_dflash_verify_cache_locs_tpu,
)
from sglang_tpu.overlap_utils_tpu import (
    resolve_device_accept_seq_lens_tpu,
)

BLOCK_SIZE = 4
MAX_CTX = 64


class _DeviceTensor(torch.Tensor):
    """CPU-backed stand-in for a tensor that lives on the accelerator.

    Everything is CPU under test, so residency cannot be read off the device
    field; this tag carries it instead and TransferRecorder classifies by tag.
    """

    @staticmethod
    def __new__(cls, data):
        return torch.Tensor._make_subclass(cls, data.detach())


_D2H_FUNCS = frozenset(
    {
        torch.Tensor.cpu,
        torch.Tensor.item,
        torch.Tensor.tolist,
        torch.Tensor.numpy,
        torch.Tensor.__int__,
        torch.Tensor.__float__,
        torch.Tensor.__index__,
    }
)


def _touches_device(value) -> bool:
    if isinstance(value, _DeviceTensor):
        return True
    if isinstance(value, (list, tuple)):
        return any(_touches_device(v) for v in value)
    if isinstance(value, dict):
        return any(_touches_device(v) for v in value.values())
    return False


class TransferRecorder(torch.overrides.TorchFunctionMode):
    """Records every host/device crossing a _DeviceTensor takes part in."""

    def __init__(self):
        super().__init__()
        self.d2h: list[str] = []
        self.h2d: list[str] = []

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func in _D2H_FUNCS:
            if _touches_device(args) or _touches_device(kwargs):
                self.d2h.append(func.__name__)
        elif func is torch.Tensor.to:
            self._record_to(args, kwargs)
        elif func is torch.Tensor.copy_ and len(args) >= 2:
            dst, src = args[0], args[1]
            if _touches_device(src) and not _touches_device(dst):
                self.d2h.append("copy_")
            elif _touches_device(dst) and not _touches_device(src):
                self.h2d.append("copy_")
        return func(*args, **kwargs)

    def _record_to(self, args, kwargs):
        # A dtype-only .to() never crosses; only a device argument does.
        target = kwargs.get("device")
        for arg in args[1:]:
            if isinstance(arg, (str, torch.device)):
                target = arg
        if target is None or not args:
            return
        if _touches_device(args[0]):
            self.d2h.append("to")
        else:
            self.h2d.append("to")


def test_transfer_recorder_observes_the_crossings_it_claims_to():
    """Without this, every empty-log assertion below could be vacuous."""
    device = _DeviceTensor(torch.tensor([1, 2], dtype=torch.int64))
    host = torch.tensor([3, 4], dtype=torch.int64)

    with TransferRecorder() as log:
        device.cpu()
        int(device[0])
        device.to("cpu")
    assert len(log.d2h) == 3, log.d2h
    assert log.h2d == []

    with TransferRecorder() as log:
        host.to(torch.device("cpu"))
    assert log.h2d == ["to"], log.h2d

    with TransferRecorder() as log:
        # A dtype-only cast stays put and must not be counted either way.
        host.to(torch.int32)
        device.to(torch.int32)
    assert log.d2h == [] and log.h2d == []


def _req_to_token(num_rows: int) -> torch.Tensor:
    """Distinct slot ids so a mis-addressed gather cannot alias a correct one."""
    return (
        torch.arange(num_rows * MAX_CTX, dtype=torch.int64).view(num_rows, MAX_CTX)
        + 1000
    )


def test_window_selection_matches_direct_lookup_for_every_commit_length():
    req_to_token = _req_to_token(3)
    rows = torch.tensor([2, 0, 1], dtype=torch.int64)
    base = torch.tensor([10, 7, 23], dtype=torch.int64)
    candidates = gather_dflash_candidate_window(req_to_token, rows, base, BLOCK_SIZE)
    assert candidates.shape == (3, 2 * BLOCK_SIZE)

    # commit_lens is in [1, block_size]; 0 covers an extend/idle predecessor.
    for commit in range(0, BLOCK_SIZE + 1):
        seq_lens = base + commit
        selected = select_dflash_verify_cache_locs_tpu(
            candidates, seq_lens, base, BLOCK_SIZE
        )
        expected = torch.stack(
            [
                req_to_token[row, start : start + BLOCK_SIZE]
                for row, start in zip(rows.tolist(), seq_lens.tolist())
            ]
        )
        torch.testing.assert_close(selected, expected)


def test_window_covers_the_reserved_two_block_span():
    """The last usable start is base + block_size, so the window needs 2 * B."""
    req_to_token = _req_to_token(1)
    rows = torch.tensor([0], dtype=torch.int64)
    base = torch.tensor([5], dtype=torch.int64)
    candidates = gather_dflash_candidate_window(req_to_token, rows, base, BLOCK_SIZE)
    last = select_dflash_verify_cache_locs_tpu(
        candidates, base + BLOCK_SIZE, base, BLOCK_SIZE
    )
    torch.testing.assert_close(last, candidates[:, BLOCK_SIZE:])


def test_offsets_are_clamped_into_the_window():
    """An out-of-contract length must stay in bounds rather than read garbage."""
    req_to_token = _req_to_token(1)
    rows = torch.tensor([0], dtype=torch.int64)
    base = torch.tensor([5], dtype=torch.int64)
    candidates = gather_dflash_candidate_window(req_to_token, rows, base, BLOCK_SIZE)
    over = select_dflash_verify_cache_locs_tpu(
        candidates, base + 3 * BLOCK_SIZE, base, BLOCK_SIZE
    )
    torch.testing.assert_close(over, candidates[:, BLOCK_SIZE:])
    under = select_dflash_verify_cache_locs_tpu(
        candidates, base - BLOCK_SIZE, base, BLOCK_SIZE
    )
    torch.testing.assert_close(under, candidates[:, :BLOCK_SIZE])


def test_padded_rows_do_not_disturb_live_rows():
    """Bucket padding leaves garbage rows; live rows must be unaffected."""
    req_to_token = _req_to_token(2)
    rows = torch.tensor([0, 1], dtype=torch.int64)
    base = torch.tensor([9, 14], dtype=torch.int64)
    candidates = gather_dflash_candidate_window(req_to_token, rows, base, BLOCK_SIZE)
    live = select_dflash_verify_cache_locs_tpu(candidates, base + 2, base, BLOCK_SIZE)

    padded_candidates = torch.full((4, 2 * BLOCK_SIZE), -7, dtype=torch.int64)
    padded_candidates[:2] = candidates
    padded_base = torch.full((4,), -7, dtype=torch.int64)
    padded_base[:2] = base
    padded_seq_lens = torch.full((4,), -7, dtype=torch.int64)
    padded_seq_lens[:2] = base + 2
    padded = select_dflash_verify_cache_locs_tpu(
        padded_candidates, padded_seq_lens, padded_base, BLOCK_SIZE
    )
    torch.testing.assert_close(padded[:2], live)


def _resolve_stub(new_seq_lens_buf: torch.Tensor) -> SimpleNamespace:
    return SimpleNamespace(
        publish_ready=None,
        new_seq_lens_buf=new_seq_lens_buf,
        spec_algo=SpeculativeAlgorithm.DFLASH,
        needs_cpu_seq_lens=True,
        fwd_prepare_d2h_stream=None,
        # The EAGLE-only device-accept paths run first and read these; without
        # them resolve raises before reaching the branch under test.
        need_topk=False,
        tpu_direct_handoff=None,
    )


def _resolve_batch(committed_base: torch.Tensor, committed: bool):
    draft = DFlashDraftInputV2.create_idle_input(device=torch.device("cpu"))
    draft.future_indices = torch.tensor([0, 1], dtype=torch.int64)
    draft.tpu_host_seq_lens_is_committed = committed
    return SimpleNamespace(
        spec_info=draft,
        seq_lens=None,
        seq_lens_cpu=committed_base,
        seq_lens_sum=int(committed_base.sum()),
        req_pool_indices_cpu=torch.tensor([0, 1], dtype=torch.int64),
    )


def _run_resolve(committed: bool, committed_base: torch.Tensor):
    published = _DeviceTensor(torch.tensor([31, 42], dtype=torch.int64))
    batch = _resolve_batch(committed_base, committed=committed)
    with (
        patch.object(overlap_utils, "_is_tpu", True),
        patch.object(overlap_utils, "_DEBUG_ASSERT", False),
        patch.object(
            overlap_utils,
            "resolve_device_accept_seq_lens_tpu",
            resolve_device_accept_seq_lens_tpu,
            create=True,
        ),
        TransferRecorder() as log,
    ):
        FutureMap.resolve_seq_lens_cpu(_resolve_stub(published), batch)
    return batch, log


def test_resolve_seq_lens_cpu_makes_no_host_transfer_on_tpu():
    """The whole point: resolving the published length must not cross buses."""
    committed_base = torch.tensor([28, 39], dtype=torch.int64)
    batch, log = _run_resolve(committed=True, committed_base=committed_base)

    assert log.d2h == [], f"unexpected device-to-host transfer: {log.d2h}"
    assert log.h2d == [], f"unexpected host-to-device transfer: {log.h2d}"
    # Device length advanced, host mirror untouched: still the committed base.
    torch.testing.assert_close(
        batch.seq_lens, torch.tensor([31, 42], dtype=torch.int64), check_device=False
    )
    assert batch.seq_lens_cpu is committed_base
    assert batch.seq_lens_sum == int(committed_base.sum())


def test_resolve_seq_lens_cpu_still_syncs_without_the_committed_flag():
    """Guards the test above: without the flag the D2H really is recorded."""
    _, log = _run_resolve(
        committed=False, committed_base=torch.tensor([28, 39], dtype=torch.int64)
    )
    assert log.d2h, "recorder failed to observe the legacy host copy"


def _prepare_batch(req_to_token: torch.Tensor):
    reqs = [
        SimpleNamespace(
            kv=SimpleNamespace(kv_allocated_len=64, kv_committed_len=committed),
            sampling_params=SimpleNamespace(top_k=1),
            decode_batch_idx=0,
            lora_id=None,
        )
        for committed in (10, 7)
    ]
    return SimpleNamespace(
        batch_size=Mock(return_value=2),
        token_to_kv_pool_allocator=SimpleNamespace(page_size=1),
        reqs=reqs,
        tree_cache=object(),
        req_to_token_pool=SimpleNamespace(
            host_resident=True, req_to_token=req_to_token
        ),
        req_pool_indices=torch.tensor([1, 0], dtype=torch.int64),
        req_pool_indices_cpu=torch.tensor([1, 0], dtype=torch.int64),
        device=torch.device("cpu"),
        maybe_evict_swa=Mock(),
        enable_overlap=True,
        spec_algorithm=SpeculativeAlgorithm.DFLASH,
        return_logprob=False,
        has_grammar=False,
    )


def test_prepare_for_decode_pregathers_the_window_and_keeps_the_committed_mirror():
    req_to_token = _req_to_token(2)
    batch = _prepare_batch(req_to_token)
    info = DFlashDraftInputV2.create_idle_input(device=torch.device("cpu"))
    spec_mock = SimpleNamespace(
        speculative_num_draft_tokens=BLOCK_SIZE, speculative_draft_window_size=None
    )

    with (
        patch.object(dflash_info_v2, "_is_tpu", True),
        patch.object(dflash_info_v2, "get_spec", return_value=spec_mock),
        patch.object(dflash_info_v2, "allocate_and_assign_decode_cache", Mock()),
    ):
        info.prepare_for_decode(batch)

    assert info.tpu_host_seq_lens_is_committed
    assert info.tpu_verify_cache_candidates.shape == (2, 2 * BLOCK_SIZE)
    torch.testing.assert_close(
        info.tpu_committed_lens, torch.tensor([10, 7], dtype=torch.int64)
    )
    torch.testing.assert_close(
        batch.seq_lens_cpu, torch.tensor([10, 7], dtype=torch.int64)
    )
    # The window must be the rows the worker will select from.
    torch.testing.assert_close(
        info.tpu_verify_cache_candidates[0], req_to_token[1, 10 : 10 + 2 * BLOCK_SIZE]
    )
    torch.testing.assert_close(
        info.tpu_verify_cache_candidates[1], req_to_token[0, 7 : 7 + 2 * BLOCK_SIZE]
    )


def test_selector_pads_to_the_bucket_and_regrows():
    """Bucket rounding and buffer reuse are the selector's only own logic.

    A capacity that stops being a bucket, or rows left over from a wider
    previous call, both surface as a mismatch against the direct op.
    """
    req_to_token = _req_to_token(6)
    buckets = (4, 8)
    selector = DeviceVerifyBlockSelector(
        device=torch.device("cpu"),
        block_size=BLOCK_SIZE,
        pad_bs=lambda bs: next(b for b in buckets if b >= bs),
    )

    # Grow past the first bucket, then shrink back below it.
    for bs in (2, 6, 1):
        rows = torch.arange(bs, dtype=torch.int64)
        base = torch.arange(bs, dtype=torch.int64) + 5
        candidates = gather_dflash_candidate_window(
            req_to_token, rows, base, BLOCK_SIZE
        )
        seq_lens = base + 2
        torch.testing.assert_close(
            selector.select(
                candidates=candidates,
                committed_lens=base,
                prefix_lens=seq_lens,
                bs=bs,
            ),
            select_dflash_verify_cache_locs_tpu(candidates, seq_lens, base, BLOCK_SIZE),
        )


def test_prepare_for_decode_stays_synchronous_when_ineligible():
    req_to_token = _req_to_token(2)
    batch = _prepare_batch(req_to_token)
    batch.has_grammar = True
    info = DFlashDraftInputV2.create_idle_input(device=torch.device("cpu"))
    spec_mock = SimpleNamespace(
        speculative_num_draft_tokens=BLOCK_SIZE, speculative_draft_window_size=None
    )

    with (
        patch.object(dflash_info_v2, "_is_tpu", True),
        patch.object(dflash_info_v2, "get_spec", return_value=spec_mock),
        patch.object(dflash_info_v2, "allocate_and_assign_decode_cache", Mock()),
    ):
        info.prepare_for_decode(batch)

    assert not info.tpu_host_seq_lens_is_committed
    assert info.tpu_verify_cache_candidates is None


def test_prepare_for_decode_uploads_the_window_without_reading_back():
    """The window costs uploads only; nothing may travel device to host."""
    batch = _prepare_batch(_req_to_token(2))
    info = DFlashDraftInputV2.create_idle_input(device=torch.device("cpu"))
    spec_mock = SimpleNamespace(
        speculative_num_draft_tokens=BLOCK_SIZE, speculative_draft_window_size=None
    )
    # The published length is the one value prepare must not consult.
    batch.seq_lens = _DeviceTensor(torch.tensor([13, 9], dtype=torch.int64))

    with (
        patch.object(dflash_info_v2, "_is_tpu", True),
        patch.object(dflash_info_v2, "get_spec", return_value=spec_mock),
        patch.object(dflash_info_v2, "allocate_and_assign_decode_cache", Mock()),
        TransferRecorder() as log,
    ):
        info.prepare_for_decode(batch)

    assert log.d2h == [], f"unexpected device-to-host transfer: {log.d2h}"
    # Exactly the candidate window and its base; anything more is a regression.
    assert len(log.h2d) == 2, f"expected 2 uploads, got {log.h2d}"


def test_device_selection_makes_no_host_transfer():
    """Picking the block's slots must stay entirely on device."""
    candidates = _DeviceTensor(
        gather_dflash_candidate_window(
            _req_to_token(2),
            torch.tensor([0, 1], dtype=torch.int64),
            torch.tensor([9, 14], dtype=torch.int64),
            BLOCK_SIZE,
        )
    )
    base = _DeviceTensor(torch.tensor([9, 14], dtype=torch.int64))
    seq_lens = _DeviceTensor(torch.tensor([11, 18], dtype=torch.int64))

    with TransferRecorder() as log:
        select_dflash_verify_cache_locs_tpu(candidates, seq_lens, base, BLOCK_SIZE)

    assert log.d2h == [], f"unexpected device-to-host transfer: {log.d2h}"
    assert log.h2d == [], f"unexpected host-to-device transfer: {log.h2d}"


@pytest.mark.parametrize("allocated_len, num_alloc_calls", [(64, 0), (12, 1)])
def test_skip_empty_host_decode_alloc_only_skips_steps_without_new_slots(
    allocated_len, num_alloc_calls
):
    """Skipping empty host decode allocations must still allocate (and record the
    allocation) whenever any request crosses its reserve."""
    batch = _prepare_batch(_req_to_token(2))
    batch.reqs[0].kv.kv_allocated_len = allocated_len
    info = DFlashDraftInputV2.create_idle_input(device=torch.device("cpu"))
    spec_mock = SimpleNamespace(
        speculative_num_draft_tokens=BLOCK_SIZE, speculative_draft_window_size=None
    )
    alloc = Mock()

    with (
        patch.object(dflash_info_v2, "_is_tpu", True),
        patch.object(dflash_info_v2, "get_spec", return_value=spec_mock),
        patch.object(dflash_info_v2, "allocate_and_assign_decode_cache", alloc),
    ):
        info.prepare_for_decode(batch)

    assert alloc.call_count == num_alloc_calls
    # committed 10 + reserve 2 * BLOCK_SIZE
    assert batch.reqs[0].kv.kv_allocated_len == max(allocated_len, 18)


def test_fused_draft_block_matches_the_step_by_step_prep():
    """The fused draft block: one graph-break-free program that equals the separate
    block-id fill, position add, verify-slot select and embedding."""
    req_to_token = _req_to_token(3)
    base = torch.tensor([10, 7, 23], dtype=torch.int64)
    candidates = gather_dflash_candidate_window(
        req_to_token, torch.tensor([2, 0, 1], dtype=torch.int64), base, BLOCK_SIZE
    )
    prefix_lens = base + torch.tensor([1, 4, 2], dtype=torch.int64)
    bonus = torch.tensor([5, 6, 7], dtype=torch.int64)
    offsets = torch.arange(BLOCK_SIZE, dtype=torch.int64)
    embedding = torch.nn.Embedding(16, 8)

    def embed(*, block_ids):
        return embedding(block_ids).flatten(0, 1)

    fused = torch.compile(build_dflash_draft_block, fullgraph=True, backend="eager")
    with torch.no_grad():
        block_ids, positions, slots, embeds = fused(
            embed=embed,
            bonus_tokens=bonus,
            prefix_lens=prefix_lens,
            candidates=candidates,
            committed_lens=base,
            block_pos_offsets=offsets,
            mask_token_id=15,
            block_size=BLOCK_SIZE,
        )
        expected_ids = torch.full((3, BLOCK_SIZE), 15, dtype=torch.int64)
        expected_ids[:, 0] = bonus
        assert torch.equal(block_ids, expected_ids)
        assert torch.equal(positions, prefix_lens[:, None] + offsets)
        assert torch.equal(
            slots,
            select_dflash_verify_cache_locs_tpu(
                candidates, prefix_lens, base, BLOCK_SIZE
            ),
        )
        assert torch.equal(embeds, embed(block_ids=expected_ids))



def test_adaptive_verify_parse_orders_rules_and_drops_full_width():
    rules = parse_adaptive_verify("5:3:2.54, 3:4:3.06, 2:8:4.05", block_size=8)
    assert rules == (
        VerifyWidthRule(min_bs=3, width=4, simulate_accept_length=3.06),
        VerifyWidthRule(min_bs=5, width=3, simulate_accept_length=2.54),
    )
    assert parse_adaptive_verify("", block_size=8) == ()


@pytest.mark.parametrize("spec", ["3", "0:4", "3:1", "3:4:5.0", "3:4,3:2"])
def test_adaptive_verify_parse_rejects_malformed(spec):
    with pytest.raises(ValueError):
        parse_adaptive_verify(spec, block_size=8)


def test_adaptive_verify_select_uses_last_rule_reached():
    rules = parse_adaptive_verify("3:4:3.06,6:3:2.54", block_size=8)
    widths = [
        None if (r := select_verify_width(rules, bs)) is None else r.width
        for bs in range(1, 9)
    ]
    assert widths == [None, None, 4, 4, 4, 3, 3, 3]


def test_mixed_verify_tails_always_verify_the_full_block():
    # A MIXED batch reserves block_size rows per tail; its tail view must never narrow.
    from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2

    worker = SimpleNamespace(
        _adaptive_verify_rules=parse_adaptive_verify("3:4:3.06", block_size=8),
        _need_mamba_verify_commit=False,
    )
    select = DFlashWorkerV2._select_verify_rule
    assert select(worker, batch=SimpleNamespace(full_verify_block=True), bs=8) is None
    decode = SimpleNamespace(has_grammar=False, return_logprob=False)
    assert select(worker, batch=decode, bs=8).width == 4


def test_adaptive_verify_shapes_map_logical_batches_to_buckets():
    rules = parse_adaptive_verify("3:4:3.06", block_size=8)
    # bs 3 and 4 run in bucket 4; bucket 2 serves only full-width batches.
    assert adaptive_verify_shapes(rules, [1, 2, 4, 8]) == {4: [4, 8]}
    assert adaptive_verify_shapes(rules, [1, 2, 3, 4]) == {4: [3, 4]}
    assert adaptive_verify_shapes((), [1, 2, 4]) == {}
    # Bucket 4 also serves bs 3 and 4, so only bucket 8 drops its full-width graph.
    assert full_verify_buckets(rules, [1, 2, 4, 8]) == [1, 2]
    assert full_verify_buckets(parse_adaptive_verify("4:4:3.06", block_size=8), [1, 2, 4, 8]) == [1, 2, 4]
    assert full_verify_buckets((), [1, 2, 4]) == [1, 2, 4]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
