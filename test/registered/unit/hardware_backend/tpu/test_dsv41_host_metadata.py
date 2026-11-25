# SPDX-License-Identifier: Apache-2.0
"""CPU tests: the DSV4.1 TPU attention metadata reads the host tensors the compile
policy uploaded instead of reading the uploads back, and stays bit-identical."""

import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from sglang.srt.model_executor.forward_batch_info import compute_position_torch
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu.compilation import host_input_mirror
from sglang_tpu.compilation.decoder_compile_policy import DecoderTpuCompilePolicy
from sglang_tpu.envs import envs as tpu_envs
from sglang_tpu.models.deepseek_v41 import attention_backend as ab

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

MAX_CONTEXT = 1 << 17
SLOTS = 8
UPLOADED = ("positions", "req_pool_indices", "out_cache_loc")


class FB(SimpleNamespace):
    """A ForwardBatch stand-in; like the dataclass, weak-referenceable."""


@pytest.fixture(autouse=True)
def _fresh_record():
    host_input_mirror.clear()
    yield
    host_input_mirror.clear()


def _req_to_token():
    base = torch.arange(SLOTS, dtype=torch.int64)[:, None] * MAX_CONTEXT
    return base + torch.arange(MAX_CONTEXT, dtype=torch.int64)[None, :] + 256


def _extend_batch(prefix, lens, slots, num_tokens, pad_b=None):
    """A bucket-padded host-staged extend, as the shim leaves it before the upload."""
    live = [(p, n) for p, n in zip(prefix, lens) if n]
    pos, _ = compute_position_torch(
        torch.tensor([p for p, _ in live]), torch.tensor([n for _, n in live])
    )
    positions = F.pad(pos.to(torch.int64), (0, num_tokens - pos.shape[0]))
    r2t = _req_to_token()
    loc = torch.zeros(num_tokens, dtype=torch.int32)
    off = 0
    for p, n, s in zip(prefix, lens, slots):
        if n:
            loc[off : off + n] = r2t[s, p : p + n].to(torch.int32)
        off += n
    pad_b = pad_b or len(prefix)
    pad = pad_b - len(prefix)
    return FB(
        forward_mode=ForwardMode.EXTEND,
        batch_size=pad_b,
        positions=positions,
        out_cache_loc=loc,
        extend_seq_lens=torch.tensor(list(lens) + [0] * pad, dtype=torch.int32),
        extend_seq_lens_cpu=list(lens) + [0] * pad,
        extend_prefix_lens_cpu=list(prefix) + [0] * pad,
        req_pool_indices=torch.tensor(list(slots) + [0] * pad, dtype=torch.int64),
        req_pool_indices_cpu=torch.tensor(list(slots) + [0] * pad, dtype=torch.int64),
        encoder_swa_row_floor=None,
        spec_info=None,
    )


def _verify_batch(seq_lens, slots, rows=6):
    """A TARGET_VERIFY batch: rows per request at seq_len + arange(rows)."""
    seq = torch.tensor(seq_lens, dtype=torch.int64)
    positions = (seq[:, None] + torch.arange(rows)).reshape(-1)
    r2t = _req_to_token()
    loc = torch.stack([r2t[s, int(L) : int(L) + rows] for s, L in zip(slots, seq)])
    return FB(
        forward_mode=ForwardMode.TARGET_VERIFY,
        batch_size=len(slots),
        positions=positions,
        out_cache_loc=loc.reshape(-1).to(torch.int32),
        # Spec batches carry a host KV-length bound: committed + rows (+ slack).
        seq_lens_cpu=seq + rows,
        extend_seq_lens=None,
        extend_seq_lens_cpu=None,
        extend_prefix_lens_cpu=None,
        req_pool_indices=torch.tensor(slots, dtype=torch.int64),
        req_pool_indices_cpu=None,
        encoder_swa_row_floor=None,
        spec_info=SimpleNamespace(draft_token_num=rows),
    )


def _upload(fb, device="meta"):
    """The compile policy's upload of the host-staged fields (meta: unreadable)."""
    policy = SimpleNamespace(model_runner=SimpleNamespace(device=torch.device(device)))
    for name in UPLOADED:
        DecoderTpuCompilePolicy._upload_field(policy, fb, name, getattr(fb, name))


def _metadata(fb):
    return ab.build_forward_metadata(
        fb,
        req_to_token=_req_to_token(),
        full_to_swa_index_mapping=None,
        request_window=None,
        swa_ring_valid_from=None,
        max_context_len=MAX_CONTEXT,
        low_ratios=ab.LOW_RATIOS,
        device=torch.device("cpu"),
    )


def _flatten(meta):
    out = {}
    for name in meta.__dataclass_fields__:
        v = getattr(meta, name)
        if isinstance(v, torch.Tensor):
            out[name] = v
        elif isinstance(v, dict):
            for r, lm in v.items():
                for i, t in enumerate(lm):
                    out[f"{name}.{r}.{i}"] = t
        elif v is not None and hasattr(v, "_fields"):
            for i, t in enumerate(v):
                out[f"{name}.{i}"] = t
    return out


def _assert_same(a, b):
    a, b = _flatten(a), _flatten(b)
    assert a.keys() == b.keys()
    for k in a:
        assert a[k].dtype == b[k].dtype, k
        assert torch.equal(a[k], b[k]), k


BATCHES = {
    "single": lambda: _extend_batch([0], [700], [3], 1024),
    "long_prefix": lambda: _extend_batch([98304], [768], [1], 1024),
    "very_long_prefix": lambda: _extend_batch([126976], [1024], [2], 1024),
    "multi": lambda: _extend_batch([256, 0, 1024], [100, 3, 256], [2, 5, 7], 512),
    "bucket_padded_rows": lambda: _extend_batch([512], [256], [4], 512, pad_b=4),
    "replay_folded": lambda: _extend_batch([98304 - 128], [128 + 640], [6], 1024),
    "verify": lambda: _verify_batch([4096, 98310, 17], [1, 2, 3]),
}


@pytest.mark.parametrize("name", list(BATCHES))
def test_metadata_from_uploaded_host_copies_is_bit_identical(name):
    reference = _metadata(BATCHES[name]())  # every field on the host: no record
    fb = BATCHES[name]()
    _upload(fb)  # fields now on "meta": a device read would raise
    _assert_same(_metadata(fb), reference)


@pytest.mark.parametrize("name", list(BATCHES))
def test_packed_upload_gives_the_same_fields(name):
    """One upload per dtype: every field is a view at its own offset and shape."""
    reference = _metadata(BATCHES[name]())
    with patch.object(
        tpu_envs, "SGLANG_TPU_DSV41_PACKED_METADATA", SimpleNamespace(get=lambda: True)
    ):
        packed = _metadata(BATCHES[name]())
    _assert_same(packed, reference)
    bases = {t.untyped_storage().data_ptr() for t in _flatten(packed).values()}
    assert len(bases) == 3  # slot tables (int32), then rows (int32, bool)


@pytest.mark.parametrize("name", ["multi", "verify"])
def test_runner_batch_copy_keeps_the_host_copies(name):
    """The runner builds metadata from a shallow copy of the uploaded batch
    (EagerRunner.load_batch); that copy must not read the uploads back."""
    import copy

    from sglang_tpu.models.deepseek_v41.hooks import (
        tpu_dsv41_carry_mirror_to_loaded_batch,
    )

    reference = _metadata(BATCHES[name]())
    fb = BATCHES[name]()
    _upload(fb)  # fields now on "meta": a device read would raise
    loaded = tpu_dsv41_carry_mirror_to_loaded_batch(
        lambda runner, batch: copy.copy(batch), None, fb
    )
    _assert_same(_metadata(loaded), reference)


def test_prebuilt_slot_tables_are_used_only_for_the_forward_slots():
    """Slot tables built ahead of a draft/verify forward replace the page-table build
    only when built for that forward's slots; other slots' tables are rebuilt."""
    fb = BATCHES["verify"]()
    reference = _metadata(BATCHES["verify"]())
    kw = dict(
        req_to_token=_req_to_token(),
        full_to_swa_index_mapping=None,
        request_window=None,
        swa_ring_valid_from=None,
        max_context_len=MAX_CONTEXT,
        low_ratios=ab.LOW_RATIOS,
        device=torch.device("cpu"),
    )
    stale = ab.build_slot_tables(req_pool_indices=torch.tensor([4, 5, 6]), **kw)
    _assert_same(ab.build_forward_metadata(fb, slot_tables=stale, **kw), reference)
    tables = ab.build_slot_tables(req_pool_indices=fb.req_pool_indices, **kw)
    with patch.object(ab, "_full_page_table", side_effect=AssertionError("rebuilt")):
        built = ab.build_forward_metadata(BATCHES["verify"](), slot_tables=tables, **kw)
    _assert_same(built, reference)


def test_slot_tables_are_reused_only_when_asked():
    """A later prepare in the same step (reuse) keeps the tables built before the
    length read; a plain prepare always rebuilds, as the page table may have grown."""
    builds = []

    def build(**kw):
        builds.append(kw["req_pool_indices"])
        return SimpleNamespace(
            req_pool_indices=kw["req_pool_indices"], bound_tokens=kw["kv_len_bound"]
        )

    backend = SimpleNamespace(
        _slot_tables=None,
        req_to_token=None,
        token_to_kv_pool=SimpleNamespace(
            full_to_swa_index_mapping=None, swa_valid_from=None
        ),
        request_window=None,
        max_context_len=MAX_CONTEXT,
        low_ratios=ab.LOW_RATIOS,
        device=torch.device("cpu"),
    )
    prepare = ab.DSV41TPUAttnBackend.prepare_slot_tables
    slots = torch.tensor([3, 1])
    with patch.object(ab, "build_slot_tables", side_effect=build):
        prepare(backend, slots)
        prepare(backend, slots, reuse=True)
        assert len(builds) == 1
        prepare(backend, torch.tensor([3, 2]), reuse=True)
        prepare(backend, torch.tensor([3, 2]))
        assert len(builds) == 3
        # Bounded tables are reused only for a bound they cover.
        prepare(backend, slots, kv_len_bound=512)
        prepare(backend, slots, reuse=True, kv_len_bound=512)
        assert len(builds) == 4
        prepare(backend, slots, reuse=True, kv_len_bound=513)
    assert len(builds) == 5


@pytest.mark.parametrize("mode", [ForwardMode.DECODE, ForwardMode.EXTEND])
def test_relabelled_idle_batch_has_no_query_rows(mode):
    # DP synchronization may represent an idle rank by a physical decode or
    # extend request in reserved pool slot 0.
    fb = _extend_batch([0], [6], [0], num_tokens=6)
    fb.out_cache_loc.zero_()
    fb.forward_mode = mode
    md = _metadata(fb)
    assert md.query_start_loc.tolist() == [0, 0]
    assert not md.valid.any()


@pytest.mark.parametrize("live", [0, 2])
@pytest.mark.parametrize("width", [5, 6])
@pytest.mark.parametrize("device_rows", [False, True])
def test_dspark_bucket_padding_has_no_query_rows(live, width, device_rows):
    # Both workers build the forward at the physical bucket, including when
    # another DP rank is busy and the local rank has no requests.
    slots = [1, 2][:live] + [0] * (4 - live)
    fb = _verify_batch([8, 40][:live] + [0] * (4 - live), slots, rows=width)
    fb.req_pool_indices_cpu = fb.req_pool_indices.clone()
    fb.out_cache_loc[live * width :] = 0
    fb.seq_lens_cpu[live:] = 0
    setting = SimpleNamespace(get=lambda: device_rows)
    with patch.object(tpu_envs, "SGLANG_TPU_SPEC_DEVICE_ROWS", setting):
        md = ab.build_forward_metadata(
            fb,
            req_to_token=_req_to_token(),
            full_to_swa_index_mapping=None,
            request_window=None,
            swa_ring_valid_from=torch.zeros(SLOTS, dtype=torch.int64),
            max_context_len=MAX_CONTEXT,
            low_ratios=ab.LOW_RATIOS,
            device=torch.device("cpu"),
        )
    expected = torch.arange(5).clamp_max(live) * width
    assert md.query_start_loc.tolist() == expected.tolist()
    assert md.valid.tolist() == [True] * (live * width) + [False] * ((4 - live) * width)
    assert md.token_rows[live * width :].eq(-1).all()


@pytest.mark.parametrize("layout", ["plain", "draft_ring", "request_window"])
def test_device_rows_match_the_host_rows(layout):
    """SGLANG_TPU_SPEC_DEVICE_ROWS builds a spec forward's position-dependent rows
    on the device: every field, dtype and the window plan must equal the host build."""
    seq_lens, slots = [4096, 98310, 17, 700], [1, 2, 3, 2]

    def batch():
        fb = _verify_batch(seq_lens, slots)
        fb.req_pool_indices_cpu = torch.tensor(slots, dtype=torch.int64)
        return fb

    kw = dict(
        req_to_token=_req_to_token(),
        full_to_swa_index_mapping=None,
        request_window=None,
        swa_ring_valid_from=None,
        max_context_len=MAX_CONTEXT,
        low_ratios=ab.LOW_RATIOS,
        device=torch.device("cpu"),
    )
    if layout == "draft_ring":
        kw["swa_ring_valid_from"] = torch.arange(SLOTS, dtype=torch.int64) * 1000
    if layout == "request_window":
        kw["request_window"] = SimpleNamespace(
            workspace_first_page=4 + 4 * SLOTS, workspace_pages=64
        )
    on = SimpleNamespace(get=lambda: True)
    with patch.object(tpu_envs, "SGLANG_TPU_SPEC_DEVICE_ROWS", on):
        # Without host slots the forward takes the host path (same plan dtypes).
        reference = ab.build_forward_metadata(_verify_batch(seq_lens, slots), **kw)
        with patch.object(ab, "_as_cpu_int64", side_effect=AssertionError("read")):
            built = ab.build_forward_metadata(batch(), **kw)
    _assert_same(built, reference)


@pytest.mark.parametrize("layout", ["draft_ring", "request_window"])
def test_device_rows_compute_no_int64_on_the_device(layout):
    """TPU vector units do not lower 64-bit integers: an int64 op in the device-built
    rows (an implicit cumsum promotion, an int64 cast) aborted the server at the
    verify warmup. No op on the device may produce an int64 tensor."""
    from torch.utils._python_dispatch import TorchDispatchMode

    class Int64OnDevice(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.ops = []

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            out = func(*args, **(kwargs or {}))
            for t in out if isinstance(out, (tuple, list)) else (out,):
                if (
                    isinstance(t, torch.Tensor)
                    and t.device.type == "meta"
                    and t.dtype == torch.int64
                ):
                    self.ops.append(str(func))
            return out

    meta = torch.device("meta")
    fb = _verify_batch([4096, 98310, 17], [1, 2, 3])
    fb.req_pool_indices_cpu = torch.tensor([1, 2, 3], dtype=torch.int64)
    fb.positions = fb.positions.to(device=meta, dtype=torch.int32)
    fb.out_cache_loc = fb.out_cache_loc.to(meta)
    kw = dict(
        req_to_token=_req_to_token(),
        full_to_swa_index_mapping=None,
        request_window=None,
        swa_ring_valid_from=None,
        max_context_len=MAX_CONTEXT,
        low_ratios=ab.LOW_RATIOS,
        device=meta,
    )
    if layout == "draft_ring":
        kw["swa_ring_valid_from"] = torch.zeros(SLOTS, dtype=torch.int64)
    else:
        kw["request_window"] = SimpleNamespace(
            workspace_first_page=4 + 4 * SLOTS, workspace_pages=64
        )
    on = SimpleNamespace(get=lambda: True)
    with patch.object(tpu_envs, "SGLANG_TPU_SPEC_DEVICE_ROWS", on):
        with Int64OnDevice() as mode:
            md = ab.build_forward_metadata(fb, **kw)
    assert md.positions.device == meta
    assert mode.ops == []


@pytest.mark.parametrize("packed_rows", [False, True])
def test_device_rows_feed_the_graph_the_same_input_form_as_host_rows(packed_rows):
    """Active ranks build a spec forward's rows on the device, idle ranks on the
    host, and both run the same compiled forward. TorchTPU compiles a view input
    as its whole base buffer, so a packed host field or an aliased one failed
    the idle rank's verify (RuntimeProgramInputMismatch): with device rows, every
    row field must be its own tensor in both builds, also when the device build
    uploads its host rows as one packed buffer."""
    slot_fields = {"full_pages", "swa_pages"}
    seq_lens, slots = [4096, 98310, 17], [1, 2, 3]
    kw = dict(
        req_to_token=_req_to_token(),
        full_to_swa_index_mapping=None,
        request_window=SimpleNamespace(
            workspace_first_page=4 + 4 * SLOTS, workspace_pages=64
        ),
        swa_ring_valid_from=None,
        max_context_len=MAX_CONTEXT,
        low_ratios=ab.LOW_RATIOS,
        device=torch.device("cpu"),
    )
    on = SimpleNamespace(get=lambda: True)
    device_fb = _verify_batch(seq_lens, slots)
    device_fb.req_pool_indices_cpu = torch.tensor(slots, dtype=torch.int64)
    flag = SimpleNamespace(get=lambda: packed_rows)
    with (
        patch.object(tpu_envs, "SGLANG_TPU_SPEC_DEVICE_ROWS", on),
        patch.object(tpu_envs, "SGLANG_TPU_DSV41_PACKED_METADATA", on),
        patch.object(tpu_envs, "SGLANG_TPU_DSV41_PACKED_ROW_UPLOAD", flag),
        patch.object(tpu_envs, "SGLANG_TPU_DSV41_VERIFY_SWA_RING", flag),
    ):
        builds = {"host": _verify_batch(seq_lens, slots), "device": device_fb}
        for name, fb in builds.items():
            md = ab.build_forward_metadata(fb, **kw)
            rows = {
                k: t
                for k, t in _flatten(md).items()
                if k not in slot_fields
                and "index_pages" not in k
                and not k.endswith(".2")  # LowRatioMetadata.index_pages
            }
            batch_ptrs = {fb.positions.data_ptr(), fb.out_cache_loc.data_ptr()}
            ptrs = [t.untyped_storage().data_ptr() for t in rows.values()]
            assert all(t._base is None for t in rows.values()), name
            if name == "device":
                # Host uploads are separate device buffers on TPU; on CPU,
                # .to("cpu") hands back the host tensor itself.
                assert len(set(ptrs)) == len(ptrs)
                assert not set(ptrs) & batch_ptrs


def test_device_ring_table_matches_the_uploaded_one():
    """The verify ring SWA table built on the device equals the host-built
    upload: positions across ring wraps and up to the table end, and a padding
    slot 0 row (an idle DP rank's dummy request)."""
    kw = dict(
        req_to_token=_req_to_token(),
        full_to_swa_index_mapping=None,
        request_window=SimpleNamespace(
            workspace_first_page=4 + 4 * SLOTS, workspace_pages=64
        ),
        swa_ring_valid_from=None,
        max_context_len=4096,
        low_ratios=ab.LOW_RATIOS,
        device=torch.device("cpu"),
    )
    on = SimpleNamespace(get=lambda: True)
    seq_lens, slots = [4084, 250, 17, 0], [1, 7, 3, 0]
    builds = {}
    for device_table in (False, True):
        flag = SimpleNamespace(get=lambda v=device_table: v)
        fb = _verify_batch(seq_lens, slots)
        fb.req_pool_indices_cpu = torch.tensor(slots, dtype=torch.int64)
        with (
            patch.object(tpu_envs, "SGLANG_TPU_SPEC_DEVICE_ROWS", on),
            patch.object(tpu_envs, "SGLANG_TPU_DSV41_VERIFY_SWA_RING", on),
            patch.object(tpu_envs, "SGLANG_TPU_DSV41_DEVICE_RING_TABLE", flag),
        ):
            builds[device_table] = ab.build_forward_metadata(fb, **kw)
    assert builds[True].swa_window_plan is None
    _assert_same(builds[False], builds[True])


def test_bounded_slot_tables_keep_every_readable_page():
    """Slot tables gathered up to a KV-length bound keep their shape and every
    page column below the bound (plus one page); past it they hold the null
    page. A forward that needs more positions than were gathered gets full
    tables (or, in check mode, an error), never a short table."""
    slots = torch.tensor([1, 2, 3])
    kw = dict(
        req_to_token=_req_to_token(),
        req_pool_indices=slots,
        full_to_swa_index_mapping=None,
        request_window=SimpleNamespace(
            workspace_first_page=4 + 4 * SLOTS, workspace_pages=64
        ),
        swa_ring_valid_from=None,
        max_context_len=MAX_CONTEXT,
        low_ratios=ab.LOW_RATIOS,
        device=torch.device("cpu"),
    )
    on = SimpleNamespace(get=lambda: True)
    full = ab.build_slot_tables(**kw)
    with patch.object(tpu_envs, "SGLANG_TPU_DSV41_BOUNDED_SLOT_TABLES", on):
        bounded = ab.build_slot_tables(**kw, kv_len_bound=1000)
    gathered = bounded.bound_tokens // 256
    assert gathered == 5 and full.bound_tokens is None
    assert bounded.full_pages.shape == full.full_pages.shape
    assert torch.equal(bounded.full_pages[:, :gathered], full.full_pages[:, :gathered])
    assert not bounded.full_pages[:, gathered:].any()
    for ratio, pages in full.index_pages.items():
        cols = gathered * 256 // ratio // 128
        assert torch.equal(bounded.index_pages[ratio][:, :cols], pages[:, :cols])
    match = dict(req_pool_indices=slots, columns=full.full_pages.shape[1])
    match["low_ratios"] = ab.LOW_RATIOS
    assert ab._matching_slot_tables(bounded, **match, kv_len_bound=1280) is bounded
    assert ab._matching_slot_tables(bounded, **match, kv_len_bound=1281) is None
    assert ab._matching_slot_tables(bounded, **match, kv_len_bound=None) is None
    with patch.object(tpu_envs, "SGLANG_TPU_DSV41_BOUNDED_SLOT_TABLES_CHECK", on):
        with pytest.raises(AssertionError, match="gathered for 1280 positions"):
            ab._matching_slot_tables(bounded, **match, kv_len_bound=1281)
    # A bound past the context gathers every column: the full table, no bound.
    with patch.object(tpu_envs, "SGLANG_TPU_DSV41_BOUNDED_SLOT_TABLES", on):
        near = ab.build_slot_tables(**kw, kv_len_bound=MAX_CONTEXT + 3)
    assert near.bound_tokens is None
    assert torch.equal(near.full_pages, full.full_pages)


def test_device_rows_check_accepts_bounded_slot_tables():
    """The device-rows check builds its host reference on equally bounded tables:
    slots whose req_to_token rows still map pages past the bound (a reused slot)
    are not a mismatch, since those columns are never read."""
    slots = torch.tensor([1, 2, 3])
    fb = _verify_batch([4096, 98310, 17], [1, 2, 3])
    fb.req_pool_indices_cpu = slots.clone()
    kw = dict(
        req_to_token=_req_to_token(),
        full_to_swa_index_mapping=None,
        request_window=None,
        swa_ring_valid_from=None,
        max_context_len=MAX_CONTEXT,
        low_ratios=ab.LOW_RATIOS,
        device=torch.device("cpu"),
    )
    on = SimpleNamespace(get=lambda: True)
    with (
        patch.object(tpu_envs, "SGLANG_TPU_SPEC_DEVICE_ROWS", on),
        patch.object(tpu_envs, "SGLANG_TPU_SPEC_DEVICE_ROWS_CHECK", on),
        patch.object(tpu_envs, "SGLANG_TPU_DSV41_BOUNDED_SLOT_TABLES", on),
    ):
        bounded = ab.build_slot_tables(
            **kw, req_pool_indices=slots, kv_len_bound=98310 + 12
        )
        assert bounded.bound_tokens is not None
        ab.build_forward_metadata(fb, **kw, slot_tables=bounded)


def test_slot_table_bound_of_an_idle_dp_rank():
    """An idle DP rank's spec step has no requests: the slot-table KV bound must
    still be defined (it crashed every rank's scheduler on an empty max)."""
    from sglang_tpu.experimental.speculative.base_spec_worker import TpuSpecWorker

    bound = TpuSpecWorker._slot_kv_len_bound
    worker = SimpleNamespace(verify_window_rows=6)
    empty = SimpleNamespace(seq_lens_cpu=torch.empty(0, dtype=torch.int64))
    assert bound(worker, empty) == 12
    assert bound(worker, SimpleNamespace(seq_lens_cpu=torch.tensor([5, 300]))) == 312


def test_device_window_declines_rows_that_may_pass_the_swa_table():
    """The host window builder raises on a row past the SWA table; the device
    build cannot, and such a row would alias the table's last page (two writes
    to one slot). When the batch's KV-length bound may pass the table, the
    forward takes the host build."""
    seq_lens, slots = [4096, 17], [1, 2]
    fb = _verify_batch(seq_lens, slots)
    fb.req_pool_indices_cpu = torch.tensor(slots, dtype=torch.int64)
    fb.seq_lens_cpu = torch.tensor([MAX_CONTEXT + 1, 23])
    kw = dict(
        req_to_token=_req_to_token(),
        full_to_swa_index_mapping=None,
        request_window=SimpleNamespace(
            workspace_first_page=4 + 4 * SLOTS, workspace_pages=64
        ),
        swa_ring_valid_from=None,
        max_context_len=MAX_CONTEXT,
        low_ratios=ab.LOW_RATIOS,
        device=torch.device("cpu"),
    )
    on = SimpleNamespace(get=lambda: True)
    with (
        patch.object(tpu_envs, "SGLANG_TPU_SPEC_DEVICE_ROWS", on),
        patch.object(ab, "_as_cpu_int64", wraps=ab._as_cpu_int64) as host_read,
    ):
        ab.build_forward_metadata(fb, **kw)
    assert host_read.called


def test_ring_verify_keeps_the_swa_table_bound():
    """A verify near the context limit has rows past the SWA table; on the ring
    their position would alias an earlier visible one. With
    SGLANG_TPU_DSV41_VERIFY_SWA_RING the build still refuses them, as the
    workspace build does, while an in-table verify attends the ring."""
    kw = dict(
        req_to_token=_req_to_token(),
        full_to_swa_index_mapping=None,
        request_window=SimpleNamespace(
            workspace_first_page=4 + 4 * SLOTS, workspace_pages=64
        ),
        swa_ring_valid_from=None,
        max_context_len=4096,
        low_ratios=ab.LOW_RATIOS,
        device=torch.device("cpu"),
    )
    on = SimpleNamespace(get=lambda: True)
    with patch.object(tpu_envs, "SGLANG_TPU_DSV41_VERIFY_SWA_RING", on):
        meta = ab.build_forward_metadata(_verify_batch([100, 17], [1, 2]), **kw)
        assert meta.swa_window_plan is None
        with pytest.raises(RuntimeError, match="past the SWA table"):
            ab.build_forward_metadata(_verify_batch([4093, 17], [1, 2]), **kw)


def test_device_rows_check_mode_catches_a_wrong_field():
    """SGLANG_TPU_SPEC_DEVICE_ROWS_CHECK passes matching device rows and raises,
    naming the field, when a device-built field differs from the host build."""
    from sglang_tpu.models.deepseek_v41 import device_rows

    def batch():
        fb = _verify_batch([4096, 98310, 17], [1, 2, 3])
        fb.req_pool_indices_cpu = torch.tensor([1, 2, 3], dtype=torch.int64)
        return fb

    kw = dict(
        req_to_token=_req_to_token(),
        full_to_swa_index_mapping=None,
        request_window=None,
        swa_ring_valid_from=None,
        max_context_len=MAX_CONTEXT,
        low_ratios=ab.LOW_RATIOS,
        device=torch.device("cpu"),
    )
    on = SimpleNamespace(get=lambda: True)
    real = device_rows.low_ratio_rows

    def off_by_one(**kwargs):
        out_loc, lengths = real(**kwargs)
        return out_loc, lengths + 1

    with (
        patch.object(tpu_envs, "SGLANG_TPU_SPEC_DEVICE_ROWS", on),
        patch.object(tpu_envs, "SGLANG_TPU_SPEC_DEVICE_ROWS_CHECK", on),
    ):
        ab.build_forward_metadata(batch(), **kw)
        with patch.object(device_rows, "low_ratio_rows", side_effect=off_by_one):
            with pytest.raises(AssertionError, match="lengths"):
                ab.build_forward_metadata(batch(), **kw)


def test_device_window_pages_match_the_host_loop():
    """device_rows.window_pages replaces extend_window_pages' per-request page loop
    with cumsum offsets; tables and row moves must match on mixed layouts
    (empty requests, repeated and padding slots, floors, page-boundary starts)."""
    import random

    from sglang_tpu.models.deepseek_v41 import device_rows
    from sglang_tpu.models.deepseek_v41.request_window import extend_window_pages

    rng = random.Random(0)
    for _ in range(200):
        bs, columns = rng.randint(1, 8), rng.choice([64, 256])
        slots = torch.tensor([rng.choice([0, 1, 2, 3, 3, 5, 7]) for _ in range(bs)])
        counts = torch.tensor([rng.choice([0, 1, 6, 8, 16]) for _ in range(bs)])
        first = torch.tensor([rng.randint(0, columns * 64 - 40) for _ in range(bs)])
        floor = torch.tensor([rng.choice([0, max(0, int(f) - 60)]) for f in first])
        args = dict(first=first, counts=counts, minimum=floor, columns=columns)
        pages, plan = extend_window_pages(
            slots=slots, base_page=100, capacity_pages=10**6, **args
        )
        got_pages, got_plan = device_rows.window_pages(
            live=device_rows.window_live(slots, counts),
            slots=slots,
            base_page=100,
            **args,
        )
        assert torch.equal(got_pages, pages)
        for got, want in zip(got_plan, plan):
            assert got.dtype == torch.int32
            assert torch.equal(got, want.to(torch.int32))


def test_device_values_are_not_read():
    fb = BATCHES["multi"]()
    _upload(fb)
    for name in UPLOADED:
        with pytest.raises(Exception):
            ab._as_cpu_int64(getattr(fb, name), 1)  # meta cannot be read back
    _metadata(fb)  # succeeds only through the host copies


def test_rewritten_field_falls_back_to_the_device_value():
    fb = BATCHES["multi"]()
    reference = _metadata(BATCHES["multi"]())
    _upload(fb, device="cpu")
    # A later stage rebinding a field to a different value: the record no longer
    # applies, and the metadata follows the rebound device value.
    fb.req_pool_indices = torch.tensor([3, 6, 1], dtype=torch.int64)
    assert host_input_mirror.host_copy(fb, "req_pool_indices") is None
    assert host_input_mirror.host_copy(fb, "positions") is not None
    ref_fb = BATCHES["multi"]()
    ref_fb.req_pool_indices = torch.tensor([3, 6, 1], dtype=torch.int64)
    _assert_same(_metadata(fb), _metadata(ref_fb))
    assert not torch.equal(
        _flatten(_metadata(ref_fb))["full_pages"], _flatten(reference)["full_pages"]
    )


def test_policy_upload_dispatcher_records_and_metadata_releases():
    """Through DecoderTpuCompilePolicy._upload_host_inputs, as serving runs it."""
    policy = object.__new__(DecoderTpuCompilePolicy)
    policy.model_runner = SimpleNamespace(device=torch.device("meta"))
    fb = BATCHES["long_prefix"]()
    fb.seq_lens = torch.tensor([98304 + 768])
    reference = _metadata(BATCHES["long_prefix"]())
    policy._upload_host_inputs(fb)
    assert fb.positions.device.type == "meta"
    for name in UPLOADED:
        assert host_input_mirror.host_copy(fb, name) is not None
    _assert_same(_metadata(fb), reference)
    # The record is released once the metadata has read it.
    assert host_input_mirror.host_copy(fb, "positions") is None


def test_record_belongs_to_one_batch():
    a, b = BATCHES["single"](), BATCHES["multi"]()
    _upload(a, device="cpu")
    assert host_input_mirror.host_copy(b, "positions") is None
    _upload(b, device="cpu")
    assert host_input_mirror.host_copy(a, "positions") is None
    assert host_input_mirror.host_copy(b, "positions") is not None


def test_unreferenceable_batch_is_not_recorded():
    fb = SimpleNamespace(**vars(BATCHES["single"]()))
    _upload(fb, device="cpu")
    assert host_input_mirror.host_copy(fb, "positions") is None


def test_only_mirrored_cpu_fields_are_recorded():
    fb = BATCHES["single"]()
    host_input_mirror.record_upload(fb, "input_ids", fb.positions, fb.positions)
    assert host_input_mirror.host_copy(fb, "input_ids") is None


def test_stale_length_lists_do_not_matter():
    # A missing prefix list zero-filled by padding must not change
    # positions; the metadata no longer reads the length lists at all.
    fb = BATCHES["long_prefix"]()
    reference = _metadata(BATCHES["long_prefix"]())
    fb.extend_prefix_lens_cpu = [0] * len(fb.extend_prefix_lens_cpu)
    _upload(fb)
    _assert_same(_metadata(fb), reference)


def test_check_mode_counts_matches():
    ab._HOST_CHECKS.clear()
    fb = BATCHES["multi"]()
    _upload(fb, device="cpu")
    with patch.object(
        tpu_envs,
        "SGLANG_TPU_DSV41_CHECK_HOST_METADATA",
        SimpleNamespace(get=lambda: True),
    ):
        _metadata(fb)
    assert ab._HOST_CHECKS == {name: 1 for name in UPLOADED}


def test_check_mode_catches_a_mismatch():
    fb = BATCHES["multi"]()
    _upload(fb, device="cpu")
    # Pair a wrong host copy with the batch's current device tensor.
    host_input_mirror.record_upload(
        fb, "req_pool_indices", torch.tensor([7, 7, 7]), fb.req_pool_indices
    )
    with patch.object(
        tpu_envs,
        "SGLANG_TPU_DSV41_CHECK_HOST_METADATA",
        SimpleNamespace(get=lambda: True),
    ):
        with pytest.raises(AssertionError, match="req_pool_indices"):
            _metadata(fb)


# ----------------------------------------------------------------------------
# Spec verify / draft rows: the TPU spec worker's host twin of select_verify_rows.

VERIFY_ROWS, CAPACITY = 6, 6


def _verify_plan(seed=0, *, live=5, padded=8):
    """A host plan as TpuSpecVerifyInput.from_batch holds it, and the device seq_lens."""
    gen = torch.Generator().manual_seed(seed)
    slots = torch.randperm(SLOTS - 1, generator=gen)[:live] + 1
    committed = torch.randint(64, 3000, (live,), generator=gen)
    columns = committed[:, None] + torch.arange(2 * CAPACITY)
    candidates = _req_to_token()[slots[:, None], columns]
    # One commit past the host base; row 0 trails it (offset clamps at 0) and
    # row 1 is a full capacity past it (offset clamps at CAPACITY).
    seq_lens = torch.zeros(padded, dtype=torch.int64)
    seq_lens[:live] = committed + torch.randint(0, CAPACITY + 1, (live,), generator=gen)
    seq_lens[0] = committed[0] - 3
    seq_lens[1] = committed[1] + CAPACITY + 2
    return SimpleNamespace(
        slots=slots,
        committed=committed,
        candidates=candidates,
        seq_lens=seq_lens,
        req_pool_cpu=F.pad(slots, (0, padded - live)),
    )


def _verify_input(plan):
    return SimpleNamespace(
        verify_cache_candidates_cpu=plan.candidates,
        committed_lens_cpu=plan.committed,
        req_pool_indices_cpu=plan.req_pool_cpu,
    )


def _device_rows(plan):
    """select_verify_rows on the zero-padded device uploads (padded_h2d)."""
    from sglang_tpu.dflash_utils_tpu import select_dflash_verify_cache_locs_tpu

    padded = plan.seq_lens.shape[0]
    candidates = F.pad(plan.candidates, (0, 0, 0, padded - plan.candidates.shape[0]))
    committed = F.pad(plan.committed, (0, padded - plan.committed.shape[0]))
    cache_locs = select_dflash_verify_cache_locs_tpu(
        candidates, plan.seq_lens, committed, VERIFY_ROWS
    )
    positions = plan.seq_lens[:, None] + torch.arange(VERIFY_ROWS)
    return positions.reshape(-1), cache_locs.reshape(-1)


@pytest.mark.parametrize("seed", range(4))
def test_host_verify_rows_match_select_verify_rows(seed):
    from sglang_tpu.experimental.speculative.host_verify_rows import host_verify_rows

    plan = _verify_plan(seed)
    host = host_verify_rows(
        seq_lens=plan.seq_lens,
        candidates_cpu=plan.candidates,
        committed_lens_cpu=plan.committed,
        req_pool_indices_cpu=plan.req_pool_cpu,
        num_rows=VERIFY_ROWS,
    )
    positions, cache_locs = _device_rows(plan)
    assert host.positions.dtype == host.out_cache_loc.dtype == torch.int64
    assert torch.equal(host.positions, positions)
    assert torch.equal(host.out_cache_loc, cache_locs)
    assert torch.equal(host.req_pool_indices, plan.req_pool_cpu)


class _CountingTensor(torch.Tensor):
    reads = 0

    def to(self, *args, **kwargs):
        type(self).reads += 1
        return torch.Tensor.to(self.as_subclass(torch.Tensor), *args, **kwargs)


def _step_rows(plan, **kwargs):
    from sglang_tpu.experimental.speculative.host_verify_rows import StepHostRows

    return StepHostRows(
        seq_lens=plan.seq_lens.as_subclass(_CountingTensor),
        verify_input=_verify_input(plan),
        num_rows=VERIFY_ROWS,
        **kwargs,
    )


def test_step_host_rows_read_seq_lens_once_for_draft_and_verify():
    """The draft block runs the first width verify rows (spec_ops._block_inputs);
    one seq_lens read must serve both forwards."""
    plan = _verify_plan()
    _CountingTensor.reads = 0
    step = _step_rows(plan)
    width = 4
    draft = step.draft_rows(width)
    verify = step.verify_rows()
    assert _CountingTensor.reads == 1
    positions, cache_locs = _device_rows(plan)
    padded = plan.seq_lens.shape[0]
    assert torch.equal(verify.positions, positions)
    assert torch.equal(verify.out_cache_loc, cache_locs)
    assert torch.equal(
        draft.positions, positions.view(padded, -1)[:, :width].reshape(-1)
    )
    assert torch.equal(
        draft.out_cache_loc, cache_locs.view(padded, -1)[:, :width].reshape(-1)
    )
    assert torch.equal(draft.req_pool_indices, plan.req_pool_cpu)


def test_step_host_rows_skip_the_read_given_host_seq_lens():
    plan = _verify_plan()
    _CountingTensor.reads = 0
    step = _step_rows(plan, seq_lens_cpu=plan.seq_lens.clone())
    assert torch.equal(step.verify_rows().positions, _device_rows(plan)[0])
    assert _CountingTensor.reads == 0


def test_step_host_rows_none_without_token_slots():
    plan = _verify_plan()
    plan.candidates = None
    assert _step_rows(plan).verify_rows() is None
    assert _step_rows(plan).draft_rows(4) is None


def _spec_verify_batch(plan, device="cpu"):
    positions, cache_locs = _device_rows(plan)
    return FB(
        forward_mode=ForwardMode.TARGET_VERIFY,
        batch_size=plan.seq_lens.shape[0],
        positions=positions.to(device),
        out_cache_loc=cache_locs.to(torch.int32).to(device),
        extend_seq_lens=None,
        extend_seq_lens_cpu=None,
        extend_prefix_lens_cpu=None,
        req_pool_indices=plan.req_pool_cpu.to(device),
        req_pool_indices_cpu=None,
        encoder_swa_row_floor=None,
        spec_info=SimpleNamespace(draft_token_num=VERIFY_ROWS),
        **{ab._REAL_B_KEY: plan.slots.shape[0]},
    )


@pytest.mark.parametrize("seed", range(3))
def test_recorded_verify_rows_replace_every_device_read(seed):
    """The verify forward's fields are device tensors built before the draft was
    dispatched; with the host twin recorded the metadata reads none of them."""
    from sglang_tpu.experimental.speculative.host_verify_rows import record_host_rows

    plan = _verify_plan(seed)
    reference = _metadata(_spec_verify_batch(plan))
    fb = _spec_verify_batch(plan, device="meta")  # a device read would raise
    record_host_rows(fb, _step_rows(plan, seq_lens_cpu=plan.seq_lens).verify_rows())
    _assert_same(_metadata(fb), reference)


def _dp_pad(fb, num_tokens, bs):
    """What ForwardBatch._pad_inputs_to_size does to the mirrored fields: torch.cat
    with a zero tail, a new tensor even when nothing is appended."""
    for name, size in (("positions", num_tokens), ("out_cache_loc", num_tokens)):
        t = getattr(fb, name)
        setattr(fb, name, torch.cat([t, t.new_zeros(size - t.shape[0])]))
    t = fb.req_pool_indices
    fb.req_pool_indices = torch.cat([t, t.new_zeros(bs - t.shape[0])])


@pytest.mark.parametrize("extra_rows", [0, 2])
def test_records_survive_the_dp_padding(extra_rows):
    """The DP MLP sync re-pads the mirrored fields before the metadata is built;
    without carrying the records every host copy fell back to a device read."""
    from sglang_tpu.experimental.speculative.host_verify_rows import record_host_rows
    from sglang_tpu.models.deepseek_v41.hooks import tpu_dsv41_carry_mirrored_padding

    plan = _verify_plan()
    reference_fb = _spec_verify_batch(plan)
    fb = _spec_verify_batch(plan, device="meta")
    record_host_rows(fb, _step_rows(plan, seq_lens_cpu=plan.seq_lens).verify_rows())
    num_tokens = fb.positions.shape[0] + extra_rows * VERIFY_ROWS
    bs = fb.req_pool_indices.shape[0] + extra_rows
    tpu_dsv41_carry_mirrored_padding(_dp_pad, fb, num_tokens, bs)
    _dp_pad(reference_fb, num_tokens, bs)
    for name in UPLOADED:
        assert host_input_mirror.host_copy(fb, name) is not None, name
    reference_fb.batch_size = fb.batch_size = bs
    _assert_same(_metadata(fb), _metadata(reference_fb))


def test_engram_commits_flush_only_where_the_length_read_synced():
    """The early Engram flush may run only after a real length read: on an idle DP
    rank (host lengths, no read) it would add a blocking device read."""
    from unittest.mock import MagicMock

    from sglang_tpu.models.deepseek_v41 import engram_host
    from sglang_tpu.models.deepseek_v41.hooks import (
        tpu_dsv41_flush_engram_at_length_read,
    )

    rows = object()
    for seq_lens_cpu, flushed in ((None, True), (torch.zeros(2), False)):
        step = SimpleNamespace(_seq_lens_cpu=seq_lens_cpu)
        with patch.object(engram_host, "flush_all_engram_commits", MagicMock()) as f:
            out = tpu_dsv41_flush_engram_at_length_read(lambda s: rows, step)
        assert out is rows
        assert f.called == flushed


def test_spec_worker_builds_host_rows_only_for_a_consumer():
    from sglang_tpu.experimental.speculative.block_spec_worker import (
        TpuBlockSpecWorker,
    )

    worker = object.__new__(TpuBlockSpecWorker)
    worker.num_verify_rows = VERIFY_ROWS
    plan = _verify_plan()
    kwargs = dict(seq_lens=plan.seq_lens, verify_input=_verify_input(plan))
    device_rows = SimpleNamespace(get=lambda: False)
    with patch.object(tpu_envs, "SGLANG_TPU_SPEC_DEVICE_ROWS", device_rows):
        with patch.object(host_input_mirror, "_consumers", 0):
            assert worker._new_step_host_rows(**kwargs) is None
        with patch.object(host_input_mirror, "_consumers", 1):
            assert worker._new_step_host_rows(**kwargs) is not None
            # Device-built rows skip the length read; host lengths (idle ranks)
            # still give rows.
            device_rows.get = lambda: True
            assert worker._new_step_host_rows(**kwargs) is None
            idle = torch.zeros(plan.seq_lens.shape, dtype=torch.int64)
            assert worker._new_step_host_rows(**kwargs, seq_lens_cpu=idle) is not None


def _idle_backend():
    backend = object.__new__(ab.DSV41TPUAttnBackend)
    backend._slot_tables = None
    backend._idle_metadata_inputs = {}
    built = []

    def build(forward_batch, *, slot_tables):
        rows = forward_batch.positions.shape[0]
        none = ("engram", "swa_window_plan", "tail", "tail_gather", "tail_inverse")
        fields = {
            name: torch.zeros(rows, dtype=torch.int32)
            for name in ab._METADATA_INPUT_FIELDS
            if name not in none
        }
        fields["low_ratio"] = {}
        metadata = ab.DSV41TPUForwardMetadata(**fields)
        built.append(metadata)
        return metadata

    backend._build_forward_metadata = build
    return backend, built


def _idle_verify_batch(pbs, rows_per_req=4):
    return FB(
        forward_mode=ForwardMode.TARGET_VERIFY,
        batch_size=pbs,
        positions=torch.zeros(pbs * rows_per_req, dtype=torch.int64),
    )


def test_idle_step_metadata_reuses_inputs_in_a_fresh_object():
    """An idle DP rank's step (SGLANG_TPU_SPEC_IDLE_INPUT_CACHE) reuses the input
    leaves built for its shape, but in a new metadata object: layers fill scratch
    fields on it, which must not leak into the next forward."""
    from sglang_tpu.experimental.speculative.idle_step import idle_step

    backend, built = _idle_backend()
    with idle_step():
        backend.init_forward_metadata(_idle_verify_batch(4))
        first = backend.forward_metadata
        first.selected[4] = torch.ones(1)
        first.engram = ("rows",)
        backend.init_forward_metadata(_idle_verify_batch(4))
        second = backend.forward_metadata
        backend.init_forward_metadata(_idle_verify_batch(8))
    assert len(built) == 2  # one per shape
    assert second is not first
    assert second.selected == {} and second.engram is None
    for name in ab._METADATA_INPUT_FIELDS:
        if name != "engram":
            assert getattr(second, name) is getattr(first, name), name
    # Outside an idle step every forward builds its own.
    backend.init_forward_metadata(_idle_verify_batch(4))
    assert len(built) == 3


def test_idle_step_metadata_with_tail_is_rebuilt():
    """Decoder-replay (tail) metadata is per forward; it is never reused."""
    from sglang_tpu.experimental.speculative.idle_step import idle_step

    backend, built = _idle_backend()
    build = backend._build_forward_metadata

    def build_with_tail(forward_batch, *, slot_tables):
        metadata = build(forward_batch, slot_tables=slot_tables)
        metadata.tail = object()
        return metadata

    backend._build_forward_metadata = build_with_tail
    with idle_step():
        backend.init_forward_metadata(_idle_verify_batch(4))
        backend.init_forward_metadata(_idle_verify_batch(4))
    assert len(built) == 2


def test_idle_step_never_reuses_metadata_of_a_forward_with_prompt_rows():
    """A fused mixed prefill + verify step runs its MIXED target forward inside the
    idle spec step on a prompt-only rank; that forward holds real prompt rows, so a
    same-shaped later one must not get its positions, slots or KV locations."""
    from sglang_tpu.experimental.speculative.idle_step import idle_step

    backend, built = _idle_backend()
    for mode in (ForwardMode.MIXED, ForwardMode.EXTEND):
        batch = FB(forward_mode=mode, batch_size=4, positions=torch.zeros(16))
        with idle_step():
            backend.init_forward_metadata(batch)
            backend.init_forward_metadata(batch)
    assert len(built) == 4
    assert backend._idle_metadata_inputs == {}


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
