"""CPU coverage for STRADDLE-PIN in the VLM host tier (SGLANG_VLM_HOST_TIER_PIN_PENDING).

While an encode's host write-back is in flight, a lookup that misses both the
device cache and the host LRU is served the encode's own embedding object (the
one the write-back job holds) instead of encoding the item again. The pin is
dropped when the write-back completes or fails. No TPU: a fake device module
stands in for the worker's stream/event, and a manual pool decides when
write-backs land.

The last section checks that the server tier is built only when one process
is the whole engine (tp_size == 1 and pp_size == 1).
"""

import contextlib
import hashlib

import pytest
import torch

from sglang.srt.managers import mm_schedule
from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu import mm_schedule_tpu
from sglang_tpu.mm_embedding import PaddedEmbedding

register_cpu_ci(est_time=8, suite="stage-a-test-cpu-intel")

HIDDEN = 8
POISON = (float("nan"), 30000.0, 0.0)


class _FakeDeviceModule:
    class Stream:
        def __init__(self, device=None):
            self.device = device

    class Event:
        def record(self):
            pass

        def synchronize(self):
            pass

    @staticmethod
    def stream(_stream):
        return contextlib.nullcontext()


class _ManualPool:
    """Write-backs land only when the test says so."""

    def __init__(self):
        self.jobs = []

    def submit(self, fn, *args):
        self.jobs.append((fn, args))

    def run(self):
        jobs, self.jobs = self.jobs, []
        for fn, args in jobs:
            fn(*args)


def _host_copy(orig):
    # On CPU, Tensor.to("cpu") returns the tensor itself; copy so the host entry
    # is a distinct buffer, as the TPU D2H is.
    def to_host(emb):
        entry = orig(emb)
        entry.buffers = [b.clone() for b in entry.buffers]
        return entry

    return staticmethod(to_host)


@pytest.fixture
def make_tier(monkeypatch):
    monkeypatch.setattr(torch, "get_device_module", lambda *_: _FakeDeviceModule)
    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 0)
    monkeypatch.setattr(
        torch.accelerator, "device_index", lambda _i: contextlib.nullcontext()
    )
    monkeypatch.setattr(
        mm_schedule_tpu._VlmHostTier,
        "_to_host",
        _host_copy(mm_schedule_tpu._VlmHostTier._to_host),
    )
    made = []

    def make(pin=True, cap_bytes=1 << 20, verify=False):
        tier = mm_schedule_tpu._VlmHostTier(cap_bytes, verify, 1e9, pin_pending=pin)
        tier._pool.shutdown(wait=True)
        tier._pool = _ManualPool()
        made.append(tier)
        return tier

    yield make


def _bits(emb) -> str:
    rows = emb.to_host() if isinstance(emb, PaddedEmbedding) else emb.to("cpu")
    return hashlib.sha256(
        rows.contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def _padded(rows: int, pad: int, seed: int) -> PaddedEmbedding:
    g = torch.Generator().manual_seed(seed)
    buffer = torch.randn(rows + pad, HIDDEN, generator=g).to(torch.bfloat16)
    return PaddedEmbedding.from_buffer(buffer, rows)


def _poison_(emb, value):
    if isinstance(emb, PaddedEmbedding):
        for span in emb.spans:
            span.buffer.fill_(value)
    else:
        emb.fill_(value)


# ---- tier-level: pin hit, release, eviction fallback, poison ----


@pytest.mark.parametrize("kind", ["padded", "tensor"])
def test_pending_write_back_is_served_from_the_pin(make_tier, kind):
    tier = make_tier(pin=True)
    emb = _padded(6, 2, 1) if kind == "padded" else torch.randn(6, HIDDEN)
    tier.write_back(("h", 6), emb)
    hit = tier.lookup(("h", 6), torch.device("cpu"))
    assert hit is not None
    served, entry = hit
    assert served is emb  # the encode's own object: nothing recomputed or copied
    assert entry is None
    assert tier.c["pin_hit"] == 1 and tier.c["host_miss"] == 0
    assert tier.c["host_hit"] == 0


def test_flag_off_keeps_the_v1_miss_while_pending(make_tier):
    tier = make_tier(pin=False)
    tier.write_back(("h", 6), _padded(6, 2, 1))
    assert tier._pending_emb == {}
    assert [fn for fn, _ in tier._pool.jobs] == [tier._copy_to_host]  # the v1 job
    assert tier.lookup(("h", 6), torch.device("cpu")) is None
    assert tier.c["host_miss"] == 1 and tier.c["pin_hit"] == 0


def test_pin_needs_a_host_tier():
    tier = mm_schedule_tpu._VlmHostTier(0, False, 60.0, pin_pending=True)
    assert not tier.pin_pending and not tier.enabled


@pytest.mark.parametrize("env, want", [(None, False), ("0", False), ("1", True)])
def test_server_tier_reads_the_pin_flag(make_tier, monkeypatch, env, want):
    monkeypatch.setenv("SGLANG_VLM_HOST_TIER_MB", "1")
    if env is None:
        monkeypatch.delenv("SGLANG_VLM_HOST_TIER_PIN_PENDING", raising=False)
    else:
        monkeypatch.setenv("SGLANG_VLM_HOST_TIER_PIN_PENDING", env)
    monkeypatch.setattr(mm_schedule_tpu, "_HOST_TIER", None)
    monkeypatch.setattr(mm_schedule_tpu, "_HOST_TIER_INIT", False)
    tier = mm_schedule_tpu._host_tier()
    try:
        assert tier.enabled and tier.pin_pending is want
    finally:
        tier._pool.shutdown(wait=True)


def test_completed_write_back_releases_the_pin_and_serves_the_host_copy(make_tier):
    tier = make_tier(pin=True)
    emb = _padded(6, 2, 1)
    want = _bits(emb)
    tier.write_back(("h", 6), emb)
    tier._pool.run()
    assert tier._pending_emb == {} and tier._pending == set()
    served, entry = tier.lookup(("h", 6), torch.device("cpu"))
    assert entry is not None and served is not emb
    assert _bits(served) == want
    assert tier.c["host_hit"] == 1 and tier.c["pin_hit"] == 0


@pytest.mark.parametrize("value", POISON)
def test_released_pin_is_never_served_after_host_eviction(make_tier, value):
    # Eviction fallback: once the write-back has landed and the host LRU later
    # evicts the entry, a lookup must miss (re-encode), not revive the old pin.
    emb = _padded(6, 2, 1)
    tier = make_tier(pin=True, cap_bytes=emb.storage_nbytes())
    tier.write_back(("a", 6), emb)
    tier._pool.run()
    tier.write_back(("b", 6), _padded(6, 2, 2))
    tier._pool.run()
    assert tier.c["evict"] == 1
    _poison_(emb, value)  # the released buffer may be reused by anything now
    assert tier.lookup(("a", 6), torch.device("cpu")) is None
    assert tier.c["host_miss"] == 1 and tier.c["pin_hit"] == 0


@pytest.mark.parametrize("value", POISON)
def test_failed_write_back_releases_the_pin(make_tier, monkeypatch, value):
    tier = make_tier(pin=True)
    emb = _padded(6, 2, 1)

    def boom(_emb):
        raise RuntimeError("D2H failed")

    monkeypatch.setattr(mm_schedule_tpu._VlmHostTier, "_to_host", staticmethod(boom))
    tier.write_back(("h", 6), emb)
    tier._pool.run()
    assert tier.c["wb_error"] == 1
    assert tier._pending_emb == {} and tier._pending == set()
    _poison_(emb, value)
    assert tier.lookup(("h", 6), torch.device("cpu")) is None


def test_pin_is_dropped_even_if_the_job_raises(make_tier, monkeypatch):
    tier = make_tier(pin=True)

    def escapes(*_):
        raise RuntimeError("unexpected")

    monkeypatch.setattr(tier, "_copy_to_host", escapes)
    tier.write_back(("h", 6), _padded(6, 2, 1))
    with pytest.raises(RuntimeError):
        tier._pool.run()
    assert tier._pending_emb == {}
    assert tier.lookup(("h", 6), torch.device("cpu")) is None


def test_a_landed_entry_wins_over_the_pin(make_tier, monkeypatch):
    # Between the LRU insert and the pin drop both exist: the host copy is served,
    # so the flag only ever replaces encodes, never a v1 host hit.
    tier = make_tier(pin=True)
    emb = _padded(6, 2, 1)
    seen = []
    copy = tier._copy_to_host

    def copy_then_lookup(key, e, t_sub):
        copy(key, e, t_sub)
        assert key in tier._lru and key in tier._pending_emb
        seen.append(tier.lookup(key, torch.device("cpu")))

    monkeypatch.setattr(tier, "_copy_to_host", copy_then_lookup)
    tier.write_back(("h", 6), emb)
    tier._pool.run()
    ((served, entry),) = seen
    assert entry is not None and served is not emb
    assert tier.c["host_hit"] == 1 and tier.c["pin_hit"] == 0


def test_too_big_write_back_releases_the_pin(make_tier):
    emb = _padded(6, 2, 1)
    tier = make_tier(pin=True, cap_bytes=emb.storage_nbytes() - 1)
    tier.write_back(("h", 6), emb)
    tier._pool.run()
    assert tier.c["wb_too_big"] == 1
    assert tier._pending_emb == {}
    assert tier.lookup(("h", 6), torch.device("cpu")) is None


@pytest.mark.parametrize("value", POISON)
def test_host_copy_is_served_once_landed_even_if_the_device_buffer_changes(
    make_tier, value
):
    tier = make_tier(pin=True)
    emb = _padded(6, 2, 1)
    want = _bits(emb)
    tier.write_back(("h", 6), emb)
    tier._pool.run()
    _poison_(emb, value)
    served, entry = tier.lookup(("h", 6), torch.device("cpu"))
    assert entry is not None
    assert _bits(served) == want


def test_resubmitted_key_keeps_the_first_pin(make_tier):
    tier = make_tier(pin=True)
    first, second = _padded(6, 2, 1), _padded(6, 2, 1)
    tier.write_back(("h", 6), first)
    tier.write_back(("h", 6), second)
    assert tier.c["wb_skip_present"] == 1
    assert tier.lookup(("h", 6), torch.device("cpu"))[0] is first


def test_verify_pin_compares_the_valid_rows(make_tier):
    tier = make_tier(pin=True, verify=True)
    emb = _padded(6, 2, 1)
    same_rows = PaddedEmbedding.from_buffer(emb.spans[0].buffer.clone(), 6)
    same_rows.spans[0].buffer[6:].fill_(7.0)  # padding rows differ: not compared
    assert tier.verify_pin(("h", 6), same_rows, emb)
    other = _padded(6, 2, 9)
    assert not tier.verify_pin(("h", 6), other, emb)
    assert tier.c["verify_pin_ok"] == 1 and tier.c["verify_pin_bad"] == 1


# ---- scheduler level: the straddle pattern through _get_chunked_prefill_embedding ----


class _Encoder:
    """Deterministic per-item encoder returning one padded buffer per item (as NaViT)."""

    def __init__(self):
        self.encoded = []

    def __call__(self, items):
        out = []
        for item in items:
            self.encoded.append(item.hash)
            rows = item.offsets[0][1] - item.offsets[0][0] + 1
            out.append(_padded(rows, 3, item.hash))
        return out


def _image(h, start, end):
    return MultimodalDataItem(
        modality=Modality.IMAGE, hash=h, offsets=[(start, end)], feature=None
    )


def _step(encoder, requests):
    """One prefill batch: requests = [(items, prefix_len, extend_len)]."""
    items, sizes, prefix, extend, offsets = [], [0], [], [], []
    for req_items, p, e in requests:
        items.extend(req_items)
        sizes.append(sizes[-1] + len(req_items))
        prefix.append(p)
        extend.append(e)
        offsets.append([it.offsets[0] for it in req_items])
    emb, _ = mm_schedule_tpu._get_chunked_prefill_embedding(
        encoder,
        items,
        sizes,
        prefix,
        extend,
        offsets,
        torch.zeros(sum(extend), dtype=torch.long),
    )
    return emb


@pytest.fixture
def scheduler_tier(make_tier, monkeypatch):
    def install(pin, verify=False, device_cache_bytes=None):
        tier = make_tier(pin=pin, cap_bytes=1 << 20, verify=verify)
        monkeypatch.setattr(mm_schedule_tpu, "_HOST_TIER", tier)
        monkeypatch.setattr(mm_schedule_tpu, "_HOST_TIER_INIT", True)
        monkeypatch.setattr(mm_schedule, "embedding_cache", None)
        mm_schedule.init_mm_embedding_cache(device_cache_bytes)
        return tier

    return install


def _straddle_run(install, pin, verify=False):
    """A's image [10, 109] straddles two chunks; B's image evicts it from the
    one-entry device cache in between; after the write-backs land, D evicts it
    again and C reuses A's image from the host copy."""
    rows_a = 100
    one_entry = _padded(rows_a, 3, 0).storage_nbytes()
    tier = install(pin, verify=verify, device_cache_bytes=one_entry)
    enc = _Encoder()
    img_a, img_b = 0xA, 0xB
    out = []
    out.append(_step(enc, [([_image(img_a, 10, 109)], 0, 64)]))  # A chunk 1: encode A
    out.append(_step(enc, [([_image(img_b, 5, 104)], 0, 128)]))  # B: encode B, evicts A
    out.append(_step(enc, [([_image(img_a, 10, 109)], 64, 64)]))  # A chunk 2: straddle
    pending_at_straddle = set(tier._pending)
    tier._pool.run()  # write-backs land
    out.append(_step(enc, [([_image(0xD, 0, 99)], 0, 128)]))  # D: evicts A again
    tier._pool.run()
    out.append(_step(enc, [([_image(img_a, 7, 106)], 0, 128)]))  # C: same image A
    return tier, enc, out, pending_at_straddle


def test_straddle_chunk_is_served_from_the_pin_and_bits_match_flag_off(
    scheduler_tier,
):
    off_tier, off_enc, off_out, off_pending = _straddle_run(scheduler_tier, pin=False)
    on_tier, on_enc, on_out, on_pending = _straddle_run(scheduler_tier, pin=True)

    assert off_pending == on_pending == {(0xA, 100), (0xB, 100)}
    # Flag off: A is encoded again for its second chunk. Flag on: it is not.
    assert off_enc.encoded == [0xA, 0xB, 0xA, 0xD]
    assert on_enc.encoded == [0xA, 0xB, 0xD]
    assert off_tier.c["encodes"] == 4 and on_tier.c["encodes"] == 3
    assert on_tier.c["pin_hit"] == 1 and off_tier.c["pin_hit"] == 0
    assert on_tier.c["host_hit"] == off_tier.c["host_hit"] == 1  # C, after landing
    # The embeddings fed to the LM are byte-identical, chunk for chunk.
    assert [_bits(e) for e in on_out] == [_bits(e) for e in off_out]
    assert on_tier._pending_emb == {}


def test_straddle_chunk_reuses_the_first_encode_buffer(scheduler_tier):
    tier, enc, out, _ = _straddle_run(scheduler_tier, pin=True)
    first, second = out[0], out[2]
    assert isinstance(first, PaddedEmbedding) and isinstance(second, PaddedEmbedding)
    assert first.spans[0].buffer is second.spans[0].buffer
    assert (first.spans[0].start, first.spans[0].length) == (0, 54)
    assert (second.spans[0].start, second.spans[0].length) == (54, 46)


def test_verify_mode_encodes_the_pin_hit_and_serves_the_pin(scheduler_tier):
    tier, enc, out, _ = _straddle_run(scheduler_tier, pin=True, verify=True)
    # Verify mode re-encodes every pin and host hit to compare it with the fresh bits.
    assert enc.encoded == [0xA, 0xB, 0xA, 0xD, 0xA]
    assert tier.c["verify_pin_ok"] == 1 and tier.c["verify_pin_bad"] == 0
    assert tier.c["verify_ok"] == 1 and tier.c["verify_bad"] == 0
    assert out[0].spans[0].buffer is out[2].spans[0].buffer


def test_flag_on_without_pending_write_back_is_unchanged(scheduler_tier):
    # Same stream but every write-back lands before the next step: no pin hits,
    # identical encodes, host hits and bits as flag off.
    def run(pin):
        tier = scheduler_tier(
            pin, device_cache_bytes=_padded(100, 3, 0).storage_nbytes()
        )
        enc = _Encoder()
        out = []
        for req in (
            ([_image(0xA, 10, 109)], 0, 64),
            ([_image(0xB, 5, 104)], 0, 128),
            ([_image(0xA, 10, 109)], 64, 64),
        ):
            out.append(_step(enc, [req]))
            tier._pool.run()
        return tier, enc, out

    off_tier, off_enc, off_out = run(False)
    on_tier, on_enc, on_out = run(True)
    assert on_enc.encoded == off_enc.encoded == [0xA, 0xB]
    assert on_tier.c["pin_hit"] == 0
    assert on_tier.c["host_hit"] == off_tier.c["host_hit"] == 1
    assert [_bits(e) for e in on_out] == [_bits(e) for e in off_out]


# ---- server tier: single-process engines only ----


def _fresh_server_tier(monkeypatch):
    monkeypatch.setenv("SGLANG_VLM_HOST_TIER_MB", "1")
    monkeypatch.setenv("SGLANG_VLM_HOST_TIER_PIN_PENDING", "1")
    monkeypatch.setattr(mm_schedule_tpu, "_HOST_TIER", None)
    monkeypatch.setattr(mm_schedule_tpu, "_HOST_TIER_INIT", False)


@pytest.mark.parametrize("tp, pp", [(2, 1), (8, 1), (1, 2)])
def test_server_tier_stays_off_in_a_multi_process_engine(monkeypatch, tp, pp):
    _fresh_server_tier(monkeypatch)
    with get_parallel().override(tp_size=tp, pp_size=pp):
        assert mm_schedule_tpu._host_tier() is None


def test_server_tier_is_built_in_a_single_process_engine(make_tier, monkeypatch):
    _fresh_server_tier(monkeypatch)
    with get_parallel().override(tp_size=1, pp_size=1):
        tier = mm_schedule_tpu._host_tier()
    try:
        assert tier.enabled and tier.pin_pending
    finally:
        tier._pool.shutdown(wait=True)


def test_multi_process_engine_encodes_every_device_miss(monkeypatch):
    # With a tier, whether A is encoded a second time depends on whether its
    # write-back landed first, and that differs between processes. Without one
    # (multi-process engine) every process encodes [A], [B], [A], as on main.
    _fresh_server_tier(monkeypatch)
    monkeypatch.setattr(mm_schedule, "embedding_cache", None)
    mm_schedule.init_mm_embedding_cache(_padded(100, 3, 0).storage_nbytes())
    enc = _Encoder()
    with get_parallel().override(tp_size=2, pp_size=1):
        for h in (0xA, 0xB, 0xA):
            _step(enc, [([_image(h, 0, 99)], 0, 128)])
    assert mm_schedule_tpu._HOST_TIER is None
    assert enc.encoded == [0xA, 0xB, 0xA]
