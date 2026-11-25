# SPDX-License-Identifier: Apache-2.0
"""CPU tests (Stage B): the DSV4.1 TPU attention metadata of one fused MIXED
forward (a prompt EXTEND batch, then verify rows of running requests, laid out by
``mixed_step.merge_mixed_batch``) is, per field, the concatenation of the plain
EXTEND forward's metadata for the prompt rows and the TARGET_VERIFY forward's
metadata for the verify rows.

Per-row fields (positions, slots, masks, compressed-record writes) concatenate;
per-request fields (page tables, SWA floors, index pages) concatenate along the
batch; offsets (query_start_loc, token_rows, window commit rows) shift by the
prompt part. The request-window SWA workspace hands out pages in batch order, so
its page ids are relabelled, not equal: those comparisons are strict xfails, and
a separate test checks the relabelled encodings address the same rows.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu.compilation import host_input_mirror
from sglang_tpu.compilation.decoder_compile_policy import DecoderTpuCompilePolicy
from sglang_tpu.experimental.speculative.host_verify_rows import host_verify_rows
from sglang_tpu.experimental.speculative.mixed_step import (
    VerifySegment,
    extend_folded_row_maps,
    merge_mixed_batch,
    mixed_positions_cpu,
)
from sglang_tpu.models.deepseek_v41 import attention_backend as ab
from sglang_tpu.models.deepseek_v41.request_window import (
    RESERVED_PAGES,
    RING_PAGES,
    SWA_PAGE_TOKENS,
    WINDOW_TOKENS,
)

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

MAX_CONTEXT = 1 << 17
SLOTS = 8
R = 6  # verify rows per request
CAPACITY = 6  # select_verify_rows candidate capacity (2 * CAPACITY columns)
REPLAY = 128  # encoder SWA replay rows of a reset prefix hit
TOKEN_PAD = 5  # token-bucket padding rows of the mixed forward
SWA_KEEP = 512  # paged SWA keeps each request's last SWA_KEEP positions


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


R2T = _req_to_token()

# Prompt requests: (slot, prefix, extend, encoder-replay reset). Slot 1 is a long
# prefix hit (folded: its last REPLAY cached tokens replay in front of it).
PROMPTS = [(1, 98304, 768, True), (2, 0, 300, False)]
# Verify requests: (slot, host committed length, device seq_len lead over it).
# 4093: the block crosses a 256 (and 64) page boundary and starts at an odd
# position (ratio-2 partner from the pair ring); 98310: far beyond the SWA window;
# 17: history shorter than the window; 250: device seq_lens 3 ahead of the host
# committed length (select_verify_rows offsets into the candidates).
VERIFIES = [(3, 4093, 0), (4, 98310, 0), (5, 17, 0), (6, 250, 3)]


def _frontier():
    """Each slot's last scheduled position + 1 (the paged SWA eviction point)."""
    out = torch.zeros(SLOTS, dtype=torch.int64)
    for slot, p, n, _ in PROMPTS:
        out[slot] = p + n
    for slot, c, lead in VERIFIES:
        out[slot] = c + lead + R
    return out


def _swa_mapping():
    """full loc -> SWA loc: kept for each slot's last SWA_KEEP positions, else 0."""
    mapping = torch.zeros(SLOTS * MAX_CONTEXT + 256, dtype=torch.int64)
    frontier = _frontier()
    pos = torch.arange(MAX_CONTEXT)
    for slot in range(1, SLOTS):
        keep = pos >= frontier[slot] - SWA_KEEP
        locs = R2T[slot][keep]
        mapping[locs] = locs + 1024
    return mapping


def _window():
    return SimpleNamespace(
        workspace_first_page=RESERVED_PAGES + RING_PAGES * SLOTS,
        workspace_pages=4096,
    )


def _ring_valid_from():
    valid_from = torch.zeros(SLOTS, dtype=torch.int64)
    valid_from[4] = 98310 - 40  # the slot's previous owner left rows below this
    valid_from[1] = 98304 - 64
    return valid_from


STORAGE = {
    "paged": dict(full_to_swa_index_mapping=None),
    "paged_evicting": dict(full_to_swa_index_mapping=_swa_mapping()),
    "request_window": dict(request_window=_window()),
    "ring": dict(swa_ring_valid_from=_ring_valid_from()),
}


def _metadata(fb, storage):
    kwargs = dict(
        full_to_swa_index_mapping=None, request_window=None, swa_ring_valid_from=None
    )
    kwargs.update(STORAGE[storage])
    return ab.build_forward_metadata(
        fb,
        req_to_token=R2T,
        max_context_len=MAX_CONTEXT,
        low_ratios=ab.LOW_RATIOS,
        device=torch.device("cpu"),
        **kwargs,
    )


# ----------------------------------------------------------------------------
# The three batches.


def _prompt(folded: bool):
    """The prompt ScheduleBatch view after the encoder-replay fold, and its maps."""
    prefix, extend, floors, keep, loc = [], [], [], [], []
    rows = 0
    for slot, p, n, reset in PROMPTS:
        r = min(p, REPLAY) if (folded and reset and p) else 0
        prefix.append(p - r)
        extend.append(n + r)
        floors += [p - r if r else 0] * (n + r)
        keep += list(range(rows + r, rows + r + n))
        loc.append(R2T[slot, p - r : p + n])
        rows += n + r
    slots = torch.tensor([s for s, *_ in PROMPTS], dtype=torch.int64)
    seq = torch.tensor([p + n for p, n in zip(prefix, extend)], dtype=torch.int64)
    prompt = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        reqs=[f"p{i}" for i in range(len(PROMPTS))],
        input_ids=torch.zeros(rows, dtype=torch.int64),
        out_cache_loc=torch.cat(loc).to(torch.int64),
        prefix_lens=prefix,
        extend_lens=extend,
        extend_num_tokens=rows,
        extend_logprob_start_lens=None,
        seq_lens=seq,
        seq_lens_cpu=seq.clone(),
        req_pool_indices=slots,
        req_pool_indices_cpu=slots.clone(),
        multimodal_inputs=None,
        encoder_swa_reset=[bool(f) for *_, f in PROMPTS] if folded else None,
        spec_info=None,
        return_logprob=False,
        top_logprobs_nums=None,
        token_ids_logprobs=None,
        extend_input_logprob_token_ids=None,
        prefill_input_ids_cpu=None,
    )
    fold = (
        SimpleNamespace(
            num_rows=rows,
            row_floor=torch.tensor(floors, dtype=torch.int64),
            keep_rows=torch.tensor(keep, dtype=torch.int64),
        )
        if folded
        else None
    )
    return prompt, fold


def _verify_rows():
    """select_verify_rows' host twin over the verify requests (live rows only)."""
    slots = torch.tensor([s for s, _, _ in VERIFIES], dtype=torch.int64)
    committed = torch.tensor([c for _, c, _ in VERIFIES], dtype=torch.int64)
    seq_lens = committed + torch.tensor([lead for *_, lead in VERIFIES])
    candidates = R2T[slots[:, None], committed[:, None] + torch.arange(2 * CAPACITY)]
    rows = host_verify_rows(
        seq_lens=seq_lens,
        candidates_cpu=candidates,
        committed_lens_cpu=committed,
        req_pool_indices_cpu=slots,
        num_rows=R,
    )
    return rows, committed, seq_lens


def _extend_fb(prompt, fold):
    """The plain EXTEND forward of the (folded) prompt batch."""
    positions = mixed_positions_cpu(prompt, None)
    return FB(
        forward_mode=ForwardMode.EXTEND,
        batch_size=len(prompt.reqs),
        positions=positions,
        out_cache_loc=prompt.out_cache_loc.to(torch.int32),
        extend_seq_lens=torch.tensor(prompt.extend_lens, dtype=torch.int32),
        extend_seq_lens_cpu=list(prompt.extend_lens),
        extend_prefix_lens_cpu=list(prompt.prefix_lens),
        req_pool_indices=prompt.req_pool_indices.clone(),
        encoder_swa_row_floor=None if fold is None else fold.row_floor,
        encoder_swa_compress_rows=None if fold is None else fold.keep_rows,
        spec_info=None,
    )


def _verify_fb():
    """Today's separate TARGET_VERIFY forward of the verify requests."""
    rows, _, _ = _verify_rows()
    return FB(
        forward_mode=ForwardMode.TARGET_VERIFY,
        batch_size=len(VERIFIES),
        positions=rows.positions.clone(),
        out_cache_loc=rows.out_cache_loc.to(torch.int32),
        extend_seq_lens=None,
        extend_seq_lens_cpu=None,
        extend_prefix_lens_cpu=None,
        req_pool_indices=rows.req_pool_indices.clone(),
        encoder_swa_row_floor=None,
        spec_info=SimpleNamespace(draft_token_num=R, custom_mask=None),
    )


def _mixed_fb(prompt, fold):
    """The fused MIXED forward: merge_mixed_batch's view, token-bucket padded."""
    rows, committed, seq_lens = _verify_rows()
    n_v = len(VERIFIES)
    verify = VerifySegment(
        reqs=[f"v{i}" for i in range(n_v)],
        req_pool_indices=rows.req_pool_indices.clone(),
        req_pool_indices_cpu=rows.req_pool_indices.clone(),
        seq_lens=seq_lens + R,
        seq_lens_cpu=seq_lens + R,
        committed_lens_cpu=committed.tolist(),
        input_ids=torch.zeros(n_v * R, dtype=torch.int64),
        positions_cpu=rows.positions.clone(),
        out_cache_loc=rows.out_cache_loc.clone(),
        num_verify_rows=R,
    )
    merged = merge_mixed_batch(
        prompt,
        verify,
        prompt_input_ids=prompt.input_ids,
        spec_info=SimpleNamespace(draft_token_num=R, custom_mask=None),
        mixed_mode=ForwardMode.MIXED,
    )
    positions = mixed_positions_cpu(prompt, verify)
    real = positions.shape[0]
    maps = None if fold is None else extend_folded_row_maps(fold, n_v * R)
    return FB(
        forward_mode=ForwardMode.MIXED,
        batch_size=len(merged.reqs),
        positions=F.pad(positions, (0, TOKEN_PAD)),
        out_cache_loc=F.pad(merged.out_cache_loc.to(torch.int32), (0, TOKEN_PAD)),
        extend_seq_lens=torch.tensor(merged.extend_lens, dtype=torch.int32),
        extend_seq_lens_cpu=list(merged.extend_lens),
        extend_prefix_lens_cpu=list(merged.prefix_lens),
        req_pool_indices=merged.req_pool_indices.clone(),
        encoder_swa_row_floor=None if maps is None else maps.row_floor,
        encoder_swa_compress_rows=None if maps is None else maps.compress_rows,
        spec_info=merged.spec_info,
    ), real


# ----------------------------------------------------------------------------
# Field split.

PER_ROW = (
    "positions",
    "token_req_pool",
    "valid",
    "compress_valid",
    "out_cache_loc",
    "low_ratio.1.out_loc",
    "low_ratio.1.lengths",
    "low_ratio.2.out_loc",
    "low_ratio.2.lengths",
)
PER_REQUEST = (
    "full_pages",
    "swa_pages",
    "swa_minimum",
    "low_ratio.1.index_pages",
    "low_ratio.2.index_pages",
)
# Shifted by the prompt part: rows (Te) or requests (Bp).
OFFSET = ("query_start_loc", "token_rows")
# Request-window row moves, [B * WINDOW_TOKENS] each.
PLAN = (
    "swa_window_plan.history_src",
    "swa_window_plan.history_dst",
    "swa_window_plan.commit_token",
    "swa_window_plan.commit_dst",
)


def _get(meta, name):
    obj = meta
    for part in name.split("."):
        if isinstance(obj, dict):
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)
    return obj


class Case:
    def __init__(self, storage, folded):
        prompt, fold = _prompt(folded)
        self.e = _metadata(_extend_fb(prompt, fold), storage)
        self.v = _metadata(_verify_fb(), storage)
        mfb, self.real = _mixed_fb(prompt, fold)
        self.m = _metadata(mfb, storage)
        self.bp, self.bv = len(PROMPTS), len(VERIFIES)
        self.te, self.tv = sum(prompt.extend_lens), R * len(VERIFIES)
        assert self.real == self.te + self.tv

    def parts(self, name):
        """(mixed prompt part, extend, mixed verify part, verify) of a field."""
        m, e, v = (_get(x, name) for x in (self.m, self.e, self.v))
        te, tv, bp, bv = self.te, self.tv, self.bp, self.bv
        if name in PER_ROW:
            return m[:te], e[:te], m[te : te + tv], v[:tv]
        if name in PER_REQUEST:
            return m[:bp], e[:bp], m[bp : bp + bv], v[:bv]
        if name == "query_start_loc":
            return m[: bp + 1], e, m[bp:] - te, v
        if name == "token_rows":
            mv = m[te : te + tv]
            return m[:te], e[:te], torch.where(mv >= 0, mv - bp, mv), v[:tv]
        if name in PLAN:
            m, e, v = (
                m.view(-1, WINDOW_TOKENS),
                e.view(-1, WINDOW_TOKENS),
                v.view(-1, WINDOW_TOKENS),
            )
            mv = m[bp : bp + bv]
            if name.endswith("commit_token"):
                ok = (
                    _get(self.m, "swa_window_plan.commit_dst").view(-1, WINDOW_TOKENS)[
                        bp : bp + bv
                    ]
                    >= 0
                )
                mv = torch.where(ok, mv - te, mv)
            return m[:bp], e[:bp], mv, v[:bv]
        raise KeyError(name)


_CASES = {}


def _case(storage, folded):
    key = (storage, folded)
    if key not in _CASES:
        _CASES[key] = Case(storage, folded)
    return _CASES[key]


def _assert_equal(a, b, what):
    assert a.dtype == b.dtype, (what, a.dtype, b.dtype)
    assert a.shape == b.shape, (what, a.shape, b.shape)
    if not torch.equal(a, b):
        diff = (a != b).nonzero()
        raise AssertionError(
            f"{what}: {diff.shape[0]} entries differ, first {diff[:4].tolist()}"
        )


# The request-window workspace numbers pages in batch order (request_window.py
# _assign_workspace_pages): the verify requests' pages follow the prompt's.
_RELABELLED = ("swa_pages", "swa_window_plan.history_dst")


def _params():
    out = []
    for storage in STORAGE:
        for folded in (False, True):
            fields = PER_ROW + PER_REQUEST + OFFSET
            if storage == "request_window":
                fields += PLAN
            for name in fields:
                marks = ()
                if storage == "request_window" and name in _RELABELLED:
                    marks = pytest.mark.xfail(
                        strict=True,
                        reason="request-window workspace pages are assigned in "
                        "batch order: the verify part's page ids shift by the "
                        "prompt's workspace pages (equivalent, see "
                        "test_request_window_pages_are_relabelled)",
                    )
                out.append(
                    pytest.param(
                        storage,
                        folded,
                        name,
                        marks=marks,
                        id=f"{storage}-{'folded' if folded else 'plain'}-{name}",
                    )
                )
    return out


@pytest.mark.parametrize("storage,folded,name", _params())
def test_verify_part_equals_target_verify(storage, folded, name):
    _, _, mv, v = _case(storage, folded).parts(name)
    _assert_equal(mv, v, f"verify part of {name}")


@pytest.mark.parametrize("storage", list(STORAGE))
@pytest.mark.parametrize("folded", [False, True])
def test_prompt_part_equals_extend(storage, folded):
    case = _case(storage, folded)
    fields = PER_ROW + PER_REQUEST + OFFSET
    if storage == "request_window":
        fields += PLAN
    for name in fields:
        mp, e, _, _ = case.parts(name)
        _assert_equal(mp, e, f"prompt part of {name}")


@pytest.mark.parametrize("storage", list(STORAGE))
def test_padding_rows_are_inert(storage):
    case = _case(storage, True)
    m, real = case.m, case.real
    assert m.positions.shape[0] == real + TOKEN_PAD
    assert not m.valid[real:].any() and not m.compress_valid[real:].any()
    assert (m.token_rows[real:] == -1).all()
    assert (m.token_req_pool[real:] == 0).all()
    for ratio in ab.LOW_RATIOS:
        assert (m.low_ratio[ratio].out_loc[real:] == -1).all()
        assert (m.low_ratio[ratio].lengths[real:] == 0).all()


def test_verify_rows_compress_and_have_no_floor():
    """The fold leaves the verify rows alone: they compress and floor at 0 (ring
    storage aside, which floors every request at its slot's valid_from)."""
    case = _case("paged", True)
    te, tv, bp = case.te, case.tv, case.bp
    assert case.m.compress_valid[te : te + tv].all()
    assert (case.m.swa_minimum[bp:] == 0).all()
    # The prompt's replay rows do not compress; its reset hit floors at its replay start.
    assert not case.m.compress_valid[:REPLAY].any()
    assert int(case.m.swa_minimum[0]) == PROMPTS[0][1] - REPLAY


def test_request_window_pages_are_relabelled():
    """Under the request window the verify part's SWA table and history copies
    differ from TARGET_VERIFY's only by a per-request constant page shift into
    still-unused workspace pages, so every position reads/writes the same ring
    rows through an equivalent workspace page."""
    for folded in (False, True):
        case = _case("request_window", folded)
        _, _, mv, v = case.parts("swa_pages")
        _, _, mh, vh = case.parts("swa_window_plan.history_dst")
        for b in range(case.bv):
            used = v[b] != 0
            assert torch.equal(mv[b] != 0, used), b
            shift = mv[b][used] - v[b][used]
            assert shift.unique().numel() == 1, b
            shift = int(shift[0])
            assert shift > 0, b
            ok = vh[b] >= 0
            assert torch.equal(mh[b] >= 0, ok), b
            assert torch.equal(mh[b][ok], vh[b][ok] + shift * SWA_PAGE_TOKENS), b
        # The mixed forward's workspace pages are disjoint across requests.
        pages = case.m.swa_pages
        live = pages[pages != 0]
        assert live.unique().numel() == live.numel()


@pytest.mark.parametrize("storage", list(STORAGE))
def test_mixed_metadata_from_host_mirror_matches(storage):
    """With the mixed positions/slots/pool rows recorded as host copies (meta
    device: a read would raise), the MIXED metadata is the same; without a
    record it reads positions from the device (MIXED is not EXTEND, so
    _host_extend_positions declines)."""
    prompt, fold = _prompt(True)
    reference = _metadata(_mixed_fb(prompt, fold)[0], storage)
    fb, _ = _mixed_fb(*_prompt(True))
    policy = SimpleNamespace(model_runner=SimpleNamespace(device=torch.device("meta")))
    for name in ("positions", "req_pool_indices", "out_cache_loc"):
        DecoderTpuCompilePolicy._upload_field(policy, fb, name, getattr(fb, name))
    got = _metadata(fb, storage)
    assert ab._host_extend_positions(fb, fb.positions.shape[0]) is None
    for name in PER_ROW + PER_REQUEST + OFFSET:
        _assert_equal(_get(got, name), _get(reference, name), name)


# ----------------------------------------------------------------------------
# Decoder SWA bounded replay (CED): the decoder rows' metadata (``tail``).

# Fill lengths of the prompt requests: slot 1's chunk ends 1000 tokens before its
# fill (mid-prompt, no decoder rows under the ``prompt`` rule); slot 2's chunk
# ends its fill (its last 128 rows are decoder rows).
PROMPT_FILLS = [98304 + 768 + 1000, 300]
TAIL_BUCKET = 256
TAIL_PER_ROW = ("positions", "token_req_pool", "valid", "out_cache_loc")
TAIL_PER_REQUEST = ("full_pages", "swa_pages", "swa_minimum")


def _with_tail(fb, fills, **extra):
    fb.decoder_fill_lens = fills
    fb.decoder_rows_bucket = TAIL_BUCKET
    for name, value in extra.items():
        setattr(fb, name, value)
    return fb


def _tail_metadata(fb, storage):
    kwargs = dict(
        full_to_swa_index_mapping=None, request_window=None, swa_ring_valid_from=None
    )
    kwargs.update(STORAGE[storage])
    return ab.build_forward_metadata(
        fb,
        req_to_token=R2T,
        max_context_len=MAX_CONTEXT,
        low_ratios=ab.LOW_RATIOS,
        device=torch.device("cpu"),
        decoder_replay=True,
        **kwargs,
    )


# Not "ring": that is the DSpark draft's SWA storage, and the draft never replays.
@pytest.mark.parametrize("storage", ["paged", "paged_evicting", "request_window"])
@pytest.mark.parametrize("folded", [False, True])
def test_fused_tail_equals_extend_tail_and_target_verify(storage, folded):
    """Under CED the fused MIXED forward's decoder rows are the prompt part's
    extend decoder rows, then every verify row; the verify rows' decoder
    metadata is the plain TARGET_VERIFY forward's (no fill - 128 floor). The
    request window relabels the verify part's workspace pages (see
    test_request_window_pages_are_relabelled), so its swa_pages are skipped."""
    from sglang_tpu.envs import envs as tpu_envs

    prompt, fold = _prompt(folded)
    prompt.decoder_fill_lens = list(PROMPT_FILLS)
    with tpu_envs.SGLANG_TPU_DSV41_DECODER_REPLAY_ROWS.override("prompt"):
        e = _tail_metadata(
            _with_tail(_extend_fb(prompt, fold), list(PROMPT_FILLS)), storage
        )
        mfb, _ = _mixed_fb(prompt, fold)
        rows, committed, _ = _verify_rows()
        fills = list(PROMPT_FILLS) + [int(c) + R for c in committed]
        m = _tail_metadata(
            _with_tail(
                mfb, fills, mixed_host_rows=SimpleNamespace(prompt_reqs=prompt.reqs)
            ),
            storage,
        )
    v = _metadata(_verify_fb(), storage)
    bp, bv, tv = len(PROMPTS), len(VERIFIES), R * len(VERIFIES)
    te = int(e.tail.valid.sum())
    assert te == 128  # slot 2's last window; slot 1 is mid-prompt
    assert m.tail_rows == e.tail_rows == TAIL_BUCKET
    assert int(m.tail.valid.sum()) == te + tv
    for name in TAIL_PER_ROW:
        mt, et, vt = (_get(x, name) for x in (m.tail, e.tail, v))
        _assert_equal(mt[:te], et[:te], f"tail prompt part of {name}")
        _assert_equal(mt[te : te + tv], vt[:tv], f"tail verify part of {name}")
    for name in TAIL_PER_REQUEST:
        if storage == "request_window" and name == "swa_pages":
            continue
        mt, et, vt = (_get(x, name) for x in (m.tail, e.tail, v))
        _assert_equal(mt[:bp], et[:bp], f"tail prompt part of {name}")
        _assert_equal(mt[bp : bp + bv], vt[:bv], f"tail verify part of {name}")
