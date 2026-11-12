from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import pytest

from sglang.srt.runtime_context import get_context
from sglang_tpu.pp_chunk_fit import (
    FitCalibration,
    FitFingerprint,
    fit_key,
    guard_points,
    guard_report,
    load_saved_fit,
    read_saved_fits,
    saved_cost_model,
    saved_fit,
    upsert_saved_fit,
    weak_fingerprint_fields,
)
from sglang_tpu.pp_chunks import (
    StepCostModel,
    chunk_buckets,
    chunk_granularity,
    chunk_pairs,
    longest_prompt,
    profile_points,
    stage_min_samples,
    uses_dynamic_chunks,
)

BUCKETS = [16, 4096, 8192, 12288, 16384]
ATTN = 1.5e-7  # ms per token of chunk per token of context


def _linear(tokens: int) -> float:
    """Prefix-free time of one request without its own causal attention."""
    return 12.0 + 0.0109 * tokens


def _measured(tokens: int, prefix: int, scale: float = 1.0) -> float:
    """What one stage reports: the ladder plus attention over the prefix
    and over the chunk itself."""
    return scale * (_linear(tokens) + ATTN * (tokens * prefix + tokens * tokens / 2))


def _stage(scale=1.0):
    out = []
    for tokens in BUCKETS:
        out.append((tokens, 0, _measured(tokens, 0, scale)))
        for prefix in (16384, 32768):
            out.append((tokens, prefix, _measured(tokens, prefix, scale)))
    return out


def _model(slack=0.1, granularity=4096) -> StepCostModel:
    return StepCostModel.fit([_stage(1.0), _stage(0.9)], slack, granularity)


def test_granularity_is_an_eighth_step_in_whole_blocks_when_aligned():
    assert chunk_granularity(4096, 16384, True) == 4096
    assert chunk_granularity(4352, 16384, True) == 4352
    assert chunk_granularity(16, 16384, True) == 2048
    assert chunk_granularity(1024, 16384, True) == 2048
    assert chunk_granularity(4096, 8192, True) == 4096


def test_granularity_ignores_the_block_without_alignment_or_above_a_step():
    assert chunk_granularity(4352, 16384, False) == 2048
    assert chunk_granularity(1114112, 16384, True) == 2048
    assert chunk_granularity(8192, 4096, True) == 512


def test_buckets_cover_every_granule_multiple():
    assert chunk_buckets(16384, 4096) == [4096, 8192, 12288]
    assert chunk_buckets(16384, 4352) == [4352, 8704, 13056]
    assert chunk_buckets(4096, 512) == [512, 1024, 1536, 2048, 2560, 3072, 3584]
    assert chunk_buckets(4096, 4096) == []


def test_profile_points_cover_every_bucket_and_prefixes_for_large_ones():
    points = profile_points(BUCKETS, 16384, 65536)
    assert [p for p in points if p[1] == 0] == [(b, 0) for b in BUCKETS]
    assert (16, 16384) not in points
    assert (4096, 16384) in points and (4096, 32768) in points
    assert (16384, 16384) in points and (16384, 32768) in points
    # Every point stays a step short of the model length.
    assert max(t + p for t, p in points) <= 65536 - 16384
    clipped = profile_points(BUCKETS, 16384, 40960)
    assert (16384, 8192) in clipped and (16384, 16384) not in clipped


def test_profile_points_skip_prefixes_beyond_the_schedule_capacity():
    # 16K behind 32K needs 64 * 128 + 2080 = 10,272 pairs of 256x256 tiles,
    # 16K behind 64K needs 64 * 256 + 2080 = 18,464.
    points = profile_points(BUCKETS, 32768, 262144, (13436, 256, 256))
    assert [p for p in points if p[1] == 0] == [(b, 0) for b in BUCKETS]
    assert (16384, 32768) in points and (16384, 65536) not in points
    assert (8192, 32768) in points and (8192, 65536) in points
    assert chunk_pairs((13436, 256, 256), 16384, 32768) == 10272
    assert chunk_pairs(None, 16384, 32768) == 0


def test_fit_takes_the_slowest_stage_and_removes_the_causal_term():
    model = _model()
    assert model.attn_ms_per_token_pair == pytest.approx(ATTN, rel=1e-6)
    assert model.linear_ms[16384] == pytest.approx(_linear(16384), rel=1e-6)
    assert model.target_ms == pytest.approx(1.1 * _measured(16384, 0))
    assert model.granularity == 4096
    assert model.buckets == BUCKETS


def test_fit_uses_the_median_slope_and_ignores_a_faster_prefix_run():
    stage = _stage()
    # One bucket whose far prefix point ran faster than its near one.
    stage = [(t, p, ms if (t, p) != (16384, 32768) else 100.0) for t, p, ms in stage]
    model = StepCostModel.fit([stage], 0.1, 4096)
    assert model.attn_ms_per_token_pair == pytest.approx(ATTN, rel=1e-6)


def test_fit_slope_ignores_an_offset_on_the_prefix_free_points():
    # Every prefix-free point carries 14 ms that the prefix points lack.
    stage = [(t, p, ms + (14.0 if p == 0 else 0.0)) for t, p, ms in _stage()]
    model = StepCostModel.fit([stage], 0.1, 4096)
    assert model.attn_ms_per_token_pair == pytest.approx(ATTN, rel=1e-6)
    # A bucket with a single prefix point falls back to its prefix-free one.
    single = [(t, p, ms) for t, p, ms in _stage() if p != 32768]
    model = StepCostModel.fit([single], 0.1, 4096)
    assert model.attn_ms_per_token_pair == pytest.approx(ATTN, rel=1e-6)


def test_fit_rejects_negative_slack_empty_input_and_mismatched_ladders():
    with pytest.raises(ValueError, match="slack"):
        StepCostModel.fit([_stage()], -0.1, 4096)
    with pytest.raises(ValueError, match="no stage"):
        StepCostModel.fit([], 0.1, 4096)
    with pytest.raises(ValueError, match="different token buckets"):
        StepCostModel.fit([_stage(), [(16384, 0, 190.0)]], 0.1, 4096)


def test_step_time_adds_the_causal_term_of_each_chunk():
    model = _model()
    # Four 4K first chunks cost less than one 16K chunk in the same bucket.
    assert model.step_ms(16384, 0.0, 4 * 4096.0**2) < model.step_ms(
        16384, 0.0, 16384.0**2
    )


def test_first_chunk_fills_the_step():
    model = _model()
    assert model.chunk(0, 16384, 0, 0.0, 0.0) == 16384
    assert model.chunk(0, 8192, 8192, 0.0, 8192.0**2) == 8192


def test_chunk_behind_a_prefix_shrinks_to_a_bucket_that_fits():
    model = _model()
    # 16K behind 16K costs 41 ms over a full step: over the 10% slack.
    assert model.chunk(16384, 16384, 0, 0.0, 0.0) == 12288
    # Deep in a prompt only a granule fits.
    assert model.chunk(112 * 1024, 16384, 0, 0.0, 0.0) == 4096
    # The last chunk of a prompt need not be aligned.
    assert model.chunk(16384, 5000, 0, 0.0, 0.0) == 5000


def test_a_clipped_chunk_is_at_least_a_quarter_step():
    model = _model(granularity=2048)
    # Very deep in a prompt the time budget would allow under 2K tokens.
    assert model.chunk(200 * 1024, 16384, 0, 0.0, 0.0) == 4096
    # The floor never exceeds the room left in the step.
    assert model.chunk(200 * 1024, 16384, 14336, 0.0, 0.0) == 2048
    assert model.chunk(200 * 1024, 16384, 16384, 0.0, 0.0) == 0


def test_a_step_limit_holds_the_step_to_a_smaller_bucket():
    # Buckets at every granule multiple, as the runner compiles them.
    buckets = list(range(2048, 16385, 2048))
    linear = {b: _linear(b) for b in buckets}
    model = StepCostModel(buckets, linear, ATTN, 1.1 * _measured(16384, 0), 2048)
    # A first chunk fills the limited step; a short request is untouched.
    assert model.chunk(0, 65536, 0, 0.0, 0.0, step_limit=4096) == 4096
    assert model.chunk(0, 1000, 0, 0.0, 0.0, step_limit=4096) == 1000
    # The limit wins over the quarter-step floor deep in a prompt.
    assert model.chunk(200 * 1024, 16384, 0, 0.0, 0.0, step_limit=2048) == 2048
    # 37 decode tokens in the step: the total still lands on the limit.
    assert model.chunk(0, 65536, 37, 0.0, 37.0, step_limit=4096) == 4096 - 37
    # A second request finds the limited step full.
    assert model.chunk(0, 65536, 4096, 0.0, 4096.0**2, step_limit=4096) == 0


def test_chunk_rounds_the_step_total_to_a_granule():
    model = _model()
    # 37 decode tokens sit in the step: a first chunk lands the total on
    # 16384, a chunk behind 16K on 12288.
    assert model.chunk(0, 65536, 37, 0.0, 37.0) == 16384 - 37
    assert model.chunk(16384, 65536, 37, 37.0 * 16384, 37.0) == 12288 - 37


def test_later_requests_get_what_the_step_has_left():
    model = _model()
    # A 4K chunk behind 16K costs 10 ms; a first chunk still fills the step.
    assert model.chunk(0, 16384, 4096, 4096.0 * 16384, 4096.0**2) == 12288
    # A 12K chunk behind 16K costs 30 ms; a 4K first chunk on top would take
    # the step past its target, so it waits.
    assert model.chunk(0, 16384, 12288, 12288.0 * 16384, 12288.0**2) == 0
    assert model.chunk(0, 16384, 16384, 0.0, 0.0) == 0


def test_a_slow_odd_bucket_does_not_hide_a_larger_fitting_one():
    model = _model()
    model.linear_ms[12288] = model.linear_ms[16384] + 10.0
    assert model.chunk(0, 16384, 4096, 4096.0 * 16384, 4096.0**2) == 12288


def test_without_an_attention_term_chunks_fill_the_bucket():
    model = StepCostModel.fit([[(b, 0, _linear(b)) for b in BUCKETS]], 0.0, 4096)
    assert model.attn_ms_per_token_pair == 0.0
    assert model.chunk(65536, 16384, 0, 0.0, 0.0) == 16384


def test_pairs_count_the_kernel_schedule_of_a_chunk():
    model = _model()
    assert model.pairs(4096, 0) == 0  # no schedule capacity known
    model.schedule = (10000, 256, 256)
    # 16 query blocks; block i attends to (i + 1) KV blocks.
    assert model.pairs(4096, 0) == sum(range(1, 17))
    # Behind 8K tokens each block also attends to the 32 prefix blocks.
    assert model.pairs(4096, 8192) == sum(range(1, 17)) + 16 * 32
    assert model.pairs(0, 8192) == 0


def test_the_schedule_capacity_bounds_a_chunk_deep_in_a_prompt():
    model = _model(granularity=2048)
    model.schedule = (10000, 256, 256)
    # 16K behind 48K needs 64 * (192 + ~32) pairs: over 10000.
    got = model.chunk(49152, 16384, 0, 0.0, 0.0)
    assert got % 2048 == 0 and 0 < got < 16384
    assert model.pairs(got, 49152) <= 10000
    assert model.pairs(got + 2048, 49152) > 10000
    # Pairs already in the step leave less room.
    less = model.chunk(49152, 16384, 2048, 0.0, 2048.0**2, None, 6000)
    assert less < got and model.pairs(less, 49152) <= 4000
    # A first chunk with a short prefix is not bounded.
    assert model.chunk(0, 16384, 0, 0.0, 0.0) == 16384


def test_longest_prompt_is_the_last_whose_final_chunk_fits():
    schedule = (400, 256, 256)
    longest = longest_prompt(schedule, 2048, 1 << 20)
    assert chunk_pairs(schedule, 2048, longest - 2048) <= 400
    assert chunk_pairs(schedule, 2048, longest + 1 - 2048) > 400
    assert longest_prompt(schedule, 2048, longest) == longest


def test_fit_keeps_the_schedule_capacity():
    model = StepCostModel.fit([_stage()], 0.1, 4096, (12345, 256, 512))
    assert model.schedule == (12345, 256, 512)
    assert "12345 attention pairs (256x512)" in model.describe()


def test_dynamic_chunks_use_resolved_defaults_and_exclude_decode(monkeypatch):
    monkeypatch.setenv("SGLANG_TPU_PP_DYNAMIC_CHUNKS", "1")
    with get_context().override_server_args(
        device="tpu",
        pp_size=8,
        chunked_prefill_size=16384,
        disaggregation_mode="null",
    ) as raw:
        assert raw.device is None
        assert raw.chunked_prefill_size is None
        assert uses_dynamic_chunks()
        get_context().override("test.decode_role", disaggregation_mode="decode")
        assert not uses_dynamic_chunks()
        get_context().override("test.prefill_role", disaggregation_mode="prefill")
        assert uses_dynamic_chunks()
        get_context().override("test.no_chunks", chunked_prefill_size=-1)
        assert not uses_dynamic_chunks()


FINGERPRINT = FitFingerprint(
    profile_points=[(2048, 0), (4096, 0), (4096, 4096)],
    schedule=(4096, 256, 512),
    kernel_version="k1",
    runtime_version="torch_tpu=1;libtpu=2",
)
REAL = FitCalibration(weights="real", routing="real", date="2026-11-07")
MODEL = StepCostModel([2048, 4096], {2048: 30.0, 4096: 55.0}, 1e-6, 0.0, 512)
# One calibrated time per guard point of FINGERPRINT.
GUARD = [(2048, 0, 32.0), (4096, 0, 57.0), (4096, 4096, 75.0)]


def _entry():
    return saved_fit(
        MODEL, slack=0.1, fingerprint=FINGERPRINT, calibration=REAL, guard_points=GUARD
    )


def _table(tmp_path, *, calibration=REAL, slack=0.1):
    path = str(tmp_path / "fits.json")
    entry = saved_fit(
        MODEL,
        slack=slack,
        fingerprint=FINGERPRINT,
        calibration=calibration,
        guard_points=GUARD,
    )
    upsert_saved_fit(path, "shape", entry)
    return path


def test_stage_min_takes_fastest_device_per_point_and_keeps_stages_apart():
    reports = [
        (1, [(2048, 0, 30.0)]),
        (0, [(2048, 0, 21.0), (4096, 0, 40.0)]),
        (0, [(2048, 0, 20.0), (4096, 0, 44.0)]),
        (1, [(2048, 0, 33.0)]),
    ]
    assert stage_min_samples(reports) == [
        [(2048, 0, 20.0), (4096, 0, 40.0)],
        [(2048, 0, 30.0)],
    ]


def test_fit_key_is_the_table_file_format():
    """Saved tables are keyed by this string; changing it orphans every entry."""
    key = fit_key(
        model_type="m",
        num_layers=48,
        hidden_size=5120,
        model_shape="cfg0123abcd",
        dtype="bfloat16",
        quantization=None,
        kv_cache_dtype="bfloat16",
        attention_backend="Rpa",
        moe_runner_backend="auto",
        moe_a2a_backend="none",
        device_kind="TPU7x",
        pp_size=8,
        tp_size=2,
        attn_cp_size=1,
        attn_dcp_size=1,
        moe_ep_size=2,
        moe_tp_size=1,
        moe_dense_tp_size=2,
        attn_dp_size=1,
        chunk_tokens=8192,
        extend_batch_cap=8,
        page_size=512,
        layer_partition="",
    )
    assert key == (
        "m/L48xH5120/cfg0123abcd/bfloat16/none/kvbfloat16/Rpa/auto/a2anone/TPU7x/pp8xtp2xcp1xdcp1xep2xmoetp1xdensetp2xdp1/chunk8192/extendbs8/page512/spliteven"
    )


def test_saved_fit_round_trips_and_takes_the_launch_slack(tmp_path):
    path = _table(tmp_path, slack=0.5)
    entry, reason = load_saved_fit(
        table_path=path, key="shape", fingerprint=FINGERPRINT, check=True
    )
    model = saved_cost_model(entry, slack=0.1, granularity=512)
    assert model.linear_ms == MODEL.linear_ms
    assert model.schedule == (4096, 256, 512)
    assert model.target_ms == pytest.approx((55.0 + 1e-6 * 4096 * 4096 / 2) * 1.1)
    assert entry.guard_points == GUARD
    assert "2026-11-07" in reason


@pytest.mark.parametrize(
    "case, expected",
    [
        ("no_table", "no fit table"),
        ("no_entry", "no saved fit for other"),
        ("dummy", "only real weights and routing"),
        ("stale", "kernel_version differs"),
        ("corrupt", "cannot read fit table"),
        ("bad_ladder", "has buckets [2048], not [2048, 4096]"),
        ("zero_guard", "guard time that is not positive"),
        ("no_guard", "no calibrated time at [(2048, 0), (4096, 0), (4096, 4096)]"),
        ("no_attention_guard", "no calibrated time at [(4096, 4096)]"),
        ("negative_attention", "negative, zero or unbounded step cost"),
        ("negative_linear", "negative, zero or unbounded step cost"),
        ("zero_step", "negative, zero or unbounded step cost"),
        ("unbounded_step", "negative, zero or unbounded step cost"),
    ],
)
def test_unusable_saved_fit_says_why(tmp_path, case, expected):
    calibration = (
        FitCalibration(weights="dummy", routing="balanced") if case == "dummy" else REAL
    )
    path = _table(tmp_path, calibration=calibration)
    key = "other" if case == "no_entry" else "shape"
    fingerprint = FINGERPRINT
    if case == "stale":
        fingerprint = FitFingerprint(
            profile_points=FINGERPRINT.profile_points,
            schedule=FINGERPRINT.schedule,
            kernel_version="k2",
            runtime_version=FINGERPRINT.runtime_version,
        )
    if case == "no_table":
        path = str(tmp_path / "missing.json")
    if case == "corrupt":
        (tmp_path / "fits.json").write_text('{"shape": {"linear_ms": ')
    if case == "bad_ladder":
        short = StepCostModel([2048], {2048: 30.0}, 1e-6, 0.0, 512)
        entry = saved_fit(
            short, slack=0.1, fingerprint=FINGERPRINT, calibration=REAL, guard_points=[]
        )
        upsert_saved_fit(path, "shape", entry)
    if case == "zero_guard":
        entry = saved_fit(
            MODEL,
            slack=0.1,
            fingerprint=FINGERPRINT,
            calibration=REAL,
            guard_points=[(4096, 0, 0.0)],
        )
        upsert_saved_fit(path, "shape", entry)
    bad_costs = {
        "negative_attention": ({2048: 30.0, 4096: 55.0}, -1e-6),
        "negative_linear": ({2048: -1.0, 4096: 55.0}, 1e-6),
        "zero_step": ({2048: 0.0, 4096: 0.0}, 0.0),
        # Finite terms whose full step overflows.
        "unbounded_step": ({2048: 30.0, 4096: 55.0}, 1e305),
    }
    if case in bad_costs:
        linear, attn = bad_costs[case]
        bad = StepCostModel([2048, 4096], linear, attn, 0.0, 512)
        entry = saved_fit(
            bad,
            slack=0.1,
            fingerprint=FINGERPRINT,
            calibration=REAL,
            guard_points=GUARD,
        )
        upsert_saved_fit(path, "shape", entry)
    if case == "no_attention_guard":
        entry = saved_fit(
            MODEL,
            slack=0.1,
            fingerprint=FINGERPRINT,
            calibration=REAL,
            guard_points=GUARD[:2],
        )
        upsert_saved_fit(path, "shape", entry)
    if case == "no_guard":
        entry = saved_fit(
            MODEL, slack=0.1, fingerprint=FINGERPRINT, calibration=REAL, guard_points=[]
        )
        upsert_saved_fit(path, "shape", entry)
    entry, reason = load_saved_fit(
        table_path=path, key=key, fingerprint=fingerprint, check=True
    )
    assert entry is None
    assert expected in reason


def test_a_fit_with_a_zero_linear_term_loads(tmp_path):
    """The fitter clamps a bucket's linear term at zero when attention dominates."""
    path = str(tmp_path / "fits.json")
    zero = StepCostModel([2048, 4096], {2048: 0.0, 4096: 55.0}, 1e-6, 0.0, 512)
    upsert_saved_fit(
        path,
        "shape",
        saved_fit(
            zero,
            slack=0.1,
            fingerprint=FINGERPRINT,
            calibration=REAL,
            guard_points=GUARD,
        ),
    )
    entry, reason = load_saved_fit(
        table_path=path, key="shape", fingerprint=FINGERPRINT, check=True
    )
    assert entry is not None, reason


def test_guard_points_cover_both_ends_and_the_attention_term():
    """Drift at either end of the ladder, or in the attention cost alone,
    must reach the guard."""
    points = [(512, 0), (2048, 0), (2048, 8192), (4096, 0), (4096, 8192), (4096, 16384)]
    assert guard_points(points) == [(512, 0), (4096, 0), (4096, 16384)]
    # Near the context length the top bucket has no prefix; a smaller one stands in.
    short = [(2048, 0), (2048, 8192), (2048, 16384), (4096, 0), (8192, 0)]
    assert guard_points(short) == [(2048, 0), (2048, 16384), (8192, 0)]
    # Without a prefixed point there is nothing to time the attention term on.
    assert guard_points([(512, 0), (4096, 0)]) == [(512, 0), (4096, 0)]


def test_guard_tolerance_is_capped_below_a_wide_slack():
    """A wide slack must not widen the guard past 10%."""
    saved = [(4096, 0, 100.0)]
    _, drift = guard_report(saved, [(4096, 0, 112.0)], slack=0.2)
    assert drift


def test_a_bad_entry_costs_only_its_own_key(tmp_path):
    """One bad or newer-schema entry in a shared table must not turn saved fits
    off for every other shape; upserting its key replaces it."""
    import json

    path = _table(tmp_path)
    table = json.loads(open(path).read())
    table["broken"] = {"linear_ms": "not a ladder"}
    table["newer"] = {**table["shape"], "field_from_a_newer_schema": 1}
    open(path, "w").write(json.dumps(table))

    def load(key):
        return load_saved_fit(
            table_path=path, key=key, fingerprint=FINGERPRINT, check=True
        )

    entry, reason = load("shape")
    assert entry is not None, reason
    for key in ("broken", "newer"):
        entry, reason = load(key)
        assert entry is None
        assert f"saved fit for {key} is unreadable" in reason
    assert sorted(read_saved_fits(path)[1]) == ["broken", "newer"]

    good = load("shape")[0]
    upsert_saved_fit(path, "broken", good)
    assert load("broken")[0] == good
    # The other bad entry stays as written, for its owner to replace.
    assert sorted(read_saved_fits(path)[1]) == ["newer"]


@pytest.mark.parametrize("text", ['{"shape": {"linear_ms": ', '["not", "a", "table"]'])
def test_a_table_that_does_not_parse_fails_as_a_whole(tmp_path, text):
    path = _table(tmp_path)
    open(path, "w").write(text)
    entry, reason = load_saved_fit(
        table_path=path, key="shape", fingerprint=FINGERPRINT, check=True
    )
    assert entry is None
    assert "cannot read fit table" in reason


def test_upsert_through_a_symlink_updates_its_target(tmp_path):
    """A table published through a symlink must stay a symlink."""
    target = tmp_path / "tables" / "fits.json"
    target.parent.mkdir()
    upsert_saved_fit(str(target), "shape", _entry())
    link = tmp_path / "fits.json"
    link.symlink_to(target)
    upsert_saved_fit(str(link), "other", _entry())
    assert link.is_symlink()
    assert sorted(read_saved_fits(str(target))[0]) == ["other", "shape"]


def test_a_failed_upsert_leaves_no_staging_file(monkeypatch, tmp_path):
    import sglang_tpu.pp_chunk_fit as pp_chunk_fit

    def full_disk(src, dst):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(pp_chunk_fit.os, "replace", full_disk)
    with pytest.raises(OSError):
        upsert_saved_fit(str(tmp_path / "fits.json"), "shape", _entry())
    assert list(tmp_path.iterdir()) == []


def test_an_unreadable_table_is_a_vote_not_an_error(monkeypatch, tmp_path):
    """A rank that raises before the saved-fit vote leaves the others waiting in
    its gather, so any read failure must come back as a reason."""
    import pathlib

    def denied(self):
        raise PermissionError(13, "Permission denied", str(self))

    monkeypatch.setattr(pathlib.Path, "is_file", denied)
    entry, reason = load_saved_fit(
        table_path=str(tmp_path / "fits.json"),
        key="shape",
        fingerprint=FINGERPRINT,
        check=True,
    )
    assert entry is None
    assert "cannot read fit table" in reason


def test_vacuous_fingerprint_fields_are_named():
    """An empty kernel version or an unknown package version matches every
    entry, so the launch must say the guard is the only check left."""
    weak = FitFingerprint(
        profile_points=FINGERPRINT.profile_points,
        schedule=None,
        kernel_version="",
        runtime_version="torch_tpu=none;libtpu=2",
    )
    assert weak_fingerprint_fields(weak) == [
        "kernel_version is empty",
        "runtime_version is torch_tpu=none;libtpu=2",
    ]
    assert weak_fingerprint_fields(FINGERPRINT) == []


def test_guard_reports_drift_at_every_point():
    saved = [(4096, 0, 50.0), (4096, 4096, 100.0)]
    timed = [(4096, 0, 52.0), (4096, 4096, 150.0), (2048, 0, 10.0)]
    report, drifted = guard_report(saved, timed, slack=0.1)
    assert report == [
        "(4096, 0) measured 52.0 ms, saved 50.0 ms (+4.0%)",
        "(4096, 4096) measured 150.0 ms, saved 100.0 ms (+50.0%)",
        "(2048, 0) measured 10.0 ms, no calibrated time",
    ]
    assert drifted == report[1:]
    # A slack below the tolerance tightens it: +4% leaves the step past a 3% target.
    _, tight = guard_report(saved, timed, slack=0.03)
    assert tight == report
    # A zero slack keeps a noise floor instead of rejecting every fit.
    _, floor = guard_report([(4096, 0, 50.0)], [(4096, 0, 50.5)], slack=0.0)
    assert floor == []


def test_upsert_keeps_other_entries_sorted(tmp_path):
    path = _table(tmp_path)
    upsert_saved_fit(
        path,
        "another",
        saved_fit(
            MODEL, slack=0.1, fingerprint=FINGERPRINT, calibration=REAL, guard_points=[]
        ),
    )
    text = (tmp_path / "fits.json").read_text()
    assert text.index('"another"') < text.index('"shape"')
    for key in ("shape", "another"):
        entry, _ = load_saved_fit(
            table_path=path, key=key, fingerprint=FINGERPRINT, check=False
        )
        assert entry.linear_ms == MODEL.linear_ms


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
