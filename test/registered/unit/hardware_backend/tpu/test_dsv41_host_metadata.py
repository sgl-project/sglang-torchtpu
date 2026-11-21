# SPDX-License-Identifier: Apache-2.0
"""CPU tests: the DSV4.1 TPU attention metadata reads the host tensors the compile
policy uploaded instead of reading the uploads back, and stays bit-identical."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from sglang.srt.model_executor.forward_batch_info import (
    ForwardMode,
    compute_position_torch,
)
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
    # Codex review: a missing prefix list zero-filled by padding must not change
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
