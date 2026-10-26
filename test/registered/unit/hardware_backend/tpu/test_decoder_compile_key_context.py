"""The compiled decode step keyed by a backend's static facts.

``PallasForwardMetadata.compile_key`` enters the decoder compile policy's
specialization key, and the warm-up runs one more pass per context length the
attention backend asks for (``warmup_context_lens``), through a hybrid wrapper
too, so every program a context-keyed backend will select is compiled before
serving.
"""

import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu.compilation import decoder_compile_policy as policy

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# The context length the fake backend keys its program by.
T = 131072


def _batch(**fields):
    base = dict(
        forward_mode=ForwardMode.DECODE,
        capture_hidden_mode=CaptureHiddenMode.NULL,
        batch_size=2,
        logits_output_mode="full",
        seq_lens=torch.tensor([5, 9]),
        orig_seq_lens=torch.tensor([5, 9]),
        seq_lens_cpu=torch.tensor([5, 9]),
        seq_lens_sum=14,
    )
    base.update(fields)
    return SimpleNamespace(**base)


def test_specialization_key_appends_the_metadatas_compile_key():
    """Both base key definitions append ``attention_metadata.compile_key``;
    metadata without one (every other backend) keys as before."""
    me = SimpleNamespace(_compiles_logits_processor=lambda: False)
    ids = torch.zeros(2, dtype=torch.int64)
    batch = _batch()
    plain = policy.DecoderTpuCompilePolicy._direct_compile_specialization_key(
        me, SimpleNamespace(), ids, ids, batch
    )
    short = policy.DecoderTpuCompilePolicy._direct_compile_specialization_key(
        me, SimpleNamespace(compile_key=(0,)), ids, ids, batch
    )
    long = policy.DecoderTpuCompilePolicy._direct_compile_specialization_key(
        me, SimpleNamespace(compile_key=(1,)), ids, ids, batch
    )
    assert plain == (ForwardMode.DECODE, CaptureHiddenMode.NULL, 2, 2, None)
    assert short == (*plain, 0) and long == (*plain, 1)
    vl = policy.Qwen3VLDecoderTpuCompilePolicy._direct_compile_specialization_key(
        me, SimpleNamespace(compile_key=(1,)), ids, ids, batch
    )
    assert vl[-1] == 1 and vl[:-1] == (
        ForwardMode.DECODE,
        CaptureHiddenMode.NULL,
        2,
        2,
        False,
        False,
        None,
    )


def test_pallas_forward_metadata_compile_key_defaults_empty():
    """A metadata that declares nothing keys the forward as before."""
    from sglang_tpu.attention_pallas_rpa_base import PallasForwardMetadata

    field = PallasForwardMetadata.__dataclass_fields__["compile_key"]
    assert field.default == ()


def test_warmup_context_helpers():
    """``warmup_context_lens`` reads the backend's request (empty by default);
    ``set_warmup_context`` gives the dummy batch the pass's context on the
    device and host lengths, leaving absent host mirrors absent."""
    assert policy.warmup_context_lens(SimpleNamespace()) == ()
    assert policy.warmup_context_lens(SimpleNamespace(warmup_context_lens=None)) == ()
    assert policy.warmup_context_lens(SimpleNamespace(warmup_context_lens=(T,))) == (T,)
    batch = _batch()
    policy.set_warmup_context(batch, T)
    assert batch.seq_lens.tolist() == [T, T] and batch.seq_lens_cpu.tolist() == [T, T]
    assert batch.orig_seq_lens.tolist() == [T, T] and batch.seq_lens_sum == 2 * T
    device_only = _batch(seq_lens_cpu=None, seq_lens_sum=None, orig_seq_lens=None)
    policy.set_warmup_context(device_only, T)
    assert device_only.seq_lens.tolist() == [T, T]
    assert device_only.seq_lens_cpu is None and device_only.seq_lens_sum is None
    assert device_only.orig_seq_lens is None


class _Backend:
    """A backend whose compiled graph is keyed by the context: records the
    context it saw while building each pass's metadata."""

    warmup_context_len = None

    def __init__(self, lens=()):
        self.warmup_context_lens = lens
        self.built = []
        self.forward_metadata = None

    def init_forward_metadata(self, forward_batch):
        self.built.append(self.warmup_context_len)
        self.forward_metadata = SimpleNamespace(
            compile_key=(int((self.warmup_context_len or 0) >= T),)
        )


class _Hybrid:
    """A hybrid wrapper as the warm-up sees it: delegates by forward mode,
    exposes the active child's metadata, declares no hooks of its own."""

    def __init__(self, prefill, decode):
        self.prefill, self.decode, self.active = prefill, decode, prefill
        self.inits = 0

    def _select_backend(self, mode):
        if mode.is_decode_or_idle() or mode.is_target_verify():
            return self.decode
        return self.prefill

    def init_forward_metadata(self, forward_batch):
        self.active = self._select_backend(forward_batch.forward_mode)
        self.active.init_forward_metadata(forward_batch)
        self.inits += 1

    @property
    def forward_metadata(self):
        return self.active.forward_metadata


def _policy(monkeypatch, backend):
    """A policy running the real ``_run_one`` over a forward that records the
    key the backend's metadata would give each pass and the pass's context."""
    monkeypatch.setattr(policy, "is_dp_attention_enabled", lambda: False)
    monkeypatch.setattr(policy, "_materialize_out", lambda out, backend, pool: None)
    keys = []

    def forward(input_ids, positions, forward_batch, **kwargs):
        keys.append(
            (backend.forward_metadata.compile_key, int(forward_batch.seq_lens.max()))
        )

    ids = torch.zeros(2, dtype=torch.int64)
    me = SimpleNamespace(
        keys=keys,
        tracer="dynamo",
        owner=SimpleNamespace(forward=forward),
        _compiled_extend_variants=set(),
        _make_warmup_inputs=lambda **kw: (
            (ids, ids, _batch(forward_mode=kw["mode"])),
            {},
        ),
        _seed_dummy_req_to_token=lambda batch: None,
        model_runner=SimpleNamespace(
            attn_backend=backend,
            prepare_dummy_forward_batch=lambda batch: batch,
            model=SimpleNamespace(
                prepare_forward_batch=lambda batch: None, forward=forward
            ),
            token_to_kv_pool=None,
        ),
    )
    me._run_one = lambda **kw: policy.DecoderTpuCompilePolicy._run_one(me, **kw)
    return me


def test_warmup_without_a_context_request_runs_one_pass(monkeypatch):
    """A backend that asks for nothing (the default) keeps its single pass and
    never sees a ``warmup_context_len``."""
    backend = _Backend()
    me = _policy(monkeypatch, backend)
    assert (
        me._run_one(num_tokens=4, bs=2, mode=ForwardMode.DECODE, spec_info=object())
        is None
    )
    assert backend.built == [None] and me.keys == [((0,), 9)]


def test_warmup_context_passes_on_a_plain_backend(monkeypatch):
    """A backend that is its own metadata builder runs the as-built pass and
    one pass per declared context, seeing each pass's length while it builds
    the metadata and nothing afterwards."""
    backend = _Backend((T,))
    assert policy.warmup_metadata_backend(backend, ForwardMode.DECODE) is backend
    me = _policy(monkeypatch, backend)
    me._run_one(num_tokens=4, bs=2, mode=ForwardMode.DECODE, spec_info=object())
    assert backend.built == [None, T] and backend.warmup_context_len is None
    assert me.keys == [((0,), 9), ((1,), T)]


def test_warmup_context_passes_reach_the_hybrid_wrappers_child(monkeypatch):
    """Behind a hybrid wrapper the warm-up's context questions go to the child
    that builds the pass's metadata, the wrapper declaring nothing itself,
    while the build still goes through the wrapper, which activates that
    child for the forward: the decode child's long-context program is
    compiled before traffic; a child asking for nothing keeps its single
    pass."""
    prefill, decode = _Backend(), _Backend((T,))
    hybrid = _Hybrid(prefill, decode)
    assert policy.warmup_metadata_backend(hybrid, ForwardMode.TARGET_VERIFY) is decode
    assert policy.warmup_metadata_backend(hybrid, ForwardMode.EXTEND) is prefill
    me = _policy(monkeypatch, hybrid)
    me._run_one(num_tokens=16, bs=2, mode=ForwardMode.TARGET_VERIFY, spec_info=object())
    assert decode.built == [None, T] and decode.warmup_context_len is None
    assert me.keys == [((0,), 9), ((1,), T)]
    assert hybrid.inits == 2 and prefill.built == []
    me._run_one(num_tokens=16, bs=2, mode=ForwardMode.EXTEND, spec_info=object())
    assert prefill.built == [None] and me.keys[2:] == [((0,), 9)]


def test_profile_runs_skip_the_context_passes(monkeypatch):
    """A profiling run (``profile_prefix`` set) times one pass as built; the
    extra context passes are a warm-up concern."""
    backend = _Backend((T,))
    me = _policy(monkeypatch, backend)
    monkeypatch.setattr(
        "sglang_tpu.pp_profile.set_profile_inputs",
        lambda policy_, batch, args, kwargs, num_tokens, prefix: (args, kwargs),
    )
    elapsed = me._run_one(
        num_tokens=4,
        bs=2,
        mode=ForwardMode.DECODE,
        spec_info=object(),
        profile_prefix=0,
    )
    assert isinstance(elapsed, float)
    assert backend.built == [None] and me.keys == [((0,), 9)]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
