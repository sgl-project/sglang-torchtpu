"""Unit tests for the TPU compiled-forward shim."""

import inspect
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.hardware_backend.tpu.compilation.bucketer import (
    Bucketer,
    PowerBucketer,
)
from sglang.srt.hardware_backend.tpu.compilation.compiled_forward_shim import (
    CompiledForwardShim,
    TpuCompiledForwardShim,
    _matches_default,
    _pad_forward_batch_inputs,
    _static_default_kwargs,
    _zero_pad_1d,
    build_bucketers,
    resolve_compile_caps,
)
from sglang.test.test_utils import CustomTestCase


def _t(*vals, dtype=torch.int32):
    return torch.tensor(list(vals), dtype=dtype)


def _bucketer(name, buckets):
    return Bucketer(name=name, buckets=buckets)


class _ForwardModeMock:
    """Minimal forward_mode stand-in for unit tests."""

    def __init__(self, mode):
        self._mode = mode

    def is_decode(self):
        return self._mode == "decode"


def _make_compiled(return_value, *, signature_of=None):
    """Build a MagicMock whose ``__signature__`` matches a real model
    method. The shim looks up ``forward_batch`` by parameter name, so
    bare MagicMocks miss the fast path and fall back to ``isinstance``;
    tests that exercise the slicing logic need a named ``forward_batch``
    in the signature.

    ``signature_of`` accepts a real callable (e.g. a stand-in
    ``model.forward`` shape) and copies its signature onto the mock.
    Defaults to the canonical body-forward signature.
    """

    def _default_body_forward(input_ids, positions, forward_batch):
        pass

    sig_source = signature_of if signature_of is not None else _default_body_forward
    compiled = MagicMock(return_value=return_value)
    compiled.__signature__ = inspect.signature(sig_source)
    return compiled


def _forward_batch(real_t, real_b, mode="extend", return_logprob=False):
    """Synthetic ForwardBatch carrying the fields the padder touches."""
    extend = mode == "extend"
    return SimpleNamespace(
        input_ids=torch.arange(real_t, dtype=torch.int32),
        positions=torch.arange(real_t, dtype=torch.int32),
        out_cache_loc=torch.arange(real_t, dtype=torch.int32),
        seq_lens=torch.full((real_b,), 100, dtype=torch.int32),
        orig_seq_lens=torch.full((real_b,), 100, dtype=torch.int32),
        req_pool_indices=torch.arange(real_b, dtype=torch.int64),
        extend_seq_lens=(
            torch.full((real_b,), real_t // real_b, dtype=torch.int32)
            if extend
            else None
        ),
        extend_prefix_lens=(torch.zeros(real_b, dtype=torch.int32) if extend else None),
        extend_seq_lens_cpu=([real_t // real_b] * real_b if extend else None),
        extend_prefix_lens_cpu=([0] * real_b if extend else None),
        extend_logprob_start_lens_cpu=([0] * real_b if extend else None),
        forward_mode=_ForwardModeMock(mode),
        return_logprob=return_logprob,
    )


class TestZeroPad1d(CustomTestCase):
    def test_pads_zeros_at_tail(self):
        out = _zero_pad_1d(_t(1, 2, 3), target_size=5)
        self.assertEqual(out.tolist(), [1, 2, 3, 0, 0])

    def test_no_op_when_already_at_size(self):
        x = _t(1, 2, 3)
        self.assertIs(_zero_pad_1d(x, target_size=3), x)
        self.assertIs(_zero_pad_1d(x, target_size=2), x)


class TestPadForwardBatchInputs(CustomTestCase):
    """Pads ForwardBatch fields to bucket size before init_forward_metadata."""

    def test_extend_single_req(self):
        # tokens=4 -> bucket 8; bs=1 already a bucket.
        fb = _forward_batch(real_t=4, real_b=1, mode="extend")
        real_t, real_b = _pad_forward_batch_inputs(
            fb,
            t_bucketer=_bucketer("Token", [8, 64]),
            b_bucketer=_bucketer("Bs", [1, 4]),
        )
        self.assertEqual((real_t, real_b), (4, 1))
        self.assertEqual(fb.input_ids.shape[0], 8)
        self.assertEqual(fb.positions.shape[0], 8)
        self.assertEqual(fb.out_cache_loc.shape[0], 4)
        self.assertEqual(fb.seq_lens.shape[0], 1)
        self.assertEqual(fb.extend_seq_lens.shape[0], 1)

    def test_extend_batched(self):
        # tokens=10 -> 16; bs=3 -> 4; ghost slots zero-filled.
        fb = _forward_batch(real_t=10, real_b=3, mode="extend")
        real_t, real_b = _pad_forward_batch_inputs(
            fb,
            t_bucketer=_bucketer("Token", [16, 64]),
            b_bucketer=_bucketer("Bs", [1, 4]),
        )
        self.assertEqual((real_t, real_b), (10, 3))
        self.assertEqual(fb.input_ids.shape[0], 16)
        self.assertEqual(fb.seq_lens.shape[0], 4)
        self.assertEqual(fb.req_pool_indices.shape[0], 4)
        self.assertEqual(fb.extend_seq_lens.shape[0], 4)
        self.assertEqual(fb.seq_lens[:3].tolist(), [100, 100, 100])
        self.assertEqual(fb.seq_lens[3].item(), 0)
        self.assertEqual(fb.extend_seq_lens[3].item(), 0)
        self.assertEqual(fb.extend_seq_lens_cpu, [3, 3, 3, 0])
        self.assertEqual(fb.extend_prefix_lens_cpu, [0, 0, 0, 0])
        self.assertEqual(fb.extend_logprob_start_lens_cpu, [0, 0, 0, 0])
        self.assertEqual(fb.batch_size, 4)

    def test_decode_uses_bs_bucketer_for_tokens(self):
        # In DECODE, token-axis tensors share the bs bucketer
        # (one query token per running seq).
        fb = _forward_batch(real_t=3, real_b=3, mode="decode")
        _pad_forward_batch_inputs(
            fb,
            t_bucketer=_bucketer("Token", [16, 64]),  # would over-pad if (wrongly) used
            b_bucketer=_bucketer("Bs", [1, 4, 8]),
        )
        self.assertEqual(fb.input_ids.shape[0], 4)
        self.assertEqual(fb.positions.shape[0], 4)
        self.assertEqual(fb.seq_lens.shape[0], 4)

    def test_no_op_when_at_bucket(self):
        # Shapes already at bucket: tensors unchanged (same identity).
        fb = _forward_batch(real_t=4, real_b=1, mode="extend")
        orig_input_ids = fb.input_ids
        orig_seq_lens = fb.seq_lens
        _pad_forward_batch_inputs(
            fb,
            t_bucketer=_bucketer("Token", [4]),
            b_bucketer=_bucketer("Bs", [1]),
        )
        self.assertIs(fb.input_ids, orig_input_ids)
        self.assertIs(fb.seq_lens, orig_seq_lens)

    def test_optional_fields_left_alone_when_none(self):
        # extend_* are None in DECODE; padder must not crash.
        fb = _forward_batch(real_t=2, real_b=2, mode="decode")
        self.assertIsNone(fb.extend_seq_lens)
        _pad_forward_batch_inputs(
            fb,
            t_bucketer=_bucketer("Token", [16]),
            b_bucketer=_bucketer("Bs", [4]),
        )
        self.assertIsNone(fb.extend_seq_lens)

    def test_return_logprob_raises(self):
        fb = _forward_batch(real_t=4, real_b=1, return_logprob=True)
        with self.assertRaisesRegex(NotImplementedError, "return_logprob"):
            _pad_forward_batch_inputs(
                fb,
                t_bucketer=_bucketer("Token", [64]),
                b_bucketer=_bucketer("Bs", [1]),
            )


class TestDefaultKwargHelpers(CustomTestCase):
    def test_static_default_kwargs_filters_to_simple_defaults(self):
        # ``_static_default_kwargs`` now takes a ``Signature`` directly —
        # the caller is responsible for resolving it. Containers and
        # custom-class defaults are filtered out (only None / bool / int /
        # float / str are considered "Dynamo-collapsible").
        def forward(a, b=None, c=False, d=3, e=1.5, f="x", g=torch.zeros(1)):
            pass

        sig = inspect.signature(forward)
        self.assertEqual(
            _static_default_kwargs(sig),
            {"b": None, "c": False, "d": 3, "e": 1.5, "f": "x"},
        )

    def test_static_default_kwargs_no_defaults_returns_empty(self):
        def forward(a, b, c):
            pass

        self.assertEqual(_static_default_kwargs(inspect.signature(forward)), {})

    def test_matches_default_is_type_strict(self):
        self.assertTrue(_matches_default(None, None))
        self.assertTrue(_matches_default(3, 3))
        self.assertFalse(_matches_default(False, 0))
        self.assertFalse(_matches_default(3.0, 3))


class TestCompiledForwardShim(CustomTestCase):
    """Adapts compiled forward calls and outputs."""

    def test_slices_logits_and_hidden_states(self):
        padded_logits = torch.arange(4 * 10, dtype=torch.float32).reshape(4, 10)
        padded_hidden = torch.arange(8 * 5, dtype=torch.float32).reshape(8, 5)
        compiled = _make_compiled(
            SimpleNamespace(
                next_token_logits=padded_logits,
                hidden_states=padded_hidden,
            )
        )
        shim = CompiledForwardShim(compiled)
        fb = SimpleNamespace(_tpu_real_t=4, _tpu_real_b=2)
        out = shim(None, None, fb)
        self.assertEqual(out.next_token_logits.shape, (2, 10))
        self.assertEqual(out.hidden_states.shape, (4, 5))
        self.assertTrue(torch.equal(out.next_token_logits, padded_logits[:2]))
        self.assertTrue(torch.equal(out.hidden_states, padded_hidden[:4]))

    def test_passes_args_through(self):
        compiled = _make_compiled(SimpleNamespace())
        shim = CompiledForwardShim(compiled)
        fb = SimpleNamespace(_tpu_real_t=4, _tpu_real_b=2)
        shim("ids", "pos", fb, "extra1", kw="extra2")
        compiled.assert_called_once_with("ids", "pos", fb, "extra1", kw="extra2")

    def test_preserves_compiled_signature_for_later_inspection(self):
        def forward(input_ids, positions, forward_batch, pp_proxy_tensors=None):
            pass

        compiled = MagicMock(return_value=SimpleNamespace())
        compiled.__signature__ = inspect.signature(forward)
        shim = CompiledForwardShim(compiled)
        self.assertIn("pp_proxy_tensors", inspect.signature(shim).parameters)

    def test_drops_default_kwargs_before_compiled_call(self):
        def forward(input_ids, positions, forward_batch, pp_proxy_tensors=None):
            pass

        compiled = MagicMock(return_value=SimpleNamespace())
        compiled.__signature__ = inspect.signature(forward)
        shim = CompiledForwardShim(compiled)
        fb = SimpleNamespace(_tpu_real_t=4, _tpu_real_b=2)
        shim("ids", "pos", fb, pp_proxy_tensors=None)
        compiled.assert_called_once_with("ids", "pos", fb)

    def test_keeps_non_default_kwargs_before_compiled_call(self):
        def forward(input_ids, positions, forward_batch, get_embedding=False):
            pass

        compiled = MagicMock(return_value=SimpleNamespace())
        compiled.__signature__ = inspect.signature(forward)
        shim = CompiledForwardShim(compiled)
        fb = SimpleNamespace(_tpu_real_t=4, _tpu_real_b=2)
        shim("ids", "pos", fb, get_embedding=True)
        compiled.assert_called_once_with("ids", "pos", fb, get_embedding=True)

    def test_no_op_when_real_size_keys_missing(self):
        # Padding hook didn't run (e.g. forward_idle): output passes through.
        out_obj = SimpleNamespace(
            next_token_logits=torch.zeros(4, 10),
            hidden_states=torch.zeros(8, 5),
        )
        shim = CompiledForwardShim(_make_compiled(out_obj))
        out = shim(None, None, SimpleNamespace())
        self.assertIs(out, out_obj)

    def test_no_op_when_already_at_real_shape(self):
        # No padding to slice off when shapes already match.
        out_obj = SimpleNamespace(
            next_token_logits=torch.zeros(2, 10),
            hidden_states=torch.zeros(4, 5),
        )
        shim = CompiledForwardShim(_make_compiled(out_obj))
        fb = SimpleNamespace(_tpu_real_t=4, _tpu_real_b=2)
        out = shim(None, None, fb)
        self.assertEqual(out.next_token_logits.shape, (2, 10))
        self.assertEqual(out.hidden_states.shape, (4, 5))

    def test_restores_real_batch_size_on_forward_batch(self):
        # Compiled-forward padding set batch_size to pad_b; the
        # shim must restore real_b so downstream consumers (sampler,
        # logprob, scheduler hooks) don't iterate over ghost slots.
        compiled = _make_compiled(
            SimpleNamespace(next_token_logits=torch.zeros(4, 10))
        )
        shim = CompiledForwardShim(compiled)
        fb = SimpleNamespace(_tpu_real_t=8, _tpu_real_b=2, batch_size=4)
        shim(None, None, fb)
        self.assertEqual(fb.batch_size, 2)

    def test_does_not_touch_batch_size_when_real_keys_missing(self):
        # forward_idle / split_prefill skip the pad hook, so the slicer
        # must not mutate batch_size in that case.
        fb = SimpleNamespace(batch_size=4)
        compiled = _make_compiled(SimpleNamespace())
        shim = CompiledForwardShim(compiled)
        shim(None, None, fb)
        self.assertEqual(fb.batch_size, 4)


class TestBuildBucketers(CustomTestCase):
    @staticmethod
    def _server_args(tpu_compile_tokens=None, tpu_compile_bs=None):
        return SimpleNamespace(
            tpu_compile_tokens=tpu_compile_tokens,
            tpu_compile_bs=tpu_compile_bs,
        )

    def test_uses_user_buckets_as_plain_bucketer(self):
        # User list yields a plain Bucketer (not the PowerBucketer fallback).
        sa = self._server_args(
            tpu_compile_tokens=[64, 512, 2048],
            tpu_compile_bs=[1, 4, 16],
        )
        t_b, b_b = build_bucketers(sa, max_t=2048, max_b=16)
        self.assertEqual(t_b.buckets, [64, 512, 2048])
        self.assertEqual(b_b.buckets, [1, 4, 16])
        self.assertIsInstance(t_b, Bucketer)
        self.assertNotIsInstance(t_b, PowerBucketer)

    def test_explicit_buckets_used_verbatim(self):
        # Explicit lists are trusted as-is; out-of-range runtime shapes raise.
        sa = self._server_args(
            tpu_compile_tokens=[64, 512],
            tpu_compile_bs=[1, 2],
        )
        t_b, b_b = build_bucketers(sa, max_t=2048, max_b=16)
        self.assertEqual(t_b.buckets, [64, 512])
        self.assertEqual(b_b.buckets, [1, 2])

    def test_falls_back_to_power_bucketer_when_unset(self):
        # min_val hardcoded: 64 (T), 1 (B).
        sa = self._server_args()
        t_b, b_b = build_bucketers(sa, max_t=2048, max_b=8)
        self.assertIsInstance(t_b, PowerBucketer)
        self.assertIsInstance(b_b, PowerBucketer)
        self.assertEqual(t_b.min_val, 64)
        self.assertEqual(t_b.max_val, 2048)
        self.assertEqual(b_b.min_val, 1)
        self.assertEqual(b_b.max_val, 8)

    def test_user_t_list_with_power_bs_fallback(self):
        # Mixed: user supplied T but not B.
        sa = self._server_args(tpu_compile_tokens=[64, 2048])
        t_b, b_b = build_bucketers(sa, max_t=2048, max_b=8)
        self.assertNotIsInstance(t_b, PowerBucketer)
        self.assertIsInstance(b_b, PowerBucketer)
        self.assertEqual(t_b.buckets, [64, 2048])
        self.assertEqual(b_b.buckets, [1, 2, 4, 8])


class TestResolveCompileCaps(CustomTestCase):
    """``resolve_compile_caps`` picks the **min** of the set caps, not the
    first one — so an over-large ``chunked_prefill_size`` doesn't
    bucket past the actual per-forward cap."""

    def test_min_of_chunked_and_max_prefill(self):
        # chunked=8192 is bigger than max_prefill=2048, but the scheduler
        # would still cap at 2048; bucketing up to 8192 would compile a
        # shape that never runs.
        max_t, max_b = resolve_compile_caps(
            chunked_prefill_size=8192,
            max_prefill_tokens=2048,
            context_len=32000,
            torch_compile_max_bs=8,
            max_running_requests=8,
        )
        self.assertEqual(max_t, 2048)
        self.assertEqual(max_b, 8)

    def test_skips_none_and_zero_candidates(self):
        max_t, _ = resolve_compile_caps(
            chunked_prefill_size=None,
            max_prefill_tokens=0,  # treated as unset
            context_len=4096,
            torch_compile_max_bs=1,
            max_running_requests=1,
        )
        self.assertEqual(max_t, 4096)

    def test_max_b_falls_back_to_one(self):
        _, max_b = resolve_compile_caps(
            chunked_prefill_size=128,
            max_prefill_tokens=None,
            context_len=None,
            torch_compile_max_bs=None,
            max_running_requests=None,
        )
        self.assertEqual(max_b, 1)

    def test_max_b_prefers_torch_compile_max_bs(self):
        # The compile knob bounds the batch grid.
        _, max_b = resolve_compile_caps(
            chunked_prefill_size=2048,
            max_prefill_tokens=2048,
            context_len=32000,
            torch_compile_max_bs=32,
            max_running_requests=2048,
        )
        self.assertEqual(max_b, 32)

    def test_raises_when_no_token_cap_set(self):
        with self.assertRaisesRegex(ValueError, "Cannot determine max_t"):
            resolve_compile_caps(
                chunked_prefill_size=None,
                max_prefill_tokens=None,
                context_len=None,
                torch_compile_max_bs=4,
                max_running_requests=4,
            )


class TestCompiledForwardShimInstall(CustomTestCase):
    """Verify TpuCompiledForwardShim installs the runtime wrappers."""

    @staticmethod
    def _model_runner_with(model):
        return SimpleNamespace(
            model=model,
            model_config=SimpleNamespace(context_len=128),
            max_running_requests=1,
            server_args=SimpleNamespace(
                tpu_compile_tokens=None,
                tpu_compile_bs=None,
                chunked_prefill_size=None,
                max_prefill_tokens=128,
            ),
        )

    def test_installs_prepare_and_compiled_forward(self):
        class _M:
            pass

        model = _M()
        original_forward = MagicMock()
        model.forward = original_forward
        runner = self._model_runner_with(model)

        TpuCompiledForwardShim(
            runner,
            output_targets=[(model, "forward", original_forward)],
        ).install()

        self.assertTrue(callable(getattr(model, "prepare_forward_batch", None)))
        self.assertIsInstance(model.forward, CompiledForwardShim)
        self.assertIs(model.forward.fn, original_forward)

    def test_chains_existing_prepare_forward_batch(self):
        class _M:
            pass

        model = _M()
        existing_prep = MagicMock()
        model.prepare_forward_batch = existing_prep
        model.forward = MagicMock()
        runner = self._model_runner_with(model)

        TpuCompiledForwardShim(
            runner,
            output_targets=[(model, "forward", model.forward)],
        ).install()
        self.assertIsNot(model.prepare_forward_batch, existing_prep)

        fb = _forward_batch(real_t=4, real_b=1, mode="extend")
        model.prepare_forward_batch(fb)
        existing_prep.assert_called_once_with(fb)
        self.assertEqual(fb._tpu_real_t, 4)
        self.assertEqual(fb._tpu_real_b, 1)

    def test_install_wraps_multiple_targets_on_different_owners(self):
        """Split-compile lands the shim on ``model.logits_processor.forward``;
        verify multi-owner ``output_targets`` works (the case the dotted-attr
        workaround used to fail on)."""

        class _Body:
            pass

        class _LP:
            pass

        class _Outer:
            pass

        model = _Outer()
        model.model = _Body()
        model.logits_processor = _LP()

        body_fn = MagicMock()
        lp_fn = MagicMock()
        model.model.forward = body_fn
        model.logits_processor.forward = lp_fn
        runner = self._model_runner_with(model)

        TpuCompiledForwardShim(
            runner,
            output_targets=[
                (model.model, "forward", body_fn),
                (model.logits_processor, "forward", lp_fn),
            ],
        ).install()

        self.assertIsInstance(model.model.forward, CompiledForwardShim)
        self.assertIsInstance(model.logits_processor.forward, CompiledForwardShim)
        self.assertIs(model.model.forward.fn, body_fn)
        self.assertIs(model.logits_processor.forward.fn, lp_fn)
        # Outer ``model.forward`` (uncompiled orchestrator) is untouched.
        self.assertFalse(hasattr(model, "forward"))


class TestCompiledForwardShimSignatureAware(CustomTestCase):
    """The unified shim locates ``forward_batch`` via signature
    introspection (one shim class for body / LP / split-prefill)."""

    def test_lp_signature_slicing(self):
        """LP signature is ``(input_ids, hidden_states, lm_head, forward_batch,
        ...)``; the shim must still slice on a 4th-positional ``forward_batch``."""

        def lp_forward(input_ids, hidden_states, lm_head, forward_batch,
                       aux_hidden_states=None):
            pass

        padded_logits = torch.arange(8 * 10, dtype=torch.float32).reshape(8, 10)
        compiled = _make_compiled(
            SimpleNamespace(next_token_logits=padded_logits, hidden_states=None),
            signature_of=lp_forward,
        )
        shim = CompiledForwardShim(compiled)

        # _fb_index reflects the LP signature (4th positional = index 3).
        self.assertEqual(shim._fb_index, 3)

        fb = SimpleNamespace(_tpu_real_t=8, _tpu_real_b=2)
        out = shim("ids", torch.zeros(8, 16), "lm_head", fb)
        self.assertEqual(out.next_token_logits.shape, (2, 10))

    def test_forward_batch_via_kwarg(self):
        """``forward_batch=`` passed as a keyword still routes to the slicer."""
        padded_logits = torch.zeros(4, 10)
        compiled = _make_compiled(
            SimpleNamespace(next_token_logits=padded_logits, hidden_states=None)
        )
        shim = CompiledForwardShim(compiled)
        fb = SimpleNamespace(_tpu_real_t=4, _tpu_real_b=2)
        out = shim("ids", "pos", forward_batch=fb)
        self.assertEqual(out.next_token_logits.shape, (2, 10))

    def test_isinstance_fallback_when_param_renamed(self):
        """If the wrapped fn names the param differently (no in-tree fn does
        today, but the shim should be robust), the call-time isinstance fallback
        recovers it from positional args."""
        # Import the real ForwardBatch and use a synthetic instance.
        from sglang.srt.model_executor.forward_batch_info import ForwardBatch

        def oddly_named(ids, hidden_states, lm_head, fb_state, **kwargs):
            pass

        compiled = _make_compiled(
            SimpleNamespace(
                next_token_logits=torch.zeros(4, 10), hidden_states=None
            ),
            signature_of=oddly_named,
        )
        shim = CompiledForwardShim(compiled)
        # `forward_batch` is not in the signature → no fast-path index cached.
        self.assertIsNone(shim._fb_index)

        # Real ForwardBatch instance (sentinel only — we never inspect any of
        # its real fields here, just isinstance).
        fb = ForwardBatch.__new__(ForwardBatch)
        fb._tpu_real_t = 4
        fb._tpu_real_b = 2
        out = shim("ids", torch.zeros(4, 8), "lm", fb)
        self.assertEqual(out.next_token_logits.shape, (2, 10))


if __name__ == "__main__":
    unittest.main()
