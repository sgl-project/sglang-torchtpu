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
        # tokens=4 -> max(tokens, bs_cap=4) -> bucket 8.
        # Extend bs pads to the fixed cap so precompile only needs one bs.
        fb = _forward_batch(real_t=4, real_b=1, mode="extend")
        real_q, real_b = _pad_forward_batch_inputs(
            fb,
            t_bucketer=_bucketer("Token", [8, 64]),
            b_bucketer=_bucketer("Bs", [1, 4]),
        )
        self.assertEqual((real_q, real_b), (4, 1))
        self.assertEqual(fb.input_ids.shape[0], 8)
        self.assertEqual(fb.positions.shape[0], 8)
        self.assertEqual(fb.out_cache_loc.shape[0], 4)
        self.assertEqual(fb.seq_lens.shape[0], 4)
        self.assertEqual(fb.extend_seq_lens.shape[0], 4)

    def test_extend_batched(self):
        # tokens=10 -> 16; bs=3 -> 4; ghost slots zero-filled.
        fb = _forward_batch(real_t=10, real_b=3, mode="extend")
        real_q, real_b = _pad_forward_batch_inputs(
            fb,
            t_bucketer=_bucketer("Token", [16, 64]),
            b_bucketer=_bucketer("Bs", [1, 4]),
        )
        self.assertEqual((real_q, real_b), (10, 3))
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
        fb = _forward_batch(real_t=4, real_b=4, mode="extend")
        orig_input_ids = fb.input_ids
        orig_seq_lens = fb.seq_lens
        _pad_forward_batch_inputs(
            fb,
            t_bucketer=_bucketer("Token", [4]),
            b_bucketer=_bucketer("Bs", [1, 4]),
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
        def forward(a, b=None, c=False, d=3, e=1.5, f="x", g=torch.zeros(1)):
            pass

        compiled = MagicMock()
        compiled.__signature__ = inspect.signature(forward)
        self.assertEqual(
            _static_default_kwargs(compiled),
            {"b": None, "c": False, "d": 3, "e": 1.5, "f": "x"},
        )

    def test_static_default_kwargs_missing_signature_returns_empty(self):
        self.assertEqual(_static_default_kwargs(MagicMock()), {})

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
        compiled = MagicMock(
            return_value=SimpleNamespace(
                next_token_logits=padded_logits,
                hidden_states=padded_hidden,
            )
        )
        shim = CompiledForwardShim(compiled)
        fb = SimpleNamespace(_tpu_real_q=4, _tpu_real_b=2)
        out = shim(None, None, fb)
        self.assertEqual(out.next_token_logits.shape, (2, 10))
        self.assertEqual(out.hidden_states.shape, (4, 5))
        self.assertTrue(torch.equal(out.next_token_logits, padded_logits[:2]))
        self.assertTrue(torch.equal(out.hidden_states, padded_hidden[:4]))

    def test_passes_args_through(self):
        compiled = MagicMock(return_value=SimpleNamespace())
        shim = CompiledForwardShim(compiled)
        fb = SimpleNamespace(_tpu_real_q=4, _tpu_real_b=2)
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
        fb = SimpleNamespace(_tpu_real_q=4, _tpu_real_b=2)
        shim("ids", "pos", fb, pp_proxy_tensors=None)
        compiled.assert_called_once_with("ids", "pos", fb)

    def test_keeps_non_default_kwargs_before_compiled_call(self):
        def forward(input_ids, positions, forward_batch, get_embedding=False):
            pass

        compiled = MagicMock(return_value=SimpleNamespace())
        compiled.__signature__ = inspect.signature(forward)
        shim = CompiledForwardShim(compiled)
        fb = SimpleNamespace(_tpu_real_q=4, _tpu_real_b=2)
        shim("ids", "pos", fb, get_embedding=True)
        compiled.assert_called_once_with("ids", "pos", fb, get_embedding=True)

    def test_no_op_when_real_size_keys_missing(self):
        # Padding hook didn't run (e.g. forward_idle): output passes through.
        out_obj = SimpleNamespace(
            next_token_logits=torch.zeros(4, 10),
            hidden_states=torch.zeros(8, 5),
        )
        shim = CompiledForwardShim(MagicMock(return_value=out_obj))
        out = shim(None, None, SimpleNamespace())
        self.assertIs(out, out_obj)

    def test_no_op_when_already_at_real_shape(self):
        # No padding to slice off when shapes already match.
        out_obj = SimpleNamespace(
            next_token_logits=torch.zeros(2, 10),
            hidden_states=torch.zeros(4, 5),
        )
        shim = CompiledForwardShim(MagicMock(return_value=out_obj))
        fb = SimpleNamespace(_tpu_real_q=4, _tpu_real_b=2)
        out = shim(None, None, fb)
        self.assertEqual(out.next_token_logits.shape, (2, 10))
        self.assertEqual(out.hidden_states.shape, (4, 5))

    def test_restores_real_batch_size_on_forward_batch(self):
        # Compiled-forward padding set batch_size to pad_b; the
        # shim must restore real_b so downstream consumers (sampler,
        # logprob, scheduler hooks) don't iterate over ghost slots.
        compiled = MagicMock(
            return_value=SimpleNamespace(next_token_logits=torch.zeros(4, 10))
        )
        shim = CompiledForwardShim(compiled)
        fb = SimpleNamespace(_tpu_real_q=8, _tpu_real_b=2, batch_size=4)
        shim(None, None, fb)
        self.assertEqual(fb.batch_size, 2)

    def test_does_not_touch_batch_size_when_real_keys_missing(self):
        # forward_idle / split_prefill skip the pad hook, so the slicer
        # must not mutate batch_size in that case.
        fb = SimpleNamespace(batch_size=4)
        compiled = MagicMock(return_value=SimpleNamespace())
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
        self.assertEqual(max_b, 64)

    def test_max_b_uses_sane_max_running_requests(self):
        _, max_b = resolve_compile_caps(
            chunked_prefill_size=2048,
            max_prefill_tokens=2048,
            context_len=32000,
            torch_compile_max_bs=32,
            max_running_requests=64,
        )
        self.assertEqual(max_b, 64)

    def test_max_b_honors_larger_torch_compile_max_bs(self):
        _, max_b = resolve_compile_caps(
            chunked_prefill_size=2048,
            max_prefill_tokens=2048,
            context_len=32000,
            torch_compile_max_bs=128,
            max_running_requests=64,
        )
        self.assertEqual(max_b, 128)

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
            runner, compiled_fns={"forward": original_forward}
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
            runner, compiled_fns={"forward": model.forward}
        ).install()
        self.assertIsNot(model.prepare_forward_batch, existing_prep)

        fb = _forward_batch(real_t=4, real_b=1, mode="extend")
        model.prepare_forward_batch(fb)
        existing_prep.assert_called_once_with(fb)
        self.assertEqual(fb._tpu_real_q, 4)
        self.assertEqual(fb._tpu_real_b, 1)


if __name__ == "__main__":
    unittest.main()
