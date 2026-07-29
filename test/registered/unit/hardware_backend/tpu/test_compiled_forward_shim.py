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
    DEFAULT_TPU_COMPILE_MAX_BS,
    CompiledForwardShim,
    _dp_attn_apply_bucket_metadata,
    _dp_attn_select_bucket_shape,
    _extend_batch_cap,
    _matches_default,
    _pad_forward_batch_inputs,
    _static_default_kwargs,
    _zero_pad_1d,
    build_bucketers,
    reconcile_tpu_batch_limits,
)
from sglang.srt.hardware_backend.tpu.compilation.decoder_compile_policy import (
    DecoderTpuCompilePolicy,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

# Pure-logic tests: CPU tensors only, and the shim's import chain is
# torch_tpu-free, so this runs on the CPU CI runner (no TPU required).
register_cpu_ci(est_time=15, suite="base-a-test-cpu")


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

    def is_idle(self):
        return self._mode == "idle"

    def is_extend(self):
        # Real ForwardMode.is_extend() is True for TARGET_VERIFY too (it is an
        # extend-family mode), so mirror that here; otherwise the shim guard's
        # `is_extend() and not is_target_verify()` short-circuit isn't tested
        # faithfully.
        return self._mode in ("extend", "mixed", "target_verify")

    def is_mixed(self):
        return self._mode == "mixed"

    def is_target_verify(self):
        return self._mode == "target_verify"

    def is_draft_extend_v2(self):
        return self._mode == "draft_extend_v2"


def _forward_batch(real_t, real_b, mode="extend", return_logprob=False):
    """Synthetic ForwardBatch carrying the fields the padder touches."""
    extend = mode in ("extend", "mixed", "draft_extend_v2")
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
        dp_padding_mode=None,
        global_num_tokens_cpu=None,
        # Real ForwardBatch always carries spec_info (None when not spec).
        spec_info=None,
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
        self.assertEqual(fb.out_cache_loc.shape[0], 8)
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

    def test_extend_batch_cap_does_not_exceed_token_ceiling(self):
        fb = _forward_batch(real_t=4, real_b=2, mode="extend")
        _pad_forward_batch_inputs(
            fb,
            t_bucketer=_bucketer("Token", [4]),
            b_bucketer=_bucketer("Bs", [1, 8]),
        )
        self.assertEqual(fb.input_ids.shape[0], 4)
        self.assertEqual(fb.seq_lens.shape[0], 4)

    def test_extend_cap_is_independent_from_decode_buckets(self):
        fb = _forward_batch(real_t=10, real_b=3, mode="extend")
        _pad_forward_batch_inputs(
            fb,
            t_bucketer=_bucketer("Token", [16, 64]),
            b_bucketer=_bucketer("Bs", [1, 4, 16, 64, 256]),
            extend_bs=4,
        )
        self.assertEqual(fb.input_ids.shape[0], 16)
        self.assertEqual(fb.seq_lens.shape[0], 4)

    def test_extend_rejects_batch_above_compile_cap(self):
        fb = _forward_batch(real_t=10, real_b=5, mode="extend")
        with self.assertRaisesRegex(ValueError, "exceeds its compile cap 4"):
            _pad_forward_batch_inputs(
                fb,
                t_bucketer=_bucketer("Token", [16, 64]),
                b_bucketer=_bucketer("Bs", [1, 4, 16, 64, 256]),
                extend_bs=4,
            )

    def test_mixed_batch_uses_decode_bucket_above_prefill_cap(self):
        fb = _forward_batch(real_t=80, real_b=56, mode="mixed")
        real_q, real_b = _pad_forward_batch_inputs(
            fb,
            t_bucketer=_bucketer("Token", [64, 128]),
            b_bucketer=_bucketer("Bs", [16, 32, 64]),
            extend_bs=16,
        )

        self.assertEqual((real_q, real_b), (80, 56))
        self.assertEqual(fb.input_ids.shape[0], 128)
        self.assertEqual(fb.seq_lens.shape[0], 64)
        self.assertEqual(fb.extend_seq_lens.shape[0], 64)
        self.assertEqual(fb.batch_size, 64)

    def test_mixed_batch_shapes_are_all_precompiled(self):
        t_bucketer = _bucketer("Token", [16, 64, 128])
        b_bucketer = _bucketer("Bs", [16, 32, 64])
        policy = DecoderTpuCompilePolicy(
            model_runner=SimpleNamespace(
                server_args=SimpleNamespace(enable_mixed_chunk=True)
            ),
            t_bucketer=t_bucketer,
            b_bucketer=b_bucketer,
        )
        policy.extend_bs = 16
        precompiled_shapes = set(policy._extend_compile_shapes())

        for real_q in (1, 16, 17, 64, 65, 128):
            for real_b in (1, 16, 17, 32, 33, 64):
                with self.subTest(real_q=real_q, real_b=real_b):
                    fb = _forward_batch(real_t=real_q, real_b=real_b, mode="mixed")
                    _pad_forward_batch_inputs(
                        fb,
                        t_bucketer=t_bucketer,
                        b_bucketer=b_bucketer,
                        extend_bs=policy.extend_bs,
                    )
                    padded_shape = (
                        fb.input_ids.shape[0],
                        fb.seq_lens.shape[0],
                    )
                    self.assertIn(padded_shape, precompiled_shapes)

    def test_target_verify_pins_token_dim_to_draft_token_num(self):
        # Verify query dim is cap * draft_token_num (fixed), NOT a token bucket:
        # the extend branch would over-pad it up to the nearest token bucket.
        fb = _forward_batch(real_t=8, real_b=2, mode="target_verify")  # bs=2, dtn=4
        fb.spec_info = SimpleNamespace(draft_token_num=4)
        real_q, real_b = _pad_forward_batch_inputs(
            fb,
            t_bucketer=_bucketer("Token", [64, 128]),  # must NOT be used for verify
            b_bucketer=_bucketer("Bs", [1, 4]),
        )
        self.assertEqual((real_q, real_b), (8, 2))
        # pad_b = max_val = 4; pad_t = 4 * draft_token_num(4) = 16, not 64.
        self.assertEqual(fb.input_ids.shape[0], 16)
        self.assertEqual(fb.positions.shape[0], 16)
        self.assertEqual(fb.seq_lens.shape[0], 4)
        self.assertEqual(fb.batch_size, 4)

    def test_draft_extend_v2_uses_forward_mode_api(self):
        fb = _forward_batch(real_t=8, real_b=2, mode="draft_extend_v2")
        fb.spec_info = SimpleNamespace(
            num_tokens_per_req=4,
            hidden_states=torch.zeros(8, 16),
        )
        _pad_forward_batch_inputs(
            fb,
            t_bucketer=_bucketer("Token", [64]),
            b_bucketer=_bucketer("Bs", [1, 4]),
        )
        self.assertEqual(fb.input_ids.shape[0], 16)
        self.assertEqual(fb.out_cache_loc.shape[0], 16)
        self.assertEqual(fb.seq_lens.shape[0], 4)
        self.assertEqual(fb.extend_seq_lens.tolist(), [4, 4, 0, 0])
        self.assertEqual(fb.extend_seq_lens_cpu, [4, 4, 0, 0])
        self.assertEqual(fb.batch_size, 4)

    def test_draft_extend_v2_rejects_unset_num_tokens_per_req(self):
        fb = _forward_batch(real_t=8, real_b=2, mode="draft_extend_v2")
        fb.spec_info = SimpleNamespace(num_tokens_per_req=-1)

        with self.assertRaisesRegex(AssertionError, "num_tokens_per_req > 0"):
            _pad_forward_batch_inputs(
                fb,
                t_bucketer=_bucketer("Token", [64]),
                b_bucketer=_bucketer("Bs", [1, 4]),
            )

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

    def test_decode_cpu_list_fields_stay_none(self):
        # Non-extend batches keep the upstream None for the extend_*_cpu
        # lists. Consumers gate on forward_mode.is_extend(), and the decode
        # precompile dummy also carries None.
        fb = _forward_batch(real_t=2, real_b=2, mode="decode")
        fb.extend_seq_lens_cpu = None
        fb.extend_prefix_lens_cpu = None
        fb.extend_logprob_start_lens_cpu = None
        _pad_forward_batch_inputs(
            fb,
            t_bucketer=_bucketer("Token", [16]),
            b_bucketer=_bucketer("Bs", [4]),
        )
        self.assertIsNone(fb.extend_seq_lens_cpu)
        self.assertIsNone(fb.extend_prefix_lens_cpu)
        self.assertIsNone(fb.extend_logprob_start_lens_cpu)

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

    def test_input_logprob_rejected(self):
        # An extend request that also asks for INPUT/prompt (echo) logprobs
        # (start_len < extend_len) still can't run on the compiled backend: the
        # LogitsProcessor would compute them over dynamically-pruned states.
        fb = _forward_batch(real_t=4, real_b=1, return_logprob=True)
        fb.extend_logprob_start_lens_cpu = [0]  # extend_len(4) - start_len(0) > 0
        with self.assertRaisesRegex(NotImplementedError, "input/prompt logprobs"):
            _pad_forward_batch_inputs(
                fb,
                t_bucketer=_bucketer("Token", [64]),
                b_bucketer=_bucketer("Bs", [1]),
            )

    def test_output_logprob_passes(self):
        # OUTPUT (next-token) logprobs only: start_len == extend_len, so there
        # are no prompt positions to score inside the compiled forward. The
        # guard must let it through (the eager Sampler produces them afterwards).
        fb = _forward_batch(real_t=4, real_b=1, return_logprob=True)
        fb.extend_logprob_start_lens_cpu = [4]  # extend_len - start_len == 0
        real_q, real_b = _pad_forward_batch_inputs(
            fb,
            t_bucketer=_bucketer("Token", [8, 64]),
            b_bucketer=_bucketer("Bs", [1, 4]),
        )
        self.assertEqual((real_q, real_b), (4, 1))
        self.assertEqual(fb.input_ids.shape[0], 8)

    def test_decode_output_logprob_passes(self):
        # Decode-mode logprob requests are output-only (not an extend batch), so
        # the input-logprob path never triggers and the request must pass.
        fb = _forward_batch(real_t=2, real_b=2, mode="decode", return_logprob=True)
        _pad_forward_batch_inputs(
            fb,
            t_bucketer=_bucketer("Token", [16]),
            b_bucketer=_bucketer("Bs", [4]),
        )
        self.assertEqual(fb.input_ids.shape[0], 4)

    def test_target_verify_output_logprob_passes(self):
        # Spec-decode target-verify is an extend-family mode (is_extend() is
        # True), but it never produces INPUT/prompt logprobs: LogitsProcessor
        # treats it like decode (_get_pruned_states -> pruned_states = all hidden
        # states, input_logprob_indices = None; from_forward_batch excludes it
        # from extend_return_logprob). So there are no dynamically-pruned states
        # and the guard must let it through even with return_logprob set. The
        # batch pads to the static verify shape (pad_b * draft_token_num).
        fb = _forward_batch(
            real_t=6, real_b=3, mode="target_verify", return_logprob=True
        )
        fb.spec_info = SimpleNamespace(draft_token_num=2, hidden_states=None)
        real_q, real_b = _pad_forward_batch_inputs(
            fb,
            t_bucketer=_bucketer("Token", [8, 64]),
            b_bucketer=_bucketer("Bs", [4]),
        )
        self.assertEqual((real_q, real_b), (6, 3))
        self.assertEqual(fb.input_ids.shape[0], 8)  # pad_b(4) * draft_token_num(2)


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
        shim = CompiledForwardShim(compiled, policy=DecoderTpuCompilePolicy())
        fb = SimpleNamespace(
            _tpu_real_q=4,
            _tpu_real_b=2,
            is_prefill_only=False,
            seq_lens=torch.zeros(4, dtype=torch.int32),
        )
        out = shim(None, None, fb)
        self.assertEqual(out.next_token_logits.shape, (2, 10))
        self.assertEqual(out.hidden_states.shape, (4, 5))
        self.assertTrue(torch.equal(out.next_token_logits, padded_logits[:2]))
        self.assertTrue(torch.equal(out.hidden_states, padded_hidden[:4]))

    def test_passes_args_through(self):
        compiled = MagicMock(return_value=SimpleNamespace())
        shim = CompiledForwardShim(compiled, policy=DecoderTpuCompilePolicy())
        fb = SimpleNamespace(
            _tpu_real_q=4,
            _tpu_real_b=2,
            is_prefill_only=False,
            seq_lens=torch.zeros(4, dtype=torch.int32),
        )
        shim("ids", "pos", fb, "extra1", kw="extra2")
        compiled.assert_called_once_with("ids", "pos", fb, "extra1", kw="extra2")

    def test_preserves_compiled_signature_for_later_inspection(self):
        def forward(input_ids, positions, forward_batch, pp_proxy_tensors=None):
            pass

        compiled = MagicMock(return_value=SimpleNamespace())
        compiled.__signature__ = inspect.signature(forward)
        shim = CompiledForwardShim(compiled, policy=DecoderTpuCompilePolicy())
        self.assertIn("pp_proxy_tensors", inspect.signature(shim).parameters)

    def test_drops_default_kwargs_before_compiled_call(self):
        def forward(input_ids, positions, forward_batch, pp_proxy_tensors=None):
            pass

        compiled = MagicMock(return_value=SimpleNamespace())
        compiled.__signature__ = inspect.signature(forward)
        shim = CompiledForwardShim(compiled, policy=DecoderTpuCompilePolicy())
        fb = SimpleNamespace(
            _tpu_real_q=4,
            _tpu_real_b=2,
            is_prefill_only=False,
            seq_lens=torch.zeros(4, dtype=torch.int32),
        )
        shim("ids", "pos", fb, pp_proxy_tensors=None)
        compiled.assert_called_once_with("ids", "pos", fb)

    def test_keeps_non_default_kwargs_before_compiled_call(self):
        def forward(input_ids, positions, forward_batch, get_embedding=False):
            pass

        compiled = MagicMock(return_value=SimpleNamespace())
        compiled.__signature__ = inspect.signature(forward)
        shim = CompiledForwardShim(compiled, policy=DecoderTpuCompilePolicy())
        fb = SimpleNamespace(
            _tpu_real_q=4,
            _tpu_real_b=2,
            is_prefill_only=False,
            seq_lens=torch.zeros(4, dtype=torch.int32),
        )
        shim("ids", "pos", fb, get_embedding=True)
        compiled.assert_called_once_with("ids", "pos", fb, get_embedding=True)

    def test_no_op_when_real_size_keys_missing(self):
        # Padding hook didn't run (e.g. forward_idle): output passes through.
        out_obj = SimpleNamespace(
            next_token_logits=torch.zeros(4, 10),
            hidden_states=torch.zeros(8, 5),
        )
        shim = CompiledForwardShim(
            MagicMock(return_value=out_obj), policy=DecoderTpuCompilePolicy()
        )
        out = shim(None, None, SimpleNamespace(is_prefill_only=False))
        self.assertIs(out, out_obj)

    def test_no_op_when_already_at_real_shape(self):
        # No padding to slice off when shapes already match.
        out_obj = SimpleNamespace(
            next_token_logits=torch.zeros(2, 10),
            hidden_states=torch.zeros(4, 5),
        )
        shim = CompiledForwardShim(
            MagicMock(return_value=out_obj), policy=DecoderTpuCompilePolicy()
        )
        fb = SimpleNamespace(
            _tpu_real_q=4,
            _tpu_real_b=2,
            is_prefill_only=False,
            seq_lens=torch.zeros(4, dtype=torch.int32),
        )
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
        shim = CompiledForwardShim(compiled, policy=DecoderTpuCompilePolicy())
        fb = SimpleNamespace(
            _tpu_real_q=8,
            _tpu_real_b=2,
            batch_size=4,
            is_prefill_only=False,
            seq_lens=torch.zeros(4, dtype=torch.int32),
        )
        shim(None, None, fb)
        self.assertEqual(fb.batch_size, 2)

    def test_does_not_touch_batch_size_when_real_keys_missing(self):
        # forward_idle / split_prefill skip the pad hook, so the slicer
        # must not mutate batch_size in that case.
        fb = SimpleNamespace(batch_size=4, is_prefill_only=False)
        compiled = MagicMock(return_value=SimpleNamespace())
        shim = CompiledForwardShim(compiled, policy=DecoderTpuCompilePolicy())
        shim(None, None, fb)
        self.assertEqual(fb.batch_size, 4)

    def test_pins_is_prefill_only_for_compiled_call_and_restores(self):
        # The compiled graph must see a batch-invariant is_prefill_only
        # (pinned False); the worker's sampling gate reads the true value
        # after the forward, so it is restored.
        seen = {}

        def compiled(input_ids, positions, forward_batch):
            seen["during"] = forward_batch.is_prefill_only
            return SimpleNamespace()

        shim = CompiledForwardShim(compiled, policy=DecoderTpuCompilePolicy())
        fb = SimpleNamespace(batch_size=0, is_prefill_only=True)
        shim(None, None, fb)
        self.assertIs(seen["during"], False)
        self.assertIs(fb.is_prefill_only, True)


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

    def test_token_buckets_above_max_t_dropped(self):
        # Token buckets feed only the EXTEND path, which never exceeds max_t
        # (chunked-prefill / context cap), so buckets above it are dropped --
        # they would compile dead shapes serving can't reach.
        sa = self._server_args(tpu_compile_tokens=[64, 512, 2048, 4096])
        t_b, _ = build_bucketers(sa, max_t=2048, max_b=8)
        self.assertEqual(t_b.buckets, [64, 512, 2048])  # 4096 dropped
        # If every user bucket is above the cap, fall back to max_t itself.
        sa2 = self._server_args(tpu_compile_tokens=[4096])
        t_b2, _ = build_bucketers(sa2, max_t=2048, max_b=8)
        self.assertEqual(t_b2.buckets, [2048])

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

    def test_extend_batch_cap_uses_reachable_minimum(self):
        self.assertEqual(
            _extend_batch_cap(
                _bucketer("Token", [64, 4096]),
                _bucketer("Bs", [1, 8192]),
            ),
            4096,
        )


class TestReconcileTpuBatchLimits(CustomTestCase):
    """The compiled grid covers scheduler concurrency without compiling past
    explicit limits when no larger serving batch is reachable (Task #40)."""

    def _r(
        self, torch_compile_max_bs=None, tpu_compile_bs=None, max_running_requests=None
    ):
        return reconcile_tpu_batch_limits(
            torch_compile_max_bs, tpu_compile_bs, max_running_requests
        )

    def test_default_neither_set(self):
        # Nothing requested -> ceiling and concurrency default to 64 (NOT the
        # ~2048 memory estimate), no explicit bucket list.
        self.assertEqual(
            self._r(),
            (DEFAULT_TPU_COMPILE_MAX_BS, None, DEFAULT_TPU_COMPILE_MAX_BS),
        )

    def test_explicit_torch_compile_max_bs_above_default_honored(self):
        tcmb, _, mrr = self._r(torch_compile_max_bs=128)
        self.assertEqual((tcmb, mrr), (128, 128))

    def test_explicit_torch_compile_max_bs_below_default_avoids_dead_shapes(self):
        self.assertEqual(self._r(torch_compile_max_bs=8), (8, None, 8))

    def test_max_running_requests_grows_ceiling(self):
        # --max-running-requests 128 -> grid ceiling 128, served (not clamped 64).
        self.assertEqual(self._r(max_running_requests=128), (128, None, 128))

    def test_tpu_compile_bs_sets_ceiling_and_concurrency(self):
        tcmb, bs, mrr = self._r(tpu_compile_bs=[1, 8])
        self.assertEqual((tcmb, mrr, bs), (8, 8, [1, 8]))

    def test_both_set_and_agree(self):
        tcmb, bs, mrr = self._r(
            tpu_compile_bs=[1, 8, 64, 128], max_running_requests=128
        )
        self.assertEqual((tcmb, mrr, max(bs)), (128, 128, 128))

    def test_conflict_grows_bucket_list(self):
        # Case E: pinned buckets top out at 64 but concurrency is 128 -> the list
        # is grown to include 128 so build_bucketers covers the batch (no crash).
        tcmb, bs, mrr = self._r(tpu_compile_bs=[1, 8, 64], max_running_requests=128)
        self.assertEqual((tcmb, mrr), (128, 128))
        self.assertIn(128, bs)
        self.assertEqual(max(bs), 128)

    def test_bucket_list_only_grows_to_scheduler_concurrency(self):
        tcmb, bs, mrr = self._r(
            torch_compile_max_bs=128,
            tpu_compile_bs=[1, 8],
            max_running_requests=32,
        )
        self.assertEqual((tcmb, bs, mrr), (128, [1, 8, 32], 32))

    def test_invariant_grid_always_covers_concurrency(self):
        cases = [
            (None, None, None),
            (None, None, 200),
            (None, [1, 8, 64], 256),
            (None, [1, 8, 64, 512], None),
            (256, None, 128),
        ]
        for tcmb_in, bs_in, mrr_in in cases:
            tcmb, bs, mrr = self._r(tcmb_in, bs_in, mrr_in)
            self.assertGreaterEqual(tcmb, mrr, f"ceiling < concurrency for {cases}")
            if bs is not None:
                self.assertGreaterEqual(
                    max(bs), mrr, f"buckets don't cover concurrency for {cases}"
                )


class TestDpAttnSelectBucketShape(CustomTestCase):
    """Every DP rank must agree on the same (pad_t, pad_b) bucket."""

    def _fb(self, mode, is_extend_in_batch=False):
        return SimpleNamespace(
            forward_mode=_ForwardModeMock(mode),
            is_extend_in_batch=is_extend_in_batch,
        )

    def test_decode_uses_bs_bucketer_for_both_axes(self):
        # decode BATCH bucket must be identical across ranks -> both axes use the
        # bs bucketer seeded with the cross-rank max_tokens_per_rank.
        t_b = _bucketer("Token", [16, 64])
        b_b = _bucketer("Bs", [4, 8, 16])
        pad_t, pad_b = _dp_attn_select_bucket_shape(
            self._fb("decode"), t_b, b_b, max_tokens_per_rank=8, real_q=3, real_b=3
        )
        self.assertEqual((pad_t, pad_b), (8, 8))

    def test_extend_pads_to_max_bs_bucket(self):
        t_b = _bucketer("Token", [16, 64])
        b_b = _bucketer("Bs", [4, 8, 16])
        pad_t, pad_b = _dp_attn_select_bucket_shape(
            self._fb("extend", is_extend_in_batch=True),
            t_b,
            b_b,
            max_tokens_per_rank=8,
            real_q=10,
            real_b=3,
        )
        # pad_b is forced to the max bs bucket. pad_t is the next token bucket
        # covering max(real_q, max_tokens_per_rank, pad_b).
        self.assertEqual(pad_b, 16)
        self.assertEqual(pad_t, 16)

    def test_idle_without_extend_in_batch_is_decode_group(self):
        t_b = _bucketer("Token", [16, 64])
        b_b = _bucketer("Bs", [4, 8, 16])
        pad_t, pad_b = _dp_attn_select_bucket_shape(
            self._fb("idle", is_extend_in_batch=False),
            t_b,
            b_b,
            max_tokens_per_rank=8,
            real_q=1,
            real_b=1,
        )
        # Treated as decode: both axes from the bs bucketer.
        self.assertEqual((pad_t, pad_b), (8, 8))


class TestDpAttnApplyBucketMetadata(CustomTestCase):
    def _fb(self, dp_size=2):
        return SimpleNamespace(
            input_ids=torch.zeros(4, dtype=torch.int32),  # CPU -> device for gpu counts
            global_num_tokens_cpu=[3] * dp_size,
            global_num_tokens_gpu=None,
            global_num_tokens_for_logprob_cpu=[3] * dp_size,
            global_num_tokens_for_logprob_gpu=None,
            dp_local_start_pos=torch.tensor([1]),
            dp_local_num_tokens=torch.tensor([2]),
            global_dp_buffer_len=0,
            is_prefill_only=True,
        )

    def test_sets_uniform_padded_global_counts(self):
        fb = self._fb(dp_size=2)
        _dp_attn_apply_bucket_metadata(fb, pad_t=8, pad_b=16)
        self.assertEqual(fb.global_num_tokens_cpu, [8, 8])
        self.assertEqual(fb.global_num_tokens_gpu.tolist(), [8, 8])
        self.assertEqual(fb.global_num_tokens_gpu.dtype, torch.int64)
        self.assertEqual(fb.global_num_tokens_for_logprob_cpu, [16, 16])

    def test_does_not_mutate_is_prefill_only(self):
        # The is_prefill_only pin lives in CompiledForwardShim.__call__ (pinned
        # False for the compiled call, restored after); applying bucket
        # metadata must leave the worker-visible flag untouched.
        fb = self._fb()
        _dp_attn_apply_bucket_metadata(fb, pad_t=8, pad_b=8)
        self.assertIs(fb.is_prefill_only, True)

    def test_clears_cached_local_offsets(self):
        fb = self._fb()
        _dp_attn_apply_bucket_metadata(fb, pad_t=8, pad_b=8)
        self.assertIsNone(fb.dp_local_start_pos)
        self.assertIsNone(fb.dp_local_num_tokens)

    def test_global_buffer_len_is_pad_t_times_dp_size(self):
        fb = self._fb(dp_size=4)
        _dp_attn_apply_bucket_metadata(fb, pad_t=8, pad_b=8)
        self.assertEqual(fb.global_dp_buffer_len, 8 * 4)


if __name__ == "__main__":
    unittest.main()
