"""Unit tests for decoder TPU compile policy warmup."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.hardware_backend.tpu.compilation.bucketer import Bucketer
from sglang.srt.hardware_backend.tpu.compilation.decoder_compile_policy import (
    DecoderTpuCompilePolicy,
    Qwen3VLDecoderTpuCompilePolicy,
    _materialize_out,
)
from sglang.srt.hardware_backend.tpu.compilation.spec_decode_compile_policy import (
    SpecDecodeCompilePolicyMixin,
    SpecDecoderTpuCompilePolicy,
)
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardMode,
    PPProxyTensors,
)
from sglang.test.test_utils import CustomTestCase


class _FakeAttnBackend:
    """Records ``init_forward_metadata`` calls so tests can assert shape."""

    def __init__(self):
        self.calls: list[tuple[int, int]] = []  # (num_tokens, bs)

    def init_forward_metadata(self, forward_batch):
        self.calls.append(
            (forward_batch.input_ids.shape[0], forward_batch.seq_lens.shape[0])
        )


def _model_runner(device="cpu"):
    """Minimal ModelRunner stand-in for unit tests.

    ``prepare_forward_batch`` is included because ``_run_one`` calls it
    before ``init_forward_metadata`` (mirrors the runtime chain installed
    by ``DecoderTpuCompilePolicy.install()``)."""
    return SimpleNamespace(
        device=device,
        model=SimpleNamespace(
            forward=MagicMock(),
            prepare_forward_batch=MagicMock(),
        ),
        attn_backend=_FakeAttnBackend(),
        req_to_token_pool=SimpleNamespace(),
        token_to_kv_pool=SimpleNamespace(),
    )


def _make_policy(model_runner=None, token_buckets=(64, 256), bs_buckets=(1, 4)):
    if model_runner is None:
        model_runner = _model_runner()
    t_bucketer = Bucketer(name="Token", buckets=list(token_buckets))
    b_bucketer = Bucketer(name="Bs", buckets=list(bs_buckets))
    return DecoderTpuCompilePolicy(
        model_runner=model_runner,
        t_bucketer=t_bucketer,
        b_bucketer=b_bucketer,
        owner=model_runner.model,
    )


def _target_and_padding(policy):
    return policy.owner, policy


def _run_cases(policy, mode):
    with patch(
        "sglang.srt.hardware_backend.tpu.compilation.decoder_compile_policy."
        "get_tensor_model_parallel_rank",
        return_value=1,
    ):
        if mode == ForwardMode.EXTEND:
            policy._precompile_extend()
        else:
            policy._precompile_decode()


class TestQwen3VLDecoderUnpadding(CustomTestCase):
    def _forward_batch(self):
        return SimpleNamespace(
            _tpu_real_q=3,
            _tpu_real_b=2,
            batch_size=4,
            seq_lens=torch.zeros(4, dtype=torch.int32),
        )

    def test_slices_raw_hidden_states_before_logits_processing(self):
        policy = Qwen3VLDecoderTpuCompilePolicy()
        out = torch.arange(20, dtype=torch.float32).reshape(5, 4)

        result = policy.unpad_outputs(out, self._forward_batch())

        self.assertEqual(result.shape, (3, 4))

    def test_rejects_dp_attention_at_install(self):
        policy = Qwen3VLDecoderTpuCompilePolicy()

        with patch(
            "sglang.srt.hardware_backend.tpu.compilation.decoder_compile_policy."
            "is_dp_attention_enabled",
            return_value=True,
        ), self.assertRaisesRegex(NotImplementedError, "does not support DP attention"):
            policy.install()

    def test_restores_extend_metadata_before_logits_processing(self):
        policy = Qwen3VLDecoderTpuCompilePolicy()
        forward_batch = self._forward_batch()
        forward_batch.input_ids = torch.arange(5)
        forward_batch.positions = torch.arange(5)
        forward_batch.out_cache_loc = torch.arange(5)
        forward_batch.extend_seq_lens = torch.tensor([2, 1, 0, 0])
        forward_batch.extend_seq_lens_cpu = [2, 1, 0, 0]
        forward_batch.extend_prefix_lens_cpu = [0, 0, 0, 0]
        forward_batch.extend_logprob_start_lens_cpu = [0, 0, 0, 0]
        forward_batch.mm_input_embeds = torch.zeros(5, 4)
        hidden_states = torch.arange(20, dtype=torch.float32).reshape(5, 4)

        result = policy.unpad_outputs(hidden_states, forward_batch)
        last_index = torch.cumsum(forward_batch.extend_seq_lens, dim=0) - 1
        sampled_hidden_states = result[last_index]

        self.assertEqual(forward_batch.batch_size, 2)
        self.assertEqual(forward_batch.extend_seq_lens.tolist(), [2, 1])
        self.assertEqual(forward_batch.extend_seq_lens_cpu, [2, 1])
        self.assertEqual(forward_batch.mm_input_embeds.shape, (3, 4))
        self.assertEqual(sampled_hidden_states.shape, (2, 4))

    def test_slices_hidden_and_aux_tuple(self):
        policy = Qwen3VLDecoderTpuCompilePolicy()
        hidden = torch.zeros(5, 4)
        aux = [torch.ones(5, 4), torch.full((5, 4), 2.0)]

        result_hidden, result_aux = policy.unpad_outputs(
            (hidden, aux), self._forward_batch()
        )

        self.assertEqual(result_hidden.shape, (3, 4))
        self.assertEqual([value.shape for value in result_aux], [(3, 4), (3, 4)])

    def test_keeps_pipeline_proxy_tensors_padded_for_next_stage(self):
        policy = Qwen3VLDecoderTpuCompilePolicy()
        out = PPProxyTensors(
            {"hidden_states": torch.zeros(5, 4), "residual": torch.ones(5, 4)}
        )

        result = policy.unpad_outputs(out, self._forward_batch())

        self.assertEqual(result["hidden_states"].shape, (5, 4))
        self.assertEqual(result["residual"].shape, (5, 4))

    def test_spec_policy_delegates_raw_vl_output_unpadding(self):
        class _SpecQwenPolicy(
            SpecDecodeCompilePolicyMixin, Qwen3VLDecoderTpuCompilePolicy
        ):
            pass

        policy = object.__new__(_SpecQwenPolicy)
        forward_batch = self._forward_batch()
        forward_batch.forward_mode = ForwardMode.TARGET_VERIFY
        out = torch.arange(20, dtype=torch.float32).reshape(5, 4)

        result = policy.unpad_outputs(out, forward_batch)

        self.assertEqual(result.shape, (3, 4))
        self.assertEqual(forward_batch.batch_size, 2)


class TestMakeDummyForwardBatch(CustomTestCase):
    """``_make_dummy_forward_batch`` shapes valid + ghost slots correctly."""

    def test_extend_shapes_and_ghost_tokens(self):
        mr = _model_runner()
        policy = _make_policy(mr)
        _, padding = _target_and_padding(policy)
        fb = padding._make_dummy_forward_batch(
            num_tokens=64, bs=4, mode=ForwardMode.EXTEND
        )
        # Token-axis tensors padded to num_tokens; bs=4 valid + 60 ghost.
        self.assertEqual(fb.input_ids.shape[0], 64)
        self.assertEqual(fb.positions.shape[0], 64)
        self.assertEqual(fb.out_cache_loc.shape[0], 64)
        self.assertEqual(fb.input_ids[:4].tolist(), [1, 1, 1, 1])
        self.assertEqual(fb.input_ids[4:].sum().item(), 0)
        self.assertEqual(fb.input_ids.dtype, torch.int64)
        self.assertEqual(fb.positions.dtype, torch.int64)
        # Per-request tensors at bs.
        self.assertEqual(fb.seq_lens.shape[0], 4)
        self.assertEqual(fb.seq_lens.dtype, torch.int64)
        self.assertEqual(fb.extend_seq_lens.shape[0], 4)
        self.assertEqual(fb.extend_seq_lens.tolist(), [1, 1, 1, 1])
        self.assertEqual(fb.extend_logprob_start_lens_cpu, [0, 0, 0, 0])
        # KV writes go to dummy slot 0 across the whole token range.
        self.assertEqual(fb.out_cache_loc.sum().item(), 0)

    def test_decode_has_no_extend_fields(self):
        mr = _model_runner()
        policy = _make_policy(mr)
        _, padding = _target_and_padding(policy)
        fb = padding._make_dummy_forward_batch(
            num_tokens=8, bs=8, mode=ForwardMode.DECODE
        )
        # Decode: one query token per running seq.
        self.assertEqual(fb.input_ids.shape[0], 8)
        self.assertEqual(fb.seq_lens.shape[0], 8)
        # Extend fields are None in decode (the path skips them).
        self.assertIsNone(fb.extend_seq_lens)
        self.assertIsNone(fb.extend_prefix_lens)
        self.assertIsNone(fb.extend_num_tokens)
        self.assertIsNone(fb.extend_logprob_start_lens_cpu)


class TestPrecompileLoops(CustomTestCase):
    """``precompile`` fires one forward per TPU precompile variant:
    EXTEND covers token buckets at the fixed batch cap, and DECODE varies bs
    (tokens == bs)."""

    def test_extend_iterates_token_buckets_at_fixed_batch_cap(self):
        mr = _model_runner()
        policy = _make_policy(mr, token_buckets=(8, 64, 256), bs_buckets=(1, 4))
        _run_cases(policy, ForwardMode.EXTEND)
        self.assertEqual(mr.model.forward.call_count, 3)
        self.assertEqual(
            mr.attn_backend.calls,
            [(8, 4), (64, 4), (256, 4)],
        )

    def test_extend_skips_token_bucket_below_bs(self):
        # A 4-token bucket can't hold the fixed 8-req extend cap.
        mr = _model_runner()
        policy = _make_policy(mr, token_buckets=(4, 64), bs_buckets=(1, 8))
        _run_cases(policy, ForwardMode.EXTEND)
        self.assertEqual(mr.model.forward.call_count, 1)
        self.assertEqual(mr.attn_backend.calls, [(64, 8)])

    def test_extend_raises_when_no_token_bucket_can_hold_bs_cap(self):
        mr = _model_runner()
        policy = _make_policy(mr, token_buckets=(4,), bs_buckets=(8,))
        with self.assertRaisesRegex(ValueError, "No TPU extend compile shapes"):
            _run_cases(policy, ForwardMode.EXTEND)

    def test_decode_loop_iterates_bs(self):
        mr = _model_runner()
        policy = _make_policy(mr, token_buckets=(64,), bs_buckets=(1, 4, 8))
        _run_cases(policy, ForwardMode.DECODE)
        self.assertEqual(mr.model.forward.call_count, 3)
        # Decode: tokens == bs for each call.
        self.assertEqual(mr.attn_backend.calls, [(1, 1), (4, 4), (8, 8)])

    def test_precompile_combines_extend_and_decode(self):
        mr = _model_runner()
        policy = _make_policy(mr, token_buckets=(8, 64), bs_buckets=(1, 4))
        with patch(
            "sglang.srt.hardware_backend.tpu.compilation.decoder_compile_policy."
            "get_tensor_model_parallel_rank",
            return_value=1,
        ):
            policy.precompile()
        # EXTEND: (8,4), (64,4) = 2. DECODE: (1,1), (4,4) = 2.
        self.assertEqual(mr.model.forward.call_count, 4)


class TestRunOneCallsCompiledForward(CustomTestCase):
    """``_run_one`` wires the synthesized batch through the same chain
    serving uses: ``prepare_forward_batch`` → ``init_forward_metadata``
    → compiled ``forward``."""

    def test_run_one_calls_prepare_then_metadata_then_forward(self):
        mr = _model_runner()
        policy = _make_policy(mr)
        policy._run_one(num_tokens=64, bs=4, mode=ForwardMode.EXTEND)
        # prepare_forward_batch runs first so model-specific prep hooks
        # (Moss-VL etc.) see precompile too.
        mr.model.prepare_forward_batch.assert_called_once()
        self.assertEqual(mr.attn_backend.calls, [(64, 4)])
        mr.model.forward.assert_called_once()
        # Args: (input_ids, positions, forward_batch). Shapes match bucket.
        args, _ = mr.model.forward.call_args
        self.assertEqual(args[0].shape, (64,))
        self.assertEqual(args[1].shape, (64,))
        self.assertEqual(args[2].batch_size, 4)

    def test_run_one_uses_no_grad_not_inference_mode(self):
        mr = _model_runner()
        policy = _make_policy(mr)
        cm = MagicMock()
        with patch(
            "sglang.srt.hardware_backend.tpu.compilation.decoder_compile_policy.torch.no_grad",
            return_value=cm,
        ) as no_grad, patch(
            "sglang.srt.hardware_backend.tpu.compilation."
            "decoder_compile_policy.torch.inference_mode",
            side_effect=AssertionError("inference_mode should not be used"),
        ):
            policy._run_one(num_tokens=64, bs=4, mode=ForwardMode.EXTEND)
        no_grad.assert_called_once_with()
        cm.__enter__.assert_called_once()

    def test_materialize_out_syncs_logits_and_flagged_kv_side_effects(self):
        logits = torch.zeros(1)
        kv0 = torch.ones(1)
        kv1 = torch.ones(1) * 2
        pool = SimpleNamespace(
            start_layer=2,
            layer_num=2,
            get_fused_kv_buffer=MagicMock(side_effect=[kv0, kv1]),
        )
        forward_batch = SimpleNamespace(
            attn_backend=SimpleNamespace(requires_kv_sync_after_compile=True),
            token_to_kv_pool=pool,
        )
        with patch("torch_tpu._internal.sync.sync.synchronize") as sync:
            _materialize_out(SimpleNamespace(next_token_logits=logits), forward_batch)
        sync.assert_called_once()
        tensors, kwargs = sync.call_args
        self.assertEqual(tensors[0], [logits, kv0, kv1])
        self.assertEqual(kwargs, {"wait": True})
        self.assertEqual(pool.get_fused_kv_buffer.call_args_list[0].args, (2,))
        self.assertEqual(pool.get_fused_kv_buffer.call_args_list[1].args, (3,))


def _spec_model_runner(device="cpu", is_draft_worker=False, num_steps=3):
    """ModelRunner stand-in with the spec config SpecDecoderTpuCompilePolicy reads."""
    return SimpleNamespace(
        device=device,
        model=SimpleNamespace(
            forward=MagicMock(),
            prepare_forward_batch=MagicMock(),
        ),
        attn_backend=_FakeAttnBackend(),
        # _run_one seeds req_to_token row 1; needs a real indexable tensor.
        req_to_token_pool=SimpleNamespace(
            req_to_token=torch.zeros(4, 4096, dtype=torch.int64)
        ),
        token_to_kv_pool=SimpleNamespace(),
        model_config=SimpleNamespace(hidden_size=16),
        dtype=torch.bfloat16,
        is_draft_worker=is_draft_worker,
        server_args=SimpleNamespace(
            speculative_num_steps=num_steps,
            # topk=1 chain: verify width == num_steps + 1.
            speculative_num_draft_tokens=num_steps + 1,
            speculative_algorithm="EAGLE3",
            speculative_eagle_topk=1,
        ),
    )


def _make_spec_policy(
    model_runner=None, token_buckets=(8, 64), bs_buckets=(1, 4), **mr_kwargs
):
    if model_runner is None:
        model_runner = _spec_model_runner(**mr_kwargs)
    return SpecDecoderTpuCompilePolicy(
        model_runner=model_runner,
        t_bucketer=Bucketer(name="Token", buckets=list(token_buckets)),
        b_bucketer=Bucketer(name="Bs", buckets=list(bs_buckets)),
        owner=model_runner.model,
    )


class TestSpecPrecompileLoops(CustomTestCase):
    """Target spec precompile fires EXTEND + DECODE + TARGET_VERIFY +
    EXTEND(FULL); the draft worker skips precompile entirely."""

    def test_target_precompiles_all_spec_modes(self):
        mr = _spec_model_runner(num_steps=3)  # draft_token_num = 4
        policy = _make_spec_policy(mr, token_buckets=(8, 64), bs_buckets=(1, 4))
        # precompile() spans two modules: the base EXTEND/DECODE warmup lives in
        # decoder_compile_policy (via super()), the spec TARGET_VERIFY/EXTEND(FULL)
        # in spec_decode_compile_policy. Patch the rank helper in both.
        with patch(
            "sglang.srt.hardware_backend.tpu.compilation.decoder_compile_policy."
            "get_tensor_model_parallel_rank",
            return_value=1,
        ), patch(
            "sglang.srt.hardware_backend.tpu.compilation.spec_decode_compile_policy."
            "get_tensor_model_parallel_rank",
            return_value=1,
        ):
            policy.precompile()
        # EXTEND (8,4),(64,4)=2  DECODE (1,1),(4,4)=2
        # TARGET_VERIFY buckets bs: (4,1),(16,4)=2  EXTEND(FULL) 2
        self.assertEqual(mr.model.forward.call_count, 8)
        self.assertIn((16, 4), mr.attn_backend.calls)
        self.assertIn((4, 1), mr.attn_backend.calls)

    def test_draft_worker_skips_precompile(self):
        mr = _spec_model_runner(is_draft_worker=True)
        policy = _make_spec_policy(mr)
        with patch(
            "sglang.srt.hardware_backend.tpu.compilation.decoder_compile_policy."
            "get_tensor_model_parallel_rank",
            return_value=1,
        ):
            policy.precompile()
        self.assertEqual(mr.model.forward.call_count, 0)
        self.assertEqual(mr.attn_backend.calls, [])

    def test_non_eagle3_algorithm_raises(self):
        # The spec policy enforces the supported algorithm at construction (only
        # EAGLE3 today); an unsupported algorithm fails loudly rather than
        # silently degrading to a non-spec policy.
        mr = _spec_model_runner()
        mr.server_args.speculative_algorithm = "NGRAM"
        with self.assertRaisesRegex(ValueError, "only supports EAGLE3"):
            _make_spec_policy(mr)


class TestSpecDummyForwardBatch(CustomTestCase):
    """Spec dummy batches carry spec_info + slot-1 KV seeding, and TARGET_VERIFY
    sizes extend metadata to draft_token_num tokens per request."""

    def test_target_verify_batch_shapes_and_capture_mode(self):
        policy = _make_spec_policy(num_steps=3)  # draft_token_num = 4
        spec_info = policy._make_verify_spec_info()
        fb = policy._make_dummy_forward_batch(
            num_tokens=16,
            bs=4,
            mode=ForwardMode.TARGET_VERIFY,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            spec_info=spec_info,
        )
        self.assertEqual(fb.capture_hidden_mode, CaptureHiddenMode.FULL)
        self.assertIsNotNone(fb.spec_info)
        # draft_token_num (=4) tokens per request.
        self.assertEqual(fb.extend_seq_lens.tolist(), [4, 4, 4, 4])
        self.assertEqual(fb.extend_num_tokens, 16)
        # Slot-1 KV seeding (avoids draft/target pool conflicts): not slot 0.
        self.assertEqual(fb.out_cache_loc.tolist(), [1] * 16)
        self.assertEqual(fb.req_pool_indices.tolist(), [1, 1, 1, 1])
        # Text-only mm_inputs is [None]*bs, not None (Dynamo guard parity).
        self.assertEqual(fb.mm_inputs, [None] * 4)

    def test_plain_extend_carries_idle_spec_stub(self):
        # With spec enabled, non-verify modes still get a non-None spec stub so
        # Dynamo traces both branches of ``if spec_info is not None``.
        policy = _make_spec_policy()
        fb = policy._make_dummy_forward_batch(
            num_tokens=64,
            bs=4,
            mode=ForwardMode.EXTEND,
            spec_info=policy._make_null_spec_info(),
        )
        self.assertIsNotNone(fb.spec_info)


class TestSpecUnpadOutputs(CustomTestCase):
    """Spec output slicing: TARGET_VERIFY logits use real_q; capture-LAST hidden
    uses real_b; everything else matches the base decoder policy."""

    def _out_and_batch(self, forward_mode, capture, real_q, real_b, rows):
        out = SimpleNamespace(
            next_token_logits=torch.zeros(rows, 8),
            hidden_states=torch.zeros(rows, 8),
        )
        fb = SimpleNamespace(
            forward_mode=forward_mode,
            capture_hidden_mode=capture,
            batch_size=rows,
            _tpu_real_q=real_q,
            _tpu_real_b=real_b,
        )
        return out, fb

    def test_target_verify_slices_logits_to_real_q(self):
        policy = _make_spec_policy()
        out, fb = self._out_and_batch(
            ForwardMode.TARGET_VERIFY,
            CaptureHiddenMode.FULL,
            real_q=16,
            real_b=4,
            rows=32,
        )
        policy.unpad_outputs(out, fb)
        # Verify emits one logit per draft token -> real_q, not real_b.
        self.assertEqual(out.next_token_logits.shape[0], 16)
        self.assertEqual(out.hidden_states.shape[0], 16)  # FULL -> real_q
        self.assertEqual(fb.batch_size, 4)

    def test_capture_last_slices_hidden_to_real_b(self):
        policy = _make_spec_policy()
        out, fb = self._out_and_batch(
            ForwardMode.DRAFT_EXTEND,
            CaptureHiddenMode.LAST,
            real_q=8,
            real_b=4,
            rows=16,
        )
        policy.unpad_outputs(out, fb)
        self.assertEqual(out.next_token_logits.shape[0], 4)  # not verify -> real_b
        self.assertEqual(out.hidden_states.shape[0], 4)  # LAST -> real_b

    def test_decode_default_slicing(self):
        policy = _make_spec_policy()
        out, fb = self._out_and_batch(
            ForwardMode.DECODE,
            CaptureHiddenMode.NULL,
            real_q=4,
            real_b=4,
            rows=8,
        )
        policy.unpad_outputs(out, fb)
        self.assertEqual(out.next_token_logits.shape[0], 4)
        self.assertEqual(out.hidden_states.shape[0], 4)


if __name__ == "__main__":
    unittest.main()
