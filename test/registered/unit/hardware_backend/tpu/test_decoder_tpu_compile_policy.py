"""Unit tests for decoder TPU compile policy warmup."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import contextlib
import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.compilation.support_torch_compile import (
    NoopCompilePolicy,
    get_torch_compile_policy,
    support_torch_compile,
)
from sglang.srt.hardware_backend.tpu.compilation.bucketer import Bucketer
from sglang.srt.hardware_backend.tpu.compilation.decoder_compile_policy import (
    DecoderTpuCompilePolicy,
    Qwen3VLDecoderTpuCompilePolicy,
    _materialize_out,
    _unpad_decoder_outputs,
)
from sglang.srt.hardware_backend.tpu.compilation.spec_decode_compile_policy import (
    Eagle3CompilePolicyMixin,
    SpecDecoderTpuCompilePolicy,
)
from sglang.srt.hardware_backend.tpu.sampler import distributed_top_k_for_batch
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardMode,
    PPProxyTensors,
)
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.speculative.eagle_info import (
    EagleDraftExtendInput,
    EagleDraftInput,
    EagleVerifyInput,
)
from sglang.srt.speculative.eagle_utils import eagle_sample
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
        token_to_kv_pool_allocator=SimpleNamespace(
            free_pages=torch.arange(4, dtype=torch.int32)
        ),
        server_args=SimpleNamespace(page_size=1, dp_size=1),
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
    with get_parallel().override(tp_rank=1):
        if mode == ForwardMode.EXTEND:
            policy._precompile_extend()
        else:
            policy._precompile_decode()


class TestDirectCompileExtraModelState(CustomTestCase):
    def test_custom_annotated_policy_forwards_extra_tensor_attrs(self):
        class CustomPolicy(DecoderTpuCompilePolicy):
            model_extra_tensor_attrs = (
                "object_state.value",
                "object_state.weight_alias",
            )

        model = SimpleNamespace(forward=MagicMock())
        support_torch_compile(model, policy=CustomPolicy)
        policy_cls = get_torch_compile_policy(model)
        self.assertIs(policy_cls, CustomPolicy)

        kv_pool = object()
        model_runner = SimpleNamespace(
            model=model,
            token_to_kv_pool=kv_pool,
            is_draft_worker=False,
        )
        policy = policy_cls(
            model_runner=model_runner,
            owner=model,
            tracer="direct",
        )
        compile_module_name = (
            "sglang.srt.hardware_backend.tpu.compilation.direct_compile"
        )
        compile_module = ModuleType(compile_module_name)
        compile_module.direct_compile = MagicMock(return_value=MagicMock())

        with patch.dict(sys.modules, {compile_module_name: compile_module}):
            policy._direct_compile(model.forward)

        compile_module.direct_compile.assert_called_once()
        kwargs = compile_module.direct_compile.call_args.kwargs
        self.assertIs(kwargs["model"], model)
        self.assertEqual(
            kwargs["model_extra_tensor_attrs"],
            CustomPolicy.model_extra_tensor_attrs,
        )
        self.assertIs(kwargs["kv_pool"], kv_pool)

    def test_specialization_uses_the_models_final_top_k_capability(self):
        model_runner = _model_runner()
        model_runner.server_args.sampling_backend = "tpu"
        selector = MagicMock(return_value=None)
        model_runner.model.logits_processor = SimpleNamespace(
            get_tpu_distributed_top_k_for_batch=selector
        )
        model_runner.model.lm_head = object()
        policy = _make_policy(model_runner)
        forward_batch = SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            batch_size=64,
        )

        fallback_key = policy._direct_compile_specialization_key(
            None,
            torch.arange(64),
            torch.arange(64),
            forward_batch,
        )
        selector.return_value = 50
        candidate_key = policy._direct_compile_specialization_key(
            None,
            torch.arange(64),
            torch.arange(64),
            forward_batch,
        )

        self.assertIsNone(fallback_key[-1])
        self.assertEqual(candidate_key[-1], 50)
        selector.assert_called_with(forward_batch, model_runner.model.lm_head)

    def test_warmup_at_default_cutoff_selects_candidate_specialization(self):
        model_runner = _model_runner()
        model_runner.server_args.sampling_backend = "tpu"
        selector = MagicMock(
            side_effect=lambda batch, _lm_head: distributed_top_k_for_batch(
                batch, sampling_backend="tpu"
            )
        )
        model_runner.model.logits_processor = SimpleNamespace(
            get_tpu_distributed_top_k_for_batch=selector
        )
        model_runner.model.lm_head = object()
        policy = _make_policy(model_runner, bs_buckets=(1, 64))
        forward_batch = policy._make_dummy_forward_batch(
            num_tokens=64,
            bs=64,
            mode=ForwardMode.DECODE,
        )

        key = policy._direct_compile_specialization_key(
            None,
            forward_batch.input_ids,
            forward_batch.positions,
            forward_batch,
        )

        self.assertEqual(key[-1], 50)
        selector.assert_called_once_with(forward_batch, model_runner.model.lm_head)


class TestDecoderOutputUnpadding(CustomTestCase):
    def test_global_top_k_rows_follow_the_real_batch_size(self):
        output = LogitsProcessorOutput(
            next_token_logits=None,
            next_token_global_top_k_logits=torch.randn(4, 8),
            next_token_global_top_k_token_ids=torch.arange(32).reshape(4, 8),
            hidden_states=torch.randn(6, 16),
        )
        forward_batch = SimpleNamespace(
            _tpu_real_q=3,
            _tpu_real_b=2,
            batch_size=4,
            input_ids=torch.arange(6),
            positions=torch.arange(6),
        )

        _unpad_decoder_outputs(output, forward_batch)

        self.assertIsNone(output.next_token_logits)
        self.assertEqual(output.next_token_global_top_k_logits.shape[0], 2)
        self.assertEqual(output.next_token_global_top_k_token_ids.shape[0], 2)
        self.assertEqual(output.hidden_states.shape[0], 3)
        self.assertEqual(forward_batch.batch_size, 2)
        self.assertEqual(forward_batch.input_ids.shape[0], 3)


class TestInstallExtendCap(CustomTestCase):
    def _install_policy(self, *, prefill_max_requests, dp_attention):
        model = SimpleNamespace(
            forward=MagicMock(),
            prepare_forward_batch=MagicMock(),
        )
        model_runner = SimpleNamespace(
            model=model,
            forward=MagicMock(),
            model_config=SimpleNamespace(context_len=4096),
            server_args=SimpleNamespace(
                chunked_prefill_size=None,
                max_prefill_tokens=2048,
                torch_compile_max_bs=64,
                prefill_max_requests=prefill_max_requests,
                unsafe_save_torch_compile_bundle=None,
                unsafe_load_torch_compile_bundle=None,
            ),
        )
        t_bucketer = Bucketer(name="Token", buckets=[64, 2048])
        b_bucketer = Bucketer(name="Bs", buckets=[16, 32, 64])
        policy = DecoderTpuCompilePolicy(model_runner=model_runner, owner=model)
        policy.backend = "eager"

        with patch(
            "sglang.srt.hardware_backend.tpu.compilation.decoder_compile_policy."
            "build_bucketers",
            return_value=(t_bucketer, b_bucketer),
        ), patch(
            "sglang.srt.hardware_backend.tpu.compilation.decoder_compile_policy."
            "is_dp_attention_enabled",
            return_value=dp_attention,
        ), patch(
            "sglang.srt.hardware_backend.tpu.compilation.decoder_compile_policy."
            "torch.compile",
            return_value=MagicMock(),
        ), patch.object(
            policy, "_install_dp_idle_as_decode"
        ):
            self.assertTrue(policy.install())

        return policy

    def test_non_dp_uses_configured_prefill_cap(self):
        policy = self._install_policy(prefill_max_requests=16, dp_attention=False)
        self.assertEqual(policy.extend_bs, 16)

    def test_non_dp_without_prefill_cap_uses_decode_max(self):
        policy = self._install_policy(prefill_max_requests=None, dp_attention=False)
        self.assertEqual(policy.extend_bs, 64)

    def test_dp_attention_uses_decode_max(self):
        policy = self._install_policy(prefill_max_requests=16, dp_attention=True)
        self.assertEqual(policy.extend_bs, 64)

    def test_target_model_keeps_outer_compile_boundary(self):
        original_outer = MagicMock(name="outer_forward")
        original_backbone = MagicMock(name="backbone_forward")
        model = SimpleNamespace(
            forward=original_outer,
            model=SimpleNamespace(forward=original_backbone),
            prepare_forward_batch=MagicMock(),
        )
        model_runner = SimpleNamespace(
            model=model,
            forward=MagicMock(),
            is_draft_worker=False,
            model_config=SimpleNamespace(context_len=4096),
            server_args=SimpleNamespace(
                chunked_prefill_size=None,
                max_prefill_tokens=2048,
                torch_compile_max_bs=64,
                prefill_max_requests=16,
                unsafe_save_torch_compile_bundle=None,
                unsafe_load_torch_compile_bundle=None,
            ),
        )
        policy = DecoderTpuCompilePolicy(model_runner=model_runner, owner=model)
        policy.backend = "eager"
        compiled_outer = MagicMock(name="compiled_outer")

        with patch(
            "sglang.srt.hardware_backend.tpu.compilation.decoder_compile_policy."
            "build_bucketers",
            return_value=(
                Bucketer(name="Token", buckets=[64, 2048]),
                Bucketer(name="Bs", buckets=[16, 32, 64]),
            ),
        ), patch(
            "sglang.srt.hardware_backend.tpu.compilation.decoder_compile_policy."
            "is_dp_attention_enabled",
            return_value=False,
        ), patch(
            "sglang.srt.hardware_backend.tpu.compilation.decoder_compile_policy."
            "torch.compile",
            return_value=compiled_outer,
        ) as compile_mock:
            self.assertTrue(policy.install())

        self.assertIs(model.model.forward, original_backbone)
        self.assertIs(policy.compiled_forward_shim.fn, compiled_outer)
        compile_mock.assert_called_once()
        self.assertIs(compile_mock.call_args.args[0].__wrapped__, original_outer)

    def test_draft_compiles_backbone_and_logits_as_separate_regions(self):
        original_outer = MagicMock(name="outer_forward")
        original_backbone = MagicMock(name="backbone_forward")
        original_logits = MagicMock(name="logits_forward")
        logits_processor = SimpleNamespace(forward=original_logits)
        model = SimpleNamespace(
            forward=original_outer,
            model=SimpleNamespace(forward=original_backbone),
            logits_processor=logits_processor,
            prepare_forward_batch=MagicMock(),
        )
        model_runner = SimpleNamespace(
            model=model,
            forward=MagicMock(),
            is_draft_worker=True,
            model_config=SimpleNamespace(context_len=4096),
            server_args=SimpleNamespace(
                chunked_prefill_size=None,
                max_prefill_tokens=2048,
                torch_compile_max_bs=64,
                prefill_max_requests=16,
                unsafe_save_torch_compile_bundle=None,
                unsafe_load_torch_compile_bundle=None,
            ),
        )
        policy = DecoderTpuCompilePolicy(model_runner=model_runner, owner=model)
        policy.backend = "eager"
        compiled_backbone = MagicMock(name="compiled_backbone")
        compiled_logits = MagicMock(name="compiled_logits")

        with patch(
            "sglang.srt.hardware_backend.tpu.compilation.decoder_compile_policy."
            "build_bucketers",
            return_value=(
                Bucketer(name="Token", buckets=[64, 2048]),
                Bucketer(name="Bs", buckets=[16, 32, 64]),
            ),
        ), patch(
            "sglang.srt.hardware_backend.tpu.compilation.decoder_compile_policy."
            "is_dp_attention_enabled",
            return_value=False,
        ), patch(
            "sglang.srt.hardware_backend.tpu.compilation.decoder_compile_policy."
            "torch.compile",
            side_effect=[compiled_backbone, compiled_logits],
        ) as compile_mock:
            self.assertTrue(policy.install())

        self.assertEqual(compile_mock.call_count, 2)
        self.assertIs(model.model.forward, compiled_backbone)
        self.assertIs(logits_processor.forward, compiled_logits)
        self.assertIs(policy.compiled_forward_shim.fn, original_outer)


class TestQwen3VLDecoderPolicy(CustomTestCase):
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

        result = policy.finalize_forward_batch(out, self._forward_batch())

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

        result = policy.finalize_forward_batch(hidden_states, forward_batch)
        last_index = torch.cumsum(forward_batch.extend_seq_lens, dim=0) - 1
        sampled_hidden_states = result[last_index]

        self.assertEqual(forward_batch.batch_size, 2)
        self.assertEqual(forward_batch.extend_seq_lens.tolist(), [2, 1])
        self.assertEqual(forward_batch.extend_seq_lens_cpu, [2, 1])
        self.assertEqual(forward_batch.mm_input_embeds.shape, (3, 4))
        self.assertEqual(sampled_hidden_states.shape, (2, 4))

    def test_direct_warmup_matches_serving_call_convention(self):
        model_runner = _model_runner()
        model_runner.model.get_image_feature = MagicMock()
        model_runner.model.get_input_embeddings = MagicMock(
            return_value=torch.nn.Embedding(2, 4)
        )
        model_runner.model.num_deepstack_embeddings = 2
        forward_batch = SimpleNamespace(
            input_ids=torch.ones(8, dtype=torch.int64),
            positions=torch.arange(8),
            forward_mode=ForwardMode.EXTEND,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            batch_size=2,
        )
        policy = Qwen3VLDecoderTpuCompilePolicy(
            model_runner=model_runner, owner=model_runner.model
        )
        policy._make_dummy_forward_batch = MagicMock(return_value=forward_batch)

        args, kwargs = policy._make_warmup_inputs(
            num_tokens=8,
            bs=2,
            mode=ForwardMode.EXTEND,
            variant="deepstack",
        )

        self.assertEqual(args, ())
        self.assertEqual(
            set(kwargs),
            {
                "input_ids",
                "positions",
                "forward_batch",
                "input_embeds",
                "input_deepstack_embeds",
            },
        )
        deepstack_key = policy._direct_compile_specialization_key(None, **kwargs)
        kwargs.pop("input_deepstack_embeds")
        self.assertNotEqual(
            deepstack_key, policy._direct_compile_specialization_key(None, **kwargs)
        )

    def test_slices_hidden_and_aux_tuple(self):
        policy = Qwen3VLDecoderTpuCompilePolicy()
        hidden = torch.zeros(5, 4)
        aux = [torch.ones(5, 4), torch.full((5, 4), 2.0)]

        result_hidden, result_aux = policy.finalize_forward_batch(
            (hidden, aux), self._forward_batch()
        )

        self.assertEqual(result_hidden.shape, (3, 4))
        self.assertEqual([value.shape for value in result_aux], [(3, 4), (3, 4)])

    def test_keeps_pipeline_proxy_tensors_padded_for_next_stage(self):
        policy = Qwen3VLDecoderTpuCompilePolicy()
        out = PPProxyTensors(
            {"hidden_states": torch.zeros(5, 4), "residual": torch.ones(5, 4)}
        )

        result = policy.finalize_forward_batch(out, self._forward_batch())

        self.assertEqual(result["hidden_states"].shape, (5, 4))
        self.assertEqual(result["residual"].shape, (5, 4))

    def test_spec_policy_delegates_raw_vl_output_unpadding(self):
        class _SpecQwenPolicy(
            Eagle3CompilePolicyMixin, Qwen3VLDecoderTpuCompilePolicy
        ):
            pass

        policy = object.__new__(_SpecQwenPolicy)
        forward_batch = self._forward_batch()
        forward_batch.forward_mode = ForwardMode.TARGET_VERIFY
        out = torch.arange(20, dtype=torch.float32).reshape(5, 4)

        result = policy.finalize_forward_batch(out, forward_batch)

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
        self.assertEqual(fb.out_cache_loc.dtype, torch.int32)
        # Per-request tensors at bs.
        self.assertEqual(fb.seq_lens.shape[0], 4)
        self.assertEqual(fb.seq_lens.dtype, torch.int64)
        self.assertEqual(fb.extend_seq_lens.shape[0], 4)
        self.assertEqual(fb.extend_seq_lens.tolist(), [1, 1, 1, 1])
        self.assertEqual(fb.extend_logprob_start_lens_cpu, [0, 0, 0, 0])
        # KV writes go to dummy slot 0 across the whole token range.
        self.assertEqual(fb.out_cache_loc.sum().item(), 0)

    def test_hybrid_swa_uses_full_attention_allocator_dtype(self):
        mr = _model_runner()
        allocator = object.__new__(SWATokenToKVPoolAllocator)
        allocator.full_attn_allocator = SimpleNamespace(
            free_pages=torch.arange(4, dtype=torch.int32)
        )
        mr.token_to_kv_pool_allocator = allocator
        policy = _make_policy(mr)

        fb = policy._make_dummy_forward_batch(
            num_tokens=8, bs=8, mode=ForwardMode.DECODE
        )

        self.assertEqual(fb.out_cache_loc.dtype, torch.int32)

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

    def test_plain_warmup_uses_serving_sampling_info(self):
        policy = _make_policy()

        fb = policy._make_dummy_forward_batch(
            num_tokens=4,
            bs=4,
            mode=ForwardMode.DECODE,
        )

        self.assertIsInstance(fb.sampling_info, SamplingBatchInfo)
        self.assertEqual(fb.sampling_info.uniform_top_k, 50)
        self.assertEqual(fb.sampling_info.top_ks.tolist(), [50] * 4)
        self.assertIsNone(fb.sampling_info.penalizer_orchestrator)


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

    def test_mixed_chunk_adds_fixed_decode_max_column(self):
        mr = _model_runner()
        mr.server_args = SimpleNamespace(enable_mixed_chunk=True)
        policy = _make_policy(
            mr,
            token_buckets=(16, 64, 128),
            bs_buckets=(16, 32, 64),
        )
        policy.extend_bs = 16

        _run_cases(policy, ForwardMode.EXTEND)

        self.assertEqual(
            mr.attn_backend.calls,
            [(16, 16), (64, 16), (128, 16), (64, 64), (128, 64)],
        )
        self.assertEqual(
            [call.args[2].forward_mode for call in mr.model.forward.call_args_list],
            [
                ForwardMode.EXTEND,
                ForwardMode.EXTEND,
                ForwardMode.EXTEND,
                ForwardMode.MIXED,
                ForwardMode.MIXED,
            ],
        )

    def test_qwen3_vl_extend_uses_configured_prefill_cap(self):
        mr = _model_runner()
        mr.server_args = SimpleNamespace(enable_mixed_chunk=False)
        mr.model.get_image_feature = MagicMock()
        mr.model.get_input_embeddings = MagicMock()
        policy = Qwen3VLDecoderTpuCompilePolicy(
            model_runner=mr,
            t_bucketer=Bucketer(name="Token", buckets=[64, 256]),
            b_bucketer=Bucketer(name="Bs", buckets=[16, 32, 64]),
            owner=mr.model,
        )
        policy.extend_bs = 16
        policy._run_one = MagicMock()

        with get_parallel().override(tp_rank=1):
            policy._precompile_extend()

        self.assertEqual(policy._run_one.call_count, 2)
        self.assertEqual(
            {call.kwargs["bs"] for call in policy._run_one.call_args_list},
            {16},
        )

    def test_extend_skips_token_bucket_below_bs(self):
        # A 4-token bucket can't hold the fixed 8-req extend cap.
        mr = _model_runner()
        policy = _make_policy(mr, token_buckets=(4, 64), bs_buckets=(1, 8))
        _run_cases(policy, ForwardMode.EXTEND)
        self.assertEqual(mr.model.forward.call_count, 1)
        self.assertEqual(mr.attn_backend.calls, [(64, 8)])

    def test_extend_batch_cap_clamped_to_token_bucket(self):
        # The extend batch cap is clamped to the largest token bucket
        # (_extend_batch_cap = min(token_max, bs_max)), so a lone small token
        # bucket still yields a valid (t, t) shape instead of an empty set.
        mr = _model_runner()
        policy = _make_policy(mr, token_buckets=(4,), bs_buckets=(8,))
        _run_cases(policy, ForwardMode.EXTEND)
        self.assertEqual(mr.model.forward.call_count, 1)
        self.assertEqual(mr.attn_backend.calls, [(4, 4)])

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
        with get_parallel().override(tp_rank=1):
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

    def test_materialize_out_syncs_global_top_k_and_flagged_kv_side_effects(self):
        global_top_k_logits = torch.zeros(1)
        global_top_k_token_ids = torch.zeros(1, dtype=torch.int32)
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
        # torch_tpu isn't installed on the CPU runners this suite targets; stub
        # the module tree so _materialize_out's
        # `from torch_tpu._internal.sync.sync import synchronize` resolves and is
        # patchable (patch.dict also overrides a real torch_tpu on a TPU host).
        tt = ModuleType("torch_tpu")
        tt._internal = ModuleType("torch_tpu._internal")
        tt._internal.sync = ModuleType("torch_tpu._internal.sync")
        tt._internal.sync.sync = ModuleType("torch_tpu._internal.sync.sync")
        tt._internal.sync.sync.synchronize = MagicMock()
        stubs = {
            "torch_tpu": tt,
            "torch_tpu._internal": tt._internal,
            "torch_tpu._internal.sync": tt._internal.sync,
            "torch_tpu._internal.sync.sync": tt._internal.sync.sync,
        }
        with patch.dict(sys.modules, stubs), patch(
            "torch_tpu._internal.sync.sync.synchronize"
        ) as sync:
            _materialize_out(
                LogitsProcessorOutput(
                    next_token_logits=None,
                    next_token_global_top_k_logits=global_top_k_logits,
                    next_token_global_top_k_token_ids=global_top_k_token_ids,
                ),
                forward_batch.attn_backend,
                forward_batch.token_to_kv_pool,
            )
        sync.assert_called_once()
        tensors, kwargs = sync.call_args
        self.assertEqual(
            tensors[0], [global_top_k_logits, global_top_k_token_ids, kv0, kv1]
        )
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
        # _run_one seeds req_to_token rows; needs a real indexable tensor
        # (draft extend seeds one row per request in the bucket).
        req_to_token_pool=SimpleNamespace(
            req_to_token=torch.zeros(16, 4096, dtype=torch.int64)
        ),
        token_to_kv_pool=SimpleNamespace(),
        token_to_kv_pool_allocator=SimpleNamespace(
            free_pages=torch.arange(4, dtype=torch.int32)
        ),
        model_config=SimpleNamespace(hidden_size=16),
        dtype=torch.bfloat16,
        is_draft_worker=is_draft_worker,
        server_args=SimpleNamespace(
            speculative_num_steps=num_steps,
            # topk=1 chain: verify width == num_steps + 1.
            speculative_num_draft_tokens=num_steps + 1,
            speculative_algorithm="EAGLE3",
            speculative_eagle_topk=1,
            page_size=1,
            dp_size=1,
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
    """Target spec precompile covers its serving forward families."""

    def test_extend_full_uses_configured_prefill_cap(self):
        mr = _spec_model_runner(num_steps=3)
        policy = _make_spec_policy(
            mr,
            token_buckets=(1024, 4096),
            bs_buckets=(16, 32, 64),
        )
        policy.extend_bs = 16

        with get_parallel().override(tp_rank=1):
            policy._precompile_extend_full()

        self.assertEqual(mr.attn_backend.calls, [(1024, 16), (4096, 16)])

    def test_target_precompiles_all_spec_modes(self):
        mr = _spec_model_runner(num_steps=3)  # draft_token_num = 4
        policy = _make_spec_policy(mr, token_buckets=(8, 64), bs_buckets=(1, 4))
        # precompile() spans two modules: the base EXTEND/DECODE warmup lives in
        # decoder_compile_policy (via super()), the spec TARGET_VERIFY/EXTEND(FULL)
        # in spec_decode_compile_policy. One topology override serves both.
        with get_parallel().override(tp_rank=1):
            policy.precompile()
        # EXTEND (8,4),(64,4)=2  DECODE (1,1),(4,4)=2
        # TARGET_VERIFY buckets bs: (4,1),(16,4)=2  EXTEND(FULL) 2
        self.assertEqual(mr.model.forward.call_count, 8)
        self.assertIn((16, 4), mr.attn_backend.calls)
        self.assertIn((4, 1), mr.attn_backend.calls)

    def test_draft_worker_skips_precompile(self):
        mr = _spec_model_runner(is_draft_worker=True)
        policy = _make_spec_policy(mr)
        with get_parallel().override(tp_rank=1):
            policy.precompile()
        self.assertEqual(mr.model.forward.call_count, 0)
        self.assertEqual(mr.attn_backend.calls, [])

    def test_draft_precompiles_all_serving_forward_families(self):
        mr = _spec_model_runner(is_draft_worker=True, num_steps=3)
        mr.server_args.page_size = 8
        policy = _make_spec_policy(mr, token_buckets=(8, 64), bs_buckets=(1, 4))
        decode_backends = [_FakeAttnBackend(), _FakeAttnBackend()]
        draft_backend = SimpleNamespace(attn_backends=decode_backends)
        draft_extend_backend = _FakeAttnBackend()

        with get_parallel().override(tp_rank=1):
            policy.precompile_draft(draft_backend, draft_extend_backend)

        self.assertEqual(mr.attn_backend.calls, [(8, 4), (64, 4)])
        self.assertEqual(decode_backends[0].calls, [(1, 1), (4, 4)])
        self.assertEqual(decode_backends[1].calls, [(1, 1), (4, 4)])
        self.assertEqual(draft_extend_backend.calls, [(4, 1), (16, 4)])
        modes = [call.args[2].forward_mode for call in mr.model.forward.call_args_list]
        self.assertEqual(
            modes,
            [ForwardMode.EXTEND] * 2
            + [ForwardMode.DECODE] * 4
            + [ForwardMode.DRAFT_EXTEND_V2] * 2,
        )
        capture_modes = [
            call.args[2].capture_hidden_mode for call in mr.model.forward.call_args_list
        ]
        self.assertEqual(
            capture_modes,
            [CaptureHiddenMode.LAST] * 6 + [CaptureHiddenMode.FULL] * 2,
        )

    def test_non_eagle3_algorithm_raises(self):
        # The spec policy enforces the supported algorithm at construction (only
        # EAGLE3 today); an unsupported algorithm fails loudly rather than
        # silently degrading to a non-spec policy.
        mr = _spec_model_runner()
        mr.server_args.speculative_algorithm = "NGRAM"
        with self.assertRaisesRegex(ValueError, "only supports EAGLE3"):
            _make_spec_policy(mr)


class TestSpecDummyForwardBatch(CustomTestCase):
    """Spec dummy batches carry spec_info and slot-1 KV seeding."""

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
        self.assertIsNone(fb.sampling_info)
        # Runtime TARGET_VERIFY is decode-like and leaves extend metadata unset.
        self.assertIsNone(fb.extend_seq_lens)
        self.assertIsNone(fb.extend_prefix_lens)
        self.assertIsNone(fb.extend_num_tokens)
        # Slot-1 KV seeding (avoids draft/target pool conflicts): not slot 0.
        self.assertEqual(fb.out_cache_loc.tolist(), [1] * 16)
        self.assertEqual(fb.out_cache_loc.dtype, torch.int64)
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
        self.assertEqual(fb.out_cache_loc.dtype, torch.int32)

    def test_draft_decode_uses_spec_v2_int64_cache_locations(self):
        policy = _make_spec_policy(is_draft_worker=True)
        fb = policy._make_dummy_forward_batch(
            num_tokens=4,
            bs=4,
            mode=ForwardMode.DECODE,
            capture_hidden_mode=CaptureHiddenMode.LAST,
            spec_info=policy._make_draft_decode_spec_info(),
        )

        self.assertEqual(fb.out_cache_loc.dtype, torch.int64)

    def test_draft_extend_uses_per_request_kv_pages(self):
        mr = _spec_model_runner(is_draft_worker=True, num_steps=3)
        mr.server_args.page_size = 8
        policy = _make_spec_policy(mr, bs_buckets=(1, 4))
        fb = policy._make_dummy_forward_batch(
            num_tokens=16,
            bs=4,
            mode=ForwardMode.DRAFT_EXTEND_V2,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            spec_info=policy._make_draft_extend_spec_info(),
        )

        self.assertEqual(fb.capture_hidden_mode, CaptureHiddenMode.FULL)
        self.assertEqual(fb.spec_info.capture_hidden_mode, CaptureHiddenMode.LAST)
        self.assertEqual(fb.positions.tolist(), [4, 5, 6, 7] * 4)
        self.assertEqual(
            fb.out_cache_loc.tolist(),
            list(range(12, 16))
            + list(range(20, 24))
            + list(range(28, 32))
            + list(range(36, 40)),
        )
        self.assertEqual(fb.out_cache_loc.dtype, torch.int64)
        self.assertEqual(fb.extend_seq_lens.tolist(), [4] * 4)
        self.assertEqual(fb.extend_prefix_lens.tolist(), [4] * 4)

        policy._seed_dummy_req_to_token(fb)
        req_to_token = mr.req_to_token_pool.req_to_token
        for row in range(1, 5):
            self.assertEqual(
                req_to_token[row, :8].tolist(), list(range(row * 8, row * 8 + 8))
            )

    def test_draft_extend_uses_multiple_pages_when_tokens_exceed_page(self):
        mr = _spec_model_runner(is_draft_worker=True, num_steps=3)
        mr.server_args.page_size = 2
        policy = _make_spec_policy(mr, bs_buckets=(1, 2))
        fb = policy._make_dummy_forward_batch(
            num_tokens=8,
            bs=2,
            mode=ForwardMode.DRAFT_EXTEND_V2,
            capture_hidden_mode=CaptureHiddenMode.LAST,
            spec_info=policy._make_draft_extend_spec_info(),
        )

        self.assertEqual(fb.positions.tolist(), [0, 1, 2, 3] * 2)
        self.assertEqual(
            fb.out_cache_loc.tolist(),
            list(range(4, 8)) + list(range(8, 12)),
        )
        self.assertEqual(fb.seq_lens.tolist(), [4, 4])

        policy._seed_dummy_req_to_token(fb)
        req_to_token = mr.req_to_token_pool.req_to_token
        self.assertEqual(req_to_token[1, :4].tolist(), list(range(4, 8)))
        self.assertEqual(req_to_token[2, :4].tolist(), list(range(8, 12)))


class TestSpecDraftDecodePadding(CustomTestCase):
    def test_pads_mrope_and_cpu_seq_lens_from_three_to_four(self):
        policy = _make_spec_policy(num_steps=3, bs_buckets=(1, 4))
        forward_batch = SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            spec_info=EagleDraftInput(
                topk_p=torch.zeros((3, 1)),
                topk_index=torch.zeros((3, 1), dtype=torch.int64),
                hidden_states=torch.zeros((3, 16)),
            ),
            seq_lens=torch.tensor([8, 7, 6], dtype=torch.int32),
            seq_lens_cpu=torch.tensor([8, 7, 6], dtype=torch.int32),
            req_pool_indices=torch.tensor([0, 1, 2]),
            req_pool_indices_cpu=torch.tensor([0, 1, 2]),
            positions=torch.arange(3),
            mrope_positions=torch.arange(9).reshape(3, 3),
            out_cache_loc=torch.arange(3),
            mm_inputs=None,
            batch_size=3,
        )

        policy.prepare_spec_forward_batch(forward_batch)

        self.assertEqual(forward_batch.batch_size, 4)
        self.assertEqual(forward_batch._tpu_real_b, 3)
        self.assertEqual(forward_batch.seq_lens.shape, (4,))
        self.assertEqual(forward_batch.seq_lens_cpu.tolist(), [8, 7, 6, 0])
        self.assertEqual(forward_batch.req_pool_indices_cpu.tolist(), [0, 1, 2, 0])
        self.assertEqual(forward_batch.mrope_positions.shape, (3, 4))
        self.assertTrue(
            torch.equal(
                forward_batch.mrope_positions[:, :3],
                torch.arange(9).reshape(3, 3),
            )
        )
        self.assertTrue(torch.all(forward_batch.mrope_positions[:, 3:] == 0))
        self.assertEqual(forward_batch.spec_info.topk_p.shape, (4, 1))
        self.assertTrue(forward_batch.spec_info.keep_outputs_padded)

    def test_idle_does_not_request_padded_outputs(self):
        policy = _make_spec_policy(num_steps=3, bs_buckets=(1, 4))
        spec_info = EagleDraftInput(
            topk_p=torch.zeros((0, 1)),
            topk_index=torch.zeros((0, 1), dtype=torch.int64),
            hidden_states=torch.zeros((0, 16)),
        )
        forward_batch = SimpleNamespace(
            forward_mode=ForwardMode.IDLE,
            spec_info=spec_info,
        )

        policy.prepare_spec_forward_batch(forward_batch)

        self.assertFalse(spec_info.keep_outputs_padded)


class TestSpecDraftExtendPadding(CustomTestCase):
    def test_normalizes_stale_cpu_lists_at_exact_bucket(self):
        policy = _make_spec_policy(num_steps=3, bs_buckets=(1, 4))
        forward_batch = SimpleNamespace(
            input_ids=torch.ones(16, dtype=torch.int32),
            seq_lens=torch.ones(4, dtype=torch.int32),
            spec_info=None,
            extend_seq_lens_cpu=[4] * 64,
            extend_prefix_lens_cpu=torch.arange(64),
            extend_logprob_start_lens_cpu=[0] * 64,
            mm_inputs=[None] * 64,
        )

        policy._pad_draft_extend(forward_batch)

        self.assertEqual(forward_batch.extend_seq_lens_cpu, [4] * 4)
        self.assertEqual(forward_batch.extend_prefix_lens_cpu, [0, 1, 2, 3])
        self.assertEqual(forward_batch.extend_logprob_start_lens_cpu, [0] * 4)
        self.assertEqual(forward_batch.mm_inputs, [None] * 4)


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
            seq_lens=torch.zeros(rows, dtype=torch.int32),
            spec_info=None,
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
        policy.finalize_forward_batch(out, fb)
        # Verify emits one logit per draft token -> real_q, not real_b.
        self.assertEqual(out.next_token_logits.shape[0], 16)
        self.assertEqual(out.hidden_states.shape[0], 16)  # FULL -> real_q
        self.assertEqual(fb.batch_size, 4)
        self.assertEqual(fb.seq_lens.shape[0], 4)

    def test_capture_last_slices_hidden_to_real_b(self):
        policy = _make_spec_policy()
        out, fb = self._out_and_batch(
            ForwardMode.DRAFT_EXTEND_V2,
            CaptureHiddenMode.LAST,
            real_q=8,
            real_b=4,
            rows=16,
        )
        policy.finalize_forward_batch(out, fb)
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
        policy.finalize_forward_batch(out, fb)
        self.assertEqual(out.next_token_logits.shape[0], 4)
        self.assertEqual(out.hidden_states.shape[0], 4)

    def test_final_draft_outputs_restore_real_batch_size(self):
        policy = _make_spec_policy()
        fb = SimpleNamespace(_tpu_real_b=2, batch_size=4, spec_info=None)
        outputs = (torch.arange(4), torch.arange(8).reshape(4, 2), None)

        unpadded = policy.finalize_spec_forward_batch(outputs, fb)

        self.assertEqual([t.shape[0] for t in unpadded[:2]], [2, 2])
        self.assertIsNone(unpadded[2])
        self.assertEqual(fb.batch_size, 2)


class TestSpecForwardBatchPreparation(CustomTestCase):
    """``prepare_spec_forward_batch`` dispatches on the draft forward mode."""

    @staticmethod
    def _policy():
        policy = _make_spec_policy()
        policy._pad_draft_decode = MagicMock()
        policy._pad_draft_extend = MagicMock()
        return policy

    def test_draft_decode_dispatches(self):
        policy = self._policy()
        spec_info = SimpleNamespace(keep_outputs_padded=False)
        forward_batch = SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            spec_info=spec_info,
        )

        policy.prepare_spec_forward_batch(forward_batch)

        policy._pad_draft_decode.assert_called_once_with(forward_batch)
        self.assertTrue(spec_info.keep_outputs_padded)

    def test_draft_extend_dispatches(self):
        policy = self._policy()
        forward_batch = SimpleNamespace(forward_mode=ForwardMode.DRAFT_EXTEND_V2)

        policy.prepare_spec_forward_batch(forward_batch)

        policy._pad_draft_extend.assert_called_once_with(forward_batch)

    def test_idle_is_noop(self):
        policy = self._policy()
        forward_batch = SimpleNamespace(forward_mode=ForwardMode.IDLE)

        policy.prepare_spec_forward_batch(forward_batch)

        policy._pad_draft_decode.assert_not_called()
        policy._pad_draft_extend.assert_not_called()

    def test_target_verify_is_noop(self):
        policy = self._policy()
        forward_batch = SimpleNamespace(forward_mode=ForwardMode.TARGET_VERIFY)

        policy.prepare_spec_forward_batch(forward_batch)

        policy._pad_draft_decode.assert_not_called()
        policy._pad_draft_extend.assert_not_called()

    def test_draft_decode_buckets_before_input_ids_are_built(self):
        policy = _make_spec_policy(num_steps=3, bs_buckets=(1, 4))
        draft_input = EagleDraftInput(
            topk_p=torch.zeros((2, 1)),
            topk_index=torch.zeros((2, 1), dtype=torch.int64),
            hidden_states=torch.zeros((2, 16)),
        )
        forward_batch = SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            spec_info=draft_input,
            input_ids=None,
            seq_lens=torch.tensor([8, 7], dtype=torch.int32),
            seq_lens_cpu=torch.tensor([8, 7], dtype=torch.int32),
            req_pool_indices=torch.tensor([0, 1]),
            req_pool_indices_cpu=torch.tensor([0, 1]),
            positions=torch.arange(2),
            mrope_positions=None,
            out_cache_loc=torch.arange(2),
            mm_inputs=None,
            batch_size=2,
        )

        policy.prepare_spec_forward_batch(forward_batch)

        self.assertEqual(forward_batch.batch_size, 4)
        self.assertEqual(forward_batch._tpu_real_b, 2)
        self.assertEqual(forward_batch.seq_lens.shape[0], 4)
        self.assertEqual(draft_input.topk_p.shape[0], 4)
        state = draft_input.padding_state
        self.assertEqual(state.logical_batch_size, 2)
        self.assertEqual(state.physical_batch_size, 4)
        self.assertEqual(state.logical_num_tokens, 2)
        self.assertEqual(state.physical_num_tokens, 4)

    def test_draft_extend_buckets_to_the_full_tree_width(self):
        policy = _make_spec_policy(num_steps=3, bs_buckets=(1, 4))
        draft_extend_input = EagleDraftExtendInput(
            hidden_states=torch.zeros((8, 8)),
            num_correct_drafts=torch.tensor([2, 1], dtype=torch.int32),
            num_accept_tokens=torch.tensor([3, 2], dtype=torch.int32),
            num_accept_tokens_cpu=[3, 2],
            input_ids=torch.arange(8),
            seq_lens=torch.tensor([8, 7], dtype=torch.int32),
            seq_lens_cpu=torch.tensor([8, 7], dtype=torch.int32),
            req_pool_indices=torch.tensor([0, 1]),
            positions=torch.arange(8),
        )
        pad_calls = []

        def fake_pad_inputs_to_size(model_runner, num_tokens, bs):
            pad_calls.append((num_tokens, bs))

        forward_batch = SimpleNamespace(
            forward_mode=ForwardMode.DRAFT_EXTEND_V2,
            spec_info=draft_extend_input,
            input_ids=draft_extend_input.input_ids,
            seq_lens=draft_extend_input.seq_lens,
            batch_size=2,
            _pad_inputs_to_size=fake_pad_inputs_to_size,
        )

        policy.prepare_spec_forward_batch(forward_batch)

        # bs=2 buckets to 4; draft extend fills draft_token_num (=4) per request.
        self.assertEqual(pad_calls, [(16, 4)])
        self.assertEqual(forward_batch.batch_size, 4)
        self.assertEqual(forward_batch._tpu_real_b, 2)
        state = draft_extend_input.padding_state
        self.assertEqual(state.logical_batch_size, 2)
        self.assertEqual(state.physical_batch_size, 4)
        self.assertEqual(state.logical_num_tokens, 8)
        self.assertEqual(state.physical_num_tokens, 16)


class TestDraftPrecompile(CustomTestCase):
    """``precompile_draft`` warms the three draft forward families."""

    def test_draft_worker_compiles_prefill_decode_and_extend(self):
        mr = _spec_model_runner(is_draft_worker=True, num_steps=3)
        policy = _make_spec_policy(mr, token_buckets=(8, 64), bs_buckets=(1, 4))
        step_backends = [_FakeAttnBackend() for _ in range(3)]
        draft_attn_backend = SimpleNamespace(attn_backends=step_backends)
        draft_extend_attn_backend = _FakeAttnBackend()

        with get_parallel().override(tp_rank=1):
            policy.precompile_draft(draft_attn_backend, draft_extend_attn_backend)

        # EXTEND(LAST) 2 token buckets at the bs cap + DECODE 2 bs x 3 steps
        # + DRAFT_EXTEND_V2 2 bs buckets.
        self.assertEqual(mr.model.forward.call_count, 2 + 6 + 2)
        # Draft decode warms each step's own backend, not the runner's.
        self.assertEqual([len(b.calls) for b in step_backends], [2, 2, 2])
        self.assertEqual(step_backends[0].calls, [(1, 1), (4, 4)])
        # DRAFT_EXTEND_V2: bs * draft_token_num tokens per shape.
        self.assertEqual(draft_extend_attn_backend.calls, [(4, 1), (16, 4)])
        # Only the draft-prefill EXTEND shapes go to the runner's own backend.
        self.assertEqual(mr.attn_backend.calls, [(8, 4), (64, 4)])

    def test_target_worker_skips_draft_precompile(self):
        mr = _spec_model_runner(is_draft_worker=False)
        policy = _make_spec_policy(mr)

        policy.precompile_draft(SimpleNamespace(attn_backends=[]), _FakeAttnBackend())

        self.assertEqual(mr.model.forward.call_count, 0)


# Speculative verification padding fixtures.
BS = 2
BUCKET_BS = 4
DRAFT_TOKEN_NUM = 4
VOCAB_SIZE = 10
LOGICAL_TOKENS = BS * DRAFT_TOKEN_NUM
PADDED_TOKENS = BUCKET_BS * DRAFT_TOKEN_NUM


def _policy():
    """A spec policy whose only job here is to own the verify padding."""
    model_runner = SimpleNamespace(
        model=SimpleNamespace(forward=None, prepare_forward_batch=None),
        is_draft_worker=False,
        model_config=SimpleNamespace(hidden_size=16),
        server_args=SimpleNamespace(
            speculative_num_steps=DRAFT_TOKEN_NUM - 1,
            speculative_num_draft_tokens=DRAFT_TOKEN_NUM,
            speculative_algorithm="EAGLE3",
            speculative_eagle_topk=1,
            page_size=1,
            dp_size=1,
        ),
    )
    return SpecDecoderTpuCompilePolicy(
        model_runner=model_runner,
        t_bucketer=Bucketer(name="Token", buckets=[PADDED_TOKENS]),
        b_bucketer=Bucketer(name="Bs", buckets=[1, BUCKET_BS]),
        owner=model_runner.model,
    )


@contextlib.contextmanager
def _serving_stubs():
    """Select TPU-native sampling and stub serving process state.

    This suite runs on CPU CI, where platform detection would otherwise leave
    ``verify_tree_greedy_func`` without a CPU implementation and its mutable
    outputs unwritten. Production TPU selects this torch-native path itself.
    """
    with patch("sglang.srt.speculative.eagle_utils._use_torch_native", True), patch(
        "sglang.srt.distributed.get_tp_group",
        return_value=SimpleNamespace(world_size=1),
    ), patch(
        "sglang.srt.layers.dp_attention.is_dp_attention_enabled",
        return_value=False,
    ), get_context().override_server_args(
        speculative_use_rejection_sampling=False,
        speculative_accept_threshold_single=1.0,
        speculative_accept_threshold_acc=1.0,
    ):
        yield


def _make_verify_input(deferred=True):
    verify_input = EagleVerifyInput.create_idle_input(
        topk=1,
        spec_steps=DRAFT_TOKEN_NUM - 1,
        num_verify_tokens=DRAFT_TOKEN_NUM,
        device="cpu",
    )
    verify_input.draft_token = torch.arange(1, LOGICAL_TOKENS + 1, dtype=torch.int32)
    verify_input.retrieve_index = torch.arange(
        LOGICAL_TOKENS, dtype=torch.int32
    ).reshape(BS, DRAFT_TOKEN_NUM)
    verify_input.retrieve_next_token = torch.full(
        (BS, DRAFT_TOKEN_NUM), -1, dtype=torch.int32
    )
    verify_input.retrieve_next_sibling = torch.full(
        (BS, DRAFT_TOKEN_NUM), -1, dtype=torch.int32
    )
    if deferred:
        verify_input.keep_outputs_padded = True
        verify_input.record_padding(
            logical_batch_size=BS,
            physical_batch_size=BUCKET_BS,
            logical_num_tokens=LOGICAL_TOKENS,
            physical_num_tokens=PADDED_TOKENS,
        )
    return verify_input


def _make_batch(temperature):
    return SimpleNamespace(
        device="cpu",
        forward_mode=ForwardMode.TARGET_VERIFY,
        has_grammar=False,
        return_logprob=False,
        seq_lens=torch.full((BS,), 16, dtype=torch.int32),
        sampling_info=SimpleNamespace(
            temperatures=torch.full((BS, 1), temperature or 1.0),
            top_ks=torch.full((BS,), VOCAB_SIZE, dtype=torch.int32),
            top_ps=torch.full((BS,), 0.9),
            need_top_k_sampling=True,  # top_ks=VOCAB_SIZE makes renorm an identity
            need_top_p_sampling=True,
            sampling_seed=None,  # unseeded: keep the pre-seed coin stream
            is_all_greedy=temperature == 0.0,
            acc_additive_penalties=None,
            acc_scaling_penalties=None,
            logit_bias=None,
        ),
    )


def _accepting_logits():
    """Target logits that accept one draft token for request 0 and the whole
    chain for request 1, so a run covers accept, reject and bonus sampling."""
    logits = torch.zeros((LOGICAL_TOKENS, VOCAB_SIZE))
    candidates = torch.arange(1, LOGICAL_TOKENS + 1).reshape(BS, DRAFT_TOKEN_NUM)
    for req, num_accepted in {0: 1, 1: DRAFT_TOKEN_NUM - 1}.items():
        for j in range(1, DRAFT_TOKEN_NUM):
            row = req * DRAFT_TOKEN_NUM + j - 1
            if j <= num_accepted:
                logits[row] = -10.0
                logits[row, candidates[req, j]] = 10.0
            elif j == num_accepted + 1:
                logits[row, candidates[req, j]] = -20.0
    return logits


class TestSpecVerifyPaddingRecord(CustomTestCase):
    """``prepare_spec_verify`` widens exactly the batch state verify samples from."""

    def test_deferral_decision_is_policy_owned(self):
        batch = _make_batch(temperature=0.7)
        verify_input = _make_verify_input(deferred=False)
        policy = _policy()

        self.assertTrue(
            policy.should_defer_verify_output_unpadding(batch, verify_input)
        )
        self.assertTrue(verify_input.keep_outputs_padded)

        verify_input.keep_outputs_padded = False
        self.assertFalse(
            NoopCompilePolicy().should_defer_verify_output_unpadding(
                batch, verify_input
            )
        )
        self.assertFalse(verify_input.keep_outputs_padded)

    def test_widens_every_per_request_input(self):
        batch = _make_batch(temperature=0.7)
        verify_input = _make_verify_input()
        logits = torch.zeros((PADDED_TOKENS, VOCAB_SIZE))

        padding = _policy().prepare_spec_verify(batch, verify_input, logits)

        self.assertIsNotNone(padding)
        self.assertEqual(batch.seq_lens.shape, (BUCKET_BS,))
        self.assertEqual(verify_input.draft_token.shape, (PADDED_TOKENS,))
        self.assertEqual(
            verify_input.retrieve_index.shape, (BUCKET_BS, DRAFT_TOKEN_NUM)
        )
        self.assertEqual(batch.sampling_info.temperatures.shape, (BUCKET_BS, 1))
        self.assertEqual(batch.sampling_info.top_ks.shape, (BUCKET_BS,))
        self.assertEqual(batch.sampling_info.top_ps.shape, (BUCKET_BS,))
        # Padded rows must leave the sampling math finite.
        self.assertEqual(batch.sampling_info.temperatures[BS:].tolist(), [[1.0], [1.0]])
        self.assertEqual(batch.sampling_info.top_ks[BS:].tolist(), [1, 1])
        self.assertEqual(batch.sampling_info.top_ps[BS:].tolist(), [1.0, 1.0])
        # A padded retrieve index points into the padded tail, never a real slot.
        self.assertTrue(torch.all(verify_input.retrieve_index[BS:] == -1))

    def test_restores_the_batch_even_when_verify_produced_nothing(self):
        batch = _make_batch(temperature=0.7)
        verify_input = _make_verify_input()
        logits = torch.zeros((PADDED_TOKENS, VOCAB_SIZE))
        before = {
            "seq_lens": batch.seq_lens,
            "draft_token": verify_input.draft_token,
            "retrieve_index": verify_input.retrieve_index,
            "temperatures": batch.sampling_info.temperatures,
            "top_ks": batch.sampling_info.top_ks,
        }

        padding = _policy().prepare_spec_verify(batch, verify_input, logits)
        _policy().finalize_spec_verify(padding, (None, None, None), (None, None))

        # Restored by identity, not by narrowing: the forward batch aliases
        # these same objects.
        self.assertIs(batch.seq_lens, before["seq_lens"])
        self.assertIs(verify_input.draft_token, before["draft_token"])
        self.assertIs(verify_input.retrieve_index, before["retrieve_index"])
        self.assertIs(batch.sampling_info.temperatures, before["temperatures"])
        self.assertIs(batch.sampling_info.top_ks, before["top_ks"])

    def test_a_raising_verify_still_puts_the_batch_back(self):
        """The worker calls finalize from a finally, so a failed verify must not
        leave the scheduler holding a widened batch."""
        batch = _make_batch(temperature=0.7)
        verify_input = _make_verify_input()
        logits = torch.zeros((PADDED_TOKENS, VOCAB_SIZE))
        policy = _policy()
        padding = policy.prepare_spec_verify(batch, verify_input, logits)
        self.assertEqual(batch.seq_lens.shape, (BUCKET_BS,))

        # What the finally block passes when eagle_sample never returned.
        verify_result, bookkeeping_result = policy.finalize_spec_verify(
            padding, (None, None, None), (None, None)
        )
        predict, accept_lens, accept_index = verify_result

        self.assertEqual(batch.seq_lens.shape, (BS,))
        self.assertEqual(verify_input.draft_token.shape, (LOGICAL_TOKENS,))
        self.assertEqual(batch.sampling_info.temperatures.shape, (BS, 1))
        self.assertEqual((predict, accept_lens, accept_index), (None, None, None))

    def test_slices_results_back_to_real_requests(self):
        batch = _make_batch(temperature=0.7)
        verify_input = _make_verify_input()
        logits = torch.zeros((PADDED_TOKENS, VOCAB_SIZE))
        padding = _policy().prepare_spec_verify(batch, verify_input, logits)

        verify_result, bookkeeping_result = _policy().finalize_spec_verify(
            padding,
            (
                torch.arange(PADDED_TOKENS, dtype=torch.int32),
                torch.arange(BUCKET_BS, dtype=torch.int32),
                torch.zeros((BUCKET_BS, DRAFT_TOKEN_NUM), dtype=torch.int32),
            ),
            (
                torch.arange(BUCKET_BS, dtype=torch.int32),
                torch.arange(BUCKET_BS, dtype=torch.int32),
            ),
        )
        predict, accept_lens, accept_index = verify_result
        new_seq_lens, bonus_tokens = bookkeeping_result

        self.assertEqual(predict.shape, (LOGICAL_TOKENS,))
        self.assertEqual(accept_lens.shape, (BS,))
        self.assertEqual(accept_index.shape, (BS, DRAFT_TOKEN_NUM))
        self.assertEqual(new_seq_lens.shape, (BS,))
        self.assertEqual(bonus_tokens.shape, (BS,))

    def test_skips_when_the_forward_did_not_defer(self):
        batch = _make_batch(temperature=0.7)
        verify_input = _make_verify_input(deferred=False)
        logits = torch.zeros((LOGICAL_TOKENS, VOCAB_SIZE))

        padding = _policy().prepare_spec_verify(batch, verify_input, logits)

        self.assertIsNone(padding)
        self.assertEqual(batch.seq_lens.shape, (BS,))

    def test_rejects_logits_that_do_not_match_the_recorded_padding(self):
        batch = _make_batch(temperature=0.7)
        verify_input = _make_verify_input()
        logits = torch.zeros((PADDED_TOKENS - 1, VOCAB_SIZE))

        with self.assertRaisesRegex(ValueError, "do not match the recorded padding"):
            _policy().prepare_spec_verify(batch, verify_input, logits)


class TestSpecVerifyPaddingIsInert(CustomTestCase):
    """Whatever lands in the padded rows must not reach the real requests.

    Both runs use the same shapes, so they consume the random stream
    identically; any difference in the results is padding leaking.
    """

    def _run(self, *, temperature, pad_seed):
        batch = _make_batch(temperature)
        verify_input = _make_verify_input()
        torch.manual_seed(pad_seed)
        logits = torch.cat(
            [_accepting_logits(), torch.randn((LOGICAL_TOKENS, VOCAB_SIZE)) * 10]
        )
        padding = _policy().prepare_spec_verify(batch, verify_input, logits)
        self.assertIsNotNone(padding)
        # Scribble different candidate ids into the padded tail as well.
        verify_input.draft_token[LOGICAL_TOKENS:] = torch.randint(
            0, VOCAB_SIZE, (LOGICAL_TOKENS,), dtype=verify_input.draft_token.dtype
        )

        logits_output = SimpleNamespace(next_token_logits=logits)
        torch.manual_seed(0)
        with _serving_stubs():
            predict, accept_lens, accept_index = eagle_sample(
                verify_input, batch, logits_output, None
            )
        verify_result, _ = _policy().finalize_spec_verify(
            padding,
            (predict, accept_lens, accept_index),
            (None, None),
        )
        return verify_result

    def _assert_inert(self, temperature, expected_accept_lens):
        first = self._run(temperature=temperature, pad_seed=1)
        second = self._run(temperature=temperature, pad_seed=99)

        predict, accept_lens, accept_index = first
        self.assertEqual(predict.shape, (LOGICAL_TOKENS,))
        self.assertEqual(accept_index.shape, (BS, DRAFT_TOKEN_NUM))
        # The crafted logits must exercise accept, reject and bonus selection.
        self.assertEqual(accept_lens.tolist(), expected_accept_lens)
        for a, b in zip(first, second):
            self.assertEqual(a.tolist(), b.tolist())

    def test_greedy_verify_ignores_the_padded_rows(self):
        # accept_lens counts accepted drafts plus the bonus token.
        self._assert_inert(temperature=0.0, expected_accept_lens=[2, DRAFT_TOKEN_NUM])

    def test_sampled_verify_ignores_the_padded_rows(self):
        self._assert_inert(temperature=0.7, expected_accept_lens=[2, DRAFT_TOKEN_NUM])

    def _run_unpadded(self, temperature):
        """The same verify with no padding at all, as the reference answer."""
        batch = _make_batch(temperature)
        verify_input = _make_verify_input(deferred=False)
        logits_output = SimpleNamespace(next_token_logits=_accepting_logits())
        self.assertIsNone(
            _policy().prepare_spec_verify(
                batch, verify_input, logits_output.next_token_logits
            )
        )
        torch.manual_seed(0)
        with _serving_stubs():
            return eagle_sample(verify_input, batch, logits_output, None)

    def test_greedy_padded_matches_the_unpadded_answer(self):
        """Greedy verify draws no random numbers, so padding may not change a
        single output bit."""
        padded = self._run(temperature=0.0, pad_seed=1)
        reference = self._run_unpadded(0.0)

        for got, want in zip(padded, reference):
            self.assertEqual(got.tolist(), want.tolist())

    def test_sampled_padded_accepts_the_same_draft_tokens(self):
        """Padding shifts which random numbers the bonus draw consumes, so the
        bonus token may differ -- but which drafts were accepted, and the tokens
        at those positions, may not."""
        predict, accept_lens, accept_index = self._run(temperature=0.7, pad_seed=1)
        ref_predict, ref_lens, ref_index = self._run_unpadded(0.7)

        self.assertEqual(accept_lens.tolist(), ref_lens.tolist())
        self.assertEqual(accept_index.tolist(), ref_index.tolist())
        for req, length in enumerate(accept_lens.tolist()):
            # The last accepted slot is the bonus/corrected token; the ones
            # before it are the drafts verify kept.
            for slot in accept_index[req, : length - 1].tolist():
                self.assertEqual(predict[slot].item(), ref_predict[slot].item())

    def test_padded_requests_keep_the_sampling_math_finite(self):
        """The padded rows run the same probability math as the real ones, so
        their parameters must keep it NaN-free -- otherwise the async NaN probe
        fires on a batch that is perfectly healthy."""
        probed = []

        def _probe(tensor, msg=""):
            probed.append((msg, bool(torch.isnan(tensor).any())))

        with patch("sglang.srt.utils.async_probe.maybe_detect_nan", _probe):
            self._run(temperature=0.7, pad_seed=1)

        self.assertTrue(probed)
        self.assertEqual([msg for msg, has_nan in probed if has_nan], [])

    def test_penalties_modify_only_the_logical_logits_prefix(self):
        batch = _make_batch(temperature=0.0)
        batch.sampling_info.acc_additive_penalties = torch.ones((BS, VOCAB_SIZE))
        batch.sampling_info.acc_scaling_penalties = torch.full((BS, VOCAB_SIZE), 2.0)
        batch.sampling_info.logit_bias = torch.ones((BS, VOCAB_SIZE))
        # Target-forward padding records this state before the policy decides
        # to keep its outputs physical through sampling.
        verify_input = _make_verify_input()
        policy = _policy()
        self.assertTrue(
            policy.should_defer_verify_output_unpadding(batch, verify_input)
        )
        logits = torch.zeros((PADDED_TOKENS, VOCAB_SIZE))
        padding = policy.prepare_spec_verify(batch, verify_input, logits)

        with _serving_stubs():
            result = eagle_sample(
                verify_input,
                batch,
                SimpleNamespace(next_token_logits=logits),
                None,
            )

        # additive 1 -> scale by 2 -> bias 1
        self.assertTrue(torch.all(logits[:LOGICAL_TOKENS] == 1.5))
        self.assertTrue(torch.all(logits[LOGICAL_TOKENS:] == 0))
        finalized, _ = policy.finalize_spec_verify(padding, result, (None, None))
        self.assertEqual(finalized[1].shape, (BS,))


if __name__ == "__main__":
    unittest.main()
