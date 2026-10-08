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
from sglang.srt.environ import envs
from sglang.srt.hardware_backend.tpu import sampler as tpu_sampler_module
from sglang.srt.hardware_backend.tpu.compilation.bucketer import Bucketer
from sglang.srt.hardware_backend.tpu.compilation.compiled_forward_shim import (
    CompiledForwardShim,
)
from sglang.srt.hardware_backend.tpu.compilation.decoder_compile_policy import (
    DecoderTpuCompilePolicy,
    Qwen3VLDecoderTpuCompilePolicy,
    _materialize_out,
    _unpad_decoder_outputs,
)
from sglang.srt.hardware_backend.tpu.compilation.pending_call import PendingCall
from sglang.srt.hardware_backend.tpu.compilation.spec_decode_compile_policy_dflash import (
    DFlashSpecDecoderTpuCompilePolicy,
    _kv_appender_specialization_key,
)
from sglang.srt.hardware_backend.tpu.compilation.spec_decode_compile_policy_eagle3 import (
    Eagle3SpecDecodeCompilePolicyMixin,
    SpecDecoderTpuCompilePolicy,
)
from sglang.srt.hardware_backend.tpu.sampler import select_logits_output_mode
from sglang.srt.hardware_backend.tpu.top_k import (
    PallasDistributedTopKConfig,
    TpuVocabShard,
)
from sglang.srt.layers.logits_processor import LogitsMetadata, LogitsProcessorOutput
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardMode,
    LogitsOutputMode,
    PPProxyTensors,
)
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.observability.compile_timing import CompileTimingRecorder
from sglang.srt.runtime_context import get_context, get_disagg, get_parallel, get_spec
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.speculative.dflash_info import DFlashVerifyInput
from sglang.srt.speculative.eagle_info import (
    EagleDraftExtendInput,
    EagleDraftInput,
    EagleVerifyInput,
)
from sglang.srt.speculative.eagle_utils import eagle_sample
from sglang.srt.speculative.eagle_worker_common import run_eagle_verify
from sglang.srt.speculative.spec_info import SpecInputType, SpeculativeAlgorithm
from sglang.test.test_utils import CustomTestCase


class _FakeAttnBackend:
    """Records ``init_forward_metadata`` calls so tests can assert shape."""

    def __init__(self):
        self.calls: list[tuple[int, int]] = []  # (num_tokens, bs)

    def init_forward_metadata(self, forward_batch):
        self.calls.append(
            (forward_batch.input_ids.shape[0], forward_batch.seq_lens.shape[0])
        )


# Config the compile policies read from the namespace bags; it reaches production
# only by being published (_PolicyTestCase below).
_BASE_CONFIG = {
    "page_size": 1,
    "dp_size": 1,
    "disaggregation_mode": "null",
}

# topk=1 EAGLE3 chain: the verify width is speculative_num_steps + 1.
_SPEC_NUM_STEPS = 3
_EAGLE3_CONFIG = {
    **_BASE_CONFIG,
    "speculative_algorithm": "EAGLE3",
    "speculative_num_steps": _SPEC_NUM_STEPS,
    "speculative_num_draft_tokens": _SPEC_NUM_STEPS + 1,
    "speculative_eagle_topk": 1,
}


_DFLASH_CONFIG = {
    **_BASE_CONFIG,
    "speculative_algorithm": "DFLASH",
    "speculative_num_steps": 0,
    "speculative_num_draft_tokens": 4,
    "speculative_draft_model_path": None,
    "speculative_draft_model_revision": None,
    "torch_compile_max_bs": 4,
}


class _PolicyTestCase(CustomTestCase):
    """Publishes the config the TPU compile policies read from the bags; each
    publish is undone in its own cleanup, so tests cannot leak config."""

    #: Merged underneath whatever a ``publish_config`` call passes.
    config_defaults = _BASE_CONFIG

    def setUp(self):
        super().setUp()
        self.publish_config()

    def publish_config(self, **fields):
        """Publish ``config_defaults`` overlaid with ``fields`` for this test."""
        override = get_context().override_server_args(
            **{**self.config_defaults, **fields}
        )
        override.install()
        self.addCleanup(override.restore)
        return override


class _SpecPolicyTestCase(_PolicyTestCase):
    """Adds the EAGLE3 spec config the spec policies read at construction."""

    config_defaults = _EAGLE3_CONFIG


def _model_runner(device="cpu"):
    """Minimal ModelRunner stand-in for unit tests.

    prepare_forward_batch is included because _run_one calls it before
    init_forward_metadata (mirrors the runtime chain installed by
    DecoderTpuCompilePolicy.install()). get_input_embeddings sizes the EAGLE3
    draft's mm_embeds warmup variant."""
    embed_tokens = torch.nn.Embedding(8, 4)
    return SimpleNamespace(
        device=device,
        model=SimpleNamespace(
            forward=MagicMock(),
            prepare_forward_batch=MagicMock(),
            get_input_embeddings=lambda: embed_tokens,
        ),
        attn_backend=_FakeAttnBackend(),
        req_to_token_pool=SimpleNamespace(),
        token_to_kv_pool=SimpleNamespace(),
        token_to_kv_pool_allocator=SimpleNamespace(
            free_pages=torch.arange(4, dtype=torch.int32)
        ),
        model_config=SimpleNamespace(model_is_mrope=False),
        is_draft_worker=False,
        spec_algorithm=SpeculativeAlgorithm.NONE,
        pp_group=SimpleNamespace(is_first_rank=True, is_last_rank=True),
        prepare_dummy_forward_batch=lambda forward_batch: forward_batch,
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


def _enable_distributed_top_k(model_runner):
    shard_width = 202_048 // 16
    shard = TpuVocabShard(
        width=shard_width,
        global_vocab_size=202_048,
        org_vocab_size=202_048,
        org_vocab_start=0,
        org_vocab_count=shard_width,
        org_vocab_padded_count=shard_width,
        added_vocab_start=202_048,
        added_vocab_count=0,
    )
    logits_processor = SimpleNamespace(
        use_tpu_distributed_top_k=True,
        do_tensor_parallel_all_gather=True,
        do_tensor_parallel_all_gather_dp_attn=False,
    )
    lm_head = SimpleNamespace(
        enable_tp=True,
        num_embeddings=202_048,
        org_vocab_size=202_048,
        tp_size=16,
        num_embeddings_per_partition=shard_width,
        shard_indices=object(),
    )
    config = PallasDistributedTopKConfig(
        shard=shard,
        world_size=16,
        model_rank=0,
        candidate_width=64,
        output_dtype=torch.bfloat16,
    )
    tpu_sampler_module._DISTRIBUTED_TOP_K_STATES[id(logits_processor)] = (
        tpu_sampler_module._DistributedTopKState(
            logits_processor=logits_processor,
            lm_head=lm_head,
            config=config,
            fallback_reason=None,
        )
    )
    model_runner.model.logits_processor = logits_processor
    model_runner.model.lm_head = lm_head
    return model_runner


def _make_distributed_top_k_policy():
    model_runner = _enable_distributed_top_k(_model_runner())
    return _make_policy(model_runner, token_buckets=(64,), bs_buckets=(64,))


def _select_logits_output_mode(policy, forward_batch):
    forward_batch.logits_output_mode = select_logits_output_mode(
        forward_batch, policy.model_runner.model
    )


def _target_and_padding(policy):
    return policy.owner, policy


def _run_cases(policy, mode):
    with get_parallel().override(
        tp_size=2,
        tp_rank=1,
        attn_tp_size=2,
        attn_tp_rank=1,
        moe_tp_size=2,
        moe_tp_rank=1,
    ):
        if mode == ForwardMode.EXTEND:
            policy._precompile_extend()
        else:
            policy._precompile_decode()


class TestDirectCompileExtraModelState(CustomTestCase):
    def test_custom_policy_forwards_kv_pool_without_replacing_runner_pool(self):
        compile_kv_pool = object()

        class CustomPolicy(DecoderTpuCompilePolicy):
            def get_direct_compile_kv_pool(self):
                return compile_kv_pool

        model_runner = _model_runner()
        runner_kv_pool = model_runner.token_to_kv_pool
        policy = CustomPolicy(model_runner=model_runner, owner=model_runner.model)
        compile_module_name = (
            "sglang.srt.hardware_backend.tpu.compilation.direct_compile"
        )
        compile_module = ModuleType(compile_module_name)
        compile_module.direct_compile = MagicMock(return_value=MagicMock())

        with patch.dict(sys.modules, {compile_module_name: compile_module}):
            policy._direct_compile(model_runner.model.forward)

        compile_module.direct_compile.assert_called_once()
        self.assertIs(
            compile_module.direct_compile.call_args.kwargs["kv_pool"], compile_kv_pool
        )
        self.assertIs(model_runner.token_to_kv_pool, runner_kv_pool)

    def test_call_and_async_warmup_trace_with_the_same_rank_tensors(self):
        """The async warmup traces the forward too, so it must pass the rank
        tensors the call path passes; without them every precompile fails."""
        model_runner = _model_runner()
        policy = DecoderTpuCompilePolicy(
            model_runner=model_runner, owner=model_runner.model
        )
        compiled = MagicMock()
        compile_module_name = (
            "sglang.srt.hardware_backend.tpu.compilation.direct_compile"
        )
        compile_module = ModuleType(compile_module_name)
        compile_module.direct_compile = MagicMock(return_value=compiled)
        rank_tensors = (torch.tensor(0), torch.tensor(1))
        policy_module = (
            "sglang.srt.hardware_backend.tpu.compilation.decoder_compile_policy"
        )

        with (
            patch.dict(sys.modules, {compile_module_name: compile_module}),
            patch(
                f"{policy_module}.get_forward_context",
                return_value=SimpleNamespace(
                    attn_backend=SimpleNamespace(forward_metadata=None)
                ),
            ),
            patch(
                f"{policy_module}.get_attention_rank_tensors",
                return_value=rank_tensors,
            ),
        ):
            call, warmup_async = policy._direct_compile(model_runner.model.forward)
            call("input_ids")
            warmup_async("input_ids")

        for recorded in (compiled.call_args, compiled.warmup_async.call_args):
            self.assertIs(recorded.kwargs["attention_rank_tensors"], rank_tensors)

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
            device="cpu",
            is_draft_worker=False,
            pp_group=SimpleNamespace(is_first_rank=True, is_last_rank=True),
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
            call, warmup_async = policy._direct_compile(model.forward)

        trace_forward = compile_module.direct_compile.call_args.args[0]
        compiled = compile_module.direct_compile.return_value
        compiled.side_effect = trace_forward
        compiled.warmup_async.side_effect = trace_forward
        backend = SimpleNamespace(forward_metadata=object())
        inputs = (torch.ones(2), None, None)
        with forward_context(ForwardContext(attn_backend=backend)):
            for invoke in (call, warmup_async):
                with self.subTest(invoke=invoke.__name__):
                    self.assertIs(invoke(*inputs), model.forward.return_value)
        self.assertEqual(model.forward.call_count, 2)

        compile_module.direct_compile.assert_called_once()
        kwargs = compile_module.direct_compile.call_args.kwargs
        self.assertIs(kwargs["model"], model)
        self.assertEqual(
            kwargs["model_extra_tensor_attrs"],
            CustomPolicy.model_extra_tensor_attrs,
        )
        self.assertIs(kwargs["kv_pool"], kv_pool)
        self.assertEqual(
            kwargs["specialization_key"], policy._direct_compile_specialization_key
        )
        self.assertIn(
            "forward_batch.extend_input_logprob_token_ids_gpu",
            kwargs["ignored_argnames"],
        )


class TestEagle3DirectCompile(_SpecPolicyTestCase):
    def test_fused_draft_trace_temporarily_restores_eager_regions(self):
        eager_backbone = MagicMock(name="eager_backbone")
        eager_model = MagicMock(name="eager_model")
        eager_logits = MagicMock(name="eager_logits")
        compiled_backbone = MagicMock(name="compiled_backbone")
        compiled_model = MagicMock(name="compiled_model")
        compiled_logits = MagicMock(name="compiled_logits")
        backbone = SimpleNamespace(forward=eager_backbone)
        logits_processor = SimpleNamespace(forward=eager_logits)
        model = SimpleNamespace(
            model=backbone,
            logits_processor=logits_processor,
            forward=eager_model,
        )
        kv_pool = object()
        model_runner = _spec_model_runner(is_draft_worker=True)
        model_runner.model = model
        model_runner.token_to_kv_pool = kv_pool
        with _target_model_config():
            policy = SpecDecoderTpuCompilePolicy(
                model_runner=model_runner,
                owner=model,
                tracer="direct",
            )
        with patch.object(
            DecoderTpuCompilePolicy, "install", return_value=True
        ) as base_install:
            self.assertTrue(policy.install())
        base_install.assert_called_once_with()

        backbone.forward = compiled_backbone
        model.forward = compiled_model
        logits_processor.forward = compiled_logits

        backend = SimpleNamespace(forward_metadata=object())
        captured = {}

        def fake_direct_compile(fn, **kwargs):
            captured.update(fn=fn, kwargs=kwargs)
            return fn

        def fused_steps(forward_batch, value):
            self.assertIs(model.forward, eager_model)
            self.assertIs(backbone.forward, eager_backbone)
            self.assertIs(logits_processor.forward, eager_logits)
            self.assertIs(backend.forward_metadata, captured["trace_metadata"])
            return (value + 1,)

        compile_module_name = (
            "sglang.srt.hardware_backend.tpu.compilation.direct_compile"
        )
        compile_module = ModuleType(compile_module_name)
        compile_module.direct_compile = fake_direct_compile

        with (
            patch.dict(sys.modules, {compile_module_name: compile_module}),
            patch(
                "sglang.srt.hardware_backend.tpu.compilation."
                "spec_decode_compile_policy_eagle3.get_forward_context",
                return_value=SimpleNamespace(attn_backend=backend),
            ),
        ):
            compiled = policy._direct_compile_fused_draft_steps(fused_steps)
            captured["trace_metadata"] = backend.forward_metadata
            output = compiled(SimpleNamespace(), torch.tensor(1))

        self.assertEqual(output[0].item(), 2)
        self.assertIs(model.forward, compiled_model)
        self.assertIs(backbone.forward, compiled_backbone)
        self.assertIs(logits_processor.forward, compiled_logits)
        self.assertIs(captured["kwargs"]["model"], model)
        self.assertIs(captured["kwargs"]["kv_pool"], kv_pool)


class TestLogitsOutputMode(_PolicyTestCase):
    def test_compiled_logits_ignore_logical_mode_after_tpu_dp_padding(self):
        policy = _make_distributed_top_k_policy()
        forward_batch = policy._make_dummy_forward_batch(
            num_tokens=64, bs=64, mode=ForwardMode.EXTEND
        )

        def consume_mode(value, batch):
            metadata = LogitsMetadata.from_forward_batch(batch)
            return value + int(metadata.forward_mode)

        with patch("sglang.srt.layers.logits_processor._is_tpu", True):
            torch._dynamo.reset()
            self.addCleanup(torch._dynamo.reset)
            compiled = torch.compile(consume_mode, backend="eager", fullgraph=True)
            value = torch.zeros(1)
            compiled(value, forward_batch)

            forward_batch._original_forward_mode = ForwardMode.IDLE
            with torch._dynamo.config.patch(error_on_recompile=True):
                result = compiled(value, forward_batch)

        self.assertEqual(result.item(), int(ForwardMode.EXTEND))

    def test_specialization_key_uses_selected_logits_mode(self):
        policy = _make_policy()
        forward_batch = SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            batch_size=64,
            logits_output_mode=LogitsOutputMode.DISTRIBUTED_TOP_K,
        )
        candidate_key = policy._direct_compile_specialization_key(
            None, torch.arange(64), None, forward_batch
        )
        forward_batch.logits_output_mode = LogitsOutputMode.FULL_VOCAB
        fallback_key = policy._direct_compile_specialization_key(
            None, torch.arange(64), None, forward_batch
        )

        self.assertIs(candidate_key[-1], LogitsOutputMode.DISTRIBUTED_TOP_K)
        self.assertIs(fallback_key[-1], LogitsOutputMode.FULL_VOCAB)
        self.assertEqual(fallback_key[:-1], candidate_key[:-1])

    def test_non_logits_boundary_ignores_logits_output_mode(self):
        policy = _make_policy()
        policy.owner = object()
        forward_batch = SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            batch_size=64,
            logits_output_mode=LogitsOutputMode.DISTRIBUTED_TOP_K,
        )
        candidate_key = policy._direct_compile_specialization_key(
            None, torch.arange(64), None, forward_batch
        )
        forward_batch.logits_output_mode = LogitsOutputMode.FULL_VOCAB

        self.assertIsNone(candidate_key[-1])
        self.assertEqual(
            candidate_key,
            policy._direct_compile_specialization_key(
                None, torch.arange(64), None, forward_batch
            ),
        )

    def test_mode_selection_uses_the_padded_physical_batch(self):
        policy = _make_distributed_top_k_policy()
        forward_batch = policy._make_dummy_forward_batch(
            num_tokens=1, bs=1, mode=ForwardMode.DECODE
        )

        self.assertIs(forward_batch.logits_output_mode, LogitsOutputMode.FULL_VOCAB)
        policy.prepare_forward_batch(forward_batch)
        _select_logits_output_mode(policy, forward_batch)

        self.assertEqual(forward_batch._tpu_real_b, 1)
        self.assertEqual(forward_batch.batch_size, 64)
        self.assertEqual(forward_batch.sampling_info.uniform_top_k, 50)
        self.assertEqual(forward_batch.sampling_info.top_ks[0].item(), 50)
        self.assertIs(
            forward_batch.logits_output_mode,
            LogitsOutputMode.DISTRIBUTED_TOP_K,
        )

    def test_compiled_logits_do_not_read_eager_logprob_metadata(self):
        policy = _make_distributed_top_k_policy()
        forward_batch = policy._make_dummy_forward_batch(
            num_tokens=64, bs=64, mode=ForwardMode.EXTEND
        )
        forward_batch.return_logprob = True
        forward_batch.top_logprobs_nums = [5] * 64
        forward_batch.token_ids_logprobs = [[3, 7]] * 64
        forward_batch.extend_input_logprob_token_ids_gpu = torch.tensor([3, 7])
        forward_batch.extend_logprob_start_lens_cpu = list(
            forward_batch.extend_seq_lens_cpu
        )

        policy.prepare_forward_batch(forward_batch)
        _select_logits_output_mode(policy, forward_batch)
        logits_processor = policy.model_runner.model.logits_processor
        logits_processor.compute_input_logprobs = False

        def consume_mode(value, batch):
            metadata = LogitsMetadata.from_forward_batch(
                batch,
                compute_input_logprobs=logits_processor.compute_input_logprobs,
            )
            offset = (
                64
                if batch.logits_output_mode is LogitsOutputMode.DISTRIBUTED_TOP_K
                else 0
            )
            return value + offset + int(metadata.extend_return_logprob)

        torch._dynamo.reset()
        self.addCleanup(torch._dynamo.reset)
        compiled = torch.compile(consume_mode, backend="eager", fullgraph=True)
        value = torch.zeros(1)
        for mode in LogitsOutputMode:
            forward_batch.logits_output_mode = mode
            compiled(value, forward_batch)

        forward_batch.top_logprobs_nums = [10] * 64
        forward_batch.token_ids_logprobs = [[11, 13]] * 64
        forward_batch.extend_input_logprob_token_ids_gpu = torch.tensor([11, 13])
        with torch._dynamo.config.patch(error_on_recompile=True):
            for mode in LogitsOutputMode:
                forward_batch.logits_output_mode = mode
                compiled(value, forward_batch)

        logits_metadata = LogitsMetadata.from_forward_batch(
            forward_batch, compute_input_logprobs=False
        )

        self.assertFalse(logits_metadata.extend_return_logprob)
        self.assertIsNone(logits_metadata.top_logprobs_nums)
        self.assertIsNone(logits_metadata.token_ids_logprobs)
        self.assertIsNone(logits_metadata.extend_input_logprob_token_ids_gpu)
        self.assertTrue(forward_batch.return_logprob)
        self.assertEqual(forward_batch.top_logprobs_nums, [10] * 64)
        self.assertEqual(forward_batch.token_ids_logprobs, [[11, 13]] * 64)
        self.assertEqual(
            forward_batch.extend_input_logprob_token_ids_gpu.tolist(), [11, 13]
        )


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


class TestInstallExtendCap(_PolicyTestCase):
    def _install_policy(self, *, prefill_max_requests, dp_attention):
        model = SimpleNamespace(
            forward=MagicMock(),
            prepare_forward_batch=MagicMock(),
        )
        # install() reads the compile grid and prefill cap from the published config.
        self.publish_config(
            chunked_prefill_size=None,
            max_prefill_tokens=2048,
            torch_compile_max_bs=64,
            prefill_max_requests=prefill_max_requests,
        )
        model_runner = SimpleNamespace(
            model=model,
            forward=MagicMock(),
            is_draft_worker=False,
            pp_group=SimpleNamespace(is_first_rank=True, is_last_rank=True),
            model_config=SimpleNamespace(context_len=4096),
            server_args=SimpleNamespace(
                prefill_max_requests=prefill_max_requests,
                unsafe_save_torch_compile_bundle=None,
                unsafe_load_torch_compile_bundle=None,
            ),
        )
        t_bucketer = Bucketer(name="Token", buckets=[64, 2048])
        b_bucketer = Bucketer(name="Bs", buckets=[16, 32, 64])
        policy = DecoderTpuCompilePolicy(model_runner=model_runner, owner=model)
        policy.backend = "eager"

        with (
            patch(
                "sglang.srt.hardware_backend.tpu.compilation.decoder_compile_policy."
                "build_bucketers",
                return_value=(t_bucketer, b_bucketer),
            ),
            patch(
                "sglang.srt.hardware_backend.tpu.compilation.decoder_compile_policy."
                "is_dp_attention_enabled",
                return_value=dp_attention,
            ),
            patch(
                "sglang.srt.hardware_backend.tpu.compilation.decoder_compile_policy."
                "torch.compile",
                return_value=MagicMock(),
            ),
            patch.object(policy, "_install_dp_idle_as_decode"),
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

    def test_context_length_does_not_cap_aggregate_chunked_prefill(self):
        model = SimpleNamespace(
            forward=MagicMock(),
            prepare_forward_batch=MagicMock(),
        )
        model_runner = SimpleNamespace(
            model=model,
            forward=MagicMock(),
            is_draft_worker=False,
            pp_group=SimpleNamespace(is_first_rank=True, is_last_rank=True),
            # The context limit applies to each request, while the flattened
            # token dimension covers the aggregate prefill batch.
            model_config=SimpleNamespace(context_len=4096),
            server_args=SimpleNamespace(
                prefill_max_requests=16,
                tpu_compile_tokens=[1024, 4096, 8192],
                tpu_compile_bs=[16],
                unsafe_save_torch_compile_bundle=None,
                unsafe_load_torch_compile_bundle=None,
            ),
        )
        # The 8192 ceiling derives from chunked_prefill_size via the bags;
        # the explicit bucket lists stay on the instance.
        self.publish_config(
            chunked_prefill_size=8192,
            max_prefill_tokens=16384,
            torch_compile_max_bs=16,
        )
        policy = DecoderTpuCompilePolicy(model_runner=model_runner, owner=model)
        policy.backend = "eager"

        with (
            patch(
                "sglang.srt.hardware_backend.tpu.compilation.decoder_compile_policy."
                "is_dp_attention_enabled",
                return_value=False,
            ),
            patch(
                "sglang.srt.hardware_backend.tpu.compilation.decoder_compile_policy."
                "torch.compile",
                return_value=MagicMock(),
            ),
        ):
            self.assertTrue(policy.install())

        self.assertEqual(policy.t_bucketer.buckets, [1024, 4096, 8192])
        self.assertEqual(policy.t_bucketer.next_bucket(8192), 8192)

    def test_target_model_keeps_outer_compile_boundary(self):
        original_outer = MagicMock(name="outer_forward")
        original_backbone = MagicMock(name="backbone_forward")
        model = SimpleNamespace(
            forward=original_outer,
            model=SimpleNamespace(forward=original_backbone),
            logits_processor=SimpleNamespace(compute_input_logprobs=True),
            prepare_forward_batch=MagicMock(),
        )
        model_runner = SimpleNamespace(
            model=model,
            forward=MagicMock(),
            is_draft_worker=False,
            pp_group=SimpleNamespace(is_first_rank=True, is_last_rank=True),
            model_config=SimpleNamespace(context_len=4096),
            server_args=SimpleNamespace(
                prefill_max_requests=16,
                unsafe_save_torch_compile_bundle=None,
                unsafe_load_torch_compile_bundle=None,
            ),
        )
        # install() reads the compile grid from the bags.
        self.publish_config(
            chunked_prefill_size=None,
            max_prefill_tokens=2048,
            torch_compile_max_bs=64,
        )
        policy = DecoderTpuCompilePolicy(model_runner=model_runner, owner=model)
        policy.backend = "eager"
        compiled_outer = MagicMock(name="compiled_outer")

        with (
            patch(
                "sglang.srt.hardware_backend.tpu.compilation.decoder_compile_policy."
                "build_bucketers",
                return_value=(
                    Bucketer(name="Token", buckets=[64, 2048]),
                    Bucketer(name="Bs", buckets=[16, 32, 64]),
                ),
            ),
            patch(
                "sglang.srt.hardware_backend.tpu.compilation.decoder_compile_policy."
                "is_dp_attention_enabled",
                return_value=False,
            ),
            patch(
                "sglang.srt.hardware_backend.tpu.compilation.decoder_compile_policy."
                "torch.compile",
                return_value=compiled_outer,
            ) as compile_mock,
        ):
            self.assertTrue(policy.install())

        self.assertIs(model.model.forward, original_backbone)
        self.assertFalse(model.logits_processor.compute_input_logprobs)
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
            pp_group=SimpleNamespace(is_first_rank=True, is_last_rank=True),
            model_config=SimpleNamespace(context_len=4096),
            server_args=SimpleNamespace(
                prefill_max_requests=16,
                unsafe_save_torch_compile_bundle=None,
                unsafe_load_torch_compile_bundle=None,
            ),
        )
        # install() reads the compile grid from the bags.
        self.publish_config(
            chunked_prefill_size=None,
            max_prefill_tokens=2048,
            torch_compile_max_bs=64,
        )
        policy = DecoderTpuCompilePolicy(model_runner=model_runner, owner=model)
        policy.backend = "eager"
        compiled_backbone = MagicMock(name="compiled_backbone")
        compiled_logits = MagicMock(name="compiled_logits")

        with (
            patch(
                "sglang.srt.hardware_backend.tpu.compilation.decoder_compile_policy."
                "build_bucketers",
                return_value=(
                    Bucketer(name="Token", buckets=[64, 2048]),
                    Bucketer(name="Bs", buckets=[16, 32, 64]),
                ),
            ),
            patch(
                "sglang.srt.hardware_backend.tpu.compilation.decoder_compile_policy."
                "is_dp_attention_enabled",
                return_value=False,
            ),
            patch(
                "sglang.srt.hardware_backend.tpu.compilation.decoder_compile_policy."
                "torch.compile",
                side_effect=[compiled_backbone, compiled_logits],
            ) as compile_mock,
        ):
            self.assertTrue(policy.install())

        self.assertEqual(compile_mock.call_count, 2)
        self.assertIs(model.model.forward, compiled_backbone)
        self.assertIs(logits_processor.forward, compiled_logits)
        self.assertIs(policy.compiled_forward_shim.fn, original_outer)

    def test_draft_compiles_direct_draft_model_without_submodule_or_logits_processor(
        self,
    ):
        original_forward = MagicMock(name="direct_draft_forward")
        model = SimpleNamespace(
            forward=original_forward,
            prepare_forward_batch=MagicMock(),
        )
        model_runner = SimpleNamespace(
            model=model,
            forward=MagicMock(),
            is_draft_worker=True,
            pp_group=SimpleNamespace(is_first_rank=True, is_last_rank=True),
            model_config=SimpleNamespace(context_len=4096),
            server_args=SimpleNamespace(
                prefill_max_requests=16,
                unsafe_save_torch_compile_bundle=None,
                unsafe_load_torch_compile_bundle=None,
            ),
        )
        # install() reads the compile grid from the bags.
        self.publish_config(
            chunked_prefill_size=None,
            max_prefill_tokens=2048,
            torch_compile_max_bs=64,
        )
        policy = DecoderTpuCompilePolicy(model_runner=model_runner, owner=model)
        policy.backend = "eager"
        compiled_model = MagicMock(name="compiled_model")

        with (
            patch(
                "sglang.srt.hardware_backend.tpu.compilation.decoder_compile_policy."
                "build_bucketers",
                return_value=(
                    Bucketer(name="Token", buckets=[64, 2048]),
                    Bucketer(name="Bs", buckets=[16, 32, 64]),
                ),
            ),
            patch(
                "sglang.srt.hardware_backend.tpu.compilation.decoder_compile_policy."
                "is_dp_attention_enabled",
                return_value=False,
            ),
            patch(
                "sglang.srt.hardware_backend.tpu.compilation.decoder_compile_policy."
                "torch.compile",
                return_value=compiled_model,
            ) as compile_mock,
        ):
            self.assertTrue(policy.install())

        self.assertEqual(compile_mock.call_count, 1)
        self.assertIs(model.forward, policy.compiled_forward_shim)
        self.assertIs(policy.compiled_forward_shim.fn, compiled_model)


class TestQwen3VLDecoderPolicy(_PolicyTestCase):
    def test_prefill_gathers_from_padded_hidden_with_real_metadata(self):
        policy = Qwen3VLDecoderTpuCompilePolicy()
        for capture in (CaptureHiddenMode.NULL, CaptureHiddenMode.LAST):
            with self.subTest(capture=capture):
                batch = self._forward_batch()
                batch.forward_mode = ForwardMode.EXTEND
                batch.capture_hidden_mode = capture
                batch.return_logprob = False
                batch.extend_seq_lens = torch.tensor([2, 1, 0, 0])
                batch.extend_seq_lens_cpu = [2, 1, 0, 0]
                hidden = torch.arange(20, dtype=torch.float32).reshape(5, 4)
                aux = [hidden + 100]
                result, result_aux = policy.finalize_forward_batch((hidden, aux), batch)
                self.assertIs(result, hidden)
                self.assertIs(result_aux[0], aux[0])
                last_index = torch.cumsum(batch.extend_seq_lens, dim=0) - 1
                torch.testing.assert_close(result[last_index], hidden[[1, 2]])
                torch.testing.assert_close(result_aux[0][last_index], aux[0][[1, 2]])

    def test_empty_prefill_request_preserves_negative_index_semantics(self):
        policy = Qwen3VLDecoderTpuCompilePolicy()
        batch = self._forward_batch()
        batch.forward_mode = ForwardMode.EXTEND
        batch.capture_hidden_mode = CaptureHiddenMode.LAST
        batch.extend_seq_lens_cpu = [0, 3]
        hidden = torch.arange(20, dtype=torch.float32).reshape(5, 4)
        result = policy.finalize_forward_batch(hidden, batch)
        torch.testing.assert_close(result[-1], hidden[2])

    def test_dp_prefill_keeps_existing_output_shape(self):
        policy = Qwen3VLDecoderTpuCompilePolicy()
        batch = self._forward_batch()
        batch.forward_mode = ForwardMode.EXTEND
        batch.capture_hidden_mode = CaptureHiddenMode.LAST
        batch.extend_seq_lens_cpu = [2, 1]
        hidden = torch.arange(20, dtype=torch.float32).reshape(5, 4)
        with patch(_DECODER_POLICY + "is_dp_attention_enabled", return_value=True):
            result = policy.finalize_forward_batch(hidden, batch)
        torch.testing.assert_close(result, hidden[:3])

    def test_full_capture_and_decode_keep_real_token_shapes(self):
        policy = Qwen3VLDecoderTpuCompilePolicy()
        for mode, capture in (
            (ForwardMode.EXTEND, CaptureHiddenMode.FULL),
            (ForwardMode.TARGET_VERIFY, CaptureHiddenMode.LAST),
            (ForwardMode.DECODE, CaptureHiddenMode.NULL),
            (ForwardMode.DRAFT_EXTEND_V2, CaptureHiddenMode.LAST),
        ):
            with self.subTest(mode=mode, capture=capture):
                batch = self._forward_batch()
                batch.forward_mode = mode
                batch.capture_hidden_mode = capture
                hidden = torch.arange(20, dtype=torch.float32).reshape(5, 4)
                result = policy.finalize_forward_batch(hidden, batch)
                torch.testing.assert_close(result, hidden[:3])

    def test_prompt_logprob_opt_in_trims_hidden_through_shim(self):
        """FULL capture must guard finalization while return_logprob is pinned false."""
        for enabled in (False, True):
            for deferred in (False, True):
                with self.subTest(enabled=enabled, deferred=deferred):
                    batch = self._forward_batch()
                    batch.forward_mode = ForwardMode.EXTEND
                    batch.input_ids = torch.arange(5)
                    batch.extend_seq_lens_cpu = [2, 1]
                    # Request only output logprobs, so this raw-hidden boundary
                    # needs no prompt gather. The opt-in still pins every
                    # extend to FULL, independently of the request's logprobs.
                    batch.extend_logprob_start_lens_cpu = [2, 1]
                    batch.return_logprob = True
                    batch.top_logprobs_nums = None
                    batch.token_ids_logprobs = None
                    batch.extend_input_logprob_token_ids_gpu = None
                    batch.mm_input_embeds = None
                    batch.input_embeds = None
                    batch.spec_info = None
                    batch.is_prefill_only = False
                    hidden = torch.arange(20, dtype=torch.float32).reshape(5, 4)

                    def compiled(input_ids, positions, forward_batch):
                        self.assertFalse(forward_batch.return_logprob)
                        return PendingCall(lambda: hidden) if deferred else hidden

                    policy = Qwen3VLDecoderTpuCompilePolicy(
                        model_runner=SimpleNamespace(
                            model=SimpleNamespace(),
                            token_to_kv_pool=SimpleNamespace(
                                layer_transfer_counter=None
                            ),
                            pp_group=SimpleNamespace(is_first_rank=True),
                        )
                    )
                    with envs.SGLANG_TPU_ENABLE_PROMPT_LOGPROB.override(enabled):
                        shim = CompiledForwardShim(compiled, policy=policy)
                    result = shim(batch.input_ids, None, batch)
                    if deferred:
                        result = result.wait()

                    if enabled:
                        torch.testing.assert_close(result, hidden[:3])
                    else:
                        self.assertIs(result, hidden)
                    self.assertIs(batch.capture_hidden_mode, CaptureHiddenMode.NULL)
                    self.assertTrue(batch.return_logprob)
                    self.assertEqual(batch.extend_logprob_start_lens_cpu, [2, 1])

    def _forward_batch(self):
        return SimpleNamespace(
            _tpu_real_q=3,
            _tpu_real_b=2,
            batch_size=4,
            seq_lens=torch.zeros(4, dtype=torch.int32),
            forward_mode=ForwardMode.DECODE,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            extend_seq_lens_cpu=None,
        )

    def test_slices_raw_hidden_states_before_logits_processing(self):
        policy = Qwen3VLDecoderTpuCompilePolicy()
        out = torch.arange(20, dtype=torch.float32).reshape(5, 4)

        result = policy.finalize_forward_batch(out, self._forward_batch())

        self.assertEqual(result.shape, (3, 4))

    def test_returns_the_same_tensor_when_no_trim_is_needed(self):
        # A no-op slice still produces a strided view, which torch_tpu
        # materializes as a same-shape identity copy in its own program between
        # decode steps. Identity, not just equality, is what proves no view was
        # created -- assertEqual on the shapes would pass either way.
        policy = Qwen3VLDecoderTpuCompilePolicy()
        forward_batch = self._forward_batch()
        forward_batch._tpu_real_q = 5
        out = torch.arange(20, dtype=torch.float32).reshape(5, 4)

        result = policy.finalize_forward_batch(out, forward_batch)

        self.assertIs(result, out)

    def test_returns_the_same_tensors_in_the_aux_hidden_state_branch(self):
        # The aux branch slices once per captured layer, so a no-op slice here
        # multiplies the identity copies.
        policy = Qwen3VLDecoderTpuCompilePolicy()
        forward_batch = self._forward_batch()
        forward_batch._tpu_real_q = 5
        hidden_states = torch.arange(20, dtype=torch.float32).reshape(5, 4)
        aux = [torch.ones(5, 4), torch.zeros(5, 4)]

        result_hidden, result_aux = policy.finalize_forward_batch(
            (hidden_states, aux), forward_batch
        )

        self.assertIs(result_hidden, hidden_states)
        self.assertIs(result_aux[0], aux[0])
        self.assertIs(result_aux[1], aux[1])

    def test_still_trims_when_the_batch_was_padded(self):
        # The skip must be conditional on the shape, not unconditional.
        policy = Qwen3VLDecoderTpuCompilePolicy()
        forward_batch = self._forward_batch()  # _tpu_real_q=3
        hidden_states = torch.arange(20, dtype=torch.float32).reshape(5, 4)
        aux = [torch.ones(5, 4)]

        result_hidden, result_aux = policy.finalize_forward_batch(
            (hidden_states, aux), forward_batch
        )

        self.assertEqual(result_hidden.shape, (3, 4))
        self.assertEqual(result_aux[0].shape, (3, 4))

    @staticmethod
    def _policy_for(*, multimodal: bool):
        """Policy whose owner model does / does not expose the VL image path."""
        model = SimpleNamespace()
        if multimodal:
            model.get_image_feature = lambda *args, **kwargs: None
        return Qwen3VLDecoderTpuCompilePolicy(model_runner=SimpleNamespace(model=model))

    def test_rejects_dp_attention_for_a_wrapper_that_does_not_serve_it(self):
        # A wrapper whose image steps or logits run eager collectives hangs
        # under DP attention; fail at install as before.
        policy = self._policy_for(multimodal=True)

        with (
            patch(
                "sglang.srt.hardware_backend.tpu.compilation.decoder_compile_policy."
                "is_dp_attention_enabled",
                return_value=True,
            ),
            patch.object(
                DecoderTpuCompilePolicy, "install", return_value=True
            ) as base_install,
        ):
            with self.assertRaises(NotImplementedError):
                policy.install()

        base_install.assert_not_called()

    def test_allows_dp_attention_for_a_wrapper_that_serves_it(self):
        # Kimi opts in: one input_embeds program per image step on every rank
        # and a compiled logits processor.
        policy = self._policy_for(multimodal=True)
        policy.model_runner.model.serves_dp_attention = True

        with (
            patch(
                "sglang.srt.hardware_backend.tpu.compilation.decoder_compile_policy."
                "is_dp_attention_enabled",
                return_value=True,
            ),
            patch.object(
                DecoderTpuCompilePolicy, "install", return_value=True
            ) as base_install,
        ):
            self.assertTrue(policy.install())

        base_install.assert_called_once()

    def test_allows_dp_attention_at_text_only_install(self):
        # This policy is also attached to the text-only backbone, where none of
        # the VL machinery the DP restriction covers is reachable.
        policy = self._policy_for(multimodal=False)

        with (
            patch(
                "sglang.srt.hardware_backend.tpu.compilation.decoder_compile_policy."
                "is_dp_attention_enabled",
                return_value=True,
            ),
            patch.object(
                DecoderTpuCompilePolicy, "install", return_value=True
            ) as base_install,
        ):
            self.assertTrue(policy.install())

        base_install.assert_called_once()

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
            logits_output_mode=LogitsOutputMode.FULL_VOCAB,
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

    def test_text_ids_warmup_uses_the_positional_serving_convention(self):
        """A multimodal model still serves text-only requests.

        Those call the decoder positionally, and Dynamo guards on len(args), so
        a graph warmed through the input_embeds keyword convention can never
        satisfy them. Warming only the embeds convention left every text graph
        to be compiled lazily at serving time, where the first batch width that
        had not been seen yet tripped fail_on_recompile and killed the server.
        """

        model_runner = _model_runner()
        model_runner.model.get_image_feature = MagicMock()
        model_runner.model.get_input_embeddings = MagicMock(
            return_value=torch.nn.Embedding(2, 4)
        )
        forward_batch = SimpleNamespace(
            input_ids=torch.ones(8, dtype=torch.int64),
            positions=torch.arange(8),
            forward_mode=ForwardMode.DECODE,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            batch_size=8,
        )
        policy = Qwen3VLDecoderTpuCompilePolicy(
            model_runner=model_runner, owner=model_runner.model
        )
        policy._make_dummy_forward_batch = MagicMock(return_value=forward_batch)

        args, kwargs = policy._make_warmup_inputs(
            num_tokens=8,
            bs=8,
            mode=ForwardMode.DECODE,
            variant="text_ids",
        )

        self.assertEqual(kwargs, {})
        self.assertEqual(len(args), 3)
        self.assertIs(args[0], forward_batch.input_ids)
        self.assertIs(args[2], forward_batch)

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
            Eagle3SpecDecodeCompilePolicyMixin, Qwen3VLDecoderTpuCompilePolicy
        ):
            pass

        policy = object.__new__(_SpecQwenPolicy)
        forward_batch = self._forward_batch()
        forward_batch.forward_mode = ForwardMode.TARGET_VERIFY
        out = torch.arange(20, dtype=torch.float32).reshape(5, 4)

        result = policy.finalize_forward_batch(out, forward_batch)

        self.assertEqual(result.shape, (3, 4))
        self.assertEqual(forward_batch.batch_size, 2)


class TestMakeDummyForwardBatch(_PolicyTestCase):
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

    def test_warmup_carries_num_token_non_padded_when_serving_does(self):
        """Real EP batches carry ``num_token_non_padded`` (init_new plus
        ModelRunner.forward) and the compiled MoE path guards on its presence,
        so a warmup batch without it recompiles on the first real decode and
        dies under fail_on_recompile (two-host tp16/ep16 R1-0528 and V3.1).
        """
        policy = _make_policy()
        policy_module = (
            "sglang.srt.hardware_backend.tpu.compilation.decoder_compile_policy"
        )

        with (
            patch(f"{policy_module}.enable_num_token_non_padded", return_value=True),
            patch(
                "sglang.srt.layers.attention.dsa.utils.is_dsa_enable_prefill_cp",
                return_value=False,
            ),
            patch("sglang.srt.layers.cp.utils.is_mla_cp_enabled", return_value=False),
        ):
            fb = policy._make_dummy_forward_batch(
                num_tokens=8, bs=8, mode=ForwardMode.DECODE
            )
        self.assertEqual(fb.global_num_token_non_padded_cpu, 8)
        self.assertEqual(fb.global_num_token_non_padded.item(), 8)
        # Replicated (not attn-TP sharded) warmup keeps the full count.
        self.assertEqual(fb.num_token_non_padded.item(), 8)

        with patch(f"{policy_module}.enable_num_token_non_padded", return_value=False):
            fb = policy._make_dummy_forward_batch(
                num_tokens=8, bs=8, mode=ForwardMode.DECODE
            )
        self.assertIsNone(fb.global_num_token_non_padded)
        self.assertIsNone(fb.num_token_non_padded)

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


class TestDpIdleDecodeMaterializationScope(_SpecPolicyTestCase):
    def test_plain_idle_decode_matches_warmup_cache_location_dtype(self):
        for dtype in (torch.int32, torch.int64):
            with self.subTest(allocator_dtype=dtype):
                policy = _make_policy()
                policy.model_runner.token_to_kv_pool_allocator.free_pages = (
                    torch.arange(4, dtype=dtype)
                )
                seen = []
                policy.model_runner.forward = lambda batch: seen.append(
                    (batch.forward_mode, batch.out_cache_loc.dtype)
                )
                policy._install_dp_idle_as_decode()
                warmup = policy._make_dummy_forward_batch(
                    num_tokens=4, bs=4, mode=ForwardMode.DECODE
                )
                batch = SimpleNamespace(
                    forward_mode=ForwardMode.IDLE,
                    global_num_tokens_cpu=[0, 1],
                    # ScheduleBatch.prepare_for_idle uses int64 regardless of
                    # the allocator dtype used by active peers and warmup.
                    out_cache_loc=torch.empty(0, dtype=torch.int64),
                )

                with _dp_attention(True):
                    policy.model_runner.forward(batch)

                self.assertEqual(
                    seen, [(ForwardMode.DECODE, warmup.out_cache_loc.dtype)]
                )
                self.assertEqual(batch.out_cache_loc.shape, (0,))

    def test_prepare_materializes_zero_row_spec_batch(self):
        policy = _make_spec_policy()
        spec_info = EagleDraftInput(
            topk_p=torch.zeros((0, 1)),
            topk_index=torch.zeros((0, 1), dtype=torch.int64),
            hidden_states=torch.zeros((0, 8)),
        )
        forward_batch = SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            _original_forward_mode=ForwardMode.IDLE,
            _original_batch_size=0,
            seq_lens=torch.empty((0,), dtype=torch.int32),
            input_ids=torch.empty((0,), dtype=torch.int64),
            positions=torch.empty((0,), dtype=torch.int64),
            out_cache_loc=torch.empty((0,), dtype=torch.int32),
            req_pool_indices=torch.empty((0,), dtype=torch.int32),
            spec_info=spec_info,
        )

        with (
            _dp_attention(True),
            patch(_DECODER_POLICY + "_pad_forward_batch_inputs", return_value=(0, 0)),
        ):
            policy.prepare_forward_batch(forward_batch)

        self.assertEqual(forward_batch.seq_lens.tolist(), [1])
        self.assertEqual(forward_batch.input_ids.tolist(), [0])


class TestPrecompileLoops(_PolicyTestCase):
    """``precompile`` fires one forward per TPU precompile variant:
    EXTEND covers token buckets at the fixed batch cap, and DECODE varies bs
    (tokens == bs)."""

    def test_compile_timings_include_waits_without_counting_other_queued_work(self):
        for async_warmup in (False, True):
            with self.subTest(async_warmup=async_warmup):
                policy = _make_policy()
                policy.model_runner.draft_model_idx = None
                policy.compile_timing_recorder = CompileTimingRecorder()
                clock = [100.0]

                def run_one(**kwargs):
                    clock[0] += 2.0

                    def execute():
                        clock[0] += 3.0

                    if async_warmup:
                        return PendingCall(execute)
                    execute()
                    return None

                with (
                    patch.object(policy, "_run_one", side_effect=run_one),
                    patch(
                        "sglang.srt.hardware_backend.tpu.compilation."
                        "decoder_compile_policy.time.perf_counter",
                        side_effect=lambda: clock[0],
                    ),
                    get_parallel().override(
                        tp_size=2,
                        tp_rank=1,
                        attn_tp_size=2,
                        attn_tp_rank=1,
                        moe_tp_size=2,
                        moe_tp_rank=1,
                    ),
                ):
                    pending = policy._precompile_decode()
                    if async_warmup:
                        self.assertEqual(policy.compile_timing_recorder.snapshot(), [])
                        self.assertEqual(len(pending), 2)
                    clock[0] += 1000.0
                    for call in pending:
                        call.wait()

                steps = policy.compile_timing_recorder.snapshot()
                self.assertEqual([step["batch_size"] for step in steps], [1, 4])
                self.assertEqual(
                    [step["duration_seconds"] for step in steps], [5.0, 5.0]
                )

    def test_extend_iterates_token_buckets_at_fixed_batch_cap(self):
        mr = _model_runner()
        policy = _make_policy(mr, token_buckets=(8, 64, 256), bs_buckets=(1, 4))
        _run_cases(policy, ForwardMode.EXTEND)
        self.assertEqual(mr.model.forward.call_count, 3)
        self.assertEqual(
            mr.attn_backend.calls,
            [(8, 4), (64, 4), (256, 4)],
        )

    def test_extend_warms_null_capture_by_default(self):
        # The other half of the prompt-logprob opt-in. Serving them needs the
        # extend graph warmed with FULL capture, which makes every prefill
        # return all its rows (~8% slower) whether or not anyone asked. With
        # the flag off the warmed variant must stay exactly what it has always
        # been -- this goes red if the choice is ever hardcoded to FULL.
        policy = _make_policy(_model_runner(), token_buckets=(8,), bs_buckets=(1, 4))
        policy._run_one = MagicMock()

        with get_parallel().override(
            tp_size=2,
            tp_rank=1,
            attn_tp_size=2,
            attn_tp_rank=1,
            moe_tp_size=2,
            moe_tp_rank=1,
        ):
            policy._precompile_extend()

        modes = {
            call.kwargs["capture_hidden_mode"]
            for call in policy._run_one.call_args_list
        }
        self.assertEqual(modes, {CaptureHiddenMode.NULL})

    def test_extend_warms_full_capture_when_prompt_logprobs_enabled(self):
        policy = _make_policy(_model_runner(), token_buckets=(8,), bs_buckets=(1, 4))
        policy._run_one = MagicMock()

        with envs.SGLANG_TPU_ENABLE_PROMPT_LOGPROB.override(True):
            with get_parallel().override(
                tp_size=2,
                tp_rank=1,
                attn_tp_size=2,
                attn_tp_rank=1,
                moe_tp_size=2,
                moe_tp_rank=1,
            ):
                policy._precompile_extend()

        modes = {
            call.kwargs["capture_hidden_mode"]
            for call in policy._run_one.call_args_list
        }
        # One variant, not two: FULL replaces NULL rather than joining it.
        self.assertEqual(modes, {CaptureHiddenMode.FULL})

    def test_warmup_modes_follow_batch_size_cutoff(self):
        policy = _make_distributed_top_k_policy()

        with patch(
            "sglang.srt.hardware_backend.tpu.sampler.distributed_top_k_min_batch_size",
            return_value=64,
        ):
            self.assertEqual(
                policy._warmup_logits_output_modes(1),
                (LogitsOutputMode.FULL_VOCAB,),
            )
            self.assertEqual(
                policy._warmup_logits_output_modes(64),
                (
                    LogitsOutputMode.DISTRIBUTED_TOP_K,
                    LogitsOutputMode.FULL_VOCAB,
                ),
            )

    def test_mixed_chunk_adds_fixed_decode_max_column(self):
        mr = _model_runner()
        # _extend_compile_cases reads enable_mixed_chunk off the schedule bag.
        self.publish_config(enable_mixed_chunk=True)
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
        self.publish_config(enable_mixed_chunk=False)
        mr.model.get_image_feature = MagicMock()
        mr.model.get_input_embeddings = MagicMock(return_value=torch.nn.Embedding(2, 4))
        policy = Qwen3VLDecoderTpuCompilePolicy(
            model_runner=mr,
            t_bucketer=Bucketer(name="Token", buckets=[64, 256]),
            b_bucketer=Bucketer(name="Bs", buckets=[16, 32, 64]),
            owner=mr.model,
        )
        policy.extend_bs = 16
        policy._run_one = MagicMock()

        with get_parallel().override(
            tp_size=2,
            tp_rank=1,
            attn_tp_size=2,
            attn_tp_rank=1,
            moe_tp_size=2,
            moe_tp_rank=1,
        ):
            policy._precompile_extend()

        # 2 token buckets x 2 conventions: the input_embeds one a VL request
        # takes, and the positional one a text-only request takes.
        self.assertEqual(policy._run_one.call_count, 4)
        self.assertEqual(
            {call.kwargs["bs"] for call in policy._run_one.call_args_list},
            {16},
        )
        self.assertEqual(
            {call.kwargs["variant"] for call in policy._run_one.call_args_list},
            {"text_only", "text_ids"},
        )

    def test_qwen3_vl_decode_warms_both_call_conventions(self):
        """Text-only requests reach the decoder positionally, so every batch
        bucket needs a graph in that convention. Warming only the input_embeds
        one left them to compile lazily at serving time, where the first
        unseen batch width tripped fail_on_recompile and killed the server."""

        warmed = self._warmed_decode_variants(decode_embeds_in_language_model=False)
        self.assertEqual(
            warmed,
            {
                (64, "text_only"),
                (128, "text_only"),
                (64, "text_ids"),
                (128, "text_ids"),
            },
        )

    def test_qwen3_vl_decode_embedding_in_graph_warms_one_convention(self):
        """With the embedding in the language model, "text_only" already is the
        serving call (input_ids + mrope positions). "text_ids" would warm 1-D
        positions an mrope decode never passes: one unused graph per bucket."""
        warmed = self._warmed_decode_variants(decode_embeds_in_language_model=True)
        self.assertEqual(warmed, {(64, "text_only"), (128, "text_only")})

    def _warmed_decode_variants(self, *, decode_embeds_in_language_model):
        mr = _model_runner()
        mr.model.get_image_feature = MagicMock()
        mr.model.get_input_embeddings = MagicMock()
        mr.model.decode_embeds_in_language_model = decode_embeds_in_language_model
        policy = Qwen3VLDecoderTpuCompilePolicy(
            model_runner=mr,
            b_bucketer=Bucketer(name="Bs", buckets=[64, 128]),
            owner=mr.model,
        )
        policy._run_one = MagicMock()

        with get_parallel().override(
            tp_size=2,
            tp_rank=1,
            attn_tp_size=2,
            attn_tp_rank=1,
            moe_tp_size=2,
            moe_tp_rank=1,
        ):
            policy._precompile_decode()

        return {
            (call.kwargs["bs"], call.kwargs["variant"])
            for call in policy._run_one.call_args_list
        }

    def test_qwen3_vl_warmup_crosses_logits_modes(self):
        mr = _model_runner()
        mr.model.get_image_feature = MagicMock()
        mr.model.get_input_embeddings = MagicMock(return_value=torch.nn.Embedding(2, 4))
        mr.model.decode_embeds_in_language_model = False
        _enable_distributed_top_k(mr)
        policy = Qwen3VLDecoderTpuCompilePolicy(
            model_runner=mr,
            t_bucketer=Bucketer(name="Token", buckets=[64]),
            b_bucketer=Bucketer(name="Bs", buckets=[64]),
            owner=mr.model,
        )
        policy._run_one = MagicMock()

        with get_parallel().override(
            tp_size=2,
            tp_rank=1,
            attn_tp_size=2,
            attn_tp_rank=1,
            moe_tp_size=2,
            moe_tp_rank=1,
        ):
            policy._precompile_extend()
            policy._precompile_decode()

        # Both calling conventions (input_embeds and positional text ids)
        # warm every logits mode, for EXTEND and DECODE alike.
        self.assertEqual(
            [
                call.kwargs["logits_output_mode"]
                for call in policy._run_one.call_args_list
            ],
            [
                LogitsOutputMode.DISTRIBUTED_TOP_K,
                LogitsOutputMode.FULL_VOCAB,
            ]
            * 4,
        )

    def test_qwen3_vl_warms_mm_embed_placement_for_every_token_bucket(self):
        """A skipped bucket JITs its select on the first real prefill.

        The multimodal merge runs eagerly outside the compiled graph, so it is
        not covered by the graph warmup above; only this loop reaches it. A
        bucket missing here reappears as a mid-serving compile stall.
        """
        mr = _model_runner()
        mr.server_args = SimpleNamespace(enable_mixed_chunk=False)
        mr.model.get_image_feature = MagicMock()
        mr.model.get_input_embeddings = MagicMock(return_value=torch.nn.Embedding(2, 4))
        policy = Qwen3VLDecoderTpuCompilePolicy(
            model_runner=mr,
            t_bucketer=Bucketer(name="Token", buckets=[64, 256]),
            b_bucketer=Bucketer(name="Bs", buckets=[16]),
            owner=mr.model,
        )
        policy.extend_bs = 16
        policy._run_one = MagicMock()

        with (
            patch(
                "sglang.srt.hardware_backend.tpu.mm_embed_tpu.apply_host_mm_embeds"
            ) as warmed,
            get_parallel().override(
                tp_size=2,
                tp_rank=1,
                attn_tp_size=2,
                attn_tp_rank=1,
                moe_tp_size=2,
                moe_tp_rank=1,
            ),
        ):
            policy._precompile_extend()

        self.assertEqual(
            [call.kwargs["input_embeds"].shape for call in warmed.call_args_list],
            [torch.Size((64, 4)), torch.Size((256, 4))],
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
        with get_parallel().override(
            tp_size=2,
            tp_rank=1,
            attn_tp_size=2,
            attn_tp_rank=1,
            moe_tp_size=2,
            moe_tp_rank=1,
        ):
            policy.precompile()
        # EXTEND: (8,4), (64,4) = 2. DECODE: (1,1), (4,4) = 2.
        self.assertEqual(mr.model.forward.call_count, 4)


class TestRunOneCallsCompiledForward(_SpecPolicyTestCase):
    """``_run_one`` wires the synthesized batch through the same chain
    serving uses: ``prepare_forward_batch`` → ``init_forward_metadata``
    → compiled ``forward``."""

    def test_pipeline_profile_executes_outer_model_synchronously(self):
        mr = _model_runner()
        policy = _make_policy(mr)
        policy.tracer = "direct"
        policy.owner = SimpleNamespace(forward=MagicMock())
        with (
            patch.object(envs.SGLANG_DIRECT_COMPILE_ASYNC, "get", return_value=True),
            patch.object(policy, "_supports_async_warmup", return_value=True),
            patch(
                "sglang.srt.hardware_backend.tpu.pp_profile.set_profile_inputs",
                side_effect=lambda policy, batch, args, kwargs, tokens, prefix: (
                    args,
                    kwargs,
                ),
            ),
            patch(
                "sglang.srt.hardware_backend.tpu.compilation."
                "decoder_compile_policy.time.perf_counter",
                side_effect=[10.0, 10.25],
            ),
        ):
            elapsed_ms = policy._run_one(
                num_tokens=64, bs=4, mode=ForwardMode.EXTEND, profile_prefix=0
            )
        self.assertEqual(elapsed_ms, 250.0)
        mr.model.forward.assert_called_once()
        policy.owner.forward.assert_not_called()
        policy.owner.forward.warmup_async.assert_not_called()

    def test_run_one_calls_prepare_then_metadata_then_forward(self):
        mr = _model_runner()

        def mark_sequence_sharded(forward_batch):
            forward_batch.attn_tp_sequence_sharded = True
            return forward_batch

        mr.prepare_dummy_forward_batch = MagicMock(side_effect=mark_sequence_sharded)
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
        self.assertTrue(args[2].attn_tp_sequence_sharded)
        mr.prepare_dummy_forward_batch.assert_called_once_with(args[2])

    def test_run_one_uses_no_grad_not_inference_mode(self):
        mr = _model_runner()
        policy = _make_policy(mr)
        cm = MagicMock()
        with (
            patch(
                "sglang.srt.hardware_backend.tpu.compilation.decoder_compile_policy.torch.no_grad",
                return_value=cm,
            ) as no_grad,
            patch(
                "sglang.srt.hardware_backend.tpu.compilation."
                "decoder_compile_policy.torch.inference_mode",
                side_effect=AssertionError("inference_mode should not be used"),
            ),
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
        with (
            patch.dict(sys.modules, stubs),
            patch("torch_tpu._internal.sync.sync.synchronize") as sync,
        ):
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

        for result in (kv0, (kv0, {"aux": kv1})):
            with self.subTest(result_type=type(result).__name__):
                with (
                    patch.dict(sys.modules, stubs),
                    patch("torch_tpu._internal.sync.sync.synchronize") as sync,
                ):
                    _materialize_out(result, SimpleNamespace(), None)
                expected = [kv0] if isinstance(result, torch.Tensor) else [kv0, kv1]
                sync.assert_called_once_with(expected, wait=True)


def _spec_model_runner(device="cpu", is_draft_worker=False):
    """ModelRunner stand-in; the spec config comes from get_spec(), not from here.

    get_input_embeddings sizes the EAGLE3 draft's mm_embeds warmup variant."""
    embed_tokens = torch.nn.Embedding(8, 4)
    return SimpleNamespace(
        device=device,
        model=SimpleNamespace(
            forward=MagicMock(),
            prepare_forward_batch=MagicMock(),
            get_input_embeddings=lambda: embed_tokens,
        ),
        attn_backend=_FakeAttnBackend(),
        # _run_one seeds req_to_token rows; needs a real indexable tensor
        # (draft extend seeds one row per request in the bucket).
        req_to_token_pool=SimpleNamespace(
            size=16, req_to_token=torch.zeros(16, 4096, dtype=torch.int64)
        ),
        token_to_kv_pool=SimpleNamespace(),
        token_to_kv_pool_allocator=SimpleNamespace(
            free_pages=torch.arange(4, dtype=torch.int32)
        ),
        model_config=SimpleNamespace(
            hidden_size=16,
            spec_hidden_size=16,
            model_is_mrope=False,
            hf_config=SimpleNamespace(
                eagle_config={"use_aux_hidden_state": False},
            ),
        ),
        dtype=torch.bfloat16,
        is_draft_worker=is_draft_worker,
        pp_group=SimpleNamespace(is_first_rank=True, is_last_rank=True),
        spec_algorithm=SpeculativeAlgorithm.EAGLE3,
        prepare_dummy_forward_batch=lambda forward_batch: forward_batch,
    )


@contextlib.contextmanager
def _target_model_config(is_multimodal=False):
    """Stand in for the target ModelConfig a spec policy reads at construction.

    The published test config points at a dummy model path, so the real
    accessor would go looking for an HF config that does not exist."""
    with patch(
        _SPEC_POLICY + "process_model_config",
        return_value=SimpleNamespace(is_multimodal=is_multimodal),
    ):
        yield


def _make_spec_policy(
    model_runner=None,
    token_buckets=(8, 64),
    bs_buckets=(1, 4),
    target_is_multimodal=False,
    **mr_kwargs,
):
    if model_runner is None:
        model_runner = _spec_model_runner(**mr_kwargs)
    with _target_model_config(target_is_multimodal):
        policy = SpecDecoderTpuCompilePolicy(
            model_runner=model_runner,
            t_bucketer=Bucketer(name="Token", buckets=list(token_buckets)),
            b_bucketer=Bucketer(name="Bs", buckets=list(bs_buckets)),
            owner=model_runner.model,
        )
    # install() normally creates this; these tests construct the policy directly.
    policy.compiled_forward_shim = CompiledForwardShim(
        model_runner.model.forward, policy=policy
    )
    return policy


_SPEC_POLICY = (
    "sglang.srt.hardware_backend.tpu.compilation.spec_decode_compile_policy_eagle3."
)
_DECODER_POLICY = "sglang.srt.hardware_backend.tpu.compilation.decoder_compile_policy."


@contextlib.contextmanager
def _dp_attention(enabled: bool, *, attn_tp_size: int = 1):
    """Override DP attention and its parallel width for policy tests."""
    with (
        patch(_SPEC_POLICY + "is_dp_attention_enabled", return_value=enabled),
        patch(_DECODER_POLICY + "is_dp_attention_enabled", return_value=enabled),
        patch(
            "sglang.srt.layers.dp_attention.is_dp_attention_enabled",
            return_value=enabled,
        ),
        patch(
            _SPEC_POLICY + "get_parallel",
            return_value=SimpleNamespace(attn_tp_size=attn_tp_size),
        ),
    ):
        yield


def _dp_idle_batch(spec_info, *, peer_tokens, **overrides):
    """Build an empty DP-rank batch."""
    fields = dict(
        forward_mode=ForwardMode.IDLE,
        _original_forward_mode=None,
        spec_info=spec_info,
        global_num_tokens_cpu=[0, peer_tokens],
        batch_size=0,
        seq_lens_sum=0,
        input_ids=torch.empty((0,), dtype=torch.int64),
        positions=torch.empty((0,), dtype=torch.int64),
        seq_lens=torch.empty((0,), dtype=torch.int64),
        seq_lens_cpu=torch.empty((0,), dtype=torch.int64),
        out_cache_loc=torch.empty((0,), dtype=torch.int64),
        req_pool_indices=torch.empty((0,), dtype=torch.int64),
        req_pool_indices_cpu=torch.empty((0,), dtype=torch.int64),
        mrope_positions=None,
        mm_inputs=None,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


class TestSpecPrecompileLoops(_SpecPolicyTestCase):
    """Target spec precompile covers its serving forward families."""

    def test_draft_prefill_uses_configured_prefill_cap(self):
        mr = _spec_model_runner(is_draft_worker=True)
        policy = _make_spec_policy(
            mr,
            token_buckets=(1024, 4096),
            bs_buckets=(16, 32, 64),
        )
        policy.extend_bs = 16

        with get_parallel().override(
            tp_size=2,
            tp_rank=1,
            attn_tp_size=2,
            attn_tp_rank=1,
            moe_tp_size=2,
            moe_tp_rank=1,
        ):
            policy._precompile_draft_prefill_extend()

        self.assertEqual(mr.attn_backend.calls, [(1024, 16), (4096, 16)])

    def test_draft_prefill_adds_embeds_variant_only_for_a_multimodal_target(self):
        """Only a multimodal target hands the draft input embeds.

        A text-only target never takes that path, so compiling it would double
        the draft EXTEND warmup for every text-only deployment."""
        seen = []
        for target_is_multimodal in (False, True):
            mr = _spec_model_runner(is_draft_worker=True)
            policy = _make_spec_policy(
                mr,
                token_buckets=(1024,),
                bs_buckets=(16,),
                target_is_multimodal=target_is_multimodal,
            )
            policy.extend_bs = 16
            with get_parallel().override(
                tp_size=2,
                tp_rank=1,
                attn_tp_size=2,
                attn_tp_rank=1,
                moe_tp_size=2,
                moe_tp_rank=1,
            ):
                policy._precompile_draft_prefill_extend()
            seen.append(len(mr.attn_backend.calls))

        self.assertEqual(seen, [1, 2])

    def test_infers_eagle_aux_hidden_width_from_config(self):
        mr = _spec_model_runner(is_draft_worker=True)
        mr.model_config = SimpleNamespace(
            hidden_size=8_192,
            spec_hidden_size=8_192,
            hf_config=SimpleNamespace(
                target_hidden_size=8_192,
                eagle_config={
                    "use_aux_hidden_state": True,
                    "eagle_aux_hidden_state_layer_ids": [7, 15, 23],
                },
            ),
        )
        policy = _make_spec_policy(mr)

        self.assertEqual(policy.eagle_hidden_size, 24_576)

    def test_extend_full_uses_configured_prefill_cap(self):
        mr = _spec_model_runner()
        policy = _make_spec_policy(
            mr,
            token_buckets=(1024, 4096),
            bs_buckets=(16, 32, 64),
        )
        policy.extend_bs = 16

        with get_parallel().override(
            tp_size=2,
            tp_rank=1,
            attn_tp_size=2,
            attn_tp_rank=1,
            moe_tp_size=2,
            moe_tp_rank=1,
        ):
            policy._precompile_extend_full()

        self.assertEqual(
            mr.attn_backend.calls,
            [(1024, 16), (1024, 16), (4096, 16), (4096, 16)],
        )

    def test_extend_full_precompiles_both_spec_info_states(self):
        mr = _spec_model_runner()
        policy = _make_spec_policy(mr, token_buckets=(8,), bs_buckets=(4,))

        with get_parallel().override(
            tp_size=2,
            tp_rank=1,
            attn_tp_size=2,
            attn_tp_rank=1,
            moe_tp_size=2,
            moe_tp_rank=1,
        ):
            policy._precompile_extend_full()

        batches = [call.args[2] for call in mr.model.forward.call_args_list]
        self.assertEqual(len(batches), 2)
        self.assertIsNone(batches[0].spec_info)
        self.assertIsInstance(batches[1].spec_info, EagleDraftInput)
        self.assertEqual(
            [batch.capture_hidden_mode for batch in batches],
            [CaptureHiddenMode.FULL, CaptureHiddenMode.FULL],
        )
        self.assertEqual(
            [batch.spec_algorithm for batch in batches],
            [SpeculativeAlgorithm.EAGLE3, SpeculativeAlgorithm.EAGLE3],
        )

    def test_target_precompiles_all_spec_modes(self):
        mr = _spec_model_runner()  # draft_token_num = 4
        policy = _make_spec_policy(mr, token_buckets=(8, 64), bs_buckets=(1, 4))
        # precompile() spans two modules: the base EXTEND/DECODE warmup lives in
        # decoder_compile_policy (via super()), the spec TARGET_VERIFY/EXTEND(FULL)
        # in spec_decode_compile_policy. One topology override serves both.
        with get_parallel().override(
            tp_size=2,
            tp_rank=1,
            attn_tp_size=2,
            attn_tp_rank=1,
            moe_tp_size=2,
            moe_tp_rank=1,
        ):
            policy.precompile()
        # EXTEND (8,4),(64,4)=2  DECODE (1,1),(4,4)=2
        # TARGET_VERIFY buckets bs: (4,1),(16,4)=2. EXTEND(FULL) covers two
        # token buckets and both real serving spec-info states, so 4.
        self.assertEqual(mr.model.forward.call_count, 10)
        self.assertIn((16, 4), mr.attn_backend.calls)
        self.assertIn((4, 1), mr.attn_backend.calls)

    def test_draft_worker_skips_precompile(self):
        mr = _spec_model_runner(is_draft_worker=True)
        policy = _make_spec_policy(mr)
        with get_parallel().override(
            tp_size=2,
            tp_rank=1,
            attn_tp_size=2,
            attn_tp_rank=1,
            moe_tp_size=2,
            moe_tp_rank=1,
        ):
            policy.precompile()
        self.assertEqual(mr.model.forward.call_count, 0)
        self.assertEqual(mr.attn_backend.calls, [])

    def test_draft_precompiles_all_serving_forward_families(self):
        mr = _spec_model_runner(is_draft_worker=True)
        # The dummy draft-extend layout pages by get_schedule().page_size.
        self.publish_config(page_size=8)
        policy = _make_spec_policy(mr, token_buckets=(8, 64), bs_buckets=(1, 4))
        decode_backends = [_FakeAttnBackend(), _FakeAttnBackend()]
        draft_backend = SimpleNamespace(attn_backends=decode_backends)
        draft_extend_backend = _FakeAttnBackend()

        with get_parallel().override(
            tp_size=2,
            tp_rank=1,
            attn_tp_size=2,
            attn_tp_rank=1,
            moe_tp_size=2,
            moe_tp_rank=1,
        ):
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

    def test_draft_precompile_installs_the_checker_after_its_last_forward(self):
        """Draft warmup finishes before installing and baselining its checker."""
        mr = _spec_model_runner(is_draft_worker=True)
        self.publish_config(page_size=8)
        policy = _make_spec_policy(mr, token_buckets=(8, 64), bs_buckets=(1, 4))
        draft_backend = SimpleNamespace(attn_backends=[_FakeAttnBackend()] * 2)
        forwards_at_install = []

        def install(**kwargs):
            forwards_at_install.append(mr.model.forward.call_count)
            self.assertEqual(kwargs["policies"], (policy,))
            self.assertTrue(kwargs["model_label"].startswith("draft "))

        with (
            get_parallel().override(
                tp_size=2,
                tp_rank=1,
                attn_tp_size=2,
                attn_tp_rank=1,
                moe_tp_size=2,
                moe_tp_rank=1,
            ),
            patch(
                "sglang.srt.hardware_backend.tpu.compilation."
                "spec_decode_compile_policy_eagle3.install_recompilation_checker",
                side_effect=install,
            ),
        ):
            policy.precompile_draft(draft_backend, _FakeAttnBackend())

        self.assertEqual(forwards_at_install, [mr.model.forward.call_count])
        self.assertGreater(mr.model.forward.call_count, 0)

    def test_draft_precompile_skips_shapes_larger_than_local_request_pool(self):
        mr = _spec_model_runner(is_draft_worker=True)
        mr.req_to_token_pool.size = 4
        policy = _make_spec_policy(mr, bs_buckets=(1, 4, 8))
        policy._precompile_draft_prefill_extend = MagicMock()
        policy._precompile_draft_decode = MagicMock()
        draft_extend_backend = _FakeAttnBackend()

        with get_parallel().override(
            tp_size=2,
            tp_rank=1,
            attn_tp_size=2,
            attn_tp_rank=1,
            moe_tp_size=2,
            moe_tp_rank=1,
        ):
            policy.precompile_draft(SimpleNamespace(), draft_extend_backend)

        self.assertEqual(draft_extend_backend.calls, [(4, 1), (16, 4)])

    def test_non_eagle3_algorithm_raises(self):
        # The spec policy enforces the supported algorithm at construction (only
        # EAGLE3 today); an unsupported algorithm fails loudly rather than
        # silently degrading to a non-spec policy.
        mr = _spec_model_runner()
        # The algorithm is read from get_spec() as the policy is constructed.
        self.publish_config(speculative_algorithm="NGRAM")
        with self.assertRaisesRegex(ValueError, "only supports EAGLE3"):
            _make_spec_policy(mr)


class TestSpecDummyForwardBatch(_SpecPolicyTestCase):
    """Spec dummy batches carry spec_info and slot-1 KV seeding."""

    def test_target_verify_batch_shapes_and_capture_mode(self):
        policy = _make_spec_policy()  # draft_token_num = 4
        spec_info = policy.make_verify_spec_info()
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
            spec_info=policy.make_null_spec_info(),
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
        mr = _spec_model_runner(is_draft_worker=True)
        self.publish_config(page_size=8)
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
        mr = _spec_model_runner(is_draft_worker=True)
        self.publish_config(page_size=2)
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


class TestDFlashCompilePolicy(_PolicyTestCase):
    """Unit tests for DFlash compile policy KV append compilation, padding, and warmup."""

    config_defaults = _DFLASH_CONFIG

    def _make_dflash_policy(
        self,
        token_buckets=(16, 64),
        bs_buckets=(1, 4),
        hidden_size=16,
        target_hidden_size=32,
        draft_token_num=4,
        is_draft_worker=True,
        pp_group=SimpleNamespace(is_first_rank=True, is_last_rank=True),
        tracer="dynamo",
    ):
        model = SimpleNamespace(
            fc=SimpleNamespace(in_features=target_hidden_size),
            forward=MagicMock(),
            prepare_forward_batch=MagicMock(),
            project_target_hidden=lambda h: h[:, :hidden_size],
        )
        model_runner = SimpleNamespace(
            device="cpu",
            dtype=torch.float32,
            model=model,
            attn_backend=_FakeAttnBackend(),
            req_to_token_pool=SimpleNamespace(
                size=16, req_to_token=torch.zeros(16, 4096, dtype=torch.int64)
            ),
            token_to_kv_pool=SimpleNamespace(),
            token_to_kv_pool_allocator=SimpleNamespace(
                free_pages=torch.arange(4, dtype=torch.int32)
            ),
            model_config=SimpleNamespace(hidden_size=hidden_size, model_is_mrope=False),
            is_draft_worker=is_draft_worker,
            pp_group=SimpleNamespace(is_first_rank=True, is_last_rank=True),
            spec_algorithm=SpeculativeAlgorithm.DFLASH,
            prepare_dummy_forward_batch=lambda forward_batch: forward_batch,
        )
        # _infer_dflash_block_size reads get_spec() at construction; publish the block size.
        self.publish_config(speculative_num_draft_tokens=draft_token_num)
        t_bucketer = Bucketer(name="Token", buckets=list(token_buckets))
        b_bucketer = Bucketer(name="Bs", buckets=list(bs_buckets))
        policy = DFlashSpecDecoderTpuCompilePolicy(
            model_runner=model_runner,
            t_bucketer=t_bucketer,
            b_bucketer=b_bucketer,
            owner=model_runner.model,
            tracer=tracer,
        )
        # install() normally creates this; these tests construct the policy directly.
        policy.compiled_forward_shim = CompiledForwardShim(model.forward, policy=policy)
        return policy, model_runner

    def test_infer_dflash_block_size_from_server_args_and_fallback(self):
        policy, mr = self._make_dflash_policy(draft_token_num=8)
        self.assertEqual(policy.draft_token_num, 8)

        # Republish without block size or draft model: the fallback comes from the spec bag.
        self.publish_config(speculative_num_draft_tokens=None)
        new_policy = DFlashSpecDecoderTpuCompilePolicy(
            model_runner=mr,
            t_bucketer=policy.t_bucketer,
            b_bucketer=policy.b_bucketer,
            owner=mr.model,
        )
        self.assertEqual(new_policy.draft_token_num, 16)

    def test_infer_dflash_hidden_sizes(self):
        policy, mr = self._make_dflash_policy(
            hidden_size=64, target_hidden_size=128, is_draft_worker=True
        )
        self.assertEqual(policy.dflash_hidden_size, 64)
        self.assertEqual(policy.dflash_target_hidden_size, 128)

        # Non-draft worker returns 0 for dflash_hidden_size.
        non_draft_policy, _ = self._make_dflash_policy(is_draft_worker=False)
        self.assertEqual(non_draft_policy.dflash_hidden_size, 0)
        self.assertEqual(non_draft_policy._infer_dflash_hidden_size(), 0)

        # Fallback when model has no fc layer.
        mr.model = SimpleNamespace()
        self.assertEqual(policy._infer_dflash_target_hidden_size(), 64)

    def test_pad_append_target_hidden_tpu_prefill_extend(self):
        policy, _ = self._make_dflash_policy(token_buckets=(16, 64))

        # 10 tokens pad to next bucket 16; dtypes pass through untouched.
        target_hidden = torch.ones(10, 32)
        cache_loc = torch.arange(10, dtype=torch.int32)
        positions = torch.arange(10, dtype=torch.int32)
        padded = policy.pad_append_target_hidden_tpu(
            target_hidden=target_hidden,
            cache_loc=cache_loc,
            positions=positions,
        )
        self.assertIsNotNone(padded)
        p_target, p_loc, p_pos, p_loc2d, p_commit = padded
        self.assertEqual(p_target.shape, (16, 32))
        self.assertEqual(p_loc.shape, (16,))
        self.assertEqual(p_loc.dtype, torch.int32)
        self.assertEqual(p_pos.shape, (16,))
        self.assertEqual(p_pos.dtype, torch.int32)
        self.assertIsNone(p_loc2d)
        self.assertIsNone(p_commit)

        # Exact match (16 tokens) requires no padding.
        target_hidden_16 = torch.ones(16, 32)
        cache_loc_16 = torch.arange(16, dtype=torch.int64)
        positions_16 = torch.arange(16, dtype=torch.int64)
        padded_16 = policy.pad_append_target_hidden_tpu(
            target_hidden=target_hidden_16,
            cache_loc=cache_loc_16,
            positions=positions_16,
        )
        self.assertIsNotNone(padded_16)
        self.assertEqual(padded_16[0].shape, (16, 32))

        # 0 tokens returns None.
        empty = policy.pad_append_target_hidden_tpu(
            target_hidden=torch.empty(0, 32),
            cache_loc=torch.empty(0, dtype=torch.int64),
            positions=torch.empty(0, dtype=torch.int64),
        )
        self.assertIsNone(empty)

    def test_pad_append_target_hidden_tpu_verify_decode(self):
        policy, _ = self._make_dflash_policy(
            token_buckets=(16,), bs_buckets=(1, 4), draft_token_num=4
        )

        # bs=2 (8 tokens) pads to bs=4 (16 tokens).
        bs, block_size = 2, 4
        target_hidden = torch.ones(bs * block_size, 32)
        cache_loc = torch.arange(bs * block_size, dtype=torch.int64)
        positions = torch.arange(bs * block_size, dtype=torch.int64)
        cache_loc_2d = torch.arange(bs * block_size, dtype=torch.int64).view(
            bs, block_size
        )
        commit_lens = torch.tensor([3, 2], dtype=torch.int32)

        padded = policy.pad_append_target_hidden_tpu(
            target_hidden=target_hidden,
            cache_loc=cache_loc,
            positions=positions,
            cache_loc_2d=cache_loc_2d,
            commit_lens=commit_lens,
        )
        self.assertIsNotNone(padded)
        p_target, p_loc, p_pos, p_loc2d, p_commit = padded
        self.assertEqual(p_target.shape, (16, 32))
        self.assertEqual(p_loc.shape, (16,))
        self.assertEqual(p_pos.shape, (16,))
        self.assertEqual(p_loc2d.shape, (4, 4))
        self.assertEqual(p_commit.shape, (4,))

        # Empty batch (bs=0) returns None.
        empty = policy.pad_append_target_hidden_tpu(
            target_hidden=torch.empty(0, 32),
            cache_loc=torch.empty(0, dtype=torch.int64),
            positions=torch.empty(0, dtype=torch.int64),
            cache_loc_2d=torch.empty(0, block_size, dtype=torch.int64),
            commit_lens=torch.empty(0, dtype=torch.int32),
        )
        self.assertIsNone(empty)

    def test_pad_append_target_hidden_tpu_verify_validation_errors(self):
        policy, _ = self._make_dflash_policy(bs_buckets=(1, 4), draft_token_num=4)
        target_hidden = torch.ones(8, 32)
        cache_loc = torch.arange(8, dtype=torch.int64)
        positions = torch.arange(8, dtype=torch.int64)
        cache_loc_2d = torch.arange(8, dtype=torch.int64).view(2, 4)

        # Missing commit_lens raises ValueError.
        with self.assertRaisesRegex(ValueError, "requires commit_lens"):
            policy.pad_append_target_hidden_tpu(
                target_hidden=target_hidden,
                cache_loc=cache_loc,
                positions=positions,
                cache_loc_2d=cache_loc_2d,
                commit_lens=None,
            )

        # Batch size mismatch raises ValueError.
        mismatched_commit = torch.tensor([3, 2, 1], dtype=torch.int32)
        with self.assertRaisesRegex(ValueError, "batch size mismatch"):
            policy.pad_append_target_hidden_tpu(
                target_hidden=target_hidden,
                cache_loc=cache_loc,
                positions=positions,
                cache_loc_2d=cache_loc_2d,
                commit_lens=mismatched_commit,
            )

    def test_ensure_kv_appender_validation_and_compilation(self):
        policy, _ = self._make_dflash_policy()
        with self.assertRaisesRegex(ValueError, "cannot be None"):
            policy._ensure_kv_appender(None)

        worker = SimpleNamespace(
            append_target_hidden_to_draft_kv_by_loc_tpu=MagicMock(),
        )
        policy.backend = "eager"
        policy._ensure_kv_appender(worker)
        self.assertIsNotNone(policy._compiled_kv_appender)
        self.assertEqual(
            worker.append_target_hidden_to_draft_kv_by_loc_tpu,
            policy.append_target_hidden_to_draft_kv_by_loc_tpu,
        )

    def test_ensure_kv_appender_honors_direct_tracer(self):
        policy, mr = self._make_dflash_policy(tracer="direct")
        worker = SimpleNamespace(
            append_target_hidden_to_draft_kv_by_loc_tpu=MagicMock(),
        )
        compile_module_name = (
            "sglang.srt.hardware_backend.tpu.compilation.direct_compile"
        )
        compile_module = ModuleType(compile_module_name)
        compile_module.direct_compile = MagicMock(return_value=MagicMock())

        with patch.dict(sys.modules, {compile_module_name: compile_module}):
            policy._ensure_kv_appender(worker)

        compile_module.direct_compile.assert_called_once()
        kwargs = compile_module.direct_compile.call_args.kwargs
        self.assertIs(kwargs["model"], mr.model)
        self.assertIs(kwargs["kv_pool"], mr.token_to_kv_pool)
        self.assertIs(
            policy._compiled_kv_appender, compile_module.direct_compile.return_value
        )

    def test_kv_appender_specialization_key_separates_verify_and_extend(self):
        common = dict(
            target_hidden=torch.zeros(8, 32),
            cache_loc=torch.zeros(8, dtype=torch.int32),
            positions=torch.zeros(8, dtype=torch.int64),
        )
        extend_key = _kv_appender_specialization_key(
            **common, cache_loc_2d=None, commit_lens=None
        )
        verify_key = _kv_appender_specialization_key(
            **common,
            cache_loc_2d=torch.zeros(2, 4, dtype=torch.int32),
            commit_lens=torch.zeros(2, dtype=torch.int32),
        )
        self.assertNotEqual(extend_key, verify_key)
        self.assertEqual(
            extend_key,
            _kv_appender_specialization_key(
                target_hidden=torch.ones(8, 32),
                cache_loc=torch.ones(8, dtype=torch.int32),
                positions=torch.ones(8, dtype=torch.int64),
                cache_loc_2d=None,
                commit_lens=None,
            ),
        )
        self.assertNotEqual(
            extend_key,
            _kv_appender_specialization_key(
                **{**common, "cache_loc": torch.zeros(8, dtype=torch.int64)},
                cache_loc_2d=None,
                commit_lens=None,
            ),
        )

    def test_precompile_draft_replaces_worker_method_and_pads_extend(self):
        policy, mr = self._make_dflash_policy(
            token_buckets=(16, 64), bs_buckets=(1, 4), draft_token_num=4
        )
        mock_append_kernel = MagicMock()
        policy._compiled_kv_appender = mock_append_kernel
        mock_block_embedder = MagicMock()
        policy._compiled_block_embedder = mock_block_embedder
        worker = SimpleNamespace(
            draft_model=mr.model,
            draft_model_runner=mr,
            append_target_hidden_to_draft_kv_by_loc_tpu=MagicMock(),
            embed_draft_block_ids=MagicMock(),
        )

        with get_parallel().override(
            tp_size=2,
            tp_rank=1,
            attn_tp_size=2,
            attn_tp_rank=1,
            moe_tp_size=2,
            moe_tp_rank=1,
        ):
            policy.precompile_draft(
                draft_attn_backend=_FakeAttnBackend(), dflash_worker=worker
            )

        # Worker methods were replaced with the compiled seams.
        self.assertEqual(
            worker.append_target_hidden_to_draft_kv_by_loc_tpu,
            policy.append_target_hidden_to_draft_kv_by_loc_tpu,
        )
        self.assertEqual(worker.embed_draft_block_ids, mock_block_embedder)

        # Extend call with 10 tokens pads to token bucket 16.
        target_hidden = torch.ones(10, 32)
        cache_loc = torch.arange(10, dtype=torch.int64)
        positions = torch.arange(10, dtype=torch.int64)
        worker.append_target_hidden_to_draft_kv_by_loc_tpu(
            target_hidden=target_hidden,
            cache_loc=cache_loc,
            positions=positions,
        )
        # 2 verify warmup calls + 2 extend warmup calls + 1 runtime call = 5 calls.
        self.assertEqual(mock_append_kernel.call_count, 5)
        kwargs = mock_append_kernel.call_args.kwargs
        self.assertEqual(kwargs["target_hidden"].shape, (16, 32))
        self.assertEqual(kwargs["positions"].shape, (16,))
        self.assertEqual(kwargs["cache_loc"].shape, (16,))
        self.assertIsNone(kwargs["cache_loc_2d"])
        self.assertIsNone(kwargs["commit_lens"])

    def test_precompile_draft_requires_worker(self):
        policy, _ = self._make_dflash_policy()
        with (
            self.assertRaisesRegex(ValueError, "cannot be None"),
            get_parallel().override(
                tp_size=2,
                tp_rank=1,
                attn_tp_size=2,
                attn_tp_rank=1,
                moe_tp_size=2,
                moe_tp_rank=1,
            ),
        ):
            policy.precompile_draft(dflash_worker=None)

    def test_precompile_draft_skipped_when_not_draft_worker(self):
        policy, _ = self._make_dflash_policy(is_draft_worker=False)
        worker = SimpleNamespace()
        policy.precompile_draft(dflash_worker=worker)
        self.assertIsNone(policy._compiled_kv_appender)

    def test_verify_pads_by_batch_size(self):
        policy, mr = self._make_dflash_policy(
            token_buckets=(16,), bs_buckets=(1, 4), draft_token_num=4
        )
        mock_append_kernel = MagicMock()
        policy._compiled_kv_appender = mock_append_kernel
        worker = SimpleNamespace(
            draft_model=mr.model,
            draft_model_runner=mr,
            append_target_hidden_to_draft_kv_by_loc_tpu=MagicMock(),
        )
        policy._ensure_kv_appender(worker)

        # Verify call with bs=2 (8 tokens) pads to bs=4 (16 tokens).
        bs = 2
        block_size = 4
        target_hidden = torch.ones(bs * block_size, 32)
        cache_loc = torch.arange(bs * block_size, dtype=torch.int64)
        positions = torch.arange(bs * block_size, dtype=torch.int64)
        cache_loc_2d = torch.arange(bs * block_size, dtype=torch.int64).view(
            bs, block_size
        )
        commit_lens = torch.tensor([3, 2], dtype=torch.int32)

        worker.append_target_hidden_to_draft_kv_by_loc_tpu(
            target_hidden=target_hidden,
            cache_loc=cache_loc,
            positions=positions,
            cache_loc_2d=cache_loc_2d,
            commit_lens=commit_lens,
        )

        kwargs = mock_append_kernel.call_args.kwargs
        self.assertEqual(kwargs["target_hidden"].shape, (16, 32))
        self.assertEqual(kwargs["positions"].shape, (16,))
        self.assertEqual(kwargs["cache_loc"].shape, (16,))
        self.assertEqual(kwargs["cache_loc_2d"].shape, (4, 4))
        self.assertEqual(kwargs["commit_lens"].shape, (4,))

    def test_append_target_hidden_uncompiled_raises(self):
        policy, _ = self._make_dflash_policy()
        target_hidden = torch.ones(4, 32)
        cache_loc = torch.arange(4, dtype=torch.int64)
        positions = torch.arange(4, dtype=torch.int64)
        with self.assertRaisesRegex(RuntimeError, "is not compiled"):
            policy.append_target_hidden_to_draft_kv_by_loc_tpu(
                target_hidden=target_hidden,
                cache_loc=cache_loc,
                positions=positions,
            )

    def test_make_null_and_verify_spec_info(self):
        policy, _ = self._make_dflash_policy(draft_token_num=6)
        null_info = policy.make_null_spec_info()
        self.assertIsInstance(null_info, DFlashVerifyInput)
        self.assertEqual(null_info.capture_hidden_mode, CaptureHiddenMode.NULL)
        self.assertEqual(null_info.draft_token_num, 6)

        verify_info = policy.make_verify_spec_info()
        self.assertIsInstance(verify_info, DFlashVerifyInput)
        self.assertEqual(verify_info.capture_hidden_mode, CaptureHiddenMode.FULL)
        self.assertEqual(verify_info.draft_token_num, 6)

    def test_make_dummy_forward_batch_draft_worker(self):
        policy, _ = self._make_dflash_policy(
            hidden_size=32, draft_token_num=4, is_draft_worker=True
        )
        spec_info = policy.make_null_spec_info()
        fb = policy._make_dummy_forward_batch(
            num_tokens=16,
            bs=4,
            mode=ForwardMode.TARGET_VERIFY,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            spec_info=spec_info,
        )
        self.assertEqual(fb.input_embeds.shape, (16, 32))
        self.assertEqual(fb.extend_seq_lens.tolist(), [4] * 4)
        self.assertEqual(fb.extend_prefix_lens.tolist(), [0] * 4)
        self.assertEqual(fb.extend_num_tokens, 16)

    def test_draft_warmup_leaves_global_num_token_non_padded_unset(self):
        """The live DFlash draft batch sets the local ``num_token_non_padded``
        and never the global tensor. direct_compile pins each input's type at
        trace time, so a warmup batch carrying the global tensor makes the
        first live draft forward fail with a type change (gpt-oss-120b DFlash
        e2e).
        """
        policy, _ = self._make_dflash_policy(
            hidden_size=32, draft_token_num=4, is_draft_worker=True
        )
        policy_module = (
            "sglang.srt.hardware_backend.tpu.compilation.decoder_compile_policy"
        )
        with (
            patch(f"{policy_module}.enable_num_token_non_padded", return_value=True),
            patch(
                "sglang.srt.layers.attention.dsa.utils.is_dsa_enable_prefill_cp",
                return_value=False,
            ),
            patch("sglang.srt.layers.cp.utils.is_mla_cp_enabled", return_value=False),
        ):
            fb = policy._make_dummy_forward_batch(
                num_tokens=16,
                bs=4,
                mode=ForwardMode.TARGET_VERIFY,
                capture_hidden_mode=CaptureHiddenMode.NULL,
                spec_info=policy.make_null_spec_info(),
            )
        self.assertIsNone(fb.global_num_token_non_padded)
        self.assertIsNotNone(fb.num_token_non_padded)

    def test_block_embed_warmup_reused_under_serving_grad_mode(self):
        torch._dynamo.reset()
        self.addCleanup(torch._dynamo.reset)
        policy, _ = self._make_dflash_policy(bs_buckets=(1, 4, 64), draft_token_num=6)
        policy.backend = "eager"
        policy.dynamic = False
        embedding = torch.nn.Embedding(16, 8)

        def embed_draft_block_ids(*, block_ids):
            return embedding(block_ids).flatten(0, 1)

        worker = SimpleNamespace(
            embed_draft_block_ids=embed_draft_block_ids,
            ensure_embed_module=lambda: embedding,
        )
        with (
            torch.enable_grad(),
            get_parallel().override(
                tp_size=2,
                tp_rank=1,
                attn_tp_size=2,
                attn_tp_rank=1,
                moe_tp_size=2,
                moe_tp_rank=1,
            ),
        ):
            policy._precompile_embed_draft_block_ids(worker)
            self.assertTrue(torch.is_grad_enabled())

        # Startup runs with gradients enabled; the scheduler disables them.
        # Every warmed bucket must remain usable once recompiles are forbidden.
        with torch.no_grad(), torch.compiler.set_stance("fail_on_recompile"):
            for bs in (1, 4, 64):
                block_ids = torch.arange(bs * 6).reshape(bs, 6) % 16
                actual = worker.embed_draft_block_ids(block_ids=block_ids)
                torch.testing.assert_close(actual, embedding(block_ids).flatten(0, 1))

    def test_precompile_draft_pd_prefill_mode_skips_verify(self):
        policy, mr = self._make_dflash_policy(
            token_buckets=(16, 64), bs_buckets=(1, 4), draft_token_num=4
        )
        mock_append_kernel = MagicMock()
        policy._compiled_kv_appender = mock_append_kernel
        mock_block_embedder = MagicMock()
        policy._compiled_block_embedder = mock_block_embedder
        worker = SimpleNamespace(
            draft_model=mr.model,
            draft_model_runner=mr,
            append_target_hidden_to_draft_kv_by_loc_tpu=MagicMock(),
            embed_draft_block_ids=MagicMock(),
        )

        # precompile_draft branches on get_disagg().disaggregation_mode.
        with (
            get_parallel().override(
                tp_size=2,
                tp_rank=1,
                attn_tp_size=2,
                attn_tp_rank=1,
                moe_tp_size=2,
                moe_tp_rank=1,
            ),
            get_disagg().override(disaggregation_mode="prefill"),
        ):
            policy.precompile_draft(
                draft_attn_backend=_FakeAttnBackend(), dflash_worker=worker
            )

        # In prefill mode: draft model forward is skipped (0 calls), only extend KV append runs (2 token buckets).
        self.assertEqual(mr.model.forward.call_count, 0)
        self.assertEqual(mock_append_kernel.call_count, 2)
        # Block embed warmup is not mode-gated: one call per bs bucket.
        self.assertEqual(mock_block_embedder.call_count, 2)

    def test_precompile_draft_pd_decode_mode_skips_extend(self):
        policy, mr = self._make_dflash_policy(
            token_buckets=(16, 64), bs_buckets=(1, 4), draft_token_num=4
        )
        mock_append_kernel = MagicMock()
        policy._compiled_kv_appender = mock_append_kernel
        mock_block_embedder = MagicMock()
        policy._compiled_block_embedder = mock_block_embedder
        worker = SimpleNamespace(
            draft_model=mr.model,
            draft_model_runner=mr,
            append_target_hidden_to_draft_kv_by_loc_tpu=MagicMock(),
            embed_draft_block_ids=MagicMock(),
        )

        with (
            get_parallel().override(
                tp_size=2,
                tp_rank=1,
                attn_tp_size=2,
                attn_tp_rank=1,
                moe_tp_size=2,
                moe_tp_rank=1,
            ),
            get_disagg().override(disaggregation_mode="decode"),
        ):
            policy.precompile_draft(
                draft_attn_backend=_FakeAttnBackend(), dflash_worker=worker
            )

        # In decode mode: draft model forward runs (2 bs buckets), only verify KV append runs (2 bs buckets).
        self.assertEqual(mr.model.forward.call_count, 2)
        self.assertEqual(mock_append_kernel.call_count, 2)
        self.assertEqual(mock_block_embedder.call_count, 2)

    def test_dflash_build_disagg_draft_input(self):
        from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2

        batch = SimpleNamespace(
            seq_lens=torch.tensor([10, 20], dtype=torch.int32),
            enable_overlap=False,
            req_pool_indices=torch.tensor([0, 1], dtype=torch.int32),
        )
        last_tokens = torch.tensor([101, 102], dtype=torch.int64)
        future_map = MagicMock()

        draft_input = SpeculativeAlgorithm.DFLASH.build_disagg_draft_input(
            batch, last_tokens, future_map
        )
        self.assertIsInstance(draft_input, DFlashDraftInputV2)
        self.assertTrue(torch.equal(draft_input.bonus_tokens, last_tokens))
        self.assertTrue(torch.equal(draft_input.new_seq_lens, batch.seq_lens))

    def test_target_precompile_pd_modes(self):
        policy, mr = self._make_dflash_policy(
            token_buckets=(8, 64),
            bs_buckets=(1, 4),
            draft_token_num=4,
            is_draft_worker=False,
            pp_group=SimpleNamespace(is_first_rank=True, is_last_rank=True),
        )

        # precompile() reads get_disagg().disaggregation_mode per call; scope each half.
        with (
            get_parallel().override(
                tp_size=2,
                tp_rank=1,
                attn_tp_size=2,
                attn_tp_rank=1,
                moe_tp_size=2,
                moe_tp_rank=1,
            ),
            get_disagg().override(disaggregation_mode="prefill"),
        ):
            policy.precompile()
        # Prefill mode: EXTEND (2), EXTEND(FULL) (4) = 6 calls; DECODE and TARGET_VERIFY skipped.
        self.assertEqual(mr.model.forward.call_count, 6)

        policy_dec, mr_dec = self._make_dflash_policy(
            token_buckets=(8, 64),
            bs_buckets=(1, 4),
            draft_token_num=4,
            is_draft_worker=False,
            pp_group=SimpleNamespace(is_first_rank=True, is_last_rank=True),
        )

        with (
            get_parallel().override(
                tp_size=2,
                tp_rank=1,
                attn_tp_size=2,
                attn_tp_rank=1,
                moe_tp_size=2,
                moe_tp_rank=1,
            ),
            get_disagg().override(disaggregation_mode="decode"),
        ):
            policy_dec.precompile()
        # Decode mode: DECODE (2), TARGET_VERIFY (2) = 4 calls; EXTEND and EXTEND(FULL) skipped.
        self.assertEqual(mr_dec.model.forward.call_count, 4)


class TestSpecDraftDecodePadding(_SpecPolicyTestCase):
    def test_pads_mrope_and_cpu_seq_lens_from_three_to_four(self):
        policy = _make_spec_policy(bs_buckets=(1, 4))
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

    def test_dp_idle_draft_extend_normalizes_before_post_write_update(self):
        policy = _make_spec_policy(bs_buckets=(1, 4))
        spec_info = EagleDraftExtendInput(
            hidden_states=torch.zeros((0, 16)),
            num_tokens_per_req=4,
            num_tokens_for_logprob_per_req=4,
        )
        forward_batch = SimpleNamespace(
            forward_mode=ForwardMode.IDLE,
            _original_forward_mode=None,
            spec_info=spec_info,
            global_num_tokens_cpu=[0, 8],
            input_ids=torch.empty((0,), dtype=torch.int64),
            seq_lens=torch.empty((0,), dtype=torch.int64),
            req_pool_indices=torch.empty((0,), dtype=torch.int32),
            req_pool_indices_cpu=torch.empty((0,), dtype=torch.int32),
        )

        with _dp_attention(True):
            policy.normalize_spec_forward_batch(forward_batch)

        self.assertEqual(forward_batch.forward_mode, ForwardMode.DRAFT_EXTEND_V2)
        self.assertEqual(forward_batch._original_forward_mode, ForwardMode.IDLE)
        self.assertEqual(forward_batch.batch_size, 2)
        self.assertEqual(forward_batch.extend_num_tokens, 8)
        self.assertEqual(forward_batch.input_ids.shape, (8,))
        self.assertEqual(forward_batch.positions.tolist(), list(range(8)))
        self.assertEqual(spec_info.hidden_states.shape, (8, 16))
        self.assertIs(spec_info.input_ids, forward_batch.input_ids)
        self.assertEqual(forward_batch.seq_lens.tolist(), [0, 0])
        self.assertEqual(forward_batch.seq_lens_cpu.tolist(), [0, 0])
        self.assertEqual(forward_batch.extend_seq_lens.tolist(), [4, 4])
        self.assertEqual(forward_batch.extend_start_loc.tolist(), [0, 4])
        self.assertEqual(forward_batch.seq_lens_sum, 0)
        self.assertTrue(spec_info.keep_outputs_padded)
        self.assertEqual(forward_batch.req_pool_indices.tolist(), [0, 0])
        self.assertEqual(forward_batch.req_pool_indices_cpu.tolist(), [0, 0])

    def test_dp_idle_verify_relabels_as_target_verify(self):
        policy = _make_spec_policy(bs_buckets=(1, 4))
        seen = []
        policy.model_runner.forward = lambda forward_batch: seen.append(
            forward_batch.forward_mode
        )
        policy._install_dp_idle_as_decode()
        spec_info = SimpleNamespace(
            keep_outputs_padded=False,
            is_verify_input=lambda: True,
        )
        forward_batch = SimpleNamespace(
            forward_mode=ForwardMode.IDLE,
            _original_forward_mode=None,
            spec_info=spec_info,
            global_num_tokens_cpu=[0, 4],
        )

        with _dp_attention(True):
            policy.model_runner.forward(forward_batch)

        self.assertEqual(seen, [ForwardMode.TARGET_VERIFY])
        self.assertEqual(forward_batch._original_forward_mode, ForwardMode.IDLE)
        self.assertTrue(spec_info.keep_outputs_padded)

    def test_dp_idle_draft_prefill_relabels_as_extend(self):
        policy = _make_spec_policy(bs_buckets=(1, 4))
        seen = []
        policy.model_runner.forward = lambda forward_batch: seen.append(
            forward_batch.forward_mode
        )
        policy._install_dp_idle_as_decode()
        spec_info = SimpleNamespace(
            keep_outputs_padded=False,
            spec_input_type=SpecInputType.EAGLE_DRAFT_EXTEND,
            is_verify_input=lambda: False,
            num_tokens_per_req=1,
        )
        forward_batch = SimpleNamespace(
            forward_mode=ForwardMode.IDLE,
            _original_forward_mode=None,
            spec_info=spec_info,
            global_num_tokens_cpu=[0, 300],
            input_ids=torch.empty((0,), dtype=torch.int64),
            seq_lens=torch.empty((0,), dtype=torch.int64),
            # Match ScheduleBatch.prepare_for_idle, not the allocator's dtype.
            out_cache_loc=torch.empty((0,), dtype=torch.int64),
            req_pool_indices=torch.empty((0,), dtype=torch.int64),
            req_pool_indices_cpu=torch.empty((0,), dtype=torch.int64),
            seq_lens_cpu=None,
        )
        with _dp_attention(True):
            policy.model_runner.forward(forward_batch)

        self.assertEqual(seen, [ForwardMode.EXTEND])
        self.assertEqual(forward_batch.batch_size, 1)
        self.assertEqual(forward_batch.seq_lens.tolist(), [300])
        self.assertEqual(forward_batch.extend_seq_lens.tolist(), [300])
        self.assertEqual(forward_batch.out_cache_loc.dtype, torch.int32)

        for name in ("extend_seq_lens", "extend_prefix_lens", "extend_start_loc"):
            self.assertEqual(getattr(forward_batch, name).dtype, torch.int32, name)
        for name in ("seq_lens", "req_pool_indices", "req_pool_indices_cpu"):
            self.assertEqual(getattr(forward_batch, name).dtype, torch.int64, name)


class TestDpSyncNumTokens(_SpecPolicyTestCase):
    """DP-agreed speculative widths use compiled shapes."""

    def test_default_policy_uses_generic_collective_alignment(self):
        counts, logprob_counts = NoopCompilePolicy().get_dp_padded_token_counts(
            SimpleNamespace(),
            SimpleNamespace(),
            [5, 0],
            [5, 0],
            attn_tp_size=4,
            cp_align_size=6,
        )

        self.assertEqual(counts, [12, 0])
        self.assertEqual(logprob_counts, [5, 0])

    def test_tpu_policy_aligns_logprob_counts_for_dpa_gather(self):
        policy = _make_policy()
        counts, logprob_counts = policy.get_dp_padded_token_counts(
            SimpleNamespace(),
            policy.model_runner,
            [5, 0],
            [5, 0],
            attn_tp_size=4,
            cp_align_size=1,
        )

        self.assertEqual(counts, [8, 0])
        self.assertEqual(logprob_counts, [8, 0])

    def test_fixed_width_spec_skips_generic_alignment_and_uses_shared_bucket(self):
        policy = _make_spec_policy(bs_buckets=(1, 4, 8, 16))
        forward_batch = SimpleNamespace(
            forward_mode=ForwardMode.TARGET_VERIFY,
            spec_info=SimpleNamespace(num_tokens_per_req=4),
            input_ids=torch.zeros(12),
        )

        with _dp_attention(True):
            counts, logprob_counts = policy.get_dp_padded_token_counts(
                forward_batch,
                policy.model_runner,
                [12, 0],
                [3, 0],
                attn_tp_size=6,
                cp_align_size=1,
            )

        self.assertEqual(counts, [16, 16])
        self.assertEqual(logprob_counts, [3, 0])

    def test_rounds_agreed_width_up_to_a_batch_bucket(self):
        policy = _make_spec_policy(bs_buckets=(1, 4, 8, 16))
        forward_batch = SimpleNamespace(
            spec_info=SimpleNamespace(num_tokens_per_req=4),
            input_ids=torch.zeros(28),
        )
        self.assertEqual(policy._bucket_dp_sync_token_width(forward_batch, 28), 32)

    def test_keeps_an_already_materialized_wider_bucket(self):
        policy = _make_spec_policy(bs_buckets=(1, 4, 8, 16))
        forward_batch = SimpleNamespace(
            spec_info=SimpleNamespace(num_tokens_per_req=1),
            input_ids=torch.zeros(16),
        )

        self.assertEqual(policy._bucket_dp_sync_token_width(forward_batch, 8), 16)

    def test_keeps_materialized_request_rows_for_draft_extend(self):
        policy = _make_spec_policy(bs_buckets=(1, 4, 8, 16))
        forward_batch = SimpleNamespace(
            spec_info=SimpleNamespace(num_tokens_per_req=4),
            input_ids=torch.zeros(4),
            req_pool_indices=torch.zeros(4, dtype=torch.int32),
        )

        self.assertEqual(policy._bucket_dp_sync_token_width(forward_batch, 4), 16)


class TestSpecDraftExtendPadding(_SpecPolicyTestCase):
    def test_pads_extend_metadata_tensors_off_bucket(self):
        """Bug regression: at an off-bucket bs, `_pad_inputs_to_size` grows the
        core request-axis tensors but left `extend_seq_lens` /
        `extend_prefix_lens` / `extend_start_loc` at real_bs, so the compiled
        DRAFT_EXTEND_V2 graph (traced with them at the bucket length) was
        invoked with shapes the trace never saw."""
        policy = _make_spec_policy(bs_buckets=(1, 4))
        forward_batch = SimpleNamespace(
            input_ids=torch.ones(12, dtype=torch.int32),
            seq_lens=torch.ones(3, dtype=torch.int32),
            spec_info=None,
            _pad_inputs_to_size=MagicMock(),
            extend_seq_lens=torch.tensor([4, 4, 4], dtype=torch.int32),
            extend_prefix_lens=torch.tensor([8, 7, 6], dtype=torch.int32),
            extend_start_loc=torch.tensor([0, 4, 8], dtype=torch.int32),
            extend_seq_lens_cpu=[4, 4, 4],
            extend_prefix_lens_cpu=[8, 7, 6],
            extend_logprob_start_lens_cpu=[0, 0, 0],
            mm_inputs=None,
        )

        policy._pad_draft_extend(forward_batch)

        self.assertEqual(forward_batch.batch_size, 4)
        self.assertEqual(forward_batch.extend_seq_lens.tolist(), [4, 4, 4, 0])
        self.assertEqual(
            forward_batch.extend_seq_lens_cpu, forward_batch.extend_seq_lens.tolist()
        )
        self.assertEqual(forward_batch.extend_prefix_lens.tolist(), [8, 7, 6, 0])
        self.assertEqual(forward_batch.extend_start_loc.tolist(), [0, 4, 8, 0])

    def test_normalizes_stale_cpu_lists_at_exact_bucket(self):
        policy = _make_spec_policy(bs_buckets=(1, 4))
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

    def test_pads_to_dp_peer_request_width(self):
        policy = _make_spec_policy(bs_buckets=(1, 4))
        forward_batch = SimpleNamespace(
            input_ids=torch.ones(4, dtype=torch.int32),
            seq_lens=torch.ones(1, dtype=torch.int32),
            spec_info=None,
            global_num_tokens_cpu=[16, 4],
            extend_seq_lens_cpu=[4],
            extend_prefix_lens_cpu=[0],
            extend_logprob_start_lens_cpu=[0],
            mm_inputs=[None],
        )
        pad_calls: list[tuple[int, int]] = []
        forward_batch._pad_inputs_to_size = lambda *, model_runner, num_tokens, bs: (
            pad_calls.append((num_tokens, bs))
        )

        with patch(
            _SPEC_POLICY + "is_dp_attention_enabled",
            return_value=True,
        ):
            policy._pad_draft_extend(forward_batch)

        self.assertEqual(pad_calls, [(16, 4)])
        self.assertEqual(forward_batch.batch_size, 4)


class TestSpecUnpadOutputs(_SpecPolicyTestCase):
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

    def test_target_verify_defers_output_unpadding_until_after_sampling(self):
        policy = _make_spec_policy()
        out, fb = self._out_and_batch(
            ForwardMode.TARGET_VERIFY,
            CaptureHiddenMode.FULL,
            real_q=4,
            real_b=1,
            rows=8,
        )
        fb.spec_info = SimpleNamespace(keep_outputs_padded=True)

        policy.finalize_forward_batch(out, fb)
        self.assertEqual(out.next_token_logits.shape[0], 8)
        self.assertEqual(out.hidden_states.shape[0], 8)

        fb.spec_info.keep_outputs_padded = False
        policy.finalize_forward_batch(out, fb)
        self.assertEqual(out.next_token_logits.shape[0], 4)
        self.assertEqual(out.hidden_states.shape[0], 4)

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

    def test_final_draft_outputs_stay_padded_until_draft_loop_finishes(self):
        policy = _make_spec_policy()
        spec_info = SimpleNamespace(keep_outputs_padded=True)
        fb = SimpleNamespace(_tpu_real_b=1, batch_size=4, spec_info=spec_info)
        outputs = (torch.arange(4), torch.arange(8).reshape(4, 2), None)

        padded = policy.finalize_spec_forward_batch(outputs, fb)

        self.assertIs(padded, outputs)
        self.assertEqual([t.shape[0] for t in padded[:2]], [4, 4])
        self.assertEqual(fb.batch_size, 4)
        self.assertTrue(spec_info.keep_outputs_padded)

    def test_dp_idle_sentinel_row_pads_to_peer_draft_width(self):
        policy = _make_spec_policy(bs_buckets=(1, 4))
        draft_input = EagleDraftInput(
            topk_p=torch.zeros((1, 1)),
            topk_index=torch.full((1, 1), 31999, dtype=torch.int64),
            hidden_states=torch.zeros((1, 16)),
        )
        forward_batch = _dp_idle_batch(
            draft_input,
            peer_tokens=1,
            batch_size=1,
            seq_lens_sum=1,
            input_ids=torch.zeros((1,), dtype=torch.int64),
            positions=torch.zeros((1,), dtype=torch.int64),
            seq_lens=torch.ones((1,), dtype=torch.int64),
            seq_lens_cpu=torch.ones((1,), dtype=torch.int64),
            out_cache_loc=torch.zeros((1,), dtype=torch.int64),
            req_pool_indices=torch.zeros((1,), dtype=torch.int64),
            req_pool_indices_cpu=torch.zeros((1,), dtype=torch.int64),
        )

        with _dp_attention(True, attn_tp_size=4):
            policy.prepare_spec_forward_batch(forward_batch)

        self.assertEqual(forward_batch.batch_size, 4)
        self.assertEqual(forward_batch.seq_lens.shape, (4,))
        self.assertEqual(forward_batch.positions.shape, (4,))
        self.assertEqual(forward_batch.out_cache_loc.shape, (12,))
        self.assertEqual(draft_input.topk_index.shape, (4, 1))
        self.assertEqual(forward_batch.forward_mode, ForwardMode.DECODE)
        self.assertTrue(draft_input.keep_outputs_padded)
        self.assertTrue(forward_batch._original_forward_mode.is_idle())
        self.assertEqual(draft_input.padding_state.logical_batch_size, 0)
        self.assertEqual(draft_input.padding_state.physical_batch_size, 4)

    def test_relabelled_idle_preserves_padded_outputs(self):
        policy = _make_spec_policy()
        out, fb = self._out_and_batch(
            ForwardMode.DECODE,
            CaptureHiddenMode.NULL,
            real_q=0,
            real_b=0,
            rows=4,
        )
        fb._original_forward_mode = ForwardMode.IDLE
        fb.spec_info = SimpleNamespace(keep_outputs_padded=True, hidden_states=None)

        policy.finalize_forward_batch(out, fb)

        self.assertEqual(out.next_token_logits.shape[0], 4)
        self.assertEqual(out.hidden_states.shape[0], 4)
        self.assertEqual(fb.batch_size, 0)
        self.assertTrue(fb.spec_info.keep_outputs_padded)


class TestSpecForwardBatchPreparation(_SpecPolicyTestCase):
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
            seq_lens=torch.tensor([1], dtype=torch.int32),
        )

        policy.prepare_spec_forward_batch(forward_batch)

        policy._pad_draft_decode.assert_called_once_with(forward_batch)
        self.assertTrue(spec_info.keep_outputs_padded)

    def test_draft_extend_dispatches(self):
        policy = self._policy()
        forward_batch = SimpleNamespace(
            forward_mode=ForwardMode.DRAFT_EXTEND_V2,
            spec_info=None,
            seq_lens=torch.tensor([1], dtype=torch.int32),
        )

        policy.prepare_spec_forward_batch(forward_batch)

        policy._pad_draft_extend.assert_called_once_with(forward_batch)

    def test_idle_is_noop(self):
        policy = self._policy()
        forward_batch = SimpleNamespace(
            forward_mode=ForwardMode.IDLE,
            spec_info=None,
            seq_lens=torch.tensor([], dtype=torch.int32),
        )

        policy.prepare_spec_forward_batch(forward_batch)

        policy._pad_draft_decode.assert_not_called()
        policy._pad_draft_extend.assert_not_called()

    def test_target_verify_is_noop(self):
        policy = self._policy()
        forward_batch = SimpleNamespace(
            forward_mode=ForwardMode.TARGET_VERIFY,
            spec_info=None,
            seq_lens=torch.tensor([1], dtype=torch.int32),
        )

        policy.prepare_spec_forward_batch(forward_batch)

        policy._pad_draft_decode.assert_not_called()
        policy._pad_draft_extend.assert_not_called()

    def test_draft_decode_buckets_before_input_ids_are_built(self):
        policy = _make_spec_policy(bs_buckets=(1, 4))
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

    def test_fused_draft_refreshes_dp_metadata_after_inputs_are_consumed(self):
        policy = _make_spec_policy(bs_buckets=(1, 8))
        forward_batch = SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            input_ids=None,
            seq_lens=torch.ones((8,), dtype=torch.int32),
            batch_size=8,
            global_num_tokens_cpu=[3, 8],
            global_num_tokens_gpu=torch.tensor([3, 8]),
            global_num_tokens_for_logprob_cpu=[3, 8],
            global_num_tokens_for_logprob_gpu=torch.tensor([3, 8]),
            dp_local_start_pos=3,
            dp_local_num_tokens=8,
            global_dp_buffer_len=11,
            is_extend_in_batch=False,
        )

        with (
            _dp_attention(True),
            patch("sglang.srt.layers.dp_attention.set_dp_buffer_len") as set_buffer_len,
            patch(
                "sglang.srt.layers.dp_attention.set_is_extend_in_batch"
            ) as set_is_extend,
        ):
            policy.refresh_fused_draft_dp_metadata(forward_batch)

        self.assertEqual(forward_batch.global_num_tokens_cpu, [8, 8])
        self.assertEqual(forward_batch.global_num_tokens_gpu.tolist(), [8, 8])
        self.assertEqual(forward_batch.global_num_tokens_for_logprob_cpu, [8, 8])
        self.assertEqual(
            forward_batch.global_num_tokens_for_logprob_gpu.tolist(), [8, 8]
        )
        self.assertEqual(forward_batch.global_dp_buffer_len, 16)
        self.assertIsNone(forward_batch.dp_local_start_pos)
        self.assertIsNone(forward_batch.dp_local_num_tokens)
        set_buffer_len.assert_called_once_with(16, 8, True, [8, 8])
        set_is_extend.assert_called_once_with(False)

    def test_draft_extend_buckets_to_the_full_tree_width(self):
        policy = _make_spec_policy(bs_buckets=(1, 4))
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
            extend_seq_lens_cpu=None,
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


class TestDraftPrecompile(_SpecPolicyTestCase):
    """``precompile_draft`` warms the three draft forward families."""

    def test_draft_worker_compiles_prefill_decode_and_extend(self):
        mr = _spec_model_runner(is_draft_worker=True)
        policy = _make_spec_policy(mr, token_buckets=(8, 64), bs_buckets=(1, 4))
        step_backends = [_FakeAttnBackend() for _ in range(3)]
        draft_attn_backend = SimpleNamespace(attn_backends=step_backends)
        draft_extend_attn_backend = _FakeAttnBackend()

        with get_parallel().override(
            tp_size=2,
            tp_rank=1,
            attn_tp_size=2,
            attn_tp_rank=1,
            moe_tp_size=2,
            moe_tp_rank=1,
        ):
            policy.precompile_draft(draft_attn_backend, draft_extend_attn_backend)

        # EXTEND(LAST) 2 token buckets at the bs cap (text-only target, so no
        # embeds variant) + DECODE 2 bs x 3 steps + DRAFT_EXTEND_V2 2 bs buckets.
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

    def test_fused_draft_warms_every_batch_bucket(self):
        mr = _spec_model_runner(is_draft_worker=True)
        policy = _make_spec_policy(mr, token_buckets=(8,), bs_buckets=(1, 4))
        step_backends = [_FakeAttnBackend() for _ in range(3)]

        class FakeDraftBackend:
            attn_backends = step_backends

            def init_forward_metadata(self, forward_batch):
                for backend in self.attn_backends:
                    backend.init_forward_metadata(forward_batch)

        seen_shapes = []

        def fused_draft_forward(forward_batch):
            seen_shapes.append(
                (
                    forward_batch.batch_size,
                    forward_batch.out_cache_loc.shape[0],
                    forward_batch.spec_info.topk_p.shape[0],
                    forward_batch.input_ids,
                    forward_batch.spec_info.keep_outputs_padded,
                )
            )
            return torch.zeros(forward_batch.batch_size)

        with (
            get_parallel().override(
                tp_size=2,
                tp_rank=1,
                attn_tp_size=2,
                attn_tp_rank=1,
                moe_tp_size=2,
                moe_tp_rank=1,
            ),
            patch(
                "sglang.srt.hardware_backend.tpu.compilation."
                "spec_decode_compile_policy_eagle3._materialize_out"
            ) as materialize,
        ):
            policy._precompile_fused_draft_decode(
                FakeDraftBackend(), fused_draft_forward
            )
        self.assertEqual(materialize.call_count, 2)

        self.assertEqual(
            seen_shapes,
            [(1, 3, 1, None, True), (4, 12, 4, None, True)],
        )
        self.assertEqual([len(backend.calls) for backend in step_backends], [2, 2, 2])

    def test_fused_draft_warmup_leaves_num_token_non_padded_unset(self):
        """The live fused draft loop runs on an init_new batch that never
        passes ModelRunner.forward, so its local ``num_token_non_padded`` is
        None even when serving enables the field. direct_compile pins each
        input's type at trace time, so a warmup batch carrying the tensor
        makes the first live draft step fail with a type change.
        """
        mr = _spec_model_runner(is_draft_worker=True)
        policy = _make_spec_policy(mr, token_buckets=(8,), bs_buckets=(4,))
        step_backends = [_FakeAttnBackend() for _ in range(3)]

        class FakeDraftBackend:
            attn_backends = step_backends

            def init_forward_metadata(self, forward_batch):
                for backend in self.attn_backends:
                    backend.init_forward_metadata(forward_batch)

        seen = []

        def fused_draft_forward(forward_batch):
            seen.append(
                (
                    forward_batch.global_num_token_non_padded_cpu,
                    forward_batch.num_token_non_padded,
                )
            )
            return torch.zeros(forward_batch.batch_size)

        policy_module = (
            "sglang.srt.hardware_backend.tpu.compilation.decoder_compile_policy"
        )
        with (
            get_parallel().override(
                tp_size=2,
                tp_rank=1,
                attn_tp_size=2,
                attn_tp_rank=1,
                moe_tp_size=2,
                moe_tp_rank=1,
            ),
            patch(f"{policy_module}.enable_num_token_non_padded", return_value=True),
            patch(
                "sglang.srt.layers.attention.dsa.utils.is_dsa_enable_prefill_cp",
                return_value=False,
            ),
            patch("sglang.srt.layers.cp.utils.is_mla_cp_enabled", return_value=False),
            patch(
                "sglang.srt.hardware_backend.tpu.compilation."
                "spec_decode_compile_policy_eagle3._materialize_out"
            ),
        ):
            policy._precompile_fused_draft_decode(
                FakeDraftBackend(), fused_draft_forward
            )

        # The global scalar mirrors init_new; the local field stays unset.
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0][0], 4 * policy.speculative_eagle_topk)
        self.assertIsNone(seen[0][1])


# Speculative verification padding fixtures.
BS = 2
BUCKET_BS = 4
# The published EAGLE3 chain (_EAGLE3_CONFIG) is what the policy reads.
DRAFT_TOKEN_NUM = _SPEC_NUM_STEPS + 1
VOCAB_SIZE = 10
LOGICAL_TOKENS = BS * DRAFT_TOKEN_NUM
PADDED_TOKENS = BUCKET_BS * DRAFT_TOKEN_NUM


def _policy():
    """A spec policy that only owns the verify padding; its chain comes from get_spec()."""
    model_runner = SimpleNamespace(
        model=SimpleNamespace(forward=None, prepare_forward_batch=None),
        is_draft_worker=False,
        pp_group=SimpleNamespace(is_first_rank=True, is_last_rank=True),
        model_config=SimpleNamespace(hidden_size=16, model_is_mrope=False),
    )
    with _target_model_config():
        return SpecDecoderTpuCompilePolicy(
            model_runner=model_runner,
            t_bucketer=Bucketer(name="Token", buckets=[PADDED_TOKENS]),
            b_bucketer=Bucketer(name="Bs", buckets=[1, BUCKET_BS]),
            owner=model_runner.model,
        )


def _cpu_top_k_renorm(probs, top_ks, need_top_k_sampling=True):
    """CPU stand-in for the TPU filter: keep values >= the k-th largest."""
    if not need_top_k_sampling:
        return probs
    ks = top_ks.clamp(1, probs.shape[-1]).long().reshape(-1, 1)
    kth = probs.sort(dim=-1, descending=True).values.gather(-1, ks - 1)
    kept = torch.where(probs >= kth, probs, torch.zeros_like(probs))
    return kept / kept.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def _cpu_top_p_renorm(probs, top_ps):
    """CPU stand-in for the TPU filter: keep values >= the first value whose
    prefix mass reaches p; an unreachable p keeps the whole row."""
    ordered = probs.sort(dim=-1, descending=True).values
    reached = ordered.cumsum(dim=-1) >= top_ps.to(probs.dtype).reshape(-1, 1)
    first = reached.to(torch.int64).argmax(dim=-1, keepdim=True)
    cutoff = torch.where(
        reached.any(dim=-1, keepdim=True),
        ordered.gather(-1, first),
        torch.zeros_like(first, dtype=probs.dtype),
    )
    kept = torch.where(probs >= cutoff, probs, torch.zeros_like(probs))
    return kept / kept.sum(dim=-1, keepdim=True).clamp_min(1e-12)


@contextlib.contextmanager
def _serving_stubs():
    """Select TPU-native sampling and stub serving process state.

    This suite runs on CPU CI, where platform detection would otherwise leave
    ``verify_tree_greedy_func`` without a CPU implementation and its mutable
    outputs unwritten. Production TPU selects this torch-native path itself.

    Keep the penalty helper eager as production TPU startup does. This makes a
    manual run inside a TPU image exercise the CPU unit-test math instead of
    sending its CPU tensors through the TPU-only ``dynamic=True`` compiler.
    """
    from sglang.srt.sampling.penaltylib import repetition_penalty

    compiled_scaling_penalties = repetition_penalty.apply_scaling_penalties
    eager_scaling_penalties = getattr(
        compiled_scaling_penalties,
        "__wrapped__",
        compiled_scaling_penalties,
    )

    with (
        patch.object(
            repetition_penalty,
            "apply_scaling_penalties",
            eager_scaling_penalties,
        ),
        patch("sglang.srt.speculative.eagle_utils._is_tpu", True),
        # The TPU renorm filter only accepts TPU tensors.
        patch(
            "sglang.srt.speculative.eagle_utils.top_k_renorm_prob_func",
            _cpu_top_k_renorm,
        ),
        patch(
            "sglang.srt.speculative.eagle_utils.top_p_renorm_prob_func",
            _cpu_top_p_renorm,
        ),
        get_parallel().override(tp_group=SimpleNamespace(world_size=1)),
        patch(
            "sglang.srt.layers.dp_attention.is_dp_attention_enabled",
            return_value=False,
        ),
        patch(
            "torch._dynamo.config.disable",
            True,
        ),
        # Layered onto the published config: eagle_sample reads the rest of the spec bag.
        get_spec().override(
            speculative_use_rejection_sampling=False,
            speculative_accept_threshold_single=1.0,
            speculative_accept_threshold_acc=1.0,
        ),
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


class TestSpecVerifyPaddingRecord(_SpecPolicyTestCase):
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

    def test_idle_peer_verify_keeps_bookkeeping_padded_until_finalize(self):
        """An idle DP peer must drain verification without mixing empty and padded rows."""
        for physical_bs in (1, 2, 4):
            with self.subTest(physical_bs=physical_bs):
                batch = _make_batch(temperature=0.0)
                batch.forward_mode = ForwardMode.IDLE
                batch.global_num_tokens_cpu = [physical_bs * DRAFT_TOKEN_NUM, 0]
                batch.seq_lens = batch.seq_lens[:0]
                logical_seq_lens = batch.seq_lens
                batch.input_ids = torch.empty(0, dtype=torch.int64)
                batch.out_cache_loc = torch.empty(0, dtype=torch.int64)
                batch.sampling_info.temperatures = batch.sampling_info.temperatures[:0]
                batch.sampling_info.top_ks = batch.sampling_info.top_ks[:0]
                batch.sampling_info.top_ps = batch.sampling_info.top_ps[:0]
                verify_input = EagleVerifyInput.create_idle_input(
                    topk=1,
                    spec_steps=DRAFT_TOKEN_NUM - 1,
                    num_verify_tokens=DRAFT_TOKEN_NUM,
                    device="cpu",
                )
                verify_input.record_padding(
                    logical_batch_size=0,
                    physical_batch_size=physical_bs,
                    logical_num_tokens=0,
                    physical_num_tokens=physical_bs * DRAFT_TOKEN_NUM,
                )
                batch.spec_info = verify_input
                policy = _policy()
                logits_output = LogitsProcessorOutput(
                    next_token_logits=torch.zeros(
                        (physical_bs * DRAFT_TOKEN_NUM, VOCAB_SIZE)
                    )
                )
                target_worker = SimpleNamespace(
                    model_runner=SimpleNamespace(compile_policy=policy),
                    forward_batch_generation=MagicMock(
                        return_value=SimpleNamespace(
                            logits_output=logits_output,
                            routed_experts_output=None,
                            indexer_topk_output=None,
                        )
                    ),
                )
                forward_batch = SimpleNamespace(spec_info=verify_input)
                module = "sglang.srt.speculative.eagle_worker_common"
                with (
                    _serving_stubs(),
                    _dp_attention(True),
                    patch(f"{module}.record_stream_for_v2_verify"),
                    patch(f"{module}.record_stream_each"),
                    patch(
                        f"{module}.eagle_prepare_for_verify",
                        return_value=(forward_batch, False),
                    ),
                    patch(f"{module}.commit_mamba_states_after_verify"),
                    patch.object(
                        policy, "finalize_forward_batch", return_value=logits_output
                    ),
                ):
                    result = run_eagle_verify(
                        batch,
                        target_worker=target_worker,
                        req_to_token_pool=None,
                        token_to_kv_pool_allocator=SimpleNamespace(
                            get_kvcache=lambda: SimpleNamespace()
                        ),
                        plan_stream=None,
                        plan_stream_ctx=contextlib.nullcontext(),
                        topk=1,
                        num_draft_tokens=DRAFT_TOKEN_NUM,
                        device="cpu",
                        metadata_ready_pre_pad=False,
                        finalize_tree_path=False,
                    )
                self.assertEqual(batch.forward_mode, ForwardMode.IDLE)
                self.assertIs(batch.seq_lens, logical_seq_lens)
                self.assertEqual(verify_input.draft_token.numel(), 0)
                for tensor in (
                    result.next_token_ids,
                    result.accept_lens,
                    result.new_seq_lens,
                    result.next_draft_input.bonus_tokens,
                ):
                    self.assertEqual(tensor.numel(), 0)

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

    def test_widens_idle_partition_to_active_peer_verify_shape(self):
        batch = _make_batch(temperature=0.7)
        batch.forward_mode = ForwardMode.IDLE
        batch.global_num_tokens_cpu = [DRAFT_TOKEN_NUM, 0]
        batch.seq_lens = batch.seq_lens[:0]
        batch.sampling_info.temperatures = batch.sampling_info.temperatures[:0]
        batch.sampling_info.top_ks = batch.sampling_info.top_ks[:0]
        batch.sampling_info.top_ps = batch.sampling_info.top_ps[:0]
        verify_input = EagleVerifyInput.create_idle_input(
            topk=1,
            spec_steps=DRAFT_TOKEN_NUM - 1,
            num_verify_tokens=DRAFT_TOKEN_NUM,
            device="cpu",
        )
        verify_input.record_padding(
            logical_batch_size=0,
            physical_batch_size=1,
            logical_num_tokens=0,
            physical_num_tokens=DRAFT_TOKEN_NUM,
        )
        policy = _policy()

        with _dp_attention(True):
            self.assertTrue(
                policy.should_defer_verify_output_unpadding(batch, verify_input)
            )

        padding = policy.prepare_spec_verify(
            batch,
            verify_input,
            torch.zeros((DRAFT_TOKEN_NUM, VOCAB_SIZE)),
        )

        self.assertIsNotNone(padding)
        self.assertEqual(batch.forward_mode, ForwardMode.TARGET_VERIFY)
        self.assertEqual(batch.seq_lens.shape, (1,))
        self.assertEqual(verify_input.draft_token.shape, (DRAFT_TOKEN_NUM,))
        self.assertEqual(verify_input.retrieve_index.shape, (1, DRAFT_TOKEN_NUM))
        verify_result, _ = policy.finalize_spec_verify(
            padding,
            (
                torch.zeros(DRAFT_TOKEN_NUM, dtype=torch.int32),
                torch.zeros(1, dtype=torch.int32),
                torch.zeros((1, DRAFT_TOKEN_NUM), dtype=torch.int32),
            ),
            (None, None),
        )
        predict, accept_lens, accept_index = verify_result
        self.assertEqual(batch.forward_mode, ForwardMode.IDLE)
        self.assertEqual(predict.shape, (0,))
        self.assertEqual(accept_lens.shape, (0,))
        self.assertEqual(accept_index.shape, (0, DRAFT_TOKEN_NUM))

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
        # The host copies of these run later on the result-copy thread; they
        # must own their storage so no slice needs materializing then.
        self.assertIsNone(predict._base)
        self.assertIsNone(accept_lens._base)
        self.assertEqual(predict.tolist(), list(range(LOGICAL_TOKENS)))
        self.assertEqual(accept_lens.tolist(), list(range(BS)))

    def test_skips_when_the_forward_did_not_defer(self):
        batch = _make_batch(temperature=0.7)
        verify_input = _make_verify_input(deferred=False)
        logits = torch.zeros((LOGICAL_TOKENS, VOCAB_SIZE))

        padding = _policy().prepare_spec_verify(batch, verify_input, logits)

        self.assertIsNone(padding)
        self.assertEqual(batch.seq_lens.shape, (BS,))

    def test_accepts_logits_already_unpadded_by_the_runner(self):
        batch = _make_batch(temperature=0.7)
        verify_input = _make_verify_input()
        logits = torch.zeros((LOGICAL_TOKENS, VOCAB_SIZE))

        padding = _policy().prepare_spec_verify(batch, verify_input, logits)

        self.assertIsNone(padding)
        self.assertFalse(verify_input.keep_outputs_padded)
        self.assertEqual(batch.seq_lens.shape, (BS,))

    def test_rejects_logits_that_do_not_match_the_recorded_padding(self):
        batch = _make_batch(temperature=0.7)
        verify_input = _make_verify_input()
        logits = torch.zeros((PADDED_TOKENS - 1, VOCAB_SIZE))

        with self.assertRaisesRegex(ValueError, "do not match the recorded padding"):
            _policy().prepare_spec_verify(batch, verify_input, logits)


class TestSpecVerifyPaddingIsInert(_SpecPolicyTestCase):
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
