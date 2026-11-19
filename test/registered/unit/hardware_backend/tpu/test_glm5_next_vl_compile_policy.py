# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Glm5NextVLDecoderTpuCompilePolicy and NEXTN MTP integration."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

import inspect
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.environ import envs
from sglang.srt.layers import dp_attention
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardMode,
    LogitsOutputMode,
)
from sglang.srt.observability.compile_timing import CompileTimingRecorder
from sglang.srt.runtime_context import get_context, get_flags, get_parallel
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.test_utils import CustomTestCase
from sglang_tpu import mm_embed_tpu
from sglang_tpu import sampler as tpu_sampler_module
from sglang_tpu.compilation.bucketer import Bucketer
from sglang_tpu.compilation.compiled_forward_shim import _static_default_kwargs
from sglang_tpu.compilation.decoder_compile_policy import (
    Glm5NextVLDecoderTpuCompilePolicy,
    Qwen3VLDecoderTpuCompilePolicy,
)
from sglang_tpu.envs import envs as tpu_envs
from sglang_tpu.experimental.compilation.chain_compile_policy import (
    TpuChainSpecDecodeCompilePolicyMixin,
)
from sglang_tpu.experimental.compilation.spec_compile_policy import (
    TpuSpecDecodeCompilePolicyMixin,
)
from sglang_tpu.experimental.sampling.sampling_batch_info import HostPaddedSamplingInfo
from sglang_tpu.hooks.glm5_next_vision import _Glm5NextLanguageModel
from sglang_tpu.top_k import PallasDistributedTopKConfig, TpuVocabShard

_BASE_CONFIG = {
    "page_size": 1,
    "dp_size": 1,
    "disaggregation_mode": "null",
}

_NEXTN_CONFIG = {
    **_BASE_CONFIG,
    "speculative_algorithm": "NEXTN",
    "speculative_num_steps": 3,
    "speculative_num_draft_tokens": 4,
    "speculative_eagle_topk": 1,
}


class _FakeLmHead(torch.nn.Module):
    def __init__(self, *, shard_width: int) -> None:
        super().__init__()
        self.enable_tp = True
        self.num_embeddings = 154_880
        self.org_vocab_size = 154_880
        self.tp_size = 8
        self.num_embeddings_per_partition = shard_width
        self.shard_indices = object()


class _RecordingEmbedding(torch.nn.Module):
    def __init__(self, inner: torch.nn.Embedding) -> None:
        super().__init__()
        self._inner = inner
        self.num_embeddings = inner.num_embeddings
        self.embedding_dim = inner.embedding_dim
        self.call_count = 0

    @property
    def weight(self) -> torch.nn.Parameter:
        return self._inner.weight

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        self.call_count += 1
        return self._inner(ids)


class _DummyDecoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(32, 8)
        self.layers_to_capture: list[int] = []

    def forward(
        self,
        input_ids=None,
        positions=None,
        forward_batch=None,
        input_embeds=None,
        pp_proxy_tensors=None,
    ):
        rows = input_embeds.shape[0] if input_embeds is not None else input_ids.shape[0]
        return torch.zeros((rows, 8), dtype=torch.float32)


class _DummyGlm5Outer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.visual = torch.nn.Linear(4, 8)
        self.model = _DummyDecoder()
        self.lm_head = torch.nn.Linear(8, 32, bias=False)
        self.pp_group = SimpleNamespace(is_first_rank=True, is_last_rank=True)
        self.logits_processor = MagicMock(name="logits_processor")
        self.capture_aux_hidden_states = False
        self.decode_embeds_in_language_model = True
        self.is_mrope_enabled = True
        self.language_model = _Glm5NextLanguageModel(self)

    def get_image_feature(self, items):
        return items

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def prepare_forward_batch(self, forward_batch):
        return None


def _make_runner(*, distributed_top_k: bool = False):
    outer = _DummyGlm5Outer()
    server_args = SimpleNamespace(
        tpu_compile_tracer="dynamo",
        enable_attn_tp_input_scattered=False,
        unsafe_save_torch_compile_bundle=None,
        unsafe_load_torch_compile_bundle=None,
        tpu_compile_tokens=[1024, 2048, 4096, 8192],
        tpu_compile_bs=[1, 4, 16, 64],
        torch_compile_max_bs=64,
        max_running_requests=64,
        chunked_prefill_size=8192,
        max_prefill_tokens=8192,
    )
    runner = SimpleNamespace(
        device="cpu",
        dtype=torch.bfloat16,
        model=outer,
        server_args=server_args,
        attn_backend=SimpleNamespace(init_forward_metadata=lambda fb: None),
        req_to_token_pool=SimpleNamespace(
            req_to_token=torch.zeros((8, 8192), dtype=torch.int32)
        ),
        token_to_kv_pool=SimpleNamespace(),
        token_to_kv_pool_allocator=SimpleNamespace(
            free_pages=torch.arange(16, dtype=torch.int32)
        ),
        model_config=SimpleNamespace(
            model_is_mrope=True,
            context_len=10240,
            hidden_size=8,
            is_multimodal=True,
        ),
        is_draft_worker=False,
        draft_model_idx=None,
        spec_algorithm=SpeculativeAlgorithm.NONE,
        pp_group=outer.pp_group,
        prepare_dummy_forward_batch=lambda fb: fb,
    )
    if distributed_top_k:
        shard_width = 154_880 // 8
        shard = TpuVocabShard(
            width=shard_width,
            global_vocab_size=154_880,
            org_vocab_size=154_880,
            org_vocab_start=0,
            org_vocab_count=shard_width,
            org_vocab_padded_count=shard_width,
            added_vocab_start=154_880,
            added_vocab_count=0,
        )
        logits_processor = SimpleNamespace(
            use_tpu_distributed_top_k=True,
            do_tensor_parallel_all_gather=True,
            do_tensor_parallel_all_gather_dp_attn=False,
        )
        lm_head = _FakeLmHead(shard_width=shard_width)
        config = PallasDistributedTopKConfig(
            shard=shard,
            world_size=8,
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
        outer.logits_processor = logits_processor
        outer.lm_head = lm_head
    return runner


def _make_glm_policy(
    runner=None,
    *,
    token_buckets=(1024, 2048, 4096, 8192),
    bs_buckets=(1, 4, 16, 64),
    recorder: CompileTimingRecorder | None = None,
) -> Glm5NextVLDecoderTpuCompilePolicy:
    if runner is None:
        runner = _make_runner()
    policy = Glm5NextVLDecoderTpuCompilePolicy(
        model_runner=runner,
        t_bucketer=Bucketer(name="Token", buckets=list(token_buckets)),
        b_bucketer=Bucketer(name="Bs", buckets=list(bs_buckets)),
        owner=runner.model.language_model,
        method_name="forward",
        method=runner.model.language_model.forward,
        name="target/language_model.forward",
        label="_Glm5NextLanguageModel",
        compile_timing_recorder=recorder,
    )
    policy.extend_bs = max(bs_buckets)
    return policy


class TestGlm5NextVLDecoderTpuCompilePolicy(CustomTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.publish_config()

    def publish_config(self, **fields):
        override = get_context().override_server_args(**{**_BASE_CONFIG, **fields})
        override.install()
        self.addCleanup(override.restore)
        return override

    def test_subclass_and_spmd_safe_contract(self):
        self.assertTrue(
            issubclass(
                Glm5NextVLDecoderTpuCompilePolicy, Qwen3VLDecoderTpuCompilePolicy
            )
        )
        self.assertTrue(Glm5NextVLDecoderTpuCompilePolicy.spmd_safe)

    def test_direct_compile_passes_spmd_safe_and_keys_all_call_shapes(self):
        from sglang_tpu.compilation.graph_identity import _key_value, canonical

        runner = _make_runner(distributed_top_k=True)
        self.addCleanup(
            tpu_sampler_module._DISTRIBUTED_TOP_K_STATES.pop,
            id(runner.model.logits_processor),
            None,
        )
        runner.server_args.tpu_compile_tracer = "direct"
        policy = _make_glm_policy(runner)
        policy.tracer = "direct"

        captured = {}

        def fake_direct_compile(fn, **kwargs):
            captured["fn"] = fn
            captured["kwargs"] = kwargs
            compiled = MagicMock(name="compiled_fn")
            compiled.warmup_async = MagicMock(name="warmup_async")
            return compiled

        with patch(
            "sglang_tpu.compilation.direct_compile.direct_compile",
            side_effect=fake_direct_compile,
        ):
            compiled_call, _ = policy._direct_compile(
                runner.model.language_model.forward
            )

        self.assertTrue(captured["kwargs"]["spmd_safe"])
        self.assertEqual(captured["kwargs"]["name"], "target/language_model.forward")
        self.assertIs(captured["kwargs"]["model"], runner.model)

        key_fn = captured["kwargs"]["specialization_key"]
        positions_1k = torch.zeros((3, 1024), dtype=torch.int64)
        positions_64 = torch.zeros((3, 64), dtype=torch.int64)
        positions_256 = torch.zeros((3, 256), dtype=torch.int64)

        extend_embeds_key = key_fn(
            None,
            None,
            positions_1k,
            SimpleNamespace(
                forward_mode=ForwardMode.EXTEND,
                capture_hidden_mode=CaptureHiddenMode.NULL,
                batch_size=64,
                logits_output_mode=LogitsOutputMode.FULL_VOCAB,
            ),
            input_embeds=torch.zeros((1024, 8), dtype=torch.bfloat16),
        )
        extend_ids_key = key_fn(
            None,
            torch.zeros(1024, dtype=torch.int64),
            positions_1k,
            SimpleNamespace(
                forward_mode=ForwardMode.EXTEND,
                capture_hidden_mode=CaptureHiddenMode.NULL,
                batch_size=64,
                logits_output_mode=LogitsOutputMode.FULL_VOCAB,
            ),
        )
        decode_full_key = key_fn(
            None,
            torch.zeros(64, dtype=torch.int64),
            positions_64,
            SimpleNamespace(
                forward_mode=ForwardMode.DECODE,
                capture_hidden_mode=CaptureHiddenMode.NULL,
                batch_size=64,
                logits_output_mode=LogitsOutputMode.FULL_VOCAB,
            ),
        )
        decode_topk_key = key_fn(
            None,
            torch.zeros(64, dtype=torch.int64),
            positions_64,
            SimpleNamespace(
                forward_mode=ForwardMode.DECODE,
                capture_hidden_mode=CaptureHiddenMode.NULL,
                batch_size=64,
                logits_output_mode=LogitsOutputMode.DISTRIBUTED_TOP_K,
            ),
        )
        verify_key = key_fn(
            None,
            torch.zeros(256, dtype=torch.int64),
            positions_256,
            SimpleNamespace(
                forward_mode=ForwardMode.TARGET_VERIFY,
                capture_hidden_mode=CaptureHiddenMode.FULL,
                batch_size=64,
                logits_output_mode=LogitsOutputMode.FULL_VOCAB,
            ),
        )

        keys = [
            extend_embeds_key,
            extend_ids_key,
            decode_full_key,
            decode_topk_key,
            verify_key,
        ]
        # Every key must be distinct and JSON-serializable for SPMDCompiler graph naming.
        serialized = [canonical(_key_value(k)) for k in keys]
        self.assertEqual(len(set(serialized)), len(keys))
        self.assertEqual(
            extend_embeds_key,
            (
                ForwardMode.EXTEND,
                CaptureHiddenMode.NULL,
                64,
                1024,
                True,
                False,
                LogitsOutputMode.FULL_VOCAB,
            ),
        )
        self.assertEqual(
            decode_topk_key,
            (
                ForwardMode.DECODE,
                CaptureHiddenMode.NULL,
                64,
                64,
                False,
                False,
                LogitsOutputMode.DISTRIBUTED_TOP_K,
            ),
        )

        # Verify CompiledForwardShim strips default kwargs (input_embeds=None,
        # pp_proxy_tensors=None) under direct tracer so serving calls match
        # warmup's exact (args, kwargs) structure.
        compiled_call.__signature__ = inspect.signature(
            runner.model.language_model.forward
        )
        self.assertEqual(
            _static_default_kwargs(compiled_call),
            {"input_embeds": None, "pp_proxy_tensors": None},
        )

    def test_language_model_owns_logits_processor_and_warms_distributed_top_k(self):
        runner = _make_runner(distributed_top_k=True)
        self.addCleanup(
            tpu_sampler_module._DISTRIBUTED_TOP_K_STATES.pop,
            id(runner.model.logits_processor),
            None,
        )
        glm_policy = _make_glm_policy(runner)
        self.assertTrue(glm_policy._owns_logits_processor())
        self.assertTrue(glm_policy._compiles_logits_processor())
        self.assertEqual(
            glm_policy._warmup_logits_output_modes(16),
            (LogitsOutputMode.FULL_VOCAB,),
        )
        self.assertEqual(
            glm_policy._warmup_logits_output_modes(64),
            (LogitsOutputMode.DISTRIBUTED_TOP_K, LogitsOutputMode.FULL_VOCAB),
        )

        # Compare against Qwen3VLDecoderTpuCompilePolicy attached to model.model
        # (with model.language_model unset, as in PR #2098): logits stay unowned.
        baseline_runner = _make_runner(distributed_top_k=True)
        self.addCleanup(
            tpu_sampler_module._DISTRIBUTED_TOP_K_STATES.pop,
            id(baseline_runner.model.logits_processor),
            None,
        )
        delattr(baseline_runner.model, "language_model")
        baseline_runner.model.decode_embeds_in_language_model = False
        qwen_policy = Qwen3VLDecoderTpuCompilePolicy(
            model_runner=baseline_runner,
            t_bucketer=Bucketer(name="Token", buckets=[1024, 2048, 4096, 8192]),
            b_bucketer=Bucketer(name="Bs", buckets=[1, 4, 16, 64]),
            owner=baseline_runner.model.model,
        )
        self.assertFalse(qwen_policy._owns_logits_processor())
        self.assertEqual(
            qwen_policy._warmup_logits_output_modes(64),
            (LogitsOutputMode.FULL_VOCAB,),
        )

    def test_extend_variants_eliminate_dead_text_ids_graphs(self):
        runner = _make_runner()
        runner.model.num_deepstack_embeddings = 2
        policy = _make_glm_policy(runner)

        self.assertEqual(policy._extend_input_variants(), ("text_only",))
        with tpu_envs.SGLANG_TPU_VL_TEXT_ONLY_EXTEND.override(False):
            self.assertEqual(policy._extend_serving_variants(), ("text_only",))
        with tpu_envs.SGLANG_TPU_VL_TEXT_ONLY_EXTEND.override(True):
            self.assertEqual(policy._extend_serving_variants(), ("text_ids",))

    def test_text_ids_warmup_uses_mrope_positions_when_mrope_enabled(self):
        runner = _make_runner()
        policy = _make_glm_policy(runner)
        dummy_batch = SimpleNamespace(
            input_ids=torch.ones(16, dtype=torch.int64),
            positions=torch.arange(16, dtype=torch.int64),
            mrope_positions=torch.zeros((3, 16), dtype=torch.int64),
            forward_mode=ForwardMode.EXTEND,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            batch_size=4,
            logits_output_mode=LogitsOutputMode.FULL_VOCAB,
        )
        policy._make_dummy_forward_batch = MagicMock(return_value=dummy_batch)

        args, kwargs = policy._make_warmup_inputs(
            num_tokens=16,
            bs=4,
            mode=ForwardMode.EXTEND,
            variant="text_ids",
        )
        self.assertEqual(kwargs, {})
        self.assertEqual(len(args), 3)
        self.assertIs(args[0], dummy_batch.input_ids)
        self.assertIs(args[1], dummy_batch.mrope_positions)
        self.assertEqual(args[1].shape, (3, 16))
        self.assertIs(args[2], dummy_batch)

    def test_precompile_decode_warms_single_variant_with_distributed_top_k(self):
        runner = _make_runner(distributed_top_k=True)
        self.addCleanup(
            tpu_sampler_module._DISTRIBUTED_TOP_K_STATES.pop,
            id(runner.model.logits_processor),
            None,
        )
        policy = _make_glm_policy(runner, bs_buckets=(1, 4, 16, 64))
        policy._run_one = MagicMock(return_value=None)

        with patch(
            "sglang_tpu.compilation.decoder_compile_policy.get_parallel",
            return_value=SimpleNamespace(tp_rank=1),
        ):
            policy._precompile_decode()

        calls = [
            (
                c.kwargs["bs"],
                c.kwargs["variant"],
                c.kwargs["logits_output_mode"],
            )
            for c in policy._run_one.call_args_list
        ]
        self.assertEqual(
            calls,
            [
                (1, "text_only", LogitsOutputMode.FULL_VOCAB),
                (4, "text_only", LogitsOutputMode.FULL_VOCAB),
                (16, "text_only", LogitsOutputMode.FULL_VOCAB),
                (64, "text_only", LogitsOutputMode.DISTRIBUTED_TOP_K),
                (64, "text_only", LogitsOutputMode.FULL_VOCAB),
            ],
        )

    def test_precompile_extend_pins_full_vocab_and_honors_prompt_logprob(self):
        runner = _make_runner(distributed_top_k=True)
        self.addCleanup(
            tpu_sampler_module._DISTRIBUTED_TOP_K_STATES.pop,
            id(runner.model.logits_processor),
            None,
        )
        recorder = CompileTimingRecorder()
        policy = _make_glm_policy(
            runner,
            token_buckets=(1024, 2048, 4096, 8192),
            bs_buckets=(1, 4, 16, 64),
            recorder=recorder,
        )
        policy._run_one = MagicMock(return_value=None)
        policy._precompile_mm_embed_placement = MagicMock()

        with (
            tpu_envs.SGLANG_TPU_ENABLE_PROMPT_LOGPROB.override(True),
            patch(
                "sglang_tpu.compilation.decoder_compile_policy.get_parallel",
                return_value=SimpleNamespace(tp_rank=1),
            ),
        ):
            policy._precompile_extend()

        # At bs=64 even with PallasDistributedTopK active on decode, EXTEND
        # compiles 4 FULL_VOCAB graphs (not 8), giving 4 EXTEND + 5 DECODE = 9.
        self.assertEqual(policy._run_one.call_count, 4)
        for call in policy._run_one.call_args_list:
            self.assertEqual(call.kwargs["variant"], "text_only")
            self.assertEqual(
                call.kwargs["logits_output_mode"], LogitsOutputMode.FULL_VOCAB
            )
            self.assertEqual(call.kwargs["capture_hidden_mode"], CaptureHiddenMode.FULL)
        policy._precompile_mm_embed_placement.assert_called_once_with()

    def test_precompile_mm_embed_placement_warms_embed_tokens_and_merge_modes(self):
        runner = _make_runner()
        embed_spy = _RecordingEmbedding(runner.model.model.embed_tokens)
        runner.model.get_input_embeddings = lambda: embed_spy
        policy = _make_glm_policy(
            runner, token_buckets=(1024, 2048), bs_buckets=(1, 16)
        )

        with (
            tpu_envs.SGLANG_TPU_VL_TEXT_ONLY_EXTEND.override(False),
            envs.SGLANG_TPU_MM_EMBED_ON_DEVICE.override(False),
            patch.object(
                mm_embed_tpu,
                "apply_host_mm_embeds",
                wraps=mm_embed_tpu.apply_host_mm_embeds,
            ) as host_spy,
            patch.object(
                mm_embed_tpu,
                "merge_mm_rows",
                wraps=mm_embed_tpu.merge_mm_rows,
            ) as device_spy,
        ):
            policy._precompile_mm_embed_placement()
            # 2 calls per bucket: standalone text-only + uninterrupted multimodal.
            self.assertEqual(embed_spy.call_count, 4)
            self.assertEqual(host_spy.call_count, 2)
            self.assertEqual(device_spy.call_count, 0)

        embed_spy.call_count = 0
        with (
            tpu_envs.SGLANG_TPU_VL_TEXT_ONLY_EXTEND.override(False),
            envs.SGLANG_TPU_MM_EMBED_ON_DEVICE.override(True),
            patch.object(
                mm_embed_tpu,
                "apply_host_mm_embeds",
                wraps=mm_embed_tpu.apply_host_mm_embeds,
            ) as host_spy,
            patch.object(
                mm_embed_tpu,
                "merge_mm_rows",
                wraps=mm_embed_tpu.merge_mm_rows,
            ) as device_spy,
        ):
            policy._precompile_mm_embed_placement()
            self.assertEqual(embed_spy.call_count, 4)
            self.assertEqual(host_spy.call_count, 0)
            self.assertEqual(device_spy.call_count, 2)

        embed_spy.call_count = 0
        with tpu_envs.SGLANG_TPU_VL_TEXT_ONLY_EXTEND.override(True):
            policy._precompile_mm_embed_placement()
            self.assertEqual(embed_spy.call_count, 0)

    def test_nextn_chain_spec_halves_target_extend_graphs_and_keeps_mm_embeds(self):
        self.publish_config(**_NEXTN_CONFIG)
        runner = _make_runner()
        runner.spec_algorithm = SpeculativeAlgorithm.from_string("NEXTN")

        SpecGlm5Policy = type(
            "SpecGlm5NextVLDecoderTpuCompilePolicy",
            (TpuChainSpecDecodeCompilePolicyMixin, Glm5NextVLDecoderTpuCompilePolicy),
            {},
        )
        self.assertTrue(SpecGlm5Policy.spmd_safe)
        with patch(
            "sglang_tpu.experimental.compilation.chain_compile_policy.process_model_config",
            return_value=runner.model_config,
        ):
            policy = SpecGlm5Policy(
                model_runner=runner,
                t_bucketer=Bucketer(name="Token", buckets=[1024, 2048, 4096, 8192]),
                b_bucketer=Bucketer(name="Bs", buckets=[1, 4, 16, 64]),
                owner=runner.model.language_model,
                method_name="forward",
                method=runner.model.language_model.forward,
                name="target/language_model.forward",
                label="_Glm5NextLanguageModel",
            )
        policy.extend_bs = 64
        policy._run_one = MagicMock(return_value=None)
        policy._precompile_mm_embed_placement = MagicMock()

        with patch(
            "sglang_tpu.compilation.spec_decode_compile_policy.get_parallel",
            return_value=SimpleNamespace(tp_rank=1),
        ):
            policy._precompile_extend()
            policy._precompile_extend_full()

        policy._precompile_mm_embed_placement.assert_called_once_with()
        # 4 token buckets * 2 has_spec_info * 1 variant = 8 EXTEND(FULL) graphs (vs 16).
        self.assertEqual(policy._run_one.call_count, 8)
        self.assertTrue(
            all(
                c.kwargs["variant"] == "text_only"
                for c in policy._run_one.call_args_list
            )
        )

        host_padded_info = HostPaddedSamplingInfo.__new__(HostPaddedSamplingInfo)
        plain_info = SamplingBatchInfo.__new__(SamplingBatchInfo)

        extend_batch = SimpleNamespace(
            sampling_info=host_padded_info,
            forward_mode=ForwardMode.EXTEND,
        )
        verify_batch = SimpleNamespace(
            sampling_info=host_padded_info,
            forward_mode=ForwardMode.TARGET_VERIFY,
        )
        plain_batch = SimpleNamespace(
            sampling_info=plain_info,
            forward_mode=ForwardMode.EXTEND,
        )
        self.assertTrue(policy.keeps_padded_mm_embeds(extend_batch))
        self.assertFalse(policy.keeps_padded_mm_embeds(verify_batch))
        self.assertFalse(policy.keeps_padded_mm_embeds(plain_batch))
        self.assertFalse(_make_glm_policy(runner).keeps_padded_mm_embeds(extend_batch))
        self.assertNotIn(
            "keeps_padded_mm_embeds", Glm5NextVLDecoderTpuCompilePolicy.__dict__
        )
        self.assertTrue(
            TpuSpecDecodeCompilePolicyMixin.keeps_padded_mm_embeds(policy, extend_batch)
        )
        self.assertFalse(
            TpuSpecDecodeCompilePolicyMixin.keeps_padded_mm_embeds(policy, verify_batch)
        )

    def test_install_fail_fasts_and_preinitializes_rank_tensors(self):
        with get_parallel().override(pp_size=2):
            policy = _make_glm_policy()
            with self.assertRaisesRegex(ValueError, "pipeline parallelism"):
                policy.install()

        policy_cp = _make_glm_policy()
        with (
            patch(
                "sglang_tpu.compilation.decoder_compile_policy.get_cp_strategy",
                return_value=object(),
            ),
            self.assertRaisesRegex(NotImplementedError, "context parallelism"),
        ):
            policy_cp.install()

        runner_scattered_spmd = _make_runner()
        runner_scattered_spmd.server_args.enable_attn_tp_input_scattered = True
        runner_scattered_spmd.server_args.tpu_compile_tracer = "direct"
        with (
            get_context().override_server_args(**_BASE_CONFIG, use_spmd_compile=True),
            self.assertRaisesRegex(
                ValueError,
                "--enable-attn-tp-input-scattered.*--use-spmd-compile",
            ),
        ):
            _make_glm_policy(runner_scattered_spmd).install()

        runner_scattered_spmd_arg = _make_runner()
        runner_scattered_spmd_arg.server_args.enable_attn_tp_input_scattered = True
        runner_scattered_spmd_arg.server_args.use_spmd_compile = True
        with self.assertRaisesRegex(
            ValueError,
            "--enable-attn-tp-input-scattered.*--use-spmd-compile",
        ):
            _make_glm_policy(runner_scattered_spmd_arg).install()

        runner_scattered_ctx = _make_runner()
        with (
            get_context().override_server_args(
                **_BASE_CONFIG,
                enable_attn_tp_input_scattered=True,
                use_spmd_compile=True,
            ),
            self.assertRaisesRegex(
                ValueError,
                "--enable-attn-tp-input-scattered.*--use-spmd-compile",
            ),
        ):
            _make_glm_policy(runner_scattered_ctx).install()

        runner_direct_spmd = _make_runner()
        runner_direct_spmd.server_args.tpu_compile_tracer = "direct"
        runner_direct_spmd.server_args.enable_attn_tp_input_scattered = False
        policy_direct_spmd = _make_glm_policy(runner_direct_spmd)
        policy_direct_spmd.tracer = "direct"
        with (
            get_context().override_server_args(**_BASE_CONFIG, use_spmd_compile=True),
            patch.object(
                Qwen3VLDecoderTpuCompilePolicy, "install", return_value=True
            ) as super_install,
        ):
            self.assertTrue(policy_direct_spmd.install())
            super_install.assert_called_once_with()

        runner_scattered = _make_runner()
        runner_scattered.server_args.enable_attn_tp_input_scattered = True
        with get_parallel().override(
            tp_size=8,
            attn_tp_size=4,
            attn_dp_size=2,
            attn_cp_size=1,
            moe_ep_size=8,
            moe_dp_size=1,
            moe_tp_size=1,
        ):
            policy_mismatch = _make_glm_policy(runner_scattered)
            with self.assertRaisesRegex(NotImplementedError, "attn_tp_size == tp_size"):
                policy_mismatch.install()

        with get_flags().dp.override(enabled=True):
            policy_dp = _make_glm_policy()
            with self.assertRaisesRegex(NotImplementedError, "DP attention"):
                policy_dp.install()

        saved_rank_tensors = dp_attention._RANK_TENSORS.copy()
        try:
            for tracer in ("dynamo", "direct"):
                with self.subTest(tracer=tracer):
                    dp_attention._RANK_TENSORS = {}
                    runner_ok = _make_runner()
                    runner_ok.server_args.tpu_compile_tracer = tracer
                    runner_ok.server_args.enable_attn_tp_input_scattered = True
                    policy_ok = _make_glm_policy(runner_ok)
                    policy_ok.tracer = tracer
                    with (
                        get_flags().dp.override(enabled=False),
                        get_parallel().override(
                            tp_size=8,
                            tp_rank=3,
                            attn_tp_size=8,
                            attn_tp_rank=3,
                            attn_dp_size=1,
                            attn_dp_rank=0,
                            attn_cp_size=1,
                            attn_cp_rank=0,
                            moe_ep_size=8,
                            moe_ep_rank=3,
                            moe_dp_size=1,
                            moe_dp_rank=0,
                            moe_tp_size=1,
                            moe_tp_rank=0,
                        ),
                        patch.object(
                            Qwen3VLDecoderTpuCompilePolicy, "install", return_value=True
                        ),
                    ):
                        self.assertTrue(policy_ok.install())
                    rank_tensor = dp_attention._RANK_TENSORS[
                        (
                            "attn_tp",
                            torch.device("cpu"),
                        )
                    ]
                    self.assertEqual(int(rank_tensor.item()), 3)
        finally:
            dp_attention._RANK_TENSORS = saved_rank_tensors


if __name__ == "__main__":
    unittest.main()
