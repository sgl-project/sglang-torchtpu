"""CPU-side coverage for TPU distributed exact top-k dispatch and merging.
Tests enforce conservative fallbacks without importing physical-TPU dependencies."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

import importlib
import unittest
from itertools import product
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import torch

from sglang.srt.environ import envs
from sglang.srt.layers import logits_processor as logits_processor_module
from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput
from sglang.srt.layers.vocab_parallel_embedding import (
    VocabParallelEmbeddingShardIndices,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode, LogitsOutputMode
from sglang.srt.model_executor.runner import eager_runner as eager_runner_module
from sglang.srt.model_executor.runner.eager_runner import EagerRunner
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.test_utils import CustomTestCase
from sglang_tpu import sampler as tpu_sampler_module
from sglang_tpu import top_k as tpu_top_k_module
from sglang_tpu.envs import envs as tpu_envs
from sglang_tpu.row_pad import pad_rows, row_bucket
from sglang_tpu.sampler import (
    TpuSampler,
    distributed_top_k_candidate_width,
    distributed_top_k_for_batch,
    sample_from_global_top_k_logits,
    select_logits_output_mode,
)
from sglang_tpu.top_k import (
    PallasDistributedTopKConfig,
    PallasDistributedTopKNegotiation,
    TpuVocabShard,
    distributed_tpu_top_k,
    negotiate_pallas_distributed_top_k_config,
    supports_distributed_tpu_top_k,
    tpu_top_k,
)


def _forward_batch(*, batch_size=1, **info_overrides):
    info = SimpleNamespace(
        sampling_params_on_cpu=False,
        uniform_top_k=50,
        is_any_greedy=False,
        need_top_k_sampling=True,
        need_top_p_sampling=False,
        need_min_p_sampling=False,
        has_custom_logit_processor=False,
        logit_bias=None,
        grammars=None,
        return_sampling_masks=None,
        sampling_mask_batch_indices=None,
        acc_additive_penalties=None,
        acc_scaling_penalties=None,
        penalizer_orchestrator=SimpleNamespace(is_required=False),
        sampling_seed=None,
        top_ks=torch.tensor([50], dtype=torch.int32),
        top_ps=torch.ones(1),
        min_ps=torch.zeros(1),
        temperatures=torch.ones(1, 1),
    )
    for name, value in info_overrides.items():
        setattr(info, name, value)
    mode = SimpleNamespace(
        is_decode=lambda: True,
        is_extend_without_speculative=lambda: False,
    )
    return SimpleNamespace(
        sampling_info=info,
        batch_size=batch_size,
        return_logprob=False,
        spec_algorithm=SpeculativeAlgorithm.NONE,
        forward_mode=mode,
        logits_output_mode=LogitsOutputMode.FULL_VOCAB,
    )


def _distributed_top_k_config(
    *, shard_width=202_048 // 16, world_size=16
) -> PallasDistributedTopKConfig:
    global_vocab_size = shard_width * world_size
    shard = TpuVocabShard(
        width=shard_width,
        global_vocab_size=global_vocab_size,
        org_vocab_size=global_vocab_size,
        org_vocab_start=0,
        org_vocab_count=shard_width,
        org_vocab_padded_count=shard_width,
        added_vocab_start=global_vocab_size,
        added_vocab_count=0,
    )
    return PallasDistributedTopKConfig(
        shard=shard,
        world_size=world_size,
        model_rank=0,
        candidate_width=64,
        output_dtype=torch.bfloat16,
    )


def _distributed_top_k_model(shard_width=202_048 // 16):
    processor = SimpleNamespace(
        use_tpu_distributed_top_k=True,
        do_tensor_parallel_all_gather=True,
        do_tensor_parallel_all_gather_dp_attn=False,
    )
    lm_head = SimpleNamespace(
        enable_tp=True,
        num_embeddings=202_048,
        org_vocab_size=202_048,
        num_embeddings_per_partition=shard_width,
        weight=torch.empty(0, dtype=torch.bfloat16),
        shard_indices=object(),
    )
    config = _distributed_top_k_config(shard_width=shard_width)
    tpu_sampler_module._DISTRIBUTED_TOP_K_STATES[id(processor)] = (
        tpu_sampler_module._DistributedTopKState(
            logits_processor=processor,
            lm_head=lm_head,
            config=config,
            fallback_reason=None,
        )
    )
    return SimpleNamespace(logits_processor=processor, lm_head=lm_head)


class TestDistributedTopKEligibility(CustomTestCase):
    def test_eager_runner_selects_logits_output_mode_only_on_tpu(self):
        model = _distributed_top_k_model()
        runner = object.__new__(EagerRunner)
        runner.model_runner = SimpleNamespace(model=model)
        batch = _forward_batch(batch_size=64)

        with patch.object(eager_runner_module, "_is_tpu", True):
            runner._set_logits_output_mode(batch)
            self.assertIs(
                batch.logits_output_mode,
                LogitsOutputMode.DISTRIBUTED_TOP_K,
            )

            batch.sampling_info.uniform_top_k = None
            runner._set_logits_output_mode(batch)
            self.assertIs(
                batch.logits_output_mode,
                LogitsOutputMode.FULL_VOCAB,
            )

        batch.logits_output_mode = LogitsOutputMode.DISTRIBUTED_TOP_K
        with patch.object(eager_runner_module, "_is_tpu", False):
            runner._set_logits_output_mode(batch)
        self.assertIs(batch.logits_output_mode, LogitsOutputMode.DISTRIBUTED_TOP_K)

    def test_candidate_width_is_static_across_requested_top_k(self):
        model = _distributed_top_k_model()
        processor, lm_head = model.logits_processor, model.lm_head

        with patch.object(tpu_sampler_module, "_DISTRIBUTED_TOP_K_MIN_BATCH_SIZE", 1):
            self.assertEqual(distributed_top_k_candidate_width(processor, lm_head), 64)
            for top_k in (2, 7, 50, 64):
                with self.subTest(top_k=top_k):
                    batch = _forward_batch(
                        batch_size=1,
                        uniform_top_k=top_k,
                        top_ks=torch.tensor([top_k], dtype=torch.int32),
                    )
                    mode = select_logits_output_mode(batch, model)
                    self.assertIs(mode, LogitsOutputMode.DISTRIBUTED_TOP_K)

            oversized_batch = _forward_batch(batch_size=1, uniform_top_k=65)
            self.assertEqual(
                distributed_top_k_for_batch(oversized_batch, sampling_backend="tpu"),
                65,
            )
            self.assertIs(
                select_logits_output_mode(oversized_batch, model),
                LogitsOutputMode.FULL_VOCAB,
            )

    def test_cached_candidate_width_is_fullgraph_safe(self):
        model = _distributed_top_k_model()
        processor, lm_head = model.logits_processor, model.lm_head

        compiled = torch.compile(
            lambda value: value + distributed_top_k_candidate_width(processor, lm_head),
            backend="eager",
            fullgraph=True,
        )

        torch.testing.assert_close(compiled(torch.tensor(1)), torch.tensor(65))

    def test_candidate_contract_is_negotiated_lazily_once(self):
        config = _distributed_top_k_config(shard_width=32, world_size=8)
        processor = SimpleNamespace(
            use_tpu_distributed_top_k=True,
            do_tensor_parallel_all_gather=True,
            do_tensor_parallel_all_gather_dp_attn=False,
            use_attn_tp_group=False,
        )
        lm_head = SimpleNamespace(weight=torch.empty(1, dtype=torch.bfloat16))
        group = SimpleNamespace(world_size=8, rank_in_group=0, ranks=list(range(8)))

        with (
            patch(
                "sglang.srt.distributed.model_parallel_is_initialized",
                return_value=True,
            ),
            patch(
                "sglang_tpu.sampler.get_parallel",
                return_value=SimpleNamespace(tp_group=group, attn_tp_group=group),
            ) as get_group,
            patch.object(
                tpu_top_k_module,
                "negotiate_pallas_distributed_top_k_config",
                return_value=PallasDistributedTopKNegotiation(config, None),
            ) as negotiate,
            patch(
                "sglang_tpu.distributed_topk_pallas."
                "register_pallas_distributed_top_k_op"
            ) as register,
        ):
            self.assertEqual(distributed_top_k_candidate_width(processor, lm_head), 64)
            self.assertEqual(distributed_top_k_candidate_width(processor, lm_head), 64)

        get_group.assert_called_once()
        negotiate.assert_called_once_with(
            lm_head,
            group=group,
            expected_dtype=torch.bfloat16,
            local_fallback_reason=None,
        )
        register.assert_called_once_with(8)

    def test_candidate_getter_waits_for_model_parallel_initialization(self):
        processor = SimpleNamespace(
            use_tpu_distributed_top_k=True,
            do_tensor_parallel_all_gather=True,
            do_tensor_parallel_all_gather_dp_attn=False,
        )
        lm_head = SimpleNamespace(weight=torch.empty(1, dtype=torch.bfloat16))

        with (
            patch(
                "sglang.srt.distributed.model_parallel_is_initialized",
                return_value=False,
            ),
            patch.object(
                tpu_top_k_module, "negotiate_pallas_distributed_top_k_config"
            ) as negotiate,
        ):
            self.assertIsNone(distributed_top_k_candidate_width(processor, lm_head))

        negotiate.assert_not_called()

    def test_candidate_contract_rejects_reinitialization_for_a_new_head(self):
        config = _distributed_top_k_config(shard_width=32, world_size=8)
        processor = SimpleNamespace(
            use_tpu_distributed_top_k=True,
            do_tensor_parallel_all_gather=True,
            do_tensor_parallel_all_gather_dp_attn=False,
            use_attn_tp_group=False,
        )
        model = SimpleNamespace(
            logits_processor=processor,
            lm_head=SimpleNamespace(weight=torch.empty(1, dtype=torch.bfloat16)),
        )
        group = SimpleNamespace(world_size=8, rank_in_group=0, ranks=list(range(8)))

        with (
            patch(
                "sglang.srt.distributed.model_parallel_is_initialized",
                return_value=True,
            ),
            patch(
                "sglang_tpu.sampler.get_parallel",
                return_value=SimpleNamespace(tp_group=group, attn_tp_group=group),
            ),
            patch.object(
                tpu_top_k_module,
                "negotiate_pallas_distributed_top_k_config",
                return_value=PallasDistributedTopKNegotiation(config, None),
            ),
            patch(
                "sglang_tpu.distributed_topk_pallas."
                "register_pallas_distributed_top_k_op"
            ),
        ):
            distributed_top_k_candidate_width(processor, model.lm_head)
            model.lm_head = SimpleNamespace(weight=torch.empty(1, dtype=torch.bfloat16))
            with self.assertRaisesRegex(RuntimeError, "cannot be reinitialized"):
                distributed_top_k_candidate_width(processor, model.lm_head)

    def test_measured_default_uses_distributed_top_k_from_batch_64(self):
        cutoff = envs.SGLANG_TPU_DISTRIBUTED_TOP_K_MIN_BATCH_SIZE
        self.assertEqual(cutoff.default, 64)

        with patch.object(tpu_sampler_module, "_DISTRIBUTED_TOP_K_MIN_BATCH_SIZE", 64):
            self.assertIsNone(
                distributed_top_k_for_batch(
                    _forward_batch(batch_size=63), sampling_backend="tpu"
                )
            )
            self.assertEqual(
                distributed_top_k_for_batch(
                    _forward_batch(batch_size=64), sampling_backend="tpu"
                ),
                50,
            )

    def test_explicitly_skipped_tensor_parallel_gather_falls_back(self):
        processor = SimpleNamespace(
            use_tpu_distributed_top_k=True,
            do_tensor_parallel_all_gather=False,
            do_tensor_parallel_all_gather_dp_attn=False,
        )
        sharded_lm_head = SimpleNamespace(
            enable_tp=True,
            num_embeddings=128,
            org_vocab_size=128,
            num_embeddings_per_partition=16,
            shard_indices=object(),
        )

        model = SimpleNamespace(logits_processor=processor, lm_head=sharded_lm_head)
        self.assertIs(
            select_logits_output_mode(_forward_batch(), model),
            LogitsOutputMode.FULL_VOCAB,
        )

    def test_full_vocab_features_fall_back(self):
        cases = {
            "mixed top-k": {"uniform_top_k": None},
            "greedy row": {"is_any_greedy": True},
            "seeded sampling": {"sampling_seed": torch.tensor([17])},
            "top-p": {"need_top_p_sampling": True},
            "min-p": {"need_min_p_sampling": True},
            "custom processor": {"has_custom_logit_processor": True},
            "logit bias": {"logit_bias": torch.zeros(1, 8)},
            "grammar": {"grammars": [object()]},
            "sampling mask": {"sampling_mask_batch_indices": torch.tensor([0])},
            "additive penalty": {"acc_additive_penalties": torch.zeros(1, 8)},
            "scaling penalty": {"acc_scaling_penalties": torch.ones(1, 8)},
            "live penalizer": {
                "penalizer_orchestrator": SimpleNamespace(is_required=True)
            },
        }
        for name, overrides in cases.items():
            with self.subTest(name=name):
                self.assertIsNone(
                    distributed_top_k_for_batch(
                        _forward_batch(**overrides), sampling_backend="tpu"
                    )
                )

    def test_non_tpu_and_logprob_batches_fall_back(self):
        batch = _forward_batch()
        self.assertIsNone(
            distributed_top_k_for_batch(batch, sampling_backend="pytorch")
        )
        batch.return_logprob = True
        self.assertIsNone(distributed_top_k_for_batch(batch, sampling_backend="tpu"))

    def test_speculative_batch_falls_back(self):
        batch = _forward_batch()
        batch.spec_algorithm = SpeculativeAlgorithm.EAGLE

        self.assertIsNone(distributed_top_k_for_batch(batch, sampling_backend="tpu"))

    def test_speculative_batch_does_not_inspect_sampling_info(self):
        class SpecBatch:
            spec_algorithm = SpeculativeAlgorithm.EAGLE3

            @property
            def sampling_info(self):
                raise AssertionError("speculative TopK gate read sampling_info")

        self.assertIsNone(
            distributed_top_k_for_batch(SpecBatch(), sampling_backend="tpu")
        )
        with patch.object(
            tpu_sampler_module, "distributed_top_k_candidate_width_for_model"
        ) as candidate_width:
            self.assertIs(
                select_logits_output_mode(SpecBatch(), object()),
                LogitsOutputMode.FULL_VOCAB,
            )
        candidate_width.assert_not_called()

    def test_draft_batch_does_not_inspect_sampling_state(self):
        class DraftBatch:
            spec_info = object()

            @property
            def spec_algorithm(self):
                raise AssertionError("draft TopK gate read spec_algorithm")

            @property
            def sampling_info(self):
                raise AssertionError("draft TopK gate read sampling_info")

        self.assertIsNone(
            distributed_top_k_for_batch(DraftBatch(), sampling_backend="tpu")
        )
        with patch.object(
            tpu_sampler_module, "distributed_top_k_candidate_width_for_model"
        ) as candidate_width:
            self.assertIs(
                select_logits_output_mode(DraftBatch(), object()),
                LogitsOutputMode.FULL_VOCAB,
            )
        candidate_width.assert_not_called()

    def test_non_speculative_selection_keeps_capability_negotiation_order(self):
        batch = _forward_batch(batch_size=64)
        model = object()
        calls = MagicMock()
        with (
            patch.object(
                tpu_sampler_module,
                "distributed_top_k_candidate_width_for_model",
            ) as candidate_width,
            patch.object(
                tpu_sampler_module, "distributed_top_k_for_batch"
            ) as requested_top_k,
        ):
            calls.attach_mock(candidate_width, "candidate_width")
            calls.attach_mock(requested_top_k, "requested_top_k")
            for width, requested, expected in (
                (None, 50, LogitsOutputMode.FULL_VOCAB),
                (64, None, LogitsOutputMode.FULL_VOCAB),
                (64, 65, LogitsOutputMode.FULL_VOCAB),
                (64, 50, LogitsOutputMode.DISTRIBUTED_TOP_K),
            ):
                with self.subTest(width=width, requested=requested):
                    calls.reset_mock()
                    candidate_width.return_value = width
                    requested_top_k.return_value = requested
                    self.assertIs(select_logits_output_mode(batch, model), expected)
                    expected_calls = [call.candidate_width(model)]
                    if width is not None:
                        expected_calls.append(
                            call.requested_top_k(batch, sampling_backend="tpu")
                        )
                    self.assertEqual(calls.mock_calls, expected_calls)

    def test_environment_can_raise_the_pre_gather_cutoff(self):
        cutoff = envs.SGLANG_TPU_DISTRIBUTED_TOP_K_MIN_BATCH_SIZE
        self.assertEqual(cutoff.default, 64)
        try:
            with cutoff.override(128):
                importlib.reload(tpu_sampler_module)

            self.assertEqual(
                tpu_sampler_module._DISTRIBUTED_TOP_K_MIN_BATCH_SIZE,
                128,
            )
            self.assertIsNone(
                tpu_sampler_module.distributed_top_k_for_batch(
                    _forward_batch(batch_size=127), sampling_backend="tpu"
                )
            )
            padded_batch = _forward_batch(batch_size=128)
            padded_batch._tpu_real_b = 127
            self.assertEqual(
                tpu_sampler_module.distributed_top_k_for_batch(
                    padded_batch, sampling_backend="tpu"
                ),
                50,
            )
            self.assertEqual(
                tpu_sampler_module.distributed_top_k_for_batch(
                    _forward_batch(batch_size=128), sampling_backend="tpu"
                ),
                50,
            )
        finally:
            # Restore the process-start configuration for the remaining tests.
            importlib.reload(tpu_sampler_module)

    def test_model_requiring_full_logits_disables_global_top_k(self):
        execution = SimpleNamespace(
            features=SimpleNamespace(enable_fp32_lm_head=False, enable_mis=False),
            deterministic=SimpleNamespace(rl_on_policy_target=None),
            kernel=SimpleNamespace(sampling_backend="tpu"),
        )
        parallel = SimpleNamespace(
            enable_dp_lm_head=False,
            enable_tp_lm_head_all_to_all=False,
            tp_size=8,
            attn_dp_size=1,
            tp_group=SimpleNamespace(cpu_group=None),
        )
        config = SimpleNamespace(vocab_size=128)

        with (
            patch.object(logits_processor_module, "_is_tpu", True),
            patch.object(logits_processor_module, "get_exec", return_value=execution),
            patch.object(
                logits_processor_module, "get_parallel", return_value=parallel
            ),
            patch.object(
                logits_processor_module.triton_symm_mem_ag,
                "recommended_max_tokens",
                return_value=128,
            ),
            patch.object(
                logits_processor_module.triton_symm_mem_ag,
                "MultimemAllGatherer",
                return_value=object(),
            ),
        ):
            processor = LogitsProcessor(config, requires_full_next_token_logits=True)

        self.assertFalse(processor.use_tpu_distributed_top_k)


class TestTpuSamplerGlobalTopK(CustomTestCase):
    def test_host_padding_keeps_candidate_width_and_exact_requested_support(self):
        probs = torch.arange(128, 0, -1, dtype=torch.float32).reshape(1, -1)
        probs /= probs.sum()
        for k in (2, 20, 50, 64):
            with self.subTest(k=k):
                info = _forward_batch(uniform_top_k=k).sampling_info
                info.top_ks = torch.tensor([k], dtype=torch.int32)
                observed = []

                def sample(weights, **kwargs):
                    observed.append(weights)
                    return weights.argmax(-1).to(torch.int32)

                with (
                    tpu_envs.SGLANG_TPU_HOST_PADDING.override(True),
                    patch("sglang_tpu.top_k.tpu_top_k", side_effect=torch.topk),
                    patch.object(
                        tpu_sampler_module, "sampling_from_probs_torch", sample
                    ),
                ):
                    result, _ = TpuSampler._sample_from_probs(
                        SimpleNamespace(), probs, info, torch.zeros(1), False
                    )
                self.assertEqual(observed[0].shape, (1, 64))
                torch.testing.assert_close(observed[0][:, :k], probs[:, :k])
                self.assertEqual(torch.count_nonzero(observed[0][:, k:]).item(), 0)
                self.assertEqual(result.tolist(), [0])

    def test_global_top_k_outputs_are_paired_and_exclusive(self):
        with self.assertRaisesRegex(ValueError, "must be provided together"):
            LogitsProcessorOutput(
                next_token_logits=None,
                next_token_global_top_k_logits=torch.ones(1, 2),
            )
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            LogitsProcessorOutput(
                next_token_logits=torch.ones(1, 8),
                next_token_global_top_k_logits=torch.ones(1, 2),
                next_token_global_top_k_token_ids=torch.ones(1, 2, dtype=torch.int32),
            )

    def test_global_top_k_position_maps_to_global_token_id(self):
        sampler = SimpleNamespace(_sync_token_ids_across_tp=MagicMock())
        output = LogitsProcessorOutput(
            next_token_logits=None,
            next_token_global_top_k_logits=torch.tensor([[3.0, 2.0, 1.0]]),
            next_token_global_top_k_token_ids=torch.tensor(
                [[91, 17, 8]], dtype=torch.int32
            ),
        )
        info = _forward_batch().sampling_info

        with patch.object(
            tpu_sampler_module,
            "sample_from_global_top_k_logits",
            return_value=torch.tensor([1], dtype=torch.int32),
        ):
            actual = TpuSampler.forward(
                sampler,
                output,
                info,
                False,
                [0],
                [[]],
                torch.tensor([7]),
            )

        torch.testing.assert_close(actual, torch.tensor([17], dtype=torch.int64))
        sampler._sync_token_ids_across_tp.assert_called_once()

    def test_global_top_k_sampling_normalizes_only_top_k(self):
        global_top_k_logits = torch.tensor([[4.0, 2.0]])
        with patch.object(
            tpu_sampler_module,
            "sampling_from_probs_torch",
            return_value=torch.tensor([1], dtype=torch.int32),
        ) as sample:
            actual = sample_from_global_top_k_logits(
                global_top_k_logits,
                torch.tensor([[2.0]]),
            )

        torch.testing.assert_close(actual, torch.tensor([1], dtype=torch.int32))
        torch.testing.assert_close(
            sample.call_args.args[0], torch.softmax(torch.tensor([[2.0, 1.0]]), -1)
        )

    def test_global_top_k_sampling_masks_candidates_above_requested_k(self):
        logits = torch.tensor([[4.0, 3.0, 2.0]])
        with patch.object(
            tpu_sampler_module,
            "sampling_from_probs_torch",
            return_value=torch.tensor([0], dtype=torch.int32),
        ) as sample:
            sample_from_global_top_k_logits(
                logits,
                torch.ones(1, 1),
                torch.tensor([2], dtype=torch.int32),
            )

        probs = sample.call_args.args[0]
        self.assertEqual(probs[0, 2].item(), 0)
        torch.testing.assert_close(probs.sum(dim=1), torch.ones(1))

    def test_fallback_uses_generic_pytorch_sampler(self):
        sampler = SimpleNamespace()
        probs = torch.rand(1, 128)
        positions = torch.tensor([7])
        info = _forward_batch(uniform_top_k=None).sampling_info
        expected = torch.tensor([17], dtype=torch.int32)

        with patch.object(
            tpu_sampler_module,
            "top_k_top_p_min_p_sampling_from_probs_torch",
            return_value=expected,
        ) as generic_sampler:
            actual, capture = TpuSampler._sample_from_probs(
                sampler, probs, info, positions, simple_sampling_case=False
            )

        self.assertIs(actual, expected)
        self.assertIsNone(capture)
        generic_sampler.assert_called_once_with(
            probs,
            info.top_ks,
            info.top_ps,
            info.min_ps,
            False,
            None,
            positions,
            return_filtered_probs=False,
        )

    def test_simple_sampling_preserves_base_return_contract(self):
        """A stochastic TPU request must return tokens plus optional mask state.

        The upstream sampler began unpacking this pair, while the TPU override
        still returned only tokens and crashed ordinary temperature sampling.
        """
        sampler = SimpleNamespace()
        probs = torch.rand(1, 128)
        positions = torch.tensor([7])
        info = _forward_batch(uniform_top_k=None).sampling_info
        expected = torch.tensor([17], dtype=torch.int32)

        with patch.object(
            tpu_sampler_module,
            "sampling_from_probs_torch",
            return_value=expected,
        ):
            actual, capture = TpuSampler._sample_from_probs(
                sampler, probs, info, positions, simple_sampling_case=True
            )

        self.assertIs(actual, expected)
        self.assertIsNone(capture)

    def test_native_top_k_captures_requested_sampling_mask_rows(self):
        sampler = SimpleNamespace()
        probs = torch.rand(2, 128)
        positions = torch.tensor([7, 8])
        info = _forward_batch(
            batch_size=2,
            uniform_top_k=2,
            sampling_mask_batch_indices=torch.tensor([1]),
            top_ks=torch.tensor([2, 2], dtype=torch.int32),
            top_ps=torch.ones(2),
            min_ps=torch.zeros(2),
            temperatures=torch.ones(2, 1),
        ).sampling_info
        global_top_k_probs = torch.tensor([[0.7, 0.3], [0.6, 0.4]])
        global_top_k_token_ids = torch.tensor([[10, 11], [20, 21]], dtype=torch.int32)

        with (
            patch(
                "sglang_tpu.top_k.tpu_top_k",
                return_value=(global_top_k_probs, global_top_k_token_ids),
            ),
            patch.object(
                tpu_sampler_module,
                "sampling_from_probs_torch",
                return_value=torch.tensor([1, 0], dtype=torch.int32),
            ),
        ):
            actual, capture = TpuSampler._sample_from_probs(
                sampler, probs, info, positions, simple_sampling_case=False
            )

        torch.testing.assert_close(actual, torch.tensor([11, 20], dtype=torch.int32))
        self.assertIsNotNone(capture)
        torch.testing.assert_close(capture.weights, torch.tensor([[0.6, 0.4]]))
        torch.testing.assert_close(
            capture.token_ids, torch.tensor([[20, 21]], dtype=torch.int32)
        )
        torch.testing.assert_close(capture.selected_weight, torch.tensor([0.6]))
        torch.testing.assert_close(capture.batch_rows, torch.tensor([1]))

    def test_filtered_sampling_uses_bsearch_mask_on_tpu(self):
        """On TPU tensors, filtered sampling routes through the sort-free
        binary-search mask + exponential-race draw instead of the torch sort path.
        The device gate is patched because CPU CI cannot allocate TPU tensors."""
        sampler = SimpleNamespace()
        probs = torch.rand(1, 128)
        positions = torch.tensor([7])
        info = _forward_batch(uniform_top_k=None).sampling_info
        filtered = torch.rand(1, 128)
        expected = torch.tensor([17], dtype=torch.int32)

        with (
            patch.object(
                tpu_sampler_module, "_use_bsearch_filtering", return_value=True
            ),
            patch(
                "sglang_tpu.sampling_mask.tpu_filter_probs",
                return_value=filtered,
            ) as filter_probs,
            patch.object(
                tpu_sampler_module,
                "sample_exponential_race",
                return_value=expected,
            ) as sample,
            patch.object(
                tpu_sampler_module,
                "top_k_top_p_min_p_sampling_from_probs_torch",
            ) as generic_sampler,
        ):
            actual, capture = TpuSampler._sample_from_probs(
                sampler, probs, info, positions, simple_sampling_case=False
            )

        self.assertIs(actual, expected)
        self.assertIsNone(capture)
        generic_sampler.assert_not_called()
        filter_probs.assert_called_once_with(
            probs,
            info.top_ks,
            info.top_ps,
            info.min_ps,
            need_top_k=info.need_top_k_sampling,
            need_top_p=info.need_top_p_sampling,
            need_min_p=info.need_min_p_sampling,
        )
        sample.assert_called_once_with(filtered)

    def test_uniform_top_k_with_top_p_filters_the_gathered_block(self):
        """With the sort-free mask available, uniform top-k plus top-p/min-p
        keeps the native-TopK fast path: the block of k gathered columns is
        filtered in place (need_top_k=False — gathering already applied it)
        instead of paying ~30 full-vocabulary reductions."""
        sampler = SimpleNamespace()
        probs = torch.rand(1, 128)
        positions = torch.tensor([7])
        info = _forward_batch(need_top_p_sampling=True).sampling_info
        global_top_k_probs = torch.tensor([[0.7, 0.3]])
        filtered_block = torch.tensor([[0.7, 0.0]])
        global_top_k_token_ids = torch.tensor([[91, 17]], dtype=torch.int32)

        with (
            patch.object(
                tpu_sampler_module, "_use_bsearch_filtering", return_value=True
            ),
            patch(
                "sglang_tpu.top_k.tpu_top_k",
                return_value=(global_top_k_probs, global_top_k_token_ids),
            ) as top_k,
            patch(
                "sglang_tpu.sampling_mask.tpu_filter_probs",
                return_value=filtered_block,
            ) as filter_probs,
            patch.object(
                tpu_sampler_module,
                "sampling_from_probs_torch",
                return_value=torch.tensor([0], dtype=torch.int32),
            ) as sample,
        ):
            actual, capture = TpuSampler._sample_from_probs(
                sampler, probs, info, positions, simple_sampling_case=False
            )

        torch.testing.assert_close(actual, torch.tensor([91], dtype=torch.int32))
        self.assertIsNone(capture)
        top_k.assert_called_once_with(probs, 50)
        filter_probs.assert_called_once_with(
            global_top_k_probs,
            info.top_ks,
            info.top_ps,
            info.min_ps,
            need_top_k=False,
            need_top_p=True,
            need_min_p=False,
        )
        sample.assert_called_once_with(
            filtered_block,
            unseeded_sampling_fn=tpu_sampler_module.sample_exponential_race,
        )

    def test_kill_switch_restores_the_torch_reference_path(self):
        """With bsearch filtering disabled, uniform top-k plus top-p must NOT
        take the native-TopK fast path (its block filter would be
        unavailable) and must fall through to the torch sort reference —
        the kill switch restores pre-PR behavior exactly."""
        sampler = SimpleNamespace()
        probs = torch.rand(1, 128)
        positions = torch.tensor([7])
        info = _forward_batch(need_top_p_sampling=True).sampling_info
        expected = torch.tensor([17], dtype=torch.int32)

        with (
            patch.object(
                tpu_sampler_module, "_use_bsearch_filtering", return_value=False
            ),
            patch("sglang_tpu.top_k.tpu_top_k") as top_k,
            patch.object(
                tpu_sampler_module,
                "top_k_top_p_min_p_sampling_from_probs_torch",
                return_value=expected,
            ) as generic_sampler,
        ):
            actual, capture = TpuSampler._sample_from_probs(
                sampler, probs, info, positions, simple_sampling_case=False
            )

        self.assertIs(actual, expected)
        self.assertIsNone(capture)
        top_k.assert_not_called()
        generic_sampler.assert_called_once()

    def test_uniform_top_k_uses_post_gather_native_top_k(self):
        sampler = SimpleNamespace()
        probs = torch.rand(1, 128)
        positions = torch.tensor([7])
        info = _forward_batch().sampling_info
        global_top_k_probs = torch.tensor([[0.7, 0.3]])
        global_top_k_token_ids = torch.tensor([[91, 17]], dtype=torch.int32)

        with (
            patch(
                "sglang_tpu.top_k.tpu_top_k",
                return_value=(global_top_k_probs, global_top_k_token_ids),
            ) as top_k,
            patch.object(
                tpu_sampler_module,
                "sampling_from_probs_torch",
                return_value=torch.tensor([1], dtype=torch.int32),
            ) as sample,
        ):
            actual, capture = TpuSampler._sample_from_probs(
                sampler, probs, info, positions, simple_sampling_case=False
            )

        torch.testing.assert_close(actual, torch.tensor([17], dtype=torch.int32))
        self.assertIsNone(capture)
        top_k.assert_called_once_with(probs, 50)
        sample.assert_called_once_with(
            global_top_k_probs,
            unseeded_sampling_fn=tpu_sampler_module.sample_exponential_race,
        )


class TestTpuTopKLayout(CustomTestCase):
    @staticmethod
    def _canonical_shard(rank, *, world_size=8, org_vocab_size=1000):
        org_padded_count = 128
        added_width = 8
        global_vocab_size = 1016
        org_start = min(rank * org_padded_count, org_vocab_size)
        org_end = min((rank + 1) * org_padded_count, org_vocab_size)
        added_start = min(
            org_vocab_size + rank * added_width,
            global_vocab_size,
        )
        added_end = min(
            org_vocab_size + (rank + 1) * added_width,
            global_vocab_size,
        )
        return TpuVocabShard(
            width=org_padded_count + added_width,
            global_vocab_size=global_vocab_size,
            org_vocab_size=org_vocab_size,
            org_vocab_start=org_start,
            org_vocab_count=org_end - org_start,
            org_vocab_padded_count=org_padded_count,
            added_vocab_start=added_start,
            added_vocab_count=added_end - added_start,
        )

    def _capability_group(self, *, world_size=8, mutate=None, descriptor_mutate=None):
        def all_gather(local):
            descriptors = []
            for rank in range(world_size):
                shard = (
                    local.shard
                    if rank == 0
                    else self._canonical_shard(rank, world_size=world_size)
                )
                if mutate is not None:
                    shard = mutate(rank, shard)
                descriptor = type(local)(
                    rank=local.rank if rank == 0 else rank,
                    shard=shard,
                    output_dtype=local.output_dtype,
                    candidate_width=local.candidate_width,
                    head_tp_size=local.head_tp_size,
                    fallback_reason=local.fallback_reason,
                )
                if descriptor_mutate is not None:
                    descriptor = descriptor_mutate(rank, descriptor)
                descriptors.append(descriptor)
            return descriptors

        return SimpleNamespace(
            world_size=world_size,
            rank_in_group=0,
            ranks=list(range(world_size)),
            all_gather_object=MagicMock(side_effect=all_gather),
        )

    def _capability_lm_head(self, *, world_size=8):
        shard = self._canonical_shard(0, world_size=world_size)
        indices = SimpleNamespace(
            org_vocab_start_index=shard.org_vocab_start,
            num_org_elements=shard.org_vocab_count,
            num_org_elements_padded=shard.org_vocab_padded_count,
            added_vocab_start_index=shard.added_vocab_start,
            num_added_elements=shard.added_vocab_count,
        )
        return SimpleNamespace(
            enable_tp=True,
            num_embeddings=shard.global_vocab_size,
            org_vocab_size=shard.org_vocab_size,
            tp_size=world_size,
            num_embeddings_per_partition=shard.width,
            shard_indices=indices,
        )

    def test_capability_requires_rank_consensus(self):
        group = self._capability_group()
        with patch.object(tpu_top_k_module, "_group_spans_world", return_value=True):
            config = negotiate_pallas_distributed_top_k_config(
                self._capability_lm_head(),
                group=group,
                expected_dtype=torch.bfloat16,
            ).config

        self.assertIsNotNone(config)
        self.assertEqual(config.candidate_width, 64)
        self.assertEqual(config.world_size, 8)
        group.all_gather_object.assert_called_once()

    def test_tp16_capability_is_supported(self):
        group = self._capability_group(world_size=16)
        with patch.object(tpu_top_k_module, "_group_spans_world", return_value=True):
            config = negotiate_pallas_distributed_top_k_config(
                self._capability_lm_head(world_size=16),
                group=group,
                expected_dtype=torch.float32,
            ).config

        self.assertIsNotNone(config)
        self.assertEqual(config.world_size, 16)
        self.assertEqual(config.output_dtype, torch.float32)

    def test_subgroup_is_rejected_before_negotiation(self):
        group = self._capability_group()
        with patch.object(tpu_top_k_module, "_group_spans_world", return_value=False):
            config = negotiate_pallas_distributed_top_k_config(
                self._capability_lm_head(),
                group=group,
                expected_dtype=torch.bfloat16,
            ).config

        self.assertIsNone(config)
        group.all_gather_object.assert_not_called()

    def test_malformed_local_rank_falls_back_after_consensus(self):
        group = self._capability_group()
        group.rank_in_group = None
        with patch.object(tpu_top_k_module, "_group_spans_world", return_value=True):
            config = negotiate_pallas_distributed_top_k_config(
                self._capability_lm_head(),
                group=group,
                expected_dtype=torch.bfloat16,
            ).config

        self.assertIsNone(config)
        group.all_gather_object.assert_called_once()

    def test_duplicate_rank_falls_back_after_consensus(self):
        group = self._capability_group(
            descriptor_mutate=lambda rank, descriptor: (
                descriptor._replace(rank=0) if rank == 7 else descriptor
            )
        )
        with patch.object(tpu_top_k_module, "_group_spans_world", return_value=True):
            config = negotiate_pallas_distributed_top_k_config(
                self._capability_lm_head(),
                group=group,
                expected_dtype=torch.bfloat16,
            ).config

        self.assertIsNone(config)
        group.all_gather_object.assert_called_once()

    def test_one_rank_metadata_disagreement_falls_back_everywhere(self):
        def mutate(rank, shard):
            if rank != 3:
                return shard
            return shard._replace(global_vocab_size=shard.global_vocab_size + 1)

        group = self._capability_group(mutate=mutate)
        with patch.object(tpu_top_k_module, "_group_spans_world", return_value=True):
            config = negotiate_pallas_distributed_top_k_config(
                self._capability_lm_head(),
                group=group,
                expected_dtype=torch.bfloat16,
            ).config
        self.assertIsNone(config)

    def test_consumer_ineligibility_participates_in_rank_consensus(self):
        group = self._capability_group()
        with patch.object(tpu_top_k_module, "_group_spans_world", return_value=True):
            result = negotiate_pallas_distributed_top_k_config(
                self._capability_lm_head(),
                group=group,
                expected_dtype=torch.bfloat16,
                local_fallback_reason="unsupported logits gather",
            )

        self.assertIsNone(result.config)
        self.assertEqual(result.fallback_reason, "rank 0: unsupported logits gather")
        group.all_gather_object.assert_called_once()

    def test_unsupported_dtype_falls_back_after_consensus(self):
        group = self._capability_group()
        with patch.object(tpu_top_k_module, "_group_spans_world", return_value=True):
            config = negotiate_pallas_distributed_top_k_config(
                self._capability_lm_head(),
                group=group,
                expected_dtype=torch.float16,
            ).config
        self.assertIsNone(config)
        group.all_gather_object.assert_called_once()

    def test_consensus_transport_failure_is_fatal(self):
        group = self._capability_group()
        group.all_gather_object.side_effect = RuntimeError("collective failed")
        with (
            patch.object(tpu_top_k_module, "_group_spans_world", return_value=True),
            self.assertRaisesRegex(RuntimeError, "collective failed"),
        ):
            negotiate_pallas_distributed_top_k_config(
                self._capability_lm_head(),
                group=group,
                expected_dtype=torch.bfloat16,
            )

    def test_distributed_path_requires_a_sharded_lm_head(self):
        layout = SimpleNamespace(
            enable_tp=True,
            num_embeddings=128,
            org_vocab_size=128,
            num_embeddings_per_partition=16,
            shard_indices=object(),
        )
        self.assertTrue(supports_distributed_tpu_top_k(layout, 16))

        layout.enable_tp = False
        self.assertFalse(supports_distributed_tpu_top_k(layout, 16))
        with self.assertRaisesRegex(ValueError, "vocab-sharded LM head"):
            TpuVocabShard.from_lm_head(layout)

    def test_padding_and_added_vocab_are_mapped(self):
        shard = VocabParallelEmbeddingShardIndices(
            padded_org_vocab_start_index=8,
            padded_org_vocab_end_index=16,
            padded_added_vocab_start_index=20,
            padded_added_vocab_end_index=24,
            org_vocab_start_index=8,
            org_vocab_end_index=14,
            added_vocab_start_index=20,
            added_vocab_end_index=22,
        )
        lm_head = SimpleNamespace(
            enable_tp=True,
            shard_indices=shard,
            num_embeddings=22,
            org_vocab_size=20,
            num_embeddings_per_partition=12,
        )

        self.assertEqual(
            TpuVocabShard.from_lm_head(lm_head),
            TpuVocabShard(
                width=12,
                global_vocab_size=22,
                org_vocab_size=20,
                org_vocab_start=8,
                org_vocab_count=6,
                org_vocab_padded_count=8,
                added_vocab_start=20,
                added_vocab_count=2,
            ),
        )

    def test_primitive_rejects_cpu_tensor_before_tpu_dependencies_load(self):
        with self.assertRaisesRegex(ValueError, "requires a TPU tensor"):
            tpu_top_k(torch.ones(1, 8), 4)

    def test_empty_distributed_input_bypasses_the_pallas_grid(self):
        shard = self._canonical_shard(0)
        group = SimpleNamespace(world_size=8, rank_in_group=0)

        class EmptyTpuTensor:
            device = SimpleNamespace(type="tpu")
            ndim = 2
            shape = (0, shard.width)
            dtype = torch.bfloat16

            @staticmethod
            def new_empty(shape, *, dtype):
                return torch.empty(shape, dtype=dtype)

        values, ids = distributed_tpu_top_k(EmptyTpuTensor(), shard, 64, group=group)
        self.assertEqual(values.shape, (0, 64))
        self.assertEqual(values.dtype, torch.float32)
        self.assertEqual(ids.shape, (0, 64))
        self.assertEqual(ids.dtype, torch.int32)


class TestTpuLmHeadRowBuckets(CustomTestCase):
    """The LM head runs eagerly on TPU for the multimodal wrappers, and each new
    row count compiles a fresh program (Gemma4 and Qwen3.5 stalled 40-100 s at
    every batch size of a draining batch). Rows pad to a power of two for the
    matmul and the caller still gets its own row count back."""

    def test_rows_pad_to_a_power_of_two_for_the_matmul(self):
        processor = LogitsProcessor.__new__(LogitsProcessor)
        processor.use_fp32_lm_head = False
        processor.rl_on_policy_target = None
        torch.manual_seed(0)
        hidden = torch.randn(5, 16)
        lm_head = SimpleNamespace(weight=torch.randn(32, 16))
        seen = []
        original = LogitsProcessor._compute_lm_head_rows

        def record(self_, hidden_states, head, bias=None):
            seen.append(tuple(hidden_states.shape))
            return original(self_, hidden_states, head, bias)

        with (
            patch.object(logits_processor_module, "_is_tpu", True),
            patch.object(LogitsProcessor, "_compute_lm_head_rows", record),
        ):
            logits = processor._compute_lm_head(hidden, lm_head)

        self.assertEqual(seen, [(8, 16)])
        self.assertEqual(tuple(logits.shape), (5, 32))
        torch.testing.assert_close(logits, hidden @ lm_head.weight.T)

        with patch.object(logits_processor_module, "_is_tpu", False):
            direct = processor._compute_lm_head(hidden, lm_head)
        torch.testing.assert_close(direct, logits)

    def test_a_traced_graph_keeps_its_own_row_count(self):
        # Shapes are static inside a compiled graph; padding there would only
        # change the HLO of every compiled model.
        processor = LogitsProcessor.__new__(LogitsProcessor)
        processor.use_fp32_lm_head = False
        processor.rl_on_policy_target = None
        hidden = torch.randn(5, 16)
        lm_head = SimpleNamespace(weight=torch.randn(32, 16))
        seen = []
        original = LogitsProcessor._compute_lm_head_rows

        def record(self_, hidden_states, head, bias=None):
            seen.append(tuple(hidden_states.shape))
            return original(self_, hidden_states, head, bias)

        with (
            patch.object(logits_processor_module, "_is_tpu", True),
            patch.object(torch.compiler, "is_compiling", return_value=True),
            patch.object(LogitsProcessor, "_compute_lm_head_rows", record),
        ):
            logits = processor._compute_lm_head(hidden, lm_head)

        self.assertEqual(seen, [(5, 16)])
        self.assertEqual(tuple(logits.shape), (5, 32))

    def test_get_logits_pads_through_the_softcap_and_slices_back(self):
        processor = LogitsProcessor.__new__(LogitsProcessor)
        processor.use_fp32_lm_head = False
        processor.rl_on_policy_target = None
        processor.do_tensor_parallel_all_gather_dp_attn = False
        processor.do_tensor_parallel_all_gather = False
        processor.logit_scale = None
        processor.final_logit_softcapping = 30.0
        processor.vocab_size = 32
        torch.manual_seed(1)
        hidden = torch.randn(5, 16)
        lm_head = SimpleNamespace(weight=torch.randn(32, 16))
        metadata = SimpleNamespace(
            next_token_logits_buffer=None, forward_mode=ForwardMode.DECODE
        )
        seen = []
        original = LogitsProcessor._compute_lm_head_rows

        def record(self_, hidden_states, head, bias=None):
            seen.append(tuple(hidden_states.shape))
            return original(self_, hidden_states, head, bias)

        scoped = MagicMock()
        scoped.scoped.return_value.__enter__ = lambda *a: None
        scoped.scoped.return_value.__exit__ = lambda *a: None
        with (
            patch.object(logits_processor_module, "_is_tpu", True),
            patch.object(logits_processor_module, "get_forward", return_value=scoped),
            patch.object(
                logits_processor_module,
                "get_parallel",
                return_value=SimpleNamespace(tp_size=1, attn_dp_size=1),
            ),
            patch.object(
                LogitsProcessor,
                "_gather_dp_attn_hidden_states",
                lambda self_, h, m: (h, h),
            ),
            patch.object(LogitsProcessor, "_compute_lm_head_rows", record),
        ):
            logits = processor._get_logits(hidden, lm_head, metadata)

        self.assertEqual(seen, [(8, 16)])
        self.assertEqual(tuple(logits.shape), (5, 32))
        expected = 30.0 * torch.tanh((hidden @ lm_head.weight.T).float() / 30.0)
        torch.testing.assert_close(logits, expected)


class TestRowPad(CustomTestCase):
    """Eager TPU ops compile per shape; rows round up to a power of two so a
    batch draining one request at a time compiles log2(bs) times, not bs."""

    def test_rounds_up_to_the_next_power_of_two(self):
        self.assertEqual(
            [row_bucket(n) for n in (0, 1, 2, 3, 5, 8, 9, 63, 64, 65)],
            [1, 1, 2, 4, 8, 8, 16, 64, 64, 128],
        )

    def test_pads_the_leading_dim_with_the_fill_value(self):
        x = torch.arange(6, dtype=torch.float32).reshape(3, 2)
        padded = pad_rows(x, 4, fill=1.0)
        self.assertEqual(tuple(padded.shape), (4, 2))
        torch.testing.assert_close(padded[:3], x)
        self.assertTrue(bool((padded[3] == 1.0).all()))
        # Already at the bucket: the same tensor, no copy.
        self.assertIs(pad_rows(x, 3), x)
        ks = pad_rows(torch.ones(3, dtype=torch.int32), 8)
        self.assertEqual(tuple(ks.shape), (8,))
        self.assertEqual(ks.dtype, torch.int32)
        self.assertEqual(ks[3:].tolist(), [0] * 5)


class TestHostSamplingWarmupCoverage(CustomTestCase):
    """Serving batches must not reach a sampling kernel variant warmup skipped.

    A mixed greedy/sampled batch has no uniform top-k and once compiled the
    full-vocabulary filter at serving time on TPU.
    """

    VOCAB = 128

    def _kernel_calls(self, infos):
        from sglang_tpu import sampling_mask as sampling_mask_module
        from sglang_tpu.sampling import warmup as warmup_module

        calls = set()

        def fake_top_k(probs, width):
            calls.add(("top_k", probs.shape[1], width))
            return torch.topk(probs, width, dim=-1)

        def fake_filter(
            probs, top_ks, top_ps, min_ps, need_top_k, need_top_p, need_min_p
        ):
            calls.add(("filter", probs.shape[1], need_top_k, need_top_p, need_min_p))
            return probs

        sampler = object.__new__(TpuSampler)
        with (
            tpu_envs.SGLANG_TPU_HOST_PADDING.override(True),
            patch.object(tpu_sampler_module, "_use_bsearch_filtering", lambda _: True),
            patch.object(tpu_top_k_module, "tpu_top_k", fake_top_k),
            patch.object(sampling_mask_module, "tpu_filter_probs", fake_filter),
        ):
            for info in infos(warmup_module):
                if info.is_all_greedy:
                    continue
                simple = not (
                    info.need_top_k_sampling
                    or info.need_top_p_sampling
                    or info.need_min_p_sampling
                )
                probs = torch.softmax(torch.randn(len(info), self.VOCAB), dim=-1)
                sampler._sample_from_probs(probs, info, None, simple)
        return calls

    def _warmup_infos(self, warmup_module):
        captured = []
        warmup_module.warmup_host_sampling(
            lambda _logits, info, *_: captured.append(info) or torch.zeros(1),
            [1, 4],
            self.VOCAB,
            "cpu",
            make_logits=lambda rows: torch.zeros(rows, self.VOCAB),
        )
        return captured

    def _serving_infos(self, warmup_module):
        sampled = (1, 20, 64, 100, warmup_module.TOP_K_ALL)
        for rows, (k0, k1), top_p, min_p in product(
            (1, 4), product((1, *sampled), sampled), (1.0, 0.9), (0.0, 0.05)
        ):
            top_ks = torch.tensor(([k0, k1] * rows)[:rows], dtype=torch.int32)
            yield SamplingBatchInfo(
                temperatures=torch.ones((rows, 1)),
                top_ps=torch.full((rows,), top_p),
                top_ks=top_ks,
                min_ps=torch.full((rows,), min_p),
                is_all_greedy=False,
                is_any_greedy=bool((top_ks == 1).any()),
                need_top_p_sampling=top_p < 1.0,
                need_top_k_sampling=bool((top_ks != warmup_module.TOP_K_ALL).any()),
                need_min_p_sampling=min_p > 0.0,
                vocab_size=self.VOCAB,
                return_sampling_masks=[False] * rows,
                device="cpu",
                uniform_top_k=int(top_ks[0])
                if bool((top_ks == top_ks[0]).all())
                else None,
            )

    def test_serving_kernel_variants_are_warmed(self):
        warmed = self._kernel_calls(self._warmup_infos)
        served = self._kernel_calls(self._serving_infos)
        self.assertLessEqual(served, warmed)


if __name__ == "__main__":
    unittest.main()
