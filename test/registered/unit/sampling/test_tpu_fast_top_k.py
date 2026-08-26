"""CPU-side coverage for TPU distributed exact top-k dispatch and merging.
Tests enforce conservative fallbacks without importing physical-TPU dependencies."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

import importlib
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.tpu import sampler as tpu_sampler_module
from sglang.srt.hardware_backend.tpu.sampler import (
    TpuSampler,
    distributed_top_k_candidate_width,
    distributed_top_k_for_batch,
    sample_from_global_top_k_logits,
)
from sglang.srt.hardware_backend.tpu.top_k import (
    TpuVocabShard,
    distributed_tpu_top_k,
    supports_distributed_tpu_top_k,
    tpu_top_k,
)
from sglang.srt.layers import logits_processor as logits_processor_module
from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput
from sglang.srt.layers.vocab_parallel_embedding import (
    VocabParallelEmbeddingShardIndices,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.test_utils import CustomTestCase


def _forward_batch(*, batch_size=1, **info_overrides):
    info = SimpleNamespace(
        uniform_top_k=50,
        is_any_greedy=False,
        need_top_k_sampling=True,
        need_top_p_sampling=False,
        need_min_p_sampling=False,
        has_custom_logit_processor=False,
        logit_bias=None,
        grammars=None,
        return_sampling_masks=None,
        sampling_mask_max_top_k=0,
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
    )


class TestDistributedTopKEligibility(CustomTestCase):
    def test_candidate_width_is_static_across_requested_top_k(self):
        processor = SimpleNamespace(
            use_tpu_distributed_top_k=True,
            do_tensor_parallel_all_gather=True,
            do_tensor_parallel_all_gather_dp_attn=False,
        )
        tp16_lm_head = SimpleNamespace(
            enable_tp=True,
            num_embeddings=202_048,
            num_embeddings_per_partition=202_048 // 16,
            shard_indices=object(),
        )

        with patch.object(tpu_sampler_module, "_DISTRIBUTED_TOP_K_MIN_BATCH_SIZE", 1):
            self.assertEqual(
                distributed_top_k_candidate_width(processor, tp16_lm_head), 64
            )
            for top_k in (2, 7, 50, 64):
                with self.subTest(top_k=top_k):
                    batch = _forward_batch(
                        batch_size=1,
                        uniform_top_k=top_k,
                        top_ks=torch.tensor([top_k], dtype=torch.int32),
                    )
                    self.assertEqual(
                        LogitsProcessor.get_tpu_distributed_top_k_for_batch(
                            processor, batch, tp16_lm_head
                        ),
                        top_k,
                    )

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
            num_embeddings_per_partition=16,
            shard_indices=object(),
        )

        self.assertIsNone(
            LogitsProcessor.get_tpu_distributed_top_k_for_batch(
                processor, _forward_batch(), sharded_lm_head
            )
        )

    def test_full_vocab_features_fall_back(self):
        cases = {
            "mixed top-k": {"uniform_top_k": None},
            "large top-k": {"uniform_top_k": 65},
            "greedy row": {"is_any_greedy": True},
            "seeded sampling": {"sampling_seed": torch.tensor([17])},
            "top-p": {"need_top_p_sampling": True},
            "min-p": {"need_min_p_sampling": True},
            "custom processor": {"has_custom_logit_processor": True},
            "logit bias": {"logit_bias": torch.zeros(1, 8)},
            "grammar": {"grammars": [object()]},
            "sampling mask": {"sampling_mask_max_top_k": 50},
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
            actual = TpuSampler._sample_from_probs(
                sampler, probs, info, positions, simple_sampling_case=False
            )

        self.assertIs(actual, expected)
        generic_sampler.assert_called_once_with(
            probs,
            info.top_ks,
            info.top_ps,
            info.min_ps,
            False,
            None,
            positions,
        )

    def test_filtered_sampling_uses_bsearch_mask_on_tpu(self):
        """On TPU tensors, filtered sampling routes through the sort-free
        binary-search mask + plain multinomial instead of the torch sort path.
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
                "sglang.srt.hardware_backend.tpu.sampling_mask.tpu_filter_probs",
                return_value=filtered,
            ) as filter_probs,
            patch.object(
                tpu_sampler_module,
                "sampling_from_probs_torch",
                return_value=expected,
            ) as sample,
            patch.object(
                tpu_sampler_module,
                "top_k_top_p_min_p_sampling_from_probs_torch",
            ) as generic_sampler,
        ):
            actual = TpuSampler._sample_from_probs(
                sampler, probs, info, positions, simple_sampling_case=False
            )

        self.assertIs(actual, expected)
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
                "sglang.srt.hardware_backend.tpu.top_k.tpu_top_k",
                return_value=(global_top_k_probs, global_top_k_token_ids),
            ) as top_k,
            patch(
                "sglang.srt.hardware_backend.tpu.sampling_mask.tpu_filter_probs",
                return_value=filtered_block,
            ) as filter_probs,
            patch.object(
                tpu_sampler_module,
                "sampling_from_probs_torch",
                return_value=torch.tensor([0], dtype=torch.int32),
            ) as sample,
        ):
            actual = TpuSampler._sample_from_probs(
                sampler, probs, info, positions, simple_sampling_case=False
            )

        torch.testing.assert_close(actual, torch.tensor([91], dtype=torch.int32))
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
        sample.assert_called_once_with(filtered_block)

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
            patch("sglang.srt.hardware_backend.tpu.top_k.tpu_top_k") as top_k,
            patch.object(
                tpu_sampler_module,
                "top_k_top_p_min_p_sampling_from_probs_torch",
                return_value=expected,
            ) as generic_sampler,
        ):
            actual = TpuSampler._sample_from_probs(
                sampler, probs, info, positions, simple_sampling_case=False
            )

        self.assertIs(actual, expected)
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
                "sglang.srt.hardware_backend.tpu.top_k.tpu_top_k",
                return_value=(global_top_k_probs, global_top_k_token_ids),
            ) as top_k,
            patch.object(
                tpu_sampler_module,
                "sampling_from_probs_torch",
                return_value=torch.tensor([1], dtype=torch.int32),
            ) as sample,
        ):
            actual = TpuSampler._sample_from_probs(
                sampler, probs, info, positions, simple_sampling_case=False
            )

        torch.testing.assert_close(actual, torch.tensor([17], dtype=torch.int32))
        top_k.assert_called_once_with(probs, 50)
        sample.assert_called_once_with(global_top_k_probs)


class TestTpuTopKLayout(CustomTestCase):
    def test_distributed_path_requires_a_sharded_lm_head(self):
        layout = SimpleNamespace(
            enable_tp=True,
            num_embeddings=128,
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
            num_embeddings_per_partition=12,
        )

        self.assertEqual(
            TpuVocabShard.from_lm_head(lm_head),
            TpuVocabShard(
                width=12,
                global_vocab_size=22,
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

    def test_distributed_merge_matches_full_top_k(self):
        local_logits = torch.tensor([[9.0, 3.0, -2.0, 7.0]])
        shard = TpuVocabShard(
            width=4,
            global_vocab_size=8,
            org_vocab_start=0,
            org_vocab_count=2,
            org_vocab_padded_count=3,
            added_vocab_start=3,
            added_vocab_count=1,
        )
        peer_logits = torch.tensor([[8.0, 6.0, 1.0, 0.0]])
        peer_ids = torch.tensor([4, 5, 6, 7], dtype=torch.int32)
        k = 3

        local_ids = torch.tensor([0, 1, -1, 3], dtype=torch.int32)
        masked_local = local_logits.masked_fill(
            local_ids.lt(0).unsqueeze(0), -torch.inf
        )
        local_sharded_top_k_logits, local_sharded_top_k_positions = torch.topk(
            masked_local, k, dim=-1
        )
        local_sharded_top_k_token_ids = local_ids[local_sharded_top_k_positions]
        peer_sharded_top_k_logits, peer_sharded_top_k_positions = torch.topk(
            peer_logits, k, dim=-1
        )
        peer_sharded_top_k_token_ids = peer_ids[peer_sharded_top_k_positions]
        group = SimpleNamespace(
            world_size=2,
            all_gather=MagicMock(
                return_value=torch.cat(
                    (
                        torch.stack(
                            (
                                local_sharded_top_k_logits.float(),
                                local_sharded_top_k_token_ids.float(),
                            ),
                            dim=-1,
                        ),
                        torch.stack(
                            (
                                peer_sharded_top_k_logits.float(),
                                peer_sharded_top_k_token_ids.float(),
                            ),
                            dim=-1,
                        ),
                    ),
                    dim=1,
                )
            ),
        )

        with (
            patch(
                "sglang.srt.hardware_backend.tpu.top_k.tpu_sharded_top_k",
                return_value=(
                    local_sharded_top_k_logits,
                    local_sharded_top_k_token_ids,
                ),
            ),
            patch(
                "sglang.srt.hardware_backend.tpu.top_k.tpu_top_k",
                side_effect=lambda values, count: torch.topk(values, count, dim=-1),
            ),
        ):
            global_top_k_logits, global_top_k_token_ids = distributed_tpu_top_k(
                local_logits.clone(),
                shard,
                k,
                group=group,
            )

        torch.testing.assert_close(global_top_k_logits, torch.tensor([[9.0, 8.0, 7.0]]))
        torch.testing.assert_close(
            global_top_k_token_ids, torch.tensor([[0, 4, 3]], dtype=torch.int32)
        )


if __name__ == "__main__":
    unittest.main()
