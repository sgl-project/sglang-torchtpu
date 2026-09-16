"""Contracts for preparing all EAGLE phases before their model submissions."""

import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.hardware_backend.tpu.attention_pallas_rpa_v3 import (
    TpuPallasRpaV3AttnBackend,
    TpuPallasRpaV3MultiStepDraftBackend,
)
from sglang.srt.hardware_backend.tpu.compilation.bucketer import Bucketer
from sglang.srt.hardware_backend.tpu.compilation.decoder_compile_policy import (
    DecoderTpuCompilePolicy,
)
from sglang.srt.hardware_backend.tpu.eagle_step_inputs import EagleRelayInputs
from sglang.srt.hardware_backend.tpu.eagle_step_preparation import EagleStepPreparer
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.model_executor.tpu_graph_runner import _augment_spec_policy
from sglang.srt.runtime_context import get_context
from sglang.srt.speculative.eagle_info import (
    EagleDraftExtendInput,
    EagleDraftInput,
    EagleVerifyInput,
)
from sglang.srt.speculative.eagle_worker_v2 import EAGLEWorkerV2
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestEagleStepPreparation(CustomTestCase):
    def test_preparer_is_built_after_draft_backend_and_policy_initialization(self):
        """The worker constructor runs before the scheduler installs attention backends."""
        worker = object.__new__(EAGLEWorkerV2)
        worker._draft_worker = SimpleNamespace()
        worker._target_worker = SimpleNamespace(model_runner=self.target)
        worker.speculative_algorithm = SpeculativeAlgorithm.EAGLE3
        worker.server_args = SimpleNamespace()
        worker.adaptive_controller = None
        worker.tpu_step_preparer = None

        def initialize_backends():
            worker._draft_worker.draft_attn_backend = self.multi
            worker._draft_worker.draft_runner = self.draft

        module = "sglang.srt.speculative.eagle_worker_v2"
        with (
            patch(module + "._is_tpu", True),
            patch(
                module + ".BaseSpecWorker.init_cuda_graphs",
                side_effect=initialize_backends,
            ),
            patch.object(
                EagleStepPreparer, "create", return_value=self.preparer
            ) as create,
        ):
            worker.init_cuda_graphs()
        create.assert_called_once_with(
            target_runner=self.target,
            draft_runner=self.draft,
            draft_attn_backend=self.multi,
            server_args=worker.server_args,
        )
        self.assertIs(worker.tpu_step_preparer, self.preparer)

    def setUp(self):
        super().setUp()
        override = get_context().override_server_args(
            speculative_algorithm="EAGLE3",
            speculative_num_steps=3,
            speculative_num_draft_tokens=4,
            speculative_eagle_topk=1,
            speculative_adaptive=False,
            speculative_use_rejection_sampling=False,
            enable_dp_attention=False,
            page_size=64,
            dp_size=1,
        )
        override.install()
        self.addCleanup(override.restore)
        req_pool = SimpleNamespace(
            req_to_token=torch.arange(8 * 8192).view(8, 8192).to(torch.int32)
        )
        self.target = self._runner(req_pool=req_pool, is_draft=False)
        self.draft = self._runner(req_pool=req_pool, is_draft=True)
        with patch(
            "sglang.srt.hardware_backend.tpu.attention_pallas_rpa_base."
            "_build_pallas_kernel_op",
            return_value=None,
        ):
            self.multi = TpuPallasRpaV3MultiStepDraftBackend(self.draft, 1, 3)
        self.preparer = EagleStepPreparer.create(
            target_runner=self.target,
            draft_runner=self.draft,
            draft_attn_backend=self.multi,
            server_args=SimpleNamespace(
                enable_dp_attention=False,
                enable_hierarchical_cache=False,
            ),
        )
        self.assertIsNotNone(self.preparer)

    def _runner(self, *, req_pool, is_draft):
        config = SimpleNamespace(
            context_len=128,
            hidden_size=4,
            spec_hidden_size=4,
            model_is_mrope=False,
            is_multimodal=False,
            hf_config=SimpleNamespace(eagle_config={"use_aux_hidden_state": False}),
        )
        runner = SimpleNamespace(
            model_config=config,
            device=torch.device("cpu"),
            page_size=64,
            req_to_token_pool=req_pool,
            is_draft_worker=is_draft,
            model=SimpleNamespace(forward=lambda: None),
            token_to_kv_pool=SimpleNamespace(
                full_to_swa_index_mapping=torch.arange(8 * 8192, dtype=torch.int64),
                tpu_commit_pending_loads=lambda: None,
            ),
            is_hybrid_swa=not is_draft,
            kv_cache_dtype=torch.bfloat16,
            spec_algorithm=SpeculativeAlgorithm.EAGLE3,
            dtype=torch.bfloat16,
            canary_manager=None,
            ngram_embedding_manager=SimpleNamespace(enabled=False),
        )
        with patch(
            "sglang.srt.hardware_backend.tpu.attention_pallas_rpa_base."
            "_build_pallas_kernel_op",
            return_value=None,
        ):
            runner.attn_backend = TpuPallasRpaV3AttnBackend(runner)
        with patch(
            "sglang.srt.hardware_backend.tpu.compilation."
            "spec_decode_compile_policy_eagle3.process_model_config",
            return_value=config,
        ):
            # TpuGraphRunner composes EAGLE over the selected decoder policy.
            policy = _augment_spec_policy(DecoderTpuCompilePolicy, runner)
            runner.compile_policy = policy(
                model_runner=runner,
                owner=runner.model,
                tracer="direct",
                b_bucketer=Bucketer(name="Bs", buckets=[1, 4, 8]),
                t_bucketer=Bucketer(name="Token", buckets=[32, 64]),
            )
        return runner

    def _batch(self, bs):
        physical = self.preparer.bucketer.next_bucket(bs)
        seq = torch.arange(65, 65 + physical, dtype=torch.int64)
        probs = torch.ones(physical, 1)
        hidden = torch.arange(physical * 4).view(physical, 4).to(torch.bfloat16)
        # The retained FutureMap dummy rows need not contain valid values.
        probs[bs:] = float("nan")
        hidden[bs:] = float("nan")
        topk = torch.arange(physical).view(physical, 1)
        bonus = torch.arange(physical, dtype=torch.int64)
        cache = torch.arange(physical * 4, dtype=torch.int64).view(physical, 4) + 32
        logical_seq = seq[:bs]
        draft = EagleDraftInput(
            topk_p=probs[:bs],
            topk_index=topk[:bs],
            bonus_tokens=bonus[:bs],
            hidden_states=hidden[:bs],
            tpu_relay_inputs=EagleRelayInputs(
                seq,
                probs,
                topk,
                bonus,
                hidden,
                cache,
                logical_seq,
            ),
        )
        return ScheduleBatch(
            reqs=[SimpleNamespace(lora_id=None) for _ in range(bs)],
            forward_mode=ForwardMode.DECODE,
            seq_lens=logical_seq,
            # This mirror deliberately precedes the last device acceptance.
            seq_lens_cpu=logical_seq.clone() - 2,
            req_pool_indices=torch.arange(1, bs + 1),
            req_pool_indices_cpu=torch.arange(1, bs + 1),
            spec_info=draft,
            sampling_info=SimpleNamespace(
                temperatures=torch.ones(bs, 1),
                top_ks=torch.ones(bs, dtype=torch.int32),
                top_ps=torch.ones(bs),
            ),
        )

    def _forward_batch(self, batch, mode):
        bs = len(batch.seq_lens)
        cache = batch.spec_info.tpu_relay_inputs.cache_locs[:bs]
        if mode.is_decode():
            spec = copy.copy(batch.spec_info)
            positions = batch.seq_lens.clone()
            cache = cache[:, :3].flatten()
            input_ids = torch.arange(bs)
            seq = batch.seq_lens
            cpu_seq = batch.seq_lens_cpu
        else:
            input_ids = torch.arange(bs * 4)
            cache = cache.flatten()
            if mode.is_target_verify():
                spec = EagleVerifyInput(
                    draft_token=input_ids,
                    draft_token_num=4,
                    topk=1,
                    spec_steps=3,
                    retrieve_index=torch.arange(bs * 4).view(bs, 4),
                    custom_mask=None,
                    positions=None,
                    retrieve_next_token=None,
                    retrieve_next_sibling=None,
                    retrieve_cum_len=None,
                    capture_hidden_mode=CaptureHiddenMode.FULL,
                    seq_lens_sum=None,
                    seq_lens_cpu=None,
                )
                positions = (batch.seq_lens[:, None] + torch.arange(4)).flatten()
                seq, cpu_seq = batch.seq_lens, batch.seq_lens_cpu
            else:
                spec = EagleDraftExtendInput(
                    hidden_states=torch.arange(bs * 16)
                    .view(bs * 4, 4)
                    .to(torch.bfloat16),
                    num_correct_drafts=torch.arange(bs, dtype=torch.int32) % 3,
                    num_accept_tokens=torch.arange(bs, dtype=torch.int32) % 3 + 1,
                    num_tokens_per_req=4,
                )
                positions = (batch.seq_lens_cpu[:, None] + torch.arange(4)).flatten()
                seq, cpu_seq = batch.seq_lens + 4, batch.seq_lens_cpu + 4
        fb = ForwardBatch(
            forward_mode=mode,
            batch_size=bs,
            input_ids=input_ids,
            req_pool_indices=batch.req_pool_indices,
            req_pool_indices_cpu=batch.req_pool_indices_cpu,
            seq_lens=seq,
            seq_lens_cpu=cpu_seq,
            seq_lens_sum=int(cpu_seq.sum()),
            out_cache_loc=cache,
            positions=positions,
            spec_info=spec,
            orig_seq_lens=torch.full((bs,), 63),
            lora_ids=[None] * bs,
        )
        if mode.is_draft_extend_v2():
            fb.extend_seq_lens = torch.full((bs,), 4, dtype=torch.int32)
            fb.extend_prefix_lens = batch.seq_lens_cpu.to(torch.int32)
            fb.extend_start_loc = torch.arange(bs, dtype=torch.int32) * 4
            fb.extend_seq_lens_cpu = [4] * bs
            fb.extend_prefix_lens_cpu = batch.seq_lens_cpu.tolist()
            fb.extend_logprob_start_lens_cpu = [0] * bs
        return fb

    def test_all_phase_arguments_match_existing_physical_padding(self):
        """Padding and stale host mirrors must leave model inputs bit-identical."""
        fields = (
            "input_ids",
            "positions",
            "seq_lens",
            "seq_lens_cpu",
            "req_pool_indices",
            "req_pool_indices_cpu",
            "out_cache_loc",
            "orig_seq_lens",
            "extend_seq_lens",
            "extend_prefix_lens",
            "extend_start_loc",
            "extend_seq_lens_cpu",
            "extend_prefix_lens_cpu",
        )
        for bs in (1, 2, 3, 4, 5):
            batch = self._batch(bs)
            step = self.preparer.prepare(batch)
            self.assertIsNotNone(step)
            for mode in (
                ForwardMode.DECODE,
                ForwardMode.TARGET_VERIFY,
                ForwardMode.DRAFT_EXTEND_V2,
            ):
                with self.subTest(bs=bs, mode=mode):
                    expected = self._forward_batch(batch, mode)
                    actual = self._forward_batch(batch, mode)
                    runner = self.target if mode.is_target_verify() else self.draft
                    policy = runner.compile_policy
                    prepare = (
                        policy.prepare_forward_batch
                        if mode.is_target_verify()
                        else policy.prepare_spec_forward_batch
                    )
                    prepare(expected)
                    with step:
                        prepare(actual)
                    for field in fields:
                        torch.testing.assert_close(
                            getattr(actual, field),
                            getattr(expected, field),
                            rtol=0,
                            atol=0,
                        )
                    for field in (
                        ("topk_p", "topk_index", "bonus_tokens", "hidden_states")
                        if mode.is_decode()
                        else (
                            ("hidden_states", "num_correct_drafts", "num_accept_tokens")
                            if mode.is_draft_extend_v2()
                            else ()
                        )
                    ):
                        torch.testing.assert_close(
                            getattr(actual.spec_info, field),
                            getattr(expected.spec_info, field),
                            rtol=0,
                            atol=0,
                        )
                    backends = (
                        self.multi.attn_backends
                        if mode.is_decode()
                        else (runner.attn_backend,)
                    )
                    for backend in backends:
                        ref = backend._build_metadata(expected, bs)
                        with step:
                            backend.init_forward_metadata(actual)
                            got = backend.forward_metadata
                        for field in (
                            "kv_lens",
                            "cu_q_lens",
                            "cu_kv_lens",
                            "page_indices",
                            "distribution",
                            "active_num_tokens",
                            "swa_page_indices",
                        ):
                            torch.testing.assert_close(
                                getattr(got, field), getattr(ref, field), rtol=0, atol=0
                            )
            self.assertEqual(batch.seq_lens.tolist(), list(range(65, 65 + bs)))

    def test_fused_draft_metadata_matches_precompiled_inputs(self):
        """The fused loop receives raw lengths and derives int32 KV lengths itself."""
        for bs in (1, 4):
            with self.subTest(bs=bs):
                batch = self._batch(bs)
                step = self.preparer.prepare(batch)
                expected = self._forward_batch(batch, ForwardMode.DECODE)
                actual = self._forward_batch(batch, ForwardMode.DECODE)
                policy = self.draft.compile_policy
                policy.prepare_spec_forward_batch(expected)
                with step:
                    policy.prepare_spec_forward_batch(actual)
                    self.multi.init_forward_metadata(actual, fused_draft=True)
                for backend in self.multi.attn_backends:
                    reference = backend._build_metadata(expected, bs, fused_draft=True)
                    torch.testing.assert_close(
                        backend.forward_metadata.kv_lens,
                        reference.kv_lens,
                        rtol=0,
                        atol=0,
                    )
                    self.assertIs(
                        backend.forward_metadata.kv_lens, step.inputs.seq_lens
                    )

    def test_page_mapping_changes_do_not_overwrite_queued_snapshots(self):
        """Allocation through aliases must invalidate reuse without mutating old work."""
        batch = self._batch(2)
        first = self.preparer.prepare(batch)
        backend = self.target.attn_backend
        key = backend, ForwardMode.TARGET_VERIFY
        original = first._metadata[key]
        same = self.preparer.prepare(batch)
        self.assertIs(original.page_indices, same._metadata[key].page_indices)
        # NumPy writes bypass the tensor's version counter.
        backend.req_to_token_pool.req_to_token.numpy()[1, 0] = 128
        changed = self.preparer.prepare(batch)._metadata[key]
        self.assertNotEqual(changed.page_indices[0], original.page_indices[0])
        self.assertEqual(original.page_indices[0], 128)
        backend.token_to_kv_pool.full_to_swa_index_mapping.numpy()[128] = 192
        remapped = self.preparer.prepare(batch)._metadata[key]
        self.assertEqual(remapped.swa_page_indices[0], 3)
        self.assertEqual(changed.swa_page_indices[0], 2)
        self.assertIsNot(remapped.swa_page_indices, changed.swa_page_indices)

    def test_context_and_relay_identity_reject_stale_steps(self):
        """Failed steps and new logical batches must not inherit prepared state."""
        batch = self._batch(2)
        first = self.preparer.prepare(batch)
        with self.assertRaisesRegex(RuntimeError, "abort step"):
            with first:
                raise RuntimeError("abort step")
        for owner in (*self.preparer.policies, *self.preparer.backends):
            self.assertIsNone(owner.prepared_eagle_step)
        batch.seq_lens = batch.seq_lens + 1
        self.assertIsNone(self.preparer.prepare(batch))
        other = self._batch(2)
        other.req_pool_indices = other.req_pool_indices.flip(0)
        other.req_pool_indices_cpu = other.req_pool_indices_cpu.flip(0)
        second = self.preparer.prepare(other)
        self.assertIsNot(first.layout, second.layout)
        self.assertEqual(first.layout.req_pool_indices_cpu.tolist(), [1, 2, 0, 0])
        self.assertEqual(second.layout.req_pool_indices_cpu.tolist(), [2, 1, 0, 0])

    def test_sampling_reuses_prepared_arguments_and_restores_logical_aliases(self):
        """Late verification binding must preserve the sampler's restoration contract."""
        policy = self.target.compile_policy
        batch = self._batch(3)
        reference_batch = self._batch(3)
        actual = self._forward_batch(batch, ForwardMode.TARGET_VERIFY)
        expected = self._forward_batch(reference_batch, ForwardMode.TARGET_VERIFY)
        logical_seq = batch.seq_lens
        logical_tokens = actual.spec_info.draft_token
        actual.spec_info.keep_outputs_padded = True
        expected.spec_info.keep_outputs_padded = True
        policy.prepare_forward_batch(expected)
        logits = torch.zeros((16, 8))
        reference_padding = policy.prepare_spec_verify(
            reference_batch, expected.spec_info, logits
        )
        with self.preparer.prepare(batch):
            policy.prepare_forward_batch(actual)
            padding = policy.prepare_spec_verify(batch, actual.spec_info, logits)
            torch.testing.assert_close(batch.seq_lens, reference_batch.seq_lens)
            for field in ("draft_token", "retrieve_index"):
                torch.testing.assert_close(
                    getattr(actual.spec_info, field), getattr(expected.spec_info, field)
                )
            for field in ("temperatures", "top_ks", "top_ps"):
                torch.testing.assert_close(
                    getattr(batch.sampling_info, field),
                    getattr(reference_batch.sampling_info, field),
                )
            policy.finalize_spec_verify(padding, (None, None, None), (None, None))
        policy.finalize_spec_verify(reference_padding, (None, None, None), (None, None))
        self.assertIs(batch.seq_lens, logical_seq)
        self.assertIs(actual.spec_info.draft_token, logical_tokens)


if __name__ == "__main__":
    unittest.main()
