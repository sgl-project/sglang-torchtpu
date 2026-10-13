"""CPU coverage for running a TPU EAGLE3 decode step at its batch bucket."""

import unittest
from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.overlap_utils import FutureMap, RelayPayload
from sglang.srt.speculative import spec_utils
from sglang.srt.speculative.spec_utils import _gather_spec_extras_torch
from sglang_tpu import (
    eagle_utils_tpu,
    patches,
    spec_decode_bucket,
)
from sglang_tpu.overlap_utils_tpu import (
    resolve_device_accept_seq_lens_tpu,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

NUM_DRAFT_TOKENS = 4
WIDTH = 2 * NUM_DRAFT_TOKENS - 1


@contextmanager
def tpu_patches(bucket=4):
    """Register a decode bucket and install the TPU FutureMap patch."""
    with ExitStack() as stack:
        for name in (
            "stash",
            "publish",
            "stash_bonus_tokens",
            "resolve_mixed_spec_tails",
        ):
            stack.enter_context(patch.object(FutureMap, name, getattr(FutureMap, name)))
        for name in ("tpu_direct_handoff", "tpu_published_seq_lens", "tpu_decode_rows"):
            stack.enter_context(patch.object(FutureMap, name, None, create=True))
        stack.enter_context(
            patch.object(patches, "_future_map_direct_handoff_patched", False)
        )
        stack.enter_context(
            patch.object(
                spec_decode_bucket,
                "_decode_batch_bucket",
                (lambda n: bucket if n <= bucket else None) if bucket else None,
            )
        )
        spec = SimpleNamespace(speculative_num_steps=3)
        stack.enter_context(
            patch.object(eagle_utils_tpu, "get_spec", return_value=spec)
        )
        stack.enter_context(
            patch.object(
                eagle_utils_tpu,
                "max_speculative_num_draft_tokens",
                return_value=NUM_DRAFT_TOKENS,
            )
        )
        stack.enter_context(
            patch.object(
                spec_utils, "gather_spec_extras_func", _gather_spec_extras_torch
            )
        )
        patches._patch_future_map_direct_handoff()
        yield


def make_batch(seqlens=(10, 20, 30), greedy=True):
    bs = len(seqlens)
    req_pool_indices = torch.arange(1, bs + 1, dtype=torch.int64)
    table = torch.arange(8 * 64, dtype=torch.int32).reshape(8, 64) + 1000
    draft = SimpleNamespace(
        future_indices=req_pool_indices,
        accept_lens_device=torch.ones(bs, dtype=torch.int32),
        device_accept_candidate_locs=None,
        device_accept_host_base=None,
        device_accept_selected_cache_locs=None,
        decode_bucket=None,
        future_dsa_topk_indices_available=False,
    )
    reqs = [
        SimpleNamespace(seqlen=s, kv=SimpleNamespace(kv_allocated_len=64), lora_id=None)
        for s in seqlens
    ]
    return SimpleNamespace(
        reqs=reqs,
        spec_info=draft,
        enable_overlap=True,
        return_logprob=False,
        return_hidden_states=False,
        has_grammar=False,
        req_to_token_pool=SimpleNamespace(host_resident=True, req_to_token=table),
        req_pool_indices=req_pool_indices,
        req_pool_indices_cpu=req_pool_indices.clone(),
        device=torch.device("cpu"),
        sampling_info=SimpleNamespace(
            is_all_greedy=greedy,
            penalizer_orchestrator=SimpleNamespace(is_required=False),
            logit_bias=None,
            has_custom_logit_processor=False,
        ),
    )


def make_future_map():
    algo = SimpleNamespace(
        is_eagle3=lambda: True,
        is_some=lambda: True,
        is_ngram=lambda: False,
        need_topk=lambda: True,
    )
    fm = FutureMap(
        device="cpu",
        spec_algo=algo,
        req_to_token_pool=SimpleNamespace(req_to_token=torch.zeros(8, 64)),
        needs_cpu_seq_lens=False,
    )
    fm.need_topk = fm.need_hidden_states = True
    fm.topk_p_buf = torch.zeros(8, 1)
    fm.topk_index_buf = torch.zeros(8, 1, dtype=torch.int64)
    fm.hidden_states_buf = torch.zeros(8, 2)
    return fm


def payload_for(rows: int, base: int) -> RelayPayload:
    return RelayPayload(
        bonus_tokens=torch.arange(rows) + base,
        topk_p=torch.ones(rows, 1),
        topk_index=(torch.arange(rows) + base)[:, None],
        hidden_states=torch.full((rows, 2), float(base)),
    )


class TestPrepare(CustomTestCase):
    def test_pads_host_inputs_to_the_bucket(self):
        batch = make_batch()
        with tpu_patches(bucket=4):
            eagle_utils_tpu.prepare_device_accept_tpu(batch, bs=3, double_alloc=8)
        draft = batch.spec_info
        bucket = draft.decode_bucket
        self.assertEqual(bucket.logical_bs, 3)
        self.assertEqual(bucket.rows_cpu.tolist(), [1, 2, 3, 0])
        self.assertEqual(bucket.valid.tolist(), [1, 1, 1, 0])
        self.assertEqual(bucket.seq_lens_cpu.tolist(), [10, 20, 30, 0])
        self.assertEqual(draft.device_accept_host_base.tolist(), [10, 20, 30, 0])
        self.assertEqual(tuple(draft.device_accept_candidate_locs.shape), (4, WIDTH))
        self.assertEqual(draft.device_accept_candidate_locs[3].tolist(), [0] * WIDTH)
        table = batch.req_to_token_pool.req_to_token
        self.assertEqual(
            draft.device_accept_candidate_locs[1].tolist(),
            table[2, 21 : 21 + WIDTH].tolist(),
        )
        # The scheduler keeps the real-sized host lengths.
        self.assertEqual(batch.seq_lens_cpu.tolist(), [10, 20, 30])
        self.assertEqual(batch.seq_lens_sum, 60)

    def test_keeps_real_size_without_bucket_or_when_ineligible(self):
        for bucket, greedy in ((None, True), (4, False)):
            with self.subTest(bucket=bucket, greedy=greedy):
                batch = make_batch(greedy=greedy)
                with tpu_patches(bucket=bucket):
                    eagle_utils_tpu.prepare_device_accept_tpu(
                        batch, bs=3, double_alloc=8
                    )
                draft = batch.spec_info
                self.assertIsNone(draft.decode_bucket)
                self.assertEqual(
                    tuple(draft.device_accept_candidate_locs.shape), (3, WIDTH)
                )

    def test_no_pending_acceptance_builds_only_the_rows(self):
        # First decode after a prefill: the resolve reads the lengths back.
        batch = make_batch()
        batch.spec_info.accept_lens_device = None
        with tpu_patches(bucket=4):
            eagle_utils_tpu.prepare_device_accept_tpu(batch, bs=3, double_alloc=8)
        draft = batch.spec_info
        self.assertEqual(draft.decode_bucket.rows_cpu.tolist(), [1, 2, 3, 0])
        self.assertIsNone(draft.decode_bucket.seq_lens_cpu)
        self.assertIsNone(draft.device_accept_candidate_locs)


class TestRegistration(CustomTestCase):
    def _register(self):
        from sglang_tpu.compilation import spec_decode_compile_policy_eagle3 as eagle3
        from sglang_tpu.compilation.bucketer import Bucketer

        policy = object.__new__(eagle3.Eagle3SpecDecodeCompilePolicyMixin)
        policy.speculative_eagle_topk = 1
        policy.b_bucketer = Bucketer("b", [1, 2, 4])
        with (
            patch.object(spec_decode_bucket, "_decode_batch_bucket", None),
            patch.object(eagle3, "is_dp_attention_enabled", return_value=False),
            patch.object(
                eagle3, "get_parallel", return_value=SimpleNamespace(pp_size=1)
            ),
        ):
            policy._register_decode_batch_bucket()
            return spec_decode_bucket.decode_batch_bucket(3)

    def test_registers_the_batch_bucket(self):
        self.assertEqual(self._register(), 4)

    def test_feature_flag_off_keeps_the_real_batch_size(self):
        from sglang_tpu.envs import envs as tpu_envs

        with tpu_envs.SGLANG_TPU_ENABLE_SPEC_BUCKETED_METADATA.override(False):
            self.assertIsNone(self._register())


class TestBucketedStep(CustomTestCase):
    def _prepare(self, fm, batch):
        eagle_utils_tpu.prepare_device_accept_tpu(batch, bs=3, double_alloc=8)

    def test_resolve_stores_and_hands_off_through_bucket_rows(self):
        fm = make_future_map()
        batch = make_batch()
        future_indices = batch.req_pool_indices
        with tpu_patches(bucket=4):
            # Rows 1..3 were published one acceptance ahead (accepted 1, 2, 3);
            # slot 0 holds a stale padding-row length that must not leak.
            fm.new_seq_lens_buf[:4] = torch.tensor([7, 11, 22, 33])
            self._prepare(fm, batch)
            self.assertTrue(resolve_device_accept_seq_lens_tpu(fm, batch))
            draft = batch.spec_info
            bucket = draft.decode_bucket
            self.assertEqual(bucket.seq_lens.tolist(), [11, 22, 33, 0])
            self.assertEqual(batch.seq_lens.tolist(), [11, 22, 33])
            table = batch.req_to_token_pool.req_to_token
            selected = draft.device_accept_selected_cache_locs
            self.assertEqual(tuple(selected.shape), (4, NUM_DRAFT_TOKENS))
            self.assertEqual(selected[2].tolist(), table[3, 33:37].tolist())
            self.assertEqual(selected[3].tolist(), [0] * NUM_DRAFT_TOKENS)
            self.assertEqual(tuple(draft.topk_p.shape), (4, 1))

            # The worker runs at the bucket and the scheduler publishes and
            # stashes through its real-sized future indices.
            image = object()
            batch.multimodal_inputs = [None, image, None]
            batch.orig_seq_lens = torch.tensor([1, 2, 3])
            spec_decode_bucket.enter_decode_bucket(batch)
            self.assertIs(batch.req_pool_indices, bucket.rows)
            self.assertIs(batch.seq_lens, bucket.seq_lens)
            # M-RoPE deltas are read per row; padding rows carry none.
            self.assertEqual(batch.multimodal_inputs, [None, image, None, None])
            self.assertIsNone(batch.orig_seq_lens)
            new_seq_lens = torch.tensor([13, 25, 34, 2])
            fm.publish(future_indices, new_seq_lens)
            self.assertEqual(fm.new_seq_lens_buf[:4].tolist(), [2, 13, 25, 34])
            payload = payload_for(4, base=50)
            fm.stash(future_indices, payload)
            self.assertIsNotNone(fm.tpu_direct_handoff)

            # Same composition next step: the bucket-sized payload is consumed
            # directly, padding row included.
            # The scheduler restores its own fields after the forward.
            batch.req_pool_indices = future_indices
            batch.req_pool_indices_cpu = future_indices.clone()
            batch.spec_info = SimpleNamespace(
                **{**vars(draft), "bonus_tokens": payload.bonus_tokens}
            )
            for req, length in zip(batch.reqs, (11, 22, 33)):
                req.seqlen = length
            self._prepare(fm, batch)
            self.assertTrue(resolve_device_accept_seq_lens_tpu(fm, batch))
            self.assertIsNone(fm.tpu_direct_handoff)
            self.assertEqual(
                batch.spec_info.decode_bucket.seq_lens.tolist(), [13, 25, 34, 0]
            )
            self.assertIs(batch.spec_info.hidden_states, payload.hidden_states)

    def test_filtered_step_flushes_the_bucket_payload_to_its_rows(self):
        fm = make_future_map()
        batch = make_batch()
        future_indices = batch.req_pool_indices
        with tpu_patches(bucket=4):
            self._prepare(fm, batch)
            self.assertTrue(resolve_device_accept_seq_lens_tpu(fm, batch))
            fm.publish(future_indices, torch.tensor([13, 25, 34, 2]))
            payload = payload_for(4, base=50)
            fm.stash(future_indices, payload)

            # Filtering rebinds future_indices; the pending rows are stored.
            filtered = make_batch(seqlens=(10, 30))
            filtered.req_pool_indices = torch.tensor([1, 3])
            filtered.req_pool_indices_cpu = torch.tensor([1, 3])
            filtered.spec_info.future_indices = filtered.req_pool_indices
            eagle_utils_tpu.prepare_device_accept_tpu(filtered, bs=2, double_alloc=8)
            self.assertTrue(resolve_device_accept_seq_lens_tpu(fm, filtered))
            self.assertEqual(fm.output_tokens_buf[:4].tolist(), [53, 50, 51, 52])
            self.assertEqual(
                filtered.spec_info.decode_bucket.rows_cpu.tolist(), [1, 3, 0, 0]
            )
            self.assertEqual(filtered.spec_info.bonus_tokens.tolist(), [50, 52, 53, 53])
            self.assertEqual(filtered.seq_lens.tolist(), [13, 34])

    def test_no_pending_acceptance_builds_the_window_from_read_back_lengths(self):
        fm = make_future_map()
        fm.needs_cpu_seq_lens = True
        batch = make_batch()
        batch.spec_info.accept_lens_device = None
        with tpu_patches(bucket=4):
            # A fresh prefill published its lengths; slot 0 is stale.
            fm.new_seq_lens_buf[:4] = torch.tensor([9, 10, 20, 30])
            self._prepare(fm, batch)
            self.assertTrue(resolve_device_accept_seq_lens_tpu(fm, batch))
            draft = batch.spec_info
            bucket = draft.decode_bucket
            self.assertEqual(bucket.seq_lens_cpu.tolist(), [10, 20, 30, 0])
            self.assertEqual(bucket.seq_lens.tolist(), [10, 20, 30, 0])
            self.assertEqual(batch.seq_lens_cpu.tolist(), [10, 20, 30])
            self.assertEqual(batch.seq_lens_sum, 60)
            table = batch.req_to_token_pool.req_to_token
            selected = draft.device_accept_selected_cache_locs
            self.assertEqual(selected[0].tolist(), table[1, 10:14].tolist())
            self.assertEqual(selected[2].tolist(), table[3, 30:34].tolist())
            self.assertEqual(selected[3].tolist(), [0] * NUM_DRAFT_TOKENS)


if __name__ == "__main__":
    unittest.main()
