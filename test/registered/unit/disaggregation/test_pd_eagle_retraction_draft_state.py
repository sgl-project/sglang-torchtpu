"""A PD decode retraction must hand an EAGLE3 request back its own draft KV, every
step's pool under multi-layer EAGLE, not whatever reused its slots; target
verification hides the loss, acceptance drops."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.mem_cache import memory_pool
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.common import restore_kv_cache
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from sglang.srt.runtime_context import get_disagg
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker, EAGLEWorkerV2
from sglang.srt.speculative.frozen_kv_mtp_worker_v2 import (
    FrozenKVMTPDraftWorker,
    FrozenKVMTPWorkerV2,
)
from sglang.srt.speculative.multi_layer_eagle_worker_v2 import (
    MultiLayerEagleDraftWorker,
    MultiLayerEagleWorkerV2,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, enter_scope, published_topology

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

POOL_SIZE = 64
PROMPT_LEN = 9
DECODED = 7  # tokens generated before the retraction
POISON = -7.0


def _pool(layer_num: int) -> MHATokenToKVPool:
    return MHATokenToKVPool(
        size=POOL_SIZE,
        page_size=1,
        head_num=2,
        head_dim=8,
        dtype=torch.float32,
        layer_num=layer_num,
        device="cpu",
        enable_memory_saver=False,
    )


def _seed(pool: MHATokenToKVPool, indices: torch.Tensor, base: float) -> None:
    for layer, (k, v) in enumerate(zip(pool.k_buffer, pool.v_buffer, strict=True)):
        pattern = torch.arange(k[indices].numel(), dtype=torch.float32).reshape_as(
            k[indices]
        )
        k[indices] = pattern + base + 1000 * layer
        v[indices] = pattern + base + 500 + 1000 * layer


def _snapshot(pool: MHATokenToKVPool, indices: torch.Tensor):
    return [
        (k[indices].clone(), v[indices].clone())
        for k, v in zip(pool.k_buffer, pool.v_buffer, strict=True)
    ]


def _poison(pool: MHATokenToKVPool) -> None:
    """Another request reuses every slot while this one is retracted."""
    for k, v in zip(pool.k_buffer, pool.v_buffer, strict=True):
        k.fill_(POISON)
        v.fill_(POISON)


def _rows_equal(pool: MHATokenToKVPool, indices: torch.Tensor, expected) -> bool:
    return all(
        torch.equal(k[indices], ek) and torch.equal(v[indices], ev)
        for (k, v), (ek, ev) in zip(
            zip(pool.k_buffer, pool.v_buffer, strict=True), expected, strict=True
        )
    )


class TestPdEagleRetractionDraftState(CustomTestCase):
    NUM_DRAFT_POOLS = 1

    def setUp(self):
        enter_scope(
            self,
            published_topology(
                disaggregation_mode="decode",
                disaggregation_decode_retraction_backup="cpu_tensor",
                speculative_algorithm="EAGLE3",
                speculative_eagle_topk=1,
                speculative_num_steps=3,
                speculative_num_draft_tokens=4,
            ),
        )
        # CPU tensors; the platform's device sync has nothing to wait for.
        enter_scope(self, patch.object(memory_pool.current_platform, "synchronize"))

        self.target = _pool(layer_num=3)
        self.drafts = [_pool(layer_num=1) for _ in range(self.NUM_DRAFT_POOLS)]
        self.allocator = TokenToKVPoolAllocator(
            POOL_SIZE, torch.float32, "cpu", self.target, need_sort=False
        )
        self.draft_kv_pools = self._alloc_draft_pools().retraction_draft_kv_pools

        seqlen = PROMPT_LEN + DECODED
        self.req = Req(
            "rid", "", list(range(1, PROMPT_LEN + 1)), SamplingParams(max_new_tokens=64)
        )
        self.req.output_ids = list(range(100, 100 + DECODED))
        self.req_pool = ReqToTokenPool(
            size=1, max_context_len=64, device="cpu", enable_memory_saver=False
        )
        self.req_pool.alloc([self.req])
        self.req.kv.kv_committed_len = self.req.kv.kv_allocated_len = seqlen
        self.old = torch.arange(1, seqlen)  # slots of the first seqlen-1 tokens
        self.new = self.old + 32  # where the resumed request lands
        self.req_pool.req_to_token[self.req.kv.req_pool_idx, : seqlen - 1] = self.old

        _seed(self.target, self.old, base=1.0)
        self.target_before = _snapshot(self.target, self.old)
        self.drafts_before = []
        for step, draft in enumerate(self.drafts):
            _seed(draft, self.old, base=9.0 + 100 * step)
            self.drafts_before.append(_snapshot(draft, self.old))

    def _alloc_draft_pools(self) -> EAGLEWorkerV2:
        """Attach the draft pool through EAGLEWorkerV2.alloc_memory_pool, as the
        scheduler does: the draft worker shares the target allocator."""
        draft_tp_worker = SimpleNamespace(
            alloc_memory_pool=Mock(),
            model_runner=SimpleNamespace(token_to_kv_pool=self.drafts[0]),
        )
        draft_worker = object.__new__(EagleDraftWorker)
        draft_worker.draft_worker = draft_tp_worker
        draft_worker.draft_runner = draft_tp_worker.model_runner
        worker = object.__new__(EAGLEWorkerV2)
        worker._draft_worker = draft_worker
        with (
            patch.object(EagleDraftWorker, "init_token_map"),
            patch.object(EagleDraftWorker, "init_lm_head"),
        ):
            worker.alloc_memory_pool(
                req_to_token_pool=Mock(), token_to_kv_pool_allocator=self.allocator
            )
        return worker

    def _retract_and_resume(self) -> None:
        """retract_decode's release_req, slot reuse, then the resume path's
        _pre_alloc and restore at new slots."""
        batch = object.__new__(ScheduleBatch)
        batch.reqs = [self.req]
        batch.req_to_token_pool = self.req_pool
        batch.token_to_kv_pool_allocator = self.allocator
        batch.tree_cache = Mock(
            token_to_kv_pool_allocator=self.allocator,
            req_to_token_pool=self.req_pool,
            **{
                "claim_kv_row.return_value": False,
                "supports_prefix_sharing.return_value": False,
            },
        )
        batch.hisparse_coordinator = None
        self.assertTrue(batch.release_req(0, 0, draft_kv_pools=self.draft_kv_pools))

        _poison(self.target)
        for draft in self.drafts:
            _poison(draft)
        self.req_pool.alloc([self.req])  # _pre_alloc on resume
        self.req_pool.req_to_token[self.req.kv.req_pool_idx, : len(self.new)] = self.new
        restore_kv_cache(
            self.req,
            None,
            self.req_pool,
            self.allocator,
            get_disagg().disaggregation_decode_retraction_backup,
        )

    def test_target_kv_survives_retraction(self):
        """Negative control: the same round trip already preserves target KV."""
        self._retract_and_resume()
        self.assertTrue(_rows_equal(self.target, self.new, self.target_before))

    def test_draft_kv_survives_retraction(self):
        """Draft KV shares the target allocator's slots, so it must travel with it."""
        self._retract_and_resume()
        for step, (draft, before) in enumerate(
            zip(self.drafts, self.drafts_before, strict=True)
        ):
            self.assertTrue(
                _rows_equal(draft, self.new, before),
                f"cpu_tensor retraction restored the target KV but not draft step "
                f"{step}: the resumed request drafts against another request's KV",
            )


class TestPdMultiLayerEagleRetractionDraftState(TestPdEagleRetractionDraftState):
    NUM_DRAFT_POOLS = 2

    def _alloc_draft_pools(self) -> MultiLayerEagleWorkerV2:
        """Attach one draft pool per step through
        MultiLayerEagleWorkerV2.alloc_memory_pool."""
        draft_worker = object.__new__(MultiLayerEagleDraftWorker)
        draft_worker.draft_worker = SimpleNamespace(alloc_memory_pool=Mock())
        draft_worker.draft_runner_list = [
            SimpleNamespace(token_to_kv_pool=draft) for draft in self.drafts
        ]
        worker = object.__new__(MultiLayerEagleWorkerV2)
        worker._draft_worker = draft_worker
        with (
            patch.object(MultiLayerEagleDraftWorker, "init_lm_head"),
            patch.object(MultiLayerEagleDraftWorker, "_init_boundary_kv_fix_state"),
        ):
            worker.alloc_memory_pool(
                req_to_token_pool=Mock(), token_to_kv_pool_allocator=self.allocator
            )
        return worker


class TestFrozenKVMTPCarriesNoDraftKV(CustomTestCase):
    def test_frozen_kv_mtp_carries_no_draft_kv(self):
        """FrozenKV MTP subclasses the EAGLE workers but reads the target KV; its
        64-token dummy draft pool cannot be indexed by the target's token ids."""
        worker = object.__new__(FrozenKVMTPWorkerV2)
        worker._draft_worker = object.__new__(FrozenKVMTPDraftWorker)
        self.assertEqual(worker.retraction_draft_kv_pools, ())


if __name__ == "__main__":
    unittest.main()
