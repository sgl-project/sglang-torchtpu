"""Prefix tails on the K3_PROGRAM prefill: prompt ends kept past their last
full page, and a later prompt's radix match continuing inside the page."""

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.envs import envs as tpu_envs
from sglang_tpu.k3_programs import sglang_bridge as bridge
from sglang_tpu.k3_programs import sglang_pools
from sglang_tpu.k3_programs import sglang_worker as worker
from sglang_tpu.k3_programs.kimi_k3.layout import KDA, MLA
from sglang_tpu.k3_programs.program import StepOutput
from sglang_tpu.k3_programs.sglang_tails import TailIndex

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

PAGE = 8192
ALIGN = 256


class TestTailIndex(CustomTestCase):
    def setUp(self):
        self.index = TailIndex(2, PAGE, ALIGN)
        self.prompt = np.arange(3 * PAGE + 3000, dtype=np.int32)

    def test_lookup_continues_a_page_aligned_hit(self):
        end = (3 * PAGE + 3000) // ALIGN * ALIGN
        slot = self.index.record(self.prompt, end)
        self.assertIsNotNone(slot)
        longer = np.concatenate([self.prompt, np.arange(500, dtype=np.int32)])
        self.index.begin_step()
        self.assertEqual(
            self.index.lookup(longer, 3 * PAGE, len(longer) - 1), (slot, end)
        )
        # A different prefix before the page boundary, or different tail tokens.
        other = longer.copy()
        other[5] += 1
        self.assertIsNone(self.index.lookup(other, 3 * PAGE, len(other) - 1))
        other = longer.copy()
        other[3 * PAGE + 10] += 1
        self.assertIsNone(self.index.lookup(other, 3 * PAGE, len(other) - 1))
        # The tail must end before the step's limit.
        self.assertIsNone(self.index.lookup(longer, 3 * PAGE, end - ALIGN))
        # Hits that are not page aligned have no tails.
        self.assertIsNone(self.index.lookup(longer, 3 * PAGE - ALIGN, len(longer) - 1))

    def test_nothing_past_the_boundary_or_already_kept(self):
        self.assertIsNone(self.index.record(self.prompt, 3 * PAGE))
        end = 3 * PAGE + 2 * ALIGN
        self.assertIsNotNone(self.index.record(self.prompt, end))
        self.assertIsNone(self.index.record(self.prompt, end))
        with self.assertRaisesRegex(ValueError, "alignment"):
            self.index.record(self.prompt, end + 1)

    def test_lru_and_pinning(self):
        a = np.arange(PAGE + 1024, dtype=np.int32)
        b = a + 1_000_000
        c = a + 2_000_000
        self.index.begin_step()
        sa = self.index.record(a, PAGE + 512)
        sb = self.index.record(b, PAGE + 512)
        # Both slots were written this step: a third tail waits.
        self.assertIsNone(self.index.record(c, PAGE + 512))
        self.index.begin_step()
        self.assertEqual(self.index.lookup(a, PAGE, PAGE + 1000)[0], sa)
        # a was read this step (pinned), so c replaces b.
        self.assertEqual(self.index.record(c, PAGE + 512), sb)
        self.assertIsNone(self.index.lookup(b, PAGE, PAGE + 1000))


class TestTailReuse(CustomTestCase):
    """A session's new tail takes its earlier turn's slot, not another
    session's live tail."""

    def test_new_turn_replaces_its_earlier_tail(self):
        index = TailIndex(2, PAGE, ALIGN)
        a = np.arange(3 * PAGE, dtype=np.int32)
        b = a + 1_000_000
        sa = index.record(a, PAGE + 512)  # session a, turn 1
        sb = index.record(b, PAGE + 512)  # session b, turn 1 (LRU now a)
        index.begin_step()
        # Session a's next turn (same floor, further) takes a's slot, not b's.
        self.assertEqual(index.record(a, PAGE + 2048), sa)
        index.begin_step()
        self.assertEqual(index.lookup(b, PAGE, PAGE + 1000), (sb, PAGE + 512))
        self.assertEqual(index.lookup(a, PAGE, 2 * PAGE), (sa, PAGE + 2048))
        # A later page: a's tail past the next boundary still replaces a's.
        index.begin_step()
        self.assertEqual(index.record(a, 2 * PAGE + 1024), sa)
        index.begin_step()
        self.assertEqual(index.lookup(b, PAGE, PAGE + 1000), (sb, PAGE + 512))

    def test_a_turn_replaces_the_tail_it_hit_in_the_same_step(self):
        index = TailIndex(2, PAGE, ALIGN)
        a = np.arange(3 * PAGE, dtype=np.int32)
        b = a + 1_000_000
        sa = index.record(a, PAGE + 512)
        sb = index.record(b, PAGE + 512)
        index.begin_step()
        # Session a's next turn hits its tail (pinning it), then records its
        # new tail in the same step: it takes its own tail's slot, not b's.
        self.assertEqual(index.lookup(a, PAGE, PAGE + 1500), (sa, PAGE + 512))
        self.assertEqual(index.record(a, PAGE + 2048, own=sa), sa)
        index.begin_step()
        self.assertEqual(index.lookup(b, PAGE, PAGE + 1000), (sb, PAGE + 512))

    def test_a_tail_another_request_reads_is_not_replaced(self):
        index = TailIndex(3, PAGE, ALIGN)
        a = np.arange(3 * PAGE, dtype=np.int32)
        sa = index.record(a, PAGE + 512)
        index.begin_step()
        # Two requests of one batch hit the same tail; the first one's new
        # tail must not take the slot the second still reads.
        self.assertEqual(index.lookup(a, PAGE, PAGE + 1500), (sa, PAGE + 512))
        self.assertEqual(index.lookup(a, PAGE, PAGE + 1500), (sa, PAGE + 512))
        self.assertNotEqual(index.record(a, PAGE + 2048, own=sa), sa)

    def test_a_diverged_prompt_does_not_replace(self):
        index = TailIndex(2, PAGE, ALIGN)
        a = np.arange(3 * PAGE, dtype=np.int32)
        sa = index.record(a, PAGE + 512)
        other = a.copy()
        other[PAGE + 100] += 1  # rewrites inside a's tail
        index.begin_step()
        self.assertNotEqual(index.record(other, PAGE + 2048), sa)
        index.begin_step()
        self.assertEqual(index.lookup(a, PAGE, PAGE + 1000), (sa, PAGE + 512))

    def test_page_digests_match_one_shot(self):
        from sglang_tpu.k3_programs.sglang_tails import _digest, _page_digests

        tokens = np.arange(3 * PAGE + 77, dtype=np.int32) * 7
        digests = _page_digests(tokens, PAGE, 3 * PAGE)
        self.assertEqual(sorted(digests), [0, PAGE, 2 * PAGE, 3 * PAGE])
        for floor, digest in digests.items():
            self.assertEqual(digest, _digest(tokens[:floor]))


class StubPrefill:
    """Records step inputs; a request finishing its prompt gets token 77."""

    def __init__(self):
        self.inputs = []

    def step(self, step_input):
        self.inputs.append(step_input)
        rows = torch.full((max(len(step_input.requests), 1), 1), -1, dtype=torch.int32)
        for i, req in enumerate(step_input.requests):
            if req.end >= req.num_prompt_tokens:
                rows[i, 0] = 77
        return StepOutput(rows)


def _model():
    return mock.patch(
        "sglang.srt.runtime_context.get_model",
        return_value=SimpleNamespace(kv_cache_dtype="auto"),
    )


class TestPrefillTails(CustomTestCase):
    def setUp(self):
        with (
            _model(),
            mock.patch(
                "sglang.srt.runtime_context.get_schedule",
                return_value=SimpleNamespace(max_mamba_cache_size=4),
            ),
            tpu_envs.SGLANG_TPU_K3_PROGRAM_PREFIX_TAILS.override(2),
        ):
            self.state = sglang_pools.K3StatePool(
                size=4, mamba_layer_ids=KDA, device="cpu"
            )
            self.kv = sglang_pools.K3UnifiedKVPool(
                size=12 * PAGE,
                page_size=PAGE,
                dtype=torch.bfloat16,
                layer_num=len(MLA),
                device="cpu",
            )
        self.plan = self.kv.plan
        self.assertEqual(self.plan.tail_slots, 2)
        self.program = StubPrefill()
        groups = bridge.kv_groups(self.plan, 1 << 20, 128, kda_block_size=ALIGN)
        self.steps = worker.PrefillSteps(
            self.program,
            self.plan,
            table_width=128,
            kda_width=groups[1].table_width,
            tails=TailIndex(2, PAGE, ALIGN),
            kv_pool=self.kv,
            state_pool=self.state,
        )
        self.r2t = np.zeros((4, 1 << 20), np.int32)

    def _req(self, rid, prompt, pool_idx, slot, pages):
        for k, page in enumerate(pages):
            self.r2t[pool_idx, k * PAGE : (k + 1) * PAGE] = np.arange(
                page * PAGE, (page + 1) * PAGE
            )
        return SimpleNamespace(
            rid=rid,
            origin_input_ids=list(prompt),
            sampling_params=SimpleNamespace(
                temperature=1.0, top_p=1.0, top_k=1, sampling_seed=None
            ),
            kv=SimpleNamespace(
                req_pool_idx=pool_idx, mamba_pool_idx=torch.tensor([slot])
            ),
        )

    def _starts(self):
        return bridge.page_starts(self.r2t, self.plan)

    def _block_mean(self, block):
        return float(self.kv.raw_pools()[0][block].float().mean())

    def test_a_tail_kept_and_continued(self):
        length = 2 * PAGE + 3000
        tail_end = length // ALIGN * ALIGN  # 2 * PAGE + 2816
        prompt_a = np.arange(length, dtype=np.int32)
        a = self._req("a", prompt_a, pool_idx=1, slot=1, pages=(1, 2, 3))
        # A's last page holds its last 3000 tokens.
        for pool in self.kv.raw_pools():
            pool[3].fill_(9.0)
        first = self.steps.step([a], [2 * PAGE], [3000], self._starts())
        self.assertEqual(first.tolist(), [77])
        cut, rest = (
            self.program.inputs[-2].requests[0],
            self.program.inputs[-1].requests[0],
        )
        self.assertEqual((cut.computed, cut.end), (2 * PAGE, tail_end))
        self.assertEqual((rest.computed, rest.end), (tail_end, length))
        # The cut names the tail's state blocks where it ends.
        tail = 0
        column = (tail_end - 1) // ALIGN
        self.assertEqual(
            [int(cut.block_tables[1 + g][column]) for g in range(3)],
            [self.plan.tail_state_block(tail, g) for g in range(3)],
        )
        self.assertEqual(
            int(rest.block_tables[1][(length - 1) // ALIGN]),
            self.plan.state_block(1, 0),
        )
        # The partial page went to the tail's KV block, in every pool.
        self.assertEqual(self._block_mean(self.plan.tail_kv_block(tail)), 9.0)
        self.assertEqual(
            float(
                self.kv.raw_pools()[-1][self.plan.tail_kv_block(tail)].float().mean()
            ),
            9.0,
        )

        # B continues A: radix hit at the page boundary (2 * PAGE), then A's tail.
        prompt_b = np.concatenate([prompt_a, np.arange(4000, dtype=np.int32) + 500_000])
        b = self._req("b", prompt_b, pool_idx=2, slot=2, pages=(1, 2, 5))
        for pool in self.kv.raw_pools():
            pool[5].fill_(0.0)
        self.steps.step([b], [2 * PAGE], [len(prompt_b) - 2 * PAGE], self._starts())
        # B's own prompt end is cut and kept too: its first call starts at A's tail.
        row = self.program.inputs[-2].requests[0]
        self.assertEqual(row.computed, tail_end)
        self.assertEqual(
            self.program.inputs[-1].requests[0].computed,
            len(prompt_b) // ALIGN * ALIGN,
        )
        self.assertEqual(self.steps.stats["tail_hit_tokens"], tail_end - 2 * PAGE)
        # B's new page holds a copy of the tail's block; its hit column names the tail's state.
        self.assertEqual(self._block_mean(5), 9.0)
        self.assertEqual(
            [int(row.block_tables[1 + g][tail_end // ALIGN - 1]) for g in range(3)],
            [self.plan.tail_state_block(tail, g) for g in range(3)],
        )

    def test_an_aligned_prompt_end_copies_the_final_state(self):
        length = PAGE + 4 * ALIGN
        prompt = np.arange(length, dtype=np.int32)
        a = self._req("a", prompt, pool_idx=1, slot=3, pages=(1, 2))
        for pool in self.state.host_pools():
            for g in range(3):
                pool[self.plan.state_block(3, g)].fill_(4.0)
        self.steps.step([a], [0], [length], self._starts())
        # One step (nothing to cut), then the slot's state copied into the tail's.
        self.assertEqual(len(self.program.inputs), 1)
        for pool in self.state.host_pools():
            self.assertEqual(
                float(pool[self.plan.tail_state_block(0, 1)].float().mean()), 4.0
            )

    def test_a_prompt_inside_its_first_alignment_keeps_no_tail(self):
        prompt = np.arange(PAGE + 100, dtype=np.int32)
        a = self._req("a", prompt, pool_idx=1, slot=1, pages=(1, 2))
        self.steps.step([a], [0], [len(prompt)], self._starts())
        self.assertEqual(len(self.program.inputs), 1)


if __name__ == "__main__":
    unittest.main()
