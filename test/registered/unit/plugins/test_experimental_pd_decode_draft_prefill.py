"""SGLANG_TPU_PD_DECODE_DRAFT_PREFILL: the hidden rows a PD prefill ships and the decode reads."""

from __future__ import annotations

import contextlib
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.model_executor.model_runner_components.spec_aux_hidden_state import (
    SpecAuxHiddenStateConfig,
)
from sglang.srt.model_executor.pool_configurator import (
    DefaultPoolConfigurator,
    HybridSWAPoolConfigurator,
    SWAChunkCapPoolConfigurator,
)
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.disaggregation.raiden import common as raiden_common
from sglang_tpu.disaggregation.raiden import pipeline as raiden_pipeline
from sglang_tpu.disaggregation.raiden import pp_common, pp_swa
from sglang_tpu.envs import envs as tpu_envs
from sglang_tpu.experimental import runs_draft_prompt_pass
from sglang_tpu.experimental.speculative import pd_draft_prefill, pd_hidden_pool
from sglang_tpu.experimental.speculative.block_spec_worker import TpuBlockSpecWorker
from sglang_tpu.experimental.speculative.pd_draft_prefill import (
    DraftPromptChunk,
    prebuilt_extend_groups,
    prebuilt_extend_view,
    swa_mapped_starts,
)
from sglang_tpu.experimental.speculative.pd_hidden_pool import (
    PdHiddenPool,
    PdHiddenWindow,
    drop_rows_before,
    hidden_lanes,
    swa_window_start,
)
from sglang_tpu.hooks.experimental_spec import budget_pd_hidden_pool, hook_specs

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

KV_HEADS = 8
PAGES, PAGE = 4, 4
FEATURE = 3 * 160


def _pool(
    attn_tp_size: int,
    group=None,
    *,
    num_pages: int = PAGES,
    paged_by_swa: bool = False,
) -> PdHiddenPool:
    return PdHiddenPool(
        num_pages=num_pages,
        page_size=PAGE,
        feature_size=FEATURE,
        total_kv_heads=KV_HEADS,
        attn_tp_size=attn_tp_size,
        attn_tp_group=group,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
        paged_by_swa=paged_by_swa,
    )


class _FakeGroup:
    """all_gather over ranks whose local tensors are given up front."""

    def __init__(self, per_rank: list[torch.Tensor]) -> None:
        self.per_rank = per_rank

    def all_gather(self, tensor: torch.Tensor, dim: int) -> torch.Tensor:
        return torch.cat(self.per_rank, dim=dim)


class TestPdHiddenPool(CustomTestCase):
    def _shipped(self, slots: list[int]) -> tuple[PdHiddenPool, torch.Tensor]:
        prefill = _pool(attn_tp_size=1)
        hidden = torch.randn(len(slots) + 2, FEATURE).to(torch.bfloat16)
        segments = prefill.write_segments(torch.tensor(slots), hidden.shape[0])
        prefill.write(hidden, segments=segments)
        return prefill, hidden[: len(slots)]

    def test_bytes_per_token_is_the_buffer_row(self):
        for attn_tp_size in (1, 8):
            pool = _pool(attn_tp_size=attn_tp_size)
            row = pool.buffer[0, 0]
            self.assertEqual(
                PdHiddenPool.bytes_per_token(
                    feature_size=FEATURE,
                    total_kv_heads=KV_HEADS,
                    attn_tp_size=attn_tp_size,
                    dtype=torch.bfloat16,
                ),
                row.numel() * row.element_size(),
            )

    def test_lanes_hold_a_row_in_head_slices(self):
        lanes = hidden_lanes(feature_size=FEATURE, total_kv_heads=KV_HEADS)
        self.assertEqual(lanes % 128, 0)
        self.assertGreaterEqual(KV_HEADS * 2 * lanes, FEATURE)

    def test_prefill_rows_round_trip_without_padding_rows(self):
        """Bucket padding rows past the real tokens must not land in slot 0."""
        slots = [5, 6, 7, 9]
        prefill, hidden = self._shipped(slots)
        self.assertTrue(torch.equal(prefill.read(torch.tensor(slots)), hidden))
        self.assertEqual(prefill.buffer.flatten(0, 1)[0].abs().sum().item(), 0.0)

    def test_rows_mapped_to_the_padding_slot_are_not_written(self):
        """A windowed draft maps tokens left of the window to SWA slot 0."""
        prefill = _pool(attn_tp_size=1)
        hidden = torch.randn(5, FEATURE).to(torch.bfloat16)
        slots = torch.tensor([0, 0, 6, 7, 9])
        prefill.write(hidden, segments=prefill.write_segments(slots, 5))
        self.assertEqual(prefill.buffer.flatten(0, 1)[0].abs().sum().item(), 0.0)
        torch.testing.assert_close(prefill.read(slots[2:]), hidden[2:])

    def test_tp_decode_joins_the_head_slices_raiden_hands_each_rank(self):
        """Raiden gives decode rank r kv head r of every row; the read joins them."""
        slots = [1, 2, 3, 12]
        prefill, hidden = self._shipped(slots)
        tp = 8
        loc = torch.tensor(slots)
        locals_ = [
            prefill.buffer[:, :, rank : rank + 1]
            .flatten(0, 1)[loc]
            .reshape(len(slots), -1)
            for rank in range(tp)
        ]
        group = _FakeGroup([t.unsqueeze(0) for t in locals_])
        decode = _pool(attn_tp_size=tp, group=group)
        self.assertEqual(decode.buffer.shape[2], 1)
        self.assertTrue(torch.equal(decode.read(loc), hidden))


class TestRaidenCarriesHiddenNotDraftKv(CustomTestCase):
    def test_hidden_pool_replaces_the_draft_pool_in_its_own_class(self):
        target = SimpleNamespace(
            start_layer=0,
            layer_num=2,
            layers_mapping=None,
            get_fused_kv_buffer=lambda layer: torch.zeros(PAGES, PAGE, 8, 2, 128),
        )
        draft = SimpleNamespace(
            start_layer=0,
            layer_num=1,
            layers_mapping=None,
            get_fused_kv_buffer=lambda layer: torch.zeros(PAGES, PAGE, 8, 2, 128),
        )
        hidden = _pool(attn_tp_size=1)
        with patch.object(pd_hidden_pool, "_pd_hidden_pool", hidden):
            groups = raiden_common.get_raiden_kv_caches(target, draft_pool=draft)
        self.assertEqual(groups.num_draft_full, 0)
        self.assertEqual(groups.num_hidden_full, 1)
        self.assertIs(groups.full[-1], hidden.buffer)
        tags = [pool["tag"] for pool in raiden_pipeline.pool_manifest(groups)]
        self.assertEqual(tags, ["fa.l0", "fa.l1", "pdhid.fa.l0"])

    def test_stages_without_the_hidden_pool_fingerprint_alike(self):
        """Only the last prefill stage and the decode hold the hidden pool."""
        target = SimpleNamespace(
            start_layer=0,
            layer_num=1,
            layers_mapping=None,
            get_fused_kv_buffer=lambda layer: torch.zeros(PAGES, PAGE, 8, 2, 128),
        )
        layout = lambda tensor: ((4, 3, 2, 1, 0), ((2, 128),), 16)  # noqa: E731
        with patch.object(pd_hidden_pool, "_pd_hidden_pool", _pool(attn_tp_size=1)):
            with_hidden = raiden_common.get_raiden_kv_caches(target, draft_pool=target)
        without = raiden_common.get_raiden_kv_caches(target)
        fingerprints = [
            raiden_pipeline.layout_fingerprint(
                raiden_pipeline.pool_manifest(groups),
                groups.full,
                layout,
                ships_pd_hidden=False,
                seq_along_lane=False,
            )
            for groups in (with_hidden, without)
        ]
        self.assertEqual(fingerprints[0], fingerprints[1])

    def test_ends_disagreeing_on_the_flag_fingerprint_apart(self):
        groups = SimpleNamespace(
            full=[torch.zeros(PAGES, PAGE, 8, 2, 128)],
            full_layer_ids=[0],
            num_draft_full=0,
            num_hidden_full=0,
            num_hidden_swa=0,
            swa=[],
            swa_layer_ids=[],
            num_draft_swa=0,
            seq_along_lane=False,
        )
        layout = lambda tensor: ((4, 3, 2, 1, 0), ((2, 128),), 16)  # noqa: E731
        pools = raiden_pipeline.pool_manifest(groups)
        self.assertNotEqual(
            raiden_pipeline.layout_fingerprint(
                pools, groups.full, layout, ships_pd_hidden=True, seq_along_lane=False
            ),
            raiden_pipeline.layout_fingerprint(
                pools, groups.full, layout, ships_pd_hidden=False, seq_along_lane=False
            ),
        )

    def test_without_the_flag_the_draft_pool_travels(self):
        target = SimpleNamespace(
            start_layer=0,
            layer_num=1,
            layers_mapping=None,
            get_fused_kv_buffer=lambda layer: torch.zeros(PAGES, PAGE, 8, 2, 128),
        )
        with patch.object(pd_hidden_pool, "_pd_hidden_pool", None):
            groups = raiden_common.get_raiden_kv_caches(target, draft_pool=target)
        self.assertEqual(groups.num_hidden_full, 0)


SWA_PAGES = 2


def _hybrid_target() -> SimpleNamespace:
    """Layer 0 full attention on PAGES pages, layer 1 sliding window on SWA_PAGES."""
    return SimpleNamespace(
        start_layer=0,
        layer_num=2,
        layers_mapping={0: (0, False), 1: (0, True)},
        get_fused_kv_buffer=lambda layer: torch.zeros(
            SWA_PAGES if layer else PAGES, PAGE, 8, 2, 128
        ),
    )


class TestRaidenMovesWindowedHiddenWithTheWindow(CustomTestCase):
    def setUp(self):
        super().setUp()
        hidden = _pool(attn_tp_size=1, num_pages=SWA_PAGES, paged_by_swa=True)
        with patch.object(pd_hidden_pool, "_pd_hidden_pool", hidden):
            target = _hybrid_target()
            self.groups = raiden_common.get_raiden_kv_caches(target, draft_pool=target)
        self.hidden = hidden
        self.pools = raiden_pipeline.pool_manifest(self.groups)

    def test_the_hidden_pool_joins_the_swa_group(self):
        self.assertEqual(
            (self.groups.num_hidden_full, self.groups.num_hidden_swa), (0, 1)
        )
        self.assertIs(self.groups.swa[-1], self.hidden.buffer)
        self.assertEqual(
            [pool["tag"] for pool in self.pools], ["fa.l0", "swa.l1", "pdhid.swa.l0"]
        )
        full, swa = pp_swa.split_manifest(self.pools)
        self.assertEqual([p["tag"] for p in full], ["fa.l0"])
        self.assertEqual([p["tag"] for p in swa], ["swa.l1", "pdhid.swa.l0"])

    def test_it_moves_the_window_pages(self):
        full, window = [1, 2, 3], [1]
        spans = pp_common.part_spans(
            self.pools,
            self.groups.full + self.groups.swa,
            full,
            window,
            1,
            KV_HEADS,
            seq_along_lane=False,
            with_swa=True,
        )
        self.assertEqual(spans[-1].tag, "pdhid.swa.l0")
        self.assertEqual(spans[-1].block_ids, window)
        dst = pp_common.part_destination_blocks(
            self.pools, [3, 2, 1], [1], start=0, count=3, source_swa_count=1
        )
        self.assertEqual(dst["transfer_pool_tags"], ["fa.l0", "swa.l1", "pdhid.swa.l0"])
        self.assertEqual(dst["dst_block_counts"], [3, 1, 1])
        # Early parts carry no window pages, so no hidden rows either.
        early = pp_common.part_spans(
            self.pools,
            self.groups.full + self.groups.swa,
            full,
            window,
            1,
            KV_HEADS,
            seq_along_lane=False,
            with_swa=False,
        )
        self.assertEqual([entry.tag for entry in early], ["fa.l0"])

    def test_a_tp_decode_slices_it_by_the_target_heads(self):
        (entry,) = pp_common.head_slice_spans(
            self.pools[-1:],
            [self.hidden.buffer],
            [3],
            [1],
            KV_HEADS,
            KV_HEADS,
            seq_along_lane=False,
        )
        self.assertEqual(entry.block_ids, [1])
        row = self.hidden.buffer[0, 0, 0].numel() * 2
        self.assertEqual({s.size_bytes for s in entry.spans}, {row})
        self.assertEqual(
            sorted(s.dst_unit_ordinal for s in entry.spans), list(range(KV_HEADS))
        )

    def test_a_page_count_off_the_swa_pool_is_rejected(self):
        hidden = _pool(attn_tp_size=1, num_pages=PAGES, paged_by_swa=True)
        with (
            patch.object(pd_hidden_pool, "_pd_hidden_pool", hidden),
            self.assertRaisesRegex(RuntimeError, "SWA KV block counts"),
        ):
            target = _hybrid_target()
            raiden_common.get_raiden_kv_caches(target, draft_pool=target)


def _runner(kv_pool, **aux) -> SimpleNamespace:
    return SimpleNamespace(
        token_to_kv_pool=kv_pool,
        spec_aux_config=SpecAuxHiddenStateConfig(
            eagle_use_aux_hidden_state=True,
            eagle_aux_hidden_state_layer_ids=[1, 2, 3],
            **aux,
        ),
        model_config=_kvc().model_config,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
    )


class _FakeSwaPool(pd_draft_prefill.SWAKVPool):
    def __init__(self, page_size: int) -> None:
        self.page_size = page_size


class TestBuildHiddenPool(CustomTestCase):
    def _build(self, runner, groups, *, paged_by_swa: bool) -> PdHiddenPool:
        with (
            patch.object(pd_draft_prefill, "get_raiden_kv_caches", lambda _: groups),
            get_parallel().override(attn_tp_size=1),
        ):
            return pd_draft_prefill.build_pd_hidden_pool(
                runner, paged_by_swa=paged_by_swa
            )

    def test_pages_follow_a_seq_along_lane_pool(self):
        # [num_blocks, 2 * kv_heads, head_dim words, packing, page_size].
        sal_layer = torch.zeros((5, 2 * KV_HEADS, 4, 2, 256), dtype=torch.uint8)
        runner = _runner(SimpleNamespace(page_size=256))
        groups = SimpleNamespace(full=[sal_layer], swa=[])
        pool = self._build(runner, groups, paged_by_swa=False)
        self.assertEqual(tuple(pool.buffer.shape[:2]), (5, 256))
        self.assertFalse(pool.paged_by_swa)

    def test_a_windowed_draft_pages_by_the_swa_pool(self):
        groups = SimpleNamespace(
            full=[torch.zeros((40, 256, 1, 2, 128))],
            swa=[torch.zeros((3, 256, 1, 2, 128))],
        )
        runner = _runner(_FakeSwaPool(256), dflash_draft_kv_is_swa=True)
        pool = self._build(runner, groups, paged_by_swa=True)
        self.assertEqual(tuple(pool.buffer.shape[:2]), (3, 256))
        self.assertTrue(pool.paged_by_swa)

    def test_paging_that_disagrees_with_the_budget_is_rejected(self):
        groups = SimpleNamespace(
            full=[torch.zeros((40, 256, 1, 2, 128))],
            swa=[torch.zeros((3, 256, 1, 2, 128))],
        )
        for paged_by_swa, draft_kv_is_swa in ((True, False), (False, True)):
            runner = _runner(_FakeSwaPool(256), dflash_draft_kv_is_swa=draft_kv_is_swa)
            with self.assertRaisesRegex(RuntimeError, "dflash_draft_kv_is_swa"):
                self._build(runner, groups, paged_by_swa=paged_by_swa)

    def test_a_stage_without_swa_pages_is_rejected(self):
        groups = SimpleNamespace(full=[torch.zeros((40, 256, 1, 2, 128))], swa=[])
        runner = _runner(SimpleNamespace(page_size=256))
        with self.assertRaisesRegex(RuntimeError, "SWA KV"):
            self._build(runner, groups, paged_by_swa=True)


class TestDraftPromptPassPlacement(CustomTestCase):
    def test_flag_moves_the_prompt_pass_from_prefill_to_decode(self):
        for flag, prefill, decode in ((False, True, False), (True, False, True)):
            with tpu_envs.SGLANG_TPU_PD_DECODE_DRAFT_PREFILL.override(flag):
                self.assertEqual(runs_draft_prompt_pass("prefill"), prefill)
                self.assertEqual(runs_draft_prompt_pass("decode"), decode)
                self.assertTrue(runs_draft_prompt_pass("null"))

    def test_groups_pack_requests_under_the_bucket_and_cap(self):
        lens = [3000, 3000, 3000, 100, 100, 100]

        def reqs(groups):
            return [[chunk.req for chunk in group] for group in groups]

        groups = prebuilt_extend_groups(lens, [0] * 6, max_tokens=8192, max_reqs=2)
        self.assertEqual(reqs(groups), [[0, 1], [2, 3], [4, 5]])
        groups = prebuilt_extend_groups(lens, [0] * 6, max_tokens=6000, max_reqs=8)
        self.assertEqual(reqs(groups), [[0, 1], [2, 3, 4, 5]])

    def test_a_long_prompt_runs_in_bucket_chunks(self):
        groups = prebuilt_extend_groups(
            [100, 19000, 100], [0, 0, 0], max_tokens=8192, max_reqs=8
        )
        chunk = DraftPromptChunk
        self.assertEqual(
            groups,
            [
                [chunk(req=0, start=0, length=100)],
                [chunk(req=1, start=0, length=8192)],
                [chunk(req=1, start=8192, length=8192)],
                # The tail packs with the next request.
                [chunk(req=1, start=16384, length=2616), chunk(2, 0, 100)],
            ],
        )

    def test_starts_skip_the_head_of_a_prompt(self):
        groups = prebuilt_extend_groups([20, 5], [12, 0], max_tokens=8, max_reqs=8)
        self.assertEqual(
            groups,
            [
                [DraftPromptChunk(req=0, start=12, length=8)],
                [DraftPromptChunk(req=1, start=0, length=5)],
            ],
        )

    def test_swa_mapped_starts_take_the_mapped_suffix(self):
        swa_loc = torch.tensor([0, 0, 7, 8, 9, 3, 4, 0, 5])
        self.assertEqual(swa_mapped_starts(swa_loc, [5, 2]), [2, 0])
        with self.assertRaisesRegex(RuntimeError, "holes"):
            swa_mapped_starts(swa_loc, [5, 4])

    def test_no_admission_hook_aborts_long_prompts(self):
        targets = [target for target, _, _ in hook_specs()]
        self.assertFalse(
            [target for target in targets if "exceed_kv_capacity" in target]
        )


def _prebuilt_batch(prefix_lens: list[int], extend_lens: list[int], slots):
    reqs = [
        SimpleNamespace(
            origin_input_ids=list(range(1000 * i, 1000 * i + prefix + extend)),
            output_ids=[90 + i],
            multimodal_inputs=None,
        )
        for i, (prefix, extend) in enumerate(zip(prefix_lens, extend_lens))
    ]
    return SimpleNamespace(
        reqs=reqs,
        prefix_lens=prefix_lens,
        extend_lens=extend_lens,
        out_cache_loc=torch.tensor(slots, dtype=torch.int64),
        req_pool_indices_cpu=torch.arange(1, len(reqs) + 1),
        seq_lens_cpu=torch.tensor(
            [p + e for p, e in zip(prefix_lens, extend_lens)], dtype=torch.int64
        ),
    )


class _Bucketer:
    def __init__(self, buckets: list[int]) -> None:
        self.buckets = buckets
        self.max_val = buckets[-1]

    def next_bucket(self, n: int) -> int:
        return next(b for b in self.buckets if b >= n)


class TestChunkedDraftPromptPass(CustomTestCase):
    """The decode draft pass over prompts longer than the largest extend bucket."""

    BUCKETS, CAP = [4, 8], 2
    PREFIX, EXTEND = [2, 0, 0], [19, 5, 6]

    def setUp(self):
        super().setUp()
        sampling = patch.object(
            pd_draft_prefill.SamplingBatchInfo,
            "from_schedule_batch",
            lambda batch, vocab_size: None,
        )
        sampling.start()
        self.addCleanup(sampling.stop)
        # Slot 0 is the padding slot; real slots are shuffled and nonzero.
        self.slots = (torch.randperm(63)[: sum(self.EXTEND)] + 1).tolist()
        self.batch = _prebuilt_batch(self.PREFIX, self.EXTEND, self.slots)

    def _worker(self, full_to_swa: torch.Tensor | None):
        hidden_pool = PdHiddenPool(
            num_pages=16,
            page_size=4,
            feature_size=FEATURE,
            total_kv_heads=KV_HEADS,
            attn_tp_size=1,
            attn_tp_group=None,
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
            paged_by_swa=full_to_swa is not None,
        )
        hidden_pool.buffer.copy_(torch.randn(hidden_pool.buffer.shape))
        t_bucketer = _Bucketer(self.BUCKETS)
        policy = SimpleNamespace(
            t_bucketer=t_bucketer,
            b_bucketer=_Bucketer([1, 2, 4]),
            _extend_compile_cap=lambda: self.CAP,
            live_cache_loc_dtype=lambda: torch.int64,
        )
        self.writes: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

        def write_draft_kv(*, target_hidden, positions, cache_loc):
            self.assertIn(target_hidden.shape[0], self.BUCKETS)
            self.writes.append((target_hidden, positions, cache_loc))

        worker = object.__new__(TpuBlockSpecWorker)
        worker.device = "cpu"
        worker._pd_hidden_pool = hidden_pool
        worker._pd_hidden_window = None
        worker._compile_policy = lambda: policy
        worker._target_worker = SimpleNamespace(
            model_runner=SimpleNamespace(model_config=SimpleNamespace(vocab_size=64))
        )
        worker._block_draft_worker = SimpleNamespace(
            full_to_swa=full_to_swa,
            draft_context=contextlib.nullcontext,
            write_draft_kv=write_draft_kv,
        )
        return worker

    def _written(self) -> dict[int, tuple[int, torch.Tensor]]:
        """Draft slot -> (position, hidden row) over every chunk, padding dropped."""
        written = {}
        for hidden, positions, cache_loc in self.writes:
            for row, slot in enumerate(cache_loc.tolist()):
                if slot != 0:
                    self.assertNotIn(slot, written)
                    written[slot] = (int(positions[row]), hidden[row])
        return written

    def _expected(self, worker, draft_slot) -> dict[int, tuple[int, torch.Tensor]]:
        """The single-pass result: each prompt token's row, read and written at its
        draft slot (the full slot, or the SWA slot of a windowed draft)."""
        expected, offset = {}, 0
        for prefix, extend in zip(self.PREFIX, self.EXTEND):
            for i in range(extend):
                slot = draft_slot(self.slots[offset + i])
                if slot != 0:
                    row = worker._pd_hidden_pool.read(torch.tensor([slot]))[0]
                    expected[slot] = (prefix + i, row)
            offset += extend
        return expected

    def _assert_same(self, written, expected):
        self.assertEqual(sorted(written), sorted(expected))
        for slot, (position, row) in expected.items():
            self.assertEqual(written[slot][0], position)
            torch.testing.assert_close(written[slot][1], row)

    def test_chunks_cover_every_prompt_token_once(self):
        worker = self._worker(full_to_swa=None)
        worker.prefill_draft_from_pd(self.batch)
        # 19 tokens at bucket 8: 8, 8, then the 3-token tail with request 1.
        self.assertEqual(len(self.writes), 4)
        self._assert_same(self._written(), self._expected(worker, lambda s: s))

    def _window_map(self) -> torch.Tensor:
        full_to_swa = torch.zeros(64, dtype=torch.int64)
        # The decode maps request 0's last 4 tokens and requests 1 and 2 whole.
        mapped = self.slots[15:]
        full_to_swa[torch.tensor(mapped)] = torch.randperm(63)[: len(mapped)] + 1
        return full_to_swa

    def test_a_windowed_draft_reads_and_writes_the_swa_tail(self):
        full_to_swa = self._window_map()
        worker = self._worker(full_to_swa=full_to_swa)
        worker.prefill_draft_from_pd(self.batch)
        self._assert_same(
            self._written(), self._expected(worker, lambda s: int(full_to_swa[s]))
        )

    def test_the_prefill_writes_rows_the_decode_reads_at_swa_slots(self):
        full_to_swa = self._window_map()
        prefill = self._worker(full_to_swa=full_to_swa)
        prefill._pd_hidden_pool.buffer.zero_()
        hidden = torch.randn(32, FEATURE).to(torch.bfloat16)
        prefill.write_pd_hidden(self.batch, hidden)
        mapped = full_to_swa[self.batch.out_cache_loc]
        torch.testing.assert_close(
            prefill._pd_hidden_pool.read(mapped[15:]), hidden[15:30]
        )
        # Tokens left of the window map to slot 0 and are dropped.
        flat = prefill._pd_hidden_pool.buffer.flatten(0, 1)
        self.assertEqual(flat[0].abs().sum().item(), 0.0)
        decode = self._worker(full_to_swa=full_to_swa)
        decode._pd_hidden_pool.buffer.copy_(prefill._pd_hidden_pool.buffer)
        decode.prefill_draft_from_pd(self.batch)
        written = self._written()
        self.assertEqual(sorted(written), sorted(mapped[15:].tolist()))
        for row, slot in enumerate(mapped[15:].tolist()):
            torch.testing.assert_close(written[slot][1], hidden[15 + row])

    def test_a_pool_paged_unlike_the_draft_is_rejected(self):
        worker = self._worker(full_to_swa=None)
        worker._block_draft_worker.full_to_swa = self._window_map()
        with self.assertRaisesRegex(RuntimeError, "full-to-SWA map appeared"):
            worker.prefill_draft_from_pd(self.batch)
        worker = self._worker(full_to_swa=self._window_map())
        worker._block_draft_worker.full_to_swa = None
        with self.assertRaisesRegex(RuntimeError, "full-to-SWA map is missing"):
            worker.write_pd_hidden(self.batch, torch.zeros(32, FEATURE))

    def test_a_short_chunk_is_the_views_chunked_request(self):
        chunks = [DraftPromptChunk(req=1, start=0, length=5)]
        chunks.append(DraftPromptChunk(req=0, start=0, length=3))
        view = prebuilt_extend_view(self.batch, chunks, vocab_size=64)
        self.assertEqual(view.prefix_lens, [0, 2])
        self.assertEqual(view.seq_lens_cpu.tolist(), [5, 5])
        self.assertEqual(
            view.input_ids.tolist(), [1000, 1001, 1002, 1003, 1004, 2, 3, 4]
        )
        self.assertEqual(
            view.out_cache_loc.tolist(), self.slots[19:24] + self.slots[:3]
        )
        self.assertEqual(view.req_pool_indices_cpu.tolist(), [2, 1])
        self.assertIs(view.chunked_req, self.batch.reqs[0])
        self.assertEqual(view.chunked_req_next_prompt_token, 5)
        tail = [DraftPromptChunk(req=0, start=16, length=3)]
        view = prebuilt_extend_view(self.batch, tail, vocab_size=64)
        self.assertEqual(view.input_ids.tolist(), [18, 19, 20])
        self.assertIsNone(view.chunked_req)
        self.assertIsNone(view.chunked_req_next_prompt_token)


def _kvc(**overrides):
    fields = dict(
        is_draft_worker=False,
        spec_algorithm=SpeculativeAlgorithm.EAGLE3,
        spec_aux_config=SpecAuxHiddenStateConfig(
            eagle_use_aux_hidden_state=True, eagle_aux_hidden_state_layer_ids=[1, 2, 3]
        ),
        model_config=SimpleNamespace(
            hidden_size=160, get_total_num_kv_heads=lambda: KV_HEADS
        ),
        model_dtype=torch.bfloat16,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


class TestPrefillWritesOnlyTheReadWindow(CustomTestCase):
    """SGLANG_TPU_ENABLE_PD_HIDDEN_WINDOW_WRITE: rows the decode never reads are not written."""

    WINDOW = PdHiddenWindow(sliding_window_size=8, page_size=4, every_chunk_end=False)

    def test_window_start_matches_the_decode_swa_tail(self):
        # decode _swa_tail_len(seq_len) == seq_len - window_start.
        for seq_len, start in [(0, 0), (5, 0), (8, 0), (11, 0), (12, 4), (42, 32)]:
            self.assertEqual(
                swa_window_start(seq_len, sliding_window_size=8, page_size=4), start
            )

    def test_a_chunk_left_of_the_prompt_window_writes_nothing(self):
        # Prompt of 40: window starts at 32; the chunk covers [0, 16).
        self.assertEqual(self.WINDOW.write_starts([0], [16], [40]), [16])
        # A chunk ending exactly at the cutoff is still entirely left of it.
        self.assertEqual(self.WINDOW.write_starts([16], [16], [40]), [16])

    def test_a_chunk_straddling_the_cutoff_writes_from_it(self):
        self.assertEqual(self.WINDOW.write_starts([24], [12], [40]), [8])
        # Page-aligned: a 42-token prompt's window starts at 32, not 34.
        self.assertEqual(self.WINDOW.write_starts([24], [18], [42]), [8])

    def test_a_chunk_inside_the_window_writes_every_row(self):
        self.assertEqual(self.WINDOW.write_starts([32], [8], [40]), [0])
        # A prompt shorter than the window is read whole.
        self.assertEqual(self.WINDOW.write_starts([0], [5], [5]), [0])

    def test_each_request_cuts_at_its_own_prompt_window(self):
        starts = self.WINDOW.write_starts(
            [0, 24, 0, 2], [16, 12, 5, 19], [40, 40, 5, 21]
        )
        self.assertEqual(starts, [16, 8, 0, 10])

    def test_prefix_sharing_keeps_the_window_before_each_chunk_end(self):
        window = PdHiddenWindow(
            sliding_window_size=8, page_size=4, every_chunk_end=True
        )
        # A later request matching at this chunk's end (16) reads [8, 16).
        self.assertEqual(window.write_starts([0], [16], [40]), [8])
        # The prompt end's window is kept too: it is the last chunk end.
        self.assertEqual(window.write_starts([24], [16], [40]), [8])
        # A chunk shorter than the window is written whole.
        self.assertEqual(window.write_starts([16, 0], [4, 3], [40, 3]), [0, 0])

    def test_dropped_rows_go_to_the_padding_slot(self):
        slots = torch.arange(1, 11)
        kept = drop_rows_before(slots, [4, 2, 4], [3, 0, 4])
        self.assertEqual(kept.tolist(), [0, 0, 0, 4, 5, 6, 0, 0, 0, 0])
        self.assertEqual(slots.tolist(), list(range(1, 11)))

    def _prefill(self, every_chunk_end: bool) -> tuple[TpuBlockSpecWorker, object]:
        case = TestChunkedDraftPromptPass()
        case.setUp()
        self.addCleanup(case.doCleanups)
        # Before eviction every prefill token has an SWA slot.
        full_to_swa = torch.arange(64, dtype=torch.int64)
        worker = case._worker(full_to_swa=full_to_swa)
        worker._pd_hidden_pool.buffer.zero_()
        worker._pd_hidden_window = PdHiddenWindow(
            sliding_window_size=4, page_size=4, every_chunk_end=every_chunk_end
        )
        return worker, case.batch

    def test_the_prefill_writes_only_window_rows_and_no_padding(self):
        worker, batch = self._prefill(every_chunk_end=False)
        # 30 live rows of [2+19, 5, 6]-token prompts, padded to the 32-row bucket.
        hidden = torch.randn(32, FEATURE).to(torch.bfloat16) + 1
        worker.write_pd_hidden(batch, hidden)
        # Request 0's window starts at 16 (offset 14); requests 1 and 2 fit the window.
        written = list(range(14, 19)) + list(range(19, 30))
        pool = worker._pd_hidden_pool
        torch.testing.assert_close(
            pool.read(batch.out_cache_loc[written]), hidden[written]
        )
        flat = pool.buffer.flatten(0, 1)
        skipped = batch.out_cache_loc[:14]
        self.assertEqual(flat[skipped].abs().sum().item(), 0.0)
        self.assertEqual(flat[0].abs().sum().item(), 0.0)
        self.assertEqual(int((flat.abs().sum((1, 2, 3)) != 0).sum()), len(written))

    def test_a_mid_prompt_chunk_left_of_the_window_writes_nothing(self):
        worker, batch = self._prefill(every_chunk_end=False)
        # Request 0's prompt continues long past this chunk.
        batch.reqs[0].origin_input_ids = list(range(100))
        worker.write_pd_hidden(batch, torch.randn(32, FEATURE).to(torch.bfloat16) + 1)
        flat = worker._pd_hidden_pool.buffer.flatten(0, 1)
        self.assertEqual(flat[batch.out_cache_loc[:19]].abs().sum().item(), 0.0)
        self.assertNotEqual(flat[batch.out_cache_loc[19:]].abs().sum().item(), 0.0)

    def test_prefix_sharing_writes_the_window_before_the_chunk_end(self):
        worker, batch = self._prefill(every_chunk_end=True)
        batch.reqs[0].origin_input_ids = list(range(100))
        hidden = torch.randn(32, FEATURE).to(torch.bfloat16) + 1
        worker.write_pd_hidden(batch, hidden)
        # This chunk ends request 0 at 21, whose window starts at 16 (offset 14).
        pool = worker._pd_hidden_pool
        torch.testing.assert_close(pool.read(batch.out_cache_loc[14:30]), hidden[14:30])
        flat = pool.buffer.flatten(0, 1)
        self.assertEqual(flat[batch.out_cache_loc[:14]].abs().sum().item(), 0.0)

    def _init_window(self, *, paged_by_swa: bool, ships: bool, enabled: bool):
        worker = object.__new__(TpuBlockSpecWorker)
        worker._pd_hidden_pool = SimpleNamespace(paged_by_swa=paged_by_swa)
        worker._target_worker = SimpleNamespace(
            model_runner=SimpleNamespace(
                sliding_window_size=2048,
                token_to_kv_pool_allocator=SimpleNamespace(page_size=256),
            )
        )
        memory = SimpleNamespace(disable_radix_cache=False)
        with (
            patch.object(
                TpuBlockSpecWorker,
                "ships_pd_hidden",
                new_callable=lambda: property(lambda self: ships),
            ),
            patch(
                "sglang_tpu.experimental.speculative.base_spec_worker.get_memory",
                return_value=memory,
            ),
            tpu_envs.SGLANG_TPU_ENABLE_PD_HIDDEN_WINDOW_WRITE.override(enabled),
        ):
            return worker._init_pd_hidden_window()

    def test_only_an_swa_paged_prefill_trims_its_writes(self):
        self.assertEqual(
            self._init_window(paged_by_swa=True, ships=True, enabled=True),
            PdHiddenWindow(
                sliding_window_size=2048, page_size=256, every_chunk_end=True
            ),
        )
        self.assertIsNone(
            self._init_window(paged_by_swa=False, ships=True, enabled=True)
        )
        self.assertIsNone(
            self._init_window(paged_by_swa=True, ships=False, enabled=True)
        )
        self.assertIsNone(
            self._init_window(paged_by_swa=True, ships=True, enabled=False)
        )


def _configurator(cls, cell_size: int = 1000):
    configurator = object.__new__(cls)
    configurator._cell_size = cell_size
    return configurator


class TestHiddenPoolKvBudget(CustomTestCase):
    def setUp(self):
        super().setUp()
        for scope in (
            tpu_envs.SGLANG_TPU_PD_DECODE_DRAFT_PREFILL.override(True),
            tpu_envs.SGLANG_TPU_ENABLE_EXPERIMENTAL_SPEC_WORKER.override(True),
            get_context().override_server_args(disaggregation_mode="decode"),
            get_parallel().override(pp_rank=0, pp_size=1),
        ):
            scope.__enter__()
            self.addCleanup(scope.__exit__, None, None, None)

    def test_the_budget_counts_the_hidden_row(self):
        row = PdHiddenPool.bytes_per_token(
            feature_size=FEATURE,
            total_kv_heads=KV_HEADS,
            attn_tp_size=1,
            dtype=torch.bfloat16,
        )
        configurator = budget_pd_hidden_pool(
            _configurator(DefaultPoolConfigurator), _kvc()
        )
        self.assertEqual(configurator._cell_size, 1000 + row)

    def test_servers_without_the_pool_keep_their_budget(self):
        cases = [
            (_kvc(is_draft_worker=True), {}),
            (_kvc(), dict(pp_rank=0, pp_size=2)),
        ]
        for kvc, parallel in cases:
            with get_parallel().override(**parallel):
                configurator = budget_pd_hidden_pool(
                    _configurator(DefaultPoolConfigurator), kvc
                )
            self.assertEqual(configurator._cell_size, 1000)
        with tpu_envs.SGLANG_TPU_PD_DECODE_DRAFT_PREFILL.override(False):
            configurator = budget_pd_hidden_pool(
                _configurator(DefaultPoolConfigurator), _kvc()
            )
        self.assertEqual(configurator._cell_size, 1000)

    def test_hybrid_swa_counts_the_row_per_full_token(self):
        row = PdHiddenPool.bytes_per_token(
            feature_size=FEATURE,
            total_kv_heads=KV_HEADS,
            attn_tp_size=1,
            dtype=torch.bfloat16,
        )
        # A windowed draft: its KV lives in the SWA slots, so _draft_cell_size is 0.
        configurator = budget_pd_hidden_pool(_hybrid_swa_configurator(), _kvc())
        self.assertEqual(configurator._draft_cell_size, row)
        self.assertEqual(
            configurator._cell_size, _hybrid_swa_configurator()._cell_size + row
        )

    def test_a_windowed_draft_counts_the_row_per_swa_token(self):
        row = PdHiddenPool.bytes_per_token(
            feature_size=FEATURE,
            total_kv_heads=KV_HEADS,
            attn_tp_size=1,
            dtype=torch.bfloat16,
        )
        aux = SpecAuxHiddenStateConfig(
            eagle_use_aux_hidden_state=True,
            eagle_aux_hidden_state_layer_ids=[1, 2, 3],
            dflash_draft_kv_is_swa=True,
        )
        base = _hybrid_swa_configurator()
        configurator = budget_pd_hidden_pool(
            _hybrid_swa_configurator(), _kvc(spec_aux_config=aux)
        )
        self.assertEqual(configurator._draft_cell_size, 0)
        self.assertEqual(configurator._draft_swa_cell_size, 300 + row)
        self.assertEqual(configurator._cell_size, base._cell_size + 0.0625 * row)
        # Full-token capacity of one budget: the row costs 1/16 per full token.
        per_full = budget_pd_hidden_pool(_hybrid_swa_configurator(), _kvc())
        budget = 1 << 30
        # 400 + 0.0625 * (1200 + 300 + 4096) vs 400 + 0.0625 * 1500 + 4096 bytes.
        self.assertEqual(int(budget // configurator._cell_size), 1_432_133)
        self.assertEqual(int(budget // per_full._cell_size), 233_943)

    def test_a_budget_it_cannot_size_is_rejected(self):
        for configurator in (
            _hybrid_swa_configurator(SWAChunkCapPoolConfigurator),
            _hybrid_swa_configurator(full_layers=0),
            _hybrid_swa_configurator(unified=True),
        ):
            with self.assertRaisesRegex(
                NotImplementedError, type(configurator).__name__
            ):
                budget_pd_hidden_pool(configurator, _kvc())


def _hybrid_swa_configurator(
    cls=HybridSWAPoolConfigurator, *, full_layers: int = 4, unified: bool = False
):
    configurator = object.__new__(cls)
    fields = dict(
        _full_per_token=100,
        _full_layers_num=full_layers,
        _swa_per_token=100,
        _swa_layers_num=12,
        _swa_full_tokens_ratio=0.0625,
        _draft_full_per_token=50,
        _draft_full_layers_num=0,
        _draft_swa_per_token=50,
        _draft_swa_layers_num=0,
        _draft_swa_full_layers_num=0,
        _draft_cell_size=0,
        _draft_swa_cell_size=300,
        _enable_unified_memory=unified,
    )
    for name, value in fields.items():
        setattr(configurator, name, value)
    configurator._recompute_cell_size()
    return configurator


if __name__ == "__main__":
    unittest.main()
