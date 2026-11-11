# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for the DeepSeek-V4.1 TPU KV pool.

Guards the physical geometry the V4.1 Pallas kernels read (SWA word pages of 64,
FP8 main-KV records, packed FP8 indexer pages of 128), kv_source layer sharing,
that every allocatable full / SWA location addresses a valid record or page, the
byte budget against what is allocated, and the ratio-2 pair-state slot reset.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

import functools
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.layers.attention.dsv4.metadata import expand_index_page_table
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.deepseek_v4_compress_state import KVAndScore
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.model_executor.pool_configurator import DSV4PoolConfigurator
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.models.deepseek_v41 import memory_pool as mp
from sglang_tpu.models.deepseek_v41.memory_pool import DSV41TPUTokenToKVPool

RATIOS = [0, 0] + [2] * 18 + [1] * 20
SOURCES = [2, 8, 14, 20]
FULL_SIZE = 4096
SWA_SIZE = 1024
NUM_REQ_SLOTS = 5

# 17 full pages (16 + the reserved one).
EXPECTED_SWA_PAGES = (SWA_SIZE + 256) // 64
EXPECTED_RECORDS = {2: 17 * 128, 1: 17 * 256}
EXPECTED_INDEX_PAGES = {2: 17, 1: 34}


def _install_server_args(testcase: unittest.TestCase, **fields) -> None:
    from sglang.srt.runtime_context import get_context

    override = get_context().override_server_args(page_size=256, **fields)
    override.install()
    testcase.addCleanup(override.restore)


def _pool_kwargs(**overrides) -> dict:
    kwargs = dict(
        max_num_reqs=NUM_REQ_SLOTS - 1,
        num_req_slots=NUM_REQ_SLOTS,
        swa_size=SWA_SIZE,
        c4_size=0,
        c128_size=0,
        c4_state_pool_size=0,
        c128_state_pool_size=0,
        page_size=256,
        swa_page_size=256,
        sliding_window=128,
        dtype=torch.bfloat16,
        c4_state_dtype=torch.float32,
        c128_state_dtype=torch.float32,
        qk_nope_head_dim=448,
        qk_rope_head_dim=64,
        indexer_head_dim=128,
        layer_num=len(RATIOS),
        device="cpu",
        enable_memory_saver=False,
        compression_ratios=RATIOS,
        start_layer=0,
        end_layer=len(RATIOS),
        kv_source_layers=SOURCES,
        full_size=FULL_SIZE,
    )
    kwargs.update(overrides)
    return kwargs


def _make_pool(testcase: unittest.TestCase, **overrides) -> DSV41TPUTokenToKVPool:
    _install_server_args(testcase)
    return DSV41TPUTokenToKVPool(**_pool_kwargs(**overrides))


def _source_of(layer_id: int) -> int:
    ratio = RATIOS[layer_id]
    return max(s for s in SOURCES if s <= layer_id and RATIOS[s] == ratio)


class TestDSV41TPUPoolGeometry(CustomTestCase):
    def setUp(self):
        self.pool = _make_pool(self)

    def test_swa_pages_every_layer(self):
        bufs = [self.pool.get_swa_pages(layer) for layer in range(len(RATIOS))]
        for buf in bufs:
            self.assertEqual(buf.dtype, torch.int32)
            self.assertEqual(tuple(buf.shape), (EXPECTED_SWA_PAGES * 128, 128))
            self.assertIsNone(buf._base)
        self.assertEqual(len({buf.data_ptr() for buf in bufs}), len(RATIOS))
        self.assertEqual(self.pool.swa_kv_pool.page_size, 64)
        self.assertEqual(self.pool.swa_page_size, 256)

    def test_main_kv_and_indexer_shapes(self):
        for source in SOURCES:
            ratio = RATIOS[source]
            words, scales = self.pool.get_main_kv(source)
            self.assertEqual(words.dtype, torch.int32)
            self.assertEqual(tuple(words.shape), (EXPECTED_RECORDS[ratio], 128))
            self.assertEqual(scales.dtype, torch.float32)
            self.assertEqual(tuple(scales.shape), (EXPECTED_RECORDS[ratio],))
            index_k = self.pool.get_index_k(source)
            self.assertEqual(index_k.dtype, torch.int32)
            self.assertEqual(
                tuple(index_k.shape), (EXPECTED_INDEX_PAGES[ratio] * 33, 128)
            )
            for buf in (words, scales, index_k):
                self.assertIsNone(buf._base)
            self.assertEqual(
                self.pool.index_k_layout(source),
                (EXPECTED_INDEX_PAGES[ratio], 33, 0, 128),
            )
            self.assertEqual(self.pool.main_page_records(source), 256 // ratio)

    def test_layer_sharing_follows_kv_sources(self):
        self.assertEqual(self.pool.sources_by_ratio, {1: [20], 2: [2, 8, 14]})
        for layer_id, ratio in enumerate(RATIOS):
            if ratio == 0:
                with self.assertRaises(AssertionError):
                    self.pool.get_main_kv(layer_id)
                continue
            source = _source_of(layer_id)
            self.assertEqual(self.pool.source_layer_of(layer_id), source)
            words, scales = self.pool.get_main_kv(layer_id)
            source_words, source_scales = self.pool.get_main_kv(source)
            self.assertIs(words, source_words)
            self.assertIs(scales, source_scales)
            self.assertIs(
                self.pool.get_index_k(layer_id), self.pool.get_index_k(source)
            )
        main_ptrs = {self.pool.get_main_kv(s)[0].data_ptr() for s in SOURCES}
        index_ptrs = {self.pool.get_index_k(s).data_ptr() for s in SOURCES}
        self.assertEqual(len(main_ptrs), len(SOURCES))
        self.assertEqual(len(index_ptrs), len(SOURCES))

    def test_pair_state_only_on_ratio2_sources(self):
        for layer_id, ratio in enumerate(RATIOS):
            state = self.pool.compress_state_pools[layer_id]
            if ratio == 2 and layer_id in SOURCES:
                self.assertEqual(state.ratio, 2)
                self.assertTrue(state.request_scoped)
                self.assertIs(self.pool.get_attention_compress_states(layer_id), state)
            else:
                self.assertIsNone(state)


class TestDSV41TPUPoolAddressing(CustomTestCase):
    """The allocators' highest location must land inside every buffer."""

    def setUp(self):
        self.pool = _make_pool(self)
        self.allocator = SWATokenToKVPoolAllocator(
            FULL_SIZE,
            SWA_SIZE,
            256,
            torch.int64,
            "cpu",
            self.pool,
            need_sort=False,
        )

    def _highest_loc(self, sub_allocator) -> int:
        return int(sub_allocator.free_pages.max()) * 256 + 255

    def test_highest_full_loc_in_range(self):
        top_full = self._highest_loc(self.allocator.full_attn_allocator)
        self.assertEqual(top_full, FULL_SIZE + 255)
        for source in SOURCES:
            ratio = RATIOS[source]
            words, scales = self.pool.get_main_kv(source)
            record = top_full // ratio
            self.assertLess(record, words.shape[0])
            # The main page id is the full page id.
            self.assertEqual(
                record // self.pool.main_page_records(source), top_full // 256
            )
            words[record] = 1
            scales[record] = 1.0

            pages, page_stride, page_offset, page_size = self.pool.index_k_layout(
                source
            )
            index_page = record // page_size
            self.assertLess(index_page, pages)
            self.assertLessEqual(
                page_offset + (index_page + 1) * page_stride,
                self.pool.get_index_k(source).shape[0],
            )

    def test_index_page_table_matches_records(self):
        # Every record of full page p sits in an indexer page the expanded table lists.
        for ratio in (1, 2):
            full_pages = torch.tensor([[1, 16, 7]], dtype=torch.int32)
            table = expand_index_page_table(
                full_pages,
                full_page_size=256,
                compress_ratio=ratio,
                index_page_size=128,
            )
            self.assertEqual(table.shape[1], 3 * (256 // ratio) // 128)
            for col, page in enumerate(full_pages[0].tolist()):
                locs = torch.arange(page * 256, (page + 1) * 256)
                records = torch.unique(locs // ratio)
                j = records - page * (256 // ratio) + col * (256 // ratio)
                via_table = table[0, j // 128] * 128 + j % 128
                torch.testing.assert_close(via_table.to(records.dtype), records)

    def test_highest_swa_loc_in_range(self):
        top_swa = self._highest_loc(self.allocator.swa_attn_allocator)
        self.assertEqual(top_swa, SWA_SIZE + 255)
        top_full = self._highest_loc(self.allocator.full_attn_allocator)
        mapping = self.allocator.full_to_swa_index_mapping
        mapping[top_full] = top_swa
        swa_loc = int(self.pool.translate_loc_from_full_to_swa(torch.tensor(top_full)))
        swa_page = swa_loc // 64
        for layer in (0, 2, 39):
            pages = self.pool.get_swa_pages(layer)
            self.assertLess(swa_page, pages.shape[0] // 128)
            # A token pair owns four word rows, including the highest
            # feature row. Both token halves must address the same pair.
            last_word_row = (swa_loc // 2) * 4 + 3
            self.assertLess(last_word_row, pages.shape[0])
            pages[last_word_row] = 1
        # Page 0 holds only the reserved logical page, so it stays the null page.
        self.assertGreaterEqual(
            int(self.allocator.swa_attn_allocator.free_pages.min()), 1
        )


class TestDSV41TPUPoolBytes(CustomTestCase):
    def setUp(self):
        self.pool = _make_pool(self)
        self.sources_by_ratio = self.pool.sources_by_ratio

    def _pair_state_bytes(self) -> int:
        return sum(
            pool.kv_score_buffer.kv_score.nbytes
            for pool in self.pool.compress_state_pools
            if pool is not None
        )

    def test_get_kv_size_bytes_matches_allocation(self):
        tensors = [self.pool.get_swa_pages(layer) for layer in range(len(RATIOS))]
        for source in SOURCES:
            tensors.extend(self.pool.get_main_kv(source))
            tensors.append(self.pool.get_index_k(source))
        kv_bytes = sum(t.nbytes for t in tensors)
        self.assertEqual(
            kv_bytes,
            DSV41TPUTokenToKVPool.kv_buffer_bytes(
                full_size=FULL_SIZE,
                swa_size=SWA_SIZE,
                num_layers=len(RATIOS),
                sources_by_ratio=self.sources_by_ratio,
            ),
        )
        self.assertEqual(
            self.pool.get_kv_size_bytes(), (kv_bytes + self._pair_state_bytes(), 0)
        )

    def test_per_token_budget_matches_allocation(self):
        self.assertEqual(
            DSV41TPUTokenToKVPool.bytes_per_swa_token(num_layers=len(RATIOS)),
            1024 * len(RATIOS),
        )
        per_full = DSV41TPUTokenToKVPool.bytes_per_full_token(
            swa_ratio=0.0,
            bytes_per_swa_token=0.0,
            sources_by_ratio=self.sources_by_ratio,
        )
        self.assertAlmostEqual(per_full, 3 * (512 + 4 + 132) / 2 + (512 + 4 + 132))
        compressed = sum(
            self.pool.get_main_kv(s)[0].nbytes
            + self.pool.get_main_kv(s)[1].nbytes
            + self.pool.get_index_k(s).nbytes
            for s in SOURCES
        )
        # Records cover full_size plus the one reserved logical page exactly.
        self.assertEqual(compressed, per_full * (FULL_SIZE + 256))
        swa = sum(self.pool.get_swa_pages(layer).nbytes for layer in range(len(RATIOS)))
        per_swa = DSV41TPUTokenToKVPool.bytes_per_swa_token(num_layers=len(RATIOS))
        self.assertEqual(swa, per_swa * (SWA_SIZE + 256))
        with_swa = DSV41TPUTokenToKVPool.bytes_per_full_token(
            swa_ratio=0.5,
            bytes_per_swa_token=per_swa,
            sources_by_ratio=self.sources_by_ratio,
        )
        self.assertAlmostEqual(with_swa, per_full + 0.5 * per_swa)

    def test_c2_state_bytes_match_pool_and_core_configurator(self):
        ring = self.pool.get_ring_size(2)
        expected = DSV41TPUTokenToKVPool.c2_state_fixed_bytes(
            num_req_slots=NUM_REQ_SLOTS,
            ring_size=ring,
            attn_head_dim=512,
            num_c2_layers=3,
        )
        self.assertEqual(expected, self._pair_state_bytes())
        configurator = object.__new__(DSV4PoolConfigurator)
        configurator.stage_owner_layers = self.sources_by_ratio
        configurator.ring_sizes = {2: ring}
        configurator.disaggregation_mode = "null"
        configurator.attn_head_dim = 512
        self.assertEqual(
            configurator._get_c2_state_fixed_bytes(NUM_REQ_SLOTS - 1), expected
        )


class TestDSV41TPUPoolRequestSlots(CustomTestCase):
    def test_slot_reuse_clears_pair_state(self):
        pool = _make_pool(self)
        req_pool = ReqToTokenPool(
            size=NUM_REQ_SLOTS - 1,
            max_context_len=16,
            device="cpu",
            enable_memory_saver=False,
        )
        req_pool.register_on_alloc_rows(pool.clear_request_slots)
        states = [pool.get_attention_compress_states(s) for s in (2, 8, 14)]
        for state in states:
            state.kv_score_buffer.kv_score.fill_(3.0)

        (slot,) = req_pool.alloc_rows(1)
        for state in states:
            ring = state.ring_size
            rows = state.kv_score_buffer[slot * ring : (slot + 1) * ring]
            self.assertTrue(torch.all(rows.kv == 0))
            self.assertTrue(torch.all(rows.score == float("-inf")))
            other = (slot % (NUM_REQ_SLOTS - 1)) + 1
            if other != slot:
                untouched = state.kv_score_buffer[other * ring : (other + 1) * ring]
                self.assertTrue(torch.all(untouched.kv_score == 3.0))

    def test_reset_row_reads_as_empty_partner(self):
        pool = _make_pool(self)
        state = pool.get_attention_compress_states(2)
        state.kv_score_buffer.kv_score.fill_(3.0)
        pool.clear_request_slots([1])
        loc = state.translate_from_req_position_to_state_loc(
            torch.tensor([1]), torch.tensor([0])
        )
        partner = state.get_state_by_state_loc(loc)
        self.assertIsInstance(partner, KVAndScore)
        self.assertTrue(torch.all(partner.score == float("-inf")))


class TestDSV41TPUPoolRejects(CustomTestCase):
    def test_unsupported_ratio(self):
        with self.assertRaisesRegex(ValueError, "compress ratios"):
            _make_pool(self, compression_ratios=[0, 4] + RATIOS[2:])

    def test_pipeline_parallel_stage(self):
        with self.assertRaisesRegex(ValueError, "pipeline parallelism"):
            _make_pool(self, layer_num=20, start_layer=20, end_layer=40)

    def test_target_layers_before_mtp_entries_are_not_a_stage(self):
        """Regression: V4.1 Flash lists 43 compress ratios (40 target + 3 MTP);
        the target pool (layer_num=40, layers [0, 40)) was rejected as a PP stage."""
        pool = _make_pool(
            self,
            compression_ratios=RATIOS + [0, 0, 0],
            layer_num=len(RATIOS),
            start_layer=0,
            end_layer=len(RATIOS),
        )
        self.assertEqual(len(pool.layer_mapping), len(RATIOS) + 3)

    def test_hisparse(self):
        with self.assertRaisesRegex(ValueError, "HiSparse"):
            _make_pool(self, enable_hisparse=True)

    def test_unified_kv(self):
        with patch.object(mp, "is_unified_kv_triton", return_value=True):
            with self.assertRaisesRegex(ValueError, "unified_kv"):
                _make_pool(self)

    def test_encoder_replay(self):
        _install_server_args(self, enable_encoder_swa_bounded_replay=True)
        with self.assertRaisesRegex(ValueError, "request window"):
            DSV41TPUTokenToKVPool(**_pool_kwargs())

    def test_cuda_kv_layout(self):
        with self.assertRaisesRegex(ValueError, "FlashMLA"):
            _make_pool(self, kv_layout="v41")


class TestDSV41TPUPoolHooks(CustomTestCase):
    def test_build_falls_through_for_v4(self):
        original = MagicMock(return_value="v4-pool")
        kvc = SimpleNamespace(model_config=SimpleNamespace(compress_ratios=[0, 4, 128]))
        self.assertEqual(mp.build_tpu_dsv41_kv_pool(original, kvc, x=1), "v4-pool")
        original.assert_called_once_with(kvc, x=1)

    def test_build_binds_pool_and_bypasses_v4_hook(self):
        import sglang.srt.mem_cache.kv_cache_configurator as kvc_mod

        built = SimpleNamespace(clear_request_slots=object())

        def core_build(kvc, **kwargs):
            self.assertIs(kvc_mod.DeepSeekV4TokenToKVPool, DSV41TPUTokenToKVPool)
            return built

        @functools.wraps(core_build)
        def v4_hook(*args, **kwargs):
            raise AssertionError("the V4 TPU hook must be bypassed for V4.1")

        req_pool = MagicMock()
        kvc = SimpleNamespace(
            model_config=SimpleNamespace(compress_ratios=RATIOS), is_draft_worker=False
        )
        pool = mp.build_tpu_dsv41_kv_pool(v4_hook, kvc, req_to_token_pool=req_pool)
        self.assertIs(pool, built)
        req_pool.register_on_alloc_rows.assert_called_once_with(
            built.clear_request_slots
        )
        self.assertIsNot(kvc_mod.DeepSeekV4TokenToKVPool, DSV41TPUTokenToKVPool)

    def test_budget_hooks(self):
        configurator = SimpleNamespace(
            stage_compress_ratios=RATIOS,
            num_layers_total=len(RATIOS),
            _unified=False,
            swa_cap_tokens=None,
            swa_ratio=0.25,
            bytes_per_swa_token=1024.0 * len(RATIOS),
            stage_owner_layers={2: [2, 8, 14], 1: [20]},
        )
        original = MagicMock()
        self.assertEqual(
            mp.get_tpu_dsv41_bytes_per_swa_token(original, configurator),
            1024.0 * len(RATIOS),
        )
        self.assertAlmostEqual(
            mp.get_tpu_dsv41_bytes_per_full_token(original, configurator),
            0.25 * 1024 * len(RATIOS) + 3 * 648 / 2 + 648,
        )
        original.assert_not_called()
        v4 = SimpleNamespace(stage_compress_ratios=[0, 4, 128])
        original.return_value = 7.0
        self.assertEqual(mp.get_tpu_dsv41_bytes_per_full_token(original, v4), 7.0)


if __name__ == "__main__":
    unittest.main()
