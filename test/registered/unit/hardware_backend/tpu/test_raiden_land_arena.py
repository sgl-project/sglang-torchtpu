"""CPU tests for the raiden reshard landing arena (#829 P1b): pool sizing, the
allocator boundary, and ScratchArena bookkeeping."""

import contextlib
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.allocator_tpu import TpuPagedTokenToKVPoolAllocator
from sglang_tpu.disaggregation.raiden import land
from sglang_tpu.disaggregation.raiden.common import get_raiden_kv_caches
from sglang_tpu.disaggregation.raiden.land import ScratchArena, land_arena_blocks
from sglang_tpu.kv_pool import TpuFusedMHATokenToKVPool, TpuSeqAlongLaneKVPool

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_FP8 = torch.float8_e4m3fn
PAGE = 128
SIZE = 2 * PAGE
# 2 * ceil(8192 / 128)
ARENA = 128


@contextlib.contextmanager
def _deployment(
    arena=True,
    mode="decode",
    backend="raiden",
    kv_cache_dtype="fp8_e4m3",
    page_size=PAGE,
    pool_cls=TpuFusedMHATokenToKVPool,
    mla=False,
):
    """The target's deployment, from which the whole group's arena follows."""
    with (
        mock.patch("sglang.srt.runtime_context.uses_mla_backend", return_value=mla),
        envs.SGLANG_RAIDEN_RESHARD_LAND_ARENA.override(arena),
        get_context().override_server_args(
            disaggregation_mode=mode,
            disaggregation_transfer_backend=backend,
            kv_cache_dtype=kv_cache_dtype,
            page_size=page_size,
        ),
        mock.patch(
            "sglang.srt.platforms.current_platform",
            SimpleNamespace(get_mha_kv_pool_cls=lambda: pool_cls),
        ),
    ):
        yield


def _pool(
    dtype=_FP8, *, layer_num=1, head_num=2, label=None, page_size=PAGE, size=SIZE
):
    return TpuFusedMHATokenToKVPool(
        size=size,
        page_size=page_size,
        dtype=dtype,
        head_num=head_num,
        head_dim=128,
        layer_num=layer_num,
        device="cpu",
        enable_memory_saver=False,
        allocation_label=label,
    )


def _swa_pool(dtype=_FP8):
    return SWAKVPool(
        size=SIZE,
        size_swa=SIZE,
        page_size=PAGE,
        dtype=dtype,
        head_num=2,
        head_dim=128,
        swa_attention_layer_ids=[0, 2],
        full_attention_layer_ids=[1, 3],
        device="cpu",
        token_to_kv_pool_class=TpuFusedMHATokenToKVPool,
    )


class TestArenaSizing(CustomTestCase):
    def test_env_off_leaves_the_pool_unchanged(self):
        # Off must not even reach the raiden predicate.
        with (
            _deployment(arena=False),
            mock.patch.object(land, "land_arena_blocks", side_effect=AssertionError),
        ):
            pool = _pool()
        self.assertEqual(pool.land_arena_blocks, 0)
        self.assertEqual(pool.num_blocks, SIZE // PAGE + 1)
        self.assertEqual(pool.kv_buffer[0].shape[0], SIZE // PAGE + 1)
        self.assertIsNone(ScratchArena.from_pool(pool))

    def test_fp8_token_major_raiden_decode_reserves_two_chunks(self):
        with _deployment():
            pool = _pool()
        self.assertEqual(pool.land_arena_blocks, ARENA)
        self.assertEqual(pool.num_blocks, SIZE // PAGE + 1 + ARENA)
        self.assertEqual(pool.kv_buffer[0].shape[0], pool.num_blocks)
        arena = ScratchArena.from_pool(pool)
        self.assertEqual(arena.first_block, SIZE // PAGE + 1)
        self.assertEqual(arena.capacity, ARENA)
        self.assertEqual(arena.first_block + arena.capacity, pool.num_blocks)

    def test_page_size_rounds_the_chunk_up(self):
        for page, blocks in ((64, 256), (256, 64), (3000, 6), (16384, 2)):
            with _deployment(page_size=page):
                self.assertEqual(land_arena_blocks(), blocks, page)

    def test_other_deployments_reserve_nothing(self):
        cases = {
            "bf16 target": (dict(kv_cache_dtype="bfloat16"), {}),
            "auto target": (dict(kv_cache_dtype="auto"), {}),
            "seq_along_lane": (dict(pool_cls=TpuSeqAlongLaneKVPool), {}),
            "generic pool": (dict(pool_cls=MHATokenToKVPool), {}),
            "swa": ({}, dict(swa=True)),
            "prefill": (dict(mode="prefill"), {}),
            "no disagg": (dict(mode="null"), {}),
            "mooncake": (dict(backend="mooncake"), {}),
            "env off": (dict(arena=False), {}),
            # The sizer charges nothing for MLA; an MHA draft must match.
            "mla target": (dict(mla=True), {}),
        }
        for name, (deployment, kwargs) in cases.items():
            with self.subTest(name), _deployment(**deployment):
                self.assertEqual(land_arena_blocks(**kwargs), 0)

    def test_bf16_target_leaves_pools_unchanged_with_env_on(self):
        with _deployment(kv_cache_dtype="bfloat16"):
            pool = _pool(torch.bfloat16)
        self.assertEqual(pool.num_blocks, SIZE // PAGE + 1)

    def test_allocator_never_hands_out_arena_blocks(self):
        with _deployment():
            pool = _pool()
        allocator = TpuPagedTokenToKVPoolAllocator(
            size=pool.size,
            page_size=PAGE,
            dtype=torch.int64,
            device="cpu",
            kvcache=pool,
            need_sort=False,
        )
        arena = ScratchArena.from_pool(pool)
        self.assertEqual(
            allocator.free_pages.tolist(), list(range(1, arena.first_block))
        )


class TestArenaUniformAcrossGroups(CustomTestCase):
    def test_draft_pool_gets_the_same_arena(self):
        with _deployment():
            target = _pool(layer_num=2, head_num=2)
            draft = _pool(layer_num=1, head_num=4)
        self.assertEqual(draft.num_blocks, target.num_blocks)
        groups = get_raiden_kv_caches(target, draft)
        self.assertEqual({int(t.shape[0]) for t in groups.full}, {target.num_blocks})

    def test_swa_group_gets_none(self):
        with _deployment():
            pool = _swa_pool()
            draft = _pool()
        self.assertEqual(pool.full_kv_pool.land_arena_blocks, ARENA)
        self.assertEqual(pool.swa_kv_pool.land_arena_blocks, 0)
        self.assertEqual(pool.swa_kv_pool.num_blocks, SIZE // PAGE + 1)
        self.assertEqual(ScratchArena.from_pool(pool).capacity, ARENA)
        groups = get_raiden_kv_caches(pool, draft)
        self.assertEqual(
            {int(t.shape[0]) for t in groups.full}, {SIZE // PAGE + 1 + ARENA}
        )
        self.assertEqual({int(t.shape[0]) for t in groups.swa}, {SIZE // PAGE + 1})

    def test_non_fp8_draft_takes_the_target_arena(self):
        # --speculative-draft-kv-cache-dtype may leave the draft non-fp8; it
        # still reserves the target's arena so the group stays uniform.
        with _deployment():
            target, draft = _pool(), _pool(torch.bfloat16)
        self.assertEqual(draft.land_arena_blocks, target.land_arena_blocks)
        self.assertEqual(target.land_arena_blocks, ARENA)
        groups = get_raiden_kv_caches(target, draft)
        self.assertEqual({int(t.shape[0]) for t in groups.full}, {target.num_blocks})

    def test_dcp_widened_draft_page_takes_the_target_arena(self):
        # Under DCP a draft pool pages dcp * page over dcp * size tokens.
        dcp = 2
        with _deployment():
            target = _pool()
            draft = _pool(page_size=PAGE * dcp, size=SIZE * dcp)
        self.assertEqual(draft.land_arena_blocks, ARENA)
        self.assertEqual(draft.num_blocks, target.num_blocks)
        groups = get_raiden_kv_caches(target, draft)
        self.assertEqual({int(t.shape[0]) for t in groups.full}, {target.num_blocks})


class TestScratchArena(CustomTestCase):
    def test_take_and_free_exactly_once(self):
        arena = ScratchArena(first_block=5, num_blocks=4)
        ids = arena.take(3)
        self.assertEqual(ids, [5, 6, 7])
        self.assertEqual((arena.held, arena.available), (3, 1))
        arena.free(ids[:2])
        self.assertEqual((arena.held, arena.available), (1, 3))
        with self.assertRaisesRegex(RuntimeError, "double free"):
            arena.free([5])
        with self.assertRaisesRegex(RuntimeError, "double free"):
            arena.free([7, 7])
        with self.assertRaisesRegex(RuntimeError, "foreign"):
            arena.free([42])
        # A rejected free changes nothing.
        self.assertEqual((arena.held, arena.available), (1, 3))
        arena.free([7])
        self.assertEqual((arena.held, arena.available), (0, 4))

    def test_exhaustion_takes_nothing(self):
        arena = ScratchArena(first_block=1, num_blocks=2)
        self.assertIsNone(arena.take(3))
        self.assertEqual(arena.available, 2)
        self.assertEqual(sorted(arena.take(2)), [1, 2])
        self.assertIsNone(arena.take(1))
        self.assertEqual(arena.take(0), [])
        with self.assertRaises(ValueError):
            arena.take(-1)

    def test_freed_ids_are_reused(self):
        arena = ScratchArena(first_block=3, num_blocks=2)
        first = arena.take(2)
        arena.free(first)
        self.assertEqual(sorted(arena.take(2)), first)

    def test_rejects_the_padding_page(self):
        with self.assertRaises(ValueError):
            ScratchArena(first_block=0, num_blocks=4)
        with self.assertRaises(ValueError):
            ScratchArena(first_block=3, num_blocks=0)


if __name__ == "__main__":
    unittest.main()
