"""Decode retraction backup of Kimi-K3's two pools on device: MLA latent pages
and KDA state slots round-trip through get_cpu_copy / load_cpu_copy.

On TPU the restore writes through the in-place page-write kernel, where the CPU
tests (test_tpu_hybrid_linear_pd_state.py) take write_pages_torch's eager
fallback. So these pin the packed page layout on device, and that each buffer
keeps its storage: the compiled step and a P/D transfer engine hold it.
"""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=60, suite="stage-a-test-tpu-unit", chips=1)

import unittest

import torch

from sglang.srt.configs.mamba_utils import (
    KimiLinearCacheParams,
    KimiLinearStateShape,
    Mamba2StateDType,
)
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.kv_pool import TpuFusedMLATokenToKVPool
from sglang_tpu.mamba_pool import AT_SLOT, TpuMambaPool

DEV = "tpu"
LAYERS = [0, 1]
# Kimi-K3's MLA page and latent widths; bf16 packs two tokens per row.
PAGE_SIZE = 128
KV_LORA_RANK, ROPE_DIM = 512, 64
# A DSpark verify window: gamma 7 plus the bonus row.
WIDTH = 8


def _fill_random(tensors, seed: int) -> list:
    """Fill device tensors with random values; return their host copies."""
    generator = torch.Generator().manual_seed(seed)
    host = [torch.randn(t.shape, generator=generator).to(t.dtype) for t in tensors]
    for tensor, value in zip(tensors, host):
        tensor.copy_(value.to(tensor.device))
    return host


class TestFusedMLARetractionBackup(CustomTestCase):
    def test_full_and_partial_pages_restore_in_place(self):
        pool = TpuFusedMLATokenToKVPool(
            size=PAGE_SIZE * 4,
            page_size=PAGE_SIZE,
            dtype=torch.bfloat16,
            kv_lora_rank=KV_LORA_RANK,
            qk_rope_head_dim=ROPE_DIM,
            layer_num=len(LAYERS),
            device=DEV,
            enable_memory_saver=False,
        )
        before = _fill_random(pool.kv_buffer, seed=0)
        ptrs = [buf.data_ptr() for buf in pool.kv_buffer]
        # Page 1 whole and 40 tokens of page 2, restored as page 3 whole and
        # the first 40 tokens of page 4.
        src = torch.arange(PAGE_SIZE, 2 * PAGE_SIZE + 40)
        dst = torch.arange(3 * PAGE_SIZE, 4 * PAGE_SIZE + 40)
        saved = pool.get_cpu_copy(src)
        pool.load_cpu_copy(saved, dst)

        self.assertEqual([buf.data_ptr() for buf in pool.kv_buffer], ptrs)
        for buf, old in zip(pool.kv_buffer, before):
            # Token t of page p is row p * PAGE_SIZE + t of the packed pages.
            got = buf.cpu().view(-1, buf.shape[-1])
            old = old.view(-1, old.shape[-1])
            self.assertTrue(torch.equal(got[dst], old[src]))
            # Every other row keeps its value, the rest of page 4 included.
            untouched = torch.ones(got.shape[0], dtype=torch.bool)
            untouched[dst] = False
            self.assertTrue(torch.equal(got[untouched], old[untouched]))


class TestMambaRetractionBackup(CustomTestCase):
    def test_window_row_restores_into_the_resumed_slot_in_place(self):
        shape = KimiLinearStateShape.create(
            tp_world_size=1, num_heads=3, head_dim=128, conv_kernel_size=4
        )
        pool = TpuMambaPool(
            size=8,
            spec_state_size=8,
            cache_params=KimiLinearCacheParams(
                dtype=Mamba2StateDType(conv=torch.float32, temporal=torch.float32),
                layers=LAYERS,
                shape=shape,
            ),
            mamba_layer_ids=LAYERS,
            device=DEV,
            speculative_num_draft_tokens=WIDTH,
            speculative_eagle_topk=1,
        )
        before = _fill_random(pool.get_state(), seed=1)
        ptrs = [state.data_ptr() for state in pool.get_state()]
        # Slot 3's request holds pool row 2; its last verify kept window row 5.
        pool.select_checkpoints(torch.tensor([3]), torch.tensor([5]))
        committed = pool.window_start + 2 * WIDTH + 5
        saved = pool.get_cpu_copy(torch.tensor(3), req_pool_indices=2)
        # The resumed request holds slot 5, whose offset a stale verify left at 0.
        pool.checkpoint_read_offsets.index_fill_(
            0, torch.tensor([5], device=pool.checkpoint_read_offsets.device), 0
        )
        pool.load_cpu_copy(saved, torch.tensor([5]))

        self.assertEqual([state.data_ptr() for state in pool.get_state()], ptrs)
        self.assertEqual(int(pool.checkpoint_read_offsets.cpu()[5]), AT_SLOT)
        for state, old in zip(pool.get_state(), before):
            got = state.cpu()
            self.assertTrue(torch.equal(got[5], old[committed]))
            # Every other row keeps its value, the window row read included.
            untouched = torch.ones(got.shape[0], dtype=torch.bool)
            untouched[5] = False
            self.assertTrue(torch.equal(got[untouched], old[untouched]))


if __name__ == "__main__":
    unittest.main()
