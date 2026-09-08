"""On-device parity checks for the block-major TPU KV pool against the fused pool.

A device run verifies writes land in the bundle itself across JAX buffer donation,
which CPU-only tests cannot distinguish from mutated view copies.
"""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=180, suite="stage-a-test-tpu-unit")

import itertools
import math
import unittest

import torch

from sglang.test.test_utils import CustomTestCase

DEV = "tpu"
PAGE_SIZE = 16
HEAD_NUM = 2
HEAD_DIM = 128
LAYER_NUM = 3
NUM_PAGES = 12


def _has_tpu() -> bool:
    return hasattr(torch, "accelerator") and torch.accelerator.is_available()


_SKIP_REASON = "Requires a TPU visible to torch.accelerator"


def _make_pools(layer_num=LAYER_NUM, seed=0):
    from sglang_tpu.kv_pool import (
        TpuBlockMajorMHATokenToKVPool,
        TpuFusedMHATokenToKVPool,
    )

    kwargs = dict(
        size=NUM_PAGES * PAGE_SIZE,
        page_size=PAGE_SIZE,
        dtype=torch.bfloat16,
        head_num=HEAD_NUM,
        head_dim=HEAD_DIM,
        layer_num=layer_num,
        device=DEV,
        enable_memory_saver=False,
    )
    fused = TpuFusedMHATokenToKVPool(**kwargs)
    bm = TpuBlockMajorMHATokenToKVPool(**kwargs)
    gen = torch.Generator().manual_seed(seed)
    bundle = torch.randn(bm.kv_bundle.shape, generator=gen).to(torch.bfloat16)
    bm.kv_bundle.copy_(bundle.to(DEV))
    for layer in range(layer_num):
        fused.kv_buffer[layer].copy_(bundle[:, layer].contiguous().to(DEV))
    return fused, bm


def _kernel_programs(name: str) -> int:
    """Distinct Pallas programs torch_tpu registered for the JAX kernel ``name``."""
    import gc

    from torch_tpu._internal.pallas.pallas import JaxCallable

    (kernel,) = [
        obj
        for obj in gc.get_objects()
        if isinstance(obj, JaxCallable) and obj.name == name
    ]
    return len(set(kernel.kernel_key_to_mlir_fingerprint.values()))


def _layer_views(bm):
    bundle = bm.kv_bundle.cpu()
    return [bundle[:, layer].contiguous() for layer in range(bm.layer_num)]


def _assert_pools_equal(test, fused, bm):
    got = _layer_views(bm)
    for layer in range(fused.layer_num):
        test.assertTrue(
            torch.equal(got[layer], fused.kv_buffer[layer].cpu()), f"layer {layer}"
        )


@unittest.skipUnless(_has_tpu(), _SKIP_REASON)
class TestBundledKernelMatchesFused(CustomTestCase):
    """The kernel, given the bundle and a traced ``layer_idx``, matches the fused pool's output and cache."""

    NUM_Q_HEADS = 4
    PAGES_PER_SEQ = 4
    TOLERANCE = 2e-2

    def _run(self, seqs, *, decode):
        from sglang_tpu.attention_pallas_rpa_base import _build_pallas_kernel_op
        from sglang_tpu.attention_pallas_rpa_v3 import RPA_V3_SPEC

        op = _build_pallas_kernel_op(RPA_V3_SPEC)
        fused, bm = _make_pools()
        num_seqs = len(seqs)
        total_q = sum(q for _, q in seqs)
        max_tokens = max(8, -(-total_q // 8) * 8)
        gen = torch.Generator().manual_seed(1)

        def rand(*shape):
            return torch.randn(*shape, generator=gen).to(torch.bfloat16).to(DEV)

        q = rand(max_tokens, self.NUM_Q_HEADS, HEAD_DIM)
        k = rand(max_tokens, HEAD_NUM, HEAD_DIM)
        v = rand(max_tokens, HEAD_NUM, HEAD_DIM)
        # Shuffled pages, none of them the padding block 0.
        perm = torch.randperm(NUM_PAGES, generator=gen) + 1
        page_indices = perm[: num_seqs * self.PAGES_PER_SEQ].to(torch.int32).to(DEV)
        kv_lens = torch.tensor([kv for kv, _ in seqs], dtype=torch.int32, device=DEV)
        cu_q_lens = torch.tensor(
            [0] + list(itertools.accumulate(q_len for _, q_len in seqs)),
            dtype=torch.int32,
            device=DEV,
        )
        cu_kv_lens = torch.tensor(
            [i * self.PAGES_PER_SEQ * PAGE_SIZE for i in range(num_seqs + 1)],
            dtype=torch.int32,
            device=DEV,
        )
        distribution = torch.tensor(
            [num_seqs] * 3 if decode else [0, num_seqs, num_seqs],
            dtype=torch.int32,
            device=DEV,
        )
        args = (
            kv_lens,
            page_indices,
            cu_q_lens,
            cu_kv_lens,
            distribution,
            None,  # attention_sink
            1.0 / math.sqrt(HEAD_DIM),
            None,  # sliding_window
            None,  # soft_cap
            None,  # k_scale
            None,  # v_scale
        )
        programs = 0
        for layer in range(LAYER_NUM):
            ref = op(q, k, v, fused.get_fused_kv_buffer(layer), *args)
            bundle, layer_idx = bm.get_fused_kv_operands(layer)
            got = op(q, k, v, bundle, *args, layer_idx=layer_idx)
            torch.testing.assert_close(
                got.cpu().float()[:total_q],
                ref.cpu().float()[:total_q],
                atol=self.TOLERANCE,
                rtol=self.TOLERANCE,
                msg=f"layer {layer}",
            )
            # A traced layer index: later layers reuse the first layer's program.
            count = _kernel_programs(f"jax_{RPA_V3_SPEC.op_name}")
            if layer == 0:
                programs = count
            self.assertEqual(count, programs, f"layer {layer} added a program")
        _assert_pools_equal(self, fused, bm)

    def test_extend_partial_last_page(self):
        self._run([(37, 21), (50, 50)], decode=False)

    def test_decode_multi_seq(self):
        self._run([(17, 1), (64, 1), (33, 1)], decode=True)


@unittest.skipUnless(_has_tpu(), _SKIP_REASON)
class TestRowPathRestoreOnDevice(CustomTestCase):
    """The row path offloads and restores directly to the bundle on device."""

    CHUNK = 5

    def _round_trip(self, indices, load_indices=None):
        fused, bm = _make_pools()
        fused.cpu_offloading_chunk_size = self.CHUNK
        bm.cpu_offloading_chunk_size = self.CHUNK
        src = torch.tensor(indices, dtype=torch.long)
        fused_cpu = fused.get_cpu_copy(src)
        bm_cpu = bm.get_cpu_copy(src)
        for bm_chunks, fused_chunks in zip(bm_cpu, fused_cpu):
            self.assertEqual(len(bm_chunks), len(fused_chunks))
            for a, b in zip(bm_chunks, fused_chunks):
                self.assertTrue(torch.equal(a, b))

        dst = src if load_indices is None else torch.tensor(load_indices)
        # Different seed, so an unwritten row is visible.
        fused, bm = _make_pools(seed=7)
        fused.load_cpu_copy(fused_cpu, dst)
        bm.load_cpu_copy(bm_cpu, dst)
        _assert_pools_equal(self, fused, bm)
        bundle = bm.kv_bundle.cpu()
        blocks, offsets = dst // PAGE_SIZE, dst % PAGE_SIZE
        for layer, chunks in enumerate(bm_cpu):
            self.assertTrue(
                torch.equal(bundle[blocks, layer, offsets], torch.cat(chunks))
            )

    def test_full_pages(self):
        self._round_trip(list(range(16, 48)))

    def test_partial_pages(self):
        self._round_trip([17, 18, *range(32, 48), 50, 61])

    def test_relocated(self):
        self._round_trip(
            [17, 18, *range(32, 48), 50, 61],
            load_indices=[81, 82, *range(96, 112), 114, 125],
        )


if __name__ == "__main__":
    unittest.main()
