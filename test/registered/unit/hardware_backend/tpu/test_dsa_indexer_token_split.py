"""DSA indexer token split (always on with attention TP > 1), host-side logic without a TPU.

The windowed indexer kernel is checked bit for bit against the unsplit one on
TPU in test/registered/tpu/unit/test_dsa_index_topk.py. These cover the rest:
which batches split and into how many rows per rank, and the all-reduce that
puts the ranks' row slices together in rank order (T=4096 at TP8 and TP32,
the 4096-token prefill bucket of Kimi-K3.1).
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace

import torch

from sglang.test.test_utils import CustomTestCase
from sglang_tpu.attention_pallas_mla_sparse import (
    TpuPallasSparseMlaAttnBackend,
    _assemble_split_rows,
)


def _split_rows(num_tokens, tp):
    """Rows per rank as the backend computes them: ceil(T / tp), rounded up to 8."""
    rows = -(-num_tokens // tp)
    return -(-rows // 8) * 8


def _simulate_all_reduce(slices, tp):
    """Run _assemble_split_rows on every rank with a summing all-reduce."""
    bufs = []
    for r in range(tp):
        _assemble_split_rows(
            slices[r], torch.tensor(r), tp, lambda t: (bufs.append(t), t)[1]
        )
    total = sum(bufs)
    return [
        _assemble_split_rows(slices[r], torch.tensor(r), tp, lambda t: total.clone())
        for r in range(tp)
    ]


class TestAssembleSplitRows(CustomTestCase):
    def test_assembly(self):
        # (T, tp, topk): the 4096-token prefill bucket at TP8 and TP32 (topk
        # 256 instead of 2048 keeps the 32-rank simulation small; the layout
        # does not depend on the width), and T=200 at tp=4, whose 56-row
        # windows put the last one 24 rows past T.
        for num_tokens, tp, topk in ((4096, 8, 256), (4096, 32, 256), (200, 4, 128)):
            g = torch.Generator().manual_seed(num_tokens * 100 + tp)
            rows = _split_rows(num_tokens, tp)
            # -1 padding included; window rows past T are all -1.
            full = torch.randint(-1, num_tokens, (tp * rows, topk), generator=g)
            full = full.to(torch.int32)
            full[num_tokens:] = -1
            outs = _simulate_all_reduce(list(full.view(tp, rows, topk)), tp)
            for r, out in enumerate(outs):
                with self.subTest(T=num_tokens, tp=tp, rank=r):
                    self.assertEqual(out.dtype, torch.int32)
                    self.assertTrue(torch.equal(out, full))


class _Mode:
    def __init__(self, extend):
        self.extend = extend

    def is_extend_or_draft_extend_or_mixed(self):
        return self.extend


class TestSplitRows(CustomTestCase):
    def _backend(self, tp, min_tokens=1024, decode_query_size=1):
        backend = TpuPallasSparseMlaAttnBackend.__new__(TpuPallasSparseMlaAttnBackend)
        backend._idx_split_group = SimpleNamespace(world_size=tp)
        backend._idx_split_size = tp
        backend._idx_split_min_tokens = max(min_tokens, 32 * tp)
        backend.forward_metadata = SimpleNamespace(decode_query_size=decode_query_size)
        return backend

    def _rows(self, backend, num_tokens, extend=True):
        batch = SimpleNamespace(forward_mode=_Mode(extend))
        return backend._indexer_split_rows(batch, num_tokens)

    def test_prefill_buckets(self):
        # The 93L TP32 run: 4096 -> 128 rows per rank, 1024 -> 32.
        self.assertEqual(self._rows(self._backend(32), 4096), 128)
        self.assertEqual(self._rows(self._backend(32), 1024), 32)
        # 13L TP8: 4096 -> 512 rows per rank; no cap, so 8192 -> 1024.
        self.assertEqual(self._rows(self._backend(8), 4096), 512)
        self.assertEqual(self._rows(self._backend(8), 8192), 1024)

    def test_unsplit_batches(self):
        backend = self._backend(32)
        self.assertEqual(self._rows(backend, 4096, extend=False), 0)  # decode
        self.assertEqual(self._rows(backend, 512), 0)  # below 1024 tokens
        verify = self._backend(32, decode_query_size=8)
        self.assertEqual(self._rows(verify, 4096), 0)

    def test_off_without_a_group(self):
        backend = self._backend(32)
        backend._idx_split_group = None
        self.assertEqual(self._rows(backend, 4096), 0)
        # Built without __init__ at all (as other tests do): no split.
        bare = TpuPallasSparseMlaAttnBackend.__new__(TpuPallasSparseMlaAttnBackend)
        self.assertEqual(self._rows(bare, 4096), 0)


if __name__ == "__main__":
    unittest.main()
