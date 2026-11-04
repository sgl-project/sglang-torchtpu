"""Attention DP2 on the K3 decode: page regions, block-table ids, and a
token-level golden layout from the DCP32 prefill through the transfer and the
swap to the DCP16 group's reads."""

import itertools
import unittest

import numpy as np

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.k3_programs import sglang_dp as dp

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

RANKS = 32


class TestDpPages(CustomTestCase):
    def test_regions_rows_and_ids(self):
        pages = dp.DpPages(pages=8, num_blocks=20)
        self.assertEqual([pages.group(p) for p in range(1, 9)], [0] * 4 + [1] * 4)
        self.assertEqual(pages.rows(1), (1, 2))
        self.assertEqual(pages.rows(5), (1, 2))  # same rows, the other group's ranks
        self.assertEqual(pages.rows(8), (7, 8))
        self.assertEqual(pages.table_ids(6), (20 + 3, 20 + 4))
        self.assertEqual(list(pages.region(1)), [5, 6, 7, 8])
        with self.assertRaises(ValueError):
            dp.DpPages(pages=7, num_blocks=20)

    def test_table_row_cuts_to_the_touched_blocks(self):
        pages = dp.DpPages(pages=8, num_blocks=20)
        row = pages.table_row([2, 3], seq_len=dp.GROUP_BLOCK_TOKENS + 1, width=6)
        self.assertEqual(row.tolist(), [3, 4, 0, 0, 0, 0])
        row = pages.table_row([6, 7], seq_len=3 * dp.GROUP_BLOCK_TOKENS, width=6)
        self.assertEqual(row.tolist(), [23, 24, 25, 0, 0, 0])
        with self.assertRaisesRegex(ValueError, "span DP groups"):
            pages.table_row([2, 6], seq_len=dp.PAGE_TOKENS + 1, width=6)


def _golden(seq_len, group, decode_pages, landing, prefill_rows, num_blocks=24, pages=16):
    """Place a sequence's chunks DCP32 on the prefill, transfer each decode
    rank's rows, swap, and read every chunk back through the DCP16 table."""
    plan = dp.DpPages(pages=pages, num_blocks=num_blocks)
    chunks = -(-seq_len // dp.CHUNK_TOKENS)
    blocks = -(-seq_len // dp.PAGE_TOKENS)
    prefill = np.full((RANKS, num_blocks), -1)
    for k in range(chunks):
        prefill[k % RANKS, prefill_rows[k // RANKS]] = k
    decode = np.full((RANKS, num_blocks), -1)
    # Rows other requests own on the decode: must survive.
    for rank in range(RANKS):
        decode[rank, num_blocks - 1] = 10_000 + rank
    for rank in range(RANKS):
        rows = plan.transfer_rows(rank, decode_pages[:blocks], landing)
        for j, row in enumerate(rows):
            decode[rank, row] = prefill[rank, prefill_rows[j]]
    sources, destinations = plan.swap_rows(decode_pages[:blocks], landing)
    other = 1 - group
    for q in range(dp.GROUP_RANKS):
        for src, dst in zip(sources, destinations):
            decode[group * dp.GROUP_RANKS + q, dst] = decode[other * dp.GROUP_RANKS + q, src]
    width = -(-seq_len // dp.GROUP_BLOCK_TOKENS) + 2
    table = plan.table_row(decode_pages, seq_len, width)
    for k in range(chunks):
        q, b = k % dp.GROUP_RANKS, k // dp.GROUP_RANKS
        ident = int(table[b])
        assert ident // num_blocks == group, (k, ident)
        got = decode[group * dp.GROUP_RANKS + q, ident % num_blocks]
        assert got == k, f"chunk {k}: group rank {q} block {b} row {ident % num_blocks} holds {got}"
    for rank in range(RANKS):
        assert decode[rank, num_blocks - 1] == 10_000 + rank


class TestGoldenLayout(CustomTestCase):
    def test_every_chunk_reads_back_in_either_group(self):
        lengths = (1, 511, 512, 8192, 8193, 16384, 16385, 40000, 65536)
        for group, seq_len in itertools.product((0, 1), lengths):
            with self.subTest(group=group, seq_len=seq_len):
                blocks = -(-seq_len // dp.PAGE_TOKENS)
                plan = dp.DpPages(pages=16, num_blocks=24)
                region, other = list(plan.region(group)), list(plan.region(1 - group))
                decode_pages = region[::-1][: blocks + 1]  # not in order, one spare
                landing = other[1 : 1 + dp.landing_pages_needed(blocks)]
                prefill_rows = [3 + 2 * j for j in range(blocks)]
                _golden(seq_len, group, decode_pages, landing, prefill_rows)

    def test_wrong_landing_group_is_rejected(self):
        plan = dp.DpPages(pages=16, num_blocks=24)
        with self.assertRaisesRegex(ValueError, "landing pages of group 1"):
            plan.transfer_rows(16, [1, 2], [3])


class TestDpAllocator(CustomTestCase):
    PAGE = 16

    def _allocator(self, pages=8):
        import torch

        from sglang_tpu.k3_programs.sglang_dp_alloc import K3DpPagedAllocator

        return K3DpPagedAllocator(
            size=pages * self.PAGE,
            page_size=self.PAGE,
            dtype=torch.bfloat16,
            device="cpu",
            kvcache=None,
            need_sort=True,
        )

    def _extend(self, alloc, tokens, last_loc=-1, prefix=0):
        import torch

        return alloc.alloc_extend(
            prefix_lens=torch.tensor([prefix]),
            prefix_lens_cpu=torch.tensor([prefix]),
            seq_lens=torch.tensor([tokens]),
            seq_lens_cpu=torch.tensor([tokens]),
            last_loc=torch.tensor([last_loc]),
            extend_num_tokens=tokens - prefix,
        )

    def test_a_request_stays_in_its_group(self):
        alloc = self._allocator()
        g0 = alloc.assign("a")
        g1 = alloc.assign("b")
        self.assertNotEqual(g0, g1)  # balanced by live requests
        alloc.pending_group = g0
        a = self._extend(alloc, 3 * self.PAGE)
        pages_a = sorted({int(x) // self.PAGE for x in a.tolist()})
        self.assertEqual({alloc.group_of_page(p) for p in pages_a}, {g0})
        alloc.pending_group = g1
        b = self._extend(alloc, 2 * self.PAGE + 1)
        pages_b = {int(x) // self.PAGE for x in b.tolist()}
        self.assertEqual({alloc.group_of_page(p) for p in pages_b}, {g1})
        # Decoding past a page boundary takes the next page from the same group.
        import torch

        out = alloc.alloc_decode(
            seq_lens=torch.tensor([3 * self.PAGE + 1]),
            seq_lens_cpu=torch.tensor([3 * self.PAGE + 1]),
            last_loc=torch.tensor([int(a[-1])]),
        )
        self.assertEqual(alloc.group_of_page(int(out[0]) // self.PAGE), g0)

    def test_landing_pages_and_capacity(self):
        alloc = self._allocator(pages=8)
        alloc.pending_group = 0
        self._extend(alloc, 4 * self.PAGE)  # group 0 full
        self.assertEqual(alloc.free_in_group(0), 0)
        self.assertEqual(alloc.available_size(), 0)  # the fuller group bounds both
        idle = self._allocator(pages=9)  # an odd last page: idle availability is all of it
        self.assertEqual(idle.available_size(), 9 * self.PAGE)
        landing = alloc.alloc_landing(1, 2)
        self.assertEqual({alloc.group_of_page(int(p)) for p in landing}, {1})
        self.assertIsNone(alloc.alloc_landing(1, 3))
        alloc.free_landing(landing)
        alloc.merge_and_sort_free()
        self.assertEqual(alloc.free_in_group(1), 4)
        self.assertIsNone(self._extend(alloc, self.PAGE))  # group 0 has no page left


class TestSwapOp(CustomTestCase):
    def test_swap_moves_landed_rows_to_the_group_on_32_devices(self):
        """The swap body under shard_map on 32 host devices (a subprocess:
        the device count is fixed at jax import)."""
        import os
        import subprocess
        import sys
        import textwrap

        script = textwrap.dedent(
            """
            import os
            os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=32"
            os.environ["JAX_PLATFORMS"] = "cpu"
            import numpy as np, jax, jax.numpy as jnp
            from jax.sharding import Mesh, PartitionSpec as P
            from sglang_tpu.k3_programs import sglang_dp as dp
            N = 24
            mesh = Mesh(np.asarray(jax.devices()[:32]), ("tp",))
            plan = dp.DpPages(pages=16, num_blocks=N)
            reqs = []
            for group, length in ((0, 40000), (1, 70000)):
                region, other = list(plan.region(group)), list(plan.region(1 - group))
                blocks = -(-length // dp.PAGE_TOKENS)
                reqs.append((group, length, region[1:1 + blocks],
                             other[-dp.landing_pages_needed(blocks):]))
            pool = np.zeros((32, N, 8), np.uint8)
            for group, length, pages, landing in reqs:
                chunks = -(-length // dp.CHUNK_TOKENS)
                for rank in range(32):
                    for j, row in enumerate(plan.transfer_rows(rank, pages, landing)):
                        k = 32 * j + rank
                        if k < chunks:
                            pool[rank, row] = [group + 1, k % 256, k // 256, 0, 0, 0, 0, 0]
            jobs = [(*plan.swap_rows(p, l), g) for g, _, p, l in reqs]
            ((s, d, g),) = dp.swap_operands(jobs)
            fn = jax.jit(jax.shard_map(dp.swap_body("tp"), mesh=mesh,
                in_specs=(P(), P(), P(), P("tp")), out_specs=(P("tp"),), check_vma=False))
            (out,) = fn(jnp.asarray(s), jnp.asarray(d), jnp.asarray(g),
                        jnp.asarray(pool.reshape(32 * N, 8)))
            out = np.asarray(out).reshape(32, N, 8)
            for group, length, pages, landing in reqs:
                table = plan.table_row(pages, length, -(-length // dp.GROUP_BLOCK_TOKENS))
                for k in range(-(-length // dp.CHUNK_TOKENS)):
                    ident = int(table[k // 16])
                    cell = out[16 * group + k % 16, ident % N]
                    assert (cell[0], cell[1], cell[2]) == (group + 1, k % 256, k // 256), (group, k)
            print("OK")
            """
        )
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, env=env, timeout=600
        )
        self.assertIn("OK", result.stdout, result.stderr[-3000:])


class TestDpHooks(CustomTestCase):
    PAGE = 16

    def setUp(self):
        import torch
        from types import SimpleNamespace

        from sglang_tpu.k3_programs.sglang_dp_alloc import K3DpPagedAllocator

        self.kvcache = SimpleNamespace(plan=SimpleNamespace(num_blocks=40))
        self.alloc = K3DpPagedAllocator(
            size=16 * self.PAGE, page_size=self.PAGE, dtype=torch.bfloat16,
            device="cpu", kvcache=self.kvcache, need_sort=True,
        )
        self.queue = SimpleNamespace(token_to_kv_pool_allocator=self.alloc, tp_rank=0)

    def _prealloc(self, rid, tokens):
        import torch
        from types import SimpleNamespace

        from sglang_tpu.hooks import k3_program as hooks

        req = SimpleNamespace(rid=rid)

        def original(queue, req, *args, **kwargs):
            return self.alloc.alloc_extend(
                prefix_lens=torch.tensor([0]), prefix_lens_cpu=torch.tensor([0]),
                seq_lens=torch.tensor([tokens]), seq_lens_cpu=torch.tensor([tokens]),
                last_loc=torch.tensor([-1]), extend_num_tokens=tokens,
            )

        kv = hooks.k3_program_dp_prealloc(original, self.queue, req)
        return req, kv

    def test_prealloc_transfer_rows_and_release(self):
        import numpy as np
        from types import SimpleNamespace

        from sglang_tpu.hooks import k3_program as hooks
        from sglang_tpu.k3_programs import sglang_dp as dp

        req, kv = self._prealloc("a", 3 * self.PAGE)
        pages = sorted({int(x) // self.PAGE for x in kv.tolist()})
        group = self.alloc.group_of_page(pages[0])
        self.assertIsNone(self.alloc.pending_group)
        landing = [int(p) for p in req._k3_dp_landing]
        self.assertEqual(len(landing), 2)
        self.assertEqual({self.alloc.group_of_page(p) for p in landing}, {1 - group})
        layout = dp.DpPages(self.alloc.dp_pages, 40)
        for rank in (0, 5, 16, 31):
            self.queue.tp_rank = rank
            rows = hooks.k3_program_dp_transfer_rows(
                lambda q, d, k, p: np.asarray(pages, np.int32),
                self.queue, SimpleNamespace(req=req), None, self.PAGE,
            )
            self.assertEqual(rows.tolist(), layout.transfer_rows(rank, pages, landing))
        self.assertEqual(req._k3_dp_transfer, (pages, landing))
        free_before = self.alloc.free_in_group(1 - group)
        hooks.k3_program_dp_release(None, req)
        self.alloc.merge_and_sort_free()
        self.assertEqual(self.alloc.free_in_group(1 - group), free_before + 2)
        self.assertEqual(self.alloc.group_requests, [0, 0])

    def test_a_request_lifecycle_returns_every_page(self):
        """Prealloc, transfer rows, the swap's landing release, core's page
        free and the release hook: availability is whole again (staged pages
        count: the leak check reads it)."""
        import numpy as np
        import torch
        from types import SimpleNamespace

        from sglang_tpu.hooks import k3_program as hooks

        full = self.alloc.available_size()
        req, kv = self._prealloc("a", 5)
        pages = sorted({int(x) // self.PAGE for x in kv.tolist()})
        hooks.k3_program_dp_transfer_rows(
            lambda q, d, k, p: np.asarray(pages, np.int32),
            self.queue, SimpleNamespace(req=req), None, self.PAGE,
        )
        self.assertLess(self.alloc.available_size(), full)
        self.alloc.free_landing(torch.as_tensor(np.asarray(req._k3_dp_transfer[1], np.int32)))
        req._k3_dp_transfer = (req._k3_dp_transfer[0], ())
        self.alloc.free(kv)
        hooks.k3_program_dp_release(None, req)
        self.assertEqual(self.alloc.available_size(), full)

    def test_admission_reserves_landing_pages(self):
        from sglang_tpu.hooks import k3_program as hooks

        need = hooks.k3_program_dp_admission(lambda q, **k: 3 * self.PAGE, self.queue, fill_len=48, prefix_len=0)
        self.assertEqual(need, 3 * self.PAGE + 2 * self.PAGE)


if __name__ == "__main__":
    unittest.main()
