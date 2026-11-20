"""Sparse MLA under decode context parallelism, host side.

The ranks run as threads over a summing all-reduce. Each rank's kernel stand-in
attends only the top-k entries its page stripe owns; gathering the heads,
attending and merging must give every rank its own heads of the unsharded
sparse attention.
The kernel on hardware: test/registered/tpu/unit/test_sparse_mla_dcp_kernel.py.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import threading
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.dcp.layout import remap_dcp_sparse_indices
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.attention_pallas_mla_sparse import TpuPallasSparseMlaAttnBackend
from sglang_tpu.attention_pallas_mla_sparse_dcp import index_page_indices

PAGE = 4  # tokens per page; the stripe unit


class _SumGroup:
    """An all-reduce over ``world`` threads."""

    def __init__(self, world):
        self.world = world
        self.local = threading.local()
        self.barrier = threading.Barrier(world)
        self.sent = [None] * world

    def all_reduce(self, x):
        me = self.local.rank
        self.sent[me] = x.clone()
        self.barrier.wait()
        out = sum(self.sent[i] for i in range(self.world))
        self.barrier.wait()
        return out

    def run(self, fn):
        results, errors = [None] * self.world, []

        def worker(rank):
            self.local.rank = rank
            try:
                results[rank] = fn(rank)
            except BaseException as e:
                errors.append(e)
                self.barrier.abort()

        threads = [
            threading.Thread(target=worker, args=(r,)) for r in range(self.world)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        if errors:
            raise errors[0]
        return results


def _attend(q, keys, rows, scale):
    """Sparse attention with LSE: q [T, H, D], keys [L, D] (the MLA value is the
    key), rows i32[T, K] key indices padded with -1."""
    out = torch.empty(q.shape[:2] + keys.shape[-1:], dtype=torch.float32)
    lse = torch.empty(q.shape[:2], dtype=torch.float32)
    for t in range(q.shape[0]):
        picked = keys[rows[t][rows[t] >= 0]]
        scores = (q[t].float() @ picked.float().T) * scale  # [H, n]
        lse[t] = torch.logsumexp(scores, dim=-1)
        out[t] = torch.softmax(scores, dim=-1) @ picked.float()
    return out, lse


def _stripe(keys, rank, world):
    """This rank's local cache: the pages p with p % world == rank, in order."""
    pages = keys.view(-1, PAGE, keys.shape[-1])
    return pages[rank::world].reshape(-1, keys.shape[-1])


class TestForwardSparseMlaDcp(CustomTestCase):
    def test_merged_heads_equal_unsharded_attention(self):
        world, heads, dim, tokens, width = 4, 2, 8, 5, 12
        length = PAGE * world * 3
        g = torch.Generator().manual_seed(0)
        keys = torch.randn(length, dim, generator=g)
        q_all = torch.randn(tokens, heads * world, dim, generator=g)
        rows = torch.stack(
            [torch.randperm(length, generator=g)[:width] for _ in range(tokens)]
        ).to(torch.int32)
        rows[0, 3:] = -1  # a short row
        # Every pick of token 1 on rank 0's pages: the other ranks own none.
        on_rank0 = [p for p in range(length) if (p // PAGE) % world == 0]
        rows[1] = torch.tensor(on_rank0[:width])
        scale = 0.3
        want, _ = _attend(q_all, keys, rows, scale)
        group = _SumGroup(world)

        def kernel(ql_nope, q_pe, *rest):
            topk, rank_t = rest[-2], rest[-1]
            # args[7], the page table: the latent stripe's, not the indexer's.
            assert rest[5] == "latent stripe", rest[5]
            rank = int(rank_t)
            local = remap_dcp_sparse_indices(topk, world, rank, interleave_size=PAGE)
            owns = (local >= 0).any(dim=1)
            local[:, 0] = torch.where(owns, local[:, 0], 0)  # the dummy key
            q = torch.cat([ql_nope, q_pe], dim=-1)[..., :dim]
            out, lse = _attend(q, _stripe(keys, rank, world), local, scale)
            return out, torch.where(owns[:, None], lse, float("-inf"))

        def rank_main(rank):
            mine = q_all[:, rank * heads : (rank + 1) * heads]
            args = (mine, torch.zeros(tokens, heads, 2)) + ("indexer",) * 12
            backend = SimpleNamespace(
                _mla_kernel_args=lambda **_: args,
                forward_metadata=SimpleNamespace(dcp_latent_page_indices="latent stripe"),
                _dcp_op=kernel,
                _dcp_group=group,
                _dcp=world,
                _cp_rank_t=torch.tensor([rank], dtype=torch.int32),
                _cp_rank_index=torch.tensor([rank]),
                _cp_rank_mask=torch.arange(world) == rank,
            )
            layer = SimpleNamespace(tp_q_head_num=heads, v_head_dim=dim)
            out = TpuPallasSparseMlaAttnBackend._forward_mla_dcp(
                backend, mine, None, layer, None, topk_indices=rows
            )
            return out.view(tokens, heads, dim)

        for rank, got in enumerate(group.run(rank_main)):
            with self.subTest(rank=rank):
                torch.testing.assert_close(
                    got.float(),
                    want[:, rank * heads : (rank + 1) * heads],
                    rtol=1e-5,
                    atol=1e-5,
                )


class TestIndexPageIndices(CustomTestCase):
    def test_index_pages_follow_the_allocator_slots(self):
        """A widened allocator page ``s`` holds slots ``[s * PAGE * dcp, (s + 1) *
        PAGE * dcp)``; the replicated index cache pages those slots by PAGE."""
        dcp, pages_per_seq = 4, 3
        stripe = torch.tensor([[5, 2, 9], [1, 7, 0]], dtype=torch.int32)
        got = index_page_indices(stripe.reshape(-1), num_seqs=2, dcp=dcp).view(2, -1)
        self.assertEqual(tuple(got.shape), (2, pages_per_seq * dcp))
        for seq in range(2):
            for pos in range(0, pages_per_seq * PAGE * dcp, PAGE):
                widened = int(stripe[seq, pos // (PAGE * dcp)])
                slot = widened * PAGE * dcp + pos % (PAGE * dcp)
                self.assertEqual(int(got[seq, pos // PAGE]), slot // PAGE)


if __name__ == "__main__":
    unittest.main()
