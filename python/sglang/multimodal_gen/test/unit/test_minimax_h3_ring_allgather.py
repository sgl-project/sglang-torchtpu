"""MiniMax H3 ring attention by K/V all-gather (backends without p2p, e.g. TPU):
the packed core attends every real key, reads the live length on device, and
zeroes the trailing padding rows."""

import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.multimodal_gen.runtime.layers import usp
from sglang.multimodal_gen.runtime.models.dits import minimax_h3 as dit
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.test.unit.test_usp_allgather_ring import (
    _attention,
    _ResumeAttention,
)

_rank = threading.local()
_HEADS, _HEAD_DIM = 4, 8


class _AllGatherRingImpl(_ResumeAttention):
    @classmethod
    def supports_ring_resume(cls) -> bool:
        return True


class _NoRingImpl:
    @classmethod
    def supports_ring_resume(cls) -> bool:
        return False


def _attention_module(impl, backend=AttentionBackendEnum.OOT):
    return SimpleNamespace(
        _attention_impl=impl, _attention_backend_enum=backend, prefix="blocks.0.attn"
    )


def _packed(total, used, seed=0):
    gen = torch.Generator().manual_seed(seed)
    q, k, v = (torch.randn(total, _HEADS, _HEAD_DIM, generator=gen) for _ in range(3))
    # Keys past the live length are padding and may hold anything.
    k[used:], v[used:] = float("nan"), float("nan")
    return q, k, v, torch.tensor([0, used, total], dtype=torch.int32)


def _ring_core(world_size, attention, q, k, v, cu_seqlens):
    """Each rank's packed core output, ranks as threads sharing collectives."""
    shard = q.shape[0] // world_size
    barrier = threading.Barrier(world_size)
    slots, outputs, live = [None] * world_size, [None] * world_size, []

    def all_gather_tensor(t, gather_dim, group):
        slots[_rank.value] = t.clone()
        barrier.wait()
        gathered = torch.cat(list(slots), dim=gather_dim)
        barrier.wait()
        return gathered

    def all_to_all_single(t, output_split_sizes, input_split_sizes, group):
        return torch.full_like(t, float(t[_rank.value]))

    ring = usp.allgather_kv_ring_attn

    def recording_ring(*args, **kwargs):
        live.append(kwargs["kv_real_len"])
        return ring(*args, **kwargs)

    def run(rank):
        _rank.value = rank
        rows = slice(rank * shard, (rank + 1) * shard)
        outputs[rank] = dit._minimax_h3_attention_core_impl(
            attention,
            q[rows],
            k[rows],
            v[rows],
            cu_seqlens=cu_seqlens,
            cu_seqlens_host=None,
            max_seqlen=q.shape[0],
            ulysses_active=False,
            ring_active=True,
        )

    with (
        patch.object(usp, "get_sp_group", lambda: SimpleNamespace(ring_group=object())),
        patch.object(usp.dist, "get_world_size", lambda group=None: world_size),
        patch.object(usp.ft_c, "all_gather_tensor", all_gather_tensor),
        patch.object(usp.ft_c, "all_to_all_single", all_to_all_single),
        patch.object(dit, "get_ring_ctx", lambda: (world_size, _rank.value)),
        patch.object(dit, "allgather_kv_ring_attn", recording_ring),
    ):
        threads = [threading.Thread(target=run, args=(r,)) for r in range(world_size)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    return torch.cat(outputs), live


class TestMiniMaxH3AllGatherRing(unittest.TestCase):
    def test_core_attends_the_live_rows_of_every_shard(self):
        # (world, total, used): no padding, padding inside the last shard, and
        # a last shard that is all padding.
        for world_size, total, used in ((2, 24, 24), (2, 24, 19), (4, 32, 21)):
            with self.subTest(world=world_size, total=total, used=used):
                q, k, v, cu_seqlens = _packed(total, used, seed=used)
                attention = _attention_module(_AllGatherRingImpl())
                self.assertTrue(dit._minimax_h3_rings_by_allgather(attention))
                out, live = _ring_core(world_size, attention, q, k, v, cu_seqlens)
                expected, _ = _attention(
                    *(t[:used].transpose(0, 1)[None] for t in (q, k, v))
                )
                expected = expected[0].transpose(0, 1)
                torch.testing.assert_close(out[:used], expected, rtol=1e-5, atol=1e-6)
                self.assertTrue(torch.all(out[used:] == 0))
                # The live length reaches the ring as device data, the same
                # program on every rank and for every prompt length.
                self.assertEqual(len(live), world_size)
                for length in live:
                    self.assertTrue(torch.is_tensor(length))
                    self.assertEqual(int(length), used)

    def test_p2p_backends_keep_the_varlen_ring(self):
        q, k, v, cu_seqlens = _packed(16, 13)
        for backend in (AttentionBackendEnum.FA, AttentionBackendEnum.AITER):
            with self.subTest(backend=backend.name):
                attention = _attention_module(_AllGatherRingImpl(), backend)
                self.assertFalse(dit._minimax_h3_rings_by_allgather(attention))
                with (
                    patch.object(dit, "get_ring_ctx", lambda: (2, 0)),
                    patch.object(
                        dit, "_ring_attention_varlen", return_value=q
                    ) as varlen,
                    patch.object(dit, "allgather_kv_ring_attn") as allgather,
                ):
                    dit._minimax_h3_attention_core_impl(
                        attention,
                        q,
                        k,
                        v,
                        cu_seqlens=cu_seqlens,
                        cu_seqlens_host=(0, 13, 16),
                        max_seqlen=13,
                        ulysses_active=False,
                        ring_active=True,
                    )
                self.assertEqual(varlen.call_args.kwargs["real_seq_len"], 13)
                allgather.assert_not_called()

    def test_backend_without_a_ring_kernel_is_refused(self):
        q, k, v, cu_seqlens = _packed(16, 13)
        attention = _attention_module(_NoRingImpl())
        self.assertFalse(dit._minimax_h3_rings_by_allgather(attention))
        with (
            patch.object(dit, "get_ring_ctx", lambda: (2, 0)),
            self.assertRaisesRegex(NotImplementedError, "all-gather"),
        ):
            dit._minimax_h3_attention_core_impl(
                attention,
                q,
                k,
                v,
                cu_seqlens=cu_seqlens,
                cu_seqlens_host=None,
                max_seqlen=16,
                ulysses_active=False,
                ring_active=True,
            )


if __name__ == "__main__":
    unittest.main()
