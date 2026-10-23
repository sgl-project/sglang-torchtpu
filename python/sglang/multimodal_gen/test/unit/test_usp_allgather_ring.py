"""All-gather ring attention equals attention over the whole sequence."""

import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.multimodal_gen.runtime.layers import usp

_rank = threading.local()


def _attention(q, k, v, real=None):
    """float32 [B, H, S, D] attention and natural-log LSE over the first `real` keys."""
    scores = torch.einsum("bhsd,bhtd->bhst", q.float(), k.float()) * q.shape[-1] ** -0.5
    if real is not None:
        scores[..., real:] = float("-inf")
    lse = torch.logsumexp(scores, dim=-1)
    probs = torch.exp(scores - lse[..., None]).nan_to_num(0.0)
    return torch.einsum("bhst,bhtd->bhsd", probs, v.float().nan_to_num(0.0)), lse


class _ResumeAttention:
    """forward_ring_resume over stacked heads-major shards, as the TPU impl."""

    def forward_ring_resume(self, query, key, value, kv_info, carry=None):
        offset, real = (int(x) for x in kv_info)
        batch, kv_heads = query.shape[0], key.shape[1]
        rows = slice(offset, offset + batch * kv_heads)
        shard = (batch, kv_heads, *key.shape[2:])
        k = key.reshape(-1, *key.shape[2:])[rows].view(shard)
        v = value.reshape(-1, *value.shape[2:])[rows].view(shard)
        repeat = query.shape[1] // kv_heads
        k, v = k.repeat_interleave(repeat, 1), v.repeat_interleave(repeat, 1)
        out, lse = _attention(query, k, v, real)
        if carry is not None:
            out_prev, lse_prev = carry
            total = torch.logaddexp(lse, lse_prev)
            w = torch.exp(lse - total).nan_to_num(0.0)[..., None]
            w_prev = torch.exp(lse_prev - total).nan_to_num(0.0)[..., None]
            out, lse = out * w + out_prev * w_prev, total
        return out.to(query.dtype), (out, lse)


def _run_ring(world_size, q_shards, k_shards, v_shards, impl, kv_real_len=None):
    barrier = threading.Barrier(world_size)
    slots = [None] * world_size
    outputs = [None] * world_size
    events = [[] for _ in range(world_size)]

    def all_gather_tensor(t, gather_dim, group):
        events[_rank.value].append("all_gather")
        slots[_rank.value] = t.clone()
        barrier.wait()
        gathered = torch.cat(list(slots), dim=gather_dim)
        barrier.wait()
        return gathered

    def all_to_all_single(t, output_split_sizes, input_split_sizes, group):
        # Equal splits: rank r receives element r of every rank's input.
        events[_rank.value].append("all_to_all")
        return torch.full_like(t, float(t[_rank.value]))

    def run(rank):
        _rank.value = rank
        outputs[rank] = usp.allgather_kv_ring_attn(
            q_shards[rank],
            k_shards[rank],
            v_shards[rank],
            attn_impl=impl,
            kv_real_len=kv_real_len,
        )

    with (
        patch.object(usp, "get_sp_group", lambda: SimpleNamespace(ring_group=object())),
        patch.object(usp.dist, "get_world_size", lambda group=None: world_size),
        patch.object(usp.ft_c, "all_gather_tensor", all_gather_tensor),
        patch.object(usp.ft_c, "all_to_all_single", all_to_all_single),
    ):
        threads = [threading.Thread(target=run, args=(r,)) for r in range(world_size)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    return outputs, events


def _inputs(world_size, shard, heads=4, kv_heads=4, batch=2):
    torch.manual_seed(0)
    q = torch.randn(batch, shard * world_size, heads, 8)
    k = torch.randn(batch, shard * world_size, kv_heads, 8)
    v = torch.randn(batch, shard * world_size, kv_heads, 8)
    return q, k, v


def _reference(q, k, v, real=None):
    # [B, S, H, D] in and out; GQA by repeating KV heads.
    k = k.repeat_interleave(q.shape[2] // k.shape[2], dim=2)
    v = v.repeat_interleave(q.shape[2] // v.shape[2], dim=2)
    out, _ = _attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), real)
    return out.transpose(1, 2)


class TestAllGatherRingAttention(unittest.TestCase):
    def test_matches_attention_over_all_keys(self):
        for world_size, kv_heads in ((2, 4), (3, 4), (2, 2)):
            with self.subTest(world=world_size, kv_heads=kv_heads):
                shard = 6
                q, k, v = _inputs(world_size, shard, kv_heads=kv_heads)
                expected = _reference(q, k, v)
                outputs, _ = _run_ring(
                    world_size,
                    q.chunk(world_size, dim=1),
                    k.chunk(world_size, dim=1),
                    v.chunk(world_size, dim=1),
                    _ResumeAttention(),
                )
                for rank, out in enumerate(outputs):
                    rows = slice(rank * shard, (rank + 1) * shard)
                    torch.testing.assert_close(
                        out, expected[:, rows], rtol=1e-5, atol=1e-6
                    )

    def test_tail_padding_is_skipped_and_pad_rows_are_zero(self):
        world_size, shard, real = 3, 6, 14  # last shard: 2 real keys, 4 pad
        q, k, v = _inputs(world_size, shard)
        k[:, real:], v[:, real:] = float("nan"), float("nan")
        expected = _reference(q, k, v, real)
        outputs, _ = _run_ring(
            world_size,
            q.chunk(world_size, dim=1),
            k.chunk(world_size, dim=1),
            v.chunk(world_size, dim=1),
            _ResumeAttention(),
            kv_real_len=real,
        )
        out = torch.cat(outputs, dim=1)
        torch.testing.assert_close(
            out[:, :real], expected[:, :real], rtol=1e-5, atol=1e-6
        )
        self.assertTrue(torch.all(out[:, real:] == 0))

    def test_ranks_make_the_same_calls(self):
        # Ranks running collectives inside one compiled program must compile
        # the same program: the shard choice travels as data, never as Python
        # values that differ per rank.
        for kv_real_len in (None, 9):
            with self.subTest(kv_real_len=kv_real_len):
                world_size, shard = 2, 6
                calls = [[] for _ in range(world_size)]

                class _Recording(_ResumeAttention):
                    def forward_ring_resume(self, q, k, v, kv_info, carry=None):
                        calls[_rank.value].append(
                            (
                                tuple(k.shape),
                                k.storage_offset(),
                                carry is None,
                                torch.is_tensor(kv_info),
                            )
                        )
                        return super().forward_ring_resume(q, k, v, kv_info, carry)

                q, k, v = _inputs(world_size, shard, batch=1)
                _, events = _run_ring(
                    world_size,
                    q.chunk(world_size, dim=1),
                    k.chunk(world_size, dim=1),
                    v.chunk(world_size, dim=1),
                    _Recording(),
                    kv_real_len=kv_real_len,
                )
                # Local shard first, then the whole gathered stack.
                self.assertEqual(
                    calls[0],
                    [
                        ((1, 4, shard, 8), 0, True, True),
                        ((2, 4, shard, 8), 0, False, True),
                    ],
                )
                self.assertEqual(calls[1], calls[0])
                # The ring position is requested before the K/V gathers.
                self.assertEqual(events[0], ["all_to_all", "all_gather", "all_gather"])


if __name__ == "__main__":
    unittest.main()
