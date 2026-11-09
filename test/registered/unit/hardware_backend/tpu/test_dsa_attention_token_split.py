"""DSA attention token split (SGLANG_TPU_ENABLE_DSA_ATTENTION_TOKEN_SPLIT), host side.

Which batches split; index_topk keeping a rank's rows local; and the
all-to-all layout with the ranks simulated as threads: each window op gets its
window's queries for every head, its rows and the window's cu_q_lens, and each
rank ends with its heads' outputs for every token. The kernels on hardware:
test/registered/tpu/unit/test_{oneshot_all_to_all,mla_sparse_prefill_window}.py.
"""

from sglang.test.ci.ci_register import register_cpu_ci, register_tpu_ci

# The q-block gate test needs JAX: it skips on the CPU runner and runs in the
# TPU unit suite.
register_cpu_ci(est_time=20, suite="base-a-test-cpu")
register_tpu_ci(est_time=20, suite="stage-a-test-tpu-unit", chips=1)

import importlib.util
import threading
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.test.test_utils import CustomTestCase
from sglang_tpu.attention_pallas_mla_sparse import TpuPallasSparseMlaAttnBackend

_HAS_JAX = importlib.util.find_spec("jax") is not None


def _split_rows(num_tokens, tp):
    """Rows per rank as the backend computes them: ceil(T / tp), rounded up to 8."""
    rows = -(-num_tokens // tp)
    return -(-rows // 8) * 8


class _Mode:
    def __init__(self, extend=True, verify=False):
        self.extend, self.verify = extend, verify

    def is_extend_or_draft_extend_or_mixed(self):
        return self.extend and not self.verify

    def is_extend_without_speculative(self):
        return self.extend and not self.verify


def _backend(tp, *, pages_per_seq=1089, qblock_op=True, split=True, dqs=1):
    """A backend as the K3.1 recipe sets it up (139264 context, the q-block
    flag on), built without __init__."""
    backend = TpuPallasSparseMlaAttnBackend.__new__(TpuPallasSparseMlaAttnBackend)
    backend._idx_split_group = SimpleNamespace(world_size=tp)
    backend._idx_split_size = tp
    backend._idx_split_min_tokens = max(1024, 32 * tp)
    backend._attn_split_world = tp if split else 0
    backend._prefill_op = object() if qblock_op else None
    backend.page_size = 128
    backend.forward_metadata = SimpleNamespace(
        decode_query_size=dqs,
        page_indices=torch.zeros(8 * pages_per_seq, dtype=torch.int32),
        cu_q_lens=torch.zeros(9, dtype=torch.int32),
        distribution=torch.zeros(3, dtype=torch.int32),
        segments="mixed",
    )
    return backend


def _rows(backend, num_tokens, **mode):
    batch = SimpleNamespace(forward_mode=_Mode(**mode))
    return backend._attention_split_rows(batch, num_tokens)


class TestAttentionSplitRows(CustomTestCase):
    def test_per_query_prefill_splits(self):
        # Without the q-block flag every prefill batch is attended per query.
        self.assertEqual(_rows(_backend(32, qblock_op=False), 4096), 128)
        self.assertEqual(_rows(_backend(32, qblock_op=False), 1024), 32)
        self.assertEqual(_rows(_backend(8, qblock_op=False), 4096), 512)

    def test_unsplit_batches(self):
        self.assertEqual(_rows(_backend(32), 4096, extend=False), 0)  # decode
        self.assertEqual(_rows(_backend(32), 4096, verify=True), 0)
        self.assertEqual(_rows(_backend(32, dqs=8), 4096), 0)
        self.assertEqual(_rows(_backend(32), 512), 0)  # indexer unsplit
        self.assertEqual(_rows(_backend(32, split=False), 4096), 0)  # flag off
        bare = TpuPallasSparseMlaAttnBackend.__new__(TpuPallasSparseMlaAttnBackend)
        self.assertEqual(_rows(bare, 4096), 0)

    def test_rows_that_leave_the_layer(self):
        # MTP top-k sharing and the top-k capturer need every token's rows.
        backend = _backend(32, qblock_op=False)
        seed = SimpleNamespace(dsa_seed_topk_capture=torch.zeros(1))
        for batch in (
            SimpleNamespace(forward_mode=_Mode(), reuse_dsa_topk_indices=True),
            SimpleNamespace(forward_mode=_Mode(), spec_info=seed),
        ):
            self.assertEqual(backend._attention_split_rows(batch, 4096), 0)
        with mock.patch(
            "sglang.srt.state_capturer.indexer_topk.get_global_indexer_capturer",
            lambda: object(),
        ):
            self.assertEqual(_rows(backend, 4096), 0)


@unittest.skipUnless(_HAS_JAX, "requires JAX")
class TestAttentionSplitRowsQblockGate(CustomTestCase):
    def test_gate(self):
        # 1089 pages (139264 context): the q-block kernel never takes these.
        self.assertEqual(_rows(_backend(32), 4096), 128)
        # 256 pages (32K context): it attends them, unless the flag is off.
        self.assertEqual(_rows(_backend(32, pages_per_seq=256), 4096), 0)
        backend = _backend(32, pages_per_seq=256, qblock_op=False)
        self.assertEqual(_rows(backend, 4096), 128)


class TestAttentionSplitNeedsTheIndexerSplit(CustomTestCase):
    def test_flag_without_the_indexer_split_raises(self):
        from sglang_tpu.envs import envs as tpu_envs

        backend = TpuPallasSparseMlaAttnBackend.__new__(TpuPallasSparseMlaAttnBackend)
        backend._idx_split_group = None  # SGLANG_TPU_DSA_INDEXER_TOKEN_SPLIT off
        backend._idx_split_size = 1
        with (
            tpu_envs.SGLANG_TPU_ENABLE_DSA_ATTENTION_TOKEN_SPLIT.override(True),
            mock.patch("torch.distributed.get_world_size", return_value=4),
            mock.patch("torch.distributed.get_rank", return_value=0),
            mock.patch(
                "sglang.srt.runtime_context.get_parallel",
                return_value=SimpleNamespace(attn_tp_rank=0),
            ),
        ):
            with self.assertRaisesRegex(
                RuntimeError, "SGLANG_TPU_DSA_INDEXER_TOKEN_SPLIT"
            ):
                backend._init_attention_token_split()


class TestIndexTopkKeepsLocalRows(CustomTestCase):
    def _index_topk(self, attention_split):
        # T=2048 at TP4: 512 rows per rank, the indexer split's default cap.
        backend = _backend(4, split=attention_split, qblock_op=False)
        local = torch.arange(512 * 16, dtype=torch.int32).view(512, 16)
        backend._index_topk_split_op = mock.MagicMock(return_value=local)
        backend._idx_split_group.all_reduce = mock.MagicMock(side_effect=lambda t: t)
        backend.token_to_kv_pool = mock.MagicMock()
        backend._kv_lens = lambda forward_batch: None
        backend._topk = 16
        with mock.patch(
            "sglang.srt.layers.dp_attention._rank_tensor",
            lambda axis, device: torch.tensor(1),
        ):
            out = backend.index_topk(
                forward_batch=SimpleNamespace(forward_mode=_Mode()),
                layer_id=0,
                query=torch.zeros(2048, 32, 128),
                key=torch.zeros(2048, 128),
                head_weights=torch.zeros(2048, 32),
                weight_scale=1.0,
            )
        return out, local, backend._idx_split_group.all_reduce

    def test_split_returns_this_ranks_rows(self):
        out, local, all_reduce = self._index_topk(attention_split=True)
        self.assertIs(out, local)
        all_reduce.assert_not_called()

    def test_without_it_the_rows_are_assembled(self):
        out, local, all_reduce = self._index_topk(attention_split=False)
        all_reduce.assert_called_once()
        self.assertEqual(tuple(out.shape), (2048, 16))
        self.assertTrue(torch.equal(out[512:1024], local))  # rank 1's slot


class _Ranks:
    """``world`` ranks as threads, and the all-to-all op they share."""

    def __init__(self, world):
        self.world = world
        self.local = threading.local()
        self.barrier = threading.Barrier(world)
        self.sent = [None] * world
        self.calls = 0

    def all_to_all(self, x):
        """The op's contract: [world, rows, 128]; block j goes to rank j."""
        me = self.local.rank
        assert x.shape[0] == self.world and x.shape[2] == 128, x.shape
        self.sent[me] = x.clone()
        self.barrier.wait()
        out = torch.stack([self.sent[i][me] for i in range(self.world)])
        self.barrier.wait()  # every rank has read before the next exchange
        self.calls += me == 0
        return out

    def run(self, fn):
        results, errors = [None] * self.world, []

        def worker(rank):
            self.local.rank = rank
            try:
                results[rank] = fn(rank)
            except BaseException as e:  # surface the first failure
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


class TestForwardMlaTokenSplit(CustomTestCase):
    # Narrow stand-ins for the 512 NoPE / 64 RoPE lanes keep 32 simulated
    # ranks light; the layout code reads the widths from the tensors.
    NOPE, ROPE, TOPK = 16, 8, 16

    def _run(self, world, num_tokens, cu_q_lens, heads=3):
        rows = _split_rows(num_tokens, world)
        g = torch.Generator().manual_seed(world * 10_000 + num_tokens)
        ql = [
            torch.randn(num_tokens, heads, self.NOPE, generator=g) for _ in range(world)
        ]
        qpe = [
            torch.randn(num_tokens, heads, self.ROPE, generator=g) for _ in range(world)
        ]
        topk = torch.randint(-1, 1 << 17, (world * rows, self.TOPK), generator=g)
        cu_q_lens = torch.tensor(cu_q_lens, dtype=torch.int32)
        ranks, seen = _Ranks(world), {}

        def window_op(ql_w, qpe_w, *rest):
            # rest: the per-query op's arguments after q_pe; [7] is the
            # cu_kv_lens slot (the window's cu_q_lens), [12] the top-k rows.
            rank = ranks.local.rank
            seen[rank] = (ql_w, qpe_w, rest[7], rest[12])
            tokens = rank * rows + torch.arange(rows)
            return ql_w * 2 + tokens.view(-1, 1, 1).float()

        def rank_main(rank):
            backend = TpuPallasSparseMlaAttnBackend.__new__(
                TpuPallasSparseMlaAttnBackend
            )
            backend._attn_split_world = world
            backend._window_op = window_op
            args = (ql[rank], qpe[rank]) + (None,) * 6 + (cu_q_lens, None, None)
            args += (0.1, 1.0, 1.0)
            local_topk = topk[rank * rows : (rank + 1) * rows]
            return backend._forward_mla_token_split(args, local_topk, rows)

        with (
            mock.patch(
                "sglang_tpu.oneshot_collectives.get_oneshot_all_to_all_op",
                lambda world_size: ranks.all_to_all,
            ),
            mock.patch(
                "sglang.srt.layers.dp_attention._rank_tensor",
                lambda axis, device: torch.tensor(ranks.local.rank),
            ),
        ):
            outs = ranks.run(rank_main)

        self.assertEqual(ranks.calls, 2)  # queries out, outputs back
        pad = (0, 0, 0, 0, 0, 0, 0, world * rows - num_tokens)
        all_ql = torch.nn.functional.pad(torch.stack(ql, 1), pad)
        all_qpe = torch.nn.functional.pad(torch.stack(qpe, 1), pad)
        tokens = torch.arange(num_tokens).view(-1, 1, 1).float()
        for rank in range(world):
            ql_w, qpe_w, window_cu, topk_w = seen[rank]
            win = slice(rank * rows, (rank + 1) * rows)
            with self.subTest(world=world, T=num_tokens, rank=rank):
                # Its window, every rank's heads in rank order.
                self.assertTrue(torch.equal(ql_w, all_ql[win].flatten(1, 2)))
                self.assertTrue(torch.equal(qpe_w, all_qpe[win].flatten(1, 2)))
                self.assertTrue(torch.equal(topk_w, topk[win]))
                want_cu = (cu_q_lens - rank * rows).clamp(0, rows)
                self.assertTrue(torch.equal(window_cu, want_cu.int()))
                self.assertEqual(
                    tuple(outs[rank].shape), (num_tokens, heads, self.NOPE)
                )
                self.assertTrue(torch.equal(outs[rank], ql[rank] * 2 + tokens))

    def test_layouts(self):
        # The 4096-token bucket at TP32 (K3.1) and TP8, with requests across
        # windows and padded tokens after the last; T=200 at 4 ranks, whose
        # 56-row windows end 24 rows past T and need tile padding in the
        # all-to-all.
        self._run(32, 4096, [0, 37, 1000, 4083, 4083, 4083, 4083, 4083, 4083])
        self._run(8, 4096, [0, 3000, 4096, 4096, 4096, 4096, 4096, 4096, 4096])
        self._run(4, 200, [0, 10, 170, 170, 170])


if __name__ == "__main__":
    unittest.main()
