"""CPU unit tests for the arithmetic that joins DCP partial attention results.

Four pieces, each checked against a reference built from a plain dense softmax:

* ``merge_across_ranks`` merges one partial (output, LSE) pair per rank into
  the full-sequence result and hands each rank back its own head block.
* ``sink_fold`` applies the attention sink once, after the merge, as
  ``out * sigmoid(lse - sink)``.
* ``merge_lse`` joins the cached-prefix pass with the chunk-local pass of an
  extend step.
* ``get_dcp_lens`` gives each rank the count of the positions it owns.

A DCP group whose ranks each compute every head themselves (attention TP 1,
as in ring-mode prefill context parallelism, where the group is the CP group)
merges without gathering or cropping heads; ``TestAllHeadsLocal`` checks that
variant of the merge and of the backend's decode and extend paths.

The collectives are faked: every rank's call runs in-process and the fake group
serves the other ranks' tensors, so the values crossing the "wire" are the ones
the function under test produced.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.attention_pallas_rpa_base import PallasForwardMetadata
from sglang_tpu.attention_pallas_rpa_batched import (
    TpuPallasRpaBatchedAttnBackend,
)
from sglang_tpu.attention_pallas_rpa_dcp import (
    gather_heads,
    merge_across_ranks,
    merge_lse,
    sink_fold,
)

register_tpu_ci(est_time=60, suite="stage-a-test-tpu-unit", chips=1)

try:
    # The layers.dcp package init pulls in the Triton DCP kernels and Triton is
    # not a TPU dependency, so a Triton-less image skips the cases that need
    # them instead of failing to collect the file.
    from sglang.srt.layers.dcp.layout import get_dcp_lens

    _DCP_IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - depends on the image
    get_dcp_lens = None
    _DCP_IMPORT_ERROR = exc

_requires_dcp_helpers = unittest.skipIf(
    _DCP_IMPORT_ERROR is not None,
    f"sglang.srt.layers.dcp is unavailable: {_DCP_IMPORT_ERROR}",
)


class _FakeDcpGroup:
    """GroupCoordinator stand-in for one rank of a DCP group.

    ``all_reduce`` answers the ``i``-th call with the sum of what every rank
    handed to its own ``i``-th call in the previous simulation pass, and
    records this pass's operand for the next one.
    """

    def __init__(self, world_size, rank, previous, record):
        self.world_size = world_size
        self.rank_in_group = rank
        self._previous = previous
        self._record = record

    def all_reduce(self, tensor):
        call = len(self._record)
        self._record.append(tensor.clone())
        if self._previous is None or any(len(r) <= call for r in self._previous):
            return tensor
        return torch.stack([r[call] for r in self._previous]).sum(dim=0)


def _simulate_group(world, fn, passes=4):
    """Run ``fn(group)`` on every rank until each collective sees the sum of
    the operands the ranks computed for it; ``passes`` >= collectives + 1."""
    previous = None
    results = None
    for _ in range(passes):
        record = [[] for _ in range(world)]
        results = [
            fn(_FakeDcpGroup(world, rank, previous, record[rank]))
            for rank in range(world)
        ]
        previous = record
    return results


def _rank_tensors(world_size, rank):
    """The rank as the device tensors the helpers take: one-hot mask, index."""
    return torch.arange(world_size) == rank, torch.tensor([rank])


def _run_cp_merge(partial_outs, partial_lses):
    """Run ``merge_across_ranks`` on every rank; return each rank's result."""
    world = len(partial_lses)
    return _simulate_group(
        world,
        lambda group: merge_across_ranks(
            partial_outs[group.rank_in_group].clone(),
            partial_lses[group.rank_in_group].clone(),
            group,
            group.world_size,
            *_rank_tensors(world, group.rank_in_group),
        ),
    )


def _softmax_over(scores, values, keys):
    """(out, lse) of attending ``keys`` only. ``keys`` is a bool mask [T,H,L]."""
    masked = torch.where(keys, scores, torch.full_like(scores, -float("inf")))
    lse = torch.logsumexp(masked, dim=-1)
    weights = torch.softmax(masked, dim=-1)
    weights = torch.nan_to_num(weights, nan=0.0)
    # values: [L, H, D] -> out[t, h, d] = sum_k w[t, h, k] * values[k, h, d]
    out = torch.einsum("thk,khd->thd", weights, values)
    return out, lse


def _random_problem(num_tokens, num_heads, num_keys, head_dim, seed):
    generator = torch.Generator().manual_seed(seed)
    scores = torch.randn(
        num_tokens, num_heads, num_keys, generator=generator, dtype=torch.float32
    )
    values = torch.randn(
        num_keys, num_heads, head_dim, generator=generator, dtype=torch.float32
    )
    return scores, values


@_requires_dcp_helpers
class TestCpLseAgOutRsMha(CustomTestCase):
    """The cross-rank merge reproduces a dense softmax over the whole sequence."""

    HEADS_PER_RANK = 3
    HEAD_DIM = 8

    def _check(self, cp, num_tokens, num_keys, seed):
        heads = self.HEADS_PER_RANK * cp
        scores, values = _random_problem(
            num_tokens, heads, num_keys, self.HEAD_DIM, seed
        )
        key_rank = torch.arange(num_keys) % cp

        partial_outs, partial_lses = [], []
        for rank in range(cp):
            owned = (key_rank == rank).view(1, 1, num_keys).expand_as(scores)
            out, lse = _softmax_over(scores, values, owned)
            partial_outs.append(out)
            partial_lses.append(lse)

        ref_out, ref_lse = _softmax_over(scores, values, torch.ones_like(scores).bool())
        results = _run_cp_merge(partial_outs, partial_lses)

        for rank in range(cp):
            head_slice = slice(
                rank * self.HEADS_PER_RANK, (rank + 1) * self.HEADS_PER_RANK
            )
            out, lse = results[rank]
            torch.testing.assert_close(
                out, ref_out[:, head_slice], atol=1e-5, rtol=1e-5
            )
            torch.testing.assert_close(
                lse, ref_lse[:, head_slice], atol=1e-5, rtol=1e-5
            )

    def test_matches_dense_softmax(self):
        for cp in (2, 4, 8):
            for num_keys in (1, 7, 64):
                with self.subTest(cp=cp, num_keys=num_keys):
                    self._check(
                        cp, num_tokens=5, num_keys=num_keys, seed=cp * 100 + num_keys
                    )

    def test_empty_shard_contributes_nothing(self):
        # A rank whose shard is empty reports lse = -inf and out = 0; the merge
        # must absorb it instead of producing NaN.
        cp, num_tokens, num_keys = 4, 3, 2
        heads = self.HEADS_PER_RANK * cp
        scores, values = _random_problem(num_tokens, heads, num_keys, self.HEAD_DIM, 7)
        key_rank = torch.arange(num_keys) % cp

        partial_outs, partial_lses = [], []
        for rank in range(cp):
            owned = (key_rank == rank).view(1, 1, num_keys).expand_as(scores)
            out, lse = _softmax_over(scores, values, owned)
            partial_outs.append(out)
            partial_lses.append(lse)
        # Ranks 2 and 3 own no key here.
        self.assertTrue(torch.isinf(partial_lses[2]).all())

        ref_out, _ = _softmax_over(scores, values, torch.ones_like(scores).bool())
        results = _run_cp_merge(partial_outs, partial_lses)
        for rank in range(cp):
            head_slice = slice(
                rank * self.HEADS_PER_RANK, (rank + 1) * self.HEADS_PER_RANK
            )
            out, _ = results[rank]
            self.assertFalse(torch.isnan(out).any())
            torch.testing.assert_close(
                out, ref_out[:, head_slice], atol=1e-5, rtol=1e-5
            )

    def test_single_rank_group_is_a_passthrough(self):
        out = torch.randn(2, 3, 4)
        lse = torch.randn(2, 3)
        group = _FakeDcpGroup(1, 0, None, [])
        merged, merged_lse = merge_across_ranks(
            out, lse, group, group.world_size, *_rank_tensors(1, 0)
        )
        self.assertIs(merged, out)
        self.assertIs(merged_lse, lse)

    def test_gather_heads_is_rank_major(self):
        # The kernel groups gathered query heads onto KV heads contiguously,
        # so rank r's heads must land in block r.
        cp = 2
        queries = [torch.randn(3, 4, 5) for _ in range(cp)]
        gathered = _simulate_group(
            cp,
            lambda group: gather_heads(
                queries[group.rank_in_group],
                group,
                group.world_size,
                _rank_tensors(cp, group.rank_in_group)[0],
            ),
        )
        expected = torch.cat(queries, dim=1)
        for rank in range(cp):
            torch.testing.assert_close(gathered[rank], expected)


class TestSinkFold(CustomTestCase):
    """The sink is a virtual key with logit ``sink`` and an all-zero value."""

    HEADS = 4
    HEAD_DIM = 6
    KEYS = 11
    TOKENS = 5

    def _reference_with_sink(self, scores, values, sinks):
        sink_column = sinks.view(1, self.HEADS, 1).expand(self.TOKENS, self.HEADS, 1)
        extended = torch.cat([sink_column, scores], dim=-1)
        weights = torch.softmax(extended, dim=-1)[..., 1:]
        return torch.einsum("thk,khd->thd", weights, values)

    def test_matches_a_softmax_with_a_sink_column(self):
        scores, values = _random_problem(
            self.TOKENS, self.HEADS, self.KEYS, self.HEAD_DIM, 11
        )
        out, lse = _softmax_over(scores, values, torch.ones_like(scores).bool())
        for sink_value in (-4.0, 0.0, 3.5):
            with self.subTest(sink=sink_value):
                sinks = torch.full((self.HEADS,), sink_value) + torch.arange(
                    self.HEADS, dtype=torch.float32
                )
                folded = sink_fold(out.clone(), lse, sinks)
                torch.testing.assert_close(
                    folded,
                    self._reference_with_sink(scores, values, sinks),
                    atol=1e-5,
                    rtol=1e-5,
                )

    def test_bfloat16_sinks_are_accepted(self):
        # layer.sinks is bf16 on TPU; the fold must run in fp32.
        scores, values = _random_problem(
            self.TOKENS, self.HEADS, self.KEYS, self.HEAD_DIM, 12
        )
        out, lse = _softmax_over(scores, values, torch.ones_like(scores).bool())
        sinks = torch.arange(self.HEADS, dtype=torch.float32) - 1.0
        folded = sink_fold(out.clone(), lse, sinks.to(torch.bfloat16))
        torch.testing.assert_close(
            folded,
            self._reference_with_sink(scores, values, sinks),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_row_that_attended_nothing_stays_zero(self):
        out = torch.zeros(1, self.HEADS, self.HEAD_DIM)
        lse = torch.full((1, self.HEADS), -float("inf"))
        folded = sink_fold(out, lse, torch.zeros(self.HEADS))
        self.assertFalse(torch.isnan(folded).any())
        torch.testing.assert_close(folded, torch.zeros_like(folded))


class TestMergeLse(CustomTestCase):
    """The extend merge joins the cached prefix with the chunk-local pass."""

    HEADS = 3
    HEAD_DIM = 8

    def _split_problem(self, prefix_len, chunk_len, seed):
        total = prefix_len + chunk_len
        scores, values = _random_problem(
            chunk_len, self.HEADS, total, self.HEAD_DIM, seed
        )
        # Query i of the chunk sits at absolute position prefix_len + i; it sees
        # the whole prefix and the chunk up to itself.
        key_pos = torch.arange(total).view(1, 1, total)
        query_pos = (prefix_len + torch.arange(chunk_len)).view(chunk_len, 1, 1)
        causal = key_pos <= query_pos
        prefix_mask = causal & (key_pos < prefix_len)
        chunk_mask = causal & (key_pos >= prefix_len)
        return scores, values, causal, prefix_mask, chunk_mask

    def test_matches_a_full_causal_reference(self):
        for prefix_len, chunk_len in ((12, 5), (1, 8), (31, 1)):
            with self.subTest(prefix_len=prefix_len, chunk_len=chunk_len):
                scores, values, causal, prefix_mask, chunk_mask = self._split_problem(
                    prefix_len, chunk_len, seed=prefix_len * 10 + chunk_len
                )
                out_a, lse_a = _softmax_over(
                    scores, values, prefix_mask.expand_as(scores)
                )
                out_b, lse_b = _softmax_over(
                    scores, values, chunk_mask.expand_as(scores)
                )
                ref_out, ref_lse = _softmax_over(
                    scores, values, causal.expand_as(scores)
                )

                out, lse = merge_lse(out_a, lse_a, out_b, lse_b)

                torch.testing.assert_close(out, ref_out, atol=1e-5, rtol=1e-5)
                torch.testing.assert_close(lse, ref_lse, atol=1e-5, rtol=1e-5)

    def test_empty_prefix_leaves_the_chunk_result_untouched(self):
        scores, values, causal, _, chunk_mask = self._split_problem(0, 6, seed=5)
        out_b, lse_b = _softmax_over(scores, values, chunk_mask.expand_as(scores))
        out_a = torch.zeros_like(out_b)
        lse_a = torch.full_like(lse_b, -float("inf"))

        out, lse = merge_lse(out_a, lse_a, out_b, lse_b)

        self.assertFalse(torch.isnan(out).any())
        torch.testing.assert_close(out, out_b, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(lse, lse_b, atol=1e-5, rtol=1e-5)
        ref_out, _ = _softmax_over(scores, values, causal.expand_as(scores))
        torch.testing.assert_close(out, ref_out, atol=1e-5, rtol=1e-5)


@_requires_dcp_helpers
class TestGlobalWindowRule(CustomTestCase):
    """Per-rank local lengths from ``get_dcp_lens`` against brute force."""

    DCP_SIZES = (2, 4, 8)
    LENGTHS = (1, 2, 7, 63, 64, 65, 127, 128, 129, 200, 513)

    @staticmethod
    def _owned_positions(length, dcp, rank):
        return [p for p in range(length) if p % dcp == rank]

    def test_local_length_counts_the_owned_positions(self):
        for dcp in self.DCP_SIZES:
            for rank in range(dcp):
                lens = torch.tensor(self.LENGTHS, dtype=torch.int32)
                got = get_dcp_lens(lens, dcp, rank).tolist()
                expected = [
                    len(self._owned_positions(L, dcp, rank)) for L in self.LENGTHS
                ]
                self.assertEqual(got, expected, f"dcp={dcp} rank={rank}")


class TestAllHeadsLocal(CustomTestCase):
    """Attention TP 1 inside a DCP group: every rank holds all heads."""

    HEADS = 6
    HEAD_DIM = 8

    def _partials(self, cp, num_tokens, num_keys, seed, keys=None):
        scores, values = _random_problem(
            num_tokens, self.HEADS, num_keys, self.HEAD_DIM, seed
        )
        keys = torch.ones(num_keys, dtype=torch.bool) if keys is None else keys
        key_rank = torch.arange(num_keys) % cp
        partials = []
        for rank in range(cp):
            owned = ((key_rank == rank) & keys).view(1, 1, num_keys).expand_as(scores)
            partials.append(_softmax_over(scores, values, owned))
        return scores, values, partials

    def test_merge_keeps_every_head_and_matches_dense_softmax(self):
        for cp in (2, 4, 8):
            for num_keys in (1, 7, 64):
                with self.subTest(cp=cp, num_keys=num_keys):
                    scores, values, partials = self._partials(
                        cp, 5, num_keys, seed=cp * 100 + num_keys
                    )
                    ref_out, ref_lse = _softmax_over(
                        scores, values, torch.ones_like(scores).bool()
                    )
                    results = _simulate_group(
                        cp,
                        lambda group: merge_across_ranks(
                            partials[group.rank_in_group][0].clone(),
                            partials[group.rank_in_group][1].clone(),
                            group,
                            group.world_size,
                            *_rank_tensors(cp, group.rank_in_group),
                            keep_all_heads=True,
                        ),
                    )
                    for out, lse in results:
                        # Every rank ends with the same full-head result.
                        self.assertEqual(out.shape, ref_out.shape)
                        self.assertFalse(torch.isnan(out).any())
                        torch.testing.assert_close(out, ref_out, atol=1e-5, rtol=1e-5)
                        torch.testing.assert_close(lse, ref_lse, atol=1e-5, rtol=1e-5)

    def test_default_merge_is_unchanged(self):
        """Without the flag the merge still crops to the rank's head block."""
        cp = 2
        scores, values, partials = self._partials(cp, 3, 9, seed=3)
        ref_out, _ = _softmax_over(scores, values, torch.ones_like(scores).bool())
        results = _run_cp_merge([p[0] for p in partials], [p[1] for p in partials])
        per_rank = self.HEADS // cp
        for rank, (out, _) in enumerate(results):
            torch.testing.assert_close(
                out,
                ref_out[:, rank * per_rank : (rank + 1) * per_rank],
                atol=1e-5,
                rtol=1e-5,
            )

    def _backend(self, cp, rank, group, dcp_op, heads_local=True):
        backend = object.__new__(TpuPallasRpaBatchedAttnBackend)
        backend._dcp = cp
        backend._dcp_group = group
        backend._dcp_heads_local = heads_local
        backend._dcp_op = dcp_op
        backend._cp_rank_mask, backend._cp_rank_index = _rank_tensors(cp, rank)
        tensor = torch.zeros(1, dtype=torch.int32)
        backend.forward_metadata = PallasForwardMetadata(
            kv_lens=tensor,
            cu_q_lens=tensor,
            cu_kv_lens=tensor,
            page_indices=tensor,
            distribution=tensor,
            active_num_tokens=tensor,
            dcp_kv_lens_global=tensor,
            dcp_cp_rank=tensor,
        )
        return backend

    def _run_forward(self, cp, mode, partials, chunk=None):
        """``_forward_dcp`` on every rank with a kernel stub that returns the
        rank's stripe partial (STRIPED scope) or the chunk partial."""
        seen = []
        num_tokens = partials[0][0].shape[0]
        q = torch.randn(num_tokens, self.HEADS, self.HEAD_DIM)

        def run(group):
            rank = group.rank_in_group

            def dcp_op(q_all, *args):
                seen.append(q_all.shape)
                scope = args[-2]
                out, lse = chunk if scope == 2 else partials[rank]
                return out.clone(), lse.clone()

            backend = self._backend(cp, rank, group, dcp_op)
            batch = SimpleNamespace(forward_mode=mode)
            with patch(
                "sglang_tpu.attention_pallas_rpa_base.gather_heads",
                side_effect=AssertionError("heads must not be gathered"),
            ):
                return backend._forward_dcp(
                    q, None, None, None, None, batch, None, 1.0, None, None, None,
                    None, self.HEADS,
                )  # fmt: skip

        results = _simulate_group(cp, run)
        # The kernel ran on the rank's own heads, never on cp * heads.
        self.assertEqual(set(seen), {q.shape})
        return results

    def test_decode_merges_stripes_without_a_head_gather(self):
        cp = 4
        scores, values, partials = self._partials(cp, 3, 21, seed=11)
        ref_out, _ = _softmax_over(scores, values, torch.ones_like(scores).bool())
        for out in self._run_forward(cp, ForwardMode.DECODE, partials):
            self.assertEqual(out.shape, ref_out.shape)
            torch.testing.assert_close(out, ref_out, atol=1e-5, rtol=1e-5)

    def test_extend_joins_the_striped_prefix_with_the_chunk(self):
        """Short-only extends in ring mode run this path: the cached prefix is
        merged across the stripes, the chunk is attended locally on every rank."""
        cp, num_keys, num_prefix = 4, 21, 13
        is_prefix = torch.arange(num_keys) < num_prefix
        scores, values, partials = self._partials(cp, 3, num_keys, 5, keys=is_prefix)
        chunk = _softmax_over(
            scores, values, (~is_prefix).view(1, 1, num_keys).expand_as(scores)
        )
        ref_out, _ = _softmax_over(scores, values, torch.ones_like(scores).bool())
        for out in self._run_forward(cp, ForwardMode.EXTEND, partials, chunk=chunk):
            self.assertEqual(out.shape, ref_out.shape)
            torch.testing.assert_close(out, ref_out, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
