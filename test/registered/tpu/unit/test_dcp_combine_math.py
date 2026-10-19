"""CPU unit tests for the arithmetic that joins DCP partial attention results.

Four pieces, each checked against a reference built from a plain dense softmax:

* ``merge_across_ranks`` merges one partial (output, LSE) pair per rank into
  the full-sequence result and hands each rank back its own head block.
* ``sink_fold`` applies the attention sink once, after the merge, as
  ``out * sigmoid(lse - sink)``.
* ``merge_lse`` joins the cached-prefix pass with the chunk-local pass of an
  extend step.
* ``get_dcp_lens`` and the global-window rule ``keep local j iff
  j*dcp + rank > Q - sw`` select exactly the positions inside the window.

The collectives are faked: every rank's call runs in-process and the fake group
serves the other ranks' tensors, so the values crossing the "wire" are the ones
the function under test produced.
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.layers.attention.attention_pallas_rpa_dcp import (
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
    """Local lengths and the global sliding-window rule against brute force."""

    DCP_SIZES = (2, 4, 8)
    WINDOWS = (64, 128)
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

    def test_kept_local_indices_match_brute_force(self):
        for dcp in self.DCP_SIZES:
            for sw in self.WINDOWS:
                for length in self.LENGTHS:
                    query = length - 1  # decode: the new token's global position
                    for rank in range(dcp):
                        owned = self._owned_positions(length, dcp, rank)
                        kept_rule = [
                            j for j in range(len(owned)) if j * dcp + rank > query - sw
                        ]
                        kept_brute = [j for j, p in enumerate(owned) if p > query - sw]
                        self.assertEqual(
                            kept_rule,
                            kept_brute,
                            f"dcp={dcp} sw={sw} len={length} rank={rank}",
                        )

    def test_the_group_keeps_exactly_the_windowed_positions(self):
        for dcp in self.DCP_SIZES:
            for sw in self.WINDOWS:
                for length in self.LENGTHS:
                    query = length - 1
                    kept = set()
                    for rank in range(dcp):
                        owned = self._owned_positions(length, dcp, rank)
                        kept.update(
                            owned[j]
                            for j in range(len(owned))
                            if j * dcp + rank > query - sw
                        )
                    expected = {p for p in range(length) if p > query - sw}
                    self.assertEqual(kept, expected, f"dcp={dcp} sw={sw} len={length}")


if __name__ == "__main__":
    unittest.main()
