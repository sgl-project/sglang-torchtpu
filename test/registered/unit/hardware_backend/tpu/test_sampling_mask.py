"""Numeric coverage for the sort-free top-k/top-p/min-p filter op on TPU.

Runs the real `tpu_filter_probs` custom op (cpu_export_jax_op export + device
execution) against a per-row sort-based reference, pinning two derived
properties:

1. The single 30-step bit-descent search — one OR'd predicate for top-k and
   top-p, min-p folded in with a `maximum` — matches the reference, including
   the documented tie-inclusive semantics and the unreachable-target edge
   cases the removed special cases used to guard (`top_p == 1.0`,
   `top_k >= vocab`, zero-padded `top_ks` rows).
2. Filtering inside a `tpu_top_k`-gathered descending [bs, k] block with
   `need_top_k=False` keeps the same probability set as full-width filtering
   with top-k enabled — the equivalence the sampler's widened native-TopK
   fast path relies on.
"""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=120, suite="stage-a-test-tpu-unit")

import unittest

import numpy as np
import torch

from sglang.test.test_utils import CustomTestCase


def _reference_filter(
    probs, top_ks, top_ps, min_ps, need_top_k, need_top_p, need_min_p
):
    """Per-row sort-based cutoff reference with the op's documented semantics:
    every filter is a lower bound on one cutoff, and the boundary value is
    kept together with its ties."""
    out = np.zeros_like(probs)
    for i, row in enumerate(probs):
        cutoff = np.float32(0.0)
        descending = np.sort(row)[::-1]
        if need_top_k:
            k = int(np.clip(top_ks[i], 1, row.size))
            cutoff = max(cutoff, descending[k - 1])
        if need_top_p:
            reached = np.nonzero(np.cumsum(descending) >= top_ps[i])[0]
            if reached.size:  # unreachable mass targets keep the whole row
                cutoff = max(cutoff, descending[reached[0]])
        if need_min_p:
            cutoff = max(cutoff, descending[0] * min_ps[i])
        out[i] = np.where(row >= cutoff, row, 0.0)
    return out


def _random_rows(rng, batch_size, vocab_size):
    logits = rng.standard_normal((batch_size, vocab_size)).astype(np.float32) * 4.0
    exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return (exp / exp.sum(axis=-1, keepdims=True)).astype(np.float32)


def _filter_on_tpu(probs, top_ks, top_ps, min_ps, need_top_k, need_top_p, need_min_p):
    from sglang_tpu.sampling_mask import tpu_filter_probs

    filtered = tpu_filter_probs(
        torch.from_numpy(probs).to("tpu"),
        torch.from_numpy(top_ks).to("tpu"),
        torch.from_numpy(top_ps).to("tpu"),
        torch.from_numpy(min_ps).to("tpu"),
        need_top_k=bool(need_top_k),
        need_top_p=bool(need_top_p),
        need_min_p=bool(need_min_p),
    )
    return filtered.cpu().numpy()


@unittest.skipUnless(torch.accelerator.is_available(), "requires TPU")
class TestFilterProbsOpMatchesSortReference(CustomTestCase):
    def _assert_matches_reference(
        self, probs, top_ks, top_ps, min_ps, need_top_k, need_top_p, need_min_p
    ):
        actual = _filter_on_tpu(
            probs, top_ks, top_ps, min_ps, need_top_k, need_top_p, need_min_p
        )
        expected = _reference_filter(
            probs, top_ks, top_ps, min_ps, need_top_k, need_top_p, need_min_p
        )
        np.testing.assert_array_equal(actual, expected)

    def test_rows_off_the_power_of_two_bucket_match_reference(self):
        """Row counts between buckets are padded with ghost rows for the op and
        sliced back, so the caller sees its own shape and unchanged values."""
        rng = np.random.default_rng(23)
        for batch_size in (1, 3, 5, 13):
            probs = _random_rows(rng, batch_size=batch_size, vocab_size=256)
            top_ks = rng.integers(1, 200, size=batch_size).astype(np.int32)
            top_ps = rng.uniform(0.1, 0.99, size=batch_size).astype(np.float32)
            min_ps = rng.uniform(0.0, 0.3, size=batch_size).astype(np.float32)
            with self.subTest(batch_size=batch_size):
                actual = _filter_on_tpu(probs, top_ks, top_ps, min_ps, 1, 1, 1)
                self.assertEqual(actual.shape, probs.shape)
                self._assert_matches_reference(probs, top_ks, top_ps, min_ps, 1, 1, 1)

    def test_randomized_mixed_filters_match_sort_reference(self):
        rng = np.random.default_rng(17)
        for trial in range(5):
            probs = _random_rows(rng, batch_size=8, vocab_size=256)
            top_ks = rng.integers(1, 200, size=8).astype(np.int32)
            top_ps = rng.uniform(0.1, 0.99, size=8).astype(np.float32)
            min_ps = rng.uniform(0.0, 0.3, size=8).astype(np.float32)
            for need_top_k, need_top_p, need_min_p in (
                (1, 0, 0),
                (0, 1, 0),
                (0, 0, 1),
                (1, 1, 0),
                (1, 1, 1),
            ):
                with self.subTest(
                    trial=trial, flags=(need_top_k, need_top_p, need_min_p)
                ):
                    self._assert_matches_reference(
                        probs,
                        top_ks,
                        top_ps,
                        min_ps,
                        need_top_k,
                        need_top_p,
                        need_min_p,
                    )

    def test_boundary_ties_are_kept_inclusively(self):
        """The cutoff keeps values tied with the k-th largest — the op's
        documented departure from arbitrary sort-order tie-breaking."""
        probs = np.array([[0.3, 0.2, 0.2, 0.2, 0.1, 0.0]], dtype=np.float32)
        actual = _filter_on_tpu(
            probs,
            np.array([2], dtype=np.int32),
            np.ones(1, dtype=np.float32),
            np.zeros(1, dtype=np.float32),
            1,
            0,
            0,
        )
        np.testing.assert_array_equal(
            actual, np.array([[0.3, 0.2, 0.2, 0.2, 0.0, 0.0]], dtype=np.float32)
        )

    def test_unreachable_targets_keep_the_whole_row(self):
        """`top_p == 1.0` on an exactly-summing row, `top_k` at and above the
        vocabulary size, and TOP_K_ALL all resolve to keep-everything without
        the removed special cases."""
        # Dyadic values: the float32 row mass is exactly 1.0.
        row = np.array([0.5, 0.25, 0.125, 0.0625, 0.03125, 0.03125], np.float32)
        probs = np.tile(row, (3, 1))
        top_ks = np.array([6, 100, 1 << 30], dtype=np.int32)
        top_ps = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        actual = _filter_on_tpu(probs, top_ks, top_ps, np.zeros(3, np.float32), 1, 1, 0)
        np.testing.assert_array_equal(actual, probs)

    def test_zero_padded_top_k_row_keeps_the_max(self):
        """A zero-padded ghost row clips to k=1 and keeps the row max, so
        padding can never produce an all-zero sampling row."""
        probs = np.array([[0.1, 0.6, 0.3]], dtype=np.float32)
        actual = _filter_on_tpu(
            probs,
            np.array([0], dtype=np.int32),
            np.ones(1, dtype=np.float32),
            np.zeros(1, dtype=np.float32),
            1,
            0,
            0,
        )
        np.testing.assert_array_equal(
            actual, np.array([[0.0, 0.6, 0.0]], dtype=np.float32)
        )


@unittest.skipUnless(torch.accelerator.is_available(), "requires TPU")
class TestGatheredBlockFilterEquivalence(CustomTestCase):
    """Validates the widened fast path: top-p/min-p evaluated inside the
    `tpu_top_k`-gathered descending [bs, k] block (need_top_k=False) keeps
    exactly the probability set that full-width filtering with top-k enabled
    keeps."""

    def _kept(self, filtered):
        return [np.sort(row[row > 0]) for row in filtered]

    def test_block_filter_matches_full_width(self):
        from sglang_tpu.top_k import tpu_top_k

        rng = np.random.default_rng(23)
        k = 50
        for label, top_ps, min_ps in (
            # Nucleus narrower than k: the block search finds the same cutoff.
            ("narrow", np.full(8, 0.6, np.float32), np.zeros(8, np.float32)),
            # Nucleus wider than k: the block mass never reaches p, the
            # search returns 0.0, and the whole block survives (= AND top-k).
            ("wide", np.full(8, 0.9999, np.float32), np.zeros(8, np.float32)),
            # min-p measures against the row max = the block's first column.
            ("min_p", np.ones(8, np.float32), np.full(8, 0.05, np.float32)),
            (
                "mixed",
                rng.uniform(0.3, 0.99, 8).astype(np.float32),
                rng.uniform(0.0, 0.2, 8).astype(np.float32),
            ),
        ):
            probs = _random_rows(rng, batch_size=8, vocab_size=512)
            top_ks = np.full(8, k, dtype=np.int32)
            need_top_p = int(top_ps.min() < 1.0)
            need_min_p = int(min_ps.max() > 0.0)
            with self.subTest(case=label):
                full = _filter_on_tpu(
                    probs, top_ks, top_ps, min_ps, 1, need_top_p, need_min_p
                )
                block, _ = tpu_top_k(torch.from_numpy(probs).to("tpu"), k)
                filtered_block = _filter_on_tpu(
                    block.cpu().numpy(),
                    top_ks,
                    top_ps,
                    min_ps,
                    0,
                    need_top_p,
                    need_min_p,
                )
                for row_full, row_block in zip(
                    self._kept(full), self._kept(filtered_block)
                ):
                    np.testing.assert_array_equal(row_full, row_block)

    def test_top_k_rows_off_the_power_of_two_bucket_match_torch(self):
        """Row counts between buckets are padded for the exported TopK op and
        sliced back, so the caller sees its own shape, values and indices."""
        from sglang_tpu.top_k import tpu_top_k

        rng = np.random.default_rng(29)
        for batch_size in (1, 3, 5, 13):
            probs = _random_rows(rng, batch_size=batch_size, vocab_size=512)
            with self.subTest(batch_size=batch_size):
                values, ids = tpu_top_k(torch.from_numpy(probs).to("tpu"), 7)
                expected = torch.topk(torch.from_numpy(probs), 7, dim=-1)
                self.assertEqual(tuple(values.shape), (batch_size, 7))
                torch.testing.assert_close(values.cpu(), expected.values)
                self.assertEqual(ids.cpu().tolist(), expected.indices.tolist())


@unittest.skipUnless(torch.accelerator.is_available(), "requires TPU")
class TestEagleVerifyRenorm(CustomTestCase):
    """EAGLE verify renormalizes top-k, then top-p on the top-k-renormalized
    distribution. Each stage must keep the sort reference's support."""

    @staticmethod
    def _renormalized(kept):
        return kept / kept.sum(axis=-1, keepdims=True)

    def test_top_k_then_top_p_stages_match_sort_reference(self):
        from sglang_tpu.eagle_utils_tpu import (
            top_k_renorm_prob_tpu,
            top_p_renorm_prob_tpu,
        )

        rng = np.random.default_rng(29)
        rows = 64
        probs = _random_rows(rng, batch_size=rows, vocab_size=4096)
        top_ks = rng.choice([1, 2, 50, 64, 1 << 30], rows).astype(np.int32)
        top_ps = rng.choice([0.5, 0.9, 1.0], rows).astype(np.float32)
        unused_ks = np.ones(rows, np.int32)
        unused_ps = np.ones(rows, np.float32)
        min_ps = np.zeros(rows, np.float32)

        after_top_k = top_k_renorm_prob_tpu(
            torch.from_numpy(probs).to("tpu"), torch.from_numpy(top_ks).to("tpu")
        )
        expected = self._renormalized(
            _reference_filter(probs, top_ks, unused_ps, min_ps, True, False, False)
        )
        actual = after_top_k.cpu().numpy()
        np.testing.assert_array_equal(actual > 0, expected > 0)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-7)

        # Feed top-p the device's own top-k output so both sides see the same input.
        after_top_p = top_p_renorm_prob_tpu(
            after_top_k, torch.from_numpy(top_ps).to("tpu")
        )
        expected = self._renormalized(
            _reference_filter(actual, unused_ks, top_ps, min_ps, False, True, False)
        )
        actual = after_top_p.cpu().numpy()
        partial = top_ps < 1.0
        np.testing.assert_array_equal(actual[partial] > 0, expected[partial] > 0)
        np.testing.assert_allclose(
            actual[partial], expected[partial], rtol=1e-5, atol=1e-7
        )
        # top_p == 1.0 on a renormalized row: whether a float32 sum reaches 1.0
        # depends on reduction order, so either side may drop a tail of
        # epsilon mass. Bound the difference instead of comparing supports.
        self.assertLess(
            np.abs(actual[~partial] - expected[~partial]).sum(-1).max(), 1e-5
        )

    def test_boundary_ties_are_kept_inclusively(self):
        from sglang_tpu.eagle_utils_tpu import (
            top_k_renorm_prob_tpu,
        )

        probs = torch.tensor([[0.4, 0.2, 0.2, 0.2]])
        actual = top_k_renorm_prob_tpu(probs.to("tpu"), torch.tensor([2]).to("tpu"))
        torch.testing.assert_close(actual.cpu(), probs)


if __name__ == "__main__":
    unittest.main()
