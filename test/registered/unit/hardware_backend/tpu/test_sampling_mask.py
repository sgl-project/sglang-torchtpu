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
    from sglang.srt.hardware_backend.tpu.sampling_mask import tpu_filter_probs

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


@unittest.skipUnless(torch.tpu.is_available(), "requires TPU")
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


@unittest.skipUnless(torch.tpu.is_available(), "requires TPU")
class TestGatheredBlockFilterEquivalence(CustomTestCase):
    """Validates the widened fast path: top-p/min-p evaluated inside the
    `tpu_top_k`-gathered descending [bs, k] block (need_top_k=False) keeps
    exactly the probability set that full-width filtering with top-k enabled
    keeps."""

    def _kept(self, filtered):
        return [np.sort(row[row > 0]) for row in filtered]

    def test_block_filter_matches_full_width(self):
        from sglang.srt.hardware_backend.tpu.top_k import tpu_top_k

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


if __name__ == "__main__":
    unittest.main()
