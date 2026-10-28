"""How reconcile_quota.py splits a shape's live slices between lanes."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import reconcile_quota  # noqa: E402

LANES = ["pr-gate", "nightly"]


def split(slices, hosts, min_hosts):
    return reconcile_quota.split_quota(
        slices=slices,
        hosts=hosts,
        chips_per_slice=4 * hosts,
        lanes=LANES,
        priority="nightly",
        min_hosts=min_hosts,
    )


class TestSplitQuota(unittest.TestCase):
    def test_reserve_spans_pools_and_quotas_sum_to_live_chips(self):
        out = split(
            slices={"pr-gate": 5, "nightly-1h": 3}, hosts=1, min_hosts={"pr-gate": 5}
        )
        self.assertEqual(sum(c for c, _ in out["pr-gate"].values()), 20)
        self.assertEqual(sum(lend for _, lend in out["pr-gate"].values()), 0)
        self.assertEqual(
            out["nightly"]["nightly-1h"][0] + out["nightly"]["pr-gate"][0], 12
        )
        for pool, live in (("pr-gate", 5), ("nightly-1h", 3)):
            self.assertEqual(
                out["pr-gate"][pool][0] + out["nightly"][pool][0], 4 * live
            )
            self.assertEqual(out["nightly"][pool][1], out["nightly"][pool][0])

    def test_reserve_fills_the_largest_pool_first(self):
        out = split(slices={"a": 3, "b": 5}, hosts=1, min_hosts={"pr-gate": 5})
        self.assertEqual(out["pr-gate"], {"b": (20, 0), "a": (0, 0)})
        self.assertEqual(out["nightly"], {"b": (0, 0), "a": (12, 12)})

    def test_multi_host_reserve_takes_whole_slices_across_pools(self):
        # Two 2x2x2 slices, one pool each: one slice reserved, not half of each.
        out = split(slices={"a": 1, "b": 1}, hosts=2, min_hosts={"pr-gate": 2})
        self.assertEqual(sorted(c for c, _ in out["pr-gate"].values()), [0, 8])
        self.assertEqual(sorted(c for c, _ in out["nightly"].values()), [0, 8])

    def test_reserve_larger_than_cluster_takes_everything(self):
        out = split(slices={"p": 2}, hosts=1, min_hosts={"pr-gate": 5})
        self.assertEqual(out["pr-gate"]["p"], (8, 0))
        self.assertEqual(out["nightly"]["p"], (0, 0))

    def test_reserve_must_be_whole_slices(self):
        with self.assertRaises(SystemExit):
            split(slices={"p": 2}, hosts=2, min_hosts={"pr-gate": 1})


class TestUsableSlices(unittest.TestCase):
    def test_multi_host_pool_holding_two_slices_gets_no_quota(self):
        # Pods could straddle the two slices and hang at ICI rendezvous.
        info = {"topology": "2x2x2", "hosts": 2, "nodes": 4, "ready": 4}
        self.assertEqual(reconcile_quota.usable_slices(pool="p", info=info), 0)

    def test_multi_host_slice_counts_only_when_every_host_is_ready(self):
        info = {"topology": "2x2x2", "hosts": 2, "nodes": 2, "ready": 1}
        self.assertEqual(reconcile_quota.usable_slices(pool="p", info=info), 0)


if __name__ == "__main__":
    unittest.main()
