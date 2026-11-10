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


def info(topology, hosts, nodes, ready, max_nodes=None):
    return {
        "topology": topology,
        "hosts": hosts,
        "nodes": nodes,
        "ready": ready,
        "max_nodes": max_nodes,
    }


class TestUsableSlices(unittest.TestCase):
    def test_fixed_single_host_pool_counts_ready_nodes(self):
        self.assertEqual(
            reconcile_quota.usable_slices(pool="p", info=info("2x2x1", 1, 5, 4)), 4
        )

    def test_autoscaled_single_host_pool_counts_its_maximum(self):
        # Quota admits before the node exists; the pending pod triggers the scale-up.
        self.assertEqual(
            reconcile_quota.usable_slices(
                pool="p", info=info("2x2x1", 1, 3, 3, max_nodes=5)
            ),
            5,
        )

    def test_autoscaled_pool_grown_past_its_maximum_by_hand_counts_live_nodes(self):
        self.assertEqual(
            reconcile_quota.usable_slices(
                pool="p", info=info("2x2x1", 1, 6, 6, max_nodes=5)
            ),
            6,
        )

    def test_multi_host_pool_holding_two_slices_gets_no_quota(self):
        # Pods could straddle the two slices and hang at ICI rendezvous.
        self.assertEqual(
            reconcile_quota.usable_slices(pool="p", info=info("2x2x2", 2, 4, 4)), 0
        )

    def test_autoscaled_multi_host_pool_with_a_two_slice_maximum_gets_no_quota(self):
        self.assertEqual(
            reconcile_quota.usable_slices(
                pool="p", info=info("2x2x2", 2, 0, 0, max_nodes=4)
            ),
            0,
        )

    def test_autoscaled_multi_host_pool_scaled_to_zero_still_counts_one_slice(self):
        self.assertEqual(
            reconcile_quota.usable_slices(
                pool="p", info=info("2x2x2", 2, 0, 0, max_nodes=2)
            ),
            1,
        )

    def test_multi_host_slice_counts_only_when_every_host_is_ready(self):
        self.assertEqual(
            reconcile_quota.usable_slices(pool="p", info=info("2x2x2", 2, 2, 1)), 0
        )


def node(pool, topology, ready=True, chips="4"):
    return {
        "metadata": {
            "labels": {
                "ci-tpu-pool": pool,
                "cloud.google.com/gke-tpu-topology": topology,
                "cloud.google.com/gke-accelerator-count": chips,
            }
        },
        "status": {
            "conditions": [{"type": "Ready", "status": "True" if ready else "False"}]
        },
    }


def node_pool(pool, topology=None, autoscaling=None):
    return {
        "name": f"ci-{pool}",
        "config": {"labels": {"ci-tpu-pool": pool}} if pool else {},
        "placementPolicy": {"tpuTopology": topology} if topology else None,
        "autoscaling": autoscaling,
    }


class TestCiPools(unittest.TestCase):
    def test_autoscaled_pool_scaled_to_zero_is_listed_from_gke(self):
        pools = reconcile_quota.ci_pools(
            nodes=[],
            node_pools=[
                node_pool("pr-gate", autoscaling={"enabled": True, "maxNodeCount": 5}),
                node_pool("nightly", topology="2x2x2"),
                node_pool(None),
            ],
        )
        self.assertEqual(
            pools["pr-gate"],
            {
                "topology": "2x2x1",
                "hosts": 1,
                "nodes": 0,
                "ready": 0,
                "chips_per_host": 4,
                "max_nodes": 5,
            },
        )
        self.assertEqual(pools["nightly"]["hosts"], 2)
        self.assertIsNone(pools["nightly"]["max_nodes"])
        self.assertEqual(set(pools), {"pr-gate", "nightly"})

    def test_nodes_add_live_counts_to_a_listed_pool(self):
        pools = reconcile_quota.ci_pools(
            nodes=[node("pr-gate", "2x2x1"), node("pr-gate", "2x2x1", ready=False)],
            node_pools=[
                node_pool("pr-gate", autoscaling={"enabled": True, "maxNodeCount": 5})
            ],
        )
        self.assertEqual((pools["pr-gate"]["nodes"], pools["pr-gate"]["ready"]), (2, 1))
        self.assertEqual(pools["pr-gate"]["max_nodes"], 5)

    def test_disabled_autoscaling_is_a_fixed_pool(self):
        pools = reconcile_quota.ci_pools(
            nodes=[],
            node_pools=[node_pool("pr-gate", autoscaling={"enabled": False})],
        )
        self.assertIsNone(pools["pr-gate"]["max_nodes"])

    def test_pool_known_only_from_nodes_is_fixed(self):
        pools = reconcile_quota.ci_pools(
            nodes=[node("by-hand", "2x2x2"), node("by-hand", "2x2x2")], node_pools=[]
        )
        self.assertEqual(
            pools["by-hand"],
            {
                "topology": "2x2x2",
                "hosts": 2,
                "nodes": 2,
                "ready": 2,
                "chips_per_host": 4,
                "max_nodes": None,
            },
        )


if __name__ == "__main__":
    unittest.main()
