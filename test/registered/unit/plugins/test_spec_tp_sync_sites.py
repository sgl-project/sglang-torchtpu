"""The fork's TpuSpecWorker sync sites stay distinct from upstream's.

The fork appends SPEC_* members to upstream's ``SpecTpSyncSite``; an upstream site
that takes one of their numbers makes the fork member its alias.
"""

import unittest

from sglang.srt.speculative.spec_tp_sync import SpecTpSyncSite, parse_spec_tp_sync
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestSpecTpSyncSites(CustomTestCase):
    def test_no_site_is_an_alias(self):
        aliases = {
            name: member.name
            for name, member in SpecTpSyncSite.__members__.items()
            if member.name != name
        }
        self.assertEqual(aliases, {})

    def test_spec_accept_greedy_slug_selects_its_own_site(self):
        """The DP-attention gpt-oss P/D recipes set this value; it must parse and must
        not select the DSpark draft broadcast."""
        sites = parse_spec_tp_sync("rng,spec-accept-greedy")
        self.assertIn(SpecTpSyncSite.SPEC_ACCEPT_GREEDY, sites)
        self.assertNotIn(SpecTpSyncSite.DSPARK_DRAFT, sites)


if __name__ == "__main__":
    unittest.main()
