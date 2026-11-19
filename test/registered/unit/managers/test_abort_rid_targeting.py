"""An abort must cover only the rid it names and that rid's own batch items.

Sequential caller rids (`<run>_0` ... `<run>_31`) are common; aborting `<run>_3`
must not abort `<run>_30` or `<run>_31`, which are separate live requests.
"""

import re
import unittest
from pathlib import Path

from sglang.srt.managers.io_struct import AbortReq
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

RUN = "run-0cdc7cd9b64049e39980c2673bcf5453"
RIDS = [f"{RUN}_{i}" for i in range(32)]


class TestAbortRidTargeting(unittest.TestCase):
    def test_named_rid_and_its_batch_items_only(self):
        self.assertEqual(
            [r for r in RIDS if AbortReq(rid=f"{RUN}_3").matches(r)], [f"{RUN}_3"]
        )
        self.assertFalse(AbortReq(rid="job-1").matches("job-10"))
        # _normalize_rid expands a batch's str rid to <rid>_<i>, one level deep.
        self.assertEqual([r for r in RIDS if AbortReq(rid=RUN).matches(r)], RIDS)
        for other in (f"{RUN}_a", f"{RUN}_", f"{RUN}_3_0", f"{RUN}x"):
            self.assertFalse(AbortReq(rid=RUN).matches(other), other)

    def test_empty_rid_matches_nothing_unless_abort_all(self):
        self.assertFalse(AbortReq().matches("_3"))
        self.assertTrue(AbortReq(abort_all=True).matches("anything"))

    def test_no_abort_path_matches_rids_by_prefix(self):
        root = Path(__file__).resolve().parents[4]
        trees = [root / "python" / "sglang", root / "sglang_tpu"]
        for tree in trees:
            self.assertTrue(tree.is_dir(), tree)
        pattern = re.compile(r"\.startswith\([^)]*rid[^)]*\)")
        offenders = [
            f"{path.relative_to(root)}:{lineno}"
            for tree in trees
            for path in tree.rglob("*.py")
            for lineno, line in enumerate(path.read_text().splitlines(), 1)
            if pattern.search(line)
        ]
        self.assertEqual(offenders, [], "match aborts with AbortReq.matches")


if __name__ == "__main__":
    unittest.main(verbosity=2)
