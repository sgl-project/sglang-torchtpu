"""Unit test for the MMMU-Pro shard collector's JSONL reader (no TPU or server).

sgl-eval streams output-rs0.jsonl with ensure_ascii=False, so a model's
generation can carry raw U+2028/U+2029/U+0085. Kimi-K2.6 shard 1 scored above
its floor on 2026-10-23 and still failed: str.splitlines() cut a record at one
of those characters ("Unterminated string starting at: line 1 column 16").
"""

import json
import tempfile
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.tpu.evals.mmmu_full import read_jsonl_ids

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestReadJsonlIds(CustomTestCase):
    def test_unicode_line_separators_stay_inside_a_record(self):
        rows = [
            {"generation": "step one\u2028step two", "id": "a"},
            {"generation": "x\x85y\u2029z", "id": "b"},
            {"generation": "plain", "id": "c"},
        ]
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "output-rs0.jsonl"
            path.write_text(
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                encoding="utf-8",
            )
            self.assertEqual(read_jsonl_ids(path), ["a", "b", "c"])


if __name__ == "__main__":
    unittest.main()
