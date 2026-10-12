import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[4]
CI_REGISTER_PATH = REPO_ROOT / "python" / "sglang" / "test" / "ci" / "ci_register.py"
DERIVE_PATH = REPO_ROOT / ".github" / "tpu-ci" / "derive_accuracy_floors.py"


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


register_cpu_ci = _load_module("ci_register", CI_REGISTER_PATH).register_cpu_ci
register_cpu_ci(est_time=1, suite="base-a-test-cpu")

derive_floors = _load_module("derive_accuracy_floors", DERIVE_PATH)


def _row(ts, value, run_id, sid="accuracy-m", label="aime25"):
    return {
        "timestamp": ts,
        "value": value,
        "run_id": run_id,
        "metric_name": label,
        "labels": {"kind": "accuracy", "id": sid},
    }


class TestDeriveAccuracyFloors(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def _series(self, rows):
        path = self.tmp / "metrics.json"
        path.write_text(json.dumps(rows))
        return derive_floors._series(path)

    def test_a_retried_run_and_a_second_dispatch_count_as_one_night(self):
        series = self._series(
            [
                _row("2026-10-01T03:00:00", 0.80, "1"),
                _row("2026-10-01T05:00:00", 0.90, "1"),  # retry of run 1
                _row("2026-10-02T03:00:00", 0.70, "2"),
                _row("2026-10-02T09:00:00", 0.85, "3"),  # [publish] dispatch
            ]
        )
        nights = series[("accuracy-m", "aime25")]
        self.assertEqual([(v, run) for _, v, run in nights], [(0.90, "1"), (0.85, "3")])

    def test_min_floor_only_raises_the_derived_floor(self):
        nights = [
            (f"2026-10-0{i}T03:00:00", v, str(i)) for i, v in enumerate([0.8, 0.9], 1)
        ]
        loose = derive_floors.derive({"n_items": 30, "min_floor": 0.675}, nights, 5)
        self.assertEqual(loose["floor"], 0.675)
        tight = derive_floors.derive({"n_items": 30, "min_floor": 0.1}, nights, 5)
        self.assertGreater(tight["floor"], 0.1)

    def test_one_night_still_gates_but_wider_than_five(self):
        def floor(values):
            nights = [
                (f"2026-10-0{i}T03:00:00", v, str(i)) for i, v in enumerate(values, 1)
            ]
            return derive_floors.derive({"n_items": 865}, nights, 5)

        one, five = floor([0.75]), floor([0.75] * 5)
        self.assertTrue(one["provisional"])
        # sigma_binomial ~= 0.0147: 3 * sqrt(2) * sigma vs 3 * sqrt(1.2) * sigma.
        self.assertAlmostEqual(one["floor"], 0.687, places=3)
        self.assertAlmostEqual(five["floor"], 0.701, places=3)

    def test_new_nights_refresh_provenance_even_if_the_floor_holds(self):
        floors = self.tmp / "floors.json"
        entry = {
            "n_items": 30,
            "hardware": "h",
            "min_floor": 0.675,
            "floor": 0.675,
            "runs": ["old"],
            "set_on": "2026-01-01",
        }
        floors.write_text(json.dumps({"floors": {"accuracy-m": {"aime25": entry}}}))
        metrics = self.tmp / "metrics.json"
        metrics.write_text(json.dumps([_row("2026-10-01T03:00:00", 0.8, "new")]))
        argv = ["derive", "--metrics", str(metrics), "--write"]
        with (
            patch.object(derive_floors, "FLOORS", floors),
            patch.object(sys, "argv", argv),
        ):
            derive_floors.main()
        got = json.loads(floors.read_text())["floors"]["accuracy-m"]["aime25"]
        self.assertEqual(got["floor"], 0.675)
        self.assertEqual(got["runs"], ["new"])
        self.assertNotEqual(got["set_on"], "2026-01-01")


if __name__ == "__main__":
    unittest.main()
