"""pr-gate checks for the TPU nightly dashboard contract (no TPU).

The contract (.github/tpu-ci/nightly-and-dashboard-redesign.md) is what keeps
the dashboard trustworthy, and every clause here is one that fails silently at
runtime -- a mislabelled metric is stored but never charted, a renamed id splits
a trend in two, a test not wired to a job simply never runs. These checks turn
each of those into a red PR instead:

  * one GitHub job <-> one test file, and every nightly-registered test is
    reachable from exactly one job (a red job must name its test);
  * nightly-accuracy / nightly-perf MODEL_CONFIGs carry the required identity
    fields (`id`, `display_name`, `model`), ids are unique, and perf measures
    through a recipe launch + bench_serving sweep only;
  * the id rename registry (dashboard_aliases.json) is well-formed: acyclic,
    targets live, and no live id is aliased away;
  * `kind` labels use only the contract values (accuracy/performance/info) --
    a hand-rolled kind is stored but charted nowhere.
"""

import ast
import importlib.util
import json
import re
import subprocess
import sys
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.tpu import model_config

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

REPO = Path(__file__).resolve().parents[3]
TPU_TESTS = REPO / "test" / "registered" / "tpu"
TPU_CI = REPO / ".github" / "tpu-ci"
WORKFLOWS = REPO / ".github" / "workflows"
ALIASES_FILE = TPU_CI / "dashboard_aliases.json"

# The dashboard's contract kinds (redesign doc B.5.3): anything else is stored
# but charted nowhere, which is exactly the silent failure this lint prevents.
CONTRACT_KINDS = {"accuracy", "performance", "info"}

def _discover(script: str, *args: str) -> list:
    """Run a .github/tpu-ci discovery script exactly as the workflows do."""
    proc = subprocess.run(
        [sys.executable, str(TPU_CI / script), *args],
        capture_output=True,
        text=True,
        cwd=str(REPO),
        check=True,
    )
    return json.loads(proc.stdout)


def _nightly_registered() -> dict:
    """{suite: {repo-relative file, ...}} for every enabled nightly TPU test."""
    spec = importlib.util.spec_from_file_location(
        "dc_ci_register", REPO / "python" / "sglang" / "test" / "ci" / "ci_register.py"
    )
    ci_register = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ci_register)

    files = [
        str(p)
        for p in TPU_TESTS.rglob("*.py")
        if "__pycache__" not in p.parts and p.name not in ("conftest.py", "__init__.py")
    ]
    registered: dict = {}
    for test in ci_register.collect_tests(files, sanity_check=False):
        if (
            test.backend != ci_register.HWBackend.TPU
            or not test.nightly
            or test.disabled is not None
        ):
            continue
        rel = str(Path(test.filename).resolve().relative_to(REPO))
        registered.setdefault(test.effective_suite, set()).add(rel)
    return registered


def _config_modules(subdir: str):
    """Load every MODEL_CONFIG module under test/registered/tpu/<subdir>."""
    for path in sorted((TPU_TESTS / subdir).glob("test_*.py")):
        spec = importlib.util.spec_from_file_location(f"dc_{path.stem}", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        yield path, mod


def _static_pd_files() -> list:
    """The `files:` values hand-wired into feature-validation-tpu.yml.

    Only literal values count: the multi-host matrix job passes an expression,
    and its files come from discover_multihost instead.
    """
    import yaml

    workflow = yaml.safe_load(
        (WORKFLOWS / "feature-validation-tpu.yml").read_text(encoding="utf-8")
    )
    files = []
    for job in workflow["jobs"].values():
        value = (job.get("with") or {}).get("files", "")
        if value and "${{" not in value:
            files.append(value)
    return files


class TestJobMapsToOneTest(CustomTestCase):
    """One GitHub job <-> one test file, rendered from the real discovery.

    A job carrying two files reports one red X for two possible culprits; a
    registered test reachable from no job runs never and reds nothing. Both
    regress silently, so the nightly's discovery is rendered here (no TPU, no
    cluster) and checked as a bijection.
    """

    @classmethod
    def setUpClass(cls):
        cls.registered = _nightly_registered()
        cls.single_host = _discover(
            "discover_tests.py", "--suite", "nightly-tpu-4-v7", "--nightly"
        )
        cls.multihost = _discover(
            "discover_multihost.py", "--suite", "nightly-tpu-multihost"
        )
        cls.ondemand = _discover(
            "discover_multihost.py", "--suite", "nightly-tpu-multihost-ondemand"
        )
        cls.static_pd = _static_pd_files()

    def test_multihost_groups_carry_exactly_one_file(self):
        # `files` is space-separated by contract; a group bundling two files
        # would make a red JobSet ambiguous about which test failed.
        for row in self.multihost + self.ondemand:
            self.assertEqual(
                len(row["files"].split()), 1, f"group {row['name']} bundles files"
            )

    def test_static_pd_jobs_carry_exactly_one_existing_file(self):
        self.assertTrue(self.static_pd, "no static P/D jobs found; did they move?")
        for value in self.static_pd:
            parts = value.split()
            self.assertEqual(len(parts), 1, f"P/D job bundles files: {value}")
            self.assertTrue((REPO / parts[0]).is_file(), f"missing file: {value}")

    def test_every_nightly_registration_reaches_exactly_one_job(self):
        # Render every job's file, then check the multiset: each registered
        # nightly test appears exactly once, and no job doubles another.
        job_files = (
            [row["file"] for row in self.single_host]
            + [row["files"] for row in self.multihost]
            + list(self.static_pd)
        )
        accuracy_perf_files = []
        for suite in self.registered:
            workflow = None
            if suite.startswith("nightly-tpu-perf-"):
                workflow = "perf-test-tpu.yml"
            elif suite.startswith("nightly-tpu-accuracy-"):
                workflow = "accuracy-test-tpu.yml"
            if workflow is None:
                continue
            # The suite must be wired into its stage workflow's matrix...
            text = (WORKFLOWS / workflow).read_text(encoding="utf-8")
            self.assertIn(suite, text, f"{suite} is not in {workflow}'s matrix")
            # ...and render to exactly one file (one file, one suite, one job).
            rows = _discover("discover_tests.py", "--suite", suite, "--nightly")
            self.assertEqual(
                len(rows), 1, f"{suite} renders {len(rows)} jobs, contract is 1"
            )
            accuracy_perf_files.append(rows[0]["file"])
        job_files += accuracy_perf_files

        duplicated = {f for f in job_files if job_files.count(f) > 1}
        self.assertFalse(duplicated, f"files reached by two jobs: {duplicated}")

        for suite, files in self.registered.items():
            if suite == "nightly-tpu-multihost-ondemand":
                # Parked out of the nightly on purpose; dispatched through
                # accuracy-test-tpu.yml's ondemand discovery instead.
                reachable = {row["files"] for row in self.ondemand}
            else:
                reachable = set(job_files)
            for file in files:
                self.assertIn(
                    file,
                    reachable,
                    f"{file} (suite {suite}) is nightly-registered but no "
                    "nightly job runs it",
                )


class TestNightlyConfigLabelLint(CustomTestCase):
    """The required identity fields on nightly-accuracy / nightly-perf configs.

    Without `id` a rename splits the series; without the kind/model labels a
    score is stored but never charted. Both are invisible at runtime, so the
    real configs are imported and checked here.
    """

    ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")

    @classmethod
    def setUpClass(cls):
        cls.accuracy = list(_config_modules("nightly-accuracy"))
        cls.perf = list(_config_modules("nightly-perf"))
        assert cls.accuracy and cls.perf, "config files moved?"

    def _all_configs(self):
        yield from ((p, m, "accuracy") for p, m in self.accuracy)
        yield from ((p, m, "perf") for p, m in self.perf)

    def test_required_fields_present(self):
        for path, mod, _kind in self._all_configs():
            config = mod.MODEL_CONFIG
            for field in ("id", "display_name", "model"):
                self.assertTrue(config.get(field), f"{path.name} misses {field!r}")
            self.assertRegex(
                config["id"],
                self.ID_RE,
                f"{path.name}: id must be a stable kebab slug (it is a join "
                "key, not a display string)",
            )

    def test_ids_are_unique(self):
        # Two tests claiming one id would merge unrelated trends into one plot.
        seen = {}
        for path, mod, _kind in self._all_configs():
            series_id = mod.MODEL_CONFIG["id"]
            self.assertNotIn(
                series_id, seen, f"{path.name} reuses id of {seen.get(series_id)}"
            )
            seen[series_id] = path.name

    def test_accuracy_gate_labels_carry_identity(self):
        for path, mod in self.accuracy:
            for params in model_config.build_gates(mod.MODEL_CONFIG):
                labels = params.metric_labels or {}
                self.assertEqual(labels.get("kind"), "accuracy", path.name)
                self.assertEqual(labels.get("id"), mod.MODEL_CONFIG["id"], path.name)
                self.assertEqual(
                    labels.get("display_name"),
                    mod.MODEL_CONFIG["display_name"],
                    path.name,
                )

    def test_perf_measures_through_recipe_and_bench_serving_only(self):
        # RecipeLauncher + bench_serving.sh is what puts the server/bench config
        # panels on the dashboard; a hand-booted server or hand-rolled bench
        # produces a chart nobody can reproduce from.
        for path, mod in self.perf:
            config = mod.MODEL_CONFIG
            launch = config.get("launch") or {}
            self.assertTrue(
                launch.get("recipe"),
                f"{path.name}: perf must launch through a recipe (RecipeLauncher)",
            )
            self.assertFalse(launch.get("external"), path.name)
            for gate in config["gates"]:
                self.assertTrue(
                    "sweep" in gate or gate.get("smoke"),
                    f"{path.name}: perf gates are sweeps or smoke legs only, "
                    f"got {gate.get('label') or gate.get('dataset')}",
                )
            # Smoke-precedes-sweep ordering and sweep shape are enforced by
            # build_gates itself; building the real config exercises that.
            self.assertTrue(model_config.build_gates(config))


class TestAliasRegistry(CustomTestCase):
    """dashboard_aliases.json: the one place an id rename is allowed to live."""

    @classmethod
    def setUpClass(cls):
        cls.aliases = json.loads(ALIASES_FILE.read_text(encoding="utf-8"))["aliases"]
        cls.live_ids = {
            mod.MODEL_CONFIG["id"]
            for subdir in ("nightly-accuracy", "nightly-perf")
            for _path, mod in _config_modules(subdir)
        }

    def test_wellformed(self):
        for old, new in self.aliases.items():
            self.assertIsInstance(old, str)
            self.assertIsInstance(new, str)
            self.assertNotEqual(old, new, f"self-alias: {old}")

    def test_no_live_id_is_aliased_away(self):
        # An alias source still claimed by a test would fork its trend: new
        # records land under the live id while renders chase the alias.
        claimed = self.live_ids & set(self.aliases)
        self.assertFalse(claimed, f"live ids used as alias sources: {claimed}")

    def test_chains_resolve_to_a_live_id(self):
        # a->b->c is legal; a cycle or a dangling target is a map that renames
        # a series into nothing.
        for start in self.aliases:
            seen, current = {start}, start
            while current in self.aliases:
                current = self.aliases[current]
                self.assertNotIn(current, seen, f"alias cycle through {start}")
                seen.add(current)
            self.assertIn(
                current,
                self.live_ids,
                f"alias {start} resolves to {current}, which no test claims",
            )


class TestReservedKindValues(CustomTestCase):
    """Every literal `kind` handed to dump_metric uses a contract value.

    The dashboard routes on kind unconditionally, so a hand-rolled value
    ("bench_detail", "kv_offload", ...) is stored but charted nowhere -- the
    metric looks dumped and never appears. AST-scanned so the check needs no
    TPU and no imports of the scanned files.
    """

    SCAN_ROOTS = (
        TPU_TESTS,
        REPO / "python" / "sglang" / "test" / "tpu",
    )

    @staticmethod
    def _literal_kinds(tree):
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and getattr(node.func, "id", getattr(node.func, "attr", ""))
                == "dump_metric"
            ):
                continue
            labels = next(
                (kw.value for kw in node.keywords if kw.arg == "labels"),
                node.args[2] if len(node.args) > 2 else None,
            )
            if not isinstance(labels, ast.Dict):
                continue
            for key, value in zip(labels.keys, labels.values):
                if (
                    isinstance(key, ast.Constant)
                    and key.value == "kind"
                    and isinstance(value, ast.Constant)
                ):
                    yield node.lineno, value.value

    def test_only_contract_kinds_are_dumped(self):
        offenders = []
        for root in self.SCAN_ROOTS:
            for path in sorted(root.rglob("*.py")):
                if "__pycache__" in path.parts:
                    continue
                tree = ast.parse(path.read_text(encoding="utf-8"))
                for lineno, kind in self._literal_kinds(tree):
                    if kind not in CONTRACT_KINDS:
                        offenders.append(f"{path.relative_to(REPO)}:{lineno}: {kind!r}")
        self.assertFalse(
            offenders,
            "off-contract kind values (allowed: accuracy/performance/info):\n"
            + "\n".join(offenders),
        )


if __name__ == "__main__":
    unittest.main()
