"""pr-gate checks for the TPU nightly dashboard contract (no TPU).

The contract (.github/tpu-ci/README.md, § Dashboard) is what keeps
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
from sglang.test.tpu.evals import model_config

register_cpu_ci(est_time=70, suite="base-a-test-cpu")

REPO = Path(__file__).resolve().parents[3]
TPU_TESTS = REPO / "test" / "registered" / "tpu"
# The taxonomy ratchet (scripts/lint/check_registered_tests.py) only admits NEW
# registered tests under test/registered/<kind>/<subsystem>/, so a new TPU
# accuracy or perf file lands outside the grandfathered tree above. Both are in
# scope here: watching only one of them would let the contract lapse silently
# for every test written from now on, which is the failure mode this file exists
# to prevent.
TAXONOMY_TPU_TESTS = {
    "nightly-accuracy": REPO / "test" / "registered" / "accuracy" / "tpu",
    "nightly-perf": REPO / "test" / "registered" / "perf" / "tpu",
}
TPU_CI = REPO / ".github" / "tpu-ci"
WORKFLOWS = REPO / ".github" / "workflows"
ALIASES_FILE = TPU_CI / "dashboard_aliases.json"

# The dashboard's contract kinds (redesign doc B.5.3): anything else is stored
# but charted nowhere, which is exactly the silent failure this lint prevents.
CONTRACT_KINDS = {"accuracy", "performance", "info"}


_DISCOVER_MANY = """
import json, sys
sys.path.insert(0, sys.argv[1])
import discover_tests
root, suites = sys.argv[2], json.loads(sys.argv[3])
print(json.dumps({s: discover_tests.discover(s, True, root) for s in suites}))
"""


def _discover_suites(suites: tuple) -> dict:
    """{suite: rendered rows} for many suites, in one discovery process.

    Same code path as ``discover_tests.py --suite <s> --nightly``, asked once
    per suite inside a single child: collecting imports every registered test,
    which the script caches per root, so a call per suite would re-walk the
    whole tree each time.
    """
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            _DISCOVER_MANY,
            str(TPU_CI),
            str(REPO / "test" / "registered"),
            json.dumps(list(suites)),
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO),
        check=True,
    )
    return json.loads(proc.stdout)


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
        for root in (TPU_TESTS, *TAXONOMY_TPU_TESTS.values())
        for p in root.rglob("*.py")
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
    """Load every MODEL_CONFIG module of a category, old tree and new."""
    roots = (TPU_TESTS / subdir, TAXONOMY_TPU_TESTS[subdir])
    for path in sorted(p for root in roots for p in root.glob("test_*.py")):
        spec = importlib.util.spec_from_file_location(f"dc_{path.stem}", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        yield path, mod


def _external_group_files() -> list:
    """Files reached by the external-cluster nightly, one entry per file.

    The external pipeline (`external-nightly-*` suites) is a job source of its
    own: no Kueue, so render_external_jobset.py round-robins each suite's
    launch groups across the slices in external-topology.yaml. Unlike the
    in-project multi-host groups, an external group deliberately bundles the
    files that share a server bring-up -- rank 0 runs them back-to-back and
    the JobSet emits one TIMINGS record PER FILE, so a red run still names the
    file that failed and the one-job-one-test attribution survives the
    bundling.
    """
    return [
        file
        for row in _discover(
            "external/render_external_jobset.py", "--list-groups", "--nightly"
        )
        for group in row["groups"]
        for file in group["files"].split()
    ]


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
        cls.multihost_accuracy = _discover(
            "discover_multihost.py", "--suite", "nightly-tpu-multihost-accuracy"
        )
        cls.ondemand = _discover(
            "discover_multihost.py", "--suite", "nightly-tpu-multihost-ondemand"
        )
        cls.static_pd = _static_pd_files()
        cls.external = _external_group_files()

    def test_multihost_groups_carry_exactly_one_file(self):
        # `files` is space-separated by contract; a group bundling two files
        # would make a red JobSet ambiguous about which test failed.
        for row in self.multihost + self.multihost_accuracy + self.ondemand:
            self.assertEqual(
                len(row["files"].split()), 1, f"group {row['name']} bundles files"
            )

    @staticmethod
    def _run_step(step, env):
        """Run one workflow step's bash; return what it wrote to GITHUB_OUTPUT."""
        import os
        import tempfile

        with tempfile.NamedTemporaryFile() as output:
            done = subprocess.run(
                ["bash", "-c", step["run"]],
                cwd=REPO,
                env={**os.environ, **env, "GITHUB_OUTPUT": output.name},
                capture_output=True,
                text=True,
            )
            written = Path(output.name).read_text()
        return done.returncode, dict(
            line.split("=", 1) for line in written.splitlines() if line
        )

    def test_each_schedule_slot_selects_its_tests(self):
        # A cron edited without its `plan` case would run nothing, or everything.
        import yaml

        workflow = yaml.safe_load((WORKFLOWS / "accuracy-test-tpu.yml").read_text())
        plan = workflow["jobs"]["plan"]["steps"][0]
        crons = [entry["cron"] for entry in workflow[True]["schedule"]]
        selected = {}
        for cron in crons:
            rc, out = self._run_step(plan, {"MODEL": "", "SCHEDULE": cron})
            self.assertEqual(rc, 0, cron)
            selected[cron] = out["model"]
        self.assertEqual(
            selected,
            {"0 10 * * *": "day-10am", "0 17 * * *": "day-5pm"},
        )
        # A called run sees its caller's schedule; the caller's input must win.
        _, out = self._run_step(plan, {"MODEL": "gemma4", "SCHEDULE": "0 1 * * *"})
        self.assertEqual(out["model"], "gemma4")
        rc, _ = self._run_step(plan, {"MODEL": "", "SCHEDULE": "0 3 * * *"})
        self.assertNotEqual(rc, 0, "an unknown schedule must not fall back to all")

    def test_scheduled_accuracy_discovers_every_model(self):
        import yaml

        workflow = yaml.safe_load((WORKFLOWS / "accuracy-test-tpu.yml").read_text())
        job = workflow["jobs"]["discover-multihost"]
        step = next(s for s in job["steps"] if s.get("id") == "d")

        def discover(model):
            rc, out = self._run_step(
                step, {"MODEL": model, "TEST_FILTER": "", "UPDATE_DASHBOARD": "false"}
            )
            self.assertEqual(rc, 0, model)
            return json.loads(out["groups"])

        names = {
            model: {g["name"] for g in discover(model)}
            for model in ("day-10am", "day-5pm", "full-set")
        }
        # collect-full-evaluations combines the Kimi MMMU shards of one run.
        kimi = {"test_tpu_kimi_k26_mmmu_shard0", "test_tpu_kimi_k26_mmmu_shard1"}
        self.assertTrue(any(kimi <= names[m] for m in ("day-10am", "day-5pm")))
        # Every multihost accuracy file runs in exactly one scheduled selection.
        slots = ("day-10am", "day-5pm")
        scheduled = [names[m] for m in slots]
        self.assertEqual(set().union(*scheduled), names["full-set"])
        self.assertEqual(sum(map(len, scheduled)), len(names["full-set"]))
        # A stale slot entry would make the est_time bot fail on a missing file.
        table = json.loads((TPU_CI / "accuracy_slots.json").read_text())
        self.assertEqual(set(table["multihost"]), names["full-set"])
        # Test authors only write "nightly"; the workflow picks the slot.
        for row in discover("full-set"):
            self.assertEqual(row["schedule"], "nightly", row["name"])

        groups = discover("full-set")
        recipes = [g["launch_script"] for g in groups]
        self.assertEqual(
            recipes.count("recipe/configs/deepseek_r1_0528/accuracy.py"), 1
        )
        self.assertEqual(recipes.count("recipe/configs/kimi_k26/mmmu.py"), 2)
        self.assertNotIn("recipe/configs/qwen3_5_397b/server.py", recipes)
        for group in groups:
            self.assertTrue((REPO / "benchmark/tpu" / group["launch_script"]).is_file())
            # A hosted runner has six hours, including startup and cleanup.
            runtime = 3600 + group["timeout_per_file"] + 600
            self.assertGreater(21600 - runtime - 900 - 600, 0, group["name"])
        # The nightly no longer runs accuracy.
        for job in workflow["jobs"].values():
            self.assertNotIn('"nightly"', job.get("if", ""))
        # Every single-host leg runs in exactly one scheduled slot; `all` runs each.
        legs = workflow["jobs"]["plan"]["steps"][-1]

        def suites(model):
            rc, out = self._run_step(legs, {"MODEL": model})
            self.assertEqual(rc, 0, model)
            return [leg["suite"] for leg in json.loads(out["singlehost"])]

        ten, five, everything = (suites(m) for m in ("day-10am", "day-5pm", "all"))
        self.assertEqual(sorted(ten + five), sorted(everything))
        self.assertEqual(sorted(suites("full-set")), sorted(everything))
        # A model name picks its own legs, never a longer name's (gpt-oss-20b).
        self.assertEqual(suites("gpt-oss-20b"), ["nightly-tpu-accuracy-gpt-oss-20b"])
        self.assertEqual(
            sorted(suites("gemma4")),
            sorted(
                s for s in everything if s.startswith("nightly-tpu-accuracy-gemma4-")
            ),
        )
        nightly = yaml.safe_load((WORKFLOWS / "nightly-test-tpu.yml").read_text())
        self.assertNotIn("accuracy", nightly["jobs"])

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
            + [row["files"] for row in self.multihost_accuracy]
            + list(self.static_pd)
            + list(self.external)
        )
        accuracy_perf_files = []
        rendered = _discover_suites(tuple(self.registered))
        for suite in self.registered:
            workflow = None
            if suite.startswith("nightly-tpu-perf-"):
                workflow = "perf-test-tpu.yml"
            elif suite.startswith("nightly-tpu-accuracy-"):
                workflow = "accuracy-test-tpu.yml"
            if workflow is None:
                continue
            # The suite must be wired into its stage workflow's matrix...
            if workflow == "accuracy-test-tpu.yml":
                text = (TPU_CI / "accuracy_slots.json").read_text(encoding="utf-8")
                where = "accuracy_slots.json"
            else:
                text = (WORKFLOWS / workflow).read_text(encoding="utf-8")
                where = f"{workflow}'s matrix"
            self.assertIn(f'"{suite}"', text, f"{suite} is not in {where}")
            # ...and render to exactly one file (one file, one suite, one job).
            rows = rendered[suite]
            self.assertEqual(
                len(rows), 1, f"{suite} renders {len(rows)} jobs, contract is 1"
            )
            accuracy_perf_files.append(rows[0]["file"])
        job_files += accuracy_perf_files

        # ...and every slot-table leg is still a registered suite.
        table = json.loads((TPU_CI / "accuracy_slots.json").read_text())
        stale = set(table["singlehost"]) - set(self.registered)
        self.assertFalse(
            stale, f"accuracy_slots.json legs with no registration: {stale}"
        )

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
            config = (
                mod.build_model_config()
                if hasattr(mod, "build_model_config")
                else mod.MODEL_CONFIG
            )
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
            series_id = (
                mod.build_model_config()
                if hasattr(mod, "build_model_config")
                else mod.MODEL_CONFIG
            )["id"]
            self.assertNotIn(
                series_id, seen, f"{path.name} reuses id of {seen.get(series_id)}"
            )
            seen[series_id] = path.name

    def test_accuracy_gate_labels_carry_identity(self):
        for path, mod in self.accuracy:
            for params in model_config.build_gates(
                (
                    mod.build_model_config()
                    if hasattr(mod, "build_model_config")
                    else mod.MODEL_CONFIG
                )
            ):
                labels = params.metric_labels or {}
                self.assertEqual(labels.get("kind"), "accuracy", path.name)
                self.assertEqual(
                    labels.get("id"),
                    (
                        mod.build_model_config()
                        if hasattr(mod, "build_model_config")
                        else mod.MODEL_CONFIG
                    )["id"],
                    path.name,
                )
                self.assertEqual(
                    labels.get("display_name"),
                    (
                        mod.build_model_config()
                        if hasattr(mod, "build_model_config")
                        else mod.MODEL_CONFIG
                    )["display_name"],
                    path.name,
                )

    @staticmethod
    def _fixture_config_module(mod) -> str:
        """The recipe a TPULauncherTestCase subclass in this module serves."""
        return next(
            (
                obj.config_module
                for obj in vars(mod).values()
                if isinstance(obj, type) and getattr(obj, "config_module", "")
            ),
            "",
        )

    def test_perf_measures_through_recipe_and_bench_serving_only(self):
        # A recipe launch plus bench_serving.sh is what puts the server/bench
        # config panels on the dashboard; a hand-booted server or hand-rolled
        # bench produces a chart nobody can reproduce from. Three launch forms
        # qualify: MODEL_CONFIG's own `launch.recipe` or `launch.config`, and
        # the unified launcher, where a TPULauncherTestCase subclass serves a
        # Python recipe module and the config carries gates only.
        for path, mod in self.perf:
            config = (
                mod.build_model_config()
                if hasattr(mod, "build_model_config")
                else mod.MODEL_CONFIG
            )
            launch = config.get("launch") or {}
            self.assertTrue(
                launch.get("recipe")
                or launch.get("config")
                or self._fixture_config_module(mod),
                f"{path.name}: perf must serve through a recipe -- launch.recipe, "
                "launch.config, or a TPULauncherTestCase subclass with config_module",
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
            (
                mod.build_model_config()
                if hasattr(mod, "build_model_config")
                else mod.MODEL_CONFIG
            )["id"]
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
        *TAXONOMY_TPU_TESTS.values(),
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
