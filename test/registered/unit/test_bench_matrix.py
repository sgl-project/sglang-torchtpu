"""Bookkeeping guards for the TPU benchmark matrix (no TPU needed).

The matrix is stitched together by convention across three places -- the
test files' MULTIHOST_LAUNCH dicts (AST-scanned), the launch wrappers under
benchmark/tpu/recipe/bench_matrix/, and config.sh's (model, recipe,
workload) axes -- and nothing else in CI checks the stitching without a TPU
slice. Each case here answers "someone extended X without updating Y":

  * a test file naming a launch wrapper that does not exist ships a group
    whose JobSet dies at `bash: no such file` after weights are mounted;
  * the two workload files of a (model, recipe) cell drifting apart in
    (suite, launch_script, weights) silently splits their shared server into
    two full bring-ups;
  * a recipe registering against the wrong slice shape (a pd cell on the
    2-host suite) only fails at engine rendezvous, an hour into the run;
  * config.sh losing a sweep axis (concurrency {1,16,32}; long requests
    {1,2,4} for agentic, none for 8k1k) would keep every downstream gate
    green while measuring a smaller matrix.
"""

import ast
import glob
import importlib.util
import os
import subprocess
import sys
import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 3))
E2E = os.path.join(REPO, "test", "registered", "tpu", "e2e")
BENCH_MATRIX = os.path.join(REPO, "benchmark", "tpu", "recipe", "bench_matrix")

# The budget constants live in one place (bench_matrix_base); this ratchet
# checks every place that must agree with them.
sys.path.insert(0, E2E)
import bench_matrix_base as bench_base  # noqa: E402

MODELS = ("gpt_oss_120b", "qwen3_coder_480b")
RECIPES = {
    "1host_pc": "external-nightly-2x2x2",
    "1host_pc_spec_kvoff": "external-nightly-2x2x2",
    "pd_2p2d_pc_spec_kvoff": "external-nightly-2x2x4",
}
WORKLOADS = ("agentic", "8k1k")
# Cells outside the MODELS x RECIPES x WORKLOADS grid: a recipe that only
# makes sense for one model / workload (the long-context P/D stress cell is a
# GPT-OSS multi-turn study). (model, recipe) -> (workloads, suite, sweep).
STANDALONE_CELLS = {
    ("gpt_oss_120b", "pd_1p1d_pc"): (
        ("agentic",),
        "external-nightly-2x2x2",
        "BS=1 8 16|LONG=1 4",
    ),
}


def _recipe_suite(basename):
    """The suite a cell file must register to, from its file name."""
    for (model, recipe), (_, suite, _) in STANDALONE_CELLS.items():
        if basename.startswith(f"test_tpu_bench_{model}_{recipe}_"):
            return recipe, suite
    recipe = next(r for r in RECIPES if f"_{r}_" in basename)
    return recipe, RECIPES[recipe]


def _scan_file(path):
    """(suite, MULTIHOST_LAUNCH, disabled) the CI scanner would read, via the
    same AST contract (literal kwargs / literal dict)."""
    tree = ast.parse(open(path).read())
    suite, launch, disabled = None, None, None
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and getattr(node.func, "id", "") == "register_tpu_ci"
        ):
            kw = {k.arg: k.value for k in node.keywords}
            if "suite" in kw:
                suite = ast.literal_eval(kw["suite"])
            if "disabled" in kw:
                disabled = ast.literal_eval(kw["disabled"])
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "MULTIHOST_LAUNCH":
                    launch = ast.literal_eval(node.value)
    return suite, launch, disabled


class TestBenchMatrix(CustomTestCase):
    def _cell_files(self):
        files = sorted(glob.glob(os.path.join(E2E, "test_tpu_bench_*.py")))
        expected = {
            os.path.join(E2E, f"test_tpu_bench_{m}_{r}_{w}.py")
            for m in MODELS
            for r in RECIPES
            for w in WORKLOADS
        } | {
            os.path.join(E2E, f"test_tpu_bench_{m}_{r}_{w}.py")
            for (m, r), (workloads, _, _) in STANDALONE_CELLS.items()
            for w in workloads
        }
        self.assertEqual(
            set(files),
            expected,
            "the matrix is 2 models x 3 recipes x 2 workloads plus "
            "STANDALONE_CELLS; a missing or extra test_tpu_bench_* file means "
            "a cell was added or dropped without updating this inventory",
        )
        return files

    def test_every_cell_names_an_existing_launch_wrapper(self):
        for path in self._cell_files():
            _, launch, _ = _scan_file(path)
            self.assertIsNotNone(launch, f"{path}: no literal MULTIHOST_LAUNCH")
            script = os.path.join(REPO, "benchmark", "tpu", launch["launch_script"])
            self.assertTrue(
                os.path.isfile(script),
                f"{path}: launch_script {launch['launch_script']} does not exist",
            )

    def test_workload_pairs_share_one_launch_group(self):
        """Files sharing (suite, launch_script, weights) share one server; a
        drifted field silently doubles the bring-up cost of the cell."""
        for model in MODELS:
            for recipe in RECIPES:
                keys = set()
                for workload in WORKLOADS:
                    path = os.path.join(
                        E2E, f"test_tpu_bench_{model}_{recipe}_{workload}.py"
                    )
                    suite, launch, _ = _scan_file(path)
                    keys.add((suite, launch["launch_script"], launch["weights"]))
                self.assertEqual(
                    len(keys),
                    1,
                    f"{model}/{recipe}: workload files disagree on the launch "
                    f"group key: {keys}",
                )

    def test_recipe_registers_against_its_slice_shape(self):
        """A pd cell on the 2-host suite (or a 1-host cell on the 4-host one)
        fails only at engine rendezvous, an hour into the run."""
        for path in self._cell_files():
            suite, _, _ = _scan_file(path)
            recipe, expected = _recipe_suite(os.path.basename(path))
            self.assertEqual(
                suite,
                expected,
                f"{path}: recipe {recipe} must register to {expected}",
            )

    def test_discovery_buckets_one_group_per_slice(self):
        """The even-split contract: 4 single-host groups over the four 2x2x2
        slices, 2 pd groups over the two 2x2x4 slices, none serialized. A
        light tail group (MULTIHOST_LAUNCH "placement": "tail") may share a
        slice, but only AFTER that slice's regular group, so a hang upstream
        costs only the tail group its run."""
        spec = importlib.util.spec_from_file_location(
            "render_external_jobset",
            os.path.join(
                REPO, ".github", "tpu-ci", "external", "render_external_jobset.py"
            ),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        cfg = mod.load_config(
            os.path.join(
                REPO, ".github", "tpu-ci", "external", "external-topology.yaml"
            )
        )
        slices = mod.list_groups(
            cfg, os.path.join(REPO, "test", "registered"), nightly_only=True
        )
        per_suite = {}
        for sl in slices:
            ours = [
                g
                for g in sl["groups"]
                if g.get("launch_script", "").startswith("recipe/bench_matrix/")
            ]
            regular = [g for g in ours if g["placement"] != "tail"]
            self.assertLessEqual(
                len(regular),
                1,
                f"{sl['name']}: {len(regular)} regular bench_matrix groups "
                f"serialize on one slice -- the matrix is sized to fill each "
                f"slice exactly once ({[g['name'] for g in regular]})",
            )
            # P/D, exec and self-launch groups share these slices and carry no
            # placement; they count as regular.
            placements = [g.get("placement", "regular") for g in sl["groups"]]
            self.assertEqual(
                placements,
                sorted(placements, key=lambda p: p == "tail"),
                f"{sl['name']}: a tail group must run after every regular "
                f"group on its slice ({[g['name'] for g in sl['groups']]})",
            )
            per_suite.setdefault(sl["suite"], 0)
            per_suite[sl["suite"]] += len(ours)
            for g in ours:
                recipe, _ = _recipe_suite(os.path.basename(g["files"].split()[0]))
                self.assertEqual(
                    g["placement"] == "tail",
                    recipe in bench_base.TAIL_RECIPES,
                    f"{g['name']}: placement must be 'tail' exactly for "
                    "bench_matrix_base.TAIL_RECIPES",
                )
                self.assertEqual(
                    g["timeout_per_file"],
                    bench_base.timeout_per_file_budget(recipe),
                    f"{g['name']}: discovery must carry the files' declared "
                    "timeout_per_file (the JobSet deadline is sized from it)",
                )
        # Expected group counts follow the files themselves: a cell disabled
        # with a reason (e.g. the raiden P/D transfer finding) is skipped by
        # discovery, and this test must not force it back on.
        expected = {}
        for path in self._cell_files():
            suite, launch, disabled = _scan_file(path)
            if disabled is not None:
                continue
            key = (suite, launch["launch_script"])
            expected.setdefault(suite, set()).add(key)
        self.assertEqual(
            per_suite.get("external-nightly-2x2x2", 0),
            len(expected.get("external-nightly-2x2x2", ())),
        )
        self.assertEqual(
            per_suite.get("external-nightly-2x2x4", 0),
            len(expected.get("external-nightly-2x2x4", ())),
        )

    def test_config_resolves_the_sweep_grid(self):
        """config.sh must keep the CI grid: concurrency {1,16,32}, long
        requests {1,2,4} for agentic and none for 8k1k, for every
        (model, recipe) -- and each standalone cell its own grid."""
        expected_long = {"agentic": "1 2 4", "8k1k": "0"}
        cells = [
            (m, r, w, f"BS=1 16 32|LONG={expected_long[w]}")
            for m in MODELS
            for r in RECIPES
            for w in WORKLOADS
        ] + [
            (m, r, w, sweep)
            for (m, r), (workloads, _, sweep) in STANDALONE_CELLS.items()
            for w in workloads
        ]
        for model, recipe, workload, expected_sweep in cells:
            with self.subTest(cell=f"{model}/{recipe}/{workload}"):
                out = subprocess.run(
                    [
                        "bash",
                        "-c",
                        'source ./config.sh && echo "BS=$SWEEP_BS|LONG=$SWEEP_LONG"',
                    ],
                    cwd=BENCH_MATRIX,
                    env={
                        **os.environ,
                        "BENCH_MODEL": model,
                        "BENCH_RECIPE": recipe,
                        "WORKLOAD": workload,
                        "PROFILE": "ci",
                    },
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(
                    out.returncode, 0, f"{model}/{recipe}/{workload}: {out.stderr}"
                )
                self.assertIn(expected_sweep, out.stdout)

    def test_reused_nightly_tools_exist(self):
        """config.sh reaches into recipe/qwen3_coder_480b/nightly/ for the
        trace generator, metrics probe, and summarizer instead of copying
        them. A rename there would otherwise surface as failed:trace-gen legs
        an hour into a TPU run -- the one self-location coupling in config.sh
        whose failure is not loud at source time."""
        nightly = os.path.join(
            REPO, "benchmark", "tpu", "recipe", "qwen3_coder_480b", "nightly"
        )
        for tool in ("gen_agentic_workload.py", "metrics_probe.py", "summarize.py"):
            self.assertTrue(
                os.path.isfile(os.path.join(nightly, tool)),
                f"{tool} moved out of {nightly}; update NIGHTLY_TOOLS_DIR in "
                "benchmark/tpu/recipe/bench_matrix/config.sh",
            )

    def test_time_budget_chain(self):
        """The chain from config.sh: worst-case sweep <= SWEEP_TIMEOUT <
        TIMEOUT_PER_FILE. Its links live in three files (test literals, the
        base's constants, config.sh's profile) that nothing else cross-checks;
        this is what turns "we changed one number" into a red test instead of
        a silently truncated benchmark."""
        # Every cell's literal must equal the base's budget constant for its
        # recipe (tail cells carry the smaller tail budget).
        for path in self._cell_files():
            _, launch, _ = _scan_file(path)
            recipe, _ = _recipe_suite(os.path.basename(path))
            self.assertEqual(
                launch.get("timeout_per_file"),
                bench_base.timeout_per_file_budget(recipe),
                f"{path}: timeout_per_file literal drifted from "
                "bench_matrix_base.timeout_per_file_budget()",
            )
            self.assertEqual(
                launch.get("placement") == "tail",
                recipe in bench_base.TAIL_RECIPES,
                f"{path}: MULTIHOST_LAUNCH placement must be 'tail' exactly "
                "for bench_matrix_base.TAIL_RECIPES",
            )
        # The sweep budget plus the health recheck must fit the file budget.
        for sweep, budget in (
            (bench_base.SWEEP_TIMEOUT_DEFAULT, bench_base.TIMEOUT_PER_FILE_BUDGET),
            (
                bench_base.TAIL_SWEEP_TIMEOUT_DEFAULT,
                bench_base.TAIL_TIMEOUT_PER_FILE_BUDGET,
            ),
        ):
            self.assertLess(sweep + bench_base.HEALTH_RECHECK_BUDGET, budget)
        # config.sh's ci profile: attempted legs x leg cap must fit the sweep
        # budget (legs with long > bs are skipped and cost nothing). Checked
        # for the matrix's largest grid and for every standalone cell, which
        # sets its own.
        budget_cells = [("gpt_oss_120b", "1host_pc_spec_kvoff", "agentic")] + [
            (m, r, w)
            for (m, r), (workloads, _, _) in STANDALONE_CELLS.items()
            for w in workloads
        ]
        for model, recipe, workload in budget_cells:
            out = subprocess.run(
                [
                    "bash",
                    "-c",
                    'source ./config.sh && echo "$SWEEP_BS|$SWEEP_LONG|$MAX_LEG_SECONDS"',
                ],
                cwd=BENCH_MATRIX,
                env={
                    **os.environ,
                    "BENCH_MODEL": model,
                    "BENCH_RECIPE": recipe,
                    "WORKLOAD": workload,
                    "PROFILE": "ci",
                },
                capture_output=True,
                text=True,
            )
            self.assertEqual(out.returncode, 0, out.stderr)
            bs_axis, long_axis, leg_cap = out.stdout.strip().split("|")
            attempted = [
                (b, l)
                for b in map(int, bs_axis.split())
                for l in map(int, long_axis.split())
                if l <= b
            ]
            worst_case = len(attempted) * int(leg_cap)
            sweep_timeout = bench_base.sweep_timeout_default(recipe)
            self.assertLessEqual(
                worst_case,
                sweep_timeout,
                f"{model}/{recipe}/{workload}: {len(attempted)} attempted ci "
                f"legs x MAX_LEG_SECONDS={leg_cap} = {worst_case}s exceeds "
                f"SWEEP_TIMEOUT {sweep_timeout}s",
            )


if __name__ == "__main__":
    unittest.main()
