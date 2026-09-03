import unittest

from check_tpu_ci_names import check_workflows, expand, matrix_legs

CALL = "./.github/workflows/_tpu-test.yml"


def workflow(job: dict) -> dict:
    """One workflow with one leg calling _tpu-test.yml."""
    return {"w.yml": {"jobs": {"leg": {"uses": CALL, **job}}}}


class TestCheckWorkflows(unittest.TestCase):
    def errors(self, workflows: dict) -> list:
        return check_workflows(workflows)

    def test_accepts_a_stage_tag_plus_a_test(self):
        self.assertEqual(
            self.errors(
                workflow(
                    {
                        "name": "single-host",
                        "with": {"display_name": "test_x", "suite": "nightly-tpu-4-v7"},
                    }
                )
            ),
            [],
        )

    def test_accepts_a_suite_as_the_last_segment(self):
        # pr-test-tpu passes no display_name; the suite is the informative name.
        self.assertEqual(
            self.errors(
                workflow(
                    {"name": "unit-test", "with": {"suite": "stage-a-test-tpu-unit"}}
                )
            ),
            [],
        )

    def test_rejects_a_leg_that_names_no_test(self):
        errors = self.errors(
            workflow({"name": "single-host", "with": {"files": "test/a.py"}})
        )
        self.assertEqual(len(errors), 1)
        self.assertIn("names no test", errors[0])

    def test_rejects_a_check_that_reads_x_slash_x(self):
        errors = self.errors(
            workflow(
                {"name": "test_x", "with": {"display_name": "test_x", "suite": "s"}}
            )
        )
        self.assertEqual(len(errors), 1)
        self.assertIn("`x / x`", errors[0])

    def test_rejects_a_computed_caller_name(self):
        errors = self.errors(
            workflow(
                {
                    "name": "${{ matrix.test.name }}",
                    "with": {"display_name": "test_x", "suite": "s"},
                }
            )
        )
        self.assertEqual(len(errors), 1)
        self.assertIn("stage tag", errors[0])

    def test_rejects_an_object_name_over_the_cap(self):
        errors = self.errors(
            workflow(
                {
                    "name": "accuracy",
                    "with": {
                        "display_name": "test_x",
                        # 46 chars: 68 with the tpu-ci- prefix and the suffixes.
                        "run_name": "nightly-tpu-accuracy-deepseek-r1-0528-multihost",
                    },
                }
            )
        )
        self.assertEqual(len(errors), 1)
        self.assertIn("68 chars > 63", errors[0])

    def test_measures_a_pd_leg_against_both_role_names(self):
        # P/D replicated jobs are `prefill`/`decode`, longer than `slice`, so a
        # 42-char name fits a single-slice leg and overruns both P/D roles.
        job = {
            "name": "pd-mh-gpt-120b",
            "with": {
                "display_name": "test_x",
                "run_name": "x" * 42,
                "pd_prefill_hosts": 2,
            },
        }
        errors = self.errors(workflow(job))
        self.assertEqual(len(errors), 2)  # one per role
        self.assertIn("prefill", errors[0])
        self.assertIn("decode", errors[1])
        del job["with"]["pd_prefill_hosts"]
        self.assertEqual(self.errors(workflow(job)), [])  # fits as a slice leg

    def test_rejects_a_run_name_of_unknowable_length(self):
        errors = self.errors(
            workflow(
                {
                    "name": "accuracy-multihost",
                    "with": {
                        "display_name": "test_x",
                        "run_name": "${{ matrix.group.name }}",
                    },
                }
            )
        )
        self.assertEqual(len(errors), 1)
        self.assertIn("not knowable", errors[0])

    def test_accepts_the_discovery_scripts_compact_run_name(self):
        self.assertEqual(
            self.errors(
                workflow(
                    {
                        "name": "single-host",
                        "with": {
                            "display_name": "test_x",
                            "run_name": "${{ matrix.test.run_name }}",
                        },
                    }
                )
            ),
            [],
        )

    def test_measures_every_matrix_leg(self):
        errors = self.errors(
            workflow(
                {
                    "name": "accuracy",
                    "strategy": {
                        "matrix": {
                            "include": [
                                {"suite": "a", "run_name": "short"},
                                {"suite": "b", "run_name": "x" * 60},
                            ]
                        }
                    },
                    "with": {
                        "display_name": "${{ matrix.suite }}",
                        "run_name": "${{ matrix.run_name }}",
                    },
                }
            )
        )
        self.assertEqual(len(errors), 1)  # only the long leg
        self.assertIn("x" * 60, errors[0])

    def test_ignores_jobs_that_do_not_call_the_reusable_workflow(self):
        self.assertEqual(
            self.errors({"w.yml": {"jobs": {"build": {"runs-on": "ubuntu-latest"}}}}),
            [],
        )


class TestMatrixExpansion(unittest.TestCase):
    def test_include_legs_are_whole_assignments(self):
        job = {"strategy": {"matrix": {"include": [{"a": "1", "b": "2"}]}}}
        self.assertEqual(matrix_legs(job), [{"a": "1", "b": "2"}])

    def test_a_plain_matrix_is_the_product_of_its_keys(self):
        job = {"strategy": {"matrix": {"a": ["1", "2"], "b": ["3"]}}}
        self.assertEqual(matrix_legs(job), [{"a": "1", "b": "3"}, {"a": "2", "b": "3"}])

    def test_a_key_computed_from_inline_literals_still_expands(self):
        # perf-test-tpu.yml picks its suite with a fromJSON('["..."]') branch.
        job = {
            "strategy": {
                "matrix": {
                    "suite": (
                        "${{ (inputs.model == 'gpt-oss-120b' "
                        "&& fromJSON('[\"nightly-tpu-perf-gpt-oss-120b\"]')) "
                        "|| fromJSON('[\"nightly-tpu-perf-gpt-oss-120b\"]') }}"
                    )
                }
            }
        }
        # `gpt-oss-120b` from the `inputs.model == ...` condition is not a value.
        self.assertEqual(
            expand("${{ matrix.suite }}", matrix_legs(job)),
            ["nightly-tpu-perf-gpt-oss-120b"],
        )

    def test_a_matrix_from_a_discovery_script_does_not_expand(self):
        job = {
            "strategy": {
                "matrix": {"test": "${{ fromJSON(needs.discover.outputs.tests) }}"}
            }
        }
        self.assertEqual(matrix_legs(job), [])
        self.assertIsNone(expand("${{ matrix.test.name }}", matrix_legs(job)))

    def test_a_literal_expands_to_itself(self):
        self.assertEqual(expand("nightly-accuracy-x", []), ["nightly-accuracy-x"])


if __name__ == "__main__":
    unittest.main()
