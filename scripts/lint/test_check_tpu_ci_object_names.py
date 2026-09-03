import unittest

from check_tpu_ci_object_names import check_workflows

CALL = "./.github/workflows/_tpu-test.yml"


def workflow(job: dict) -> dict:
    """One workflow with one leg calling _tpu-test.yml."""
    return {"w.yml": {"jobs": {"leg": {"uses": CALL, **job}}}}


class TestObjectNameBudget(unittest.TestCase):
    def errors(self, workflows: dict) -> list:
        return check_workflows(workflows)

    def test_accepts_a_name_inside_the_cap(self):
        self.assertEqual(
            self.errors(
                workflow(
                    {"with": {"run_name": "nightly-accuracy-gpt-oss-120b-mmlupro"}}
                )
            ),
            [],
        )

    def test_rejects_a_name_over_the_cap(self):
        errors = self.errors(
            workflow(
                # 46 chars: 68 with the tpu-ci- prefix and the object suffixes.
                {
                    "with": {
                        "run_name": "nightly-tpu-accuracy-deepseek-r1-0528-multihost"
                    }
                }
            )
        )
        self.assertEqual(len(errors), 1)
        self.assertIn("68 chars > 63", errors[0])

    def test_falls_back_to_the_suite_when_no_run_name_is_passed(self):
        errors = self.errors(workflow({"with": {"suite": "x" * 43}}))
        self.assertEqual(len(errors), 1)
        self.assertIn("`suite`", errors[0])

    def test_measures_a_pd_leg_against_both_role_names(self):
        # P/D replicated jobs are `prefill`/`decode`, longer than `slice`, so a
        # 42-char name fits a single-slice leg and overruns both P/D roles.
        job = {"with": {"run_name": "x" * 42, "pd_prefill_hosts": 2}}
        errors = self.errors(workflow(job))
        self.assertEqual(len(errors), 2)  # one per role
        self.assertIn("prefill", errors[0])
        self.assertIn("decode", errors[1])
        del job["with"]["pd_prefill_hosts"]
        self.assertEqual(self.errors(workflow(job)), [])  # fits as a slice leg

    def test_measures_every_matrix_leg(self):
        errors = self.errors(
            workflow(
                {
                    "strategy": {
                        "matrix": {
                            "include": [
                                {"run_name": "short"},
                                {"run_name": "x" * 60},
                            ]
                        }
                    },
                    "with": {"run_name": "${{ matrix.run_name }}"},
                }
            )
        )
        self.assertEqual(len(errors), 1)  # only the long leg
        self.assertIn("x" * 60, errors[0])

    def test_measures_a_suite_a_matrix_spells_out_in_an_expression(self):
        # perf-test-tpu.yml picks its suite with a fromJSON('["..."]') branch;
        # the `inputs.model == '...'` half is a condition, not a candidate.
        errors = self.errors(
            workflow(
                {
                    "strategy": {
                        "matrix": {
                            "suite": (
                                "${{ (inputs.model == 'gpt-oss-120b' "
                                "&& fromJSON('[\"%s\"]')) || fromJSON('[\"s\"]') }}"
                                % ("x" * 43)
                            )
                        }
                    },
                    "with": {"suite": "${{ matrix.suite }}"},
                }
            )
        )
        self.assertEqual(len(errors), 1)
        self.assertIn("x" * 43, errors[0])

    def test_rejects_a_run_name_of_unknowable_length(self):
        # The file stem discovery also emits; this is how a leg overruns the cap.
        errors = self.errors(
            workflow({"with": {"run_name": "${{ matrix.group.name }}"}})
        )
        self.assertEqual(len(errors), 1)
        self.assertIn("not knowable", errors[0])

    def test_accepts_the_discovery_scripts_compact_run_name(self):
        self.assertEqual(
            self.errors(
                workflow({"with": {"run_name": "${{ matrix.group.run_name }}"}})
            ),
            [],
        )

    def test_ignores_a_leg_with_neither_name(self):
        self.assertEqual(self.errors(workflow({"with": {"files": "test/a.py"}})), [])

    def test_ignores_jobs_that_do_not_call_the_reusable_workflow(self):
        self.assertEqual(
            self.errors({"w.yml": {"jobs": {"build": {"runs-on": "ubuntu-latest"}}}}),
            [],
        )


if __name__ == "__main__":
    unittest.main()
