"""SPMDCompiler.finish runs its before_distribute steps once every rank traced
successfully, and never after a failed trace or under verification. Runs on the
host.
"""

import unittest

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.compilation import spmd_compiler

register_tpu_ci(est_time=10, suite="stage-a-test-tpu-unit", chips=1)


def _compiler(peer_error=None, verify=False):
    compiler = object.__new__(spmd_compiler.SPMDCompiler)
    compiler.verify_spmd_safe = verify
    compiler.gather = lambda value: [value, peer_error if value is None else value]
    compiler.submissions = {}
    compiler.trace_seconds = {}
    compiler.rank = 1
    compiler._distribute = lambda: None
    compiler._finish_verification = lambda error: None
    compiler.before_distribute = []
    return compiler


class TestBeforeDistribute(CustomTestCase):
    def _run(self, compiler):
        calls = []
        compiler.before_distribute.append(calls.append)
        compiler.finish()
        return calls

    def test_runs_with_the_compiler_gather_after_agreed_traces(self):
        compiler = _compiler()
        self.assertEqual(self._run(compiler), [compiler.gather])

    def test_skipped_when_any_rank_failed_to_trace(self):
        compiler = _compiler(peer_error="trace failed on rank 0")
        calls = []
        compiler.before_distribute.append(calls.append)
        with self.assertRaises(RuntimeError):
            compiler.finish()
        self.assertEqual(calls, [])

    def test_skipped_under_verification(self):
        self.assertEqual(self._run(_compiler(verify=True)), [])


if __name__ == "__main__":
    unittest.main()
