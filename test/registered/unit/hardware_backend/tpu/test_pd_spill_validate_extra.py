"""Lower-priority unit tests of the offload's refusals: several problems are reported together, and the wording of the pair-level reasons. Fixtures come from test_pd_spill_validate.py."""

import importlib.util
import sys
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu import pd_spill_validate

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _load_sibling(name: str):
    path = Path(__file__).with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_main = _load_sibling("test_pd_spill_validate")
SWITCH = _main.SWITCH
_engine = _main._engine
_validate = _main._validate


class TestEngineRefusalsExtra(CustomTestCase):
    def test_every_problem_is_reported_at_once(self):
        engine = _engine()
        engine.memory.enable_unified_memory = True
        engine.kv_manager.pp_size = 2
        with self.assertRaises(pd_spill_validate.UnsupportedSpillConfig) as raised:
            _validate(engine)
        self.assertIn("--enable-unified-memory", str(raised.exception))
        self.assertIn("pipeline parallelism", str(raised.exception))


class TestPairRefusals(CustomTestCase):
    def test_switch_must_match_on_both_roles(self):
        for mine, theirs in ((True, False), (False, True)):
            with self.subTest(mine=mine, theirs=theirs), SWITCH.override(mine):
                reason = pd_spill_validate.peer_switch_mismatch(
                    peer_role="decode", peer_enabled=theirs
                )
                self.assertIn("same value on both roles", reason)
                self.assertIn(
                    "on for " + ("this engine" if mine else "its decode"), reason
                )
        for value in (True, False):
            with SWITCH.override(value):
                self.assertIsNone(
                    pd_spill_validate.peer_switch_mismatch(
                        peer_role="decode", peer_enabled=value
                    )
                )

    def test_resharding_is_refused_only_with_the_switch_on(self):
        with SWITCH.override(True):
            self.assertIn("resharding", pd_spill_validate.resharding_refusal())
        with SWITCH.override(False):
            self.assertIsNone(pd_spill_validate.resharding_refusal())


if __name__ == "__main__":
    unittest.main()
