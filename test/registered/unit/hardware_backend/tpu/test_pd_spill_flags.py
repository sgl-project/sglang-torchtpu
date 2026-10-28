"""The offload-to-decode-DRAM switch implies every component flag; explicit flags win.

Guards the bookkeeping between SGLANG_ENABLE_PREFILL_KV_OFFLOAD_TO_DECODE_DRAM and the four
component flags it stands for: a component that stops following the switch, or
a switch default that overrides an explicitly set component value, fails here.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

import unittest
from contextlib import ExitStack

from sglang.test.test_utils import CustomTestCase
from sglang_tpu import pd_spill_flags as flags
from sglang_tpu.envs import envs as tpu_envs

_COMPONENTS = (
    "SGLANG_DISAGG_PREFIX_VICTIM_PUSH",
    "SGLANG_DISAGG_PREFILL_PULL_DECODE_PREFIX",
)


class TestSpillSwitch(CustomTestCase):
    def _all_cleared(self, stack):
        for name in (*_COMPONENTS, "SGLANG_TPU_PD_SWA_HOST_WINDOWS"):
            field = getattr(tpu_envs, name)
            stack.enter_context(field.override(field.default))
            field.clear()

    def test_switch_off_leaves_every_component_at_its_own_flag(self):
        with ExitStack() as stack:
            self._all_cleared(stack)
            stack.enter_context(
                tpu_envs.SGLANG_ENABLE_PREFILL_KV_OFFLOAD_TO_DECODE_DRAM.override(False)
            )
            self.assertFalse(flags.victim_push_enabled())
            self.assertFalse(flags.decode_prefix_pull_enabled())
            self.assertFalse(flags.prefill_swa_path_end_backup_enabled())
            self.assertEqual(flags.decode_swa_host_windows(), 0)
            with tpu_envs.SGLANG_TPU_PD_SWA_HOST_WINDOWS.override(64):
                # One value drives both roles.
                self.assertTrue(flags.prefill_swa_path_end_backup_enabled())
                self.assertEqual(flags.decode_swa_host_windows(), 64)
            with tpu_envs.SGLANG_DISAGG_PREFIX_VICTIM_PUSH.override(True):
                self.assertTrue(flags.victim_push_enabled())
                self.assertFalse(flags.decode_prefix_pull_enabled())

    def test_switch_on_implies_every_component(self):
        with ExitStack() as stack:
            self._all_cleared(stack)
            stack.enter_context(
                tpu_envs.SGLANG_ENABLE_PREFILL_KV_OFFLOAD_TO_DECODE_DRAM.override(True)
            )
            self.assertTrue(flags.victim_push_enabled())
            self.assertTrue(flags.decode_prefix_pull_enabled())
            self.assertTrue(flags.prefill_swa_path_end_backup_enabled())
            self.assertEqual(flags.decode_swa_host_windows(), 128)

    def test_explicit_window_count_wins_over_the_switch_default(self):
        with ExitStack() as stack:
            self._all_cleared(stack)
            stack.enter_context(
                tpu_envs.SGLANG_ENABLE_PREFILL_KV_OFFLOAD_TO_DECODE_DRAM.override(True)
            )
            with tpu_envs.SGLANG_TPU_PD_SWA_HOST_WINDOWS.override(64):
                self.assertEqual(flags.decode_swa_host_windows(), 64)
            with tpu_envs.SGLANG_TPU_PD_SWA_HOST_WINDOWS.override(0):
                self.assertEqual(flags.decode_swa_host_windows(), 0)
                self.assertFalse(flags.prefill_swa_path_end_backup_enabled())


if __name__ == "__main__":
    unittest.main()
