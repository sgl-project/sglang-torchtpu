"""Unit tests for TPU env overrides.

Covers the LIBTPU_INIT_ARGS override contract: default flags are appended, the
VMEM prefetch serialization applies only to small explicit KV pools, and a
caller-supplied value for the same flag name — even a different value —
is never clobbered or duplicated. Also covers the TPU_PREMAPPED_BUFFER_SIZE
default that the launcher writes for spawned workers.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import os
import subprocess
import sys
import textwrap
import unittest
from unittest.mock import patch

from sglang.test.test_utils import CustomTestCase
from sglang_tpu import env_override
from sglang_tpu.env_override import (
    _DECODE_CP_SPARSE_CORE_OVERRIDES,
    _DEFAULT_TPU_PREMAPPED_BUFFER_SIZE,
    _TPU_PREMAPPED_BUFFER_SIZE,
    _VMEM_PREFETCH_MIN_LAYERS,
    _VMEM_PREFETCH_OVERRIDES,
    _VMEM_PREFETCH_SAFE_MIN_TOKENS,
    apply_decode_cp_sparse_core_override,
    apply_small_kv_pool_vmem_prefetch_override,
    apply_tpu_env_overrides,
    premapped_buffer_size_seen_by_torch_tpu,
)


class TestLibtpuInitArgsOverrides(CustomTestCase):
    def _apply(
        self,
        initial: str | None,
        max_total_tokens: int | None,
        num_hidden_layers: int = _VMEM_PREFETCH_MIN_LAYERS + 1,
    ) -> str:
        env = {} if initial is None else {"LIBTPU_INIT_ARGS": initial}
        with patch.dict(os.environ, env, clear=False):
            if initial is None:
                os.environ.pop("LIBTPU_INIT_ARGS", None)
            apply_tpu_env_overrides()
            apply_small_kv_pool_vmem_prefetch_override(
                max_total_tokens=max_total_tokens,
                num_hidden_layers=num_hidden_layers,
            )
            return os.environ.get("LIBTPU_INIT_ARGS", "")

    def test_small_pool_serializes_vmem_prefetches(self):
        # The device deadlocks large programs at small KV pools when VMEM
        # prefetches overlap; a small explicit pool must get the serializing
        # flags.
        args = self._apply(None, _VMEM_PREFETCH_SAFE_MIN_TOKENS - 1).split()
        for flag in _VMEM_PREFETCH_OVERRIDES:
            self.assertIn(flag, args)

    def test_large_or_unset_pool_keeps_default_schedule(self):
        # Unaffected deployments must keep the default prefetch schedule
        # (bit-identical numerics): no serializing flags at the boundary
        # value or when no explicit pool size is requested.
        for max_total_tokens in (None, _VMEM_PREFETCH_SAFE_MIN_TOKENS):
            args = self._apply(None, max_total_tokens).split()
            for flag in _VMEM_PREFETCH_OVERRIDES:
                self.assertNotIn(flag, args)

    def test_small_model_keeps_default_schedule(self):
        # Models below the measured layer threshold never deadlocked at any
        # pool size; capping their prefetches broke an unrelated small-pool
        # CI configuration, so the containment must not touch them.
        args = self._apply(
            None,
            _VMEM_PREFETCH_SAFE_MIN_TOKENS - 1,
            num_hidden_layers=_VMEM_PREFETCH_MIN_LAYERS - 1,
        ).split()
        for flag in _VMEM_PREFETCH_OVERRIDES:
            self.assertNotIn(flag, args)

    def test_user_value_for_same_flag_name_wins(self):
        # A user-tuned value must be preserved, and the default for the same
        # flag must not be appended alongside it (libtpu would see both).
        user_flag = "--xla_gf_vmem_max_outstanding_prefetches=4"
        args = self._apply(user_flag, _VMEM_PREFETCH_SAFE_MIN_TOKENS - 1).split()
        self.assertIn(user_flag, args)
        self.assertNotIn("--xla_gf_vmem_max_outstanding_prefetches=1", args)

    def test_unrelated_user_flags_are_preserved(self):
        user_flag = "--xla_tpu_enable_sparse_core_collective_offload_all_gather=false"
        args = self._apply(user_flag, _VMEM_PREFETCH_SAFE_MIN_TOKENS - 1).split()
        self.assertIn(user_flag, args)
        for flag in _VMEM_PREFETCH_OVERRIDES:
            self.assertIn(flag, args)

    def test_decode_cp_turns_off_sparse_core_offloads(self):
        # Under DCP the SparseCore all-gather offload returns a wrong first
        # token after long prompts; without DCP the libtpu defaults stay.
        for dcp_size, expected in ((2, True), (1, False)):
            with self.subTest(dcp_size=dcp_size):
                with patch.dict(os.environ, {"LIBTPU_INIT_ARGS": ""}, clear=False):
                    apply_decode_cp_sparse_core_override(dcp_size=dcp_size)
                    args = os.environ["LIBTPU_INIT_ARGS"].split()
                for flag in _DECODE_CP_SPARSE_CORE_OVERRIDES:
                    self.assertEqual(flag in args, expected)


class TestPremappedBufferSizeDefault(CustomTestCase):
    """The launcher writes the premapped-pool default so spawned workers
    have it when ``import torch`` loads torch_tpu (issue #1394)."""

    def setUp(self):
        # Writing the default records this process's PID; keep it per-test.
        # (start/addCleanup rather than enterContext, which needs Python 3.11.)
        patcher = patch.object(env_override, "_premapped_default_writer_pid", None)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _apply(self, initial: str | None) -> tuple[str | None, str | None]:
        """Return the env value after the override and what torch_tpu saw."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(_TPU_PREMAPPED_BUFFER_SIZE, None)
            if initial is not None:
                os.environ[_TPU_PREMAPPED_BUFFER_SIZE] = initial
            apply_tpu_env_overrides()
            return (
                os.environ.get(_TPU_PREMAPPED_BUFFER_SIZE),
                premapped_buffer_size_seen_by_torch_tpu(),
            )

    def test_unset_gets_default(self):
        env_value, _ = self._apply(None)
        self.assertEqual(env_value, str(_DEFAULT_TPU_PREMAPPED_BUFFER_SIZE))

    def test_user_value_wins(self):
        # 0 is a valid user choice: it turns the pool off on purpose.
        for value in ("17179869184", "0"):
            with self.subTest(value=value):
                self.assertEqual(self._apply(value), (value, value))

    def test_default_written_after_torch_import_is_not_seen_in_process(self):
        # torch is already imported here, as in any sglang process, so
        # torch_tpu read the variable as unset. Only spawned children get the
        # default; this process must not report it.
        self.assertIn("torch", sys.modules)
        default = str(_DEFAULT_TPU_PREMAPPED_BUFFER_SIZE)
        with self.subTest("writer process"):
            self.assertEqual(self._apply(None), (default, None))
        with self.subTest("other process"):
            other_pid = os.getpid() + 1
            with (
                patch.object(env_override, "_premapped_default_writer_pid", other_pid),
                patch.dict(os.environ, {_TPU_PREMAPPED_BUFFER_SIZE: default}),
            ):
                self.assertEqual(premapped_buffer_size_seen_by_torch_tpu(), default)

    def test_spawned_child_overrides_torch_tpus_zero(self):
        # When the variable is unset, torch_tpu's import writes "0" with C
        # setenv. os.environ never sees that write, but child processes
        # inherit it. The launcher's default must replace the "0" for children
        # spawned after the call. Run in a subprocess so the C-level write does
        # not reach this test process.
        script = textwrap.dedent(
            f"""
            import ctypes, os, subprocess, sys
            ctypes.CDLL(None).setenv(b"{_TPU_PREMAPPED_BUFFER_SIZE}", b"0", 1)
            assert "{_TPU_PREMAPPED_BUFFER_SIZE}" not in os.environ
            from sglang_tpu.env_override import apply_tpu_env_overrides
            apply_tpu_env_overrides()
            child = subprocess.run(
                [sys.executable, "-c",
                 "import os; print(os.environ['{_TPU_PREMAPPED_BUFFER_SIZE}'])"],
                capture_output=True, text=True, check=True,
            )
            print(child.stdout.strip())
            """
        )
        env = {k: v for k, v in os.environ.items() if k != _TPU_PREMAPPED_BUFFER_SIZE}
        out = subprocess.run(
            [sys.executable, "-c", script],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(
            out.stdout.strip().splitlines()[-1],
            str(_DEFAULT_TPU_PREMAPPED_BUFFER_SIZE),
        )


if __name__ == "__main__":
    unittest.main()
