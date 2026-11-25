"""K3_PROGRAM on a server that does not select it: the plugin registers the
algorithm, and the K3_PROGRAM hooks on paths every server runs (the pool,
allocator and release hooks, the raiden pool grouping, the platform's pool
selectors' check) fall through, without importing the programs. Those import
jax and torch_tpu, which CPU runners do not install."""

import json
import os
import subprocess
import sys
import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=40, suite="base-a-test-cpu")

# Runs in a fresh interpreter with the TPU stack blocked, as on a CPU runner,
# and prints what it saw as JSON on its last line.
_PROBE = r"""
import json, sys
for name in ("jax", "jaxlib", "torch_tpu", "libtpu", "tpu_sync_torch"):
    sys.modules[name] = None

from types import SimpleNamespace
from unittest import mock

from sglang.srt.runtime_context import get_context
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang_tpu import hooks
from sglang_tpu.disaggregation.raiden import common as raiden_common
from sglang_tpu.hooks import k3_megakernel
from sglang_tpu.hooks import k3_program as k3_hooks
from sglang_tpu import tpu_platform

seen = {}
with mock.patch(
    "sglang.srt.platforms.current_platform", SimpleNamespace(is_tpu=lambda: True)
):
    hooks.register()
algo = SpeculativeAlgorithm.from_string("K3_PROGRAM")
seen["algorithm"] = type(algo).__module__ + "." + type(algo).__qualname__

seen["active_unpublished"] = k3_hooks.k3_program_active()
for name in ("EAGLE", "k3_program"):
    override = get_context().override_server_args(speculative_algorithm=name)
    override.install()
    try:
        seen["active_" + name] = k3_hooks.k3_program_active()
        seen["platform_" + name] = tpu_platform._k3_program_selected()
    finally:
        override.restore()
seen["owns_weights"] = k3_megakernel._owns_weights()
seen["selected_eagle"] = k3_hooks.k3_program_selected(
    SimpleNamespace(speculative_algorithm="EAGLE")
)
seen["selected_k3"] = k3_hooks.k3_program_selected(
    SimpleNamespace(speculative_algorithm="K3_PROGRAM")
)

pool = SimpleNamespace(full_kv_pool=object())
queue = SimpleNamespace(token_to_kv_pool_allocator=object(), tp_rank=0)
scheduler = SimpleNamespace(waiting_queue=[object()])
seen["fallthrough"] = [
    k3_hooks.k3_program_kv_layer_ids(lambda p: "kv", pool) == "kv",
    k3_hooks.k3_program_dp_allocator(lambda p: "alloc", None) == "alloc",
    k3_hooks.k3_program_dp_prealloc(lambda q, r: "pre", queue, None) == "pre",
    k3_hooks.k3_program_dp_transfer_rows(
        lambda q, d, k, p: "rows", queue, None, None, 1
    )
    == "rows",
    k3_hooks.k3_program_dp_admission(lambda q: 7, queue) == 7,
    k3_hooks.k3_program_dp_group_cap(lambda s, b: "batch", scheduler, None)
    == "batch",
    k3_hooks.k3_program_dp_release("released", SimpleNamespace(rid="r"))
    == "released",
    raiden_common._k3_program_groups(pool) is None,
]

seen["k3_modules"] = sorted(m for m in sys.modules if m.startswith("sglang_tpu.k3_programs"))
seen["jax_loaded"] = any(
    sys.modules.get(m) is not None for m in ("jax", "jaxlib", "torch_tpu")
)
try:  # The block holds: the programs themselves do need jax.
    import sglang_tpu.k3_programs.sglang_worker  # noqa: F401

    seen["worker_import"] = "imported"
except ImportError as e:
    seen["worker_import"] = type(e).__name__
print(json.dumps(seen))
"""


class TestK3ProgramImportGate(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # As on a CPU runner: no platform selected, no torch_tpu backend autoload.
        env = dict(os.environ, TORCH_DEVICE_BACKEND_AUTOLOAD="0", SGLANG_PLATFORM="")
        result = subprocess.run(
            [sys.executable, "-c", _PROBE],
            capture_output=True,
            text=True,
            env=env,
            timeout=600,
        )
        if result.returncode != 0:
            raise AssertionError(result.stderr[-4000:])
        cls.seen = json.loads(result.stdout.strip().splitlines()[-1])

    def test_registration_imports_no_program_module(self):
        self.assertEqual(
            self.seen["algorithm"],
            "sglang_tpu.k3_programs.sglang_algorithm.K3ProgramAlgo",
        )
        self.assertEqual(
            self.seen["k3_modules"],
            ["sglang_tpu.k3_programs", "sglang_tpu.k3_programs.sglang_algorithm"],
        )
        self.assertFalse(self.seen["jax_loaded"])
        self.assertEqual(self.seen["worker_import"], "ModuleNotFoundError")

    def test_k3_program_checks_read_the_published_name(self):
        self.assertFalse(self.seen["active_unpublished"])
        self.assertFalse(self.seen["active_EAGLE"])
        self.assertFalse(self.seen["platform_EAGLE"])
        self.assertTrue(self.seen["active_k3_program"])
        self.assertTrue(self.seen["platform_k3_program"])
        self.assertFalse(self.seen["owns_weights"])
        self.assertFalse(self.seen["selected_eagle"])
        self.assertTrue(self.seen["selected_k3"])

    def test_every_server_hooks_fall_through(self):
        self.assertEqual(self.seen["fallthrough"], [True] * 8)


if __name__ == "__main__":
    unittest.main()
