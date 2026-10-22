# SPDX-License-Identifier: Apache-2.0
"""Contract test for the out-of-tree TPU platform (wave 0 of the plugin migration).

``sglang_tpu.platform.TpuOOTPlatform`` declares ``PlatformEnum.OOT``, which turns
on every ``is_out_of_tree()`` gate in ``python/sglang/srt``. Those gates call
platform methods the in-tree TPU platform never had to answer, and the base
``SRTPlatform`` / ``DeviceMixin`` implementations of several are bare
``raise NotImplementedError``. The contract: for every platform method reachable
from an OOT gate, the TPU side of the class hierarchy has an opinion. "Not
supported on TPU" is an opinion; inheriting the base stub means nobody looked.

The gate inventory is recomputed from source on every run by
``tools/plugin/gate_audit.py``, so an upstream sync that adds a gate calling a
new platform method fails here. That alarm is the point of the file.

Runs on CPU; needs no TPU hardware and does not import ``torch_tpu``.
"""

import ast
import importlib.util
import os
import sys
import unittest
from typing import Optional
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")


_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir, os.pardir)
)

# The CPU gate installs python/ only; an installed sglang_tpu still wins.
if _REPO_ROOT not in sys.path:
    sys.path.append(_REPO_ROOT)


def _load_tool(filename: str, name: str):
    """Import a tools/plugin/ script without putting tools/ on sys.path."""
    path = os.path.join(_REPO_ROOT, "tools", "plugin", filename)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


gate_audit = _load_tool("gate_audit.py", "_tpu_gate_audit")
tpu_debt = _load_tool("check_tpu_branches.py", "_tpu_debt_ratchet")
inventory = _load_tool("inventory.py", "_tpu_inventory")


# Gate-reachable methods the TPU platform does not answer; the list only shrinks.
KNOWN_UNANSWERED = {
    "get_device_core_count": (
        "Defined on no platform class; its callers (triton, wave, batch-invariant "
        "ops) are unreachable from TPU. Belongs upstream."
    ),
}


def _defining_class(cls: type, name: str) -> Optional[type]:
    for klass in cls.__mro__:
        if name in klass.__dict__:
            return klass
    return None


class TestTpuOOTPlatformContract(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        from sglang.srt.platforms.device_mixin import PlatformEnum
        from sglang.srt.platforms.interface import SRTPlatform
        from sglang_tpu.platform import TpuOOTPlatform
        from sglang_tpu.tpu_platform import TpuSRTPlatform

        cls.PlatformEnum = PlatformEnum
        cls.SRTPlatform = SRTPlatform
        cls.TpuSRTPlatform = TpuSRTPlatform
        cls.TpuOOTPlatform = TpuOOTPlatform
        cls.platform = TpuOOTPlatform()
        # Anything resolved above these in the MRO is an unconsidered default.
        cls.tpu_owned = (TpuOOTPlatform, TpuSRTPlatform)

        cls.gates = gate_audit.audit_tree(_REPO_ROOT)
        cls.required = gate_audit.required_platform_methods(cls.gates)

    # -- identity ----------------------------------------------------------

    def test_declares_out_of_tree(self):
        self.assertEqual(self.TpuOOTPlatform._enum, self.PlatformEnum.OOT)
        self.assertTrue(self.platform.is_out_of_tree())

    def test_still_reports_as_tpu(self):
        """The core tree's is_tpu() sites must keep seeing True."""
        self.assertTrue(self.platform.is_tpu())
        self.assertFalse(self.platform.is_cuda())
        self.assertFalse(self.platform.is_cpu())

    def test_subclasses_the_in_tree_platform(self):
        """Every factory is inherited, not copied."""
        self.assertTrue(issubclass(self.TpuOOTPlatform, self.TpuSRTPlatform))

    # -- the gate contract -------------------------------------------------

    def test_gate_audit_finds_the_core_gates(self):
        self.assertGreater(len(self.gates), 0, "gate audit found nothing")
        self.assertGreater(len(self.required), 0, "no platform methods reached")

    def test_every_gate_reachable_method_has_a_tpu_opinion(self):
        unanswered = {}
        for method, sites in self.required.items():
            if method in KNOWN_UNANSWERED:
                continue
            owner = _defining_class(self.TpuOOTPlatform, method)
            if owner is None:
                unanswered[method] = ("not defined on any class in the MRO", sites)
            elif owner not in self.tpu_owned:
                unanswered[method] = (
                    f"resolves to {owner.__name__}, not the TPU platform",
                    sites,
                )

        if unanswered:
            lines = [
                "Platform methods reached from an is_out_of_tree() gate that the "
                "TPU platform does not answer:"
            ]
            for method, (why, sites) in sorted(unanswered.items()):
                lines.append(f"  {method}: {why}")
                for site in sites:
                    lines.append(f"      gate at {site}")
            lines.append(
                "Implement each on sglang_tpu.platform.TpuOOTPlatform, or add it "
                "to KNOWN_UNANSWERED in this file with the reason it cannot be."
            )
            self.fail("\n".join(lines))

    def test_known_unanswered_list_is_not_stale(self):
        now_answered = [
            method
            for method in KNOWN_UNANSWERED
            if _defining_class(self.TpuOOTPlatform, method) in self.tpu_owned
        ]
        self.assertEqual(
            now_answered,
            [],
            "these are implemented now; delete them from KNOWN_UNANSWERED",
        )

    def test_known_unanswered_entries_are_still_reachable(self):
        unreachable = [m for m in KNOWN_UNANSWERED if m not in self.required]
        self.assertEqual(
            unreachable,
            [],
            "no is_out_of_tree() gate reaches these; drop them from KNOWN_UNANSWERED",
        )

    # -- backend initialisation -------------------------------------------

    def test_init_backend_defers_outside_the_worker_bootstrap(self):
        """The import-time call from model_runner.py must not initialize the backend."""
        from sglang_tpu.platform import _WORKER_BOOTSTRAP_MARKER

        with patch.dict(os.environ, {"WORLD_SIZE": "1"}, clear=False):
            os.environ.pop(_WORKER_BOOTSTRAP_MARKER, None)
            with patch.object(self.TpuSRTPlatform, "init_backend") as in_tree_init:
                self.platform.init_backend()
            in_tree_init.assert_not_called()

    def test_init_backend_runs_inside_the_worker_bootstrap(self):
        """The worker bootstrap's call must do the real work."""
        from sglang_tpu.platform import _WORKER_BOOTSTRAP_MARKER

        with patch.dict(os.environ, {_WORKER_BOOTSTRAP_MARKER: "dp1_dprank0_tp1"}):
            with patch.object(self.TpuSRTPlatform, "init_backend") as in_tree_init:
                self.platform.init_backend()
            in_tree_init.assert_called_once()

    def test_worker_bootstrap_marker_is_set_by_patch_torch(self):
        """The marker is only meaningful while patch_torch sets it."""
        from sglang_tpu.platform import _WORKER_BOOTSTRAP_MARKER

        source = self._read_core("utils/patch_torch.py")
        self.assertIn(f'os.environ["{_WORKER_BOOTSTRAP_MARKER}"] = ', source)

    # -- DSA / sparse attention -------------------------------------------

    def test_dsa_kv_pool_cls_fails_with_a_tpu_specific_message(self):
        """Mode B has no DSA pool yet; the refusal must name TPU, not be the base stub."""
        owner = _defining_class(self.TpuOOTPlatform, "get_dsa_kv_pool_cls")
        self.assertIn(owner, self.tpu_owned)
        with self.assertRaises(NotImplementedError) as caught:
            self.platform.get_dsa_kv_pool_cls()
        self.assertIn("TPU", str(caught.exception))

    # -- speculative decoding ---------------------------------------------

    def test_supported_speculative_algorithms_are_real_algorithms(self):
        """Catches an upstream rename of a SpeculativeAlgorithm member."""
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
        from sglang_tpu.platform import _SUPPORTED_SPEC_ALGORITHMS

        known = {member.name for member in SpeculativeAlgorithm}
        self.assertTrue(
            _SUPPORTED_SPEC_ALGORITHMS <= known,
            f"unknown algorithms claimed: {sorted(_SUPPORTED_SPEC_ALGORITHMS - known)}",
        )

    def test_dspark_is_claimed_only_with_experimental_spec_worker(self):
        """DSPARK is served by TpuBlockSpecWorker when the experimental spec worker is enabled."""
        from sglang_tpu.envs import envs

        self.assertTrue(self.platform.supports_speculative_algorithm("DSPARK"))
        with envs.SGLANG_TPU_ENABLE_EXPERIMENTAL_SPEC_WORKER.override(False):
            self.assertFalse(self.platform.supports_speculative_algorithm("DSPARK"))

    def test_claimed_draft_backends_are_registered(self):
        from sglang_tpu.platform import _SUPPORTED_DRAFT_ATTENTION_BACKENDS

        missing = _SUPPORTED_DRAFT_ATTENTION_BACKENDS - (
            self._draft_backend_keys() | self._registered_draft_backends()
        )
        self.assertEqual(
            missing,
            set(),
            "claimed draft backends neither draft_utils nor the plugin's draft "
            f"registry provides: {sorted(missing)}",
        )

    def test_default_draft_backend_matches_the_core_default(self):
        """Must match the in-tree default in speculative_hook.py."""
        from sglang_tpu.platform import (
            _DEFAULT_DRAFT_ATTENTION_BACKEND,
            _SUPPORTED_DRAFT_ATTENTION_BACKENDS,
        )

        self.assertIn(
            _DEFAULT_DRAFT_ATTENTION_BACKEND, _SUPPORTED_DRAFT_ATTENTION_BACKENDS
        )
        self.assertEqual(
            self.platform.get_default_speculative_draft_attention_backend("DFLASH"),
            _DEFAULT_DRAFT_ATTENTION_BACKEND,
        )
        source = self._read_core("arg_groups/speculative_hook.py")
        self.assertIn(f'"{_DEFAULT_DRAFT_ATTENTION_BACKEND}"', source)

    def test_speculative_queries_reject_unsupported_inputs(self):
        self.assertFalse(self.platform.supports_speculative_algorithm("NGRAM"))
        self.assertTrue(self.platform.supports_speculative_algorithm("eagle3"))
        self.assertFalse(
            self.platform.supports_speculative_draft_attention_backend(
                "DFLASH", "flashinfer"
            )
        )
        self.assertFalse(
            self.platform.supports_speculative_draft_attention_backend(
                "NGRAM", "pallas_tpu_rpa_v3"
            )
        )
        with self.assertRaises(ValueError):
            self.platform.get_default_speculative_draft_attention_backend("NGRAM")

    def test_speculative_cache_locs_fn_is_not_overridden(self):
        """Core still branches on _is_tpu for this; one implementation only."""
        self.assertIsNone(self.platform.get_speculative_cache_locs_fn())

    # -- memory ------------------------------------------------------------

    def test_device_memory_capacity_matches_mode_a(self):
        """Mode B reported HBM here; above 60 GiB that floors the mamba/GDN
        pre-capture activation reserve at 10 GiB, shrinking the KV pool."""
        from sglang.srt.utils import common

        for platform in (self.TpuSRTPlatform(), self.platform):
            with (
                self.subTest(platform=type(platform).__name__),
                patch.object(common, "current_platform", platform),
                patch.object(
                    type(platform), "get_device_total_memory", return_value=96 << 30
                ),
            ):
                self.assertIsNone(common.get_device_memory_capacity("tpu"))

    # -- profiler ----------------------------------------------------------

    def test_profiler_activity_str_is_the_xprof_name_not_a_torch_activity(self):
        """profiler_manager maps the name only if torch.profiler has it; "TPU" is xprof's."""
        import torch

        name = self.platform.get_torch_profiler_activity_str()
        self.assertEqual(name, "TPU")
        self.assertFalse(hasattr(torch.profiler.ProfilerActivity, name))
        self.assertIsInstance(
            self.platform.get_torch_profiler_activity(),
            torch.profiler.ProfilerActivity,
        )

    # -- activation and packaging -----------------------------------------

    def test_activate_declines_unless_selected_by_name(self):
        from sglang_tpu import activate

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SGLANG_PLATFORM", None)
            self.assertIsNone(activate())
        with patch.dict(os.environ, {"SGLANG_PLATFORM": "tpu"}):
            self.assertIsNone(activate())

    def test_advertised_class_path_resolves_to_this_class(self):
        import pkgutil

        from sglang_tpu import PLATFORM_CLASS_PATH

        self.assertIs(pkgutil.resolve_name(PLATFORM_CLASS_PATH), self.TpuOOTPlatform)

    def test_entry_point_name_matches_the_packaging_metadata(self):
        from sglang_tpu import ENTRY_POINT_NAME

        with open(os.path.join(_REPO_ROOT, "pyproject.toml"), encoding="utf-8") as f:
            content = f.read()
        self.assertIn(f'\n{ENTRY_POINT_NAME} = "sglang_tpu:activate"', content)

    def test_in_tree_entry_point_still_serves_mode_a(self):
        with open(
            os.path.join(_REPO_ROOT, "python", "pyproject.toml"), encoding="utf-8"
        ) as f:
            content = f.read()
        self.assertIn('tpu = "sglang.srt.hardware_backend.tpu:activate"', content)

    def test_scheduler_initialization_hook_is_declared(self):
        from sglang.srt.plugins.hook_registry import HookType
        from sglang_tpu.hooks import HOOKS
        from sglang_tpu.hooks.attn_residual import (
            aggregate,
            aggregate_fused_add,
            prepare_attn_residual_kernel,
            register_attn_residual_ops,
        )
        from sglang_tpu.hooks.attn_tp import device_supports_input_scattered
        from sglang_tpu.hooks.moe_topk import (
            biased_grouped_topk,
            biased_topk,
            register_router_topk,
        )
        from sglang_tpu.hooks.pipeline_parallel import (
            check_pipeline_parallel_compat,
            skip_spec_op_precompile_without_draft,
        )
        from sglang_tpu.hooks.scheduler import initialize_dense_moe_config
        from sglang_tpu.hooks.static_fp8 import bind_kimi_k3_static_fp8
        from sglang_tpu.hooks.twolevel_allreduce import (
            all_reduce as twolevel_all_reduce,
        )
        from sglang_tpu.hooks.twolevel_allreduce import (
            merged_moe_all_reduce as twolevel_merged_moe_all_reduce,
        )

        self.assertIn(
            (
                "sglang.srt.layers.layer_boundary.adapters.attention."
                "_device_supports_input_scattered",
                device_supports_input_scattered,
                HookType.AROUND,
            ),
            HOOKS,
        )
        self.assertIn(
            (
                "sglang.srt.managers.scheduler.Scheduler.init_moe_gemm_config",
                initialize_dense_moe_config,
                HookType.AFTER,
            ),
            HOOKS,
        )

        self.assertIn(
            (
                "sglang.srt.arg_groups.validation_hook.check_pipeline_parallel_compat",
                check_pipeline_parallel_compat,
                HookType.AROUND,
            ),
            HOOKS,
        )
        self.assertIn(
            (
                "sglang.srt.managers.scheduler.Scheduler.precompile_tpu_spec_ops",
                skip_spec_op_precompile_without_draft,
                HookType.AROUND,
            ),
            HOOKS,
        )
        self.assertIn(
            (
                "sglang.srt.distributed.parallel_state.GroupCoordinator.all_reduce",
                twolevel_all_reduce,
                HookType.AROUND,
            ),
            HOOKS,
        )
        self.assertIn(
            (
                "sglang.srt.models.kimi_k3.merged_moe_all_reduce",
                twolevel_merged_moe_all_reduce,
                HookType.AROUND,
            ),
            HOOKS,
        )
        self.assertIn(
            (
                "sglang.srt.layers.moe.topk.TopK.__init__",
                register_router_topk,
                HookType.AFTER,
            ),
            HOOKS,
        )
        self.assertIn(
            (
                "sglang.srt.layers.moe.topk.biased_topk_impl",
                biased_topk,
                HookType.AROUND,
            ),
            HOOKS,
        )
        self.assertIn(
            (
                "sglang.srt.layers.moe.topk.biased_grouped_topk_impl",
                biased_grouped_topk,
                HookType.AROUND,
            ),
            HOOKS,
        )
        self.assertIn(
            (
                "sglang.srt.models.kimi_k3.KimiK3LinearForCausalLM.post_load_weights",
                bind_kimi_k3_static_fp8,
                HookType.AFTER,
            ),
            HOOKS,
        )
        self.assertIn(
            (
                "sglang.srt.models.kimi_k3.KimiK3LinearModel.__init__",
                register_attn_residual_ops,
                HookType.AFTER,
            ),
            HOOKS,
        )
        self.assertIn(
            (
                "sglang.srt.models.kimi_k3.KimiK3LinearForCausalLM.post_load_weights",
                prepare_attn_residual_kernel,
                HookType.AFTER,
            ),
            HOOKS,
        )
        self.assertIn(
            (
                "sglang.srt.layers.attn_residual._aggregate",
                aggregate,
                HookType.AROUND,
            ),
            HOOKS,
        )
        self.assertIn(
            (
                "sglang.srt.layers.attn_residual._aggregate_fused_add",
                aggregate_fused_add,
                HookType.AROUND,
            ),
            HOOKS,
        )

    # -- helpers -----------------------------------------------------------

    def _read_core(self, relpath: str) -> str:
        path = os.path.join(_REPO_ROOT, "python", "sglang", "srt", relpath)
        with open(path, "r", encoding="utf-8") as handle:
            return handle.read()

    def _registered_draft_backends(self) -> set:
        """Names sglang_tpu registers with register_draft_attention_backend."""
        path = os.path.join(_REPO_ROOT, "sglang_tpu", "tpu_platform.py")
        with open(path, "r", encoding="utf-8") as handle:
            tree = ast.parse(handle.read())
        names = set()
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "register_draft_attention_backend"
                and node.args
                and isinstance(node.args[0], ast.Constant)
            ):
                names.add(node.args[0].value)
        return names

    def _draft_backend_keys(self) -> set:
        """Dispatch-dict keys in speculative/draft_utils.py, parsed not imported."""
        tree = ast.parse(self._read_core("speculative/draft_utils.py"))
        keys = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Dict):
                for key in node.keys:
                    if isinstance(key, ast.Constant) and isinstance(key.value, str):
                        keys.add(key.value)
        return keys


def _declared_entry_points(pyproject_relpath: str, dist_name: str) -> list:
    """Entry points a pyproject declares, as installed-metadata lookalikes."""
    import pkgutil
    import re
    from types import SimpleNamespace

    section = re.compile(r'^\[project\.entry-points\."([^"]+)"\]$')
    dist = SimpleNamespace(name=dist_name)
    eps, group = [], None
    with open(os.path.join(_REPO_ROOT, pyproject_relpath), encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if line.startswith("["):
                match = section.match(line)
                group = match.group(1) if match else None
            elif group and "=" in line and not line.startswith("#"):
                name, value = (part.strip().strip('"') for part in line.split("=", 1))
                eps.append(
                    SimpleNamespace(
                        name=name,
                        value=value,
                        group=group,
                        dist=dist,
                        load=lambda value=value: pkgutil.resolve_name(value),
                    )
                )
    return eps


class TestTpuPluginHooksLoad(CustomTestCase):
    """The plugin's hooks reach HookRegistry under every way of selecting TPU."""

    def _loaded_general_plugins(self, core_pyproject: str, selected: Optional[str]):
        from sglang.srt import plugins

        installed = _declared_entry_points(
            core_pyproject, "sglang"
        ) + _declared_entry_points("pyproject.toml", "sglang-tpu")

        def entry_points(group):
            return [ep for ep in installed if ep.group == group]

        env = {k: v for k, v in os.environ.items() if k != "SGLANG_PLATFORM"}
        if selected is not None:
            env["SGLANG_PLATFORM"] = selected
        with (
            patch.dict(os.environ, env, clear=True),
            patch.object(plugins, "entry_points", entry_points),
        ):
            loaded = plugins.load_plugins_by_group(
                plugins.GENERAL_PLUGINS_GROUP,
                excluded_dists=plugins._get_excluded_dists(),
            )
        return [func for func, _dist in loaded.values()]

    def test_hooks_load_in_both_modes_and_with_the_platform_unset(self):
        """Selecting the in-tree tpu platform skips the sglang-tpu dist entirely."""
        from sglang_tpu.hooks import register

        for core_pyproject in ("python/pyproject.toml", "python/pyproject_tpu.toml"):
            for selected in ("tpu", "tpu_oot", None):
                with self.subTest(pyproject=core_pyproject, platform=selected):
                    self.assertIn(
                        register,
                        self._loaded_general_plugins(core_pyproject, selected),
                    )

    def test_register_is_idempotent_and_tpu_only(self):
        """Both entry points run when the platform is unset; non-TPU hosts load them too."""
        from types import SimpleNamespace

        from sglang.srt.plugins.hook_registry import HookRegistry, HookType
        from sglang_tpu import hooks

        target = "sglang.srt.utils.common.get_device_name"
        fake_hook = (target, lambda result, *a, **k: result, HookType.AFTER)
        for is_tpu, expected in ((True, 1), (False, 0)):
            with self.subTest(is_tpu=is_tpu):
                HookRegistry.reset()
                self.addCleanup(HookRegistry.reset)
                platform = SimpleNamespace(is_tpu=lambda is_tpu=is_tpu: is_tpu)
                with (
                    patch.object(hooks, "HOOKS", [fake_hook]),
                    patch.object(hooks, "_registered", False),
                    patch("sglang.srt.platforms.current_platform", platform),
                ):
                    hooks.register()
                    hooks.register()
                self.assertEqual(len(HookRegistry._hooks.get(target, [])), expected)


class TestTpuDebtRatchet(CustomTestCase):
    """The ratchet must catch an increase; its failure mode is a silent pass."""

    def test_tpu_is_recognised_only_as_a_whole_path_word(self):
        self.assertTrue(
            tpu_debt.names_tpu("python/sglang/srt/layers/moe/tpu_moe_utils.py")
        )
        self.assertTrue(
            tpu_debt.names_tpu("python/sglang/multimodal_gen/runtime/platforms/tpu.py")
        )
        self.assertFalse(
            tpu_debt.names_tpu("python/sglang/srt/managers/auxiliary_output.py")
        )

    def test_shims_are_not_debt(self):
        """Shims import the plugin by design."""
        self.assertTrue(
            tpu_debt._is_shim("python/sglang/srt/hardware_backend/tpu/kv_pool.py")
        )
        self.assertTrue(
            tpu_debt._is_shim("python/sglang/srt/model_executor/tpu_graph_runner.py")
        )
        self.assertFalse(tpu_debt._is_shim("python/sglang/srt/speculative/x.py"))

    def test_imports_are_not_counted_as_branches(self):
        source = "from sglang.srt.utils import is_tpu\nimport sglang.srt.utils as u\n"
        self.assertEqual(self._count(tpu_debt.count_tpu_branches, source), 0)

    def test_calls_and_attribute_reads_are_counted(self):
        source = (
            "from sglang.srt.utils import is_tpu\n"
            "def f(p):\n"
            "    if is_tpu():\n"
            "        return p.is_tpu()\n"
            "    return _is_tpu\n"
        )
        self.assertEqual(self._count(tpu_debt.count_tpu_branches, source), 3)

    def test_plugin_imports_are_counted_at_any_scope(self):
        source = (
            "import sglang_tpu\n"
            "from sglang_tpu.kv_pool import X\n"
            "from sglang_tpuish import Y\n"
            "def f():\n"
            "    from sglang_tpu import patches\n"
        )
        self.assertEqual(self._count(tpu_debt.count_plugin_imports, source), 3)

    def test_an_added_branch_is_a_violation(self):
        baseline = tpu_debt.Debt(branches={"python/sglang/srt/a.py": 2})
        current = tpu_debt.Debt(branches={"python/sglang/srt/a.py": 3})
        violations, _ = tpu_debt.compare(current, baseline, partial=False)
        self.assertEqual(len(violations), 1)
        self.assertIn("baseline 2", violations[0])

    def test_an_added_plugin_import_is_a_violation(self):
        baseline = tpu_debt.Debt(imports={"python/sglang/srt/a.py": 1})
        current = tpu_debt.Debt(imports={"python/sglang/srt/a.py": 2})
        violations, _ = tpu_debt.compare(current, baseline, partial=False)
        self.assertEqual(len(violations), 1)
        self.assertIn("sglang_tpu", violations[0])

    def test_a_removed_branch_is_an_improvement_not_a_violation(self):
        baseline = tpu_debt.Debt(branches={"python/sglang/srt/a.py": 2})
        current = tpu_debt.Debt(branches={"python/sglang/srt/a.py": 1})
        violations, improvements = tpu_debt.compare(current, baseline, partial=False)
        self.assertEqual(violations, [])
        self.assertEqual(len(improvements), 1)

    def test_a_new_tpu_module_is_a_violation_even_under_files_mode(self):
        current = tpu_debt.Debt(modules=["python/sglang/srt/layers/tpu_thing.py"])
        for partial in (False, True):
            violations, _ = tpu_debt.compare(current, tpu_debt.Debt(), partial=partial)
            self.assertEqual(len(violations), 1, f"partial={partial}")

    def test_a_missing_file_is_not_read_as_progress_under_files_mode(self):
        baseline = tpu_debt.Debt(
            branches={"python/sglang/srt/a.py": 2},
            imports={"python/sglang/srt/a.py": 1},
            modules=["python/sglang/srt/b/tpu.py"],
        )
        _, improvements = tpu_debt.compare(tpu_debt.Debt(), baseline, partial=True)
        self.assertEqual(improvements, [])

    def test_committed_baseline_matches_the_tree(self):
        """Catches a hand-edited or stale baseline."""
        current = tpu_debt.collect(_REPO_ROOT)
        baseline = tpu_debt.load_baseline(_REPO_ROOT)
        violations, _ = tpu_debt.compare(current, baseline, partial=False)
        self.assertEqual(
            violations,
            [],
            "run `python tools/plugin/check_tpu_branches.py --write-baseline`",
        )

    def _count(self, counter, source: str) -> int:
        import tempfile

        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as handle:
            handle.write(source)
            path = handle.name
        try:
            return counter(path)
        finally:
            os.unlink(path)


# Attributes a patch creates on a core class rather than replaces.
ADDED_BY_PATCHES = {
    "sglang.srt.managers.overlap_utils.FutureMap.tpu_direct_handoff",
    "sglang.srt.managers.overlap_utils.FutureMap.tpu_published_seq_lens",
    "sglang.srt.mem_cache.unified_radix_cache.UnifiedRadixCache._write_back_copy_ahead",
}


class TestTpuPatchInventory(CustomTestCase):
    def test_every_patched_core_attribute_exists(self):
        """After an upstream rename, a setattr patch adds a dead attribute and the
        renamed code runs unpatched, silently."""
        import pkgutil

        targets = [
            (patch.name, target)
            for patch in inventory.collect_patches(_REPO_ROOT)
            for target in patch.targets
            if target.startswith("sglang.")
        ]
        self.assertGreater(len(targets), 0)
        missing = []
        for name, target in targets:
            try:
                pkgutil.resolve_name(target)
                exists = True
            except (ImportError, AttributeError):
                exists = False
            if exists == (target in ADDED_BY_PATCHES):
                missing.append(f"{name}: {target}")
        self.assertEqual(
            missing,
            [],
            "Retarget a patch whose core attribute is gone; list one the patch adds "
            "on purpose in ADDED_BY_PATCHES, and drop it once core defines it.",
        )


if __name__ == "__main__":
    unittest.main()
