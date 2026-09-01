# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the shared-file changes that let torch_tpu -- a lazy,
one-device-per-process backend -- serve. CPU-runnable unless marked."""

import ast
import os
import pathlib
import subprocess
import sys
import textwrap
import unittest
from unittest.mock import MagicMock, patch

import torch

from sglang.test.test_utils import CustomTestCase

_HAS_TORCH_TPU = hasattr(torch, "tpu")

_LAYERNORM = "sglang.multimodal_gen.runtime.layers.layernorm"
_MANAGER = "sglang.multimodal_gen.runtime.disaggregation.transport.manager"
_VOCAB_EMBEDDING = "sglang.multimodal_gen.runtime.layers.vocab_parallel_embedding"

# These three name a literal backend, so the override alone reaches them.
_HOST_SAFE_COMPILE_SITES = (
    (_LAYERNORM, "LayerNorm.forward_native"),
    (_LAYERNORM, "_ScaleResidualNormScaleShift.forward_native"),
    (_LAYERNORM, "_NormScaleShift.forward_native"),
)
# This one names the platform backend, which torch.compile resolves eagerly.
_TPU_BACKEND_COMPILE_SITES = ((_VOCAB_EMBEDDING, "get_masked_input_and_mask"),)


def _run_probe(source: str, **env_overrides: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source)],
        capture_output=True,
        text=True,
        timeout=600,
        env={**os.environ, **env_overrides},
    )


class TestAnnotationsDoNotTouchTheAccelerator(CustomTestCase):
    """A signature annotation is evaluated at `def` time, so
    `torch.get_device_module().X` there initializes a lazy backend on import."""

    def test_transfer_manager_signatures_stay_device_free(self):
        completed = _run_probe(f"""
            import functools, importlib, sys, typing
            import torch

            importlib.import_module({_MANAGER!r})
            del sys.modules[{_MANAGER!r}]

            def _forbidden(*args, **kwargs):
                raise AssertionError("torch.get_device_module() resolved")

            torch.get_device_module = _forbidden
            module = importlib.import_module({_MANAGER!r})
            for qualname in (
                "DiffusionTransferManager.load_tensors_async",
                "DiffusionTransferManager.stage_tensors_async",
            ):
                target = functools.reduce(getattr, qualname.split("."), module)
                typing.get_type_hints(target)
            """)
        self.assertEqual(
            completed.returncode, 0, f"subprocess stderr:\n{completed.stderr[-4000:]}"
        )


class TestVariableAnnotations(CustomTestCase):
    """A ratchet on the annotation form. Inert inside a method body -- Python
    never evaluates it -- but at class or module scope it runs at import,
    which is the bug fixed in `manager.py`."""

    @staticmethod
    def _annotations(tree: ast.AST):
        # ast.walk visits ast.arg directly, so every parameter kind is covered.
        for node in ast.walk(tree):
            if isinstance(node, (ast.AnnAssign, ast.arg)):
                if node.annotation is not None:
                    yield node.annotation
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.returns is not None:
                    yield node.returns

    def test_runtime_sources_are_free_of_the_annotation_form(self):
        # `runtime` is a namespace package, so its own `__file__` is None.
        from sglang.multimodal_gen.runtime.layers import layernorm

        root = pathlib.Path(layernorm.__file__).parents[1]
        offenders = []
        scanned = 0
        for path in sorted(root.rglob("*.py")):
            scanned += 1
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for annotation in self._annotations(tree):
                if any(
                    isinstance(n, ast.Attribute) and n.attr == "get_device_module"
                    for n in ast.walk(annotation)
                ):
                    offenders.append(
                        f"{path.relative_to(root)}:{annotation.lineno}: "
                        f"{ast.unparse(annotation)}"
                    )
        self.assertEqual(offenders, [], "\n".join(offenders))
        self.assertGreater(scanned, 100, f"only {scanned} files under {root}")


class TestCompileSitesAreDisabledOnTpu(CustomTestCase):
    """`backend="tpu"` cannot lower a symbolic-shape graph -- it raises
    `Cannot call storage_offset() on tensor with symbolic sizes/strides` once
    torch.compile generalizes past the first shape."""

    def _assert_left_undecorated(self, sites) -> None:
        completed = _run_probe(
            f"""
            import functools, importlib

            for module_name, qualname in {sites!r}:
                module = importlib.import_module(module_name)
                fn = functools.reduce(getattr, qualname.split("."), module)
                # torch.compile(disable=True) returns the function untouched.
                assert not hasattr(fn, "_torchdynamo_orig_callable"), qualname
            """,
            SGLANG_DIFFUSION_PLATFORM_OVERRIDE="tpu",
        )
        self.assertEqual(
            completed.returncode, 0, f"subprocess stderr:\n{completed.stderr[-4000:]}"
        )

    def test_layernorm_sites_are_left_undecorated(self):
        self._assert_left_undecorated(_HOST_SAFE_COMPILE_SITES)

    @unittest.skipUnless(_HAS_TORCH_TPU, "resolving backend='tpu' needs torch_tpu")
    def test_the_platform_backend_site_is_left_undecorated(self):
        self._assert_left_undecorated(_TPU_BACKEND_COMPILE_SITES)


class TestLayernormDoesNotImportTritonNormOnTpu(CustomTestCase):
    """TPU has no triton runtime, so layernorm must not import triton-backed names."""

    def test_layernorm_does_not_import_norm_triton_on_tpu(self):
        completed = _run_probe(
            """
            import sys
            from sglang.multimodal_gen.runtime.layers import layernorm
            assert "sglang.kernels.ops.diffusion.norm.norm_triton" not in sys.modules, (
                "norm_triton was imported on TPU"
            )
            """,
            SGLANG_DIFFUSION_PLATFORM_OVERRIDE="tpu",
        )
        self.assertEqual(
            completed.returncode, 0, f"subprocess stderr:\n{completed.stderr[-2000:]}"
        )


class TestCustomOpDispatchesTpu(CustomTestCase):
    """`forward_tpu` is defined on CustomOp but was unreachable: dispatch_forward
    had no TPU branch, so an override would have been silently ignored."""

    def _dispatch_with_tpu_platform(self, op_cls):
        from sglang.multimodal_gen.runtime.layers import custom_op

        platform = MagicMock()
        for name in ("is_hip", "is_npu", "is_xpu", "is_musa"):
            getattr(platform, name).return_value = False
        platform.is_tpu.return_value = True
        with (
            patch.object(custom_op, "_is_cuda", False),
            patch.object(custom_op, "current_platform", platform),
        ):
            return op_cls()._forward_method()

    def test_subclass_override_is_reached(self):
        from sglang.multimodal_gen.runtime.layers.custom_op import CustomOp

        class OpWithTpuKernel(CustomOp):
            def forward_native(self):
                return "native"

            def forward_tpu(self):
                return "tpu"

        self.assertEqual(self._dispatch_with_tpu_platform(OpWithTpuKernel), "tpu")

    def test_default_still_falls_back_to_native(self):
        from sglang.multimodal_gen.runtime.layers.custom_op import CustomOp

        class OpWithoutTpuKernel(CustomOp):
            def forward_native(self):
                return "native"

        self.assertEqual(self._dispatch_with_tpu_platform(OpWithoutTpuKernel), "native")


class TestGlooDefaultGroupIsNotDeviceBound(CustomTestCase):
    """A tpu-bound default group makes gloo look up a backend it has not got."""

    def _init_with_platform(self, **predicates):
        from unittest.mock import MagicMock, patch

        from sglang.multimodal_gen.runtime import platforms
        from sglang.multimodal_gen.runtime.distributed import parallel_state

        platform = MagicMock()
        for name in ("is_mps", "is_musa", "is_npu", "is_cpu", "is_xpu", "is_tpu"):
            getattr(platform, name).return_value = predicates.get(name, False)
        platform.device_name = "test"
        world = MagicMock(world_size=1)
        with (
            patch.object(platforms, "current_platform", platform, create=True),
            patch("torch.distributed.is_initialized", return_value=False),
            patch("torch.distributed.init_process_group") as init_pg,
            patch("torch.distributed.get_world_size", return_value=1),
            patch.object(parallel_state, "_WORLD", world),
            patch.object(parallel_state, "_sync_srt_world_group"),
        ):
            parallel_state.init_distributed_environment(
                world_size=1,
                rank=0,
                distributed_init_method="tcp://127.0.0.1:1",
                local_rank=0,
                backend="gloo",
                device_id=torch.device("cpu"),
            )
        init_pg.assert_called_once()
        return init_pg.call_args.kwargs

    def test_tpu_omits_device_id(self):
        self.assertNotIn("device_id", self._init_with_platform(is_tpu=True))

    def test_cuda_like_platform_still_passes_device_id(self):
        self.assertIn("device_id", self._init_with_platform())


@unittest.skipUnless(_HAS_TORCH_TPU, "torch_tpu not installed")
class TestTpuHostBehaviour(CustomTestCase):
    def test_launcher_imports_do_not_initialize_pjrt(self):
        # The parent process must never own a device: PJRT is one client per
        # process and the spawned worker needs the libtpu lockfile.
        completed = _run_probe("""
            import torch
            import sglang.multimodal_gen.runtime.managers.gpu_worker
            import sglang.multimodal_gen.runtime.managers.scheduler
            from sglang.multimodal_gen.runtime.server_args import ServerArgs
            assert not torch.tpu.is_initialized(), 'PJRT initialized at import'
            """)
        self.assertEqual(
            completed.returncode, 0, f"subprocess stderr:\n{completed.stderr[-2000:]}"
        )


if __name__ == "__main__":
    unittest.main()
