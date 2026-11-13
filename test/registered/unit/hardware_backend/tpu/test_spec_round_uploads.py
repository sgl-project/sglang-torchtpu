"""Uploads in the TPU spec decode round must not block the host.

On torch_tpu a blocking H2D copy (``x.to(device)``, ``torch.tensor(v,
device=...)``, ``dst.copy_(src)``) returns only after every program already
queued on the TPU has finished. Each function listed below runs while the
target verify or the draft is still queued, so one blocking upload in it stalls
the host for that whole program and the TPU then idles while the host catches
up. On GLM-5.3-Flash MTP (TP8, EP8, one request) four such uploads stretched a
decode round from 9.7 ms to 16.3 ms. They must go through
``sglang_tpu.host_upload`` instead.

test/registered/tpu/unit/test_host_upload.py checks on a TPU that those helpers
do not wait; this file checks that the hot-path functions use them.
"""

import ast
import importlib.util
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# (module path under sglang_tpu/, class or None, function). Hand-listed: "runs
# while TPU work is queued" is a fact about the round's order, not something a
# scan can infer.
_HOT_PATH = [
    ("compilation/compiled_forward_shim.py", None, "_pad_mamba_tracking"),
    ("experimental/speculative/draft_forward.py", None, "forward_draft_rows"),
    (
        "experimental/speculative/chain_draft_worker.py",
        "TpuChainDraftWorker",
        "fused_draft_batch",
    ),
    ("mamba_pool.py", "TpuMambaPool", "select_checkpoints"),
    ("mamba_pool.py", "TpuMambaPool", "snapshot_checkpoints"),
]


def _package_dir() -> Path:
    spec = importlib.util.find_spec("sglang_tpu")
    assert spec is not None and spec.origin is not None, "sglang_tpu not importable"
    return Path(spec.origin).parent


def _find_function(tree: ast.Module, cls: str | None, name: str):
    scope = tree.body
    if cls is not None:
        owner = next(
            (n for n in scope if isinstance(n, ast.ClassDef) and n.name == cls), None
        )
        if owner is None:
            return None
        scope = owner.body
    return next(
        (
            n
            for n in scope
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name
        ),
        None,
    )


def _is_true(node) -> bool:
    return isinstance(node, ast.Constant) and node.value is True


def _names_a_device(node) -> bool:
    """A ``.to()`` / ``device=`` argument that names a device, not a dtype."""
    if isinstance(node, ast.Constant):
        return isinstance(node.value, str) and node.value != "cpu"
    return "device" in ast.unparse(node)


def _blocking_uploads(fn: ast.AST) -> list[str]:
    """Source of every call in fn that copies host data to a device blocking."""
    bad = []
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        kwargs = {k.arg: k.value for k in node.keywords if k.arg}
        if _is_true(kwargs.get("non_blocking")):
            continue
        if isinstance(func, ast.Attribute) and func.attr == "tensor":
            # torch.tensor(v, device=d) builds on the host, then copies blocking.
            if "device" in kwargs and _names_a_device(kwargs["device"]):
                bad.append(ast.unparse(node))
        elif isinstance(func, ast.Attribute) and func.attr == "to":
            targets = node.args[:1] + ([kwargs["device"]] if "device" in kwargs else [])
            if any(_names_a_device(t) for t in targets):
                bad.append(ast.unparse(node))
        elif isinstance(func, ast.Attribute) and func.attr == "copy_":
            bad.append(ast.unparse(node))
    return bad


class TestSpecRoundUploadsDoNotBlock(CustomTestCase):
    def test_detector_sees_each_blocking_form(self):
        # A detector whose AST matching drifted would report every function clean.
        offender = ast.parse(
            "def f(self, x, buf):\n"
            "    a = x.to(self.device)\n"
            "    b = torch.tensor(3, dtype=torch.int32, device=x.device)\n"
            "    buf.copy_(x)\n"
            "    ok = x.to(torch.int32)\n"
            "    ok = x.to(self.device, non_blocking=True)\n"
            "    ok = torch.tensor([1, 2])\n"
        )
        self.assertEqual(len(_blocking_uploads(offender)), 3)

    def test_hot_path_functions_upload_without_blocking(self):
        root = _package_dir()
        for path, cls, name in _HOT_PATH:
            label = f"sglang_tpu/{path}:{cls + '.' if cls else ''}{name}"
            with self.subTest(function=label):
                tree = ast.parse((root / path).read_text())
                fn = _find_function(tree, cls, name)
                self.assertIsNotNone(
                    fn, f"{label} is gone; update _HOT_PATH to its new name"
                )
                self.assertEqual(
                    _blocking_uploads(fn),
                    [],
                    msg=(
                        f"{label} copies host data to the TPU with a blocking "
                        "call, which waits for every queued TPU program. Use "
                        "sglang_tpu.host_upload.upload or device_scalar."
                    ),
                )


if __name__ == "__main__":
    unittest.main()
