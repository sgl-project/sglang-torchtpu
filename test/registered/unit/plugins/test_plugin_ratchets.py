# SPDX-License-Identifier: Apache-2.0
"""Tests for the plugin debt ratchet: hooks, runtime patches, fork-only core API.

``tools/plugin/check_plugin_ratchets.py`` counts, from source, what sglang_tpu
needs from core, and ``fork_only_api.py`` finds the core API it uses that the
pinned upstream commit lacks. A counter the parser silently misses lets debt
grow unchecked, so these tests pin the parsing down on small synthetic trees.

Runs on CPU; needs no TPU hardware.
"""

import collections
import contextlib
import importlib.util
import io
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

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


ratchets = _load_tool("check_plugin_ratchets.py", "_plugin_ratchets_under_test")
inventory = ratchets.inventory
fork_only_api = ratchets.fork_only_api


# An upstream tree, the fork's changes on top of it, and a plugin that reaches
# into both. Fork-only: module forkmod, symbols a:fork_only_fn and
# pkg:fork_star (through a star import), methods SRTPlatform.fork_method and
# DeviceMixin.dm_fork. SRTPlatform.fork_unused is fork-only but only an
# unrelated plugin class defines it.
_UPSTREAM = {
    "python/sglang/__init__.py": "",
    "python/sglang/srt/__init__.py": "",
    "python/sglang/srt/a.py": "x = 1\n",
    "python/sglang/srt/pkg/__init__.py": "from .impl import *\n",
    "python/sglang/srt/pkg/impl.py": "def starred(): ...\n",
    "python/sglang/srt/lazy.py": "def __getattr__(name): ...\n",
    "python/sglang/srt/platforms/__init__.py": "",
    "python/sglang/srt/platforms/device_mixin.py": """
        class DeviceMixin:
            def dm_shared(self): ...
        """,
    "python/sglang/srt/platforms/interface.py": """
        from sglang.srt.platforms.device_mixin import DeviceMixin

        class SRTPlatform(DeviceMixin):
            def shared(self): ...
        """,
}
_FORK = {
    "python/sglang/srt/a.py": "x = 1\ndef fork_only_fn(): ...\n",
    "python/sglang/srt/forkmod.py": "",
    "python/sglang/srt/pkg/impl.py": "def starred(): ...\ndef fork_star(): ...\n",
    "python/sglang/srt/platforms/device_mixin.py": """
        class DeviceMixin:
            def dm_shared(self): ...
            def dm_fork(self): ...
        """,
    "python/sglang/srt/platforms/interface.py": """
        from sglang.srt.platforms.device_mixin import DeviceMixin

        class SRTPlatform(DeviceMixin):
            def shared(self): ...
            def fork_method(self): ...
            def fork_unused(self): ...
        """,
}
_PLUGIN = {
    "sglang_tpu/__init__.py": "",
    "sglang_tpu/platform.py": """
        from sglang.srt.platforms.interface import SRTPlatform
        from sglang.srt.a import x, fork_only_fn
        from sglang.srt import forkmod
        from sglang.srt.pkg import starred, fork_star
        from sglang.srt.lazy import served_by_getattr
        from sglang.srt.missing import optional_dependency

        class Base(SRTPlatform):
            def shared(self): ...
            def fork_method(self): ...

        class Leaf(Base):
            def dm_fork(self): ...

        class Unrelated:
            def fork_unused(self): ...
        """,
    "sglang_tpu/hooks/__init__.py": """
        from sglang.srt.plugins.hook_registry import HookType
        from sglang_tpu.hooks.extra import hook_specs as extra_hooks

        def _wrap(*args): ...

        HOOKS: list[tuple] = [
            ("sglang.srt.a.x", _wrap, HookType.AROUND),
        ]
        HOOKS.extend(extra_hooks())
        """,
    "sglang_tpu/hooks/extra.py": """
        def hook_specs():
            from sglang.srt.plugins.hook_registry import HookType

            pkg = "sglang.srt.pkg"
            return [
                (f"{pkg}.starred", print, HookType.AFTER),
                (pkg + ".impl.starred", print, HookType.BEFORE),
            ]
        """,
    "sglang_tpu/patches.py": """
        import sglang.srt.a as a_module

        def apply_tpu_patches():
            _patch_a()

        def _patch_a():
            a_module.x = 2
        """,
}

_FORK_ONLY = fork_only_api.ForkOnlyApi(
    modules=["sglang.srt.forkmod"],
    symbols=["sglang.srt.a:fork_only_fn", "sglang.srt.pkg:fork_star"],
    platform_methods=["DeviceMixin.dm_fork", "SRTPlatform.fork_method"],
)


def _write(root: str, files: dict[str, str]) -> None:
    for relpath, text in files.items():
        path = os.path.join(root, relpath)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(textwrap.dedent(text).lstrip("\n"))


def _git(root: str, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", root, "-c", "user.name=t", "-c", "user.email=t@t", *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _fork_repo(root: str) -> str:
    """A git repo whose first commit is "upstream"; returns that commit."""
    _git(root, "init", "-q")
    _write(root, _UPSTREAM)
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "--no-gpg-sign", "-m", "upstream")
    upstream = _git(root, "rev-parse", "HEAD")
    _write(
        root,
        {**_FORK, **_PLUGIN, fork_only_api.UPSTREAM_PIN_PATH: f"# pin\n{upstream}\n"},
    )
    return upstream


def _run(*argv: str) -> tuple[int, str]:
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
        code = ratchets.main(list(argv))
    return code, out.getvalue()


class TestHookInventory(CustomTestCase):
    def test_static_hooks_cover_the_runtime_registry(self):
        """Hooks a provider adds through ``HOOKS.extend(provider())`` were invisible
        to the source count (68 counted, 156 registered), so they escaped the ratchet."""
        import sglang_tpu.hooks

        static = collections.Counter(
            h.target for h in inventory.collect_hooks(_REPO_ROOT)
        )
        runtime = collections.Counter(target for target, _, _ in sglang_tpu.hooks.HOOKS)
        self.assertEqual(runtime - static, collections.Counter())

    def test_provider_targets_resolve_through_local_strings(self):
        with tempfile.TemporaryDirectory() as root:
            _write(root, _PLUGIN)
            targets = [h.target for h in inventory.collect_hooks(root)]
        self.assertEqual(
            targets,
            ["sglang.srt.a.x", "sglang.srt.pkg.starred", "sglang.srt.pkg.impl.starred"],
        )

    def test_hook_module_is_the_longest_prefix_with_a_source_file(self):
        with tempfile.TemporaryDirectory() as root:
            _write(root, {**_UPSTREAM, **_PLUGIN})
            cases = {
                "sglang.srt.pkg.impl.starred": "sglang.srt.pkg.impl",
                "sglang.srt.pkg.starred": "sglang.srt.pkg",
                "sglang.srt.platforms.interface.SRTPlatform.shared": (
                    "sglang.srt.platforms.interface"
                ),
                "sglang_tpu.hooks.extra.hook_specs": "sglang_tpu.hooks.extra",
                # A deleted module: the closest package that remains.
                "sglang.srt.gone.Thing.method": "sglang.srt",
            }
            for target, module in cases.items():
                self.assertEqual(ratchets.hook_module(target, root), module, target)


class TestForkOnlyApi(CustomTestCase):
    def test_finds_fork_only_modules_symbols_and_platform_methods(self):
        """Star imports and ``__getattr__`` count as definitions; an import that fails
        in the fork too, and a method on a non-platform class, do not count."""
        with (
            tempfile.TemporaryDirectory() as upstream,
            tempfile.TemporaryDirectory() as fork,
        ):
            _write(upstream, _UPSTREAM)
            _write(fork, {**_UPSTREAM, **_FORK, **_PLUGIN})
            api = fork_only_api.collect(fork, fork_only_api.disk_tree(upstream))
        self.assertEqual(api, _FORK_ONLY)

    def test_reads_the_pinned_commit_from_git_objects(self):
        with tempfile.TemporaryDirectory() as root:
            upstream = _fork_repo(root)
            api = fork_only_api.collect(root, fork_only_api.git_tree(root, upstream))
            self.assertEqual(api, _FORK_ONLY)
            with self.assertRaises(fork_only_api.UpstreamUnavailable):
                fork_only_api.git_tree(root, "0" * 40)


class TestCompare(CustomTestCase):
    BASELINE = ratchets.PluginDebt(
        hooks={"sglang.srt.a": 2, "sglang.srt.b": 1},
        patches={"_patch_a": ["sglang.srt.a.f", "sglang.srt.a.g"]},
        fork_only=fork_only_api.ForkOnlyApi(modules=["m"], symbols=["s:x"]),
    )

    def test_growth_is_a_violation(self):
        current = self.BASELINE._replace(
            hooks={"sglang.srt.a": 3, "sglang.srt.b": 1, "sglang.srt.c": 1},
            patches={
                "_patch_a": ["sglang.srt.a.f", "sglang.srt.a.g", "sglang.srt.a.h"],
                "_patch_new": [],
            },
            fork_only=fork_only_api.ForkOnlyApi(
                modules=["m"],
                symbols=["s:x", "s:y"],
                platform_methods=["SRTPlatform.p"],
            ),
        )
        violations, improvements = ratchets.compare(current, self.BASELINE)
        self.assertEqual(improvements, [])
        for expected in (
            "sglang.srt.a: 3 hook(s), baseline 2",
            "sglang.srt.c: 1 hook(s), baseline 0",
            "_patch_a() now also assigns sglang.srt.a.h",
            "new runtime patch _patch_new()",
            "s:y: new fork-only core symbol",
            "SRTPlatform.p: new fork-only platform method",
        ):
            self.assertTrue(any(expected in v for v in violations), expected)
        self.assertEqual(len(violations), 6)

    def test_shrinking_is_an_improvement_not_a_violation(self):
        current = self.BASELINE._replace(
            hooks={"sglang.srt.a": 1},
            patches={"_patch_a": ["sglang.srt.a.f"]},
            fork_only=fork_only_api.ForkOnlyApi(symbols=["s:x"]),
        )
        violations, improvements = ratchets.compare(current, self.BASELINE)
        self.assertEqual(violations, [])
        self.assertEqual(len(improvements), 4)

    def test_unreadable_upstream_skips_only_the_fork_only_counters(self):
        current = self.BASELINE._replace(hooks={"sglang.srt.a": 3}, fork_only=None)
        violations, _ = ratchets.compare(current, self.BASELINE)
        self.assertEqual(len(violations), 1)
        self.assertIn("sglang.srt.a: 3 hook(s)", violations[0])


class TestCommandLine(CustomTestCase):
    def test_a_new_hook_warns_and_write_baseline_locks_it_in(self):
        with tempfile.TemporaryDirectory() as root:
            _fork_repo(root)
            self.assertEqual(_run("--repo-root", root, "--write-baseline")[0], 0)
            self.assertEqual(_run("--repo-root", root)[0], 0)

            hooks = os.path.join(root, "sglang_tpu/hooks/__init__.py")
            with open(hooks, "a", encoding="utf-8") as handle:
                handle.write(
                    'HOOKS.append(("sglang.srt.a.fork_only_fn", _wrap, HookType.AROUND))\n'
                )
            code, out = _run("--repo-root", root)
            self.assertEqual(code, 1)
            self.assertIn("sglang.srt.a: 2 hook(s), baseline 1", out)
            code, out = _run("--repo-root", root, "--warn-only")
            self.assertEqual(code, 0)
            self.assertIn("sglang.srt.a: 2 hook(s), baseline 1", out)

            self.assertEqual(_run("--repo-root", root, "--write-baseline")[0], 0)
            self.assertEqual(_run("--repo-root", root)[0], 0)

    def test_a_missing_pinned_commit_skips_fork_only_counters_without_failing(self):
        """A shallow clone lacks the pinned commit; pre-commit must still pass."""
        with tempfile.TemporaryDirectory() as root:
            _fork_repo(root)
            self.assertEqual(_run("--repo-root", root, "--write-baseline")[0], 0)
            _write(root, {fork_only_api.UPSTREAM_PIN_PATH: "0" * 40 + "\n"})
            code, out = _run("--repo-root", root)
            self.assertEqual(code, 0)
            self.assertIn("fork-only counters skipped", out)
            self.assertEqual(_run("--repo-root", root, "--write-baseline")[0], 2)


if __name__ == "__main__":
    unittest.main()
