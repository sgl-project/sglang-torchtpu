# SPDX-License-Identifier: Apache-2.0
"""List the core API sglang_tpu uses that upstream SGLang does not have.

Mode C runs the plugin on a stock upstream install, so each of these blocks it.
Parsed, not imported:

- fork-only modules: ``sglang.*`` modules the plugin imports that exist in this
  fork but not upstream;
- fork-only symbols: names the plugin imports from a module upstream has, which
  this fork defines and upstream does not;
- fork-only platform methods: methods the plugin's platform classes define that
  this fork's ``SRTPlatform`` / ``DeviceMixin`` declare and upstream's do not.

"Upstream" is the sgl-project/sglang commit last merged into this fork, pinned
in ``tools/plugin/upstream_pin.txt``. The sync is a real merge, so that commit
is in every full clone and is read from local git objects; ``--upstream-root``
reads an upstream checkout instead. Name resolution is approximate: it follows
star imports and takes a module-level ``__getattr__`` to define every name.

Usage:
    python tools/plugin/fork_only_api.py
    python tools/plugin/fork_only_api.py --upstream-root ~/sglang
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import gc
import importlib.util
import os
import subprocess
import sys
import warnings
from typing import Callable, Iterable, Iterator, NamedTuple, Optional

# Core files are parsed, not imported; their SyntaxWarnings are not this tool's output.
warnings.simplefilter("ignore", SyntaxWarning)

_TOOLS = os.path.dirname(os.path.abspath(__file__))

#: Paths are relative to the repo root, with forward slashes.
UPSTREAM_PIN_PATH = "tools/plugin/upstream_pin.txt"
PLUGIN_ROOT = "sglang_tpu"
CORE_PACKAGE = "sglang"
#: Where the ``sglang`` package sits, in this fork and in an upstream checkout alike.
CORE_DIR = f"python/{CORE_PACKAGE}"

#: The core platform base classes, module -> class.
PLATFORM_BASES = {
    "sglang.srt.platforms.interface": "SRTPlatform",
    "sglang.srt.platforms.device_mixin": "DeviceMixin",
}


class UpstreamUnavailable(Exception):
    """The upstream tree cannot be read: no pin, or its objects are not local."""


class ForkOnlyApi(NamedTuple):
    """NamedTuple, not msgspec.Struct: runs on the bare pre-commit interpreter."""

    modules: list[str] = []
    #: "module:name"
    symbols: list[str] = []
    #: "SRTPlatform.method" / "DeviceMixin.method"
    platform_methods: list[str] = []


def _load(filename: str):
    """Import a sibling tools/plugin/ script without putting tools/ on sys.path."""
    name = f"_fork_only_api_{filename[:-3]}"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, os.path.join(_TOOLS, filename)
        )
        sys.modules[name] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sys.modules[name])
    return sys.modules[name]


@contextlib.contextmanager
def _without_cyclic_gc() -> Iterator[None]:
    # Parsing ~800 files allocates millions of AST nodes; the cyclic collector's
    # passes over them cost more than the parsing. ASTs hold no cycles.
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if was_enabled:
            gc.enable()


# -- source trees --------------------------------------------------------------


class SourceTree:
    """The ``python/sglang`` sources of one SGLang tree, read on demand."""

    def __init__(
        self, files: set[str], read_many: Callable[[list[str]], dict[str, str]]
    ):
        self.files = files
        self.dirs: set[str] = set()
        for path in files:
            parent = path.rpartition("/")[0]
            while parent and parent not in self.dirs:
                self.dirs.add(parent)
                parent = parent.rpartition("/")[0]
        self._read_many = read_many
        self._sources: dict[str, str] = {}

    def module_path(self, module: str) -> Optional[str]:
        base = f"python/{module.replace('.', '/')}"
        for path in (f"{base}.py", f"{base}/__init__.py"):
            if path in self.files:
                return path
        return None

    def has_module(self, module: str) -> bool:
        """A module, a package, or a namespace package directory."""
        return (
            self.module_path(module) is not None
            or f"python/{module.replace('.', '/')}" in self.dirs
        )

    def prefetch(self, modules: Iterable[str]) -> None:
        """Read these modules' sources in one batch."""
        paths = {self.module_path(module) for module in modules}
        wanted = sorted(p for p in paths if p and p not in self._sources)
        if wanted:
            self._sources.update(self._read_many(wanted))

    def parse(self, module: str) -> Optional[ast.Module]:
        path = self.module_path(module)
        if path is None:
            return None
        if path not in self._sources:
            self._sources.update(self._read_many([path]))
        try:
            return ast.parse(self._sources[path], filename=path)
        except SyntaxError:  # pragma: no cover - CI lints syntax separately
            return None


def _python_files(root: str, subdir: str) -> Iterator[str]:
    """``.py`` files under ``root/subdir``, relative to ``root``, sorted."""
    for dirpath, dirnames, filenames in os.walk(os.path.join(root, subdir)):
        dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
        rel = os.path.relpath(dirpath, root).replace(os.sep, "/")
        for name in sorted(filenames):
            if name.endswith(".py"):
                yield f"{rel}/{name}"


def disk_tree(root: str) -> SourceTree:
    """The tree checked out at ``root``, the directory that holds ``python/``."""
    files = set(_python_files(root, CORE_DIR))
    if not files:
        raise UpstreamUnavailable(f"no {CORE_DIR} under {root}")

    def read_many(paths: list[str]) -> dict[str, str]:
        sources = {}
        for path in paths:
            with open(os.path.join(root, path), encoding="utf-8") as handle:
                sources[path] = handle.read()
        return sources

    return SourceTree(files, read_many)


def git_tree(repo_root: str, rev: str) -> SourceTree:
    """The tree of commit ``rev`` in this repo's object store. Never fetches."""

    def git(*args: str, stdin: bytes = b"") -> bytes:
        result = subprocess.run(
            ["git", "-C", repo_root, *args],
            input=stdin,
            capture_output=True,
            env={**os.environ, "GIT_NO_LAZY_FETCH": "1"},
        )
        if result.returncode != 0:
            raise UpstreamUnavailable(
                f"`git {args[0]}` cannot read upstream commit {rev[:10]} (shallow "
                f"clone? `git fetch origin {rev}` or pass --upstream-root): "
                f"{result.stderr.decode().strip()}"
            )
        return result.stdout

    listing = git("ls-tree", "-r", "--name-only", rev, "--", CORE_DIR)
    files = {path for path in listing.decode().splitlines() if path.endswith(".py")}
    if not files:
        raise UpstreamUnavailable(f"upstream commit {rev[:10]} has no {CORE_DIR}")

    def read_many(paths: list[str]) -> dict[str, str]:
        batch = "".join(f"{rev}:{path}\n" for path in paths).encode()
        out = git("cat-file", "--batch", stdin=batch)
        sources = {}
        offset = 0
        for path in paths:
            newline = out.index(b"\n", offset)
            header = out[offset:newline].split()
            if len(header) != 3 or header[1] != b"blob":
                raise UpstreamUnavailable(f"{rev[:10]}:{path} is not available locally")
            end = newline + 1 + int(header[2])
            sources[path] = out[newline + 1 : end].decode("utf-8")
            offset = end + 1
        return sources

    return SourceTree(files, read_many)


def read_pin(repo_root: str) -> str:
    """The pinned upstream commit: the pin file's first non-comment line."""
    path = os.path.join(repo_root, UPSTREAM_PIN_PATH)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line and not line.startswith("#"):
                    return line
    raise UpstreamUnavailable(f"{UPSTREAM_PIN_PATH} names no commit")


def upstream_tree(repo_root: str, upstream_root: Optional[str] = None) -> SourceTree:
    if upstream_root:
        return disk_tree(upstream_root)
    return git_tree(repo_root, read_pin(repo_root))


# -- what a module defines -----------------------------------------------------


def _statements(body: list[ast.stmt], into_defs: bool) -> Iterator[ast.stmt]:
    """Statements at any block depth; into function and class bodies if asked."""
    for stmt in body:
        yield stmt
        if not into_defs and isinstance(
            stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            continue
        for field in ("body", "orelse", "finalbody"):
            nested = getattr(stmt, field, None)
            if isinstance(nested, list):
                yield from _statements(nested, into_defs)
        for block in (
            getattr(stmt, "handlers", None) or getattr(stmt, "cases", None) or []
        ):
            yield from _statements(block.body, into_defs)


def _bound_names(target: ast.expr) -> Iterator[str]:
    if isinstance(target, ast.Name):
        yield target.id
    elif isinstance(target, (ast.Tuple, ast.List)):
        for element in target.elts:
            yield from _bound_names(element)
    elif isinstance(target, ast.Starred):
        yield from _bound_names(target.value)


def _absolute(module: str, is_package: bool, node: ast.ImportFrom) -> str:
    if not node.level:
        return node.module or ""
    parts = module.split(".")
    base = parts[: len(parts) - node.level + int(is_package)]
    return ".".join(base + ([node.module] if node.module else []))


def defined_names(
    tree: SourceTree, module: str, _seen: Optional[set[str]] = None
) -> tuple[set[str], bool]:
    """Names ``module`` binds at import, and whether a ``__getattr__`` serves any name."""
    seen = set() if _seen is None else _seen
    if module in seen:
        return set(), False
    seen.add(module)
    parsed = tree.parse(module)
    if parsed is None:
        return set(), False
    is_package = tree.module_path(module).endswith("/__init__.py")
    names: set[str] = set()
    wildcard = False
    for stmt in _statements(parsed.body, into_defs=False):
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(stmt.name)
            wildcard |= stmt.name == "__getattr__"
        elif isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                names.update(_bound_names(target))
        elif isinstance(stmt, (ast.AnnAssign, ast.AugAssign)):
            names.update(_bound_names(stmt.target))
        elif isinstance(stmt, ast.Import):
            names.update(a.asname or a.name.split(".", 1)[0] for a in stmt.names)
        elif isinstance(stmt, ast.ImportFrom):
            for alias in stmt.names:
                if alias.name != "*":
                    names.add(alias.asname or alias.name)
                    continue
                source = _absolute(module, is_package, stmt)
                star_names, star_wildcard = defined_names(tree, source, seen)
                names |= star_names
                wildcard |= star_wildcard
    return names, wildcard


class _Resolver:
    """``defined_names`` per (tree, module), computed once."""

    def __init__(self) -> None:
        self._cache: dict[tuple[int, str], tuple[set[str], bool]] = {}

    def defines(self, tree: SourceTree, module: str, name: str) -> bool:
        if tree.has_module(f"{module}.{name}"):
            return True
        key = (id(tree), module)
        if key not in self._cache:
            self._cache[key] = defined_names(tree, module)
        names, wildcard = self._cache[key]
        return wildcard or name in names


# -- the plugin's imports ------------------------------------------------------


def _is_core(module: str) -> bool:
    return module == CORE_PACKAGE or module.startswith(CORE_PACKAGE + ".")


def core_imports(parsed: ast.Module) -> Iterator[tuple[str, Optional[str]]]:
    """``(module, name)`` per imported core name; ``name`` is None for ``import m``."""
    for node in _statements(parsed.body, into_defs=True):
        if isinstance(node, ast.Import):
            yield from ((a.name, None) for a in node.names if _is_core(a.name))
        elif (
            isinstance(node, ast.ImportFrom)
            and not node.level
            and node.module
            and _is_core(node.module)
        ):
            yield from ((node.module, a.name) for a in node.names if a.name != "*")


def fork_only_imports(
    imported: set[tuple[str, Optional[str]]], fork: SourceTree, upstream: SourceTree
) -> tuple[list[str], list[str]]:
    """Fork-only ``(modules, "module:name" symbols)`` the plugin imports.

    Only what resolves in this fork counts: an import that fails here too is an
    optional dependency behind a guard, not fork API.
    """
    upstream.prefetch(module for module, _ in imported)
    resolver = _Resolver()
    modules: set[str] = set()
    symbols: set[str] = set()
    for module, name in imported:
        if name is not None and fork.has_module(f"{module}.{name}"):
            module, name = f"{module}.{name}", None
        if not fork.has_module(module):
            continue
        if not upstream.has_module(module):
            modules.add(module)
        elif (
            name is not None
            and not resolver.defines(upstream, module, name)
            and resolver.defines(fork, module, name)
        ):
            symbols.add(f"{module}:{name}")
    return sorted(modules), sorted(symbols)


# -- platform methods ----------------------------------------------------------


def _methods(cls: ast.ClassDef) -> set[str]:
    return {
        stmt.name
        for stmt in cls.body
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _class_methods(tree: SourceTree, module: str, cls: str) -> set[str]:
    parsed = tree.parse(module)
    for node in parsed.body if parsed else []:
        if isinstance(node, ast.ClassDef) and node.name == cls:
            return _methods(node)
    return set()


#: Class qualname -> (resolved base qualnames, method names).
PluginClasses = dict[str, tuple[list[str], set[str]]]


def plugin_classes(relpath: str, parsed: ast.Module) -> PluginClasses:
    """The top-level classes of one plugin file, bases resolved through its imports."""
    inventory = _load("inventory.py")
    module = relpath[: -len(".py")].replace("/", ".").removesuffix(".__init__")
    local = [node for node in parsed.body if isinstance(node, ast.ClassDef)]
    aliases = {
        **inventory._import_aliases(parsed),
        **{node.name: f"{module}.{node.name}" for node in local},
    }
    classes = {}
    for node in local:
        bases = [inventory._dotted(base, aliases) for base in node.bases]
        classes[f"{module}.{node.name}"] = ([b for b in bases if b], _methods(node))
    return classes


def plugin_platform_methods(classes: PluginClasses) -> set[str]:
    """Methods of plugin classes that derive, in the plugin, from a core platform base."""
    platform = {f"{module}.{cls}" for module, cls in PLATFORM_BASES.items()}
    while True:
        derived = {
            q for q, (bases, _) in classes.items() if platform.intersection(bases)
        }
        if derived <= platform:
            break
        platform |= derived
    return set().union(
        *(methods for q, (_, methods) in classes.items() if q in platform)
    )


def fork_only_platform_methods(
    overridden: set[str], fork: SourceTree, upstream: SourceTree
) -> list[str]:
    upstream_methods = set().union(
        *(
            _class_methods(upstream, module, cls)
            for module, cls in PLATFORM_BASES.items()
        )
    )
    found = []
    for module, cls in PLATFORM_BASES.items():
        fork_only = _class_methods(fork, module, cls) - upstream_methods
        found += [f"{cls}.{name}" for name in fork_only & overridden]
    return sorted(found)


# -- entry points --------------------------------------------------------------


def _scan_plugin(
    repo_root: str,
) -> tuple[set[tuple[str, Optional[str]]], PluginClasses]:
    """Every core import of the plugin, and its classes; one parse per file."""
    imported: set[tuple[str, Optional[str]]] = set()
    classes: PluginClasses = {}
    for relpath in _python_files(repo_root, PLUGIN_ROOT):
        with open(os.path.join(repo_root, relpath), encoding="utf-8") as handle:
            try:
                parsed = ast.parse(handle.read(), filename=relpath)
            except SyntaxError:  # pragma: no cover - CI lints syntax separately
                continue
        imported.update(core_imports(parsed))
        classes.update(plugin_classes(relpath, parsed))
    return imported, classes


def collect(repo_root: str, upstream: SourceTree) -> ForkOnlyApi:
    """Raises UpstreamUnavailable when an upstream source cannot be read."""
    with _without_cyclic_gc():
        fork = disk_tree(repo_root)
        imported, classes = _scan_plugin(repo_root)
        modules, symbols = fork_only_imports(imported, fork, upstream)
        methods = fork_only_platform_methods(
            plugin_platform_methods(classes), fork, upstream
        )
    return ForkOnlyApi(modules=modules, symbols=symbols, platform_methods=methods)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo-root", default=os.path.dirname(os.path.dirname(_TOOLS)))
    parser.add_argument(
        "--upstream-root",
        help="an sgl-project/sglang checkout to compare against instead of the pin",
    )
    args = parser.parse_args(argv)
    try:
        api = collect(args.repo_root, upstream_tree(args.repo_root, args.upstream_root))
    except UpstreamUnavailable as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    for title, items in (
        ("Fork-only core modules imported by sglang_tpu", api.modules),
        ("Fork-only core symbols imported by sglang_tpu", api.symbols),
        ("Fork-only platform methods sglang_tpu overrides", api.platform_methods),
    ):
        print(f"## {title}: {len(items)}\n")
        print("\n".join(f"- `{item}`" for item in items) or "(none)")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
