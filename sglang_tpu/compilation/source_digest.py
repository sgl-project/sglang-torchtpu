"""Digest of the source packages that can shape a traced graph.

The lowered-graph store keys every entry by this digest (``lowered_graph_cache``)
and computes it at every startup by reading each file: a reused executable
must come from the code that runs now, and only the contents can tell.

A deployment whose own packages define models or patch these names them in
``SGLANG_TPU_LOWERED_GRAPH_SOURCE_PACKAGES`` (comma-separated).
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
from collections.abc import Iterable, Mapping
from pathlib import Path

PACKAGES_ENV = "SGLANG_TPU_LOWERED_GRAPH_SOURCE_PACKAGES"

# Packages whose files can run or be read while a decoder graph is traced.
# Every file counts: JSON profiles and kernel sources shape graphs too.
SOURCE_PACKAGES = ("sglang.srt", "sglang.kernels", "sglang_tpu")


def extra_source_packages() -> tuple[str, ...]:
    """The deployment's own packages (``PACKAGES_ENV``)."""
    names = (name.strip() for name in os.environ.get(PACKAGES_ENV, "").split(","))
    return tuple(name for name in names if name)


def source_roots(packages: Iterable[str] | None = None) -> dict[str, Path]:
    """Each package's source directory. A package that cannot be found, or
    that spans several directories (a namespace package), raises.

    By default, ``SOURCE_PACKAGES`` and the deployment's own packages.
    """
    if packages is None:
        packages = (*SOURCE_PACKAGES, *extra_source_packages())
    roots = {}
    for package in packages:
        spec = importlib.util.find_spec(package)
        locations = list(spec.submodule_search_locations or ()) if spec else []
        if len(locations) != 1:
            raise RuntimeError(
                f"cannot locate the sources of {package}: found {locations}"
            )
        roots[package] = Path(locations[0])
    return roots


def _files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    )


def hash_sources(roots: Mapping[str, Path]) -> str:
    """Hash every file under each named root by relative path."""
    hasher = hashlib.sha256()
    for name, root in sorted(roots.items()):
        for path in _files(root):
            relative = f"{name}/{path.relative_to(root).as_posix()}".encode()
            contents = path.read_bytes()
            hasher.update(len(relative).to_bytes(8, "big"))
            hasher.update(relative)
            hasher.update(len(contents).to_bytes(8, "big"))
            hasher.update(contents)
    return hasher.hexdigest()


def sources_digest() -> str:
    """The digest of ``SOURCE_PACKAGES`` and the deployment's own packages."""
    return hash_sources(source_roots())
