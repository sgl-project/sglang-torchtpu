"""Digest of the source packages that can shape a traced graph.

The lowered-graph store keys every entry by this digest (``lowered_graph_cache``).
Hashing every file costs a recursive read at each startup, so an immutable
image computes it once at build time (``python -m
sglang_tpu.compilation.source_digest --write PATH``) and the store reads the
manifest named by ``SGLANG_TPU_LOWERED_GRAPH_SOURCES_PATH``. Each root's file
count, total size and newest mtime are recorded too; a root whose stat
fingerprint changed since (a source overlay on a dev image) is hashed again,
so a stale manifest can only cost time.

A deployment whose own packages define models or patch these names them in
``SGLANG_TPU_LOWERED_GRAPH_SOURCE_PACKAGES`` (comma-separated), at build time
and at runtime alike.

Imports nothing heavy: the image build runs it before any TPU is present.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from collections.abc import Iterable, Mapping
from pathlib import Path

MANIFEST_ENV = "SGLANG_TPU_LOWERED_GRAPH_SOURCES_PATH"
MANIFEST_FORMAT = 1
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


def stat_fingerprint(roots: Mapping[str, Path]) -> dict[str, list[int | str]]:
    """Per root: its path, file count, total size and newest mtime (ns).

    A stat walk, without reading any file.
    """
    fingerprint = {}
    for name, root in sorted(roots.items()):
        stats = [path.stat() for path in _files(root)]
        fingerprint[name] = [
            str(root),
            len(stats),
            sum(stat.st_size for stat in stats),
            max((stat.st_mtime_ns for stat in stats), default=0),
        ]
    return fingerprint


def write_manifest(path: Path, roots: Mapping[str, Path] | None = None) -> str:
    roots = source_roots() if roots is None else roots
    digest = hash_sources(roots)
    manifest = {
        "format": MANIFEST_FORMAT,
        "digest": digest,
        "roots": stat_fingerprint(roots),
    }
    path.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    return digest


def sources_digest(
    roots: Mapping[str, Path] | None = None,
    environ: Mapping[str, str] = os.environ,
) -> tuple[str, str]:
    """The source digest and where it came from: ``"manifest"`` when the
    build-time manifest still matches every root's stat fingerprint, else
    ``"hashed"``."""
    roots = source_roots() if roots is None else roots
    manifest_path = environ.get(MANIFEST_ENV, "").strip()
    if manifest_path:
        try:
            manifest = json.loads(Path(manifest_path).read_text())
        except (OSError, ValueError):
            manifest = None
        if (
            isinstance(manifest, dict)
            and manifest.get("format") == MANIFEST_FORMAT
            and manifest.get("roots") == stat_fingerprint(roots)
        ):
            return str(manifest["digest"]), "manifest"
    return hash_sources(roots), "hashed"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write", type=Path, required=True, help="manifest path")
    args = parser.parse_args(argv)
    print(f"{args.write}: {write_manifest(args.write)}")


if __name__ == "__main__":
    main()
