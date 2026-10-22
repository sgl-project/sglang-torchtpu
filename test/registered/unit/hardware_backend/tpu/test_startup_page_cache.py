"""CPU checks for startup file cleanup and host memory diagnostics."""

import logging
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu import page_cache, startup_memory

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


@pytest.mark.parametrize(
    "current,legacy,name",
    [
        ("active", "other", "active"),
        (None, "active", "active"),
        (None, None, "default"),
        ("", "other", "default"),
    ],
)
def test_disk_cache_preserves_files_and_other_namespaces(
    monkeypatch, tmp_path, current, legacy, name
):
    root = tmp_path / name
    root.mkdir()
    (root / "nested").mkdir()
    paths = [root / "a", root / "nested/b"]
    for path in paths:
        path.write_bytes(b"payload")
    (root / "link").symlink_to(paths[0])
    other = tmp_path / "unrelated"
    other.mkdir()
    (other / "keep").touch()
    monkeypatch.setattr(page_cache, "_CACHE_ROOT", tmp_path)
    monkeypatch.setattr(page_cache, "_filesystem_type", lambda _: "ext4")
    for key, value in [
        ("TORCH_TPU_TIER2_COMPILATION_CACHE", current),
        ("TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE", legacy),
    ]:
        monkeypatch.delenv(key, raising=False)
        if value is not None:
            monkeypatch.setenv(key, value)
    advised = []
    monkeypatch.setattr(
        page_cache, "_advise_file", lambda p: advised.append(p) or p.stat().st_size
    )
    page_cache.release_compile_page_cache()
    assert sorted(advised) == paths
    assert all(p.read_bytes() == b"payload" for p in paths)
    monkeypatch.setenv("SGLANG_TPU_RELEASE_COMPILE_PAGE_CACHE", "0")
    advised.clear()
    page_cache.release_compile_page_cache()
    assert advised == []


def test_checkpoint_paths_are_deduplicated_and_do_not_recurse(monkeypatch, tmp_path):
    (tmp_path / "a.safetensors").touch()
    (tmp_path / "config.json").touch()
    draft = tmp_path / "draft"
    draft.mkdir()
    (draft / "b.safetensors").touch()
    advised = []
    monkeypatch.setattr(page_cache, "_advise_file", lambda p: advised.append(p) or 0)
    page_cache.release_checkpoint_page_cache([str(tmp_path), str(tmp_path)])
    assert advised == [tmp_path / "a.safetensors"]
    advised.clear()
    page_cache.release_checkpoint_page_cache([str(tmp_path), str(draft)])
    assert sorted(advised) == [tmp_path / "a.safetensors", draft / "b.safetensors"]
    advised.clear()
    monkeypatch.setenv("SGLANG_TPU_RELEASE_CHECKPOINT_PAGE_CACHE", "0")
    page_cache.release_checkpoint_page_cache([str(tmp_path), str(draft)])
    assert advised == []


def test_advice_errors_propagate_and_close_files(monkeypatch, tmp_path):
    target = tmp_path / "shard"
    target.touch()
    descriptors = []

    def fail(fd, *_):
        descriptors.append(fd)
        raise OSError("advice failed")

    monkeypatch.setattr(page_cache.os, "posix_fadvise", fail)
    with pytest.raises(OSError, match="advice failed"):
        page_cache._advise_files([target])
    with pytest.raises(OSError):
        os.fstat(descriptors[0])


def test_startup_discovers_files_only_on_the_host_leader(monkeypatch):
    events = []
    group = SimpleNamespace(barrier=lambda: events.append("barrier"))
    monkeypatch.setattr(
        page_cache, "resolve_compilations", lambda _: events.append("saves")
    )
    monkeypatch.setattr(
        page_cache, "get_parallel", lambda: SimpleNamespace(world_group=group)
    )
    monkeypatch.setattr(
        page_cache, "get_model", lambda: SimpleNamespace(model_path="target")
    )
    monkeypatch.setattr(
        page_cache,
        "get_spec",
        lambda: SimpleNamespace(speculative_draft_model_path="draft"),
    )
    monkeypatch.setattr(
        page_cache, "release_checkpoint_page_cache", lambda paths: events.append(paths)
    )
    monkeypatch.setattr(
        page_cache, "release_compile_page_cache", lambda: events.append("compiled")
    )
    for rank in range(8):
        monkeypatch.setenv("LOCAL_RANK", str(rank))
        page_cache.release_startup_page_cache()
    assert events[:4] == ["saves", "barrier", ["target", "draft"], "compiled"]
    assert events[4:] == ["saves", "barrier"] * 7


@pytest.mark.parametrize(
    "env_name",
    ["TORCH_TPU_TIER2_COMPILATION_CACHE", "TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE"],
)
def test_compile_cache_disabled_skips(monkeypatch, tmp_path, caplog, env_name):
    (tmp_path / "disabled").mkdir()
    monkeypatch.setattr(page_cache, "_CACHE_ROOT", tmp_path)
    monkeypatch.delenv("TORCH_TPU_TIER2_COMPILATION_CACHE", raising=False)
    monkeypatch.delenv("TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE", raising=False)
    monkeypatch.setenv(env_name, "disabled")
    monkeypatch.setattr(
        page_cache, "_filesystem_type", lambda _: pytest.fail("cache disabled")
    )
    with caplog.at_level(logging.INFO):
        page_cache.release_compile_page_cache()
    assert not caplog.records


@pytest.mark.parametrize("failure", [FileNotFoundError, PermissionError])
@pytest.mark.parametrize("operation", ["lstat", "open", "walk"])
def test_compile_cache_disappearance_and_other_errors(
    monkeypatch, tmp_path, caplog, failure, operation
):
    root = tmp_path / "active"
    root.mkdir()
    target = root / "program"
    target.write_bytes(b"x")
    monkeypatch.setattr(page_cache, "_CACHE_ROOT", tmp_path)
    monkeypatch.delenv("TORCH_TPU_TIER2_COMPILATION_CACHE", raising=False)
    monkeypatch.setenv("TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE", "active")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "1")
    monkeypatch.setattr(page_cache, "_filesystem_type", lambda _: "ext4")
    if operation == "walk":

        def walk(_, *, onerror):
            onerror(failure("cache changed"))
            return []

        monkeypatch.setattr(page_cache.os, "walk", walk)
    else:
        original = getattr(Path, operation)

        def changed(path, *args, **kwargs):
            if path == target:
                raise failure("cache changed")
            return original(path, *args, **kwargs)

        monkeypatch.setattr(Path, operation, changed)
    if failure is FileNotFoundError:
        with caplog.at_level(logging.INFO):
            page_cache.release_compile_page_cache()
        assert "advised_gb=0.000" in caplog.text
    else:
        with pytest.raises(PermissionError, match="cache changed"):
            page_cache.release_compile_page_cache()


def test_tmpfs_mount_resolution_and_skip(monkeypatch, tmp_path, caplog):
    disk = tmp_path / "disk cache"
    disk.mkdir()
    link = tmp_path / "shm/cache"
    link.parent.mkdir()
    link.symlink_to(disk, target_is_directory=True)
    escaped = str(disk).replace(" ", r"\040")
    mountinfo = (
        "1 0 0:1 / / rw - overlay overlay rw\n"
        f"2 1 0:2 / {link.parent} rw - tmpfs tmpfs rw\n"
        f"3 1 0:3 / {escaped} rw - xfs disk rw\n"
    )
    original_read = Path.read_text
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda path, *a, **kw: (
            mountinfo
            if str(path) == "/proc/self/mountinfo"
            else original_read(path, *a, **kw)
        ),
    )
    assert page_cache._filesystem_type(link) == "xfs"
    assert page_cache._filesystem_type(link.parent) == "tmpfs"
    mountinfo += f"4 1 0:4 / {escaped} rw - tmpfs tmpfs rw\n"
    assert page_cache._filesystem_type(link) == "tmpfs"

    monkeypatch.setattr(page_cache, "_CACHE_ROOT", link.parent)
    monkeypatch.setenv("TORCH_TPU_TIER2_COMPILATION_CACHE", "active")
    (link.parent / "active").mkdir()
    with caplog.at_level(logging.INFO):
        page_cache.release_compile_page_cache()
    assert not caplog.records


@pytest.mark.parametrize("page_size", [4096, 65536])
def test_memory_log_aggregates_zones_and_warns(
    monkeypatch, tmp_path, caplog, page_size
):
    # 40,000 2 MiB blocks exceed 60 GB; small blocks do not count.
    order = (2 * 1024 * 1024 // page_size).bit_length() - 1
    counts = ["999999"] * order + ["20000", "0"]
    buddy = tmp_path / "buddyinfo"
    buddy.write_text(
        f"Node 0, zone DMA {' '.join(counts)}\n"
        f"Node 0, zone Normal {' '.join(counts)}\n"
        f"Node 1, zone Normal {' '.join(['0'] * (order + 2))}\n"
    )
    monkeypatch.setattr(startup_memory, "_BUDDYINFO", buddy)
    monkeypatch.setattr(startup_memory.os, "sysconf", lambda _: page_size)
    monkeypatch.setattr(startup_memory, "_CGROUP_ROOT", tmp_path)
    (tmp_path / "memory.max").write_text("800000000000")
    (tmp_path / "memory.current").write_text("610000000000")
    with caplog.at_level(logging.INFO):
        startup_memory.log_startup_memory()
    assert "cgroup_headroom_gb=190.000" in caplog.text
    assert "node=0 free_2m_gb=83.886" in caplog.text
    warnings = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 2
    assert "headroom" in warnings[0] and "below 200 GB" in warnings[0]
    assert "node 1" in warnings[1] and "below 60 GB" in warnings[1]

    caplog.clear()
    (tmp_path / "memory.max").write_text("max")
    buddy.write_text("invalid buddyinfo")
    with caplog.at_level(logging.INFO):
        startup_memory.log_startup_memory()
    assert "headroom is unlimited" in caplog.text
    assert "NUMA free 2 MiB+ blocks unavailable" in caplog.text


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
