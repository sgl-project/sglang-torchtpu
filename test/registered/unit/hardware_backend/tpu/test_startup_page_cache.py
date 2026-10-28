"""CPU checks for startup file cleanup."""

import logging
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import pytest

from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu import page_cache
from sglang_tpu.envs import envs as tpu_envs

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
        page_cache,
        "_advise_drop_file_cache",
        lambda p: advised.append(p) or p.stat().st_size,
    )
    page_cache.release_compile_artifacts()
    assert sorted(advised) == paths
    assert all(p.read_bytes() == b"payload" for p in paths)
    monkeypatch.setenv("SGLANG_TPU_RELEASE_COMPILE_ARTIFACTS", "0")
    advised.clear()
    page_cache.release_compile_artifacts()
    assert advised == []


def test_checkpoint_paths_are_deduplicated_and_do_not_recurse(monkeypatch, tmp_path):
    (tmp_path / "a.safetensors").touch()
    (tmp_path / "config.json").touch()
    draft = tmp_path / "draft"
    draft.mkdir()
    (draft / "b.safetensors").touch()
    advised = []
    monkeypatch.setattr(
        page_cache, "_advise_drop_file_cache", lambda p: advised.append(p) or 0
    )
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


def test_startup_discovers_files_only_on_the_host_leader(tmp_path):
    target = tmp_path / "target"
    draft = tmp_path / "draft"
    for directory in (target, draft):
        directory.mkdir()
        (directory / "weights.safetensors").touch()
    advised = []
    group = Mock()
    with (
        get_context().override_server_args(
            model_path=str(target), speculative_draft_model_path=str(draft)
        ),
        get_parallel().override(world_group=group),
        tpu_envs.SGLANG_TPU_RELEASE_COMPILE_ARTIFACTS.override(False),
        patch.object(page_cache, "resolve_compilations", Mock()) as join_writers,
        patch.object(
            page_cache.os,
            "posix_fadvise",
            side_effect=lambda fd, *_: advised.append(os.fstat(fd).st_ino),
        ),
    ):
        group.local_rank = 1
        page_cache.release_startup_page_cache()
        assert advised == []
        group.local_rank = 0
        page_cache.release_startup_page_cache()
    assert sorted(advised) == sorted(
        (directory / "weights.safetensors").stat().st_ino
        for directory in (target, draft)
    )
    assert join_writers.call_count == 2
    assert group.barrier.call_count == 2


@pytest.mark.parametrize(
    "env_name",
    ["TORCH_TPU_TIER2_COMPILATION_CACHE", "TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE"],
)
def test_compile_cache_disabled_skips(monkeypatch, tmp_path, caplog, env_name):
    (tmp_path / "disabled").mkdir()
    monkeypatch.setattr(page_cache, "_CACHE_ROOT", tmp_path)
    monkeypatch.delenv("TORCH_TPU_TIER2_COMPILATION_CACHE", raising=False)
    monkeypatch.delenv("TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE", raising=False)
    # Runner-set Python cache dirs are roots release_compile_artifacts also scans.
    monkeypatch.delenv("TORCHINDUCTOR_CACHE_DIR", raising=False)
    monkeypatch.delenv("JAX_COMPILATION_CACHE_DIR", raising=False)
    monkeypatch.setenv(env_name, "disabled")
    monkeypatch.setattr(
        page_cache, "_filesystem_type", lambda _: pytest.fail("cache disabled")
    )
    with caplog.at_level(logging.INFO):
        page_cache.release_compile_artifacts()
    assert not caplog.records


def test_disappeared_cache_paths_are_skipped(tmp_path):
    root = tmp_path / "active"
    root.mkdir()
    target = root / "program"
    target.write_bytes(b"payload")
    paths = page_cache._cache_files(root)
    assert paths == [target]

    target.unlink()
    root.rmdir()
    assert page_cache._cache_files(root) == []
    assert page_cache._advise_files(paths) == 0


def test_tmpfs_mount_resolution_and_release(monkeypatch, tmp_path):
    if not Path("/dev/shm").is_dir():
        pytest.skip("requires /dev/shm")
    disk = tmp_path / "disk cache"
    disk.mkdir()
    with TemporaryDirectory(dir="/dev/shm") as directory:
        root = Path(directory)
        if page_cache._filesystem_type(root) not in ("tmpfs", "ramfs"):
            pytest.skip("/dev/shm is not memory-backed")
        link = root / "disk link"
        link.symlink_to(disk, target_is_directory=True)
        assert page_cache._filesystem_type(link) == page_cache._filesystem_type(disk)

        (root / "active").mkdir()
        (root / "active/program.bin").write_bytes(b"payload")
        monkeypatch.setattr(page_cache, "_CACHE_ROOT", root)
        monkeypatch.setenv("TORCH_TPU_TIER2_COMPILATION_CACHE", "active")
        with patch.object(page_cache.os, "posix_fadvise") as advise:
            page_cache.release_compile_artifacts()
        advise.assert_not_called()
        assert not (root / "active/program.bin").exists()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
