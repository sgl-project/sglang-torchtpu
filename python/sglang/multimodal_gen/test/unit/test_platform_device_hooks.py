import os
import unittest
from unittest.mock import MagicMock, patch

from sglang.multimodal_gen.runtime.platforms.cpu import CpuPlatform
from sglang.multimodal_gen.runtime.platforms.interface import Platform
from sglang.multimodal_gen.runtime.utils.perf_logger import MemorySnapshot

MIB = 1024**2


def _device_module_mock(*, available: bool) -> MagicMock:
    module = MagicMock()
    module.is_available.return_value = available
    module.memory_allocated.return_value = 1 * MIB
    module.memory_reserved.return_value = 2 * MIB
    module.max_memory_allocated.return_value = 3 * MIB
    module.max_memory_reserved.return_value = 4 * MIB
    return module


class TestPlatformMemoryHooks(unittest.TestCase):
    def test_default_snapshot_converts_bytes_to_mb(self):
        module = _device_module_mock(available=True)
        with patch("torch.get_device_module", return_value=module):
            snapshot = Platform.get_memory_snapshot()
        self.assertEqual(
            snapshot,
            MemorySnapshot(
                allocated_mb=1.0,
                reserved_mb=2.0,
                peak_allocated_mb=3.0,
                peak_reserved_mb=4.0,
            ),
        )

    def test_default_snapshot_is_zero_when_device_unavailable(self):
        module = _device_module_mock(available=False)
        with patch("torch.get_device_module", return_value=module):
            snapshot = Platform.get_memory_snapshot()
        self.assertEqual(snapshot.allocated_mb, 0.0)
        self.assertEqual(snapshot.peak_reserved_mb, 0.0)
        module.memory_allocated.assert_not_called()

    def test_default_empty_cache_and_reset_delegate_to_device_module(self):
        module = _device_module_mock(available=True)
        with patch("torch.get_device_module", return_value=module):
            Platform.empty_cache()
            Platform.reset_peak_memory_stats()
        module.empty_cache.assert_called_once_with()
        module.reset_peak_memory_stats.assert_called_once_with()

    def test_cpu_platform_has_safe_memory_hooks(self):
        # torch.cpu.is_available() is True but torch.cpu has none of the
        # memory APIs; the CPU overrides must not touch the device module.
        CpuPlatform.empty_cache()
        CpuPlatform.reset_peak_memory_stats()
        snapshot = CpuPlatform.get_memory_snapshot()
        self.assertEqual(snapshot.peak_allocated_mb, 0.0)

    def test_capture_memory_snapshot_delegates_to_current_platform(self):
        from sglang.multimodal_gen.runtime.utils import perf_logger

        sentinel = MemorySnapshot(
            allocated_mb=1.0,
            reserved_mb=1.0,
            peak_allocated_mb=1.0,
            peak_reserved_mb=1.0,
        )
        with patch.object(
            perf_logger.current_platform, "get_memory_snapshot", return_value=sentinel
        ):
            self.assertIs(perf_logger.capture_memory_snapshot(), sentinel)


class TestInitWorkerDeviceEnv(unittest.TestCase):
    def test_sets_device_and_rendezvous_env(self):
        module = _device_module_mock(available=True)
        env_keys = ("MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK", "RANK", "WORLD_SIZE")
        saved = {key: os.environ.get(key) for key in env_keys}
        try:
            with patch("torch.get_device_module", return_value=module):
                Platform.init_worker_device_env(
                    local_rank=2, rank=3, world_size=8, master_port=12355
                )
            module.set_device.assert_called_once_with(2)
            self.assertEqual(os.environ["MASTER_ADDR"], "localhost")
            self.assertEqual(os.environ["MASTER_PORT"], "12355")
            self.assertEqual(os.environ["LOCAL_RANK"], "2")
            self.assertEqual(os.environ["RANK"], "3")
            self.assertEqual(os.environ["WORLD_SIZE"], "8")
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


if __name__ == "__main__":
    unittest.main()
