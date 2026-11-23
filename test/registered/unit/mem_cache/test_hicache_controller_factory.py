"""The platform chooses the HiCache controller class in ``create_for_backend``.

A platform's class replaces the one ``create_for_backend`` was called on; a
None answer keeps that class, so every in-tree io backend builds what it did.
"""

import unittest
from unittest.mock import MagicMock, patch

from sglang.srt.managers.cache_controller import HiCacheController
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _recording(base: type) -> type:
    class _Controller(base):
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    return _Controller


class TestCreateForBackend(CustomTestCase):
    # (base class, the platform seam create_for_backend asks)
    _CASES = (
        (HiCacheController, "get_hicache_controller_cls"),
        (HybridCacheController, "get_hybrid_cache_controller_cls"),
    )

    def test_platform_class_is_built_with_the_callers_arguments(self):
        for base, seam in self._CASES:
            with self.subTest(base=base.__name__):
                platform_cls = _recording(base)
                platform = MagicMock()
                getattr(platform, seam).return_value = platform_cls
                with patch("sglang.srt.platforms.current_platform", platform):
                    ctrl = base.create_for_backend("pool", 64, io_backend="dma")

                self.assertIs(type(ctrl), platform_cls)
                self.assertEqual(ctrl.args, ("pool", 64))
                self.assertEqual(ctrl.kwargs, {"io_backend": "dma"})
                getattr(platform, seam).assert_called_once_with("dma")

    def test_no_platform_class_keeps_the_called_class(self):
        for base, seam in self._CASES:
            with self.subTest(base=base.__name__):
                own_cls = _recording(base)
                platform = MagicMock()
                getattr(platform, seam).return_value = None
                with patch("sglang.srt.platforms.current_platform", platform):
                    ctrl = own_cls.create_for_backend("pool", io_backend="kernel")

                self.assertIs(type(ctrl), own_cls)
                self.assertEqual(ctrl.kwargs, {"io_backend": "kernel"})


if __name__ == "__main__":
    unittest.main()
