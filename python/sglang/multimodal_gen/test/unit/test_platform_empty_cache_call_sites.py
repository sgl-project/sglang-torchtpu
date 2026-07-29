"""Regression tests: converted empty_cache call sites resolve and execute.

Catches the class of bug where a call site is converted to
``current_platform.empty_cache()`` without importing ``current_platform``
(NameError only surfaces on the memory-release runtime path, not at import).
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch.nn as nn


class TestEmptyCacheCallSites(unittest.TestCase):
    def test_component_manager_release_path_executes(self):
        from sglang.multimodal_gen.runtime.managers.memory_managers import (
            component_manager as cm,
        )

        manager = object.__new__(cm.ComponentResidencyManager)
        use = SimpleNamespace(memory_intensive=True)
        module = nn.Linear(2, 2)  # on CPU => "released from device"
        platform = MagicMock()
        with patch.object(cm, "current_platform", platform):
            cm.ComponentResidencyManager._empty_cache_after_large_release(
                manager,
                use=use,
                strategy=None,
                module=module,
                was_on_cuda=True,
            )
        platform.empty_cache.assert_called_once_with()

    def test_lora_pipeline_references_platform(self):
        from sglang.multimodal_gen.runtime.pipelines_core import lora_pipeline

        self.assertTrue(hasattr(lora_pipeline, "current_platform"))


if __name__ == "__main__":
    unittest.main()
