"""Platform decisions must trace without graph breaks or stale scoped facts."""

import unittest
from unittest.mock import patch

import torch

from sglang.srt.runtime_context import get_platform, override_platform
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestPlatformContextCompile(CustomTestCase):
    def test_fullgraph_platform_override_and_restore(self):
        # R2 model warmup failed before kernels on this platform-fact read.
        # Reuse one compiled callable across scopes to catch stale constants too.
        def forward(x):
            return x + 3 if get_platform().is_blackwell else x - 2

        compiled = torch.compile(forward, backend="eager", fullgraph=True)
        x = torch.arange(4, dtype=torch.float32)
        native = get_platform().is_blackwell
        torch.testing.assert_close(compiled(x), forward(x))
        with override_platform(is_blackwell=not native):
            torch.testing.assert_close(compiled(x), forward(x))
            with override_platform(is_blackwell=native):
                torch.testing.assert_close(compiled(x), forward(x))
            torch.testing.assert_close(compiled(x), forward(x))
        torch.testing.assert_close(compiled(x), forward(x))

    def test_distinct_native_probes_in_one_graph(self):
        # Different arguments to the constant resolver must not alias results.
        # Mock only hardware discovery, not PlatformContext or its overrides.
        from sglang.srt.utils import common

        def forward(x):
            platform = get_platform()
            if platform.is_blackwell:
                x = x + 10
            if platform.is_sm100:
                x = x + 20
            if platform.device_sm == 90:
                x = x + 30
            return x

        with (
            patch.object(common, "is_blackwell_supported", return_value=False),
            patch.object(common, "is_sm100_supported", return_value=True),
            patch.object(common, "get_device_sm", return_value=90),
        ):
            x = torch.zeros(4)
            compiled = torch.compile(forward, backend="eager", fullgraph=True)
            torch.testing.assert_close(compiled(x), x + 50)


if __name__ == "__main__":
    unittest.main()
