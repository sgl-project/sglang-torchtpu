"""CPU CI entry for the multimodal_gen checkpoint weight reader tests."""

import sys

import pytest

from sglang.multimodal_gen.test.unit.test_weight_readers import (  # noqa: F401
    TestCapabilities,
    TestSafetensorsDevicePlacement,
    TestSelection,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
