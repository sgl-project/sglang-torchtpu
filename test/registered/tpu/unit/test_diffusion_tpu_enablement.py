"""TPU CI entry for the multimodal_gen TPU enablement tests."""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=120, suite="stage-a-test-tpu-unit", chips=1)

import unittest

from sglang.multimodal_gen.test.unit.test_tpu_enablement import (  # noqa: F401
    TestAnnotationsDoNotTouchTheAccelerator,
    TestCompileSitesAreDisabledOnTpu,
    TestCustomOpDispatchesTpu,
    TestGlooDefaultGroupIsNotDeviceBound,
    TestLayernormDoesNotImportTritonNormOnTpu,
    TestTpuHostBehaviour,
    TestVariableAnnotations,
)

if __name__ == "__main__":
    unittest.main()
