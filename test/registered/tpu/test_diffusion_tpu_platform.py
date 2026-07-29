"""TPU CI entry for the sglang-diffusion (multimodal_gen) TPU platform.

Runs the diffusion TPU unit tests on the TPU CI slice: platform plugin
resolution and capability surface, the platform device/lifecycle hooks, the
TpuBackend compile-kwargs wiring, and the pallas attention backend — the
latter includes an on-TPU numeric parity test against the SDPA reference
when a TPU is present (it self-skips on CPU).
"""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=240, suite="stage-a-test-tpu-unit")

import unittest

from sglang.multimodal_gen.test.unit.test_platform_capability_hooks import (  # noqa: F401
    TestCapabilityHookConsumers,
    TestCapabilityHookDefaults,
)
from sglang.multimodal_gen.test.unit.test_platform_device_hooks import (  # noqa: F401
    TestInitWorkerDeviceEnv,
    TestPlatformMemoryHooks,
)
from sglang.multimodal_gen.test.unit.test_torch_compile_kwargs_tpu import (  # noqa: F401
    TestBuildTorchCompileKwargsTpu,
)
from sglang.multimodal_gen.test.unit.test_tpu_pallas_backend import (  # noqa: F401
    TestBackendInterface,
    TestNumericParity,
)
from sglang.multimodal_gen.test.unit.test_tpu_platform import (  # noqa: F401
    TestTpuPlatformPlugin,
    TestTpuPlatformSurface,
)

if __name__ == "__main__":
    unittest.main()
