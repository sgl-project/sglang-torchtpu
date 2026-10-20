"""CPU checks of the GLM-5.3-Flash megakernel's serving hooks."""

import unittest
from unittest import mock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.hooks import glm_flash_megakernel as hook

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _Kernel:
    def __init__(self, req):
        self.req = req

    def invalidate(self):
        self.req = None


class TestGlmFlashMegakernelHooks(CustomTestCase):
    def test_freeing_the_kernels_request_drops_it(self):
        mk = _Kernel(req=3)
        with mock.patch.object(hook, "_KERNELS", [mk]):
            hook.drop_freed_glm_flash_megakernel_request(object(), [5, 3])
        self.assertIsNone(mk.req)

    def test_freeing_another_request_keeps_it(self):
        mk = _Kernel(req=3)
        with mock.patch.object(hook, "_KERNELS", [mk]):
            hook.drop_freed_glm_flash_megakernel_request(object(), [5])
        self.assertEqual(mk.req, 3)


if __name__ == "__main__":
    unittest.main()
