"""rebind_tensor: a base tensor takes its update's storage; anything else copies."""

import unittest

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.compilation.tensor_rebind import rebind_tensor

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestRebindTensor(CustomTestCase):
    def test_base_tensor_takes_update_storage(self):
        live = torch.zeros(4, 3)
        holder = [live]
        update = live.clone() + 1
        rebind_tensor(live, update)
        self.assertIs(holder[0], live)
        self.assertEqual(live.data_ptr(), update.data_ptr())
        torch.testing.assert_close(holder[0], torch.ones(4, 3))

    def test_view_copies(self):
        base = torch.zeros(4, 3)
        view = base[1:3]
        rebind_tensor(view, torch.ones(2, 3))
        torch.testing.assert_close(base[1:3], torch.ones(2, 3))
        torch.testing.assert_close(base[0], torch.zeros(3))

    def test_base_with_live_view_copies(self):
        live = torch.zeros(4, 3)
        view = live.view(12)
        update = torch.ones(4, 3)
        rebind_tensor(live, update)
        self.assertNotEqual(live.data_ptr(), update.data_ptr())
        torch.testing.assert_close(view, torch.ones(12))

    def test_inference_flag_mismatch_copies(self):
        live = torch.zeros(4)
        with torch.inference_mode():
            update = torch.ones(4)
        rebind_tensor(live, update)
        self.assertFalse(live.is_inference())
        torch.testing.assert_close(live, torch.ones(4))


if __name__ == "__main__":
    unittest.main()
