"""TpuShortConvPool keeps each PLE layer's short-conv state as its own root
tensor, slot-major, so Raiden can register it as a pool (no folded views)."""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=5, suite="stage-a-test-tpu-unit", chips=1)

import unittest

import torch

from sglang_tpu.ple_state_pool import TpuShortConvPool


class TestTpuShortConvPool(unittest.TestCase):
    def _pool(self, layer_ids=(1, 5)):
        return TpuShortConvPool(
            size=4,
            state_shape=(6, 3),
            layer_ids=list(layer_ids),
            dtype=torch.float32,
            device="cpu",
        )

    def test_transfer_entries_are_root_slot_major_tensors(self):
        pool = self._pool()
        entries = list(pool.iter_transfer_state_entries())
        self.assertEqual([(n, lid) for n, _, _, lid in entries], [("ple_short_conv", 1), ("ple_short_conv", 5)])
        for _, tensor, axis, _ in entries:
            self.assertIsNone(tensor._base)
            self.assertIsNone(axis)
            self.assertEqual(tuple(tensor.shape), (5, 6, 3))
        self.assertIs(pool.layer_cache(5), pool.layer_states[1])

    def test_slot_reset_copy_and_cpu_round_trip(self):
        pool = self._pool()
        for i, state in enumerate(pool.layer_states):
            state.copy_(torch.arange(state.numel(), dtype=torch.float32).view_as(state) + 100 * i)
        pool.copy_slots(torch.tensor([1]), torch.tensor([3]))
        for state in pool.layer_states:
            self.assertTrue(torch.equal(state[3], state[1]))
        pool.reset_slots(torch.tensor([1, 2]))
        for state in pool.layer_states:
            self.assertEqual(float(state[1:3].abs().sum()), 0.0)
        saved = pool.get_cpu_slots(torch.tensor([3, 4]))
        self.assertEqual(tuple(saved.shape), (2, 2, 6, 3))  # core's [layers, slots, ...]
        pool.reset_slots(torch.tensor([3, 4]))
        pool.load_cpu_slots(saved, torch.tensor([3, 4]))
        self.assertTrue(torch.equal(pool.get_cpu_slots(torch.tensor([3, 4])), saved))

    def test_disabled_without_ple_layers(self):
        pool = self._pool(layer_ids=())
        self.assertFalse(pool.enabled)
        self.assertEqual(list(pool.iter_transfer_state_entries()), [])
        self.assertIsNone(pool.get_cpu_slots(torch.tensor([1])))


if __name__ == "__main__":
    unittest.main()
