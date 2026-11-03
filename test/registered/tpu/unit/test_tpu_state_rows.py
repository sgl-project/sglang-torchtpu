"""The Pallas state-row read/write ops match index_select / the CPU row write.

Decode reads and writes a few conv-state rows per stream; the ops must touch
only those rows, and under direct_compile the pool must stay on its buffer
(donated, updated in place) whatever its slot count.
"""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=60, suite="stage-a-test-tpu-unit", chips=1)

import unittest

import torch

from sglang.test.test_utils import CustomTestCase

_TPU_AVAILABLE = hasattr(torch, "accelerator") and torch.accelerator.is_available()


class _Pool:
    def __init__(self, tensors):
        self.tensors = list(tensors)

    def get_state(self):
        return list(self.tensors)

    def set_state(self, state):
        self.tensors = list(state)


@unittest.skipUnless(_TPU_AVAILABLE, "requires TPU")
class TestTpuStateRows(CustomTestCase):
    def test_read_matches_index_select(self):
        from sglang_tpu.mamba_pool import read_state_rows

        state = torch.randn(41, 3, 1, 256)
        rows = torch.tensor([7, 0, 40, 7, 3, 0, 0, 0], dtype=torch.int32)
        out = read_state_rows(state.to("tpu"), rows.to("tpu")).cpu()
        torch.testing.assert_close(out, state[rows.long()], rtol=0, atol=0)

    def test_write_touches_only_live_rows(self):
        from sglang_tpu.mamba_pool import write_state_rows

        state = torch.randn(41, 3, 1, 256)
        rows = torch.tensor([5, 0, 2, 0, 17, 9, 17, 33])
        live = torch.tensor([True, False, True, False, True, True, False, True])
        new = torch.randn(rows.shape[0], 3 * 256)
        expected = state.clone()
        write_state_rows(expected, rows, new, live)
        device_state = state.to("tpu")
        write_state_rows(device_state, rows.to("tpu"), new.to("tpu"), live.to("tpu"))
        torch.testing.assert_close(device_state.cpu(), expected, rtol=0, atol=0)

    def test_compiled_read_write_updates_pool_in_place(self):
        from sglang_tpu.compilation.direct_compile import direct_compile
        from sglang_tpu.mamba_pool import read_state_rows, write_state_rows

        for slots in (17, 181):
            pool = _Pool([torch.randn(slots, 3, 1, 256).to("tpu") for _ in range(2)])
            reference = [t.cpu() for t in pool.tensors]

            def step(rows, live, x):
                for state in pool.tensors:
                    old = read_state_rows(state, rows).reshape(rows.shape[0], 3, 256)
                    new = torch.cat((old[:, 1:], x[:, None]), 1)
                    write_state_rows(state, rows, new, live)
                return x.sum().reshape(1)

            compiled = direct_compile(
                step, specialization_key=lambda *a, **k: slots, kv_pool=pool
            )
            rows = torch.tensor([3, 1, 0, 0], dtype=torch.int32)
            live = torch.tensor([True, True, False, False])
            ptrs = [t.data_ptr() for t in pool.tensors]
            for value in (1.0, 2.0):
                x = torch.full((4, 256), value)
                compiled(rows.to("tpu"), live.to("tpu"), x.to("tpu")).cpu()
                for ref in reference:
                    old = ref[rows.long()].reshape(4, 3, 256)
                    new = torch.cat((old[:, 1:], x[:, None]), 1)
                    write_state_rows(ref, rows, new, live)
            for tensor, ref in zip(pool.tensors, reference, strict=True):
                torch.testing.assert_close(tensor.cpu(), ref, rtol=0, atol=0)
            # Donated: the pool keeps its buffer instead of a fresh copy per call.
            self.assertEqual([t.data_ptr() for t in pool.tensors], ptrs)


if __name__ == "__main__":
    unittest.main()
