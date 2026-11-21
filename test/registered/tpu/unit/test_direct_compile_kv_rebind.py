"""direct_compile hands donated KV state back without a TPU op.

The KV-pool inputs are donated, so each updated buffer comes back as an output
in the same HBM, and the artifact rebinds the live tensor to it. Over several
steps the pool must match eager CPU bit for bit, every holder of a pool tensor
must see the update, a replay must dispatch no TPU op, and a raiden host tier
registered before the steps must still move the current pages (it registers
the HBM once; the tensors' data_ptr changes with each rebind).
"""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=10, suite="stage-a-test-tpu-unit", chips=1)

import importlib.util
import unittest

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from sglang.test.test_utils import CustomTestCase

PAGES, WIDTH, ROWS = 64, 256, 8


def _no_tpu() -> bool:
    return importlib.util.find_spec("torch_tpu") is None or not hasattr(torch, "tpu")


class _Pool:
    def __init__(self, buffers):
        self.buffers = list(buffers)

    def get_state(self):
        return list(self.buffers)

    def set_state(self, state):
        self.buffers = list(state)


def _pool_buffers(device):
    return [
        torch.zeros((PAGES, WIDTH), dtype=torch.bfloat16, device=device),
        torch.zeros((PAGES, WIDTH), dtype=torch.uint8, device=device),
        torch.zeros((PAGES, WIDTH), dtype=torch.float32, device=device),
        torch.zeros((PAGES,), dtype=torch.int32, device=device),
        # Read but never written: comes back untouched.
        torch.full((PAGES,), 7, dtype=torch.int32, device=device),
    ]


def _step(buffers, rows, values):
    """One forward's KV writes: row scatters of several dtypes and a whole-buffer update."""
    bf16, byte, f32, counter, frozen = buffers
    bf16.index_copy_(0, rows, values.to(torch.bfloat16))
    byte.index_copy_(0, rows, (values & 255).to(torch.uint8))
    f32.index_add_(0, rows, values.to(torch.float32))
    counter.add_(1)
    return (values.sum(dim=-1) + frozen.index_select(0, rows)).to(torch.int32)


class _TpuDispatches(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.ops = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        leaves = [*args, *(kwargs or {}).values(), out]
        if any(isinstance(x, torch.Tensor) and x.device.type == "tpu" for x in leaves):
            self.ops.append(func.overloadpacket.__name__)
        return out


@unittest.skipIf(_no_tpu(), "needs torch_tpu")
class TestDirectCompileKVRebind(CustomTestCase):
    def test_donated_kv_state_matches_eager_without_dispatches(self):
        from sglang_tpu.compilation.direct_compile import direct_compile

        pool = _Pool(_pool_buffers("tpu"))
        held = pool.get_state()
        reference = _pool_buffers("cpu")
        compiled = direct_compile(
            lambda rows, values: _step(pool.buffers, rows, values),
            specialization_key=lambda rows, values: (tuple(values.shape),),
            kv_pool=pool,
        )
        gen = torch.Generator().manual_seed(466)
        for step in range(5):
            rows = torch.randperm(PAGES, generator=gen)[:ROWS]
            # Small integers: exact in bf16, so CPU and TPU casts agree bit for bit.
            values = torch.randint(-200, 200, (ROWS, WIDTH), generator=gen)
            rows_tpu, values_tpu = rows.to("tpu"), values.to("tpu")
            if step < 2:
                out = compiled(rows_tpu, values_tpu)
            else:
                with _TpuDispatches() as dispatches:
                    out = compiled(rows_tpu, values_tpu)
                self.assertEqual(dispatches.ops, [], f"step {step}")
            expected = _step(reference, rows, values)
            self.assertTrue(torch.equal(out.cpu(), expected), f"step {step}")
            for index, (live, ref) in enumerate(zip(pool.buffers, reference)):
                self.assertIs(live, held[index])
                self.assertTrue(
                    torch.equal(live.cpu(), ref), f"step {step} buf {index}"
                )
        self.assertEqual(len(compiled.cache), 1)

    @unittest.skipIf(
        importlib.util.find_spec("tpu_sync") is None, "needs the raiden engine"
    )
    def test_host_tier_registered_before_rebinds_moves_current_pages(self):
        from sglang_tpu.compilation.direct_compile import direct_compile
        from sglang_tpu.raiden_transfer import RaidenKVTransfer

        pool = _Pool(
            [
                torch.zeros((16, 128, WIDTH), dtype=torch.bfloat16, device="tpu")
                for _ in range(2)
            ]
        )
        pool.page_buffers = pool.get_state
        transfer = RaidenKVTransfer(pool, num_host_pages=2)

        def step(x):
            for index, buffer in enumerate(pool.buffers):
                buffer.add_((x + index).to(buffer.dtype).reshape(1, 1, 1))
                buffer[3].add_(1)
            return x + 1

        compiled = direct_compile(step, specialization_key=lambda x: (), kv_pool=pool)
        x = torch.ones((1,), device="tpu")
        with torch.no_grad():
            for _ in range(4):
                x = compiled(x)
        page3 = [buffer[3].cpu() for buffer in pool.buffers]
        transfer.backup([3], [0]).synchronize()
        transfer.load([0], [7]).synchronize()
        for buffer, expected in zip(pool.buffers, page3):
            self.assertTrue(torch.equal(buffer[7].cpu(), expected))


if __name__ == "__main__":
    unittest.main()
