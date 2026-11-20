import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from sglang.test.ci.ci_register import register_tpu_ci
from sglang_tpu.fused_eager import fused_eager_ops, submit
from sglang_tpu.sampler import sample_exponential_race

register_tpu_ci(est_time=60, suite="stage-a-test-tpu-unit", chips=1)

# A ~200K vocabulary row: rows of this width are where a fused softmax was seen
# to reduce in another order, so the regions keep reductions out (fused_eager.py).
_VOCAB = 202048


def _bits(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.cpu()
    return tensor.view(torch.int32) if tensor.dtype == torch.float32 else tensor


def _softcap_(logits, scale, cap):
    # A model's in-place logit scale and tanh softcap.
    logits.mul_(scale)
    logits.div_(cap).tanh_().mul_(cap)
    return logits


class TestFusedEagerOps(unittest.TestCase):
    def test_regions_keep_the_values_of_the_per_op_programs(self):
        device = torch.device("tpu")
        for rows in (1, 3, 77, 100, 128):
            generator = torch.Generator().manual_seed(rows)
            logits = (torch.randn(rows, _VOCAB, generator=generator) * 12).to(device)

            want = _softcap_(logits.clone(), 0.5, 30.0)
            with fused_eager_ops() as fused:
                got = _softcap_(logits.clone(), 0.5, 30.0)
            self.assertTrue(fused)
            submit((got,))
            self.assertTrue(torch.equal(_bits(got), _bits(want)), rows)

            probs = torch.softmax(want, dim=-1)
            # The race defers itself; the per-op draw runs with fusion off.
            torch.tpu.manual_seed(rows)
            fused_ids = sample_exponential_race(probs)
            fused_next = torch.rand(8, device=device)
            torch.tpu.manual_seed(rows)
            with patch(
                "sglang_tpu.sampler.fused_eager_ops",
                lambda enabled=True: fused_eager_ops(False),
            ):
                eager_ids = sample_exponential_race(probs)
            eager_next = torch.rand(8, device=device)
            self.assertTrue(torch.equal(fused_ids.cpu(), eager_ids.cpu()), rows)
            # The generator advanced the same way.
            self.assertTrue(torch.equal(fused_next.cpu(), eager_next.cpu()), rows)

            lengths = torch.randint(1, 1 << 20, (rows,), generator=generator)
            fields = (lengths.to(device), lengths.int().to(device))
            with fused_eager_ops():
                padded = [F.pad(field, (0, 128 - rows)) for field in fields]
            submit(padded)
            for field, got in zip(fields, padded):
                want = F.pad(field.cpu(), (0, 128 - rows))
                self.assertTrue(torch.equal(got.cpu(), want), rows)

    def test_no_region_while_tracing_or_already_deferring(self):
        from torch_tpu._internal import execution_mode

        with patch("torch.compiler.is_compiling", return_value=True):
            with fused_eager_ops() as fused:
                self.assertFalse(fused)
        with fused_eager_ops(enabled=False) as fused:
            self.assertFalse(fused)
        with execution_mode.set_eager_mode(execution_mode.EagerMode.DEFER_AND_FUSE):
            with fused_eager_ops() as fused:
                self.assertFalse(fused)
        # Host tensors are skipped.
        submit((torch.zeros(2), None, 3))


if __name__ == "__main__":
    unittest.main()
