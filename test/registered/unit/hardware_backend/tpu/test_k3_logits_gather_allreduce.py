"""Kimi-K3 logits gather by all-reduce (sglang_tpu/kimi_k3/logits_gather.py), without a TPU.

Simulates W tensor-parallel ranks in one process: each rank builds its
zero-padded [W, rows, n] buffer, a patched all-reduce sums the W buffers, and
every rank's result must equal the all-gather ``torch.cat(shards, dim=-1)`` bit
for bit. The collective itself (and the XLA all-gather failure this replaces)
needs TPU coverage: the 93L TP32 serving runs.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.test.test_utils import CustomTestCase
from sglang_tpu.envs import envs as tpu_envs
from sglang_tpu.kimi_k3 import logits_gather as lg

_ALL_REDUCE = "sglang.srt.distributed.tensor_model_parallel_all_reduce"


class _Holder(torch.nn.Module):
    def __init__(self, rank):
        super().__init__()
        self.register_buffer(
            "_tpu_logits_tp_rank", torch.tensor([rank], dtype=torch.int32)
        )


def _gatherer(rank, world, orig=None):
    return lg._AllReduceLogitsGatherer(orig, _Holder(rank), world)


class TestAllReduceGather(CustomTestCase):
    def _check(self, world, rows, n, dtype):
        g = torch.Generator().manual_seed(world * 1000 + rows)
        shards = [torch.randn(rows, n, generator=g).to(dtype) for _ in range(world)]
        expected = torch.cat(shards, dim=-1)
        # The W ranks' zero-padded buffers sum to the stacked shards.
        total = torch.stack(shards)
        for r in range(world):
            # Fake all-reduce: rank r must contribute only its own slot (zeros
            # elsewhere); it gets back the sum over all ranks.
            def fake_all_reduce(buf, r=r):
                own = torch.zeros_like(total)
                own[r] = shards[r]
                self.assertTrue(torch.equal(buf, own), f"rank={r} buffer")
                return total.clone()

            with mock.patch(_ALL_REDUCE, fake_all_reduce):
                out = _gatherer(r, world)(shards[r])
            self.assertEqual(out.dtype, dtype)
            self.assertTrue(
                torch.equal(out, expected), f"world={world} rows={rows} rank={r}"
            )

    def test_bit_identical_to_all_gather(self):
        for world in (2, 32):
            for rows in (1, 8):
                for dtype in (torch.bfloat16, torch.float32):
                    with self.subTest(world=world, rows=rows, dtype=dtype):
                        self._check(world, rows, 256, dtype)

    def test_non_2d_keeps_the_original_gatherer(self):
        orig = mock.Mock(return_value="orig")
        x = torch.zeros(4, dtype=torch.bfloat16)
        with mock.patch(_ALL_REDUCE) as all_reduce:
            self.assertEqual(_gatherer(0, 4, orig)(x), "orig")
        orig.assert_called_once_with(x)
        all_reduce.assert_not_called()


class TestBind(CustomTestCase):
    def _model(self, **lp_attrs):
        model = torch.nn.Module()
        model.dummy = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        model.logits_processor = torch.nn.Module()
        model.logits_processor._logits_gatherer = mock.Mock()
        attrs = {"do_tensor_parallel_all_gather": True, **lp_attrs}
        for name, value in attrs.items():
            setattr(model.logits_processor, name, value)
        return model

    def _bind(self, model, tp_size=4, tp_rank=2, enabled=True):
        parallel = SimpleNamespace(tp_size=tp_size, tp_rank=tp_rank)
        with (
            tpu_envs.SGLANG_TPU_LOGITS_GATHER_ALLREDUCE.override(enabled),
            mock.patch(
                "sglang.srt.runtime_context.get_parallel", return_value=parallel
            ),
        ):
            return lg.bind_logits_allreduce_gather(model)

    def test_default_on(self):
        self.assertTrue(tpu_envs.SGLANG_TPU_LOGITS_GATHER_ALLREDUCE.get())

    def test_binds_once_and_registers_the_rank(self):
        model = self._model()
        orig = model.logits_processor._logits_gatherer
        self.assertTrue(self._bind(model))
        lp = model.logits_processor
        self.assertIsInstance(lp._logits_gatherer, lg._AllReduceLogitsGatherer)
        self.assertIs(lp._logits_gatherer.orig, orig)
        self.assertEqual(lp._tpu_logits_tp_rank.tolist(), [2])
        self.assertFalse(self._bind(model))  # already bound

    def test_skips(self):
        cases = {
            "env off": (self._model(), dict(enabled=False)),
            "tp 1": (self._model(), dict(tp_size=1, tp_rank=0)),
            "no tp gather": (self._model(do_tensor_parallel_all_gather=False), {}),
            "attn tp group": (self._model(use_attn_tp_group=True), {}),
            "dp attention": (
                self._model(do_tensor_parallel_all_gather_dp_attn=True),
                {},
            ),
        }
        for name, (model, kwargs) in cases.items():
            with self.subTest(name):
                orig = model.logits_processor._logits_gatherer
                self.assertFalse(self._bind(model, **kwargs))
                self.assertIs(model.logits_processor._logits_gatherer, orig)


if __name__ == "__main__":
    unittest.main()
