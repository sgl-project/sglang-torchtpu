"""Both Kimi-K3 programs in one engine: the prefill program shares the decode
program's routed experts and identical dense families, and the worker sends
each batch to its program."""

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.k3_programs import sglang_bridge as bridge
from sglang_tpu.k3_programs import sglang_colocated as colocated
from sglang_tpu.k3_programs import sglang_worker as worker
from sglang_tpu.k3_programs.kimi_k3 import pools

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

STEP = 8192


class StubRankLoader:
    """A rank loader whose prefill iterator reads expert parts through
    ``_experts_at`` (as the real one at any expert parallelism)."""

    def __init__(self):
        self.expert_reads = 0

    def _experts_at(self, *args, **kwargs):
        for family in colocated.EXPERT_FAMILIES:
            self.expert_reads += 1
            yield ("moe", f"{family}@0/1", np.zeros(4))

    def iter_prefill_rank(self, presharded, cache, rank, **kwargs):
        yield from self._experts_at()
        yield ("dense", "latent_down", np.ones(2))
        # A family both programs read alike, and one the prefill rebuilds.
        yield ("kda", "k_projection", np.ones(2))
        yield ("kda", "k_o", np.full(2, 5.0))


class StubPrefillProgram:
    def __init__(self):
        self._rank_loader = StubRankLoader()
        self.weights = {}
        self.shared_experts = False

    def load_weights(self, presharded):
        for group, name, array in self._rank_loader.iter_prefill_rank(
            presharded, None, 0
        ):
            self.weights.setdefault(group, {})[name.split("@")[0]] = torch.from_numpy(
                array
            )
        for family in colocated.EXPERT_FAMILIES:
            assert family in self.weights["moe"], family


def _config(ep, decode_ep=None):
    return mock.patch(
        "sglang_tpu.k3_programs.kimi_k3.config.active",
        return_value=SimpleNamespace(
            prefill_expert_parallel=ep,
            decode_expert_parallel=ep if decode_ep is None else decode_ep,
        ),
    )


class TestSharedExperts(CustomTestCase):
    def setUp(self):
        self.experts = {
            f: torch.full((3,), i, dtype=torch.float32)
            for i, f in enumerate(colocated.EXPERT_FAMILIES)
        }
        self.dense = {
            name: torch.full((2,), 7.0)
            for names in colocated.SHARED_DENSE.values()
            for name in names
        }
        self.decoder = SimpleNamespace(
            ops=SimpleNamespace(
                weights=dict(self.experts, **self.dense, k_o=torch.zeros(2)),
                load_weights=mock.Mock(),
            )
        )
        self.prefiller = SimpleNamespace(program=StubPrefillProgram())

    def test_prefill_takes_the_decoders_experts(self):
        for ep in colocated.SHARED_EXPERT_PARALLEL:
            with self.subTest(ep=ep):
                self.setUp()
                self._check_shared(ep)

    def _check_shared(self, ep):
        loader = self.prefiller.program._rank_loader
        with _config(ep):
            colocated.share_experts(self.prefiller, self.decoder)
        self.assertTrue(self.prefiller.program.shared_experts)
        self.decoder.ops.load_weights.assert_called_once()
        self.prefiller.program.load_weights("/presharded")
        moe = self.prefiller.program.weights["moe"]
        for family in colocated.EXPERT_FAMILIES:
            self.assertIs(moe[family], self.experts[family])
        # No expert file was read for the prefill; its own dense weights were.
        self.assertEqual(loader.expert_reads, 0)
        self.assertIn("latent_down", self.prefiller.program.weights["dense"])
        # The shared dense families are the decoder's tensors; the output
        # projection the prefill rebuilds is its own.
        weights = self.prefiller.program.weights
        for group, names in colocated.SHARED_DENSE.items():
            for name in names:
                self.assertIs(weights[group][name], self.dense[name])
        self.assertEqual(weights["kda"]["k_o"].tolist(), [5.0, 5.0])
        # The loader's expert jobs are back after the load.
        self.assertEqual(
            len(list(loader._experts_at())), len(colocated.EXPERT_FAMILIES)
        )

    def test_only_at_equal_expert_parallelism(self):
        for ep, decode_ep in ((16, 8), (8, 16), (32, 32)):
            with self.subTest(ep=ep, decode_ep=decode_ep):
                with (
                    _config(ep, decode_ep),
                    self.assertRaisesRegex(ValueError, f"not {ep} and {decode_ep}"),
                ):
                    colocated.share_experts(self.prefiller, self.decoder)


class TestColocatedDispatch(CustomTestCase):
    def test_extend_to_prefill_decode_to_decode(self):
        plan = bridge.UnifiedPoolPlan(40, 4, pools.cache_layout("auto"))
        prefill = mock.Mock(spec=worker.PrefillSteps)
        prefill.step.return_value = np.array([42])
        prefill.clock = worker.StepClock("prefill")
        decode = mock.Mock(spec=worker.DecodeSteps)
        decode.plan = plan
        decode.clock = worker.StepClock("decode")
        decode.step.return_value = (
            np.array([1, 2, -1, -1, -1, -1, -1, -1]),
            np.array([2], np.int32),
        )
        steps = colocated.ColocatedSteps(prefill, decode)
        r2t = torch.zeros((2, 1 << 20), dtype=torch.int32)
        target = SimpleNamespace(
            model_runner=None,
            get_memory_pool=lambda: (
                SimpleNamespace(req_to_token=r2t, mamba_pool=None),
                None,
            ),
        )
        worker.K3ProgramWorker.build_steps = lambda target_worker: steps
        self.addCleanup(setattr, worker.K3ProgramWorker, "build_steps", None)
        w = worker.K3ProgramWorker(None, 0, 0, target)
        w.alloc_memory_pool()
        req = SimpleNamespace(rid="r", grammar=None)
        batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            reqs=[req],
            prefix_lens=[0],
            extend_lens=[100],
            seq_lens=torch.tensor([100]),
            return_logprob=False,
        )
        self.assertEqual(
            w.forward_batch_generation(batch).next_token_ids.tolist(), [42]
        )
        prefill.step.assert_called_once()
        batch.forward_mode = ForwardMode.DECODE
        result = w.forward_batch_generation(batch)
        decode.step.assert_called_once()
        self.assertEqual(result.accept_lens.tolist(), [2])
        w.clear_cache_pool()
        prefill.forget_all.assert_called_once()
        decode.forget_all.assert_called_once()


if __name__ == "__main__":
    unittest.main()
