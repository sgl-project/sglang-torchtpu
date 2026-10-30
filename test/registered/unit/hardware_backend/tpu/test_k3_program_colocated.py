"""Both Kimi-K3 programs in one engine: the prefill program shares the decode
program's routed experts, and the worker sends each batch to its program."""

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
    ``_expert_jobs`` (as the real one at prefill expert parallelism 8)."""

    def __init__(self):
        self.expert_reads = 0

    def _expert_jobs(self, *args, **kwargs):
        for family in colocated.EXPERT_FAMILIES:
            self.expert_reads += 1
            yield ("moe", f"{family}@0/1", np.zeros(4))

    def iter_prefill_rank(self, presharded, cache, rank, **kwargs):
        yield from self._expert_jobs()
        yield ("dense", "latent_down", np.ones(2))


class StubPrefillProgram:
    def __init__(self):
        self._rank_loader = StubRankLoader()
        self.weights = {}

    def load_weights(self, presharded):
        for group, name, array in self._rank_loader.iter_prefill_rank(
            presharded, None, 0
        ):
            self.weights.setdefault(group, {})[name.split("@")[0]] = torch.from_numpy(
                array
            )
        for family in colocated.EXPERT_FAMILIES:
            assert family in self.weights["moe"], family


def _config(ep):
    return mock.patch(
        "sglang_tpu.k3_programs.kimi_k3.config.active",
        return_value=SimpleNamespace(prefill_expert_parallel=ep),
    )


class TestSharedExperts(CustomTestCase):
    def setUp(self):
        self.experts = {
            f: torch.full((3,), i, dtype=torch.float32)
            for i, f in enumerate(colocated.EXPERT_FAMILIES)
        }
        self.decoder = SimpleNamespace(
            ops=SimpleNamespace(
                weights=dict(self.experts, dense=torch.zeros(1)),
                load_weights=mock.Mock(),
            )
        )
        self.prefiller = SimpleNamespace(program=StubPrefillProgram())

    def test_prefill_takes_the_decoders_experts(self):
        loader = self.prefiller.program._rank_loader
        with _config(8):
            colocated.share_experts(self.prefiller, self.decoder)
        self.decoder.ops.load_weights.assert_called_once()
        self.prefiller.program.load_weights("/presharded")
        moe = self.prefiller.program.weights["moe"]
        for family in colocated.EXPERT_FAMILIES:
            self.assertIs(moe[family], self.experts[family])
        # No expert file was read for the prefill; its own dense weights were.
        self.assertEqual(loader.expert_reads, 0)
        self.assertIn("latent_down", self.prefiller.program.weights["dense"])
        # The loader's expert jobs are back after the load.
        self.assertEqual(
            len(list(loader._expert_jobs())), len(colocated.EXPERT_FAMILIES)
        )

    def test_only_at_prefill_expert_parallelism_eight(self):
        with _config(16), self.assertRaisesRegex(ValueError, "parallelism 8, not 16"):
            colocated.share_experts(self.prefiller, self.decoder)


class TestColocatedDispatch(CustomTestCase):
    def test_extend_to_prefill_decode_to_decode(self):
        plan = bridge.UnifiedPoolPlan(40, 4, pools.cache_layout("auto"))
        prefill = mock.Mock(spec=worker.PrefillSteps)
        prefill.step.return_value = np.array([42])
        decode = mock.Mock(spec=worker.DecodeSteps)
        decode.plan = plan
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
