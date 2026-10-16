"""Require native cache hits for the complete warmed small-K sampler path."""

import time
import unittest

import torch

from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.sampler import TpuSampler

register_tpu_ci(est_time=180, suite="stage-a-test-tpu-unit")


def check_top_k_sampler_warmup() -> dict:
    vocab_size = 202_048
    max_rows = 4
    generator = torch.Generator().manual_seed(113)
    cpu_probs = torch.softmax(
        torch.randn(max_rows, vocab_size, generator=generator), dim=-1
    )
    cases = []
    for rows in range(1, max_rows + 1):
        probs = cpu_probs[:rows].to("tpu")
        positions = torch.zeros(rows, dtype=torch.int64, device="tpu")
        for k in range(2, 65):
            for top_p, min_p in (
                (False, False),
                (True, False),
                (False, True),
                (True, True),
            ):
                info = SamplingBatchInfo(
                    temperatures=torch.ones(rows, 1, device="tpu"),
                    top_ks=torch.full((rows,), k, dtype=torch.int32, device="tpu"),
                    top_ps=torch.full((rows,), 0.9, device="tpu"),
                    min_ps=torch.full((rows,), 0.05, device="tpu"),
                    is_all_greedy=False,
                    is_any_greedy=False,
                    need_top_k_sampling=True,
                    need_top_p_sampling=top_p,
                    need_min_p_sampling=min_p,
                    uniform_top_k=k,
                    vocab_size=vocab_size,
                    device="tpu",
                )
                cases.append((probs, info, positions))
    torch.tpu.synchronize()
    sampler = object.__new__(TpuSampler)
    rng_state = torch.tpu.get_rng_state()
    start = time.perf_counter()
    sampler.warmup_small_top_k(
        max_batch_size=max_rows, vocab_size=vocab_size, device=torch.device("tpu")
    )
    warmup_seconds = time.perf_counter() - start
    assert torch.equal(rng_state, torch.tpu.get_rng_state())
    programs_before = len(torch.tpu._get_cache_stats().per_entry_stats)
    misses_before = torch.tpu._get_cache_misses()
    outputs = []
    for probs, info, positions in cases:
        output, _ = sampler._sample_from_probs(probs, info, positions, False)
        outputs.append(output.cpu())
    programs = len(torch.tpu._get_cache_stats().per_entry_stats) - programs_before
    misses = torch.tpu._get_cache_misses() - misses_before
    assert programs == 0, f"First use after warmup compiled {programs} new programs"
    assert misses == 0, f"First use after warmup had {misses} native cache misses"
    for (probs, info, _), ids in zip(cases, outputs):
        support = cpu_probs[: probs.shape[0]].topk(info.uniform_top_k, dim=-1).indices
        assert (ids[:, None] == support).any(dim=1).all()
    return {
        "cases": len(cases),
        "warmup_seconds": warmup_seconds,
        "post_warmup_programs": programs,
        "post_warmup_cache_misses": misses,
        "rng_preserved": True,
        "support_passed": True,
    }


class TestTopKSamplerWarmup(CustomTestCase):
    def test_every_small_k_and_logical_row_hits_native_cache(self):
        check_top_k_sampler_warmup()


if __name__ == "__main__":
    unittest.main()
