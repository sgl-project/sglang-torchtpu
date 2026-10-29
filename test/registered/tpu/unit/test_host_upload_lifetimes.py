"""Check temporary host uploads against the native TPU backend."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch_tpu  # noqa: F401

from sglang.srt.managers import overlap_utils
from sglang.srt.managers.overlap_utils import RelayPayload
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_tpu_ci
from sglang_tpu.compilation.bucketer import Bucketer
from sglang_tpu.compilation.decoder_compile_policy import DecoderTpuCompilePolicy
from sglang_tpu.experimental.sampling.sampling_batch_info import HostPaddedSamplingInfo
from sglang_tpu.host_metadata_cache import HostMetadataCache
from sglang_tpu.host_padding_relay import HostPaddingFutureMap
from sglang_tpu.sampler import sampling_params_to_device

register_tpu_ci(
    est_time=30,
    suite="stage-a-test-tpu-unit",
    chips=1,
    labels=["singlehost", "infra", "compile"],
)


def sampling_info(rows, info_type=SamplingBatchInfo, iteration=0):
    return info_type(
        temperatures=torch.arange(rows, dtype=torch.float32).reshape(-1, 1) + iteration,
        top_ps=torch.full((rows,), 0.75),
        top_ks=torch.full((rows,), 23 + iteration, dtype=torch.int32),
        min_ps=torch.full((rows,), 0.25),
        sampling_seed=torch.arange(rows, dtype=torch.int64) + iteration,
        is_all_greedy=False,
        is_any_greedy=False,
        need_top_p_sampling=True,
        need_top_k_sampling=True,
        need_min_p_sampling=True,
        vocab_size=1024,
        device=torch.device("tpu:0"),
        sampling_params_on_cpu=True,
    )


class TestHostUploadLifetimes(unittest.TestCase):
    @torch.no_grad()
    def test_forward_input_upload_preserves_replaced_fields(self):
        policy = object.__new__(DecoderTpuCompilePolicy)
        policy.model_runner = SimpleNamespace(device=torch.device("tpu:0"))
        fields = (
            "input_ids",
            "positions",
            "out_cache_loc",
            "seq_lens",
            "req_pool_indices",
        )
        for padded in (False, True):
            for width in (1, 2, 4, 8, 128, 4096):
                with self.subTest(padded=padded, width=width):
                    expected = {
                        name: torch.arange(width, dtype=torch.int64) + offset
                        for offset, name in enumerate(fields)
                    }
                    batch = ForwardBatch(
                        forward_mode=ForwardMode.EXTEND,
                        batch_size=width,
                        seq_lens_sum=int(expected["seq_lens"].sum()),
                        **{name: value.clone() for name, value in expected.items()},
                    )
                    if padded:
                        batch.sampling_info = sampling_info(
                            width, HostPaddedSamplingInfo
                        )
                        policy._upload_host_padded_inputs(batch)
                    else:
                        policy._upload_host_inputs(batch)
                    for name, value in expected.items():
                        self.assertTrue(
                            torch.equal(getattr(batch, name).cpu(), value), name
                        )

    @torch.no_grad()
    def test_sampling_upload_preserves_live_and_padded_rows(self):
        for iteration in range(40):
            for width in (1, 2, 4, 8, 128, 4096):
                info = sampling_info(width, iteration=iteration)
                uploaded = sampling_params_to_device(info, padded_size=width * 2)
                for name, fill in (
                    ("temperatures", 1),
                    ("top_ps", 1),
                    ("top_ks", 1),
                    ("min_ps", 0),
                    ("sampling_seed", 0),
                ):
                    with self.subTest(width=width, name=name):
                        value = getattr(info, name)
                        expected = value.new_full((width * 2, *value.shape[1:]), fill)
                        expected[:width] = value
                        self.assertTrue(
                            torch.equal(getattr(uploaded, name).cpu(), expected)
                        )

    @torch.no_grad()
    def test_metadata_replacement_preserves_inflight_copies(self):
        cache = HostMetadataCache()
        for iteration in range(40):
            for width in (256, 1024, 4096, 65536):
                pending = [
                    cache.get_by_value(
                        name="pages",
                        value=torch.full((width,), i, dtype=torch.int32),
                        device="tpu:0",
                    )
                    for i in range(16)
                ]
                for i, resident in enumerate(pending):
                    observed = resident.cpu()
                    self.assertTrue(
                        torch.equal(observed, torch.full_like(observed, i)),
                        f"Metadata mismatch at iteration {iteration}, width {width}, copy {i}",
                    )

    @torch.no_grad()
    def test_repeated_stash_gather_preserves_tokens(self):
        bucketer = Bucketer(name="Bs", buckets=[1, 2, 4, 8, 16, 32, 64, 128, 256])
        pool = SimpleNamespace(
            req_to_token=torch.zeros((4096, 1), dtype=torch.int32),
            _tpu_relay_bucketer=bucketer,
        )
        with patch.object(overlap_utils, "_DEBUG_ASSERT", True):
            relay = HostPaddingFutureMap(
                torch.device("tpu:0"), SpeculativeAlgorithm.NONE, pool
            )
            # Reuse request slots after gather poisons them, with both full and
            # partially filled buckets. Lost index uploads expose poisoned rows.
            for iteration in range(40):
                for width in bucketer.buckets:
                    live = min(width, 128)
                    indices = torch.arange(1, live + 1, dtype=torch.int64)
                    expected = torch.arange(width, dtype=torch.int64) + iteration
                    relay.stash(indices, RelayPayload(expected.to(relay.device)))
                    observed = relay.gather(indices).cpu()[:live]
                    self.assertTrue(
                        torch.equal(observed, expected[:live]),
                        f"Relay mismatch at iteration {iteration}, bucket {width}",
                    )


if __name__ == "__main__":
    unittest.main()
