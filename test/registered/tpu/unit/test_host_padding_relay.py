"""Check temporary relay index uploads against the native TPU backend."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch_tpu  # noqa: F401

from sglang.srt.managers import overlap_utils
from sglang.srt.managers.overlap_utils import RelayPayload
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_tpu_ci
from sglang_tpu.compilation.bucketer import Bucketer
from sglang_tpu.host_padding_relay import HostPaddingFutureMap

register_tpu_ci(
    est_time=30,
    suite="stage-a-test-tpu-unit",
    chips=1,
    labels=["singlehost", "infra", "compile"],
)


class TestHostPaddingRelay(unittest.TestCase):
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
