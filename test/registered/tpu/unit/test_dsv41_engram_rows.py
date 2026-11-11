# SPDX-License-Identifier: Apache-2.0
"""Regression for the real Engram extend scan abort under production flags.

With libtpu 0.0.50 and the required associative-scan workaround, the INT64
scan used by repeat_interleave aborts compilation. Exercise the row hook
through the actual Hasher forward, including padding, hashes and history
commits, against the original CPU Hasher on two stateful replays.
"""

import importlib.util
import os
import unittest

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["SGLANG_PLATFORM"] = "tpu_oot"

from sglang_tpu.env_override import apply_tpu_env_overrides

apply_tpu_env_overrides()

import torch
from sglang.srt.layers.engram import EngramHasher
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase

from sglang_tpu.models.deepseek_v41.hooks import hook_specs

register_tpu_ci(
    est_time=240,
    suite="stage-a-test-tpu-unit",
    chips=1,
    labels=["singlehost", "deepseek-v4.1-flash", "accuracy", "compile"],
)


class _Batch:
    pass


def _hasher():
    # No tokenizer or checkpoint download: exact signed64 production operations
    # with nontrivial large multipliers, two layers, four-gram/eight-head layout.
    hasher = EngramHasher.__new__(EngramHasher)
    torch.nn.Module.__init__(hasher)
    hasher.max_ngram_size = 4
    hasher.pad_id = 0
    hasher.pad_row = 2
    hasher.image_token_id = 127
    hasher.register_buffer("token_map", torch.arange(256, dtype=torch.int64))
    hasher.register_buffer(
        "multipliers",
        torch.tensor([[100000000003 + i * 2 for i in range(4)]] * 2),
    )
    hasher.register_buffer(
        "primes", torch.arange(1009, 1057, dtype=torch.int64).reshape(2, 3, 8)
    )
    hasher.register_buffer(
        "offsets", torch.arange(24, dtype=torch.int64).repeat(2, 1) * 1100
    )
    hasher.register_buffer("history", torch.zeros(3, 3, dtype=torch.int32))
    return hasher


class _Forward(torch.nn.Module):
    def __init__(self, hasher, counts):
        super().__init__()
        self.hasher = hasher
        self.batch = _Batch()
        self.batch.forward_mode = ForwardMode.EXTEND
        self.batch.extend_seq_lens_cpu = counts
        self.batch.engram_history = None

    def forward(self, ids, lens, starts, req, positions, out_loc):
        batch = self.batch
        batch.req_pool_indices = req
        batch.extend_seq_lens = lens
        batch.extend_start_loc = starts
        batch.positions = positions
        batch.out_cache_loc = out_loc
        hashes = self.hasher(ids, batch)
        return hashes, self.hasher.history.clone()


@unittest.skipUnless(importlib.util.find_spec("torch_tpu"), "requires TorchTPU")
class TestEngramExtendRows(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        import torch_tpu  # noqa: F401

        from sglang_tpu.patches import _register_tpu_compile_backend

        _register_tpu_compile_backend()
        from sglang.srt.platforms import current_platform
        from sglang.srt.plugins.hook_registry import HookRegistry

        assert current_platform.is_out_of_tree(), "requires the actual OOT caller path"
        targets = {
            "sglang.srt.layers.engram.EngramHasher.prepare_extend_rows",
            "sglang.srt.layers.engram.EngramHasher.prepare_lookback_blocked",
        }
        # Consume the real plugin registrations, not hand-written replacements.
        for spec in hook_specs():
            if spec[0] in targets:
                HookRegistry.register(*spec)
        HookRegistry.apply_hooks()
        assert targets <= HookRegistry._patched, "Engram hook application failed"

    def test_real_hasher_hashes_and_history(self):
        hasher = _hasher()
        device_hasher = _hasher().to("tpu")
        for width in (1, 3, 4, 5, 8):
            for rows in (0, 128):
                with self.subTest(prefix_width=width, rows=rows):
                    torch._dynamo.reset()
                    # Width four includes all 16 possible boolean patterns.
                    mask = (
                        torch.arange(rows)[:, None] >> torch.arange(width)[None, :]
                    ) % 2 == 1
                    expected = hasher.prepare_lookback_blocked(mask)
                    compiled_prefix = torch.compile(
                        device_hasher.prepare_lookback_blocked,
                        backend="tpu",
                        fullgraph=True,
                        dynamic=False,
                    )
                    for replay in range(2):
                        actual = compiled_prefix(mask.to("tpu")).cpu()
                        self.assertEqual(actual.dtype, torch.bool)
                        self.assertTrue(torch.equal(actual, expected))
        cases = [(128, [1] * 8), (128, [0, 2, 1, 0, 3, 1, 0, 1])]
        cases += [(tokens, [tokens // 8] * 8) for tokens in (128, 256, 512, 1024)]
        for tokens, counts in cases:
            with self.subTest(tokens=tokens, counts=counts):
                # Distinct diagnostic graphs, not a production cache-limit change.
                torch._dynamo.reset()
                reference = _Forward(_hasher(), counts)
                actual = _Forward(_hasher(), counts).to("tpu")
                compiled = torch.compile(
                    actual, backend="tpu", fullgraph=True, dynamic=False
                )
                real = sum(counts)
                starts = torch.tensor(
                    [sum(counts[:i]) for i in range(8)], dtype=torch.int32
                )
                lens = torch.tensor(counts, dtype=torch.int32)
                # Two live owners; all remaining requests are graph-padding
                # owners with out_loc=0, so duplicate pad writes are identical.
                req = torch.tensor([0, 1, 2, 2, 2, 2, 2, 2], dtype=torch.int64)
                for replay in range(2):
                    ids = torch.zeros(tokens, dtype=torch.int64)
                    ids[:real] = (torch.arange(real) + 17 + replay) % 126
                    if real and counts[0]:
                        # Position zero for warmup, then interior barriers on
                        # full prefills; both replays change the boundary.
                        image_position = min(counts[0] - 1, replay + 1)
                        ids[image_position] = 127
                    positions = torch.zeros(tokens, dtype=torch.int64)
                    out_loc = torch.zeros(tokens, dtype=torch.int32)
                    cursor = 0
                    for row, count in enumerate(counts):
                        positions[cursor : cursor + count] = (
                            torch.arange(count) + replay
                        )
                        if row < 2:
                            out_loc[cursor : cursor + count] = torch.arange(count) + 1
                        else:
                            # Equal pad histories avoid duplicate-index ordering
                            # ambiguity on both reference and device.
                            ids[cursor : cursor + count] = 0
                        cursor += count
                    inputs = (ids, lens, starts, req, positions, out_loc)
                    expected = reference(*inputs)
                    got = compiled(*(value.to("tpu") for value in inputs))
                    for result, target in zip(got, expected):
                        self.assertTrue(torch.equal(result.cpu(), target))


if __name__ == "__main__":
    unittest.main()
