"""Async MM warmup artifacts must be reusable by serving with new valid lengths."""

import unittest

import torch

from sglang.srt.hardware_backend.tpu.compilation.warmup_queue import collect_warmups
from sglang.srt.hardware_backend.tpu.mm_compile import MmCompiledOps
from sglang.srt.hardware_backend.tpu.mm_embed_tpu import scatter_mm_on_device
from sglang.srt.hardware_backend.tpu.mm_embedding import (
    PaddedEmbedding,
    concat_embeddings,
)
from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=30, suite="stage-a-test-tpu-unit")


def check_async_mm_ops(embedding):
    device = embedding.weight.device
    dtype = embedding.weight.dtype
    hidden = embedding.weight.shape[1]
    ops = MmCompiledOps()
    cases = []
    with torch.no_grad():
        with collect_warmups() as queue:
            for q in (128, 512):
                ids = (
                    torch.arange(q, dtype=torch.int64, device="cpu")
                    % embedding.num_embeddings
                )
                ids[0], ids[-1] = -1, embedding.num_embeddings + 7
                ids = ids.to(device)
                text = torch.zeros(q, hidden, dtype=dtype, device=device)
                for b in (256, 512):
                    src = torch.zeros(b, hidden, dtype=dtype, device=device)
                    indices = torch.zeros(q, dtype=torch.int32, device=device)
                    mask = torch.zeros(q, 1, dtype=torch.bool, device=device)
                    queue.calls.append(ops.merge.warmup_async(text, src, indices, mask))
                for bs in (1, 2):
                    lengths = torch.tensor(
                        [q - bs + 1] + [1] * (bs - 1), dtype=torch.int32, device=device
                    )
                    queue.calls.append(ops.gather.warmup_async(text, lengths))
                cases.append((q, ids))

        expected_cache_sizes = (4, 4)
        assert (
            tuple(len(fn.cache) for fn in (ops.merge, ops.gather))
            == expected_cache_sizes
        )
        for q, ids in cases:
            ids.clamp_(0, embedding.num_embeddings - 1)
            text = embedding(ids)
            reference = text.cpu()
            for valid_rows in (17, 23):
                sources, host_rows = [], []
                for b in (256, 512):
                    host = torch.full((b, hidden), torch.nan, dtype=dtype)
                    host[3 : 3 + valid_rows] = (
                        torch.arange(valid_rows * hidden)
                        .reshape(valid_rows, hidden)
                        .to(dtype)
                    )
                    sources.append(
                        PaddedEmbedding.from_buffer(
                            host.to(device), 3 + valid_rows
                        ).slice(3, 3 + valid_rows)
                    )
                    host_rows.append(host[3 : 3 + valid_rows])
                src = concat_embeddings(sources)
                mask = torch.zeros(q, 1, dtype=torch.bool)
                mask[1 : 1 + 2 * valid_rows] = True
                actual = text.clone()
                scatter_mm_on_device(actual, mask, src, merge_rows=ops.merge)
                expected = reference.clone()
                expected[mask[:, 0]] = torch.cat(host_rows)
                assert torch.equal(actual.cpu(), expected)
                # New sequence lengths reuse the gather specialization too.
                for lengths in ([q - 3], [valid_rows, q - valid_rows - 3]):
                    lengths_cpu = torch.tensor(lengths, dtype=torch.int32)
                    selected = ops.gather(actual, lengths_cpu.to(device))
                    assert torch.equal(
                        selected.cpu(), expected[torch.cumsum(lengths_cpu, 0) - 1]
                    )
        assert (
            tuple(len(fn.cache) for fn in (ops.merge, ops.gather))
            == expected_cache_sizes
        )


@unittest.skipUnless(hasattr(torch, "tpu") and torch.tpu.is_available(), "requires TPU")
class TestMmCompiledOps(unittest.TestCase):
    def test_async_warmup_and_serving_are_bitwise_equal(self):
        embedding = torch.nn.Embedding(32, 128, device="tpu", dtype=torch.bfloat16)
        check_async_mm_ops(embedding)


if __name__ == "__main__":
    unittest.main()
