"""Changing media lengths within warmed buckets must not compile new programs."""

import unittest

import torch
from torch_tpu._internal import execution_mode
from torch_tpu._internal.device import _device_ops_backend as backend
from torch_tpu._internal.sync.sync import synchronize

from sglang.srt.hardware_backend.tpu.mm_embed_tpu import (
    embed_mm_on_tpu,
    warmup_device_mm_embeds,
)
from sglang.srt.managers.schedule_batch import Modality
from sglang.srt.multimodal.embedding import PaddedEmbedding, concat_embeddings
from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=15, suite="stage-a-test-tpu-unit")


@unittest.skipUnless(hasattr(torch, "tpu") and torch.tpu.is_available(), "requires TPU")
class TestBucketedPlacement(unittest.TestCase):
    def test_lengths_offsets_and_repeated_items_reuse_warmed_programs(self):
        old_mode = execution_mode.eager_mode
        old_fallback = execution_mode.enable_cpu_fallback
        execution_mode.eager_mode = execution_mode.EagerMode.DEFER_NEVER
        execution_mode.enable_cpu_fallback = False
        try:
            self._check_bucketed_placement()
        finally:
            execution_mode.eager_mode = old_mode
            execution_mode.enable_cpu_fallback = old_fallback

    def _check_bucketed_placement(self):
        device = torch.device("tpu")
        # Include the deepstack column split and its wider destination grid.
        sources_cpu = [
            torch.arange(v * 192).reshape(v, 192).to(torch.bfloat16)
            for v in (16, 32, 64)
        ]
        sources = [PaddedEmbedding.from_tensor(x.to(device)) for x in sources_cpu]
        prompts = {
            p: torch.zeros((p, 64), dtype=torch.bfloat16, device=device)
            for p in (32, 64)
        }
        synchronize(
            [part.data for src in sources for part in src.parts]
            + list(prompts.values()),
            wait=True,
        )
        warmup_device_mm_embeds(sources, list(prompts), 64, torch.bfloat16, device)
        misses = backend._get_cache_misses()
        for source_cpu, source in zip(sources_cpu, sources):
            for rows, prompt in prompts.items():
                for offset, count in ((0, 2), (3, 5), (7, 3)):
                    # Repeat a cached item with a second, noncontiguous prompt span.
                    chunk = source.slice_rows(offset, offset + count)
                    chunk = concat_embeddings([chunk, chunk.slice_rows(0, 1)])
                    normal, extra = chunk.split_columns(64)
                    mask = torch.zeros((rows, 1), dtype=torch.bool)
                    mask[2 : 2 + count] = True
                    mask[-2] = True
                    actual, deepstack = embed_mm_on_tpu(
                        input_embeds=prompt,
                        deepstack_shape=torch.Size((rows, 128)),
                        modalities=[Modality.IMAGE],
                        embeddings=[normal],
                        masks=[mask],
                        deepstack_embeddings=[extra],
                        use_deepstack={Modality.IMAGE: True},
                    )
                    expected = torch.zeros(rows, 192, dtype=torch.bfloat16)
                    expected[2 : 2 + count] = source_cpu[offset : offset + count]
                    expected[-2] = source_cpu[offset]
                    self.assertTrue(torch.equal(actual.cpu(), expected[:, :64]))
                    self.assertTrue(torch.equal(deepstack.cpu(), expected[:, 64:]))
        self.assertEqual(backend._get_cache_misses(), misses)


if __name__ == "__main__":
    unittest.main()
