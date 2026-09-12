"""Bitwise placement checks with padded sources and CPU row metadata."""

import unittest
from unittest.mock import patch

import torch

from sglang.srt.hardware_backend.tpu import mm_embed_tpu
from sglang.srt.managers.schedule_batch import Modality
from sglang.srt.mem_cache.multimodal_cache import EmbeddingResult, MultiModalStaticCache
from sglang.srt.multimodal.embedding import (
    PaddedEmbedding,
    concat_embeddings,
    split_embedding,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDeviceEmbedPlacement(unittest.TestCase):
    device = torch.device("cpu")

    def test_chunked_permuted_repeated_sources_and_deepstack(self):
        # NaN padding must never replace text or reach deepstack's zero rows.
        source_cpu = torch.arange(16 * 12).reshape(16, 12).to(torch.bfloat16)
        source_cpu[9:] = float("nan")
        source = source_cpu.to(self.device)
        embedding = PaddedEmbedding.from_tensor(
            source, 9, torch.tensor([8, 0, 4, 3, 7, 1, 6])
        )
        self.assertTrue(
            torch.equal(embedding.to_cpu(), source_cpu[[8, 0, 4, 3, 7, 1, 6]])
        )
        first, second = split_embedding(embedding, [3, 4])
        chunk = concat_embeddings(
            [second.slice_rows(1, 3), first, second.slice_rows(0, 1)]
        )
        normal, extra = chunk.split_columns(4)
        prompt = torch.full((16, 4), -1, dtype=torch.bfloat16)
        mask = torch.zeros(16, 1, dtype=torch.bool)
        mask[[0, 2, 5, 6, 11, 15]] = True
        expected = prompt.clone()
        expected[mask[:, 0]] = source_cpu[[7, 1, 8, 0, 4, 3], :4]
        expected_extra = torch.zeros(16, 8, dtype=torch.bfloat16)
        expected_extra[mask[:, 0]] = source_cpu[[7, 1, 8, 0, 4, 3], 4:]
        with patch.object(
            mm_embed_tpu,
            "apply_device_mm_embeds",
            wraps=mm_embed_tpu.apply_device_mm_embeds,
        ) as copy:
            actual, deepstack = mm_embed_tpu.embed_mm_on_tpu(
                input_embeds=prompt.to(self.device),
                deepstack_shape=torch.Size((16, 8)),
                modalities=[Modality.IMAGE],
                embeddings=[normal],
                masks=[mask],
                deepstack_embeddings=[extra],
                use_deepstack={Modality.IMAGE: True},
            )
        self.assertTrue(torch.equal(actual.cpu(), expected))
        self.assertTrue(torch.equal(deepstack.cpu(), expected_extra))
        # All three logical chunks share one source per column group.
        self.assertEqual(copy.call_count, 2)
        self.assertIs(first.parts[0].data, source)

    def test_plain_device_and_host_modalities(self):
        prompt = torch.zeros(8, 4, dtype=torch.bfloat16)
        image = torch.arange(12).reshape(3, 4).float()
        audio = torch.full((2, 4), 17.0)
        image_mask = torch.zeros(8, 1, dtype=torch.bool)
        audio_mask = torch.zeros_like(image_mask)
        image_mask[[0, 3, 7]] = True
        audio_mask[[1, 5]] = True
        result, _ = mm_embed_tpu.embed_mm_on_tpu(
            input_embeds=prompt.to(self.device),
            deepstack_shape=None,
            modalities=[Modality.IMAGE, Modality.AUDIO],
            embeddings=[image.to(self.device), audio],
            masks=[image_mask, audio_mask],
            deepstack_embeddings=[None, None],
            use_deepstack={},
        )
        prompt[image_mask[:, 0]] = image.to(prompt.dtype)
        prompt[audio_mask[:, 0]] = audio.to(prompt.dtype)
        self.assertTrue(torch.equal(result.cpu(), prompt))

    def test_empty_rows_and_invalid_lengths(self):
        source = torch.ones(16, 4).to(self.device)
        empty = PaddedEmbedding.from_tensor(source, 0)
        prompt = torch.zeros(8, 4).to(self.device)
        mask = torch.zeros(8, 1, dtype=torch.bool)
        result = mm_embed_tpu._merge_mm_embedding(prompt, empty, mask)
        self.assertIs(result, prompt)
        mask[1] = True
        with self.assertRaises(ValueError):
            mm_embed_tpu._merge_mm_embedding(prompt, empty, mask)
        with self.assertRaises(ValueError):
            PaddedEmbedding.from_tensor(source, 3, torch.tensor([3]))

    def test_cache_charges_padding_and_evicts(self):
        source = torch.zeros(16, 4).to(self.device)
        embedding = PaddedEmbedding.from_tensor(source, 2)
        cache = MultiModalStaticCache(source.numel() * source.element_size())
        self.assertTrue(
            cache.set(1, EmbeddingResult(embedding=embedding.slice_rows(0, 1)))
        )
        self.assertEqual(cache.current_size, cache.max_size)
        self.assertTrue(
            cache.set(2, EmbeddingResult(embedding=embedding.slice_rows(1, 2)))
        )
        # Both entries share one bucket; freeing one must keep its full charge.
        self.assertIsNotNone(cache.get_single(1))
        self.assertEqual(cache.current_size, cache.max_size)
        cache.free(1, None)
        self.assertEqual(cache.current_size, cache.max_size)
        self.assertIs(cache.get_single(2).embedding.parts[0].data, source)
        other = PaddedEmbedding.from_tensor(torch.ones_like(source), 1)
        cache.set(3, EmbeddingResult(embedding=other))
        self.assertIsNone(cache.get_single(2))
        self.assertEqual(cache.current_size, cache.max_size)
        cache.clear()
        self.assertEqual(cache.current_size, 0)
        cache.set(4, EmbeddingResult(embedding=embedding))
        cache.free(4, None)
        self.assertEqual(cache.current_size, 0)


if __name__ == "__main__":
    unittest.main()
