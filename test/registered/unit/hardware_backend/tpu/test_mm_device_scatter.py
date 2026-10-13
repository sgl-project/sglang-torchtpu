"""The MM placement switch preserves host and device embedding results."""

import os
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch_tpu._internal import execution_mode

from sglang.srt.environ import envs
from sglang.srt.managers import mm_schedule, mm_utils
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.mem_cache.multimodal_cache import MultiModalStaticCache
from sglang.test.ci.ci_register import register_tpu_ci
from sglang_tpu import mm_embed_tpu
from sglang_tpu.graph_runner.vit_tpu_compile_runner import (
    Qwen3VLVisionTpuCompilePolicy,
    VisionPaddingContext,
    ViTTpuCompileRunner,
    _Qwen3VLRunContext,
)
from sglang_tpu.patches import _register_aten_fallbacks

register_tpu_ci(est_time=30, suite="stage-a-test-tpu-unit", chips=1)


@contextmanager
def _disable_cpu_fallback():
    original = execution_mode.enable_cpu_fallback
    execution_mode.enable_cpu_fallback = False
    try:
        yield
    finally:
        execution_mode.enable_cpu_fallback = original


@unittest.skipUnless(
    hasattr(torch, "accelerator") and torch.accelerator.is_available(), "requires TPU"
)
class TestDeviceEncoderScatter(unittest.TestCase):
    def setUp(self):
        _register_aten_fallbacks()
        self.enterContext(_disable_cpu_fallback())

    def test_non_chunked_embed(self):
        for on_device in (None, False, True):
            with self.subTest(on_device=on_device):
                self._check_embed(on_device, chunk_size=20)

    def test_chunked_embed(self):
        for on_device in (None, False, True):
            with self.subTest(on_device=on_device):
                self._check_embed(on_device, chunk_size=4)

    @torch.no_grad()
    def _check_embed(self, on_device, chunk_size):
        encoder_device = "tpu" if on_device else "cpu"
        # Both modalities cross chunk boundaries; the second image arrives later.
        specs = [
            (Modality.IMAGE, 2, 6),
            (Modality.AUDIO, 8, 12),
            (Modality.IMAGE, 14, 16),
        ]
        items, sources = [], {}
        tokens = torch.arange(20) % 8
        weight = torch.arange(32).reshape(8, 4).to(torch.bfloat16)
        for i, (modality, start, end) in enumerate(specs):
            key = 100 + i
            items.append(
                MultimodalDataItem(
                    modality=modality, hash=key, pad_value=key, offsets=[(start, end)]
                )
            )
            width = 12 if modality == Modality.IMAGE else 4
            sources[key] = (
                torch.arange((end - start + 1) * width).reshape(-1, width).float() / 10
                + i * 10
            )
            tokens[start : end + 1] = key
        expected = weight[tokens.clamp(0, 7)].clone()
        expected_deepstack = torch.zeros(20, 8, dtype=weight.dtype)
        for item in items:
            start, end = item.offsets[0]
            source = sources[item.hash].to(weight.dtype)
            expected[start : end + 1] = source[:, :4]
            if item.modality == Modality.IMAGE:
                expected_deepstack[start : end + 1] = source[:, 4:]
        outputs = {key: value.to(encoder_device) for key, value in sources.items()}
        image_encoder = Mock(
            side_effect=lambda batch: torch.cat([outputs[item.hash] for item in batch])
        )
        audio_encoder = Mock(
            side_effect=lambda batch: [outputs[item.hash] for item in batch]
        )
        embedding = torch.nn.Embedding.from_pretrained(weight.to("tpu"))
        model = SimpleNamespace(
            deepstack_visual_indexes=[0, 1],
            separate_deepstack_embeds=lambda x: (x[:, :4], x[:, 4:]),
        )
        cache = MultiModalStaticCache(1 << 20)
        with (
            patch.dict(os.environ),
            patch.object(mm_utils, "_is_tpu", True),
            patch.object(mm_schedule, "_is_tpu", True),
            patch.object(mm_schedule, "embedding_cache", cache),
            patch.object(
                mm_embed_tpu, "embed_mm_on_host", wraps=mm_embed_tpu.embed_mm_on_host
            ) as host_merge,
        ):
            if on_device is None:
                os.environ.pop("SGLANG_TPU_MM_EMBED_ON_DEVICE", None)
            else:
                os.environ["SGLANG_TPU_MM_EMBED_ON_DEVICE"] = str(int(on_device))
            for repeat in range(2):
                for prefix in range(0, 20, chunk_size):
                    with self.subTest(repeat=repeat, prefix=prefix):
                        actual, info = mm_utils.embed_mm_inputs(
                            [MultimodalInputs(mm_items=items)],
                            [prefix],
                            [chunk_size],
                            tokens[prefix : prefix + chunk_size].clone().to("tpu"),
                            embedding,
                            multimodal_model=model,
                            data_embedding_func_mapping={
                                Modality.IMAGE: image_encoder,
                                Modality.AUDIO: audio_encoder,
                            },
                            use_deepstack={Modality.IMAGE: True},
                        )
                        deepstack = info["input_deepstack_embeds"]
                        for result, reference in (
                            (actual, expected),
                            (deepstack, expected_deepstack),
                        ):
                            self.assertEqual(result.device.type, "tpu")
                            self.assertEqual(result.dtype, weight.dtype)
                            self.assertTrue(
                                torch.equal(
                                    result.cpu(),
                                    reference[prefix : prefix + chunk_size],
                                )
                            )
                self.assertEqual(image_encoder.call_count, 2 if chunk_size < 20 else 1)
                self.assertEqual(audio_encoder.call_count, 1)
                for item in items:
                    cached = cache.get_single(item.hash)
                    self.assertIsNotNone(cached)
                    self.assertEqual(cached.embedding.device.type, encoder_device)
                    self.assertEqual(cached.embedding.dtype, sources[item.hash].dtype)
                    self.assertTrue(
                        torch.equal(cached.embedding.cpu(), sources[item.hash])
                    )
            self.assertEqual(
                host_merge.call_count, 2 * (20 // chunk_size) * int(not on_device)
            )

    @torch.no_grad()
    def test_qwen_output_device_and_row_selection(self):
        vit = SimpleNamespace(blocks=[], spatial_merge_size=2)
        policy = Qwen3VLVisionTpuCompilePolicy(vit)
        runner = ViTTpuCompileRunner(vit)
        expected = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4)
        padded = torch.cat([expected, torch.full_like(expected, -100)]).to("tpu")
        padding = VisionPaddingContext(
            real_seq_len=12, bucket_len=24, padded=True, merge_sq=4
        )
        with patch.object(
            runner, "_blocks_and_merger_fallback", return_value=padded[:3]
        ):
            for on_device in (False, True):
                for indices in (None, torch.tensor([2, 0])):
                    for path in ("unpad", "eager"):
                        with (
                            self.subTest(
                                on_device=on_device, indices=indices, path=path
                            ),
                            envs.SGLANG_TPU_MM_EMBED_ON_DEVICE.override(on_device),
                        ):
                            if path == "unpad":
                                actual = policy.unpad_outputs(
                                    padded, _Qwen3VLRunContext(padding, indices)
                                )
                            else:
                                actual = runner.run(
                                    padded[:3],
                                    torch.tensor([0, 3], dtype=torch.int32),
                                    output_indices=indices,
                                )
                            self.assertEqual(
                                actual.device.type, "tpu" if on_device else "cpu"
                            )
                            self.assertEqual(actual.dtype, expected.dtype)
                            selected = (
                                expected if indices is None else expected[indices]
                            )
                            self.assertTrue(torch.equal(actual.cpu(), selected))


if __name__ == "__main__":
    unittest.main()
