"""Unit tests for TPU handling of the multimodal encoder call.

Every image in a prefill batch is concatenated into one vision-encoder call,
and TPU precompiles that call for a bounded set of patch counts, so a batch
above the largest one recompiles -- fatal once the scheduler arms
fail_on_recompile.

The budget is injected here; resolving it against the live policy is covered in
test_vit_tpu_compile_runner.py, which needs torch_tpu. No server / TPU / weights
involved.

Host placement also needs every encoder output on the host, whichever device
the encoder finished on (#741). A mock stands in for the TPU tensor here;
test_mm_device_scatter.py runs the same path with real TPU tensors.
"""

import unittest
from unittest.mock import Mock, patch

import torch

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.tpu import mm_schedule_tpu
from sglang.srt.managers import mm_schedule
from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.mem_cache.multimodal_cache import MultiModalStaticCache
from sglang.srt.multimodal.evs import EVSEmbeddingResult
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_MERGE_AREA = 4  # spatial_merge_size ** 2: patches per emitted token.


def _image(patches: int) -> MultimodalDataItem:
    """An IMAGE item whose grid multiplies out to patches."""
    return MultimodalDataItem(
        modality=Modality.IMAGE,
        offsets=[(0, patches)],
        model_specific_data={"image_grid_thw": torch.tensor([1, 1, patches])},
    )


def _tpu_output(rows: int) -> Mock:
    """An encoder output left on the TPU; .to("cpu") returns its host copy."""
    output = Mock(spec=torch.Tensor)
    output.to.return_value = torch.full((rows, 3), float(rows))
    return output


class _RecordingEmbedder:
    """Stands in for the ViT: records per-call patch totals, returns tokens."""

    def __init__(self, as_list: bool = False):
        self.calls: list[int] = []
        self.as_list = as_list

    def __call__(self, items):
        patches = [mm_schedule_tpu._item_patch_count(item) for item in items]
        self.calls.append(sum(patches))
        outputs = [
            torch.full((count // _MERGE_AREA, 3), float(count)) for count in patches
        ]
        return outputs if self.as_list else torch.concat(outputs, dim=0)


class TestSplitItemsByPatchBudget(CustomTestCase):
    def test_items_near_the_budget_get_their_own_group(self):
        items = [_image(4752), _image(4836), _image(4784)]
        groups = mm_schedule_tpu._split_items_by_patch_budget(items, 4900)
        self.assertEqual([len(group) for group in groups], [1, 1, 1])

    def test_small_items_are_packed_rather_than_split_one_by_one(self):
        groups = mm_schedule_tpu._split_items_by_patch_budget([_image(1000)] * 7, 4900)
        totals = [
            sum(mm_schedule_tpu._item_patch_count(item) for item in group)
            for group in groups
        ]
        self.assertEqual(totals, [4000, 3000])

    def test_an_item_larger_than_the_budget_still_forms_one_group(self):
        # Packing cannot help; the item must still be encoded, not dropped.
        groups = mm_schedule_tpu._split_items_by_patch_budget([_image(14372)], 4900)
        self.assertEqual([len(group) for group in groups], [1])

    def test_grouping_preserves_input_order(self):
        items = [_image(3000), _image(3000), _image(1000)]
        groups = mm_schedule_tpu._split_items_by_patch_budget(items, 4900)
        flattened = [item for group in groups for item in group]
        self.assertEqual([id(item) for item in flattened], [id(item) for item in items])


class TestEncodeVisionItems(CustomTestCase):
    def _encode(self, items, embedder, budget=4900):
        with patch.object(mm_schedule_tpu, "_patch_budget", return_value=budget):
            return mm_schedule_tpu.encode_vision_items(embedder, items)

    def test_a_batch_within_the_budget_is_not_split(self):
        embedder = _RecordingEmbedder()
        self._encode([_image(2000), _image(2000)], embedder)
        self.assertEqual(embedder.calls, [4000])

    def test_an_oversized_batch_runs_as_calls_that_each_fit_a_bucket(self):
        embedder = _RecordingEmbedder()
        self._encode([_image(4752), _image(4836), _image(4784)], embedder)
        self.assertEqual(embedder.calls, [4752, 4836, 4784])

    def test_split_output_keeps_row_order_and_total_length(self):
        embedder = _RecordingEmbedder()
        items = [_image(4752), _image(4836)]
        out = self._encode(items, embedder)
        self.assertEqual(out.shape[0], (4752 + 4836) // _MERGE_AREA)
        # Each call fills its rows with its own patch total, so the boundary
        # position proves the groups were concatenated in item order.
        self.assertEqual(out[0, 0].item(), 4752.0)
        self.assertEqual(out[4752 // _MERGE_AREA, 0].item(), 4836.0)

    def test_per_item_list_outputs_are_flattened_not_concatenated(self):
        embedder = _RecordingEmbedder(as_list=True)
        out = self._encode([_image(4752), _image(4836), _image(4784)], embedder)
        self.assertIsInstance(out, list)
        self.assertEqual(
            [int(tensor[0, 0].item()) for tensor in out], [4752, 4836, 4784]
        )

    def test_an_item_above_the_budget_is_still_encoded_in_one_call(self):
        # Nothing to split; the compile runner pads/recompiles and warns for itself.
        embedder = _RecordingEmbedder()
        self._encode([_image(14372)], embedder)
        self.assertEqual(embedder.calls, [14372])

    def test_items_without_patch_grids_use_model_owned_batching(self):
        cases = (
            (Modality.IMAGE, torch.zeros(3, 28, 28)),
            (Modality.VIDEO, torch.zeros(2, 3, 28, 28)),
        )
        for modality, feature in cases:
            with self.subTest(modality=modality):
                items = [
                    MultimodalDataItem(
                        modality=modality,
                        feature=feature,
                        offsets=[(0, 1)],
                    )
                ]
                expected = torch.ones(2, 3)
                embedder = Mock(return_value=expected)

                result = self._encode(items, embedder, budget=1)

                self.assertIs(result, expected)
                embedder.assert_called_once_with(items)


class TestEncoderOutputToHost(CustomTestCase):
    def test_device_outputs_are_copied_to_the_host(self):
        tensor = _tpu_output(2)
        per_item = [_tpu_output(1), _tpu_output(3)]
        evs = EVSEmbeddingResult(embedding=_tpu_output(4), num_tokens_per_frame=[3, 1])
        with envs.SGLANG_TPU_MM_EMBED_ON_DEVICE.override(False):
            landed_tensor = mm_schedule_tpu.encoder_output_to_host(tensor)
            landed_list = mm_schedule_tpu.encoder_output_to_host(per_item)
            landed_evs = mm_schedule_tpu.encoder_output_to_host(evs)

        self.assertIs(landed_tensor, tensor.to.return_value)
        self.assertEqual(len(landed_list), len(per_item))
        for landed, output in zip(landed_list, per_item):
            self.assertIs(landed, output.to.return_value)
        # EVS keeps its per-frame token counts; only the tensor moves.
        self.assertIsInstance(landed_evs, EVSEmbeddingResult)
        self.assertEqual(landed_evs.num_tokens_per_frame, [3, 1])
        self.assertIs(landed_evs.embedding, evs.embedding.to.return_value)
        for output in (tensor, *per_item, evs.embedding):
            output.to.assert_called_once_with("cpu")

    def test_host_outputs_pass_through_without_a_copy(self):
        # Qwen3-VL already landed on the host at its bucketed shape.
        output = torch.ones(2, 3)
        with envs.SGLANG_TPU_MM_EMBED_ON_DEVICE.override(False):
            self.assertIs(mm_schedule_tpu.encoder_output_to_host(output), output)

    def test_device_placement_leaves_outputs_on_the_device(self):
        output = _tpu_output(2)
        with envs.SGLANG_TPU_MM_EMBED_ON_DEVICE.override(True):
            self.assertIs(mm_schedule_tpu.encoder_output_to_host(output), output)
        output.to.assert_not_called()

    def test_split_batch_outputs_land_before_concatenation(self):
        # A mock cannot be concatenated, so this also pins the order: landing
        # after the concat would run a variable-length cat on the TPU.
        outputs = [_tpu_output(2), _tpu_output(3)]
        embedder = Mock(side_effect=outputs)
        with (
            envs.SGLANG_TPU_MM_EMBED_ON_DEVICE.override(False),
            patch.object(mm_schedule_tpu, "_patch_budget", return_value=4900),
        ):
            out = mm_schedule_tpu.encode_vision_items(
                embedder, [_image(4752), _image(4836)]
            )

        self.assertEqual(embedder.call_count, 2)
        expected = torch.cat([output.to.return_value for output in outputs])
        self.assertTrue(torch.equal(out, expected))


class TestSharedDispatch(CustomTestCase):
    def test_non_tpu_backends_keep_the_single_batched_call(self):
        embedder = _RecordingEmbedder()
        items = [_image(4752), _image(4836), _image(4784)]
        with patch.object(mm_schedule, "_is_tpu", False):
            mm_schedule._encode_vision_items(embedder, items)
        self.assertEqual(embedder.calls, [14372])

    def test_tpu_full_path_caches_the_host_copy(self):
        # An item with several placeholder spans takes the full path, which
        # calls the encoder directly rather than through _encode_vision_items.
        output = _tpu_output(4)
        item = MultimodalDataItem(
            modality=Modality.VIDEO, hash=7, offsets=[(0, 1), (3, 4)]
        )
        cache = MultiModalStaticCache(1 << 20)
        with (
            patch.object(mm_schedule, "_is_tpu", True),
            patch.object(mm_schedule, "embedding_cache", cache),
            envs.SGLANG_TPU_MM_EMBED_ON_DEVICE.override(False),
        ):
            chunk, _ = mm_schedule._get_chunked_embedding_full(
                Mock(return_value=output),
                [item],
                item.offsets,
                0,
                5,
                torch.zeros(5, dtype=torch.long),
                torch.device("cpu"),
            )

        output.to.assert_called_once_with("cpu")
        self.assertIs(cache.get([item.hash]).embedding, output.to.return_value)
        self.assertTrue(torch.equal(chunk, output.to.return_value))


if __name__ == "__main__":
    unittest.main()
