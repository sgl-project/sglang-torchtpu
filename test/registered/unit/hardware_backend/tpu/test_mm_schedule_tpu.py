"""Unit tests for TPU patch-budget batching of the multimodal encoder call.

Every image in a prefill batch is concatenated into one vision-encoder call,
and TPU precompiles that call for a bounded set of patch counts, so a batch
above the largest one recompiles -- fatal once the scheduler arms
fail_on_recompile.

The budget is injected here; resolving it against the live policy is covered in
test_vit_tpu_compile_runner.py, which needs torch_tpu. No server / TPU / weights
involved.
"""

import unittest
from unittest.mock import Mock, patch

import torch

from sglang.srt.managers import mm_schedule
from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu import mm_schedule_tpu

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_MERGE_AREA = 4  # spatial_merge_size ** 2: patches per emitted token.


def _image(patches: int) -> MultimodalDataItem:
    """An IMAGE item whose grid multiplies out to patches."""
    return MultimodalDataItem(
        modality=Modality.IMAGE,
        offsets=[(0, patches)],
        model_specific_data={"image_grid_thw": torch.tensor([1, 1, patches])},
    )


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


class TestSharedDispatch(CustomTestCase):
    def test_non_tpu_backends_keep_the_single_batched_call(self):
        embedder = _RecordingEmbedder()
        items = [_image(4752), _image(4836), _image(4784)]
        with patch.object(mm_schedule, "_is_tpu", False):
            mm_schedule._encode_vision_items(embedder, items)
        self.assertEqual(embedder.calls, [14372])


if __name__ == "__main__":
    unittest.main()
