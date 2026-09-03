"""Unit tests for TPU ViT patch-budget batching in mm_schedule.

Every image in a prefill batch is concatenated into one vision-encoder call,
but TPU precompiles that call as graphs bucketed by patch count. A batch above
the top bucket gets no padding and recompiles, which is fatal once the
scheduler arms fail_on_recompile.

The budget is injected here; resolving it against the live policy is covered in
the TPU-suite test, which needs torch_tpu. No server / GPU / weights involved.
"""

import unittest
from unittest.mock import patch

import torch

from sglang.srt.managers import mm_schedule
from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
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


class _RecordingEmbedder:
    """Stands in for the ViT: records per-call patch totals, returns tokens."""

    def __init__(self, as_list: bool = False):
        self.calls: list[int] = []
        self.as_list = as_list

    def __call__(self, items):
        patches = [mm_schedule._item_patch_count(item) for item in items]
        self.calls.append(sum(patches))
        outputs = [
            torch.full((count // _MERGE_AREA, 3), float(count)) for count in patches
        ]
        return outputs if self.as_list else torch.concat(outputs, dim=0)


class TestSplitItemsByPatchBudget(CustomTestCase):
    def test_items_near_the_budget_get_their_own_group(self):
        items = [_image(4752), _image(4836), _image(4784)]
        groups = mm_schedule._split_items_by_patch_budget(items, 4900)
        self.assertEqual([len(group) for group in groups], [1, 1, 1])

    def test_small_items_are_packed_rather_than_split_one_by_one(self):
        groups = mm_schedule._split_items_by_patch_budget([_image(1000)] * 7, 4900)
        totals = [
            sum(mm_schedule._item_patch_count(item) for item in group)
            for group in groups
        ]
        self.assertEqual(totals, [4000, 3000])

    def test_an_item_larger_than_the_budget_still_forms_one_group(self):
        # Packing cannot help; the caller warns instead of silently dropping it.
        groups = mm_schedule._split_items_by_patch_budget([_image(14372)], 4900)
        self.assertEqual([len(group) for group in groups], [1])

    def test_grouping_preserves_input_order(self):
        items = [_image(3000), _image(3000), _image(1000)]
        groups = mm_schedule._split_items_by_patch_budget(items, 4900)
        flattened = [item for group in groups for item in group]
        self.assertEqual([id(item) for item in flattened], [id(item) for item in items])


class TestEncodeVisionItems(CustomTestCase):
    def _encode(self, items, embedder, budget=4900, is_tpu=True):
        with patch.object(mm_schedule, "_is_tpu", is_tpu), patch.object(
            mm_schedule, "_tpu_vit_patch_budget", return_value=budget
        ):
            return mm_schedule._encode_vision_items(embedder, items)

    def test_non_tpu_backends_keep_the_single_batched_call(self):
        embedder = _RecordingEmbedder()
        items = [_image(4752), _image(4836), _image(4784)]
        self._encode(items, embedder, is_tpu=False)
        self.assertEqual(embedder.calls, [14372])

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

    def test_an_image_above_the_budget_warns_that_it_will_recompile(self):
        embedder = _RecordingEmbedder()
        with self.assertLogs(mm_schedule.logger, level="WARNING") as captured:
            self._encode([_image(14372)], embedder)
        self.assertIn("above the top precompiled ViT", "".join(captured.output))

    def test_a_batch_that_fits_does_not_warn(self):
        embedder = _RecordingEmbedder()
        with patch.object(mm_schedule.logger, "warning") as warn:
            self._encode([_image(2000), _image(2000)], embedder)
        warn.assert_not_called()


if __name__ == "__main__":
    unittest.main()
