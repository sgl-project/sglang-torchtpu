"""Unit tests for ``get_new_expanded_mm_items`` per-image splitting.

This is the load-bearing behavioral path for multi-image requests: a bundled
``MultimodalDataItem`` (one item carrying N image offsets + a concatenated
feature) must be split back into N per-image items so RadixAttention can cache
each image independently and chunked-prefill can encode them one at a time.

The MoonViT-style models (e.g. nvidia/LocateAnything-3B) carry their per-image
grids under ``image_grid_hws`` rather than ``image_grid_thw``; the splitter must
recognize both keys, fall back cleanly when no usable grid is present, and not
mis-split a degenerate flat grid. No server / GPU / weight loading involved.
"""

import pickle
import unittest
from unittest.mock import patch

import numpy as np
import torch

from sglang.srt.managers.mm_utils import get_new_expanded_mm_items
from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _bundled_item(grid_key=None, grid=None, feature_len=10, num_images=2):
    """A bundled IMAGE item: `num_images` offsets, one concatenated feature."""
    model_specific_data = {}
    if grid_key is not None:
        model_specific_data[grid_key] = grid
    # Distinct per-row values so slice boundaries are checkable.
    feature = torch.arange(feature_len * 3, dtype=torch.float32).reshape(feature_len, 3)
    offsets = [(0, 5), (5, feature_len)][:num_images]
    return MultimodalDataItem(
        modality=Modality.IMAGE,
        offsets=offsets,
        feature=feature,
        model_specific_data=model_specific_data,
    )


class TestGetNewExpandedMMItems(CustomTestCase):
    def test_image_grid_hws_splits_per_image(self):
        # grid rows [[2,3],[4,1]] -> prod = [6, 4] patches -> feature_len 10.
        item = _bundled_item(
            grid_key="image_grid_hws",
            grid=[[2, 3], [4, 1]],
            feature_len=10,
        )
        out = get_new_expanded_mm_items([item])

        self.assertEqual(len(out), 2)
        self.assertEqual([len(o.offsets) for o in out], [1, 1])
        self.assertEqual(out[0].offsets, [(0, 5)])
        self.assertEqual(out[1].offsets, [(5, 10)])
        # Feature sliced 0:6 and 6:10 along dim-0.
        self.assertEqual(out[0].feature.shape[0], 6)
        self.assertEqual(out[1].feature.shape[0], 4)
        self.assertTrue(torch.equal(out[0].feature, item.feature[0:6]))
        self.assertTrue(torch.equal(out[1].feature, item.feature[6:10]))
        # Split items must re-hash (pad value is recomputed per image).
        self.assertTrue(all(o.hash is None for o in out))

    def test_image_grid_hws_tensor_splits_per_image(self):
        # Same as above but the grid arrives as a rank-2 tensor (HF emits these).
        item = _bundled_item(
            grid_key="image_grid_hws",
            grid=torch.tensor([[2, 3], [4, 1]], dtype=torch.long),
            feature_len=10,
        )
        out = get_new_expanded_mm_items([item])

        self.assertEqual(len(out), 2)
        self.assertTrue(torch.equal(out[0].feature, item.feature[0:6]))
        self.assertTrue(torch.equal(out[1].feature, item.feature[6:10]))

    def test_image_grid_thw_still_splits(self):
        # The pre-existing image_grid_thw path must keep working:
        # [[1,2,3],[1,4,1]] -> [6,4].
        item = _bundled_item(
            grid_key="image_grid_thw",
            grid=[[1, 2, 3], [1, 4, 1]],
            feature_len=10,
        )
        out = get_new_expanded_mm_items([item])

        self.assertEqual(len(out), 2)
        self.assertTrue(torch.equal(out[0].feature, item.feature[0:6]))
        self.assertTrue(torch.equal(out[1].feature, item.feature[6:10]))

    def test_missing_grid_falls_back_to_simple_split(self):
        # No grid, but feature dim-0 == num offsets -> simple per-row split.
        item = _bundled_item(grid_key=None, feature_len=2, num_images=2)
        out = get_new_expanded_mm_items([item])

        self.assertEqual(len(out), 2)
        self.assertTrue(torch.equal(out[0].feature, item.feature[0:1]))
        self.assertTrue(torch.equal(out[1].feature, item.feature[1:2]))

    def test_flat_1d_grid_does_not_mis_split(self):
        # A flat 1-D grid (`tensor([2, 2])`) has length == num_items so it passes
        # the length check, but prod(dim=-1) would collapse it to a scalar and
        # corrupt the slice boundaries. The rank-2 guard must reject it. With
        # feature_len != num_items, the simple-split fallback also declines, so
        # the bundled item is passed through unchanged (never mis-sliced).
        item = _bundled_item(
            grid_key="image_grid_hws",
            grid=torch.tensor([2, 2], dtype=torch.long),
            feature_len=10,
        )
        out = get_new_expanded_mm_items([item])

        self.assertEqual(len(out), 1)
        self.assertIs(out[0], item)

    def test_numpy_grid_splits_per_image(self):
        # image_grid_hws can arrive as a numpy array from the HF image processor.
        item = _bundled_item(
            grid_key="image_grid_hws",
            grid=np.array([[2, 3], [4, 1]], dtype=np.int64),
            feature_len=10,
        )
        out = get_new_expanded_mm_items([item])

        self.assertEqual(len(out), 2)
        self.assertTrue(torch.equal(out[0].feature, item.feature[0:6]))
        self.assertTrue(torch.equal(out[1].feature, item.feature[6:10]))

    def test_non_bundled_item_passes_through(self):
        # A single-image item (one offset) is not bundled and is returned as-is.
        item = MultimodalDataItem(
            modality=Modality.IMAGE,
            offsets=[(0, 5)],
            feature=torch.arange(18, dtype=torch.float32).reshape(6, 3),
            model_specific_data={"image_grid_hws": [[2, 3]]},
        )
        out = get_new_expanded_mm_items([item])

        self.assertEqual(len(out), 1)
        self.assertIs(out[0], item)

    @patch("sglang.srt.managers.mm_utils._is_tpu", True)
    def test_split_tensors_own_storage(self):
        # Large tensors keep pickle metadata overhead negligible.
        hidden_size = 1536
        cases = [
            (Modality.IMAGE, "image_grid_thw", count, 1) for count in (2, 8, 16)
        ] + [(Modality.VIDEO, "video_grid_thw", 2, 2)]

        for modality, grid_key, count, frames in cases:
            for field in ("feature", "precomputed_embeddings"):
                with self.subTest(modality=modality, count=count, field=field):
                    rows = frames * 256
                    parent = torch.arange(
                        count * rows * hidden_size, dtype=torch.float32
                    ).reshape(count * rows, hidden_size)
                    item = MultimodalDataItem(
                        modality=modality,
                        offsets=[(i, i + 1) for i in range(count * frames)],
                        model_specific_data={grid_key: [[frames, 16, 16]] * count},
                        **{field: parent},
                    )
                    items = get_new_expanded_mm_items([item])
                    logical_bytes = parent.numel() * parent.element_size()

                    # Check both IPC protocols and receiver-side storage.
                    variants = [items]
                    for protocol in (pickle.DEFAULT_PROTOCOL, pickle.HIGHEST_PROTOCOL):
                        payload = pickle.dumps(items, protocol=protocol)
                        self.assertLess(len(payload), logical_bytes * 1.2)
                        variants.append(pickle.loads(payload))

                    for split_items in variants:
                        self.assertEqual(len(split_items), count)
                        for split_item, expected in zip(
                            split_items, parent.split(rows)
                        ):
                            tensor = getattr(split_item, field)
                            self.assertTrue(torch.equal(tensor, expected))
                            self.assertEqual(
                                tensor.untyped_storage().nbytes(),
                                tensor.numel() * tensor.element_size(),
                            )

    @patch("sglang.srt.managers.mm_utils._is_tpu", True)
    def test_simple_split_tensors_own_storage(self):
        count = 8
        hidden_size = 1536

        for field in ("feature", "precomputed_embeddings"):
            parent = torch.arange(count * hidden_size, dtype=torch.float32).reshape(
                count, hidden_size
            )
            metadata_parent = parent + 1
            item = MultimodalDataItem(
                modality=Modality.IMAGE,
                offsets=[(i, i + 1) for i in range(count)],
                model_specific_data={"item_data": metadata_parent},
                **{field: parent},
            )
            items = get_new_expanded_mm_items([item])
            logical_bytes = sum(
                tensor.numel() * tensor.element_size()
                for tensor in (parent, metadata_parent)
            )

            self.assertEqual(len(items), count)
            for split_item, expected, expected_metadata in zip(
                items, parent.split(1), metadata_parent.split(1)
            ):
                tensor = getattr(split_item, field)
                self.assertTrue(torch.equal(tensor, expected))
                self.assertEqual(
                    tensor.untyped_storage().nbytes(),
                    tensor.numel() * tensor.element_size(),
                )
                metadata = split_item.model_specific_data["item_data"]
                self.assertTrue(torch.equal(metadata, expected_metadata))
                self.assertEqual(
                    metadata.untyped_storage().nbytes(),
                    metadata.numel() * metadata.element_size(),
                )

            for protocol in (pickle.DEFAULT_PROTOCOL, pickle.HIGHEST_PROTOCOL):
                payload = pickle.dumps(items, protocol=protocol)
                self.assertLess(len(payload), logical_bytes * 1.2)


if __name__ == "__main__":
    unittest.main()
