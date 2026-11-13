"""Images larger than the encoder's biggest row bucket are downscaled by the TPU processor."""

import io
import unittest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

from PIL import Image

from sglang.test.test_utils import CustomTestCase


class TestInkling2ImageCap(CustomTestCase):
    def test_fit_image_size(self):
        from sglang_tpu.multimodal.inkling2_processor import fit_image_size

        self.assertEqual(fit_image_size(800, 600, 40, 4000), (800, 600))
        self.assertEqual(
            fit_image_size(5120, 5760, 40, 32767), (5120, 5760)
        )  # 18432 patches fit
        w, h = fit_image_size(10000, 9000, 40, 32767)  # 56250 patches do not
        self.assertLessEqual((-(-w // 40)) * (-(-h // 40)), 32767)
        self.assertAlmostEqual(w / h, 10000 / 9000, places=2)
        self.assertGreater(w * h, 0.85 * 32767 * 1600)

    def test_oversized_image_is_capped(self):
        from sglang_tpu.multimodal.inkling2_processor import (
            MAX_IMAGE_PATCHES,
            RowCappedImageProcessor,
        )

        buf = io.BytesIO()
        Image.new("RGB", (10000, 9000), (200, 30, 30)).save(buf, format="PNG")
        span = RowCappedImageProcessor()._encode(buf.getvalue(), 200005)
        _, rows, cols = span.vision_grid_thw[0].tolist()
        self.assertLessEqual(rows * cols, MAX_IMAGE_PATCHES)
        self.assertEqual(len(span.int_tokens), rows * cols)
        self.assertEqual(span.vision_patches_bthwc.shape[0], rows * cols)

    def test_small_image_is_unchanged(self):
        from sglang_tpu.multimodal.inkling2_processor import RowCappedImageProcessor

        buf = io.BytesIO()
        Image.new("RGB", (123, 77)).save(buf, format="PNG")
        span = RowCappedImageProcessor()._encode(buf.getvalue(), 200005)
        self.assertEqual(span.vision_grid_thw[0].tolist(), [1, 2, 4])
        self.assertEqual(span.vision_original_thw[0].tolist(), [1, 77, 123])


if __name__ == "__main__":
    unittest.main()
