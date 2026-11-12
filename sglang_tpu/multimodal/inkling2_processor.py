# SPDX-License-Identifier: Apache-2.0
"""Inkling2 image preprocessing bound to the TPU model class.

Loaded through ``SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE=sglang_tpu.multimodal``.
"""

from __future__ import annotations

import io
import logging
import math
from typing import ClassVar

from PIL import Image

from sglang.srt.models.registry import ModelRegistry
from sglang.tml.multimodal.image_processing import (
    Inkling2ImageProcessor,
    _load_image_bytes,
)
from sglang.tml.processor import Inkling2MultimodalProcessor
from sglang_tpu.models.inkling2 import Inkling2ForConditionalGeneration
from sglang_tpu.models.inkling2_vision import ROW_BUCKETS

if not ModelRegistry.is_arch_supported(Inkling2ForConditionalGeneration.__name__):
    ModelRegistry.register("sglang_tpu.models")

logger = logging.getLogger(__name__)

# 40px patches per image the encoder can hold: each is 2x2 native ViT rows, plus one CLS row.
MAX_IMAGE_PATCHES = (ROW_BUCKETS[-1] - 1) // 4


def fit_image_size(
    width: int, height: int, patch: int, max_patches: int
) -> tuple[int, int]:
    """Largest same-aspect size whose patch grid fits ``max_patches``; unchanged if it already fits."""
    cols, rows = math.ceil(width / patch), math.ceil(height / patch)
    if cols * rows <= max_patches:
        return width, height
    scale = math.sqrt(max_patches / (cols * rows))
    while True:
        width, height = (
            max(1, math.floor(width * scale)),
            max(1, math.floor(height * scale)),
        )
        if math.ceil(width / patch) * math.ceil(height / patch) <= max_patches:
            return width, height
        scale = 0.98


class RowCappedImageProcessor(Inkling2ImageProcessor):
    """Reference preprocessing, with images downscaled to the encoder's largest bucket."""

    max_patches = MAX_IMAGE_PATCHES

    def _encode(self, image, image_token_id, expected_size=None, virtual_size=None):
        if virtual_size is None:
            data = _load_image_bytes(image)
            with Image.open(io.BytesIO(data)) as decoded:
                width, height = decoded.size
                fitted = fit_image_size(
                    width, height, self.patch_size, self.max_patches
                )
                if fitted != (width, height):
                    logger.warning(
                        "Inkling2 image %dx%d exceeds %d patches; downscaled to %dx%d",
                        width,
                        height,
                        self.max_patches,
                        *fitted,
                    )
                    # PIL Lanczos here; the reference resize needs numba, absent on TPU hosts.
                    buf = io.BytesIO()
                    decoded.convert("RGBA").resize(fitted, Image.LANCZOS).save(
                        buf, format="PNG"
                    )
                    image = buf.getvalue()
        return super()._encode(image, image_token_id, expected_size, virtual_size)


class Inkling2TpuMultimodalProcessor(Inkling2MultimodalProcessor):
    models: ClassVar[list] = [Inkling2ForConditionalGeneration]

    def __init__(self, hf_config, server_args, processor, *args, **kwargs):
        super().__init__(hf_config, server_args, processor, *args, **kwargs)
        # The reference processor is stateless; the capped subclass only adds the size guard.
        self.image_processor = RowCappedImageProcessor()
