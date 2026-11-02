# SPDX-License-Identifier: Apache-2.0
"""Inkling2 image preprocessing bound to the TPU model class.

Loaded through ``SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE=sglang_tpu.multimodal``.
"""

from __future__ import annotations

from typing import ClassVar

from sglang.srt.models.registry import ModelRegistry
from sglang.tml.processor import Inkling2MultimodalProcessor
from sglang_tpu.models.inkling2 import Inkling2ForConditionalGeneration

# The tokenizer process resolves the architecture to pick this processor, but
# TpuPlatform.init_backend registers the TPU models only in scheduler workers.
if not ModelRegistry.is_arch_supported(Inkling2ForConditionalGeneration.__name__):
    ModelRegistry.register("sglang_tpu.models")


class Inkling2TpuMultimodalProcessor(Inkling2MultimodalProcessor):
    models: ClassVar[list] = [Inkling2ForConditionalGeneration]
