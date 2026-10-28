# SPDX-License-Identifier: Apache-2.0
"""Tokenizer loader for the native Muse multimodal processor.

The native processor owns image/video preprocessing, so HF only supplies the
checkpoint tokenizer and its chat template.
"""


class MuseSparkNativeProcessor:
    @classmethod
    def from_pretrained(cls, name, *args, **kwargs):
        from sglang.srt.utils.hf_transformers.tokenizer import get_tokenizer

        kwargs.pop("use_fast", None)
        return get_tokenizer(name, *args, **kwargs)


def register_muse_spark_processor() -> None:
    """Map the core Muse Spark config to the tokenizer-only HF processor."""
    from sglang.srt.configs.muse_spark import MuseSparkConfig
    from sglang.srt.multimodal.customized_mm_processor_utils import (
        register_customized_processor,
    )

    register_customized_processor(MuseSparkNativeProcessor)(MuseSparkConfig)
