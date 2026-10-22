# SPDX-License-Identifier: Apache-2.0
"""Fixed raw-patch buckets for Muse Spark's replicated NaViT encoder."""

import logging

import torch

from sglang.srt.compilation import torch_compile_bundle
from sglang.srt.compilation.support_torch_compile import CompilePolicy
from sglang_tpu.models.muse_spark_vision import PATCH_BUCKETS, vision_metadata
from sglang_tpu.patches import _register_tpu_compile_backend

logger = logging.getLogger(__name__)


class MuseVisionCompilePolicy(CompilePolicy):
    platform = "tpu"
    backend = "tpu"
    spmd_safe = True

    def install(self):
        _register_tpu_compile_backend()
        cfg = self.model_runner.server_args
        bundled = (
            cfg.unsafe_save_torch_compile_bundle or cfg.unsafe_load_torch_compile_bundle
        )
        compiler = torch_compile_bundle.Compiler if bundled else torch.compile
        self.owner.forward = compiler(
            torch.no_grad()(self.method),
            backend=self.backend,
            fullgraph=True,
            dynamic=False,
        )
        return True

    def precompile(self):
        encoder = self.owner
        config = encoder.config
        param = next(encoder.parameters())
        for bucket in PATCH_BUCKETS:
            # Metadata values do not specialize the graph; lengths remain fixed.
            meta = vision_metadata(
                16,
                bucket // 16,
                encoder.head_dim,
                config.pos_emb_height,
                config.pos_emb_width,
                config.rope_parameters["rope_theta"],
                config.merge_size,
                bucket,
            )
            patches = torch.zeros(
                bucket,
                3 * config.patch_size**2 * config.patch_temporal,
                dtype=param.dtype,
                device=param.device,
            )
            args = (patches, *(t.to(param.device) for t in meta[1:]))
            with torch.no_grad():
                value = encoder(*args)
                value.cpu()
            logger.info("Muse NaViT precompiled raw_patch_bucket=%d", bucket)
