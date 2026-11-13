# SPDX-License-Identifier: Apache-2.0
"""Fixed row buckets for Inkling2's replicated vision encoder."""

import logging

import torch

from sglang.srt.compilation import torch_compile_bundle
from sglang.srt.compilation.support_torch_compile import CompilePolicy
from sglang_tpu.models.inkling2_vision import ROW_BUCKETS, vision_metadata
from sglang_tpu.patches import _register_tpu_compile_backend

logger = logging.getLogger(__name__)


class Inkling2VisionCompilePolicy(CompilePolicy):
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
        param = encoder.proj_out.weight
        for bucket in ROW_BUCKETS:
            # Metadata values do not specialize the graph; only the bucket does.
            meta = vision_metadata(
                ((1, 1),),
                merge=encoder.merge_size,
                head_dim=encoder.head_dim,
                posemb_grid=encoder.posemb_grid_size,
                bucket=bucket,
            )
            pixels = torch.zeros(
                bucket,
                3 * encoder.patch_size**2,
                dtype=param.dtype,
                device=param.device,
            )
            args = (pixels, *(t.to(param.device) for t in meta))
            with torch.no_grad():
                encoder.forward(*args)
            logger.info("[Inkling2 vision] compiled row bucket %d", bucket)
