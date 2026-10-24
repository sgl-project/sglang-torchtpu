"""Compile policies the plugin attaches to core model classes.

Core model modules must not import ``sglang_tpu`` (the ratchet in
``tools/plugin/check_tpu_branches.py``), so a TPU compile boundary that core does
not declare itself with ``support_torch_compile`` is declared here instead.

``TpuGraphRunner`` calls :func:`annotate_core_compile_boundaries` right before it
discovers policies. The model is loaded by then, so this only looks at modules
already in ``sys.modules``: it never imports a model module for a model that is
not being served.
"""

import sys
from typing import Callable, NamedTuple

from sglang.srt.compilation.support_torch_compile import (
    CompilePolicy,
    get_torch_compile_policy,
    support_torch_compile,
)


class _CoreCompileBoundary(NamedTuple):
    module: str
    class_name: str
    # Imported lazily: the policy modules pull in TorchTPU.
    policy: Callable[[], type[CompilePolicy]]


def _qwen3_vl_decoder_policy() -> type[CompilePolicy]:
    from sglang_tpu.compilation.decoder_compile_policy import (
        Qwen3VLDecoderTpuCompilePolicy,
    )

    return Qwen3VLDecoderTpuCompilePolicy


def _kimi_k3_vision_policy() -> type[CompilePolicy]:
    from sglang_tpu.graph_runner.kimi_k3_vit_tpu_compile_runner import (
        KimiK3VisionTpuCompilePolicy,
    )

    return KimiK3VisionTpuCompilePolicy


CORE_COMPILE_BOUNDARIES: tuple[_CoreCompileBoundary, ...] = (
    # The Qwen3-VL MoE decoder (Qwen3-VL-30B-A3B, 235B-A22B). qwen3_vl annotates
    # only the dense Qwen3LLMModel and the vision tower, and the vision tower's
    # annotation alone suppresses TpuGraphRunner's whole-model fallback. Without
    # this entry the MoE decoder runs eager under --enable-torch-compile while
    # only the ViT compiles. Annotating Qwen3MoeLLMModel rather than its base
    # Qwen3MoeModel keeps the text-only Qwen3-MoE models on the fallback. Models
    # built on this class compile their decoder too: InternS1-Pro subclasses it,
    # and the Qwen3-Omni thinker uses it as-is.
    _CoreCompileBoundary(
        module="sglang.srt.models.qwen3_vl_moe",
        class_name="Qwen3MoeLLMModel",
        policy=_qwen3_vl_decoder_policy,
    ),
    # Kimi-K3's vision encoder. Its layer loop runs on the TPU runner the
    # plugin attaches as graph_runners (sglang_tpu.hooks.kimi_k3_vision); this
    # compiles that runner per padded patch bucket. The language model carries
    # its own annotation, so this does not suppress the decoder's compile.
    _CoreCompileBoundary(
        module="sglang.srt.models.kimi_k3_vl",
        class_name="MoonViT3dEncoder",
        policy=_kimi_k3_vision_policy,
    ),
)


def annotate_core_compile_boundaries() -> None:
    """Attach each boundary's policy to its class, if the class is loaded.

    A class that already carries a policy keeps it, so a boundary that core
    later annotates itself wins over this table.
    """
    for boundary in CORE_COMPILE_BOUNDARIES:
        module = sys.modules.get(boundary.module)
        if module is None:
            continue
        cls = getattr(module, boundary.class_name)
        if get_torch_compile_policy(cls) is None:
            support_torch_compile(cls, policy=boundary.policy())
