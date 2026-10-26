# SPDX-License-Identifier: Apache-2.0
"""TPU adaptations of the EPD encode server (--encoder-only).

Why: the encode server has no Scheduler or ModelRunner, so it skips the torch_tpu
     worker bootstrap, TpuGraphRunner never installs the vision compile
     policies, and its batches can exceed the largest precompiled patch bucket.
How: BEFORE MMEncoder.__init__ bootstraps the torch_tpu worker and requires
     --enable-torch-compile; AFTER it installs and warms the vision policies; AROUND
     MMEncoder.encode_indices splits a batch to the patch budget; AFTER
     MMEncoder.init_moe_gemm_config resolves the dense MoE flags; BEFORE
     launch_dp_runtime rejects encoder DP.
Upstream PR or issue: not filed.
Remove when: core exposes a platform seam for encoder compilation and batching.
"""

from __future__ import annotations

from typing import Callable

import torch

from sglang.srt.disaggregation.encoder.preprocessor import EncoderPreprocessResult
from sglang.srt.disaggregation.encoder.server import MMEncoder
from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils.patch_torch import initialize_torch_tpu_worker


def bootstrap_encoder_worker(
    encoder: MMEncoder,
    server_args: ServerArgs,
    schedule_path=None,
    dist_init_method=None,
    rank: int = 0,
    gpu_id: int | None = None,
):
    # Uncompiled, the ViT recompiles per image shape.
    if not server_args.enable_torch_compile:
        raise ValueError("The TPU encode server requires --enable-torch-compile.")
    # Must precede set_device and the distributed init in MMEncoder.__init__.
    initialize_torch_tpu_worker(server_args, rank)


def install_encoder_vision_compile(result, encoder: MMEncoder, *args, **kwargs):
    # Deferred: torch_tpu is absent on the CPU hosts that also import these hooks.
    from sglang_tpu.compilation.encoder_compile_policy import (
        install_vision_compile_policies,
    )

    install_vision_compile_policies(encoder.model)
    return result


def split_indices_by_patch_budget(
    patch_counts: list[int], indices: list[int], budget: int
) -> list[list[int]]:
    """Greedily group indices so each group's patch total fits budget.

    An index above the budget on its own gets a group of its own.
    """
    groups: list[list[int]] = []
    current: list[int] = []
    current_patches = 0
    for index, patches in zip(indices, patch_counts):
        if current and current_patches + patches > budget:
            groups.append(current)
            current, current_patches = [], 0
        current.append(index)
        current_patches += patches
    if current:
        groups.append(current)
    return groups


def encode_indices_within_patch_budget(
    original,
    encoder: MMEncoder,
    mm_feature,
    preprocess_result: EncoderPreprocessResult,
    indices: list[int],
    modality: Modality,
    get_feature_fn: Callable[[list[MultimodalDataItem]], torch.Tensor],
) -> torch.Tensor:
    if modality == Modality.AUDIO:
        return original(
            encoder, mm_feature, preprocess_result, indices, modality, get_feature_fn
        )
    from sglang_tpu.graph_runner.vit_tpu_compile_runner import (
        precompiled_max_patch_count,
    )

    # Split grid rows, not items: the aggregate processor path packs every image
    # of a request into one item with a [num_images, 3] grid.
    patch_counts = [
        encoder.preprocessor.get_num_patches(preprocess_result.grid_thw[i], modality)
        for i in indices
    ]
    groups = split_indices_by_patch_budget(
        patch_counts, indices, precompiled_max_patch_count()
    )
    # no_grad, not inference_mode: the compiled boundary is traced and warmed
    # under no_grad, and inference tensors would miss its guards.
    with torch.no_grad():
        outputs = [
            get_feature_fn(
                encoder._build_model_mm_items(
                    mm_feature, preprocess_result, group, modality
                )
            )
            for group in groups
        ]
    return torch.cat([output.reshape(-1, output.shape[-1]) for output in outputs])


def reject_encoder_dp(server_args: ServerArgs):
    # Every DP worker would bootstrap as rank 0 of a world of 1 and bind one chip.
    raise ValueError(
        "Encoder DP mode is not supported on TPU; scale the encoder with "
        "--tp-size instead."
    )
