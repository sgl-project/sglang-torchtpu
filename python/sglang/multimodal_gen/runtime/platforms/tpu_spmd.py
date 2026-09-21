# SPDX-License-Identifier: Apache-2.0
"""TPU-isolated SPMD subclasses and runtime adapters for diffusion pipelines.

All multi-rank TPU adaptations (CFG positive/negative text shape alignment,
SP masked text sharding, replicated VAE decode materialization, and CPU-group
executor barriers) are encapsulated here as subclasses and TPU-only runtime
hooks so shared GPU model, stage, pipeline_config, and executor files remain
100% untouched.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

from sglang.multimodal_gen.configs.pipeline_configs.ernie_image import (
    ErnieImagePipelineConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.qwen_image import (
    QwenImagePipelineConfig,
    shard_rotary_emb_for_sp,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_classifier_free_guidance_world_size as get_cfg_world_size,
    get_sp_world_size,
    get_world_group,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.distributed.sp_shard_utils import (
    build_shard_plan,
    join_seqs,
    shard_like,
    should_shard_text,
)
from sglang.multimodal_gen.runtime.models.dits.qwen_image import (
    QwenImageTransformer2DModel,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import (
    StageParallelismType,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.decoding import DecodingStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.text_encoding import (
    TextEncodingStage,
)
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.server_args import ServerArgs, get_global_server_args


def requires_uniform_cfg_branch_shapes() -> bool:
    """True when CFG-parallel ranks must trace identical shapes and collectives."""
    return (
        current_platform.is_tpu()
        and model_parallel_is_initialized()
        and get_cfg_world_size() > 1
    )


def _entry_at(entries: list | None, index: int):
    if entries is None or index >= len(entries):
        return None
    return entries[index]


def _pad_zeros(tensor: torch.Tensor, target_len: int, dim: int) -> torch.Tensor:
    """Right-pad `tensor` along `dim` to `target_len` with zeros (False for bool)."""
    pad_len = target_len - tensor.shape[dim]
    if pad_len <= 0:
        return tensor
    pad_shape = list(tensor.shape)
    pad_shape[dim] = pad_len
    return torch.cat([tensor, tensor.new_zeros(pad_shape)], dim=dim)


def _text_seq_lens_or_full(
    pipeline_config,
    batch: Req,
    encoder_index: int,
    text_seq_len: int,
    batch_size: int,
    *,
    negative: bool = False,
) -> list[int]:
    seq_lens_by_encoder = (
        batch.negative_prompt_seq_lens if negative else batch.prompt_seq_lens
    )
    if (
        seq_lens_by_encoder is None
        or encoder_index >= len(seq_lens_by_encoder)
        or not seq_lens_by_encoder[encoder_index]
    ):
        return [text_seq_len] * batch_size
    return pipeline_config.require_text_seq_lens(
        batch, encoder_index, negative=negative, expected_batch_size=batch_size
    )


def _resolve_tpu_text_conditioning_mask(
    batch: Req,
    encoder_index: int,
    txt_seq_lens: list[int],
    text_seq_len: int,
    batch_size: int,
    device: torch.device,
    *,
    negative: bool = False,
) -> torch.Tensor | None:
    fully_packed = all(seq_len == text_seq_len for seq_len in txt_seq_lens)
    if fully_packed and not requires_uniform_cfg_branch_shapes():
        return None

    masks_by_encoder = (
        batch.negative_prompt_embeds_mask if negative else batch.prompt_embeds_mask
    )
    if masks_by_encoder is not None and encoder_index < len(masks_by_encoder):
        mask = masks_by_encoder[encoder_index]
        if mask is not None:
            if tuple(mask.shape) != (batch_size, text_seq_len):
                kind = "negative" if negative else "positive"
                raise ValueError(
                    f"{kind} text conditioning mask for encoder {encoder_index} "
                    f"has shape {tuple(mask.shape)}, expected "
                    f"{(batch_size, text_seq_len)}."
                )
            return mask.to(device=device, dtype=torch.bool)

    positions = torch.arange(text_seq_len, device=device)
    seq_lens = torch.tensor(txt_seq_lens, device=device, dtype=torch.long)
    return positions.unsqueeze(0) < seq_lens.unsqueeze(1)


class TpuDecodingStageMixin:
    """TPU subclass behavior for `DecodingStage`."""

    @property
    def parallelism_type(self) -> StageParallelismType:
        if current_platform.is_tpu():
            server_args = get_global_server_args()
            if server_args is not None and server_args.enable_cfg_parallel:
                # Every rank in a TPU SPMD slice must enqueue the same device
                # programs in lockstep; running VAE decode on all ranks avoids
                # main-rank divergence before host materialization.
                return StageParallelismType.REPLICATED
        return super().parallelism_type

    def decode(
        self, latents: torch.Tensor, server_args: ServerArgs, vae_dtype=None
    ) -> torch.Tensor:
        image = super().decode(latents, server_args, vae_dtype=vae_dtype)
        if current_platform.is_tpu():
            # Materialize lazy XLA tensors to CPU inside the stage while all
            # TPU ranks are synchronized in REPLICATED execution.
            image = image.cpu().float()
        return image


class TpuDecodingStage(TpuDecodingStageMixin, DecodingStage):
    pass


class TpuTextEncodingStageMixin:
    """TPU subclass behavior for `TextEncodingStage`."""

    def _align_cfg_branch_text_lengths(
        self, batch: Req, server_args: ServerArgs
    ) -> None:
        pipeline_config = server_args.pipeline_config
        sp_size = max(get_sp_world_size(), 1)
        num_encoders = min(len(batch.prompt_embeds), len(batch.negative_prompt_embeds))

        supports_mask = (
            hasattr(pipeline_config, "_prepare_encoder_hidden_states_mask")
            or getattr(pipeline_config, "supports_text_padding_mask", lambda: False)()
        )

        for encoder_index in range(num_encoders):
            pos = batch.prompt_embeds[encoder_index]
            neg = batch.negative_prompt_embeds[encoder_index]
            if pos.ndim < 2 or neg.ndim < 2:
                continue

            longest = max(pos.shape[-2], neg.shape[-2])
            target_len = -(-longest // sp_size) * sp_size
            if pos.shape[-2] == target_len and neg.shape[-2] == target_len:
                continue

            if not supports_mask:
                raise ValueError(
                    f"{type(pipeline_config).__name__} produces text lengths that "
                    f"differ across CFG branches (conditional={pos.shape[-2]}, "
                    f"unconditional={neg.shape[-2]}) but does not honor a text "
                    "padding mask, so the branches cannot be padded to a common "
                    "length for cfg_parallel."
                )

            self._pad_branch_text_conditioning(
                batch, encoder_index, target_len, negative=False
            )
            self._pad_branch_text_conditioning(
                batch, encoder_index, target_len, negative=True
            )

    @staticmethod
    def _pad_branch_text_conditioning(
        batch: Req, encoder_index: int, target_len: int, *, negative: bool
    ) -> None:
        embeds_list = batch.negative_prompt_embeds if negative else batch.prompt_embeds
        embeds_mask_list = (
            batch.negative_prompt_embeds_mask if negative else batch.prompt_embeds_mask
        )
        attention_mask_list = (
            batch.negative_attention_mask if negative else batch.prompt_attention_mask
        )

        embeds = embeds_list[encoder_index]
        source_len = embeds.shape[-2]
        if source_len == target_len:
            return

        attention_mask = _entry_at(attention_mask_list, encoder_index)
        mask = _entry_at(embeds_mask_list, encoder_index)
        if mask is None and attention_mask is not None:
            if attention_mask.shape[-1] == source_len:
                mask = attention_mask.to(dtype=torch.bool)
        if mask is None:
            kind = "negative" if negative else "positive"
            raise ValueError(
                f"Cannot pad {kind} text conditioning for encoder {encoder_index} "
                f"from {source_len} to {target_len}: no embedding-aligned mask was "
                "recorded."
            )

        embeds_list[encoder_index] = _pad_zeros(embeds, target_len, dim=-2)
        if embeds_mask_list is not None and encoder_index < len(embeds_mask_list):
            embeds_mask_list[encoder_index] = _pad_zeros(mask, target_len, dim=-1)

        if attention_mask is not None and attention_mask.shape[-1] == source_len:
            attention_mask_list[encoder_index] = _pad_zeros(
                attention_mask, target_len, dim=-1
            )

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        batch = super().forward(batch, server_args)
        if (
            requires_uniform_cfg_branch_shapes()
            and batch.do_classifier_free_guidance
            and batch.negative_prompt_embeds
        ):
            self._align_cfg_branch_text_lengths(batch, server_args)
        return batch


class TpuTextEncodingStage(TpuTextEncodingStageMixin, TextEncodingStage):
    pass


class TpuQwenImagePipelineConfigMixin:
    """TPU subclass behavior for `QwenImagePipelineConfig`."""

    def _prepare_text_conditioning(
        self,
        batch,
        encoder_index: int,
        text_seq_len: int,
        batch_size: int,
        *,
        negative: bool = False,
    ):
        if not requires_uniform_cfg_branch_shapes():
            return super()._prepare_text_conditioning(
                batch,
                encoder_index,
                text_seq_len,
                batch_size,
                negative=negative,
            )

        txt_seq_lens = _text_seq_lens_or_full(
            self,
            batch,
            encoder_index,
            text_seq_len,
            batch_size,
            negative=negative,
        )
        encoder_hidden_states_mask = _resolve_tpu_text_conditioning_mask(
            batch,
            encoder_index,
            txt_seq_lens,
            text_seq_len,
            batch_size,
            batch.prompt_embeds[0].device,
            negative=negative,
        )
        return txt_seq_lens, encoder_hidden_states_mask

    def _prepare_cond_kwargs(
        self, batch, prompt_embeds, rotary_emb, device, dtype, *, negative=False
    ):
        if not requires_uniform_cfg_branch_shapes():
            return super()._prepare_cond_kwargs(
                batch, prompt_embeds, rotary_emb, device, dtype, negative=negative
            )

        batch_size = prompt_embeds.shape[0]
        text_seq_len = prompt_embeds.shape[1]
        height = batch.height
        width = batch.width
        vae_scale_factor = self.get_vae_scale_factor()

        img_shapes = [
            [
                (
                    1,
                    height // vae_scale_factor // 2,
                    width // vae_scale_factor // 2,
                )
            ]
        ] * batch_size
        txt_seq_lens, encoder_hidden_states_mask = self._prepare_text_conditioning(
            batch, 0, text_seq_len, batch_size, negative=negative
        )

        if rotary_emb is None:
            return {
                "img_shapes": img_shapes,
                "txt_seq_lens": txt_seq_lens,
                "freqs_cis": None,
                "encoder_hidden_states_mask": encoder_hidden_states_mask,
            }

        rope_txt_seq_lens = [max(text_seq_len, *txt_seq_lens)] * batch_size
        img_cache, txt_cache = self.get_freqs_cis(
            img_shapes, rope_txt_seq_lens, rotary_emb, device, dtype
        )
        img_cache = shard_rotary_emb_for_sp(img_cache)
        return {
            "txt_seq_lens": txt_seq_lens,
            "freqs_cis": (img_cache, txt_cache),
            "img_shapes": img_shapes,
            "encoder_hidden_states_mask": encoder_hidden_states_mask,
        }


class TpuQwenImagePipelineConfig(
    TpuQwenImagePipelineConfigMixin, QwenImagePipelineConfig
):
    pass


class TpuErnieImagePipelineConfigMixin:
    """TPU subclass behavior for `ErnieImagePipelineConfig`."""

    def _prepare_encoder_hidden_states_mask(
        self,
        batch,
        txt_seq_lens: list[int],
        text_seq_len: int,
        device,
        *,
        negative: bool = False,
    ):
        if not requires_uniform_cfg_branch_shapes():
            return super()._prepare_encoder_hidden_states_mask(
                batch, txt_seq_lens, text_seq_len, device
            )
        return _resolve_tpu_text_conditioning_mask(
            batch,
            0,
            txt_seq_lens,
            text_seq_len,
            len(txt_seq_lens),
            device,
            negative=negative,
        )

    def _prepare_cond_kwargs(
        self, batch, prompt_embeds, rotary_emb, device, dtype, *, negative: bool = False
    ):
        if not requires_uniform_cfg_branch_shapes():
            return super()._prepare_cond_kwargs(
                batch, prompt_embeds, rotary_emb, device, dtype, negative=negative
            )

        batch_size = prompt_embeds.shape[0]
        text_seq_len = prompt_embeds.shape[1]
        vae_scale_factor = self.get_vae_scale_factor()
        img_shapes = [
            [
                (
                    1,
                    batch.height // vae_scale_factor,
                    batch.width // vae_scale_factor,
                )
            ]
        ] * batch_size

        txt_seq_lens = self.require_text_seq_lens(
            batch, 0, negative=negative, expected_batch_size=batch_size
        )
        encoder_hidden_states_mask = self._prepare_encoder_hidden_states_mask(
            batch, txt_seq_lens, text_seq_len, device, negative=negative
        )

        if rotary_emb is None:
            return {
                "img_shapes": img_shapes,
                "txt_seq_lens": txt_seq_lens,
                "freqs_cis": None,
                "encoder_hidden_states_mask": encoder_hidden_states_mask,
            }

        rope_txt_seq_lens = [max(text_seq_len, *txt_seq_lens)] * batch_size
        freqs_cis = self.get_freqs_cis(
            img_shapes, rope_txt_seq_lens, rotary_emb, device, dtype
        )
        if isinstance(freqs_cis, tuple):
            img_cache, txt_cache = freqs_cis
            img_cache = shard_rotary_emb_for_sp(img_cache)
            freqs_cis = (img_cache, txt_cache)

        return {
            "txt_seq_lens": txt_seq_lens,
            "freqs_cis": freqs_cis,
            "img_shapes": img_shapes,
            "encoder_hidden_states_mask": encoder_hidden_states_mask,
        }


class TpuErnieImagePipelineConfig(
    TpuErnieImagePipelineConfigMixin, ErnieImagePipelineConfig
):
    pass


class TpuQwenImageTransformer2DModelMixin:
    """TPU subclass behavior for `QwenImageTransformer2DModel`."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        encoder_hidden_states_mask: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_shapes: Optional[List[Tuple[int, int, int]]] = None,
        txt_seq_lens: Optional[List[int]] = None,
        freqs_cis: Tuple[torch.Tensor, torch.Tensor] = None,
        guidance: torch.Tensor = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_block_samples=None,
        additional_t_cond=None,
        return_dict: bool = True,
        modulation_cache_key: tuple[int, float] | None = None,
    ):
        if current_platform.is_tpu():
            if isinstance(encoder_hidden_states_mask, list):
                encoder_hidden_states_mask = encoder_hidden_states_mask[0]
            if (
                encoder_hidden_states is not None
                and encoder_hidden_states_mask is not None
                and should_shard_text(encoder_hidden_states.shape[1])
            ):
                txt_shard = build_shard_plan(encoder_hidden_states.shape[1])
                local_txt_mask = shard_like(
                    encoder_hidden_states_mask.to(
                        device=hidden_states.device, dtype=torch.bool
                    ),
                    txt_shard,
                    dim=1,
                )
                batch_size, image_seq_len = hidden_states.shape[:2]
                image_mask = torch.ones(
                    (batch_size, image_seq_len),
                    dtype=torch.bool,
                    device=hidden_states.device,
                )
                local_joint_mask = join_seqs(
                    local_txt_mask, image_mask, txt_shard.local_pad, dim=1
                )
                attention_kwargs = dict(attention_kwargs or {})
                attention_kwargs["attn_mask"] = local_joint_mask
                if txt_shard.local_pad > 0:
                    attention_kwargs["attn_mask_meta"] = {
                        "local_pad": txt_shard.local_pad
                    }
                # Pass encoder_hidden_states_mask=None so super().forward() takes
                # its `elif should_shard_text(...)` branch, sharding
                # encoder_hidden_states and freqs_cis while forwarding our
                # sharded `attn_mask` in `block_attention_kwargs`.
                encoder_hidden_states_mask = None

        return super().forward(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            timestep=timestep,
            img_shapes=img_shapes,
            txt_seq_lens=txt_seq_lens,
            freqs_cis=freqs_cis,
            guidance=guidance,
            attention_kwargs=attention_kwargs,
            controlnet_block_samples=controlnet_block_samples,
            additional_t_cond=additional_t_cond,
            return_dict=return_dict,
            modulation_cache_key=modulation_cache_key,
        )


class TpuQwenImageTransformer2DModel(
    TpuQwenImageTransformer2DModelMixin, QwenImageTransformer2DModel
):
    pass


_SUBCLASS_CACHE: dict[tuple[type, type], type] = {}
_INSTALLED = False


def _get_tpu_subclass(cls: type, mixin_cls: type, default_subcls: type) -> type:
    if issubclass(cls, mixin_cls):
        return cls
    if cls is default_subcls.__mro__[2]:
        return default_subcls
    key = (cls, mixin_cls)
    subcls = _SUBCLASS_CACHE.get(key)
    if subcls is None:
        subcls = type(f"Tpu{cls.__name__}", (mixin_cls, cls), {})
        _SUBCLASS_CACHE[key] = subcls
    return subcls


def _install_subclass_factory(
    base_cls: type, mixin_cls: type, default_subcls: type
) -> None:
    if getattr(base_cls, "_tpu_subclass_installed", False):
        return
    orig_new = base_cls.__new__

    def _tpu_aware_new(cls, *args, **kwargs):
        if current_platform.is_tpu() and not issubclass(cls, mixin_cls):
            cls = _get_tpu_subclass(cls, mixin_cls, default_subcls)
        if orig_new is object.__new__:
            return object.__new__(cls)
        return orig_new(cls, *args, **kwargs)

    base_cls.__new__ = staticmethod(_tpu_aware_new)
    base_cls._tpu_subclass_installed = True


def _install_tpu_cpu_barrier() -> None:
    if getattr(dist, "_tpu_cpu_barrier_installed", False):
        return
    orig_barrier = dist.barrier

    def _tpu_safe_barrier(group=dist.GroupMember.WORLD, *args, **kwargs):
        if (
            current_platform.is_tpu()
            and (group is None or group == dist.GroupMember.WORLD)
            and model_parallel_is_initialized()
        ):
            world_group = get_world_group()
            if world_group is not None and world_group.cpu_group is not None:
                return orig_barrier(group=world_group.cpu_group, *args, **kwargs)
        return orig_barrier(group=group, *args, **kwargs)

    dist.barrier = _tpu_safe_barrier
    dist._tpu_cpu_barrier_installed = True


def upgrade_instance_to_tpu_subclass(obj: Any) -> Any:
    """Upgrade an already-instantiated config/stage/model to its TPU subclass."""
    if obj is None or not current_platform.is_tpu():
        return obj
    for base_cls, mixin_cls, default_subcls in (
        (
            QwenImagePipelineConfig,
            TpuQwenImagePipelineConfigMixin,
            TpuQwenImagePipelineConfig,
        ),
        (
            ErnieImagePipelineConfig,
            TpuErnieImagePipelineConfigMixin,
            TpuErnieImagePipelineConfig,
        ),
        (DecodingStage, TpuDecodingStageMixin, TpuDecodingStage),
        (TextEncodingStage, TpuTextEncodingStageMixin, TpuTextEncodingStage),
        (
            QwenImageTransformer2DModel,
            TpuQwenImageTransformer2DModelMixin,
            TpuQwenImageTransformer2DModel,
        ),
    ):
        if isinstance(obj, base_cls) and not isinstance(obj, mixin_cls):
            obj.__class__ = _get_tpu_subclass(type(obj), mixin_cls, default_subcls)
            break
    return obj


def install_tpu_spmd_overrides() -> None:
    """Register TPU subclasses and CPU-group barrier routing when running on TPU."""
    global _INSTALLED
    if not _INSTALLED:
        _install_subclass_factory(
            DecodingStage, TpuDecodingStageMixin, TpuDecodingStage
        )
        _install_subclass_factory(
            TextEncodingStage, TpuTextEncodingStageMixin, TpuTextEncodingStage
        )
        _install_subclass_factory(
            QwenImagePipelineConfig,
            TpuQwenImagePipelineConfigMixin,
            TpuQwenImagePipelineConfig,
        )
        _install_subclass_factory(
            ErnieImagePipelineConfig,
            TpuErnieImagePipelineConfigMixin,
            TpuErnieImagePipelineConfig,
        )
        _install_subclass_factory(
            QwenImageTransformer2DModel,
            TpuQwenImageTransformer2DModelMixin,
            TpuQwenImageTransformer2DModel,
        )
        _install_tpu_cpu_barrier()
        _INSTALLED = True

    if current_platform.is_tpu():
        server_args = get_global_server_args()
        if server_args is not None and getattr(server_args, "pipeline_config", None):
            upgrade_instance_to_tpu_subclass(server_args.pipeline_config)
