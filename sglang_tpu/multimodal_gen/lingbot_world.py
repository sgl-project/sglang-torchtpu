# SPDX-License-Identifier: Apache-2.0
"""LingBot-World on TPU workers: KV-cache attention outside the compiled
blocks; RoPE tables, DMD schedule and camera condition on the host.

Runtime patch, tracked debt:
- Why: each chunk writes its K/V at a new cache position. The write position,
  the cursors and the visible length are Python ints
  (``CausalSelfAttentionKVCache.update_and_get_attention_kv``), which a
  compiled region guards on value, and ``TpuOOTPlatform`` keeps shapes static:
  under ``--regional-compile`` the stock blocks compile 4 or 5 new graphs a
  chunk, up to the recompile limit. Per chunk the stock path also builds the
  Ulysses shard's RoPE tables on the device (int64 position math, float64
  tables, a ``tolist()`` read per axis), finds each DMD sigma with an int64
  ``argmin`` on the device (torch_tpu eager aborts int64 reductions such as
  ``mask.sum()``: "Unsupported conversion from vmreg/vreg to U64"), builds the
  Plucker camera condition from float64 poses on the device with a host read
  of ``max_norm > 0``, and draws the DMD re-noise in bfloat16, where
  torch_tpu's generator has 128 distinct values.
- How: :func:`install`
  (1) rebinds ``LingBotWorldCausalSelfAttention._attend_with_kv_cache`` for TPU
      tensors passed ``current_start=None`` to the stock method under
      ``torch.compiler.disable``: the Ulysses all-to-alls, the cache write and
      the attention over the visible window run eagerly between compiled block
      halves. ``CausalLingBotWorldTransformer3DModel.forward`` leaves the chunk
      start on the caches, so the block graphs compile in the first chunk, and
      a block's second half once more for the camera condition's other state
      (present or absent); later chunks and sessions compile nothing. The
      output is a fresh contiguous tensor:
      torch_tpu guards each compiled input's storage offset, dynamo its strides;
  (2) builds ``_compute_rope_for_sequence_shard_with_offset``'s float64 tables
      with the stock code on the host and copies the float32 result over;
  (3) keeps ``_prepare_causal_dmd_timesteps``'s schedule on the host, where
      ``pred_noise_to_pred_video`` and ``SelfForcingFlowMatchScheduler.
      add_noise`` find each sigma; the device runs the elementwise updates;
  (4) draws ``_add_noise_for_next_timestep``'s noise in float32 on the request
      generator's device and casts it, as CUDA's bfloat16 ``randn`` samples in
      float32; a CPU generator (``generator_device="cpu"``) works too;
  (5) builds ``prepare_world_condition``'s camera condition with the stock code
      on the host and copies it over.
- Upstream PR or issue: none.
- Remove when: (1) the causal KV cache takes its write position and visible
  length as device tensors over a fixed-size window; (2), (3) and (5) torch_tpu
  runs int64 and float64 device math natively; (4) torch_tpu's bfloat16
  ``randn`` samples in float32.
"""

from __future__ import annotations

import torch

from sglang.multimodal_gen.configs.pipeline_configs.lingbot_world import (
    LingBotWorldI2VConfig,
)
from sglang.multimodal_gen.runtime.models.dits.lingbot_world import (
    CausalLingBotWorldTransformer3DModel,
    LingBotWorldCausalSelfAttention,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.lingbot_world.lingbot_world_causal_denoising import (
    LingBotWorldCausalDMDDenoisingStage,
)

# Devices that take these paths.
_DEVICE_TYPES = ("tpu",)

_Model = CausalLingBotWorldTransformer3DModel
_Stage = LingBotWorldCausalDMDDenoisingStage
_STOCK_ATTEND = LingBotWorldCausalSelfAttention._attend_with_kv_cache
_STOCK_SHARD_ROPE = _Model._compute_rope_for_sequence_shard_with_offset
_STOCK_TIMESTEPS = _Stage._prepare_causal_dmd_timesteps
_STOCK_ADD_NOISE = _Stage._add_noise_for_next_timestep
_STOCK_WORLD_CONDITION = LingBotWorldI2VConfig.prepare_world_condition


def _on_tpu(device) -> bool:
    return torch.device(device).type in _DEVICE_TYPES


@torch.compiler.disable
def _attend_eagerly(self, q, k, v, kv_cache, current_start, update_cache_only):
    x = _STOCK_ATTEND(self, q, k, v, kv_cache, current_start, update_cache_only)
    return x.clone(memory_format=torch.contiguous_format)


def _attend_with_kv_cache(
    self, q, k, v, kv_cache, current_start, update_cache_only=False
):
    """``_attend_with_kv_cache``, run eagerly for TPU tensors that take the
    chunk start from the cache."""
    if q.device.type not in _DEVICE_TYPES or current_start is not None:
        return _STOCK_ATTEND(self, q, k, v, kv_cache, current_start, update_cache_only)
    return _attend_eagerly(self, q, k, v, kv_cache, current_start, update_cache_only)


def _compute_rope_for_sequence_shard_with_offset(
    self, local_len, token_start, frame_stride_local, width_local, device
):
    """The Ulysses shard's RoPE tables, built on the host for TPU devices."""
    args = (local_len, token_start, frame_stride_local, width_local)
    if not _on_tpu(device):
        return _STOCK_SHARD_ROPE(self, *args, device)
    cos, sin = _STOCK_SHARD_ROPE(self, *args, torch.device("cpu"))
    return cos.to(device), sin.to(device)


def _prepare_causal_dmd_timesteps(self, batch, server_args, scheduler, device):
    """The DMD schedule, kept on the host for TPU devices."""
    if not _on_tpu(device):
        return _STOCK_TIMESTEPS(self, batch, server_args, scheduler, device)
    return batch.timesteps.cpu()


def _add_noise_for_next_timestep(self, batch, **kwargs):
    """The DMD re-noise, drawn in float32 on the generator's device for TPU
    latents."""
    if not _on_tpu(kwargs["device"]):
        return _STOCK_ADD_NOISE(self, batch, **kwargs)
    x0, device = kwargs["x0_btchw"], kwargs["device"]
    generator = self._single_generator(batch)
    noise = torch.randn(
        kwargs["raw_latent_shape"],
        dtype=torch.float32,
        generator=generator,
        device=device if generator is None else generator.device,
    ).to(device=device, dtype=x0.dtype)
    return (
        kwargs["scheduler"]
        .add_noise(x0.flatten(0, 1), noise.flatten(0, 1), kwargs["next_timestep"])
        .unflatten(0, x0.shape[:2])
    )


def prepare_world_condition(self, batch, device, dtype):
    """The camera condition, built on the host for TPU devices."""
    if not _on_tpu(device):
        return _STOCK_WORLD_CONDITION(self, batch, device, dtype)
    condition = _STOCK_WORLD_CONDITION(self, batch, torch.device("cpu"), dtype)
    if condition is None:
        return None
    return {name: value.to(device) for name, value in condition.items()}


def install() -> None:
    LingBotWorldCausalSelfAttention._attend_with_kv_cache = _attend_with_kv_cache
    _Model._compute_rope_for_sequence_shard_with_offset = (
        _compute_rope_for_sequence_shard_with_offset
    )
    _Stage._prepare_causal_dmd_timesteps = _prepare_causal_dmd_timesteps
    _Stage._add_noise_for_next_timestep = _add_noise_for_next_timestep
    LingBotWorldI2VConfig.prepare_world_condition = prepare_world_condition
