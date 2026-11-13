# SPDX-License-Identifier: Apache-2.0
"""Cosmos3 on TPU workers: one compiled program per GEN layer, size and prompt bucket.

Runtime patch, tracked debt:
- Why: torch_tpu compiles static shapes. Each GEN layer cross-attends to the
  text pathway's K/V at the prompt's exact token count, so every prompt
  length compiles all GEN layers again. The text and visual RoPE ids are
  built on the device from ``text_mask.sum``, read back with ``.item()`` for
  every sample of every forward.
- How: :func:`install`
  (1) wraps the transformer's ``__call__``: the prompt's tokens right-pad to a
      multiple of 512 (masked, as the stock pads them), and the real token
      count goes to every GEN cross-attention as a device int32 tensor. The
      text pathway is causal and zeroes masked rows, so the real K/V rows are
      the stock ones;
  (2) builds the RoPE ids on the host, from the same functions, and moves
      them to the device (no device-to-host read per forward);
  (3) replaces ``USPAttention.forward_with_replicated_kv_prefix`` for
      attentions that carry that count: queries attend the real text keys,
      then the visual keys, as resumable kernel steps (Ulysses splits the
      heads as the stock does). These are the keys the stock attends, and
      padded text keys are never read. Control, action and sound calls, and
      ring or K/V-gather SP, run the stock transformer and attention;
  (4) runs text CFG as the serial forwards when both prompts tokenize to the
      same length, so the GEN layers keep batch 1;
  (5) hands the compiled GEN layers their tensors laid out as new ones, as
      each program is guarded on its inputs' storage offsets (torch_tpu) and
      strides (Dynamo): the text pathway's K/V (views of the fused QKV
      projection on a conditioning-cache miss, copies on a hit) are copied
      once per prompt, and each compiled layer copies any other view, e.g.
      this rank's Ulysses rows of the assembled control/sound stream.
- Upstream PR or issue: none.
- Remove when: the text K/V cache takes a platform prompt bucket with its real
  length as data, and the RoPE ids are built from host lengths.
"""

from __future__ import annotations

import functools

import torch
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_ring_parallel_world_size,
    get_sp_world_size,
    get_ulysses_parallel_rank,
)
from sglang.multimodal_gen.runtime.layers.attention.layer import USPAttention
from sglang.multimodal_gen.runtime.layers.usp import (
    _usp_input_all_to_all,
    _usp_output_all_to_all,
)
from sglang.multimodal_gen.runtime.models.dits import cosmos3video as dit
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages import (
    cosmos3 as stage,
)
from sglang_tpu.multimodal_gen import tpu_splash_attn

# Prompts of up to 512 tokens share a program; the kernel's text step needs
# >= 512 keys.
TEXT_ALIGNMENT = 512
# Devices that take these paths.
_DEVICE_TYPES = ("tpu",)
# The real text-key count, int32 [0, n], shared by every GEN cross-attention.
_TEXT_KEYS = "_tpu_text_keys"

_STOCK_POSITIONS = dit.Cosmos3OmniTransformer._compute_rope_position_ids
_STOCK_PREFIX_ATTENTION = USPAttention.forward_with_replicated_kv_prefix
_STOCK_CFG_BATCHED = stage.Cosmos3DenoisingStage._predict_noise_cfg_batched
_STOCK_COMPILE = stage.Cosmos3DenoisingStage._maybe_enable_torch_compile
_STOCK_UND_FORWARD = dit.Cosmos3LanguageModel.forward


def compute_rope_position_ids(self, text_mask, T, Hp, Wp, fps, device, **kwargs):
    """``_compute_rope_position_ids`` on the host, moved to ``device``."""
    if torch.device(device).type not in _DEVICE_TYPES:
        return _STOCK_POSITIONS(self, text_mask, T, Hp, Wp, fps, device, **kwargs)
    text_pos, vis_pos = _STOCK_POSITIONS(
        self, text_mask.cpu(), T, Hp, Wp, fps, torch.device("cpu"), **kwargs
    )
    return text_pos.to(device), vis_pos.to(device)


def _gen_attentions(transformer) -> list[USPAttention]:
    return [
        getattr(layer, "_orig_mod", layer).cross_attention.attn
        for layer in transformer.gen_layers
    ]


def _shards_heads_only(attn) -> bool:
    """Whether ``attn`` runs without SP or with Ulysses alone."""
    if attn.skip_sequence_parallel or get_sp_world_size() == 1:
        return True
    return get_ring_parallel_world_size() == 1 and attn.sp_attention_mode != "kv_gather"


def _tie_text_keys(transformer, keys: torch.Tensor | None) -> None:
    """Points every GEN cross-attention's count buffer at ``keys``.

    ``None`` sends the layers to the stock attention. ``Module.to`` copies each
    layer's buffer separately, so the layers are tied again on every call.
    """
    for attn in _gen_attentions(transformer):
        if getattr(attn, _TEXT_KEYS, None) is not keys:
            attn.register_buffer(_TEXT_KEYS, keys, persistent=False)


def _text_keys(transformer, device) -> torch.Tensor:
    """The count tensor shared by every GEN cross-attention.

    Updated in place per call: the compiled layers read it as an input.
    """
    keys = transformer.__dict__.get(_TEXT_KEYS)
    if keys is None or keys.device != torch.device(device):
        keys = torch.zeros(2, dtype=torch.int32, device=device)
        transformer.__dict__[_TEXT_KEYS] = keys
    _tie_text_keys(transformer, keys)
    return keys


@functools.lru_cache(maxsize=64)
def _counts(real: int, device) -> torch.Tensor:
    return torch.tensor([0, real], dtype=torch.int32).to(device)


def _transformer_call(self, *args, **kwargs):
    """Pads the prompt's tokens to the bucket before the transformer runs."""
    latents = kwargs.get("hidden_states")
    text_ids, text_mask = kwargs.get("text_ids"), kwargs.get("text_mask")
    if (
        args
        or not isinstance(latents, torch.Tensor)
        or latents.device.type not in _DEVICE_TYPES
        or text_ids is None
        or text_mask is None
        # Control clips, actions and sound run the stock path.
        or any(
            kwargs.get(name) is not None
            for name in ("control_latents", "action_latents", "sound_latents")
        )
        # So do ring and K/V-gather SP.
        or not _shards_heads_only(_gen_attentions(self)[0])
    ):
        _tie_text_keys(self, None)
        return torch.nn.Module.__call__(self, *args, **kwargs)
    real = kwargs.get("max_text_seq_len")
    if real is None:
        real = int(text_mask.cpu().sum(dim=1).max())
    real = min(int(real), text_ids.shape[1])
    bucket = -(-max(real, 1) // TEXT_ALIGNMENT) * TEXT_ALIGNMENT
    # Rows past ``real`` are masked; past the bucket they are dropped.
    text_ids, text_mask = text_ids[:, :real], text_mask[:, :real]
    text_ids = F.pad(text_ids, (0, bucket - real))
    text_mask = F.pad(text_mask, (0, bucket - real))
    _text_keys(self, latents.device).copy_(_counts(real, latents.device))
    kwargs.update(text_ids=text_ids, text_mask=text_mask, max_text_seq_len=bucket)
    return torch.nn.Module.__call__(self, **kwargs)


def _uses_kernel(impl, q, keys) -> bool:
    if not getattr(impl, "use_splash", False) or q.device.type != "tpu":
        return False
    # Every step takes the kernel or none does: their carries differ.
    return all(
        tpu_splash_attn._ring_kernel_fallback_reason(q, key, key) is None
        for key in keys
    )


def prefix_attention(self, q, k_prefix, v_prefix, k_suffix, v_suffix):
    """``forward_with_replicated_kv_prefix`` over the prefix's real keys.

    ``q``/``k_suffix``/``v_suffix`` are this rank's rows ``[B, S, H, D]``;
    ``k_prefix``/``v_prefix`` hold every head, replicated over SP ranks.
    """
    text_keys = getattr(self, _TEXT_KEYS, None)
    if text_keys is None or q.device.type not in _DEVICE_TYPES:
        return _STOCK_PREFIX_ATTENTION(self, q, k_prefix, v_prefix, k_suffix, v_suffix)
    sp = 1 if self.skip_sequence_parallel else get_sp_world_size()
    if sp > 1:
        q, k_suffix, v_suffix = (
            _usp_input_all_to_all(t, head_dim=2) for t in (q, k_suffix, v_suffix)
        )
        # The prefix heads paired with this rank's suffix heads, as the stock.
        heads = k_suffix.shape[2]
        start = get_ulysses_parallel_rank() * heads
        k_prefix, v_prefix = (t.narrow(2, start, heads) for t in (k_prefix, v_prefix))
    q = q.transpose(1, 2)
    keys = [k_prefix.transpose(1, 2), k_suffix.transpose(1, 2)]
    values = [v_prefix.transpose(1, 2), v_suffix.transpose(1, 2)]
    counts = [text_keys, _counts(keys[1].shape[2], q.device)]
    impl = self.attn_impl
    if _uses_kernel(impl, q, keys):
        carry = None
        for key, value, count in zip(keys, values, counts, strict=True):
            out, carry = impl.forward_ring_resume(q, key, value, count, carry)
    else:
        valid = torch.cat(
            [
                torch.arange(key.shape[2], device=q.device) < count[1]
                for key, count in zip(keys, counts, strict=True)
            ]
        )
        # Zero masked keys and values: a zero probability times NaN is NaN.
        # float32, as the platform's SDPA fallback.
        out = F.scaled_dot_product_attention(
            q.float(),
            torch.where(valid[:, None], torch.cat(keys, 2), 0).float(),
            torch.where(valid[:, None], torch.cat(values, 2), 0).float(),
            attn_mask=valid[None, None, None],
            scale=impl.softmax_scale,
            enable_gqa=True,
        ).to(q.dtype)
    out = out.transpose(1, 2)
    return _usp_output_all_to_all(out, head_dim=2) if sp > 1 else out


def predict_noise_cfg_batched(self, *, max_text_seq_len=None, **kwargs):
    """``_predict_noise_cfg_batched`` as the serial forwards on ``_DEVICE_TYPES``.

    Both take ``uncond + g * (cond - uncond)``; batch 1 reuses the GEN programs.
    """
    if kwargs["latents"].device.type not in _DEVICE_TYPES:
        return _STOCK_CFG_BATCHED(self, max_text_seq_len=max_text_seq_len, **kwargs)
    return self._predict_noise_text_cfg_serial(
        cond_text_seq_len=max_text_seq_len,
        uncond_text_seq_len=max_text_seq_len,
        **kwargs,
    )


def _contiguous_strides(shape) -> tuple[int, ...]:
    strides, step = [], 1
    for size in reversed(shape):
        strides.append(step)
        step *= max(size, 1)
    return tuple(reversed(strides))


def _fresh(t):
    """``t``, or a contiguous copy if ``t`` is laid out unlike a new tensor.

    A view whose storage offset or strides (size-1 dims included) change per
    call or per rank compiles the program again.
    """
    if (
        isinstance(t, torch.Tensor)
        and t.device.type in _DEVICE_TYPES
        and (t.storage_offset() != 0 or t.stride() != _contiguous_strides(t.shape))
    ):
        return t.clone(memory_format=torch.contiguous_format)
    return t


def und_forward(self, *args, **kwargs):
    """``Cosmos3LanguageModel.forward`` with every layer's K/V fresh."""
    kv = _STOCK_UND_FORWARD(self, *args, **kwargs)
    return [tuple(map(_fresh, layer_kv)) for layer_kv in kv]


def _fresh_gen_layer_inputs(layer, args, kwargs):
    return tuple(map(_fresh, args)), {
        name: _fresh(value) for name, value in kwargs.items()
    }


def maybe_enable_torch_compile(self, transformer, server_args):
    """``_maybe_enable_torch_compile``; each compiled GEN layer then copies the
    views it is called with (:func:`_fresh`) before its program runs."""
    _STOCK_COMPILE(self, transformer, server_args)
    if getattr(transformer, "_gen_layers_torch_compiled", False):
        # On the compiled wrapper, so the hook runs outside the program.
        for layer in transformer.gen_layers:
            layer.register_forward_pre_hook(_fresh_gen_layer_inputs, with_kwargs=True)


def install() -> None:
    dit.Cosmos3OmniTransformer._compute_rope_position_ids = compute_rope_position_ids
    dit.Cosmos3OmniTransformer.__call__ = _transformer_call
    USPAttention.forward_with_replicated_kv_prefix = prefix_attention
    stage.Cosmos3DenoisingStage._predict_noise_cfg_batched = predict_noise_cfg_batched
    stage.Cosmos3DenoisingStage._maybe_enable_torch_compile = maybe_enable_torch_compile
    dit.Cosmos3LanguageModel.forward = und_forward
