# SPDX-License-Identifier: Apache-2.0
"""Qwen-Image 2.1 on TPU workers: one compiled program per image size.

Runtime patch, tracked debt:
- Why: torch_tpu compiles static shapes. Qwen-Image 2.1 runs its prefix (the
  prompt and any condition images) through every block once per request, and
  each later step attends to the prefix's cached K/V at its exact length. So
  every prompt length compiles every block again, twice (prefill and cached
  step), and the eager text encoder compiles once per prompt length too. Its
  complex RoPE reads interleaved pairs, which torch_tpu re-lays out (see
  rope.py), and torch_tpu splits each compiled block at its four Ulysses
  all-to-alls.
- How: :func:`install`
  (1) wraps the encoding stage's ``build_layout`` so the prefix pads to a
      multiple of 512 rows. The layout carries the real prefix length, the
      Ulysses rank and each condition image's rows as device tensors, and the
      RoPE as real cos/sin tables (built on the host);
  (2) replaces ``QwenImage21Attention.forward``. The prefix prefills as one
      causal attention over its padded rows, then each condition image's rows
      attend every key up to the image's end (the stock segment rules). Target
      queries attend [prefix keys | target keys] as two resumable kernel
      steps: this rank's heads of the prefix's real keys, then the target's.
      Padded keys are never read;
  (3) rotates pairs in the pair-swap form and pads the target to a multiple
      of the SP degree (its real length is the target step's key count);
  (4) right-pads the prompt tokens to a multiple of 256 (the encoder is causal,
      so real tokens never see the padding, and the stage keeps only real rows),
      and the prompt's hidden rows the DiT reads to the same multiple;
  (5) slices each block's prefix caches and copies its RoPE share before its
      compiled region (the stock block indexes the caches by an int the
      compiler guards on, one graph per layer, and the share is a view at a
      rank-dependent offset), so all layers and ranks run one program, and
      declares the blocks spmd-safe regions for ``--regional-compile``.
- Upstream PR or issue: none.
- Remove when: the DiT takes a platform-provided prefix bucket and key counts,
  and RoPE goes through a platform rotation.
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
from sglang.multimodal_gen.runtime.layers.usp import (
    _usp_input_all_to_all,
    _usp_output_all_to_all,
)
from sglang.multimodal_gen.runtime.models.dits import qwen_image21 as dit
from sglang_tpu.multimodal_gen import tpu_splash_attn
from sglang_tpu.multimodal_gen.rope import _pair_swap_matrix

# Text prompts stay within one bucket; the kernel's prefix step needs >= 512 keys.
PREFIX_ALIGNMENT = 512
PROMPT_TOKEN_ALIGNMENT = 256


def _real_rope(rope: torch.Tensor) -> torch.Tensor:
    """Complex ``[N, D/2]`` rotations as float32 ``[N, 2, D]`` cos/sin, pair-repeated."""
    return torch.stack(
        (rope.real.repeat_interleave(2, -1), rope.imag.repeat_interleave(2, -1)), 1
    ).float()


def _keys(first: int, real: int, device) -> torch.Tensor:
    return torch.tensor([first, real], dtype=torch.int32).to(device)


def pad_layout(layout: dict, alignment: int, device) -> dict:
    """``build_layout``'s host layout with the prefix padded to ``alignment`` rows.

    The compiled DiT reads only tensors and values fixed by the bucket and the
    image sizes, so a new prompt length reuses its programs.
    """
    prefix_len = layout["prefix_rope"].shape[0]
    extra = -prefix_len % alignment
    # Padded rows take text row 0 and the identity rotation; no real query reads them.
    prefix_rope = torch.cat(
        (
            layout["prefix_rope"],
            layout["prefix_rope"].new_ones(extra, *layout["prefix_rope"].shape[1:]),
        )
    )
    return dict(
        encoder_seq_len=layout["encoder_seq_len"],
        text_indices=F.pad(layout["text_indices"], (0, extra)).to(device),
        image_indices=layout["image_indices"].to(device),
        prefix_rope=_real_rope(prefix_rope).to(device),
        target_rope=_real_rope(layout["target_rope"]).to(device),
        # Each condition image's rows and the keys they attend, [0, image end).
        image_runs=[
            (torch.arange(start, end).to(device), _keys(0, end, device))
            for start, end, is_image in layout["segments"]
            if is_image
        ],
        # [Ulysses rank, real keys]: the rank picks this rank's heads of the
        # replicated prefix K/V on the device.
        prefix_keys=_keys(get_ulysses_parallel_rank(), prefix_len, device),
        target_keys=_keys(0, layout["target_rope"].shape[0], device),
    )


def apply_rope_pair_swap(x: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
    """The complex rotation of ``qwen_image21.apply_rope`` on a ``[N, 2, D]`` table.

    ``x * cos + (x @ R) * sin`` in float32 forms (ac - bd, ad + bc) per pair,
    the complex product; ``x @ R`` is a signed pair swap, exact in bfloat16.
    """
    cos, sin = rope[None, :, 0, None], rope[None, :, 1, None]
    swapped = torch.matmul(x, _pair_swap_matrix(x.shape[-1], x.dtype, x.device))
    return (x.float() * cos + swapped.float() * sin).to(x.dtype)


def _prefill_attention(qp, kp, vp, layout, scale):
    """The prefix's self-attention by the stock segment rules, ``[1, H, P, D]``.

    Text rows are causal; each condition image's rows see every key up to the
    image's end. Padded rows see only real or padded keys and no real row
    reads them.
    """
    out = tpu_splash_attn.tpu_attention(qp, kp, vp, scale, is_causal=True)
    for rows, keys in layout["image_runs"]:
        image_out, _ = tpu_splash_attn.tpu_ring_attention_step(
            qp.index_select(2, rows), kp, vp, keys, None, scale
        )
        out = out.index_copy(2, rows, image_out)
    return out


def _uses_kernel(q, *keys) -> bool:
    # Both steps take the kernel or neither does: their carries differ.
    return q.device.type == "tpu" and all(
        tpu_splash_attn._ring_kernel_fallback_reason(q, key, key) is None
        for key in keys
    )


def _target_attention(usp, q, k, v, kp, vp, layout):
    """Target queries over [real prefix keys | target keys], ``[B, S, H, D]``.

    ``kp``/``vp`` are the replicated prefix K/V of every head, ``[1, H, P, D]``.
    """
    sharded = not usp.skip_sequence_parallel and get_sp_world_size() > 1
    if sharded:
        if get_ring_parallel_world_size() > 1 or usp.sp_attention_mode == "kv_gather":
            raise NotImplementedError(
                "Qwen-Image 2.1 on TPU shards the sequence with Ulysses only"
            )
        q, k, v = (_usp_input_all_to_all(t, head_dim=2) for t in (q, k, v))
    q, k, v = (t.transpose(1, 2) for t in (q, k, v))
    heads, scale = q.shape[1], usp.attn_impl.softmax_scale
    # As a stack of per-rank head shards, this rank's prefix heads start at row
    # rank * heads. The rank is device data, so every rank runs one program.
    kp, vp = (t.view(-1, heads, *t.shape[2:]) for t in (kp, vp))
    prefix_keys = layout["prefix_keys"] * layout["prefix_keys"].new_tensor([heads, 1])
    if _uses_kernel(q, kp, k):
        _, carry = tpu_splash_attn.tpu_ring_attention_step(
            q, kp, vp, prefix_keys, None, scale
        )
        out, _ = tpu_splash_attn.tpu_ring_attention_step(
            q, k, v, layout["target_keys"], carry, scale
        )
    else:
        rows = prefix_keys[0] + torch.arange(heads, device=q.device)
        kp, vp = (t.flatten(0, 1).index_select(0, rows)[None] for t in (kp, vp))
        prefix = torch.arange(kp.shape[2], device=q.device) < prefix_keys[1]
        target = torch.arange(k.shape[2], device=q.device) < layout["target_keys"][1]
        out = F.scaled_dot_product_attention(
            q.float(),
            torch.cat((kp, k), 2).float(),
            torch.cat((vp, v), 2).float(),
            attn_mask=torch.cat((prefix, target))[None, None, None],
            scale=scale,
        ).to(q.dtype)
    out = out.transpose(1, 2)
    return _usp_output_all_to_all(out, head_dim=2) if sharded else out


def _attend(attn, q, k, v, rope, prefix, layout, cache):
    if cache:
        kp, vp = cache["key"], cache["value"]
        prefix_output = None
    else:
        # Heads-major, the layout every later step reads from the cache.
        qp, kp, vp = (
            t.transpose(1, 2).contiguous()
            for t in attn.qkv(prefix, layout["prefix_rope"])
        )
        out = _prefill_attention(qp, kp, vp, layout, attn.local_attn.softmax_scale)
        prefix_output = attn.to_out[0](out.transpose(1, 2).flatten(2))[0]
        if cache is not None:
            cache.update(key=kp, value=vp)
    q = dit.apply_qk_norm_rope(q, attn.norm_q, rope)
    k = dit.apply_qk_norm_rope(k, attn.norm_k, rope)
    return _target_attention(attn.target_attn, q, k, v, kp, vp, layout), prefix_output


def dit_forward(
    self,
    hidden_states,
    encoder_hidden_states,
    timestep,
    layouts,
    condition_latents=None,
    prefix_caches=None,
    **kwargs,
):
    """The DiT forward with the target padded to a multiple of the SP degree.

    Padded target rows rotate by zero and are not among the target's real keys
    (``target_keys``); their outputs are dropped.
    """
    target_len = hidden_states.shape[1]
    extra = -target_len % get_sp_world_size()
    if extra:
        hidden_states = F.pad(hidden_states, (0, 0, 0, extra))
        layouts = [
            dict(
                layout, target_rope=F.pad(layout["target_rope"], (0, 0, 0, 0, 0, extra))
            )
            for layout in layouts
        ]
    output = _STOCK_DIT_FORWARD(
        self,
        hidden_states,
        encoder_hidden_states,
        timestep,
        layouts,
        condition_latents=condition_latents,
        prefix_caches=prefix_caches,
        **kwargs,
    )
    return output[:, :target_len] if extra else output


_STOCK_DIT_FORWARD = dit.QwenImage21Transformer2DModel.forward


def _dit_call(self, *args, **kwargs):
    """Pads the prompt's hidden rows to a multiple of 256 before the DiT runs.

    The prefill projects ``encoder_hidden_states[:, :encoder_seq_len]``, the
    prompt's exact row count, so a compiled DiT (the TPU default compiles the
    whole module) would specialize on every prompt length. The padding happens
    here, outside the compiled call; the layout's ``text_indices`` read only
    real rows.
    """
    states = kwargs.get("encoder_hidden_states")
    layouts = kwargs.get("layouts")
    if states is not None and layouts is not None:
        packed = states[0] if isinstance(states, list) else states
        extra = -packed.shape[1] % PROMPT_TOKEN_ALIGNMENT
        if extra:
            packed = F.pad(packed, (0, 0, 0, extra))
        kwargs["encoder_hidden_states"] = (
            [packed] if isinstance(states, list) else packed
        )
        kwargs["layouts"] = [
            dict(layout, encoder_seq_len=packed.shape[1]) for layout in layouts
        ]
    return torch.nn.Module.__call__(self, *args, **kwargs)


def attention_forward(self, x, ropes, prefixes, layouts, caches):
    """``QwenImage21Attention.forward`` over padded layouts."""
    q, k, v = self.project_qkv(x)
    outputs, prefix_outputs = [], []
    for sample, layout in enumerate(layouts):
        out, prefix_out = _attend(
            self,
            q[sample : sample + 1],
            k[sample : sample + 1],
            v[sample : sample + 1],
            ropes[sample],
            prefixes[sample],
            layout,
            caches[sample],
        )
        outputs.append(out)
        prefix_outputs.append(prefix_out)
    return self.to_out[0](dit.cat_outputs(outputs).flatten(2))[0], prefix_outputs


def block_forward(
    self,
    hidden_states,
    modulation,
    prefix_states,
    prefix_modulation,
    layouts,
    ropes,
    caches,
):
    """``QwenImage21TransformerBlock.forward`` on caches already sliced to this layer.

    The stock forward slices them by ``self._layer_id``, an int the compiler
    guards on, so every layer compiled its own graph (64 per image size, the
    recompile limit). :func:`_block_call` slices outside the compiled region.
    """
    scale1, gate1, scale2, gate2 = modulation
    prefixes = [
        dit.apply_modulation(
            state["hidden_states"], self.img_norm1, prefix_modulation[0]
        )
        if not cache
        else None
        for state, cache in zip(prefix_states, caches, strict=True)
    ]
    attention, prefix_attentions = self.attn(
        dit.apply_modulation(hidden_states, self.img_norm1, scale1),
        ropes,
        prefixes,
        layouts,
        caches,
    )
    hidden_states = dit.residual_gate_add(hidden_states, attention, gate1)
    hidden_states = dit.residual_gate_add(
        hidden_states,
        self.img_mlp(dit.apply_modulation(hidden_states, self.img_norm2, scale2)),
        gate2,
    )
    for state, attention in zip(prefix_states, prefix_attentions, strict=True):
        if attention is not None:
            _, pg1, ps2, pg2 = prefix_modulation
            prefix = dit.residual_gate_add(state["hidden_states"], attention, pg1)
            state["hidden_states"] = dit.residual_gate_add(
                prefix,
                self.img_mlp(dit.apply_modulation(prefix, self.img_norm2, ps2)),
                pg2,
            )
    return hidden_states


def _block_call(
    self,
    hidden_states,
    modulation,
    prefix_states,
    prefix_modulation,
    layouts,
    ropes,
    caches,
):
    caches = [cache[self._layer_id] for cache in caches]
    # A copy at offset 0: the compiled program bakes an input's storage offset.
    ropes = [rope.clone() for rope in ropes]
    return torch.nn.Module.__call__(
        self,
        hidden_states,
        modulation,
        prefix_states,
        prefix_modulation,
        layouts,
        ropes,
        caches,
    )


class _PromptPadding:
    """A processor whose token rows are right-padded to a multiple of ``multiple``."""

    def __init__(self, processor, multiple: int):
        self._processor = processor
        self._multiple = multiple

    def __getattr__(self, name):
        # copy and pickle look up dunders before __init__ sets _processor.
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._processor, name)

    def __call__(self, *args, **kwargs):
        inputs = self._processor(*args, **kwargs)
        extra = -inputs["input_ids"].shape[1] % self._multiple
        if extra:
            tokenizer = self._processor.tokenizer
            pad_id = tokenizer.pad_token_id
            if pad_id is None:
                pad_id = tokenizer.eos_token_id
            inputs["input_ids"] = F.pad(inputs["input_ids"], (0, extra), value=pad_id)
            inputs["attention_mask"] = F.pad(inputs["attention_mask"], (0, extra))
        return inputs


def _padded_build_layout(build_layout):
    @functools.wraps(build_layout)
    def build(image_slots, image_shapes, axes_dims, device):
        layout = build_layout(image_slots, image_shapes, axes_dims, "cpu")
        return pad_layout(layout, PREFIX_ALIGNMENT, device)

    build.__wrapped_for_tpu__ = True
    return build


def _padded_prompt_init(init):
    @functools.wraps(init)
    def wrapped(self, text_encoder, processor, vae, scheduler):
        init(self, text_encoder, processor, vae, scheduler)
        self.processor = _PromptPadding(self.processor, PROMPT_TOKEN_ALIGNMENT)

    wrapped.__wrapped_for_tpu__ = True
    return wrapped


def install() -> None:
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages import (
        qwen_image21 as stage,
    )

    if not getattr(stage.build_layout, "__wrapped_for_tpu__", False):
        stage.build_layout = _padded_build_layout(stage.build_layout)
    init = stage.QwenImage21EncodingStage.__init__
    if not getattr(init, "__wrapped_for_tpu__", False):
        stage.QwenImage21EncodingStage.__init__ = _padded_prompt_init(init)
    dit.QwenImage21Attention.forward = attention_forward
    dit.QwenImage21Transformer2DModel.forward = dit_forward
    dit.QwenImage21Transformer2DModel.__call__ = _dit_call
    dit.apply_rope = apply_rope_pair_swap
    block_cls = dit.QwenImage21TransformerBlock
    if not getattr(block_cls.forward, "__wrapped_for_tpu__", False):
        block_forward.__wrapped_for_tpu__ = True
        block_cls.forward = block_forward
        block_cls.__call__ = _block_call
    # Every rank runs each block with the same shapes, offsets and arguments.
    dit.QwenImage21Transformer2DModel._compile_regions_spmd_safe = True
