# SPDX-License-Identifier: Apache-2.0
"""GLM-Image's autoregressive prior-token generation on TPU workers.

Runtime patch, tracked debt:
- Why: GLM-Image draws its prior tokens (about 1.3k at 1024x1024) with
  Transformers' ``generate``. Its masks and stop checks reduce int64 tensors
  on the device, which torch_tpu does not lower ("Unsupported conversion from
  vmreg/vreg to U64"), and the dynamic KV cache gives every step new shapes.
- How: :func:`install` runs ``GlmImageForConditionalGeneration.generate`` on
  TPU, for text-to-image requests, with a static KV cache: the only compiled
  shapes are a 256-token prompt chunk and one token.
  - Positions are the stock ones (``get_rope_index`` and its decode grid
    positions), computed on the host; the RoPE tables are built on the host.
  - When the logits processors are temperature, top-k and top-p (the
    checkpoint's generation config), each step draws on the device: the top-k
    logits, the top-p cut and an exponential-race draw over them, with the
    noise drawn on the host from torch's seeded generator. Only the token id
    returns to the host. Any other processor list runs Transformers'
    processors on the host over the full row.
- Upstream PR or issue: none.
- Remove when: torch_tpu lowers int64 reductions and the prior tokens come
  from a TPU decode engine.
"""

from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
from transformers import GlmImageForConditionalGeneration
from transformers.generation import logits_process
from transformers.generation.logits_process import LogitsProcessorList
from transformers.models.glm_image import modeling_glm_image as hf

from sglang.multimodal_gen.runtime.utils.torch_compile import (
    build_torch_compile_kwargs,
)

CHUNK = 256
_CAPACITY_ALIGNMENT = 1024
# Devices that take these paths.
_DEVICE_TYPES = ("tpu",)

_STOCK_GENERATE = GlmImageForConditionalGeneration.generate


def device_sampling(processors):
    """``(temperature, top_k, top_p)`` if the processors are exactly those three."""
    kinds = [type(p) for p in processors]
    expected = [
        logits_process.TemperatureLogitsWarper,
        logits_process.TopKLogitsWarper,
        logits_process.TopPLogitsWarper,
    ]
    if kinds != expected:
        return None
    temperature, top_k, top_p = processors
    if top_p.min_tokens_to_keep != 1 or top_k.filter_value != top_p.filter_value:
        return None
    return float(temperature.temperature), int(top_k.top_k), float(top_p.top_p)


def candidate_probs(logits, temperature, top_k, top_p):
    """The top-k candidates' probabilities after the top-p cut, ``[1, k]``.

    The candidates come in descending order; a candidate stays if the
    probability ranked above it sums to less than ``top_p`` (Transformers'
    ascending-order rule), so the first always stays.
    """
    values, ids = torch.topk(logits.float() / temperature, top_k, dim=-1)
    probs = values.softmax(-1)
    keep = (probs.cumsum(-1) - probs) < top_p
    probs = torch.where(keep, probs, 0.0)
    return probs / probs.sum(-1, keepdim=True), ids


def _attend(q, keys, values, visible, scale):
    """``[1, H, n, D]`` queries over cache rows ``[C, KVH, D]``: ``[1, n, H * D]``."""
    _, heads, n, dim = q.shape
    q = q[0].unflatten(0, (keys.shape[1], -1))
    scores = (q.float() @ keys.permute(1, 2, 0)[:, None].float()) * scale
    probs = scores.masked_fill(~visible, float("-inf")).softmax(-1)
    out = probs.to(values.dtype) @ values.transpose(0, 1)[:, None]
    return out.flatten(0, 1).transpose(0, 1).reshape(1, n, heads * dim)


class _Decoder:
    """Static-KV-cache decoding of one GLM-Image text-to-image prompt."""

    def __init__(self, lm):
        self.lm = lm
        self.text = lm.model.language_model
        self.capacity = 0
        kwargs = build_torch_compile_kwargs(mode=None)
        kwargs.update(fullgraph=True, dynamic=False)
        self.forward = torch.compile(self._forward, **kwargs)
        self.draw = torch.compile(self._draw, **kwargs)
        self.last_row = torch.zeros(1, dtype=torch.int32).to(self._device)

    @property
    def _device(self):
        return self.lm.lm_head.weight.device

    def reserve(self, capacity: int) -> None:
        if capacity <= self.capacity:
            return
        attn = self.text.layers[0].self_attn
        heads = self.text.config.num_key_value_heads
        weight = self.lm.lm_head.weight
        shape = (capacity, heads, attn.head_dim)
        self.keys = [weight.new_zeros(shape) for _ in self.text.layers]
        self.values = [weight.new_zeros(shape) for _ in self.text.layers]
        self.capacity = capacity

    def set_positions(self, positions):
        """The stock RoPE tables of cache rows ``0..``, from ``[3, rows]`` positions."""
        weight = self.lm.lm_head.weight
        rows = F.pad(positions, (0, self.capacity - positions.shape[1]))
        rotary = copy.deepcopy(self.text.rotary_emb).to("cpu")
        cos, sin = rotary(torch.empty(0, dtype=weight.dtype), rows[:, None])
        self.cos, self.sin = cos[0].to(weight.device), sin[0].to(weight.device)

    def _forward(self, tokens_slots, last):
        """Writes tokens ``tokens_slots[0]`` at cache rows ``tokens_slots[1]``.

        Returns the float32 logits of row ``last``. A query sees the cache rows
        up to its own, so a chunk's padded tail never reaches its real rows.
        """
        tokens, slots = tokens_slots[0], tokens_slots[1]
        h = self.text.embed_tokens(tokens)[None]
        cos, sin = (t.index_select(0, slots)[None] for t in (self.cos, self.sin))
        visible = (
            torch.arange(self.capacity, dtype=torch.int32, device=slots.device)
            <= slots[:, None]
        )
        index = slots.long()
        for layer, keys, values in zip(self.text.layers, self.keys, self.values):
            attn = layer.self_attn
            x = layer.input_layernorm(h)
            q, k, v = (
                proj(x).unflatten(-1, (-1, attn.head_dim)).transpose(1, 2)
                for proj in (attn.q_proj, attn.k_proj, attn.v_proj)
            )
            q, k = hf.apply_rotary_pos_emb(q, k, cos, sin)
            keys.index_copy_(0, index, k[0].transpose(0, 1))
            values.index_copy_(0, index, v[0].transpose(0, 1))
            out = attn.o_proj(_attend(q, keys, values, visible, attn.scaling))
            h = h + layer.post_self_attn_layernorm(out)
            out = layer.mlp(layer.post_attention_layernorm(h))
            h = h + layer.post_mlp_layernorm(out)
        h = self.text.norm(h[0].index_select(0, last))
        return self.lm.lm_head(h).float()

    def _draw(self, logits, noise, sampling):
        """The token drawn from ``candidate_probs`` with exponential noise, int32."""
        probs, ids = candidate_probs(logits, *sampling)
        pick = torch.argmax(probs / noise, dim=-1, keepdim=True)
        return ids.gather(-1, pick).to(torch.int32).view(1)

    def generate(
        self, input_ids, attention_mask, image_grid_thw, max_new_tokens, kwargs
    ):
        lm = self.lm
        ids = input_ids.cpu()
        if ids.shape[0] != 1 or (
            attention_mask is not None and not bool(attention_mask.cpu().all())
        ):
            raise NotImplementedError("TPU generation takes one unpadded prompt")
        config, model_kwargs = lm._prepare_generation_config(None, **kwargs)
        n = ids.shape[1]
        lm._prepare_special_tokens(config, True, device="cpu")
        processors = lm._get_logits_processor(
            generation_config=config,
            input_ids_seq_length=n,
            encoder_input_ids=ids,
            prefix_allowed_tokens_fn=None,
            logits_processor=LogitsProcessorList(),
            device="cpu",
            model_kwargs=model_kwargs,
        )
        sampling = device_sampling(processors) if config.do_sample else None
        eos = config.eos_token_id
        eos = set() if eos is None else {eos} if isinstance(eos, int) else set(eos)

        # The stock positions: prompt rows, then each decoded grid token's.
        grids = image_grid_thw.cpu()
        positions, _ = lm.model.get_rope_index(
            ids, grids, torch.tensor([grids.shape[0]])
        )
        decode = lm.model._cached_decode_position_ids
        lm.model._cached_decode_position_ids = None
        if decode is None:
            raise NotImplementedError("TPU generation decodes image grids only")
        # Row r of the cache holds prompt token r, then decoded token r - n.
        positions = torch.cat((positions[:, 0], decode[0]), dim=-1)
        padded = -(-n // CHUNK) * CHUNK
        rows = max(padded, n + max_new_tokens)
        self.reserve(-(-rows // _CAPACITY_ALIGNMENT) * _CAPACITY_ALIGNMENT)
        self.set_positions(positions[:, : self.capacity])

        device = self._device
        tokens = F.pad(ids[0], (0, padded - n)).to(torch.int32)
        prompt = torch.stack((tokens, torch.arange(padded, dtype=torch.int32)))
        for start in range(0, padded, CHUNK):
            chunk = prompt[:, start : start + CHUNK].contiguous()
            last = torch.tensor([min(n - 1 - start, CHUNK - 1)], dtype=torch.int32)
            logits = self.forward(chunk.to(device), last.to(device))
        out = []
        for step in range(max_new_tokens):
            if sampling is not None:
                noise = torch.empty(1, sampling[1]).exponential_()
                token = int(self.draw(logits, noise.to(device), sampling))
            else:
                history = torch.cat((ids, torch.tensor([out], dtype=ids.dtype)), -1)
                scores = processors(history, logits.cpu())
                if config.do_sample:
                    token = int(torch.multinomial(scores.softmax(-1)[0], 1))
                else:
                    token = int(scores.argmax(-1))
            out.append(token)
            if token in eos:
                break
            step_row = torch.tensor([[token], [n + step]], dtype=torch.int32)
            logits = self.forward(step_row.to(device), self.last_row)
        generated = torch.tensor([out], dtype=input_ids.dtype)
        return torch.cat((ids, generated), -1).to(input_ids.device)


def _generate(self, input_ids=None, attention_mask=None, **kwargs):
    max_new_tokens = kwargs.pop("max_new_tokens", None)
    image_grid_thw = kwargs.pop("image_grid_thw", None)
    if (
        self.lm_head.weight.device.type not in _DEVICE_TYPES
        or input_ids is None
        or image_grid_thw is None
        or max_new_tokens is None
        or kwargs.get("pixel_values") is not None
    ):
        if max_new_tokens is not None:
            kwargs["max_new_tokens"] = max_new_tokens
        if image_grid_thw is not None:
            kwargs["image_grid_thw"] = image_grid_thw
        return _STOCK_GENERATE(
            self, input_ids=input_ids, attention_mask=attention_mask, **kwargs
        )
    for name in ("pixel_values", "images_per_sample", "mm_token_type_ids"):
        kwargs.pop(name, None)
    decoder = self.__dict__.get("_tpu_decoder")
    if decoder is None:
        decoder = self.__dict__["_tpu_decoder"] = _Decoder(self)
    with torch.no_grad():
        return decoder.generate(
            input_ids, attention_mask, image_grid_thw, max_new_tokens, kwargs
        )


def install() -> None:
    GlmImageForConditionalGeneration.generate = _generate
