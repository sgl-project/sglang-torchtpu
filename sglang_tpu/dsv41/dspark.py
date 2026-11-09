# SPDX-License-Identifier: Apache-2.0
"""DeepSeek V4.1 Flash DSpark drafter on the stock TPU ops (no megakernel).

The checkpoint bundles the drafter as ``mtp.{0,1,2}.*``. This is a port of
core ``DeepseekV4ForCausalLMDSpark`` (models/deepseek_v4_dspark.py) for the
V4.1 shifted-mHC configuration:

* three stages, each a ratio-0 (SWA-only) MQA attention plus a fused-EP MoE
  with ``dspark_n_routed_experts`` experts, top ``dspark_num_experts_per_tok``;
  shifted mHC chains the stages exactly like consecutive target layers;
* block drafting: one parallel forward over ``[r, noise x (gamma - 1)]`` at
  positions ``n .. n + gamma - 1``; every row predicts one draft
  (sample_from_anchor). Inside the block attention is bidirectional; each row
  also sees the last ``sliding_window`` committed positions ``n - W .. n - 1``;
* the context KV of committed positions is not computed by drafting: the
  target captures the hc-mean of the attention input of layers
  ``dspark_target_layer_ids`` and ``main_norm(main_proj(.))`` feeds every
  stage's ``wkv`` + ``kv_norm`` + RoPE (``inject``);
* a rank-``dspark_markov_rank`` Markov head adds ``w2(w1(prev_token))`` to the
  shared target lm_head logits row by row.

Draft context storage: the drafter lives inside the target model and owns a
small per-request ring of context KV per stage, ``[slots * RING + 1, D]`` bf16
(the last row is a write sink for masked rows). Only the last ``RING`` committed
positions of a request are kept; ``RING >= sliding_window + gamma``. A per-slot
floor marks the first position whose context KV was injected for the request
that holds the slot; earlier positions (a radix-cache prefix, or a previous
owner of the slot) are masked, exactly like positions below zero.

Device-specific leaves (RoPE, the grouped wo_a projection, the mHC boundary
ops and the fused EP MoE) are the target's own TPU ops on TPU tensors; CPU
tensors take the plain torch reference so the drafter can be unit tested.
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Callable, Iterable
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.deepseek_v4 import MqaAttentionBase, make_hc_mixing_params
from sglang.srt.models.dspark import project_through_lm_head
from sglang.srt.utils import add_prefix
from sglang_tpu.dsv41.mhc import ShiftedCarry, ShiftedMhc, hc_combine

logger = logging.getLogger(__name__)

# Ring entries per request slot and stage. A power of two above window + gamma.
RING = 256


def dspark_config_of(config) -> dict | None:
    """The bundled DSpark drafter geometry, or None when the checkpoint has none."""
    if getattr(config, "dspark_block_size", None) is None:
        return None
    targets = list(getattr(config, "dspark_target_layer_ids", None) or ())
    stages = int(getattr(config, "num_nextn_predict_layers", 0) or 0)
    if not targets or stages != len(targets):
        raise ValueError(
            "V4.1 DSpark needs one stage per target layer: "
            f"num_nextn_predict_layers={stages}, dspark_target_layer_ids={targets}"
        )
    return {
        "gamma": int(config.dspark_block_size),
        "noise_token_id": int(config.dspark_noise_token_id),
        "target_layer_ids": targets,
        "markov_rank": int(config.dspark_markov_rank),
        "n_routed_experts": int(
            getattr(config, "dspark_n_routed_experts", 0) or config.n_routed_experts
        ),
        "num_experts_per_tok": int(
            getattr(config, "dspark_num_experts_per_tok", 0)
            or config.num_experts_per_tok
        ),
    }


def dspark_enabled(config) -> bool:
    """Serve the bundled drafter: --speculative-algorithm DSPARK on a V4.1 checkpoint."""
    from sglang.srt.runtime_context import get_spec

    algorithm = get_spec().speculative_algorithm
    name = getattr(algorithm, "name", algorithm)
    if name is None or str(name).upper() != "DSPARK":
        return False
    return dspark_config_of(config) is not None


class DeviceOps(NamedTuple):
    """Device leaves the drafter calls inside its compiled graphs.

    rope(x, positions, freqs, inverse) and project(o, layer). Their custom ops
    must exist before Dynamo traces a draft/inject program: registering one
    (jax.jit export) while tracing is an unsupported graph break, fatal under
    fullgraph compile. prepare_tpu_ops() resolves them at
    model load; without device ops, CPU tensors take the torch reference.
    """

    rope: Callable
    project: Callable


_DEVICE_OPS: DeviceOps | None = None


def set_device_ops(ops: DeviceOps | None) -> None:
    global _DEVICE_OPS
    _DEVICE_OPS = ops


def prepare_tpu_ops() -> None:
    """Register the target's RoPE and wo_a projection ops now, outside any trace."""
    from sglang_tpu.dsv4.dsv4_attention_backend import (
        _register_rope_op,
        apply_tpu_dsv4_rope,
    )
    from sglang_tpu.dsv41.custom_ops import register_ops

    _register_rope_op()
    project_op = register_ops().project

    def project(o, layer):
        return project_op(o, layer.wo_a.weight, layer.wo_a.weight_scale)

    def rope(x, positions, freqs, inverse):
        return apply_tpu_dsv4_rope(x, positions, freqs, inverse=inverse)

    set_device_ops(DeviceOps(rope=rope, project=project))


def _on_tpu(x: torch.Tensor) -> bool:
    return x.device.type == "tpu"


def _device_ops(x: torch.Tensor) -> DeviceOps | None:
    ops = _DEVICE_OPS
    if ops is None and _on_tpu(x):
        raise RuntimeError("V4.1 DSpark: call prepare_tpu_ops() before tracing")
    return ops


def rotate(x, positions, freqs, *, inverse=False):
    """DSV4 RoPE on the last rope dims (interleaved pairs), as the target applies it."""
    ops = _device_ops(x)
    if ops is not None:
        return ops.rope(x, positions, freqs, inverse)
    if freqs.is_complex():
        table = freqs[positions.long()]
    else:
        # The TPU cos/sin cache [pos, cos | sin], see get_or_build_cos_sin_cache.
        half = freqs.shape[-1] // 2
        rows = freqs[positions.long()].float()
        table = torch.complex(rows[..., :half], rows[..., half:])
    if inverse:
        table = table.conj()
    rope = 2 * table.shape[-1]
    head = x[..., :-rope]
    tail = torch.view_as_complex(x[..., -rope:].float().unflatten(-1, (-1, 2)))
    shape = (table.shape[0],) + (1,) * (tail.ndim - 2) + (table.shape[-1],)
    tail = torch.view_as_real(tail * table.view(shape)).flatten(-2)
    return torch.cat([head, tail.to(x.dtype)], dim=-1)


def project_groups(layer: MqaAttentionBase, o: torch.Tensor) -> torch.Tensor:
    """Grouped wo_a on the inverse-rotated attention output [T, H, D] -> [T, G * R]."""
    ops = _device_ops(o)
    if ops is not None:
        return ops.project(o, layer)
    groups = layer.n_local_groups
    grouped = o.reshape(o.shape[0], groups, -1)
    weight = layer.wo_a.weight.view(groups, layer.o_lora_rank, -1)
    out = torch.einsum("tgd,grd->tgr", grouped.float(), weight.float())
    return out.reshape(o.shape[0], -1).to(o.dtype)


def block_attention(q, block_kv, ctx_kv, ctx_valid, sink, scale):
    """Each row attends to its request's valid context and to the whole block.

    q [B, W, H, D], block_kv [B, W, D], ctx_kv [B, C, D], ctx_valid [B, C],
    sink [H] (denominator-only logits). Returns [B, W, H, D] in q's dtype.
    """
    keys = torch.cat([ctx_kv, block_kv], dim=1).float()
    visible = torch.cat([ctx_valid, ctx_valid.new_ones(block_kv.shape[:2])], dim=1)
    scores = torch.einsum("bwhd,bkd->bwhk", q.float(), keys) * scale
    scores = scores.masked_fill(~visible[:, None, None, :], float("-inf"))
    sink = sink.float().view(1, 1, -1, 1)
    peak = torch.maximum(scores.amax(-1, keepdim=True), sink)
    weights = torch.exp(scores - peak)
    denom = weights.sum(-1, keepdim=True) + torch.exp(sink - peak)
    out = torch.einsum("bwhk,bkd->bwhd", weights / denom, keys)
    return out.to(q.dtype)


class DSparkAttention(MqaAttentionBase):
    """Ratio-0 MQA over [context window ++ block]; never touches the target pools."""

    def __init__(self, config, layer_id, quant_config, prefix):
        super().__init__(
            config,
            layer_id,
            quant_config,
            prefix,
            compress_ratio=0,
            fuse_wqa_wkv=False,
            wo_a_fp8=quant_config is not None,
            wo_a_keeps_quant_config=quant_config is not None,
            wo_b_reduce_results=False,
            rope_original_seq_len=0,
        )

    def context_kv(self, x, positions):
        """Normed, rotated [T, D] KV of rows x; the same map as the target SWA write."""
        kv, _ = self.wkv(x)
        return rotate(self.kv_norm(kv), positions, self.freqs_cis)

    def forward(self, x, positions, ctx_kv, ctx_valid, width):
        rows = x.shape[0]
        q, _ = self.wq_a(x)
        q, _ = self.wq_b(self.q_norm(q))
        q = q.view(rows, self.n_local_heads, self.head_dim)
        if self.q_head_norm:
            q = q * torch.rsqrt(
                q.float().square().mean(-1, keepdim=True) + self.eps
            ).to(q.dtype)
        q = rotate(q, positions, self.freqs_cis)
        kv = self.context_kv(x, positions)
        batch = rows // width
        o = block_attention(
            q.view(batch, width, self.n_local_heads, self.head_dim),
            kv.view(batch, width, self.head_dim),
            ctx_kv,
            ctx_valid,
            self.attn_sink,
            self.softmax_scale,
        ).view(rows, self.n_local_heads, self.head_dim)
        o = rotate(o, positions, self.freqs_cis, inverse=True)
        return self.wo_b(project_groups(self, o))[0]


def _default_moe(config, layer_id, quant_config, prefix):
    from sglang_tpu.dsv41.moe import DSV41MoE

    return DSV41MoE(config, layer_id, quant_config, prefix)


class DSparkStage(nn.Module):
    """One draft stage: the DSV41Layer sublayer order with a block attention."""

    def __init__(self, config, stage_id, quant_config, prefix, moe_factory):
        super().__init__()
        self.stage_id = stage_id
        self.hidden_size = config.hidden_size
        self.self_attn = DSparkAttention(
            config, stage_id, quant_config, add_prefix("self_attn", prefix)
        )
        # Layer id >= num_hash_layers: score routing with the correction bias.
        self.mlp = moe_factory(
            config,
            config.num_hidden_layers + stage_id,
            quant_config,
            add_prefix("mlp", prefix),
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.hc_mult, self.hc_eps = config.hc_mult, config.hc_eps
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.rms_norm_eps = config.rms_norm_eps
        (
            self.hc_attn_fn,
            self.hc_ffn_fn,
            self.hc_attn_base,
            self.hc_ffn_base,
            self.hc_attn_scale,
            self.hc_ffn_scale,
        ) = make_hc_mixing_params(config.hc_mult, config.hidden_size)
        self.attn_mhc, self.ffn_mhc = ShiftedMhc(self, "attn"), ShiftedMhc(self, "ffn")

    def forward(
        self, residual, carry, positions, ctx_kv, ctx_valid, width, input_ids, valid
    ):
        if carry is None:
            post, comb, value, pre = self.attn_mhc.start(residual)
        else:
            residual, post, comb, value, pre = self.attn_mhc.advance(carry)
        output = self.self_attn(
            self.input_layernorm(value), positions, ctx_kv, ctx_valid, width
        )
        attention = ShiftedCarry(output, residual, post, comb, pre)
        residual, post, comb, value, pre = self.ffn_mhc.advance(attention)
        output = self.mlp(self.post_attention_layernorm(value), input_ids, valid)
        return ShiftedCarry(output, residual, post, comb, pre)


class DSV41DSparkDrafter(nn.Module):
    """The bundled V4.1 DSpark drafter; owned by the target model (``model.dspark``)."""

    def __init__(
        self,
        config,
        quant_config,
        prefix: str = "",
        *,
        moe_factory: Callable | None = None,
    ):
        super().__init__()
        geometry = dspark_config_of(config)
        if geometry is None:
            raise ValueError("The checkpoint does not bundle a DSpark drafter")
        if not getattr(config, "hc_pre_from_prev_sublayer", False):
            raise ValueError("The V4.1 DSpark port expects shifted mHC")
        self.gamma = geometry["gamma"]
        self.noise_token_id = geometry["noise_token_id"]
        self.target_layer_ids = tuple(geometry["target_layer_ids"])
        self.markov_rank = geometry["markov_rank"]
        if self.markov_rank <= 0:
            raise ValueError("V4.1 DSpark requires a Markov head")
        self.window = int(config.sliding_window)
        if RING < self.window + self.gamma + 1:
            raise ValueError(f"RING={RING} cannot hold window + gamma")
        self.hidden_size = config.hidden_size
        self.hc_mult = config.hc_mult
        self.vocab_size = config.vocab_size
        stage_config = copy.copy(config)
        stage_config.n_routed_experts = geometry["n_routed_experts"]
        stage_config.num_experts_per_tok = geometry["num_experts_per_tok"]
        moe_factory = moe_factory or _default_moe
        self.stages = nn.ModuleList(
            [
                DSparkStage(
                    stage_config,
                    stage,
                    quant_config,
                    add_prefix(f"stages.{stage}", prefix),
                    moe_factory,
                )
                for stage in range(len(self.target_layer_ids))
            ]
        )
        self.main_proj = ReplicatedLinear(
            config.hidden_size * len(self.target_layer_ids),
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("main_proj", prefix),
        )
        self.main_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # markov_head.embed / .head; core's default runs w2 in BF16 then adds in FP32.
        self.markov_w1 = nn.Parameter(
            torch.empty(config.vocab_size, self.markov_rank, dtype=torch.bfloat16),
            requires_grad=False,
        )
        self.markov_w2 = nn.Parameter(
            torch.empty(config.vocab_size, self.markov_rank, dtype=torch.bfloat16),
            requires_grad=False,
        )
        # Shared target modules (attach) and per-slot context state (allocate_state).
        self._embed: nn.Module | None = None
        self._lm_head: nn.Module | None = None
        self.ring: list[torch.Tensor] | None = None
        self.floor: torch.Tensor | None = None
        self.num_slots = 0

    # ---- wiring -------------------------------------------------------------

    def attach(self, *, embed_tokens: nn.Module, lm_head: nn.Module) -> None:
        # Plain attributes, not submodules: the target owns and loads them.
        object.__setattr__(self, "_embed", embed_tokens)
        object.__setattr__(self, "_lm_head", lm_head)

    def allocate_state(self, num_slots: int, device) -> None:
        """Context rings for every request slot (slot 0 is the padding row).

        On TPU this also registers the drafter's device ops, before any warm-up
        traces a draft or inject program.
        """
        if str(device).split(":")[0] == "tpu":
            prepare_tpu_ops()
        dim = self.stages[0].self_attn.head_dim
        rows = num_slots * RING + 1
        self.ring = [
            torch.zeros((rows, dim), dtype=torch.bfloat16, device=device)
            for _ in self.stages
        ]
        # A large floor masks every position of a slot no request has seeded.
        self.floor = torch.full((num_slots,), 1 << 30, dtype=torch.int32, device=device)
        self.num_slots = num_slots

    def state_bytes(self) -> int:
        if self.ring is None:
            return 0
        return sum(t.numel() * t.element_size() for t in self.ring) + self.floor.nbytes

    # ---- context injection ----------------------------------------------------

    def project_context(self, aux_hidden: torch.Tensor) -> torch.Tensor:
        """main_norm(main_proj(cat(hc-mean attention inputs))) -> [T, H]."""
        projected, _ = self.main_proj(aux_hidden)
        return self.main_norm(projected)

    def inject(
        self,
        aux_hidden: torch.Tensor,
        positions: torch.Tensor,
        slots: torch.Tensor,
        write: torch.Tensor,
    ) -> torch.Tensor:
        """Write committed rows' context KV into every stage's ring.

        aux_hidden [T, n_targets * H], positions [T], slots [T] request slot of
        each row, write [T] bool. Returns a value read after the last write
        (the KV free fence tail).
        """
        main_x = self.project_context(aux_hidden)
        sink = self.ring[0].shape[0] - 1
        index = torch.where(write, slots.long() * RING + positions.long() % RING, sink)
        for stage, ring in zip(self.stages, self.ring):
            kv = stage.self_attn.context_kv(main_x, positions)
            ring.index_copy_(0, index, kv.to(ring.dtype))
        return self.ring[-1][sink, :1]

    def set_floor(self, slots: torch.Tensor, floors: torch.Tensor) -> torch.Tensor:
        self.floor.index_copy_(0, slots.long(), floors.to(torch.int32))
        return self.floor[:1]

    # ---- draft forward ----------------------------------------------------------

    def block_positions(self, seq_lens: torch.Tensor) -> torch.Tensor:
        offsets = torch.arange(self.gamma, device=seq_lens.device, dtype=torch.int64)
        return (seq_lens.long()[:, None] + offsets).reshape(-1)

    def gather_context(self, seq_lens, slots, live):
        """[B, W, D] per stage and the [B, W] validity of positions n - W .. n - 1."""
        offsets = torch.arange(self.window, device=seq_lens.device, dtype=torch.int64)
        positions = seq_lens.long()[:, None] - self.window + offsets
        floor = self.floor[slots.long()].long()
        valid = (positions >= floor[:, None]) & (positions >= 0) & live[:, None]
        index = slots.long()[:, None] * RING + positions % RING
        return [ring[index] for ring in self.ring], valid

    def block_hidden(self, bonus_tokens, seq_lens, slots, live):
        """Final normed hidden [B * gamma, H] of the block [r, noise x (gamma - 1)]."""
        batch, width = bonus_tokens.shape[0], self.gamma
        noise = torch.full(
            (batch, width - 1),
            self.noise_token_id,
            dtype=torch.int64,
            device=bonus_tokens.device,
        )
        ids = torch.cat([bonus_tokens.long()[:, None], noise], dim=1).reshape(-1)
        positions = self.block_positions(seq_lens)
        valid = live[:, None].expand(batch, width).reshape(-1)
        contexts, ctx_valid = self.gather_context(seq_lens, slots, live)
        embedded = self._embed(ids)
        residual = embedded[:, None, :].expand(-1, self.hc_mult, -1).contiguous()
        carry = None
        for stage, ctx_kv in zip(self.stages, contexts):
            carry = stage(
                residual, carry, positions, ctx_kv, ctx_valid, width, ids, valid
            )
        residual = self.stages[-1].ffn_mhc.commit(carry)
        value = hc_combine(residual.flatten(1), carry.pre, self.hc_mult, residual.dtype)
        return self.norm(value)

    def base_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """Shared target lm_head logits [N, V] (FP32 for the Markov addition)."""
        logits = project_through_lm_head(hidden, self._lm_head)
        return logits[..., : self.vocab_size].float()

    def markov_bias(self, tokens: torch.Tensor) -> torch.Tensor:
        latent = self.markov_w1[tokens.long()]
        return F.linear(latent, self.markov_w2).float()

    def apply_step_logits(self, logits, *, token_ids, hidden_states=None):
        """core run_markov_block's head interface (sampled drafting)."""
        del hidden_states
        return logits + self.markov_bias(token_ids)

    def greedy_drafts(self, hidden, bonus_tokens):
        """Greedy Markov chain over the gamma rows: [B, gamma] drafts."""
        batch = bonus_tokens.shape[0]
        base = self.base_logits(hidden).view(batch, self.gamma, -1)
        previous = bonus_tokens.long()
        drafts = []
        for step in range(self.gamma):
            previous = torch.argmax(base[:, step] + self.markov_bias(previous), dim=-1)
            drafts.append(previous)
        return torch.stack(drafts, dim=1)

    def draft(self, bonus_tokens, seq_lens, slots, live):
        """One draft step: greedy drafts [B, gamma] and the block hidden [B * gamma, H]."""
        hidden = self.block_hidden(bonus_tokens, seq_lens, slots, live)
        return self.greedy_drafts(hidden, bonus_tokens), hidden

    # ---- weights ----------------------------------------------------------------

    def remap(self, name: str) -> str | None:
        """Checkpoint ``mtp.<s>.*`` -> drafter parameter name (None = not used)."""
        if not name.startswith("mtp."):
            return None
        _, stage, rest = name.split(".", 2)
        if rest.startswith("confidence_head.") or rest.endswith(".gate.bias_vl"):
            return None  # ragged verify is off (K = 5 fixed); vision routing bias.
        if rest == "markov_head.embed.weight":
            return "markov_w1"
        if rest == "markov_head.head.weight":
            return "markov_w2"
        if rest.startswith(("main_proj.", "main_norm.")) or rest == "norm.weight":
            name = rest
        else:
            for old, new in (
                ("attn.", "self_attn."),
                ("ffn.", "mlp."),
                ("attn_norm.", "input_layernorm."),
                ("ffn_norm.", "post_attention_layernorm."),
            ):
                if rest.startswith(old):
                    rest = new + rest[len(old) :]
                    break
            rest = rest.replace(".gate.bias", ".gate.e_score_correction_bias")
            rest = rest.replace(".w1.", ".gate_proj.")
            rest = rest.replace(".w2.", ".down_proj.")
            rest = rest.replace(".w3.", ".up_proj.")
            name = f"stages.{stage}.{rest}"
        if name.endswith(".scale"):
            name = name.removesuffix(".scale") + ".weight_scale_inv"
        return name

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params = dict(self.named_parameters())
        loaded = set()
        for checkpoint_name, tensor in weights:
            name = self.remap(checkpoint_name)
            if name is None:
                continue
            expert = _expert_target(name)
            if expert is not None:
                param_name, shard, expert_id = expert
                if param_name not in params:
                    raise KeyError(f"DSpark expert parameter {param_name} missing")
                param = params[param_name]
                param.weight_loader(
                    param, tensor, param_name, shard_id=shard, expert_id=expert_id
                )
                loaded.add(param_name)
                continue
            for proj, shard in ((".gate_proj.", 0), (".up_proj.", 1)):
                if proj in name:
                    stacked = name.replace(proj, ".gate_up_proj.")
                    if stacked in params:
                        param = params[stacked]
                        param.weight_loader(param, tensor, shard)
                        loaded.add(stacked)
                        break
            else:
                if name not in params:
                    raise KeyError(
                        f"Unexpected DSpark tensor {checkpoint_name} -> {name}"
                    )
                param = params[name]
                getattr(param, "weight_loader", default_weight_loader)(param, tensor)
                loaded.add(name)
        return loaded


def _expert_target(name: str):
    """(param, shard, expert) for ``stages.<s>.mlp.experts.<e>.<proj>.<w>``."""
    marker = ".mlp.experts."
    if marker not in name:
        return None
    head, tail = name.split(marker, 1)
    expert, proj, kind = tail.split(".", 2)
    shard = {"gate_proj": "w1", "down_proj": "w2", "up_proj": "w3"}[proj]
    family = "w2" if shard == "w2" else "w13"
    return f"{head}.mlp.experts.{family}_{kind}", shard, int(expert)
