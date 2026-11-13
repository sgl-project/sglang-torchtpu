# SPDX-License-Identifier: Apache-2.0
"""Inkling2 DSpark MTP draft on TPU (reference: python/sglang/tml/mtp.py of sglang-tml-internal).

A 5-layer block draft over the target's embedding and lm_head:
  - context KV: per draft layer, a softmax mixture of the captured target features
    (``fuse_weight`` row ``i``), RMS-normed and projected by that layer's own K/V
    projection (``kv_proj_ctx``, ``k_norm_ctx``);
  - block: ``[root, mask_embed x 7]`` attended bidirectionally;
  - drafts: a GRU-style head over the target lm_head, seeded per request with ``f_rnn``,
    the mixture of the target features of the last committed token.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn

from sglang.srt.compilation.compiled_op import compiled_op
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import QKVParallelLinear
from sglang.srt.layers.logits_processor import should_apply_lm_head_quant_method
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.dflash import (
    DFlashAttention,
    DFlashDecoderLayer,
    DFlashDraftModel,
)
from sglang.srt.models.dspark import (
    DSparkDraftModel,
    gather_and_crop_vocab,
    project_through_lm_head,
)


def _mix_rmsnorm(targets, alpha, weight, eps):
    """RMSNorm(sum_t alpha[t] * targets[:, t]) for targets [rows, taps, hidden]."""
    mixed = (targets.float() * alpha.float()[None, :, None]).sum(1)
    out = (
        mixed
        * torch.rsqrt(mixed.square().mean(-1, keepdim=True) + eps)
        * weight.float()
    )
    return out.to(targets.dtype)


@compiled_op()
def seed_f_rnn(
    f_rnn_buf, alpha, norm_weight, target_hidden, rows, req_pool_indices, live, *, eps
):
    """f_rnn_buf with each live request's row replaced by its target features at ``rows``.

    target_hidden [tokens, K * H]; rows, req_pool_indices, live [n]. Dense one-hot write:
    padded entries share an index, and a compiled duplicate-index scatter is nondeterministic.
    """
    hidden = f_rnn_buf.shape[1]
    features = target_hidden.index_select(0, rows.long()).view(
        rows.shape[0], -1, hidden
    )
    features = _mix_rmsnorm(features, alpha, norm_weight, eps)
    slots = torch.arange(f_rnn_buf.shape[0], device=features.device)
    onehot = (req_pool_indices.long()[:, None] == slots[None]) & live[:, None]
    scattered = torch.einsum("nr,nh->rh", onehot.float(), features.float())
    return torch.where(onehot.any(0)[:, None], scattered, f_rnn_buf.float()).to(
        f_rnn_buf.dtype
    )


class Inkling2MTPHead(nn.Module):
    markov_head_type = "rnn_hidden"
    uses_recurrent_context = True

    def __init__(self, vocab_size, markov_rank, hidden_size, rms_norm_eps):
        super().__init__()
        self.hidden_size = hidden_size
        self.markov_rank = markov_rank
        self.unpadded_vocab_size = vocab_size
        self.markov_w1 = nn.Embedding(vocab_size, markov_rank)
        self.rnn_proj = nn.Linear(2 * markov_rank + 2 * hidden_size, 3 * markov_rank)
        self.hidden_out = nn.Linear(markov_rank, hidden_size)
        self.dflare_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.final_logit_softcapping = None
        # lm_head (this rank's vocab rows) folded through hidden_out: [V / tp, rank], [V / tp].
        # Buffers so direct_compile lifts them as graph inputs (set once by attach_fold).
        self.register_buffer("_fold_weight", None, persistent=False)
        self.register_buffer("_fold_bias", None, persistent=False)
        self._gather_logits = None
        # Eager sampled drafting reads the rows the last draft forward staged.
        self._staged_f_rnn = None

    @torch.no_grad()
    def attach_fold(self, weight, multiplier):
        # float64 on the host: TPU has no f64, and the fold is done once.
        head = weight.detach().cpu().double()
        folded_weight = head @ self.hidden_out.weight.detach().cpu().double()
        folded_bias = head @ self.hidden_out.bias.detach().cpu().double()
        if multiplier:
            folded_weight /= multiplier
            folded_bias /= multiplier
        self._fold_weight = folded_weight.to(device=weight.device, dtype=weight.dtype)
        self._fold_bias = folded_bias.to(device=weight.device, dtype=weight.dtype)

    def get_prev_embeddings(self, token_ids):
        return self.markov_w1(token_ids.long())

    def _rnn_step(self, state, prev_emb, hidden, f_rnn):
        inputs = torch.cat([state, prev_emb, hidden, f_rnn], dim=-1)
        gate_raw, cand_raw, out_raw = self.rnn_proj(inputs).chunk(3, dim=-1)
        gate = torch.sigmoid(gate_raw)
        state = gate * state + (1.0 - gate) * torch.tanh(cand_raw)
        return state, torch.tanh(out_raw)

    def _step_logits(self, base, latent):
        bias = torch.addmm(
            self._fold_bias, latent.to(self._fold_weight.dtype), self._fold_weight.T
        )
        logits = base + self._gather_logits(bias)
        if self.final_logit_softcapping:
            cap = self.final_logit_softcapping
            logits = torch.tanh(logits / cap) * cap
        return logits

    def sample_block(
        self,
        base_logits,
        *,
        first_prev_tokens,
        hidden_states,
        sampler,
        collect_corrected=True,
        f_rnn=None,
    ):
        """Drafts [bs, gamma] (and their logits) from base logits [bs, gamma, V]."""
        bs = hidden_states.shape[0]
        f_rnn = (self._staged_f_rnn if f_rnn is None else f_rnn)[:bs].to(
            hidden_states.dtype
        )
        state = hidden_states.new_zeros(bs, self.markov_rank)
        prev_tokens = first_prev_tokens
        tokens, corrected = [], []
        for step in range(base_logits.shape[1]):
            previous = self.get_prev_embeddings(prev_tokens).to(hidden_states.dtype)
            state, latent = self._rnn_step(
                state, previous, hidden_states[:, step, :], f_rnn
            )
            logits = self._step_logits(base_logits[:, step, :], latent)
            prev_tokens = sampler(logits, step)
            tokens.append(prev_tokens)
            if collect_corrected:
                corrected.append(logits)
        return (
            torch.stack(tokens, dim=1).to(torch.int64),
            torch.stack(corrected, dim=1) if corrected else None,
        )


class Inkling2MTPAttention(DFlashAttention):
    """DFlash attention plus the context-only K/V projection the KV writer uses."""

    def __init__(
        self, config, layer_id: int, quant_config=None, prefix: str = ""
    ) -> None:
        super().__init__(config, layer_id, quant_config=quant_config, prefix=prefix)
        self.kv_proj_ctx = QKVParallelLinear(
            hidden_size=int(config.hidden_size),
            head_size=self.head_dim,
            total_num_heads=0,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=False,
        )
        self.k_norm_ctx = RMSNorm(self.head_dim, eps=float(config.rms_norm_eps))

    def kv_proj_only(self, hidden_states):
        kv, _ = self.kv_proj_ctx(hidden_states)
        k, v = kv.split([self.kv_size, self.kv_size], dim=-1)
        return k, v

    def apply_k_norm(self, k):
        return self.k_norm_ctx(k.reshape(-1, self.head_dim)).view_as(k)


class Inkling2MTPDecoderLayer(DFlashDecoderLayer):
    attention_cls = Inkling2MTPAttention

    def __init__(self, config, layer_id: int, **kwargs) -> None:
        super().__init__(config, layer_id, **kwargs)
        self.dflare_norm = RMSNorm(
            int(config.hidden_size), eps=float(config.rms_norm_eps)
        )


class Inkling2ForConditionalGenerationMTP(DSparkDraftModel):
    decoder_layer_cls = Inkling2MTPDecoderLayer
    fall_back_to_pt_during_load = False
    allow_patterns_overrides = ["mtp.safetensors"]
    safetensors_index_file = None
    supports_pre_gather_target_hidden_projection = False
    supports_fused_context_kv = False

    def __init__(self, config, quant_config=None, prefix=""):
        # The target checkpoint's quant metadata covers only its experts; the draft
        # is BF16 (load_weights checks every tensor), so it is built unquantized.
        del quant_config
        if config.q_bias or config.o_bias:
            raise ValueError(
                "Inkling2 MTP does not support attention projection biases"
            )
        # DSparkDraftMixin builds a stock markov head; this draft brings its own.
        DFlashDraftModel.__init__(self, config, quant_config=None, prefix=prefix)
        hidden, eps = config.hidden_size, config.rms_norm_eps
        self.block_size = config.block_size
        self.gamma = self.block_size - 1
        self.sample_from_anchor = False
        self.num_stages = config.num_hidden_layers
        self.num_context_features = len(config.target_layer_ids)
        self.logits_mup_width_multiplier = None
        self.fc = self.hidden_norm = None
        self.lm_head = None
        self.confidence_head = None
        self._fused_kv_write_cache = None
        self.markov_head = Inkling2MTPHead(
            config.padded_vocab_size, config.markov_rank, hidden, eps
        )
        self.markov_head.final_logit_softcapping = config.final_logit_softcapping
        self.markov_head.unpadded_vocab_size = config.unpadded_vocab_size
        self.fuse_weight = nn.Parameter(
            torch.zeros(self.num_stages + 1, self.num_context_features),
            requires_grad=False,
        )
        self.mask_embed = nn.Parameter(torch.zeros(hidden), requires_grad=False)
        # softmax(fuse_weight), filled after loading.
        self.register_buffer(
            "fuse_alpha",
            torch.zeros(self.num_stages + 1, self.num_context_features),
            persistent=False,
        )
        # Per request pool row: the recurrent seed f_rnn, written after extend and verify.
        # A buffer so direct_compile lifts it as a graph input (the forward only reads it).
        self.register_buffer("f_rnn_buf", None, persistent=False)

    @property
    def target_hidden_width(self) -> int:
        """Width of the captured target features the KV writer receives (unprojected)."""
        return self.num_context_features * int(self.config.hidden_size)

    def set_block_size(self, block_size):
        if block_size != self.block_size:
            raise ValueError("Inkling2 MTP requires the checkpoint's block size")

    def attach_shared_modules(self, *, embed_tokens, lm_head):
        if should_apply_lm_head_quant_method(lm_head, lm_head.quant_method):
            raise ValueError("Inkling2 MTP requires an unquantized target LM head")
        self.embed_tokens, self.lm_head = embed_tokens, lm_head
        self.markov_head.attach_fold(lm_head.weight, self.logits_mup_width_multiplier)
        self.markov_head._gather_logits = self._gather_logits

    def _gather_logits(self, logits):
        # Real vocab only, like the target's logits: sampled acceptance pairs the two.
        return gather_and_crop_vocab(logits, self.lm_head)[
            ..., : self.markov_head.unpadded_vocab_size
        ]

    def compute_base_logits(self, hidden):
        if self.logits_mup_width_multiplier:
            hidden = hidden / self.logits_mup_width_multiplier
        return self._gather_logits(project_through_lm_head(hidden, self.lm_head)), None

    def forward_embed(self, input_ids):
        """Block rows [root, mask x gamma]: the root's embedding, then ``mask_embed``."""
        ids = input_ids.view(-1, self.block_size)
        root = self.embed_tokens(ids[:, 0]).to(self.mask_embed.dtype)
        is_root = (torch.arange(self.block_size, device=ids.device) == 0)[None, :, None]
        hidden = torch.where(is_root, root[:, None, :], self.mask_embed[None, None, :])
        return hidden.reshape(-1, hidden.shape[-1])

    def allocate_f_rnn(self, num_req_rows: int) -> None:
        self.f_rnn_buf = self.mask_embed.new_zeros(
            num_req_rows, self.config.hidden_size
        )

    def _mix_norm(self, targets, row, norm):
        return _mix_rmsnorm(
            targets, self.fuse_alpha[row], norm.weight, norm.variance_epsilon
        )

    def _split_targets(self, target_hidden):
        return target_hidden.view(
            target_hidden.shape[0], self.num_context_features, self.config.hidden_size
        )

    @torch.no_grad()
    def store_f_rnn(self, *, target_hidden, rows, req_pool_indices, live):
        """Seed f_rnn of each live request from its target features at hidden row ``rows``."""
        norm = self.markov_head.dflare_norm
        self.f_rnn_buf.copy_(
            seed_f_rnn(
                self.f_rnn_buf,
                self.fuse_alpha[-1],
                norm.weight,
                target_hidden,
                rows,
                req_pool_indices,
                live,
                eps=norm.variance_epsilon,
            )
        )

    @torch.no_grad()
    def forward(self, input_ids, positions, forward_batch, input_embeds=None, **kwargs):
        bs = input_ids.numel() // self.block_size
        self.markov_head._staged_f_rnn = self.f_rnn_buf.index_select(
            0, forward_batch.req_pool_indices.view(-1)[:bs].long()
        )
        return DFlashDraftModel.forward(
            self,
            input_ids,
            positions,
            forward_batch,
            input_embeds=self.forward_embed(input_ids),
            **kwargs,
        )

    def sample_draft_tokens_tpu(self, hidden_states, input_ids):
        """Greedy [bs, gamma] drafts inside the draft graph; row 0 of each block is the root."""
        hidden = hidden_states.view(-1, self.block_size, hidden_states.shape[-1])[:, 1:]
        bs = hidden.shape[0]
        base_logits, _ = self.compute_base_logits(hidden.reshape(bs * self.gamma, -1))
        drafts, _ = self.markov_head.sample_block(
            base_logits.view(bs, self.gamma, -1),
            first_prev_tokens=input_ids.view(bs, self.block_size)[:, 0],
            hidden_states=hidden,
            sampler=lambda logits, step: torch.argmax(logits, dim=-1),
            collect_corrected=False,
        )
        return drafts

    def project_target_hidden(self, target_hidden):
        # Every layer mixes the raw features itself (prepare_context_hidden_for_kv).
        return target_hidden

    def prepare_context_hidden_for_kv(self, layer, ctx_hidden):
        layer_id = next(
            i for i, candidate in enumerate(self.layers) if candidate is layer
        )
        context = self._mix_norm(
            self._split_targets(ctx_hidden), layer_id, layer.dflare_norm
        )
        return context.to(layer.self_attn.kv_proj_ctx.weight.dtype)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params = {
            name: param
            for name, param in self.named_parameters()
            if not name.startswith(("embed_tokens.", "lm_head."))
        }
        mapping = (
            (".q_proj.", ".qkv_proj.", "q"),
            (".k_proj.", ".qkv_proj.", "k"),
            (".v_proj.", ".qkv_proj.", "v"),
            (".k_proj_ctx.", ".kv_proj_ctx.", "k"),
            (".v_proj_ctx.", ".kv_proj_ctx.", "v"),
            (".gate_proj.", ".gate_up_proj.", 0),
            (".up_proj.", ".gate_up_proj.", 1),
        )
        loaded, shards = set(), {}
        for checkpoint_name, weight in weights:
            name, shard = checkpoint_name, None
            for source, destination, shard_id in mapping:
                if source in name:
                    name, shard = name.replace(source, destination), shard_id
                    break
            if name not in params:
                raise ValueError(
                    f"Unrecognized MTP checkpoint tensor: {checkpoint_name}"
                )
            param = params[name]
            if weight.dtype != torch.bfloat16 or param.dtype != weight.dtype:
                raise ValueError(
                    f"{checkpoint_name}: MTP requires BF16 parameters and checkpoint weights"
                )
            loader = getattr(param, "weight_loader", default_weight_loader)
            if shard is None:
                loader(param, weight)
            else:
                loader(param, weight, shard)
                shards.setdefault(name, set()).add(shard)
            loaded.add(name)
        missing = set(params) - loaded
        for name, pieces in shards.items():
            expected = (
                {"q", "k", "v"}
                if ".qkv_proj." in name
                else {"k", "v"}
                if ".kv_proj_ctx." in name
                else {0, 1}
            )
            if pieces != expected:
                missing.add(f"{name} shards {sorted(expected - pieces)}")
        if missing:
            raise ValueError(f"Missing MTP checkpoint tensors: {sorted(missing)}")
        self.fuse_alpha.copy_(torch.softmax(self.fuse_weight.float(), dim=-1))
        return loaded


EntryClass = Inkling2ForConditionalGenerationMTP
