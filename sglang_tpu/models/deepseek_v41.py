# SPDX-License-Identifier: Apache-2.0
"""Flash text model: DP-local attention and shared FFN, fused routed EP.

The core V4.1 config normalizes its architecture to DeepseekV4ForCausalLM.
The registry entry below dispatches V4 back to its original implementation.
Checkpoint names and the core loader remain unchanged.
"""

import functools

from torch import nn

from sglang.srt.compilation.support_torch_compile import support_torch_compile
from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsv4.dsv41_sparse import (
    DeepseekV41Compressor,
    DeepseekV41Indexer,
)
from sglang.srt.layers.engram import Engram, EngramHasher, EngramLayout
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM as CoreCausalLM
from sglang.srt.models.deepseek_v4 import (
    MqaAttentionBase,
    get_attn_backend,
    get_attn_tp_context,
    make_hc_mixing_params,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import add_prefix
from sglang_tpu.dsv41.mhc import (
    ShiftedCarry,
    ShiftedMhc,
    hc_combine,
    pack_layer_weights,
)
from sglang_tpu.dsv41.moe import DSV41MoE


class DSV41Attention(MqaAttentionBase):
    def __init__(self, config, layer_id, quant_config, prefix):
        super().__init__(
            config,
            layer_id,
            quant_config,
            prefix,
            wo_a_fp8=True,
            wo_a_keeps_quant_config=True,
            wo_b_reduce_results=False,
        )
        self.compressor = None
        self.indexer = None
        if layer_id in config.kv_source_layer_ids:
            self.compressor = DeepseekV41Compressor(
                config.hidden_size,
                config.head_dim,
                self.compress_ratio,
                config.rms_norm_eps,
                fused_compress=False,
            )
            self.compressor.wkv.float()
            if self.compress_ratio == 2:
                self.compressor.wgate.float()
        if layer_id in config.index_source_layer_ids:
            self.indexer = DeepseekV41Indexer(
                config,
                layer_id,
                config.head_dim,
                quant_config,
                add_prefix("indexer", prefix),
            )
        if self.compressor is not None and self.indexer is None:
            raise ValueError("A KV source must also provide its index-key projection")

    def forward(self, positions, hidden_states, forward_batch):
        backend = get_attn_backend()
        if self.fuse_wqa_wkv:
            qkv, _ = self.wqkv_a(hidden_states)
            q_lora = qkv[:, : self.q_lora_rank]
        else:
            q_lora, _ = self.wq_a(hidden_states)
            qkv = None
        q_lora, q, kv = backend.prepare_mqa_qkv(
            self, hidden_states, q_lora, qkv, positions
        )
        backend.forward_low_ratio_sources(
            layer=self,
            x=hidden_states,
            q_lora=q_lora,
            positions=positions,
            forward_batch=forward_batch,
        )
        output = backend.attention(self, q, kv, positions)
        # CSA2 already applied inverse RoPE. wo_a receives the unrotated value.
        projected = backend.project_output(self, output)
        return self.wo_b(projected)[0]


class DSV41Layer(nn.Module):
    def __init__(self, config, layer_id, quant_config, layout, prefix):
        super().__init__()
        self.layer_id, self.config = layer_id, config
        self.hidden_size = config.hidden_size
        self.self_attn = DSV41Attention(
            config, layer_id, quant_config, add_prefix("self_attn", prefix)
        )
        self.mlp = DSV41MoE(config, layer_id, quant_config, add_prefix("mlp", prefix))
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.hc_mult, self.hc_eps = config.hc_mult, config.hc_eps
        self.hc_sinkhorn_iters, self.rms_norm_eps = (
            config.hc_sinkhorn_iters,
            config.rms_norm_eps,
        )
        (
            self.hc_attn_fn,
            self.hc_ffn_fn,
            self.hc_attn_base,
            self.hc_ffn_base,
            self.hc_attn_scale,
            self.hc_ffn_scale,
        ) = make_hc_mixing_params(config.hc_mult, config.hidden_size)
        self.attn_mhc, self.ffn_mhc = ShiftedMhc(self, "attn"), ShiftedMhc(self, "ffn")
        self.engram = None
        if layout is not None and layer_id in layout.layer_ids:
            self.engram = Engram(
                config, layer_id, layout, quant_config, add_prefix("engram", prefix)
            )

    def forward(self, residual, carry, input_ids, positions, forward_batch):
        if self.engram is not None:
            previous_pre = carry.pre if carry is not None else None
            if carry is not None:
                residual = self.attn_mhc.commit(carry)
            # The host coordinator has already gathered these rows. The index
            # tensor supplies only the local token/head shape to the adapter.
            heads = (self.config.engram_max_ngram_size - 1) * self.config.engram_n_heads
            indices = input_ids[:, None].expand(-1, heads)
            residual = self.engram(residual, indices, forward_batch)
            post, comb, value, pre = self.attn_mhc.start(residual, previous_pre)
        elif carry is None:
            post, comb, value, pre = self.attn_mhc.start(residual)
        else:
            residual, post, comb, value, pre = self.attn_mhc.advance(carry)
        output = self.self_attn(positions, self.input_layernorm(value), forward_batch)
        attention = ShiftedCarry(output, residual, post, comb, pre)
        residual, post, comb, value, pre = self.ffn_mhc.advance(attention)
        valid = get_attn_backend().forward_metadata.valid
        output = self.mlp(self.post_attention_layernorm(value), input_ids, valid)
        return ShiftedCarry(output, residual, post, comb, pre)


class DSV41Model(nn.Module):
    fall_back_to_pt_during_load = False

    def __init__(self, config, quant_config, prefix):
        super().__init__()
        self.config, self.hidden_size = config, config.hidden_size
        self.start_layer, self.end_layer = 0, config.num_hidden_layers
        self.hc_mult = config.hc_mult
        self.engram_layout = layout = EngramLayout.from_config(config)
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            enable_tp=False,
            prefix=add_prefix("embed_tokens", prefix),
        )
        self.layers = nn.ModuleList(
            [
                DSV41Layer(
                    config, i, quant_config, layout, add_prefix(f"layers.{i}", prefix)
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.engram_hasher = (
            EngramHasher.from_config(config, layout) if layout else None
        )
        self.hc_pre_from_prev_sublayer = True
        self.hc_head_fn = self.hc_head_scale = self.hc_head_base = None
        self.dspark_layers_to_capture = self.late_layer_start = None
        self.gemm_output_zero_allocator_size = 0
        self.host_engram = None
        starts = sorted({0, *(layout.layer_ids if layout else ())})
        boundaries = starts + [self.end_layer]
        self.segment_starts = tuple(starts)
        self.segments = tuple(
            functools.partial(self.run_segment, start, stop)
            for start, stop in zip(boundaries, boundaries[1:])
        )

    def get_input_embeddings(self):
        return self.embed_tokens

    def run_segment(
        self, start, stop, carry, input_ids, positions, forward_batch, input_embeds
    ):
        residual = None
        if start == 0:
            embedded = (
                self.embed_tokens(input_ids) if input_embeds is None else input_embeds
            )
            residual = embedded[:, None, :].expand(-1, self.hc_mult, -1).contiguous()
        for layer in self.layers[start:stop]:
            carry = layer(residual, carry, input_ids, positions, forward_batch)
        # No host Engram boundary follows the final layer. Keep its mHC commit,
        # combine and norm in this same device graph instead of compiling a
        # separate tail for every prefill/decode shape.
        return self.finish(carry) if stop == self.end_layer else carry

    def finish(self, carry):
        residual = self.layers[-1].ffn_mhc.commit(carry)
        value = hc_combine(residual.flatten(1), carry.pre, self.hc_mult, residual.dtype)
        return self.norm(value), residual.flatten(1)

    def forward(
        self,
        input_ids,
        positions,
        forward_batch,
        input_embeds=None,
        pp_proxy_tensors=None,
    ):
        from torch_tpu._internal.sync.sync import synchronize

        carry = None
        for start, segment in zip(self.segment_starts, self.segments):
            layer = self.layers[start]
            if layer.engram is not None:
                if self.host_engram is None:
                    raise RuntimeError("Host Engram coordinator is not bound")
                self.host_engram.stage_layer(
                    layer.engram.layer_hash_index, input_ids.shape[0]
                )
            carry = segment(carry, input_ids, positions, forward_batch, input_embeds)
            # Submit this segment before host lookup for the next Engram layer.
            # Its output tensors carry the device dependency into that segment.
            synchronize(list(carry), wait=False)
        return carry


class DeepseekV4ForCausalLM(CoreCausalLM):
    def __new__(cls, config, quant_config=None, prefix=""):
        if getattr(config, "model_type", None) != "deepseek_v41":
            return CoreCausalLM(config, quant_config, prefix)
        return super().__new__(cls)

    def __init__(self, config, quant_config=None, prefix=""):
        nn.Module.__init__(self)
        if not config.hc_pre_from_prev_sublayer:
            raise ValueError("V4.1 requires shifted mHC")
        if not envs.SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE.get():
            raise ValueError("V4.1 requires SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1")
        parallel = get_parallel()
        self.config, self.quant_config = config, quant_config
        self.tp_size, self.pp_group = parallel.tp_size, parallel.pp_group
        self.wo_a_fp8, self.num_fused_shared_experts = True, 0
        self.vision = None
        self.model = DSV41Model(config, quant_config, add_prefix("model", prefix))
        self.lm_head = (
            self.model.embed_tokens
            if config.tie_word_embeddings
            else ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("lm_head", prefix),
                use_attn_tp_group=parallel.enable_dp_lm_head,
            )
        )
        self.logits_processor = LogitsProcessor(config)
        self.capture_aux_hidden_states = False
        self.start_layer, self.end_layer = self.model.start_layer, self.model.end_layer
        self._mhc_prewarmed_at_load = True
        get_attn_tp_context().init_context(config.q_lora_rank, is_dsa=True)
        from sglang_tpu.dsv41.compile_policy import DSV41CompilePolicy

        support_torch_compile(self, policy=DSV41CompilePolicy)

    def post_load_weights(self, is_nextn=False, weight_names=None):
        if is_nextn:
            raise NotImplementedError("MTP is outside the initial V4.1 validation")
        # Ordinary FP8 postprocessing prepares wo_a's channelwise weight and
        # scale after this loader hook, as for the other SGLang dense linears.
        pack_layer_weights(self.model.layers)

    def wants_prefill_autotune(self):
        return False

    def precompile_kernels_after_loading(self):
        return None


EntryClass = DeepseekV4ForCausalLM
