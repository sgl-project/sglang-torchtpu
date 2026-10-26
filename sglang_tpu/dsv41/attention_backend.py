# SPDX-License-Identifier: Apache-2.0
"""V4.1 source ownership and CSA2 attention on SGLang request/page metadata."""

import torch
import torch.nn.functional as F

from sglang_tpu.dsv4.dsv4_attention_backend import (
    DSV4TPUAttnBackend,
    apply_tpu_dsv4_rope,
)
from sglang_tpu.dsv41.custom_ops import (
    publish_source,
    register_cache_ops,
    register_ops,
)


def pair_compress(kv, score, req, positions, valid, state, ring_size):
    """Pair adjacent positions, including an even token from the previous chunk.

    Read the old ring before writing this chunk. Only its last ``ring_size`` rows
    per request may write live slots; discarded and padded rows use the spare row.
    The returned value is meaningful only at odd, valid positions.
    """
    dim = kv.shape[-1]
    previous = req * ring_size + (positions - 1) % ring_size
    previous = torch.where(valid & (positions > 0), previous, state.shape[0] - 1)
    carried = state[previous]
    in_batch = torch.cat(
        [
            valid.new_zeros(1),
            valid[1:]
            & valid[:-1]
            & (req[1:] == req[:-1])
            & (positions[1:] == positions[:-1] + 1),
        ]
    )
    previous_kv = torch.where(in_batch[:, None], torch.roll(kv, 1, 0), carried[:, :dim])
    previous_score = torch.where(
        in_batch[:, None], torch.roll(score, 1, 0), carried[:, dim:]
    )
    # A missing partner must not create NaNs in unused even/padding rows.
    previous_score = torch.where(
        (valid & (positions % 2 == 1))[:, None], previous_score, score
    )
    weights = torch.stack([previous_score, score], 1).softmax(1)
    pooled = (torch.stack([previous_kv, kv], 1) * weights).sum(1)

    keep = valid
    if kv.shape[0] > ring_size:
        keep = torch.cat(
            [
                valid[:-ring_size]
                & ((req[:-ring_size] != req[ring_size:]) | ~valid[ring_size:]),
                valid[-ring_size:],
            ]
        )
    slots = torch.where(
        keep, req * ring_size + positions % ring_size, state.shape[0] - 1
    )
    state.index_copy_(0, slots.to(torch.int64), torch.cat([kv, score], -1))
    state[-1, :dim] = 0
    state[-1, dim:] = float("-inf")
    return pooled


def compressor_project(compressor, x):
    # Both C1 and C2 keep their checkpoint projection weights in FP32.
    kv = F.linear(x.float(), compressor.wkv.weight)
    if compressor.compress_ratio == 1:
        return kv, None
    return kv, F.linear(x.float(), compressor.wgate.weight)


def quantize_index_query(values):
    """Keep E4M3 queries; fold the per-head power-of-two scale into weights."""
    x = values.float()
    amax = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4)
    scale = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))
    return (x / scale).clamp(-448, 448).to(torch.float8_e4m3fn), scale.squeeze(-1)


class DSV41TPUAttnBackend(DSV4TPUAttnBackend):
    def __init__(self, model_runner):
        from sglang.srt.runtime_context import get_exec, get_parallel, get_spec

        parallel = get_parallel()
        if any(
            getattr(parallel, name) != 1
            for name in ("pp_size", "attn_cp_size", "attn_dcp_size", "attn_tp_size")
        ):
            raise NotImplementedError("V4.1 CSA2 requires attention TP1, PP1/CP1/DCP1")
        if get_spec().speculative_algorithm is not None:
            raise NotImplementedError("V4.1 initial validation does not enable MTP")
        features = get_exec().features
        if (
            features.enable_encoder_swa_bounded_replay
            or features.enable_decoder_swa_bounded_replay
        ):
            raise NotImplementedError("V4.1 bounded replay is outside this validation")
        config = model_runner.model_config.hf_config
        if set(config.compress_ratios) - {0, 1, 2}:
            raise ValueError("The Flash CSA2 adapter accepts compression ratios 0/1/2")
        if config.sliding_window != 128 or config.index_topk != 512:
            raise ValueError("The selected CSA2 kernel requires SWA128 and TopK512")
        super().__init__(model_runner)
        self.index_sources = {}
        for layer_id, ratio in enumerate(config.compress_ratios):
            if ratio:
                sources = [
                    source
                    for source in config.index_source_layer_ids
                    if source <= layer_id and config.compress_ratios[source] == ratio
                ]
                if not sources:
                    raise ValueError(
                        f"Layer {layer_id} has no ratio-{ratio} index source"
                    )
                self.index_sources[layer_id] = sources[-1]
        self.ops = register_ops()
        self.cache_ops = register_cache_ops()

    def init_forward_metadata(self, forward_batch):
        super().init_forward_metadata(forward_batch)
        metadata = self.forward_metadata
        # Gathered candidates are scoped to this forward, never reused across batches.
        metadata.prepared_index = {}
        metadata.index_versions = {}
        metadata.candidate_version = 0
        bs, tokens = forward_batch.batch_size, forward_batch.input_ids.shape[0]
        if not bs:
            return
        if forward_batch.forward_mode.is_decode_or_idle():
            rows = torch.arange(bs, dtype=torch.int32)
            metadata.index_max_queries = 1
        elif forward_batch.forward_mode.is_extend_without_speculative():
            lengths = torch.as_tensor(
                forward_batch.extend_seq_lens_cpu, dtype=torch.int32
            )
            if lengths.numel() < bs:
                lengths = torch.nn.functional.pad(lengths, (0, bs - lengths.numel()))
            # Use T rather than a per-batch host value: avoids extra graph variants.
            metadata.index_max_queries = tokens
            rows = torch.repeat_interleave(
                torch.arange(bs, dtype=torch.int32), lengths[:bs]
            )
        else:
            raise NotImplementedError(
                "V4.1 supports prefill and decode in this validation"
            )
        if rows.numel() < tokens:
            rows = torch.nn.functional.pad(rows, (0, tokens - rows.numel()), value=-1)
        metadata.token_rows = rows[:tokens].to(self.device)
        metadata.token_requests = forward_batch.req_pool_indices[
            metadata.token_rows.clamp_min(0).long()
        ].int()
        metadata.valid = (forward_batch.out_cache_loc > 0) & (metadata.token_rows >= 0)
        metadata.index_requests = torch.where(
            metadata.valid, metadata.token_rows, -1
        ).int()
        metadata.out_loc = forward_batch.out_cache_loc.int()
        metadata.full_pages = metadata.c4_block_tables_flat.view(bs, -1)
        metadata.swa_pages = metadata.swa_block_tables_flat.view(bs, -1)
        metadata.low_indices = {}
        metadata.gathered = {}
        metadata.candidates = None
        metadata.swa_metadata = None
        metadata.swa_records = None
        metadata.last_read = None

    def init_forward_metadata_replay_cuda_graph(
        self, *args, forward_batch=None, **kwargs
    ):
        if forward_batch is None:
            raise ValueError("V4.1 replay requires its ForwardBatch")
        self.init_forward_metadata(forward_batch)

    def forward_low_ratio_sources(self, *, layer, x, q_lora, positions, forward_batch):
        if not x.shape[0] or not layer.compress_ratio:
            return
        metadata, pool = self.forward_metadata, self.token_to_kv_pool
        ratio, pos = layer.compress_ratio, positions.int()
        if layer.compressor is not None:
            kv, score = compressor_project(layer.compressor, x)
            if ratio == 2:
                state = pool.get_attention_compress_states(layer.layer_id)
                kv = pair_compress(
                    kv,
                    score,
                    metadata.token_requests,
                    pos,
                    metadata.valid,
                    state.kv_score_buffer.kv_score,
                    state.ring_size,
                )
            latent = layer.compressor.finish(kv)
            group_pos = pos - pos % ratio
            emit = metadata.valid & ((pos + 1) % ratio == 0)
            records = torch.where(emit, metadata.out_loc // ratio, -1).int()
            indexer = layer.indexer
            if indexer is None or not indexer.owns_k:
                raise ValueError(
                    "Every V4.1 KV source must own its index-key projection"
                )
            key = indexer.k_norm(indexer.wk(latent))
            key = apply_tpu_dsv4_rope(key, group_pos, layer.freqs_cis)
            latent = apply_tpu_dsv4_rope(latent, group_pos, layer.freqs_cis)
            after = metadata.last_read if metadata.last_read is not None else pos
            publish_source(
                self.cache_ops, pool, layer.layer_id, records, latent, key, after
            )
            source = pool.source_layer_of(layer.layer_id)
            metadata.index_versions[source] = metadata.index_versions.get(source, 0) + 1
            # A rewritten K source invalidates its compact candidate copies.
            for cached in tuple(metadata.prepared_index):
                if cached[0] == source:
                    del metadata.prepared_index[cached]

        if layer.indexer is not None:
            indexer = layer.indexer
            q, _ = indexer.wq_b(q_lora)
            q = q.view(q.shape[0], indexer.n_heads, indexer.index_head_dim)
            q, q_scale = quantize_index_query(
                apply_tpu_dsv4_rope(q, pos, layer.freqs_cis)
            )
            weights = indexer.head_weights_raw(x).float() * indexer.head_weight_scale
            weights = weights * q_scale
            publish = indexer.is_candidate_source and indexer.candidate_topk_blocks > 0
            consume = indexer.uses_candidates and indexer.candidate_topk_blocks > 0
            if consume and metadata.candidates is None:
                raise RuntimeError("Candidate source has not run before its consumer")
            candidates = metadata.candidates
            if candidates is None:
                candidates = torch.full(
                    (x.shape[0], max(1, indexer.candidate_topk_blocks)),
                    -1,
                    dtype=torch.int32,
                    device=x.device,
                )
            lengths = torch.where(metadata.valid, (pos + 1) // ratio, 0).int()
            page_records = self.page_size // ratio
            storage_records = pool.index_page_records(layer.layer_id)
            # Even no-candidate layers use the same D128 FP8 scorer geometry.
            block_size = max(8, indexer.candidate_block_size)
            words = pool.low_ratio_cache(layer.layer_id, index=True)
            if consume:
                source = pool.source_layer_of(layer.layer_id)
                cache_key = (
                    source,
                    metadata.index_versions.get(source, 0),
                    metadata.candidate_version,
                    page_records,
                    storage_records,
                    block_size,
                )
                prepared = metadata.prepared_index.get(cache_key)
                if prepared is None:
                    prepared = self.ops.prepare_index(
                        words,
                        metadata.full_pages,
                        metadata.index_requests,
                        lengths,
                        candidates,
                        page_records,
                        storage_records,
                        self.index_topk,
                        block_size,
                        metadata.index_max_queries,
                    )
                    metadata.prepared_index[cache_key] = prepared
                selected = self.ops.select_prepared(
                    q,
                    weights,
                    lengths,
                    *prepared,
                    metadata.full_pages.shape[1] * page_records,
                    self.index_topk,
                    block_size,
                    layer.layer_id,
                )
            else:
                selected, blocks = self.ops.select(
                    q,
                    weights,
                    words,
                    metadata.full_pages,
                    metadata.index_requests,
                    lengths,
                    candidates,
                    page_records,
                    storage_records,
                    self.index_topk,
                    block_size,
                    publish,
                    layer.layer_id,
                )
                if publish:
                    metadata.candidates = blocks
                    metadata.candidate_version += 1
                    metadata.prepared_index.clear()
            metadata.low_indices[layer.layer_id] = selected

    def attention(self, layer, q, k, positions):
        if not q.shape[0]:
            return q
        metadata, pool = self.forward_metadata, self.token_to_kv_pool
        pos, starts = positions.int(), metadata.cu_seqlens_q
        if metadata.swa_metadata is None:
            metadata.swa_metadata, metadata.swa_records = self.ops.metadata(
                metadata.swa_pages,
                starts,
                pos,
                metadata.valid,
                self.page_size,
            )
        swa = pool.get_swa_buffer(layer.layer_id)
        swa.copy_(self.cache_ops.swa(swa, metadata.swa_records, k, self.page_size))
        sink = layer.attn_sink.float()
        common = (swa, starts, pos, sink, layer.freqs_cis, metadata.swa_metadata)
        if not layer.compress_ratio:
            output = self.ops.swa(q, *common, self.page_size, layer.softmax_scale)
        else:
            index_source = self.index_sources[layer.layer_id]
            key = (pool.source_layer_of(layer.layer_id), index_source)
            gathered = metadata.gathered.get(key)
            if gathered is None:
                output, words, scales, counts = self.ops.gather(
                    q,
                    pool.low_ratio_cache(layer.layer_id),
                    pool.low_ratio_scales(layer.layer_id),
                    metadata.low_indices[index_source],
                    metadata.full_pages,
                    *common,
                    self.page_size,
                    layer.compress_ratio,
                    layer.softmax_scale,
                )
                metadata.gathered[key] = (words, scales, counts)
            else:
                output = self.ops.attend(
                    q,
                    *gathered,
                    *common,
                    self.page_size,
                    layer.compress_ratio,
                    layer.softmax_scale,
                )
        metadata.last_read = output
        return output.masked_fill(~metadata.valid[:, None, None], 0)

    def project_output(self, layer, output):
        return self.ops.project(output, layer.wo_a.weight, layer.wo_a.weight_scale)
