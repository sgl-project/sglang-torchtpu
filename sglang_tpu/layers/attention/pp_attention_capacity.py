# SPDX-License-Identifier: Apache-2.0
"""Host-side limits for the batched attention kernel's pipeline schedules."""

from functools import lru_cache

import torch

from sglang_tpu.managers.pp_chunks import chunk_pairs


@lru_cache(maxsize=None)
def _capacity_for_shape(
    mode,
    num_q_heads,
    num_kv_heads,
    head_dim,
    num_seqs,
    total_q_tokens,
    pages_per_seq,
    page_size,
    dtype_q,
    dtype_kv,
    layout,
):
    import jax.numpy as jnp
    from jax.experimental.pallas import tpu as pltpu

    from sglang_tpu.layers.attention.pallas.batched_rpa import configs
    from sglang_tpu.layers.attention.pallas.batched_rpa.wrapper import (
        calculate_block_sizes,
    )
    from sglang_tpu.utils.pallas_export import tpu_info_for_cpu_export

    with tpu_info_for_cpu_export():
        model = configs.ModelConfigs(
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            mask_value=0.0,
        )
        serve = configs.ServingConfigs(
            num_seqs=num_seqs,
            page_size=page_size,
            total_q_tokens=total_q_tokens,
            num_page_indices=num_seqs * pages_per_seq,
            dtype_q=jnp.dtype(dtype_q),
            dtype_kv=jnp.dtype(dtype_kv),
            dtype_out=jnp.dtype(dtype_q),
            kv_layout=layout,
        )
        vmem = pltpu.get_tpu_info().vmem_capacity_bytes
        decode, mixed = calculate_block_sizes(model, serve, vmem)
        block = decode if mode == "decode" else mixed
        cfg = configs.RpaConfigs(
            block=block,
            model=model,
            serve=serve,
            mode=configs.RpaCase.DECODE if mode == "decode" else configs.RpaCase.MIXED,
            vmem_limit_bytes=vmem,
        )
        return _capacity_from_config(cfg, pages_per_seq, page_size)


def _capacity_from_config(cfg, pages_per_seq, page_size):
    block = cfg.block
    per_lane = cfg.max_steps_ub * cfg.max_schedule_size_multiplier - block.n_buffer - 1
    if block.batch_size > 1:
        per_lane -= -(-pages_per_seq * page_size // block.bkv_sz)
    return max(0, per_lane) * block.batch_size, block.bq_sz, block.bkv_sz


def capacity_for_batch(
    runner, backend, *, mode, num_seqs, total_q_tokens, attn_tp_size=1
):
    """Use the kernel's actual padded batch shape, dtype, layout and tile choice."""
    from sglang_tpu.layers.attention.attention_pallas_rpa_batched import kv_layout

    config = runner.model_config
    return _capacity_for_shape(
        mode,
        config.get_num_attention_heads(attn_tp_size),
        config.get_num_kv_heads(attn_tp_size),
        config.head_dim,
        num_seqs,
        total_q_tokens,
        -(-backend.max_context_len // backend.page_size),
        backend.page_size,
        str(config.dtype).removeprefix("torch."),
        str(runner.token_to_kv_pool.dtype).removeprefix("torch."),
        kv_layout(),
    )


def attention_schedule_capacity(runner, policy):
    """Batched attention prefill capacity for the largest compiled pipeline step."""
    from sglang_tpu.layers.attention.attention_pallas_rpa_batched import (
        TpuPallasRpaBatchedAttnBackend,
    )

    backend = runner.attn_backend
    if not isinstance(backend, TpuPallasRpaBatchedAttnBackend):
        return None
    return capacity_for_batch(
        runner,
        backend,
        mode="mixed",
        num_seqs=policy._extend_compile_cap(),
        total_q_tokens=policy.t_bucketer.max_val,
    )


def check_schedule_lengths(query_lens, kv_lens, capacity, mode):
    """Reject a step whose query/KV block pairs exceed its schedule buffers."""
    if len(query_lens) != len(kv_lens):
        raise ValueError("Batched attention query and KV lengths must align.")
    pairs = 0
    for query, kv in zip(query_lens, kv_lens):
        query, kv = int(query), int(kv)
        if query < 0 or kv < query:
            raise ValueError(
                f"Invalid batched attention lengths: query={query}, KV={kv}."
            )
        pairs += chunk_pairs(capacity, query, kv - query)
    if pairs > capacity[0]:
        raise ValueError(
            f"Batched {mode} attention needs {pairs} query/KV block pairs, "
            f"but this compiled batch can schedule {capacity[0]}. "
            "Reduce the prefill chunk, batch size or context length."
        )


def _host_lengths(value, count, name):
    if value is None or (
        isinstance(value, torch.Tensor) and value.device.type != "cpu"
    ):
        raise ValueError(f"Pipeline attention requires host {name}.")
    return (
        value[:count].tolist()
        if isinstance(value, torch.Tensor)
        else list(value[:count])
    )


def check_batch_schedule(runner, backend, batch):
    """Check live pipeline lengths before constructing device attention metadata."""
    active = backend._non_empty_batch_size(batch)
    if active == 0:
        return
    kv_lens = _host_lengths(batch.seq_lens_cpu, active, "sequence lengths")
    mode = "decode" if batch.forward_mode.is_decode() else "mixed"
    query_lens = (
        [1] * active
        if mode == "decode"
        else _host_lengths(batch.extend_seq_lens_cpu, active, "extend lengths")
    )
    capacity = capacity_for_batch(
        runner,
        backend,
        mode=mode,
        num_seqs=int(batch.seq_lens.shape[0]),
        total_q_tokens=int(batch.input_ids.shape[0]),
    )
    check_schedule_lengths(query_lens, kv_lens, capacity, mode)


def check_cp_schedule(runner, backend, host):
    """Check a context-parallel extend's attention rows on this rank."""
    from sglang.srt.runtime_context import get_parallel

    rows = host.num_rows
    capacity = capacity_for_batch(
        runner,
        backend,
        mode="mixed",
        num_seqs=2 * host.pad_b,
        total_q_tokens=host.local_rows_padded,
        attn_tp_size=get_parallel().attn_tp_size,
    )
    check_schedule_lengths(
        host.pseudo_q_lens[:rows], host.pseudo_kv_lens[:rows], capacity, "mixed"
    )
