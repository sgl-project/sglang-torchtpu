# SPDX-License-Identifier: Apache-2.0
"""Synthetic single-request inputs for TPU pipeline timing."""

from contextlib import contextmanager

import torch

from sglang.srt.layers.aux_hidden_states import AUX_HIDDEN_STATES_PROXY_KEY
from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
from sglang_tpu.pp_wave import num_relayed_aux_slots


@contextmanager
def profile_cache(policy, max_tokens):
    """Reserve distinct KV write slots and restore the dummy request row."""
    runner = policy.model_runner
    allocator = runner.token_to_kv_pool_allocator
    page = runner.page_size
    size = -(-max_tokens // page) * page
    prefix_cpu = torch.zeros(1, dtype=torch.int64)
    seq_cpu = torch.tensor([size], dtype=torch.int64)
    slots = allocator.alloc_extend(
        prefix_cpu.to(runner.device),
        prefix_cpu,
        seq_cpu.to(runner.device),
        seq_cpu,
        torch.full((1,), -1, dtype=torch.int64, device=runner.device),
        size,
    )
    if slots is None:
        raise RuntimeError(f"Pipeline profiling requires {size} free KV tokens.")
    policy._pp_profile_slots = slots
    try:
        table = runner.req_to_token_pool.req_to_token
        row = policy._dummy_kv_slot()
        saved = table[row].clone()
        try:
            yield
        finally:
            table[row] = saved
    finally:
        try:
            allocator.free(slots)
        finally:
            del policy._pp_profile_slots


def set_profile_inputs(policy, batch, args, kwargs, tokens, prefix):
    """Build one real-sized chunk behind a prefix of reserved padding pages."""
    runner = policy.model_runner
    device = runner.device
    generator = torch.Generator(device="cpu").manual_seed(0)
    batch.input_ids = torch.randint(
        0,
        runner.model_config.vocab_size,
        (tokens,),
        generator=generator,
    ).to(device)
    batch.positions = torch.arange(
        prefix, prefix + tokens, device=device, dtype=torch.int64
    )
    batch.seq_lens = torch.tensor([prefix + tokens], dtype=torch.int64, device=device)
    batch.orig_seq_lens = batch.seq_lens
    batch.seq_lens_cpu = torch.tensor([prefix + tokens], dtype=torch.int64)
    batch.seq_lens_sum = prefix + tokens
    batch.extend_num_tokens = tokens
    batch.extend_seq_lens = torch.tensor([tokens], dtype=torch.int32, device=device)
    batch.extend_prefix_lens = torch.tensor([prefix], dtype=torch.int32, device=device)
    batch.extend_start_loc = torch.zeros(1, dtype=torch.int32, device=device)
    batch.extend_seq_lens_cpu = [tokens]
    batch.extend_prefix_lens_cpu = [prefix]
    batch.extend_logprob_start_lens_cpu = [0]
    slots = policy._pp_profile_slots[:tokens]
    batch.out_cache_loc = slots.to(batch.out_cache_loc.dtype)
    row = policy._dummy_kv_slot()
    table = runner.req_to_token_pool.req_to_token
    table[row, :prefix] = 0
    table[row, prefix : prefix + tokens] = slots.to(
        device=table.device, dtype=table.dtype
    )
    args = (batch.input_ids, batch.positions, batch)
    # args is rebuilt positionally, so drop keyword copies the vision policy's
    # warmup passes; input_embeds too, since the entry point recomputes it and a
    # stale one describes the pre-profile batch.
    for key in ("input_ids", "positions", "forward_batch", "input_embeds"):
        kwargs.pop(key, None)
    tensors = [
        batch.input_ids,
        batch.positions,
        batch.seq_lens,
        batch.extend_seq_lens,
        batch.extend_prefix_lens,
        batch.out_cache_loc,
        table,
    ]
    if "pp_proxy_tensors" in kwargs:
        # Local import: decoder_compile_policy reaches this module through the
        # profiling entry point.
        from sglang_tpu.compilation.decoder_compile_policy import (
            max_extend_tokens,
        )

        hidden = runner.model_config.hidden_size
        # The wave's buffer height and hand-off dtype (see _pp_init_wave), not
        # this chunk's token count and the model config dtype: serving always
        # delivers the full buffer and the shim trims it to the padded bucket,
        # so a chunk-sized proxy here would trace a row count serving never
        # repeats.
        rows = max_extend_tokens(context_len=runner.model_config.context_len)

        transfer_dtype = getattr(runner.model, "pp_transfer_dtype", runner.dtype)

        def randn(cols: int, dtype: torch.dtype) -> torch.Tensor:
            return torch.randn(
                (rows, cols),
                generator=generator,
                dtype=dtype,
            ).to(device)

        proxy = {
            "hidden_states": randn(hidden, transfer_dtype),
            "residual": randn(hidden, transfer_dtype),
        }
        # Serving hands the aux capture slots on with the activations, so
        # the profiled chunk must carry them too.
        num_aux = num_relayed_aux_slots(runner.spec_aux_config, runner.model)
        if num_aux:
            proxy[AUX_HIDDEN_STATES_PROXY_KEY] = randn(hidden * num_aux, runner.dtype)
        kwargs["pp_proxy_tensors"] = PPProxyTensors(proxy)
        tensors.extend(proxy.values())
    from torch_tpu._internal.sync.sync import synchronize

    synchronize(
        [tensor for tensor in tensors if tensor.device.type == "tpu"], wait=True
    )
    return args, kwargs
