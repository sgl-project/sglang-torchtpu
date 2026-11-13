# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1 Engram lookup from one host table shared by the node's ranks.

With SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1 (layout "shared") every rank maps
the same memfd copy of each layer's table, so under attention DP a rank looks
up its own tokens only: no id all-gather, no value all-reduce, no table in HBM
and no gather of a 100 GB table inside the compiled graph. Before each compiled
forward, on the host:

1. ``host_hash_ids`` hashes this rank's (padded) tokens with core's torch path
   on CPU and commits the CPU history.
2. ``gather_staged_rows`` copies each layer's raw FP8 rows and E8M0 scale bytes
   for those ids to the device, one [T * C, dim] buffer per Engram layer.
3. The compiled forward receives ``local_row_ids`` in place of the hash ids
   (token t, column c reads staged row t * C + c), so row selections the model
   applies to the hash ids (the late-layer tail) apply to them unchanged, and
   ``dequant_staged`` gathers and dequantizes those rows in the graph.
"""

from __future__ import annotations

import contextlib
from types import SimpleNamespace

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardBatch

# Per-call inputs the host stage sets; None outside a staged V4.1 forward.
ForwardBatch.tpu_engram_local_ids = None
ForwardBatch.tpu_engram_staged = None


def shared_host_tables(model) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    """(weight, scale) of each Engram layer's shared host table, by hash index;
    empty unless every Engram table is a shared host table. Tags each embedding
    with its hash index for ``staged_lookup``."""
    tables = {}
    for layer in model.model.layers:
        engram = layer.engram
        if engram is None:
            continue
        embed = engram.embed
        if not embed._shared:
            return ()
        embed.tpu_hash_index = engram.layer_hash_index
        tables[engram.layer_hash_index] = (embed.weight, embed.scale)
    return tuple(tables[i] for i in sorted(tables))


def local_row_ids(
    num_tokens: int, num_layers: int, num_cols: int, device
) -> torch.Tensor:
    """[T, L, C] int32: token t, column c reads staged row t * C + c."""
    rows = torch.arange(num_tokens * num_cols, dtype=torch.int32).view(
        num_tokens, 1, num_cols
    )
    return rows.expand(num_tokens, num_layers, num_cols).contiguous().to(device)


def gather_staged_rows(
    hash_ids: torch.Tensor,
    tables: tuple[tuple[torch.Tensor, torch.Tensor], ...],
    device,
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    """Per hash index, (raw [T * C, dim], scale [T * C, dim // block]) uint8 on
    ``device``: the rows ``hash_ids`` [T, L, C] select from each full table."""
    staged = []
    for index, (weight, scale) in enumerate(tables):
        ids = hash_ids[:, index].reshape(-1).to(torch.int64)
        raw, scales = _host_buffers(index, ids.numel(), weight.shape[1], scale.shape[1])
        torch.index_select(weight.view(torch.uint8), 0, ids, out=raw)
        torch.index_select(scale.view(torch.uint8), 0, ids, out=scales)
        staged.append((raw.to(device), scales.to(device)))
    return tuple(staged)


# (hash index, rows) -> reused host gather buffers; one entry per compiled bucket.
_HOST_BUFFERS: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}


def _host_buffers(index: int, rows: int, dim: int, blocks: int):
    key = (index, rows)
    if key not in _HOST_BUFFERS:
        _HOST_BUFFERS[key] = (
            torch.empty(rows, dim, dtype=torch.uint8),
            torch.empty(rows, blocks, dtype=torch.uint8),
        )
    return _HOST_BUFFERS[key]


def dequant_staged(
    raw: torch.Tensor, scale: torch.Tensor, local_ids: torch.Tensor
) -> torch.Tensor:
    """Rows ``local_ids`` of the staged bytes as bf16 [*local_ids.shape, dim].

    Integer e8m0 dequant: the power of two is (byte << 23) bitcast to fp32,
    exact for bytes 1..254 (byte 0 gives 0 instead of 2**-127; byte 255 never
    appears in a real table).
    """
    rows = raw[local_ids].view(torch.float8_e4m3fn).float()
    scales = scale[local_ids].to(torch.int32)
    power = (scales << 23).view(torch.float32)
    values = rows.unflatten(-1, (scales.shape[-1], -1)) * power.unsqueeze(-1)
    return values.flatten(-2).to(torch.bfloat16)


def staged_lookup(embedding, indices, forward_batch) -> torch.Tensor:
    """EngramEmbedding.forward from the rows the shim staged for this batch."""
    raw, scale = forward_batch.tpu_engram_staged[embedding.tpu_hash_index]
    return dequant_staged(raw, scale, indices)


def move_hasher_to_host(hasher) -> None:
    """Hash on the host from now on: tables and history go to CPU. The compiled
    forward reads the staged local ids instead of calling the hasher."""
    if hasher.token_map.device.type != "cpu":
        hasher.to("cpu")
    if hasher.history is not None and hasher.history.device.type != "cpu":
        hasher.history = hasher.history.to("cpu")


def _cpu(t):
    return t.detach().to("cpu") if isinstance(t, torch.Tensor) else t


def host_hash_ids(hasher, input_ids: torch.Tensor, forward_batch) -> torch.Tensor:
    """[T, L, C] int64 hash ids of the padded batch, history committed on CPU.

    Runs EngramHasher.forward on CPU copies of the fields it reads. Padded
    tokens hash as padding and padded requests commit to the spare history row,
    as on the device. Batches the hasher does not serve (idle) get zeros, which
    stage row 0 and are never read for a real token.
    """
    num_tokens = input_ids.shape[0]
    num_layers, num_cols = hasher.offsets.shape
    mode = forward_batch.forward_mode
    serves = mode.is_decode() or mode.is_target_verify() or mode.is_extend()
    if num_tokens == 0 or not serves:
        return torch.zeros(num_tokens, num_layers, num_cols, dtype=torch.int64)
    batch = SimpleNamespace(
        forward_mode=mode,
        spec_info=forward_batch.spec_info,
        positions=_cpu(forward_batch.positions),
        req_pool_indices=_cpu(forward_batch.req_pool_indices),
        out_cache_loc=_cpu(forward_batch.out_cache_loc),
        extend_seq_lens=_cpu(forward_batch.extend_seq_lens),
        extend_start_loc=_cpu(forward_batch.extend_start_loc),
        # Bucket padding lengthens the device lengths; the host list would not.
        extend_seq_lens_cpu=None,
        engram_history=_padded_history(
            _cpu(forward_batch.engram_history), forward_batch.req_pool_indices
        ),
        tpu_engram_local_ids=None,
    )
    return hasher(_cpu(input_ids), batch)


def _padded_history(history, req_pool_indices):
    """Scheduler history covers the real requests only, but the batch is padded
    to its bucket's request count, so the hasher indexes rows past it. Padding
    requests commit to the spare history row, so zero rows serve them."""
    if history is None or history.shape[0] >= req_pool_indices.shape[0]:
        return history
    missing = req_pool_indices.shape[0] - history.shape[0]
    return torch.cat([history, history.new_zeros(missing, history.shape[1])])


@contextlib.contextmanager
def without_cuda_host_register():
    """_HostTable pins its mapping with cudaHostRegister for GPU reads; the TPU
    lookup gathers on the host, so the pin is a no-op here."""
    stub = SimpleNamespace(cudaHostRegister=lambda *args: 0)
    original = torch.cuda.cudart
    torch.cuda.cudart = lambda: stub
    try:
        yield
    finally:
        torch.cuda.cudart = original
