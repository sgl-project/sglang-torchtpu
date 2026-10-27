# SPDX-License-Identifier: Apache-2.0
"""Stage pipeline-local expert parameters on the host during checkpoint loading,
and convert a stage's FP8 experts to the runtime layout a few at a time."""

from __future__ import annotations

import functools
from contextlib import contextmanager
from contextvars import ContextVar

import torch

_SHARDS_PER_EXPERT = {"w1": 2, "w3": 2, "w2": 1}
_UPLOAD_CHUNK_BYTES = 256 << 20
_MAX_STAGED = 32
_active_stager: ContextVar[ExpertParamStager | None] = ContextVar(
    "tpu_expert_stager", default=None
)


class ExpertWriteTracker:
    """Track distinct expert shards until every expert of a parameter is loaded."""

    def __init__(self):
        self._seen: dict[int, set[tuple[int, str]]] = {}

    def record(self, param, expert_id: int | None, shard_id: str | None) -> bool:
        shards = _SHARDS_PER_EXPERT.get(shard_id)
        if shards is None or expert_id is None:
            return False
        seen = self._seen.setdefault(id(param), set())
        seen.add((expert_id, shard_id))
        if len(seen) != param.shape[0] * shards:
            return False
        del self._seen[id(param)]
        return True


class ExpertParamStager:
    """Reuse host buffers and upload a parameter when its last expert arrives."""

    def __init__(self):
        self._staged: dict[int, tuple[torch.nn.Parameter, torch.Tensor]] = {}
        self._pool: dict[tuple, list[torch.Tensor]] = {}
        self._tracker = ExpertWriteTracker()
        self._written: set[int] = set()

    def staging(self, param) -> bool:
        return id(param) not in self._written

    def host_param(self, param: torch.nn.Parameter) -> torch.Tensor:
        entry = self._staged.get(id(param))
        if entry is None:
            if len(self._staged) >= _MAX_STAGED:
                oldest = next(iter(self._staged.values()))[0]
                self.flush(oldest)
            key = (tuple(param.shape), param.dtype)
            free = self._pool.get(key)
            buffer = (
                free.pop()
                if free
                else torch.empty(param.shape, dtype=param.dtype, device="cpu")
            )
            host = torch.Tensor._make_subclass(type(param), buffer, False)
            host.__dict__.update(param.__dict__)
            entry = (param, host)
            self._staged[id(param)] = entry
        return entry[1]

    def written(self, param, expert_id: int, shard_id: str) -> None:
        if self._tracker.record(param, expert_id, shard_id):
            self.flush(param)

    def flush(self, param) -> None:
        entry = self._staged.pop(id(param), None)
        if entry is None:
            return
        param, host = entry
        device = param.device
        per_expert = max(1, host[0].numel() * host.element_size())
        step = max(1, _UPLOAD_CHUNK_BYTES // per_expert)
        chunks = [
            host[start : start + step].to(device, copy=True)
            for start in range(0, host.shape[0], step)
        ]
        param.data = torch.cat(chunks, dim=0) if len(chunks) > 1 else chunks[0]
        if device.type == "tpu":
            from torch_tpu._internal.sync.sync import synchronize

            synchronize([param.data], wait=True)
        self._written.add(id(param))
        buffer = torch.Tensor._make_subclass(torch.Tensor, host.data, False)
        self._pool.setdefault((tuple(host.shape), host.dtype), []).append(buffer)

    def flush_all(self) -> None:
        for param, _ in list(self._staged.values()):
            self.flush(param)
        self._pool.clear()


def get_expert_stager() -> ExpertParamStager | None:
    return _active_stager.get()


def _stage_expert_writes(load_impl):
    """``FusedMoE._weight_loader_impl`` that writes a staged expert shard to
    the parameter's host buffer instead of the device."""

    @functools.wraps(load_impl)
    def _weight_loader_impl(
        self, param, loaded_weight, weight_name, shard_id, expert_id
    ):
        stager = _active_stager.get()
        if (
            stager is not None
            and param.device.type == "tpu"
            and "input_scale" not in weight_name
            and stager.staging(param)
        ):
            load_impl(
                self,
                param=stager.host_param(param),
                loaded_weight=loaded_weight,
                weight_name=weight_name,
                shard_id=shard_id,
                expert_id=expert_id,
            )
            stager.written(param, expert_id, shard_id)
            return
        load_impl(
            self,
            param=param,
            loaded_weight=loaded_weight,
            weight_name=weight_name,
            shard_id=shard_id,
            expert_id=expert_id,
        )

    return _weight_loader_impl


@contextmanager
def expert_loading_context():
    """Stage expert shards on the host while a pipeline stage loads, and
    upload the remaining staged parameters before weight post-processing."""
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

    stager = ExpertParamStager()
    token = _active_stager.set(stager)
    load_impl = FusedMoE._weight_loader_impl
    FusedMoE._weight_loader_impl = _stage_expert_writes(load_impl)
    try:
        yield stager
        stager.flush_all()
    finally:
        FusedMoE._weight_loader_impl = load_impl
        _active_stager.reset(token)


_REQUANT_CHUNK_BYTES = 1 << 30
_TRANSPOSE_CHUNK_ELEMENTS = 1 << 28


def prepare_pipeline_stage_fp8_moe_weights(
    layer: torch.nn.Module, *, block_quant: bool, weight_block_size
) -> bool:
    """Bring a stage's FP8 MoE weights to the gmm_v2 layout ([expert, K, N])
    with bounded device intermediates.

    Block-FP8 checkpoints converting to channelwise runtime scales are
    dequantized, requantized and transposed per expert chunk into new storage;
    other weights are transposed per expert chunk. Returns whether the layer
    keeps blockwise runtime scales.
    """
    from sglang.srt.utils import print_warning_once
    from sglang_tpu.fp8_moe import _get_tpu_fp8_runtime

    if block_quant and _get_tpu_fp8_runtime() == "channelwise":
        print_warning_once(
            "TPU FP8 MoE is converting block-FP8 checkpoint weights to "
            "channelwise runtime scales by default for performance. Use "
            "--tpu-fp8-runtime=channelwise to make this explicit, or "
            "--tpu-fp8-runtime=blockwise to preserve blockwise runtime scales."
        )
        requantize_stage_experts_to_channelwise(
            layer, weight_block_size, transpose=True
        )
        return False
    for name in ("w13_weight", "w2_weight"):
        param = getattr(layer, name)
        transpose_expert_chunks(param)
    return block_quant


# TODO(yinlin09): Convert TP/EP weights here as well,
# replacing fp8_moe._requantize_tpu_fp8_moe_block_to_channelwise.
def requantize_stage_experts_to_channelwise(
    layer: torch.nn.Module,
    block_size,
    *,
    transpose: bool = False,
) -> None:
    """Convert a few experts at a time and concatenate their runtime weights."""
    from sglang.srt.platforms import current_platform
    from sglang_tpu.fp8_moe import (
        _clear_replaced_param,
        _quantize_moe_weight_channelwise,
        block_quant_dequant,
        copy_or_rebind_param,
    )

    free_bytes = current_platform.get_available_memory()[0]
    target_transient_bytes = int(
        min(_REQUANT_CHUNK_BYTES, max(256 << 20, free_bytes // 4))
    )
    for w_name, scale_inv_name in (
        ("w13_weight", "w13_weight_scale_inv"),
        ("w2_weight", "w2_weight_scale_inv"),
    ):
        w_param = getattr(layer, w_name)
        scale_inv = getattr(layer, scale_inv_name).data
        fp8_dtype = w_param.dtype
        num_experts, n, k = w_param.shape
        chunk = max(1, target_transient_bytes // max(1, n * k * 4))
        weights, scales = [], []
        for c0 in range(0, num_experts, chunk):
            c1 = min(c0 + chunk, num_experts)
            w_chunk = block_quant_dequant(
                w_param.data[c0:c1], scale_inv[c0:c1], block_size, torch.float32
            )
            q_chunk, s_chunk = _quantize_moe_weight_channelwise(w_chunk, fp8_dtype)
            del w_chunk
            if transpose:
                q_chunk = q_chunk.transpose(1, 2).contiguous()
            weights.append(q_chunk)
            scales.append(s_chunk)
            _wait_for(q_chunk, s_chunk)
        _replace_with_expert_chunks(w_param, weights)
        copy_or_rebind_param(layer, f"{w_name}_scale", torch.cat(scales, dim=0))
        _clear_replaced_param(layer, scale_inv_name)
    layer.weight_block_size = None


def transpose_expert_chunks(param: torch.nn.Parameter) -> None:
    """Transpose each expert's matrix with bounded device intermediates."""
    weight = param.data
    step = max(1, _TRANSPOSE_CHUNK_ELEMENTS // max(1, weight[0].numel()))
    chunks = []
    for start in range(0, weight.shape[0], step):
        chunk = weight[start : start + step].transpose(1, 2).contiguous()
        _wait_for(chunk)
        chunks.append(chunk)
    _replace_with_expert_chunks(param, chunks)


def _replace_with_expert_chunks(
    param: torch.nn.Parameter, chunks: list[torch.Tensor]
) -> None:
    """Free the parameter's data, then set it to the chunks joined along the
    expert axis; the chunks must not share storage with the freed data."""
    from sglang_tpu.fp8_moe import _free_replaced_storage

    old = param.data
    param.data = torch.empty(0, dtype=old.dtype, device=old.device)
    _free_replaced_storage(old, param.data)
    param.data = torch.cat(chunks, dim=0) if len(chunks) > 1 else chunks[0]
    chunks.clear()


def _wait_for(*tensors: torch.Tensor) -> None:
    """Wait for the programs producing device tensors, so one chunk's
    intermediates are released before the next chunk is queued."""
    if tensors[0].device.type != "tpu":
        return
    from torch_tpu._internal.sync.sync import synchronize

    synchronize(list(tensors), wait=True)
