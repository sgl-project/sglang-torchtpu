"""Receive trainer weights over tpu-sync into this TP rank's model parameters.

Each TP rank stages one bf16 buffer per HF tensor, holding its tensor-parallel slice; the trainer's tpu-sync
controller reshards into them, and ``install`` copies them into the model inside a weight-update session,
concatenating q/k/v and gate/up into sglang's fused parameters. Served by ``handlers`` through
/plugin_weight_update; tpu-sync endpoints are unauthenticated.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

import msgspec
import torch

logger = logging.getLogger(__name__)

# Row-parallel layers split their input dim (1); every other 2-D weight is split on its output dim (0).
_ROW_PARALLEL_SUFFIXES = (".o_proj.weight", ".down_proj.weight")
_KV_PROJ_SUFFIXES = (".self_attn.k_proj.weight", ".self_attn.v_proj.weight")
# sglang parameter suffix -> the HF tensors it concatenates on dim 0, in order.
_FUSED_PARAMS = (
    (
        ".self_attn.qkv_proj.weight",
        (
            ".self_attn.q_proj.weight",
            ".self_attn.k_proj.weight",
            ".self_attn.v_proj.weight",
        ),
    ),
    (".mlp.gate_up_proj.weight", (".mlp.gate_proj.weight", ".mlp.up_proj.weight")),
)
_TIED_LM_HEAD = ("lm_head.weight", "model.embed_tokens.weight")


class TpSlice(msgspec.Struct, frozen=True):
    """One TP rank's slice of a full HF tensor, in tpu-sync's mesh terms (mesh axes ``["fsdp", "tp"]``)."""

    local_shape: list[int]
    mesh_shape: list[int]
    sharding_spec: list[str]
    # empty: tpu-sync derives the slice from the rank's mesh position
    global_shard_indices: list[int]


def tp_slice(
    name: str, global_shape: list[int], *, tp_size: int, tp_rank: int, num_kv_heads: int
) -> TpSlice:
    """The slice of HF tensor ``name`` that sglang's TP rank ``tp_rank`` holds."""
    local_shape = list(global_shape)
    if len(global_shape) != 2 or tp_size == 1:
        return TpSlice(
            local_shape, [1] * len(global_shape), [""] * len(global_shape), []
        )
    if name.endswith(_ROW_PARALLEL_SUFFIXES):
        local_shape[1] //= tp_size
        return TpSlice(local_shape, [1, tp_size], ["", "tp"], [])
    if name.endswith(_KV_PROJ_SUFFIXES) and num_kv_heads < tp_size:
        # sglang replicates each KV head on tp_size // num_kv_heads ranks; a rank receives its whole head
        local_shape[0] //= num_kv_heads
        return TpSlice(
            local_shape,
            [num_kv_heads, 1],
            ["tp", ""],
            [tp_rank // (tp_size // num_kv_heads)],
        )
    local_shape[0] //= tp_size
    return TpSlice(local_shape, [tp_size, 1], ["tp", ""], [])


def install_plan(
    param_names: list[str], hf_names: list[str]
) -> dict[str, tuple[str, ...]]:
    """sglang parameter name -> the HF tensors that fill it; raises unless every name on both sides is matched."""
    received = set(hf_names)
    plan: dict[str, tuple[str, ...]] = {}
    for param in param_names:
        sources = (param,)
        for fused_suffix, source_suffixes in _FUSED_PARAMS:
            if param.endswith(fused_suffix):
                prefix = param[: -len(fused_suffix)]
                sources = tuple(prefix + suffix for suffix in source_suffixes)
        if param == _TIED_LM_HEAD[0] and param not in received:
            sources = (_TIED_LM_HEAD[1],)
        plan[param] = sources
    missing = {
        param: sources
        for param, sources in plan.items()
        if not received.issuperset(sources)
    }
    unused = received.difference(
        source for sources in plan.values() for source in sources
    )
    if missing or unused:
        raise ValueError(
            f"tpu-sync weights do not match the model: {len(missing)} parameters lack HF tensors "
            f"(e.g. {list(missing.items())[:3]}), {len(unused)} HF tensors fill no parameter (e.g. {sorted(unused)[:3]})"
        )
    return plan


class TpuSyncReceiver:
    """This TP rank's staging buffers, their tpu-sync registration, and the plan that installs them."""

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        controller_address: str,
        job_name: str,
        weight_shapes: Mapping[str, list[int]],
        parallelism: int,
        tp_size: int,
        tp_rank: int,
        num_kv_heads: int,
        bind_ip: str,
    ) -> None:
        params = dict(model.named_parameters())
        self._plan = install_plan(sorted(params), sorted(weight_shapes))
        names = sorted(weight_shapes)
        slices = {
            name: tp_slice(
                name,
                weight_shapes[name],
                tp_size=tp_size,
                tp_rank=tp_rank,
                num_kv_heads=num_kv_heads,
            )
            for name in names
        }
        _check_shapes(self._plan, params, slices)
        device = next(model.parameters()).device
        # the trainer sends bf16
        self._staging = {
            name: torch.empty(
                slices[name].local_shape, dtype=torch.bfloat16, device=device
            )
            for name in names
        }
        torch.accelerator.synchronize()
        self._synchronizer = _new_weight_synchronizer(
            [self._staging[name] for name in names],
            parallelism=parallelism,
            bind_ip=bind_ip,
        )
        variables = [
            _variable_metadata(
                name=name,
                shape=list(weight_shapes[name]),
                mesh_shape=slices[name].mesh_shape,
                layout=list(range(len(slices[name].local_shape) - 1, -1, -1)),
                item_size=self._staging[name].element_size(),
                layer_idx=index,
                sharding_spec=slices[name].sharding_spec,
                global_shard_indices=slices[name].global_shard_indices,
            )
            for index, name in enumerate(names)
        ]
        _controller_client(controller_address).register_work_unit(
            _unit_id(job_name, str(tp_rank)),
            [f"{bind_ip}:{self._synchronizer.local_port}"],
            f"{bind_ip}:{self._synchronizer.listener_port}",
            mesh_shape=[1, tp_size],
            variables=variables,
            mesh_axes=["fsdp", "tp"],
        )
        logger.info(
            "tpu-sync receiver %s/%d registered %d staging tensors with %s",
            job_name,
            tp_rank,
            len(names),
            controller_address,
        )

    @torch.no_grad()
    def install(self, model: torch.nn.Module, *, check_parity: bool) -> int:
        """Copy the last transfer into ``model``; with ``check_parity``, return how many parameters it changed."""
        # applies the tiling plan delivered with the transfer
        self._synchronizer.h2d()
        params = dict(model.named_parameters())
        changed = 0
        for param_name, sources in self._plan.items():
            received = (
                torch.cat([self._staging[s] for s in sources])
                if len(sources) > 1
                else self._staging[sources[0]]
            )
            target = params[param_name]
            if check_parity:
                changed += int(not torch.equal(target, received.to(target.dtype)))
            target.copy_(received)
        torch.accelerator.synchronize()
        return changed


def _check_shapes(
    plan: Mapping[str, tuple[str, ...]], params: Mapping[str, torch.Tensor], slices
) -> None:
    for param_name, sources in plan.items():
        local = [slices[s].local_shape for s in sources]
        rows = sum(shape[0] for shape in local)
        expected = [rows, *local[0][1:]] if len(sources) > 1 else local[0]
        if list(params[param_name].shape) != expected:
            raise ValueError(
                f"tpu-sync slice of {sources} is {expected} but sglang's {param_name} is "
                f"{list(params[param_name].shape)} (padded or differently sharded layouts are not supported)"
            )


def _new_weight_synchronizer(
    tensors: list[torch.Tensor], *, parallelism: int, bind_ip: str
):
    from tpu_sync.api.torch.weight_synchronizer import WeightSynchronizer

    # auto_h2d=False: install copies to HBM only while the engine is paused inside a session
    return WeightSynchronizer(
        [[t] for t in tensors],
        local_port=0,
        parallelism=parallelism,
        listener_port=0,
        bind_ip=bind_ip,
        unsafe_skip_buffer_lock=True,
        auto_h2d=False,
    )


def _controller_client(address: str):
    from tpu_sync.rpc import raiden_controller

    return raiden_controller.RaidenControllerClientFacade(address)


def _unit_id(job_name: str, rank: str):
    from tpu_sync.api.common import RaidenId

    return RaidenId(job_name=job_name, job_replica_id=rank, data_name="weights")


def _variable_metadata(**fields):
    from tpu_sync.rpc import raiden_service_pb2

    return raiden_service_pb2.VariableMetadataProto(**fields)
