# SPDX-License-Identifier: Apache-2.0
"""Stream only this rank's routed-expert tensors with Run:AI.

With expert parallelism every rank keeps ``num_experts / ep_size`` of the
routed experts, but ``RunaiModelStreamerLoader`` fetches every expert tensor
on every rank. This loader builds the fetch plan from the safetensors shard
headers and hands Run:AI only the kept tensor names, so the skipped byte
ranges are never requested from storage. Headers also name tensors a stale
index omits, and they are read through Run:AI, so object-storage checkpoints
take the same path.

The filter is conservative: a tensor is skipped only when its name carries a
numeric ``experts.<id>`` segment, ends in a per-expert weight or scale suffix,
and the id is outside this rank's contiguous range. Every other tensor,
including ``mtp`` layers, fused expert tensors without an id, and per-expert
biases or input scales, is kept. Every expert layer must name the config's
full expert set with uniform suffixes, otherwise the loader falls back to the
stock full-checkpoint stream. Enabled by default for Run:AI loads under EP;
set ``SGLANG_TPU_EP_SHARDED_WEIGHT_LOAD=0`` to force the stock stream.

Hook header (AFTER ``get_model_loader``):
  Why: ``RunaiModelStreamerLoader`` fetches every routed expert on every rank;
       under EP each rank keeps ``1/ep_size`` of them, so startup reads
       ``ep_size`` times the bytes it needs.
  How: AFTER ``get_model_loader`` swaps in
       ``EPShardedRunaiModelStreamerLoader`` on Run:AI loads when EP is on,
       experts are placed contiguously, and the checkpoint index names them.
       ``SGLANG_TPU_EP_SHARDED_WEIGHT_LOAD=0`` keeps the stock stream.
  Upstream PR or issue: not filed (plugin-only loader; SGLang core has no EP
       weight filter).
  Remove when: upstream SGLang's ``RunaiModelStreamerLoader`` accepts an EP
       weight filter.
"""

from __future__ import annotations

import inspect
import re
from collections.abc import Iterable, Mapping

import msgspec

from sglang.srt.model_loader.loader import RunaiModelStreamerLoader
from sglang.srt.model_loader.weight_utils import (
    RUNAI_STREAMER_TENSOR_ATTR,
    maybe_add_mtp_safetensors,
)
from sglang.srt.runtime_context import get_parallel, get_server_args
from sglang_tpu._logging import get_logger
from sglang_tpu.envs import envs

logger = get_logger(__name__)

INDEX_NAME = "model.safetensors.index.json"
# A numeric segment after "experts." is the checkpoint's global routed-expert
# id. "shared_experts" has no digit and fused tensors ("experts.gate_up_proj")
# have no id, so both stay in every rank's plan. The greedy layer prefix makes
# the last ".experts." segment win when a name contains more than one.
_EXPERT_NAME = re.compile(r"^(?P<layer>.*)\.experts\.(?P<id>\d+)(?P<suffix>\..+)?$")
# Checkpoint tensors of the multi-token-prediction head. Their expert layout
# need not follow the target model's EP ownership.
_MTP_NAME = re.compile(r"(?:^|\.)mtp\.")
# Per-expert tensor suffixes that only the owning rank needs. Anything else
# under experts.<id> (biases, input scales, router tensors, fused tensors) is
# fetched by every rank; skipping an unknown suffix would drop a tensor the
# model may need. ".scale" is the DeepSeek FP8 block scale next to ".weight".
_EXPERT_WEIGHT_SUFFIXES = (".weight", ".weight_packed", ".weight_scale", ".scale")


class ExpertFetchPlan(msgspec.Struct, frozen=True):
    """Tensors to fetch and to skip, plus the local expert id range."""

    kept: tuple[str, ...]
    skipped: tuple[str, ...]
    local_experts: tuple[int, int]


def expert_id_of(name: str) -> int | None:
    """Global routed-expert id encoded in a checkpoint tensor name."""
    match = _EXPERT_NAME.match(name)
    return int(match["id"]) if match else None


def _validate_expert_layers(
    layers: Mapping[str, Mapping[int, set[str]]], num_routed_experts: int
) -> None:
    """Reject a checkpoint whose expert ids or suffixes do not match the config.

    A layer that names a subset of the config's experts (a different id space,
    a fused per-group layout, or a different expert count) cannot be sharded
    from the index alone; the caller then keeps the stock stream.
    """
    expected = set(range(num_routed_experts))
    for layer, experts in layers.items():
        if set(experts) != expected:
            missing = sorted(expected - set(experts))
            raise ValueError(
                f"layer {layer} names {len(experts)} of {num_routed_experts} "
                f"expert ids; first missing ids {missing[:4]}"
            )
        suffixes = next(iter(experts.values()))
        if any(expert_suffixes != suffixes for expert_suffixes in experts.values()):
            raise ValueError(f"layer {layer} names experts with different tensors")


def plan_expert_fetch(
    names: Iterable[str],
    *,
    num_routed_experts: int,
    ep_size: int,
    ep_rank: int,
) -> ExpertFetchPlan:
    """Split the checkpoint's tensor names into this rank's and the rest.

    ``names`` comes from the shard headers, not the index, so a tensor a
    stale index omits is still kept. Experts are placed contiguously by EP
    rank, as ``sglang.srt.layers.moe.utils.local_routed_expert_range``
    assumes.
    """
    if num_routed_experts <= 0 or num_routed_experts % ep_size:
        raise ValueError(
            f"{num_routed_experts} routed experts do not divide EP size {ep_size}"
        )
    local = num_routed_experts // ep_size
    start, end = ep_rank * local, (ep_rank + 1) * local
    kept, skipped = [], []
    layers: dict[str, dict[int, set[str]]] = {}
    for name in names:
        if _MTP_NAME.search(name):
            kept.append(name)
            continue
        match = _EXPERT_NAME.match(name)
        if match is None or not name.endswith(_EXPERT_WEIGHT_SUFFIXES):
            kept.append(name)
            continue
        expert = int(match["id"])
        if expert >= num_routed_experts:
            raise ValueError(
                f"expert id {expert} of {name} is outside "
                f"{num_routed_experts} routed experts"
            )
        layers.setdefault(match["layer"], {}).setdefault(expert, set()).add(
            match["suffix"]
        )
        if start <= expert < end:
            kept.append(name)
        else:
            skipped.append(name)
    _validate_expert_layers(layers, num_routed_experts)
    return ExpertFetchPlan(
        kept=tuple(sorted(kept)),
        skipped=tuple(sorted(skipped)),
        local_experts=(start, end),
    )


def num_routed_experts_of(config) -> int | None:
    """Routed-expert count of a text config, or None when it has none.

    The field is named differently across checkpoints (DeepSeek, Qwen,
    Mixtral, gpt-oss), so the known spellings are probed in order.
    """
    for name in (
        "n_routed_experts",
        "num_experts",
        "num_local_experts",
        "moe_num_experts",
    ):
        value = getattr(config, name, None)
        if isinstance(value, int) and value > 0:
            return value
    return None


def shardable(
    *,
    env_enabled: bool,
    model_type: str | None,
    is_draft_model: bool,
    draft_model_idx: int | None,
    ep_size: int,
    num_routed_experts: int | None,
    enable_eplb: bool,
    redundant_experts: int,
    init_expert_location: str,
    ep_join_mode: str | None,
) -> bool:
    """Whether this rank may fetch a subset of the expert tensors."""
    if not env_enabled or is_draft_model or draft_model_idx is not None:
        return False
    if model_type == "deepseek_v41":
        # The V4.1 hook installs its own loader for the Engram row tables.
        return False
    if ep_size <= 1 or num_routed_experts is None:
        return False
    if num_routed_experts % ep_size:
        return False
    if ep_join_mode in ("scale", "recover"):
        # An elastic-EP joiner may store experts on a widened storage rank
        # offset; the checkpoint id no longer names this rank's slot.
        return False
    if enable_eplb or redundant_experts or init_expert_location != "trivial":
        # A non-trivial expert location moves experts off their contiguous
        # EP slot, so the checkpoint id no longer names this rank's storage.
        return False
    return True


class EPShardedRunaiModelStreamerLoader(RunaiModelStreamerLoader):
    """Run:AI loader that requests only this rank's routed-expert tensors."""

    def __init__(
        self,
        load_config,
        model_config,
        *,
        num_routed_experts: int,
        ep_size: int,
        ep_rank: int,
    ):
        super().__init__(load_config)
        self._primary_model_config = model_config
        self._num_routed_experts = num_routed_experts
        self._ep_size = ep_size
        self._ep_rank = ep_rank

    def _get_weights_iterator(self, source):
        if not self._is_primary_source(source):
            yield from super()._get_weights_iterator(source)
            return

        from runai_model_streamer import SafetensorsStreamer

        folder, files = self._prepare_weights(source.model_or_path, source.revision)
        if source.model_config is not None:
            files = maybe_add_mtp_safetensors(
                files, folder, INDEX_NAME, source.model_config.hf_config
            )

        with SafetensorsStreamer() as streamer:
            plan = self._plan(streamer, files)
            if plan is None:
                streamer.stream_files(
                    files,
                    device=self.target_device_str,
                    is_distributed=bool(self._is_distributed),
                )
            else:
                if self._is_distributed:
                    logger.info(
                        "[ep-sharded-load] ignoring distributed streaming: "
                        "per-rank fetch plans differ, so each rank streams "
                        "independently"
                    )
                streamer.stream_files(
                    files,
                    device="cpu",
                    is_distributed=False,
                    tensor_names=set(plan.kept),
                )
            for name, tensor in streamer.get_tensors():
                setattr(tensor, RUNAI_STREAMER_TENSOR_ATTR, True)
                yield name, tensor

    def _is_primary_source(self, source) -> bool:
        return (
            source.model_config is self._primary_model_config
            and source.prefix == ""
            and self.load_config.draft_model_idx is None
        )

    def _plan(self, streamer, files) -> ExpertFetchPlan | None:
        """The fetch plan from the shard headers, or None for the stock stream."""
        if "tensor_names" not in inspect.signature(streamer.stream_files).parameters:
            logger.warning(
                "[ep-sharded-load] this Run:AI has no tensor_names support; "
                "using the stock full-checkpoint stream"
            )
            return None
        try:
            import runai_model_streamer.safetensors_streamer.safetensors_pytorch as sp

            metas = sp.SafetensorsMetadata.from_files(
                streamer.file_streamer, files, None
            )
            names = [tensor.name for meta in metas for tensor in meta.tensors_metadata]
        except (ImportError, AttributeError, OSError, RuntimeError, ValueError) as exc:
            logger.warning(
                "[ep-sharded-load] cannot read safetensors headers (%s); using "
                "the stock full-checkpoint stream",
                exc,
            )
            return None
        try:
            plan = plan_expert_fetch(
                names,
                num_routed_experts=self._num_routed_experts,
                ep_size=self._ep_size,
                ep_rank=self._ep_rank,
            )
        except ValueError as exc:
            logger.warning(
                "[ep-sharded-load] cannot shard this checkpoint (%s); using "
                "the stock full-checkpoint stream",
                exc,
            )
            return None
        if not plan.skipped:
            logger.info(
                "[ep-sharded-load] no non-local expert tensors in the shards; "
                "using the stock full-checkpoint stream"
            )
            return None
        logger.info(
            "[ep-sharded-load] fetching %d tensors from %d shards; skipping %d "
            "non-local routed-expert tensors for experts [%d, %d) of %d",
            len(plan.kept),
            len(files),
            len(plan.skipped),
            plan.local_experts[0],
            plan.local_experts[1],
            self._num_routed_experts,
        )
        return plan


def select_tpu_ep_sharded_loader(result, load_config, model_config=None):
    """AFTER get_model_loader: swap in the EP-sharded Run:AI loader."""
    if type(result) is not RunaiModelStreamerLoader or model_config is None:
        return result
    if not envs.SGLANG_TPU_EP_SHARDED_WEIGHT_LOAD.get():
        return result
    parallel = get_parallel()
    num_routed_experts = num_routed_experts_of(model_config.hf_text_config)
    server_args = get_server_args()
    if not shardable(
        env_enabled=True,
        model_type=model_config.hf_config.model_type,
        is_draft_model=model_config.is_draft_model,
        draft_model_idx=load_config.draft_model_idx,
        ep_size=parallel.moe_ep_size,
        num_routed_experts=num_routed_experts,
        enable_eplb=server_args.enable_eplb,
        redundant_experts=server_args.ep_num_redundant_experts,
        init_expert_location=server_args.init_expert_location,
        ep_join_mode=server_args.ep_join_mode,
    ):
        return result
    local = num_routed_experts // parallel.moe_ep_size
    logger.info(
        "EP-sharded Run:AI weight streaming: experts [%d, %d) of %d on this rank.",
        parallel.moe_ep_rank * local,
        (parallel.moe_ep_rank + 1) * local,
        num_routed_experts,
    )
    return EPShardedRunaiModelStreamerLoader(
        load_config,
        model_config,
        num_routed_experts=num_routed_experts,
        ep_size=parallel.moe_ep_size,
        ep_rank=parallel.moe_ep_rank,
    )
