# SPDX-License-Identifier: Apache-2.0
"""Stream only this rank's routed-expert tensors with Run:AI.

The Run:AI counterpart of ``ep_local_expert_reader``, which serves
``DefaultModelLoader``: the same model opt-in
(``supports_tpu_ep_local_expert_reads``), expert range
(``tpu_local_routed_expert_range``), tensor-name rule (``local_expert_id``) and
``SGLANG_TPU_DISABLE_EP_LOCAL_EXPERT_READS`` switch. Tensor names come from the
safetensors shard headers, read through Run:AI, so object-storage checkpoints
take the same path. Run:AI is handed only the kept names, so the skipped byte
ranges are never requested.

Hook header (AFTER ``get_model_loader``):
  Why: ``RunaiModelStreamerLoader`` fetches every routed expert on every rank;
       under EP each rank keeps ``1/ep_size`` of them, so startup reads
       ``ep_size`` times the bytes it needs.
  How: AFTER ``get_model_loader`` swaps in ``EPLocalRunaiModelStreamerLoader``
       for Run:AI loads. It filters the primary source by the model's EP-local
       expert range and takes the stock stream when the model has none.
  Upstream PR or issue: not filed (plugin-only loader).
  Remove when: upstream ``RunaiModelStreamerLoader`` takes a per-source expert
       range like ``DefaultModelLoader.Source.local_expert_range``.
"""

from __future__ import annotations

import inspect
from collections.abc import Iterable, Iterator, Mapping

import msgspec
import torch

from sglang.srt.model_loader.loader import RunaiModelStreamerLoader
from sglang.srt.model_loader.weight_utils import (
    RUNAI_STREAMER_TENSOR_ATTR,
    maybe_add_mtp_safetensors,
)
from sglang_tpu._logging import get_logger
from sglang_tpu.ep_local_expert_reader import (
    is_unused_target_key,
    local_expert_id,
    tpu_local_routed_expert_range,
)

logger = get_logger(__name__)

INDEX_NAME = "model.safetensors.index.json"


class ExpertFetchPlan(msgspec.Struct, frozen=True):
    """Checkpoint tensors one EP rank fetches and skips."""

    kept: frozenset[str]
    skipped: frozenset[str]
    # This rank's routed-expert tensors; the load fails if one is not read.
    local_experts: frozenset[str]


def plan_expert_fetch(
    names: Iterable[str],
    expert_range: tuple[int, int] | None,
    *,
    skip_unused_target_keys: bool = False,
) -> ExpertFetchPlan:
    """Skip the routed experts outside ``expert_range``; None skips nothing."""
    kept, skipped, local = set(), set(), set()
    for name in names:
        if skip_unused_target_keys and is_unused_target_key(name):
            skipped.add(name)
            continue
        eid = None if expert_range is None else local_expert_id(name, expert_range)
        if eid == -1:
            skipped.add(name)
            continue
        kept.add(name)
        if eid is not None:
            local.add(name)
    return ExpertFetchPlan(
        kept=frozenset(kept), skipped=frozenset(skipped), local_experts=frozenset(local)
    )


class EPLocalRunaiModelStreamerLoader(RunaiModelStreamerLoader):
    """Run:AI loader that requests only this rank's routed-expert tensors."""

    # A subclass that reads some tensors outside the stream (``_split_kept``)
    # sets this so it plans even when the model has no EP-local range.
    _plans_without_expert_range = False

    def __init__(self, load_config):
        super().__init__(load_config)
        self._primary_model_config = None
        self._expert_range: tuple[int, int] | None = None

    def _get_all_weights(self, model_config, model):
        self._primary_model_config = model_config
        if self.load_config.draft_model_idx is None and not model_config.is_draft_model:
            self._expert_range = tpu_local_routed_expert_range(model)
        yield from super()._get_all_weights(model_config, model)

    def _get_weights_iterator(self, source):
        if not self._is_primary_source(source) or (
            self._expert_range is None and not self._plans_without_expert_range
        ):
            yield from super()._get_weights_iterator(source)
            return

        from runai_model_streamer import SafetensorsStreamer

        folder, files = self._prepare_weights(source.model_or_path, source.revision)
        if source.model_config is not None:
            files = maybe_add_mtp_safetensors(
                files, folder, INDEX_NAME, source.model_config.hf_config
            )

        with SafetensorsStreamer() as streamer:
            shard_of = self._shard_of_names(streamer, files)
            is_target_load = not getattr(
                self._primary_model_config, "is_draft_model", False
            )
            plan = (
                None
                if shard_of is None
                else plan_expert_fetch(
                    shard_of,
                    self._expert_range,
                    skip_unused_target_keys=is_target_load,
                )
            )
            streamed, extra = (
                (None, iter(()))
                if plan is None
                else self._split_kept(kept=plan.kept, shard_of=shard_of)
            )
            if plan is None or (not plan.skipped and streamed == plan.kept):
                # Nothing to skip: the stock stream, distributed if configured.
                streamer.stream_files(
                    files,
                    device=self.target_device_str,
                    is_distributed=bool(self._is_distributed),
                )
                for name, tensor in streamer.get_tensors():
                    setattr(tensor, RUNAI_STREAMER_TENSOR_ATTR, True)
                    yield name, tensor
                return
            kept_shards = {shard_of[name] for name in streamed if name in shard_of}
            active_files = [f for f in files if f in kept_shards]
            self._log_plan(plan=plan, streamed=streamed, num_files=len(active_files))
            # Run:AI distributed streaming needs identical tensor_names on every
            # rank, and EP ranks keep different experts.
            seen = set()
            if active_files:
                streamer.stream_files(
                    active_files,
                    device="cpu",
                    is_distributed=False,
                    tensor_names=streamed,
                )
                for name, tensor in streamer.get_tensors():
                    seen.add(name)
                    setattr(tensor, RUNAI_STREAMER_TENSOR_ATTR, True)
                    yield name, tensor
        missing = sorted((plan.local_experts & streamed) - seen)
        if missing:
            raise RuntimeError(
                f"EP-local Run:AI read missed {len(missing)} local expert tensors, "
                f"e.g. {missing[:3]}. SGLANG_TPU_DISABLE_EP_LOCAL_EXPERT_READS=1 "
                "streams whole shards."
            )
        yield from extra

    def _split_kept(
        self, *, kept: frozenset[str], shard_of: Mapping[str, str]
    ) -> tuple[frozenset[str], Iterator[tuple[str, torch.Tensor]]]:
        """Names Run:AI streams, and the weights read after the stream."""
        del shard_of
        return kept, iter(())

    def _is_primary_source(self, source) -> bool:
        return (
            source.model_config is self._primary_model_config
            and source.prefix == ""
            and self.load_config.draft_model_idx is None
        )

    def _shard_of_names(self, streamer, files: list[str]) -> dict[str, str] | None:
        """Tensor name -> shard path from the shard headers; None for the stock
        stream when this Run:AI cannot filter or the headers do not read."""
        if "tensor_names" not in inspect.signature(streamer.stream_files).parameters:
            logger.warning(
                "[ep-local-runai] this Run:AI has no tensor_names support; "
                "streaming whole shards"
            )
            return None
        try:
            import runai_model_streamer.safetensors_streamer.safetensors_pytorch as sp

            metas = sp.SafetensorsMetadata.from_files(
                streamer.file_streamer, files, None
            )
        except (ImportError, AttributeError, OSError, RuntimeError, ValueError) as exc:
            logger.warning(
                "[ep-local-runai] cannot read safetensors headers (%s); "
                "streaming whole shards",
                exc,
            )
            return None
        return {
            tensor.name: path
            for path, meta in zip(files, metas, strict=True)
            for tensor in meta.tensors_metadata
        }

    def _log_plan(
        self, *, plan: ExpertFetchPlan, streamed: frozenset[str], num_files: int
    ) -> None:
        logger.info(
            "[ep-local-runai] streaming %d tensors from %d shards; skipping %d "
            "routed-expert tensors outside experts %s; %d read outside the stream",
            len(streamed),
            num_files,
            len(plan.skipped),
            self._expert_range,
            len(plan.kept - streamed),
        )


def select_tpu_ep_local_runai_loader(result, load_config, model_config=None):
    """AFTER get_model_loader: swap in the EP-local Run:AI loader."""
    del model_config
    if type(result) is not RunaiModelStreamerLoader:
        return result
    return EPLocalRunaiModelStreamerLoader(load_config)
