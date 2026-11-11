# SPDX-License-Identifier: Apache-2.0
"""RunAI loader for DeepSeek-V4.1 Flash with EP-local reads.

What each rank reads is decided by ``ep_local_expert_reader``; this class only
wires it into ``RunaiModelStreamerLoader``. Forked from
sgl-project/sglang-torchtpu#1744 (e0a43ce363, sglang_tpu/dsv41/loader.py).
"""

from __future__ import annotations

import inspect
import json
import logging
from pathlib import Path
from time import perf_counter

from sglang.srt.model_loader.loader import RunaiModelStreamerLoader
from sglang_tpu.models.deepseek_v41.ep_local_expert_reader import (
    EpLocalExpertReader,
    plan_expert_reads,
)

logger = logging.getLogger(__name__)


class DSV41RunaiModelStreamerLoader(RunaiModelStreamerLoader):
    """Reuse construction/postprocessing; specialize only primary weight I/O."""

    def __init__(self, load_config, model_config):
        super().__init__(load_config)
        self._primary_model_config = model_config

    def load_model(self, *, model_config, device_config):
        # Quant-method postprocessing establishes canonical TPU FP8 [K,N]
        # weights. Binding earlier (model.post_load_weights) sees checkpoint
        # block scales instead, and would silently miss the fused QB path.
        model = super().load_model(
            model_config=model_config, device_config=device_config
        )
        from sglang_tpu.models.deepseek_v41.preprocess import bind_model_preprocessing

        bind_model_preprocessing(model)
        return model

    def _get_weights_iterator(self, source):
        config = getattr(source.model_config, "hf_config", None)
        if (
            source.model_config is not self._primary_model_config
            or getattr(config, "model_type", None) != "deepseek_v41"
            or getattr(source.model_config, "is_draft_model", False)
            or source.model_or_path
            != getattr(
                self._primary_model_config,
                "model_weights",
                self._primary_model_config.model_path,
            )
            or source.revision != self._primary_model_config.revision
            or source.prefix
            or self.load_config.draft_model_idx is not None
        ):
            yield from super()._get_weights_iterator(source)
            return

        from runai_model_streamer import SafetensorsStreamer

        from sglang.srt.model_loader.weight_utils import RUNAI_STREAMER_TENSOR_ATTR
        from sglang.srt.runtime_context import get_parallel
        from sglang.srt.utils.runai_utils import is_runai_obj_uri

        if (
            "tensor_names"
            not in inspect.signature(SafetensorsStreamer.stream_files).parameters
        ):
            raise RuntimeError(
                "V4.1 selective loading requires RunAI tensor_names support (>= 0.17.0)"
            )
        folder, files = self._prepare_weights(source.model_or_path, source.revision)
        if is_runai_obj_uri(folder):
            raise ValueError(
                "V4.1 EP-local reads need a local or mounted checkpoint path"
            )
        weight_map = json.loads(
            (Path(folder) / "model.safetensors.index.json").read_text()
        )["weight_map"]
        parallel = get_parallel()
        plan = plan_expert_reads(
            weight_map,
            num_layers=config.num_hidden_layers,
            num_experts=config.n_routed_experts,
            ep_size=parallel.moe_ep_size,
            ep_rank=parallel.moe_ep_rank,
        )
        reader = EpLocalExpertReader(
            weight_map,
            files,
            plan,
            tp_rank=parallel.tp_rank,
            tp_size=parallel.tp_size,
        )
        stream_files = reader.stream_files
        logger.info(
            "V4.1 EP rank %d/%d: reading %d/%d tensors (%d local expert tensors, "
            "%d Engram row ranges) from %d shards",
            parallel.moe_ep_rank,
            parallel.moe_ep_size,
            len(plan.tensor_names),
            plan.total_tensors,
            plan.local_expert_tensors,
            len(reader.row_tables),
            len(stream_files),
        )
        # EP sets differ per rank, and RunAI distributed streaming requires
        # identical tensor_names, so each rank streams independently. Keep
        # owned=True (the default): loaders may retain CPU expert tensors.
        start = perf_counter()
        with SafetensorsStreamer() as streamer:
            streamer.stream_files(
                stream_files,
                device="cpu",
                is_distributed=False,
                tensor_names=reader.streamed,
            )
            for name, tensor in streamer.get_tensors():
                setattr(tensor, RUNAI_STREAMER_TENSOR_ATTR, True)
                yield name, tensor
        logger.info(
            "V4.1 EP rank %d: streamed tensors consumed in %.1f s",
            parallel.moe_ep_rank,
            perf_counter() - start,
        )
        start = perf_counter()
        yield from reader.iter_row_tables()
        logger.info(
            "V4.1 EP rank %d: Engram row ranges consumed in %.1f s",
            parallel.moe_ep_rank,
            perf_counter() - start,
        )
