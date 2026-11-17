# SPDX-License-Identifier: Apache-2.0
"""Opt-in GLM NextN shard selection before payload reads and startup prefetch.

Hook header (AFTER ``get_model_loader``):
  Why: a native draft stored beside its target otherwise reads every shard.
  How: the existing loader factory hook selects a DefaultModelLoader subclass;
       the exact GLM NextN model supplies its own checkpoint prefix.
  Upstream PR or issue: not filed (experimental plugin-only loading policy).
  Remove when: the upstream loader supports model-owned source file selection.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os

from sglang.srt.model_loader.loader import DefaultModelLoader
from sglang.srt.model_loader.weight_utils import maybe_add_mtp_safetensors
from sglang_tpu.envs import envs

logger = logging.getLogger(__name__)
_INDEX_NAME = "model.safetensors.index.json"


class GlmNextNModelLoader(DefaultModelLoader):
    """Preserve stock loading except for an audited primary GLM draft source.

    The opt-in assumes a complete tensor index. An incomplete index can hide
    required draft tensors in apparently target-only shards, so it stays off
    by default. Selected shards are read whole; unindexed sidecars survive.
    """

    def __init__(self, load_config):
        super().__init__(load_config)
        self._primary_model_config = None
        self._mtp_prefix = None

    def _set_primary_model(self, model_config, model):
        self._primary_model_config = model_config
        self._mtp_prefix = None
        if (
            not envs.SGLANG_ENABLE_MTP_SHARD_FILTER.get()
            or self.load_config.draft_model_idx is not None
        ):
            return
        from sglang.srt.models.glm4_moe import GlmMoeDsaForCausalLMNextN

        if type(model) is not GlmMoeDsaForCausalLMNextN:
            return
        if getattr(model.config, "num_nextn_predict_layers", 0) != 1 or not getattr(
            model.config, "num_hidden_layers", 0
        ):
            return
        # Reuse the consuming model's rule, including legacy layer numbering.
        self._mtp_prefix = model._initialize_nextn_conf(True).nextn_layer_prefix + "."

    def _can_filter(self, source):
        return (
            envs.SGLANG_ENABLE_MTP_SHARD_FILTER.get()
            and self._mtp_prefix is not None
            and source.model_config is self._primary_model_config
            and not source.prefix
        )

    def _filter_source(self, resolved):
        if not resolved.use_safetensors or not self._can_filter(resolved.source):
            return resolved
        index_path = os.path.join(resolved.hf_folder, _INDEX_NAME)
        if not os.path.isfile(index_path):
            return resolved
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
        needed = {
            os.path.join(resolved.hf_folder, shard)
            for name, shard in weight_map.items()
            if name.startswith(self._mtp_prefix)
        }
        if not needed or not needed.issubset(resolved.weight_files):
            return resolved
        indexed = {os.path.join(resolved.hf_folder, s) for s in weight_map.values()}
        selected = tuple(
            path
            for path in resolved.weight_files
            if path in needed or path not in indexed
        )
        logger.info(
            "GLM NextN checkpoint reads: %d/%d shards",
            len(selected),
            len(resolved.weight_files),
        )
        return dataclasses.replace(resolved, weight_files=selected)

    def resolve_model_weights(self, model_config, model):
        self._set_primary_model(model_config, model)
        return tuple(
            self._filter_source(source)
            for source in super().resolve_model_weights(model_config, model)
        )

    def _get_all_weights(self, model_config, model):
        self._set_primary_model(model_config, model)
        yield from super()._get_all_weights(model_config, model)

    def _get_weights_iterator(self, source, *, resolved_source=None, **kwargs):
        if resolved_source is None and self._can_filter(source):
            folder, files, use_safetensors = self._prepare_weights(
                source.model_or_path,
                source.revision,
                source.fall_back_to_pt,
                source.allow_patterns_overrides,
            )
            if use_safetensors:
                files = maybe_add_mtp_safetensors(
                    files, folder, _INDEX_NAME, source.model_config.hf_config
                )
            resolved_source = self.ResolvedSource(
                source, folder, tuple(files), use_safetensors
            )
        if resolved_source is not None:
            resolved_source = self._filter_source(resolved_source)
        return super()._get_weights_iterator(
            source, resolved_source=resolved_source, **kwargs
        )


def select_glm_nextn_loader(result, load_config, model_config=None):
    """AFTER get_model_loader: leave other loader implementations untouched."""
    if (
        type(result) is not DefaultModelLoader
        or not envs.SGLANG_ENABLE_MTP_SHARD_FILTER.get()
        or load_config.draft_model_idx is not None
        or model_config is None
    ):
        return result
    from sglang.srt.models.glm4_moe import GlmMoeDsaForCausalLMNextN

    architectures = getattr(model_config.hf_config, "architectures", None) or []
    if architectures[:1] != [GlmMoeDsaForCausalLMNextN.__name__]:
        return result
    return GlmNextNModelLoader(load_config)
