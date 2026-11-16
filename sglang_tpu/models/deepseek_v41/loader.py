# SPDX-License-Identifier: Apache-2.0
"""Run:AI loader for DeepSeek-V4.1 Flash: EP-local experts, TP-local Engram rows.

Routed experts follow the generic EP-local Run:AI read
(``sglang_tpu.ep_weight_shard``). Engram tables leave the stream and are read
as this rank's row range (``engram_row_reader``).
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping

import torch

from sglang.srt.model_loader.loader import BaseModelLoader, DummyModelLoader
from sglang_tpu.ep_weight_shard import EPLocalRunaiModelStreamerLoader
from sglang_tpu.models.deepseek_v41.engram_row_reader import (
    is_engram_table,
    iter_row_tables,
)

# DeepseekV4ForCausalLM.load_weights skips every MTP tensor of the target model.
_MTP_PREFIX = "mtp."


class DSV41RunaiModelStreamerLoader(EPLocalRunaiModelStreamerLoader):
    """EP-local Run:AI stream without MTP and Engram tables, then Engram rows."""

    _plans_without_expert_range = True

    def _shard_of_names(self, streamer, files: list[str]) -> dict[str, str]:
        shard_of = super()._shard_of_names(streamer, files)
        if shard_of is None:
            # A whole-shard stream would hand every rank each ~100 GB table.
            raise RuntimeError(
                "V4.1 Engram row reads need Run:AI tensor_names support "
                "(>= 0.17.0) and readable shard headers"
            )
        return shard_of

    def _split_kept(
        self, *, kept: frozenset[str], shard_of: Mapping[str, str]
    ) -> tuple[frozenset[str], Iterator[tuple[str, torch.Tensor]]]:
        from sglang.srt.runtime_context import get_parallel

        tables = frozenset(name for name in kept if is_engram_table(name))
        streamed = frozenset(
            name
            for name in kept
            if name not in tables and not name.startswith(_MTP_PREFIX)
        )
        parallel = get_parallel()
        rows = iter_row_tables(
            shard_of=shard_of,
            names=tables,
            tp_rank=parallel.tp_rank,
            tp_size=parallel.tp_size,
        )
        return streamed, rows


class DSV41DeferredDummyLoader(DummyModelLoader):
    """Boots a V4.1 target on dummy weights whose real ones (``real_loader``,
    the loader get_model_loader chose for --load-format) load while the
    direct-tracer compile runs (deferred_load): records the raw parameters
    before the post-load pass and leaves the host Engram tables unfilled."""

    def __init__(self, load_config, real_loader):
        # DummyModelLoader rejects model_loader_extra_config; it belongs to the
        # real loader here.
        BaseModelLoader.__init__(self, load_config)
        self.real_loader = real_loader

    def load_model(self, *, model_config, device_config):
        from sglang.srt.model_loader.loader import (
            _get_quantization_config,
            _initialize_model,
            post_load_weights,
        )
        from sglang.srt.model_loader.utils import set_default_torch_dtype
        from sglang.srt.model_loader.weight_utils import initialize_dummy_weights
        from sglang_tpu.models.deepseek_v41 import deferred_load

        quant_config = _get_quantization_config(model_config, self.load_config)
        with set_default_torch_dtype(model_config.dtype):
            with torch.device(device_config.device):
                model = _initialize_model(model_config, self.load_config, quant_config)
            deferred_load.defer(model, self.real_loader)
            initialize_dummy_weights(deferred_load.device_only(model))
            post_load_weights(model)
            for module in model.modules():
                quant_method = getattr(module, "quant_method", None)
                if quant_method is None or (
                    hasattr(module, "is_weights_quantized")
                    and module.is_weights_quantized()
                ):
                    continue
                quant_method.process_weights_after_loading(module)
        return model.eval()
