# SPDX-License-Identifier: Apache-2.0
"""Column-parallel MTP embedding/hidden fusion projection (``eh_proj``).

In DeepSeek-family NextN / MTP draft models (DeepSeek-V3/R1, GLM-5, GLM-5.3-Flash),
``eh_proj`` fuses the normed draft token embedding ``e`` and normed previous
hidden state ``h`` via a ``2 * H -> H`` linear projection before the MTP decoder
layer. Checkpoints keep ``eh_proj`` in unquantized bf16, and core builds it as a
replicated ``nn.Linear`` unless ``quant_config`` is ``quark``.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from sglang.srt.layers.dp_attention import is_dp_attention_enabled
from sglang.srt.layers.linear import ColumnParallelLinear
from sglang.srt.layers.utils.common import add_prefix
from sglang.srt.runtime_context import get_parallel


class ColumnParallelMtpFusionProj(ColumnParallelLinear):
    """Column-sharded unquantized MTP fusion projection returning a single tensor.

    ``DeepseekModelNextN.forward`` unpacks ``(output, _)`` only when
    ``isinstance(self.eh_proj, ReplicatedLinear)`` and expects a single tensor
    otherwise (matching ``nn.Linear``). Under DP attention, ranks outside the
    attention-TP group hold distinct token batches, so the column shard and
    all-gather stay within ``attn_tp_group``.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        params_dtype: torch.dtype,
        prefix: str,
        use_attn_tp_group: bool,
    ) -> None:
        self.prefix = prefix
        self._use_attn_tp_group = use_attn_tp_group
        super().__init__(
            input_size=input_size,
            output_size=output_size,
            bias=False,
            gather_output=not use_attn_tp_group,
            params_dtype=params_dtype,
            quant_config=None,
            prefix=prefix,
            # Under attention TP, forward gathers on attn_tp_group itself.
            parallel_group="attn_tp" if use_attn_tp_group else None,
        )

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        output, _ = super().forward(input_)
        if self._use_attn_tp_group:
            output = get_parallel().attn_tp_group.all_gather(output, dim=-1)
        return output


def shard_nextn_eh_proj(
    result: None,
    model: Any,
    config: Any = None,
    quant_config: Any = None,
    prefix: str = "",
    *args: Any,
    **kwargs: Any,
) -> None:
    """Replace a NextN draft model's replicated ``nn.Linear`` ``eh_proj``
    with ``ColumnParallelMtpFusionProj`` when token-sharing TP divides it."""
    del config, quant_config, args, kwargs
    eh_proj = getattr(model, "eh_proj", None)
    if type(eh_proj) is not nn.Linear or eh_proj.bias is not None:
        return result

    parallel = get_parallel()
    sub_tp = is_dp_attention_enabled() or getattr(parallel, "attn_cp_size", 1) > 1
    tp_size = parallel.attn_tp_size if sub_tp else parallel.tp_size
    if tp_size <= 1 or eh_proj.out_features % tp_size != 0:
        return result

    with torch.device(eh_proj.weight.device):
        model.eh_proj = ColumnParallelMtpFusionProj(
            eh_proj.in_features,
            eh_proj.out_features,
            params_dtype=eh_proj.weight.dtype,
            prefix=add_prefix("eh_proj", prefix),
            use_attn_tp_group=sub_tp,
        )
    return result
