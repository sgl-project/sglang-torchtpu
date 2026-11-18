# SPDX-License-Identifier: Apache-2.0
"""Column-parallel MTP embedding/hidden fusion projection (eh_proj).

shard_nextn_eh_proj
  Why: DeepseekModelNextN (unless quant_config is quark), Glm4MoeModelNextN and
       Glm4MoeLiteModelNextN fuse the normed draft token embedding (e) and normed
       previous hidden state (h) via an unsharded nn.Linear(2 * H, H) eh_proj,
       so every decode/extend step of a 3-step MTP drafter streams the full
       2 * H * H bf16 weight matrix from HBM on every TP rank (e.g. 4 * 67.1 MB
       = 268.4 MB/step on GLM-5.3-Flash TP8, costing ~80 us/call vs ~31 us/call
       when column-sharded across TP8).
  How: AFTER the __init__ of each of those models (DeepseekModelNextN also
       backs the GLM-5.2 and GLM-5.3-Flash drafts): when eh_proj is a bias-free
       nn.Linear and the token-sharing TP width (attn_tp_size under DP
       attention, else tp_size) is > 1 and divides out_features, replace
       eh_proj with ColumnParallelMtpFusionProj (column-sharded weight +
       all-gather returning a single tensor).
       Bit-identical to the replicated projection.
  Upstream PR or issue: none yet (seam: construct these models' eh_proj as
       ColumnParallelLinear upstream, as nemotron_h_mtp.py already does).
  Remove when: these NextN models shard unquantized eh_proj across TP in core.
"""

from __future__ import annotations

from typing import Any


def shard_nextn_eh_proj(result: None, *args: Any, **kwargs: Any) -> None:
    """AFTER DeepseekModelNextN / Glm4MoeModelNextN / Glm4MoeLiteModelNextN __init__."""
    from sglang_tpu.mtp_fusion_proj import shard_nextn_eh_proj as _shard

    return _shard(result, *args, **kwargs)
