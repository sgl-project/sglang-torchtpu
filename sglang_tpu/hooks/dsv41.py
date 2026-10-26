# SPDX-License-Identifier: Apache-2.0
"""V4.1 physical cache accounting and an unfused mHC primitive.

Why: The core TPU pool budget enumerates only C4/C128, and shifted mHC imports
     a Triton-only combine directly instead of using its existing MhcOps table.
How: AFTER the per-full-token budget adds C1/C2 physical bytes; AROUND combine
     selects tensor math on TPU. Model ordering and model configuration stay in core.
Upstream PR or issue: not filed (local V4.1 integration; needs pool-byte and mHC seams).
Remove when: the configurator queries pool storage sizes and shifted mHC uses MhcOps.
"""


def cache_bytes(result, configurator):
    if not configurator._is_tpu_dsv4:
        return result
    return result + sum(
        (configurator.attn_head_dim + 4 + 2 * configurator.indexer_head_dim)
        * len(configurator.stage_owner_layers.get(ratio, ()))
        / ratio
        for ratio in (1, 2)
    )


def hc_combine(original, x_flat, pre, hc, out_dtype):
    if x_flat.device.type != "tpu":
        return original(x_flat, pre, hc, out_dtype)
    from sglang_tpu.dsv41.mhc import hc_combine as combine

    return combine(x_flat, pre, hc, out_dtype)


def hook_specs():
    from sglang.srt.plugins.hook_registry import HookType

    return [
        (
            "sglang.srt.model_executor.pool_configurator.DSV4PoolConfigurator._get_bytes_per_full_token",
            cache_bytes,
            HookType.AFTER,
        ),
        (
            "sglang.kernels.ops.layernorm.mhc.hc_combine",
            hc_combine,
            HookType.AROUND,
        ),
    ]
