# SPDX-License-Identifier: Apache-2.0
"""V4.1 physical cache accounting and an unfused mHC primitive.

Why: The core TPU pool budget enumerates only C4/C128 and sizes C2 pair state
     as flat 2-D GPU rings, and shifted mHC imports a Triton-only combine
     directly instead of using its existing MhcOps table.
How: AFTER the per-full-token budget adds C1/C2 physical bytes (including TPU
     8-row sublane alignment on page-major index owners); AROUND C2 fixed state
     bytes matches TPUPairStatePool; AROUND combine selects tensor math on TPU.
Upstream PR or issue: not filed (local V4.1 integration; needs pool-byte and mHC seams).
Remove when: the configurator queries pool storage sizes and shifted mHC uses MhcOps.
"""


def cache_bytes(result, configurator):
    # The plugin's DSV4 budget hook already owns this method on TPU; V4 configs
    # have no C1/C2 owner layers, so the sum below adds nothing for them.
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import DSV41_INDEX_PAGE_SIZE

    # Match PackedFP8IndexPool(full_pages=...): feature-major E4M3 plus one
    # FP32 scale, with per-full-page rows aligned to a multiple of 8 sublanes.
    page_rows = (33 * DSV41_INDEX_PAGE_SIZE + 127) // 128
    page_size = int(getattr(configurator, "page_size", 256))
    total = 0.0
    for ratio in (1, 2):
        owners = len(configurator.stage_owner_layers.get(ratio, ()))
        if not owners:
            continue
        records_per_full = page_size // ratio
        subpages = records_per_full // DSV41_INDEX_PAGE_SIZE
        a, b = 8, subpages
        while b:
            a, b = b, a % b
        stride_align = 8 // a
        stride = (page_rows + stride_align - 1) // stride_align * stride_align
        index_bytes = (subpages * stride * 512) / records_per_full
        scale_bytes = 4096.0 / records_per_full
        total += (
            (configurator.attn_head_dim + scale_bytes + index_bytes) * owners / ratio
        )
    return result + total


def c2_state_bytes(original, configurator, max_running_requests: int) -> int:
    owners = len(configurator.stage_owner_layers.get(2, ()))
    if not owners:
        return original(configurator, max_running_requests)
    num_req_slots = configurator._get_num_req_slots(max_running_requests)
    ring_size = configurator.ring_sizes[2]
    ring_stride = max(8, (int(ring_size) + 7) // 8 * 8)
    feat_stride = ((2 * int(configurator.attn_head_dim) + 127) // 128) * 128
    return (num_req_slots + 1) * ring_stride * feat_stride * 4 * owners


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
            "sglang.srt.model_executor.pool_configurator.DSV4PoolConfigurator._get_c2_state_fixed_bytes",
            c2_state_bytes,
            HookType.AROUND,
        ),
        (
            "sglang.kernels.ops.layernorm.mhc.hc_combine",
            hc_combine,
            HookType.AROUND,
        ),
    ]
