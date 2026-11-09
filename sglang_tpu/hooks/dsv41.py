# SPDX-License-Identifier: Apache-2.0
"""V4.1 physical cache accounting and an unfused mHC primitive.

Why: The core TPU pool budget enumerates only C4/C128, and shifted mHC imports
     a Triton-only combine directly instead of using its existing MhcOps table.
How: AFTER the per-full-token budget adds C1/C2 physical bytes; AROUND combine
     selects tensor math on TPU. Model ordering and model configuration stay in core.
last_loc_on_host
  Why: get_last_loc picks Triton unless the attention backend is on a fixed
       allowlist of non-Triton names; V4.1's "dsv4" backend is not on it, so
       speculative decode-cache allocation (spec_utils.allocate_and_assign_
       decode_cache, reached by the experimental TPU spec worker) launched a
       Triton kernel on TPU, failing the first speculative decode. On TPU the
       request table is host-resident (or a TPU tensor); Triton never applies.
  How: AROUND get_last_loc: a CPU/TPU request table takes get_last_loc_torch;
       other devices keep the original dispatch. The registry also rebinds
       spec_utils' `from ... import get_last_loc`.
Upstream PR or issue: not filed (local V4.1 integration; needs pool-byte and mHC seams).
Remove when: the configurator queries pool storage sizes and shifted mHC uses MhcOps;
             get_last_loc dispatches on the request table's device (or the platform).
"""


def cache_bytes(result, configurator):
    # The plugin's DSV4 budget hook already owns this method on TPU; V4 configs
    # have no C1/C2 owner layers, so the sum below adds nothing for them.
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import DSV41_INDEX_PAGE_SIZE

    # Match PackedFP8IndexPool: feature-major E4M3 plus one FP32 scale,
    # rounded to complete int32[128] owner rows per physical core page.
    # 64 records need 17 rows (136 B/record); 128 need 33 (132 B/record).
    page_rows = (33 * DSV41_INDEX_PAGE_SIZE + 127) // 128
    index_bytes = page_rows * 512 / DSV41_INDEX_PAGE_SIZE
    return result + sum(
        (configurator.attn_head_dim + 4 + index_bytes)
        * len(configurator.stage_owner_layers.get(ratio, ()))
        / ratio
        for ratio in (1, 2)
    )


def hc_combine(original, x_flat, pre, hc, out_dtype):
    if x_flat.device.type != "tpu":
        return original(x_flat, pre, hc, out_dtype)
    from sglang_tpu.dsv41.mhc import hc_combine as combine

    return combine(x_flat, pre, hc, out_dtype)


def last_loc_on_host(
    original, req_to_token, req_pool_indices_tensor, prefix_lens_tensor
):
    if req_to_token.device.type not in ("cpu", "tpu"):
        return original(req_to_token, req_pool_indices_tensor, prefix_lens_tensor)
    from sglang.srt.mem_cache.allocation import get_last_loc_torch

    return get_last_loc_torch(req_to_token, req_pool_indices_tensor, prefix_lens_tensor)


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
        (
            "sglang.srt.mem_cache.allocation.get_last_loc",
            last_loc_on_host,
            HookType.AROUND,
        ),
    ]
