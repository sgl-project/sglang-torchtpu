# SPDX-License-Identifier: Apache-2.0
"""HookRegistry entries (``sglang.srt.plugins`` entry point). TPU worker initialization hooks.

Every hook is tracked debt and carries a header: Why / How (hook type and
target) / Upstream PR or issue / Remove when. A REPLACE on a method longer than
~30 lines means "write the seam instead".
"""

from __future__ import annotations

import logging

from sglang.srt.plugins.hook_registry import HookType
from sglang_tpu.hooks.attn_residual import (
    aggregate,
    aggregate_fused_add,
    prepare_attn_residual_kernel,
    register_attn_residual_ops,
)
from sglang_tpu.hooks.attn_tp import device_supports_input_scattered
from sglang_tpu.hooks.cpu_pinning import pin_scheduler_main_thread
from sglang_tpu.hooks.draft_kv_pool import (
    draft_cell_size_per_token,
    draft_mha_pool_class,
)
from sglang_tpu.hooks.dsa_kv_pool import size_tpu_dsa_kv_pool
from sglang_tpu.hooks.experimental_spec import hook_specs
from sglang_tpu.hooks.glm_flash_megakernel import (
    drop_freed_glm_flash_megakernel_request,
    route_glm_flash_megakernel,
)
from sglang_tpu.hooks.hicache_swa_metrics import hook_specs as hicache_swa_metrics_hooks
from sglang_tpu.hooks.host_padding import hook_specs as host_padding_hooks
from sglang_tpu.hooks.k3_megakernel import (
    attach_k3_megakernel,
    k3_megakernel_forward,
    load_k3_non_layer_weights,
    prepare_k3_capture_kernels,
    skip_k3_decoder_layers,
)
from sglang_tpu.hooks.kda_weights import (
    bind_glm5_next_kda_weights,
    rehome_kimi_k3_kda_weights,
)
from sglang_tpu.hooks.kimi_k3_vision import (
    attach_k3_vision_runner,
    bind_k3_vision_runner,
    encode_with_k3_vision_runner,
    pass_through_projected_features,
)
from sglang_tpu.hooks.mhc import (
    hc_ffn_post_pre,
    hc_post,
    hc_pre,
    register_mhc_ops,
    wire_hc_ffn_post_pre,
)
from sglang_tpu.hooks.moe_topk import (
    biased_grouped_topk,
    biased_topk,
    register_router_topk,
)
from sglang_tpu.hooks.pipeline_parallel import (
    check_pipeline_parallel_compat,
    skip_spec_op_precompile_without_draft,
)
from sglang_tpu.hooks.scheduler import initialize_dense_moe_config
from sglang_tpu.hooks.spec_tp_sync import broadcast_undeferred
from sglang_tpu.hooks.spec_worker import select_experimental_spec_worker
from sglang_tpu.hooks.static_fp8 import bind_kimi_k3_static_fp8
from sglang_tpu.hooks.swa_miss_metrics import hook_specs as swa_miss_metrics_hooks
from sglang_tpu.hooks.twolevel_allreduce import all_reduce as twolevel_all_reduce
from sglang_tpu.hooks.twolevel_allreduce import (
    merged_moe_all_reduce as twolevel_merged_moe_all_reduce,
)

logger = logging.getLogger(__name__)

# (target, hook, hook_type) triples.

HOOKS: list[tuple] = [
    (
        "sglang.srt.managers.scheduler.Scheduler.run_event_loop",
        pin_scheduler_main_thread,
        HookType.BEFORE,
    ),
    (
        "sglang.srt.managers.scheduler.Scheduler.init_moe_gemm_config",
        initialize_dense_moe_config,
        HookType.AFTER,
    ),
    (
        "sglang.srt.arg_groups.validation_hook.check_pipeline_parallel_compat",
        check_pipeline_parallel_compat,
        HookType.AROUND,
    ),
    (
        "sglang.srt.managers.scheduler.Scheduler.precompile_tpu_spec_ops",
        skip_spec_op_precompile_without_draft,
        HookType.AROUND,
    ),
    (
        "sglang.srt.speculative.spec_info.SpeculativeAlgorithm.create_worker",
        select_experimental_spec_worker,
        HookType.AFTER,
    ),
    (
        "sglang.srt.models.kimi_k3.KimiK3LinearModel.__init__",
        register_attn_residual_ops,
        HookType.AFTER,
    ),
    (
        "sglang.srt.models.kimi_k3.make_pp_layers",
        skip_k3_decoder_layers,
        HookType.AROUND,
    ),
    (
        "sglang.srt.models.kimi_k3.KimiK3ForConditionalGeneration.load_weights",
        load_k3_non_layer_weights,
        HookType.AROUND,
    ),
    (
        "sglang.srt.models.kimi_k3.KimiK3LinearForCausalLM.load_weights",
        load_k3_non_layer_weights,
        HookType.AROUND,
    ),
    (
        "sglang.srt.models.kimi_k3.KimiK3LinearForCausalLM.post_load_weights",
        prepare_attn_residual_kernel,
        HookType.AFTER,
    ),
    (
        "sglang.srt.models.kimi_k3.KimiK3LinearForCausalLM.post_load_weights",
        attach_k3_megakernel,
        HookType.AFTER,
    ),
    (
        "sglang.srt.models.kimi_k3.KimiK3LinearModel.forward",
        k3_megakernel_forward,
        HookType.AROUND,
    ),
    (
        "sglang.srt.models.kimi_k3.KimiK3LinearForCausalLM.set_dspark_layers_to_capture",
        prepare_k3_capture_kernels,
        HookType.AFTER,
    ),
    (
        "sglang.srt.models.kimi_k3_vl.MoonViT3dEncoder.__init__",
        attach_k3_vision_runner,
        HookType.AFTER,
    ),
    (
        "sglang.srt.models.kimi_k3.KimiK3ForConditionalGeneration.__init__",
        bind_k3_vision_runner,
        HookType.AFTER,
    ),
    (
        "sglang.srt.models.kimi_k3_vl.KimiK3VisionTower._forward_eager",
        encode_with_k3_vision_runner,
        HookType.AROUND,
    ),
    (
        "sglang.srt.models.kimi_k3_vl.KimiK3MultiModalProjector.forward",
        pass_through_projected_features,
        HookType.AROUND,
    ),
    (
        "sglang.srt.layers.attn_residual._aggregate",
        aggregate,
        HookType.AROUND,
    ),
    (
        "sglang.srt.layers.attn_residual._aggregate_fused_add",
        aggregate_fused_add,
        HookType.AROUND,
    ),
    (
        "sglang.srt.models.glm5_next.Glm5NextDecoderLayer.__init__",
        register_mhc_ops,
        HookType.AFTER,
    ),
    (
        "sglang.srt.layers.layer_boundary.residual.mhc.MHCState.__init__",
        wire_hc_ffn_post_pre,
        HookType.BEFORE,
    ),
    (
        "sglang.srt.layers.layer_boundary.adapters.attention."
        "_device_supports_input_scattered",
        device_supports_input_scattered,
        HookType.AROUND,
    ),
    (
        "sglang.srt.models.glm5_next.Glm5NextForConditionalGeneration.load_weights",
        bind_glm5_next_kda_weights,
        HookType.AFTER,
    ),
    (
        "sglang.srt.models.kimi_k3.KimiK3LinearForCausalLM.post_load_weights",
        rehome_kimi_k3_kda_weights,
        HookType.AFTER,
    ),
    (
        "sglang_tpu.graph_runner.tpu_graph_runner.TpuGraphRunner.__init__",
        route_glm_flash_megakernel,
        HookType.AFTER,
    ),
    (
        "sglang.srt.mem_cache.memory_pool.ReqToTokenPool.free_rows",
        drop_freed_glm_flash_megakernel_request,
        HookType.BEFORE,
    ),
    (
        "sglang.srt.models.glm5_next.Glm5NextDecoderLayer._hc_pre",
        hc_pre,
        HookType.AROUND,
    ),
    (
        "sglang.srt.models.glm5_next.Glm5NextDecoderLayer.hc_ffn_post_pre",
        hc_ffn_post_pre,
        HookType.AROUND,
    ),
    (
        "sglang.srt.models.glm5_next.Glm5NextDecoderLayer.hc_post",
        hc_post,
        HookType.AROUND,
    ),
    (
        "sglang.srt.distributed.parallel_state.GroupCoordinator.all_reduce",
        twolevel_all_reduce,
        HookType.AROUND,
    ),
    (
        "sglang.srt.models.kimi_k3.merged_moe_all_reduce",
        twolevel_merged_moe_all_reduce,
        HookType.AROUND,
    ),
    (
        "sglang.srt.layers.moe.topk.TopK.__init__",
        register_router_topk,
        HookType.AFTER,
    ),
    (
        "sglang.srt.layers.moe.topk.biased_topk_impl",
        biased_topk,
        HookType.AROUND,
    ),
    (
        "sglang.srt.layers.moe.topk.biased_grouped_topk_impl",
        biased_grouped_topk,
        HookType.AROUND,
    ),
    (
        "sglang.srt.models.kimi_k3.KimiK3LinearForCausalLM.post_load_weights",
        bind_kimi_k3_static_fp8,
        HookType.AFTER,
    ),
    (
        "sglang.srt.mem_cache.kv_cache_configurator.KVCacheConfigurator._tpu_mha_pool_class",
        draft_mha_pool_class,
        HookType.AROUND,
    ),
    (
        "sglang.srt.model_executor.pool_configurator.DefaultPoolConfigurator.__init__",
        size_tpu_dsa_kv_pool,
        HookType.AFTER,
    ),
    (
        "sglang.srt.speculative.dflash_utils.dflash_draft_cell_size_per_token",
        draft_cell_size_per_token,
        HookType.AROUND,
    ),
    (
        "sglang.srt.speculative.spec_tp_sync.SpecTpSync.sync",
        broadcast_undeferred,
        HookType.AROUND,
    ),
]

HOOKS.extend(host_padding_hooks())

HOOKS.extend(hicache_swa_metrics_hooks())

HOOKS.extend(swa_miss_metrics_hooks())

HOOKS.extend(hook_specs())

_registered = False


def register() -> None:
    # Two entry points name this function (tpu_hooks in sglang, tpu_oot_hooks in
    # sglang-tpu); with SGLANG_PLATFORM unset both run in the same process.
    global _registered
    if _registered:
        return
    _registered = True

    if not HOOKS:
        logger.debug("sglang_tpu: no hooks registered.")
        return

    from sglang.srt.platforms import current_platform

    # Every host installs the plugin, so the entry points also run off TPU.
    if not current_platform.is_tpu():
        return

    # Register the tokenizer adapter before tokenizer and worker processor loading.
    from sglang_tpu.models.muse_spark_hf_processor import (
        register_muse_spark_processor,
    )

    register_muse_spark_processor()

    from sglang.srt.plugins.hook_registry import HookRegistry

    for target, hook, hook_type in HOOKS:
        HookRegistry.register(target, hook, hook_type)
    logger.info("sglang_tpu: registered %d hook(s).", len(HOOKS))
