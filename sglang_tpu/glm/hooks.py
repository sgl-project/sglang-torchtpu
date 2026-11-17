# SPDX-License-Identifier: Apache-2.0
"""GLM-specific TPU hooks; loader hook debt is documented in loader.py."""


def _select_glm_nextn_loader(result, load_config, model_config=None):
    # Keep loader/model imports out of plugin startup and the default-off path.
    from sglang_tpu.envs import envs

    if not envs.SGLANG_ENABLE_MTP_SHARD_FILTER.get():
        return result
    from sglang_tpu.glm.loader import select_glm_nextn_loader

    return select_glm_nextn_loader(result, load_config, model_config)
