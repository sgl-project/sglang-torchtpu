# SPDX-License-Identifier: Apache-2.0
"""Select the V4.1 transport without changing core loader construction.

Why: RunaiModelStreamerLoader has no model-owned tensor selection seam.
How: AFTER get_model_loader selects a subclass only for the V4.1 target;
     the subclass inherits construction, weight loading and postprocessing.
Upstream PR or issue: not filed (local selective-loading prototype).
Remove when: upstream accepts a model-owned checkpoint selection seam.
"""


def select_loader(result, load_config, model_config=None):
    from sglang.srt.model_loader.loader import RunaiModelStreamerLoader

    config = getattr(model_config, "hf_config", None)
    if (
        type(result) is not RunaiModelStreamerLoader
        or getattr(config, "model_type", None) != "deepseek_v41"
        or getattr(model_config, "is_draft_model", False)
        or load_config.draft_model_idx is not None
    ):
        return result
    from sglang_tpu.dsv41.loader import DSV41RunaiModelStreamerLoader

    return DSV41RunaiModelStreamerLoader(load_config, model_config)


def hook_specs():
    from sglang.srt.plugins.hook_registry import HookType

    return [
        (
            "sglang.srt.model_loader.loader.get_model_loader",
            select_loader,
            HookType.AFTER,
        )
    ]
