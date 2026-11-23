# SPDX-License-Identifier: Apache-2.0
"""TPU server-argument decisions, made by wrapping core resolution steps.

Each wrapper is a `register_resolution_hook` on the step that used to make the
decision inline, so the core `arg_groups` handlers carry no TPU branch. The
wrappers register when this module is imported; `tpu_platform` imports it.
"""

from __future__ import annotations

from typing import Any, Callable

from sglang.srt.arg_groups.resolution_hooks import register_resolution_hook

# The hooks import lazily: `tpu_platform` imports this module during platform
# discovery, before the resolving view's platform registry is usable.

# The architectures `handle_model_specific_adjustments` gives the Kimi hybrid
# (KDA) speculative defaults (`apply_kimi_k3_spec_backend_defaults`).
_KIMI_HYBRID_ARCHITECTURES = frozenset(
    {
        "KimiLinearForCausalLM",
        "KimiK3ForConditionalGeneration",
        "KimiK31ForConditionalGeneration",
    }
)


@register_resolution_hook("handle_model_specific_adjustments")
def _reject_kda_speculative_decoding(
    server_args: Any, previous: Callable[[Any], None]
) -> None:
    # Before the step, so it does not pin GPU verify backends first.
    _check_kda_speculative_decoding(server_args)
    previous(server_args)


def _check_kda_speculative_decoding(server_args: Any) -> None:
    from sglang.srt.arg_groups.model_override_base import (
        model_config_of,
        resolving_view,
    )
    from sglang.srt.connector import ConnectorType
    from sglang.srt.runtime_context import get_platform
    from sglang.srt.utils.common import parse_connector_type
    from sglang_tpu.experimental import experimental_spec_enabled

    cfg = resolving_view(server_args)
    if not get_platform().is_tpu or cfg.speculative_algorithm is None:
        return
    # The experimental worker verifies these on the TPU KDA checkpoint path
    # (sglang_tpu.hooks.experimental_spec.kimi_spec_defaults).
    if (
        cfg.speculative_algorithm in ("DSPARK", "DFLASH")
        and experimental_spec_enabled()
    ):
        return
    # The step builds no model config for an instance connector.
    if parse_connector_type(cfg.model_path) == ConnectorType.INSTANCE:
        return
    architecture = model_config_of(server_args).hf_config.architectures[0]
    if architecture not in _KIMI_HYBRID_ARCHITECTURES:
        return
    # No KDA target-verify path on TPU to populate intermediate_ssm.
    raise ValueError(
        "Speculative decoding is not supported for Kimi hybrid (KDA) models "
        "on TPU yet; drop --speculative-algorithm."
    )
