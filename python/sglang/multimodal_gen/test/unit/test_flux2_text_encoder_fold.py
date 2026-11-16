"""FLUX.2's Mistral3 text encoder keeps its sizes in a nested ``text_config``.

Encoder tensor-parallel folding reads the hidden size, head count and MLP width
from the top level of the encoder config; ``Flux2MistralTextConfig`` copies them
up after the checkpoint config is applied, as the loader does. Pure config logic
(the fold group is monkeypatched).
"""

from types import SimpleNamespace

import pytest

from sglang.multimodal_gen.configs.models.encoders.flux_2 import (
    Flux2MistralTextConfig,
)
from sglang.multimodal_gen.runtime.models.encoders import base as encoder_base

transformers = pytest.importorskip("transformers")

# FLUX.2-dev's text_encoder/config.json: the language model sizes only under text_config.
TEXT_CONFIG = {
    "hidden_size": 5120,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "intermediate_size": 32768,
    "num_hidden_layers": 40,
}


def _loaded_config():
    """The encoder config after the loader applies the checkpoint config
    (``TextEncoderLoader.build_model_config``), before the config hook."""
    config = Flux2MistralTextConfig()
    hf_config = transformers.Mistral3Config(text_config=TEXT_CONFIG)
    for key, value in hf_config.__dict__.items():
        setattr(config.arch_config, key, value)
    return config


def test_fold_policy_keeps_folding_across_eight_ranks(monkeypatch):
    monkeypatch.setattr(
        encoder_base,
        "get_folding_tp_group",
        lambda config: SimpleNamespace(world_size=8),
    )
    # `auto` on peer-connected GPUs: the 5120-wide encoder now folds as well.
    monkeypatch.setattr(encoder_base, "group_has_measured_topology", lambda g: True)
    for policy in ("fold", "auto"):
        for hook, expected in ((False, None), (True, "replica")):
            config = _loaded_config()
            if hook:
                config.post_diffusers_config_update()
            config.parallel_folding_mode = "replica"
            encoder_base.finalize_encoder_folding(config, policy)
            assert config.parallel_folding_mode == expected, (policy, hook)
