"""GPT-OSS weight loading over runai-streamed tensors.

`_load_weights_mxfp4` hands expert tensors to the expert loader as they arrive
and copies the non-expert set it holds for the sorted pass out of the streamer
buffer. Pure CPU: the model is a stub and the streamer is a generator that
overwrites its buffer after every yield.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.model_loader.weight_utils import RUNAI_STREAMER_TENSOR_ATTR
from sglang.srt.models import gpt_oss
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_NAMES = [
    "model.layers.0.mlp.experts.gate_up_proj_blocks",
    "model.embed_tokens.weight",
    "model.layers.0.mlp.experts.down_proj_scales",
    "model.layers.0.input_layernorm.weight",
]


def _streamed(names):
    """Yield one tensor per name from a single reused buffer, as the runai
    streamer does, filling the buffer with the item's index."""
    buffer = torch.empty(4, dtype=torch.uint8)
    for i, name in enumerate(names):
        buffer.fill_(i)
        view = buffer.view(4)
        setattr(view, RUNAI_STREAMER_TENSOR_ATTR, True)
        yield name, view
    buffer.fill_(255)


def _stub():
    seen = {"experts": [], "normal": []}

    def load_experts(weights):
        for name, weight in weights:
            seen["experts"].append((name, weight.clone()))
        return set()

    def load_normal(weights, **kwargs):
        seen["normal"] = [(name, weight) for name, weight in weights]

    stub = SimpleNamespace(
        quant_config=SimpleNamespace(get_name=lambda: "mxfp4"),
        _load_mxfp4_experts_weights=load_experts,
        _load_normal_weights=load_normal,
    )
    return stub, seen


class TestStreamedWeightLoading(CustomTestCase):
    def test_experts_consumed_before_buffer_reuse(self):
        stub, seen = _stub()
        gpt_oss.GptOssForCausalLM._load_weights_mxfp4(
            stub, _streamed(_NAMES), is_nextn=False, weight_name_mapping={}
        )
        self.assertEqual([n for n, _ in seen["experts"]], [_NAMES[0], _NAMES[2]])
        for (name, weight), expected in zip(seen["experts"], (0, 2)):
            self.assertTrue(bool((weight == expected).all()), name)

    def test_normal_weights_survive_buffer_reuse(self):
        stub, seen = _stub()
        gpt_oss.GptOssForCausalLM._load_weights_mxfp4(
            stub, _streamed(_NAMES), is_nextn=False, weight_name_mapping={}
        )
        self.assertEqual([n for n, _ in seen["normal"]], [_NAMES[1], _NAMES[3]])
        for (name, weight), expected in zip(seen["normal"], (1, 3)):
            self.assertTrue(bool((weight == expected).all()), name)
            self.assertFalse(getattr(weight, RUNAI_STREAMER_TENSOR_ATTR, False))

    def test_materialize_is_identity_for_ordinary_tensors(self):
        t = torch.zeros(2)
        self.assertIs(gpt_oss._materialize_streamed(t), t)


if __name__ == "__main__":
    unittest.main()
