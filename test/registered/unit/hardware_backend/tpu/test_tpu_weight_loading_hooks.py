"""The TPU platform's model-loader hooks keep the loader's former TPU behavior."""

import sys
import types
import unittest
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch

import sglang.srt.model_loader.loader as loader_mod
from sglang.srt.model_loader import utils as loader_utils
from sglang.srt.model_loader.loader import DefaultModelLoader
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.platform import TpuOOTPlatform
from sglang_tpu.tpu_platform import TpuSRTPlatform

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

_PLATFORMS = (TpuSRTPlatform, TpuOOTPlatform)
# Imports torch_tpu at module load, so the CPU test stands in a recorder.
_FUSED_MOE_DECODE = "sglang_tpu.moe.pallas.fused_moe_decode.fused_moe_decode_op"


class _QuantMethod:
    def __init__(self, events):
        self.events = events

    def process_weights_after_loading(self, module):
        self.events.append(("process", module.tag))


def _quantized_model(events):
    model = torch.nn.Module()
    for tag in ("a", "b"):
        layer = torch.nn.Module()
        layer.tag = tag
        layer.quant_method = _QuantMethod(events)
        model.add_module(tag, layer)
    return model


@contextmanager
def _tpu_loader(platform_cls, events):
    fused_moe_decode = types.ModuleType(_FUSED_MOE_DECODE)
    fused_moe_decode.prepare_model_for_fused_moe_decode = lambda model: events.append(
        ("prepare_fused_moe_decode", model)
    )
    with (
        patch.dict(sys.modules, {_FUSED_MOE_DECODE: fused_moe_decode}),
        patch.object(
            platform_cls, "synchronize", lambda self: events.append(("synchronize",))
        ),
        patch.object(loader_mod, "current_platform", platform_cls()),
        patch.object(loader_mod, "device_loading_context", lambda *_: nullcontext()),
        patch.object(
            loader_mod,
            "finalize_gfx115_weight_padding",
            side_effect=lambda _: events.append(("finalize",)),
        ),
    ):
        yield


class TestTpuWeightLoadingHooks(CustomTestCase):
    def test_postprocess_drains_every_module_and_prepares_tpu_targets_once(self):
        """Synchronize after each module's post-processing on any target, then
        prepare fused_moe_decode once, only when the target is the TPU."""
        for platform_cls in _PLATFORMS:
            for device_type in ("tpu", "cpu"):
                with self.subTest(platform=platform_cls.__name__, device=device_type):
                    events = []
                    model = _quantized_model(events)
                    with _tpu_loader(platform_cls, events):
                        DefaultModelLoader.postprocess_weights(
                            model, SimpleNamespace(type=device_type)
                        )
                    prepared = (
                        [("prepare_fused_moe_decode", model)]
                        if device_type == "tpu"
                        else []
                    )
                    self.assertEqual(
                        events,
                        [("process", "a"), ("synchronize",)]
                        + [("process", "b"), ("synchronize",)]
                        + prepared
                        + [("finalize",)],
                    )

    def test_host_tensors_load_synchronously(self):
        """Threaded host->device parameter copies race torch_tpu's enqueue path."""
        for platform_cls in _PLATFORMS:
            with (
                self.subTest(platform=platform_cls.__name__),
                patch.object(loader_utils, "current_platform", platform_cls()),
            ):
                self.assertFalse(loader_utils.should_async_load(torch.zeros(1)))


if __name__ == "__main__":
    unittest.main()
