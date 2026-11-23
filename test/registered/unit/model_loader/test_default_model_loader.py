import os
import tempfile
import unittest
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import safetensors.torch
import torch

import sglang.srt.model_loader.loader as loader_mod
from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.model_loader import utils as loader_utils
from sglang.srt.model_loader import weight_utils
from sglang.srt.model_loader.loader import DefaultModelLoader, ModelOptModelLoader
from sglang.srt.platforms.device_mixin import DeviceMixin
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=13, suite="base-a-test-cpu")


class TestDefaultModelLoader(CustomTestCase):
    def test_load_weights_only_precedes_postprocessing(self):
        events = []
        model = Mock()
        model.quant_config = None
        module = Mock()
        module.quant_method.process_weights_after_loading.side_effect = lambda _: (
            events.append("postprocess")
        )
        model.load_weights.side_effect = lambda _: events.append("load")
        model.named_modules.return_value = [("layer", module)]

        with (
            patch.object(loader_mod, "is_cuda_alike", return_value=False),
            patch.object(
                loader_mod,
                "device_loading_context",
                side_effect=lambda *_: nullcontext(),
            ),
        ):
            DefaultModelLoader.load_weights_and_postprocess(
                model,
                iter(()),
                torch.device("cpu"),
            )

        self.assertEqual(events, ["load", "postprocess"])

    def test_reload_requires_explicit_initial_load_marker(self):
        self.assertFalse(loader_mod.QuantizedRLModelLoader.is_reload_scenario(Mock()))
        model = torch.nn.Module()
        model.original_weights_rebuild_keys = set()
        model.recorded_loader = {}
        for marker in (None, False, 1, Mock()):
            with self.subTest(marker=marker):
                model.flash_rl_initial_load_complete = marker
                self.assertFalse(
                    loader_mod.QuantizedRLModelLoader.is_reload_scenario(model)
                )
        model.flash_rl_initial_load_complete = True
        self.assertTrue(loader_mod.QuantizedRLModelLoader.is_reload_scenario(model))

    def test_boot_paths_preserve_load_order_and_custom_override(self):
        class CustomModelLoader(DefaultModelLoader):
            def load_weights_and_postprocess(self, model, weights, target_device):
                events.append("override")
                DefaultModelLoader.load_weights_and_postprocess(
                    model, weights, target_device
                )

        for loader_class in (DefaultModelLoader, CustomModelLoader):
            with self.subTest(loader=loader_class.__name__):
                events = []
                loader = object.__new__(loader_class)
                loader.load_config = object()
                model = Mock(quant_config=None)
                model.eval.return_value = model
                model.load_weights.side_effect = lambda weights: events.append(
                    ("load", list(weights))
                )
                module = Mock()
                module.quant_method.process_weights_after_loading.side_effect = (
                    lambda _: events.append("postprocess")
                )
                model.named_modules.return_value = [("layer", module)]
                model_config = SimpleNamespace(modelopt_quant=None, dtype=torch.float32)
                resolved_source = SimpleNamespace(source=object())

                with (
                    patch.object(loader_mod, "is_cuda_alike", return_value=False),
                    patch.object(
                        loader_mod, "_get_quantization_config", return_value=None
                    ),
                    patch.object(loader_mod, "_initialize_model", return_value=model),
                    patch.object(
                        loader_mod,
                        "device_loading_context",
                        side_effect=lambda *_: nullcontext(),
                    ),
                    patch.object(
                        loader, "_get_all_weights", return_value=iter([("direct", 1)])
                    ),
                    patch.object(
                        loader,
                        "_get_weights_iterator",
                        return_value=iter([("deferred", 2)]),
                    ) as get_weights,
                ):
                    self.assertIs(
                        loader.load_model(
                            model_config=model_config,
                            device_config=SimpleNamespace(device="cpu"),
                        ),
                        model,
                    )
                    loader.commit_model_weights(
                        model=model,
                        model_config=model_config,
                        resolved_sources=(resolved_source,),
                        target_device=torch.device("cpu"),
                        startup_prefetch_active=True,
                    )

                expected = []
                for weights in ([("direct", 1)], [("deferred", 2)]):
                    if loader_class is CustomModelLoader:
                        expected.append("override")
                    expected.extend([("load", weights), "postprocess"])
                self.assertEqual(events, expected)
                get_weights.assert_called_once_with(
                    resolved_source.source,
                    resolved_source=resolved_source,
                    startup_prefetch_started=True,
                    startup_prefetch_active=True,
                )


class TestModelOptLoaderSelection(CustomTestCase):
    def test_every_modelopt_quantization_selects_the_modelopt_loader(self):
        """A pre-quantized checkpoint under any ModelOpt quantization value must
        load through ModelOptModelLoader, not fall through to the default loader."""
        for quantization in (
            "modelopt",
            "modelopt_fp8",
            "modelopt_fp4",
            "modelopt_mixed",
        ):
            with self.subTest(quantization=quantization):
                model_config = SimpleNamespace(
                    quantization=quantization, _is_already_quantized=lambda: True
                )
                self.assertIsInstance(
                    loader_mod.get_model_loader(LoadConfig(), model_config),
                    ModelOptModelLoader,
                )


class _RecordingPlatform(DeviceMixin):
    def __init__(self, events, *, expert_range=None, reader=None, async_load=True):
        self.events = events
        self.expert_range = expert_range
        self.reader = reader
        self.async_load = async_load

    def after_module_weights_processed(self, module):
        self.events.append(("module_hook", module.tag))

    def after_model_weights_processed(self, model, target_device):
        self.events.append(("model_hook", target_device))

    def get_local_expert_range(self, model):
        self.events.append(("range", model))
        return self.expert_range

    def create_local_expert_reader(
        self, hf_weights_files, local_expert_range, *, disable_mmap
    ):
        self.events.append(
            ("create_reader", tuple(hf_weights_files), local_expert_range, disable_mmap)
        )
        return self.reader

    def supports_async_weight_loading(self):
        return self.async_load


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
    model.load_weights = lambda weights: events.append(("load",))
    return model


@contextmanager
def _recording_loader(events, platform):
    @contextmanager
    def device_loading_context(module, target_device):
        events.append(("enter", module.tag))
        yield module
        events.append(("exit", module.tag))

    with (
        patch.object(loader_mod, "current_platform", platform),
        patch.object(loader_mod, "device_loading_context", device_loading_context),
        patch.object(
            loader_mod,
            "finalize_gfx115_weight_padding",
            side_effect=lambda _: events.append(("finalize",)),
        ),
        patch.object(loader_mod, "is_cuda_alike", return_value=False),
    ):
        yield


def _processed(tag):
    return [("enter", tag), ("process", tag), ("exit", tag), ("module_hook", tag)]


class TestPlatformWeightLoadingHooks(CustomTestCase):
    """Each hook fires once, where the device branch it replaced ran."""

    _TARGET = torch.device("meta")

    def test_postprocess_hooks_follow_each_module_and_then_the_model(self):
        events = []
        with _recording_loader(events, _RecordingPlatform(events)):
            DefaultModelLoader.load_weights_and_postprocess(
                _quantized_model(events), iter(()), self._TARGET
            )
        self.assertEqual(
            events,
            [("load",)]
            + _processed("a")
            + _processed("b")
            + [("model_hook", self._TARGET), ("finalize",)],
        )

    def test_quantized_rl_reload_runs_no_hook(self):
        events = []
        with (
            _recording_loader(events, _RecordingPlatform(events)),
            patch.object(
                loader_mod.QuantizedRLModelLoader,
                "is_reload_scenario",
                return_value=True,
            ),
        ):
            DefaultModelLoader.postprocess_weights(
                _quantized_model(events), self._TARGET
            )
        self.assertEqual(events, [])

    def test_cpu_quantization_load_runs_only_the_module_hook(self):
        events = []
        model = _quantized_model(events)
        loader = SimpleNamespace(load_config=None, _get_all_weights=lambda *_: ())
        with (
            _recording_loader(events, _RecordingPlatform(events)),
            patch.object(loader_mod, "_get_quantization_config", return_value=None),
            patch.object(loader_mod, "_initialize_model", return_value=model),
        ):
            loader_mod.load_model_with_cpu_quantization(
                loader,
                model_config=SimpleNamespace(dtype=torch.float32),
                device_config=SimpleNamespace(device="cpu"),
            )
        self.assertEqual(events, [("load",)] + _processed("a") + _processed("b"))

    def test_source_takes_the_expert_range_from_the_platform(self):
        model_config = SimpleNamespace(model_path="/dummy", revision=None)
        model = torch.nn.Module()
        events = []
        platform = _RecordingPlatform(events, expert_range=(2, 4))
        with patch.object(loader_mod, "current_platform", platform):
            source = DefaultModelLoader.Source.init_new(model_config, model)
        self.assertEqual(source.local_expert_range, (2, 4))
        self.assertEqual(events, [("range", model)])

        with patch.object(loader_mod, "current_platform", DeviceMixin()):
            source = DefaultModelLoader.Source.init_new(model_config, model)
        self.assertIsNone(source.local_expert_range)

    def test_should_async_load_asks_the_platform_for_host_tensors(self):
        self.assertTrue(DeviceMixin().supports_async_weight_loading())
        for async_load in (True, False):
            platform = _RecordingPlatform([], async_load=async_load)
            with (
                self.subTest(async_load=async_load),
                patch.object(loader_utils, "current_platform", platform),
            ):
                self.assertIs(
                    loader_utils.should_async_load(torch.zeros(1)), async_load
                )


class _Reader:
    def __init__(self, events, handled):
        self.events = events
        self.handled = handled

    def handles(self, st_file):
        return st_file == self.handled

    def read(self, st_file):
        self.events.append(("read", os.path.basename(st_file)))
        return {"from_reader": torch.zeros(1)}

    def verify(self):
        self.events.append(("verify",))


class TestPlatformLocalExpertReader(CustomTestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.files = []
        for i in range(2):
            path = os.path.join(tmp.name, f"shard{i}.safetensors")
            safetensors.torch.save_file({f"w{i}": torch.full((2,), float(i))}, path)
            self.files.append(path)

    def _load(self, platform, iterator, expert_range):
        if iterator == "buffered":
            it = weight_utils.buffered_multi_thread_safetensors_weights_iterator(
                self.files,
                max_workers=2,
                disable_mmap=True,
                local_expert_range=expert_range,
            )
        else:
            it = weight_utils.safetensors_weights_iterator(
                self.files, disable_mmap=True, local_expert_range=expert_range
            )
        with patch.object(weight_utils, "current_platform", platform):
            return set(dict(it))

    def test_platform_reader_serves_the_shards_it_claims(self):
        for iterator in ("single", "buffered"):
            with self.subTest(iterator=iterator):
                events = []
                reader = _Reader(events, handled=self.files[0])
                platform = _RecordingPlatform(events, reader=reader)
                loaded = self._load(platform, iterator, (2, 4))
                self.assertEqual(loaded, {"from_reader", "w1"})
                self.assertEqual(
                    events,
                    [
                        ("create_reader", tuple(self.files), (2, 4), True),
                        ("read", "shard0.safetensors"),
                        ("verify",),
                    ],
                )

    def test_whole_shards_without_a_range_or_a_platform_reader(self):
        for iterator in ("single", "buffered"):
            with self.subTest(iterator=iterator):
                events = []
                platform = _RecordingPlatform(events, reader=_Reader(events, None))
                self.assertEqual(self._load(platform, iterator, None), {"w0", "w1"})
                self.assertEqual(events, [])
                self.assertEqual(
                    self._load(DeviceMixin(), iterator, (2, 4)), {"w0", "w1"}
                )


if __name__ == "__main__":
    unittest.main()
