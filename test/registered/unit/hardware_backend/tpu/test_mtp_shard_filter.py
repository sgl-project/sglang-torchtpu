"""Plugin-owned GLM NextN reads: original bytes, source guards and opt-in."""

import json
import os
import tempfile
import unittest
from types import SimpleNamespace

import safetensors.torch
import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.model_loader import loader as loader_module
from sglang.srt.model_loader.loader import DefaultModelLoader
from sglang.srt.models.deepseek_nextn import DeepseekV3ForCausalLMNextN
from sglang.srt.models.glm4_moe import (
    GlmMoeDsaForCausalLM,
    GlmMoeDsaForCausalLMNextN,
)
from sglang.srt.plugins.hook_registry import _wrap_fn
from sglang.srt.runtime_context import get_context
from sglang_tpu.envs import envs
from sglang_tpu.hooks import HOOKS

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

_INDEX_NAME = "model.safetensors.index.json"


def _write_index(folder, weight_map):
    with open(os.path.join(folder, _INDEX_NAME), "w") as f:
        json.dump({"weight_map": weight_map}, f)


def _model_config(folder, layers=78):
    return SimpleNamespace(
        model_path=folder,
        revision=None,
        quantization=None,
        hf_config=SimpleNamespace(
            architectures=[GlmMoeDsaForCausalLMNextN.__name__],
            num_hidden_layers=layers,
            num_nextn_predict_layers=1,
        ),
    )


def _model(config, cls=GlmMoeDsaForCausalLMNextN):
    # No accelerator allocations: exercise the real class's NextN loading rule.
    model = cls.__new__(cls)
    torch.nn.Module.__init__(model)
    model.config = config.hf_config
    return model


def _loader(
    *,
    model_config=None,
    multithread=False,
    load_format="safetensors",
    draft_model_idx=None,
):
    # Use the registered callbacks and actual hook wrapper, including Run:AI's
    # factory hook from main; do not mutate global plugin registration in tests.
    factory = loader_module.get_model_loader
    for target, hook, hook_type in HOOKS:
        if target == "sglang.srt.model_loader.loader.get_model_loader":
            factory = _wrap_fn(factory, hook, hook_type)
    return factory(
        LoadConfig(
            load_format=load_format,
            draft_model_idx=draft_model_idx,
            model_loader_extra_config=(
                {}
                if load_format == "runai_streamer"
                else {"enable_multithread_load": multithread}
            ),
        ),
        model_config,
    )


class TestMtpShardFilter(CustomTestCase):
    def test_skips_target_payload_and_keeps_original_mtp_bytes(self):
        """Both ordinary and pre-resolved startup must skip target-only payloads."""
        with (
            envs.SGLANG_ENABLE_MTP_SHARD_FILTER.override(True),
            tempfile.TemporaryDirectory() as folder,
            get_context().override_server_args(
                device="cpu",
                weight_loader_disable_mmap=True,
                weight_loader_prefetch_checkpoints=False,
            ),
        ):
            target = os.path.join(folder, "target.safetensors")
            with open(target, "wb") as f:
                f.write(b"target payload must not be read")
            mtp = os.path.join(folder, "draft.safetensors")
            mixed = os.path.join(folder, "mixed.safetensors")
            tensors = {
                "model.layers.78.eh_proj.weight": torch.arange(16)
                .reshape(4, 4)
                .to(torch.float8_e4m3fn),
                "model.layers.78.eh_proj.weight_scale_inv": torch.tensor(
                    [0.1, 0.2, 0.3, 0.4]
                ),
            }
            shared = {
                "model.layers.77.norm.weight": torch.ones(4),
                "model.layers.78.shared_head.norm.weight": torch.arange(4).float(),
            }
            safetensors.torch.save_file(tensors, mtp)
            safetensors.torch.save_file(shared, mixed)
            _write_index(
                folder,
                {
                    "model.layers.0.weight": "target.safetensors",
                    **{name: "draft.safetensors" for name in tensors},
                    **{name: "mixed.safetensors" for name in shared},
                },
            )
            config = _model_config(folder)
            model = _model(config)
            for multithread in (False, True):
                loader = _loader(model_config=config, multithread=multithread)
                # The ordinary load path must initialize the policy itself.
                ordinary = dict(loader._get_all_weights(config, model))
                resolved = loader.resolve_model_weights(config, model)[0]
                self.assertEqual(set(resolved.weight_files), {mtp, mixed})
                pre_resolved = dict(
                    loader._get_weights_iterator(
                        resolved.source,
                        resolved_source=resolved,
                        startup_prefetch_started=True,
                    )
                )
                for loaded in (ordinary, pre_resolved):
                    self.assertEqual(set(loaded), set(tensors) | set(shared))
                    for name, expected in {**tensors, **shared}.items():
                        self.assertTrue(
                            torch.equal(
                                loaded[name].view(torch.uint8),
                                expected.view(torch.uint8),
                            ),
                            name,
                        )
            model.secondary_weights = (
                DefaultModelLoader.Source(
                    model_or_path=folder,
                    revision=None,
                    prefix="aux.",
                    model_config=config,
                ),
            )
            sources = loader.resolve_model_weights(config, model)
            self.assertEqual(set(sources[1].weight_files), {target, mtp, mixed})

    def test_model_layout_and_transport_guards(self):
        """Only the audited model/source filters; sidecars and unknown layouts survive."""

        class UnreviewedSubclass(GlmMoeDsaForCausalLMNextN):
            pass

        with (
            envs.SGLANG_ENABLE_MTP_SHARD_FILTER.override(True),
            tempfile.TemporaryDirectory() as folder,
        ):
            files = []
            for name in ("target.safetensors", "draft.safetensors", "mtp.safetensors"):
                path = os.path.join(folder, name)
                open(path, "w").close()
                files.append(path)
            for layers, prefix in ((78, 78), (1, 0)):
                _write_index(
                    folder,
                    {
                        "model.embed_tokens.weight": "target.safetensors",
                        f"model.layers.{prefix}.eh_proj.weight": "draft.safetensors",
                    },
                )
                config = _model_config(folder, layers)
                for cls in (
                    GlmMoeDsaForCausalLMNextN,
                    GlmMoeDsaForCausalLM,
                    DeepseekV3ForCausalLMNextN,
                    UnreviewedSubclass,
                ):
                    loader = _loader(model_config=config)
                    source = DefaultModelLoader.ResolvedSource(
                        DefaultModelLoader.Source(folder, None, model_config=config),
                        folder,
                        tuple(files),
                        True,
                    )
                    loader._set_primary_model(config, _model(config, cls))
                    expected = files[1:] if cls is GlmMoeDsaForCausalLMNextN else files
                    self.assertEqual(
                        list(loader._filter_source(source).weight_files), expected
                    )
                    # PT/BIN sources never enter the safetensors selection policy.
                    pt = DefaultModelLoader.ResolvedSource(
                        source.source, folder, tuple(files), False
                    )
                    self.assertEqual(
                        loader._filter_source(pt).weight_files, tuple(files)
                    )
            loader._set_primary_model(config, _model(config))
            _write_index(folder, {"model.weight": "target.safetensors"})
            self.assertEqual(loader._filter_source(source).weight_files, tuple(files))
            os.unlink(os.path.join(folder, _INDEX_NAME))
            self.assertEqual(loader._filter_source(source).weight_files, tuple(files))
            self.assertIs(
                type(_loader(model_config=config, draft_model_idx=0)),
                DefaultModelLoader,
            )
            config.hf_config.architectures = [DeepseekV3ForCausalLMNextN.__name__]
            self.assertIs(type(_loader(model_config=config)), DefaultModelLoader)
            # Run:AI keeps main's EP-local loader rather than the draft loader.
            from sglang_tpu.ep_weight_shard import EPLocalRunaiModelStreamerLoader

            self.assertIs(
                type(_loader(load_format="runai_streamer")),
                EPLocalRunaiModelStreamerLoader,
            )

    def test_disabled_filter_preserves_partial_index_weights(self):
        """Default/disabled loading must not hide a draft norm omitted by the index."""
        with (
            envs.SGLANG_ENABLE_MTP_SHARD_FILTER.override(False),
            tempfile.TemporaryDirectory() as folder,
            get_context().override_server_args(
                device="cpu",
                weight_loader_disable_mmap=True,
                weight_loader_prefetch_checkpoints=False,
            ),
        ):
            # Exercise the unset production default; override restores the
            # caller's environment, including an explicitly set value.
            envs.SGLANG_ENABLE_MTP_SHARD_FILTER.clear()
            target = os.path.join(folder, "target.safetensors")
            draft = os.path.join(folder, "draft.safetensors")
            hidden_norm = "model.layers.78.hnorm.weight"
            safetensors.torch.save_file(
                {"model.layers.0.weight": torch.ones(2), hidden_norm: torch.ones(2)},
                target,
            )
            safetensors.torch.save_file(
                {"model.layers.78.eh_proj.weight": torch.ones(2)}, draft
            )
            _write_index(
                folder,
                {
                    "model.layers.0.weight": "target.safetensors",
                    "model.layers.78.eh_proj.weight": "draft.safetensors",
                },
            )
            loader = _loader()
            self.assertIs(type(loader), DefaultModelLoader)
            config = _model_config(folder)
            model = _model(config)
            resolved = loader.resolve_model_weights(config, model)[0]
            self.assertEqual(set(resolved.weight_files), {target, draft})
            for loaded in (
                dict(loader._get_all_weights(config, model)),
                dict(
                    loader._get_weights_iterator(
                        resolved.source, resolved_source=resolved
                    )
                ),
            ):
                self.assertTrue(torch.equal(loaded[hidden_norm], torch.ones(2)))


if __name__ == "__main__":
    unittest.main()
