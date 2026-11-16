"""Unit tests for srt/model_loader/weight_utils.py shard-index consistency."""

import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import safetensors.torch
import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.environ import envs
from sglang.srt.model_loader.loader import DefaultModelLoader
from sglang.srt.model_loader.weight_utils import (
    filter_duplicate_safetensors_files,
    maybe_add_mtp_safetensors,
    prepare_mtp_safetensors,
)
from sglang.srt.runtime_context import get_context
from sglang.srt.utils import runai_utils

register_cpu_ci(est_time=11, suite="base-a-test-cpu")

INDEX_NAME = "model.safetensors.index.json"


def _write_index(folder, weight_map):
    with open(os.path.join(folder, INDEX_NAME), "w") as f:
        json.dump({"weight_map": weight_map}, f)


def _touch(folder, name):
    path = os.path.join(folder, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w").close()
    return path


class TestFilterDuplicateSafetensorsFiles(CustomTestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.folder = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def test_missing_shard_raises(self):
        # Index lists two shards, only one on disk (interrupted download).
        _write_index(
            self.folder,
            {
                "w1": "model-00001-of-00002.safetensors",
                "w2": "model-00002-of-00002.safetensors",
            },
        )
        present = _touch(self.folder, "model-00001-of-00002.safetensors")

        with self.assertRaises(RuntimeError) as cm:
            filter_duplicate_safetensors_files(
                hf_weights_files=[present],
                hf_folder=self.folder,
                index_file=INDEX_NAME,
            )
        self.assertIn("model-00002-of-00002.safetensors", str(cm.exception))

    def test_complete_checkpoint_filters_non_indexed(self):
        # All indexed shards present; a non-indexed duplicate is still filtered out.
        _write_index(
            self.folder,
            {
                "w1": "model-00001-of-00002.safetensors",
                "w2": "model-00002-of-00002.safetensors",
            },
        )
        shard1 = _touch(self.folder, "model-00001-of-00002.safetensors")
        shard2 = _touch(self.folder, "model-00002-of-00002.safetensors")
        extra = _touch(self.folder, "consolidated.safetensors")

        result = filter_duplicate_safetensors_files(
            hf_weights_files=[shard1, shard2, extra],
            hf_folder=self.folder,
            index_file=INDEX_NAME,
        )
        self.assertEqual(sorted(result), sorted([shard1, shard2]))

    def test_missing_shard_outside_allow_patterns_is_ignored(self):
        # Cosmos3-style checkpoints use one root index for multiple subfolder
        # weight sources. Loading the transformer source should not require the
        # vision encoder shard to already be present; the secondary source
        # downloads and loads it separately.
        _write_index(
            self.folder,
            {
                "llm": "transformer/diffusion_pytorch_model.safetensors",
                "vit": "vision_encoder/model.safetensors",
            },
        )
        shard = "transformer/diffusion_pytorch_model.safetensors"
        _touch(self.folder, shard)

        for folder in (self.folder, "s3://bucket/model"):
            with self.subTest(folder=folder):
                transformer = os.path.join(folder, shard)
                result = filter_duplicate_safetensors_files(
                    hf_weights_files=[transformer],
                    hf_folder=folder,
                    index_file=os.path.join(self.folder, INDEX_NAME),
                    allow_patterns=["transformer/*.safetensors"],
                )
                self.assertEqual(result, [transformer])

    def test_missing_shard_inside_allow_patterns_raises(self):
        _write_index(
            self.folder,
            {
                "llm1": "transformer/model-00001-of-00002.safetensors",
                "llm2": "transformer/model-00002-of-00002.safetensors",
                "vit": "vision_encoder/model.safetensors",
            },
        )
        shard = "transformer/model-00001-of-00002.safetensors"
        _touch(self.folder, shard)

        for folder in (self.folder, "s3://bucket/model"):
            with (
                self.subTest(folder=folder),
                self.assertRaisesRegex(
                    RuntimeError, r"model-00002-of-00002\.safetensors"
                ),
            ):
                filter_duplicate_safetensors_files(
                    hf_weights_files=[os.path.join(folder, shard)],
                    hf_folder=folder,
                    index_file=os.path.join(self.folder, INDEX_NAME),
                    allow_patterns=["transformer/*.safetensors"],
                )

    def test_local_index_allows_subset_of_existing_shards(self):
        _write_index(
            self.folder,
            {"w1": "first.safetensors", "w2": "second.safetensors"},
        )
        first = _touch(self.folder, "first.safetensors")
        _touch(self.folder, "second.safetensors")

        result = filter_duplicate_safetensors_files([first], self.folder, INDEX_NAME)

        self.assertEqual(result, [first])

    def test_single_file_model_no_index_returns_unchanged(self):
        # No index on disk (single-file / dummy / object-storage): early return.
        single = _touch(self.folder, "model.safetensors")

        result = filter_duplicate_safetensors_files(
            hf_weights_files=[single],
            hf_folder=self.folder,
            index_file=INDEX_NAME,
        )
        self.assertEqual(result, [single])


class TestMaybeAddMtpSafetensors(CustomTestCase):
    def test_remote_mtp_guards(self):
        folder = "s3://bucket/model"
        model = f"{folder}/model.safetensors"
        mtp = f"{folder}/mtp.safetensors"
        cases = (
            ("Glm4MoeForCausalLM", 1, [model, mtp], [model, mtp], False),
            ("Glm4MoeForCausalLM", 1, [model], [model], True),
            ("Glm4MoeForCausalLM", 0, [model], [model, mtp], False),
            ("LlamaForCausalLM", 1, [model], [model, mtp], False),
        )
        for arch, nextn, selected, available, may_list in cases:
            with (
                self.subTest(arch=arch, nextn=nextn, selected=selected),
                patch.object(
                    runai_utils, "list_safetensors", return_value=available
                ) as listing,
            ):
                config = SimpleNamespace(
                    architectures=[arch], num_nextn_predict_layers=nextn
                )
                self.assertEqual(
                    maybe_add_mtp_safetensors(selected, folder, INDEX_NAME, config),
                    selected,
                )
                if not may_list:
                    listing.assert_not_called()

    def test_local_unindexed_mtp(self):
        with tempfile.TemporaryDirectory() as folder:
            model = _touch(folder, "model.safetensors")
            mtp = _touch(folder, "mtp.safetensors")
            config = SimpleNamespace(
                architectures=["Glm4MoeLiteForCausalLMNextN"],
                num_nextn_predict_layers=1,
            )
            with patch.object(runai_utils, "list_safetensors") as listing:
                self.assertEqual(
                    maybe_add_mtp_safetensors([model], folder, INDEX_NAME, config),
                    [model, mtp],
                )
                listing.assert_not_called()


class TestNextNShardSelection(CustomTestCase):
    def test_loader_skips_target_payload_and_keeps_original_mtp_bytes(self):
        """Draft startup must not open target-only shards of a full checkpoint."""
        with tempfile.TemporaryDirectory() as folder:
            target = _touch(folder, "target.safetensors")
            # Index discovery may stat this shard, but no draft payload reader
            # may open it. A regression to reading every shard fails decoding it.
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
            config = SimpleNamespace(
                architectures=["GlmMoeDsaForCausalLMNextN"],
                num_hidden_layers=78,
                num_nextn_predict_layers=1,
            )
            model_config = SimpleNamespace(
                model_path=folder, revision=None, hf_config=config
            )
            with (
                envs.SGLANG_ENABLE_MTP_SHARD_FILTER.override(True),
                get_context().override_server_args(
                    device="cpu",
                    weight_loader_disable_mmap=True,
                    weight_loader_prefetch_checkpoints=False,
                ),
            ):
                for multithread in (False, True):
                    loader = DefaultModelLoader(
                        LoadConfig(
                            load_format="safetensors",
                            model_loader_extra_config={
                                "enable_multithread_load": multithread
                            },
                        )
                    )
                    source = DefaultModelLoader.Source.init_new(
                        model_config, torch.nn.Module()
                    )
                    resolved = loader.resolve_model_weights(
                        model_config, torch.nn.Module()
                    )[0]
                    self.assertEqual(set(resolved.weight_files), {mtp, mixed})
                    for resolution in (None, resolved):
                        with self.subTest(
                            multithread=multithread, pre_resolved=resolution is not None
                        ):
                            loaded = dict(
                                loader._get_weights_iterator(
                                    source, resolved_source=resolution
                                )
                            )
                            self.assertEqual(set(loaded), set(tensors) | set(shared))
                            for name, expected in {**tensors, **shared}.items():
                                self.assertTrue(
                                    torch.equal(
                                        loaded[name].view(torch.uint8),
                                        expected.view(torch.uint8),
                                    ),
                                    name,
                                )
                model = torch.nn.Module()
                model.secondary_weights = (
                    DefaultModelLoader.Source(
                        model_or_path=folder,
                        revision=None,
                        prefix="aux.",
                        model_config=model_config,
                    ),
                )
                sources = loader.resolve_model_weights(model_config, model)
                self.assertEqual(set(sources[1].weight_files), {target, mtp, mixed})

    def test_layout_guards_and_unindexed_sidecars(self):
        """Targets/unknown layouts retain all files; draft sidecars survive filtering."""
        with (
            envs.SGLANG_ENABLE_MTP_SHARD_FILTER.override(True),
            tempfile.TemporaryDirectory() as folder,
        ):
            target = _touch(folder, "target.safetensors")
            draft = _touch(folder, "draft.safetensors")
            sidecar = _touch(folder, "mtp.safetensors")
            files = [target, sidecar, draft]
            for layers, prefix in ((78, 78), (1, 0)):
                _write_index(
                    folder,
                    {
                        "model.embed_tokens.weight": "target.safetensors",
                        f"model.layers.{prefix}.eh_proj.weight": "draft.safetensors",
                    },
                )
                for arch in (
                    "DeepseekV3ForCausalLMNextN",
                    "GlmMoeDsaForCausalLMNextN",
                    "Glm4MoeForCausalLMNextN",
                    "Glm4MoeLiteForCausalLMNextN",
                    "GlmMoeDsaForCausalLM",
                    "DeepseekV4ForCausalLMNextN",
                ):
                    config = SimpleNamespace(
                        architectures=[arch],
                        text_config=SimpleNamespace(
                            num_hidden_layers=layers, num_nextn_predict_layers=1
                        ),
                    )
                    with self.subTest(arch=arch, layers=layers):
                        selected = prepare_mtp_safetensors(
                            files, folder, INDEX_NAME, config
                        )
                        self.assertEqual(
                            selected,
                            files
                            if arch
                            in ("GlmMoeDsaForCausalLM", "DeepseekV4ForCausalLMNextN")
                            else [sidecar, draft],
                        )
            # An index without the native NextN prefix must not silently drop
            # embeddings or other weights from a standalone/custom draft layout.
            config.architectures = ["GlmMoeDsaForCausalLMNextN"]
            _write_index(folder, {"model.weight": "target.safetensors"})
            self.assertEqual(
                prepare_mtp_safetensors(files, folder, INDEX_NAME, config), files
            )
            os.unlink(os.path.join(folder, INDEX_NAME))
            self.assertEqual(
                prepare_mtp_safetensors(files, folder, INDEX_NAME, config), files
            )

    def test_disabled_filter_preserves_partial_index_weights(self):
        """Default/disabled filtering must not hide an unindexed MTP tensor."""
        with (
            envs.SGLANG_ENABLE_MTP_SHARD_FILTER.override(False),
            tempfile.TemporaryDirectory() as folder,
        ):
            target = os.path.join(folder, "target.safetensors")
            draft = os.path.join(folder, "draft.safetensors")
            hidden_norm = "model.layers.78.hnorm.weight"
            tensors = {
                "model.layers.0.weight": torch.ones(2),
                hidden_norm: torch.ones(2),
            }
            safetensors.torch.save_file(tensors, target)
            safetensors.torch.save_file(
                {"model.layers.78.eh_proj.weight": torch.ones(2)}, draft
            )
            # File discovery still finds both shards, but this incomplete tensor
            # index hides a draft norm in the apparently target-only shard.
            _write_index(
                folder,
                {
                    "model.layers.0.weight": "target.safetensors",
                    "model.layers.78.eh_proj.weight": "draft.safetensors",
                },
            )
            for arch in (
                "GlmMoeDsaForCausalLMNextN",
                "DeepseekV3ForCausalLMNextN",
                "Glm4MoeForCausalLMNextN",
                "Glm4MoeLiteForCausalLMNextN",
            ):
                config = SimpleNamespace(
                    architectures=[arch],
                    num_hidden_layers=78,
                    num_nextn_predict_layers=1,
                )
                for quant in (
                    None,
                    {"quant_method": "fp8", "weight_block_size": [128, 128]},
                    {"quant_method": "quark"},
                ):
                    config.quantization_config = quant
                    self.assertEqual(
                        prepare_mtp_safetensors(
                            [target, draft], folder, INDEX_NAME, config
                        ),
                        [target, draft],
                    )
            model_config = SimpleNamespace(
                model_path=folder,
                revision=None,
                hf_config=SimpleNamespace(
                    architectures=["GlmMoeDsaForCausalLMNextN"],
                    num_hidden_layers=78,
                    num_nextn_predict_layers=1,
                ),
            )
            with get_context().override_server_args(
                device="cpu",
                weight_loader_disable_mmap=True,
                weight_loader_prefetch_checkpoints=False,
            ):
                loader = DefaultModelLoader(
                    LoadConfig(
                        load_format="safetensors",
                        model_loader_extra_config={"enable_multithread_load": False},
                    )
                )
                source = DefaultModelLoader.Source.init_new(
                    model_config, torch.nn.Module()
                )
                resolved = loader.resolve_model_weights(
                    model_config, torch.nn.Module()
                )[0]
                self.assertEqual(set(resolved.weight_files), {target, draft})
                for resolution in (None, resolved):
                    loaded = dict(
                        loader._get_weights_iterator(source, resolved_source=resolution)
                    )
                    self.assertIn(hidden_norm, loaded)
                    self.assertTrue(
                        torch.equal(loaded[hidden_norm], tensors[hidden_norm])
                    )


if __name__ == "__main__":
    unittest.main()
