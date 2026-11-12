"""Unit tests for srt/configs/model_config_parser_registry.py"""

import unittest
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from transformers import LlamaConfig, PretrainedConfig

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.configs.model_config_parser_registry import (
    _MODEL_CONFIG_PARSER_REGISTRY,
    ModelConfigParserBase,
    get_model_config_parser,
    model_config_parser_kwargs,
    register_model_config_parser,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _FakeParser(ModelConfigParserBase):
    def parse(self, model, trust_remote_code, revision=None, **kwargs):
        return PretrainedConfig()


class _AnotherFakeParser(ModelConfigParserBase):
    def parse(self, model, trust_remote_code, revision=None, **kwargs):
        return PretrainedConfig()


class TestModelConfigParserRegistry(CustomTestCase):
    def setUp(self):
        self._saved_registry = dict(_MODEL_CONFIG_PARSER_REGISTRY)
        _MODEL_CONFIG_PARSER_REGISTRY.clear()

    def tearDown(self):
        _MODEL_CONFIG_PARSER_REGISTRY.clear()
        _MODEL_CONFIG_PARSER_REGISTRY.update(self._saved_registry)

    def test_register_then_get_roundtrip(self):
        register_model_config_parser("fake")(_FakeParser)
        self.assertIsInstance(get_model_config_parser("fake"), _FakeParser)

    def test_default_parsers_need_no_server_options(self):
        register_model_config_parser("fake")(_FakeParser)
        self.assertEqual(model_config_parser_kwargs("auto", object()), {})
        self.assertEqual(model_config_parser_kwargs("fake", object()), {})

    def test_parser_options_reach_model_config_without_mutating_the_caller(self):
        received = {}

        class OptionsParser(ModelConfigParserBase):
            def server_args_kwargs(self, server_args):
                return {"hidden_size": server_args.hidden_size}

            def parse(self, model, trust_remote_code, revision=None, **kwargs):
                received.update(kwargs)
                return LlamaConfig(
                    architectures=["LlamaForCausalLM"],
                    hidden_size=kwargs["hidden_size"],
                    intermediate_size=32,
                    num_attention_heads=2,
                    num_hidden_layers=2,
                    vocab_size=128,
                )

        register_model_config_parser("options")(OptionsParser)
        options = model_config_parser_kwargs("options", SimpleNamespace(hidden_size=16))
        with TemporaryDirectory() as checkpoint:
            config = ModelConfig(
                model_path=checkpoint,
                model_config_parser="options",
                model_config_parser_kwargs=options,
                override_config_file="custom.json",
            )
        self.assertEqual(config.hidden_size, 16)
        self.assertEqual(
            received, {"hidden_size": 16, "_configuration_file": "custom.json"}
        )
        self.assertEqual(options, {"hidden_size": 16})

    def test_register_rejects_non_subclass(self):
        class NotAParser:
            pass

        with self.assertRaises(ValueError) as ctx:
            register_model_config_parser("bad")(NotAParser)
        self.assertIn("ModelConfigParserBase", str(ctx.exception))

    def test_unknown_name_raises_with_registered_list(self):
        register_model_config_parser("fake")(_FakeParser)
        register_model_config_parser("another")(_AnotherFakeParser)
        with self.assertRaises(ValueError) as ctx:
            get_model_config_parser("does-not-exist")
        msg = str(ctx.exception)
        self.assertIn("does-not-exist", msg)
        self.assertIn("another", msg)
        self.assertIn("fake", msg)


if __name__ == "__main__":
    unittest.main()
