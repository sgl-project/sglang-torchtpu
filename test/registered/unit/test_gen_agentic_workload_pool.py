"""Unit test for the agentic trace generator's content-token pool (no TPU).

The pool is the vocabulary the synthetic round bodies are sampled from. A
control token in it lands in message content, where it can corrupt a chat
template turn boundary or be rejected outright (GPT-OSS's server rejects
content containing harmony tags like <|channel|> with a 400), so every
special AND added token must be excluded.
"""

import importlib.util
import os
import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_GEN = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 3)),
    "benchmark",
    "tpu",
    "recipe",
    "qwen3_coder_480b",
    "nightly",
    "gen_agentic_workload.py",
)


def _load_generator():
    spec = importlib.util.spec_from_file_location("_gen_agentic_workload", _GEN)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeTokenizer:
    """GPT-OSS-shaped: only some added tokens are flagged special."""

    def __init__(self, vocab, special_ids, added):
        self._vocab = vocab
        self.all_special_ids = special_ids
        self._added = added

    def get_vocab(self):
        return dict(self._vocab)

    def get_added_vocab(self):
        return dict(self._added)


class TestContentTokenPool(CustomTestCase):
    def test_excludes_special_and_added_tokens(self):
        gen = _load_generator()
        vocab = {f"t{i}": i for i in range(10)}
        vocab.update({"<|endoftext|>": 100, "<|channel|>": 101, "<|message|>": 102})
        tok = _FakeTokenizer(
            vocab,
            special_ids=[100],
            added={"<|endoftext|>": 100, "<|channel|>": 101, "<|message|>": 102},
        )
        self.assertEqual(gen._content_token_pool(tok), list(range(10)))

    def test_tokenizer_without_added_vocab(self):
        class _Minimal:
            all_special_ids = [2]

            def get_vocab(self):
                return {"a": 3, "b": 1, "<s>": 2}

        self.assertEqual(_load_generator()._content_token_pool(_Minimal()), [1, 3])


if __name__ == "__main__":
    unittest.main()
