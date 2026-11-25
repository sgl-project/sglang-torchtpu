"""A Kimi tokenizer's chat segment encodings reused across requests
(sglang_tpu/hooks/kimi_tokenizer.py): the same ids, new text encoded once."""

import unittest
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.envs import envs as tpu_envs
from sglang_tpu.hooks import kimi_tokenizer

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class FakeKimiTokenizer:
    """The remote-code tokenizer's shape: chat segments encoded one by one."""

    def __init__(self):
        self.calls = []

    def _encode_text_piece(self, text, allow_special_tokens=True):
        self.calls.append((text, allow_special_tokens))
        return [ord(c) + (0 if allow_special_tokens else 100_000) for c in text]

    def _encode_chat_segments(self, segments):
        token_ids = []
        for segment in segments:
            token_ids.extend(
                self._encode_text_piece(
                    segment.text, allow_special_tokens=segment.allow_special
                )
            )
        return token_ids


def segments(*texts, allow=True):
    return [SimpleNamespace(text=t, allow_special=allow) for t in texts]


class TestSegmentEncodingCache(CustomTestCase):
    def test_same_ids_and_only_new_segments_encoded(self):
        plain, cached = FakeKimiTokenizer(), FakeKimiTokenizer()
        self.assertTrue(kimi_tokenizer.install(cached, 1000))
        turn1 = segments("<|user|>", "hello there", "<|assistant|>")
        turn2 = turn1 + segments("a reply", "<|user|>", "and more")
        for turn in (turn1, turn2):
            self.assertEqual(
                cached._encode_chat_segments(turn), plain._encode_chat_segments(turn)
            )
        # The second turn encoded only the segments it added.
        self.assertEqual(
            [text for text, _ in cached.calls],
            ["<|user|>", "hello there", "<|assistant|>", "a reply", "and more"],
        )
        cache = cached._encode_text_piece
        self.assertEqual((cache.hits, cache.misses), (4, 5))

    def test_allow_special_is_part_of_the_key(self):
        tok = FakeKimiTokenizer()
        kimi_tokenizer.install(tok, 1000)
        a = tok._encode_chat_segments(segments("<|x|>"))
        b = tok._encode_chat_segments(segments("<|x|>", allow=False))
        self.assertNotEqual(a, b)
        self.assertEqual(b, FakeKimiTokenizer()._encode_text_piece("<|x|>", False))
        self.assertEqual(len(tok.calls), 2)

    def test_each_call_returns_a_new_list(self):
        tok = FakeKimiTokenizer()
        kimi_tokenizer.install(tok, 1000)
        first = tok._encode_text_piece("abc")
        first.append(-1)
        self.assertEqual(tok._encode_text_piece("abc"), [97, 98, 99])
        second = tok._encode_text_piece("abc")
        second[0] = -1
        self.assertEqual(tok._encode_text_piece("abc"), [97, 98, 99])

    def test_least_recently_used_out_past_the_budget(self):
        tok = FakeKimiTokenizer()
        kimi_tokenizer.install(tok, 10)
        tok._encode_text_piece("aaaa")
        tok._encode_text_piece("bbbb")
        tok._encode_text_piece("aaaa")  # a is now the most recently used
        tok._encode_text_piece("cccc")  # 12 tokens: b goes
        tok.calls.clear()
        tok._encode_text_piece("aaaa")
        tok._encode_text_piece("cccc")
        self.assertEqual(tok.calls, [])
        tok._encode_text_piece("bbbb")
        self.assertEqual(tok.calls, [("bbbb", True)])
        # A text longer than the whole budget is encoded but never kept.
        tok.calls.clear()
        long = "z" * 11
        self.assertEqual(tok._encode_text_piece(long), [ord("z")] * 11)
        tok._encode_text_piece(long)
        self.assertEqual(tok.calls, [(long, True), (long, True)])
        self.assertLessEqual(tok._encode_text_piece._tokens, 10)

    def test_install_needs_the_method_and_runs_once(self):
        self.assertFalse(kimi_tokenizer.install(SimpleNamespace(), 10))
        tok = FakeKimiTokenizer()
        self.assertTrue(kimi_tokenizer.install(tok, 10))
        cache = tok._encode_text_piece
        self.assertFalse(kimi_tokenizer.install(tok, 10))
        self.assertIs(tok._encode_text_piece, cache)
        with self.assertRaises(ValueError):
            kimi_tokenizer.SegmentEncodingCache(lambda *a, **k: [], 0)

    def test_hook_follows_the_env(self):
        manager = SimpleNamespace(tokenizer=FakeKimiTokenizer())
        with tpu_envs.SGLANG_TPU_KIMI_TOKENIZER_SEGMENT_CACHE_TOKENS.override(0):
            kimi_tokenizer.cache_kimi_segments(None, manager)
        self.assertNotIsInstance(
            manager.tokenizer.__dict__.get("_encode_text_piece"),
            kimi_tokenizer.SegmentEncodingCache,
        )
        with tpu_envs.SGLANG_TPU_KIMI_TOKENIZER_SEGMENT_CACHE_TOKENS.override(64):
            kimi_tokenizer.cache_kimi_segments(None, manager)
        self.assertIsInstance(
            manager.tokenizer._encode_text_piece, kimi_tokenizer.SegmentEncodingCache
        )
        self.assertEqual(manager.tokenizer._encode_text_piece.max_tokens, 64)
        # No tokenizer (skip_tokenizer_init): nothing to do.
        with tpu_envs.SGLANG_TPU_KIMI_TOKENIZER_SEGMENT_CACHE_TOKENS.override(64):
            kimi_tokenizer.cache_kimi_segments(None, SimpleNamespace(tokenizer=None))


if __name__ == "__main__":
    unittest.main()
