"""Reuse a Kimi tokenizer's encodings of chat segments across requests.

Why: a Kimi-K3 chat renders to segments (structural markers, each message's
     text) that the remote-code tokenizer encodes one by one
     (``_encode_text_piece``: a Python pass over every character, then
     tiktoken). An AgentX turn resends the whole session, so every request
     encodes 300k-570k tokens of text the session's last turn already
     encoded: about 1 ms a thousand prompt tokens ahead of the prefill.
How: AFTER TokenizerManager.init_tokenizer_and_processor, when
     SGLANG_TPU_KIMI_TOKENIZER_SEGMENT_CACHE_TOKENS is set and the tokenizer
     has ``_encode_text_piece``, an instance attribute memoizes it on
     ``(allow_special_tokens, text)`` (least recently used out past that many
     cached tokens). The encoding is a function of those arguments alone, so
     the ids are the same; each call returns a new list.
Upstream seam issue: not filed (the tokenizer is the model's remote code).
Remove when: the tokenizer caches segment encodings itself, or sglang
     tokenizes only a session's new messages.
"""

from __future__ import annotations

import collections
import logging
import threading
from typing import Any, Callable

logger = logging.getLogger(__name__)


class SegmentEncodingCache:
    """``encode(text, allow_special_tokens)`` through a memo of up to
    ``max_tokens`` cached token ids."""

    def __init__(self, encode: Callable[..., list[int]], max_tokens: int) -> None:
        if max_tokens <= 0:
            raise ValueError(f"a segment cache of {max_tokens} tokens")
        self._encode = encode
        self.max_tokens = max_tokens
        self._entries: collections.OrderedDict[tuple[bool, str], tuple[int, ...]] = (
            collections.OrderedDict()
        )
        self._tokens = 0
        self._lock = threading.Lock()
        self.hits = self.misses = 0

    def __call__(self, text: str, allow_special_tokens: bool = True) -> list[int]:
        if type(text) is not str:
            return self._encode(text, allow_special_tokens=allow_special_tokens)
        key = (bool(allow_special_tokens), text)
        with self._lock:
            ids = self._entries.get(key)
            if ids is not None:
                self._entries.move_to_end(key)
                self.hits += 1
                return list(ids)
        encoded = self._encode(text, allow_special_tokens=allow_special_tokens)
        ids = tuple(encoded)
        with self._lock:
            self.misses += 1
            if len(ids) <= self.max_tokens and key not in self._entries:
                self._entries[key] = ids
                self._tokens += len(ids)
                while self._tokens > self.max_tokens:
                    _, old = self._entries.popitem(last=False)
                    self._tokens -= len(old)
        return list(ids)


def install(tokenizer: Any, max_tokens: int) -> bool:
    """Memoize ``tokenizer._encode_text_piece`` (an instance attribute over
    the class's method); False when the tokenizer has none."""
    encode = getattr(tokenizer, "_encode_text_piece", None)
    if encode is None or isinstance(encode, SegmentEncodingCache):
        return False
    tokenizer._encode_text_piece = SegmentEncodingCache(encode, max_tokens)
    return True


def cache_kimi_segments(result: None, manager: Any, *args, **kwargs) -> None:
    """AFTER TokenizerManager.init_tokenizer_and_processor."""
    from sglang_tpu.envs import envs as tpu_envs

    max_tokens = tpu_envs.SGLANG_TPU_KIMI_TOKENIZER_SEGMENT_CACHE_TOKENS.get()
    tokenizer = getattr(manager, "tokenizer", None)
    if max_tokens > 0 and tokenizer is not None and install(tokenizer, max_tokens):
        logger.info(
            "Kimi tokenizer: segment encodings cached (up to %d tokens)", max_tokens
        )
    return result


def hook_specs() -> list[tuple]:
    from sglang.srt.plugins.hook_registry import HookType

    return [
        (
            "sglang.srt.managers.tokenizer_manager.TokenizerManager.init_tokenizer_and_processor",
            cache_kimi_segments,
            HookType.AFTER,
        ),
    ]
