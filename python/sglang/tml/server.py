from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from functools import lru_cache

from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.function_call.inkling_detector import InklingDetector
from sglang.srt.parser.inkling_tokenizer import IMAGE_TOKEN_ID
from sglang.srt.tokenizer.tiktoken_tokenizer import TiktokenTokenizer
from sglang.tml.rendering import (
    EncodedTextTokenSpan,
    OpenAIMessage,
    Renderer,
    TokenSpan,
    json,
)
from sglang.tml.rendering.chat import _content_from_oss
from sglang.tml.rendering.tokenizers import Tokenizer


class _Inkling2Detector(InklingDetector):
    get_auto_tool_call_structural_tag = (
        BaseFormatDetector.get_auto_tool_call_structural_tag
    )


class Inkling2FunctionCallParser(FunctionCallParser):
    ToolCallParserEnum = {"inkling": _Inkling2Detector}


@lru_cache(maxsize=8)
def _load_tokenizer(name_or_path: str, revision: str | None) -> Tokenizer:
    if revision is None:
        return Tokenizer.from_pretrained(name_or_path)
    from pathlib import Path

    if Path(name_or_path).is_dir():
        return Tokenizer.from_pretrained(name_or_path)
    from huggingface_hub import hf_hub_download

    return Tokenizer.from_file(
        hf_hub_download(name_or_path, "tokenizer.json", revision=revision)
    )


def rendering_tokenizer(tokenizer) -> Tokenizer:
    if isinstance(tokenizer, TiktokenTokenizer):
        return Tokenizer.from_tiktoken(tokenizer.tokenizer)
    return _load_tokenizer(
        tokenizer.name_or_path, getattr(tokenizer, "init_kwargs", {}).get("revision")
    )


@dataclass(frozen=True)
class _RequestImage:
    url: str


class _RequestRenderer(Renderer):
    def _body(self, content):
        if isinstance(content, _RequestImage):
            return "image", TokenSpan(EncodedTextTokenSpan([IMAGE_TOKEN_ID]))
        return super()._body(content)


def _message(value: Mapping) -> OpenAIMessage:
    value = dict(value)
    if value["role"] == "developer":
        value["role"] = "system"
    if (
        value["role"] == "assistant"
        and value.get("tool_calls")
        and value.get("content") == ""
    ):
        value["content"] = None
    content = value.get("content")
    if not isinstance(content, list):
        return OpenAIMessage.from_openai_chat_dict(value)
    parts = []
    for part in content:
        if part["type"] == "image_url":
            image = part["image_url"]
            url = image if isinstance(image, str) else image["url"]
            if not isinstance(url, str) or not url:
                raise ValueError("Image URL must be a nonempty string")
            parts.append(_RequestImage(url))
        elif part["type"] == "tool_call":
            if value["role"] != "assistant":
                raise ValueError("Tool call content parts require role='assistant'")
            parts.extend(
                OpenAIMessage.from_oss_dict(
                    {"role": "assistant", "tool_calls": [part["tool_call"]]}
                ).content
            )
        else:
            parts.append(_content_from_oss(part))
    value["content"] = None
    message = OpenAIMessage.from_openai_chat_dict(value)
    reasoning_count = int(bool(value.get("reasoning_content")))
    return replace(
        message,
        content=[
            *message.content[:reasoning_count],
            *parts,
            *message.content[reasoning_count:],
        ],
    )


def render_inkling2_messages(
    messages,
    tokenizer,
    *,
    tools=None,
    reasoning_effort: float = 0.9,
    assistant_prefix: str | None = None,
) -> list[int]:
    history = []
    if tools:
        tools = [
            {
                **tool,
                "function": {
                    key: value
                    for key, value in tool["function"].items()
                    if value is not None
                },
            }
            for tool in tools
        ]
        history.append(
            OpenAIMessage.from_oss_dict(
                {"role": "tool_declare", "content": json.dumps(tools)}
            )
        )
    for value in messages:
        message = _message(value)
        history.extend(
            replace(part, author=message.author) for part in message.to_messages()
        )
    renderer = _RequestRenderer(tokenizer)
    spans, _ = renderer.render_for_completion_with_effort(history, reasoning_effort)
    tokens = [token for span in spans for token in span.span.tokens]
    if assistant_prefix is not None:
        tokens.extend(
            [
                tokenizer.encode_special("message_model"),
                tokenizer.encode_special("content_text"),
                *tokenizer.encode_ordinary(assistant_prefix),
            ]
        )
    return tokens
