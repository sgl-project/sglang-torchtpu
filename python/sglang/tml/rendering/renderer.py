from __future__ import annotations

from typing import TYPE_CHECKING

from sglang.tml.rendering import json
from sglang.tml.rendering.chat import (
    AuthorKind,
    ImagePointer,
    InvokeTool,
    Message,
    ModelEndSampling,
    Text,
    Thinking,
    ThinkingEffort,
    ToolDeclareJson,
    ToolError,
    _tool_spec_function,
    normalize_messages,
)
from sglang.tml.rendering.parser import Parser, StreamingParser
from sglang.tml.rendering.spans import (
    EncodedTextTokenSpan,
    ImageAssetPointerTokenSpan,
    TokenSpan,
    TokenSpans,
    compact_spans,
)

if TYPE_CHECKING:
    from sglang.tml.multimodal import Inkling2ImageProcessor
    from sglang.tml.rendering.tokenizers import Tokenizer

DEFAULT_THINKING_EFFORT_PREFIX = "Thinking effort level:"


class Renderer:
    def __init__(
        self,
        tokenizer: Tokenizer,
        image_tokenizer: Inkling2ImageProcessor | None = None,
    ):
        self.tokenizer = tokenizer
        self.image_tokenizer = image_tokenizer

    def encode_special(self, name: str) -> int:
        return self.tokenizer.encode_special(name)

    def encode_string(self, text: str) -> list[int]:
        return self.tokenizer.encode_ordinary(text)

    def _special(self, name: str) -> TokenSpan:
        return TokenSpan(EncodedTextTokenSpan([self.encode_special(name)]))

    def _text(self, text: str) -> TokenSpan:
        return TokenSpan(EncodedTextTokenSpan(self.encode_string(text)))

    def _image(self, image: ImagePointer) -> TokenSpan:
        if self.image_tokenizer is None:
            raise ValueError("Image rendering requires an image tokenizer")
        if image.width <= 0 or image.height <= 0:
            raise ValueError("Image dimensions must be positive")
        if (image.override_width is None) != (image.override_height is None):
            raise ValueError("Image override must set both width and height or neither")
        if image.override_width is not None:
            if image.override_width <= 0 or image.override_height <= 0:
                raise ValueError("Image override dimensions must be positive")
            width, height = image.override_width, image.override_height
        else:
            width, height = image.width, image.height
        virtual_width, virtual_height = self.image_tokenizer.resolve_virtual_dims(
            (width, height)
        )
        num_tokens = self.image_tokenizer.count_tokens_for_image(
            virtual_width, virtual_height
        )
        stamp = (virtual_width, virtual_height) != (image.width, image.height)
        return TokenSpan(
            ImageAssetPointerTokenSpan(
                image.location,
                image.format,
                image.width,
                image.height,
                num_tokens,
                virtual_width if stamp else None,
                virtual_height if stamp else None,
            )
        )

    def _body(self, content):
        if isinstance(content, Text):
            return "text", self._text(content.text)
        if isinstance(content, Thinking):
            return "thinking", self._text(content.text)
        if isinstance(content, ToolError):
            return "tool_error", self._text(content.text)
        if isinstance(content, ThinkingEffort):
            effort = round((content.millieffort / 1000) * 100) / 100
            return "text", self._text(f"{DEFAULT_THINKING_EFFORT_PREFIX} {effort:g}")
        if isinstance(content, ImagePointer):
            return "image", self._image(content)
        if isinstance(content, InvokeTool):
            if content.text is not None:
                return "invoke_tool_text", self._text(content.text)
            args = {arg.name: json.loads(arg.value) for arg in content.args}
            body = (
                '{"name":'
                + json.dumps(content.name)
                + ',"args":'
                + json.dumps(args)
                + "}"
            )
            return "invoke_tool_json", self._text(body)
        if isinstance(content, ToolDeclareJson):
            tools = [
                {"type": spec.type_name, **_tool_spec_function(spec)}
                for spec in content.tool_specs
            ]
            return "xml", self._text(json.dumps(tools, legacy_floats=False))
        raise ValueError(f"Unsupported content type: {type(content).__name__}")

    def _render(self, messages) -> TokenSpans:
        spans = []
        for message in messages:
            if isinstance(message.content, ModelEndSampling):
                spans.append(self._special("content_model_end_sampling"))
                continue
            kind, body = self._body(message.content)
            if not (
                isinstance(message.content, InvokeTool) and message.content.headerless
            ):
                spans.append(self._special(f"message_{message.author.kind.value}"))
                if message.author.name is not None:
                    spans.append(self._text(message.author.name))
            spans.extend(
                (self._special(f"content_{kind}"), body, self._special("end_message"))
            )
        return compact_spans(spans)

    def render_for_completion(self, messages) -> tuple[TokenSpans, Parser]:
        messages = normalize_messages(messages)
        return self._render(messages), Parser(self.tokenizer, messages)

    def render_for_completion_with_effort(
        self, messages, effort: float
    ) -> tuple[TokenSpans, Parser]:
        messages = normalize_messages(messages)
        if any(isinstance(message.content, ThinkingEffort) for message in messages):
            raise ValueError("Messages already contain ThinkingEffort")
        position = next(
            (
                index
                for index, message in enumerate(messages)
                if message.author.kind != AuthorKind.System
            ),
            len(messages),
        )
        messages.insert(position, Message.thinking_effort(effort))
        return self.render_for_completion(messages)

    def bridge_to_next_turn(
        self, prefix, message_history, next_messages
    ) -> tuple[TokenSpans, Parser]:
        history = normalize_messages(message_history)
        messages = normalize_messages(next_messages)
        return compact_spans([*prefix, *self._render(messages)]), Parser(
            self.tokenizer, [*history, *messages]
        )

    def get_generation_prefix(self) -> list[TokenSpan]:
        return []

    def parser_for_completion(self, messages=()) -> Parser:
        return Parser(self.tokenizer, messages)

    def streaming_parser(self, messages=()) -> StreamingParser:
        return StreamingParser(self.tokenizer, messages)

    def stop(self) -> list[int]:
        return [self.encode_special("content_model_end_sampling")]

    def end_of_message_condition(self, content_type=None) -> list[int]:
        name = (
            "content_model_end_sampling"
            if content_type in (ModelEndSampling, "model_end_sampling")
            else "end_message"
        )
        return [self.encode_special(name)]
