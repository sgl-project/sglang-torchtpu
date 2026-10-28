from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum, auto
from json import JSONDecodeError
from typing import TYPE_CHECKING

from sglang.tml.rendering import json
from sglang.tml.rendering.chat import (
    Author,
    AuthorKind,
    InvokeTool,
    Message,
    MessageChannel,
    ModelEndSampling,
    ParseUpdate,
    StreamingContent,
    StreamingMessageHeader,
    StructuredToolCall,
    Text,
    Thinking,
    ToolArg,
    ToolError,
    normalize_messages,
)
from sglang.tml.rendering.spans import EncodedTextTokenSpan, TokenSpan

if TYPE_CHECKING:
    from sglang.tml.rendering.tokenizers import Tokenizer


@dataclass(frozen=True)
class ParseErrorKind:
    category: str
    name: str
    field: str | None = None


class ParseError(ValueError):
    def __init__(
        self,
        message: str,
        *,
        category="Content",
        kind="Other",
        field=None,
        sampled_tokens=(),
    ):
        super().__init__(message)
        self.kind = ParseErrorKind(category, kind, field)
        self.sampled_tokens = list(sampled_tokens)


@dataclass(frozen=True)
class ParseResult(Sequence[Message]):
    messages: Sequence[Message]
    num_tokens: int

    def __len__(self) -> int:
        return len(self.messages)

    def __getitem__(self, index):
        return self.messages[index]


def _parse_content(kind: str, body: str):
    if kind == "text":
        return Text(body)
    if kind == "thinking":
        return Thinking(body)
    if kind == "tool_error":
        return ToolError(body)
    if kind == "invoke_tool_text":
        return InvokeTool(body)
    if kind != "invoke_tool_json":
        raise ParseError(f"Unsupported content type: {kind}", kind="UnknownContentType")
    try:
        value = json.loads(body)
    except JSONDecodeError as error:
        remainder = body[error.pos :]
        incomplete = error.pos >= len(body) or error.msg.startswith("Unterminated")
        if error.msg == "Expecting value":
            incomplete |= remainder == "-" or any(
                value.startswith(remainder) and remainder != value
                for value in ("true", "false", "null")
            )
        if error.msg.startswith("Invalid \\u"):
            incomplete |= bool(re.fullmatch(r"u[0-9a-fA-F]{0,4}", remainder))
        if error.msg in ("Expecting ',' delimiter", "Extra data"):
            incomplete |= bool(
                re.search(
                    r"(?:^|[\s\[:,])-?(?:0|[1-9]\d*)(?:\.\d+)?(?:\.|[eE][+-]?)$", body
                )
            )
        raise ParseError(
            str(error), kind="JsonIncomplete" if incomplete else "JsonSyntax"
        ) from error
    except ValueError as error:
        raise ParseError(str(error), kind="JsonSyntax") from error
    if not isinstance(value, dict):
        raise ParseError(
            "Tool payload must be a dictionary",
            kind="ToolSchemaWrongType",
            field="payload",
        )
    if not isinstance(value.get("name"), str):
        raise ParseError(
            "Tool payload requires a string name",
            kind="ToolSchemaMissingField",
            field="name",
        )
    if "args" not in value:
        raise ParseError(
            "Tool payload requires args", kind="ToolSchemaMissingField", field="args"
        )
    if not isinstance(value["args"], dict):
        raise ParseError(
            "Tool args must be a dictionary", kind="ToolSchemaWrongType", field="args"
        )
    extra = sorted(value.keys() - {"name", "args"})
    if extra:
        raise ParseError(f"unsupported tool-call field `{extra[0]}`", category="Other")
    args = [
        ToolArg(key, json.dumps(item, legacy_floats=False))
        for key, item in sorted(value["args"].items())
    ]
    return InvokeTool(StructuredToolCall(value["name"], args))


@dataclass(frozen=True)
class ContentStart:
    kind: str
    name_tokens: Sequence[int] = ()


class Boundary(Enum):
    MESSAGE = auto()
    TURN = auto()


class FramingParser:
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        self.reset()

    def reset(self):
        self._state = "initial"
        self._tokens = []

    def is_at_message_boundary(self) -> bool:
        return self._state == "initial"

    def error_context(self) -> list[int]:
        return self._tokens.copy()

    def parse_token(self, token: int) -> ContentStart | Boundary | int | None:
        special = self.tokenizer.decode_special(token)
        if self._state == "content":
            if special == "end_message":
                self._state = "initial"
                return Boundary.MESSAGE
            self._tokens.append(token)
            return token
        if self._state == "name":
            if special is None:
                self._tokens.append(token)
                return None
            event = ContentStart(special.removeprefix("content_"), self._tokens)
            self._state = "content"
            self._tokens = []
            return event
        if special == "content_model_end_sampling":
            return Boundary.TURN
        if special == "message_model":
            self._tokens = []
            self._state = "name"
        elif special == "content_invoke_tool_text":
            self._tokens = []
            self._state = "content"
            return ContentStart("invoke_tool_text")
        else:
            raise ParseError(
                f"Expected a model message header, got {special or token}",
                category="Structure",
                kind="MissingMessageHeader",
                sampled_tokens=[token],
            )
        return None


class StreamingParser:
    def __init__(self, tokenizer: Tokenizer, messages=()):
        self.tokenizer = tokenizer
        self._framing = FramingParser(tokenizer)
        self._initial_ordinal = 0
        self._initial_ids = set()
        for message in normalize_messages(messages):
            if isinstance(message.content, InvokeTool):
                self._initial_ordinal += 1
                identifier = message.content.tool_call_id
                if identifier is not None and identifier.strip():
                    self._initial_ids.add(identifier)
            if message.message_metadata is not None:
                identifier = message.message_metadata.tool_call_id
                if identifier is not None and identifier.strip():
                    self._initial_ids.add(identifier)
        self.reset()

    def reset(self):
        self._framing.reset()
        self._author = None
        self._body = ""
        self._content_kind = ""
        self._pending_utf8 = []
        self._ordinal = self._initial_ordinal
        self._ids = self._initial_ids.copy()

    def is_at_message_boundary(self) -> bool:
        return self._framing.is_at_message_boundary() and not self._pending_utf8

    def _tool_call_id(self, name: str) -> str:
        while True:
            identifier = f"functions.{name or 'unknown'}:{self._ordinal}"
            self._ordinal += 1
            if identifier not in self._ids:
                self._ids.add(identifier)
                return identifier

    def _complete(self, content) -> ParseUpdate:
        if isinstance(content, InvokeTool):
            content = content.copy(tool_call_id=self._tool_call_id(content.name))
        return ParseUpdate(
            Message(
                content,
                self._author or Author(AuthorKind.Model),
                channel_enum=MessageChannel.Main,
            )
        )

    def _decode_name(self, tokens):
        if not tokens:
            return None
        try:
            return self.tokenizer.decode_strict(tokens)
        except (UnicodeDecodeError, ValueError) as error:
            raise ParseError(
                str(error), kind="Utf8Content", sampled_tokens=tokens
            ) from error

    def _check_pending(self):
        if self._pending_utf8:
            raise ParseError(
                f"Incomplete UTF-8 sequence at end of message: {len(self._pending_utf8)} tokens remaining",
                category="Structure",
                kind="InternalStateError",
                sampled_tokens=self._pending_utf8,
            )

    def _push_token(self, token: int) -> list[ParseUpdate]:
        event = self._framing.parse_token(token)
        if event is Boundary.MESSAGE:
            self._check_pending()
            return [self._complete(_parse_content(self._content_kind, self._body))]
        if event is Boundary.TURN:
            return [self._complete(ModelEndSampling())]
        if isinstance(event, ContentStart):
            self._author = Author(
                AuthorKind.Model, self._decode_name(event.name_tokens)
            )
            self._content_kind = event.kind
            self._body = ""
            return [ParseUpdate(StreamingMessageHeader(self._author))]
        if isinstance(event, int):
            self._pending_utf8.append(event)
            try:
                text = self.tokenizer.decode_bytes(self._pending_utf8).decode("utf-8")
            except UnicodeDecodeError as error:
                if (
                    error.reason == "unexpected end of data"
                    and len(self._pending_utf8) < 8
                ):
                    return []
                raise ParseError(
                    str(error), kind="Utf8Content", sampled_tokens=self._pending_utf8
                ) from error
            except ValueError as error:
                raise ParseError(
                    str(error), kind="Utf8Content", sampled_tokens=self._pending_utf8
                ) from error
            self._pending_utf8.clear()
            self._body += text
            if text and self._content_kind in ("text", "thinking"):
                content = (
                    Thinking(text) if self._content_kind == "thinking" else Text(text)
                )
                return [ParseUpdate(StreamingContent(0, content))]
            return []
        return []

    def parse_token(self, token: int) -> list[ParseUpdate]:
        try:
            return self._push_token(token)
        except ParseError as error:
            if not error.sampled_tokens:
                error.sampled_tokens = self._framing.error_context()
            self.reset()
            raise

    def parse_tokens(self, tokens: Sequence[int]) -> list[ParseUpdate]:
        result = []
        for token in tokens:
            result.extend(self.parse_token(token))
        return result

    def parse_span(self, span: TokenSpan) -> list[ParseUpdate]:
        if not isinstance(span.span, EncodedTextTokenSpan):
            tokens = (
                [] if self.is_at_message_boundary() else self._framing.error_context()
            )
            self.reset()
            raise ParseError(
                "Parser only supports text tokens",
                category="Other",
                sampled_tokens=tokens,
            )
        return self.parse_tokens(span.span.tokens)

    def flush(self) -> list[ParseUpdate]:
        try:
            self._check_pending()
            if not self.is_at_message_boundary():
                raise ParseError(
                    "Unterminated message",
                    category="Structure",
                    kind="UnterminatedMessage",
                    sampled_tokens=self._framing.error_context(),
                )
        finally:
            self.reset()
        return []


class Parser:
    def __init__(self, tokenizer: Tokenizer, messages=()):
        self._stream = StreamingParser(tokenizer, messages)
        self._in_use = False

    def parse_tokens(self, tokens: Sequence[int]) -> ParseResult:
        return self.parse([TokenSpan(EncodedTextTokenSpan(tokens))])

    def parse(self, spans: Sequence[TokenSpan]) -> ParseResult:
        stream = self.parse_stream()
        updates = stream.feed(spans)
        updates.extend(stream.finish())
        result = [update.update for update in updates if update.is_complete()]
        return ParseResult(result, sum(span.num_tokens for span in spans))

    def parse_stream(self) -> _IncrementalParseStream:
        if self._in_use:
            raise RuntimeError("parser is in use by a completion stream")
        self._in_use = True
        return _IncrementalParseStream(self)

    def parse_partial(self, spans: Sequence[TokenSpan]) -> list[ParseUpdate]:
        stream = self.parse_stream()
        updates = stream.feed(spans)
        try:
            updates.extend(stream.finish())
        except ParseError as error:
            if error.kind.name not in ("UnterminatedMessage", "InternalStateError"):
                raise
        return updates


class _IncrementalParseStream:
    def __init__(self, parser: Parser):
        self._parser = parser
        self._closed = False

    def _release(self):
        self._closed = True
        self._parser._in_use = False
        self._parser._stream.reset()

    def feed(self, spans: Sequence[TokenSpan]) -> list[ParseUpdate]:
        if self._closed:
            raise ParseError("completion stream is closed", category="Other")
        try:
            return [
                update
                for span in spans
                for update in self._parser._stream.parse_span(span)
            ]
        except ParseError:
            self._release()
            raise

    def finish(self) -> list[ParseUpdate]:
        if self._closed:
            raise ParseError("completion stream is already closed", category="Other")
        try:
            return self._parser._stream.flush()
        finally:
            self._release()

    def __del__(self):
        if not self._closed:
            self._release()
