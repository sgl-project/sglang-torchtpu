from __future__ import annotations

import base64
import io
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum

from sglang.tml.rendering.spans import ImageFormat


class AuthorKind(str, Enum):
    User = "user"
    Model = "model"
    System = "system"
    Tool = "tool"


class MessageChannel(str, Enum):
    Main = "main"


@dataclass(frozen=True)
class Author:
    kind: AuthorKind
    name: str | None = None

    def __post_init__(self):
        object.__setattr__(self, "kind", AuthorKind(self.kind))


@dataclass(frozen=True)
class Text:
    text: str


@dataclass(frozen=True)
class Thinking:
    text: str


@dataclass(frozen=True)
class ToolError:
    text: str


@dataclass(frozen=True)
class ThinkingEffort:
    millieffort: int

    def __post_init__(self):
        if not isinstance(self.millieffort, int) or not 0 <= self.millieffort <= 1000:
            raise ValueError("millieffort must be an integer in [0, 1000]")


@dataclass(frozen=True)
class ModelEndSampling:
    pass


@dataclass(frozen=True)
class ImagePointer:
    location: str
    format: ImageFormat
    width: int
    height: int
    override_width: int | None = None
    override_height: int | None = None


@dataclass(frozen=True)
class ToolArg:
    name: str
    value: str


@dataclass(frozen=True)
class ToolMetadata:
    name: str
    value: str


@dataclass(frozen=True)
class StructuredToolCall:
    name: str
    args: Sequence[ToolArg] = ()
    metadata: Sequence[ToolMetadata] = ()
    tool_call_id: str | None = None


@dataclass(frozen=True, init=False)
class InvokeTool:
    name: str
    args: Sequence[ToolArg]
    metadata: Sequence[ToolMetadata]
    tool_call_id: str | None
    text: str | None
    headerless: bool

    def __init__(self, content: StructuredToolCall | str):
        if isinstance(content, StructuredToolCall):
            values = (
                content.name,
                tuple(content.args),
                tuple(content.metadata),
                content.tool_call_id,
                None,
                False,
            )
        elif isinstance(content, str):
            values = ("", (), (), None, content, True)
        else:
            raise TypeError("InvokeTool content must be StructuredToolCall or str")
        for name, value in zip(self.__dataclass_fields__, values):
            object.__setattr__(self, name, value)

    def copy(self, **changes) -> InvokeTool:
        unexpected = changes.keys() - self.__dataclass_fields__.keys()
        if unexpected:
            raise TypeError(f"Unexpected InvokeTool fields: {sorted(unexpected)}")
        result = object.__new__(InvokeTool)
        for name in self.__dataclass_fields__:
            object.__setattr__(result, name, changes.get(name, getattr(self, name)))
        return result


@dataclass(frozen=True)
class ToolSpecJson:
    type_name: str
    name: str
    description: str
    parameters: str | Mapping

    def __post_init__(self):
        if isinstance(self.parameters, Mapping):
            object.__setattr__(
                self,
                "parameters",
                json.dumps(
                    self.parameters,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    allow_nan=False,
                ),
            )
        elif not isinstance(self.parameters, str):
            raise TypeError("parameters must be a JSON object or serialized JSON")


def _tool_spec_function(spec: ToolSpecJson) -> dict:
    from sglang.tml.rendering.json import loads

    return {
        "name": spec.name,
        "description": spec.description,
        "parameters": loads(spec.parameters) if spec.parameters else {},
    }


@dataclass(frozen=True)
class ToolDeclareJson:
    tool_specs: Sequence[ToolSpecJson]


@dataclass(frozen=True)
class MessageMetadata:
    tool_call_id: str | None = None


Content = (
    Text
    | Thinking
    | ToolError
    | ThinkingEffort
    | ModelEndSampling
    | ImagePointer
    | InvokeTool
    | ToolDeclareJson
)


@dataclass(frozen=True)
class Message:
    content: Content
    author: Author
    channel: str = ""
    channel_enum: MessageChannel | None = None
    message_metadata: MessageMetadata | None = None

    def copy(self, **changes) -> Message:
        return replace(self, **changes)

    @staticmethod
    def user_text(text: str) -> Message:
        return Message(Text(text), Author(AuthorKind.User))

    @staticmethod
    def model_text(text: str) -> Message:
        return Message(Text(text), Author(AuthorKind.Model))

    @staticmethod
    def system_text(text: str) -> Message:
        return Message(Text(text), Author(AuthorKind.System))

    @staticmethod
    def thinking_effort(effort: float) -> Message:
        if not math.isfinite(effort) or not 0 <= effort <= 1:
            raise ValueError("thinking effort must be a finite number in [0, 1]")
        return Message(
            ThinkingEffort(round(effort * 1000)),
            Author(AuthorKind.System),
            channel_enum=MessageChannel.Main,
        )


@dataclass(frozen=True)
class MessageList(Sequence[Message]):
    messages: Sequence[Message]

    def __len__(self):
        return len(self.messages)

    def __getitem__(self, index):
        return self.messages[index]

    @classmethod
    def from_oss(cls, messages: Sequence[Mapping]) -> MessageList:
        return cls(
            [
                message
                for item in messages
                for message in OpenAIMessage.from_oss_dict(item).to_messages()
            ]
        )

    def to_oss(self) -> list[dict]:
        return [message.to_oss_dict() for message in OpenAIMessage.from_messages(self)]

    @classmethod
    def from_oss_messages(cls, messages: Sequence[Mapping]) -> MessageList:
        return cls.from_oss(messages)

    def to_oss_messages(self) -> list[dict]:
        return self.to_oss()


@dataclass(frozen=True)
class StreamingMessageHeader:
    author: Author
    channel: MessageChannel | None = None


@dataclass(frozen=True)
class StreamingContent:
    content_index: int
    content: Text | Thinking


@dataclass(frozen=True)
class ParseUpdate:
    update: StreamingMessageHeader | StreamingContent | Message

    @classmethod
    def complete(cls, message: Message) -> ParseUpdate:
        return cls(message)

    @classmethod
    def new_message(cls, header: StreamingMessageHeader) -> ParseUpdate:
        return cls(header)

    @classmethod
    def append_content(cls, content: StreamingContent) -> ParseUpdate:
        return cls(content)

    def is_complete(self) -> bool:
        return isinstance(self.update, Message)

    def is_new_message(self) -> bool:
        return isinstance(self.update, StreamingMessageHeader)

    def is_append_content(self) -> bool:
        return isinstance(self.update, StreamingContent)

    def message(self) -> Message | None:
        return self.update if self.is_complete() else None

    def header(self) -> StreamingMessageHeader | None:
        return self.update if self.is_new_message() else None

    def content(self) -> StreamingContent | None:
        return self.update if self.is_append_content() else None


_ROLES = {
    "user": AuthorKind.User,
    "assistant": AuthorKind.Model,
    "system": AuthorKind.System,
    "tool": AuthorKind.Tool,
}
_IMAGE_FORMATS = {
    "JPEG": ImageFormat.Jpeg,
    "JPG": ImageFormat.Jpeg,
    "WEBP": ImageFormat.Webp,
}


def _image_from_oss(value: str | Mapping) -> ImagePointer:
    from PIL import Image

    location = value if isinstance(value, str) else value["url"]
    if location.startswith("data:"):
        header, payload = location.split(",", 1)
        if "base64" not in header.lower().split(";")[1:]:
            raise ValueError("Image data URI must contain base64 data")
        with Image.open(io.BytesIO(base64.b64decode(payload))) as image:
            fmt = _IMAGE_FORMATS.get(image.format, ImageFormat.Png)
            return ImagePointer(location, fmt, image.width, image.height)
    if isinstance(value, str) or "width" not in value or "height" not in value:
        raise ValueError("Remote image URLs must carry explicit width and height")
    return ImagePointer(
        location,
        _IMAGE_FORMATS.get(value.get("format", "png").upper(), ImageFormat.Png),
        value["width"],
        value["height"],
    )


def _content_from_oss(part: Mapping) -> Content:
    kind = part["type"]
    if kind == "text":
        return Text(part["text"])
    if kind == "thinking":
        return Thinking(part["thinking"])
    if kind == "tool_error":
        return ToolError(part["text"])
    if kind == "image_url":
        return _image_from_oss(part["image_url"])
    raise ValueError(f"Unsupported content part type: {kind}")


_MISSING = object()


def _tool_field(value, name: str, default=_MISSING):
    if isinstance(value, Mapping):
        result = value.get(name, default)
    else:
        try:
            result = getattr(value, name)
        except AttributeError:
            result = default
    if result is _MISSING:
        raise ValueError(f"Tool call missing {name}")
    return result


@dataclass(frozen=True)
class OpenAIMessage:
    author: Author
    content: Sequence[Content]
    channel: str = "main"
    tool_call_id: str | None = None
    tool_name: str | None = None

    @property
    def role(self) -> str:
        if self.author.kind == AuthorKind.System and self.author.name == "tool_declare":
            return "tool_declare"
        return next(role for role, kind in _ROLES.items() if kind == self.author.kind)

    @classmethod
    def from_oss_dict(cls, value: Mapping) -> OpenAIMessage:
        role = value["role"]
        if role == "tool_declare":
            specs = json.loads(value["content"])
            if not isinstance(specs, list):
                raise ValueError("tool_declare content must be a JSON array")
            tools = []
            for spec in specs:
                if not isinstance(spec, dict):
                    raise ValueError("Tool declarations must be JSON objects")
                function = spec.get("function")
                if not isinstance(function, dict):
                    raise ValueError("Tool declarations require a function object")
                parameters = function.get("parameters", {})
                if not isinstance(parameters, dict):
                    raise ValueError("Tool parameters must be a JSON object")
                tools.append(
                    ToolSpecJson(
                        spec.get("type", "function"),
                        function["name"],
                        function.get("description", ""),
                        parameters,
                    )
                )
            return cls(
                Author(AuthorKind.System, "tool_declare"), [ToolDeclareJson(tools)]
            )
        if role not in _ROLES:
            raise ValueError(f"Unsupported role: {role}")
        raw_content = value.get("content")
        if raw_content is None:
            content = []
        elif isinstance(raw_content, str):
            content = [Text(raw_content)]
        elif isinstance(raw_content, list):
            content = [_content_from_oss(part) for part in raw_content]
        else:
            raise ValueError(
                "Message content must be a string or a list of content parts"
            )
        for tool_call in value.get("tool_calls") or []:
            function = _tool_field(tool_call, "function")
            args = _tool_field(function, "arguments", "{}")
            args = json.loads(args) if isinstance(args, str) else args
            if not isinstance(args, dict):
                raise ValueError("tool_call arguments must decode to a JSON object")
            content.append(
                InvokeTool(
                    StructuredToolCall(
                        _tool_field(function, "name"),
                        [
                            ToolArg(name, json.dumps(arg, allow_nan=False))
                            for name, arg in args.items()
                        ],
                        tool_call_id=_tool_field(tool_call, "id", None),
                    )
                )
            )
        tool_name = value.get("name") if role == "tool" else None
        return cls(
            Author(_ROLES[role], tool_name),
            content,
            tool_call_id=value.get("tool_call_id"),
            tool_name=tool_name,
        )

    @classmethod
    def from_oss_messages(cls, messages: Sequence[Mapping]) -> list[OpenAIMessage]:
        return [cls.from_oss_dict(message) for message in messages]

    @classmethod
    def from_openai_chat_dict(cls, value: Mapping) -> OpenAIMessage:
        result = cls.from_oss_dict(value)
        reasoning = value.get("reasoning_content")
        if reasoning is not None and not isinstance(reasoning, str):
            raise ValueError("reasoning_content must be a string")
        if reasoning:
            content = list(result.content)
            if value.get("content") == "":
                content.pop(0)
            result = replace(result, content=[Thinking(reasoning), *content])
        return result

    def to_messages(self) -> list[Message]:
        author = self.author
        if author.kind == AuthorKind.Tool:
            author = Author(author.kind, self.tool_name or author.name)
        metadata = (
            MessageMetadata(self.tool_call_id)
            if self.tool_call_id is not None
            else None
        )
        result = [
            Message(
                item,
                Author(author.kind, item.name)
                if author.kind == AuthorKind.Model and isinstance(item, InvokeTool)
                else author,
                channel_enum=MessageChannel.Main,
                message_metadata=metadata,
            )
            for item in self.content
        ]
        if author.kind == AuthorKind.Model and result:
            result.append(
                Message(
                    ModelEndSampling(),
                    Author(AuthorKind.Model),
                    channel_enum=MessageChannel.Main,
                )
            )
        return result

    @classmethod
    def from_messages(cls, messages: Sequence[Message]) -> list[OpenAIMessage]:
        result = []
        pending = None
        for message in messages:
            if isinstance(message.content, ModelEndSampling):
                if pending is not None:
                    result.append(pending)
                    pending = None
                continue
            author = message.author
            if author.kind == AuthorKind.Model and isinstance(
                message.content, InvokeTool
            ):
                author = Author(AuthorKind.Model)
            channel = (
                message.channel_enum.value
                if message.channel_enum is not None
                else message.channel or "main"
            )
            same_turn = (
                pending is not None
                and pending.author.kind == author.kind
                and author.kind != AuthorKind.Tool
            )
            if same_turn and author.kind == AuthorKind.Model:
                if pending.channel != channel:
                    raise ValueError(
                        "Cannot represent multiple channels in one model turn"
                    )
                if (
                    pending.author.name is not None
                    and author.name is not None
                    and pending.author.name != author.name
                ):
                    raise ValueError(
                        "Cannot represent multiple authors in one model turn"
                    )
                if pending.author.name is None and author.name is not None:
                    pending = replace(pending, author=author)
            elif same_turn:
                same_turn = (
                    pending.channel == channel and pending.author.name == author.name
                )
            if same_turn:
                pending = replace(pending, content=[*pending.content, message.content])
            else:
                if pending is not None:
                    result.append(pending)
                pending = cls(
                    author,
                    [message.content],
                    channel,
                    message.message_metadata.tool_call_id
                    if message.message_metadata is not None
                    else None,
                    message.author.name
                    if message.author.kind == AuthorKind.Tool
                    else None,
                )
        if pending is not None:
            result.append(pending)
        return result

    def to_oss_dict(self) -> dict:
        if len(self.content) == 1 and isinstance(self.content[0], ToolDeclareJson):
            from sglang.tml.rendering.json import dumps

            tools = [
                {
                    "type": spec.type_name,
                    "function": _tool_spec_function(spec),
                }
                for spec in self.content[0].tool_specs
            ]
            return {"role": self.role, "content": dumps(tools, legacy_floats=False)}
        parts = []
        tools = []
        for content in self.content:
            if isinstance(content, (Text, ToolError)):
                parts.append({"type": "text", "text": content.text})
            elif isinstance(content, Thinking):
                parts.append({"type": "thinking", "thinking": content.text})
            elif isinstance(content, ImagePointer):
                value = {"url": content.location}
                if not content.location.startswith("data:"):
                    value.update(
                        width=content.width,
                        height=content.height,
                        format=content.format.value,
                    )
                parts.append({"type": "image_url", "image_url": value})
            elif isinstance(content, InvokeTool):
                if content.text is not None:
                    raise ValueError(
                        "Raw text tool calls have no OpenAI tool_calls representation"
                    )
                args = {arg.name: json.loads(arg.value) for arg in content.args}
                tool = {
                    "type": "function",
                    "function": {
                        "name": content.name,
                        "arguments": json.dumps(args),
                    },
                }
                if content.tool_call_id is not None:
                    tool["id"] = content.tool_call_id
                tools.append(tool)
            elif isinstance(content, ModelEndSampling):
                parts.append({"type": "model_end_sampling"})
            else:
                raise ValueError(
                    f"Content {type(content).__name__} has no OpenAI representation"
                )
        value = {
            "role": self.role,
            "content": parts[0]["text"]
            if len(parts) == 1 and parts[0]["type"] == "text"
            else parts,
        }
        if tools:
            value["tool_calls"] = tools
        if self.tool_call_id is not None:
            value["tool_call_id"] = self.tool_call_id
        if self.tool_name is not None:
            value["name"] = self.tool_name
        return value

    def to_openai_chat_dict(self) -> dict:
        reasoning = [item.text for item in self.content if isinstance(item, Thinking)]
        content = [item for item in self.content if not isinstance(item, Thinking)]
        value = replace(self, content=content).to_oss_dict()
        if value["content"] == []:
            value["content"] = None
        elif isinstance(value["content"], list):
            for part in value["content"]:
                if part["type"] == "image_url":
                    part["image_url"] = {"url": part["image_url"]["url"]}
        if reasoning:
            value["reasoning_content"] = "\n".join(reasoning)
        for call in value.get("tool_calls", []):
            if "id" not in call:
                raise ValueError("Missing tool_call_id for OpenAI tool call")
        if self.author.kind == AuthorKind.Tool:
            identifier = (
                self.tool_call_id if self.tool_call_id is not None else self.tool_name
            )
            if identifier is None:
                raise ValueError("Missing tool_call_id for OpenAI tool result")
            value["tool_call_id"] = identifier
        else:
            value.pop("name", None)
            value.pop("tool_call_id", None)
        return value


def normalize_messages(
    messages: Sequence[Message | OpenAIMessage | Mapping],
) -> list[Message]:
    result = []
    for message in messages:
        if isinstance(message, Mapping):
            message = OpenAIMessage.from_openai_chat_dict(message)
        if isinstance(message, OpenAIMessage):
            result.extend(message.to_messages())
        elif isinstance(message, Message):
            result.append(message)
        else:
            raise TypeError(
                "Expected Message, OpenAIMessage, or OpenAI message dictionary"
            )
    return result
