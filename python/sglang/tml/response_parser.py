import codecs
import json
from dataclasses import dataclass, field
from typing import Literal

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.core_types import ToolCallItem
from sglang.srt.function_call.inkling_detector import InklingDetector
from sglang.tml.rendering.parser import (
    Boundary,
    ContentStart,
    FramingParser,
    ParseError,
)


@dataclass(frozen=True)
class ParsedText:
    index: int
    kind: Literal["text", "reasoning"]
    text: str


@dataclass(frozen=True)
class ParsedToolCall:
    index: int
    call: ToolCallItem


@dataclass
class ParsedOutput:
    parts: list[ParsedText | ParsedToolCall] = field(default_factory=list)

    @property
    def reasoning_text(self) -> str:
        return "".join(
            part.text
            for part in self.parts
            if isinstance(part, ParsedText) and part.kind == "reasoning"
        )

    @property
    def normal_text(self) -> str:
        return "".join(
            part.text
            for part in self.parts
            if isinstance(part, ParsedText) and part.kind == "text"
        )

    @property
    def calls(self) -> list[ToolCallItem]:
        return [part.call for part in self.parts if isinstance(part, ParsedToolCall)]

    def append_text(self, index: int, kind: Literal["text", "reasoning"], text: str):
        if self.parts and isinstance(self.parts[-1], ParsedText):
            previous = self.parts[-1]
            if previous.index == index and previous.kind == kind:
                self.parts[-1] = ParsedText(index, kind, previous.text + text)
                return
        self.parts.append(ParsedText(index, kind, text))

    def extend(self, other: "ParsedOutput") -> None:
        for part in other.parts:
            if isinstance(part, ParsedText):
                self.append_text(part.index, part.kind, part.text)
            else:
                self.parts.append(part)


class InklingResponseParser:
    def __init__(
        self,
        tokenizer,
        tools: list[Tool],
        *,
        stream_reasoning: bool = True,
        continue_final_message: bool = False,
    ):
        from sglang.tml.server import rendering_tokenizer

        self.tokenizer = rendering_tokenizer(tokenizer)
        self._parser = FramingParser(self.tokenizer)
        self._tools = tools or []
        self._tool_detector = InklingDetector()
        self._stream_reasoning = stream_reasoning
        self._kind = None
        self._body = ""
        self._offset = 0
        self._call_index = 0
        self._block_index = -1
        self._decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self._content_decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self._decoded_length = 0
        self._text_length = 0
        if continue_final_message:
            self._start_text(ParsedOutput())

    def _start_text(self, result: ParsedOutput) -> None:
        for name in ("message_model", "content_text"):
            self._consume(self.tokenizer.encode_special(name), result)

    def parse(
        self, token_ids, *, text=None, meta_info=None, incremental: bool = False
    ) -> ParsedOutput:
        meta_info = meta_info or {}
        finish_reason = meta_info.get("finish_reason")
        if incremental:
            if finish_reason and finish_reason.get("type") == "abort":
                token_ids = token_ids[
                    : max(0, meta_info["completion_tokens"] - self._offset)
                ]
            self._offset += len(token_ids)
        else:
            if len(token_ids) < self._offset:
                raise RuntimeError("Cumulative output token IDs cannot shrink")
            token_ids, self._offset = token_ids[self._offset :], len(token_ids)
        if text is not None:
            self._text_length = (
                self._text_length + len(text) if incremental else len(text)
            )
        result = ParsedOutput()
        for token in token_ids:
            decoded = self._decoder.decode(self.tokenizer.decode_bytes([token]))
            remaining = max(0, self._text_length - self._decoded_length)
            if (
                text is not None
                and finish_reason
                and (len(decoded) > remaining or remaining == 0)
            ):
                self._content_decoder.reset()
                if self.tokenizer.decode_special(token) is None:
                    for tail_token in self.tokenizer.encode_ordinary(
                        decoded[:remaining]
                    ):
                        self._parse_token(tail_token, result)
                break
            self._decoded_length += len(decoded)
            self._parse_token(token, result)
        return result

    def _parse_token(self, token: int, result: ParsedOutput) -> None:
        special = self.tokenizer.decode_special(token)
        if special in ("content_model_end_sampling", "endoftext"):
            result.extend(self.finish())
            return
        if self._parser.is_at_message_boundary():
            if special is None:
                self._start_text(result)
            elif special.startswith("content_"):
                self._consume(self.tokenizer.encode_special("message_model"), result)
        self._consume(token, result)

    def _consume(self, token: int, result: ParsedOutput) -> None:
        try:
            event = self._parser.parse_token(token)
        except ParseError as error:
            raise RuntimeError("Invalid generated Inkling framing") from error
        if isinstance(event, ContentStart):
            self._block_index += 1
            self._kind = event.kind
            self._body = ""
            self._content_decoder.reset()
            if self._kind == "text" or (
                self._kind == "thinking" and self._stream_reasoning
            ):
                self._append_content("", result)
        elif isinstance(event, int):
            self._append_content(
                self._content_decoder.decode(self.tokenizer.decode_bytes([event])),
                result,
            )
        elif event is Boundary.MESSAGE:
            self._append_content(self._content_decoder.decode(b"", final=True), result)
            if self._kind == "thinking":
                result.append_text(self._block_index, "reasoning", self._body)
            elif self._kind == "invoke_tool_json":
                self._parse_tool(result)
            elif self._kind == "invoke_tool_text" and self._tools:
                result.parts.append(
                    ParsedToolCall(
                        self._block_index,
                        ToolCallItem(
                            tool_index=self._call_index,
                            name="",
                            parameters=json.dumps(
                                {"text": self._body}, ensure_ascii=False
                            ),
                        ),
                    )
                )
                self._call_index += 1
            elif self._kind in ("invoke_tool_text", "tool_error"):
                result.append_text(self._block_index, "text", self._body)
            elif self._kind != "text":
                raise RuntimeError(f"Unsupported generated content type: {self._kind}")
            self._kind = None

    def _append_content(self, text: str, result: ParsedOutput) -> None:
        if self._kind == "text":
            result.append_text(self._block_index, "text", text)
        elif self._kind == "thinking" and self._stream_reasoning:
            result.append_text(self._block_index, "reasoning", text)
        else:
            self._body += text

    def _parse_tool(self, result: ParsedOutput) -> None:
        if self._tools:
            call = self._tool_detector._canonical_json_call(
                self._body, self._tools, self._call_index
            )
            if call is None:
                try:
                    payload = json.loads(self._body)
                except ValueError:
                    payload = None
                if isinstance(payload, dict) and payload.get("name"):
                    call = self._tool_detector._tool_call_item(
                        payload, self._tools, self._call_index
                    )
            if call is not None:
                result.parts.append(ParsedToolCall(self._block_index, call))
                self._call_index += 1
                return
        result.append_text(self._block_index, "text", self._body)

    def finish(self) -> ParsedOutput:
        result = ParsedOutput()
        if self._kind is not None:
            self._consume(self.tokenizer.encode_special("end_message"), result)
        self._parser.reset()
        return result
