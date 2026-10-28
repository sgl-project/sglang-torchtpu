import re

from sglang.srt.parser.inkling_tokenizer import INKLING_CONTROL_TOKENS
from sglang.srt.parser.reasoning_parser import InklingDetector, StreamingParseResult

_JSON_EVENTS = re.compile(
    '"|' + "|".join(re.escape(token) for token in INKLING_CONTROL_TOKENS)
)


class InklingJsonDetector(InklingDetector):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._in_json_string = False
        self._json_escape = False

    def detect_and_parse(self, text: str) -> StreamingParseResult:
        self._in_json_string = False
        self._json_escape = False
        return super().detect_and_parse(text)

    def _parse_blocks(self, text: str) -> StreamingParseResult:
        content: list[str] = []
        reasoning: list[str] = []

        def parse_framing(fragment: str) -> None:
            result = super(InklingJsonDetector, self)._parse_blocks(fragment)
            content.append(result.normal_text)
            reasoning.append(result.reasoning_text)

        pos = 0
        while pos < len(text):
            if self._in_json_string:
                start = pos
                while pos < len(text):
                    char = text[pos]
                    pos += 1
                    if self._json_escape:
                        self._json_escape = False
                    elif char == "\\":
                        self._json_escape = True
                    elif char == '"':
                        self._in_json_string = False
                        break
                content.append(text[start:pos])
                continue

            match = _JSON_EVENTS.search(text, pos)
            if match is None:
                parse_framing(text[pos:])
                break
            parse_framing(text[pos : match.start()])
            token = match.group()
            if token == '"' and self._kind in (None, "content"):
                self._in_json_string = True
                content.append(token)
            else:
                parse_framing(token)
            pos = match.end()

        return StreamingParseResult(
            normal_text="".join(content), reasoning_text="".join(reasoning)
        )

    def finish(self) -> StreamingParseResult:
        result = (
            self._parse_blocks(self._buffer)
            if self._in_json_string
            else StreamingParseResult()
        )
        tail = super().finish()
        result.normal_text += tail.normal_text
        result.reasoning_text += tail.reasoning_text
        self._in_json_string = False
        self._json_escape = False
        return result
