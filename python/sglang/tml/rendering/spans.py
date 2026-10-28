from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from enum import Enum


class ImageFormat(str, Enum):
    Png = "png"
    Jpeg = "jpeg"
    Webp = "webp"


@dataclass(frozen=True)
class EncodedTextTokenSpan:
    tokens: Sequence[int]


@dataclass(frozen=True)
class ImageAssetPointerTokenSpan:
    location: str
    format: ImageFormat
    width: int
    height: int
    num_tokens: int
    virtual_width: int | None = None
    virtual_height: int | None = None


@dataclass(frozen=True)
class TokenSpan:
    span: EncodedTextTokenSpan | ImageAssetPointerTokenSpan

    @property
    def num_tokens(self) -> int:
        if isinstance(self.span, EncodedTextTokenSpan):
            return len(self.span.tokens)
        return self.span.num_tokens


@dataclass(frozen=True)
class TokenSpans(Sequence[TokenSpan]):
    spans: Sequence[TokenSpan]

    def __len__(self) -> int:
        return len(self.spans)

    def __getitem__(self, index):
        return self.spans[index]

    def __iter__(self) -> Iterator[TokenSpan]:
        return iter(self.spans)

    @property
    def num_tokens(self) -> int:
        return sum(span.num_tokens for span in self.spans)


def compact_spans(spans: Sequence[TokenSpan]) -> TokenSpans:
    result = []
    tokens = []
    for span in spans:
        if isinstance(span.span, EncodedTextTokenSpan):
            tokens.extend(span.span.tokens)
        else:
            if tokens:
                result.append(TokenSpan(EncodedTextTokenSpan(tokens)))
                tokens = []
            result.append(span)
    if tokens:
        result.append(TokenSpan(EncodedTextTokenSpan(tokens)))
    return TokenSpans(result)
