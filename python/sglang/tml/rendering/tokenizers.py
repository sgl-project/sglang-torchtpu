from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import tiktoken


def _byte_decoder() -> dict[str, int]:
    visible = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    encoded = dict(zip(visible, visible))
    for byte in range(256):
        if byte not in encoded:
            encoded[byte] = 256 + len(encoded) - len(visible)
    return {chr(char): byte for byte, char in encoded.items()}


class Tokenizer:
    def __init__(self, tokenizer_file: str | Path):
        data = json.loads(Path(tokenizer_file).read_text())
        model = data["model"]
        pretokenizer = data["pre_tokenizer"]
        if (
            model["type"] != "BPE"
            or data["normalizer"] is not None
            or data["decoder"]["type"] != "ByteLevel"
            or pretokenizer["type"] != "Sequence"
            or len(pretokenizer["pretokenizers"]) != 2
        ):
            raise ValueError("Expected an Inkling2 byte-level BPE tokenizer")
        split, byte_level = pretokenizer["pretokenizers"]
        if (
            split["type"] != "Split"
            or split["behavior"] != "Isolated"
            or split["invert"]
            or byte_level["type"] != "ByteLevel"
            or byte_level["add_prefix_space"]
            or byte_level["use_regex"]
            or model["dropout"] is not None
        ):
            raise ValueError("Unsupported Inkling2 pretokenizer configuration")
        decoder = _byte_decoder()
        ranks = {
            bytes(decoder[char] for char in token): token_id
            for token, token_id in model["vocab"].items()
        }
        rank_ids = set(ranks.values())
        if len(ranks) != len(model["vocab"]) or len(rank_ids) != len(ranks):
            raise ValueError("Duplicate tokenizer vocabulary entries")
        spellings: dict[str, int] = {}
        for token in sorted(data["added_tokens"], key=lambda item: item["id"]):
            spelling, token_id = token["content"], token["id"]
            if not token["special"] or not (
                spelling.startswith("<|") and spelling.endswith("|>")
            ):
                raise ValueError(f"Unsupported added token: {spelling!r}")
            if spelling in spellings or token_id in rank_ids:
                raise ValueError(f"Duplicate special token: {spelling!r}")
            rank_ids.add(token_id)
            spellings[spelling] = token_id
        for name in ("begin_of_text", "endoftext", "content_model_end_sampling"):
            if f"<|{name}|>" not in spellings:
                raise ValueError(f"Missing Inkling2 special token: {name}")
        self._set_encoding(
            tiktoken.Encoding(
                name="inkling2",
                pat_str=split["pattern"]["Regex"],
                mergeable_ranks=ranks,
                special_tokens=spellings,
            )
        )

    def _set_encoding(self, encoding: tiktoken.Encoding) -> None:
        self._encoding = encoding
        self._special_ids = {
            spelling[2:-2]: token
            for spelling, token in encoding._special_tokens.items()
            if spelling.startswith("<|") and spelling.endswith("|>")
        }
        self._special_names = {token: name for name, token in self._special_ids.items()}

    @classmethod
    def from_tiktoken(cls, encoding: tiktoken.Encoding) -> Tokenizer:
        tokenizer = cls.__new__(cls)
        tokenizer._set_encoding(encoding)
        return tokenizer

    @classmethod
    def from_pretrained(cls, model_path: str | Path) -> Tokenizer:
        directory = Path(model_path)
        if directory.is_dir():
            return cls(directory / "tokenizer.json")
        from huggingface_hub import hf_hub_download

        return cls(hf_hub_download(str(model_path), "tokenizer.json"))

    @classmethod
    def from_file(cls, tokenizer_file: str | Path) -> Tokenizer:
        return cls(tokenizer_file)

    @property
    def bos_token(self) -> str:
        return "begin_of_text"

    @property
    def eos_token(self) -> str:
        return "endoftext"

    @property
    def all_special_tokens(self) -> list[str]:
        return [f"<|{name}|>" for name in self._special_ids]

    def encode_ordinary(self, text: str) -> list[int]:
        text.encode("utf-8")
        return self._encoding.encode_ordinary(text)

    def decode_bytes(self, token_ids: Sequence[int]) -> bytes:
        try:
            return self._encoding.decode_bytes(list(token_ids))
        except KeyError as exc:
            raise ValueError(f"Unknown token ID: {exc.args[0]}") from exc

    def decode(self, token_ids: Sequence[int]) -> str:
        return self.decode_bytes(token_ids).decode("utf-8", errors="replace")

    def decode_strict(self, token_ids: Sequence[int]) -> str:
        return self.decode_bytes(token_ids).decode("utf-8")

    def encode_special(self, name: str) -> int:
        try:
            return self._special_ids[name]
        except KeyError as exc:
            raise ValueError(f"Unknown special token: {name!r}") from exc

    def decode_special(self, token_id: int) -> str | None:
        return self._special_names.get(token_id)

    def is_special_token(self, token_id: int) -> bool:
        return token_id in self._special_names

    def special_tokens(self) -> list[str]:
        return list(self._special_ids)
