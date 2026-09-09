"""Framing and validation shared by TPU disaggregation transports."""

from __future__ import annotations

from typing import Any, ClassVar, TypeVar

import msgspec

WireMessageT = TypeVar("WireMessageT", bound="WireMessage")


class WireMessage(msgspec.Struct, forbid_unknown_fields=True):
    """Base class for typed messages encoded as ``[TAG, JSON payload]``.

    Subclasses define their tag and payload fields. The shared encoder and
    decoder enforce the frame layout, message tag, payload schema, and field
    validation.
    """

    TAG: ClassVar[bytes]

    def to_frames(self) -> list[bytes]:
        """Encode this message as its tag frame and JSON payload frame."""
        return [self.TAG, msgspec.json.encode(self)]

    @classmethod
    def from_frames(cls: type[WireMessageT], frames: list[Any]) -> WireMessageT:
        """Validate two wire frames and decode their typed payload."""
        if len(frames) != 2:
            raise ValueError(
                f"malformed {cls.__name__} message: expected 2 frames, "
                f"got {len(frames)}"
            )
        tag_frame, payload_frame = frames
        tag = bytes(tag_frame)
        if tag != cls.TAG:
            raise ValueError(
                f"malformed {cls.__name__} message: expected tag {cls.TAG!r}, "
                f"got {tag!r}"
            )
        try:
            return msgspec.json.decode(bytes(payload_frame), type=cls)
        except (msgspec.DecodeError, ValueError) as exc:
            raise ValueError(f"malformed {cls.__name__} payload: {exc}") from exc


def require_non_negative(value: int, field: str) -> None:
    """Reject a negative integer with its wire field name in the error."""
    if value < 0:
        raise ValueError(f"invalid {field}: expected >= 0, got {value}")
