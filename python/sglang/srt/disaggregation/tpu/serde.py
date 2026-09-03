"""Typed control and data headers for TPU disaggregated KV transfers."""

from __future__ import annotations

import enum
from typing import Any, ClassVar

import msgspec

from sglang.srt.disaggregation.tpu.transfer import PrefillTransferInfo
from sglang.srt.disaggregation.tpu.wire import WireMessage, require_non_negative


class MessageTag(bytes, enum.Enum):
    """Identify the TPU message type in the first ZMQ frame."""

    TRANSFER = b"TRANSFER"
    KV_DATA_CHANNEL = b"KV_DATA_CHANNEL"


class Transfer(WireMessage):
    """Tell prefill that one decode rank is ready to receive a room."""

    TAG: ClassVar[bytes] = MessageTag.TRANSFER

    room: int
    decode_ip: str
    decode_port: int
    num_pages: int
    required_dst_info_num: int
    decode_prefix_len: int
    dst_tp_rank: int
    decode_attn_tp_size: int

    def __post_init__(self) -> None:
        for field, value in (
            ("room", self.room),
            ("num_pages", self.num_pages),
            ("decode_prefix_len", self.decode_prefix_len),
            ("dst_tp_rank", self.dst_tp_rank),
        ):
            require_non_negative(value, field)
        if not 1 <= self.decode_port <= 65535:
            raise ValueError(
                f"invalid decode_port: expected 1..65535, got {self.decode_port}"
            )
        if self.required_dst_info_num < 1:
            raise ValueError("invalid required_dst_info_num: expected >= 1")
        if self.decode_attn_tp_size < 1:
            raise ValueError("invalid decode_attn_tp_size: expected >= 1")


class KVDataChannelHeader(msgspec.Struct, forbid_unknown_fields=True):
    """Metadata for one channel of a TPU KV data transfer.

    Unlike ``WireMessage``, this message has raw data trailers. Its wire layout
    is ``[KV_DATA_CHANNEL, JSON header, *aux frames, *layer frames]``. Keeping
    tensor bytes outside JSON lets the receiver copy each layer directly into
    pinned staging memory.
    """

    TAG: ClassVar[bytes] = MessageTag.KV_DATA_CHANNEL

    info: PrefillTransferInfo
    index_start: int
    num_pages: int
    channel: int
    num_channels: int
    layer_start: int
    num_layers: int
    num_aux: int

    def __post_init__(self) -> None:
        if self.index_start < 0:
            raise ValueError("invalid index_start: expected >= 0")
        require_non_negative(self.num_pages, "num_pages")
        if self.num_channels < 1:
            raise ValueError("invalid num_channels: expected >= 1")
        if not 0 <= self.channel < self.num_channels:
            raise ValueError(
                f"invalid channel: expected 0..{self.num_channels - 1}, "
                f"got {self.channel}"
            )
        if self.num_layers < 1:
            raise ValueError("invalid num_layers: expected >= 1")
        if self.num_aux < 0:
            raise ValueError("invalid num_aux: expected >= 0")
        if self.channel != 0 and self.num_aux:
            raise ValueError("invalid num_aux: only channel 0 may carry aux frames")
        for field, value in (
            ("info.room", self.info.room),
            ("info.src_tp_rank", self.info.src_tp_rank),
            ("info.dst_tp_rank", self.info.dst_tp_rank),
        ):
            require_non_negative(value, field)
        for field, value in (
            ("info.num_pages", self.info.num_pages),
            ("info.kv_packing", self.info.kv_packing),
            ("info.src_attn_tp_size", self.info.src_attn_tp_size),
            ("info.dst_attn_tp_size", self.info.dst_attn_tp_size),
            ("info.total_num_head", self.info.total_num_head),
            ("info.n_layers", self.info.n_layers),
            ("info.page_size", self.info.page_size),
        ):
            if value < 1:
                raise ValueError(f"invalid {field}: expected >= 1")
        if self.info.n_layers != self.num_layers:
            raise ValueError("inconsistent num_layers in header and transfer info")
        if self.info.n_lane_to_send <= 0:
            raise ValueError(
                "invalid transfer_info: transfer must contain at least one lane"
            )
        expected_start = self.channel * self.num_layers // self.num_channels
        if self.layer_start != expected_start:
            raise ValueError(
                f"invalid layer_start: expected {expected_start}, "
                f"got {self.layer_start}"
            )

    @property
    def _num_layer_frames(self) -> int:
        end = (self.channel + 1) * self.num_layers // self.num_channels
        return end - self.layer_start

    def to_frames(self, aux_frames: list[Any], layer_frames: list[Any]) -> list[Any]:
        """Encode the header and append its validated raw data frames."""
        if len(aux_frames) != self.num_aux:
            raise ValueError(
                f"expected {self.num_aux} aux frames, got {len(aux_frames)}"
            )
        if len(layer_frames) != self._num_layer_frames:
            raise ValueError(
                f"expected {self._num_layer_frames} layer frames, "
                f"got {len(layer_frames)}"
            )
        return [
            self.TAG,
            msgspec.json.encode(self),
            *aux_frames,
            *layer_frames,
        ]

    @classmethod
    def from_frames(
        cls, frames: list[Any]
    ) -> tuple[KVDataChannelHeader, list[Any], list[Any]]:
        """Decode the header and return separate aux and layer frame slices."""
        if len(frames) < 2:
            raise ValueError(
                "malformed KV_DATA_CHANNEL message: expected at least 2 "
                f"frames, got {len(frames)}"
            )
        tag_frame, header_frame, *payload_frames = frames
        if bytes(tag_frame) != cls.TAG:
            raise ValueError(
                "malformed KV_DATA_CHANNEL message: unexpected tag "
                f"{bytes(tag_frame)!r}"
            )
        try:
            header = msgspec.json.decode(bytes(header_frame), type=cls)
        except (msgspec.DecodeError, ValueError) as exc:
            raise ValueError(f"malformed KV_DATA_CHANNEL header: {exc}") from exc
        expected = 2 + header.num_aux + header._num_layer_frames
        if len(frames) != expected:
            raise ValueError(
                "malformed KV_DATA_CHANNEL message: expected "
                f"{expected} frames, got {len(frames)}"
            )
        return (
            header,
            payload_frames[: header.num_aux],
            payload_frames[header.num_aux :],
        )
