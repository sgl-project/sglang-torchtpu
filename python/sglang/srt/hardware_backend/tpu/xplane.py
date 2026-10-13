"""Timestamp rebasing for libtpu's XSpace (.xplane.pb) files.

libtpu writes every line's timestamp relative to its own session start and
records the wall-clock start only as a ``profile_start_time`` stat, which XProf
ignores. Traces from several processes then all begin at zero, and a collective
that finished at one instant lands up to hundreds of milliseconds apart across
chips. Rebasing each line onto the wall clock lines them up.

Works on the protobuf wire format directly so it needs no xplane protos at
runtime; events are copied as opaque bytes, which keeps a 300 MB file to about
a second.
"""

from __future__ import annotations

from typing import Iterator, Optional, Tuple

# Field numbers from tsl/profiler/protobuf/xplane.proto.
_XSPACE_PLANES = 1
_XPLANE_NAME = 2
_XPLANE_LINES = 3
_XPLANE_STAT_METADATA = 5
_XPLANE_STATS = 6
_XLINE_TIMESTAMP_NS = 3
_XSTAT_METADATA_ID = 1
_XSTAT_UINT64 = 3
_XSTAT_INT64 = 4
_XSTAT_METADATA_NAME = 2

_WIRE_VARINT = 0
_WIRE_LEN = 2

_TASK_ENV_PLANE = "Task Environment"
_START_TIME_STAT = "profile_start_time"


def rebase_to_wall_clock(space: bytes) -> Tuple[bytes, Optional[int]]:
    """Shift every line in ``space`` by its ``profile_start_time``.

    Returns the rewritten XSpace and the start time applied, or the input
    unchanged and None when the file carries no start time.
    """
    view = memoryview(space)
    start_ns = _profile_start_time(view)
    if start_ns is None:
        return space, None

    out = bytearray()
    for field, _, plane, raw in _fields(view):
        if field != _XSPACE_PLANES:
            out += raw
            continue
        new_plane = bytearray()
        for plane_field, _, line, plane_raw in _fields(plane):
            if plane_field == _XPLANE_LINES:
                new_plane += _len_field(_XPLANE_LINES, _shift_line(line, start_ns))
            else:
                new_plane += plane_raw
        out += _len_field(_XSPACE_PLANES, new_plane)
    return bytes(out), start_ns


def _shift_line(line: memoryview, start_ns: int) -> bytes:
    # An absent timestamp_ns is proto3's zero, i.e. the session start.
    timestamp_ns = 0
    body = bytearray()
    for field, _, value, raw in _fields(line):
        if field == _XLINE_TIMESTAMP_NS:
            timestamp_ns = value
        else:
            body += raw
    body += _varint(_XLINE_TIMESTAMP_NS << 3 | _WIRE_VARINT)
    body += _varint(timestamp_ns + start_ns)
    return bytes(body)


def _profile_start_time(space: memoryview) -> Optional[int]:
    for field, _, plane, _ in _fields(space):
        if field != _XSPACE_PLANES:
            continue
        name, stat_ids, stats = "", {}, []
        for plane_field, _, value, _ in _fields(plane):
            if plane_field == _XPLANE_NAME:
                name = bytes(value).decode(errors="replace")
            elif plane_field == _XPLANE_STAT_METADATA:
                stat_id, stat_name = _map_entry_name(value)
                stat_ids[stat_name] = stat_id
            elif plane_field == _XPLANE_STATS:
                stats.append(value)
        if name != _TASK_ENV_PLANE or _START_TIME_STAT not in stat_ids:
            continue
        for stat in stats:
            values = {f: v for f, _, v, _ in _fields(stat)}
            if values.get(_XSTAT_METADATA_ID) == stat_ids[_START_TIME_STAT]:
                start_ns = values.get(_XSTAT_INT64, values.get(_XSTAT_UINT64))
                return start_ns or None
    return None


def _map_entry_name(entry: memoryview) -> Tuple[Optional[int], Optional[str]]:
    """Key and metadata name of a map<int64, X*Metadata> entry."""
    key = name = None
    for field, _, value, _ in _fields(entry):
        if field == 1:
            key = value
        elif field == 2:
            for meta_field, _, meta_value, _ in _fields(value):
                if meta_field == _XSTAT_METADATA_NAME:
                    name = bytes(meta_value).decode(errors="replace")
    return key, name


def _fields(buf: memoryview) -> Iterator[Tuple[int, int, object, memoryview]]:
    """Yield (field, wire type, value, raw bytes of the whole field)."""
    i, n = 0, len(buf)
    while i < n:
        start = i
        key, i = _read_varint(buf, i)
        field, wire = key >> 3, key & 7
        if wire == _WIRE_VARINT:
            value, i = _read_varint(buf, i)
        elif wire == 1:
            value, i = buf[i : i + 8], i + 8
        elif wire == _WIRE_LEN:
            length, i = _read_varint(buf, i)
            value, i = buf[i : i + length], i + length
        elif wire == 5:
            value, i = buf[i : i + 4], i + 4
        else:
            raise ValueError(f"unsupported protobuf wire type {wire}")
        if i > n:
            raise ValueError("truncated protobuf message")
        yield field, wire, value, buf[start:i]


def _read_varint(buf: memoryview, i: int) -> Tuple[int, int]:
    result = shift = 0
    while True:
        byte = buf[i]
        i += 1
        result |= (byte & 0x7F) << shift
        if byte < 0x80:
            return result, i
        shift += 7


def _varint(value: int) -> bytes:
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _len_field(field: int, payload: bytes) -> bytes:
    return _varint(field << 3 | _WIRE_LEN) + _varint(len(payload)) + payload
