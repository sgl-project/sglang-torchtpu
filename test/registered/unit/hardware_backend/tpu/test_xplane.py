"""Unit tests for rebasing libtpu XSpace timestamps onto the wall clock."""

import unittest

from sglang.srt.hardware_backend.tpu.xplane import rebase_to_wall_clock
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

START_NS = 1_790_428_911_372_434_336


def _varint(value):
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _int(field, value):
    return _varint(field << 3) + _varint(value)


def _len(field, payload):
    if isinstance(payload, str):
        payload = payload.encode()
    return _varint(field << 3 | 2) + _varint(len(payload)) + payload


def _event(metadata_id, offset_ps, duration_ps):
    return _len(4, _int(1, metadata_id) + _int(2, offset_ps) + _int(3, duration_ps))


def _line(name, events, timestamp_ns=None):
    body = _len(2, name)
    if timestamp_ns is not None:
        body += _int(3, timestamp_ns)
    return _len(3, body + b"".join(events))


def _task_env_plane(start_ns):
    # stat_metadata map entry {7: XStatMetadata(id=7, name=...)} and the stat.
    stat_meta = _len(
        5, _int(1, 7) + _len(2, _int(1, 7) + _len(2, "profile_start_time"))
    )
    stat = _len(6, _int(1, 7) + _int(4, start_ns))
    return _len(1, _len(2, "Task Environment") + stat_meta + stat)


def _device_plane():
    return _len(
        1,
        _len(2, "/device:TPU:0")
        + _line("XLA Ops", [_event(1, 205_059_304_950, 10_585_910)])
        + _line("Steps", [_event(2, 0, 5)], timestamp_ns=455_495),
    )


def _line_timestamps(space):
    """{line name: timestamp_ns}, an absent timestamp read as proto3's 0."""
    from sglang.srt.hardware_backend.tpu.xplane import _fields

    stamps = {}
    for field, _, plane, _ in _fields(memoryview(space)):
        if field != 1:
            continue
        for plane_field, _, line, _ in _fields(plane):
            if plane_field != 3:
                continue
            values = {f: v for f, _, v, _ in _fields(line)}
            stamps[bytes(values[2]).decode()] = values.get(3, 0)
    return stamps


class TestRebaseToWallClock(CustomTestCase):
    def test_every_line_moves_by_the_start_time(self):
        # A line libtpu left at the session origin (no timestamp) and one with
        # its own relative offset must both land on the wall clock.
        space = _device_plane() + _task_env_plane(START_NS) + _len(4, "decode-0-0")

        rebased, start_ns = rebase_to_wall_clock(space)

        self.assertEqual(start_ns, START_NS)
        self.assertEqual(
            _line_timestamps(rebased),
            {"XLA Ops": START_NS, "Steps": START_NS + 455_495},
        )

    def test_events_and_other_fields_are_untouched(self):
        # Events carry the offsets XProf reads per op; rewriting them would
        # double-shift, so they must come through byte for byte.
        event = _event(1, 205_059_304_950, 10_585_910)
        space = _device_plane() + _task_env_plane(START_NS) + _len(4, "decode-0-0")

        rebased, _ = rebase_to_wall_clock(space)

        self.assertIn(event, rebased)
        self.assertTrue(rebased.endswith(_len(4, "decode-0-0")))

    def test_without_start_time_the_file_is_returned_as_is(self):
        space = _device_plane()

        rebased, start_ns = rebase_to_wall_clock(space)

        self.assertIsNone(start_ns)
        self.assertIs(rebased, space)

    def test_truncated_file_raises(self):
        space = _device_plane() + _task_env_plane(START_NS)

        with self.assertRaises((ValueError, IndexError)):
            rebase_to_wall_clock(space[:-3])


if __name__ == "__main__":
    unittest.main(verbosity=2)
