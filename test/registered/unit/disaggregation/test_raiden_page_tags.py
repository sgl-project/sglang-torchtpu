"""Per-page tags ride the BLOCKS message from prefill to decode when a page tag
source is registered, and leave the wire format unchanged when none is."""

import threading
import unittest
from unittest.mock import Mock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu.disaggregation.raiden import sender
from sglang_tpu.disaggregation.raiden.conn import RaidenKVManager, RaidenKVReceiver
from sglang_tpu.disaggregation.raiden.serde import Blocks
from sglang_tpu.disaggregation.raiden.state import DecodeState

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

ROOM = 41
TAG_BYTES = 4


def _tags(kind, pages):
    """A fake source: one 4-byte record per page, naming the pool and the page."""
    prefix = b"K" if kind == "kv" else b"S"
    return b"".join(prefix + page.to_bytes(3, "little") for page in pages)


def _blocks(**tag_fields):
    return Blocks(
        room=ROOM,
        blocks=[5, 6, 7],
        endpoint="127.0.0.1:31020",
        swa_blocks=[7],
        swa_endpoint="127.0.0.1:31022",
        aux_blob=b"",
        **tag_fields,
    )


def _decode_manager():
    """A decode manager with its room tables, no sockets or engines."""
    mgr = object.__new__(RaidenKVManager)
    mgr._prefill = None
    mgr._decode = DecodeState()
    mgr._pipeline = None
    mgr.attn_tp_rank = 0
    mgr.chunked_transfer = True
    mgr._state_lock = threading.Lock()
    mgr._aborted = {}
    mgr._completed_recv = set()
    mgr._engine = Mock()
    mgr._swa_engine = Mock()
    mgr._draft_engine = None
    mgr._unified_swa_pool = None
    return mgr


def _receiver(mgr, sources=1):
    receiver = object.__new__(RaidenKVReceiver)
    receiver.kv_mgr = mgr
    receiver.bootstrap_room = ROOM
    receiver.bootstrap_infos = [{}] * sources
    return receiver


class TestPageTagFields(unittest.TestCase):
    def tearDown(self):
        sender.set_page_tag_source(None)

    def test_without_a_source_the_message_is_unchanged(self):
        fields = sender.page_tag_fields([5, 6, 7], [7])

        self.assertEqual(fields, {})
        self.assertEqual(_blocks(**fields).to_frames(), _blocks().to_frames())

    def test_a_source_tags_full_and_window_pages(self):
        sender.set_page_tag_source(_tags)

        fields = sender.page_tag_fields([5, 6, 7], [7])

        self.assertEqual(fields["page_tags"], _tags("kv", [5, 6, 7]))
        self.assertEqual(fields["swa_page_tags"], _tags("swa", [7]))
        # The window holds the request's last page, index 2 of 3.
        self.assertEqual(fields["swa_page_tags_first"], 2)

    def test_a_full_attention_room_sends_no_window_tags(self):
        sender.set_page_tag_source(_tags)

        fields = sender.page_tag_fields([5, 6], [])

        self.assertEqual(fields, {"page_tags": _tags("kv", [5, 6])})

    def test_tags_survive_the_wire(self):
        sender.set_page_tag_source(_tags)
        message = _blocks(**sender.page_tag_fields([5, 6, 7], [7]))

        self.assertEqual(Blocks.from_frames(message.to_frames()), message)


class TestDecodeReadsPageTags(unittest.TestCase):
    def test_decode_keeps_the_tags_blocks_carried(self):
        mgr = _decode_manager()
        tags = {
            "page_tags": _tags("kv", [5, 6, 7]),
            "swa_page_tags": _tags("swa", [7]),
            "swa_page_tags_first": 2,
        }

        mgr._on_blocks(_blocks(**tags).to_frames())

        self.assertEqual(
            _receiver(mgr).page_tags(),
            (tags["page_tags"], tags["swa_page_tags"], 2),
        )

    def test_untagged_blocks_read_as_empty_tags(self):
        mgr = _decode_manager()

        mgr._on_blocks(_blocks().to_frames())

        self.assertEqual(_receiver(mgr).page_tags(), (b"", b"", 0))
        self.assertNotIn(ROOM, mgr._decode.page_tags)

    def test_only_a_single_source_blocks_path_carries_tags(self):
        mgr = _decode_manager()
        self.assertTrue(_receiver(mgr).carries_page_tags())
        # Resharding: several prefill sources push without BLOCKS.
        self.assertFalse(_receiver(mgr, sources=2).carries_page_tags())
        mgr._pipeline = Mock()
        self.assertFalse(_receiver(mgr).carries_page_tags())


if __name__ == "__main__":
    unittest.main()
