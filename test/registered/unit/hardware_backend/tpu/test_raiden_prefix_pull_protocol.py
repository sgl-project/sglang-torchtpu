"""CPU end-to-end check of the reverse Raiden transfer on a fake platform.

Eight fake decode ranks (attention TP 8) and two fake TP=1 prefill stages run
the real ReversePull protocol over an in-memory router. A fake controller
stands in for tpu-sync: it plans each coordinate from the registered spans with
the native planner's coverage rules, copies the bytes, and reports the native
completions both sides poll. The stages must end up with the decode's prefix
and window pages, head for head.
"""

import threading
import time
import unittest
from types import SimpleNamespace

import numpy as np

from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu.disaggregation.prefix_pull_decode import HeldPrefix
from sglang_tpu.disaggregation.raiden import pp_reverse
from sglang_tpu.disaggregation.raiden.host_pool import StagingAdmission
from sglang_tpu.disaggregation.raiden.pp_reverse import DONE, FAILED, ReversePull

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

TP = 8
PAGE = 128
HEADS = 16
DRAFT_HEADS = 8
ROOM = 4242
# Prefill stage -> its full-attention and sliding-window layers.
STAGES = {0: (["fa.l0"], ["swa.l1"]), 1: (["fa.l2"], ["swa.l3", "draft.swa.l0"])}
NUM_PAGES = 12


class _Tensor:
    def __init__(self, array):
        self.array = array
        self.shape = array.shape

    def element_size(self):
        return 1


def _heads(tag):
    return DRAFT_HEADS if tag.startswith("draft.") else HEADS


def _pool(tag, tensor):
    stride = int(np.prod(tensor.shape[1:]))
    return dict(tag=tag, block_stride_bytes=stride, num_blocks=tensor.shape[0])


class _Platform:
    """Router, controllers and native completion of the fake deployment."""

    def __init__(self):
        rng = np.random.default_rng(7)
        tags = [t for full, swa in STAGES.values() for t in full + swa]
        # The model's KV of every layer, all heads: [pages, 2 * heads, 2, 4, PAGE].
        self.truth = {
            tag: rng.integers(
                0, 256, (NUM_PAGES, 2 * _heads(tag), 2, 4, PAGE), np.uint8
            )
            for tag in tags
        }
        self.endpoints = {}
        self.registrations = {}
        self.native = {}  # endpoint key -> {"fa"/"swa": (sent, received, failed)}
        self.lock = threading.Lock()

    def send(self, endpoint, frames):
        self.endpoints[endpoint].on_message(frames)

    def register(self, controller, req_id, uuid, unit, block_ids, pool_spans):
        key = (controller, req_id, uuid)
        self.registrations.setdefault(key, {})[unit.data_replica_idx] = (
            list(block_ids),
            {entry.tag: entry for entry in pool_spans},
        )

    def coordinate(self, **kw):
        """The source controller's plan and the engines' copy, at once."""
        controller = kw["src_controller_address"]
        regs = self.registrations[(controller, kw["req_id"], kw["uuid"])]
        assert sorted(regs) == list(range(TP)), "every decode rank registers"
        (dst_unit,) = kw["dst_units"]
        stage = self.stages[dst_unit.data_replica_idx]
        cursor = 0
        for tag, count in zip(kw["transfer_pool_tags"], kw["dst_block_counts"]):
            dst_ids = kw["dst_device_block_ids"][cursor : cursor + count]
            cursor += count
            dst = stage.local[tag].array
            live = int(np.prod(dst.shape[1:]))
            covered = [[] for _ in dst_ids]
            for unit in kw["src_units"]:
                rank = unit.data_replica_idx
                block_ids, entries = regs[rank]
                src = self.decodes[rank].local[tag].array
                for span in entries[tag].spans:
                    page = src[block_ids[span.src_block_ordinal]].reshape(-1)
                    index, start = divmod(span.dst_offset_bytes, live)
                    out = dst[dst_ids[index]].reshape(-1)
                    out[start : start + span.size_bytes] = page[
                        span.src_offset_bytes : span.src_offset_bytes + span.size_bytes
                    ]
                    covered[index].append((start, start + span.size_bytes))
            for intervals in covered:
                until = 0
                for lo, hi in sorted(intervals):
                    assert lo == until, "coverage gap or overlap"
                    until = hi
                assert until == live
        half = "swa" if kw["req_id"].startswith("swa:") else "fa"
        done = SimpleNamespace(req_id=kw["req_id"])
        with self.lock:
            for unit in kw["src_units"]:
                self._settle(("decode", unit.data_replica_idx), half, 0, done)
            self._settle(("prefill", dst_unit.data_replica_idx), half, 1, done)
        return True

    def _settle(self, who, half, kind, transfer):
        lists = self.native.setdefault(who, {}).setdefault(half, ([], [], []))
        lists[kind].append(transfer)

    def drain(self, who):
        with self.lock:
            halves = self.native.pop(who, {})
        return halves.get("fa"), halves.get("swa")


def _worker(platform, *, producer, rank, tp_rank, pools_by_tag, held=None):
    full_tags = [t for t in pools_by_tag if not pp_reverse.is_swa_pool(t)]
    swa_tags = [t for t in pools_by_tag if pp_reverse.is_swa_pool(t)]
    role = "prefill" if producer else "decode"
    controller = f"{role}-ctl"
    unit_fields = dict(
        job_name="sglang",
        job_replica_id=controller,
        data_name=role,
        data_replica_idx=rank,
    )
    swa_unit_fields = unit_fields | {"job_replica_id": f"{role}-swa-ctl"}

    def client(ctl):
        return SimpleNamespace(
            register_request_blocks=lambda **kw: platform.register(ctl, **kw),
            complete_request_blocks=lambda *a: None,
            cancel_request_blocks_if_unclaimed=lambda *a: True,
        )

    endpoint = f"tcp://{role}:{31000 + rank}"
    t = SimpleNamespace(
        producer=producer,
        manager=SimpleNamespace(
            attn_tp_rank=tp_rank,
            attn_tp_size=1 if producer else TP,
            pp_rank=rank if producer else 0,
            local_ip=role,
            rank_port=31000 + rank,
            _timeout_s=30.0,
            kv_args=SimpleNamespace(total_kv_head_num=HEADS, page_size=PAGE),
            _norm_req_id=lambda r: r,
        ),
        pools=[_pool(tag, pools_by_tag[tag]) for tag in full_tags],
        tensors=[pools_by_tag[tag] for tag in full_tags],
        swa=SimpleNamespace(
            pools=[_pool(tag, pools_by_tag[tag]) for tag in swa_tags],
            tensors=[pools_by_tag[tag] for tag in swa_tags],
            tags=swa_tags,
            client=client(f"{role}-swa-ctl"),
            unit=SimpleNamespace(**swa_unit_fields),
            unit_fields=swa_unit_fields,
            controller=f"{role}-swa-ctl",
            max_blocks=8,
        ),
        client=client(controller),
        unit=SimpleNamespace(**unit_fields),
        unit_fields=unit_fields,
        controller=controller,
        admission=StagingAdmission(4 * NUM_PAGES, 4 * 8),
        max_blocks=NUM_PAGES,
        seq_along_lane=True,
        draft_kv_heads=DRAFT_HEADS,
        draft_swa_kv_heads=None,
        replies={},
        lock=threading.RLock(),
        hbm_mem_type=2,
        _send=platform.send,
        _coordinate=lambda address, **kw: platform.coordinate(**kw),
    )
    reverse = ReversePull(t, held)
    reverse.local = pools_by_tag
    platform.endpoints[endpoint] = reverse
    return reverse, endpoint


class TestPrefixPullProtocol(unittest.TestCase):
    def setUp(self):
        self.platform = p = _Platform()
        # The decode holds the prefix [0, 6 pages) at pages decode_full, and the
        # window ending there (two pages) at decode_swa, in every rank's slice.
        self.decode_full = [9, 2, 7, 0, 11, 4]
        self.decode_swa = [5, 1]
        end = len(self.decode_full) * PAGE
        held = HeldPrefix(
            end=end,
            page_size=PAGE,
            full_pages=tuple(self.decode_full),
            window_start=end - 2 * PAGE,
            swa_pages=tuple(self.decode_swa),
        )
        self.end = end
        p.decodes = {}
        decode_endpoints = []
        for rank in range(TP):
            local = {}
            for tag, truth in p.truth.items():
                rows = truth.shape[1] // TP
                local[tag] = _Tensor(
                    np.ascontiguousarray(truth[:, rank * rows : (rank + 1) * rows])
                )
            reverse, endpoint = _worker(
                p,
                producer=False,
                rank=rank,
                tp_rank=rank,
                pools_by_tag=local,
                held=lambda room: held if room == ROOM else None,
            )
            p.decodes[rank] = reverse
            decode_endpoints.append(endpoint)
        p.stages = {}
        for stage, (full_tags, swa_tags) in STAGES.items():
            local = {
                tag: _Tensor(np.zeros_like(p.truth[tag]))
                for tag in full_tags + swa_tags
            }
            reverse, _ = _worker(
                p, producer=True, rank=stage, tp_rank=0, pools_by_tag=local
            )
            reverse.transfer.replies[ROOM] = {
                endpoint: dict(attn_tp_size=TP, attn_tp_rank=r)
                for r, endpoint in enumerate(decode_endpoints)
            }
            p.stages[stage] = reverse

    def _run(self, timeout=10.0):
        p = self.platform
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            for stage, reverse in p.stages.items():
                reverse.record(*p.drain(("prefill", stage)))
                reverse.poll()
            for rank, reverse in p.decodes.items():
                reverse.record(*p.drain(("decode", rank)))
                reverse.poll()
            states = {s: r.status(ROOM)[0] for s, r in p.stages.items()}
            if all(state != "pending" for state in states.values()):
                return states
            time.sleep(0.005)
        self.fail("prefix pull did not settle")

    def test_every_stage_assembles_the_decode_prefix_and_window(self):
        p = self.platform
        start = 2 * PAGE
        prefill_full = [3, 10, 6, 8]  # pages of [start, end) on every stage
        prefill_swa = [7, 2]
        for reverse in p.stages.values():
            reverse.start(
                ROOM, start=start, end=self.end, full=prefill_full, swa=prefill_swa
            )
        self.assertEqual(self._run(), {0: DONE, 1: DONE})
        for stage, (full_tags, swa_tags) in STAGES.items():
            local = p.stages[stage].local
            for tag in full_tags:
                for src, dst in zip(self.decode_full[2:], prefill_full):
                    np.testing.assert_array_equal(
                        local[tag].array[dst], p.truth[tag][src]
                    )
            for tag in swa_tags:
                for src, dst in zip(self.decode_swa, prefill_swa):
                    np.testing.assert_array_equal(
                        local[tag].array[dst], p.truth[tag][src]
                    )
        # Every decode rank completed its offers and returned its staging.
        for reverse in p.decodes.values():
            self.assertEqual(reverse.offers, {})
            self.assertEqual(reverse.transfer.admission.pages(), 0)

    def test_a_range_the_decode_does_not_hold_fails_loudly(self):
        stage = self.platform.stages[0]
        stage.start(ROOM, start=0, end=self.end - PAGE, full=[1] * 5, swa=[2, 3])
        self.assertEqual(self._run()[0], FAILED)
        self.assertIn("does not end at the held prefix", stage.status(ROOM)[1])

    def test_busy_decode_staging_is_retried(self):
        p = self.platform
        for reverse in p.decodes.values():
            reverse.transfer.admission.admit("other", 4 * NUM_PAGES)
        stage = p.stages[0]
        stage.start(ROOM, start=0, end=self.end, full=list(range(6)), swa=[6, 7])
        time.sleep(0.02)
        stage.poll()
        self.assertEqual(stage.status(ROOM)[0], "pending")
        for reverse in p.decodes.values():
            reverse.transfer.admission.release("other")
        self.assertEqual(self._run()[0], DONE)


if __name__ == "__main__":
    unittest.main()
