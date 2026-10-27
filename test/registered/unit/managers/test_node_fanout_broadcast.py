"""Two-level request broadcast with shared-memory and inline tensor inputs.

Four gloo ranks stand in for two nodes of two ranks each. Rank 0 receives one
request whose features the tokenizer already put in /dev/shm and one whose
features arrive inline (the tokenizer's fallback). After the scheduler broadcast
every rank must hold SHM handles for the first request
and inline tensors for the fallback. After ``_finalize_shm_features`` every rank
must see the exact pixels while every segment is unlinked. Also exercise empty
polls, entirely inline inputs, and a remote node that cannot allocate SHM.
"""

import datetime
import errno
import mmap
import os
import pickle
import sys
import unittest
from array import array
from contextlib import ExitStack
from multiprocessing import shared_memory
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed
import torch.multiprocessing

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.io_struct import (  # noqa: E402
    BatchTokenizedEmbeddingReqInput,
    MMInputsProcessError,
    TokenizedEmbeddingReqInput,
)
from sglang.srt.managers.mm_utils import (  # noqa: E402
    ShmPointerMMData,
    wrap_shm_features,
)
from sglang.srt.managers.schedule_batch import (  # noqa: E402
    Modality,
    MultimodalDataItem,
    MultimodalProcessorOutput,
)
from sglang.srt.managers.scheduler_components import node_fanout  # noqa: E402
from sglang.srt.managers.scheduler_components.node_fanout import (  # noqa: E402
    node_fanout_enabled,
)
from sglang.srt.managers.scheduler_components.request_receiver import (  # noqa: E402
    SchedulerRequestReceiver,
)
from sglang.srt.sampling.sampling_params import SamplingParams  # noqa: E402
from sglang.srt.utils import broadcast_pyobj  # noqa: E402

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

WORLD, NNODES, NUM_IMAGES = 4, 2, 7


def _request(rid, features) -> TokenizedEmbeddingReqInput:
    return TokenizedEmbeddingReqInput(
        rid=rid,
        input_text="",
        input_ids=array("q", [1]),
        mm_inputs=(
            MultimodalProcessorOutput(
                mm_items=[
                    MultimodalDataItem(modality=Modality.IMAGE, feature=f, hash=100 + i)
                    for i, f in enumerate(features)
                ]
            )
            if features
            else None
        ),
        token_type_ids=None,
        sampling_params=SamplingParams(max_new_tokens=1),
    )


def _receiver() -> SchedulerRequestReceiver:
    return SchedulerRequestReceiver(
        recv_from_tokenizer=None,
        recv_from_rpc=None,
        recv_skipper=None,
        input_blocker=None,
        mm_receiver=None,
        server_args=SimpleNamespace(),
        model_config=SimpleNamespace(is_multimodal=True),
        max_recv_per_poll=-1,
        stream_output=lambda *args, **kwargs: None,
        get_last_batch=lambda: None,
    )


def _run_rank(rank: int, init_file: str) -> None:
    torch.set_num_threads(1)
    torch.distributed.init_process_group(
        backend="gloo",
        init_method=Path(init_file).as_uri(),
        rank=rank,
        world_size=WORLD,
        timeout=datetime.timedelta(seconds=60),
    )
    try:
        receiver = _receiver()
        parallel = SimpleNamespace(
            attn_dp_enabled=False,
            tp_size=WORLD,
            pp_size=1,
            nnodes=NNODES,
            tp_group=SimpleNamespace(
                rank=rank,
                rank_in_group=rank,
                ranks=list(range(WORLD)),
                cpu_group=torch.distributed.group.WORLD,
            ),
        )
        with (
            patch(
                "sglang.srt.managers.scheduler_components.node_fanout.get_parallel",
                return_value=parallel,
            ),
            patch(
                "sglang.srt.managers.scheduler_components.node_fanout.get_serving",
                return_value=SimpleNamespace(skip_tokenizer_init=False),
            ),
            patch(
                "sglang.srt.managers.scheduler_components.request_receiver.get_parallel",
                return_value=parallel,
            ),
            patch(
                "sglang.srt.managers.mm_utils.get_serving",
                return_value=SimpleNamespace(skip_tokenizer_init=False),
            ),
            patch.dict(
                os.environ, {"SGLANG_ENABLE_NODE_LEADER_REQUEST_BROADCAST": "1"}
            ),
        ):
            _exercise_receiver(rank, receiver)
    finally:
        torch.distributed.destroy_process_group()


def _exercise_receiver(rank, receiver):
    assert receiver._broadcast_reqs_across_ranks([] if rank == 0 else None) == []
    for base in (30, 90):
        _check_mixed_requests(rank, receiver, base)

    # Exercise a larger mapping granularity even on a 4-KiB-page CI host.
    with patch.object(
        node_fanout.mmap,
        "ALLOCATIONGRANULARITY",
        max(mmap.ALLOCATIONGRANULARITY, 65536),
    ):
        _check_packed_batch(rank, receiver)

    # No SHM on any node: this must not enter a materialization collective.
    # This tensor is large enough to use broadcast_pyobj's separate buffer path.
    reqs = (
        [_request("inline-only", [torch.full((640, 640, 3), 42, dtype=torch.uint8)])]
        if rank == 0
        else None
    )
    with patch(
        "sglang.srt.managers.scheduler_components.node_fanout._allocate_segment"
    ) as allocate:
        reqs = receiver._broadcast_reqs_across_ranks(reqs)
        receiver._finalize_shm_features(reqs)
        allocate.assert_not_called()
    assert torch.all(reqs[0].mm_inputs.mm_items[0].feature == 42)

    for index, fault in enumerate(("allocation", "source", "receiver")):
        _check_failure(rank, receiver, fault)
        # Reuse the same fan-out groups for a valid SHM request after each fault.
        _check_mixed_requests(rank, receiver, 110 + 10 * index)

    # The opt-out continues to use the flat broadcast.
    with patch.dict(os.environ, {"SGLANG_ENABLE_NODE_LEADER_REQUEST_BROADCAST": "0"}):
        reqs = receiver._broadcast_reqs_across_ranks(
            [_request("text-only", None)] if rank == 0 else None
        )
    assert reqs[0].rid == "text-only"


def _check_failure(rank, receiver, fault):
    reqs = None
    if rank == 0:
        req = _request(f"failed-{fault}", [torch.ones(8, dtype=torch.uint8)])
        wrap_shm_features(req, force=True)
        if fault == "source":
            req.mm_inputs.mm_items[0].feature.close_and_unlink()
        # Keep a healthy inline request in the batch to verify per-request errors.
        reqs = pickle.loads(pickle.dumps([req, _request("healthy", [torch.arange(4)])]))
    with ExitStack() as stack:
        if fault == "allocation":
            stack.enter_context(
                patch.object(node_fanout, "_allocate_segment", return_value=None)
            )
        elif fault == "receiver" and rank == WORLD - 1:
            stack.enter_context(
                patch(
                    "sglang.srt.managers.mm_utils.shared_memory.SharedMemory",
                    side_effect=FileNotFoundError("injected receiver open failure"),
                )
            )
        reqs = receiver._broadcast_reqs_across_ranks(reqs)
    names = [item.feature.shm_name for item in reqs[0].mm_inputs.mm_items]
    receiver._finalize_shm_features(reqs)
    assert isinstance(reqs[0].mm_inputs, MMInputsProcessError)
    assert torch.equal(reqs[1].mm_inputs.mm_items[0].feature, torch.arange(4))
    torch.distributed.barrier()
    _assert_unlinked(names)


def _check_mixed_requests(rank, receiver, base):
    reqs = None
    if rank == 0:
        pixels = [
            torch.full((321 + i, 323, 3), base + i, dtype=torch.uint8)
            for i in range(NUM_IMAGES)
        ]
        # The tokenizer's /dev/shm handles, as rank 0 unpickles them.
        shm_req = _request(f"vlm-{base}", pixels)
        wrap_shm_features(shm_req, force=True)
        shm_req = pickle.loads(pickle.dumps(shm_req))
        inline = [t.clone() + 100 for t in pixels]
        # Exceeds broadcast_pyobj's 1-MiB separate-buffer threshold. Reopening
        # SHM pointers must not repickle this tensor or replace its storage.
        inline[0] = torch.full((2 << 20,), base + 100, dtype=torch.uint8)
        reqs = [
            shm_req,
            _request(f"inline-{base}", inline),
            _request(f"text-{base}", None),
        ]
    sent_inline = []

    def observe_broadcast(data, *args, **kwargs):
        if data is not None:
            sent_inline.append(data[1].mm_inputs.mm_items[0].feature)
        return broadcast_pyobj(data, *args, **kwargs)

    with patch.object(node_fanout, "broadcast_pyobj", side_effect=observe_broadcast):
        reqs = receiver._broadcast_reqs_across_ranks(reqs)
    if rank % (WORLD // NNODES) == 0:
        assert reqs[1].mm_inputs.mm_items[0].feature is sent_inline[-1]
    assert [r.rid for r in reqs] == [
        f"vlm-{base}",
        f"inline-{base}",
        f"text-{base}",
    ]
    assert reqs[2].mm_inputs is None
    items = reqs[0].mm_inputs.mm_items
    assert all(isinstance(it.feature, ShmPointerMMData) for it in items)
    assert all(torch.is_tensor(it.feature) for it in reqs[1].mm_inputs.mm_items)
    names = [it.feature.shm_name for it in items]
    if rank >= WORLD // NNODES:
        # Forwarded tensors land in one segment on the second node.
        forwarded = reqs[0].mm_inputs.mm_items
        assert len({it.feature.shm_name for it in forwarded}) == 1
        assert all(
            it.feature.offset % mmap.ALLOCATIONGRANULARITY == 0 for it in forwarded
        )
    receiver._finalize_shm_features(reqs)
    for shift, req in ((0, reqs[0]), (100, reqs[1])):
        for i, it in enumerate(req.mm_inputs.mm_items):
            assert torch.is_tensor(it.feature)
            assert torch.all(it.feature == base + i + shift)
            assert it.hash == 100 + i
    # Check rank isolation for a tensor at a nonzero packed-segment offset.
    last = reqs[0].mm_inputs.mm_items[-1].feature
    writer_rank = WORLD // NNODES
    if rank == writer_rank:
        last.zero_()
    torch.distributed.barrier()
    assert torch.all(last == (0 if rank == writer_rank else base + NUM_IMAGES - 1))
    _assert_unlinked(names)


def _check_packed_batch(rank, receiver):
    expected = _request(
        "packed",
        [
            torch.arange(5, dtype=torch.uint8),
            [torch.arange(12, dtype=torch.float32).reshape(3, 4).T],
            (
                torch.arange(7, dtype=torch.bfloat16),
                torch.arange(3, dtype=torch.float32),
            ),
            None,
        ],
    )
    expected.mm_inputs.mm_items[2].precomputed_embeddings = (
        torch.arange(6, dtype=torch.float32).reshape(2, 3),
        torch.arange(4, dtype=torch.bfloat16),
    )
    expected.mm_inputs.mm_items[3].precomputed_embeddings = torch.arange(
        8, dtype=torch.float32
    )
    reqs = None
    if rank == 0:
        req = pickle.loads(pickle.dumps(expected))
        wrap_shm_features(req, force=True)
        # Rust ingress uses tuple shapes; exercise one at a nonzero packed offset.
        pointer = req.mm_inputs.mm_items[1].feature[0]
        pointer.shape = tuple(pointer.shape)
        reqs = pickle.loads(
            pickle.dumps(
                [BatchTokenizedEmbeddingReqInput(batch=[req, _request("text", None)])]
            )
        )
    reqs = receiver._broadcast_reqs_across_ranks(reqs)
    batch = reqs[0].batch
    assert [req.rid for req in batch] == ["packed", "text"]
    names, offsets = [], []
    for item in batch[0].mm_inputs.mm_items:
        for value in (item.feature, item.precomputed_embeddings):
            if value is None:
                continue
            pointers = value if isinstance(value, (list, tuple)) else [value]
            for pointer in pointers:
                assert isinstance(pointer, ShmPointerMMData)
                assert pointer.precomputed_hash == (
                    None if isinstance(value, (list, tuple)) else item.hash
                )
                names.append(pointer.shm_name)
                offsets.append(pointer.offset)
    if rank >= WORLD // NNODES:
        assert len(set(names)) == 1
        assert len(set(offsets)) == len(offsets)
        assert any(offset > 0 for offset in offsets)
        assert all(offset % mmap.ALLOCATIONGRANULARITY == 0 for offset in offsets)
    receiver._finalize_shm_features(reqs)
    assert batch[1].mm_inputs is None
    for actual, reference in zip(
        batch[0].mm_inputs.mm_items, expected.mm_inputs.mm_items, strict=True
    ):
        assert actual.hash == reference.hash
        for attr in ("feature", "precomputed_embeddings"):
            value, wanted = getattr(actual, attr), getattr(reference, attr)
            assert type(value) is type(wanted)
            if wanted is None:
                continue
            pairs = (
                zip(value, wanted, strict=True)
                if isinstance(wanted, (list, tuple))
                else [(value, wanted)]
            )
            for tensor, source in pairs:
                assert tensor.shape == source.shape
                assert tensor.dtype == source.dtype
                assert torch.equal(tensor, source)
    torch.distributed.barrier()
    _assert_unlinked(names)


def _assert_unlinked(names):
    for name in set(names):
        try:
            shared_memory.SharedMemory(name=name).close()
        except FileNotFoundError:
            continue
        raise AssertionError(f"segment {name} was not unlinked")


class TestNodeFanoutBroadcast(unittest.TestCase):
    def test_transport_eligibility(self):
        node_fanout.print_warning_once.cache_clear()
        self.addCleanup(node_fanout.print_warning_once.cache_clear)
        cases = [
            ({}, False, None, False, None),
            ({}, False, "0", False, None),
            ({}, False, "1", True, None),
            ({"nnodes": 1}, False, "1", False, "nnodes > 1"),
            ({"tp_size": 1}, False, "1", False, "tp_size > 1"),
            ({"pp_size": 2}, False, "1", False, "pp_size == 1"),
            ({"tp_size": 3}, False, "1", False, "divisible by nnodes"),
            ({"attn_dp_enabled": True}, False, "1", False, "DP attention"),
            ({}, True, "1", False, "tokenizer initialization"),
            # A prior rejected configuration must not cache the decision.
            ({}, False, "1", True, None),
        ]
        for overrides, skip_tokenizer, flag, expected, reason in cases:
            with self.subTest(
                overrides=overrides, skip_tokenizer=skip_tokenizer, flag=flag
            ):
                config = dict(nnodes=2, tp_size=4, pp_size=1, attn_dp_enabled=False)
                config.update(overrides)
                with (
                    patch.dict(os.environ),
                    patch(
                        "sglang.srt.managers.scheduler_components.node_fanout.get_parallel",
                        return_value=SimpleNamespace(**config),
                    ),
                    patch(
                        "sglang.srt.managers.scheduler_components.node_fanout.get_serving",
                        return_value=SimpleNamespace(
                            skip_tokenizer_init=skip_tokenizer
                        ),
                    ),
                    patch("sglang.srt.utils.common.logger.warning") as warning,
                ):
                    if flag is None:
                        os.environ.pop(
                            "SGLANG_ENABLE_NODE_LEADER_REQUEST_BROADCAST", None
                        )
                    else:
                        os.environ["SGLANG_ENABLE_NODE_LEADER_REQUEST_BROADCAST"] = flag
                    for _ in range(2):
                        self.assertEqual(node_fanout_enabled(), expected)
                    if reason is None:
                        warning.assert_not_called()
                    else:
                        warning.assert_called_once()
                        message = warning.call_args.args[0]
                        self.assertIn(
                            "SGLANG_ENABLE_NODE_LEADER_REQUEST_BROADCAST", message
                        )
                        self.assertIn(reason, message)

    def test_disabled_flag_does_not_require_runtime_context(self):
        with (
            patch.dict(
                os.environ, {"SGLANG_ENABLE_NODE_LEADER_REQUEST_BROADCAST": "0"}
            ),
            patch.object(node_fanout, "get_parallel") as parallel,
            patch.object(node_fanout, "get_serving") as serving,
        ):
            self.assertFalse(node_fanout_enabled())
            parallel.assert_not_called()
            serving.assert_not_called()

    def test_ineligible_dispatch_warns_without_creating_groups(self):
        node_fanout.print_warning_once.cache_clear()
        self.addCleanup(node_fanout.print_warning_once.cache_clear)
        for attn_dp in (False, True):
            with self.subTest(attn_dp=attn_dp):
                parallel = SimpleNamespace(
                    nnodes=2,
                    tp_size=4 if attn_dp else 1,
                    pp_size=1,
                    attn_dp_enabled=attn_dp,
                    attn_tp_rank=0,
                    attn_cp_rank=0,
                    enable_dp_attention_local_control_broadcast=True,
                )
                with (
                    patch.dict(
                        os.environ,
                        {"SGLANG_ENABLE_NODE_LEADER_REQUEST_BROADCAST": "1"},
                    ),
                    patch.object(node_fanout, "get_parallel", return_value=parallel),
                    patch(
                        "sglang.srt.managers.scheduler_components.request_receiver.get_parallel",
                        return_value=parallel,
                    ),
                    patch(
                        "sglang.srt.managers.scheduler_components.request_receiver.attn_cp_tp_broadcast_pyobj",
                        side_effect=lambda reqs: reqs,
                    ),
                    patch.object(node_fanout.dist, "new_group") as new_group,
                    patch("sglang.srt.utils.common.logger.warning") as warning,
                ):
                    receiver = _receiver()
                    for _ in range(2):
                        self.assertEqual(receiver._broadcast_reqs_across_ranks([]), [])
                    warning.assert_called_once()
                    self.assertIn(
                        "DP attention" if attn_dp else "tp_size > 1",
                        warning.call_args.args[0],
                    )
                    new_group.assert_not_called()

    def test_segment_creation_failure_is_recoverable(self):
        with patch.object(
            node_fanout.shared_memory,
            "SharedMemory",
            side_effect=OSError(errno.ENOSPC, "injected allocation failure"),
        ):
            self.assertIsNone(node_fanout._allocate_segment(mmap.ALLOCATIONGRANULARITY))

    @unittest.skipUnless(sys.platform == "linux", "requires posix_fallocate")
    def test_reservation_failure_closes_and_unlinks_segment(self):
        created = []
        allocate = shared_memory.SharedMemory

        def record_segment(*args, **kwargs):
            shm = allocate(*args, **kwargs)
            created.append(shm)
            return shm

        try:
            with (
                patch.object(
                    node_fanout.shared_memory,
                    "SharedMemory",
                    side_effect=record_segment,
                ),
                patch.object(
                    node_fanout.os,
                    "posix_fallocate",
                    side_effect=OSError(errno.ENOSPC, "injected reservation failure"),
                ),
            ):
                self.assertIsNone(
                    node_fanout._allocate_segment(mmap.ALLOCATIONGRANULARITY)
                )
            self.assertEqual(len(created), 1)
            self.assertEqual(created[0]._fd, -1)
            _assert_unlinked([created[0].name])
        finally:
            for shm in created:
                shm.close()
                try:
                    shm.unlink()
                except FileNotFoundError:
                    pass

    def test_pixels_reach_every_rank_through_node_shm(self):
        with TemporaryDirectory() as directory:
            torch.multiprocessing.spawn(
                _run_rank, args=(str(Path(directory) / "gloo-init"),), nprocs=WORLD
            )

    def test_handles_are_small_on_the_wire(self):
        req = _request("vlm", [torch.zeros((640, 640, 3), dtype=torch.uint8)])
        from sglang.srt.managers.mm_utils import wrap_shm_features

        wrap_shm_features(req, force=True)
        try:
            self.assertLess(len(pickle.dumps(req)), 8192)
        finally:
            for it in req.mm_inputs.mm_items:
                it.feature.close_and_unlink()


if __name__ == "__main__":
    unittest.main()
