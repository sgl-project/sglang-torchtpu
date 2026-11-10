"""CPU tests for deferred Engram token waits."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu.engram import (
    _MM_PAD_SHIFT_VALUE,
    HostEngramCoordinator,
    HostEngramInputs,
    PendingTokens,
    PreparedEngramUpload,
    capture_inputs,
    hash_on_host,
    pack_rows,
)

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

NGRAM = 3


def _reference_capture(
    requests,
    starts: list[int],
    lengths: list[int],
    ngram: int,
    pending: PendingTokens | None = None,
) -> HostEngramInputs:
    """Blocking reference for token windows and positions."""
    windows, all_positions = [], []
    shifts = np.arange(ngram, dtype=np.int64)
    for req, start, length in zip(requests, starts, lengths, strict=True):
        if start < 0 or length < 0:
            raise ValueError("Engram chunk start and length must be non-negative")
        if not length:
            continue
        stop = start + length
        lo = max(0, start - ngram + 1)
        prompt, output = req.origin_input_ids, req.output_ids
        prompt_len = len(prompt)
        known_end = prompt_len + len(output)
        # Do not copy a whole million-token context on every decode step.
        source = list(prompt[lo : min(stop, prompt_len)])
        if stop > prompt_len:
            source.extend(output[max(0, lo - prompt_len) : stop - prompt_len])
        for position in range(max(lo, known_end), stop):
            if pending is None:
                raise RuntimeError(
                    f"Engram input token at position {position} is not available on host"
                )
            source.append(pending.read(req, position))
        positions = np.arange(start, stop, dtype=np.int64)
        past = positions[:, None] - shifts
        indices = np.maximum(past - lo, 0)
        windows.append(np.asarray(source, dtype=np.int64)[indices])
        all_positions.append(positions)
    return HostEngramInputs(
        tokens=(
            np.concatenate(windows) if windows else np.empty((0, ngram), dtype=np.int64)
        ),
        positions=(
            np.concatenate(all_positions)
            if all_positions
            else np.empty(0, dtype=np.int64)
        ),
    )


def _request(prompt, output=()):
    return SimpleNamespace(origin_input_ids=list(prompt), output_ids=list(output))


def _tables():
    return dict(
        token_map=np.asarray([3, 1, 6, 0, 2, 5, 4, 7], dtype=np.int64),
        multipliers=np.asarray(
            [[100000000000000003, 13, 31], [7, 99999999999999999, 19]],
            dtype=np.int64,
        ),
        primes=np.asarray([[[11, 13], [17, 19]], [[23, 29], [31, 37]]]),
        offsets=np.asarray([[0, 11, 24, 41], [0, 23, 52, 83]]),
        pad_id=3,
    )


def _result(tokens):
    """A sampled result whose token copy completes at synchronize()."""
    result = SimpleNamespace(next_token_ids=None)
    event = Mock()

    def complete():
        result.next_token_ids = torch.tensor(tokens)

    event.synchronize.side_effect = complete
    result.copy_done = event
    return result, event


def test_capture_defers_in_flight_token_and_matches_blocking_capture():
    rng = np.random.default_rng(0)
    for trial in range(200):
        reqs, starts, lengths, pend_reqs, pend_pos, sampled = [], [], [], [], [], []
        for _ in range(int(rng.integers(1, 6))):
            prompt = rng.integers(0, 8, int(rng.integers(1, 7))).tolist()
            output = rng.integers(0, 8, int(rng.integers(0, 4))).tolist()
            req = _request(prompt, output)
            known = len(prompt) + len(output)
            if rng.random() < 0.7:  # decode: one token still in flight
                pend_reqs.append(req)
                pend_pos.append(known)
                sampled.append(int(rng.integers(0, 8)))
                starts.append(known)
                lengths.append(1)
            else:  # prefill chunk inside the known tokens
                start = int(rng.integers(0, known))
                starts.append(start)
                lengths.append(int(rng.integers(0, known - start + 1)))
            reqs.append(req)
        order = rng.permutation(len(reqs))
        reqs = [reqs[i] for i in order]
        starts = [starts[i] for i in order]
        lengths = [lengths[i] for i in order]
        result, event = _result(sampled)
        pending = PendingTokens(pend_reqs, pend_pos, result)
        deferred = capture_inputs(reqs, starts, lengths, NGRAM, pending)
        event.synchronize.assert_not_called()
        had_holes = bool(deferred.holes)
        resolved = deferred.resolve(pending)
        assert event.synchronize.call_count == int(had_holes)
        result2, _ = _result(sampled)
        stock = _reference_capture(
            reqs, starts, lengths, NGRAM, PendingTokens(pend_reqs, pend_pos, result2)
        )
        np.testing.assert_array_equal(resolved.tokens, stock.tokens)
        np.testing.assert_array_equal(resolved.positions, stock.positions)
        torch.testing.assert_close(
            hash_on_host(resolved, **_tables()), hash_on_host(stock, **_tables())
        )


def test_capture_rejects_missing_token_like_blocking_capture():
    req = _request([1, 2], [3])
    with pytest.raises(RuntimeError, match="not available on host"):
        capture_inputs([req], [3], [1], NGRAM, None)
    with pytest.raises(ValueError, match="non-negative"):
        capture_inputs([req], [-1], [1], NGRAM, None)
    # A pending result for another request fails when resolved, like stock.
    result, _ = _result([5])
    pending = PendingTokens([_request([1])], [1], result)
    deferred = capture_inputs([req], [3], [1], NGRAM, pending)
    with pytest.raises(RuntimeError, match="not available on host"):
        deferred.resolve(pending)


def _table(rows, dim=64, seed=1):
    gen = torch.Generator().manual_seed(seed)
    weight = torch.randint(0, 256, (rows, dim), dtype=torch.uint8, generator=gen)
    scale = torch.randint(100, 140, (rows, dim // 32), dtype=torch.uint8, generator=gen)
    return weight.view(torch.float8_e4m3fn), scale.view(torch.float8_e8m0fnu)


def _coordinator(
    *, image_token_id=None, raw_transfer_mod=None, layout_mod=None, sync_fn=None
):
    tables = _tables()
    hasher = SimpleNamespace(
        max_ngram_size=NGRAM,
        **{k: torch.as_tensor(v) for k, v in tables.items() if k != "pad_id"},
        pad_id=tables["pad_id"],
        image_token_id=image_token_id,
    )
    embeddings = []
    for layer in (0, 1):
        weight, scale = _table(rows=130, seed=layer + 3)
        embeddings.append(
            (layer, SimpleNamespace(weight=weight, scale=scale, row_start=0))
        )
    return (
        HostEngramCoordinator(
            hasher,
            embeddings,
            "cpu",
            raw_transfer_mod=raw_transfer_mod,
            layout_mod=layout_mod,
            sync_fn=sync_fn,
        ),
        embeddings,
    )


def _batch(requests, *, decode=False, starts=None, lengths=None):
    mode = SimpleNamespace(
        is_decode=lambda: decode,
        is_extend_without_speculative=lambda: not decode,
        is_idle=lambda: False,
    )
    return SimpleNamespace(
        reqs=requests,
        spec_algorithm=SimpleNamespace(is_none=lambda: True),
        forward_mode=mode,
        prefix_lens=starts,
        extend_lens=lengths,
        seq_lens_cpu=torch.tensor([s + 1 for s in starts]) if decode else None,
    )


def test_coordinator_decode_waits_at_first_stage_and_stages_reference_rows():
    coordinator, embeddings = _coordinator()
    a, b = _request([1, 2, 3]), _request([4, 5])
    requests, positions = coordinator.prepare(
        _batch([a, b], starts=[0, 0], lengths=[3, 2])
    )
    result, event = _result([6, 7])
    coordinator.remember(requests, positions, result)
    coordinator.clear_step()
    coordinator.prepare(_batch([b, a], decode=True, starts=[2, 3]))
    event.synchronize.assert_not_called()
    assert coordinator.hashes is None
    coordinator.begin_step(4)
    staged = []
    for layer, embedding in embeddings:
        coordinator.stage_layer(layer, 4)
        staged.append(embedding._tpu_engram_packed.clone())
    event.synchronize.assert_called_once()
    coordinator.clear_step()
    assert coordinator.deferred is None and coordinator.hashes is None

    expected_result, _ = _result([6, 7])
    expected_inputs = _reference_capture(
        [b, a], [2, 3], [1, 1], NGRAM, PendingTokens([a, b], [3, 2], expected_result)
    )
    expected_hashes = hash_on_host(expected_inputs, **_tables())
    for (layer, embedding), rows in zip(embeddings, staged, strict=True):
        expected = pack_rows(
            embedding.weight,
            embedding.scale,
            expected_hashes[:, layer],
            row_start=0,
            padded_tokens=4,
        )
        assert torch.equal(rows, expected)


def test_prefill_hashes_at_prepare_and_rejects_oversized_step():
    coordinator, _ = _coordinator()
    req = _request([1, 2, 3, 4])
    coordinator.prepare(_batch([req], starts=[0], lengths=[4]))
    assert coordinator.deferred is None and coordinator.hashes.shape[0] == 4
    with pytest.raises(ValueError, match="exceed"):
        coordinator.begin_step(2)
    coordinator.clear_step()
    coordinator.remember(
        [req], [4], SimpleNamespace(copy_done=None, next_token_ids=[5])
    )
    coordinator.prepare(_batch([req], decode=True, starts=[4]))
    assert coordinator.deferred is not None
    with pytest.raises(ValueError, match="exceed"):
        coordinator.begin_step(0)


def test_hash_on_host_and_coordinator_block_sentinel_and_image_tokens():
    tables = _tables()
    # In _tables(), token_map[0] == 3 == pad_id.
    # Once a window hits a negative sentinel (< 0), an MM pad sentinel (>= 1_000_000),
    # or image_token_id, that shift and all older predecessors map to pad_id.
    inputs_with_sentinels = HostEngramInputs(
        tokens=np.asarray(
            [
                [-1, 5, 999],
                [2, _MM_PAD_SHIFT_VALUE + 42, 777],
                [2, 4, 6],
            ],
            dtype=np.int64,
        ),
        positions=np.asarray([5, 6, 7], dtype=np.int64),
    )
    equivalent_padded = HostEngramInputs(
        tokens=np.asarray(
            [
                [0, 0, 0],
                [2, 0, 0],
                [2, 0, 0],
            ],
            dtype=np.int64,
        ),
        positions=np.asarray([5, 6, 7], dtype=np.int64),
    )
    got = hash_on_host(inputs_with_sentinels, **tables, image_token_id=4)
    expected = hash_on_host(equivalent_padded, **tables)
    assert torch.equal(got, expected)

    # An unblocked token outside token_map (< _MM_PAD_SHIFT_VALUE) still raises.
    bad_inputs = HostEngramInputs(
        tokens=np.asarray([[2, 99, 1]], dtype=np.int64),
        positions=np.asarray([3], dtype=np.int64),
    )
    with pytest.raises(ValueError, match="outside the compressed token map"):
        hash_on_host(bad_inputs, **tables, image_token_id=4)

    # Coordinator threads image_token_id and MM pad sentinels end-to-end.
    coordinator, embeddings = _coordinator(image_token_id=4)
    req = _request([1, _MM_PAD_SHIFT_VALUE + 7, 2, 4, 5])
    coordinator.prepare(_batch([req], starts=[0], lengths=[5]))
    coordinator.stage(8)
    ref_req = _request([1, 0, 2, 0, 5])
    ref_inputs = _reference_capture([ref_req], [0], [5], NGRAM)
    # Position 1 ([MM_PAD, 1, _]) blocks shift 0 and older shift 1 -> [0, 0, 0].
    # Position 2 ([2, MM_PAD, 1]) blocks shift 1 and older shift 2 -> [2, 0, 0].
    # Position 3 ([4, 2, MM_PAD]) has image_token_id=4 at shift 0 -> [0, 0, 0].
    # Position 4 ([5, 4, 2]) has image_token_id=4 at shift 1 -> [5, 0, 0].
    ref_inputs.tokens[1] = [0, 0, 0]
    ref_inputs.tokens[2] = [2, 0, 0]
    ref_inputs.tokens[3] = [0, 0, 0]
    ref_inputs.tokens[4] = [5, 0, 0]
    ref_hashes = hash_on_host(ref_inputs, **tables)
    for layer, embedding in embeddings:
        expected_packed = pack_rows(
            embedding.weight,
            embedding.scale,
            ref_hashes[:, layer],
            row_start=0,
            padded_tokens=8,
        )
        assert torch.equal(embedding._tpu_engram_packed, expected_packed)
    coordinator.clear_step()


class _FakeRawHostBuffer:
    def __init__(self, size_bytes: int):
        self._buf = np.zeros(size_bytes, dtype=np.uint8)
        self.size_bytes = size_bytes
        self.data_ptr = int(self._buf.ctypes.data)


def _fake_transport(*, tile=128, corrupt_d2h=False, defer_h2d=False):
    pending_copies = []
    synced = []

    class FakeTransfer:
        def __init__(self, device_tensor, host_buffer, unsafe_skip_buffer_lock=False):
            self.device_tensor = device_tensor
            self.host_buffer = host_buffer
            self.unsafe_skip_buffer_lock = unsafe_skip_buffer_lock

        def d2h(self):
            arr = self.host_buffer._buf.view(np.int32)
            n = self.device_tensor.numel()
            arr[:n] = self.device_tensor.numpy()
            if corrupt_d2h and n:
                arr[0] ^= 1

        def h2d_async(self):
            arr = self.host_buffer._buf.view(np.int32)
            n = self.device_tensor.numel()
            snapshot = arr[:n].copy()
            done = [False]

            def apply():
                if not done[0]:
                    self.device_tensor.copy_(torch.from_numpy(snapshot))
                    done[0] = True

            if not defer_h2d:
                apply()
            else:
                pending_copies.append(apply)
            return SimpleNamespace(wait=apply)

    raw_mod = SimpleNamespace(
        RawHostBuffer=_FakeRawHostBuffer,
        PreparedTorchRawTransfer=FakeTransfer,
    )
    layout_mod = SimpleNamespace(
        get_device_layout_if_materialized=lambda t: ([0], [[tile]], 32)
    )

    def sync_fn(tensors, wait=True):
        synced.append((list(tensors), wait))

    return raw_mod, layout_mod, sync_fn, pending_copies, synced


def test_prepared_engram_upload_dma_lifecycle_and_slot_ping_pong():
    bad_raw, layout_mod, sync_fn, _, _ = _fake_transport(corrupt_d2h=True)
    with pytest.raises(RuntimeError, match="layout validation failed"):
        PreparedEngramUpload(
            16, "cpu", raw_transfer_mod=bad_raw, layout_mod=layout_mod, sync_fn=sync_fn
        )

    raw_mod, layout_mod, sync_fn, pending_copies, synced = _fake_transport(
        tile=128, defer_h2d=True
    )
    upload = PreparedEngramUpload(
        10, "cpu", raw_transfer_mod=raw_mod, layout_mod=layout_mod, sync_fn=sync_fn
    )
    # Physical allocation rounds 10 words up to the 128-word tile (512 bytes).
    assert upload.host_buffer.size_bytes == 128 * 4
    upload.acquire().copy_(torch.arange(10, dtype=torch.int32))
    upload.start()
    # Deferred H2D has not modified device tensor until wait()/activate().
    assert not torch.equal(upload.device, torch.arange(10, dtype=torch.int32))
    upload.activate()
    assert torch.equal(upload.device, torch.arange(10, dtype=torch.int32))
    # Cannot start a new upload on an activated slot before recording its consumer.
    with pytest.raises(RuntimeError, match="Record the Engram consumer"):
        upload.start()
    with pytest.raises(ValueError, match="requires a consumer output"):
        upload.record_consumers([])
    consumer_marker = torch.tensor([99])
    upload.record_consumers([consumer_marker])
    assert len(synced) == 0
    upload.start()
    assert len(synced) == 1
    assert synced[0][0][0] is consumer_marker and synced[0][1] is True
    upload.wait()

    # Coordinator double-buffers (2 slots per layer/bucket) and pipelines next layer.
    raw_mod, layout_mod, sync_fn, pending_copies, synced = _fake_transport(
        defer_h2d=True
    )
    coordinator, embeddings = _coordinator(
        raw_transfer_mod=raw_mod, layout_mod=layout_mod, sync_fn=sync_fn
    )
    req1 = _request([1, 2, 3, 4])
    coordinator.prepare(_batch([req1], starts=[0], lengths=[4]))
    coordinator.begin_step(4)
    # begin_step pre-stages layer 0 before stage_layer(0) is called.
    assert 0 in coordinator._staged_uploads and 1 not in coordinator._staged_uploads
    coordinator.stage_layer(0, 4)
    slot0_step1 = embeddings[0][1]._tpu_engram_packed
    seg0_out = torch.tensor([10])
    coordinator.finish_segment(0, 4, [seg0_out])
    # finish_segment(0) immediately stages layer 1 while segment 0 runs.
    assert 1 in coordinator._staged_uploads
    coordinator.stage_layer(1, 4)
    slot1_step1 = embeddings[1][1]._tpu_engram_packed
    seg1_out = torch.tensor([20])
    coordinator.finish_segment(1, 4, [seg1_out])
    step1_layer0_snapshot = slot0_step1.clone()
    coordinator.clear_step()

    # Step 2 uses the alternate slot without waiting on Step 1's consumers or
    # overwriting Step 1's device buffer while Step 2's upload is in flight.
    req2 = _request([5, 6, 7, 1])
    coordinator.prepare(_batch([req2], starts=[0], lengths=[4]))
    coordinator.begin_step(4)
    assert len(synced) == 0
    assert torch.equal(slot0_step1, step1_layer0_snapshot)
    coordinator.stage_layer(0, 4)
    slot0_step2 = embeddings[0][1]._tpu_engram_packed
    assert slot0_step2 is not slot0_step1
    assert torch.equal(slot0_step1, step1_layer0_snapshot)
    coordinator.finish_segment(0, 4, [torch.tensor([30])])
    coordinator.stage_layer(1, 4)
    slot1_step2 = embeddings[1][1]._tpu_engram_packed
    assert slot1_step2 is not slot1_step1
    coordinator.finish_segment(1, 4, [torch.tensor([40])])
    coordinator.clear_step()

    # Step 3 reuses Slot 0, which now synchronizes Step 1's recorded consumers.
    req3 = _request([2, 4, 6, 0])
    coordinator.prepare(_batch([req3], starts=[0], lengths=[4]))
    coordinator.begin_step(4)
    assert len(synced) == 1 and synced[0][0][0] is seg0_out
    coordinator.stage_layer(0, 4)
    assert embeddings[0][1]._tpu_engram_packed is slot0_step1
    coordinator.finish_segment(0, 4, [torch.tensor([50])])
    assert len(synced) == 2 and synced[1][0][0] is seg1_out
    coordinator.stage_layer(1, 4)
    coordinator.finish_segment(1, 4, [torch.tensor([60])])
    coordinator.clear_step()


def test_coordinator_prefetches_next_chunk_and_handles_prefetch_miss():
    raw_mod, layout_mod, sync_fn, _, _ = _fake_transport(defer_h2d=True)
    coordinator, embeddings = _coordinator(
        raw_transfer_mod=raw_mod, layout_mod=layout_mod, sync_fn=sync_fn
    )
    prompt = [1, 2, 3, 4, 5, 6, 7, 0, 2, 4, 6, 1]
    req = _request(prompt)

    # Chunk 1: [0:4]. At finish_forward(4), Chunk 2 [4:8] is prefetched into
    # the inactive upload slots while Chunk 1's active slots stay intact.
    coordinator.prepare(_batch([req], starts=[0], lengths=[4]))
    coordinator.begin_step(4)
    for layer, embedding in embeddings:
        coordinator.stage_layer(layer, 4)
        coordinator.finish_segment(layer, 4, [embedding._tpu_engram_packed])
    chunk1_devices = [emb._tpu_engram_packed for _, emb in embeddings]
    chunk1_values = [t.clone() for t in chunk1_devices]
    coordinator.finish_forward(4)
    assert coordinator._prefetch is not None
    assert coordinator._prefetch.starts == [4]
    assert coordinator._prefetch.lengths == [4]
    assert set(coordinator._prefetch.uploads.keys()) == {0, 1}
    for dev, snap in zip(chunk1_devices, chunk1_values, strict=True):
        assert torch.equal(dev, snap)
    coordinator.clear_step()

    # Chunk 2: [4:8] matches the prefetch and reuses its hashes and staged uploads.
    prefetched_hashes = coordinator._prefetch.hashes
    prefetched_uploads = dict(coordinator._prefetch.uploads)
    coordinator.prepare(_batch([req], starts=[4], lengths=[4]))
    assert coordinator._prefetch is None
    assert coordinator.hashes is prefetched_hashes
    assert coordinator._staged_uploads == prefetched_uploads
    coordinator.begin_step(4)
    for layer, embedding in embeddings:
        coordinator.stage_layer(layer, 4)
        assert embedding._tpu_engram_packed is prefetched_uploads[layer].device
        coordinator.finish_segment(layer, 4, [embedding._tpu_engram_packed])
    coordinator.finish_forward(4)
    # Chunk 3 [8:12] is also prefetched at the end of Chunk 2.
    assert coordinator._prefetch is not None
    assert coordinator._prefetch.starts == [8]
    coordinator.clear_step()

    expected_chunk2_inputs = _reference_capture([req], [4], [4], NGRAM)
    expected_chunk2_hashes = hash_on_host(expected_chunk2_inputs, **_tables())
    for layer, embedding in embeddings:
        expected = pack_rows(
            embedding.weight,
            embedding.scale,
            expected_chunk2_hashes[:, layer],
            row_start=0,
            padded_tokens=4,
        )
        assert torch.equal(embedding._tpu_engram_packed, expected)

    # Prefetch miss: scheduling a different request discards the prefetch and
    # computes accurate rows for the actual batch.
    other_req = _request([7, 6, 5, 4])
    coordinator.prepare(_batch([other_req], starts=[0], lengths=[4]))
    assert coordinator._prefetch is None
    assert coordinator._staged_uploads == {}
    coordinator.stage(4)
    expected_other_inputs = _reference_capture([other_req], [0], [4], NGRAM)
    expected_other_hashes = hash_on_host(expected_other_inputs, **_tables())
    for layer, embedding in embeddings:
        expected = pack_rows(
            embedding.weight,
            embedding.scale,
            expected_other_hashes[:, layer],
            row_start=0,
            padded_tokens=4,
        )
        assert torch.equal(embedding._tpu_engram_packed, expected)
    coordinator.clear_step()
