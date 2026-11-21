# SPDX-License-Identifier: Apache-2.0
"""CPU tests for DSV4.1 host Engram overlap, double-buffered DMA, prefetch, and hooks."""

from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from sglang.srt.layers.engram import compute_engram_hash_ids
from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu.compilation.pending_call import PendingCall
from sglang_tpu.models.deepseek_v41 import hooks
from sglang_tpu.models.deepseek_v41.engram_host import (
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


def _request(prompt, output=(), *, beam_group=None, req_pool_idx=None):
    return SimpleNamespace(
        origin_input_ids=list(prompt),
        output_ids=list(output),
        beam_group=beam_group,
        kv=SimpleNamespace(req_pool_idx=req_pool_idx),
    )


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
    tables = _tables()
    torch_tables = {k: torch.as_tensor(v) for k, v in tables.items() if k != "pad_id"}
    for _ in range(200):
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
        host_hashed = hash_on_host(resolved, **tables)
        assert torch.equal(host_hashed, hash_on_host(stock, **tables))
        # Verify exact agreement with core torch compute_engram_hash_ids.
        if len(resolved.positions):
            blocked = torch.as_tensor(resolved.positions)[:, None] < torch.arange(NGRAM)
            core_hashed = compute_engram_hash_ids(
                torch.as_tensor(resolved.tokens),
                blocked,
                tables["pad_id"],
                torch_tables["token_map"],
                torch_tables["multipliers"],
                torch_tables["primes"],
                torch_tables["offsets"],
            )
            assert torch.equal(host_hashed, core_hashed)


def test_capture_rejects_missing_token_like_blocking_capture():
    req = _request([1, 2], [3])
    with pytest.raises(RuntimeError, match="not available on host"):
        capture_inputs([req], [3], [1], NGRAM, None)
    with pytest.raises(ValueError, match="non-negative"):
        capture_inputs([req], [-1], [1], NGRAM, None)
    result, _ = _result([5])
    pending = PendingTokens([_request([1])], [1], result)
    deferred = capture_inputs([req], [3], [1], NGRAM, pending)
    with pytest.raises(RuntimeError, match="not available on host"):
        deferred.resolve(pending)


def _table(rows, dim=64, seed=1):
    gen = torch.Generator().manual_seed(seed)
    weight = torch.randint(0, 127, (rows, dim), dtype=torch.uint8, generator=gen)
    scale = torch.randint(100, 140, (rows, dim // 32), dtype=torch.uint8, generator=gen)
    return weight.view(torch.float8_e4m3fn), scale.view(torch.float8_e8m0fnu)


def _coordinator(
    *, image_token_id=None, batch_transfer_mod=None, layout_mod=None, sync_fn=None
):
    tables = _tables()
    hasher = SimpleNamespace(
        max_ngram_size=NGRAM,
        **{k: torch.as_tensor(v) for k, v in tables.items() if k != "pad_id"},
        pad_id=tables["pad_id"],
        image_token_id=image_token_id,
        history=None,
    )
    embeddings = []
    for layer in (0, 1):
        weight, scale = _table(rows=130, seed=layer + 3)
        embeddings.append(
            (
                layer,
                SimpleNamespace(
                    weight=weight,
                    scale=scale,
                    dim=weight.shape[1],
                    row_start=0,
                    _shared=True,
                    host_table=SimpleNamespace(layout="shared"),
                    tpu_hash_index=layer,
                ),
            )
        )
    return (
        HostEngramCoordinator(
            hasher,
            embeddings,
            "cpu",
            batch_transfer_mod=batch_transfer_mod,
            layout_mod=layout_mod,
            sync_fn=sync_fn,
        ),
        embeddings,
        hasher,
    )


def _batch(
    requests,
    *,
    decode=False,
    idle=False,
    starts=None,
    lengths=None,
    spec_none=True,
    req_pool_indices_cpu=None,
    seq_lens_cpu=None,
):
    mode = SimpleNamespace(
        is_decode=lambda: decode,
        is_extend_without_speculative=lambda: not decode and not idle,
        is_extend=lambda: not decode and not idle,
        is_target_verify=lambda: False,
        is_idle=lambda: idle,
    )
    if seq_lens_cpu is None and decode and starts is not None:
        seq_lens_cpu = torch.tensor([s + 1 for s in starts])
    return SimpleNamespace(
        reqs=requests,
        spec_algorithm=SimpleNamespace(is_none=lambda: spec_none),
        forward_mode=mode,
        prefix_lens=starts,
        extend_lens=lengths,
        seq_lens_cpu=seq_lens_cpu,
        req_pool_indices_cpu=req_pool_indices_cpu,
    )


def test_coordinator_decode_waits_at_first_stage_and_stages_reference_rows():
    coordinator, embeddings, _ = _coordinator()
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
    staged, uploads, num_cols = coordinator.stage_all(4)
    assert num_cols == 4
    assert len(staged) == 2 and len(uploads) == 2
    event.synchronize.assert_called_once()
    staged_copies = [t.clone() for t in staged]
    coordinator.record_consumers(uploads, [torch.tensor([1])])
    coordinator.clear_step()
    assert coordinator.deferred is None and coordinator.hashes is None

    expected_result, _ = _result([6, 7])
    expected_inputs = _reference_capture(
        [b, a], [2, 3], [1, 1], NGRAM, PendingTokens([a, b], [3, 2], expected_result)
    )
    expected_hashes = hash_on_host(expected_inputs, **_tables())
    for (layer, embedding), rows in zip(embeddings, staged_copies, strict=True):
        expected = pack_rows(
            embedding.weight,
            embedding.scale,
            expected_hashes[:, layer],
            row_start=0,
            padded_tokens=4,
        )
        assert torch.equal(rows, expected)


def test_prefill_hashes_at_prepare_and_rejects_oversized_step():
    coordinator, _, _ = _coordinator()
    req = _request([1, 2, 3, 4])
    coordinator.prepare(_batch([req], starts=[0], lengths=[4]))
    assert coordinator.deferred is None and coordinator.hashes.shape[0] == 4
    with pytest.raises(ValueError, match="exceed"):
        coordinator.stage_all(2)
    coordinator.clear_step()
    coordinator.remember(
        [req], [4], SimpleNamespace(copy_done=None, next_token_ids=[5])
    )
    coordinator.prepare(_batch([req], decode=True, starts=[4]))
    assert coordinator.deferred is not None
    with pytest.raises(ValueError, match="exceed"):
        coordinator.stage_all(0)


def test_hash_on_host_and_coordinator_block_sentinel_and_image_tokens():
    tables = _tables()
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

    bad_inputs = HostEngramInputs(
        tokens=np.asarray([[2, 99, 1]], dtype=np.int64),
        positions=np.asarray([3], dtype=np.int64),
    )
    with pytest.raises(ValueError, match="outside the compressed token map"):
        hash_on_host(bad_inputs, **tables, image_token_id=4)

    coordinator, embeddings, _ = _coordinator(image_token_id=4)
    req = _request([1, _MM_PAD_SHIFT_VALUE + 7, 2, 4, 5])
    coordinator.prepare(_batch([req], starts=[0], lengths=[5]))
    staged, uploads, _ = coordinator.stage_all(8)
    coordinator.record_consumers(uploads, [torch.tensor([1])])
    ref_req = _request([1, 0, 2, 0, 5])
    ref_inputs = _reference_capture([ref_req], [0], [5], NGRAM)
    ref_inputs.tokens[1] = [0, 0, 0]
    ref_inputs.tokens[2] = [2, 0, 0]
    ref_inputs.tokens[3] = [0, 0, 0]
    ref_inputs.tokens[4] = [5, 0, 0]
    ref_hashes = hash_on_host(ref_inputs, **tables)
    for (layer, embedding), dev_packed in zip(embeddings, staged, strict=True):
        expected_packed = pack_rows(
            embedding.weight,
            embedding.scale,
            ref_hashes[:, layer],
            row_start=0,
            padded_tokens=8,
        )
        assert torch.equal(dev_packed, expected_packed)
    coordinator.clear_step()


def _fake_transport(*, tile=128, corrupt_d2h=False, defer_h2d=False):
    pending_copies = []
    synced = []
    events = []

    def _phys_bytes(t: torch.Tensor) -> int:
        return (-(-t.numel() // tile) * tile) * 4

    def batch_transfer_d2h_sync(src_arrs, dst_arrs):
        assert len(src_arrs) == len(dst_arrs) == 1
        src, dst = src_arrs[0], dst_arrs[0]
        assert dst.device.type == "cpu" and dst.is_contiguous()
        assert dst.nbytes >= _phys_bytes(src)
        n = src.numel()
        dst[:n].copy_(src)
        if corrupt_d2h and n:
            dst[0] ^= 1

    def batch_transfer_h2d(src_arrs, dst_arrs):
        assert len(src_arrs) == len(dst_arrs) == 1
        src, dst = src_arrs[0], dst_arrs[0]
        assert src.device.type == "cpu" and src.is_contiguous()
        assert src.nbytes >= _phys_bytes(dst)
        events.append("h2d_async")
        n = dst.numel()
        snapshot = src[:n].clone()
        done = [False]

        def apply():
            if not done[0]:
                dst.copy_(snapshot)
                done[0] = True

        if not defer_h2d:
            apply()
        else:
            pending_copies.append(apply)
        return SimpleNamespace(wait=apply)

    bt_mod = SimpleNamespace(
        batch_transfer_d2h_sync=batch_transfer_d2h_sync,
        batch_transfer_h2d=batch_transfer_h2d,
    )
    layout_mod = SimpleNamespace(
        get_device_layout_if_materialized=lambda t: ([0], [[tile]], 32)
    )

    def sync_fn(tensors, wait=True):
        events.append(("sync", wait))
        synced.append((list(tensors), wait))

    return bt_mod, layout_mod, sync_fn, pending_copies, synced, events


def test_prepared_engram_upload_dma_lifecycle_and_slot_ping_pong():
    bad_bt, layout_mod, sync_fn, _, _, _ = _fake_transport(corrupt_d2h=True)
    with pytest.raises(RuntimeError, match="layout validation failed"):
        PreparedEngramUpload(
            16, "cpu", batch_transfer_mod=bad_bt, layout_mod=layout_mod, sync_fn=sync_fn
        )

    bt_mod, layout_mod, sync_fn, _, synced, _ = _fake_transport(
        tile=128, defer_h2d=True
    )
    upload = PreparedEngramUpload(
        10, "cpu", batch_transfer_mod=bt_mod, layout_mod=layout_mod, sync_fn=sync_fn
    )
    # Physical allocation rounds 10 words up to the 128-word tile (512 bytes),
    # and host_words is a zero-copy prefix view over host_buffer.
    assert upload.host_buffer.nbytes == 128 * 4
    assert upload.host_words.data_ptr() == upload.host_buffer.data_ptr()
    upload.acquire().copy_(torch.arange(10, dtype=torch.int32))
    upload.start()
    assert not torch.equal(upload.device, torch.arange(10, dtype=torch.int32))
    upload.activate()
    assert torch.equal(upload.device, torch.arange(10, dtype=torch.int32))
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

    # Coordinator double-buffers (2 slots per layer/bucket) across steps 1/2/3.
    bt_mod, layout_mod, sync_fn, _, synced, _ = _fake_transport(defer_h2d=True)
    coordinator, _, _ = _coordinator(
        batch_transfer_mod=bt_mod, layout_mod=layout_mod, sync_fn=sync_fn
    )
    req1 = _request([1, 2, 3, 4])
    coordinator.prepare(_batch([req1], starts=[0], lengths=[4]))
    staged1, uploads1, _ = coordinator.stage_all(4)
    slot0_step1, slot1_step1 = staged1
    step1_out = torch.tensor([10])
    coordinator.record_consumers(uploads1, [step1_out])
    step1_layer0_snapshot = slot0_step1.clone()
    coordinator.clear_step()

    # Step 2 uses alternate slots without waiting on Step 1's consumers.
    req2 = _request([5, 6, 7, 1])
    coordinator.prepare(_batch([req2], starts=[0], lengths=[4]))
    staged2, uploads2, _ = coordinator.stage_all(4)
    assert len(synced) == 0
    assert staged2[0] is not slot0_step1 and staged2[1] is not slot1_step1
    assert torch.equal(slot0_step1, step1_layer0_snapshot)
    step2_out = torch.tensor([20])
    coordinator.record_consumers(uploads2, [step2_out])
    coordinator.clear_step()

    # Step 3 reuses Slot 0 for both layers, synchronizing Step 1's consumer.
    req3 = _request([2, 4, 6, 0])
    coordinator.prepare(_batch([req3], starts=[0], lengths=[4]))
    staged3, uploads3, _ = coordinator.stage_all(4)
    assert len(synced) == 2
    assert all(s[0][0] is step1_out and s[1] is True for s in synced)
    assert staged3[0] is slot0_step1 and staged3[1] is slot1_step1
    coordinator.record_consumers(uploads3, [torch.tensor([30])])
    coordinator.clear_step()

    # Switching buckets does NOT stall on the previous bucket's in-flight consumers;
    # consumer waits happen lazily in start() only when a slot is actually reused.
    req4 = _request([1, 2])
    coordinator.prepare(_batch([req4], starts=[0], lengths=[2]))
    _, uploads4, _ = coordinator.stage_all(8)
    assert len(synced) == 2  # no cross-bucket wait
    coordinator.record_consumers(uploads4, [torch.tensor([40])])
    coordinator.clear_step()

    # Returning to bucket 4 picks the older alternate slot (Step 2) and waits on
    # Step 2's consumer in start(), leaving Step 3's consumer untouched until next reuse.
    req5 = _request([3, 4, 5, 6])
    coordinator.prepare(_batch([req5], starts=[0], lengths=[4]))
    staged5, uploads5, _ = coordinator.stage_all(4)
    assert len(synced) == 4
    assert all(s[0][0] is step2_out and s[1] is True for s in synced[2:])
    assert staged5[0] is not slot0_step1 and staged5[1] is not slot1_step1
    coordinator.record_consumers(uploads5, [torch.tensor([50])])
    coordinator.clear_step()


def test_coordinator_prefetches_next_chunk_and_handles_prefetch_miss():
    bt_mod, layout_mod, sync_fn, _, _, _ = _fake_transport(defer_h2d=True)
    coordinator, embeddings, _ = _coordinator(
        batch_transfer_mod=bt_mod, layout_mod=layout_mod, sync_fn=sync_fn
    )
    prompt = [1, 2, 3, 4, 5, 6, 7, 0, 2, 4, 6, 1]
    req = _request(prompt)

    # Chunk 1: [0:4]. At finish_forward(4), Chunk 2 [4:8] is prefetched into
    # the inactive upload slots while Chunk 1's active slots stay intact.
    coordinator.prepare(_batch([req], starts=[0], lengths=[4]))
    staged1, uploads1, _ = coordinator.stage_all(4)
    chunk1_values = [t.clone() for t in staged1]
    coordinator.record_consumers(uploads1, [torch.tensor([1])])
    coordinator.finish_forward(4)
    assert coordinator._prefetch is not None
    assert coordinator._prefetch.starts == [4]
    assert coordinator._prefetch.lengths == [4]
    assert set(coordinator._prefetch.uploads.keys()) == {0, 1}
    for dev, snap in zip(staged1, chunk1_values, strict=True):
        assert torch.equal(dev, snap)
    coordinator.clear_step()

    # Chunk 2: [4:8] matches the prefetch and reuses its hashes and staged uploads.
    prefetched_hashes = coordinator._prefetch.hashes
    prefetched_uploads = dict(coordinator._prefetch.uploads)
    coordinator.prepare(_batch([req], starts=[4], lengths=[4]))
    assert coordinator._prefetch is None
    assert coordinator.hashes is prefetched_hashes
    assert coordinator._staged_uploads == prefetched_uploads
    staged2, uploads2, _ = coordinator.stage_all(4)
    for layer, dev in enumerate(staged2):
        assert dev is prefetched_uploads[layer].device
    coordinator.record_consumers(uploads2, [torch.tensor([2])])
    coordinator.finish_forward(4)
    # Chunk 3 [8:12] is also prefetched at the end of Chunk 2.
    assert coordinator._prefetch is not None
    assert coordinator._prefetch.starts == [8]
    coordinator.clear_step()

    expected_chunk2_inputs = _reference_capture([req], [4], [4], NGRAM)
    expected_chunk2_hashes = hash_on_host(expected_chunk2_inputs, **_tables())
    for (layer, embedding), dev in zip(embeddings, staged2, strict=True):
        expected = pack_rows(
            embedding.weight,
            embedding.scale,
            expected_chunk2_hashes[:, layer],
            row_start=0,
            padded_tokens=4,
        )
        assert torch.equal(dev, expected)

    # Prefetch miss: scheduling a different request discards the prefetch.
    other_req = _request([7, 6, 5, 4])
    coordinator.prepare(_batch([other_req], starts=[0], lengths=[4]))
    assert coordinator._prefetch is None
    assert coordinator._staged_uploads == {}
    staged_other, uploads_other, _ = coordinator.stage_all(4)
    coordinator.record_consumers(uploads_other, [torch.tensor([3])])
    expected_other_inputs = _reference_capture([other_req], [0], [4], NGRAM)
    expected_other_hashes = hash_on_host(expected_other_inputs, **_tables())
    for (layer, embedding), dev in zip(embeddings, staged_other, strict=True):
        expected = pack_rows(
            embedding.weight,
            embedding.scale,
            expected_other_hashes[:, layer],
            row_start=0,
            padded_tokens=4,
        )
        assert torch.equal(dev, expected)
    coordinator.clear_step()


@dataclass
class _FakeLogitsOutput:
    next_token_logits: torch.Tensor


def test_hooks_end_to_end_overlap_fallback_idle_and_pending_call():
    bt_mod, layout_mod, sync_fn, _, synced, events = _fake_transport(defer_h2d=True)
    coordinator, embeddings, hasher = _coordinator(
        batch_transfer_mod=bt_mod, layout_mod=layout_mod, sync_fn=sync_fn
    )
    hasher.token_map = hasher.token_map.cpu()

    call_log = []

    def fake_embed_tokens(ids):
        call_log.append(("embed", ids.clone()))
        return torch.zeros(ids.shape[0], 8, dtype=torch.bfloat16)

    layers = [
        SimpleNamespace(engram=SimpleNamespace(layer_hash_index=idx, embed=emb))
        for idx, emb in embeddings
    ]
    model = SimpleNamespace(
        model=SimpleNamespace(layers=layers, engram_hasher=hasher),
        vision=None,
        get_input_embeddings=lambda: fake_embed_tokens,
        _tpu_host_engram_coordinator=coordinator,
    )
    hf_config = SimpleNamespace(
        model_type="deepseek_v41",
        vision_n_layers=0,
        vocab_size=8,
        image_token_id=None,
        engram_pad_token_id=3,
    )
    runner = SimpleNamespace(
        model=model,
        model_config=SimpleNamespace(hf_config=hf_config, use_engram=True),
        is_draft_worker=False,
        device="cpu",
    )
    from sglang_tpu.compilation.compiled_forward_shim import CompiledForwardShim
    from sglang_tpu.models.deepseek_v41.compile_policy import (
        DeepseekV4DecoderTpuCompilePolicy,
        _EngramCompiledForwardShim,
    )

    policy = object.__new__(DeepseekV4DecoderTpuCompilePolicy)
    policy.model_runner = runner
    policy._engram_hasher = hasher
    policy._embeds_outside_graph = True
    shim = object.__new__(_EngramCompiledForwardShim)
    shim.policy = policy
    worker = SimpleNamespace(model_runner=runner)

    def make_fb(mode, n_tokens):
        return SimpleNamespace(
            input_ids=torch.zeros(n_tokens, dtype=torch.int64),
            positions=torch.arange(n_tokens, dtype=torch.int64),
            req_pool_indices=torch.zeros(1, dtype=torch.int64),
            out_cache_loc=torch.arange(n_tokens, dtype=torch.int64),
            extend_seq_lens=torch.tensor([n_tokens], dtype=torch.int64),
            extend_start_loc=torch.tensor([0], dtype=torch.int64),
            engram_history=None,
            spec_info=None,
            input_embeds=None,
            mm_inputs=None,
            forward_mode=mode,
            tpu_engram_local_ids=None,
            tpu_engram_staged=None,
        )

    fake_parallel = SimpleNamespace(
        pp_size=1, attn_cp_size=1, attn_dcp_size=1, attn_dp_size=2
    )
    fake_exec = SimpleNamespace(
        features=SimpleNamespace(enable_encoder_swa_bounded_replay=False)
    )

    with (
        patch("sglang.srt.runtime_context.get_parallel", return_value=fake_parallel),
        patch("sglang.srt.runtime_context.get_exec", return_value=fake_exec),
    ):
        # 1. Chunked prefill: flush_consumers(wait=False) happens BEFORE prefetch h2d_async.
        prompt_req = _request([1, 2, 3, 4, 5, 6, 7, 0])
        chunk1_batch = _batch([prompt_req], starts=[0], lengths=[4])
        chunk1_fb = make_fb(chunk1_batch.forward_mode, 4)

        def run_shim(fb, fn, *, unpad_idle=False):
            policy._embed_outside_graph(fb)

            def orig_shim(s, wrapped_fn, ids, pos, fbatch, *a, **kw):
                out = wrapped_fn(ids, pos, fbatch)
                if unpad_idle and not isinstance(out, PendingCall):
                    # Simulate IDLE DP unpadding slicing logits to [:0].
                    return _FakeLogitsOutput(
                        next_token_logits=out.next_token_logits[:0]
                    )
                return out

            with patch.object(CompiledForwardShim, "_forward", orig_shim):
                return shim._forward(fn, fb.input_ids, fb.positions, fb)

        events.clear()
        raw_logits_1 = torch.ones(4, 8)

        def worker_orig_chunk1(w, b):
            run_shim(
                chunk1_fb,
                lambda ids, pos, fb: _FakeLogitsOutput(next_token_logits=raw_logits_1),
            )
            return SimpleNamespace(copy_done=None, next_token_ids=[5])

        hooks.tpu_dsv41_engram_worker_forward(worker_orig_chunk1, worker, chunk1_batch)
        # Flush (sync, False) must precede the chunk-2 prefetch h2d_async calls.
        flush_idx = events.index(("sync", False))
        assert events[flush_idx + 1 :] == ["h2d_async", "h2d_async"]
        assert coordinator._prefetch is not None and coordinator._prefetch.starts == [4]
        assert chunk1_fb.tpu_engram_staged is None
        assert chunk1_fb.input_embeds is not None

        # 2. Overlap decode: embed_tokens runs BEFORE copy_done.synchronize(),
        #    and clamps negative/MM sentinel ids to fallback_id (3).
        dec_req = _request([1, 2, 3])
        prefill_b = _batch([dec_req], starts=[0], lengths=[3])
        res_pending, copy_event = _result([6])

        def track_sync():
            call_log.append(("copy_done_sync", None))
            res_pending.next_token_ids = torch.tensor([6])

        copy_event.synchronize.side_effect = track_sync
        hooks.tpu_dsv41_engram_worker_forward(
            lambda w, b: res_pending, worker, prefill_b
        )
        dec_b = _batch([dec_req], decode=True, starts=[3])
        dec_fb = make_fb(dec_b.forward_mode, 4)
        dec_fb.input_ids = torch.tensor(
            [6, -1, _MM_PAD_SHIFT_VALUE + 5, 2], dtype=torch.int64
        )
        call_log.clear()

        def worker_orig_decode(w, b):
            run_shim(
                dec_fb,
                lambda ids, pos, fb: _FakeLogitsOutput(
                    next_token_logits=torch.ones(4, 8)
                ),
            )
            return SimpleNamespace(copy_done=None, next_token_ids=[7])

        hooks.tpu_dsv41_engram_worker_forward(worker_orig_decode, worker, dec_b)
        assert [tag for tag, _ in call_log] == ["embed", "copy_done_sync"]
        assert torch.equal(
            call_log[0][1], torch.tensor([6, 3, 3, 2], dtype=torch.int64)
        )

        # 3. IDLE DP batch: even though shim unpads output to [:0], wrapped_fn
        #    records the un-sliced raw logits tensor as slot consumer.
        idle_b = _batch([], idle=True)
        idle_fb = make_fb(idle_b.forward_mode, 4)
        raw_idle_logits = torch.full((4, 8), 42.0)

        def worker_orig_idle(w, b):
            out = run_shim(
                idle_fb,
                lambda ids, pos, fb: _FakeLogitsOutput(
                    next_token_logits=raw_idle_logits
                ),
                unpad_idle=True,
            )
            assert out.next_token_logits.shape[0] == 0
            return SimpleNamespace(copy_done=None, next_token_ids=torch.empty(0))

        hooks.tpu_dsv41_engram_worker_forward(worker_orig_idle, worker, idle_b)
        for upload in coordinator._active_uploads.values():
            assert upload.consumers == [raw_idle_logits]

        # 4. PendingCall (warmup/precompile): queuing 3 calls on the same bucket
        #    before calling wait() does not fail needs_consumer, and wait() re-pins
        #    staged fields and records real consumers.
        warm_mode = SimpleNamespace(
            is_decode=lambda: False,
            is_extend_without_speculative=lambda: False,
            is_extend=lambda: False,
            is_target_verify=lambda: False,
            is_idle=lambda: True,
        )
        pendings = []
        observed_staged = []
        for idx in range(3):
            w_fb = make_fb(warm_mode, 4)
            marker = torch.full((4, 8), float(idx + 1))

            def make_pending_fn(expected_marker, fb_ref):
                def _fn(ids, pos, fb):
                    def _exec():
                        assert fb_ref.tpu_engram_staged is not None
                        assert fb_ref.tpu_engram_local_ids is not None
                        observed_staged.append(fb_ref.tpu_engram_staged)
                        return _FakeLogitsOutput(next_token_logits=expected_marker)

                    return PendingCall(_exec)

                return _fn

            p = run_shim(w_fb, make_pending_fn(marker, w_fb))
            assert isinstance(p, PendingCall)
            # Restored immediately after shim returns.
            assert w_fb.tpu_engram_staged is None
            pendings.append((p, marker, w_fb))

        for p, marker, w_fb in pendings:
            res = p.wait()
            assert res.next_token_logits is marker
            assert w_fb.tpu_engram_staged is None
        assert len(observed_staged) == 3

        # 5. Folded CED extend fallback still populates PendingTokens so the
        #    following overlap decode step resolves its hole cleanly.
        ced_exec = SimpleNamespace(
            features=SimpleNamespace(enable_encoder_swa_bounded_replay=True)
        )
        ced_req = _request([1, 2, 3])
        ced_b = _batch([ced_req], starts=[0], lengths=[3])
        ced_res, ced_event = _result([4])
        with patch("sglang.srt.runtime_context.get_exec", return_value=ced_exec):
            hooks.tpu_dsv41_engram_worker_forward(lambda w, b: ced_res, worker, ced_b)
            assert not coordinator.started
        # Next decode step resolves token 4 from ced_res via PendingTokens.
        dec_after_ced = _batch([ced_req], decode=True, starts=[3])
        coordinator.prepare_if_supported(dec_after_ced, parallel=fake_parallel)
        ced_event.synchronize.assert_not_called()
        staged_ced, uploads_ced, _ = coordinator.stage_all(4)
        ced_event.synchronize.assert_called_once()
        coordinator.record_consumers(uploads_ced, [torch.tensor([1])])
        coordinator.clear_step()
        assert len(staged_ced) == 2


def test_fallback_decode_updates_pending_and_clears_stale_pending():
    coordinator, _, hasher = _coordinator()
    model = SimpleNamespace(
        model=SimpleNamespace(engram_hasher=hasher, layers=[]),
        _tpu_host_engram_coordinator=coordinator,
    )
    worker = SimpleNamespace(
        model_runner=SimpleNamespace(
            model_config=SimpleNamespace(use_engram=True),
            is_draft_worker=False,
            model=model,
            device=torch.device("cpu"),
        )
    )
    fake_parallel = SimpleNamespace(pp_size=1, attn_cp_size=1, attn_dcp_size=1)
    fake_exec = SimpleNamespace(
        features=SimpleNamespace(enable_encoder_swa_bounded_replay=False)
    )

    req_a = _request([1, 2, 3])
    beam_req = _request([4, 5, 6, 7], beam_group=object())

    with (
        patch("sglang.srt.runtime_context.get_parallel", return_value=fake_parallel),
        patch("sglang.srt.runtime_context.get_exec", return_value=fake_exec),
    ):
        # Step N-2: fast-path extend on req_a -> samples token 5 at position 3.
        ext_b = _batch([req_a], starts=[0], lengths=[3])
        res_n2, _ = _result([5])
        hooks.tpu_dsv41_engram_worker_forward(lambda w, b: res_n2, worker, ext_b)

        # Step N-1: fallback decode sharing a batch with a beam_group request
        # (including appended beam_tail rows in seq_lens_cpu).
        # Under overlap, token 5 is now committed to req_a.output_ids? Not yet for
        # step N-1, and step N-1 samples token 6 at position 4 (seq_len=4).
        # Next_token_ids is sliced to base rows [beam_req, req_a] -> [7, 6].
        beam_dec_b = _batch(
            [beam_req, req_a],
            decode=True,
            seq_lens_cpu=torch.tensor([5, 4, 5, 5]),
        )
        res_n1, ev_n1 = _result([7, 6])
        hooks.tpu_dsv41_engram_worker_forward(lambda w, b: res_n1, worker, beam_dec_b)
        assert not coordinator.started

        # Step N: beam_req finished; req_a returns to fast-path decode at start=4.
        # Under overlap, req_a.output_ids has step N-2's token [5] (known_end=4),
        # while step N-1's token 6 (pos 4) is still in flight in res_n1.
        req_a.output_ids.append(5)
        dec_n = _batch([req_a], decode=True, starts=[4])
        prepared = coordinator.prepare_if_supported(dec_n, parallel=fake_parallel)
        assert prepared == ([req_a], [5])
        ev_n1.synchronize.assert_not_called()
        hashes = coordinator.resolve()
        ev_n1.synchronize.assert_called_once()
        expected = hash_on_host(
            HostEngramInputs(
                tokens=np.asarray([[6, 5, 3]], dtype=np.int64),
                positions=np.asarray([4], dtype=np.int64),
            ),
            **_tables(),
        )
        assert torch.equal(hashes, expected)
        coordinator.clear_step()

        # Non-recordable fallback (e.g. verify or missing seq_lens_cpu) clears
        # stale PendingTokens rather than keeping an old step's positions.
        assert coordinator.pending is not None
        hooks.tpu_dsv41_engram_worker_forward(
            lambda w, b, is_verify=False: SimpleNamespace(
                copy_done=None, next_token_ids=None
            ),
            worker,
            dec_n,
            is_verify=True,
        )
        assert coordinator.pending is None

        coordinator.remember([req_a], [5], res_n1)
        no_lens_b = _batch([beam_req], decode=True, seq_lens_cpu=None)
        no_lens_b.seq_lens_cpu = None
        hooks.tpu_dsv41_engram_worker_forward(lambda w, b: res_n1, worker, no_lens_b)
        assert coordinator.pending is None


def test_fast_path_commits_hasher_history_for_fallback_decode():
    from sglang.srt.layers.engram import EngramHasher
    from sglang_tpu.models.deepseek_v41.engram_host import host_hash_ids

    tables = _tables()

    def make_real_hasher():
        # Build a lightweight EngramHasher without loading an HF tokenizer.
        h = EngramHasher.__new__(EngramHasher)
        torch.nn.Module.__init__(h)
        h.max_ngram_size = NGRAM
        h.pad_id = tables["pad_id"]
        h.image_token_id = 7
        for k in ("token_map", "multipliers", "primes", "offsets"):
            h.register_buffer(k, torch.as_tensor(tables[k], dtype=torch.int64))
        h.history = None
        h.pad_row = 0
        h.init_history(num_req_slots=16, device="cpu")
        return h

    fast_hasher = make_real_hasher()
    ref_hasher = make_real_hasher()
    # Poison fast_hasher.history with stale tokens from prior slot occupants.
    fast_hasher.history.fill_(5)
    # Slot 9 belongs to a beam request that stays on the fallback path.
    fast_hasher.history[9] = torch.tensor([2, 6], dtype=torch.int32)
    ref_hasher.history[9] = torch.tensor([2, 6], dtype=torch.int32)

    embeddings = []
    for layer in (0, 1):
        weight, scale = _table(rows=130, seed=layer + 3)
        embeddings.append(
            (
                layer,
                SimpleNamespace(
                    weight=weight,
                    scale=scale,
                    dim=weight.shape[1],
                    row_start=0,
                    _shared=True,
                    host_table=SimpleNamespace(layout="shared"),
                    tpu_hash_index=layer,
                ),
            )
        )
    coordinator = HostEngramCoordinator(fast_hasher, embeddings, "cpu")

    # 4 requests exercising: normal prompt (A), 1-token prompt pos<n-1 (B),
    # 2-chunk prefill with prefetch hit (C), and wide 62-bit MM pad sentinel (D).
    wide_mm_pad = (1 << 50) + 4
    req_a = _request([1, 2, 3, 4], req_pool_idx=1)
    req_b = _request([6], req_pool_idx=2)
    req_c = _request([0, 1, 2, 3, 4, 5, 6, 7], req_pool_idx=3)
    req_d = _request([1, wide_mm_pad, 5], req_pool_idx=4)
    slots = [1, 2, 3, 4]

    # Step 1: fast-path prefill for A, B, D + chunk 1 of C ([0:4]).
    b1 = _batch(
        [req_a, req_b, req_c, req_d],
        starts=[0, 0, 0, 0],
        lengths=[4, 1, 4, 3],
        req_pool_indices_cpu=torch.tensor(slots, dtype=torch.int64),
    )
    coordinator.prepare(b1)
    _, uploads1, _ = coordinator.stage_all(16)
    coordinator.record_consumers(uploads1, [torch.tensor([1])])
    coordinator.clear_step()

    # Step 2: chunked prefill on C: chunk 1 ([0:4]) -> prefetch -> chunk 2 ([4:8]).
    b_c1 = _batch(
        [req_c],
        starts=[0],
        lengths=[4],
        req_pool_indices_cpu=torch.tensor([3], dtype=torch.int64),
    )
    coordinator.prepare(b_c1)
    _, uploads_c1, _ = coordinator.stage_all(4)
    coordinator.record_consumers(uploads_c1, [torch.tensor([2])])
    coordinator.finish_forward(4)
    coordinator.clear_step()

    b_c2 = _batch(
        [req_c],
        starts=[4],
        lengths=[4],
        req_pool_indices_cpu=torch.tensor([3], dtype=torch.int64),
    )
    coordinator.prepare(b_c2)
    assert coordinator._staged_uploads  # prefetch hit
    _, uploads_c2, _ = coordinator.stage_all(4)
    coordinator.record_consumers(uploads_c2, [torch.tensor([3])])
    coordinator.clear_step()

    # Step 3: fast-path overlap decode on A (token 7 at pos 4) and B (token 2 at pos 1),
    # exercising the req.kv.req_pool_idx fallback when req_pool_indices_cpu is None.
    res_prev, _ = _result([7, 2])
    coordinator.remember([req_a, req_b], [4, 1], res_prev)
    b_dec_fast = _batch(
        [req_a, req_b],
        decode=True,
        starts=[4, 1],
        req_pool_indices_cpu=None,
    )
    coordinator.prepare(b_dec_fast)
    _, uploads_df, _ = coordinator.stage_all(4)
    coordinator.record_consumers(uploads_df, [torch.tensor([4])])
    coordinator.clear_step()

    # Set up reference hasher history identically from the true token histories:
    # A ended at [..., 4, 7]; B ended at [0, 6, 2] -> last 2 = [6, 2];
    # C ended at [..., 6, 7]; D ended at [..., wide_mm_pad, 5] (clamped to MM_PAD).
    ref_hasher.history[1] = torch.tensor([4, 7], dtype=torch.int32)
    ref_hasher.history[2] = torch.tensor([6, 2], dtype=torch.int32)
    ref_hasher.history[3] = torch.tensor([6, 7], dtype=torch.int32)
    ref_hasher.history[4] = torch.tensor(
        ref_hasher.history_row([wide_mm_pad, 5]), dtype=torch.int32
    )

    # Step 4: fallback decode batch (e.g. beam request at slot 9 joined A, B, C, D).
    decode_mode = SimpleNamespace(
        is_decode=lambda: True,
        is_target_verify=lambda: False,
        is_extend=lambda: False,
        # The merged EngramHasher checks is_idle() first.
        is_idle=lambda: False,
    )
    fallback_fb = SimpleNamespace(
        forward_mode=decode_mode,
        spec_info=None,
        positions=torch.tensor([5, 2, 8, 3, 10, 0], dtype=torch.int64),
        req_pool_indices=torch.tensor([1, 2, 3, 4, 9, 0], dtype=torch.int64),
        out_cache_loc=torch.tensor([10, 11, 12, 13, 14, 0], dtype=torch.int64),
        extend_seq_lens=None,
        extend_start_loc=None,
        engram_history=None,
    )
    next_ids = torch.tensor([3, 5, 1, 2, 4, 0], dtype=torch.int64)
    fast_out = host_hash_ids(fast_hasher, next_ids, fallback_fb)
    ref_out = host_hash_ids(ref_hasher, next_ids, fallback_fb)
    assert torch.equal(fast_out, ref_out)
    assert torch.equal(
        fast_hasher.history[[1, 2, 3, 4, 9]], ref_hasher.history[[1, 2, 3, 4, 9]]
    )


def test_direct_tracer_carrying_packed_engram_tensors_end_to_end():
    """Direct tracer carries packed int32 staged tensors via attention metadata."""
    from unittest import mock

    from sglang_tpu.models.deepseek_v41 import compile_policy, hooks
    from sglang_tpu.models.deepseek_v41.engram_host import (
        local_row_ids,
        pack_rows,
        unpack_rows,
    )

    weight, scale = _table(rows=40, seed=19)
    indices = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.int64)
    packed = pack_rows(weight, scale, indices, row_start=0, padded_tokens=4)
    ids = local_row_ids(4, 1, 4, "cpu")
    metadata = SimpleNamespace(engram=None)
    ctx = SimpleNamespace(attn_backend=SimpleNamespace(forward_metadata=metadata))
    embedding = SimpleNamespace(
        dim=weight.shape[1],
        tpu_hash_index=0,
        host_table=SimpleNamespace(layout="shared"),
    )

    def fake_compiled_fn(input_ids, positions, forward_batch):
        # Direct tracer rebuilds ForwardBatch from dataclass fields only,
        # dropping the non-dataclass staging attributes.
        rebuilt = SimpleNamespace(tpu_engram_local_ids=None, tpu_engram_staged=None)
        got_ids = hooks.tpu_dsv41_hasher_forward(
            lambda *a, **k: None, "hasher", input_ids, rebuilt
        )
        # Simulate non-CPU device check by calling staged_lookup through hook
        # with metadata fallback (rebuilt has None staging attrs).
        out = hooks.tpu_dsv41_engram_embed_forward(
            lambda *a, **k: None, embedding, ids[:, 0], rebuilt
        )
        return got_ids, out

    carried = compile_policy._carrying_engram(fake_compiled_fn)
    fb = SimpleNamespace(tpu_engram_local_ids=ids, tpu_engram_staged=(packed,))
    with (
        mock.patch.object(compile_policy, "get_forward_context", return_value=ctx),
        mock.patch(
            "sglang.srt.model_executor.forward_context.has_forward_context",
            return_value=True,
        ),
        mock.patch(
            "sglang.srt.model_executor.forward_context.get_forward_context",
            return_value=ctx,
        ),
    ):
        _, out = carried(torch.zeros(4, dtype=torch.int64), torch.arange(4), fb)

    assert metadata.engram is not None
    assert metadata.engram[0] is ids
    assert metadata.engram[1] == (packed,)
    assert metadata.engram[1][0] is packed
    assert torch.equal(out, unpack_rows(packed, tokens=4, heads=4, dim=weight.shape[1]))


def test_tpu_consumer_event_releases_output_tensors_across_slots_and_warmup():
    """TPU stream Events gate slot reuse without pinning output tensors in HBM."""
    import gc
    import sys
    import weakref

    from sglang_tpu.compilation.compiled_forward_shim import CompiledForwardShim
    from sglang_tpu.models.deepseek_v41.compile_policy import (
        DeepseekV4DecoderTpuCompilePolicy,
        _EngramCompiledForwardShim,
    )

    # 1. Default TPU coordinator selects torch.get_device_module("tpu").Event
    #    unless a custom sync_fn overrides synchronization.
    fake_tpu_event_cls = type("FakeTpuEvent", (), {})
    fake_sync_mod = SimpleNamespace(synchronize=lambda tensors, wait=True: None)
    _, default_embeddings, default_hasher = _coordinator()
    with (
        patch(
            "sglang_tpu.models.deepseek_v41.engram_host.torch.device",
            return_value=SimpleNamespace(type="tpu"),
        ),
        patch(
            "sglang_tpu.models.deepseek_v41.engram_host.torch.get_device_module",
            return_value=SimpleNamespace(Event=fake_tpu_event_cls),
        ) as get_dev_mod,
        patch.dict(
            sys.modules,
            {
                "torch_tpu": SimpleNamespace(),
                "torch_tpu._internal": SimpleNamespace(),
                "torch_tpu._internal.sync": SimpleNamespace(),
                "torch_tpu._internal.sync.sync": fake_sync_mod,
            },
        ),
    ):
        tpu_coord = HostEngramCoordinator(default_hasher, default_embeddings, "tpu")
        get_dev_mod.assert_called_once_with("tpu")
        assert tpu_coord._event_cls is fake_tpu_event_cls
        custom_sync_coord = HostEngramCoordinator(
            default_hasher, default_embeddings, "tpu", sync_fn=lambda t, wait=True: None
        )
        assert custom_sync_coord._event_cls is None

    # 2. End-to-end shim + coordinator with recorded stream Events.
    bt_mod, layout_mod, sync_fn, _, synced, events = _fake_transport(defer_h2d=True)
    created_events = []

    class FakeEvent:
        def __init__(self):
            self.idx = len(created_events)
            self.recorded = 0
            self.synced = 0
            created_events.append(self)

        def record(self):
            self.recorded += 1
            events.append(("event_record", self.idx))

        def synchronize(self):
            self.synced += 1
            events.append(("event_sync", self.idx))

    _, embeddings, hasher = _coordinator()
    coordinator = HostEngramCoordinator(
        hasher,
        embeddings,
        "cpu",
        batch_transfer_mod=bt_mod,
        layout_mod=layout_mod,
        sync_fn=sync_fn,
        event_cls=FakeEvent,
    )
    layers = [
        SimpleNamespace(engram=SimpleNamespace(layer_hash_index=idx, embed=emb))
        for idx, emb in embeddings
    ]
    model = SimpleNamespace(
        model=SimpleNamespace(layers=layers, engram_hasher=hasher),
        vision=None,
        get_input_embeddings=lambda: (
            lambda ids: torch.zeros(ids.shape[0], 8, dtype=torch.bfloat16)
        ),
        _tpu_host_engram_coordinator=coordinator,
    )
    runner = SimpleNamespace(
        model=model,
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                model_type="deepseek_v41",
                vision_n_layers=0,
                vocab_size=8,
                image_token_id=None,
                engram_pad_token_id=3,
            ),
            use_engram=True,
        ),
        is_draft_worker=False,
        device="cpu",
    )
    policy = object.__new__(DeepseekV4DecoderTpuCompilePolicy)
    policy.model_runner = runner
    policy._engram_hasher = hasher
    policy._embeds_outside_graph = False
    shim = object.__new__(_EngramCompiledForwardShim)
    shim.policy = policy
    worker = SimpleNamespace(model_runner=runner)

    def make_fb(mode, n_tokens):
        return SimpleNamespace(
            input_ids=torch.zeros(n_tokens, dtype=torch.int64),
            positions=torch.arange(n_tokens, dtype=torch.int64),
            req_pool_indices=torch.zeros(1, dtype=torch.int64),
            out_cache_loc=torch.arange(n_tokens, dtype=torch.int64),
            extend_seq_lens=torch.tensor([n_tokens], dtype=torch.int64),
            extend_start_loc=torch.tensor([0], dtype=torch.int64),
            engram_history=None,
            spec_info=None,
            input_embeds=None,
            mm_inputs=None,
            forward_mode=mode,
            tpu_engram_local_ids=None,
            tpu_engram_staged=None,
        )

    def run_shim(fb, fn, *, unpad_idle=False):
        def orig_shim(s, wrapped_fn, ids, pos, fbatch, *a, **kw):
            out = wrapped_fn(ids, pos, fbatch)
            if unpad_idle and not isinstance(out, PendingCall):
                return _FakeLogitsOutput(next_token_logits=out.next_token_logits[:0])
            return out

        with patch.object(CompiledForwardShim, "_forward", orig_shim):
            return shim._forward(fn, fb.input_ids, fb.positions, fb)

    fake_parallel = SimpleNamespace(
        pp_size=1, attn_cp_size=1, attn_dcp_size=1, attn_dp_size=1
    )
    fake_exec = SimpleNamespace(
        features=SimpleNamespace(enable_encoder_swa_bounded_replay=False)
    )
    weak_outputs = []

    with (
        patch("sglang.srt.runtime_context.get_parallel", return_value=fake_parallel),
        patch("sglang.srt.runtime_context.get_exec", return_value=fake_exec),
    ):
        # Step 1 (chunked prefill): event.record() precedes prefetch h2d_async,
        # sync_fn is never called, and next_token_logits is NOT retained.
        prompt_req = _request([1, 2, 3, 4, 5, 6, 7, 0])
        b1 = _batch([prompt_req], starts=[0], lengths=[4])
        fb1 = make_fb(b1.forward_mode, 4)
        events.clear()

        def step1_worker(w, b):
            def _fn(ids, pos, fb):
                logits = torch.ones(4, 16)
                weak_outputs.append(weakref.ref(logits))
                return _FakeLogitsOutput(next_token_logits=logits)

            out = run_shim(fb1, _fn)
            del out
            return SimpleNamespace(copy_done=None, next_token_ids=[5])

        hooks.tpu_dsv41_engram_worker_forward(step1_worker, worker, b1)
        gc.collect()
        assert weak_outputs[-1]() is None
        assert len(synced) == 0
        rec_idx = events.index(("event_record", 0))
        assert events[rec_idx + 1 :] == ["h2d_async", "h2d_async"]
        ev0 = created_events[0]
        assert ev0.recorded == 1 and ev0.synced == 0
        for upload in coordinator._active_uploads.values():
            assert upload.consumer_event is ev0
            assert upload.consumers == []

        # Step 2 (chunk-2 prefetch hit on alternate slots): does NOT wait on ev0.
        b2 = _batch([prompt_req], starts=[4], lengths=[4])
        fb2 = make_fb(b2.forward_mode, 4)
        hooks.tpu_dsv41_engram_worker_forward(
            lambda w, b: (
                run_shim(
                    fb2,
                    lambda ids, pos, fb: _FakeLogitsOutput(
                        next_token_logits=torch.ones(4, 16)
                    ),
                ),
                SimpleNamespace(copy_done=None, next_token_ids=[6]),
            )[1],
            worker,
            b2,
        )
        assert ev0.synced == 0
        ev1 = created_events[1]
        assert ev1.recorded == 1 and ev1.synced == 0

        # Step 3 (reuses Step 1's slots on bucket 4): synchronizes ev0 per layer.
        other_req = _request([3, 1, 4, 2])
        b3 = _batch([other_req], starts=[0], lengths=[4])
        fb3 = make_fb(b3.forward_mode, 4)
        events.clear()
        hooks.tpu_dsv41_engram_worker_forward(
            lambda w, b: (
                run_shim(
                    fb3,
                    lambda ids, pos, fb: _FakeLogitsOutput(
                        next_token_logits=torch.ones(4, 16)
                    ),
                ),
                SimpleNamespace(copy_done=None, next_token_ids=[7]),
            )[1],
            worker,
            b3,
        )
        assert ev0.synced == 2
        assert ev1.synced == 0
        # Both slot waits happen before their replacement h2d_async transfers.
        assert events[:4] == [
            ("event_sync", 0),
            "h2d_async",
            ("event_sync", 0),
            "h2d_async",
        ]

        # Step 4 (IDLE DP unpadding on bucket 16): drops un-sliced output tensor.
        idle_b = _batch([], idle=True)
        idle_fb = make_fb(idle_b.forward_mode, 16)

        def idle_worker(w, b):
            def _fn(ids, pos, fb):
                logits = torch.full((16, 16), 9.0)
                weak_outputs.append(weakref.ref(logits))
                return _FakeLogitsOutput(next_token_logits=logits)

            out = run_shim(idle_fb, _fn, unpad_idle=True)
            assert out.next_token_logits.shape[0] == 0
            del out
            return SimpleNamespace(copy_done=None, next_token_ids=torch.empty(0))

        hooks.tpu_dsv41_engram_worker_forward(idle_worker, worker, idle_b)
        gc.collect()
        assert weak_outputs[-1]() is None
        ev_idle = created_events[-1]
        for upload in coordinator._active_uploads.values():
            assert upload.consumer_event is ev_idle
            assert upload.consumers == []
        assert len(synced) == 0

        # Step 5 (PendingCall warmup on bucket 8): 2 executed variants + 1 no-op
        # compile-only PendingCall(lambda: None) preserve real consumer_events
        # and retain zero warmup output tensors.
        warm_mode = SimpleNamespace(
            is_decode=lambda: False,
            is_extend_without_speculative=lambda: False,
            is_extend=lambda: False,
            is_target_verify=lambda: False,
            is_idle=lambda: True,
        )
        warmup_calls = []
        for idx in range(3):
            w_fb = make_fb(warm_mode, 8)

            def make_warm_fn(call_idx):
                def _fn(ids, pos, fb):
                    if call_idx == 2:
                        return PendingCall(lambda: None)

                    def _exec():
                        logits = torch.full((8, 16), float(call_idx + 1))
                        weak_outputs.append(weakref.ref(logits))
                        return _FakeLogitsOutput(next_token_logits=logits)

                    return PendingCall(_exec)

                return _fn

            warmup_calls.append(run_shim(w_fb, make_warm_fn(idx)))

        event_count_before_wait = len(created_events)
        for pending in warmup_calls:
            res = pending.wait()
            del res
        gc.collect()
        # Only the 2 executed warmup variants recorded events; the 3rd no-op
        # PendingCall left Slot 0's recorded event intact.
        assert len(created_events) == event_count_before_wait + 2
        ev_w0, ev_w1 = created_events[-2], created_events[-1]
        for layer_slots in coordinator._uploads_by_layer.values():
            slot0, slot1 = layer_slots[8]
            assert slot0.consumer_event is ev_w0 and slot0.consumers == []
            assert slot1.consumer_event is ev_w1 and slot1.consumers == []
        assert all(ref() is None for ref in weak_outputs)
        # Only pre-wait slot reuse during PendingCall queuing touched sync_fn
        # (on the pre-allocated upload.device buffers, never output tensors).
        assert [t for t, _ in synced] == [
            [coordinator._uploads_by_layer[0][8][0].device],
            [coordinator._uploads_by_layer[1][8][0].device],
        ]


def test_encoder_swa_bounded_replay_fast_path_folded_and_multichunk_prefetch():
    """Folded and cache-miss extends under --enable-encoder-swa-bounded-replay
    stay on the host fast path with exact folded windows and chunk prefetch."""
    coordinator, embeddings, hasher = _coordinator()
    hasher.history = torch.zeros(8, NGRAM - 1, dtype=torch.int32)
    hasher.pad_row = 7
    fake_parallel = SimpleNamespace(pp_size=1, attn_cp_size=1, attn_dcp_size=1)

    # 1. Cache-miss 2-chunk prefill with replay enabled: fast path + prefetch hit.
    miss_req = _request([1, 2, 3, 4, 5, 6, 7, 0], req_pool_idx=1)
    b_miss1 = _batch(
        [miss_req],
        starts=[0],
        lengths=[4],
        req_pool_indices_cpu=torch.tensor([1], dtype=torch.int64),
    )
    b_miss1.encoder_swa_reset = [True]
    prep1 = coordinator.prepare_if_supported(
        b_miss1,
        parallel=fake_parallel,
        enable_encoder_swa_bounded_replay=True,
        folds_encoder_swa_replay=True,
        has_request_window=True,
    )
    assert prep1 == ([miss_req], [4])
    _, up1, _ = coordinator.stage_all(4)
    coordinator.record_consumers(up1, [torch.tensor([1])])
    coordinator.finish_forward(4)
    assert coordinator._prefetch is not None and coordinator._prefetch.starts == [4]
    coordinator.clear_step()

    b_miss2 = _batch(
        [miss_req],
        starts=[4],
        lengths=[4],
        req_pool_indices_cpu=torch.tensor([1], dtype=torch.int64),
    )
    b_miss2.encoder_swa_reset = [False]
    prep2 = coordinator.prepare_if_supported(
        b_miss2,
        parallel=fake_parallel,
        enable_encoder_swa_bounded_replay=True,
        folds_encoder_swa_replay=True,
        has_request_window=True,
    )
    assert prep2 == ([miss_req], [8])
    assert coordinator._staged_uploads  # prefetch hit!
    _, up2, _ = coordinator.stage_all(4)
    coordinator.record_consumers(up2, [torch.tensor([2])])
    coordinator.clear_step()

    # 2. Mixed folded hit + non-reset continuation chunk in one batch:
    #    hit_req has 260 tokens, prefix_len=200 (even) -> replays 128 tokens [72:200]
    #    before chunk [200:220] (folded [72:220] = 148 rows), then chunk 2 [220:240]
    #    (not reset -> 20 rows, prefetches chunk 3 [240:260]).
    rng = np.random.default_rng(123)
    hit_prompt = rng.integers(0, 8, 260).tolist()
    hit_req = _request(hit_prompt, req_pool_idx=2)
    cont_prompt = rng.integers(0, 8, 16).tolist()
    cont_req = _request(cont_prompt, req_pool_idx=3)

    b_fold = _batch(
        [hit_req, cont_req],
        starts=[200, 8],
        lengths=[20, 8],
        req_pool_indices_cpu=torch.tensor([2, 3], dtype=torch.int64),
    )
    b_fold.encoder_swa_reset = [True, False]
    prep_fold = coordinator.prepare_if_supported(
        b_fold,
        parallel=fake_parallel,
        enable_encoder_swa_bounded_replay=True,
        folds_encoder_swa_replay=True,
        has_request_window=True,
    )
    # Sampled token positions stay at prefix_len + extend_len = [220, 16].
    assert prep_fold == ([hit_req, cont_req], [220, 16])
    assert coordinator._next_prompt_spec is None  # suppressed on folded chunk
    staged_fold, up_fold, _ = coordinator.stage_all(256)
    folded_copies = [t.clone() for t in staged_fold]
    coordinator.record_consumers(up_fold, [torch.tensor([3])])
    coordinator.clear_step()

    expected_fold_inputs = _reference_capture(
        [hit_req, cont_req], [72, 8], [148, 8], NGRAM
    )
    expected_fold_hashes = hash_on_host(expected_fold_inputs, **_tables())
    for (layer, embedding), got_rows in zip(embeddings, folded_copies, strict=True):
        expected_rows = pack_rows(
            embedding.weight,
            embedding.scale,
            expected_fold_hashes[:, layer],
            row_start=0,
            padded_tokens=256,
        )
        assert torch.equal(got_rows, expected_rows)
    assert torch.equal(
        hasher.history[2],
        torch.tensor(hit_prompt[218:220], dtype=torch.int32),
    )
    assert torch.equal(
        hasher.history[3],
        torch.tensor(cont_prompt[14:16], dtype=torch.int32),
    )

    # Chunk 2 of hit_req ([220:240], encoder_swa_reset=[False]) resumes prefetching!
    b_hit_c2 = _batch(
        [hit_req],
        starts=[220],
        lengths=[20],
        req_pool_indices_cpu=torch.tensor([2], dtype=torch.int64),
    )
    b_hit_c2.encoder_swa_reset = [False]
    assert coordinator.prepare_if_supported(
        b_hit_c2,
        parallel=fake_parallel,
        enable_encoder_swa_bounded_replay=True,
        folds_encoder_swa_replay=True,
        has_request_window=True,
    ) == ([hit_req], [240])
    _, up_hc2, _ = coordinator.stage_all(32)
    coordinator.record_consumers(up_hc2, [torch.tensor([4])])
    coordinator.finish_forward(32)
    assert coordinator._prefetch is not None and coordinator._prefetch.starts == [240]
    coordinator.clear_step()

    # 3. Odd cached-prefix boundary returns None so core raises its ValueError.
    b_odd = _batch([hit_req], starts=[199], lengths=[10])
    b_odd.encoder_swa_reset = [True]
    assert (
        coordinator.prepare_if_supported(
            b_odd,
            parallel=fake_parallel,
            enable_encoder_swa_bounded_replay=True,
            folds_encoder_swa_replay=True,
            has_request_window=True,
        )
        is None
    )

    # 4. Non-folding backend (folds_encoder_swa_replay=False, has_request_window=True):
    #    cache miss uses fast path; prefix hit falls back (two-pass replay + extend).
    assert coordinator.prepare_if_supported(
        b_miss1,
        parallel=fake_parallel,
        enable_encoder_swa_bounded_replay=True,
        folds_encoder_swa_replay=False,
        has_request_window=True,
    ) == ([miss_req], [4])
    coordinator.clear_step()
    assert (
        coordinator.prepare_if_supported(
            b_fold,
            parallel=fake_parallel,
            enable_encoder_swa_bounded_replay=True,
            folds_encoder_swa_replay=False,
            has_request_window=True,
        )
        is None
    )


def test_speculative_extend_and_ced_combined_fast_path():
    """DSpark speculative EXTEND + IDLE use fast-path DMA/prefetch without pinning pending."""
    coordinator, _, hasher = _coordinator()
    hasher.history = torch.zeros(8, NGRAM - 1, dtype=torch.int32)
    hasher.pad_row = 7
    model = SimpleNamespace(
        model=SimpleNamespace(engram_hasher=hasher, layers=[]),
        _tpu_host_engram_coordinator=coordinator,
    )
    runner = SimpleNamespace(
        model=model,
        model_config=SimpleNamespace(use_engram=True),
        is_draft_worker=False,
        device=torch.device("cpu"),
        attn_backend=SimpleNamespace(folds_encoder_swa_replay=True),
        token_to_kv_pool=SimpleNamespace(request_window=object()),
    )
    worker = SimpleNamespace(model_runner=runner, is_dllm=lambda: False)
    fake_parallel = SimpleNamespace(pp_size=1, attn_cp_size=1, attn_dcp_size=1)
    ced_exec = SimpleNamespace(
        features=SimpleNamespace(enable_encoder_swa_bounded_replay=True)
    )

    prompt = list(range(8)) * 25  # 200 tokens
    spec_req = _request(prompt, req_pool_idx=2)
    # Seed stale pending before speculative extend; hook must clear it to None.
    coordinator.pending = PendingTokens(
        [spec_req], [1], SimpleNamespace(copy_done=None, next_token_ids=[3])
    )

    b_spec_fold = _batch(
        [spec_req],
        starts=[160],
        lengths=[40],
        spec_none=False,
        req_pool_indices_cpu=torch.tensor([2], dtype=torch.int64),
    )
    b_spec_fold.encoder_swa_reset = [True]
    observed_started = []

    def fake_spec_orig(w, b):
        observed_started.append(coordinator.started)
        staged, uploads, _ = coordinator.stage_all(256)
        coordinator.record_consumers(uploads, [torch.tensor([1])])
        assert len(staged) == 2
        return SimpleNamespace(copy_done=None, next_token_ids=torch.tensor([4]))

    with (
        patch("sglang.srt.runtime_context.get_parallel", return_value=fake_parallel),
        patch("sglang.srt.runtime_context.get_exec", return_value=ced_exec),
    ):
        hooks.tpu_dsv41_engram_worker_forward(fake_spec_orig, worker, b_spec_fold)
        assert observed_started == [True]
        assert coordinator.pending is None
        assert torch.equal(
            hasher.history[2], torch.tensor(prompt[198:200], dtype=torch.int32)
        )

        # IDLE DP batch under DSpark also takes the fast path.
        b_spec_idle = _batch([], idle=True, spec_none=False)
        observed_started.clear()
        hooks.tpu_dsv41_engram_worker_forward(
            lambda w, b: (
                observed_started.append(coordinator.started),
                SimpleNamespace(copy_done=None, next_token_ids=torch.empty(0)),
            )[1],
            worker,
            b_spec_idle,
        )
        assert observed_started == [True]

    # Speculative DECODE and MIXED batches cleanly return None.
    b_spec_dec = _batch([spec_req], decode=True, starts=[200], spec_none=False)
    assert coordinator.prepare_if_supported(b_spec_dec, parallel=fake_parallel) is None
    b_spec_mixed = _batch([spec_req], starts=[160], lengths=[40], spec_none=False)
    b_spec_mixed.forward_mode.is_mixed = lambda: True
    assert (
        coordinator.prepare_if_supported(b_spec_mixed, parallel=fake_parallel) is None
    )


def test_target_verify_single_threaded_numpy_staging_matches_engram_hasher():
    """stage_verify matches EngramHasher.forward across block sizes & skips D2H on CPU slots."""
    from sglang.srt.layers.engram import EngramHasher
    from sglang_tpu.models.deepseek_v41.compile_policy import _stage_host_engram
    from sglang_tpu.models.deepseek_v41.engram_host import host_hash_ids

    tables = _tables()

    def make_real_hasher():
        h = EngramHasher.__new__(EngramHasher)
        torch.nn.Module.__init__(h)
        h.max_ngram_size = NGRAM
        h.pad_id = tables["pad_id"]
        h.image_token_id = 7
        for k in ("token_map", "multipliers", "primes", "offsets"):
            h.register_buffer(k, torch.as_tensor(tables[k], dtype=torch.int64))
        h.history = None
        h.pad_row = 0
        h.init_history(num_req_slots=16, device="cpu")
        return h

    fast_hasher = make_real_hasher()
    ref_hasher = make_real_hasher()
    embeddings = []
    for layer in (0, 1):
        weight, scale = _table(rows=130, seed=layer + 9)
        embeddings.append(
            (
                layer,
                SimpleNamespace(
                    weight=weight,
                    scale=scale,
                    dim=weight.shape[1],
                    row_start=0,
                    _shared=True,
                    host_table=SimpleNamespace(layout="shared"),
                    tpu_hash_index=layer,
                ),
            )
        )
    coordinator = HostEngramCoordinator(fast_hasher, embeddings, "cpu")
    verify_mode = SimpleNamespace(
        is_decode=lambda: False,
        is_target_verify=lambda: True,
        is_extend=lambda: False,
        is_idle=lambda: False,
    )

    rng = np.random.default_rng(77)
    saw_skip_pos_d2h = False
    saw_short_prompt_fallback = False
    for step_idx in range(60):
        bs = int(rng.integers(1, 7))
        # Test both block < NGRAM (e.g. 1, 2) and block >= NGRAM (e.g. 6).
        block = int(rng.integers(1, 8))
        hist_vals = rng.integers(0, 8, (17, NGRAM - 1), dtype=np.int32)
        # Inject MM pad sentinel into one slot's history periodically.
        if step_idx % 3 == 0:
            hist_vals[2, 0] = _MM_PAD_SHIFT_VALUE
        fast_hasher.history.copy_(torch.from_numpy(hist_vals))
        ref_hasher.history.copy_(torch.from_numpy(hist_vals))

        slots = rng.choice(np.arange(1, 16), size=bs, replace=False).astype(np.int64)
        # Bucket-pad the tail row with reserved slot 0 (select_verify_rows sets
        # positions = seq_lens[:, None] + arange(block), so padded seq_len=0 gives 0..block-1).
        slots[-1] = 0
        base_pos = rng.integers(0, 50, size=bs, dtype=np.int64)
        base_pos[-1] = 0
        positions = (base_pos[:, None] + np.arange(block)[None, :]).reshape(-1)
        tokens = rng.integers(0, 8, size=(bs * block,), dtype=np.int64)
        tokens[-block:] = 0

        class _NoD2HTensor:
            """Simulates a device tensor whose .detach()/.to('cpu') would fail if called."""

            def __init__(self, shape):
                self.shape = shape

            def detach(self):
                raise AssertionError("Unexpected D2H copy on device tensor")

        req_slots_cpu = torch.from_numpy(slots)
        input_ids_t = torch.from_numpy(tokens)
        positions_t = torch.from_numpy(positions)

        bs_live = bs - 1
        # Simulate 1-step overlap lag on live_seq_lens_cpu (0..block tokens behind device).
        lag = rng.integers(0, block, size=bs_live, dtype=np.int64)
        live_host_lens = np.maximum(0, base_pos[:bs_live] - lag)
        can_skip_pos_d2h = bool(np.all(live_host_lens >= NGRAM - 1))
        saw_skip_pos_d2h = saw_skip_pos_d2h or can_skip_pos_d2h
        saw_short_prompt_fallback = saw_short_prompt_fallback or (not can_skip_pos_d2h)
        fast_fb = SimpleNamespace(
            forward_mode=verify_mode,
            spec_info=SimpleNamespace(
                draft_token_num=block,
                live_seq_lens_cpu=torch.from_numpy(live_host_lens),
            ),
            positions=(
                _NoD2HTensor(positions_t.shape) if can_skip_pos_d2h else positions_t
            ),
            req_pool_indices=_NoD2HTensor(req_slots_cpu.shape),
            req_pool_indices_cpu=req_slots_cpu,
            out_cache_loc=_NoD2HTensor((bs * block,)),
            tpu_engram_local_ids=None,
            tpu_engram_staged=None,
        )
        uploads = _stage_host_engram(coordinator, fast_hasher, input_ids_t, fast_fb)
        staged_fast = [t.clone() for t in fast_fb.tpu_engram_staged]
        coordinator.record_consumers(uploads, [torch.tensor([1])])

        ref_fb = SimpleNamespace(
            forward_mode=verify_mode,
            spec_info=SimpleNamespace(draft_token_num=block),
            positions=positions_t,
            req_pool_indices=req_slots_cpu,
            out_cache_loc=torch.zeros(bs * block, dtype=torch.int64),
            extend_seq_lens=None,
            extend_start_loc=None,
            engram_history=None,
        )
        ref_hashes = host_hash_ids(ref_hasher, input_ids_t, ref_fb)
        for (layer, embedding), got_packed in zip(embeddings, staged_fast, strict=True):
            expected_packed = pack_rows(
                embedding.weight,
                embedding.scale,
                ref_hashes[:, layer],
                row_start=0,
                padded_tokens=bs * block,
            )
            assert torch.equal(got_packed, expected_packed)
    assert saw_skip_pos_d2h and saw_short_prompt_fallback

    # Ragged verify (explicit layout or non-uniform token count) must assert.
    ragged_layout_fb = SimpleNamespace(
        forward_mode=verify_mode,
        spec_info=SimpleNamespace(draft_token_num=4),
        # The sync keeps the ragged layout on the forward batch.
        ragged_verify_layout=object(),
        positions=torch.arange(8, dtype=torch.int64),
        req_pool_indices=torch.tensor([1, 2], dtype=torch.int64),
        req_pool_indices_cpu=torch.tensor([1, 2], dtype=torch.int64),
    )
    with pytest.raises(AssertionError, match="ragged verify"):
        coordinator.stage_verify(torch.zeros(8, dtype=torch.int64), ragged_layout_fb)

    unequal_tokens_fb = SimpleNamespace(
        forward_mode=verify_mode,
        spec_info=SimpleNamespace(draft_token_num=4, ragged_verify_layout=None),
        positions=torch.arange(6, dtype=torch.int64),
        req_pool_indices=torch.tensor([1, 2], dtype=torch.int64),
        req_pool_indices_cpu=torch.tensor([1, 2], dtype=torch.int64),
    )
    with pytest.raises(AssertionError, match="ragged verify"):
        coordinator.stage_verify(torch.zeros(6, dtype=torch.int64), unequal_tokens_fb)


def test_deferred_verify_history_commits_order_across_verify_extend_and_prebuilt():
    """Deferred verify commits preserve slot 0, flush before reused-slot EXTEND,
    drain on IDLE steps, and flush before PD PREBUILT history writes."""
    from sglang.srt.layers.engram import EngramHasher
    from sglang_tpu.models.deepseek_v41.compile_policy import _stage_host_engram
    from sglang_tpu.models.deepseek_v41.engram_host import has_pending_engram_commits

    tables = _tables()
    hasher = EngramHasher.__new__(EngramHasher)
    torch.nn.Module.__init__(hasher)
    hasher.max_ngram_size = NGRAM
    hasher.pad_id = tables["pad_id"]
    hasher.image_token_id = None
    for k in ("token_map", "multipliers", "primes", "offsets"):
        hasher.register_buffer(k, torch.as_tensor(tables[k], dtype=torch.int64))
    hasher.init_history(num_req_slots=8, device="cpu")
    # Sentinel in slot 0: bucket-padded commit_lens=0 rows must NEVER overwrite it.
    hasher.history[0] = torch.tensor([7, 7], dtype=torch.int32)
    hasher.history[2] = torch.tensor([1, 2], dtype=torch.int32)
    hasher.history[4] = torch.tensor([3, 4], dtype=torch.int32)

    embeddings = []
    for layer in (0, 1):
        weight, scale = _table(rows=130, seed=layer + 15)
        embeddings.append(
            (
                layer,
                SimpleNamespace(
                    weight=weight,
                    scale=scale,
                    dim=weight.shape[1],
                    row_start=0,
                    _shared=True,
                    host_table=SimpleNamespace(layout="shared"),
                    tpu_hash_index=layer,
                ),
            )
        )
    coordinator = HostEngramCoordinator(hasher, embeddings, "cpu")
    manager = SimpleNamespace(engram_hasher=hasher)

    # 1. Queue verify commit on slots [2, 4, 0 (padded)]:
    #    - slot 2 accepts 3 tokens [5, 6, 1, ...] -> new history [6, 1]
    #    - slot 4 accepts 1 token  [5, 2, 3, ...] -> window [3, 4, 5, ...] -> [4, 5]
    #    - slot 0 (padded) has commit_lens=0 -> leaves history[0] == [7, 7]!
    verify_ids_2d = torch.tensor(
        [[5, 6, 1, 0], [5, 2, 3, 0], [6, 6, 6, 6]], dtype=torch.int64
    )
    req_slots = torch.tensor([2, 4, 0], dtype=torch.int64)
    commit_lens = torch.tensor([3, 1, 0], dtype=torch.int32)

    hooks.tpu_dsv41_defer_engram_commit(
        lambda *a, **kw: pytest.fail("Should defer instead of calling original"),
        manager,
        verify_ids_2d=verify_ids_2d,
        req_pool_indices=req_slots,
        commit_lens=commit_lens,
    )
    assert has_pending_engram_commits(hasher)
    # Not applied yet before next host staging point!
    assert torch.equal(hasher.history[2], torch.tensor([1, 2], dtype=torch.int32))

    # 2. Slot 2 finishes and is IMMEDIATELY reused by a new request [0, 3, 5] in EXTEND!
    #    Fast-path stage_all must flush the old verify commit FIRST, then write [3, 5]
    #    into slot 2 while preserving slot 4's newly flushed [4, 5] and slot 0's [7, 7]!
    reused_req = _request([0, 3, 5], req_pool_idx=2)
    ext_b = _batch(
        [reused_req],
        starts=[0],
        lengths=[3],
        spec_none=False,
        req_pool_indices_cpu=torch.tensor([2], dtype=torch.int64),
    )
    coordinator.prepare(ext_b)
    ext_fb = SimpleNamespace(
        forward_mode=ext_b.forward_mode,
        tpu_engram_local_ids=None,
        tpu_engram_staged=None,
    )
    up_ext = _stage_host_engram(
        coordinator, hasher, torch.tensor([0, 3, 5, 0], dtype=torch.int64), ext_fb
    )
    coordinator.record_consumers(up_ext, [torch.tensor([1])])
    coordinator.clear_step()

    assert not has_pending_engram_commits(hasher)
    assert torch.equal(hasher.history[0], torch.tensor([7, 7], dtype=torch.int32))
    assert torch.equal(hasher.history[2], torch.tensor([3, 5], dtype=torch.int32))
    assert torch.equal(hasher.history[4], torch.tensor([4, 5], dtype=torch.int32))

    # 3. IDLE DP step drains queued verify commits so _PENDING_COMMITS never leaks.
    hooks.tpu_dsv41_defer_engram_commit(
        lambda *a, **kw: None,
        manager,
        verify_ids_2d=torch.zeros((2, 4), dtype=torch.int64),
        req_pool_indices=torch.zeros((2,), dtype=torch.int64),
        commit_lens=torch.zeros((2,), dtype=torch.int32),
    )
    assert has_pending_engram_commits(hasher)
    idle_mode = SimpleNamespace(
        is_decode=lambda: False,
        is_target_verify=lambda: False,
        is_extend=lambda: False,
        is_idle=lambda: True,
    )
    idle_fb = SimpleNamespace(
        forward_mode=idle_mode,
        tpu_engram_local_ids=None,
        tpu_engram_staged=None,
    )
    up_idle = _stage_host_engram(
        coordinator, hasher, torch.zeros(4, dtype=torch.int64), idle_fb
    )
    coordinator.record_consumers(up_idle, [torch.tensor([2])])
    assert not has_pending_engram_commits(hasher)
    assert torch.equal(hasher.history[0], torch.tensor([7, 7], dtype=torch.int32))

    # 4. PD PREBUILT hook flushes pending commits BEFORE _prepare_engram_history writes.
    hooks.tpu_dsv41_defer_engram_commit(
        lambda *a, **kw: None,
        manager,
        verify_ids_2d=torch.tensor([[6, 2, 0, 0]], dtype=torch.int64),
        req_pool_indices=torch.tensor([5], dtype=torch.int64),
        commit_lens=torch.tensor([2], dtype=torch.int32),
    )
    assert has_pending_engram_commits(hasher)
    prebuilt_batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_prebuilt=lambda: True)
    )

    def fake_prepare_prebuilt(mgr, b):
        # Pending commits must already have landed when original runs!
        assert not has_pending_engram_commits(hasher)
        assert torch.equal(hasher.history[5], torch.tensor([6, 2], dtype=torch.int32))
        hasher.history[5] = torch.tensor([1, 4], dtype=torch.int32)

    hooks.tpu_dsv41_flush_engram_before_prebuilt(
        fake_prepare_prebuilt, manager, prebuilt_batch
    )
    assert torch.equal(hasher.history[5], torch.tensor([1, 4], dtype=torch.int32))


def test_speculative_verify_discards_same_bucket_next_chunk_prefetch():
    """Intervening verify/fallback staging on the same bucket must discard prefetch.

    Regression for DSpark + chunked prefill: speculative extend leaves a
    next-chunk prefetch in the inactive bucket slot; verify skips
    prepare_if_supported and stages via use_prestaged=False on the same padded
    token bucket, which would otherwise overwrite the prefetch slot while
    _prefetch.matches still passes on host metadata alone.
    """
    from sglang.srt.layers.engram import EngramHasher
    from sglang_tpu.models.deepseek_v41.compile_policy import _stage_host_engram

    tables = _tables()
    hasher = EngramHasher.__new__(EngramHasher)
    torch.nn.Module.__init__(hasher)
    hasher.max_ngram_size = NGRAM
    hasher.pad_id = tables["pad_id"]
    hasher.image_token_id = 7
    for k in ("token_map", "multipliers", "primes", "offsets"):
        hasher.register_buffer(k, torch.as_tensor(tables[k], dtype=torch.int64))
    hasher.init_history(num_req_slots=8, device="cpu")
    hasher.history[2] = torch.tensor([0, 0], dtype=torch.int32)

    embeddings = []
    for layer in (0, 1):
        weight, scale = _table(rows=130, seed=layer + 23)
        embeddings.append(
            (
                layer,
                SimpleNamespace(
                    weight=weight,
                    scale=scale,
                    dim=weight.shape[1],
                    row_start=0,
                    _shared=True,
                    host_table=SimpleNamespace(layout="shared"),
                    tpu_hash_index=layer,
                ),
            )
        )
    coordinator = HostEngramCoordinator(hasher, embeddings, "cpu")
    model = SimpleNamespace(
        model=SimpleNamespace(engram_hasher=hasher, layers=[]),
        _tpu_host_engram_coordinator=coordinator,
    )
    runner = SimpleNamespace(
        model=model,
        model_config=SimpleNamespace(use_engram=True),
        is_draft_worker=False,
        device=torch.device("cpu"),
        attn_backend=SimpleNamespace(folds_encoder_swa_replay=False),
        token_to_kv_pool=SimpleNamespace(request_window=None),
    )
    worker = SimpleNamespace(model_runner=runner, is_dllm=lambda: False)
    fake_parallel = SimpleNamespace(pp_size=1, attn_cp_size=1, attn_dcp_size=1)
    fake_exec = SimpleNamespace(
        features=SimpleNamespace(enable_encoder_swa_bounded_replay=False)
    )

    req_chunked = _request([1, 2, 3, 4, 5, 6, 7, 0], req_pool_idx=1)
    b_c1 = _batch(
        [req_chunked],
        starts=[0],
        lengths=[4],
        spec_none=False,
        req_pool_indices_cpu=torch.tensor([1], dtype=torch.int64),
    )
    b_c2 = _batch(
        [req_chunked],
        starts=[4],
        lengths=[4],
        spec_none=False,
        req_pool_indices_cpu=torch.tensor([1], dtype=torch.int64),
    )
    verify_mode = SimpleNamespace(
        is_decode=lambda: False,
        is_target_verify=lambda: True,
        is_extend=lambda: False,
        is_idle=lambda: False,
    )
    b_ver = SimpleNamespace(
        forward_mode=verify_mode,
        spec_algorithm=SimpleNamespace(is_none=lambda: False),
        reqs=[],
    )
    verify_fb = SimpleNamespace(
        forward_mode=verify_mode,
        spec_info=SimpleNamespace(
            draft_token_num=4,
            live_seq_lens_cpu=torch.tensor([10], dtype=torch.int64),
        ),
        positions=torch.tensor([10, 11, 12, 13], dtype=torch.int64),
        req_pool_indices=torch.tensor([2], dtype=torch.int64),
        req_pool_indices_cpu=torch.tensor([2], dtype=torch.int64),
        tpu_engram_local_ids=None,
        tpu_engram_staged=None,
    )
    verify_ids = torch.zeros(4, dtype=torch.int64)

    with (
        patch("sglang.srt.runtime_context.get_parallel", return_value=fake_parallel),
        patch("sglang.srt.runtime_context.get_exec", return_value=fake_exec),
    ):
        # Step 1: speculative chunk 1 at padded bucket 4 prefetches chunk 2.
        def run_c1(w, b):
            fb = SimpleNamespace(
                forward_mode=b.forward_mode,
                extend_seq_lens_cpu=[4, 0],
                extend_prefix_lens_cpu=[0, 0],
                tpu_engram_local_ids=None,
                tpu_engram_staged=None,
            )
            uploads = _stage_host_engram(
                coordinator, hasher, torch.tensor([1, 2, 3, 4], dtype=torch.int64), fb
            )
            coordinator.record_consumers(uploads, [torch.tensor([1])])
            coordinator.finish_forward(4)
            return SimpleNamespace(copy_done=None, next_token_ids=torch.tensor([0]))

        hooks.tpu_dsv41_engram_worker_forward(run_c1, worker, b_c1)
        assert coordinator._prefetch is not None
        assert coordinator._prefetch.padded_tokens == 4

        # Step 2: intervening TARGET_VERIFY on the exact same padded bucket 4.
        verify_copies = []

        def run_ver(w, b, is_verify=False):
            uploads = _stage_host_engram(coordinator, hasher, verify_ids, verify_fb)
            verify_copies.extend(t.clone() for t in verify_fb.tpu_engram_staged)
            coordinator.record_consumers(uploads, [torch.tensor([2])])
            return SimpleNamespace(copy_done=None, next_token_ids=None)

        hooks.tpu_dsv41_engram_worker_forward(run_ver, worker, b_ver, is_verify=True)
        assert coordinator._prefetch is None

        # Step 3: speculative chunk 2 must NOT reuse the verify-overwritten slot.
        c2_copies = []
        saw_prestaged = []

        def run_c2(w, b):
            saw_prestaged.append(bool(coordinator._staged_uploads))
            fb = SimpleNamespace(
                forward_mode=b.forward_mode,
                extend_seq_lens_cpu=[4, 0],
                extend_prefix_lens_cpu=[4, 0],
                tpu_engram_local_ids=None,
                tpu_engram_staged=None,
            )
            uploads = _stage_host_engram(
                coordinator, hasher, torch.tensor([5, 6, 7, 0], dtype=torch.int64), fb
            )
            c2_copies.extend(t.clone() for t in fb.tpu_engram_staged)
            coordinator.record_consumers(uploads, [torch.tensor([3])])
            return SimpleNamespace(copy_done=None, next_token_ids=torch.tensor([0]))

        hooks.tpu_dsv41_engram_worker_forward(run_c2, worker, b_c2)
        assert saw_prestaged == [False]

    expected_c2_hashes = hash_on_host(
        _reference_capture([req_chunked], [4], [4], NGRAM),
        **tables,
        image_token_id=hasher.image_token_id,
    )
    for (layer, embedding), got_rows, ver_rows in zip(
        embeddings, c2_copies, verify_copies, strict=True
    ):
        expected_rows = pack_rows(
            embedding.weight,
            embedding.scale,
            expected_c2_hashes[:, layer],
            row_start=0,
            padded_tokens=4,
        )
        assert torch.equal(got_rows, expected_rows)
        assert not torch.equal(got_rows, ver_rows)

    # Fallback stage_from_hashes on the same bucket also discards pending prefetch.
    coordinator.prepare(b_c1)
    _, up1, _ = coordinator.stage_all(4)
    coordinator.record_consumers(up1, [torch.tensor([4])])
    coordinator.finish_forward(4)
    coordinator.clear_step()
    assert coordinator._prefetch is not None
    _, up_fb, _ = coordinator.stage_from_hashes(
        torch.zeros((4, 2, 4), dtype=torch.int64), 4
    )
    coordinator.record_consumers(up_fb, [torch.tensor([5])])
    assert coordinator._prefetch is None


def test_stage_all_falls_back_when_folded_extend_lens_drift():
    """stage_all validates predicted extend ranges against ForwardBatch and falls back."""
    from sglang.srt.layers.engram import EngramHasher
    from sglang_tpu.models.deepseek_v41.compile_policy import _stage_host_engram
    from sglang_tpu.models.deepseek_v41.engram_host import host_hash_ids

    tables = _tables()

    def make_real_hasher():
        h = EngramHasher.__new__(EngramHasher)
        torch.nn.Module.__init__(h)
        h.max_ngram_size = NGRAM
        h.pad_id = tables["pad_id"]
        h.image_token_id = 7
        for k in ("token_map", "multipliers", "primes", "offsets"):
            h.register_buffer(k, torch.as_tensor(tables[k], dtype=torch.int64))
        h.history = None
        h.pad_row = 0
        h.init_history(num_req_slots=8, device="cpu")
        return h

    fast_hasher = make_real_hasher()
    ref_hasher = make_real_hasher()
    fast_hasher.history[2] = torch.tensor([7, 7], dtype=torch.int32)
    ref_hasher.history[2] = torch.tensor([7, 7], dtype=torch.int32)

    embeddings = []
    for layer in (0, 1):
        weight, scale = _table(rows=130, seed=layer + 31)
        embeddings.append(
            (
                layer,
                SimpleNamespace(
                    weight=weight,
                    scale=scale,
                    dim=weight.shape[1],
                    row_start=0,
                    _shared=True,
                    host_table=SimpleNamespace(layout="shared"),
                    tpu_hash_index=layer,
                ),
            )
        )
    coordinator = HostEngramCoordinator(fast_hasher, embeddings, "cpu")
    fake_parallel = SimpleNamespace(pp_size=1, attn_cp_size=1, attn_dcp_size=1)

    rng = np.random.default_rng(456)
    prompt = rng.integers(0, 8, 260).tolist()
    # Ensure drifted slice ending at 215 has a distinct 2-token tail from 220.
    prompt[213:215] = [1, 2]
    prompt[218:220] = [6, 5]
    hit_req = _request(prompt, req_pool_idx=2)

    b_fold = _batch(
        [hit_req],
        starts=[200],
        lengths=[20],
        req_pool_indices_cpu=torch.tensor([2], dtype=torch.int64),
    )
    b_fold.encoder_swa_reset = [True]
    prep = coordinator.prepare_if_supported(
        b_fold,
        parallel=fake_parallel,
        enable_encoder_swa_bounded_replay=True,
        folds_encoder_swa_replay=True,
        has_request_window=True,
    )
    assert prep == ([hit_req], [220])
    assert coordinator._prepared_extend_ranges == ([72], [148])

    # 1. Matching TPU/DP zero-padded batch buckets stay on the fast path.
    matching_fb = SimpleNamespace(
        forward_mode=b_fold.forward_mode,
        extend_seq_lens_cpu=[148, 0, 0, 0],
        extend_prefix_lens_cpu=[72, 0, 0, 0],
        encoder_swa_replay=False,
    )
    assert coordinator._matches_forward_extend_ranges(matching_fb)

    # 2. Core fold drift (actual batch has [200:215] instead of predicted [72:220]):
    #    _stage_host_engram falls back cleanly to host_hash_ids without corrupting history.
    actual_tokens = torch.tensor(prompt[200:215] + [0] * 17, dtype=torch.int64)
    actual_positions = torch.tensor(list(range(200, 215)) + [0] * 17, dtype=torch.int64)
    actual_hist = torch.tensor([prompt[198:200]], dtype=torch.int32)
    extend_mode = SimpleNamespace(
        is_decode=lambda: False,
        is_target_verify=lambda: False,
        is_extend=lambda: True,
        is_idle=lambda: False,
    )
    drifted_fb = SimpleNamespace(
        forward_mode=extend_mode,
        spec_info=None,
        positions=actual_positions,
        req_pool_indices=torch.tensor([2, 0, 0, 0], dtype=torch.int64),
        out_cache_loc=torch.arange(32, dtype=torch.int64),
        extend_seq_lens=torch.tensor([15, 0, 0, 0], dtype=torch.int32),
        extend_start_loc=torch.tensor([0, 15, 15, 15], dtype=torch.int32),
        extend_seq_lens_cpu=[15, 0, 0, 0],
        extend_prefix_lens_cpu=[200, 0, 0, 0],
        engram_history=actual_hist,
        encoder_swa_replay=False,
        tpu_engram_local_ids=None,
        tpu_engram_staged=None,
    )
    uploads = _stage_host_engram(coordinator, fast_hasher, actual_tokens, drifted_fb)
    got_copies = [t.clone() for t in drifted_fb.tpu_engram_staged]
    coordinator.record_consumers(uploads, [torch.tensor([1])])
    assert not coordinator.started

    ref_hashes = host_hash_ids(ref_hasher, actual_tokens, drifted_fb)
    for (layer, embedding), got_rows in zip(embeddings, got_copies, strict=True):
        expected_rows = pack_rows(
            embedding.weight,
            embedding.scale,
            ref_hashes[:, layer],
            row_start=0,
            padded_tokens=32,
        )
        assert torch.equal(got_rows, expected_rows)
    # History matches the actual [200:215] batch ([1, 2]), NOT predicted [6, 5].
    assert torch.equal(fast_hasher.history[2], torch.tensor([1, 2], dtype=torch.int32))
    assert torch.equal(fast_hasher.history[2], ref_hasher.history[2])

    # 3. Unexpected encoder_swa_replay=True forward also falls back cleanly.
    coordinator.prepare_if_supported(
        b_fold,
        parallel=fake_parallel,
        enable_encoder_swa_bounded_replay=True,
        folds_encoder_swa_replay=True,
        has_request_window=True,
    )
    replay_fb = SimpleNamespace(
        forward_mode=extend_mode,
        extend_seq_lens_cpu=[148, 0, 0, 0],
        extend_prefix_lens_cpu=[72, 0, 0, 0],
        encoder_swa_replay=True,
    )
    assert coordinator.stage_all(256, forward_batch=replay_fb) is None
    assert not coordinator.started


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-p", "no:cacheprovider"]))


# One Engram layer with two hash columns: primes 5 and 7 (above
# engram_vocab_size - 1), so the column-aligned split over TP 2 is [0, 5) and
# [5, 12) while an equal-row split would be [0, 6) and [6, 12).
_ENGRAM_CONFIG = SimpleNamespace(
    engram_layer_ids=[1],
    engram_max_ngram_size=2,
    engram_n_heads=2,
    engram_vocab_size=5,
    engram_num_embeddings=[12],
    engram_head_dim=128,
)


def _host_engram_embedding(monkeypatch, *, tp_rank: int, tp_size: int):
    """A real EngramEmbedding host table built through the TPU init hook, as on a
    TPU host (no CUDA)."""
    from sglang.srt.arg_groups.engram_config import (
        EngramConfig,
        EngramLookup,
        EngramPlacement,
        EngramReplication,
    )
    from sglang.srt.environ import envs
    from sglang.srt.layers import engram

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        engram,
        "get_parallel",
        lambda: SimpleNamespace(tp_size=tp_size, tp_rank=tp_rank, tp_group=None),
    )
    config = EngramConfig(
        placement=EngramPlacement.CPU,
        replication=EngramReplication.NONE,
        lookup=EngramLookup.ALL_REDUCE,
    )
    monkeypatch.setattr(engram, "resolve_engram_config", lambda columns: config)
    # Skip dropping the checkpoint page cache; there is no model directory here.
    monkeypatch.setattr(engram, "_page_cache_dropped", True)
    layout = engram.EngramLayout.from_config(_ENGRAM_CONFIG)
    embedding = engram.EngramEmbedding.__new__(engram.EngramEmbedding)
    with (
        envs.SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE.override(True),
        envs.SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT.override("shared"),
    ):
        hooks.tpu_dsv41_engram_embed_init(
            engram.EngramEmbedding.__init__,
            embedding,
            layout.num_embeddings[0],
            layout.head_dim,
            1,
            column_bounds=layout.column_bounds(0),
        )
    return embedding


def test_engram_host_table_init_needs_no_cuda(monkeypatch):
    """Regression: the TPU init hook built the host table with only
    cudaHostRegister stubbed, and the core later started asking
    torch.cuda.current_device() for its host read order, so every rank died
    at model init on a machine without CUDA."""
    embedding = _host_engram_embedding(monkeypatch, tp_rank=0, tp_size=1)

    assert embedding.host_table is not None
    assert not embedding.shared_host_table_without_ats


def test_engram_row_reader_loads_each_ranks_rows(monkeypatch, tmp_path):
    """Regression: the TPU loader read each rank's Engram rows as an equal-row
    split after the core moved EngramEmbedding to hash-column boundaries, so
    loading rejected every rank's rows. Each rank must read exactly the rows
    its EngramEmbedding owns, and get the checkpoint's bytes."""
    from safetensors.torch import save_file

    from sglang.kernels.ops.quantization.fp4_fake_quant import FP8_BLOCK_SIZE
    from sglang_tpu.models.deepseek_v41 import engram_row_reader as reader

    weight = torch.arange(12 * 128, dtype=torch.float32).remainder(200)
    weight = weight.view(12, 128).to(torch.float8_e4m3fn)
    blocks = 128 // FP8_BLOCK_SIZE
    scale = torch.arange(12 * blocks, dtype=torch.float32).remainder(20).exp2()
    scale = scale.view(12, blocks).to(torch.float8_e8m0fnu)
    names = {"weight": "layers.1.engram.embed.weight"}
    names["scale"] = "layers.1.engram.embed.scale"
    path = str(tmp_path / "model-00001.safetensors")
    save_file({names["weight"]: weight, names["scale"]: scale}, path)
    raw = open(path, "rb").read()

    def read_range(_path, offset, out):
        out.copy_(
            torch.frombuffer(
                bytearray(raw[offset : offset + out.numel()]), dtype=torch.uint8
            )
        )

    monkeypatch.setattr(reader, "_read_range", read_range)
    for tp_rank in range(2):
        embedding = _host_engram_embedding(monkeypatch, tp_rank=tp_rank, tp_size=2)
        ranges = reader.engram_row_ranges(
            names.values(), _ENGRAM_CONFIG, tp_rank=tp_rank, tp_size=2
        )
        shard_of = dict.fromkeys(names.values(), path)
        for name, rows in reader.iter_row_tables(shard_of=shard_of, ranges=ranges):
            param = embedding.weight if name == names["weight"] else embedding.scale
            reader.load_local_rows(None, embedding, param, rows)
        owned = slice(embedding.row_start, embedding.row_start + embedding.rows)
        assert torch.equal(
            embedding.weight.view(torch.uint8), weight[owned].view(torch.uint8)
        )
        assert torch.equal(
            embedding.scale.view(torch.uint8), scale[owned].view(torch.uint8)
        )
