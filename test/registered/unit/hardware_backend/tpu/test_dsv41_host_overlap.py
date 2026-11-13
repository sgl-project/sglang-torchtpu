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


def _request(prompt, output=(), *, beam_group=None):
    return SimpleNamespace(
        origin_input_ids=list(prompt),
        output_ids=list(output),
        beam_group=beam_group,
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
):
    mode = SimpleNamespace(
        is_decode=lambda: decode,
        is_extend_without_speculative=lambda: not decode and not idle,
        is_extend=lambda: not decode and not idle,
        is_target_verify=lambda: False,
        is_idle=lambda: idle,
    )
    return SimpleNamespace(
        reqs=requests,
        spec_algorithm=SimpleNamespace(is_none=lambda: spec_none),
        forward_mode=mode,
        prefix_lens=starts,
        extend_lens=lengths,
        seq_lens_cpu=torch.tensor([s + 1 for s in starts]) if decode else None,
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

    # Switching buckets releases consumers on the old bucket so HBM is not pinned.
    req4 = _request([1, 2])
    coordinator.prepare(_batch([req4], starts=[0], lengths=[2]))
    _, uploads4, _ = coordinator.stage_all(8)
    assert len(synced) == 6  # both slots across both layers on bucket 4 released
    coordinator.record_consumers(uploads4, [torch.tensor([40])])
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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-p", "no:cacheprovider"]))
