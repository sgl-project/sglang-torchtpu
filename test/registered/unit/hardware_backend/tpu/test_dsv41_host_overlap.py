"""CPU tests for deferred Engram token waits."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu.engram import (
    HostEngramCoordinator,
    HostEngramInputs,
    PendingTokens,
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


def _coordinator():
    tables = _tables()
    hasher = SimpleNamespace(
        max_ngram_size=NGRAM,
        **{k: torch.as_tensor(v) for k, v in tables.items() if k != "pad_id"},
        pad_id=tables["pad_id"],
    )
    embeddings = []
    for layer in (0, 1):
        weight, scale = _table(rows=130, seed=layer + 3)
        embeddings.append(
            (layer, SimpleNamespace(weight=weight, scale=scale, row_start=0))
        )
    return HostEngramCoordinator(hasher, embeddings, "cpu"), embeddings


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
