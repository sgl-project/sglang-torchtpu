"""CPU checks of the V4.1 stock verify path and the DSpark worker's host logic.

* C2 pair compression (attention_backend.pair_compress) over request-major
  verify blocks of K + 1 rows with partial acceptance equals one-token decode:
  rejected rows never leak into a later pooled value when the ring is the
  speculative width (8 for K = 5); width 2 fails (positive control).
* Host Engram rows of verify blocks, drafted tokens included, equal the hashes
  of the same tokens committed one by one (capture_inputs).
* The worker's committed-token tail, extend rows and context floors.
* The verify rows' D2H + hash are deferred to the first stage_layer and hash
  the same inputs as hashing them at once.
* Sampled batches keep the greedy draft chain and no draft logits (target-only
  acceptance) unless --speculative-use-rejection-sampling, which TPU allows for
  DSPARK only.
"""

from __future__ import annotations

import gc
import unittest
from array import array
from types import SimpleNamespace

import numpy as np
import torch

from sglang.srt.managers.schedule_batch import Req, ReqKvInfo, ScheduleBatch
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.dsv41.attention_backend import pair_compress, verify_token_rows
from sglang_tpu.dsv41.dspark import RING
from sglang_tpu.dsv41.moe import _ep_op_name, moe_token_rows
from sglang_tpu.engram import (
    HostEngramCoordinator,
    capture_inputs,
    hash_on_host,
    verify_inputs,
)
from sglang_tpu.experimental.speculative.dspark.dsv41_dspark_worker import (
    ExtendContext,
    TpuDsv41DsparkWorker,
    advance_tails,
    commit_rows,
    committed_tail,
    dp_verify_counts,
    extend_rows,
)

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

K = 5
DIM = 8


def expected_pool(kv, score, position):
    """C2 pooled value at an odd position: softmax over the pair's scores."""
    weights = torch.stack([score[position - 1], score[position]]).softmax(0)
    return weights[0] * kv[position - 1] + weights[1] * kv[position]


class TestVerifyPairCompress(CustomTestCase):
    def _run(self, ring_size, accepts):
        """Two requests: prefill 9 / 4 tokens, then verify steps with given accepts."""
        torch.manual_seed(0)
        slots = {0: 1, 1: 2}  # request -> pool slot (0 is the padding slot)
        lengths = {0: 9, 1: 4}
        total = 9 + 4 + (K + 1) * len(accepts) + 2
        truth = {r: (torch.randn(total, DIM), torch.randn(total, DIM)) for r in slots}
        state = torch.zeros(3 * ring_size + 1, 2 * DIM)
        state[-1, DIM:] = float("-inf")
        got = {r: {} for r in slots}

        def call(rows):
            reqs = torch.tensor([slots[r] if r is not None else 0 for r, _, _ in rows])
            pos = torch.tensor([p for _, p, _ in rows])
            valid = torch.tensor([r is not None for r, _, _ in rows])
            kv = torch.stack([v[0] for _, _, v in rows])
            score = torch.stack([v[1] for _, _, v in rows])
            return pair_compress(kv, score, reqs, pos, valid, state, ring_size)

        prefill = [
            (r, p, (truth[r][0][p], truth[r][1][p]))
            for r in slots
            for p in range(lengths[r])
        ] + [(None, 0, (torch.zeros(DIM), torch.zeros(DIM)))] * 3
        pooled = call(prefill)
        for (r, p, _), value in zip(prefill, pooled):
            if r is not None:
                got[r][p] = value
        for accept in accepts:
            rows = []
            for r in slots:
                n = lengths[r]
                for j in range(K + 1):
                    if j <= accept[r]:
                        value = (truth[r][0][n + j], truth[r][1][n + j])
                    else:  # a rejected draft: unrelated KV / score
                        value = (torch.randn(DIM) * 9, torch.randn(DIM) * 9)
                    rows.append((r, n + j, value))
            rows += [(None, 0, (torch.zeros(DIM), torch.zeros(DIM)))] * (K + 1)
            pooled = call(rows)
            for (r, p, _), value in zip(rows, pooled):
                if r is not None and p <= lengths[r] + accept[r]:
                    got[r][p] = value
            for r in slots:
                lengths[r] += accept[r] + 1
        errors = 0
        for r in slots:
            kv, score = truth[r]
            for p, value in got[r].items():
                if p % 2 == 1:
                    errors += not torch.allclose(
                        value, expected_pool(kv, score, p), atol=1e-5
                    )
        return errors

    def test_verify_blocks_equal_sequential_decode(self):
        accepts = [{0: 2, 1: 5}, {0: 0, 1: 1}, {0: 5, 1: 3}, {0: 1, 1: 0}]
        self.assertEqual(self._run(ring_size=8, accepts=accepts), 0)

    def test_two_row_ring_fails_partial_accepts(self):
        accepts = [{0: 2, 1: 5}, {0: 0, 1: 1}, {0: 5, 1: 3}, {0: 1, 1: 0}]
        self.assertGreater(self._run(ring_size=2, accepts=accepts), 0)

    def test_verify_token_rows(self):
        self.assertEqual(
            verify_token_rows(3, K + 1).tolist(),
            [0] * (K + 1) + [1] * (K + 1) + [2] * (K + 1),
        )


def tables(ngram=4, heads=2, vocab=50, seed=0):
    rng = np.random.default_rng(seed)
    return {
        "token_map": rng.integers(0, 40, size=vocab),
        "multipliers": rng.integers(1, 1 << 20, size=(3, ngram)),
        "primes": np.array([[[1009, 1013]] * (ngram - 1)] * 3)[:, :, :heads],
        "offsets": rng.integers(0, 1000, size=(3, (ngram - 1) * heads)),
    }


class TestVerifyEngram(CustomTestCase):
    def test_verify_rows_hash_like_committed_tokens(self):
        rng = np.random.default_rng(3)
        ngram, table = 4, tables()
        for n in (0, 2, 3, 17):
            prompt = rng.integers(0, 50, size=max(n, 1)).tolist()[:n]
            verify = rng.integers(0, 50, size=K + 1).tolist()
            tail = ([0] * ngram + prompt)[-(ngram - 1) :]
            got = hash_on_host(
                verify_inputs(
                    np.array([tail]), np.array([verify]), np.array([n]), ngram
                ),
                **table,
                pad_id=2,
            )
            req = SimpleNamespace(origin_input_ids=prompt, output_ids=verify)
            expected = hash_on_host(
                capture_inputs([req], [n], [K + 1], ngram), **table, pad_id=2
            )
            torch.testing.assert_close(got, expected, rtol=0, atol=0)

    def test_two_requests_are_request_major(self):
        inputs = verify_inputs(
            np.array([[1, 2, 3], [4, 5, 6]]),
            np.array([[7, 8, 9, 10, 11, 12], [13, 14, 15, 16, 17, 18]]),
            np.array([10, 20]),
            4,
        )
        self.assertEqual(
            inputs.positions.tolist(), list(range(10, 16)) + list(range(20, 26))
        )
        self.assertEqual(inputs.tokens[0].tolist(), [7, 3, 2, 1])
        self.assertEqual(inputs.tokens[6 + 2].tolist(), [15, 14, 13, 6])


def make_coordinator(n_layers=2, rows=50, dim=64, heads=6):
    gen = torch.Generator().manual_seed(0)
    embeddings = [
        (
            index,
            SimpleNamespace(
                weight=(torch.randn(rows, dim, generator=gen) * 0.5).to(
                    torch.float8_e4m3fn
                ),
                scale=torch.randint(
                    100, 140, (rows, dim // 32), generator=gen, dtype=torch.uint8
                ),
            ),
        )
        for index in range(n_layers)
    ]
    hasher = SimpleNamespace(
        max_ngram_size=3,
        token_map=torch.arange(10),
        multipliers=torch.ones(n_layers, 3, dtype=torch.int64),
        primes=torch.full((n_layers, 2, heads // 2), 7, dtype=torch.int64),
        offsets=torch.zeros(n_layers, heads, dtype=torch.int64),
        pad_id=2,
    )
    hashes = torch.randint(0, rows, (3, n_layers, heads), generator=gen)
    return HostEngramCoordinator(hasher, embeddings, "cpu"), hashes


class TestDeferredEngram(CustomTestCase):
    def _stage(self):
        w = TpuDsv41DsparkWorker.__new__(TpuDsv41DsparkWorker)
        w._tails = torch.tensor([[1, 2, 3], [4, 5, 6]])
        w._ngram = 4
        w.num_verify_rows = K + 1
        log = []

        class Coord:
            def prepare_verify(self, inputs):
                log.append(("hash", inputs))

            def prepare_verify_deferred(self, resolve):
                log.append(("defer",))
                self.resolve = resolve

        def engram_rows(tails, slots, verify_tokens, seq_lens):
            log.append(("rows",))
            return torch.cat(
                [tails[slots], verify_tokens.view(-1, K + 1), seq_lens[:, None]], 1
            )

        w._engram, w._engram_rows = Coord(), engram_rows
        state = SimpleNamespace(
            req_pool_indices=torch.tensor([1, 0]), seq_lens=torch.tensor([40, 0])
        )
        draft = SimpleNamespace(verify_tokens=torch.arange(2 * (K + 1)) + 10)
        w._stage_verify_engram(1, state, draft)
        return w._engram, log

    def test_worker_defers_the_d2h_and_hashes_the_same_inputs(self):
        coord, log = self._stage()
        # The rows op is issued at once; the D2H + hash wait for first use.
        self.assertEqual([e[0] for e in log], ["rows", "defer"])
        coord.resolve()
        self.assertEqual([e[0] for e in log], ["rows", "defer", "hash"])
        # The live request (bs = 1): slot 1's tail, verify tokens 10..15, n = 40.
        want = verify_inputs(
            np.array([[4, 5, 6]]), np.arange(10, 16)[None], np.array([40]), 4
        )
        b = log[-1][1]
        np.testing.assert_array_equal(want.tokens, b.tokens)
        np.testing.assert_array_equal(want.positions, b.positions)
        self.assertEqual(b.positions.tolist(), list(range(40, 46)))
        self.assertEqual(b.tokens[0].tolist(), [10, 6, 5, 4])

    def test_coordinator_resolves_once_on_first_stage_layer(self):
        coord, hashes = make_coordinator()
        calls = []

        def resolve():
            calls.append(1)
            coord.hashes = hashes.clone()

        coord.hashes = hashes.clone()
        coord.stage(8)
        eager = [e._tpu_engram_packed.clone() for _, e in coord.embeddings]
        coord.clear_step()
        self.assertFalse(coord.rows_prepared)

        coord.prepare_verify_deferred(resolve)
        self.assertTrue(coord.rows_prepared)
        self.assertEqual(calls, [])
        coord.stage(8)
        self.assertEqual(calls, [1])  # the second layer reuses the hashes
        for want, (_, e) in zip(eager, coord.embeddings):
            torch.testing.assert_close(e._tpu_engram_packed, want, rtol=0, atol=0)

        # clear_step drops an unused deferral without calling it.
        coord.clear_step()
        coord.prepare_verify_deferred(resolve)
        coord.clear_step()
        self.assertFalse(coord.rows_prepared)
        self.assertEqual(calls, [1])

        # A resolve that prepares no rows is an error.
        coord.prepare_verify_deferred(lambda: None)
        with self.assertRaises(RuntimeError):
            coord.stage_layer(0, 8)


class TestGreedyDraft(CustomTestCase):
    """Default (no --speculative-use-rejection-sampling): sampled rows keep the greedy chain, no q."""

    def _draft(self, greedy_draft, sampled=True):
        w = TpuDsv41DsparkWorker.__new__(TpuDsv41DsparkWorker)
        w._greedy_draft = greedy_draft
        w._all_drafts = {2: torch.full((2,), K)}
        chain = torch.arange(2 * K).view(2, K)
        calls = []
        w._draft_op = lambda *args: (chain, "hidden")

        def sample_drafts(hidden, bonus, temperatures, top_ks):
            calls.append(hidden)
            return chain + 100, torch.zeros(2, K, 7)

        w._sample_drafts = sample_drafts
        params = torch.ones(2) if sampled else None
        bucket = SimpleNamespace(
            state=SimpleNamespace(
                bonus_tokens=torch.tensor([7, 8]),
                seq_lens=torch.tensor([3, 4]),
                req_pool_indices=torch.tensor([1, 2]),
            ),
            live_mask=torch.ones(2, dtype=torch.bool),
            temperatures=params,
            top_ks=params,
        )
        return w.draft(None, bucket, None, None), calls

    def test_sampled_rows_draft_greedily_without_logits(self):
        result, calls = self._draft(True)
        self.assertEqual(calls, [])
        self.assertIsNone(result.draft_logits)  # verify -> accept_drafts_target_only
        self.assertTrue(
            torch.equal(result.draft_tokens, torch.arange(2 * K).view(2, K))
        )
        self.assertEqual(result.verify_tokens.view(2, K + 1)[:, 0].tolist(), [7, 8])

    def test_flag_off_samples_drafts_from_q(self):
        result, calls = self._draft(False)
        self.assertEqual(calls, ["hidden"])
        self.assertIsNotNone(result.draft_logits)
        greedy, _ = self._draft(False, sampled=False)
        self.assertIsNone(greedy.draft_logits)


class TestRejectionSamplingFlag(CustomTestCase):
    """--speculative-use-rejection-sampling passes the TPU gate for DSPARK only."""

    def _validate(self, algorithm, rejection, device="tpu"):
        from sglang.srt.arg_groups.speculative_hook import (
            _validate_tpu_speculative_args,
        )

        args = SimpleNamespace(
            device=device,
            speculative_algorithm=algorithm,
            speculative_eagle_topk=1,
            speculative_use_rejection_sampling=rejection,
            kv_canary="none",
        )
        _validate_tpu_speculative_args(args)

    def test_dspark_may_sample_drafts_on_tpu(self):
        self._validate("DSPARK", True)
        self._validate("DSPARK", False)

    def test_other_tpu_algorithms_still_refuse(self):
        for algorithm in ("EAGLE", "EAGLE3", "DFLASH"):
            with self.assertRaises(ValueError):
                self._validate(algorithm, True)
            self._validate(algorithm, False)

    def test_non_tpu_devices_are_not_gated_here(self):
        self._validate("EAGLE", True, device="cuda")


class TestFusedEpShapes(CustomTestCase):
    def test_moe_token_rows(self):
        decode = [1, 2, 4, 8, 16]
        self.assertEqual([moe_token_rows(t) for t in decode], decode)
        verify = [bs * (K + 1) for bs in decode]
        self.assertEqual([moe_token_rows(t) for t in verify], [8, 16, 24, 48, 96])
        draft = [bs * K for bs in decode]
        self.assertEqual([moe_token_rows(t) for t in draft], [8, 16, 24, 40, 80])
        self.assertEqual(moe_token_rows(8192), 8192)

    def test_op_name_follows_the_routing_geometry(self):
        target = _ep_op_name(6, True, "sqrtsoftplus", 1.5, False, 2048)
        drafter = _ep_op_name(3, True, "sqrtsoftplus", 1.5, False, 2048)
        self.assertNotEqual(target, drafter)
        self.assertEqual(target, _ep_op_name(6, True, "sqrtsoftplus", 1.5, False, 2048))
        self.assertNotEqual(
            target, _ep_op_name(6, True, "sqrtsoftplus", 1.5, True, 2048)
        )
        self.assertRegex(target, r"^sglang_tpu_pallas::[A-Za-z0-9_]+$")


class TestWorkerHostLogic(CustomTestCase):
    def test_advance_tails_follows_commits(self):
        rng = np.random.default_rng(5)
        seq = rng.integers(0, 100, size=200).tolist()
        tails = torch.zeros((4, 3), dtype=torch.int64)
        slots = torch.tensor([1, 3, 0])
        n = {1: 10, 3: 50}
        for slot in (1, 3):
            tails[slot] = torch.tensor(seq[n[slot] - 3 : n[slot]])
        for step in range(6):
            accept = {1: step % (K + 1), 3: (5 * step + 2) % (K + 1)}
            verify = []
            for slot in (1, 3):
                p = n[slot]
                drafts = [seq[p + 1 + j] if j < accept[slot] else 999 for j in range(K)]
                verify.append([seq[p]] + drafts)
            verify.append([0] * (K + 1))
            accept_lens = torch.tensor(
                [accept[1] + 1, accept[3] + 1, 0], dtype=torch.int32
            )
            advance_tails(tails, slots, accept_lens, torch.tensor(verify).reshape(-1))
            for slot in (1, 3):
                n[slot] += accept[slot] + 1
                self.assertEqual(tails[slot].tolist(), seq[n[slot] - 3 : n[slot]])

    def test_commit_rows_write_only_accepted(self):
        row_slots, write = commit_rows(
            torch.tensor([4, 0]), torch.tensor([2, 0]), K + 1
        )
        self.assertEqual(row_slots.tolist(), [4] * (K + 1) + [0] * (K + 1))
        self.assertEqual(write.tolist(), [True, True] + [False] * (2 * K))

    def test_committed_tail(self):
        req = SimpleNamespace(origin_input_ids=[5, 6, 7, 8], output_ids=[9, 10])
        self.assertEqual(committed_tail(req, 6, 3), [8, 9, 10])
        self.assertEqual(committed_tail(req, 4, 3), [6, 7, 8])
        self.assertEqual(committed_tail(req, 2, 3), [0, 5, 6])
        self.assertEqual(committed_tail(req, 7, 3), [0, 9, 10])  # missing token logged

    def test_extend_rows_write_last_ring_positions(self):
        host = extend_rows(
            prefix_lens=[0, 7],
            extend_lens=[RING + 10, 3],
            slots=[1, 2],
            floors=[0, 7],
            tails=[[1, 2, 3], [4, 5, 6]],
            rows=RING + 32,
        )
        live = RING + 13
        self.assertEqual(host["positions"][:3].tolist(), [0, 1, 2])
        self.assertEqual(host["positions"][RING + 10 : live].tolist(), [7, 8, 9])
        self.assertFalse(host["write"][:10].any())
        self.assertTrue(host["write"][10:live].all())
        self.assertFalse(host["write"][live:].any())
        self.assertEqual(host["slots"][live - 1].item(), 2)
        self.assertEqual(host["floors"][live - 1].item(), 7)
        self.assertEqual(host["tails"][0].tolist(), [1, 2, 3])
        self.assertEqual(host["tails"].shape, (RING + 32, 3))

    def test_floor_continues_chunks_and_resets_otherwise(self):
        ctx, req = ExtendContext(), make_req("r", list(range(400)))
        self.assertEqual(ctx.floor(req, 3, 100, 50), 100)  # radix prefix of 100
        self.assertEqual(ctx.floor(req, 3, 150, 50), 100)  # next chunk
        self.assertEqual(ctx.floor(req, 5, 200, 10), 200)  # moved slot
        self.assertEqual(ctx.floor(req, 5, 180, 10), 180)  # restarted (retraction)


def make_req(rid, tokens, output=()):
    """A real scheduler Req (attribute drift on Req fails here, not on TPU)."""
    req = Req(rid, "", array("q", tokens), SamplingParams())
    req.output_ids.extend(output)
    return req


def make_extend_batch(reqs, prefix_lens, extend_lens, slots):
    """A real ScheduleBatch with the fields prepare_for_extend fills."""
    batch = ScheduleBatch(reqs=reqs)
    batch.prefix_lens = list(prefix_lens)
    batch.extend_lens = list(extend_lens)
    batch.req_pool_indices_cpu = torch.tensor(slots, dtype=torch.int64)
    for req, slot in zip(reqs, slots):
        req.kv.req_pool_idx = slot
    return batch


class TestDpVerifyCounts(CustomTestCase):
    """Ranks holding 0 / 1 / 2 requests verify at one shared bucket (2)."""

    def _pad_sizes(self, counts_for_sync, raw):
        from sglang.srt.model_executor.forward_batch_info import ForwardBatch
        from sglang.srt.speculative.dflash_info import DFlashVerifyInput
        from sglang.srt.speculative.spec_info import spec_scale_global_num_tokens

        spec = DFlashVerifyInput(
            draft_token=torch.zeros(1, dtype=torch.int64),
            positions=torch.zeros(1, dtype=torch.int64),
            draft_token_num=K + 1,
            custom_mask=None,
            capture_hidden_mode=None,
        )
        padded_bs = 2  # TpuSpecWorker._decode_bucket: bucket of max(raw)
        rows = padded_bs * (K + 1)  # every rank's verify forward, idle ones too
        scaled, _ = spec_scale_global_num_tokens(spec, counts_for_sync, counts_for_sync)
        shell = ForwardBatch.__new__(ForwardBatch)
        sizes = []
        for rank in range(len(raw)):
            ids = torch.zeros(rows, dtype=torch.int64)
            try:  # prepare_mlp_sync_batch pads this rank's rows to its entry
                sizes.append(shell._pad_tensor_to_size(ids, scaled[rank]).shape[0])
            except RuntimeError:
                sizes.append(None)
        return sizes

    def test_shared_bucket_counts_pad_nothing(self):
        raw = [0, 1, 2, 1, 0, 2, 1, 1]
        counts = dp_verify_counts(raw, 2)
        self.assertEqual(counts, [2] * 8)
        self.assertEqual(self._pad_sizes(counts, raw), [12] * 8)

    def test_raw_counts_reproduce_the_negative_pad(self):
        raw = [0, 1, 2, 1, 0, 2, 1, 1]
        sizes = self._pad_sizes(raw, raw)
        self.assertEqual([s is None for s in sizes], [r < 2 for r in raw])

    def test_no_dp_and_bad_bucket(self):
        self.assertIsNone(dp_verify_counts(None, 4))
        with self.assertRaises(ValueError):
            dp_verify_counts([1, 3], 2)


class TestExtendContextRealBatch(CustomTestCase):
    def test_slots_floors_tails_from_the_batch(self):
        a = make_req("a", list(range(1000, 1320)))
        b = make_req("b", list(range(2000, 2052)))
        before = {id(r): set(vars(r)) for r in (a, b)}
        ctx = ExtendContext()
        # Chunk 1: a from scratch, b after a 40-token radix prefix.
        host = ctx.rows(
            make_extend_batch([a, b], [0, 40], [300, 12], [3, 5]), tail=3, rows=512
        )
        self.assertEqual(host["slots"][:300].unique().tolist(), [3])
        self.assertEqual(host["slots"][300:312].unique().tolist(), [5])
        self.assertEqual(host["floors"][299].item(), 0)
        self.assertEqual(host["floors"][300].item(), 40)
        self.assertEqual(host["tails"][0].tolist(), [1297, 1298, 1299])
        self.assertEqual(host["tails"][300].tolist(), [2049, 2050, 2051])
        self.assertFalse(host["write"][312:].any())
        # Chunk 2 of a continues; its floor stays 0.
        host = ctx.rows(make_extend_batch([a], [300], [20], [3]), tail=3, rows=128)
        self.assertEqual(host["floors"][0].item(), 0)
        self.assertEqual(host["tails"][0].tolist(), [1317, 1318, 1319])
        # Retraction: the scheduler swaps req.kv; a re-extends from a 100-token
        # prefix on another slot with its decoded tokens in output_ids.
        a.output_ids.extend([7, 8, 9])
        a.kv = ReqKvInfo()
        host = ctx.rows(make_extend_batch([a], [100], [223], [7]), tail=3, rows=256)
        self.assertEqual(host["floors"][0].item(), 100)
        self.assertEqual(host["slots"][0].item(), 7)
        self.assertEqual(host["tails"][0].tolist(), [7, 8, 9])
        # No attribute is added to Req; finished requests leave the floor map.
        self.assertEqual({id(r): set(vars(r)) for r in (a, b)}, before)
        self.assertFalse(hasattr(a, "req_pool_idx"))
        del a, b
        gc.collect()
        self.assertEqual(len(ctx._seen), 0)

    def test_misaligned_batch_raises(self):
        a = make_req("a", [1, 2, 3])
        batch = make_extend_batch([a], [0], [3], [2])
        batch.req_pool_indices_cpu = torch.empty(0, dtype=torch.int64)
        with self.assertRaises(ValueError):
            ExtendContext().rows(batch, tail=3, rows=8)


if __name__ == "__main__":
    unittest.main()
