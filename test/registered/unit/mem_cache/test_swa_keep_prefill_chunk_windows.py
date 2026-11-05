from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

import os
import random
import unittest
from array import array
from unittest import mock

import torch
from swa_cache_world import PAGE, POISON, SWA, WINDOW, KVWorld, kv_value

from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import FINISH_LENGTH, Req
from sglang.srt.mem_cache.common import checkpoint_kv_cache, release_kv_cache
from sglang.srt.sampling.sampling_params import SamplingParams

# The served shape scaled 1/32: 512-token pages -> 16, the 2047-token window
# (global_attn_cfg 2048 - 1) -> 63, 2048-token chunks -> 64, 4096 -> 128.
WIN_CHUNK, BIG_CHUNK = 64, 128


def _bf16(x: float) -> float:
    return float(torch.tensor(x).to(torch.bfloat16))


def _tokens(seed: int, n: int) -> list[int]:
    rng = random.Random(seed)
    return [rng.randrange(1, 50000) for _ in range(n)]


class PrefillWorld(KVWorld):
    """A real UnifiedRadixCache driven like the scheduler drives a request:
    admission match + lock, chunked prefill with checkpoint_kv_cache after
    every chunk, decode steps (one KV slot per token), release_kv_cache at the
    finish. Every slot gets the cold value of its token."""

    def __init__(self, *, keep: bool, full_tokens=8192, swa_tokens=4096):
        super().__init__(full_tokens=full_tokens, swa_tokens=swa_tokens)
        self.keep = keep

    def _env(self):
        # patch.dict restores the env even when the body raises
        return mock.patch.dict(
            os.environ, {"SGLANG_OPT_SWA_KEEP_PREFILL_CHUNK_WINDOWS": str(self.keep)}
        )

    # -- request lifecycle
    def admit(self, tokens: list[int], chunk: int, outputs=()) -> Req:
        # ``outputs``: tokens generated before a retraction; reset_for_retract
        # keeps output_ids, so the re-prefill covers prompt + outputs.
        fill = list(tokens) + list(outputs)
        with self._env():
            req = Req(
                rid=f"r{self.rid}",
                origin_input_text="",
                origin_input_ids=array("q", tokens),
                sampling_params=SamplingParams(temperature=0, max_new_tokens=1 << 20),
            )
            req.output_ids.extend(outputs)
            self.rid += 1
            self.req_to_token_pool.alloc([req])
            req.init_next_round_input(tree_cache=self.cache)
            req.lock_receipt = self.cache.inc_lock_ref(req.last_node).to_dec_params()
            cached = len(req.prefix_indices)
            row = req.kv.req_pool_idx
            if cached:
                self.req_to_token_pool.write(
                    (row, slice(0, cached)), req.prefix_indices
                )
            req.kv.kv_committed_len = req.kv.kv_allocated_len = cached
            req.cached = cached
            req.chunk_ends = []
            pos = cached
            while pos < len(fill):
                end = min(pos + chunk, len(fill))
                req.set_extend_range(pos, end)
                self.req_to_token_pool.write(
                    (row, slice(pos, end)), self._alloc(fill[pos:end], pos)
                )
                req.kv.kv_committed_len = req.kv.kv_allocated_len = end
                pos = end
                if pos < len(fill):
                    checkpoint_kv_cache(req, self.cache)
                    req.chunk_ends.append(pos)
                    req.init_next_round_input()
            req.output_ids.append(self._next_token(req))
            checkpoint_kv_cache(req, self.cache)
            req.chunk_ends.append(pos)
            return req

    def decode(self, req: Req, steps: int, evict_every: int = 0) -> list[int]:
        """``steps`` decode steps; with ``evict_every`` the scheduler's decode
        out-of-window eviction runs at that interval. Returns the request's SWA
        eviction cursor after every step."""
        cursors = []
        with self._env():
            for _ in range(steps):
                self.decode_step(req)
                if evict_every and req.decode_batch_idx % evict_every == 0:
                    self.cache.evict_sliding_windows(
                        req, req.seqlen - 1, eviction_interval=1
                    )
                cursors.append(req.kv.get_evicted_seqlen(SWA))
        return cursors

    def finish(self, req: Req) -> None:
        with self._env():
            req.finished_reason = FINISH_LENGTH(length=0)
            release_kv_cache(req, self.cache)

    def match_profile(self, tokens: list[int], seed: int) -> list[int]:
        """Matched length of ``tokens[:k]`` + a fresh page, for every k."""
        tail = _tokens(seed, PAGE)
        return [
            len(self.match(tokens[:k] + tail).device_indices)
            for k in range(1, len(tokens) + 1)
        ]

    def live_ranges(self, req: Req) -> list[tuple[int, int, bool]]:
        """(start, end, swa live) along the request's root path, root first."""
        swa = self.cache.components[SWA]
        node = self.cache.tree_core.node_by_id(req.last_node)
        out, end = [], 0
        path = []
        while node is not None and node.parent is not None:
            path.append(node)
            node = node.parent
        for n in reversed(path):
            out.append((end, end + len(n.key), n.component_data[SWA].value is not None))
            end += len(n.key)
        return out

    def dead_tokens(self, req: Req) -> list[tuple[int, int]]:
        """Merged [start, end) ranges with no live SWA on the request's path."""
        out = []
        for s, e, live in self.live_ranges(req):
            if live:
                continue
            if out and out[-1][1] == s:
                out[-1] = (out[-1][0], e)
            else:
                out.append((s, e))
        return out


class TestKeepPrefillChunkWindows(unittest.TestCase):
    PROMPT = 16 * PAGE  # 256 tokens: two 128-token chunks, four 64-token chunks

    def test_off_reproduces_the_big_chunk_free(self):
        """Without the flag a chunk longer than window + page frees its first
        rows at its insert (R7Q): pre_len 127 - 63 = 64 -> [0, 64) and
        [128, 192) are SWA tombstones, so a match can only end at a chunk end."""
        w = PrefillWorld(keep=False)
        a = w.admit(_tokens(1, self.PROMPT), BIG_CHUNK)
        self.assertEqual(a.chunk_ends, [128, 256])
        self.assertEqual(w.dead_tokens(a), [(0, 64), (128, 192)])
        prof = w.match_profile(_tokens(1, self.PROMPT), seed=90)
        self.assertEqual(prof[96 - 1], 0, "shared 96 -> nothing (3584 -> 0)")
        self.assertEqual(prof[112 - 1], 0, "the window ending at 112 is gone")
        self.assertEqual(prof[192 - 1], 128, "shared 192 -> back to the chunk end")
        w.finish(a)
        w.cache.sanity_check()

    def test_window_chunks_keep_every_page_live(self):
        """Control: 64-token chunks free nothing at their inserts, flag or not."""
        for keep in (False, True):
            w = PrefillWorld(keep=keep)
            a = w.admit(_tokens(1, self.PROMPT), WIN_CHUNK)
            self.assertEqual(w.dead_tokens(a), [])

    def test_on_big_chunks_match_like_window_chunks(self):
        """With the flag, one 128-token chunk insert leaves the same match point
        for every prefix length as two 64-token inserts - including a request
        whose chunks start after a prefix hit."""
        base = _tokens(2, self.PROMPT)
        second = base[:80] + _tokens(3, 300)
        profiles = {}
        for name, keep, chunk in (
            ("q2048", False, WIN_CHUNK),
            ("v2", True, BIG_CHUNK),
            ("r7q", False, BIG_CHUNK),
        ):
            w = PrefillWorld(keep=keep)
            a = w.admit(base, chunk)
            b = w.admit(second, chunk)
            self.assertEqual(b.cached, 80 if name != "r7q" else 0, name)
            if name == "v2":
                self.assertEqual(b.chunk_ends, [208, 336, 380])
                self.assertEqual(w.dead_tokens(a), [])
                self.assertEqual(w.dead_tokens(b), [])
            profiles[name] = (
                w.match_profile(base, seed=91),
                w.match_profile(second, seed=92),
            )
            w.finish(a)
            w.finish(b)
            w.cache.sanity_check()
        self.assertEqual(profiles["v2"], profiles["q2048"])
        self.assertNotEqual(profiles["r7q"], profiles["q2048"], "the probe can see R7Q")

    def test_on_branch_insert_keeps_the_chunk_live(self):
        """A FULL hit past the last live window (evicted SWA) makes the insert
        stop at the branch point and then free on the whole forward key; with
        the flag that free also keeps everything above the tree-owned prefix."""
        base = _tokens(4, self.PROMPT)
        again = base[:200] + _tokens(5, 100)
        profiles, dead = {}, {}
        for name, keep, chunk in (
            ("q2048", False, WIN_CHUNK),
            ("v2", True, BIG_CHUNK),
            ("r7q", False, BIG_CHUNK),
        ):
            w = PrefillWorld(keep=keep)
            a = w.admit(base, chunk)
            w.finish(a)
            w.evict(swa_tokens=10**6)  # tombstone every unlocked window
            pre = w.match(again)
            self.assertEqual(len(pre.device_indices), 0)
            self.assertIsNotNone(pre.swa_branching_seqlen, name)
            b = w.admit(again, chunk)
            dead[name] = w.dead_tokens(b)
            profiles[name] = w.match_profile(again, seed=93)
            w.finish(b)
            w.cache.sanity_check()
        self.assertEqual(dead["v2"], [])
        self.assertEqual(dead["q2048"], [])
        self.assertNotEqual(dead["r7q"], [])
        self.assertEqual(profiles["v2"], profiles["q2048"])

    def test_default_off(self):
        env = {
            k: v
            for k, v in os.environ.items()
            if k != "SGLANG_OPT_SWA_KEEP_PREFILL_CHUNK_WINDOWS"
        }
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertFalse(envs.SGLANG_OPT_SWA_KEEP_PREFILL_CHUNK_WINDOWS.get())

    def test_retain_floor_keeps_the_lower_floor(self):
        """swa_retain_floor (mamba checkpoints) combines by min: a floor above the
        tree-owned prefix keeps the chunk live, one below it keeps more."""
        for floor in (10**9, 0, None):
            w = PrefillWorld(keep=True)
            with mock.patch.object(w.cache, "swa_retain_floor", lambda req: floor):
                a = w.admit(_tokens(1, self.PROMPT), BIG_CHUNK)
            self.assertEqual(w.dead_tokens(a), [], floor)

    def _reprefill_dead_tokens(self, n_out: int) -> dict:
        """A retracted request (reset_for_retract keeps output_ids) re-prefills
        its 112-token prompt + ``n_out`` outputs as one chunk on a cold tree, so
        its prefill-end insert has pre_len 112 + n_out - 1 with nothing of it
        tree-owned: without the flag that insert frees [0, 48) (pre_len - 63,
        page-floored)."""
        n, dead = 7 * PAGE, {}
        for keep in (False, True):
            w = PrefillWorld(keep=keep)
            a = w.admit(_tokens(10, n), BIG_CHUNK, outputs=_tokens(11, n_out))
            self.assertEqual(a.chunk_ends, [n + n_out])
            dead[keep] = w.dead_tokens(a)
            w.finish(a)
            w.cache.sanity_check()
        return dead

    def test_prefill_end_insert_keeps_the_chunk_live(self):
        """pre_len == len(origin_input_ids) (key = prompt + first output) is the
        prefill-end insert: with the flag it frees nothing above the tree-owned
        prefix."""
        dead = self._reprefill_dead_tokens(1)
        self.assertEqual(dead, {False: [(0, 48)], True: []})

    def test_first_decode_insert_frees_like_off(self):
        """pre_len == len(origin_input_ids) + 1 is the first decode insert: it
        frees out of window exactly as without the flag, though the flag kept
        the same rows live one token earlier."""
        kept = self._reprefill_dead_tokens(1)[True]
        self.assertEqual(kept, [], "the probe can see the flag")
        dead = self._reprefill_dead_tokens(2)
        self.assertEqual(dead, {False: [(0, 48)], True: [(0, 48)]})

    def test_decode_eviction_is_unchanged(self):
        """The flag only touches prefill-phase inserts: decode out-of-window
        eviction frees the same slots, flag on or off."""
        runs = {}
        for keep in (False, True):
            w = PrefillWorld(keep=keep)
            a = w.admit(_tokens(6, 200), BIG_CHUNK)
            free0 = w.allocator.swa_available_size()
            cursors = w.decode(a, 3 * WINDOW, evict_every=8)
            runs[keep] = (cursors, free0 - w.allocator.swa_available_size())
            final = a.seqlen - 1 - max(WINDOW, PAGE)
            self.assertGreaterEqual(cursors[-1], (final - 8) // PAGE * PAGE)
            w.finish(a)
            w.cache.sanity_check()
        self.assertEqual(runs[True][0], runs[False][0])
        self.assertGreater(runs[True][0][-1], 200, "decode slots were freed")

    def test_pool_accounting_returns_to_baseline(self):
        w = PrefillWorld(keep=True)
        full0 = w.allocator.full_available_size()
        swa0 = w.allocator.swa_available_size()
        reqs = [
            w.admit(_tokens(7, 300), BIG_CHUNK),
            w.admit(_tokens(7, 150) + _tokens(8, 250), BIG_CHUNK),
        ]
        for r in reqs:
            w.decode(r, 2 * WINDOW, evict_every=16)
        w.cache.sanity_check()
        for r in reqs:
            w.finish(r)
        w.cache.sanity_check()
        self.assertEqual(w.cache.tree_core.component_protected_size_[SWA], 0)
        w.evict(swa_tokens=10**6, full_tokens=10**6)
        self.assertEqual(w.allocator.full_available_size(), full0)
        self.assertEqual(w.allocator.swa_available_size(), swa0)

    def test_warm_window_inside_a_big_chunk_equals_cold(self):
        """Hits that end at 64-aligned points inside a 128-token chunk read the
        values the chunk's prefill wrote; freed slots hold poison."""
        w = PrefillWorld(keep=True)
        base = _tokens(9, self.PROMPT)
        a = w.admit(base, BIG_CHUNK)
        w.finish(a)
        others = [w.admit(_tokens(50 + i, 200), BIG_CHUNK) for i in range(2)]
        for o in others:
            w.finish(o)
        w.poison_free_swa()
        want = [_bf16(kv_value(t, p)) for p, t in enumerate(base)]
        for k in (64, 80, 96, 112, 192, 208, 224, 240):
            m = w.match(base[:k] + _tokens(60 + k, PAGE))
            n = len(m.device_indices)
            self.assertEqual(n, k, f"a hit ends at {k}")
            full = m.device_indices.to(torch.int64)
            self.assertEqual(w.full_values(full), want[:n])
            lo = max(0, n - (WINDOW + 1))
            got = w.swa_values(full[lo:])
            self.assertEqual(got, want[lo:n], f"warm SWA window at {k} == cold")
            self.assertNotIn(POISON, got)
        w.cache.sanity_check()


if __name__ == "__main__":
    unittest.main()
