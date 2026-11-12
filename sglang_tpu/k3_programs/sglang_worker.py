# SPDX-License-Identifier: Apache-2.0
"""The Kimi-K3 programs as an sglang speculative worker.

The decode program (``kimi_k3.decode.KimiDecoder``) runs a whole step in its
ops: DSpark drafts, the target's verify, acceptance and sampling, keeping each
request's next input on the device. sglang's speculative workers have the
same shape from the scheduler's side -- a decode batch in, each request's
accepted tokens out -- so the program serves as the worker of a
plugin-registered algorithm, ``K3_PROGRAM``, the way vLLM-TorchTPU's
``MegakernelModelRunner`` serves it under vLLM (``runner.py`` at ``c78529c``).
On a P/D prefill server the same worker runs the prefill program
(:class:`PrefillSteps`) on sglang's chunked prefill. A decode step hands over:

* every request of the batch goes in as host rows (:mod:`sglang_step`):
  ``computed`` its committed KV, ``scheduled`` the next token plus the drafts
  the schedule gives the batch size, its block tables over the unified pools
  (:mod:`sglang_bridge`);
* requests that left the batch since the last step go in as finished, and
  the program drops their state;
* the step's ``[>= batch, 8]`` tokens, ``-1`` padded, come back as the flat
  ``[batch * 8]`` rows and accept lengths sglang's output processor reads.

The scheduler sizes each round's KV slots from the speculative settings
``K3ProgramAlgo.handle_server_args`` pins (eight tokens a round) and advances
``kv_committed_len`` by the accept lengths. The program reads tokens from the
host only on a request's first step (the prefill's first token, its anchor),
so a request decodes from where the P/D transfer left it.
"""

from __future__ import annotations

import collections
import logging
import math
import time
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch

from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker
from sglang.srt.speculative.spec_info import SpecInput, SpecInputType
from sglang.srt.speculative.spec_registry import CustomSpecAlgo
from sglang_tpu.k3_programs import sglang_bridge as bridge
from sglang_tpu.k3_programs import sglang_dp, sglang_step
from sglang_tpu.k3_programs.program import MegakernelProgram, StepInput

logger = logging.getLogger(__name__)

ALGORITHM = "K3_PROGRAM"
# Tokens a step returns per request: up to seven accepted drafts and the bonus.
WIDTH = 8
# Colocated: prefill first tokens kept for decode steps (far above the requests
# that can sit between a prefill and their first decode step).
FIRST_TOKENS_KEPT = 4096


def drafts_for(schedule: Sequence[tuple[int, int, int]] | None, batch: int) -> int:
    """The drafts a step of ``batch`` requests verifies (0: a plain step)."""
    for low, high, drafts in schedule or ():
        if low <= batch <= high:
            return drafts
    return 0


class K3ProgramInput(SpecInput):
    """The batch's speculative input. The program keeps each request's next
    input and drafts itself, so there is nothing to carry between rounds."""

    accept_lens_device = None
    future_indices = None
    # What overlap scheduling relays through its future map (the EAGLE
    # draft-input fields): only each request's last token is set, unused.
    bonus_tokens = None
    topk_p = None
    topk_index = None
    hidden_states = None
    draft_probs = None
    dsa_topk_indices = None
    future_dsa_topk_indices_available = False
    device_accept_selected_cache_locs = None

    def __init__(self) -> None:
        super().__init__(SpecInputType.NGRAM_VERIFY)

    def filter_batch(self, new_indices, new_indices_cpu=None) -> None:
        # Overlap scheduling reads the batch's lengths through its future
        # indices, which follow the batch's requests.
        if self.future_indices is not None:
            self.future_indices = self.future_indices[new_indices]

    def merge_batch(self, spec_info: K3ProgramInput) -> None:
        if self.future_indices is None:
            self.future_indices = spec_info.future_indices
            return
        assert spec_info.future_indices is not None, (
            "K3_PROGRAM: merging a batch without future indices under overlap"
        )
        self.future_indices = torch.cat([self.future_indices, spec_info.future_indices])


class K3ProgramAlgo(CustomSpecAlgo):
    """``K3_PROGRAM``: the members a registered algorithm needs beyond
    ``CustomSpecAlgo``'s defaults."""

    def need_topk(self) -> bool:
        return False

    def supports_kv_free_fence(self, *, copies_draft_tail: bool) -> bool:
        # The tokens and accept lengths the result copies derive from the
        # step's emitted tokens, the output of its last op (which commits the
        # target's and the draft's KV); the device runs ops in launch order.
        return True

    def create_future_map(
        self,
        device,
        req_to_token_pool,
        needs_cpu_seq_lens: bool = True,
        needs_confidence_relay: bool = False,
    ):
        from sglang.srt.managers.overlap_utils import FutureMap

        return FutureMap(
            device, self, req_to_token_pool, needs_cpu_seq_lens, needs_confidence_relay
        )

    def handle_server_args(self, server_args: Any) -> None:
        """The settings the programs serve under: a round's KV reservation
        for eight tokens, KV pages of one logical block and the scheduler
        synchronous (the worker runs each step to its tokens). The decode
        runs without the radix cache (a request's KDA state lives in the
        program's slots). A prefill with the radix cache uses the hybrid
        cache's ``extra_buffer`` strategy, chunked a page at a time so every
        full chunk ends on a checkpoint the cache keeps."""
        from sglang_tpu.k3_programs.kimi_k3.layout import cache_layout
        from sglang_tpu.k3_programs.sglang_config import program_kv_cache_dtype

        server_args.speculative_num_steps = WIDTH - 1
        server_args.speculative_eagle_topk = 1
        server_args.speculative_num_draft_tokens = WIDTH
        layout = cache_layout(program_kv_cache_dtype(server_args.kv_cache_dtype))
        server_args.page_size = bridge.TP_SIZE * layout.tokens_per_block
        role = getattr(server_args, "disaggregation_mode", None)
        from sglang_tpu.envs import envs as tpu_envs

        server_args.disable_overlap_schedule = not (
            role in ("decode", None, "null")
            and tpu_envs.SGLANG_TPU_K3_PROGRAM_OPT_OVERLAP.get()
        )
        colocated = role in (None, "null")
        if (
            role == "decode"
            and tpu_envs.SGLANG_TPU_K3_PROGRAM_ENABLE_DECODE_PREFIX_CACHE.get()
            and not getattr(server_args, "disable_radix_cache", False)
        ):
            # KV pages only (sglang_decode_cache): the transfer delivers each
            # request's KDA state, so the tree tracks none.
            server_args.disaggregation_decode_enable_radix_cache = True
            server_args.mamba_radix_cache_strategy = "extra_buffer"
            server_args.mamba_track_interval = server_args.page_size * (1 << 16)
            return
        if not (role == "prefill" or colocated) or getattr(
            server_args, "disable_radix_cache", False
        ):
            server_args.disable_radix_cache = True
            return
        server_args.mamba_radix_cache_strategy = "extra_buffer"
        server_args.chunked_prefill_size = server_args.page_size
        if colocated:
            # The decode program keeps a request's KDA state in its own slots:
            # no decode step may be tracked into the radix cache (the interval
            # lies past any context), only the prefill's chunk checkpoints.
            server_args.mamba_track_interval = server_args.page_size * (1 << 16)
        else:
            server_args.mamba_track_interval = server_args.page_size

    def build_disagg_draft_input(self, batch, last_tokens_tensor, future_map):
        # A request the P/D transfer delivered: the program takes its first
        # token from the host on its first step. Under overlap scheduling the
        # future map carries the batch's lengths and last tokens (as for
        # DSpark's disaggregated decode).
        spec = K3ProgramInput()
        spec.bonus_tokens = last_tokens_tensor
        if getattr(batch, "enable_overlap", False):
            from sglang.srt.managers.overlap_utils import RelayPayload

            spec.future_indices = batch.req_pool_indices
            future_map.publish(spec.future_indices, batch.seq_lens)
            future_map.stash(spec.future_indices, RelayPayload.from_draft_input(spec))
        return spec


class _Prompts:
    """Each live request's prompt as an int32 array, built once (sglang keeps
    token ids in Python lists; converting a 300k-token prompt every step costs
    more host time than the program's whole decode step)."""

    def __init__(self) -> None:
        self._arrays: dict[str, np.ndarray] = {}

    def get(self, req: Any) -> np.ndarray:
        array = self._arrays.get(req.rid)
        if array is None:
            array = np.asarray(req.origin_input_ids, dtype=np.int32)
            self._arrays[req.rid] = array
        return array

    def drop(self, rids) -> None:
        for rid in rids:
            self._arrays.pop(rid, None)


# Colocated programs' step costs for prefill_deferral_rounds (tpu7x 2x2x4, fp8
# KV, EP16 experts): a prefill batch ~1 s per 16384 rows, a DSpark decode
# round ~10 ms plus ~1.6 ms a request.
PREFILL_STEP_S, PREFILL_ROW_S = 0.05, 6e-5
DECODE_STEP_S, DECODE_REQUEST_S = 0.010, 0.0016


def prefill_deferral_rounds(prefill_rows: int, running: int, share: float) -> int:
    """Decode rounds to run after a prefill batch of ``prefill_rows`` so that
    prefill takes about ``share`` of a colocated engine's time while
    ``running`` requests decode: none when nothing decodes (a prefill delays no
    one), more after a long batch than after a short one (a fixed count lets a
    cold prompt's full chunks starve decoding, or holds a short turn back)."""
    if running <= 0 or prefill_rows <= 0:
        return 0
    prefill_s = PREFILL_STEP_S + PREFILL_ROW_S * prefill_rows
    decode_s = DECODE_STEP_S + DECODE_REQUEST_S * running
    return max(1, min(512, math.ceil(prefill_s * (1 - share) / share / decode_s)))


def dp_pages(plan) -> "sglang_dp.DpPages | None":
    """The DP2 page regions over ``plan``'s KV pages under ``attention_dp=2``
    (None without attention DP)."""
    from sglang_tpu.k3_programs.kimi_k3.config import active

    if active().attention_dp == 1:
        return None
    if active().attention_dp != sglang_dp.GROUPS:
        raise ValueError(
            f"K3 decode: attention DP{active().attention_dp} is unsupported (DP2 only)"
        )
    if plan.layout.tokens_per_block != sglang_dp.CHUNK_TOKENS:
        raise ValueError(
            "K3 decode: attention DP2 runs with the fp8 KV cache (512 tokens a row)"
        )
    return sglang_dp.DpPages(plan.kv_blocks // 2 * 2, plan.num_blocks)


class StepClock:
    """Where a role's steps spend host time: each phase between marks, and
    the gap since the previous step (the scheduler's own work; gaps over a
    second are idle and not counted). Logged every ``every`` steps when
    ``enabled`` (the worker enables rank 0)."""

    IDLE = 1.0
    SLOW = 1.0  # a step this long (seconds) is logged on its own

    def __init__(self, name: str, every: int = 1000) -> None:
        self.name, self.every = name, every
        self.enabled = False
        self.sums: collections.Counter = collections.Counter()
        self.step_sums: collections.Counter = collections.Counter()
        self.steps = self.rows = 0
        self._last_end: float | None = None
        self._at = 0.0

    def start(self) -> None:
        now = time.perf_counter()
        if self._last_end is not None and now - self._last_end < self.IDLE:
            self.sums["scheduler"] += now - self._last_end
        self._at = now

    def mark(self, phase: str) -> None:
        now = time.perf_counter()
        self.sums[phase] += now - self._at
        self.step_sums[phase] += now - self._at
        self._at = now

    def stop(self, rows: int) -> None:
        self._last_end = time.perf_counter()
        self.steps += 1
        self.rows += rows
        if self.enabled and sum(self.step_sums.values()) > self.SLOW:
            logger.info(
                "K3 %s: slow step (%d rows); ms: %s",
                self.name,
                rows,
                ", ".join(f"{k} {1e3 * v:.1f}" for k, v in self.step_sums.items()),
            )
        self.step_sums.clear()
        if self.steps % self.every:
            return
        if self.enabled:
            phases = ", ".join(
                f"{phase} {1e3 * total / self.every:.2f}"
                for phase, total in self.sums.items()
            )
            logger.info(
                "K3 %s: %d steps, %.1f rows a step; ms a step: %s",
                self.name,
                self.every,
                self.rows / self.every,
                phases,
            )
        self.sums.clear()
        self.rows = 0


_COLO_STEPS: list = []


def colo_steps() -> "DecodeSteps | None":
    """The process's colocated DP2 decode steps (B rows), once built."""
    return _COLO_STEPS[-1] if _COLO_STEPS else None


class DecodeSteps:
    """A decode program's steps from sglang requests (no sglang objects
    beyond the requests' fields, so it runs on CPU against a stub program)."""

    # Departed request ids remembered to reject a resumed one.
    _DEPARTED_MEMORY = 4096

    def __init__(
        self,
        program: MegakernelProgram,
        plan: bridge.UnifiedPoolPlan,
        *,
        table_width: int,
        schedule: Sequence[tuple[int, int, int]] | None,
    ) -> None:
        self.program = program
        self.plan = plan
        self.table_width = table_width
        self.schedule = schedule
        # Attention DP2 (sglang_dp): two ids a page, 8192-token table blocks;
        # ``swap`` moves rows landed on the other group (set at build).
        self.dp = dp_pages(plan)
        self.table_tokens = (
            sglang_dp.GROUP_BLOCK_TOKENS if self.dp else plan.logical_block_tokens
        )
        self.swap = None
        # Colocated attention DP2 (sglang_dp.ColoRows, set at build): B rows.
        self.colo: sglang_dp.ColoRows | None = None
        self._live: set[str] = set()
        self._departed: dict[str, None] = {}
        self._prompts = _Prompts()
        self.clock = StepClock("decode")
        # Colocated: the prefill's first tokens by request, for a request whose
        # prefill result the overlapped scheduler has not processed yet.
        self.first_tokens: dict[str, int] = {}

    def step(self, reqs: Sequence[Any], starts: np.ndarray) -> tuple:
        """One step for ``reqs`` (sglang ``Req``s, batch order) with their
        ``req_to_token`` rows' :func:`sglang_bridge.page_starts`: the flat
        ``[len(reqs) * WIDTH]`` int64 tokens and the ``[len(reqs)]`` int32
        accept lengths."""
        tokens = self.launch(reqs, starts)
        rows = np.asarray(tokens.cpu(), dtype=np.int64)
        self.clock.mark("tokens")
        _, lens = sglang_step.accepted(rows, len(reqs))
        if not all(lens):
            idle = [reqs[i].rid for i, n in enumerate(lens) if not n]
            raise RuntimeError(f"K3 decode: requests {idle} took no token this step")
        return rows.reshape(-1), np.asarray(lens, np.int32)

    def launch(
        self, reqs: Sequence[Any], starts: np.ndarray, *, lagging: bool = False
    ) -> torch.Tensor:
        """Launch one step for ``reqs``: its ``[len(reqs), WIDTH]`` tokens on
        the device, -1 padded (no host sync). ``lagging``: under overlap
        scheduling a running request's committed length trails its device
        position by up to a step, so its tables reach a step further (within
        the KV sglang allocated); the program tracks its position itself."""
        scheduled = 1 + drafts_for(self.schedule, len(reqs))
        requests = []
        swaps: list = []
        jobs: list = []  # colocated DP2 copies before the step
        for req in reqs:
            if req.rid in self._departed:
                # The program dropped its KDA state when it left the batch; the
                # slot block holds the prefill's, not the state after its outputs.
                raise RuntimeError(
                    f"K3 decode: request {req.rid} came back after leaving the "
                    "batch (a retraction); the programs do not resume requests"
                )
            prompt = self._prompts.get(req)
            if req.rid in self._live:
                computed = int(req.kv.kv_committed_len)
                num_tokens = len(prompt) + len(req.output_ids)
                end = computed + scheduled
                token_ids = prompt  # the program holds its next input
                if lagging:
                    end += WIDTH
                    allocated = int(getattr(req.kv, "kv_allocated_len", 0) or 0)
                    if allocated:
                        end = min(end, allocated)
            else:
                # A request's first step: its whole prompt computed (prefilled
                # or transferred), then the prefill's token.
                outputs = list(req.output_ids)
                first = self.first_tokens.pop(req.rid, None)
                if not outputs:
                    if first is None:
                        raise RuntimeError(
                            f"K3 decode: request {req.rid} has no first token"
                        )
                    outputs = [first]
                computed = len(prompt)
                num_tokens = computed + len(outputs)
                end = computed + scheduled
                token_ids = np.concatenate((prompt, np.asarray(outputs, np.int32)))
            # Near the context limit the scheduled (and lagging) extent can
            # pass the last page the tables hold; the program stops there.
            end = min(end, self.table_width * self.table_tokens)
            slot = int(req.kv.mamba_pool_idx.reshape(-1)[0])
            if self.colo is not None:
                row = starts[req.kv.req_pool_idx]
                pages = (
                    np.asarray(row[: -(-end // sglang_dp.PAGE_TOKENS)], np.int64)
                    // sglang_dp.PAGE_TOKENS
                )
                if (pages == 0).any():  # past the KV sglang allocated so far
                    pages = pages[: int(np.argmax(pages == 0))]
                if req.rid in self._live and req.rid in self.colo:
                    self.colo.extend(req.rid, pages, end)
                else:
                    jobs.append(self.colo.admit(req.rid, pages, computed, end))
                tables = (
                    self.colo.table_row(req.rid, end, self.table_width),
                    *bridge.kda_table_rows(slot, self.plan),
                )
            elif self.dp is None:
                tables = bridge.block_tables(
                    starts[req.kv.req_pool_idx], end, slot, self.plan, self.table_width
                )
            else:
                row = starts[req.kv.req_pool_idx]
                pages = (
                    np.asarray(row[: -(-end // sglang_dp.PAGE_TOKENS)], np.int64)
                    // sglang_dp.PAGE_TOKENS
                )
                tables = (
                    self.dp.table_row(pages, end, self.table_width),
                    *bridge.kda_table_rows(slot, self.plan),
                )
                transfer = getattr(req, "_k3_dp_transfer", None)
                if (
                    req.rid not in self._live
                    and transfer is not None
                    and len(transfer[1])
                ):
                    swaps.append((req, transfer))
            requests.append(
                sglang_step.request_input(
                    rid=req.rid,
                    token_ids=token_ids,
                    num_tokens=num_tokens,
                    num_prompt_tokens=len(prompt),
                    computed=computed,
                    scheduled=scheduled,
                    block_tables=tables,
                    sampling_params=req.sampling_params,
                )
            )
        if swaps:
            self._swap(swaps)
        if jobs:
            self._run_jobs(jobs)
        current = {req.rid for req in reqs}
        finished, self._live = self._live - current, current
        self._forget(finished)
        self.clock.mark("inputs")
        tokens = self.program.step(StepInput(requests, finished)).tokens
        self.clock.mark("dispatch")
        return tokens[: len(reqs)]

    def _swap(self, swaps) -> None:
        """Move each new request's rows that landed on the other DP group to
        its own group (one op), then give the landing pages back (held until
        the step that reads them has run)."""
        from sglang_tpu.k3_programs import sglang_dp_alloc

        if self.swap is None:
            raise RuntimeError(
                "K3 decode: DP2 transfers landed rows but no swap op is bound"
            )
        jobs = []
        for _, (pages, landing) in swaps:
            sources, destinations = self.dp.swap_rows(pages, landing)
            jobs.append((sources, destinations, self.dp.group(int(pages[0]))))
        self.swap(jobs)
        allocator = sglang_dp_alloc.active()
        for req, (pages, landing) in swaps:
            if allocator is not None:
                allocator.free_landing(torch.as_tensor(np.asarray(landing, np.int32)))
            req._k3_dp_transfer = (pages, ())

    def _run_jobs(self, jobs) -> None:
        if self.swap is None:
            raise RuntimeError("K3 decode: DP2 B rows need the swap op, none is bound")
        jobs = [job for job in jobs if job is not None and len(job[0])]
        if jobs:
            self.swap(jobs)

    def colo_release(self, rid: str) -> None:
        """Colocated DP2: request ``rid`` left (its KV goes to the radix
        cache): its decode-written halves back to the DCP32 rows, now, ahead
        of any prefill reading them; its B rows freed."""
        if self.colo is not None and rid in self.colo:
            self._run_jobs([self.colo.release(rid)])

    def _forget(self, finished: set[str]) -> None:
        for rid in finished:
            self.colo_release(rid)
        self._prompts.drop(finished)
        for rid in finished:
            self._departed[rid] = None
        while len(self._departed) > self._DEPARTED_MEMORY:
            del self._departed[next(iter(self._departed))]

    def forget_all(self) -> None:
        """Drop every request's state (a cache flush)."""
        if self._live:
            self.program.step(StepInput((), set(self._live)))
        for rid in list(self._live):
            self.colo_release(rid)
        self._prompts.drop(self._live)
        self._live = set()
        self._departed.clear()
        self.first_tokens.clear()


class PrefillSteps:
    """A prefill program's steps from sglang's chunked prefill: each extend
    batch's chunks in, the first token of every prompt that finishes out. The
    program keeps a request's KDA state between its chunks and writes the
    state after each chunk into the request's slot blocks, its KV and the
    draft's into its pages, for the P/D transfer.

    With prefix ``tails`` (:mod:`sglang_tails`), a prompt's last chunk is cut
    at its last hit-aligned token past its last full page: the program
    checkpoints the KDA state there into a tail slot (vLLM's ``align`` mode:
    a step ending inside a block is folded into the next window, which writes
    the state at its end into the blocks the step named), and the partial
    page goes to the tail's KV block. A later prompt continuing it starts
    there: its new page takes a copy of the tail's block, and its hit column
    names the tail's state."""

    def __init__(
        self,
        program: MegakernelProgram,
        plan: bridge.UnifiedPoolPlan,
        *,
        table_width: int,
        kda_width: int = 1,
        tails: Any = None,
        kv_pool: Any = None,
        state_pool: Any = None,
    ) -> None:
        self.program = program
        self.plan = plan
        self.table_width = table_width
        self.kda_width = kda_width
        self.tails = tails
        self.kv_pool, self.state_pool = kv_pool, state_pool
        self._live: set[str] = set()
        self._prompts = _Prompts()
        self.stats = collections.Counter()
        self.clock = StepClock("prefill", every=200)
        # Colocated: shared with the decode steps (DecodeSteps.first_tokens).
        self.first_tokens: dict[str, int] | None = None

    def _row(self, req, prompt, computed, end, starts, columns=()):
        """The program's row for ``req`` computing ``[computed, end)``; each
        of ``columns`` ``(column, tail)`` names a tail's state blocks in that
        column of the KDA tables (the rest name the request's slot)."""
        tables = bridge.block_tables(
            starts[req.kv.req_pool_idx],
            end,
            int(req.kv.mamba_pool_idx.reshape(-1)[0]),
            self.plan,
            self.table_width,
            self.kda_width,
        )
        if columns:
            tables = list(tables)
            for group in range(bridge.KDA_PER_POOL):
                row = tables[1 + group].copy()
                for column, tail in columns:
                    row[column] = self.plan.tail_state_block(tail, group)
                tables[1 + group] = row
            tables = tuple(tables)
        return sglang_step.request_input(
            rid=req.rid,
            token_ids=prompt,
            num_tokens=len(prompt),
            num_prompt_tokens=len(prompt),
            computed=int(computed),
            scheduled=int(end - computed),
            block_tables=tables,
            sampling_params=req.sampling_params,
        )

    def step(
        self,
        reqs: Sequence[Any],
        prefix_lens: Sequence[int],
        extend_lens: Sequence[int],
        starts: np.ndarray,
    ) -> np.ndarray:
        """One extend batch, with the ``req_to_token`` rows'
        :func:`sglang_bridge.page_starts`: the ``[len(reqs)]`` int64 first
        tokens (0 for a request whose prompt continues in a later chunk)."""
        page, align = self.plan.logical_block_tokens, self.plan.layout.tokens_per_block
        tails = self.tails
        if tails is not None:
            tails.begin_step()
        first_rows, split_rows, split_index = [], [], []
        hit_copies, tail_kv_copies, tail_state_copies = [], [], []
        for i, (req, prefix, extend) in enumerate(zip(reqs, prefix_lens, extend_lens)):
            prompt = self._prompts.get(req)
            computed, end = int(prefix), int(prefix) + int(extend)
            columns = []
            hit_tail = None
            if tails is not None and req.rid not in self._live:
                limit = min(end - 1, len(prompt) - 1) // align * align
                hit = tails.lookup(prompt, computed, limit)
                if hit is not None:
                    tail, tail_end = hit
                    hit_tail = tail
                    page_block = (
                        int(starts[req.kv.req_pool_idx][computed // page]) // page
                    )
                    hit_copies.append((self.plan.tail_kv_block(tail), page_block))
                    columns.append((tail_end // align - 1, tail))
                    self.stats["tail_hit_tokens"] += tail_end - computed
                    computed = tail_end
            cut = None
            if tails is not None and end == len(prompt):
                tail_end = end // align * align
                if tail_end > end // page * page and tail_end > computed:
                    tail = tails.record(prompt, tail_end, own=hit_tail)
                    if tail is not None:
                        cut = (tail_end, tail)
            if cut is None:
                first_rows.append(
                    self._row(req, prompt, computed, end, starts, columns)
                )
                continue
            tail_end, tail = cut
            page_block = int(starts[req.kv.req_pool_idx][tail_end // page]) // page
            tail_kv_copies.append((page_block, self.plan.tail_kv_block(tail)))
            if tail_end == end:
                # The prompt ends on the alignment: its final state (the slot's)
                # is the tail's too.
                first_rows.append(
                    self._row(req, prompt, computed, end, starts, columns)
                )
                slot = int(req.kv.mamba_pool_idx.reshape(-1)[0])
                for group in range(bridge.KDA_PER_POOL):
                    tail_state_copies.append(
                        (
                            self.plan.state_block(slot, group),
                            self.plan.tail_state_block(tail, group),
                        )
                    )
                continue
            # Up to the tail's end, its state checkpointed into the tail's
            # slot; then the rest of the prompt.
            columns.append(((tail_end - 1) // align, tail))
            first_rows.append(
                self._row(req, prompt, computed, tail_end, starts, columns)
            )
            split_rows.append(self._row(req, prompt, tail_end, end, starts))
            split_index.append(i)
        current = {req.rid for req in reqs}
        finished, self._live = self._live - current, current
        self._prompts.drop(finished)
        if hit_copies:
            src, dst = zip(*hit_copies)
            self.kv_pool.copy_blocks(src, dst)
        tokens = self.program.step(StepInput(first_rows, finished)).tokens
        first = np.asarray(tokens[: len(reqs)].cpu(), dtype=np.int64)[:, 0]
        if split_rows:
            more = self.program.step(StepInput(split_rows)).tokens
            more = np.asarray(more[: len(split_rows)].cpu(), dtype=np.int64)[:, 0]
            for row, i in enumerate(split_index):
                first[i] = more[row]
        if tail_kv_copies:
            src, dst = zip(*tail_kv_copies)
            self.kv_pool.copy_blocks(src, dst)
        if tail_state_copies:
            src, dst = zip(*tail_state_copies)
            self.state_pool.copy_blocks(src, dst)
        for i, (computed, scheduled) in enumerate(zip(prefix_lens, extend_lens)):
            finishes = computed + scheduled >= len(reqs[i].origin_input_ids)
            if finishes != (first[i] >= 0):
                raise RuntimeError(
                    f"K3 prefill: request {reqs[i].rid} "
                    + ("finished its prompt without" if finishes else "got")
                    + " a first token"
                )
            if finishes and self.first_tokens is not None:
                self.first_tokens[reqs[i].rid] = int(first[i])
                # A request that ends at its first token never decodes: keep
                # only the latest entries (a decode step takes its own).
                while len(self.first_tokens) > FIRST_TOKENS_KEPT:
                    del self.first_tokens[next(iter(self.first_tokens))]
        return np.maximum(first, 0)

    def forget_all(self) -> None:
        if self._live:
            self.program.step(StepInput((), set(self._live)))
        self._prompts.drop(self._live)
        self._live = set()


class K3ProgramWorker(BaseSpecWorker):
    """A P/D role's worker: :class:`DecodeSteps` or :class:`PrefillSteps`
    over the program ``build_steps`` returns once the target's pools exist."""

    # Set by the plugin before the scheduler builds the worker: (target worker)
    # -> the role's steps over the unified pools.
    build_steps: Callable[[Any], DecodeSteps | PrefillSteps] | None = None

    def __init__(self, server_args, gpu_id: int, nccl_port: int, target_worker):
        super().__init__()
        self.server_args = server_args
        self._target_worker = target_worker
        self.model_runner = target_worker.model_runner
        self.speculative_num_steps = WIDTH - 1
        self.speculative_num_draft_tokens = WIDTH
        self.steps: DecodeSteps | PrefillSteps | None = None
        from sglang_tpu.envs import envs as tpu_envs
        from sglang_tpu.k3_programs.sglang_deferred import DeferredGuard

        self.guard = DeferredGuard(
            enabled=not tpu_envs.SGLANG_TPU_K3_PROGRAM_DISABLE_DEFERRED_GUARD.get()
        )

    @property
    def draft_worker(self):
        # The program drafts in its own ops.
        return None

    def alloc_memory_pool(self, **kwargs) -> None:
        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            self._target_worker.get_memory_pool()
        )
        if K3ProgramWorker.build_steps is None:
            raise RuntimeError("K3_PROGRAM: no decode program was configured")
        self.steps = K3ProgramWorker.build_steps(self._target_worker)
        self._warm_slot_bookkeeping()
        rank = getattr(self.model_runner, "tp_rank", 0)
        self.guard.log = rank == 0
        for role in (
            getattr(self.steps, "prefill", None),
            getattr(self.steps, "decode", None),
            self.steps,
        ):
            if role is not None and hasattr(role, "clock"):
                role.clock.enabled = rank == 0

    def _warm_slot_bookkeeping(self) -> None:
        """Compile sglang's slot bookkeeping on the device before serving (the
        TPU KDA backend's warmup, which K3_PROGRAM does not build): the
        request pool's mapping writes, once per admitted-request count, and
        under P/D the state payload's per-row reads (a Python int row is a
        constant of each program)."""
        pool = self.req_to_token_pool
        mappings = [
            getattr(pool, name, None)
            for name in (
                "req_index_to_mamba_index_mapping",
                "req_index_to_mamba_ping_pong_track_buffer_mapping",
            )
        ]
        mappings = [
            m
            for m in mappings
            if isinstance(m, torch.Tensor) and m.device.type != "cpu"
        ]
        if not mappings:
            return
        from sglang.srt.runtime_context import get_disagg

        with torch.no_grad():
            for mapping in mappings:
                width = mapping.shape[1:]
                for count in range(1, pool.size + 1):
                    # Request row 0 and state slot 0 are reserved: no request's.
                    zeros = torch.zeros((count, *width), dtype=torch.int64)
                    mapping[[0] * count] = zeros.to(mapping.device, mapping.dtype)
                    mapping[0:1].cpu()
            if get_disagg().disaggregation_mode != "null":
                for row in range(mappings[0].shape[0]):
                    mappings[0][row].cpu()

    def clear_cache_pool(self) -> None:
        if self.steps is not None:
            self.steps.forget_all()

    def forward_batch_generation(
        self, batch, on_publish=None, pp_proxy_tensors=None
    ) -> GenerationBatchResult:
        assert self.steps is not None
        from sglang_tpu.k3_programs.sglang_colocated import ColocatedSteps

        steps = self.steps
        if isinstance(steps, ColocatedSteps):
            steps = steps.prefill if batch.forward_mode.is_extend() else steps.decode
        prefill = isinstance(steps, PrefillSteps)
        if prefill != batch.forward_mode.is_extend():
            raise NotImplementedError(
                f"K3_PROGRAM's {'prefill' if prefill else 'decode'} role runs no "
                f"{batch.forward_mode} batches"
            )
        if batch.return_logprob or any(req.grammar is not None for req in batch.reqs):
            # The program returns tokens only, and its next inputs live on the
            # device: a grammar truncating a run on the host would leave them
            # behind (vLLM-TorchTPU hands it the bitmasks instead).
            raise NotImplementedError(
                "K3_PROGRAM serves without logprobs or structured outputs"
            )
        clock = steps.clock
        clock.start()
        # One slot per logical block of each row (a strided view of the host
        # table; a device table copies only these).
        starts = (
            bridge.page_starts(self.req_to_token_pool.req_to_token, self.steps.plan)
            .cpu()
            .numpy()
        )
        clock.mark("page_starts")
        # sglang's deferred writes since the last step run as a program of
        # their own, not inside the program's next op (sglang_deferred).
        future_map = getattr(getattr(on_publish, "func", None), "__self__", None)
        self.guard(
            (
                ("batch", batch),
                ("req_to_token_pool", self.req_to_token_pool),
                ("allocator", self.token_to_kv_pool_allocator),
                ("future_map", future_map),
            )
        )
        clock.mark("guard")
        if prefill:
            # A prefix reloaded from the host tier (sglang_hicache) lands
            # before the program reads it; sglang waits in the target
            # worker's forward, which K3_PROGRAM never runs.
            self._wait_host_loads(batch)
            state_pool = self.req_to_token_pool.mamba_pool
            # A prefix hit's cached state into the request's slot (sglang's
            # model runner does this before a forward; the program reads it).
            cow_src = getattr(batch, "mamba_cow_src_indices", None)
            if cow_src is not None and len(cow_src):
                state_pool.copy_from(cow_src, batch.mamba_cow_dst_indices)
            clock.mark("cow")
            first = steps.step(batch.reqs, batch.prefix_lens, batch.extend_lens, starts)
            clock.mark("step")
            self._track_checkpoints(batch, state_pool)
            relay = K3ProgramInput()
            if on_publish is not None:
                on_publish(batch.seq_lens)
                # Overlap scheduling relays each request's last token.
                device = future_map.device if future_map is not None else "cpu"
                relay.bonus_tokens = torch.from_numpy(first).to(device)
            clock.mark("track")
            clock.stop(sum(int(n) for n in batch.extend_lens))
            return GenerationBatchResult(
                next_token_ids=torch.from_numpy(first),
                accept_lens=torch.ones(len(batch.reqs), dtype=torch.int32),
                new_seq_lens=batch.seq_lens,
                next_draft_input=relay,
                speculative_num_draft_tokens=WIDTH,
            )
        if on_publish is not None:
            # Overlap scheduling: everything stays on the device; the
            # scheduler copies the tokens out while the next step runs.
            tokens = steps.launch(batch.reqs, starts, lagging=True)
            accept_lens = (tokens >= 0).sum(dim=1, dtype=torch.int32)
            last = tokens.gather(1, (accept_lens - 1).clamp(min=0).long()[:, None])
            new_seq_lens = batch.seq_lens.to(tokens.device) + accept_lens.to(
                batch.seq_lens.dtype
            )
            on_publish(new_seq_lens)
            relay = K3ProgramInput()
            relay.bonus_tokens = last.reshape(-1).long()
            clock.mark("result")
            clock.stop(len(batch.reqs))
            return GenerationBatchResult(
                next_token_ids=tokens.reshape(-1).long(),
                accept_lens=accept_lens,
                new_seq_lens=new_seq_lens,
                next_draft_input=relay,
                speculative_num_draft_tokens=WIDTH,
                speculative_output_stride=WIDTH,
            )
        flat, lens = steps.step(batch.reqs, starts)
        accept_lens = torch.from_numpy(lens)
        new_seq_lens = batch.seq_lens + accept_lens.to(batch.seq_lens.device)
        clock.mark("result")
        clock.stop(len(batch.reqs))
        return GenerationBatchResult(
            next_token_ids=torch.from_numpy(flat),
            accept_lens=accept_lens,
            new_seq_lens=new_seq_lens,
            next_draft_input=K3ProgramInput(),
            speculative_num_draft_tokens=WIDTH,
            speculative_output_stride=WIDTH,
        )

    def _wait_host_loads(self, batch) -> None:
        """Wait for the host-tier loads ``batch`` consumes (none: no wait)."""
        counter = getattr(self._target_worker, "hicache_layer_transfer_counter", None)
        if counter is None:
            return
        counter.set_consumer(getattr(batch, "hicache_consumer_index", -1))
        counter.wait()

    @staticmethod
    def _track_checkpoints(batch, state_pool) -> None:
        """The state after each tracked chunk into the request's track slot,
        which the radix cache keeps as the checkpoint at that length."""
        mask = getattr(batch, "mamba_track_mask", None)
        if mask is None:
            return
        rows = [i for i, tracked in enumerate(mask.tolist()) if tracked]
        if not rows:
            return
        seqlens = batch.mamba_track_seqlens_cpu
        for i in rows:
            end = batch.prefix_lens[i] + batch.extend_lens[i]
            # The cache tracks the state at the last checkpoint of the chunk:
            # with page-sized chunks, the chunk's end (+1 when it is not the
            # final state position, as sglang marks a forced track).
            if int(seqlens[i]) not in (end, end + 1):
                raise RuntimeError(
                    f"K3 prefill: request {batch.reqs[i].rid} tracks length "
                    f"{int(seqlens[i])}, not its chunk end {end}: chunks must be "
                    "one page"
                )
        working = [batch.reqs[i].kv.mamba_pool_idx.reshape(-1)[0] for i in rows]
        state_pool.copy_from(
            torch.stack(working), batch.mamba_track_indices.reshape(-1)[rows]
        )


def register() -> None:
    """Register ``K3_PROGRAM`` (idempotent)."""
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
    from sglang.srt.speculative.spec_registry import _REGISTRY

    if ALGORITHM in _REGISTRY:
        return
    from sglang_tpu.k3_programs.sglang_program import (
        build_steps,
        install_kernel_config,
    )

    K3ProgramWorker.build_steps = build_steps
    # Before any kernel module is imported (they read the config at import).
    install_kernel_config()

    @SpeculativeAlgorithm.register(
        ALGORITHM, spec_class=K3ProgramAlgo, supports_overlap=True
    )
    def _factory(server_args):
        return K3ProgramWorker
