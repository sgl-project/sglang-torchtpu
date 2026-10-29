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
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch

from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker
from sglang.srt.speculative.spec_info import SpecInput, SpecInputType
from sglang.srt.speculative.spec_registry import CustomSpecAlgo
from sglang_tpu.k3_programs import sglang_bridge as bridge
from sglang_tpu.k3_programs import sglang_step
from sglang_tpu.k3_programs.program import MegakernelProgram, StepInput

logger = logging.getLogger(__name__)

ALGORITHM = "K3_PROGRAM"
# Tokens a step returns per request: up to seven accepted drafts and the bonus.
WIDTH = 8


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

    def __init__(self) -> None:
        super().__init__(SpecInputType.NGRAM_VERIFY)

    def filter_batch(self, new_indices, new_indices_cpu=None) -> None:
        pass

    def merge_batch(self, spec_info: K3ProgramInput) -> None:
        pass


class K3ProgramAlgo(CustomSpecAlgo):
    """``K3_PROGRAM``: the members a registered algorithm needs beyond
    ``CustomSpecAlgo``'s defaults."""

    def need_topk(self) -> bool:
        return False

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
        server_args.disable_overlap_schedule = True
        role = getattr(server_args, "disaggregation_mode", None)
        if role != "prefill" or getattr(server_args, "disable_radix_cache", False):
            server_args.disable_radix_cache = True
            return
        server_args.mamba_radix_cache_strategy = "extra_buffer"
        server_args.mamba_track_interval = server_args.page_size
        server_args.chunked_prefill_size = server_args.page_size

    def build_disagg_draft_input(self, batch, last_tokens_tensor, future_map):
        # A request the P/D transfer delivered: the program takes its first
        # token from the host on its first step.
        return K3ProgramInput()


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
        self._live: set[str] = set()
        self._departed: dict[str, None] = {}
        self._prompts = _Prompts()

    def step(self, reqs: Sequence[Any], starts: np.ndarray) -> tuple:
        """One step for ``reqs`` (sglang ``Req``s, batch order) with their
        ``req_to_token`` rows' :func:`sglang_bridge.page_starts`: the flat
        ``[len(reqs) * WIDTH]`` int64 tokens and the ``[len(reqs)]`` int32
        accept lengths."""
        scheduled = 1 + drafts_for(self.schedule, len(reqs))
        requests = []
        for req in reqs:
            if req.rid in self._departed:
                # The program dropped its KDA state when it left the batch; the
                # slot block holds the prefill's, not the state after its outputs.
                raise RuntimeError(
                    f"K3 decode: request {req.rid} came back after leaving the "
                    "batch (a retraction); the programs do not resume requests"
                )
            computed = int(req.kv.kv_committed_len)
            prompt = self._prompts.get(req)
            num_tokens = len(prompt) + len(req.output_ids)
            if req.rid in self._live:
                token_ids = prompt  # the program holds its next input
            else:
                outputs = np.asarray(req.output_ids, dtype=np.int32)
                token_ids = np.concatenate((prompt, outputs))
            requests.append(
                sglang_step.request_input(
                    rid=req.rid,
                    token_ids=token_ids,
                    num_tokens=num_tokens,
                    num_prompt_tokens=len(prompt),
                    computed=computed,
                    scheduled=scheduled,
                    block_tables=bridge.block_tables(
                        starts[req.kv.req_pool_idx],
                        computed + scheduled,
                        int(req.kv.mamba_pool_idx.reshape(-1)[0]),
                        self.plan,
                        self.table_width,
                    ),
                    sampling_params=req.sampling_params,
                )
            )
        current = {req.rid for req in reqs}
        finished, self._live = self._live - current, current
        self._forget(finished)
        tokens = self.program.step(StepInput(requests, finished)).tokens
        rows = np.asarray(tokens[: len(reqs)].cpu(), dtype=np.int64)
        _, lens = sglang_step.accepted(rows, len(reqs))
        if not all(lens):
            idle = [reqs[i].rid for i, n in enumerate(lens) if not n]
            raise RuntimeError(f"K3 decode: requests {idle} took no token this step")
        return rows.reshape(-1), np.asarray(lens, np.int32)

    def _forget(self, finished: set[str]) -> None:
        self._prompts.drop(finished)
        for rid in finished:
            self._departed[rid] = None
        while len(self._departed) > self._DEPARTED_MEMORY:
            del self._departed[next(iter(self._departed))]

    def forget_all(self) -> None:
        """Drop every request's state (a cache flush)."""
        if self._live:
            self.program.step(StepInput((), set(self._live)))
        self._prompts.drop(self._live)
        self._live = set()
        self._departed.clear()


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
            if tails is not None and req.rid not in self._live:
                limit = min(end - 1, len(prompt) - 1) // align * align
                hit = tails.lookup(prompt, computed, limit)
                if hit is not None:
                    tail, tail_end = hit
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
                    tail = tails.record(prompt, tail_end)
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

    def clear_cache_pool(self) -> None:
        if self.steps is not None:
            self.steps.forget_all()

    def forward_batch_generation(
        self, batch, on_publish=None, pp_proxy_tensors=None
    ) -> GenerationBatchResult:
        assert self.steps is not None
        prefill = isinstance(self.steps, PrefillSteps)
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
        # One slot per logical block of each row (a strided view of the host
        # table; a device table copies only these).
        starts = (
            bridge.page_starts(self.req_to_token_pool.req_to_token, self.steps.plan)
            .cpu()
            .numpy()
        )
        if prefill:
            state_pool = self.req_to_token_pool.mamba_pool
            # A prefix hit's cached state into the request's slot (sglang's
            # model runner does this before a forward; the program reads it).
            cow_src = getattr(batch, "mamba_cow_src_indices", None)
            if cow_src is not None and len(cow_src):
                state_pool.copy_from(cow_src, batch.mamba_cow_dst_indices)
            first = self.steps.step(
                batch.reqs, batch.prefix_lens, batch.extend_lens, starts
            )
            self._track_checkpoints(batch, state_pool)
            if on_publish is not None:
                on_publish(batch.seq_lens)
            return GenerationBatchResult(
                next_token_ids=torch.from_numpy(first),
                accept_lens=torch.ones(len(batch.reqs), dtype=torch.int32),
                new_seq_lens=batch.seq_lens,
                next_draft_input=K3ProgramInput(),
                speculative_num_draft_tokens=WIDTH,
            )
        flat, lens = self.steps.step(batch.reqs, starts)
        accept_lens = torch.from_numpy(lens)
        new_seq_lens = batch.seq_lens + accept_lens.to(batch.seq_lens.device)
        if on_publish is not None:
            on_publish(new_seq_lens)
        return GenerationBatchResult(
            next_token_ids=torch.from_numpy(flat),
            accept_lens=accept_lens,
            new_seq_lens=new_seq_lens,
            next_draft_input=K3ProgramInput(),
            speculative_num_draft_tokens=WIDTH,
            speculative_output_stride=WIDTH,
        )

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

    @SpeculativeAlgorithm.register(ALGORITHM, spec_class=K3ProgramAlgo)
    def _factory(server_args):
        return K3ProgramWorker
