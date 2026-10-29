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
        for eight tokens, KV pages of one logical block, the scheduler
        synchronous (the worker runs each step to its tokens) and no radix
        cache (a request's KDA state lives in the program's slots)."""
        from sglang_tpu.k3_programs.kimi_k3.layout import cache_layout
        from sglang_tpu.k3_programs.sglang_config import program_kv_cache_dtype

        server_args.speculative_num_steps = WIDTH - 1
        server_args.speculative_eagle_topk = 1
        server_args.speculative_num_draft_tokens = WIDTH
        layout = cache_layout(program_kv_cache_dtype(server_args.kv_cache_dtype))
        server_args.page_size = bridge.TP_SIZE * layout.tokens_per_block
        server_args.disable_overlap_schedule = True
        server_args.disable_radix_cache = True

    def build_disagg_draft_input(self, batch, last_tokens_tensor, future_map):
        # A request the P/D transfer delivered: the program takes its first
        # token from the host on its first step.
        return K3ProgramInput()


class DecodeSteps:
    """A decode program's steps from sglang requests (no sglang objects
    beyond the requests' fields, so it runs on CPU against a stub program)."""

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

    def step(self, reqs: Sequence[Any], req_to_token: np.ndarray) -> tuple:
        """One step for ``reqs`` (sglang ``Req``s, batch order) with host
        ``req_to_token`` rows: the flat ``[len(reqs) * WIDTH]`` int64 tokens
        and the ``[len(reqs)]`` int32 accept lengths."""
        scheduled = 1 + drafts_for(self.schedule, len(reqs))
        requests = []
        for req in reqs:
            computed = int(req.kv.kv_committed_len)
            tables = bridge.block_tables(
                req_to_token[req.kv.req_pool_idx],
                computed + scheduled,
                int(req.kv.mamba_pool_idx.reshape(-1)[0]),
                self.plan,
                self.table_width,
            )
            requests.append(
                sglang_step.request_input(
                    rid=req.rid,
                    origin_input_ids=req.origin_input_ids,
                    output_ids=req.output_ids,
                    computed=computed,
                    scheduled=scheduled,
                    block_tables=tables,
                    sampling_params=req.sampling_params,
                )
            )
        current = {req.rid for req in reqs}
        finished, self._live = self._live - current, current
        tokens = self.program.step(StepInput(requests, finished)).tokens
        rows = np.asarray(tokens[: len(reqs)].cpu(), dtype=np.int64)
        _, lens = sglang_step.accepted(rows, len(reqs))
        if not all(lens):
            idle = [reqs[i].rid for i, n in enumerate(lens) if not n]
            raise RuntimeError(f"K3 decode: requests {idle} took no token this step")
        return rows.reshape(-1), np.asarray(lens, np.int32)

    def forget_all(self) -> None:
        """Drop every request's state (a cache flush)."""
        if self._live:
            self.program.step(StepInput((), set(self._live)))
        self._live = set()


class PrefillSteps:
    """A prefill program's steps from sglang's chunked prefill: each extend
    batch's chunks in, the first token of every prompt that finishes out. The
    program keeps a request's KDA state between its chunks and writes the
    state after each chunk into the request's slot blocks, its KV and the
    draft's into its pages, for the P/D transfer."""

    def __init__(
        self,
        program: MegakernelProgram,
        plan: bridge.UnifiedPoolPlan,
        *,
        table_width: int,
    ) -> None:
        self.program = program
        self.plan = plan
        self.table_width = table_width
        self._live: set[str] = set()

    def step(
        self,
        reqs: Sequence[Any],
        prefix_lens: Sequence[int],
        extend_lens: Sequence[int],
        req_to_token: np.ndarray,
    ) -> np.ndarray:
        """One extend batch: the ``[len(reqs)]`` int64 first tokens (0 for a
        request whose prompt continues in a later chunk)."""
        requests = []
        for req, computed, scheduled in zip(reqs, prefix_lens, extend_lens):
            tables = bridge.block_tables(
                req_to_token[req.kv.req_pool_idx],
                int(computed) + int(scheduled),
                int(req.kv.mamba_pool_idx.reshape(-1)[0]),
                self.plan,
                self.table_width,
            )
            requests.append(
                sglang_step.request_input(
                    rid=req.rid,
                    origin_input_ids=req.origin_input_ids,
                    output_ids=(),
                    computed=int(computed),
                    scheduled=int(scheduled),
                    block_tables=tables,
                    sampling_params=req.sampling_params,
                )
            )
        current = {req.rid for req in reqs}
        finished, self._live = self._live - current, current
        tokens = self.program.step(StepInput(requests, finished)).tokens
        first = np.asarray(tokens[: len(reqs)].cpu(), dtype=np.int64)[:, 0]
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
        req_to_token = self.req_to_token_pool.req_to_token.cpu().numpy()
        if prefill:
            first = self.steps.step(
                batch.reqs, batch.prefix_lens, batch.extend_lens, req_to_token
            )
            if on_publish is not None:
                on_publish(batch.seq_lens)
            return GenerationBatchResult(
                next_token_ids=torch.from_numpy(first),
                accept_lens=torch.ones(len(batch.reqs), dtype=torch.int32),
                new_seq_lens=batch.seq_lens,
                next_draft_input=K3ProgramInput(),
                speculative_num_draft_tokens=WIDTH,
            )
        flat, lens = self.steps.step(batch.reqs, req_to_token)
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


def register() -> None:
    """Register ``K3_PROGRAM`` (idempotent)."""
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
    from sglang.srt.speculative.spec_registry import _REGISTRY

    if ALGORITHM in _REGISTRY:
        return
    from sglang_tpu.k3_programs.sglang_program import build_steps

    K3ProgramWorker.build_steps = build_steps

    @SpeculativeAlgorithm.register(ALGORITHM, spec_class=K3ProgramAlgo)
    def _factory(server_args):
        return K3ProgramWorker
