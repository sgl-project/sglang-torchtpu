# SPDX-License-Identifier: Apache-2.0
"""The Kimi-K3 decode program as an sglang speculative worker.

The decode program (``kimi_k3.decode.KimiDecoder``) runs a whole step in its
ops: DSpark drafts, the target's verify, acceptance and sampling, keeping each
request's next input on the device. sglang's speculative workers have the
same shape from the scheduler's side -- a decode batch in, each request's
accepted tokens out -- so the program serves as the worker of a
plugin-registered algorithm, ``K3_PROGRAM``, the way vLLM-TorchTPU's
``MegakernelModelRunner`` serves it under vLLM (``runner.py`` at ``c78529c``):

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
        """Size the scheduler's per-round KV reservation for eight tokens."""
        server_args.speculative_num_steps = WIDTH - 1
        server_args.speculative_eagle_topk = 1
        server_args.speculative_num_draft_tokens = WIDTH

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


class K3ProgramWorker(BaseSpecWorker):
    """The decode role's worker: :class:`DecodeSteps` over the program
    ``build_steps`` returns once the target's pools exist."""

    # Set by the plugin before the scheduler builds the worker: (target worker)
    # -> DecodeSteps over the unified pools.
    build_steps: Callable[[Any], DecodeSteps] | None = None

    def __init__(self, server_args, gpu_id: int, nccl_port: int, target_worker):
        super().__init__()
        self.server_args = server_args
        self._target_worker = target_worker
        self.model_runner = target_worker.model_runner
        self.speculative_num_steps = WIDTH - 1
        self.speculative_num_draft_tokens = WIDTH
        self.steps: DecodeSteps | None = None

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
        if not batch.forward_mode.is_decode():
            raise NotImplementedError(
                f"K3_PROGRAM serves the P/D decode role: no {batch.forward_mode} "
                "batches (prompts arrive through the KV transfer)"
            )
        assert self.steps is not None
        if batch.return_logprob or any(req.grammar is not None for req in batch.reqs):
            # The program returns tokens only, and its next inputs live on the
            # device: a grammar truncating a run on the host would leave them
            # behind (vLLM-TorchTPU hands it the bitmasks instead).
            raise NotImplementedError(
                "K3_PROGRAM decodes without logprobs or structured outputs"
            )
        req_to_token = self.req_to_token_pool.req_to_token
        flat, lens = self.steps.step(batch.reqs, req_to_token.cpu().numpy())
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

    @SpeculativeAlgorithm.register(ALGORITHM, spec_class=K3ProgramAlgo)
    def _factory(server_args):
        return K3ProgramWorker
