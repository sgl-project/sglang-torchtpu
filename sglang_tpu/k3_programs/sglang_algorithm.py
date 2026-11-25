# SPDX-License-Identifier: Apache-2.0
"""``K3_PROGRAM``, the Kimi-K3 programs' speculative algorithm, as sglang
registers it: the settings it serves under and the batch input it carries.

The plugin registers the algorithm in every process it loads in, so this
module imports no program module (jax, torch_tpu): the worker
(:mod:`sglang_worker`) loads when the scheduler builds the algorithm's worker,
and the programs when a server that selects the algorithm loads its model
(``hooks/k3_program.py``).
"""

from __future__ import annotations

from typing import Any

import torch

from sglang.srt.speculative.spec_info import SpecInput, SpecInputType
from sglang.srt.speculative.spec_registry import CustomSpecAlgo

ALGORITHM = "K3_PROGRAM"
# Tokens a step returns per request: up to seven accepted drafts and the bonus.
WIDTH = 8


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
        # The tokens (and accept lengths) the result copies are, or derive
        # from, the step's emitted tokens, the output of its last op (which
        # commits the target's and the draft's KV); the device runs ops in
        # launch order.
        return True

    def create_future_map(
        self,
        device,
        req_to_token_pool,
        needs_cpu_seq_lens: bool = True,
        needs_confidence_relay: bool = False,
    ):
        """sglang's future map, or under host bookkeeping (no host length
        mirror or confidence relay asked for, as the programs' worker asks
        none) one that relays nothing (:mod:`sglang_host_relay`)."""
        from sglang.srt.managers.overlap_utils import FutureMap
        from sglang_tpu.envs import envs as tpu_envs

        cls = FutureMap
        if (
            tpu_envs.SGLANG_TPU_K3_PROGRAM_OPT_HOST_BOOKKEEPING.get()
            and not needs_cpu_seq_lens
            and not needs_confidence_relay
        ):
            from sglang_tpu.k3_programs.sglang_host_relay import HostRelayFutureMap

            cls = HostRelayFutureMap
        return cls(
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
        from sglang_tpu.k3_programs import sglang_bridge as bridge
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


def register() -> None:
    """Register ``K3_PROGRAM`` (idempotent). The worker class loads when the
    scheduler builds the algorithm's worker, after the model load built the
    programs (``sglang_program.build_from_server`` installs the kernel config
    before the programs import a kernel module)."""
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
    from sglang.srt.speculative.spec_registry import _REGISTRY

    if ALGORITHM in _REGISTRY:
        return

    @SpeculativeAlgorithm.register(
        ALGORITHM, spec_class=K3ProgramAlgo, supports_overlap=True
    )
    def _factory(server_args):
        from sglang_tpu.k3_programs.sglang_program import build_steps
        from sglang_tpu.k3_programs.sglang_worker import K3ProgramWorker

        if K3ProgramWorker.build_steps is None:
            K3ProgramWorker.build_steps = build_steps
        return K3ProgramWorker
