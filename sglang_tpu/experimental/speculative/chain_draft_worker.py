"""Autoregressive chain drafter holder for TpuChainSpecWorker."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable

import torch

from sglang.srt.layers.dp_attention import set_is_extend_in_batch
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
    enable_num_token_non_padded,
)
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.model_executor.pool_configurator import MemoryPoolConfig
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang_tpu.experimental.compilation.chain_compile_policy import (
    TpuChainSpecDecodeCompilePolicyMixin,
)
from sglang_tpu.experimental.speculative.chain_draft_input import (
    ChainDraftForwardInput,
    chain_draft_forward_input,
)
from sglang_tpu.experimental.speculative.draft_worker import TpuDraftWorker

# (forward_batch, carry, req_pool_indices, positions, cache_locs, seq_lens,
#  temperatures, top_ks, uniforms) -> (drafts [pbs, N], dense logits [pbs, N, V] | None).
FusedDraft = Callable[..., tuple[torch.Tensor, torch.Tensor | None]]


class TpuChainDraftWorker(TpuDraftWorker):
    """A draft that proposes one token per forward from a carried hidden (MTP, EAGLE3).

    The subclass validates its draft model, attaches the target modules and maps
    draft logits to target tokens. This class owns the carried hidden and runs
    the whole draft chain as one compiled program.

    Attributes:
      carry_size: Width of the draft hidden carried between forwards.
      target_feature_size: Width of the target feature the draft reads on
        committed rows (MTP: the target final hidden; EAGLE3: its aux layers).
      fused_draft_tensor_attrs: Non-parameter draft-model tensors the fused draft
        reads (direct_compile model_extra_tensor_attrs).
    """

    carry_size: int
    target_feature_size: int
    fused_draft_tensor_attrs: tuple[str, ...] = ()

    def __init__(self, target_runner: ModelRunner) -> None:
        super().__init__(target_runner)
        self._carry_table: torch.Tensor | None = None
        self._fused_draft: FusedDraft | None = None
        # Per batch bucket; the fused program overwrites it before the first draft forward.
        self._hidden_placeholders: dict[int, torch.Tensor] = {}

    @abstractmethod
    def first_draft_logits(self, carry: torch.Tensor) -> torch.Tensor:
        """Draft-vocab logits [pbs, V_d] of d_1 from the carried [pbs, carry_size] hidden."""

    @abstractmethod
    def target_tokens(self, draft_ids: torch.Tensor) -> torch.Tensor:
        """Map [pbs] draft-vocab ids to target-vocab ids."""

    @abstractmethod
    def target_logits(self, draft_logits: torch.Tensor) -> torch.Tensor:
        """Dense target-vocab [pbs, V] logits; -inf where the draft vocab has no token."""

    def alloc_memory_pool(
        self,
        memory_pool_config: MemoryPoolConfig | None = None,
        req_to_token_pool: ReqToTokenPool | None = None,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator | None = None,
    ) -> None:
        super().alloc_memory_pool(
            memory_pool_config=memory_pool_config,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        )
        assert req_to_token_pool is not None
        # Indexed by req pool row; padded rows read and write the reserved row 0.
        self._carry_table = torch.zeros(
            (req_to_token_pool.req_to_token.shape[0], self.carry_size),
            dtype=self.model_runner.dtype,
            device=self.model_runner.device,
        )

    @property
    def carry(self) -> torch.Tensor:
        """[req pool rows, carry_size] draft hidden of each request's last committed row."""
        assert self._carry_table is not None, "alloc_memory_pool has not run"
        return self._carry_table

    def precompile_draft(self) -> None:
        compile_policy = self._compile_policy()
        self._fused_draft = compile_policy.compile_fused_draft(
            self._fused_draft_steps,
            model_extra_tensor_attrs=self.fused_draft_tensor_attrs,
        )
        compile_policy.precompile_draft(
            self.model_runner.attn_backend,
            target_feature_size=self.target_feature_size,
            fused_draft_warmup=self._warmup_fused_draft,
        )

    def fused_draft_batch(
        self,
        *,
        spec_algorithm: SpeculativeAlgorithm,
        input_ids: torch.Tensor,
        req_pool_indices: torch.Tensor,
        req_pool_indices_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        out_cache_loc: torch.Tensor,
    ) -> ForwardBatch:
        """The DECODE batch the fused draft plans attention from; warmup builds it the same way."""
        bs = seq_lens.shape[0]
        hidden = self._hidden_placeholders.get(bs)
        if hidden is None:
            hidden = torch.zeros(
                (bs, self.carry_size),
                dtype=self.model_runner.dtype,
                device=self.model_runner.device,
            )
            self._hidden_placeholders[bs] = hidden
        return ForwardBatch(
            forward_mode=ForwardMode.DECODE,
            out_cache_loc_is_physical=True,
            batch_size=bs,
            input_ids=input_ids,
            req_pool_indices=req_pool_indices,
            req_pool_indices_cpu=req_pool_indices_cpu,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            seq_lens_sum=int(seq_lens_cpu.sum()),
            out_cache_loc=out_cache_loc,
            # Step 0 sits at seq_lens; the fused loop moves later steps along the verify rows.
            positions=seq_lens,
            input_embeds=None,
            spec_algorithm=spec_algorithm,
            spec_info=chain_draft_forward_input(hidden_states=hidden, width=1),
            capture_hidden_mode=CaptureHiddenMode.FULL,
            num_token_non_padded=(
                torch.tensor(bs, dtype=torch.int32, device=seq_lens.device)
                if enable_num_token_non_padded()
                else None
            ),
            global_num_token_non_padded_cpu=bs,
        )

    def run_fused_draft(
        self,
        forward_batch: ForwardBatch,
        *,
        req_pool_indices: torch.Tensor,
        positions: torch.Tensor,
        cache_locs: torch.Tensor,
        seq_lens: torch.Tensor,
        temperatures: torch.Tensor | None,
        top_ks: torch.Tensor | None,
        uniforms: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Drafts [pbs, N] (and dense q logits when sampling) from one compiled program."""
        assert self._fused_draft is not None, "precompile_draft has not run"
        shim = self._compile_policy().compiled_forward_shim
        assert shim is not None
        # The fused program inlines the forward shim, whose HiCache reload gate runs
        # only eagerly; a PD decode drafts before any target forward waits on it.
        shim.wait_for_hicache_reload()
        # The fused program guards on this flag; it is a decode program, but the last
        # extend (or an extend warmup) under DP attention may have left it True.
        set_is_extend_in_batch(False)
        backend = self.model_runner.attn_backend
        backend.init_forward_metadata(forward_batch)
        forward_batch.mark_forward_metadata_ready()
        with forward_context(ForwardContext(attn_backend=backend)):
            return self._fused_draft(
                forward_batch,
                self.carry,
                req_pool_indices,
                positions,
                cache_locs,
                seq_lens,
                temperatures,
                top_ks,
                uniforms,
            )

    def _fused_draft_steps(
        self,
        forward_batch: ForwardBatch,
        carry: torch.Tensor,
        req_pool_indices: torch.Tensor,
        positions: torch.Tensor,
        cache_locs: torch.Tensor,
        seq_lens: torch.Tensor,
        temperatures: torch.Tensor | None,
        top_ks: torch.Tensor | None,
        uniforms: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Traced once per (batch bucket, sampled): d_1 from the carry, then N - 1 draft forwards."""
        spec_info = forward_batch.spec_info
        assert isinstance(spec_info, ChainDraftForwardInput)
        model = self.model_runner.model
        hidden = carry.index_select(0, req_pool_indices)
        logits = self.first_draft_logits(hidden)
        tokens: list[torch.Tensor] = []
        dense_logits: list[torch.Tensor] = []
        for step in range(self.num_drafts):
            if step > 0:
                # Step s reads d_s at verify row s - 1 and writes its draft KV there;
                # RPA writes at kv_len - 1, so kv_len is the absolute seq_lens + s.
                spec_info.hidden_states = hidden
                spec_info.rpa_draft_kv_lens = (seq_lens + step).to(torch.int32)
                forward_batch.input_ids = tokens[-1]
                forward_batch.positions = positions[:, step - 1]
                forward_batch.out_cache_loc = cache_locs[:, step - 1]
                logits_output = model.forward(
                    tokens[-1], positions[:, step - 1], forward_batch
                )
                logits, hidden = (
                    logits_output.next_token_logits,
                    logits_output.hidden_states,
                )
            token, dense = self._choose_token(
                logits,
                temperatures=temperatures,
                top_ks=top_ks,
                uniforms=None if uniforms is None else uniforms[:, step],
            )
            tokens.append(token)
            if dense is not None:
                dense_logits.append(dense)
        spec_info.rpa_draft_kv_lens = None
        return (
            torch.stack(tokens, dim=1),
            torch.stack(dense_logits, dim=1) if dense_logits else None,
        )

    def _choose_token(
        self,
        logits: torch.Tensor,
        *,
        temperatures: torch.Tensor | None,
        top_ks: torch.Tensor | None,
        uniforms: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Next target token [pbs], plus the dense logits q was drawn from when sampling."""
        if temperatures is None or top_ks is None or uniforms is None:
            return self.target_tokens(torch.argmax(logits, dim=-1)), None
        dense = self.target_logits(logits)
        probs = torch.softmax(
            dense.float() / temperatures.clamp_min(1e-5)[:, None], dim=-1
        )
        # Inverse CDF with one uniform per row; ranks share the uniforms, so no sync.
        drawn = (probs.cumsum(dim=-1) < uniforms[:, None]).sum(dim=-1)
        drawn = drawn.clamp_max(dense.shape[-1] - 1)
        token = torch.where(top_ks <= 1, torch.argmax(dense, dim=-1), drawn)
        return token, dense

    def _warmup_fused_draft(self, bs: int, sampled: bool) -> None:
        """One fused draft call per (batch bucket, sampled) on the dummy KV slot."""
        device = self.model_runner.device
        width = self.num_drafts + 1
        # Pool row 1 and slot 1 are the policies' dummy request and KV slot.
        self.model_runner.req_to_token_pool.req_to_token[1, :width] = 1
        ones = torch.ones((bs,), dtype=torch.int64, device=device)
        forward_batch = self.fused_draft_batch(
            spec_algorithm=self.model_runner.spec_algorithm,
            input_ids=torch.zeros((bs,), dtype=torch.int64, device=device),
            req_pool_indices=ones,
            req_pool_indices_cpu=torch.ones((bs,), dtype=torch.int64),
            seq_lens=torch.zeros((bs,), dtype=torch.int64, device=device),
            seq_lens_cpu=torch.full((bs,), width, dtype=torch.int64),
            out_cache_loc=ones,
        )
        grid = torch.ones((bs, width), dtype=torch.int64, device=device)
        temperatures = top_ks = uniforms = None
        if sampled:
            temperatures = torch.ones((bs,), dtype=torch.float32, device=device)
            top_ks = torch.ones((bs,), dtype=torch.int32, device=device)
            uniforms = torch.rand((bs, self.num_drafts), device=device)
        with torch.no_grad():
            tokens, dense = self.run_fused_draft(
                forward_batch,
                req_pool_indices=ones,
                positions=grid - 1,
                cache_locs=grid,
                seq_lens=torch.zeros((bs,), dtype=torch.int64, device=device),
                temperatures=temperatures,
                top_ks=top_ks,
                uniforms=uniforms,
            )
        tokens.cpu()
        if dense is not None:
            dense.float().cpu()

    def _compile_policy(self) -> TpuChainSpecDecodeCompilePolicyMixin:
        compile_policy = self.model_runner.compile_policy
        if not isinstance(compile_policy, TpuChainSpecDecodeCompilePolicyMixin):
            raise TypeError(
                "The chain draft needs TpuChainSpecDecodeCompilePolicyMixin, got "
                f"{type(compile_policy).__name__}."
            )
        return compile_policy
