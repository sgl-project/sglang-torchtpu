"""Unit tests for DFlashWorkerV2.forward_batch_generation on TPU."""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=120, suite="stage-a-test-tpu-unit")

import importlib.util
import unittest
from types import SimpleNamespace
from typing import Optional
from unittest.mock import MagicMock

import torch
import torch.nn as nn

import sglang.srt.speculative.dflash_worker_v2 as dflash_worker_mod
import sglang.srt.speculative.spec_utils as spec_utils
from sglang.srt.hardware_backend.tpu.compilation.bucketer import Bucketer
from sglang.srt.hardware_backend.tpu.kv_pool import TpuFusedMHATokenToKVPool
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardMode,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.srt.speculative.dflash_info import DFlashVerifyInput
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2
from sglang.srt.speculative.draft_worker_common import make_draft_input_v2
from sglang.test.test_utils import CustomTestCase


def _detect_tpu() -> bool:
    """True only when torch_tpu is installed and a TPU device is usable."""
    if importlib.util.find_spec("torch_tpu") is None:
        return False
    try:
        import torch_tpu  # noqa: F401

        torch.zeros(1, dtype=torch.int32).to("tpu")
        return True
    except (RuntimeError, ValueError, ImportError, AssertionError):
        return False


_TPU_AVAILABLE = _detect_tpu()


class _ForceTpuContext:
    """Context manager to ensure TPU code paths in DFlash worker and spec utils."""

    def __enter__(self):
        self.prev_worker = dflash_worker_mod._is_tpu
        self.prev_spec = spec_utils._is_tpu, spec_utils._use_torch_native
        dflash_worker_mod._is_tpu = True
        spec_utils._is_tpu = True
        spec_utils._use_torch_native = True
        self.par_override = get_parallel().override(
            world_size=1,
            world_rank=0,
            tp_size=1,
            tp_rank=0,
            pp_size=1,
            pp_rank=0,
            moe_ep_size=1,
            moe_ep_rank=0,
            moe_dp_size=1,
            moe_dp_rank=0,
            moe_tp_size=1,
            moe_tp_rank=0,
            attn_tp_size=1,
            attn_tp_rank=0,
            attn_cp_size=1,
            attn_cp_rank=0,
        )
        self.par_override.__enter__()
        return self

    def __exit__(self, *args):
        self.par_override.__exit__(*args)
        dflash_worker_mod._is_tpu = self.prev_worker
        spec_utils._is_tpu, spec_utils._use_torch_native = self.prev_spec


class _MockAttnBackend:
    """Mock attention backend for draft and target model runners."""

    def __init__(self):
        self.req_to_token_pool = None

    def init_forward_metadata(self, forward_batch):
        pass


class _DFlashLayerAttn(nn.Module):
    """DFlash layer attention matching gpt-oss-20b-DFlash parameters."""

    def __init__(
        self,
        num_kv_heads: int = 8,
        head_dim: int = 128,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.device = device or torch.device("tpu")
        self.attn = SimpleNamespace(k_scale=None, v_scale=None)

    def kv_proj_only(self, layer_ctx_hidden: torch.Tensor):
        num_tokens = layer_ctx_hidden.shape[0]
        k = torch.ones(
            (num_tokens, self.num_kv_heads * self.head_dim),
            dtype=torch.float32,
            device=self.device,
        )
        v = (
            torch.ones(
                (num_tokens, self.num_kv_heads * self.head_dim),
                dtype=torch.float32,
                device=self.device,
            )
            * 2.0
        )
        return k, v

    def apply_k_norm(self, k: torch.Tensor):
        return k

    def apply_k_rope(self, positions: torch.Tensor, k: torch.Tensor):
        return k


def _create_dflash_worker_for_tpu(
    block_size: int = 8,
    hidden_size: int = 2880,
    num_attention_heads: int = 64,
    num_key_value_heads: int = 8,
    head_dim: int = 128,
    vocab_size: int = 201088,
    mask_token_id: int = 200000,
    num_context_features: int = 5,
    device: Optional[torch.device] = None,
) -> DFlashWorkerV2:
    """Instantiate DFlashWorkerV2 matching gpt-oss-20b-DFlash config with real TPU KVPool."""
    dev = device or torch.device("tpu")

    worker = object.__new__(DFlashWorkerV2)
    worker.block_size = block_size
    worker.speculative_num_draft_tokens = block_size
    worker.use_compact_draft_cache = False
    worker.draft_window_size = None
    worker.page_size = 16
    worker.device = dev
    worker.gpu_id = 0
    worker.ps = SimpleNamespace(tp_rank=0, world_size=1)
    worker.nccl_port = 0
    worker._need_mamba_verify_commit = False
    worker._warned_sampling_fallback = False
    worker._logged_first_verify = True
    worker._mask_token = "<|MASK|>"
    worker._mask_token_id = mask_token_id
    worker._mask_token_id_override = mask_token_id
    worker._noise_embed_scale = 1.0

    worker._block_pos_offsets = torch.arange(block_size, dtype=torch.int64, device=dev)
    worker._draft_block_ids_buf = None
    worker._draft_block_positions_buf = None
    worker._draft_block_tokens_buf = None
    worker._draft_verify_out_cache_loc_buf = None
    worker._draft_block_end_buf = None
    worker._draft_extend_seq_lens_buf = None
    worker._draft_seq_lens_cpu_buf = None
    worker._draft_block_spec_info = DFlashVerifyInput(
        draft_token=torch.empty((0,), dtype=torch.long, device=dev),
        positions=torch.empty((0,), dtype=torch.int64, device=dev),
        draft_token_num=block_size,
        custom_mask=None,
        capture_hidden_mode=CaptureHiddenMode.NULL,
    )
    worker._draft_greedy_gathered_max_buf = None
    worker._draft_greedy_gathered_ids_buf = None
    worker._draft_greedy_gather_cap = 0
    worker._draft_greedy_local_max_buf = None
    worker._draft_greedy_local_arg_buf = None
    worker._draft_greedy_local_cap = 0
    worker._draft_greedy_best_rank_buf = None
    worker._draft_greedy_rank_index_buf = None
    worker._draft_greedy_selected_ids_buf = None
    worker._draft_greedy_index_cap = 0

    worker._use_fused_kv_materialize = False
    worker._fused_kv_helper = None
    worker._use_triton_prepare_block = False
    worker._use_triton_accept_bonus = False
    worker._use_triton_compact_rebuild = False
    worker._accept_bonus_buffer_cap = 0
    worker._accept_bonus_buffer_slot = 0
    worker._accept_len_buf = None
    worker._commit_lens_bufs = []
    worker._bonus_id_bufs = []
    worker._out_tokens_bufs = []
    worker._new_seq_lens_bufs = []
    worker._draft_sampler = None

    class _MockEmbed(nn.Module):
        def forward(self, ids):
            return torch.zeros(
                (ids.shape[0], hidden_size), dtype=torch.float32, device=dev
            )

    class _MockLMHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(
                torch.zeros((vocab_size, hidden_size), dtype=torch.bfloat16, device=dev)
            )

        def forward(self, hidden_states):
            bs = hidden_states.shape[0]
            logits = torch.zeros((bs, vocab_size), dtype=torch.float32, device=dev)
            return logits

    target_model = SimpleNamespace(
        get_input_embeddings=lambda: _MockEmbed(),
        lm_head=_MockLMHead(),
    )

    req_to_token_table = torch.full((8, 64), -1, dtype=torch.int32, device=dev)
    req_to_token_table[0, :16] = torch.arange(0, 16, dtype=torch.int32, device=dev)
    req_to_token_table[1, :16] = torch.arange(100, 116, dtype=torch.int32, device=dev)
    req_to_token_table[2, :16] = torch.arange(200, 216, dtype=torch.int32, device=dev)

    target_model_runner = SimpleNamespace(
        model=target_model,
        model_config=SimpleNamespace(
            vocab_size=vocab_size,
            model_is_mrope=False,
            is_matryoshka=False,
            hidden_size=hidden_size,
        ),
        prefill_attention_backend_str="torch_native",
        attn_backend=_MockAttnBackend(),
        req_to_token_pool=SimpleNamespace(req_to_token=req_to_token_table),
        decode_cuda_graph_runner=None,
        ngram_embedding_manager=SimpleNamespace(enabled=False),
        server_args=SimpleNamespace(enable_lora=False),
        ps=SimpleNamespace(attn_dcp_size=1, attn_dcp_rank=0),
        compile_policy=SimpleNamespace(
            b_bucketer=Bucketer(name="Bs", buckets=[1, 2, 4, 8, 16])
        ),
        device=dev,
    )

    target_worker = SimpleNamespace(
        model_runner=target_model_runner,
        device=dev,
        forward_batch_generation=MagicMock(),
    )
    worker._target_worker = target_worker
    worker.model_runner = target_model_runner

    layer0 = SimpleNamespace(
        self_attn=_DFlashLayerAttn(
            num_kv_heads=num_key_value_heads, head_dim=head_dim, device=dev
        )
    )
    draft_fc = nn.Linear(
        num_context_features * hidden_size, hidden_size, bias=False, device=dev
    )
    draft_hidden_norm = RMSNorm(hidden_size, eps=1e-5).to(dev)

    def _project_tpu(h):
        return draft_hidden_norm(draft_fc(torch.cat(h, dim=-1)))

    draft_model = SimpleNamespace(
        layers=[layer0],
        fc=draft_fc,
        hidden_norm=draft_hidden_norm,
        project_target_hidden=lambda h: draft_hidden_norm(draft_fc(h)),
        project_target_hidden_tpu=_project_tpu,
        prepare_context_hidden_for_kv=lambda layer, h: h,
    )
    worker.draft_model = draft_model

    draft_r2t_table = torch.full((8, 64), -1, dtype=torch.int32, device=dev)
    draft_attn_backend = _MockAttnBackend()
    draft_token_to_kv_pool = TpuFusedMHATokenToKVPool(
        size=512,
        page_size=16,
        dtype=torch.float32,
        head_num=num_key_value_heads,
        head_dim=head_dim,
        layer_num=1,
        device=str(dev),
        enable_memory_saver=False,
    )

    draft_model_runner = SimpleNamespace(
        model=draft_model,
        model_config=SimpleNamespace(
            vocab_size=vocab_size,
            hf_config=SimpleNamespace(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                head_dim=64,
                block_size=block_size,
                dflash_config={
                    "mask_token_id": mask_token_id,
                    "target_layer_ids": [1, 6, 11, 16, 21],
                },
                num_target_layers=24,
            ),
        ),
        attn_backend=draft_attn_backend,
        req_to_token_pool=SimpleNamespace(req_to_token=draft_r2t_table),
        token_to_kv_pool=draft_token_to_kv_pool,
        forward=MagicMock(),
        device=dev,
    )
    worker.draft_model_runner = draft_model_runner

    draft_worker = SimpleNamespace(
        model_runner=draft_model_runner,
        init_cuda_graphs=MagicMock(),
    )
    worker._draft_worker = draft_worker
    worker.draft_model.lm_head = target_model.lm_head

    return worker


def _make_schedule_batch(
    forward_mode: ForwardMode,
    seq_lens: torch.Tensor,
    req_pool_indices: torch.Tensor,
    out_cache_loc: Optional[torch.Tensor] = None,
    extend_lens: Optional[list] = None,
    prefix_lens: Optional[list] = None,
    spec_info=None,
) -> SimpleNamespace:
    """Helper to create a ScheduleBatch mock with all required attributes."""
    seq_lens_cpu = seq_lens.cpu().to(torch.int32)
    reqs = [
        SimpleNamespace(
            lora_id=None,
            rid=f"req_{i}",
            grammar=None,
            token_type_ids=None,
            dimensions=None,
            return_pooled_hidden_states=False,
            multi_item_delimiter_indices=None,
            dllm_block_offset=0,
            bootstrap_room=None,
        )
        for i in range(len(seq_lens))
    ]
    return SimpleNamespace(
        device=seq_lens.device,
        forward_mode=forward_mode,
        is_extend_in_batch=(forward_mode == ForwardMode.EXTEND),
        seq_lens=seq_lens,
        seq_lens_cpu=seq_lens_cpu,
        seq_lens_sum=int(seq_lens.sum().item()),
        req_pool_indices=req_pool_indices,
        req_pool_indices_cpu=req_pool_indices.cpu().tolist(),
        reqs=reqs,
        out_cache_loc=out_cache_loc,
        extend_lens=extend_lens,
        prefix_lens=prefix_lens,
        extend_num_tokens=sum(extend_lens) if extend_lens else None,
        extend_logprob_start_lens=None,
        extend_input_logprob_token_ids=None,
        orig_seq_lens=None,
        out_cache_loc_dsv4=None,
        mamba_track_indices=None,
        mamba_track_mask=None,
        mamba_track_seqlens=None,
        mamba_cow_src_indices=None,
        mamba_cow_dst_indices=None,
        mamba_clear_indices=None,
        encoder_lens=None,
        encoder_out_cache_loc=None,
        input_embeds=None,
        replace_embeds=None,
        replace_positions=None,
        positions=None,
        padded_spec_info=None,
        sampling_info=None,
        return_logprob=False,
        top_logprobs_nums=None,
        custom_mask=None,
        spec_info=spec_info,
        input_ids=None,
        has_grammar=False,
        can_run_dp_cuda_graph=False,
        can_run_dp_breakable_cuda_graph=False,
        global_forward_mode=forward_mode,
        is_prefill_only=False,
        spec_algorithm=None,
        tbo_split_seq_index=None,
        token_ids_logprobs=None,
        multimodal_inputs=None,
        encoder_cached=None,
        encoder_lens_cpu=None,
        global_num_tokens=None,
        global_num_tokens_for_logprob=None,
        dllm_config=None,
        model_config=SimpleNamespace(is_matryoshka=False, hidden_size=2880),
    )


@unittest.skipUnless(_TPU_AVAILABLE, "Requires TPU hardware")
class TestDFlashWorkerForwardTpu(CustomTestCase):
    """Test full forward batch lifecycle of DFlashWorkerV2 on TPU."""

    def setUp(self):
        super().setUp()
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))

    def test_forward_batch_generation_lifecycle_on_tpu(self):
        """Verify sequential execution of prefill, verify decode, chained decode, and idle."""
        device = torch.device("tpu")
        block_size = 8
        hidden_size = 2880
        num_context_features = 5
        vocab_size = 201088

        worker = _create_dflash_worker_for_tpu(
            block_size=block_size,
            hidden_size=hidden_size,
            num_context_features=num_context_features,
            vocab_size=vocab_size,
            device=device,
        )

        # -------------------------------------------------------------
        # Phase 1: Prefill / Extend forward
        # -------------------------------------------------------------
        extend_lens = [3, 4]
        prefix_lens = [0, 0]
        seq_lens_prefill = torch.tensor([3, 4], dtype=torch.int64, device=device)
        total_extend_tokens = sum(extend_lens)
        out_cache_loc = torch.tensor(
            [0, 1, 2, 100, 101, 102, 103], dtype=torch.int64, device=device
        )
        req_pool_indices = torch.tensor([0, 1], dtype=torch.int64, device=device)

        prefill_batch = _make_schedule_batch(
            forward_mode=ForwardMode.EXTEND,
            seq_lens=seq_lens_prefill,
            req_pool_indices=req_pool_indices,
            out_cache_loc=out_cache_loc,
            extend_lens=extend_lens,
            prefix_lens=prefix_lens,
        )

        prefill_target_hidden = torch.ones(
            (total_extend_tokens, hidden_size * num_context_features),
            dtype=torch.float32,
            device=device,
        )
        bonus_tokens_step1 = torch.tensor([105, 109], dtype=torch.int64, device=device)
        prefill_target_res = GenerationBatchResult(
            logits_output=LogitsProcessorOutput(
                next_token_logits=None,
                hidden_states=prefill_target_hidden,
            ),
            next_token_ids=bonus_tokens_step1,
            accept_lens=None,
            can_run_cuda_graph=False,
            next_draft_input=None,
            speculative_num_draft_tokens=block_size,
            new_seq_lens=seq_lens_prefill,
        )
        worker._target_worker.forward_batch_generation.return_value = prefill_target_res

        with _ForceTpuContext():
            prefill_res = worker.forward_batch_generation(prefill_batch)

        self.assertTrue(torch.equal(prefill_res.new_seq_lens, seq_lens_prefill))
        self.assertIsNone(prefill_res.logits_output.hidden_states)
        self.assertIsInstance(prefill_res.next_draft_input, DFlashDraftInputV2)
        self.assertTrue(
            torch.equal(prefill_res.next_draft_input.bonus_tokens, bonus_tokens_step1)
        )

        # Check fused KV buffer writes for prefill
        fused_kv = worker.draft_model_runner.token_to_kv_pool.get_fused_kv_buffer(
            0
        ).flatten(0, 1)
        self.assertTrue((fused_kv[out_cache_loc] != 0).any())

        # -------------------------------------------------------------
        # Phase 2: First Decode / Verify step
        # -------------------------------------------------------------
        bs = 2
        seq_lens_t = torch.tensor([3, 4], dtype=torch.int64, device=device)
        decode_batch_1 = _make_schedule_batch(
            forward_mode=ForwardMode.DECODE,
            seq_lens=seq_lens_t,
            req_pool_indices=req_pool_indices,
            spec_info=prefill_res.next_draft_input,
        )

        draft_hidden_step1 = torch.zeros(
            (bs * block_size, hidden_size), dtype=torch.float32, device=device
        )
        draft_tokens_step1 = torch.zeros(
            (bs, block_size - 1), dtype=torch.int64, device=device
        )
        worker.draft_model_runner.forward.return_value = SimpleNamespace(
            logits_output=SimpleNamespace(
                hidden_states=draft_hidden_step1,
                next_draft_tokens=draft_tokens_step1,
            ),
            can_run_graph=False,
        )

        verify_logits_step1 = torch.zeros(
            (bs * block_size, vocab_size), dtype=torch.float32, device=device
        )
        verify_logits_step1[0, 106] = 20.0
        verify_logits_step1[1, 107] = 20.0
        verify_logits_step1[2, 115] = 20.0
        verify_logits_step1[3, 0] = 20.0

        verify_logits_step1[block_size + 0, 110] = 20.0
        verify_logits_step1[block_size + 1, 114] = 20.0
        verify_logits_step1[block_size + 2, 0] = 20.0
        verify_logits_step1[block_size + 3, 0] = 20.0

        verify_hidden_step1 = (
            torch.ones(
                (bs * block_size, hidden_size * num_context_features),
                dtype=torch.float32,
                device=device,
            )
            * 4.0
        )
        worker._target_worker.forward_batch_generation.return_value = SimpleNamespace(
            logits_output=LogitsProcessorOutput(
                next_token_logits=verify_logits_step1,
                hidden_states=verify_hidden_step1,
            ),
            can_run_cuda_graph=False,
            routed_experts_output=None,
            indexer_topk_output=None,
        )

        with _ForceTpuContext():
            decode_res_1 = worker.forward_batch_generation(decode_batch_1)

        self.assertIsInstance(decode_res_1, GenerationBatchResult)
        self.assertEqual(decode_res_1.accept_lens.shape[0], bs)
        self.assertEqual(decode_res_1.new_seq_lens.shape[0], bs)
        self.assertIsInstance(decode_res_1.next_draft_input, DFlashDraftInputV2)
        self.assertEqual(decode_res_1.next_draft_input.bonus_tokens.shape[0], bs)

        # -------------------------------------------------------------
        # Phase 3: Chained Decode / Verify step
        # -------------------------------------------------------------
        seq_lens_t_2 = decode_res_1.new_seq_lens
        decode_batch_2 = _make_schedule_batch(
            forward_mode=ForwardMode.DECODE,
            seq_lens=seq_lens_t_2,
            req_pool_indices=req_pool_indices,
            spec_info=decode_res_1.next_draft_input,
        )

        draft_hidden_step2 = torch.zeros(
            (bs * block_size, hidden_size), dtype=torch.float32, device=device
        )
        draft_tokens_step2 = torch.zeros(
            (bs, block_size - 1), dtype=torch.int64, device=device
        )
        worker.draft_model_runner.forward.return_value = SimpleNamespace(
            logits_output=SimpleNamespace(
                hidden_states=draft_hidden_step2,
                next_draft_tokens=draft_tokens_step2,
            ),
            can_run_graph=False,
        )

        verify_logits_step2 = torch.zeros(
            (bs * block_size, vocab_size), dtype=torch.float32, device=device
        )
        verify_hidden_step2 = (
            torch.ones(
                (bs * block_size, hidden_size * num_context_features),
                dtype=torch.float32,
                device=device,
            )
            * 5.0
        )
        worker._target_worker.forward_batch_generation.return_value = SimpleNamespace(
            logits_output=LogitsProcessorOutput(
                next_token_logits=verify_logits_step2,
                hidden_states=verify_hidden_step2,
            ),
            can_run_cuda_graph=False,
            routed_experts_output=None,
            indexer_topk_output=None,
        )

        with _ForceTpuContext():
            decode_res_2 = worker.forward_batch_generation(decode_batch_2)

        self.assertIsInstance(decode_res_2, GenerationBatchResult)
        self.assertEqual(decode_res_2.accept_lens.shape[0], bs)
        self.assertEqual(decode_res_2.next_token_ids.shape[0], bs * block_size)

        # -------------------------------------------------------------
        # Phase 4: Non-power-of-2 Batch Size Verify Step (bs=3)
        # -------------------------------------------------------------
        bs_odd = 3
        seq_lens_odd = torch.tensor([5, 8, 12], dtype=torch.int64, device=device)
        req_pool_indices_odd = torch.tensor([0, 1, 2], dtype=torch.int64, device=device)
        bonus_odd = torch.tensor([101, 102, 103], dtype=torch.int64, device=device)
        draft_input_odd = make_draft_input_v2(
            bonus_tokens=bonus_odd,
            new_seq_lens=seq_lens_odd,
        )
        decode_batch_odd = _make_schedule_batch(
            forward_mode=ForwardMode.DECODE,
            seq_lens=seq_lens_odd,
            req_pool_indices=req_pool_indices_odd,
            spec_info=draft_input_odd,
        )

        draft_hidden_odd = torch.zeros(
            (bs_odd * block_size, hidden_size), dtype=torch.float32, device=device
        )
        draft_tokens_odd = torch.zeros(
            (bs_odd, block_size - 1), dtype=torch.int64, device=device
        )
        worker.draft_model_runner.forward.return_value = SimpleNamespace(
            logits_output=SimpleNamespace(
                hidden_states=draft_hidden_odd,
                next_draft_tokens=draft_tokens_odd,
            ),
            can_run_graph=False,
        )

        verify_logits_odd = torch.zeros(
            (bs_odd * block_size, vocab_size), dtype=torch.float32, device=device
        )
        verify_hidden_odd = torch.ones(
            (bs_odd * block_size, hidden_size * num_context_features),
            dtype=torch.float32,
            device=device,
        )
        worker._target_worker.forward_batch_generation.return_value = SimpleNamespace(
            logits_output=LogitsProcessorOutput(
                next_token_logits=verify_logits_odd,
                hidden_states=verify_hidden_odd,
            ),
            can_run_cuda_graph=False,
            routed_experts_output=None,
            indexer_topk_output=None,
        )

        with _ForceTpuContext():
            decode_res_odd = worker.forward_batch_generation(decode_batch_odd)

        self.assertEqual(decode_res_odd.accept_lens.shape[0], bs_odd)
        self.assertEqual(decode_res_odd.new_seq_lens.shape[0], bs_odd)
        self.assertEqual(decode_res_odd.next_token_ids.shape[0], bs_odd * block_size)
        self.assertEqual(decode_res_odd.next_draft_input.bonus_tokens.shape[0], bs_odd)

        # -------------------------------------------------------------
        # Phase 5: Idle step
        # -------------------------------------------------------------
        idle_batch = _make_schedule_batch(
            forward_mode=ForwardMode.IDLE,
            seq_lens=torch.empty((0,), dtype=torch.int64, device=device),
            req_pool_indices=torch.empty((0,), dtype=torch.int64, device=device),
        )

        with _ForceTpuContext():
            idle_res = worker.forward_batch_generation(idle_batch)

        self.assertEqual(idle_res.next_token_ids.numel(), 0)
        self.assertEqual(idle_res.accept_lens.numel(), 0)
        self.assertFalse(idle_res.can_run_cuda_graph)
        self.assertIsInstance(idle_res.next_draft_input, DFlashDraftInputV2)
        self.assertEqual(idle_res.next_draft_input.bonus_tokens.numel(), 0)
        self.assertEqual(idle_res.next_draft_input.new_seq_lens.numel(), 0)


if __name__ == "__main__":
    unittest.main()
