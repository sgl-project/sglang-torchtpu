"""Unit tests for one DSparkWorkerV2 decode step on TPU.

The accept tail runs through two compiled graphs on TPU (DsparkTpuAcceptor),
which the CUDA build folds into the target-verify cuda graph instead. The
kernel-level parity of those graphs is pinned in test_dspark_tpu_fused_accept;
what these cases guard is the wiring around them -- bucket staging, the padded
verify outputs being trimmed back, and the fused path agreeing with the eager
one when driven from a real worker.
"""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=120, suite="stage-a-test-tpu-unit")

import importlib.util
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch
import torch.nn as nn

from sglang.srt.distributed.device_communicators.custom_all_reduce_utils import (
    update_environment_variables,
)
from sglang.srt.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.environ import envs
from sglang.srt.hardware_backend.tpu.compilation.bucketer import Bucketer
from sglang.srt.hardware_backend.tpu.compilation.spec_decode_compile_policy_dflash import (
    DFlashSpecDecodeCompilePolicyMixin,
)
from sglang.srt.hardware_backend.tpu.dspark_utils_tpu import DsparkTpuVerifyWindow
from sglang.srt.hardware_backend.tpu.kv_pool import TpuFusedMHATokenToKVPool
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.mem_cache.kv_index_translator import KVIndexTranslator
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
from sglang.srt.models.deepseek_v4_dspark import DSparkV4MarkovHead
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.speculative.draft_worker_common import (
    make_draft_block_spec_info,
    make_draft_input_v2,
)
from sglang.srt.speculative.dspark_components.dspark_draft import DraftBlockProposer
from sglang.srt.speculative.dspark_components.dspark_kv_inject import (
    TargetHiddenKvInjector,
)
from sglang.srt.speculative.dspark_components.dspark_observability import (
    DsparkStepObservers,
)
from sglang.srt.speculative.dspark_components.dspark_planner import DSparkVerifyPlanner
from sglang.srt.speculative.dspark_components.dspark_verify import TargetVerifyExecutor
from sglang.srt.speculative.dspark_components.dspark_worker_v2 import DSparkWorkerV2
from sglang.srt.speculative.spec_tp_sync import SpecTpSync
from sglang.test.test_utils import CustomTestCase

_VOCAB = 256
_HIDDEN = 64
_MARKOV_RANK = 8
_GAMMA = 3
_WIDTH = _GAMMA + 1  # verify_num_draft_tokens
_HEAD_DIM = 128
_NUM_KV_HEADS = 2
_PAGE_SIZE = 16
_MAX_REQS = 8
_BUCKETS = [1, 2, 4, 8, 16]


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


def _ensure_single_rank_distributed() -> None:
    """Bring up a real one-rank TP group.

    DSparkV4MarkovHead embeds through VocabParallelEmbedding, which reaches
    get_tp_group() even at tp_size 1, so a get_parallel() override alone is not
    enough -- the group object has to exist.
    """
    if torch.distributed.is_initialized():
        return
    update_environment_variables({"MASTER_ADDR": "localhost", "MASTER_PORT": "12355"})
    init_distributed_environment(world_size=1, rank=0, local_rank=0, backend="gloo")
    initialize_model_parallel(tensor_model_parallel_size=1)


class _MockAttnBackend:
    def __init__(self):
        self.req_to_token_pool = None

    def init_forward_metadata(self, forward_batch):
        pass


class _MockCompilePolicy:
    """Carries the real deferred-unpadding hooks over a stub bucketer.

    Binding the production methods keeps the padding contract honest: whether a
    step runs padded is then decided by padding_state, exactly as in serving.
    """

    should_defer_verify_output_unpadding = (
        DFlashSpecDecodeCompilePolicyMixin.should_defer_verify_output_unpadding
    )
    prepare_spec_verify = DFlashSpecDecodeCompilePolicyMixin.prepare_spec_verify
    finalize_dflash_verify = DFlashSpecDecodeCompilePolicyMixin.finalize_dflash_verify

    def __init__(self, b_bucketer):
        self.b_bucketer = b_bucketer
        self.is_draft_worker = False

    def finalize_forward_batch(self, outputs, forward_batch):
        return outputs


class _DraftLayerAttn(nn.Module):
    """Just the surface dflash_write_draft_kv touches on a draft layer."""

    def __init__(self, device: torch.device):
        super().__init__()
        self.num_kv_heads = _NUM_KV_HEADS
        self.head_dim = _HEAD_DIM
        self.device = device
        self.attn = SimpleNamespace(k_scale=None, v_scale=None)
        # The fused KV write gathers cos/sin off rotary_emb rather than calling
        # apply_k_rope, and asserts a neox table over head_dim.
        self.rotary_emb = get_rope(
            _HEAD_DIM,
            rotary_dim=_HEAD_DIM,
            max_position=4096,
            base=10000,
            is_neox_style=True,
            dtype=torch.float32,
        )
        self.rotary_emb.cos_sin_cache = self.rotary_emb.cos_sin_cache.to(device)

    def kv_proj_only(self, layer_ctx_hidden: torch.Tensor):
        num_tokens = layer_ctx_hidden.shape[0]
        width = self.num_kv_heads * self.head_dim
        k = torch.ones((num_tokens, width), dtype=torch.float32, device=self.device)
        return k, k * 2.0

    def apply_k_norm(self, k: torch.Tensor):
        return k

    def apply_k_rope(self, positions: torch.Tensor, k: torch.Tensor):
        return k


class _DSparkDraftModel(nn.Module):
    """Minimal DSpark draft: a real markov head over a linear base-logit head."""

    uses_own_vocab_modules = True
    sample_from_anchor = True
    confidence_head = None

    def __init__(self, device: torch.device):
        super().__init__()
        # Build on CPU and move: parameters allocated straight onto TPU draw
        # from the device RNG, which torch.manual_seed does not pin, so two
        # workers in the same test would not share weights.
        self.markov_head = DSparkV4MarkovHead(
            vocab_size=_VOCAB, markov_rank=_MARKOV_RANK
        )
        # VocabParallelEmbedding allocates with torch.empty and expects a
        # checkpoint load, so without this the head runs on uninitialized memory.
        with torch.no_grad():
            self.markov_head.markov_w1.weight.normal_(std=0.02)
        self.lm_head = nn.Linear(_HIDDEN, _VOCAB, bias=False)
        self.embed_tokens = nn.Embedding(_VOCAB, _HIDDEN)
        self.to(device)
        self.layers = [SimpleNamespace(self_attn=_DraftLayerAttn(device))]

    def compute_base_logits(self, hidden_states: torch.Tensor):
        return self.lm_head(hidden_states).float(), None

    def project_target_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden

    def prepare_context_hidden_for_kv(self, layer, hidden: torch.Tensor):
        return hidden


def _make_model_runner(*, device, model, is_draft_worker, req_to_token, kv_pool=None):
    return SimpleNamespace(
        is_draft_worker=is_draft_worker,
        kv_index_translator=KVIndexTranslator(
            req_to_token=req_to_token,
            token_to_kv_pool_allocator=None,
            token_to_kv_pool=kv_pool,
            page_size=_PAGE_SIZE,
            device=device,
        ),
        lora_manager=None,
        ngram_embedding_manager=SimpleNamespace(enabled=False),
        prefill_attention_backend_str="torch_native",
        ps=SimpleNamespace(attn_dcp_size=1, attn_dcp_rank=0),
        model=model,
        model_config=SimpleNamespace(
            vocab_size=_VOCAB,
            model_is_mrope=False,
            is_matryoshka=False,
            hidden_size=_HIDDEN,
            hf_config=SimpleNamespace(hidden_size=_HIDDEN, vocab_size=_VOCAB),
        ),
        attn_backend=_MockAttnBackend(),
        req_to_token_pool=SimpleNamespace(req_to_token=req_to_token, size=_MAX_REQS),
        token_to_kv_pool=kv_pool,
        decode_cuda_graph_runner=None,
        server_args=SimpleNamespace(enable_lora=False),
        compile_policy=_MockCompilePolicy(Bucketer(name="Bs", buckets=_BUCKETS)),
        forward=MagicMock(),
        device=device,
    )


def _create_dspark_worker_for_tpu(device: torch.device) -> DSparkWorkerV2:
    """Assemble DSparkWorkerV2 without build_draft_tp_worker loading a checkpoint."""
    worker = object.__new__(DSparkWorkerV2)
    worker.server_args = ServerArgs(model_path="dummy")
    worker.gpu_id = 0
    worker.nccl_port = 0
    worker.ps = SimpleNamespace(tp_rank=0, tp_size=1, world_size=1, attn_tp_size=1)
    worker.device = device
    worker.page_size = _PAGE_SIZE
    worker.gamma = _GAMMA
    worker.verify_num_draft_tokens = _WIDTH
    worker.speculative_num_draft_tokens = _WIDTH
    worker.sample_from_anchor = True
    worker.query_token_num = _GAMMA
    worker._mask_token_id = 1
    worker._draft_is_moe = False
    worker._draft_dp_context_enabled = False
    worker._is_pd_prefill = False
    worker._decode_graph_allowed = False
    worker._target_is_mambaish = False
    worker._need_mamba_verify_commit = False
    worker._simulate_acc_len = 0.0
    worker._forced_budget_frac = None
    worker._draft_sampler = None
    worker._verify_epilogue = None
    worker._tp_sync = SpecTpSync(SimpleNamespace(world_size=1, rank_in_group=0))
    worker._draft_graph_group = None

    req_to_token = torch.zeros(
        (_MAX_REQS, 4 * _PAGE_SIZE), dtype=torch.int32, device=device
    )
    for req in range(_MAX_REQS):
        req_to_token[req] = torch.arange(
            req * 4 * _PAGE_SIZE,
            (req + 1) * 4 * _PAGE_SIZE,
            dtype=torch.int32,
            device=device,
        )

    draft_model = _DSparkDraftModel(device)
    worker.draft_model = draft_model
    target_model = SimpleNamespace(
        get_input_embeddings=lambda: draft_model.embed_tokens,
        lm_head=draft_model.lm_head,
    )

    target_model_runner = _make_model_runner(
        device=device,
        model=target_model,
        is_draft_worker=False,
        req_to_token=req_to_token,
    )
    draft_model_runner = _make_model_runner(
        device=device,
        model=draft_model,
        is_draft_worker=True,
        req_to_token=req_to_token,
        kv_pool=TpuFusedMHATokenToKVPool(
            size=_MAX_REQS * 4 * _PAGE_SIZE,
            page_size=_PAGE_SIZE,
            dtype=torch.float32,
            head_num=_NUM_KV_HEADS,
            head_dim=_HEAD_DIM,
            layer_num=1,
            device=str(device),
            enable_memory_saver=False,
        ),
    )
    worker.model_runner = target_model_runner
    worker.draft_model_runner = draft_model_runner
    worker._target_worker = SimpleNamespace(
        model_runner=target_model_runner,
        device=device,
        forward_batch_generation=MagicMock(),
    )
    worker._draft_worker = SimpleNamespace(model_runner=draft_model_runner)

    worker._block_pos_offsets = torch.arange(_WIDTH, dtype=torch.int64, device=device)
    worker._draft_block_spec_info = make_draft_block_spec_info(
        draft_token_num=_GAMMA, device=device
    )
    worker._tpu_verify_window = DsparkTpuVerifyWindow(
        device=device,
        verify_num_draft_tokens=_WIDTH,
        block_pos_offsets=worker._block_pos_offsets,
    )

    worker._verify_planner = DSparkVerifyPlanner(
        draft_model=draft_model,
        gamma=_GAMMA,
        model_runner=target_model_runner,
        device=device,
        tp_rank=0,
        verify_num_draft_tokens=_WIDTH,
        tp_sync=worker._tp_sync,
    )
    worker._kv_injector = TargetHiddenKvInjector(
        draft_model=draft_model,
        draft_model_runner=draft_model_runner,
        model_runner=target_model_runner,
        device=device,
        verify_num_draft_tokens=_WIDTH,
        block_pos_offsets=worker._block_pos_offsets,
    )
    worker._proposer = DraftBlockProposer(
        draft_model=draft_model,
        draft_model_runner=draft_model_runner,
        gamma=_GAMMA,
        mask_token_id=worker._mask_token_id,
        draft_block_spec_info=worker._draft_block_spec_info,
        tp_sync=worker._tp_sync,
    )
    worker._verify_executor = TargetVerifyExecutor(
        target_worker=worker._target_worker,
        gamma=_GAMMA,
        verify_num_draft_tokens=_WIDTH,
        model_runner=target_model_runner,
        kv_injector=worker._kv_injector,
        tp_sync=worker._tp_sync,
        verify_epilogue=None,
        simulate_acc_len=0.0,
    )
    worker._observers = DsparkStepObservers(
        planner=worker._verify_planner,
        gamma=_GAMMA,
        verify_num_draft_tokens=_WIDTH,
        tp_rank=0,
        device=device,
        simulate_acc_len=0.0,
    )
    return worker


def _make_schedule_batch(*, seq_lens, req_pool_indices, spec_info):
    return SimpleNamespace(
        device=seq_lens.device,
        forward_mode=ForwardMode.DECODE,
        is_extend_in_batch=False,
        seq_lens=seq_lens,
        seq_lens_cpu=seq_lens.cpu().to(torch.int64),
        seq_lens_sum=int(seq_lens.sum().item()),
        req_pool_indices=req_pool_indices,
        req_pool_indices_cpu=req_pool_indices.cpu(),
        reqs=[
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
            for i in range(int(seq_lens.numel()))
        ],
        out_cache_loc=None,
        extend_lens=None,
        prefix_lens=None,
        extend_num_tokens=None,
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
        encoder_cached=None,
        encoder_lens_cpu=None,
        tree_cache=SimpleNamespace(page_size=_PAGE_SIZE),
        input_embeds=None,
        replace_embeds=None,
        replace_positions=None,
        positions=None,
        padded_spec_info=None,
        sampling_info=None,
        return_logprob=False,
        top_logprobs_nums=None,
        token_ids_logprobs=None,
        custom_mask=None,
        spec_info=spec_info,
        input_ids=None,
        has_grammar=False,
        can_run_dp_cuda_graph=False,
        can_run_dp_breakable_cuda_graph=False,
        can_run_dp_prefill_cuda_graph=False,
        can_run_decode_cuda_graph=False,
        return_hidden_states_mode=CaptureHiddenMode.NULL,
        enable_overlap=False,
        forward_iter=0,
        global_forward_mode=ForwardMode.DECODE,
        is_prefill_only=False,
        spec_algorithm=None,
        tbo_split_seq_index=None,
        multimodal_inputs=None,
        global_num_tokens=None,
        global_num_tokens_for_logprob=None,
        spec_verify_tier_num_tokens=0,
        global_spec_verify_tier_num_tokens=None,
        dllm_config=None,
        model_config=SimpleNamespace(is_matryoshka=False, hidden_size=_HIDDEN),
    )


def _stub_draft_forward(worker, *, bs, device):
    """Hand the proposer a fixed hidden state; the markov head samples from it.

    next_draft_tokens stays None so propose() takes the compute_base_logits +
    sample_draft_block tail rather than reading an in-graph sampler buffer.
    """
    hidden = torch.linspace(-1.0, 1.0, bs * _GAMMA * _HIDDEN, dtype=torch.float32)
    worker.draft_model_runner.forward.return_value = SimpleNamespace(
        logits_output=SimpleNamespace(
            hidden_states=hidden.view(bs * _GAMMA, _HIDDEN).to(device),
            next_draft_tokens=None,
        ),
        can_run_graph=False,
    )


def _stub_accepting_target(worker, *, bs, device, accept_lens, pad_bs=None):
    """Drive the target so row r accepts exactly accept_lens[r] draft tokens.

    Random logits make every row mismatch at step 0, which leaves correct_len
    at zero and the scatter index pinned to column 0 -- the accept tail would
    then compare equal no matter how it is wired. Deriving the target argmax
    from the candidates the draft actually proposed is what puts a spread of
    accept lengths (and a live cap) through the step.

    pad_bs also stamps the bucket padding the real compile policy records, so
    the accept runs wide and the caller trims back.
    """
    physical_bs = bs if pad_bs is None else pad_bs

    def _forward(*, batch, forward_batch, **kwargs):
        if pad_bs is not None:
            forward_batch.spec_info.record_padding(
                logical_batch_size=bs,
                physical_batch_size=pad_bs,
                logical_num_tokens=bs * _WIDTH,
                physical_num_tokens=pad_bs * _WIDTH,
            )
        candidates = forward_batch.spec_info.draft_token.view(bs, _WIDTH).cpu()
        # Greedy accept matches candidates[:, 1:] against target_predict[:, :-1].
        target_predict = torch.zeros((bs, _WIDTH), dtype=torch.int64)
        for row, keep in enumerate(accept_lens):
            target_predict[row, :keep] = candidates[row, 1 : 1 + keep]
            if keep < _WIDTH - 1:
                target_predict[row, keep] = (candidates[row, keep + 1] + 1) % _VOCAB
        logits = torch.zeros((physical_bs * _WIDTH, _VOCAB), dtype=torch.float32)
        rows = torch.arange(bs * _WIDTH, dtype=torch.int64)
        logits[rows, target_predict.reshape(-1)] = 1.0
        return SimpleNamespace(
            logits_output=LogitsProcessorOutput(
                next_token_logits=logits.to(device),
                hidden_states=torch.zeros(
                    (physical_bs * _WIDTH, _HIDDEN),
                    dtype=torch.float32,
                    device=device,
                ),
            ),
            can_run_cuda_graph=False,
            routed_experts_output=None,
            indexer_topk_output=None,
        )

    worker._target_worker.forward_batch_generation.side_effect = _forward


def _initial_draft_input(*, bs, device, seq_lens) -> DFlashDraftInputV2:
    return make_draft_input_v2(
        bonus_tokens=torch.arange(2, 2 + bs, dtype=torch.int64, device=device),
        new_seq_lens=seq_lens,
    )


@unittest.skipUnless(_TPU_AVAILABLE, "Requires TPU hardware")
class TestDSparkWorkerDecodeTpu(CustomTestCase):
    def setUp(self):
        super().setUp()
        # compiled_op is gated on this env, which only the ServerArgs pipeline
        # sets; without it the accept tail runs op-by-op and the graphs are
        # never exercised.
        self.enterContext(envs.SGLANG_ENABLE_TORCH_COMPILE.override(True))
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
        _ensure_single_rank_distributed()
        self.device = torch.device("tpu")

    def _build_worker(self):
        """Same weights every time: the fused/eager and padded/plain pairs only
        compare if both runs propose the same draft block."""
        torch.manual_seed(0)
        return _create_dspark_worker_for_tpu(self.device)

    def _run_step(self, worker, *, bs, seq_lens, draft_input, accept_lens, pad_bs=None):
        batch = _make_schedule_batch(
            seq_lens=seq_lens,
            req_pool_indices=torch.arange(bs, dtype=torch.int64, device=self.device),
            spec_info=draft_input,
        )
        _stub_draft_forward(worker, bs=bs, device=self.device)
        _stub_accepting_target(
            worker, bs=bs, device=self.device, accept_lens=accept_lens, pad_bs=pad_bs
        )
        return worker.forward_batch_generation(batch)

    def test_decode_step_chains_on_tpu(self):
        """Two bucket-sized steps then a non-bucket one, each feeding the next."""
        worker = self._build_worker()
        bs = 2
        seq_lens = torch.tensor([5, 9], dtype=torch.int64, device=self.device)
        draft_input = _initial_draft_input(bs=bs, device=self.device, seq_lens=seq_lens)
        accept_lens = [0, _GAMMA]

        for _ in range(2):
            result = self._run_step(
                worker,
                bs=bs,
                seq_lens=seq_lens,
                draft_input=draft_input,
                accept_lens=accept_lens,
            )
            self.assertIsInstance(result, GenerationBatchResult)
            self.assertEqual(result.next_token_ids.shape[0], bs * _WIDTH)
            # commit_lens is correct_len + 1, so the block bounds it.
            torch.testing.assert_close(
                result.accept_lens.cpu(),
                torch.tensor([n + 1 for n in accept_lens], dtype=torch.int32),
            )
            torch.testing.assert_close(
                result.new_seq_lens,
                seq_lens + result.accept_lens.to(seq_lens.dtype),
            )
            seq_lens = result.new_seq_lens
            draft_input = result.next_draft_input

        bs_odd = 3
        seq_lens_odd = torch.tensor([7, 11, 13], dtype=torch.int64, device=self.device)
        accept_odd = [0, 1, _GAMMA]
        result_odd = self._run_step(
            worker,
            bs=bs_odd,
            seq_lens=seq_lens_odd,
            draft_input=_initial_draft_input(
                bs=bs_odd, device=self.device, seq_lens=seq_lens_odd
            ),
            accept_lens=accept_odd,
        )
        torch.testing.assert_close(
            result_odd.accept_lens.cpu(),
            torch.tensor([n + 1 for n in accept_odd], dtype=torch.int32),
        )
        torch.testing.assert_close(
            result_odd.new_seq_lens,
            seq_lens_odd + result_odd.accept_lens.to(seq_lens_odd.dtype),
        )

    def test_padded_verify_outputs_are_trimmed(self):
        """A bucket-padded verify must land the same rows as an unpadded one.

        bs=3 buckets up to 4, so the accept graphs run one row wide and the
        staging buffers carry a junk row. If that row leaks past the trim, or
        the sentinel verify_lens stops making the cap a no-op, the live rows
        move.
        """
        bs = 3
        seq_lens = torch.tensor([7, 11, 13], dtype=torch.int64, device=self.device)
        accept_lens = [0, 1, _GAMMA]

        outcomes = []
        for pad_bs in (4, None):
            worker = self._build_worker()
            outcomes.append(
                self._run_step(
                    worker,
                    bs=bs,
                    seq_lens=seq_lens,
                    draft_input=_initial_draft_input(
                        bs=bs, device=self.device, seq_lens=seq_lens
                    ),
                    accept_lens=accept_lens,
                    pad_bs=pad_bs,
                )
            )

        padded_res, plain_res = outcomes
        self.assertEqual(padded_res.accept_lens.shape[0], bs)
        self.assertEqual(padded_res.next_token_ids.shape[0], bs * _WIDTH)
        for name in ("accept_lens", "new_seq_lens", "next_token_ids"):
            with self.subTest(field=name):
                torch.testing.assert_close(
                    getattr(padded_res, name), getattr(plain_res, name)
                )

    def test_fused_accept_matches_eager_in_worker(self):
        """Clearing _tpu_acceptor must not change the step's accept outcome."""
        bs = 3
        seq_lens = torch.tensor([7, 11, 13], dtype=torch.int64, device=self.device)
        accept_lens = [0, 1, _GAMMA]

        outcomes = []
        for use_fused in (True, False):
            worker = self._build_worker()
            self.assertIsNotNone(worker._verify_executor._tpu_acceptor)
            if not use_fused:
                worker._verify_executor._tpu_acceptor = None
            outcomes.append(
                self._run_step(
                    worker,
                    bs=bs,
                    seq_lens=seq_lens,
                    draft_input=_initial_draft_input(
                        bs=bs, device=self.device, seq_lens=seq_lens
                    ),
                    accept_lens=accept_lens,
                )
            )

        fused, eager = outcomes
        for name in (
            "accept_lens",
            "block_accept_lens",
            "new_seq_lens",
            "next_token_ids",
        ):
            with self.subTest(field=name):
                torch.testing.assert_close(getattr(fused, name), getattr(eager, name))


if __name__ == "__main__":
    unittest.main()
