"""Fused mixed prefill + verify (SGLANG_TPU_MIXED_VERIFY_FUSED), end to end on CPU:
every DP rank kind's MIXED batch built by the worker's own builder
(TpuSpecWorker._forward_mixed_target -> mixed_step.build_mixed_layout), through the
real ScheduleBatch -> ForwardBatch.init_new, the runner's DP MLP-sync padding
(ForwardBatch.prepare_mlp_sync_batch) and the compile policy's preparation
(DecoderTpuCompilePolicy.prepare_forward_batch: _dp_attn_select_bucket_shape's
MIXED branch, _pad_batch_fields, then MixedVerifyInput.record_padding's upload and
verify-token splice).

The four rank kinds of one step (prompt + verify, verify-only, prompt-only, idle)
must build, agree on the padded shapes, and hand the compiled graph the same input
form: field set, None-ness, dtype and view-ness (TorchTPU compiles a view input as
its base buffer) of every tensor field. Two TPU smokes failed on the idle rank only,
inside ForwardBatch.init_new (is_prefill_only inherited from ScheduleBatch.init_new([]),
and the dummy verify request lacking optional request fields); this runs that path.
"""

import dataclasses
from array import array
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from sglang.srt.layers.dp_attention import DpPaddingMode
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.model_executor import forward_batch_info
from sglang.srt.model_executor.cuda_graph_config import CudaGraphConfig, PhaseConfig
from sglang.srt.model_executor.forward_batch_info import (
    LogitsOutputMode,
)
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative.mixed_verify import MixedVerifyPlan
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu.compilation.bucketer import Bucketer
from sglang_tpu.experimental.speculative import mixed_step
from sglang_tpu.experimental.speculative.base_spec_worker import TpuSpecWorker
from sglang_tpu.experimental.speculative.host_verify_rows import HostVerifyRows

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

DP = 4
R = 3  # verify rows per request
VOCAB = 1024
HIDDEN = 8
T_BUCKETS = [8, 16, 32]
B_BUCKETS = [1, 2, 4]
EXT_CAP = 4  # min(token max 32, batch max 4)
PBS = 2  # verify bucket: bucket(max verify requests 2)
L = EXT_CAP + R * PBS
PAD_B = EXT_CAP + PBS
KINDS = ("prompt+verify", "verify-only", "prompt-only", "idle")
RANK = {kind: rank for rank, kind in enumerate(KINDS)}

# Per rank: prompt requests (prefix, rows) and verify requests (committed length).
PROMPTS = {
    "prompt+verify": [(10, 3), (0, 2)],
    "verify-only": [],
    "prompt-only": [(4, 4)],
    "idle": [],
}
VERIFIES = {
    "prompt+verify": [40, 70],
    "verify-only": [55],
    "prompt-only": [],
    "idle": [],
}


def _rows(kind):
    return sum(n for _, n in PROMPTS[kind]) + R * len(VERIFIES[kind])


GLOBAL_PROMPT_ROWS = tuple(sum(n for _, n in PROMPTS[k]) for k in KINDS)
GLOBAL_VERIFY_BS = tuple(len(VERIFIES[k]) for k in KINDS)
# The MLP-synced counts of the step (dp_attn._mixed_verify_num_tokens).
GNT = [_rows(k) for k in KINDS]  # [11, 3, 4, 0]
GNT_LOGPROB = [len(PROMPTS[k]) or len(VERIFIES[k]) for k in KINDS]
PAD_T = next(b for b in T_BUCKETS if b >= max(max(GNT), PAD_B))


def _plan(kind):
    return MixedVerifyPlan(
        n_prompt_rows=GLOBAL_PROMPT_ROWS[RANK[kind]],
        n_verify=GLOBAL_VERIFY_BS[RANK[kind]],
        num_verify_rows=R,
        global_prompt_rows=GLOBAL_PROMPT_ROWS,
        global_verify_bs=GLOBAL_VERIFY_BS,
    )


# --- pools, config, runner -----------------------------------------------------


def _model_config():
    return SimpleNamespace(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        context_len=1024,
        is_generation=True,
        is_encoder_decoder=False,
        is_multimodal=False,
        is_matryoshka=False,
        model_is_mrope=False,
        requires_mm_token_modalities=False,
        use_ngram_embedding=False,
        use_engram=False,
        hf_config=SimpleNamespace(
            model_type="deepseek_v41",
            vision_n_layers=0,
            vocab_size=VOCAB,
            engram_pad_token_id=0,
        ),
    )


MODEL_CONFIG = _model_config()
POOL = SimpleNamespace(device="cpu", req_to_token=torch.zeros((16, 256)))
ALLOCATOR = SimpleNamespace(
    device="cpu", page_size=1, free_pages=torch.zeros(4, dtype=torch.int32)
)


def _req(rid, prompt_len, output_len=0):
    req = Req(rid, "", array("q", range(1, prompt_len + 1)), SamplingParams(), False)
    req.sampling_params.normalize(None)
    req.output_ids.extend(range(output_len))
    req.refresh_fill_ids()
    return req


def _new_batch(reqs):
    return ScheduleBatch.init_new(
        reqs,
        POOL,
        ALLOCATOR,
        None,
        MODEL_CONFIG,
        False,
        SpeculativeAlgorithm.DSPARK,
    )


def _sync(batch, kind):
    batch.mixed_verify = _plan(kind)
    batch.global_num_tokens = list(GNT)
    batch.global_num_tokens_for_logprob = list(GNT_LOGPROB)
    batch.is_extend_in_batch = True
    return batch


def _prompt_batch(kind):
    """The rank's prompt part as the scheduler prepares an extend (host tokens)."""
    spec = PROMPTS[kind]
    reqs = [_req(f"{kind}-p{i}", p + n) for i, (p, n) in enumerate(spec)]
    for req, (p, n) in zip(reqs, spec):
        req.prefix_len = p
        req.extend_end = p + n
    batch = _new_batch(reqs)
    batch.forward_mode = ForwardMode.EXTEND
    rows = sum(n for _, n in spec)
    batch.input_ids = torch.arange(100, 100 + rows, dtype=torch.int64)
    batch.input_tensor_device = "cpu"
    batch.out_cache_loc = torch.arange(500, 500 + rows, dtype=torch.int32)
    batch.prefix_lens = [p for p, _ in spec]
    batch.extend_lens = [n for _, n in spec]
    batch.extend_num_tokens = rows
    batch.extend_logprob_start_lens = [n for _, n in spec]
    seq = torch.tensor([p + n for p, n in spec], dtype=torch.int64)
    batch.seq_lens = seq
    batch.seq_lens_cpu = seq.clone()
    batch.seq_lens_sum = int(seq.sum())
    batch.orig_seq_lens = seq.to(torch.int32)
    slots = torch.arange(5, 5 + len(spec), dtype=torch.int64)
    batch.req_pool_indices = slots
    batch.req_pool_indices_cpu = slots.clone()
    batch.sampling_info = SamplingBatchInfo.from_schedule_batch(batch, VOCAB)
    return _sync(batch, kind)


def _verify_batch(kind):
    """The rank's running batch prepared for decode (the verify part)."""
    commits = VERIFIES[kind]
    reqs = [_req(f"{kind}-v{i}", c - 2, 2) for i, c in enumerate(commits)]
    batch = _new_batch(reqs)
    batch.forward_mode = ForwardMode.DECODE
    n = len(reqs)
    batch.input_ids = torch.zeros((n,), dtype=torch.int64)
    batch.out_cache_loc = torch.zeros((n,), dtype=torch.int32)
    seq = torch.tensor(commits, dtype=torch.int64)
    batch.seq_lens = seq
    batch.seq_lens_cpu = seq.clone()
    batch.seq_lens_sum = int(seq.sum())
    batch.orig_seq_lens = seq.to(torch.int32)
    slots = torch.arange(9, 9 + n, dtype=torch.int64)
    batch.req_pool_indices = slots
    batch.req_pool_indices_cpu = slots.clone()
    batch.sampling_info = SamplingBatchInfo.from_schedule_batch(batch, VOCAB)
    return _sync(batch, kind)


def _idle_batch(kind):
    """The scheduler's idle batch of a rank with neither part."""
    batch = _new_batch([])
    batch.prepare_for_idle()
    return _sync(batch, kind)


def _host_rows(kind):
    """The verify bucket's host rows (HostVerifyRows' layout); padding requests at
    pool row 0, KV slot 0."""
    commits = VERIFIES[kind] + [0] * (PBS - len(VERIFIES[kind]))
    slots = list(range(9, 9 + len(VERIFIES[kind]))) + [0] * (PBS - len(VERIFIES[kind]))
    positions = torch.tensor([c + j for c in commits for j in range(R)])
    locs = torch.tensor([(1000 + c + j) if c else 0 for c in commits for j in range(R)])
    return HostVerifyRows(
        positions=positions, out_cache_loc=locs, req_pool_indices=torch.tensor(slots)
    )


class _Embedding(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(VOCAB, HIDDEN))

    def forward(self, ids):
        return torch.nn.functional.embedding(ids, self.weight)


def _policy(runner):
    """The DSV4.1 target's compile policy (bucketing only; no compile)."""
    from sglang_tpu.experimental.compilation.block_compile_policy import (
        TpuBlockSpecDecodeCompilePolicyMixin,
    )
    from sglang_tpu.models.deepseek_v41.compile_policy import (
        DeepseekV4DecoderTpuCompilePolicy,
    )

    cls = type(
        "SpecDeepseekV4DecoderTpuCompilePolicy",
        (TpuBlockSpecDecodeCompilePolicyMixin, DeepseekV4DecoderTpuCompilePolicy),
        {},
    )
    policy = object.__new__(cls)
    policy.model_runner = runner
    policy.t_bucketer = Bucketer(name="Token", buckets=T_BUCKETS)
    policy.b_bucketer = Bucketer(name="Bs", buckets=B_BUCKETS)
    # Under DP attention the extend cap is the batch bucket max.
    policy.extend_bs = policy.b_bucketer.max_val
    policy.is_v41 = True
    policy._engram_hasher = None
    policy._embeds_outside_graph = True
    policy.is_draft_worker = False
    policy.draft_token_num = R
    return policy


def _runner():
    embedding = _Embedding()
    runner = SimpleNamespace(
        device="cpu",
        is_draft_worker=False,
        model_config=MODEL_CONFIG,
        runner_ids=MagicMock(),
        prefill_attention_backend_str="torch_native",
        ngram_embedding_manager=SimpleNamespace(enabled=False),
        lora_manager=None,
        hisparse_coordinator=None,
        token_to_kv_pool_allocator=ALLOCATOR,
        attn_backend=SimpleNamespace(
            folds_encoder_swa_replay=False,
            decoder_replay=False,
            get_cpu_graph_seq_len_fill_value=lambda: 1,
            get_cuda_graph_seq_len_fill_value=lambda: 1,
            late_layer_num_tokens=lambda fb: None,
        ),
        attn_tp_sequence_sharded=lambda n: False,
        model=SimpleNamespace(vision=None, get_input_embeddings=lambda: embedding),
    )
    runner.compile_policy = _policy(runner)
    return runner


# --- the worker and its target forward ------------------------------------------


class Target:
    """The target worker's forward up to the compiled graph: the runner's eager
    preparation (DP MLP-sync padding), then the policy's (bucket, pad, upload,
    record_padding); records the graph inputs, returns L logits rows and the
    padded hidden."""

    def __init__(self, runner):
        from sglang.srt.model_executor.model_runner import ModelRunner

        self.model_runner = runner
        self._prepare_eager = ModelRunner._prepare_eager_forward_batch
        self.forwards = []

    def forward_batch_generation(self, *, batch, forward_batch, is_verify, **kw):
        assert batch is None and is_verify
        runner = self.model_runner
        self._prepare_eager(runner, forward_batch)
        runner.compile_policy.prepare_forward_batch(forward_batch)
        self.forwards.append(forward_batch)
        rows = int(forward_batch.input_ids.shape[0])
        index = forward_batch.logit_rows
        return SimpleNamespace(
            logits_output=LogitsProcessorOutput(
                next_token_logits=torch.zeros((index.shape[0], VOCAB)),
                hidden_states=torch.zeros((rows, HIDDEN)),
            )
        )


class Worker:
    num_verify_rows = R
    _target_has_linear_attn = False
    device = torch.device("cpu")

    _forward_mixed_target = TpuSpecWorker._forward_mixed_target

    def __init__(self, runner):
        self.target_worker = Target(runner)

    def _compile_policy(self):
        return self.target_worker.model_runner.compile_policy


def _patches(kind):
    stack = ExitStack()
    stack.enter_context(
        get_context().override_server_args(
            device="cpu",
            cuda_graph_config=CudaGraphConfig(prefill=PhaseConfig(bs=[])),
        )
    )
    stack.enter_context(
        get_parallel().override(
            tp_size=DP,
            tp_rank=RANK[kind],
            attn_dp_size=DP,
            attn_dp_rank=RANK[kind],
            attn_tp_size=1,
            attn_tp_rank=0,
            attn_cp_size=1,
            attn_cp_rank=0,
            attn_dcp_size=1,
            moe_ep_size=DP,
            moe_dp_size=1,
            moe_tp_size=1,
            tp_group=SimpleNamespace(cpu_group=None),
        )
    )
    stack.enter_context(
        patch(
            "sglang.srt.layers.dp_attention.is_dp_attention_enabled",
            return_value=True,
        )
    )
    stack.enter_context(
        patch(
            "sglang.srt.model_executor.forward_batch_info.enable_num_token_non_padded",
            return_value=True,
        )
    )
    # CPU runners have no driver to pin the synced token counts with.
    stack.enter_context(patch.object(forward_batch_info, "_is_cpu", True))
    # As on TPU: init_new keeps req_pool_indices_cpu on every mode, and DP
    # attention pads every rank to the same length (MAX_LEN).
    stack.enter_context(patch.object(forward_batch_info, "_is_tpu", True))
    stack.enter_context(
        patch.object(
            DpPaddingMode,
            "get_dp_padding_mode",
            classmethod(lambda cls, *args: cls.MAX_LEN),
        )
    )
    return stack


def run_rank(kind):
    """One rank's fused MIXED forward, as _forward_mixed_fused hands it to the
    target: the verify part (or the idle stand-in) with the bucket state, rows
    and host rows of its verify half; the prompt part on the stash."""
    with _patches(kind):
        runner = _runner()
        worker = Worker(runner)
        prompt = _prompt_batch(kind) if PROMPTS[kind] else None
        verify = _verify_batch(kind) if VERIFIES[kind] else None
        container = prompt or verify or _idle_batch(kind)
        part = mixed_step.MixedPromptPart(
            plan=container.mixed_verify,
            prompt=prompt,
            sync_fields=mixed_step.mixed_sync_fields(container),
        )
        sync = mixed_step.verify_sync_fields(container.mixed_verify)
        batch = (
            verify if verify is not None else mixed_step.idle_like(container, **sync)
        )
        host = _host_rows(kind)
        verify_tokens = torch.arange(900, 900 + PBS * R, dtype=torch.int64)
        out = worker._forward_mixed_target(
            part,
            batch=batch,
            bucket_state=SimpleNamespace(seq_lens=torch.zeros(PBS, dtype=torch.int64)),
            verify_tokens=verify_tokens,
            rows=SimpleNamespace(
                flat_cache_locs=host.out_cache_loc.clone(),
                flat_positions=host.positions.clone(),
            ),
            verify_input=None,
            spec_info="verify-spec",
            logits_output_mode=LogitsOutputMode.FULL_VOCAB,
            host_rows=host,
        )
    (fb,) = worker.target_worker.forwards
    return fb, out, part


_BUILT = {}


def _built(kind):
    if kind not in _BUILT:
        _BUILT[kind] = run_rank(kind)
    return _BUILT[kind]


@pytest.fixture
def ranks():
    """Every rank kind's forward (built once; a kind that fails to build fails
    its own test_every_rank_kind_builds case with the error)."""
    return {kind: _built(kind) for kind in KINDS}


# --- input form -----------------------------------------------------------------


def _form(value):
    """What direct_compile sees of one ForwardBatch field (a registered pytree
    dataclass): a tensor leaf is a graph input resolved by path on every call,
    so its presence, dtype, rank and view-ness must match the warmup's; None
    and scalars are static (baked at trace, never resolved per call)."""
    if isinstance(value, torch.Tensor):
        return ("tensor", value.dtype, value.dim(), value._base is None)
    if value is None or isinstance(value, (bool, int, float, str)):
        return ("static",)
    if isinstance(value, (list, tuple)):
        return (
            "seq",
            tuple(_form(v) for v in value if isinstance(v, torch.Tensor)),
        )
    return (type(value).__name__,)


def _fields(fb):
    names = [f.name for f in dataclasses.fields(fb)]
    return {name: getattr(fb, name) for name in names}


@pytest.mark.parametrize("kind", KINDS)
def test_every_rank_kind_builds_and_counts_its_real_rows(kind):
    fb, out, part = _built(kind)
    assert fb.forward_mode == ForwardMode.MIXED
    assert fb.is_prefill_only is False
    real = 0 if kind == "idle" else _rows(kind)
    assert fb.global_num_token_non_padded_cpu == real
    assert int(fb.global_num_token_non_padded) == real
    assert int(fb.num_token_non_padded) == real  # local count (attn TP 1)
    assert part.ran
    # The verify rows verify() takes, in the TARGET_VERIFY layout.
    assert out.next_token_logits.shape == (PBS * R, VOCAB)
    assert out.hidden_states.shape == (PBS * R, HIDDEN)


def test_every_rank_kind_pads_to_the_same_shapes(ranks):
    shapes = {
        kind: (
            tuple(fb.input_ids.shape),
            tuple(fb.positions.shape),
            tuple(fb.out_cache_loc.shape),
            tuple(fb.seq_lens.shape),
            tuple(fb.req_pool_indices.shape),
            tuple(fb.input_embeds.shape),
            tuple(fb.logit_rows.shape),
            fb.batch_size,
            len(fb.extend_seq_lens_cpu),
            len(fb.extend_prefix_lens_cpu),
            len(fb.mm_inputs),
            tuple(fb.global_num_tokens_cpu),
        )
        for kind, (fb, _, _) in ranks.items()
    }
    expected = (
        (PAD_T,),
        (PAD_T,),
        (PAD_T,),
        (PAD_B,),
        (PAD_B,),
        (PAD_T, HIDDEN),
        (L,),
        PAD_B,
        PAD_B,
        PAD_B,
        PAD_B,
        (PAD_T,) * DP,
    )
    assert shapes == {kind: expected for kind in KINDS}


def test_every_rank_kind_feeds_the_graph_the_same_input_form(ranks):
    forms = {
        kind: {name: _form(v) for name, v in _fields(fb).items()}
        for kind, (fb, _, _) in ranks.items()
    }
    reference = forms[KINDS[0]]
    diffs = {
        kind: {
            name: (reference.get(name), form.get(name))
            for name in set(reference) | set(form)
            if reference.get(name) != form.get(name)
        }
        for kind, form in forms.items()
    }
    assert diffs == {kind: {} for kind in KINDS}


@pytest.mark.parametrize("kind", KINDS)
def test_graph_tensor_inputs_are_tensors_of_their_own(kind):
    fb, _, _ = _built(kind)
    views = sorted(
        name
        for name, v in _fields(fb).items()
        if isinstance(v, torch.Tensor) and v._base is not None
    )
    assert views == []
    assert fb.spec_info.logits_select_index is fb.logit_rows


def test_verify_tokens_land_in_the_verify_rows(ranks):
    for kind, (fb, _, _) in ranks.items():
        prompt_rows = sum(n for _, n in PROMPTS[kind])
        n_seg = len(VERIFIES[kind]) or (0 if PROMPTS[kind] else 1)
        verify = fb.input_ids[prompt_rows : prompt_rows + n_seg * R].tolist()
        assert verify == list(range(900, 900 + n_seg * R)), kind
