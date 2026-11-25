"""TpuSpecWorker fused mixed prefill + verify steps (SGLANG_TPU_MIXED_VERIFY_FUSED):
one MIXED target forward per rank in place of the verify forward, its output split,
the prompt half after it, the graph input form, and host Engram staging (CPU)."""

from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import LogitsOutputMode
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.speculative.mixed_verify import MixedVerifyPlan
from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu.experimental.speculative import base_spec_worker, mixed_step
from sglang_tpu.experimental.speculative.base_spec_worker import TpuSpecWorker
from sglang_tpu.experimental.speculative.host_verify_rows import HostVerifyRows

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

R = 3  # verify rows per request
EXT_CAP = 4  # prompt request cap of the MIXED batch
EXTEND_CAP = 2  # the extend compile cap (prompt sampling rows)
PBS = 2  # verify bucket (pad_bv)
PAD_T = 32  # token bucket of the MIXED forward
L = EXT_CAP + R * PBS
VOCAB = 5
HIDDEN = 2
GNT = [11, 6, 0, 3]  # the step's MLP-synced counts
PLAN = MixedVerifyPlan(5, 2, R, (5, 0, 0, 5), (2, 2, 0, 0))


class Bucketer:
    buckets = (8, 16, 32)
    max_val = 32

    def next_bucket(self, n):
        return next(b for b in self.buckets if b >= n)


class Policy:
    t_bucketer = Bucketer()
    b_bucketer = SimpleNamespace(next_bucket=lambda n: 1 if n <= 1 else 2, max_val=2)

    def live_cache_loc_dtype(self):
        return torch.int32

    def _extend_compile_cap(self):
        return EXTEND_CAP


def _req(rid):
    return SimpleNamespace(
        rid=rid,
        lora_id=None,
        extend_end=0,
        full_untruncated_fill_ids=[],
        origin_input_ids=list(range(1, 40)),
        output_ids=[],
        kv=SimpleNamespace(kv_committed_len=0),
    )


def _prompt():
    """Two prompt requests: 3 rows from 10 (slot 5), 2 rows from 0 (slot 6)."""
    reqs = [_req("p0"), _req("p1")]
    return SimpleNamespace(
        name="prompt",
        forward_mode=ForwardMode.EXTEND,
        reqs=reqs,
        input_ids=torch.tensor([21, 22, 23, 31, 32]),
        prefill_input_ids_cpu=None,
        out_cache_loc=torch.tensor([100, 101, 102, 200, 201], dtype=torch.int32),
        prefix_lens=[10, 0],
        extend_lens=[3, 2],
        extend_num_tokens=5,
        extend_logprob_start_lens=None,
        seq_lens=torch.tensor([13, 2]),
        seq_lens_cpu=torch.tensor([13, 2]),
        seq_lens_sum=15,
        orig_seq_lens=None,
        req_pool_indices=torch.tensor([5, 6]),
        req_pool_indices_cpu=torch.tensor([5, 6]),
        multimodal_inputs=None,
        encoder_swa_reset=[False, False],
        decoder_fill_lens=None,
        engram_history=None,
        decoding_reqs=None,
        spec_info=None,
        return_logprob=False,
        top_logprobs_nums=None,
        token_ids_logprobs=None,
        extend_input_logprob_token_ids=None,
        has_grammar=False,
        sampling_info="prompt-sampling",
        mixed_verify=PLAN,
        mixed_verify_batch=None,
        global_num_tokens=list(GNT),
        global_num_tokens_for_logprob=[2, 0, 0, 2],
        is_extend_in_batch=True,
    )


def _verify_batch(name="verify", n=2, **kw):
    fields = dict(
        name=name,
        forward_mode=ForwardMode.DECODE,
        reqs=[_req(f"v{i}") for i in range(n)],
        input_ids=torch.zeros((0,), dtype=torch.int64),
        out_cache_loc=torch.zeros((0,), dtype=torch.int64),
        seq_lens=torch.zeros((n,), dtype=torch.int64),
        seq_lens_cpu=torch.zeros((n,), dtype=torch.int64),
        req_pool_indices=torch.zeros((n,), dtype=torch.int64),
        req_pool_indices_cpu=torch.zeros((n,), dtype=torch.int64),
        multimodal_inputs=None,
        spec_info="verify-draft-input",
        has_grammar=False,
        sampling_info="verify-sampling",
        mixed_verify=PLAN,
        global_num_tokens=list(GNT),
        global_num_tokens_for_logprob=[2, 0, 0, 2],
        is_extend_in_batch=True,
    )
    fields.update(kw)
    return SimpleNamespace(**fields)


def _host_rows(n_live):
    """The verify bucket's host rows: live requests at 40 / 70 (slots 7, 8)."""
    starts = [40, 70][:n_live] + [0] * (PBS - n_live)
    slots = [7, 8][:n_live] + [0] * (PBS - n_live)
    positions = torch.tensor([s + j for s in starts for j in range(R)])
    locs = torch.tensor([(1000 + s + j) if s else 0 for s in starts for j in range(R)])
    return HostVerifyRows(
        positions=positions, out_cache_loc=locs, req_pool_indices=torch.tensor(slots)
    )


class FakeForwardBatch:
    """ForwardBatch.init_new stand-in: the merged batch's fields as given."""

    @classmethod
    def init_new(cls, batch, runner, *, capture_hidden_mode, **kw):
        fb = SimpleNamespace(
            merged=batch,
            forward_mode=batch.forward_mode,
            input_ids=batch.input_ids,
            positions=batch.spec_info.positions,
            out_cache_loc=batch.out_cache_loc,
            seq_lens=batch.seq_lens,
            req_pool_indices=batch.req_pool_indices,
            extend_seq_lens_cpu=list(batch.extend_lens),
            extend_prefix_lens_cpu=list(batch.prefix_lens),
            spec_info=batch.spec_info,
            capture_hidden_mode=capture_hidden_mode,
            global_num_tokens=batch.global_num_tokens,
            global_num_token_non_padded=None,
            global_num_token_non_padded_cpu=len(batch.input_ids),
            encoder_swa_row_floor=None,
            encoder_swa_compress_rows=None,
            logit_rows=None,
        )
        return fb


class FakeTarget:
    """The target forward: logits row i holds the forward row it selects, the
    hidden row j holds j; records each forward."""

    def __init__(self, calls):
        self.calls = calls
        self.forwards = []
        self.model_runner = SimpleNamespace(
            sample=self._sample, attn_backend=SimpleNamespace()
        )

    def forward_batch_generation(self, *, batch, forward_batch, is_verify, **kw):
        assert batch is None and is_verify
        self.calls.append(("mixed_forward", forward_batch.forward_mode.name))
        self.forwards.append(forward_batch)
        # The policy pads and uploads, then hands the batch to its spec input.
        forward_batch.spec_info.record_padding(
            logical_batch_size=0,
            physical_batch_size=0,
            logical_num_tokens=0,
            physical_num_tokens=PAD_T,
        )
        index = forward_batch.logit_rows
        logits = index.to(torch.float32)[:, None].repeat(1, VOCAB)
        hidden = torch.arange(PAD_T, dtype=torch.float32)[:, None].repeat(1, HIDDEN)
        return SimpleNamespace(
            logits_output=LogitsProcessorOutput(
                next_token_logits=logits, hidden_states=hidden
            )
        )

    def _sample(self, logits_output, view):
        rows = logits_output.next_token_logits.shape[0]
        self.calls.append(("sample", rows, view.sampling_info))
        return logits_output.next_token_logits[:, 0].to(torch.int64) + 1000


class FakeWorker:
    num_verify_rows = R
    _target_has_linear_attn = False
    _dp_attention = True
    device = torch.device("cpu")

    _forward_mixed = TpuSpecWorker._forward_mixed
    _forward_mixed_fused = TpuSpecWorker._forward_mixed_fused
    _forward_target_verify = TpuSpecWorker._forward_target_verify
    _forward_mixed_target = TpuSpecWorker._forward_mixed_target
    _finish_mixed_prompt = TpuSpecWorker._finish_mixed_prompt
    _sample_mixed_prompt = TpuSpecWorker._sample_mixed_prompt

    def __init__(self):
        self.calls = []
        self._mixed_steps = 0
        self._mixed_prompt = None
        self.target_worker = FakeTarget(self.calls)
        self.verify_outputs = []
        self.stash_seen = []

    def _compile_policy(self):
        return Policy()

    def _pad_bs(self, n):
        return 1 if n <= 1 else 2

    def _device_zeros(self, rows):
        return torch.zeros((rows,), dtype=torch.int64)

    def _check_supported(self, **kw):
        pass

    def _verify_half(self, batch, n_live, *, idle):
        self.stash_seen.append(self._mixed_prompt)
        self.calls.append(("draft", batch.name, list(batch.global_num_tokens)))
        verify_tokens = torch.arange(500, 500 + PBS * R)
        rows = SimpleNamespace(
            flat_cache_locs=torch.arange(PBS * R), flat_positions=torch.zeros(PBS * R)
        )
        out = self._forward_target_verify(
            batch=batch,
            bucket_state=SimpleNamespace(seq_lens=torch.zeros(PBS)),
            verify_tokens=verify_tokens,
            rows=rows,
            verify_input=SimpleNamespace(req_pool_indices_cpu=torch.zeros(PBS)),
            spec_info="verify-spec",
            logits_output_mode=LogitsOutputMode.FULL_VOCAB,
            host_rows=_host_rows(n_live),
        )
        self.verify_outputs.append(out)
        self.calls.append(("accept_commit", out.next_token_logits.shape[0]))
        return out

    def _forward_decode(self, batch, *, grammar_barrier=None):
        assert batch.is_extend_in_batch is False
        self._verify_half(batch, len(batch.reqs), idle=False)
        return SimpleNamespace(tag="verify", new_seq_lens="verify-lens")

    def _forward_idle_decode(self, batch):
        with patch.object(batch, "forward_mode", ForwardMode.DECODE):
            self._verify_half(batch, 0, idle=True)

    def sync_target_tokens(self, ids):
        self.calls.append(("sync", int(ids.shape[0])))
        return ids

    def inject_mixed_prompt(self, prompt, hidden):
        self.calls.append(("inject", prompt is not None, int(hidden.shape[0])))
        self.injected = hidden
        return "fence"

    def mixed_prompt_result(self, prompt, result, kv_fence_tail):
        result.new_seq_lens = "prompt-lens"
        result.kv_fence_tail = kv_fence_tail
        result.mixed_verify_result = None
        return result

    def _idle_result(self):
        return SimpleNamespace(tag="idle", mixed_verify_result=None)


def _idle_like(batch, **overrides):
    idle = _verify_batch(f"idle-like-{batch.name}", n=0, forward_mode=ForwardMode.IDLE)
    idle.global_num_tokens = list(batch.global_num_tokens)
    for k, v in overrides.items():
        setattr(idle, k, v)
    return idle


def _patches(fused=True, fold=None):
    stack = ExitStack()
    stack.enter_context(patch.object(mixed_step, "idle_like", _idle_like))
    # The test's token buckets are tiny; the draft KV row multiple scales with them.
    stack.enter_context(patch.object(mixed_step, "DRAFT_KV_ROW_MULTIPLE", 8))
    stack.enter_context(patch.object(mixed_step, "fused_mixed_forward", lambda: fused))
    stack.enter_context(
        patch.object(base_spec_worker, "ForwardBatch", FakeForwardBatch)
    )
    stack.enter_context(
        patch(
            "sglang_tpu.compilation.compiled_forward_shim.mixed_verify_caps",
            lambda plan, t, b: (EXT_CAP, PBS),
        )
    )
    features = SimpleNamespace(enable_encoder_swa_bounded_replay=fold is not None)
    stack.enter_context(
        patch(
            "sglang.srt.runtime_context.get_exec",
            lambda: SimpleNamespace(features=features),
        )
    )
    stack.enter_context(
        patch(
            "sglang.srt.model_executor.encoder_swa_replay.fold_encoder_swa_replay",
            lambda worker, batch: fold,
        )
    )
    return stack


def _run(batch, *, fused=True, fold=None, worker=None):
    worker = worker or FakeWorker()
    if fold is not None:
        worker.target_worker.model_runner.attn_backend.folds_encoder_swa_replay = True
    published = []
    with _patches(fused, fold):
        result = TpuSpecWorker._forward_mixed(
            worker, batch, grammar_barrier=None, on_publish_verify=published.append
        )
    return worker, result, published


VERIFY_COUNTS = [2, 2, 0, 0]


# (a) dispatch and stash lifecycle ------------------------------------------------


def test_fused_dispatch_stashes_the_prompt_for_the_verify_half_and_clears_it():
    prompt = _prompt()
    prompt.mixed_verify_batch = _verify_batch()
    worker, result, published = _run(prompt)
    (part,) = worker.stash_seen
    assert part is not None and part.prompt is prompt and part.plan is PLAN
    assert part.sync_fields["global_num_tokens"] == GNT
    assert part.ran and worker._mixed_prompt is None


def test_stash_is_cleared_when_the_verify_half_raises():
    prompt = _prompt()
    prompt.mixed_verify_batch = _verify_batch()
    worker = FakeWorker()

    def boom(*a, **k):
        raise ValueError("draft failed")

    worker._forward_decode = boom
    with _patches(), pytest.raises(ValueError):
        TpuSpecWorker._forward_mixed(
            worker, prompt, grammar_barrier=None, on_publish_verify=None
        )
    assert worker._mixed_prompt is None


# (b) the same call order on every rank kind ------------------------------------


def _order(calls):
    return [c[0] for c in calls]


SAME_ORDER = ["draft", "mixed_forward", "accept_commit", "sync", "inject"]


def test_prompt_and_verify_rank():
    prompt = _prompt()
    verify = _verify_batch()
    prompt.mixed_verify_batch = verify
    worker, result, published = _run(prompt)
    calls = [c for c in worker.calls if c[0] != "sample"]
    assert _order(calls) == SAME_ORDER
    assert calls[0] == ("draft", "verify", VERIFY_COUNTS)
    assert ("sync", 2) in calls and ("inject", True, 8) in calls
    assert result.new_seq_lens == "prompt-lens"
    assert result.mixed_verify_result.tag == "verify"
    assert published == ["verify-lens"]
    # The verify half's sync fields are restored on the verify part.
    assert verify.global_num_tokens == GNT and verify.is_extend_in_batch is True
    # Stage A's verify-forward batch state.
    assert verify.forward_mode == ForwardMode.TARGET_VERIFY
    assert verify.spec_info == "verify-spec"


def test_verify_only_rank():
    verify = _verify_batch()
    worker, result, published = _run(verify)
    assert _order(worker.calls) == SAME_ORDER
    assert ("sync", 0) in worker.calls and ("inject", False, 8) in worker.calls
    assert result.tag == "verify" and published == []
    (fb,) = worker.target_worker.forwards
    assert fb.merged.extend_lens == [R, R] and fb.global_num_tokens == GNT


def test_prompt_only_rank_runs_the_idle_verify_half():
    prompt = _prompt()
    worker, result, published = _run(prompt)
    calls = [c for c in worker.calls if c[0] != "sample"]
    assert _order(calls) == SAME_ORDER
    assert calls[0] == ("draft", "idle-like-prompt", VERIFY_COUNTS)
    (fb,) = worker.target_worker.forwards
    # Prompt rows only: no verify request.
    assert fb.merged.extend_lens == [3, 2]
    assert result.new_seq_lens == "prompt-lens" and published == []


def test_idle_rank_runs_a_dummy_mixed_forward_with_no_real_rows():
    idle = _verify_batch("idle", n=0, forward_mode=ForwardMode.IDLE)
    worker, result, _ = _run(idle)
    assert _order(worker.calls) == SAME_ORDER
    (fb,) = worker.target_worker.forwards
    # One dummy verify request (pool row 0), masked by the non-padded count.
    assert fb.merged.extend_lens == [R]
    assert fb.merged.req_pool_indices_cpu.tolist() == [0]
    assert fb.global_num_token_non_padded_cpu == 0
    assert result.tag == "idle"


def test_every_rank_kind_hands_verify_the_same_shapes():
    shapes = set()
    for batch in (
        _prompt(),
        _verify_batch(),
        _verify_batch("idle", n=0, forward_mode=ForwardMode.IDLE),
    ):
        worker, _, _ = _run(batch)
        (out,) = worker.verify_outputs
        shapes.add(
            (
                tuple(out.next_token_logits.shape),
                tuple(out.hidden_states.shape),
                tuple(worker.target_worker.forwards[0].logit_rows.shape),
                tuple(worker.injected.shape),
            )
        )
    assert shapes == {((PBS * R, VOCAB), (PBS * R, HIDDEN), (L,), (8, HIDDEN))}


# (c) output slicing and the prompt split ----------------------------------------


def test_verify_outputs_are_the_verify_rows_and_prompt_outputs_stay_on_the_part():
    prompt = _prompt()
    prompt.mixed_verify_batch = _verify_batch()
    worker, result, _ = _run(prompt)
    (out,) = worker.verify_outputs
    (fb,) = worker.target_worker.forwards
    P = 5
    # Logits rows [ext_cap, L): each verify row's forward row.
    assert out.next_token_logits[:, 0].tolist() == [float(P + j) for j in range(6)]
    # Hidden at the verify offsets after the prompt rows.
    assert out.hidden_states[:, 0].tolist() == [float(P + j) for j in range(6)]
    # The index: each prompt request's last row, then the verify rows.
    assert fb.logit_rows.tolist() == [2, 4, 0, 0] + list(range(5, 11))
    # Prompt sampling reads the prompt requests' last rows at the extend cap.
    assert ("sample", 2, "prompt-sampling") in worker.calls
    assert result.next_token_ids.tolist() == [1002, 1004]
    # The draft inject reads the prompt rows at the prompt's extend bucket.
    assert worker.injected[:, 0].tolist() == [0, 1, 2, 3, 4] + [7] * 3
    # Positions: prompt rows from their prefix, verify rows from the host rows.
    assert fb.positions.tolist() == [10, 11, 12, 0, 1, 40, 41, 42, 70, 71, 72]
    assert fb.merged.prefix_lens == [10, 0, 40, 70]
    # The prompt's host ids, then the device verify tokens spliced in.
    assert fb.input_ids.tolist() == [21, 22, 23, 31, 32] + list(range(500, 506))
    assert fb.merged.seq_lens.tolist() == [13, 2, 43, 73]


def test_padding_verify_rows_clamp_into_the_forward():
    index = mixed_step.verify_hidden_index(30, PBS, R, PAD_T)
    assert index.tolist() == [30, 31, 31, 31, 31, 31]


def test_folded_prompt_rows_drop_replay_rows_and_carry_row_maps():
    prompt = _prompt()
    prompt.encoder_swa_reset = [True, False]
    folded_batch = SimpleNamespace(**vars(prompt))
    # Request 0 replays 4 rows before its 3: 7 folded rows, then request 1's 2.
    folded_batch.prefix_lens = [6, 0]
    folded_batch.extend_lens = [7, 2]
    folded_batch.extend_num_tokens = 9
    folded_batch.input_ids = torch.tensor([1, 2, 3, 4, 21, 22, 23, 31, 32])
    folded_batch.out_cache_loc = torch.arange(9, dtype=torch.int32)
    fold = SimpleNamespace(
        batch=folded_batch,
        num_rows=9,
        keep_rows=torch.tensor([4, 5, 6, 7, 8]),
        row_floor=torch.tensor([6] * 7 + [0, 0]),
    )
    prompt.mixed_verify_batch = _verify_batch()
    worker, _, _ = _run(prompt, fold=fold)
    (fb,) = worker.target_worker.forwards
    (out,) = worker.verify_outputs
    assert fb.encoder_swa_row_floor.tolist() == [6] * 7 + [0] * 8
    assert fb.encoder_swa_compress_rows.tolist() == [4, 5, 6, 7, 8] + list(range(9, 15))
    assert out.hidden_states[:, 0].tolist() == [float(9 + j) for j in range(6)]
    assert worker.injected[:, 0].tolist() == [4, 5, 6, 7, 8] + [7] * 3
    assert fb.mixed_host_rows.prompt_folded
    assert fb.mixed_host_rows.prompt_starts == [6, 0]


# (d) result assembly ---------------------------------------------------------


def test_prompt_result_carries_the_verify_result_and_publishes_it():
    prompt = _prompt()
    prompt.mixed_verify_batch = _verify_batch()
    _, result, published = _run(prompt)
    assert result.kv_fence_tail == "fence"
    assert result.mixed_verify_result.new_seq_lens == "verify-lens"
    assert published == ["verify-lens"]


# (f) fused off: Stage A untouched ---------------------------------------------


def test_fused_off_runs_stage_a():
    prompt = _prompt()
    prompt.mixed_verify_batch = _verify_batch()
    worker = FakeWorker()
    halves = []
    worker._forward_decode = lambda b, grammar_barrier=None: (
        halves.append("decode") or (SimpleNamespace(new_seq_lens="v"))
    )
    worker.extend = lambda b, pp_proxy_tensors: (
        halves.append("extend")
        or (SimpleNamespace(mixed_verify_result=None, new_seq_lens="p"))
    )
    with _patches(fused=False):
        TpuSpecWorker._forward_mixed(
            worker, prompt, grammar_barrier=None, on_publish_verify=None
        )
    assert halves == ["decode", "extend"]
    assert worker.target_worker.forwards == [] and worker._mixed_prompt is None


def test_target_verify_without_a_stash_builds_the_verify_forward():
    worker = SimpleNamespace(
        _mixed_prompt=None,
        _build_verify_batch=lambda **kw: (_ for _ in ()).throw(StopIteration("tv")),
    )
    with pytest.raises(StopIteration, match="tv"):
        TpuSpecWorker._forward_target_verify(
            worker,
            batch=None,
            bucket_state=None,
            verify_tokens=None,
            rows=None,
            verify_input=None,
            spec_info=None,
            logits_output_mode=LogitsOutputMode.FULL_VOCAB,
        )


# Graph input form ------------------------------------------------------------


def _layout(kind):
    prompt = _prompt() if kind in ("prompt+verify", "prompt-only") else None
    n_verify = 2 if kind in ("prompt+verify", "verify-only") else 0
    base = _verify_batch(n=n_verify)
    # Host rows as views of one packed buffer, as a careless build would hand them.
    packed = torch.cat(
        [_host_rows(n_verify).positions, _host_rows(n_verify).out_cache_loc]
    )
    rows = mixed_step.HostRows(
        positions=packed[: PBS * R],
        out_cache_loc=packed[PBS * R :],
        req_pool_indices=_host_rows(n_verify).req_pool_indices,
    )
    spec = mixed_step.make_mixed_verify_input(
        positions=torch.zeros((0,), dtype=torch.int64), num_verify_rows=R
    )
    layout = mixed_step.build_mixed_layout(
        base=base,
        prompt=prompt,
        folded=None,
        host_rows=rows,
        n_verify=n_verify,
        verify_reqs=base.reqs,
        verify_tokens=torch.arange(PBS * R),
        num_verify_rows=R,
        ext_cap=EXT_CAP,
        pad_bv=PBS,
        out_cache_loc_dtype=torch.int32,
        spec_info=spec,
    )
    return layout, spec


KINDS = ("prompt+verify", "verify-only", "prompt-only", "idle")


@pytest.mark.parametrize("kind", KINDS)
def test_every_path_feeds_the_mixed_graph_fresh_tensors(kind):
    """TorchTPU compiles a view input as its whole base buffer: every MIXED
    input of every rank kind is a tensor of its own, as the warmup's are."""
    layout, spec = _layout(kind)
    fields = {
        name: getattr(layout.batch, name) for name in mixed_step.MIXED_INPUT_FIELDS
    }
    fields.update(positions=layout.positions, logits_index=layout.logits_index)
    for name, t in fields.items():
        assert isinstance(t, torch.Tensor), name
        assert t._base is None, (kind, name)
    assert layout.logits_index.shape == (L,)
    # After the policy's upload: the spliced ids and uploads are fresh too.
    spec.positions = layout.positions
    spec.verify_tokens = torch.arange(100, 100 + PBS * R)
    spec.verify_row_start = layout.prompt_rows
    spec.num_verify_tokens = layout.verify_rows
    fb = SimpleNamespace(
        input_ids=torch.nn.functional.pad(layout.batch.input_ids, (0, 4)),
        positions=layout.positions,
        out_cache_loc=layout.batch.out_cache_loc,
        req_pool_indices=layout.batch.req_pool_indices,
        seq_lens=layout.batch.seq_lens,
    )
    mixed_step.finish_mixed_inputs(fb, spec, torch.device("cpu"))
    for name in ("input_ids", "positions", "out_cache_loc", "req_pool_indices"):
        assert getattr(fb, name)._base is None, (kind, name)
    P, V = layout.prompt_rows, layout.verify_rows
    assert fb.input_ids[P : P + V].tolist() == list(range(100, 100 + V))


def test_splice_puts_the_verify_tokens_in_their_rows():
    ids = torch.tensor([1, 2, 3, 0, 0, 0, 0, 0])
    out = mixed_step.splice_verify_tokens(
        ids, torch.tensor([7, 8, 9, 10]), mixed_step.splice_index(8, 3, 3)
    )
    assert out.tolist() == [1, 2, 3, 7, 8, 9, 0, 0]


def test_mixed_verify_input_counts_tokens_not_requests():
    spec = mixed_step.make_mixed_verify_input(
        positions=torch.zeros(0), num_verify_rows=R
    )
    assert spec.num_tokens_per_req == 1 and spec.num_tokens_for_logprob_per_req == 1
    assert spec.capture_hidden_mode.is_full() and spec.draft_token_num == R


def test_inject_rows_never_take_a_fine_bucket():
    t = SimpleNamespace(
        max_val=2048,
        next_bucket=lambda n: next(b for b in (256, 512, 1024, 1152, 2048) if b >= n),
    )
    plan = MixedVerifyPlan(900, 0, R, (900, 1100), (0, 0))
    assert mixed_step.inject_rows(plan, t, 8) == 2048
    plan = MixedVerifyPlan(900, 0, R, (900, 300), (0, 0))
    assert mixed_step.inject_rows(plan, t, 8) == 1024


def test_logits_processor_prefers_the_logit_rows_field_for_mixed():
    """The target ignores spec_info in its compiled graph: a MIXED forward reads
    its index from ForwardBatch.logit_rows alone (smokes j-0d12ddc2 /
    j-2ac8a7c7 returned one row per request when the read required spec_info)."""
    from sglang.srt.layers import logits_processor as lp

    source = open(lp.__file__).read()
    branch = source[
        source.index("elif forward_batch.forward_mode == ForwardMode.MIXED") :
    ]
    branch = branch[: branch.index("else:")]
    assert "mixed_logits_select_index(forward_batch)" in branch
    index = torch.tensor([1, 2])
    assert (
        lp.mixed_logits_select_index(SimpleNamespace(logit_rows=index, spec_info=None))
        is index
    )


# (e) host Engram: prompt rows as an extend, verify rows never commit -------------

NGRAM = 3


def _tables():
    return dict(
        token_map=np.asarray([3, 1, 6, 0, 2, 5, 4, 7], dtype=np.int64),
        multipliers=np.asarray(
            [[100000000000000003, 13, 31], [7, 99999999999999999, 19]],
            dtype=np.int64,
        ),
        primes=np.asarray([[[11, 13], [17, 19]], [[23, 29], [31, 37]]]),
        offsets=np.asarray([[0, 11, 24, 41], [0, 23, 52, 83]]),
        pad_id=3,
    )


def _coordinator():
    from sglang_tpu.models.deepseek_v41.engram_host import HostEngramCoordinator

    tables = _tables()
    hasher = SimpleNamespace(
        max_ngram_size=NGRAM,
        **{k: torch.as_tensor(v) for k, v in tables.items() if k != "pad_id"},
        pad_id=tables["pad_id"],
        image_token_id=None,
        history=torch.zeros((12, NGRAM - 1), dtype=torch.int32),
        pad_row=11,
    )
    embeddings = []
    for layer in (0, 1):
        gen = torch.Generator().manual_seed(layer + 3)
        weight = torch.randint(0, 127, (130, 64), dtype=torch.uint8, generator=gen)
        scale = torch.randint(100, 140, (130, 2), dtype=torch.uint8, generator=gen)
        embeddings.append(
            (
                layer,
                SimpleNamespace(
                    weight=weight.view(torch.float8_e4m3fn),
                    scale=scale.view(torch.float8_e8m0fnu),
                ),
            )
        )
    return HostEngramCoordinator(hasher, embeddings, "cpu"), embeddings, hasher


PROMPT = [1, 2, 3, 4, 5, 6, 7, 0, 2, 4, 6, 1, 3, 5, 7, 2, 4, 6]


def _engram_req(prompt=PROMPT, slot=None):
    return SimpleNamespace(
        origin_input_ids=list(prompt),
        output_ids=[],
        beam_group=None,
        kv=SimpleNamespace(req_pool_idx=slot),
    )


def _engram_batch(reqs, starts, lengths, slots):
    mode = SimpleNamespace(
        is_decode=lambda: False,
        is_extend_without_speculative=lambda: True,
        is_extend=lambda: True,
        is_target_verify=lambda: False,
        is_idle=lambda: False,
    )
    return SimpleNamespace(
        reqs=reqs,
        spec_algorithm=SimpleNamespace(is_none=lambda: True),
        forward_mode=mode,
        prefix_lens=starts,
        extend_lens=lengths,
        seq_lens_cpu=None,
        req_pool_indices_cpu=torch.tensor(slots),
    )


def _mixed_rows(reqs, starts, lengths, slots, *, n_verify=2, folded=False):
    tokens = torch.tensor([5, 6, 7, 1, 2, 3, 0, 0, 0])  # one padding request
    positions = torch.tensor([20, 21, 22, 30, 31, 32])[: n_verify * R]
    return mixed_step.MixedHostRows(
        prompt_reqs=reqs,
        prompt_starts=starts,
        prompt_lengths=lengths,
        prompt_slots=slots,
        prompt_folded=folded,
        verify_tokens=tokens,
        verify_positions=positions,
        verify_slots=torch.tensor([3, 4])[:n_verify],
        num_verify_rows=R,
    )


def _expected_hashes(coordinator, hasher_history, rows):
    from sglang_tpu.models.deepseek_v41.engram_host import (
        capture_inputs,
        hash_on_host,
        verify_inputs_from_host,
    )

    tables = _tables()
    verify = hash_on_host(
        verify_inputs_from_host(
            hasher_history,
            rows.verify_tokens[: rows.verify_rows],
            rows.verify_positions,
            rows.verify_slots,
            block=R,
            ngram=NGRAM,
        ),
        **tables,
    )
    prompt = hash_on_host(
        capture_inputs(
            rows.prompt_reqs, rows.prompt_starts, rows.prompt_lengths, NGRAM
        ).resolve(None),
        **tables,
    )
    return torch.cat([prompt, verify])


def _packed(embeddings, hashes, padded):
    from sglang_tpu.models.deepseek_v41.engram_host import pack_rows

    return [
        pack_rows(
            e.weight, e.scale, hashes[:, layer], row_start=0, padded_tokens=padded
        )
        for layer, e in embeddings
    ]


def test_mixed_staging_commits_prompt_history_and_never_verify_history():
    coordinator, embeddings, hasher = _coordinator()
    hasher.history[3] = torch.tensor([7, 6], dtype=torch.int32)
    hasher.history[4] = torch.tensor([5, 4], dtype=torch.int32)
    before = hasher.history.clone()
    req = _engram_req()
    rows = _mixed_rows([req], [4], [5], [6])
    expected = _expected_hashes(coordinator, before, rows)
    staged, uploads, _ = coordinator.stage_mixed(16, rows)
    for dev, ref in zip(staged, _packed(embeddings, expected, 16), strict=True):
        assert torch.equal(dev, ref)
    # Verify slots keep their history (the deferred commit after accept writes it).
    assert torch.equal(hasher.history[3], before[3])
    assert torch.equal(hasher.history[4], before[4])
    # The prompt slot holds its chunk's last n - 1 tokens, as an extend commits.
    assert hasher.history[6].tolist() == [PROMPT[7], PROMPT[8]]


def test_policy_routes_a_fused_mixed_forward_to_mixed_staging():
    from sglang_tpu.models.deepseek_v41 import compile_policy

    coordinator = SimpleNamespace(
        started=True,
        stage_mixed=lambda padded, rows: (("staged",), ["upload"], 4),
        stage_all=lambda *a, **k: pytest.fail("stage_all"),
    )
    fb = SimpleNamespace(
        mixed_host_rows="rows",
        forward_mode=ForwardMode.MIXED,
        tpu_engram_staged=None,
        tpu_engram_local_ids=None,
    )
    with (
        patch.object(compile_policy.engram_host, "move_hasher_to_host"),
        patch.object(
            compile_policy.engram_host,
            "host_hash_ids",
            side_effect=AssertionError("host_hash_ids commits verify rows"),
        ),
        patch.object(compile_policy.engram_host, "local_row_ids", return_value="ids"),
    ):
        uploads = compile_policy._stage_host_engram(
            coordinator, SimpleNamespace(), torch.zeros(16), fb
        )
    assert uploads == ["upload"] and fb.tpu_engram_staged == ("staged",)


def _make_prefetch(coordinator, req, slot):
    coordinator.prepare(_engram_batch([req], [0], [8], [slot]))
    _, uploads, _ = coordinator.stage_all(16)
    coordinator.record_consumers(uploads, [torch.tensor([1])])
    coordinator.finish_forward(16)
    prefetch = coordinator._prefetch
    assert prefetch is not None and prefetch.starts == [8]
    coordinator.clear_step()
    return prefetch


def test_next_chunk_prefetch_survives_a_fused_mixed_staging_of_another_prompt():
    coordinator, embeddings, _ = _coordinator()
    req_a, req_b = _engram_req(), _engram_req(PROMPT[::-1])
    prefetch = _make_prefetch(coordinator, req_a, 6)
    rows = _mixed_rows([req_b], [0], [5], [7])
    _, uploads, _ = coordinator.stage_mixed(16, rows)
    coordinator.record_consumers(uploads, [torch.tensor([2])])
    coordinator.clear_step()
    assert coordinator._prefetch is prefetch
    for upload in prefetch.uploads.values():
        assert all(upload is not u for u in uploads)
    # The next extend of chunk 2 still takes it.
    coordinator.prepare(_engram_batch([req_a], [8], [8], [6]))
    assert coordinator.hashes is prefetch.hashes


def test_fused_mixed_staging_takes_the_prefetched_chunk_and_appends_verify_rows():
    coordinator, embeddings, hasher = _coordinator()
    req = _engram_req(PROMPT * 2)
    prefetch = _make_prefetch(coordinator, req, 6)
    before = hasher.history.clone()
    rows = _mixed_rows([req], [8], [8], [6])
    expected = _expected_hashes(coordinator, before, rows)
    staged, uploads, _ = coordinator.stage_mixed(16, rows)
    assert coordinator._prefetch is None
    for (layer, _), dev, ref in zip(
        embeddings, staged, _packed(embeddings, expected, 16), strict=True
    ):
        assert dev is prefetch.uploads[layer].device
        assert torch.equal(dev, ref)
    # The next chunk prefetches after the fused forward, as after an extend.
    assert coordinator.has_next_prompt_prefetch()


def test_fused_compile_cases_cover_every_verify_bucket_that_fits():
    t = SimpleNamespace(buckets=(128, 256, 512), max_val=512)
    b = SimpleNamespace(
        next_bucket=lambda n: next(x for x in (4, 8, 16) if x >= n), max_val=16
    )
    cases = mixed_step.fused_compile_cases(t, b, attn_tp_size=1)
    ext_cap = 16  # min(token max, batch max)
    assert {pbs for _, _, pbs in cases} == {4, 8, 16}
    assert all(cap == ext_cap and tokens >= cap + pbs for tokens, cap, pbs in cases)
    assert len(cases) == 9


def test_fused_compile_cases_floor_under_decoder_replay(monkeypatch):
    """Under decoder replay fused MIXED compiles only from its token floor up (the
    floor is clamped to the bucketer's largest bucket)."""
    t = SimpleNamespace(buckets=(256, 512, 1024), max_val=1024)
    b = SimpleNamespace(
        next_bucket=lambda n: next(x for x in (4, 8, 16) if x >= n), max_val=16
    )
    monkeypatch.setattr(mixed_step, "mixed_min_token_rows", lambda: 1024)
    cases = mixed_step.fused_compile_cases(t, b, attn_tp_size=1)
    assert {tokens for tokens, _, _ in cases} == {1024}
    assert len(cases) == 3
    small = SimpleNamespace(buckets=(128, 256, 512), max_val=512)
    assert {n for n, _, _ in mixed_step.fused_compile_cases(small, b, 1)} == {512}


def test_dummy_mixed_rows_fit_the_request_window_workspace():
    """A dummy MIXED batch (ext_cap + verify bucket requests) must claim no more
    SWA request-window workspace pages than a real step can: one request holds
    the rows, the rest are zero-row padding."""
    import torch

    from sglang_tpu.experimental.compilation.spec_compile_policy import (
        _concentrate_dummy_rows,
    )
    from sglang_tpu.models.deepseek_v41.request_window import (
        _assign_workspace_pages,
        _workspace_pages,
    )

    bs, rows = 128, 1024
    batch = SimpleNamespace(
        extend_num_tokens=rows,
        extend_seq_lens=torch.ones(bs, dtype=torch.int32),
        seq_lens=torch.ones(bs, dtype=torch.int64),
    )
    _concentrate_dummy_rows(batch, bs)
    assert (
        batch.extend_seq_lens_cpu[0] == rows and sum(batch.extend_seq_lens_cpu) == rows
    )
    assert int(batch.seq_lens.sum()) == rows and batch.seq_lens_sum == rows
    capacity = _workspace_pages(num_slots=17, chunk_tokens=1024)  # 84
    _assign_workspace_pages(
        first=torch.zeros(bs, dtype=torch.int64),
        counts=torch.tensor(batch.extend_seq_lens_cpu, dtype=torch.int64),
        columns=4096,
        base_page=0,
        capacity_pages=capacity,
    )  # raises if over capacity
    with pytest.raises(RuntimeError, match="workspace"):
        # One row per request (the old dummy) needs a page each.
        _assign_workspace_pages(
            first=torch.zeros(bs, dtype=torch.int64),
            counts=torch.ones(bs, dtype=torch.int64),
            columns=4096,
            base_page=0,
            capacity_pages=capacity,
        )


def test_merged_mixed_batch_is_never_prefill_only():
    """An idle rank's empty prompt view has is_prefill_only = all([]) = True;
    the merged MIXED batch must not inherit it."""
    view = SimpleNamespace(is_prefill_only=True, forward_mode=None)
    merged = mixed_step.merge_mixed_batch(
        view, None, prompt_input_ids=None, spec_info=None, mixed_mode="MIXED"
    )
    assert merged.is_prefill_only is False and merged.forward_mode == "MIXED"


def test_dummy_verify_req_answers_optional_request_fields():
    req = mixed_step._DummyVerifyReq()
    for name in (
        "token_type_ids",
        "dimensions",
        "multi_item_delimiter_indices",
        "token_indices_to_pool",
        "decision_layout",
        "some_future_optional_field",
    ):
        assert getattr(req, name) is None
    assert req.return_pooled_hidden_states is False
    with pytest.raises(AttributeError):
        req._private


def test_fused_mixed_outputs_skip_the_decoder_trim():
    """The policy must not trim a fused MIXED forward's L logits rows (smoke
    j-0d12ddc2-6d5d-4edc-8667: 64 rows came back, 112 expected)."""
    import torch

    from sglang.srt.model_executor.forward_mode import ForwardMode
    from sglang_tpu.compilation import compiled_forward_shim as shim
    from sglang_tpu.experimental.compilation import spec_compile_policy as scp
    from sglang_tpu.experimental.compilation.spec_compile_policy import (
        TpuSpecDecodeCompilePolicyMixin,
    )

    stack_patch = patch.object(scp, "_fused_mixed_steps", return_value=True)
    stack_patch.start()
    policy = object.__new__(TpuSpecDecodeCompilePolicyMixin)
    fb = SimpleNamespace(
        forward_mode=ForwardMode.MIXED,
        mixed_verify=object(),
        logit_rows=None,
        # The batch's spec_info need not carry the worker's flag.
        spec_info=SimpleNamespace(keep_outputs_padded=False),
    )
    setattr(fb, shim._REAL_Q_KEY, 40)
    setattr(fb, shim._REAL_B_KEY, 3)
    out = SimpleNamespace(
        next_token_logits=torch.zeros(112, 8), hidden_states=torch.zeros(128, 4)
    )
    with patch.object(shim, "_restore_forward_batch_inputs") as restore:
        got = policy.finalize_forward_batch(out, fb)
    assert got.next_token_logits.shape[0] == 112 and got.hidden_states.shape[0] == 128
    restore.assert_called_once_with(fb, 40, 3)
    # The logits row index alone (a dataclass field) also keeps the rows.
    fb.mixed_verify, fb.logit_rows = None, torch.zeros(112, dtype=torch.int64)
    with patch.object(shim, "_restore_forward_batch_inputs"):
        assert policy.finalize_forward_batch(out, fb).next_token_logits.shape[0] == 112
    # Not fused (no plan, no index): the normal path runs.
    fb.logit_rows = None
    with patch.object(
        TpuSpecDecodeCompilePolicyMixin.__mro__[1],
        "finalize_forward_batch",
        return_value="trimmed",
    ):
        assert policy.finalize_forward_batch(out, fb) == "trimmed"
    stack_patch.stop()


def test_eager_runner_load_batch_carries_the_mixed_attributes():
    """EagerRunner.load_batch hands the model a dataclasses.replace copy; the
    plan and host Engram rows must survive it (smoke j-f23d753e: the shim padded
    the fused forward as an extend)."""
    src = SimpleNamespace(mixed_verify="plan", mixed_host_rows="rows")
    copy = SimpleNamespace()
    assert mixed_step.carry_mixed_fields(copy, object(), src) is copy
    assert copy.mixed_verify == "plan" and copy.mixed_host_rows == "rows"
    same = SimpleNamespace()
    assert mixed_step.carry_mixed_fields(same, object(), same) is same


def test_eager_runner_keeps_a_fused_mixed_forward_mixed():
    from contextlib import nullcontext

    from sglang.srt.model_executor.forward_mode import ForwardMode

    seen = []
    runner = SimpleNamespace(
        model_context=nullcontext,
        _execute_extend=lambda fb, pp: seen.append((fb.forward_mode, pp)) or "out",
    )
    original = lambda r, fb, *a, **k: "original"  # noqa: E731
    fb = SimpleNamespace(forward_mode=ForwardMode.MIXED, mixed_verify="plan")
    with patch.object(mixed_step, "fused_mixed_forward", return_value=True):
        assert mixed_step.execute_fused_mixed(original, runner, fb, "pp") == "out"
        assert seen == [(ForwardMode.MIXED, "pp")]
        # A MIXED batch without the plan (legacy mixed chunk) keeps core's path.
        legacy = SimpleNamespace(forward_mode=ForwardMode.MIXED, mixed_verify=None)
        assert mixed_step.execute_fused_mixed(original, runner, legacy) == "original"
    with patch.object(mixed_step, "fused_mixed_forward", return_value=False):
        assert mixed_step.execute_fused_mixed(original, runner, fb) == "original"


def test_fine_bucket_sizes_the_window_workspace():
    """With the fine bucket a fused step may hold 128 rows more than a chunk:
    the request-window workspace and its KV budget count them."""
    from sglang_tpu.envs import envs as tpu_envs
    from sglang_tpu.models.deepseek_v41.request_window import (
        _workspace_pages,
        window_chunk_tokens,
    )

    with patch.object(mixed_step, "fused_mixed_forward", return_value=True):
        with tpu_envs.SGLANG_TPU_MIXED_VERIFY_FINE_BUCKET.override(True):
            assert window_chunk_tokens(1024) == 1152
            assert _workspace_pages(num_slots=17, chunk_tokens=1152) == 86
        assert window_chunk_tokens(1024) == 1024  # fine bucket off
    with tpu_envs.SGLANG_TPU_MIXED_VERIFY_FINE_BUCKET.override(True):
        with patch.object(mixed_step, "fused_mixed_forward", return_value=False):
            assert window_chunk_tokens(1024) == 1024  # fused off
    assert window_chunk_tokens(None) == 0


def test_draft_append_warmup_skips_fine_buckets_when_fused():
    from sglang_tpu.experimental.compilation import block_compile_policy as bcp

    policy = object.__new__(bcp.TpuBlockSpecDecodeCompilePolicyMixin)
    policy.t_bucketer = SimpleNamespace(buckets=[512, 1024, 1152, 1280, 2048])
    with patch.object(bcp, "_fused_mixed_steps", return_value=True):
        assert policy._draft_append_token_buckets() == [512, 1024, 1280, 2048]
    with patch.object(bcp, "_fused_mixed_steps", return_value=False):
        assert policy._draft_append_token_buckets() == [512, 1024, 1152, 1280, 2048]


def test_load_batch_copy_takes_over_the_verify_token_splice():
    """The fused spec input must splice into the copy EagerRunner.load_batch hands
    the policy and the model, not the original batch."""
    spec = SimpleNamespace()
    src = SimpleNamespace(mixed_verify="plan", mixed_host_rows=None, spec_info=spec)
    spec._batch_ref = src
    copy = SimpleNamespace(spec_info=spec)
    mixed_step.carry_mixed_fields(copy, object(), src)
    assert spec._batch_ref is copy
    # A spec input bound elsewhere (or already used) is left alone.
    other = SimpleNamespace(spec_info=SimpleNamespace(_batch_ref=None))
    mixed_step.carry_mixed_fields(other, object(), src)
    assert other.spec_info._batch_ref is None


def test_load_batch_copy_takes_over_the_host_mirror_record():
    import torch

    from sglang_tpu.compilation import host_input_mirror

    class Batch:  # weak-referenceable
        pass

    src, copy = Batch(), Batch()
    host, dev = torch.arange(3), torch.arange(3) + 10
    src.positions = copy.positions = dev
    src.spec_info = copy.spec_info = None
    host_input_mirror.clear()
    host_input_mirror.record_upload(src, "positions", host, dev)
    # Not a mixed batch: the runner's behavior is untouched (no-op with mixed off).
    src.mixed_verify = None
    assert mixed_step.carry_mixed_fields(copy, object(), src) is copy
    assert host_input_mirror.host_copy(copy, "positions") is None
    assert host_input_mirror.host_copy(src, "positions") is host
    src.mixed_verify = "plan"
    mixed_step.carry_mixed_fields(copy, object(), src)
    assert host_input_mirror.host_copy(copy, "positions") is host
    host_input_mirror.clear()


def test_load_batch_carry_leaves_other_batches_alone():
    spec = SimpleNamespace()
    src = SimpleNamespace(spec_info=spec, mixed_host_rows="rows")
    spec._batch_ref = src
    copy = SimpleNamespace(spec_info=spec)
    assert mixed_step.carry_mixed_fields(copy, object(), src) is copy
    assert spec._batch_ref is src and not hasattr(copy, "mixed_host_rows")


def test_every_non_field_forward_batch_attribute_is_carried():
    """Non-field attributes the DSpark worker path sets on a ForwardBatch are lost
    in EagerRunner.load_batch's dataclasses.replace copy unless carried; a new one
    must be added to mixed_step._MIXED_ATTRS (or handled in carry_mixed_fields)."""
    import dataclasses
    import pathlib
    import re

    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

    fields = {f.name for f in dataclasses.fields(ForwardBatch)} | set(dir(ForwardBatch))
    root = pathlib.Path(mixed_step.__file__).parent
    files = [
        "base_spec_worker.py",
        "block_spec_worker.py",
        "mixed_step.py",
        "host_verify_rows.py",
        "draft_forward.py",
    ]
    pat = re.compile(
        r"\b(?:forward_batch|fb|verify_forward_batch)\.([A-Za-z_]\w*)\s*=(?!=)"
    )
    set_attrs = set()
    for name in files:
        path = root / name
        if path.exists():
            set_attrs |= {a for a in pat.findall(path.read_text()) if a not in fields}
    assert set_attrs <= set(mixed_step._MIXED_ATTRS), sorted(
        set_attrs - set(mixed_step._MIXED_ATTRS)
    )
