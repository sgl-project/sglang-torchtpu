"""Explicit host-input placement; meta checks device boundaries without a TPU."""

import sys
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.managers.overlap_utils import resolve_forward_inputs
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.platforms.interface import SRTPlatform
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu.allocator_tpu import TpuPagedTokenToKVPoolAllocator
from sglang_tpu.compilation.bucketer import Bucketer
from sglang_tpu.compilation.decoder_compile_policy import (
    DecoderTpuCompilePolicy,
)
from sglang_tpu.tpu_platform import TpuSRTPlatform

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@contextmanager
def backend(tpu):
    with ExitStack() as stack:
        for module in (
            "managers.overlap_utils",
            "mem_cache.allocation",
            "model_executor.forward_batch_info",
        ):
            stack.enter_context(patch(f"sglang.srt.{module}._is_tpu", tpu))
        stack.enter_context(
            patch(
                "sglang.srt.managers.schedule_batch.current_platform",
                TpuSRTPlatform() if tpu else SRTPlatform(),
            )
        )
        yield


@pytest.fixture(autouse=True)
def config():
    with (
        get_context().override_server_args(
            enable_torch_compile=True, mamba_radix_cache_strategy="no_buffer"
        ),
        get_parallel().override(moe_ep_size=1),
        envs.SGLANG_TPU_HOST_PADDING_NO_D2H.override(True),
        backend(True),
    ):
        yield


def allocator(device="cpu"):
    return TpuPagedTokenToKVPoolAllocator(
        size=32,
        page_size=4,
        dtype=torch.bfloat16,
        device=device,
        kvcache=None,
        need_sort=False,
    )


def make_batch(mode, lengths, prefixes, mm=None, input_device="cpu"):
    bs = len(lengths)
    n = bs if mode.is_decode() else sum(lengths)
    batch = ScheduleBatch(
        reqs=[
            SimpleNamespace(rid=str(i), lora_id=None, token_type_ids=None, session=None)
            for i in range(bs)
        ],
        device="meta",
        batch_input_device=input_device,
        forward_mode=mode,
        spec_algorithm=SpeculativeAlgorithm.NONE,
        input_ids=torch.arange(n),
        seq_lens=torch.tensor([p + n for p, n in zip(prefixes, lengths)]),
        seq_lens_cpu=torch.tensor([p + n for p, n in zip(prefixes, lengths)]),
        orig_seq_lens_host=torch.tensor(
            [p + n for p, n in zip(prefixes, lengths)], dtype=torch.int32
        ),
        orig_seq_lens=torch.tensor(
            [p + n for p, n in zip(prefixes, lengths)], dtype=torch.int32
        ),
        out_cache_loc=torch.arange(n, dtype=torch.int32),
        req_pool_indices=torch.empty(bs, dtype=torch.int64, device="meta"),
        req_pool_indices_cpu=torch.arange(bs),
        extend_lens=lengths,
        prefix_lens=prefixes,
        extend_num_tokens=n,
        extend_logprob_start_lens=[0] * bs,
        multimodal_inputs=mm or [None] * bs,
        req_to_token_pool=SimpleNamespace(
            req_to_token=torch.zeros(8, 64, dtype=torch.int32)
        ),
        token_to_kv_pool_allocator=allocator(),
        model_config=SimpleNamespace(
            is_encoder_decoder=False, use_ngram_embedding=False
        ),
    )
    runner = SimpleNamespace(
        device="meta",
        is_draft_worker=False,
        kv_index_translator=SimpleNamespace(rebind_write_loc=lambda _: None),
        ngram_embedding_manager=SimpleNamespace(enabled=False),
        model_config=SimpleNamespace(
            model_is_mrope=True, requires_mm_token_modalities=False
        ),
        lora_manager=None,
        attn_dcp_size=1,
        ps=SimpleNamespace(attn_dcp_size=1),
        prefill_attention_backend_str="torch_native",
    )
    return batch, runner


def build(batch, runner):
    with patch(
        "sglang.srt.model_executor.forward_batch_info.support_triton",
        return_value=False,
    ):
        return ForwardBatch.init_new(
            batch,
            runner,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            return_hidden_states_before_norm=False,
        )


def prepare(fb, runner):
    policy = object.__new__(DecoderTpuCompilePolicy)
    policy.model_runner = runner
    policy.t_bucketer = Bucketer("t", [8, 16])
    policy.b_bucketer = Bucketer("b", [1, 2, 4])
    policy.extend_bs = 4
    policy.prepare_forward_batch(fb)


@pytest.mark.parametrize(
    "mode,lengths",
    [
        (ForwardMode.EXTEND, [3, 2]),
        (ForwardMode.EXTEND, [4, 4]),
        (ForwardMode.DECODE, [4, 5, 6]),
    ],
)
def test_host_inputs_pad_before_transfer_and_leave_scheduler_ownership(mode, lengths):
    batch, runner = make_batch(mode, lengths, [0] * len(lengths))
    fb = build(batch, runner)
    assert fb.req_pool_indices is batch.req_pool_indices_cpu
    assert batch.req_pool_indices.device.type == "meta"
    assert fb.positions.device.type == fb.mrope_positions.device.type == "cpu"
    if mode.is_extend():
        assert (
            fb.extend_seq_lens.device.type == fb.extend_start_loc.device.type == "cpu"
        )
    original = {
        name: getattr(batch, name)
        for name in ("input_ids", "seq_lens", "out_cache_loc")
    }
    values = {name: t.clone() for name, t in original.items()}
    # The real eager runner copies the dataclass before padding.
    prepared = replace(fb)
    prepare(prepared, runner)
    expected_tokens = 4 if mode.is_decode() else 8
    assert prepared.input_ids.shape == (expected_tokens,)
    assert prepared.seq_lens.shape == (4,)
    for name in (*original, "req_pool_indices", "positions", "mrope_positions"):
        assert getattr(prepared, name).device.type == "meta"
    assert prepared.seq_lens_cpu.device.type == "cpu"
    for name, t in original.items():
        assert getattr(batch, name) is t
        torch.testing.assert_close(t, values[name])


def test_mixed_image_text_mrope_stays_on_host_until_padding():
    positions = torch.arange(30).reshape(3, 10)
    mm = SimpleNamespace(
        mrope_positions=positions, mrope_position_delta=torch.tensor([[7]])
    )
    batch, runner = make_batch(ForwardMode.EXTEND, [3, 2], [2, 1], [mm, None])
    fb = build(batch, runner)
    torch.testing.assert_close(
        fb.mrope_positions,
        torch.cat((positions[:, 2:5], torch.tensor([[1, 2]]).repeat(3, 1)), dim=1),
    )
    prepare(fb, runner)
    assert fb.mrope_positions.shape == (3, 8)
    assert fb.mrope_positions.device.type == "meta"


def test_moe_counts_keep_device_contract_during_host_staging():
    batch, runner = make_batch(ForwardMode.DECODE, [3, 4], [0, 0])
    with patch(
        "sglang.srt.model_executor.forward_batch_info.enable_num_token_non_padded",
        return_value=True,
    ):
        fb = build(batch, runner)
    fb.set_local_num_token_non_padded(sharded=False)
    prepare(fb, runner)
    assert fb.global_num_token_non_padded.device.type == "meta"
    assert fb.num_token_non_padded.device.type == "meta"


def test_forward_construction_default_keeps_device_metadata():
    batch, runner = make_batch(ForwardMode.EXTEND, [3, 2], [0, 0], input_device=None)
    # Meta cannot execute the variable-length position constructor; keep its
    # device/shape contract while checking the default placement decisions.
    with patch(
        "sglang.srt.model_executor.forward_batch_info.compute_position",
        return_value=(torch.empty(5, device="meta"), torch.empty(2, device="meta")),
    ):
        fb = build(batch, runner)
    assert fb.extend_seq_lens.device.type == "meta"
    assert fb.req_pool_indices is batch.req_pool_indices


@pytest.mark.parametrize("staged", [False, True])
def test_allocator_output_device_is_opt_in(staged):
    pool = allocator("meta")
    zero = torch.zeros(2, dtype=torch.int64)
    lengths = torch.tensor([3, 1])
    options = {"output_device": "cpu"} if staged else {}
    locs = pool.alloc_extend(zero, zero, lengths, lengths, zero - 1, 4, **options)
    assert locs.device.type == ("cpu" if staged else "meta")
    if staged:
        assert locs.tolist() == [4, 5, 6, 8]
    # Last locations are from the existing host request table in both cases.
    locs = pool.alloc_decode(lengths + 1, lengths + 1, torch.tensor([6, 8]), **options)
    assert locs.device.type == ("cpu" if staged else "meta")
    if staged:
        assert locs.tolist() == [7, 9]
    assert pool.free_pages.tolist() == list(range(3, 9))


def test_platform_default_and_tpu_fallbacks():
    batch, _ = make_batch(ForwardMode.EXTEND, [2], [0])
    assert SRTPlatform().get_batch_input_device(batch) is None
    platform = TpuSRTPlatform()
    assert platform.get_batch_input_device(batch) == "cpu"
    # The D2H host-pad flag alone must not enable host staging.
    with (
        envs.SGLANG_TPU_HOST_PADDING_NO_D2H.override(False),
        envs.SGLANG_TPU_HOST_PADDING.override(True),
    ):
        assert platform.get_batch_input_device(batch) is None
    for name, value in (
        ("spec_algorithm", SpeculativeAlgorithm.EAGLE),
        ("dllm_config", object()),
        ("hisparse_coordinator", object()),
    ):
        saved = getattr(batch, name)
        setattr(batch, name, value)
        assert platform.get_batch_input_device(batch) == "meta"
        setattr(batch, name, saved)
    with get_context().override_server_args(enable_torch_compile=False):
        assert platform.get_batch_input_device(batch) == "meta"
    for name in ("beam_group", "positional_embed_overrides"):
        setattr(batch.reqs[0], name, object())
        assert platform.get_batch_input_device(batch) == "meta"
        setattr(batch.reqs[0], name, None)
    batch.req_to_token_pool.req_to_token = torch.empty(8, 64, device="meta")
    assert platform.get_batch_input_device(batch) == "meta"


def test_merge_fallback_never_reads_device_metadata_back_to_host():
    host, _ = make_batch(ForwardMode.DECODE, [3, 4], [0, 0])
    device, _ = make_batch(ForwardMode.DECODE, [3], [0], input_device=None)
    for name in ("input_ids", "seq_lens", "orig_seq_lens", "out_cache_loc"):
        setattr(device, name, getattr(device, name).to("meta"))
    host._align_input_devices(device)
    assert host.batch_input_device == device.batch_input_device == "meta"
    for name in ("input_ids", "seq_lens", "orig_seq_lens", "out_cache_loc"):
        assert getattr(host, name).device.type == "meta"


def test_relay_keeps_decode_and_mixed_tokens_on_device():
    batch, _ = make_batch(ForwardMode.EXTEND, [3], [0])
    batch.prefill_input_ids_cpu = torch.tensor([1, 2, 3])
    batch.input_ids = None
    future = SimpleNamespace(
        spec_algo=SpeculativeAlgorithm.NONE,
        output_tokens_buf=torch.empty(8, dtype=torch.int64, device="meta"),
    )
    resolve_forward_inputs(batch, future)
    assert batch.input_ids.device.type == "cpu"
    batch.prefill_input_ids_cpu = torch.tensor([1, 2, 3])
    batch.mix_running_indices = torch.empty(1, dtype=torch.int64, device="meta")
    resolve_forward_inputs(batch, future)
    assert batch.input_ids.device.type == "cpu" and batch.input_ids.tolist() == [
        1,
        2,
        3,
        0,
    ]
    assert batch.mixed_input_tokens.device.type == "meta"
    batch.input_ids = None
    resolve_forward_inputs(batch, future)
    assert batch.input_ids.device.type == "meta"


def test_filter_keeps_host_metadata_and_device_relay_keys_in_request_order():
    batch, _ = make_batch(ForwardMode.DECODE, [3, 4, 5], [0, 0, 0])
    for req in batch.reqs:
        req.return_logprob = False
        req.grammar = None
        req.return_hidden_states_mode = CaptureHiddenMode.NULL
    calls = []
    batch.sampling_info = SimpleNamespace(
        filter_batch=lambda host, device: calls.append((host, device))
    )
    original = batch.seq_lens
    batch.filter_batch(keep_indices=[2, 0])
    assert batch.seq_lens.tolist() == [5, 3]
    assert original.tolist() == [3, 4, 5]
    assert batch.input_ids.tolist() == [2, 0]
    assert batch.req_pool_indices_cpu.tolist() == [2, 0]
    assert batch.orig_seq_lens_host.tolist() == [5, 3]
    assert batch.req_pool_indices.device.type == "meta"
    assert calls[0][1].device.type == "meta"


def test_seeded_sampler_gets_local_device_positions():
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    from sglang.srt.layers.sampler import Sampler
    from sglang_tpu.sampler import TpuSampler

    sampler = TpuSampler.__new__(TpuSampler)
    output = LogitsProcessorOutput(next_token_logits=torch.empty(2, 16, device="meta"))
    info = SimpleNamespace(
        sampling_seed=torch.empty(2, device="meta"), is_all_greedy=False
    )
    positions = torch.tensor([3, 4])
    seen = []

    def sample(
        self,
        logits_output,
        sampling_info,
        return_logprob,
        top_logprobs_nums,
        token_ids_logprobs,
        local_positions,
    ):
        seen.append(local_positions)
        return torch.empty(2, device="meta", dtype=torch.int64)

    with patch.object(Sampler, "forward", sample):
        sampler.forward(output, info, False, [0, 0], [[], []], positions)
    assert seen[0].device.type == "meta"
    assert positions.tolist() == [3, 4]


@pytest.mark.parametrize(
    "device,input_device,expected",
    [("cpu", "cpu", {}), ("meta", None, {}), ("meta", "cpu", {"output_device": "cpu"})],
)
def test_shared_allocation_only_passes_output_device_for_staged_accelerator(
    device, input_device, expected
):
    from unittest.mock import MagicMock

    from sglang.srt.mem_cache import allocation

    pool = SimpleNamespace(
        page_size=4,
        forward_write_target=nullcontext,
        alloc_extend=MagicMock(return_value=torch.arange(2)),
        alloc_decode=MagicMock(return_value=torch.arange(2)),
    )
    tree = SimpleNamespace(token_to_kv_pool_allocator=pool)
    batch = SimpleNamespace(device=device, batch_input_device=input_device)
    lengths = torch.tensor([1, 1])
    with (
        patch.object(allocation, "evict_from_tree_cache"),
        patch.object(allocation, "page_interleave_shard_size", return_value=1),
    ):
        allocation.alloc_paged_token_slots_extend(
            tree,
            lengths - 1,
            lengths - 1,
            lengths,
            lengths,
            lengths - 2,
            2,
            batch=batch,
        )
        allocation.alloc_paged_token_slots_decode(
            tree, lengths, lengths, lengths - 2, batch=batch
        )
    assert pool.alloc_extend.call_args.kwargs == expected
    assert pool.alloc_decode.call_args.kwargs == expected


@pytest.mark.parametrize("dynamic_index", [False, True])
def test_future_map_store_materializes_host_values_on_relay_device(dynamic_index):
    from sglang.srt.managers.overlap_utils import FutureMap

    relay = FutureMap.__new__(FutureMap)
    relay._use_dynamic_index = dynamic_index
    source = torch.tensor([3, 7], dtype=torch.int64)
    buffer = torch.empty(4, dtype=torch.int32, device="meta")
    indices = torch.tensor([0, 2], device="meta")
    seen = []
    operation = "index_copy_" if dynamic_index else "__setitem__"
    original = getattr(torch.Tensor, operation)

    def observe(self, *args):
        if self is buffer:
            seen.append(args[-1])
        return original(self, *args)

    with patch.object(torch.Tensor, operation, observe):
        relay._store(buffer, indices, source)
    assert len(seen) == 1
    assert seen[0].device == buffer.device
    assert seen[0].dtype == buffer.dtype
    assert source.tolist() == [3, 7]
    assert source.dtype == torch.int64


def test_non_tpu_cpu_forward_preserves_request_indices_identity():
    batch, runner = make_batch(ForwardMode.EXTEND, [3], [0], input_device=None)
    runner.device = batch.device = "cpu"
    batch.req_pool_indices = torch.tensor([2])
    batch.req_pool_indices_cpu = torch.tensor([2])
    with backend(False):
        fb = build(batch, runner)
    assert fb.req_pool_indices is batch.req_pool_indices


def swa_allocator(*, ring=False):
    from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator

    pool = object.__new__(SWATokenToKVPoolAllocator)
    pool.page_size = 4
    pool.device = "meta"
    pool.full_attn_allocator = allocator("meta")
    pool.swa_attn_allocator = allocator("meta")
    pool._swa_req_ring = ring
    pool.full_to_swa_index_mapping = torch.zeros(37, dtype=torch.int64)
    pool.full_to_swa_index_mapping[-1] = -1
    pool._kvcache = SimpleNamespace(
        full_to_swa_index_mapping=pool.full_to_swa_index_mapping,
        translate_loc_from_full_to_swa=lambda idx: pool.full_to_swa_index_mapping[idx],
    )
    return pool


@pytest.mark.parametrize("ring", [False, True])
def test_swa_host_allocation_preserves_mapping_and_page_accounting(ring):
    pool = swa_allocator(ring=ring)
    zero = torch.zeros(2, dtype=torch.int64)
    lengths = torch.tensor([3, 4])
    # Shift SWA pages so the test detects accidental full-pool index reuse.
    pool.swa_attn_allocator.alloc(4)
    with patch("sglang.srt.mem_cache.allocator.swa._is_tpu", True):
        locs = pool.alloc_extend(
            zero, zero, lengths, lengths, zero - 1, 7, output_device="cpu"
        )
        assert locs.tolist() == [4, 5, 6, 8, 9, 10, 11]
        locs = pool.alloc_decode(
            lengths + 1, lengths + 1, torch.tensor([6, 11]), output_device="cpu"
        )
    assert locs.tolist() == [7, 12]
    assert pool.full_attn_allocator.free_pages.tolist() == list(range(4, 9))
    if ring:
        assert pool.full_to_swa_index_mapping[:-1].count_nonzero() == 0
        assert pool.swa_attn_allocator.free_pages.tolist() == list(range(2, 9))
    else:
        assert pool.full_to_swa_index_mapping[locs.long()].tolist() == [11, 16]
        assert pool.swa_attn_allocator.free_pages.tolist() == list(range(5, 9))


def test_swa_placement_excludes_subclasses_and_device_mapping():
    from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator

    batch, _ = make_batch(ForwardMode.EXTEND, [2], [0])
    pool = swa_allocator()
    batch.token_to_kv_pool_allocator = pool
    platform = TpuSRTPlatform()
    assert platform.get_batch_input_device(batch) == "cpu"
    pool.full_to_swa_index_mapping = pool.full_to_swa_index_mapping.to("meta")
    assert platform.get_batch_input_device(batch) == "meta"

    class DifferentAllocator(SWATokenToKVPoolAllocator):
        pass

    batch.token_to_kv_pool_allocator = object.__new__(DifferentAllocator)
    batch.token_to_kv_pool_allocator.page_size = 4
    assert platform.get_batch_input_device(batch) == "meta"


@pytest.mark.parametrize("tpu", [False, True])
def test_pd_prebuilt_initializes_placement_for_first_decode(tpu):
    from sglang.srt.disaggregation import decode_schedule_batch_mixin as prebuilt

    batch, _ = make_batch(ForwardMode.PREBUILT, [3], [0], input_device=None)
    batch.model_config.vocab_size = 32
    req = batch.reqs[0]
    req.kv = SimpleNamespace(req_pool_idx=1)
    req.extend_range = SimpleNamespace(length=3)
    req.prefix_indices = []
    req.origin_input_ids = [1, 2, 3]
    req.output_ids = []
    req.retracted_stain = True
    req.multimodal_inputs = None
    with (
        patch.object(prebuilt, "_is_tpu", tpu),
        patch(
            "sglang.srt.platforms.current_platform",
            TpuSRTPlatform() if tpu else SRTPlatform(),
        ),
        patch.object(prebuilt.SamplingBatchInfo, "from_schedule_batch"),
    ):
        batch.prepare_for_prebuilt()
    assert batch.batch_input_device == ("cpu" if tpu else None)
    for name in ("seq_lens", "orig_seq_lens"):
        assert getattr(batch, name).device.type == ("cpu" if tpu else "meta")
    assert batch.seq_lens_cpu.tolist() == [3]
    assert batch.req_pool_indices.device.type == "meta"


@pytest.mark.parametrize("prompt_length", [5, 9])
def test_mixed_prompt_is_bucketed_before_injecting_device_tail(prompt_length):
    batch, runner = make_batch(ForwardMode.MIXED, [prompt_length, 1, 1], [0, 8, 12])
    batch.device = runner.device = "cpu"
    batch.prefill_input_ids_cpu = torch.arange(prompt_length)
    batch.mix_running_indices = torch.tensor([3, 9])
    relay = SimpleNamespace(output_tokens_buf=torch.arange(16) + 100)
    resolve_forward_inputs(batch, relay)
    assert batch.input_ids.tolist() == list(range(prompt_length)) + [0, 0]
    assert batch.mixed_input_tokens.tolist() == [103, 109]
    fb = build(batch, runner)
    prepare(fb, runner)
    assert fb.input_ids.tolist()[: prompt_length + 2] == list(range(prompt_length)) + [
        103,
        109,
    ]
    assert fb.input_ids.tolist()[prompt_length + 2 :] == [0] * (
        fb.input_ids.numel() - prompt_length - 2
    )
    assert fb.mixed_input_tokens is None


def test_decode_placement_recovers_without_reading_device_lengths():
    batch, _ = make_batch(ForwardMode.DECODE, [8, 12], [0, 0], input_device="meta")
    batch.seq_lens = torch.empty(2, dtype=torch.int64, device="meta")
    batch.orig_seq_lens = torch.empty(2, dtype=torch.int32, device="meta")
    batch.orig_seq_lens_host = torch.tensor([18, 22], dtype=torch.int32)
    # This TPU-only method is absent from the CPU platform used by the test.
    with patch(
        "sglang.srt.managers.schedule_batch.current_platform.get_batch_input_device",
        return_value="cpu",
        create=True,
    ):
        batch._refresh_input_device()
    assert batch.batch_input_device == "cpu"
    assert batch.seq_lens.tolist() == [8, 12]
    assert batch.orig_seq_lens.tolist() == [18, 22]


def test_decode_advances_original_length_mirror_after_recovery():
    batch, _ = make_batch(ForwardMode.DECODE, [8, 12], [0, 0])
    batch.orig_seq_lens_host = torch.tensor([18, 22], dtype=torch.int32)
    batch.orig_seq_lens = batch.orig_seq_lens_host
    batch.sampling_info = SimpleNamespace(
        penalizer_orchestrator=SimpleNamespace(is_required=False)
    )
    for req in batch.reqs:
        req.decode_batch_idx = 0
    with (
        patch(
            "sglang.srt.managers.schedule_batch.alloc_for_decode",
            return_value=torch.arange(2),
        ),
        patch("sglang.srt.managers.schedule_batch.strip_beam_tail"),
        patch("sglang.srt.managers.schedule_batch.append_beam_tail"),
        patch(
            "sglang.srt.managers.schedule_batch.current_platform.get_batch_input_device",
            return_value="cpu",
            create=True,
        ),
    ):
        batch.prepare_for_decode()
    assert batch.seq_lens.tolist() == [9, 13]
    assert batch.orig_seq_lens.tolist() == [19, 23]
    assert batch.orig_seq_lens_host.tolist() == [19, 23]


def test_decode_allocator_uses_host_request_indices_without_device_read():
    from sglang.srt.mem_cache import allocation

    batch, _ = make_batch(ForwardMode.DECODE, [8, 12], [0, 0])
    batch.tree_cache = SimpleNamespace(page_size=4)
    batch.req_to_token_pool.write = lambda *args: None
    batch.req_to_token_pool.alloc_aux_to_lengths = lambda **kwargs: None
    for req in batch.reqs:
        req.kv = SimpleNamespace(kv_allocated_len=0, kv_committed_len=0)
    batch.req_to_token_pool.req_to_token[0, 7] = 23
    batch.req_to_token_pool.req_to_token[1, 11] = 29
    batch.maybe_evict_swa = lambda: None
    with patch.object(
        allocation,
        "alloc_paged_token_slots_decode",
        return_value=torch.tensor([24, 30]),
    ) as alloc:
        result = allocation.alloc_for_decode(batch, 1)
    assert result.tolist() == [24, 30]
    assert alloc.call_args.kwargs["last_loc"].tolist() == [23, 29]


def test_eagle3_prefill_stages_on_host_but_decode_and_multimodal_do_not():
    platform = TpuSRTPlatform()
    batch, _ = make_batch(ForwardMode.EXTEND, [2], [0])
    batch.spec_algorithm = SpeculativeAlgorithm.EAGLE3
    assert platform.get_batch_input_device(batch) == "cpu"
    batch.forward_mode = ForwardMode.MIXED
    assert platform.get_batch_input_device(batch) == "meta"
    batch.forward_mode = ForwardMode.EXTEND
    batch.model_config.is_multimodal = True
    assert platform.get_batch_input_device(batch) == "meta"
    decode, _ = make_batch(ForwardMode.DECODE, [3], [0])
    decode.spec_algorithm = SpeculativeAlgorithm.EAGLE3
    assert platform.get_batch_input_device(decode) == "meta"


def test_eagle3_prefill_staging_follows_the_feature_flag():
    from sglang_tpu.envs import envs as tpu_envs

    platform = TpuSRTPlatform()
    batch, _ = make_batch(ForwardMode.EXTEND, [2], [0])
    batch.spec_algorithm = SpeculativeAlgorithm.EAGLE3
    with tpu_envs.SGLANG_TPU_ENABLE_SPEC_BUCKETED_METADATA.override(False):
        assert platform.get_batch_input_device(batch) == "meta"
    assert platform.get_batch_input_device(batch) == "cpu"


def test_spec_decode_returns_a_host_staged_prefill_to_the_device():
    batch, _ = make_batch(ForwardMode.EXTEND, [3, 4], [0, 0])
    batch._move_inputs_to_device()
    assert batch.batch_input_device == "meta"
    for name in ("input_ids", "seq_lens", "orig_seq_lens", "out_cache_loc"):
        assert getattr(batch, name).device.type == "meta"
    # Already on the device: nothing moves.
    batch.input_ids = torch.arange(3)
    batch._move_inputs_to_device()
    assert batch.input_ids.device.type == "cpu"


def test_draft_prefill_tail_tokens_land_at_explicit_slots_after_bucketing():
    batch, runner = make_batch(ForwardMode.EXTEND, [3, 2], [0, 0])
    batch.device = runner.device = "cpu"
    fb = build(batch, runner)
    # Host-rotated ids with each request's last slot left for the device token.
    fb.input_ids = torch.tensor([11, 12, 0, 21, 0])
    fb.mixed_input_tokens = torch.tensor([13, 22])
    fb.mixed_input_slots = torch.tensor([2, 4])
    prepare(fb, runner)
    assert fb.input_ids.tolist()[:5] == [11, 12, 13, 21, 22]
    assert fb.input_ids.tolist()[5:] == [0] * (fb.input_ids.numel() - 5)
    assert fb.mixed_input_tokens is None and fb.mixed_input_slots is None
    assert fb._tpu_host_staged


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
