# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Wave-7 TPU scheduler & host optimizations (default-off)."""

from __future__ import annotations

import os
from array import array
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.disaggregation.mode import DisaggregationMode
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.model_executor.forward_mode import CaptureHiddenMode, ForwardMode
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang_tpu.allocator_tpu import TpuPagedTokenToKVPoolAllocator
from sglang_tpu.envs import envs as tpu_envs
from sglang_tpu.host_padding_relay import HostPaddingFutureMap


@contextmanager
def _temp_env(**kwargs: str):
    old = {k: os.environ.get(k) for k in kwargs}
    try:
        for k, v in kwargs.items():
            os.environ[k] = v
        yield
    finally:
        for k, prev in old.items():
            if prev is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = prev


class _FakeBucketer:
    buckets = [32, 64]
    max_val = 64

    @staticmethod
    def next_bucket(n: int) -> int:
        for b in (32, 64):
            if n <= b:
                return b
        return max(64, n)


def _make_fake_req(
    rid: str,
    input_ids: list[int],
    *,
    prefix_len: int = 0,
    extend_len: int = 1,
    finished: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        rid=rid,
        origin_input_ids=array("q", input_ids),
        output_ids=array("q"),
        prefix_indices=torch.zeros(prefix_len, dtype=torch.int32),
        num_matched_prefix_tokens=prefix_len,
        extend_end=prefix_len + extend_len,
        extend_len=extend_len,
        inflight_middle_chunks=0,
        beam_group=None,
        return_sampling_mask=False,
        token_indices_to_pool=None,
        return_logprob=False,
        return_hidden_states=False,
        grammar=None,
        is_retracted=False,
        require_reasoning=False,
        multimodal_inputs=None,
        session=None,
        mamba_lazy_is_insert=True,
        kv=SimpleNamespace(
            holds_mamba=False,
            holds_kv=True,
            kv_allocated_len=prefix_len + extend_len,
            kv_committed_len=prefix_len + extend_len,
            is_kv_released=False,
            cache_protected_len=0,
        ),
        time_stats=SimpleNamespace(
            set_prefill_finished_time=MagicMock(),
            set_last_decode_finish_time=MagicMock(),
            set_completion_time=MagicMock(),
        ),
        finished=lambda _f=finished: _f,
        update_finish_state=MagicMock(),
    )


def _make_fake_batch(
    reqs: list[SimpleNamespace],
    *,
    forward_mode: ForwardMode,
    pool_indices: list[int],
    seq_lens: list[int],
    out_cache_loc: list[int],
    prefill_tokens: list[int] | None = None,
) -> SimpleNamespace:
    b = SimpleNamespace(
        reqs=list(reqs),
        forward_mode=forward_mode,
        req_pool_indices=torch.tensor(pool_indices, dtype=torch.int64),
        req_pool_indices_cpu=torch.tensor(pool_indices, dtype=torch.int64),
        seq_lens=torch.tensor(seq_lens, dtype=torch.int64),
        seq_lens_cpu=torch.tensor(seq_lens, dtype=torch.int64),
        orig_seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
        out_cache_loc=torch.tensor(out_cache_loc, dtype=torch.int32),
        prefill_input_ids_cpu=(
            torch.tensor(prefill_tokens, dtype=torch.int64)
            if prefill_tokens is not None
            else None
        ),
        input_ids=None,
        is_prefill_only=False,
        return_logprob=False,
        return_hidden_states=False,
        return_hidden_states_mode=CaptureHiddenMode.NULL,
        has_grammar=False,
        input_embeds=None,
        replace_embeds=None,
        multimodal_inputs=[None] * len(reqs),
        chunked_req=None,
        decoding_reqs=None,
        prefill_stats="dummy_stats",
        batch_is_full=False,
        device="cpu",
        spec_algorithm=SpeculativeAlgorithm.NONE,
        spec_info=None,
    )
    b.is_empty = lambda: len(b.reqs) == 0
    b.batch_size = lambda: len(b.reqs)

    def _filter():
        keep = [i for i, r in enumerate(b.reqs) if not r.finished()]
        b.reqs = [b.reqs[i] for i in keep]
        if keep:
            idx = torch.tensor(keep, dtype=torch.int64)
            b.req_pool_indices = b.req_pool_indices[idx]
            b.req_pool_indices_cpu = b.req_pool_indices_cpu[idx]
            b.seq_lens = b.seq_lens[idx]
            b.seq_lens_cpu = b.seq_lens_cpu[idx]
            b.orig_seq_lens = b.orig_seq_lens[idx]
        b.out_cache_loc = None

    def _merge(other):
        b.reqs = b.reqs + other.reqs
        b.req_pool_indices = torch.cat([b.req_pool_indices, other.req_pool_indices])
        b.req_pool_indices_cpu = torch.cat(
            [b.req_pool_indices_cpu, other.req_pool_indices_cpu]
        )
        b.seq_lens = torch.cat([b.seq_lens, other.seq_lens])
        b.seq_lens_cpu = torch.cat([b.seq_lens_cpu, other.seq_lens_cpu])
        b.orig_seq_lens = torch.cat([b.orig_seq_lens, other.orig_seq_lens])
        b.out_cache_loc = None

    b.filter_batch = _filter
    b.merge_batch = _merge
    return b


def test_w7_flags_default_off():
    assert tpu_envs.SGLANG_TPU_EAGER_FILTER_RUNNING_BATCH.get() is False
    assert tpu_envs.SGLANG_TPU_PIGGYBACK_SINGLE_TOKEN_PREFILL.get() is False
    assert tpu_envs.SGLANG_TPU_PACK_PREFILL_TO_BUCKET.get() is False
    assert tpu_envs.SGLANG_TPU_PACK_PREFILL_TARGET_TOKENS.get() == 0
    assert tpu_envs.SGLANG_TPU_PACK_PREFILL_MAX_DELAY_STEPS.get() == 14


def test_piggyback_single_token_prefill_standalone_and_merged():
    for handoff in ("0", "1"):
        with _temp_env(
            SGLANG_TPU_PIGGYBACK_SINGLE_TOKEN_PREFILL="1",
            SGLANG_TPU_SPEC_DIRECT_HANDOFF=handoff,
        ):
            pool = SimpleNamespace(
                req_to_token=torch.zeros((16, 64), dtype=torch.int32),
                req_generation=torch.zeros(16, dtype=torch.int64),
                _tpu_relay_bucketer=_FakeBucketer(),
            )
            fmap = HostPaddingFutureMap("cpu", SpeculativeAlgorithm.NONE, pool)

            alloc = TpuPagedTokenToKVPoolAllocator.__new__(
                TpuPagedTokenToKVPoolAllocator
            )
            alloc.defer_decode_upload = True

            sched = Scheduler.__new__(Scheduler)
            sched.is_generation = True
            sched.spec_algorithm = SpeculativeAlgorithm.NONE
            sched.require_mlp_sync = False
            sched.enable_hisparse = False
            sched.dllm_config = None
            sched.chunked_reqs = []
            sched.max_running_requests = 8
            sched.future_map = fmap
            sched.token_to_kv_pool_allocator = alloc

            def _update_running(batch):
                batch.filter_batch()
                if not batch.is_empty():
                    batch.seq_lens = batch.seq_lens + 1
                    batch.seq_lens_cpu = batch.seq_lens_cpu + 1
                    batch.orig_seq_lens = batch.orig_seq_lens + 1
                    batch.out_cache_loc = torch.tensor(
                        [200 + i for i in range(len(batch.reqs))], dtype=torch.int32
                    )
                    batch.forward_mode = ForwardMode.DECODE
                return batch

            sched.update_running_batch = _update_running

            fake_parallel = SimpleNamespace(pp_size=1, attn_cp_size=1, attn_dcp_size=1)
            fake_exec = SimpleNamespace(
                mamba=SimpleNamespace(enable_mamba_extra_buffer=False)
            )

            with (
                patch("sglang.srt.managers.scheduler.current_platform") as mock_plat,
                patch(
                    "sglang.srt.managers.scheduler.get_parallel",
                    return_value=fake_parallel,
                ),
                patch("sglang.srt.managers.scheduler.get_exec", return_value=fake_exec),
            ):
                mock_plat.is_tpu.return_value = True

                # 1) Standalone 1-token prefill when running_batch is empty.
                empty_running = _make_fake_batch(
                    [],
                    forward_mode=ForwardMode.DECODE,
                    pool_indices=[],
                    seq_lens=[],
                    out_cache_loc=[],
                )
                r0 = _make_fake_req("r0", [42], prefix_len=0, extend_len=1)
                r1 = _make_fake_req("r1", [99], prefix_len=0, extend_len=1)
                nb0 = _make_fake_batch(
                    [r0, r1],
                    forward_mode=ForwardMode.EXTEND,
                    pool_indices=[1, 2],
                    seq_lens=[1, 1],
                    out_cache_loc=[10, 11],
                    prefill_tokens=[42, 99],
                )
                nb_out, rb_out, ret = sched._maybe_piggyback_single_token_prefill(
                    nb0, empty_running
                )
                assert nb_out is None
                assert ret is rb_out
                assert ret.forward_mode == ForwardMode.DECODE
                assert ret.prefill_input_ids_cpu is None
                assert torch.equal(
                    ret.out_cache_loc, torch.tensor([10, 11], dtype=torch.int32)
                )
                assert r0._tpu_piggybacked_prefill is True
                assert r1._tpu_piggybacked_prefill is True
                gathered0 = fmap.gather(ret.req_pool_indices_cpu)
                assert gathered0[:2].tolist() == [42, 99]

                # Simulate previous decode step stashing sampled tokens for [1, 2].
                sampled_padded = torch.zeros(32, dtype=torch.int64)
                sampled_padded[:2] = torch.tensor([501, 502], dtype=torch.int64)
                fmap.stash_bonus_tokens(ret.req_pool_indices_cpu, sampled_padded)

                # 2) Merge a new 1-token prefill r2 into active decode [r0, r1].
                r2 = _make_fake_req("r2", [777], prefix_len=0, extend_len=1)
                nb1 = _make_fake_batch(
                    [r2],
                    forward_mode=ForwardMode.EXTEND,
                    pool_indices=[3],
                    seq_lens=[1],
                    out_cache_loc=[33],
                    prefill_tokens=[777],
                )
                nb_out2, rb_out2, ret2 = sched._maybe_piggyback_single_token_prefill(
                    nb1, rb_out
                )
                assert nb_out2 is None
                assert ret2 is rb_out2
                assert ret2.forward_mode == ForwardMode.DECODE
                assert [r.rid for r in ret2.reqs] == ["r0", "r1", "r2"]
                assert ret2.seq_lens_cpu.tolist() == [2, 2, 1]
                assert ret2.out_cache_loc.tolist() == [200, 201, 33]
                gathered2 = fmap.gather(ret2.req_pool_indices_cpu)
                assert gathered2[:3].tolist() == [501, 502, 777]


def test_piggyback_rejects_multitoken_or_overflow_without_mutating_running():
    with _temp_env(SGLANG_TPU_PIGGYBACK_SINGLE_TOKEN_PREFILL="1"):
        sched = Scheduler.__new__(Scheduler)
        sched.is_generation = True
        sched.spec_algorithm = SpeculativeAlgorithm.NONE
        sched.require_mlp_sync = False
        sched.enable_hisparse = False
        sched.dllm_config = None
        sched.chunked_reqs = []
        sched.max_running_requests = 2
        sched.update_running_batch = MagicMock()

        fake_parallel = SimpleNamespace(pp_size=1, attn_cp_size=1, attn_dcp_size=1)
        fake_exec = SimpleNamespace(
            mamba=SimpleNamespace(enable_mamba_extra_buffer=False)
        )
        with (
            patch("sglang.srt.managers.scheduler.current_platform") as mock_plat,
            patch(
                "sglang.srt.managers.scheduler.get_parallel",
                return_value=fake_parallel,
            ),
            patch("sglang.srt.managers.scheduler.get_exec", return_value=fake_exec),
        ):
            mock_plat.is_tpu.return_value = True

            # Multi-token prefill (extend_len=4) stays EXTEND without touching running_batch.
            r_run = _make_fake_req("run0", [10], prefix_len=0, extend_len=1)
            running = _make_fake_batch(
                [r_run],
                forward_mode=ForwardMode.DECODE,
                pool_indices=[1],
                seq_lens=[5],
                out_cache_loc=[10],
            )
            r_multi = _make_fake_req("m0", [1, 2, 3, 4], prefix_len=0, extend_len=4)
            nb_multi = _make_fake_batch(
                [r_multi],
                forward_mode=ForwardMode.EXTEND,
                pool_indices=[2],
                seq_lens=[4],
                out_cache_loc=[20, 21, 22, 23],
                prefill_tokens=[1, 2, 3, 4],
            )
            nb_o, rb_o, pig = sched._maybe_piggyback_single_token_prefill(
                nb_multi, running
            )
            assert pig is None and nb_o is nb_multi and rb_o is running
            sched.update_running_batch.assert_not_called()

            # Overflow > max_running_requests rejected BEFORE update_running_batch.
            r_run2 = _make_fake_req("run1", [11], prefix_len=0, extend_len=1)
            running_full = _make_fake_batch(
                [r_run, r_run2],
                forward_mode=ForwardMode.DECODE,
                pool_indices=[1, 2],
                seq_lens=[5, 5],
                out_cache_loc=[10, 11],
            )
            r_one = _make_fake_req("o0", [12], prefix_len=0, extend_len=1)
            nb_one = _make_fake_batch(
                [r_one],
                forward_mode=ForwardMode.EXTEND,
                pool_indices=[3],
                seq_lens=[1],
                out_cache_loc=[30],
                prefill_tokens=[12],
            )
            nb_o2, _, pig2 = sched._maybe_piggyback_single_token_prefill(
                nb_one, running_full
            )
            assert pig2 is None and nb_o2 is nb_one
            sched.update_running_batch.assert_not_called()


def test_piggybacked_prefill_sets_ttft_and_checkpoints_once_in_decode_result():
    proc = SchedulerBatchResultProcessor.__new__(SchedulerBatchResultProcessor)
    for k, v in (
        ("still_in_batch", lambda _r: True),
        ("kv_weight_version_tracker", None),
        ("model_config", SimpleNamespace(self_opened_think_start_ids=())),
        ("tree_cache", MagicMock()),
        ("token_to_kv_pool_allocator", MagicMock()),
        (
            "metrics_reporter",
            SimpleNamespace(
                num_generated_tokens=0,
                forward_ct_decode=0,
                report_decode_stats=MagicMock(),
            ),
        ),
        ("output_streamer", SimpleNamespace(stream_output=MagicMock())),
        ("beam_coordinator", SimpleNamespace(commit_decode=lambda _b: set())),
    ):
        object.__setattr__(proc, k, v)

    r_piggy = _make_fake_req("piggy", [10], prefix_len=0, extend_len=1)
    r_piggy._tpu_piggybacked_prefill = True
    r_dec = _make_fake_req("dec", [20], prefix_len=0, extend_len=1)
    r_dec.output_ids.append(21)

    batch = _make_fake_batch(
        [r_dec, r_piggy],
        forward_mode=ForwardMode.DECODE,
        pool_indices=[1, 2],
        seq_lens=[2, 1],
        out_cache_loc=[10, 20],
    )
    res = GenerationBatchResult(
        logits_output=None,
        next_token_ids=torch.tensor([31, 41], dtype=torch.int64),
        can_run_cuda_graph=False,
    )

    with (
        patch.object(SchedulerBatchResultProcessor, "_handle_finish_state_updated_req"),
        patch(
            "sglang.srt.managers.scheduler_components.batch_result_processor.get_observability",
            return_value=SimpleNamespace(enable_metrics=False),
        ),
        patch(
            "sglang.srt.managers.scheduler_components.batch_result_processor.checkpoint_kv_cache"
        ) as mock_ckpt,
    ):
        proc.process_batch_result_decode(batch, res)
        assert r_piggy._tpu_piggybacked_prefill is False
        r_piggy.time_stats.set_prefill_finished_time.assert_called_once()
        r_piggy.time_stats.set_last_decode_finish_time.assert_not_called()
        r_dec.time_stats.set_last_decode_finish_time.assert_called_once()
        mock_ckpt.assert_called_once_with(r_piggy, proc.tree_cache)

        # Next decode step on r_piggy uses normal decode timing and no re-checkpoint.
        res2 = GenerationBatchResult(
            logits_output=None,
            next_token_ids=torch.tensor([32, 42], dtype=torch.int64),
            can_run_cuda_graph=False,
        )
        proc.process_batch_result_decode(batch, res2)
        assert r_piggy.time_stats.set_prefill_finished_time.call_count == 1
        assert r_piggy.time_stats.set_last_decode_finish_time.call_count == 1
        assert mock_ckpt.call_count == 1


def test_pack_prefill_to_bucket_delays_underfilled_and_respects_grace_and_cap():
    with _temp_env(
        SGLANG_TPU_PACK_PREFILL_TO_BUCKET="1",
        SGLANG_TPU_PACK_PREFILL_MAX_DELAY_STEPS="5",
        SGLANG_TPU_PIGGYBACK_SINGLE_TOKEN_PREFILL="1",
    ):
        sched = Scheduler.__new__(Scheduler)
        sched.chunked_prefill_size = 8192
        sched.page_size = 128
        sched.is_mixed_chunk = False
        sched.disaggregation_mode = DisaggregationMode.NULL
        sched.require_mlp_sync = False
        sched.max_running_requests = 64
        sched.chunked_reqs = []
        sched._tpu_prefill_delay_steps = 0
        sched._tpu_prefill_drain_grace_steps = 0

        with patch("sglang.srt.managers.scheduler.current_platform") as mock_plat:
            mock_plat.is_tpu.return_value = True

            # Low concurrency (GSM8K threads=32): zero delay.
            run_32 = SimpleNamespace(
                reqs=[_make_fake_req(f"d{i}", [1]) for i in range(32)],
                is_empty=lambda: False,
                is_prefill_only=False,
            )
            sched.waiting_queue = [_make_fake_req("q0", [1] * 7400)]
            assert sched._should_defer_underfilled_prefill(run_32) is False

            # Saturated c=64 (running=63 + 1 queued 7400-token req < 8064): defers up to cap=5.
            run_63 = SimpleNamespace(
                reqs=[_make_fake_req(f"d{i}", [1]) for i in range(63)],
                is_empty=lambda: False,
                is_prefill_only=False,
            )
            for _ in range(5):
                assert sched._should_defer_underfilled_prefill(run_63) is True
            # Step 6 hits max_delay=5 and flushes!
            assert sched._should_defer_underfilled_prefill(run_63) is False

            # Second request arrives -> pending = 14800 >= 8064 -> fires immediately!
            sched.waiting_queue = [
                _make_fake_req("q0", [1] * 7400),
                _make_fake_req("q1", [1] * 7400),
            ]
            run_62 = SimpleNamespace(
                reqs=[_make_fake_req(f"d{i}", [1]) for i in range(62)],
                is_empty=lambda: False,
                is_prefill_only=False,
            )
            assert sched._should_defer_underfilled_prefill(run_62) is False

            # Drain grace (running=62 + 1 queued = 63 == max_reqs - 1): 3 grace steps, then flush!
            sched.waiting_queue = [_make_fake_req("q_last", [1] * 7400)]
            assert sched._should_defer_underfilled_prefill(run_62) is True
            assert sched._should_defer_underfilled_prefill(run_62) is True
            assert sched._should_defer_underfilled_prefill(run_62) is True
            assert sched._should_defer_underfilled_prefill(run_62) is False

            # 1-token prefill with piggyback=1 never delays!
            sched.waiting_queue = [_make_fake_req("q_1tok", [42])]
            assert sched._should_defer_underfilled_prefill(run_63) is False

            # chunk16k pair-packing: defers 1x 7400 (< 8320), fires immediately on 2x 7400 (14800 >= 8320)!
            sched.chunked_prefill_size = 16384
            sched.waiting_queue = [_make_fake_req("q16_0", [1] * 7400)]
            assert sched._should_defer_underfilled_prefill(run_63) is True
            sched.waiting_queue = [
                _make_fake_req("q16_0", [1] * 7400),
                _make_fake_req("q16_1", [1] * 7400),
            ]
            assert sched._should_defer_underfilled_prefill(run_62) is False


def test_eager_filter_running_batch_and_d1_page_table_version():
    # 1) ReqToTokenPool.write(bump_page_table=False) preserves _tpu_page_table_version
    #    on intra-page decode writes, while SWA clears bump _tpu_swa_map_version.
    pool = ReqToTokenPool(
        size=8, max_context_len=512, device="cpu", enable_memory_saver=False
    )
    v0 = pool.req_to_token._tpu_page_table_version
    pool.write((1, 1), torch.tensor(129, dtype=torch.int32), bump_page_table=False)
    assert pool.req_to_token._tpu_page_table_version == v0
    pool.write((1, 128), torch.tensor(256, dtype=torch.int32), bump_page_table=True)
    assert pool.req_to_token._tpu_page_table_version == v0 + 1

    swa_alloc = SWATokenToKVPoolAllocator.__new__(SWATokenToKVPoolAllocator)
    swa_alloc.full_to_swa_index_mapping = torch.zeros(1025, dtype=torch.int64)
    swa_alloc._swa_req_ring = False
    swa_alloc.page_size = 128
    swa_alloc.free_group = []
    swa_alloc.swa_free_group = []
    swa_alloc.swa_attn_allocator = MagicMock()
    swa_alloc.full_attn_allocator = MagicMock()
    s0 = getattr(swa_alloc.full_to_swa_index_mapping, "_tpu_swa_map_version", 0)
    swa_alloc.clear_full_to_swa_mapping(torch.tensor([128, 129], dtype=torch.int64))
    s1 = swa_alloc.full_to_swa_index_mapping._tpu_swa_map_version
    assert s1 == s0 + 1
    swa_alloc.clear()
    assert swa_alloc.full_to_swa_index_mapping._tpu_swa_map_version == s1 + 1


def test_get_next_batch_to_run_eager_filters_before_prefill_admission():
    sched = Scheduler.__new__(Scheduler)
    sched.scheduler_stage_metrics = None
    sched.process_pending_chunked_abort = lambda: None
    sched._process_hicache_events = lambda: None
    sched.enable_hisparse = False
    sched.dllm_config = None
    sched.chunked_reqs = []
    sched.prefill_decode_interval = 0
    sched._prefill_decode_interval_remaining = 0
    sched._tpu_prefill_delay_steps = 0
    sched._tpu_prefill_drain_grace_steps = 0
    sched.require_mlp_sync = False
    sched.spec_algorithm = SpeculativeAlgorithm.NONE
    sched.enable_fpm = False
    sched.kv_free_fence = None
    sched._negotiate_prefill_delayer_while_deferring = lambda: None
    sched.dp_attn_adapter = SimpleNamespace(
        maybe_prepare_mlp_sync_batch=lambda b, need_sync=False: b,
        maybe_convert_decode_to_extend=lambda b: b,
    )
    sched.ngram_embedding_manager = SimpleNamespace(
        prepare_for_forward=lambda b, chunked_req=None: b
    )
    sched.update_running_batch = lambda b: b

    seen_running_bs: list[int] = []

    def _fake_get_prefill(rb):
        seen_running_bs.append(len(rb.reqs))
        return SimpleNamespace(batch_to_run=None, running_batch=rb)

    sched.get_new_batch_prefill = _fake_get_prefill

    with (
        patch("sglang.srt.managers.scheduler.current_platform") as mock_plat,
        patch("sglang.srt.managers.scheduler.set_schedule_time_batch"),
    ):
        mock_plat.is_tpu.return_value = True

        # Upstream PR #44351 filters finished requests from running_batch before every
        # prefill admission (and KVFreeFence protects in-flight pages on TPU).
        r_live = _make_fake_req("live", [1], finished=False)
        r_done = _make_fake_req("done", [2], finished=True)
        rb0 = _make_fake_batch(
            [r_live, r_done],
            forward_mode=ForwardMode.DECODE,
            pool_indices=[1, 2],
            seq_lens=[5, 5],
            out_cache_loc=[10, 20],
        )
        rb0.batch_is_full = True
        sched.get_next_batch_to_run(rb0, last_batch=None)
        assert seen_running_bs[-1] == 1
        assert rb0.batch_is_full is False

        with _temp_env(SGLANG_TPU_EAGER_FILTER_RUNNING_BATCH="1"):
            rb1 = _make_fake_batch(
                [r_live, r_done],
                forward_mode=ForwardMode.DECODE,
                pool_indices=[1, 2],
                seq_lens=[5, 5],
                out_cache_loc=[10, 20],
            )
            rb1.batch_is_full = True
            sched.get_next_batch_to_run(rb1, last_batch=None)
            assert seen_running_bs[-1] == 1
            assert rb1.batch_is_full is False

            sched.kv_free_fence = object()
            rb2 = _make_fake_batch(
                [r_live, r_done],
                forward_mode=ForwardMode.DECODE,
                pool_indices=[1, 2],
                seq_lens=[5, 5],
                out_cache_loc=[10, 20],
            )
            rb2.batch_is_full = True
            sched.get_next_batch_to_run(rb2, last_batch=None)
            assert seen_running_bs[-1] == 1
            assert rb2.batch_is_full is False
