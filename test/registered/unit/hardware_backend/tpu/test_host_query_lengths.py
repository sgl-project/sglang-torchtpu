import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.speculative.spec_info import SpecInputType
from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu.attention_pallas_rpa_base import (
    BaseTpuPallasRpaAttnBackend,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _bare_backend():
    backend = object.__new__(BaseTpuPallasRpaAttnBackend)
    backend.kernel_spec = SimpleNamespace(uses_draft_extend_v2=True)
    backend._is_hybrid_swa = False
    backend._host_metadata_cache = None
    # Single-token decode: the fixed-width DECODE range is covered separately in
    # test_pallas_rpa_v3_backend.py; these tests are about the query-length source.
    backend.decode_query_size = 1
    return backend


@pytest.mark.parametrize("lengths", [[4, 4, 0, 0], [4, 4, 4, 4]])
@pytest.mark.parametrize(
    "mirror_kind", ["list", "tensor", "missing", "partial", "rank2", "device_rank2"]
)
def test_eagle_refresh_requires_complete_host_query_shape_without_readback(
    lengths, mirror_kind
):
    backend = _bare_backend()
    backend.page_size = 64
    backend.max_context_len = 128
    backend._is_hybrid_swa = False
    backend._is_extend_including_draft = lambda mode: True
    backend._get_cached_rpa_distribution = lambda *args: torch.zeros(
        3, dtype=torch.int32
    )
    backend.req_to_token_pool = SimpleNamespace(
        req_to_token=torch.arange(8 * 128, dtype=torch.int32).view(8, 128)
    )
    mirror = {
        "list": lengths,
        "tensor": torch.tensor(lengths),
        "missing": None,
        "partial": lengths[:2],
        "rank2": torch.tensor(lengths).reshape(4, 1),
        "device_rank2": lengths,
    }[mirror_kind]
    # The meta tensor cannot be copied to CPU, including on invalid input.
    device_lengths = torch.empty(
        (4, 1) if mirror_kind == "device_rank2" else (4,), device="meta"
    )
    batch = SimpleNamespace(
        seq_lens=torch.tensor([68, 73, 0, 0]),
        seq_lens_cpu=torch.tensor([68, 73, 0, 0]),
        req_pool_indices_cpu=torch.tensor([4, 1, 0, 0]),
        extend_seq_lens=device_lengths,
        extend_seq_lens_cpu=mirror,
        forward_mode=ForwardMode.DRAFT_EXTEND_V2,
        spec_info=SimpleNamespace(
            spec_input_type=SpecInputType.EAGLE_DRAFT_EXTEND,
            num_tokens_per_req=4,
        ),
    )
    with patch.object(envs.SGLANG_TPU_ENABLE_HOST_QUERY_LENS, "get", return_value=True):
        if mirror_kind in ("missing", "partial", "rank2", "device_rank2"):
            with pytest.raises(AssertionError):
                backend._build_metadata(batch, non_empty_bs=2)
            return
        metadata = backend._build_metadata(batch, non_empty_bs=2)
    expected = torch.cat(
        (torch.zeros(1, dtype=torch.int32), torch.tensor(lengths).cumsum(0).int())
    )
    torch.testing.assert_close(metadata.cu_q_lens, expected)
    assert metadata.active_num_tokens.item() == sum(lengths[:2])


@pytest.mark.parametrize("padding", ["general", "eagle"])
@pytest.mark.parametrize("physical_bs", [2, 4])
@pytest.mark.parametrize("mirror_kind", ["missing", "partial", "list"])
@pytest.mark.parametrize("enabled", [False, True])
def test_query_mirror_survives_padding_before_metadata(
    padding, physical_bs, mirror_kind, enabled
):
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
    from sglang_tpu.compilation.bucketer import Bucketer
    from sglang_tpu.compilation.compiled_forward_shim import (
        _pad_forward_batch_inputs,
        _restore_forward_batch_inputs,
    )
    from sglang_tpu.compilation.spec_decode_compile_policy_eagle3 import (
        Eagle3SpecDecodeCompilePolicyMixin,
    )

    mirror = {
        "missing": None,
        "partial": [4],
        "list": [4, 4],
    }[mirror_kind]
    batch = ForwardBatch(
        forward_mode=ForwardMode.DRAFT_EXTEND_V2,
        batch_size=2,
        input_ids=torch.arange(8),
        positions=torch.arange(8),
        req_pool_indices=torch.tensor([0, 1]),
        req_pool_indices_cpu=torch.tensor([0, 1]),
        seq_lens=torch.tensor([68, 73]),
        seq_lens_cpu=torch.tensor([68, 73]),
        seq_lens_sum=141,
        out_cache_loc=torch.arange(8),
        extend_seq_lens=torch.tensor([4, 4], dtype=torch.int32),
        extend_seq_lens_cpu=mirror,
        extend_prefix_lens=torch.tensor([64, 69]),
        extend_prefix_lens_cpu=[64, 69],
        extend_start_loc=torch.tensor([0, 4]),
        extend_num_tokens=8,
        spec_info=SimpleNamespace(
            num_tokens_per_req=4,
            spec_input_type=SpecInputType.EAGLE_DRAFT_EXTEND,
            is_draft_input=lambda: False,
        ),
    )
    backend = _bare_backend()
    with (
        patch.object(
            envs.SGLANG_TPU_ENABLE_HOST_QUERY_LENS, "get", return_value=enabled
        ),
        patch(
            "sglang.srt.layers.dp_attention.is_dp_attention_enabled", return_value=False
        ),
    ):
        if padding == "general":
            _pad_forward_batch_inputs(
                batch,
                Bucketer(name="Token", buckets=[16]),
                Bucketer(name="Batch", buckets=[physical_bs]),
                extend_bs=physical_bs,
            )
        else:
            policy = object.__new__(Eagle3SpecDecodeCompilePolicyMixin)
            policy.b_bucketer = Bucketer(name="Batch", buckets=[physical_bs])
            policy.draft_token_num = 4
            policy._dp_peer_draft_extend_rows = lambda _: 0
            policy.model_runner = SimpleNamespace(
                attn_backend=SimpleNamespace(
                    get_cpu_graph_seq_len_fill_value=lambda: 0,
                    get_cuda_graph_seq_len_fill_value=lambda: 0,
                )
            )
            policy._pad_draft_extend(batch)

        def build_metadata():
            return backend._build_metadata_host(
                batch,
                64,
                torch.device("cpu"),
                physical_bs,
                torch.arange(4 * 128, dtype=torch.int32).reshape(4, 128),
                2,
                torch.zeros(3, dtype=torch.int32),
                non_empty_bs=2,
            )

        if enabled and mirror_kind in ("missing", "partial"):
            assert batch.extend_seq_lens_cpu is None
            with pytest.raises(AssertionError):
                build_metadata()
            return
        metadata = build_metadata()
    expected = torch.cat(
        (torch.zeros(1, dtype=torch.int32), batch.extend_seq_lens.cumsum(0).int())
    )
    torch.testing.assert_close(metadata.cu_q_lens, expected)
    assert metadata.active_num_tokens.item() == 8
    if mirror_kind in ("missing", "partial"):
        assert batch.extend_seq_lens_cpu is None
    else:
        assert batch.extend_seq_lens_cpu == batch.extend_seq_lens.tolist()
    _restore_forward_batch_inputs(batch, 8, 2)
    assert batch.extend_seq_lens_cpu == (
        None if mirror_kind in ("missing", "partial") else [4, 4]
    )
    assert batch.extend_seq_lens.tolist() == [4, 4]


@pytest.mark.parametrize(
    "mode,spec_type,width",
    [
        (ForwardMode.EXTEND, None, 0),
        (ForwardMode.EXTEND, SpecInputType.EAGLE_DRAFT_EXTEND, 4),
        (ForwardMode.DRAFT_EXTEND_V2, SpecInputType.DFLASH_DRAFT, 4),
        (ForwardMode.DRAFT_EXTEND_V2, SpecInputType.EAGLE_DRAFT_EXTEND, -1),
    ],
)
@pytest.mark.parametrize(
    "mirror,enabled,expected",
    [
        ([5, 9], True, [0, 5, 14]),
        # Fall back to device lengths if the host mirror is incomplete or disabled.
        # Different host and device values reveal which source was used.
        (None, True, [0, 6, 13]),
        ([5], True, [0, 6, 13]),
        ([5, 9], False, [0, 6, 13]),
    ],
)
def test_other_extend_paths_use_complete_mirror_or_device_lengths(
    mode, spec_type, width, mirror, enabled, expected
):
    backend = _bare_backend()
    batch = SimpleNamespace(
        forward_mode=mode,
        spec_info=(
            None
            if spec_type is None
            else SimpleNamespace(spec_input_type=spec_type, num_tokens_per_req=width)
        ),
        seq_lens=torch.tensor([68, 73]),
        seq_lens_cpu=torch.tensor([68, 73]),
        req_pool_indices_cpu=torch.tensor([0, 1]),
        extend_seq_lens=torch.tensor([6, 7]),
        extend_seq_lens_cpu=mirror,
    )
    with patch.object(
        envs.SGLANG_TPU_ENABLE_HOST_QUERY_LENS, "get", return_value=enabled
    ):
        metadata = backend._build_metadata_host(
            batch,
            64,
            torch.device("cpu"),
            2,
            torch.arange(2 * 128, dtype=torch.int32).reshape(2, 128),
            2,
            torch.zeros(3, dtype=torch.int32),
            non_empty_bs=2,
        )
    torch.testing.assert_close(
        metadata.cu_q_lens, torch.tensor(expected, dtype=torch.int32)
    )


@pytest.mark.parametrize("idle", [False, True])
def test_eagle_query_producer_precedes_idle_normalization(idle):
    from unittest.mock import Mock

    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang.srt.speculative import eagle_worker_common as common

    bs = 0 if idle else 2
    batch = SimpleNamespace(
        seq_lens=torch.full((bs,), 64),
        seq_lens_cpu=None,
        forward_mode=ForwardMode.IDLE if idle else ForwardMode.DECODE,
        model_config=SimpleNamespace(vocab_size=100),
    )
    forward = SimpleNamespace(
        seq_lens=batch.seq_lens,
        seq_lens_cpu=None,
        extend_seq_lens_cpu=None,
        forward_mode=ForwardMode.IDLE if idle else ForwardMode.DRAFT_EXTEND_V2,
        mark_forward_metadata_ready=lambda: None,
    )

    def normalize(fb):
        if idle:
            assert fb.extend_seq_lens_cpu is None
            # An idle DP rank acquires the peer's physical request layout.
            fb.seq_lens = torch.full((4,), 64)
            fb.extend_seq_lens = torch.full((4,), 4)
            fb.extend_seq_lens_cpu = [4] * 4
            fb.forward_mode = ForwardMode.DRAFT_EXTEND_V2
        else:
            assert fb.extend_seq_lens_cpu == [4, 4]
            fb.extend_seq_lens = torch.full((2,), 4)

    runner = SimpleNamespace(
        spec_algorithm=SimpleNamespace(is_standalone=lambda: False),
        compile_policy=SimpleNamespace(
            normalize_spec_forward_batch=normalize,
            prepare_spec_forward_batch=lambda fb: None,
        ),
        attn_backend=SimpleNamespace(init_forward_metadata=Mock()),
    )
    with (
        patch.object(common.ForwardBatch, "init_new", return_value=forward),
        patch.object(common, "maybe_detect_oob"),
    ):
        result = common.prepare_for_draft_extend(
            SimpleNamespace(num_front_tokens=0),
            batch,
            torch.zeros(bs * 4, dtype=torch.int64),
            4,
            runner,
            None,
            return_hidden_states_before_norm=False,
        )
    assert result.extend_seq_lens_cpu == result.extend_seq_lens.tolist()


def test_padding_does_not_mutate_a_borrowed_cpu_query_list():
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

    original = [4, 4]
    batch = ForwardBatch(
        forward_mode=ForwardMode.DRAFT_EXTEND_V2,
        batch_size=2,
        input_ids=torch.arange(8),
        positions=torch.arange(8),
        req_pool_indices=torch.tensor([0, 1]),
        seq_lens=torch.tensor([68, 73]),
        seq_lens_sum=141,
        out_cache_loc=torch.arange(8),
        extend_seq_lens=torch.tensor(original),
        extend_seq_lens_cpu=original,
        spec_info=SimpleNamespace(num_tokens_per_req=4, is_draft_input=lambda: False),
    )
    runner = SimpleNamespace(
        attn_backend=SimpleNamespace(
            get_cpu_graph_seq_len_fill_value=lambda: 0,
            get_cuda_graph_seq_len_fill_value=lambda: 0,
        )
    )
    batch._pad_inputs_to_size(runner, 16, 4)
    assert original == [4, 4]
    assert batch.extend_seq_lens_cpu == batch.extend_seq_lens.tolist() == [4, 4, 4, 4]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
