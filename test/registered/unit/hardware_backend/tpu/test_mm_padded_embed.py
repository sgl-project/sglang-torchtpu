"""CPU coverage for TPU merge indexing and embedding preparation."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.tpu import mm_embed_tpu
from sglang.srt.hardware_backend.tpu.mm_embed_tpu import scatter_mm_on_device
from sglang.srt.hardware_backend.tpu.mm_embedding import (
    EmbeddingSpan,
    PaddedEmbedding,
    concat_embeddings,
)
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-test-cpu-intel")

NUM_TOKENS = 64


def test_embed_preserves_cpu_placeholders_and_merges_multiple_images():
    cpu_ids = torch.tensor([1, 99, 99, 2, 100, 100, 3])
    device_ids = torch.cat([cpu_ids, torch.zeros(1, dtype=torch.long)])
    text_embedding = torch.nn.Embedding.from_pretrained(
        torch.arange(32).reshape(8, 4).float()
    )
    vision = [torch.full((2, 4), 40.0), torch.full((2, 4), 50.0)]
    mm_inputs = MultimodalInputs(
        mm_items=[
            MultimodalDataItem(
                modality=Modality.IMAGE,
                pad_value=pad,
                offsets=[offset],
                precomputed_embeddings=embedding,
            )
            for pad, offset, embedding in zip([99, 100], [(1, 2), (4, 5)], vision)
        ]
    )
    expected = text_embedding(device_ids.clamp(0, 7))
    expected[1:3], expected[4:6] = vision
    with (
        envs.SGLANG_TPU_MM_EMBED_ON_DEVICE.override(True),
        patch.object(mm_embed_tpu, "get_server_args", return_value=None),
    ):
        result, _ = mm_embed_tpu.embed_mm_inputs_tpu(
            forward_batch=SimpleNamespace(mm_input_ids_cpu=cpu_ids),
            mm_inputs_list=[mm_inputs],
            extend_prefix_lens=[0],
            extend_seq_lens=[7],
            input_ids=device_ids,
            input_embedding=text_embedding,
            data_embedding_func_mapping={Modality.IMAGE: Mock()},
        )
    assert torch.equal(result, expected)
    assert cpu_ids.tolist() == [1, 99, 99, 2, 100, 100, 3]
    assert device_ids.tolist() == [1, 7, 7, 2, 7, 7, 3, 0]


def test_adaptive_split_preserves_cpu_token_order_with_text_only_requests():
    def make_input(precomputed):
        return MultimodalInputs(
            mm_items=[
                MultimodalDataItem(
                    modality=Modality.IMAGE,
                    precomputed_embeddings=torch.ones(2, 1) if precomputed else None,
                )
            ]
        )

    mm_inputs = [make_input(True), make_input(False), make_input(True)]
    cpu_ids = torch.arange(13)
    forward_batch = SimpleNamespace(
        mm_inputs=[None, mm_inputs[0], None, mm_inputs[1], mm_inputs[2]],
        extend_seq_lens_cpu=[2, 3, 2, 3, 3],
        mm_input_ids_cpu=cpu_ids,
    )
    groups = []

    def embed_group(*, input_ids, input_ids_cpu, **kwargs):
        assert torch.equal(input_ids, input_ids_cpu)
        groups.append(input_ids_cpu.tolist())
        return input_ids_cpu.float().unsqueeze(-1), {}

    with (
        patch.object(mm_embed_tpu, "_embed_mm_inputs", side_effect=embed_group),
        patch.object(mm_embed_tpu, "get_server_args", return_value=True),
        patch.object(
            mm_embed_tpu,
            "get_disagg",
            return_value=SimpleNamespace(enable_adaptive_dispatch_to_encoder=True),
        ),
    ):
        result, _ = mm_embed_tpu.embed_mm_inputs_tpu(
            mm_inputs_list=mm_inputs,
            extend_prefix_lens=[0, 0, 0],
            extend_seq_lens=[3, 3, 3],
            input_ids=cpu_ids.clone(),
            forward_batch=forward_batch,
            input_embedding=torch.nn.Embedding.from_pretrained(
                cpu_ids.float().unsqueeze(-1)
            ),
        )
    assert groups == [[2, 3, 4, 10, 11, 12], [7, 8, 9]]
    assert torch.equal(result[:, 0], cpu_ids.float())


def test_token_rewrite_invalidates_cpu_ids_for_later_modalities():
    cpu_ids = torch.tensor([1, 2, 3])
    device_ids = cpu_ids.clone()
    rewritten_ids = torch.tensor([3, 2, 1])
    mm_inputs = MultimodalInputs(
        mm_items=[
            MultimodalDataItem(modality=modality, pad_value=99, offsets=[(1, 1)])
            for modality in (Modality.IMAGE, Modality.VIDEO, Modality.AUDIO)
        ]
    )
    calls = []

    def rewrite(*, input_ids, input_ids_cpu, **kwargs):
        calls.append(input_ids_cpu)
        return None, None, rewritten_ids

    with (
        patch.object(mm_embed_tpu, "get_embedding_and_mask", side_effect=rewrite),
        envs.SGLANG_TPU_MM_EMBED_ON_DEVICE.override(True),
    ):
        mm_embed_tpu._embed_mm_inputs(
            mm_inputs_list=[mm_inputs],
            extend_prefix_lens=[0],
            extend_seq_lens=[3],
            input_ids=device_ids,
            input_ids_cpu=cpu_ids,
            input_embedding=torch.nn.Embedding(4, 1),
            data_embedding_func_mapping={
                modality: Mock() for modality in Modality.all()
            },
        )
    assert calls[0] is cpu_ids
    assert calls[1:] == [None, None]


def _make_mask(pattern: str) -> torch.Tensor:
    mask = torch.zeros(NUM_TOKENS, dtype=torch.bool)
    if pattern == "interleaved":
        mask[::3] = True
    elif pattern == "blocks":
        mask[5:20] = True
        mask[40:41] = True
    elif pattern == "all_true":
        mask[:] = True
    return mask.unsqueeze(-1)


@pytest.mark.parametrize("width", [8, 24])
@pytest.mark.parametrize("src_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "mask_pattern", ["interleaved", "blocks", "all_true", "all_false"]
)
def test_scatter_matches_masked_scatter_bitwise(width, src_dtype, mask_pattern):
    """The row-index mm embedding merge must stay bitwise identical to
    masked_scatter_ semantics, whose internal transients it avoids."""
    torch.default_generator.manual_seed(0)
    mask = _make_mask(mask_pattern)
    dest = torch.randn(NUM_TOKENS, width).to(torch.bfloat16)
    src = torch.randn(int(mask.sum()), width, dtype=src_dtype)

    expected = dest.clone()
    expected.masked_scatter_(mask.expand_as(expected), src.to(expected.dtype))

    actual = dest.clone()
    scatter_mm_on_device(dest=actual, mask=mask, src=src)
    assert torch.equal(actual, expected)


def test_scatter_row_count_mismatch_fails_loud():
    """A mask/src row-count mismatch must raise, not silently corrupt rows."""
    dest = torch.zeros(8, 4)
    src_short_mask = _make_mask("all_false")[:8]
    src_short_mask[1] = True
    with pytest.raises((RuntimeError, IndexError)):
        scatter_mm_on_device(dest=dest, mask=src_short_mask, src=torch.ones(3, 4))
    mask_heavy = src_short_mask.clone()
    mask_heavy[2:6] = True
    with pytest.raises((RuntimeError, IndexError)):
        scatter_mm_on_device(dest=dest, mask=mask_heavy, src=torch.ones(1, 4))


def test_device_scatter_empty_token_batch():
    dest = torch.empty(0, 8)
    scatter_mm_on_device(dest, torch.empty(0, 1, dtype=torch.bool), torch.empty(0, 8))
    assert dest.shape == (0, 8)


def test_device_scatter_padded_cached_view_preserves_backing_storage():
    cache = torch.arange(64 * 16, dtype=torch.float32).reshape(64, 16)
    original_cache = cache.clone()
    src = cache[7:25:2, ::2]
    dest = torch.full((32, 8), -1.0, dtype=torch.bfloat16)
    mask = torch.zeros(32, 1, dtype=torch.bool)
    mask[3:30:3] = True
    expected = dest.clone()
    expected.masked_scatter_(mask.expand_as(expected), src.to(expected.dtype))

    scatter_mm_on_device(dest, mask, src)

    assert torch.equal(dest, expected)
    assert torch.equal(cache, original_cache)


def test_device_scatter_full_bucket_source_alias():
    dest = torch.arange(32 * 8, dtype=torch.float32).reshape(32, 8)
    expected = dest.clone()

    scatter_mm_on_device(dest, torch.ones(32, 1, dtype=torch.bool), dest)

    assert torch.equal(dest, expected)


@pytest.mark.parametrize(
    "physical_rows,token_bucket,valid_lengths",
    [(256, 512, [256, 240]), (512, 1024, [512, 480]), (512, 256, [200, 180])],
)
def test_padded_merge_preserves_physical_bucket_across_valid_lengths(
    physical_rows, token_bucket, valid_lengths
):
    signatures = []
    for valid_rows in valid_lengths:
        buffer = torch.arange(physical_rows * 8).reshape(physical_rows, 8).float()
        buffer[valid_rows:] = torch.nan
        src = PaddedEmbedding.from_buffer(buffer, valid_rows)
        mask = torch.zeros(token_bucket, 1, dtype=torch.bool)
        mask[3 : 3 + valid_rows] = True
        dest = torch.full((token_bucket, 8), -7.0, dtype=torch.bfloat16)
        expected = dest.clone().masked_scatter_(
            mask.expand_as(dest), buffer[:valid_rows].to(dest.dtype)
        )
        with patch.object(
            mm_embed_tpu, "_scatter_mm_rows", wraps=mm_embed_tpu._scatter_mm_rows
        ) as merge:
            scatter_mm_on_device(dest, mask, src)
        assert torch.equal(dest, expected)
        args = merge.call_args.args
        assert args[1] is buffer
        signatures.append(tuple((value.shape, value.dtype) for value in args))
    assert signatures[0] == signatures[1]


def test_padded_merge_reuses_one_buffer_for_multiple_cached_spans():
    buffer = torch.arange(64).reshape(16, 4).float()
    src = PaddedEmbedding((EmbeddingSpan(buffer, 5, 3), EmbeddingSpan(buffer, 1, 2)))
    mask = torch.zeros(16, 1, dtype=torch.bool)
    mask[[1, 4, 5, 9, 13]] = True
    dest = torch.full((16, 4), -1.0)
    expected = dest.clone().masked_scatter_(
        mask.expand_as(dest), torch.cat([buffer[5:8], buffer[1:3]])
    )
    with patch.object(
        mm_embed_tpu, "_scatter_mm_rows", wraps=mm_embed_tpu._scatter_mm_rows
    ) as merge:
        scatter_mm_on_device(dest, mask, src)
    assert torch.equal(dest, expected)
    merge.assert_called_once()


def test_padded_and_unpadded_chunks_merge_in_order():
    first = PaddedEmbedding.from_buffer(torch.arange(32).reshape(8, 4).float(), 3)
    second = torch.full((2, 4), 50.0)
    src = concat_embeddings([first.slice(1, 3), second])
    mask = torch.tensor([False, True, False, True, True, True]).unsqueeze(-1)
    dest = torch.full((6, 4), -1.0)
    expected = dest.clone().masked_scatter_(mask.expand_as(dest), src.to_host())
    scatter_mm_on_device(dest, mask, src)
    assert torch.equal(dest, expected)
