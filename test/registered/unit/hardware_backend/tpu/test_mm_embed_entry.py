"""The explicit MM backend hook preserves shared orchestration contracts."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.tpu import mm_embed_tpu
from sglang.srt.hardware_backend.tpu.mm_embedding import PaddedEmbedding
from sglang.srt.managers import mm_schedule, mm_utils
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.mem_cache.multimodal_cache import MultiModalStaticCache
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="stage-a-test-cpu-intel")


def _forward_batch(mm_inputs, mode="extend"):
    return SimpleNamespace(
        forward_mode=SimpleNamespace(
            is_decode=lambda: mode == "decode",
            is_target_verify=lambda: mode == "verify",
        ),
        contains_mm_inputs=lambda: bool(mm_inputs),
        mm_inputs=mm_inputs,
        extend_prefix_lens_cpu=[0, 1],
        extend_seq_lens_cpu=[1, 3],
        spec_algorithm=SimpleNamespace(is_eagle=lambda: True),
        input_embeds=torch.empty(4, 2),
    )


@pytest.mark.parametrize("custom_backend", [False, True])
@pytest.mark.parametrize("adaptive", [False, True])
def test_backend_selection_preserves_shared_batch_lifecycle(custom_backend, adaptive):
    input_ids = torch.tensor([1, 2, 3, 4])
    embedding = torch.nn.Embedding(8, 2)
    mm_input = MultimodalInputs(mm_items=[])
    forward_batch = _forward_batch([None, mm_input])
    merged = torch.arange(8).reshape(4, 2).float()
    deepstack = torch.ones(4, 6)
    result = (merged, {"input_deepstack_embeds": deepstack})
    callback = Mock(return_value=result)
    language_model = Mock()
    language_model.get_input_embeddings.return_value = embedding
    language_model.pp_group.is_first_rank = True

    with (
        patch.object(mm_utils, "get_server_args", return_value=True),
        patch.object(
            mm_utils,
            "get_disagg",
            return_value=SimpleNamespace(enable_adaptive_dispatch_to_encoder=adaptive),
        ),
        patch.object(mm_utils, "embed_mm_inputs", return_value=result) as ordinary,
        patch.object(
            mm_utils, "_embed_mm_inputs_with_split", return_value=result
        ) as split,
    ):
        output = mm_utils.general_mm_embed_routine(
            input_ids,
            forward_batch,
            language_model,
            mm_embedder=callback if custom_backend else None,
            use_deepstack={Modality.IMAGE: True},
        )

    selected = callback if custom_backend else split if adaptive else ordinary
    selected.assert_called_once()
    for unused in (callback, ordinary, split):
        if unused is not selected:
            unused.assert_not_called()
    kwargs = selected.call_args.kwargs
    assert kwargs["mm_inputs_list"] == [mm_input]
    assert kwargs["extend_prefix_lens"] == [1]
    assert kwargs["extend_seq_lens"] == [3]
    assert kwargs["input_ids"] is input_ids
    assert kwargs["input_embedding"] is embedding
    if custom_backend:
        assert kwargs["forward_batch"] is forward_batch
    assert forward_batch.mm_inputs is None
    assert torch.equal(forward_batch.mm_input_embeds, merged)
    assert forward_batch.mm_input_embeds.data_ptr() != merged.data_ptr()
    assert torch.equal(forward_batch.input_embeds, merged)
    language_model.assert_called_once_with(
        input_ids=None,
        forward_batch=forward_batch,
        input_embeds=forward_batch.input_embeds,
        input_deepstack_embeds=deepstack,
    )
    assert output is language_model.return_value


@pytest.mark.parametrize("mode", ["decode", "verify", "text"])
def test_embedding_callback_skipped_outside_mm_prefill(mode):
    mm_input = MultimodalInputs(mm_items=[])
    mm_inputs = [] if mode == "text" else [None, mm_input]
    batch = _forward_batch(mm_inputs, mode)
    ids = torch.tensor([1, 2, 3, 4])
    embedding = torch.nn.Embedding(8, 2)
    expected = embedding(ids)
    language_model = Mock()
    language_model.get_input_embeddings.return_value = embedding
    language_model.pp_group.is_first_rank = True
    callback = Mock(side_effect=AssertionError("must skip multimodal preparation"))

    mm_utils.general_mm_embed_routine(ids, batch, language_model, mm_embedder=callback)

    callback.assert_not_called()
    assert batch.mm_inputs is mm_inputs
    assert torch.equal(language_model.call_args.kwargs["input_embeds"], expected)


@pytest.mark.parametrize("on_device", [False, True])
@pytest.mark.parametrize("precomputed", [False, True])
def test_explicit_tpu_entry_returns_merged_tensor_to_shared_routine(
    on_device, precomputed
):
    cpu_ids = torch.tensor([1, 2, 99, 99])
    ids = cpu_ids.clone()
    vision = torch.tensor([[8.0, 9.0], [10.0, 11.0]])
    mm_input = MultimodalInputs(
        mm_items=[
            MultimodalDataItem(
                modality=Modality.IMAGE,
                hash=123,
                offsets=[(2, 3)],
                pad_value=99,
                feature=torch.zeros(1),
                precomputed_embeddings=vision if precomputed else None,
            )
        ]
    )
    batch = _forward_batch([mm_input])
    batch.extend_prefix_lens_cpu = [0]
    batch.extend_seq_lens_cpu = [4]
    batch.mm_input_ids_cpu = cpu_ids
    embedding = torch.nn.Embedding(8, 2)
    expected = embedding(ids.clamp(0, 7))
    expected[2:4] = vision
    buffer = torch.full((8, 2), torch.nan)
    buffer[:2] = vision
    encoder = Mock(return_value=[PaddedEmbedding.from_buffer(buffer, 2)])
    language_model = Mock()
    language_model.get_input_embeddings.return_value = embedding
    language_model.pp_group.is_first_rank = True

    with (
        envs.SGLANG_TPU_MM_EMBED_ON_DEVICE.override(on_device),
        patch.object(mm_schedule, "embedding_cache", MultiModalStaticCache(4096)),
        patch.object(mm_utils, "get_server_args", return_value=None),
        patch.object(mm_embed_tpu, "get_server_args", return_value=None),
        patch.object(
            mm_utils,
            "embed_mm_inputs",
            side_effect=AssertionError("must use TPU backend"),
        ),
    ):
        mm_utils.general_mm_embed_routine(
            ids,
            batch,
            language_model,
            data_embedding_funcs={Modality.IMAGE: encoder},
            mm_embedder=mm_embed_tpu.embed_mm_inputs_tpu,
        )

    if precomputed:
        encoder.assert_not_called()
    else:
        encoder.assert_called_once()
    assert torch.equal(batch.mm_input_embeds, expected)
    assert torch.equal(language_model.call_args.kwargs["input_embeds"], expected)
    assert cpu_ids.tolist() == [1, 2, 99, 99]
