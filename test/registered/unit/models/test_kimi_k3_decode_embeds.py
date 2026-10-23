import contextlib
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.compilation.support_torch_compile import get_torch_compile_policy
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models import kimi_k3
from sglang.srt.models.kimi_k3 import KimiK3ForConditionalGeneration
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.compilation.decoder_compile_policy import (
    KimiK3VLDecoderTpuCompilePolicy,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _forward(
    *,
    decode_in_graph: bool,
    is_decode: bool,
    is_target_verify=False,
    has_mm_inputs=False,
    mm_extend_in_batch=False,
    materialized_from=None,
):
    """Call the wrapper's forward on a stand-in self with a mocked LM."""
    language_model = MagicMock(return_value="lm-out")
    fake_self = SimpleNamespace(
        decode_embeds_in_language_model=decode_in_graph,
        text_prefill_in_language_model=decode_in_graph,
        language_model=language_model,
        get_image_feature=MagicMock(),
    )
    forward_batch = SimpleNamespace(
        forward_mode=SimpleNamespace(
            is_decode=lambda: is_decode,
            is_target_verify=lambda: is_target_verify,
        ),
        contains_mm_inputs=lambda: has_mm_inputs,
        is_tpu_mm_extend_in_batch=mm_extend_in_batch,
        # Set when DP attention runs a decode (or idle) batch as extends.
        _original_forward_mode=materialized_from,
    )
    input_ids = torch.tensor([1, 2, 3])
    positions = torch.tensor([0, 1, 2])
    with patch.object(
        kimi_k3, "general_mm_embed_routine", return_value="routine-out"
    ) as routine:
        out = KimiK3ForConditionalGeneration.forward(
            fake_self, input_ids, positions, forward_batch
        )
    return out, language_model, routine, input_ids, positions, forward_batch


class TestKimiK3DecodeEmbedsInGraph(CustomTestCase):
    def test_decode_hands_raw_input_ids_to_the_language_model(self):
        out, lm, routine, ids, pos, fb = _forward(decode_in_graph=True, is_decode=True)
        self.assertEqual(out, "lm-out")
        routine.assert_not_called()
        # The compiled decode variants are warmed without input_embeds.
        lm.assert_called_once_with(ids, pos, fb, pp_proxy_tensors=None)

    def test_target_verify_hands_raw_input_ids_to_the_language_model(self):
        out, lm, routine, ids, pos, fb = _forward(
            decode_in_graph=True, is_decode=False, is_target_verify=True
        )
        self.assertEqual(out, "lm-out")
        routine.assert_not_called()
        lm.assert_called_once_with(ids, pos, fb, pp_proxy_tensors=None)

    def test_text_only_prefill_embeds_in_the_language_model(self):
        out, lm, routine, ids, pos, fb = _forward(decode_in_graph=True, is_decode=False)
        self.assertEqual(out, "lm-out")
        routine.assert_not_called()
        lm.assert_called_once_with(ids, pos, fb, pp_proxy_tensors=None)

    def test_image_prefill_uses_the_embed_routine(self):
        out, lm, routine, *_ = _forward(
            decode_in_graph=True, is_decode=False, has_mm_inputs=True
        )
        self.assertEqual(out, "routine-out")
        routine.assert_called_once()
        lm.assert_not_called()

    def test_text_prefill_takes_the_routine_when_a_dp_peer_has_an_image(self):
        out, lm, routine, *_ = _forward(
            decode_in_graph=True, is_decode=False, mm_extend_in_batch=True
        )
        self.assertEqual(out, "routine-out")
        routine.assert_called_once()

    def test_image_decode_run_as_extends_keeps_the_text_program(self):
        # A DP peer's text extend turns this rank's decode into one-token
        # extends. Its image request embeds nothing new, so it must take the
        # peers' input_ids program, not the routine (CDK j-0f28a1ee hung here).
        out, lm, routine, ids, pos, fb = _forward(
            decode_in_graph=True,
            is_decode=False,
            has_mm_inputs=True,
            materialized_from=ForwardMode.DECODE,
        )
        self.assertEqual(out, "lm-out")
        routine.assert_not_called()
        lm.assert_called_once_with(ids, pos, fb, pp_proxy_tensors=None)

    def test_decode_run_as_extends_follows_a_dp_peers_image_step(self):
        out, lm, routine, *_ = _forward(
            decode_in_graph=True,
            is_decode=False,
            has_mm_inputs=True,
            mm_extend_in_batch=True,
            materialized_from=ForwardMode.DECODE,
        )
        self.assertEqual(out, "routine-out")
        routine.assert_called_once()
        lm.assert_not_called()

    def test_decode_uses_the_embed_routine_when_disabled(self):
        out, lm, routine, *_ = _forward(decode_in_graph=False, is_decode=True)
        self.assertEqual(out, "routine-out")
        routine.assert_called_once()

    def test_logits_processor_resolves_to_the_language_model(self):
        # The TPU compile policy and sampler look it up on model_runner.model.
        lp = object()
        fake_self = SimpleNamespace(language_model=SimpleNamespace(logits_processor=lp))
        self.assertIs(
            KimiK3ForConditionalGeneration.logits_processor.fget(fake_self), lp
        )
        fake_self = SimpleNamespace(language_model=None)
        with self.assertRaisesRegex(AttributeError, "encoder-only"):
            KimiK3ForConditionalGeneration.logits_processor.fget(fake_self)


@contextlib.contextmanager
def _published_server_args(**fields):
    override = get_context().override_server_args(**fields)
    override.install()
    try:
        yield
    finally:
        override.restore()


class TestKimiK3CompileAnnotation(CustomTestCase):
    @staticmethod
    def _build(*, tpu: bool, **server_args) -> KimiK3ForConditionalGeneration:
        # A fresh class per build: an annotation on it cannot leak across tests.
        class LanguageModel(torch.nn.Linear):
            def __init__(self, *_, **__):
                super().__init__(1, 1)

        config = SimpleNamespace(
            encoder_only=False, vision_config=None, text_config=None
        )
        with (
            patch.object(kimi_k3, "_is_tpu", tpu),
            patch.object(kimi_k3, "KimiK3VisionTower", lambda _: torch.nn.Identity()),
            patch.object(
                kimi_k3, "KimiK3MultiModalProjector", lambda _: torch.nn.Identity()
            ),
            patch.object(kimi_k3, "KimiK3LinearForCausalLM", LanguageModel),
            _published_server_args(**server_args),
        ):
            return KimiK3ForConditionalGeneration(config)

    def test_unsupported_tpu_compile_setups_fail_before_loading(self):
        for server_args in (
            {"enable_torch_compile": True, "tpu_compile_tracer": "direct"},
            {"enable_torch_compile": True, "pp_size": 2},
        ):
            with self.subTest(**server_args):
                with self.assertRaisesRegex(ValueError, "does not support"):
                    self._build(tpu=True, **server_args)
        # Without torch.compile neither matters.
        self._build(tpu=True, tpu_compile_tracer="direct")

    def test_tpu_compiles_the_wrappers_language_model_instance(self):
        """The text-only entry, which shares the language model class, keeps the
        generic decoder policy; the vision tower runs without DP collectives."""
        model = self._build(tpu=True)
        self.assertIs(
            get_torch_compile_policy(model.language_model),
            KimiK3VLDecoderTpuCompilePolicy,
        )
        self.assertIsNone(get_torch_compile_policy(type(model.language_model)))
        self.assertFalse(model.use_data_parallel)

    def test_other_platforms_keep_the_dp_vision_runner_uncompiled(self):
        model = self._build(tpu=False)
        self.assertIsNone(get_torch_compile_policy(model.language_model))
        self.assertTrue(model.use_data_parallel)


class TestKimiK3TpuImageFeaturePlacement(CustomTestCase):
    def _image_features(self, *, on_device: bool):
        """Run the TPU (non-DP) image path with a mocked tower and projector."""
        model = KimiK3ForConditionalGeneration.__new__(KimiK3ForConditionalGeneration)
        torch.nn.Module.__init__(model)
        model.use_data_parallel = False
        # A meta "device" tells host tensors apart from the ones moved to it.
        model.vision_tower = MagicMock(device=torch.device("meta"))
        model.vision_tower.patch_embed.proj.weight.dtype = torch.float32
        model.mm_projector = MagicMock()
        item = MultimodalDataItem(
            modality=Modality.IMAGE,
            offsets=[(0, 1)],
            feature=torch.ones(16, 3),
            model_specific_data={"image_grid_thw": torch.tensor([[1, 4, 4]])},
        )
        with (
            patch.object(kimi_k3, "_is_tpu", True),
            envs.SGLANG_TPU_MM_EMBED_ON_DEVICE.override(on_device),
            get_parallel().override(tp_size=1),
        ):
            out = model.get_image_feature([item])
        model.vision_tower.assert_called_once()
        (pixel_values, grid_thws), kwargs = model.vision_tower.call_args
        self.assertEqual(pixel_values.device.type, "meta")
        # The tower reads the grids on the host: no device copy to sync on.
        self.assertEqual(grid_thws.device.type, "cpu")
        self.assertEqual(grid_thws.tolist(), [[1, 4, 4]])
        self.assertEqual(kwargs["grid_thw_list"], [[1, 4, 4]])
        return out, model.mm_projector.return_value

    def test_features_land_on_the_host(self):
        out, features = self._image_features(on_device=False)
        features.to.assert_called_once_with("cpu")
        self.assertIs(out, features.to.return_value)

    def test_features_stay_on_device_with_mm_embed_on_device(self):
        out, features = self._image_features(on_device=True)
        features.to.assert_not_called()
        self.assertIs(out, features)


if __name__ == "__main__":
    unittest.main()
