# SPDX-License-Identifier: Apache-2.0
"""LTX-2.3 requests that turn classifier-free guidance off (guidance_scale <= 1)."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.ltx_2.denoising import (
    LTX2DenoisingStage,
)


class _ModelCalled(Exception):
    pass


def _record_model_kwargs(ctx, step, model_kwargs):
    raise _ModelCalled(model_kwargs)


class TestLtx2GuiderWithoutCfg(unittest.TestCase):
    def test_cfg_off_runs_the_positive_prompt_only(self):
        # Without CFG the negative prompt is never encoded; the stage-1 guider
        # would index it. The step must run one positive-prompt pass instead.
        stage = object.__new__(LTX2DenoisingStage)
        prompt, audio_prompt = torch.ones(1, 4, 8), torch.ones(1, 4, 8)
        batch = SimpleNamespace(
            extra={"ltx2_stage1_guider_params": {"video_cfg_scale": 3.0}},
            do_classifier_free_guidance=False,
            prompt_embeds=[prompt],
            audio_prompt_embeds=[audio_prompt],
            negative_prompt_embeds=[],
            negative_audio_prompt_embeds=[],
        )
        ctx = SimpleNamespace(
            stage="stage1",
            is_ltx23_variant=True,
            use_ltx23_legacy_one_stage=False,
            latents=torch.zeros(1, 4, 8),
            audio_latents=torch.zeros(1, 4, 8),
            audio_scheduler=object(),
            scheduler=SimpleNamespace(sigmas=torch.tensor([1.0, 0.5])),
        )
        server_args = SimpleNamespace(enable_cfg_parallel=False)
        inputs = SimpleNamespace(latent_model_input=torch.zeros(1, 4, 8))
        with (
            patch.object(stage, "_prepare_ltx2_model_inputs", return_value=inputs),
            patch.object(stage, "_build_ltx2_base_model_kwargs", return_value={}),
            patch.object(stage, "_get_ltx_prompt_attention_mask", return_value=None),
            patch.object(stage, "_build_ltx2_model_kwargs", lambda c, b, **kw: kw),
            patch.object(
                stage, "_ltx2_model_forward_context", lambda c, s: nullcontext()
            ),
            patch.object(stage, "_ltx2_call_current_model", _record_model_kwargs),
        ):
            with self.assertRaises(_ModelCalled) as call:
                stage._run_denoising_step(
                    ctx, SimpleNamespace(step_index=0), batch, server_args
                )
        model_kwargs = call.exception.args[0]
        self.assertIs(model_kwargs["encoder_hidden_states"], prompt)
        self.assertIs(model_kwargs["audio_encoder_hidden_states"], audio_prompt)


if __name__ == "__main__":
    unittest.main()
