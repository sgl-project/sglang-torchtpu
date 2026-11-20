"""CPU contracts for the MLA DFlash draft hooks (sglang_tpu.hooks.dflash_mla)."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.models.dflash import (
    DFlashAttention,
    DFlashDecoderLayer,
    DFlashLagunaDecoderLayer,
)
from sglang.srt.models.dflash import DFlashMLAAttention as CoreDFlashMLAAttention
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.hooks.dflash_mla import (
    dflash_mla_head_dims,
    select_dflash_mla_attention,
)
from sglang_tpu.models.dflash_mla import DFlashMLAAttention

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _draft_config(mode: str | None, archs=("DFlash2DraftModel",)):
    dflash_config = {"block_size": 8}
    if mode is not None:
        dflash_config["attention_mode"] = mode
    return SimpleNamespace(
        architectures=list(archs),
        dflash_config=dflash_config,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        num_attention_heads=64,
        num_key_value_heads=64,
    )


def _model_config(hf_config):
    # _derive_model_shapes' result for the Kimi-K3.1 draft: config.head_dim is the rope dim.
    return SimpleNamespace(
        hf_config=hf_config,
        hf_text_config=hf_config,
        attention_arch=AttentionArch.MLA,
        head_dim=64,
        v_head_dim=128,
        swa_head_dim=64,
        swa_v_head_dim=128,
        get_total_num_kv_heads=lambda: hf_config.num_key_value_heads,
    )


def _attention_cls(layer_type, config, core_cls, positional=False):
    """The attention class the layer constructor receives when core passes core_cls."""
    layer = layer_type.__new__(layer_type)
    if positional:
        return select_dflash_mla_attention(
            lambda layer, config, layer_id, attention_cls, **kwargs: attention_cls,
            layer,
            config,
            0,
            core_cls,
        )
    return select_dflash_mla_attention(
        lambda layer, config, **kwargs: kwargs["attention_cls"],
        layer,
        config=config,
        layer_id=0,
        attention_cls=core_cls,
    )


class TestDFlashMlaHooks(CustomTestCase):
    def test_head_dims(self):
        """Only an MLA DFlash draft takes the expanded (nope + rope) head dims."""
        mha, mla = AttentionArch.MHA, AttentionArch.MLA
        for hf_config, want, arch in (
            (_draft_config("mla"), (192, 128), mha),
            (_draft_config(None), (64, 128), mla),
            (_draft_config("gqa"), (64, 128), mla),
            (
                _draft_config("mla", archs=("KimiK31ForConditionalGeneration",)),
                (64, 128),
                mla,
            ),
        ):
            with self.subTest(hf_config=hf_config):
                mc = _model_config(hf_config)
                dflash_mla_head_dims(None, mc)
                self.assertEqual((mc.head_dim, mc.v_head_dim), want)
                self.assertEqual((mc.swa_head_dim, mc.swa_v_head_dim), want)
                self.assertIs(mc.attention_arch, arch)

    def test_mla_draft_needs_one_kv_head_per_attention_head(self):
        hf_config = _draft_config("mla")
        hf_config.num_key_value_heads = 8
        with self.assertRaisesRegex(ValueError, "num_key_value_heads"):
            dflash_mla_head_dims(None, _model_config(hf_config))

    def test_attention_class(self):
        """MLA drafts build the TPU DFlashMLAAttention whichever class core
        passes; GQA drafts and a subclass's own attention are kept."""
        laguna = DFlashLagunaDecoderLayer.attention_cls
        for layer_type, mode, core_cls, want in (
            (DFlashDecoderLayer, "mla", CoreDFlashMLAAttention, DFlashMLAAttention),
            (DFlashDecoderLayer, "mla", DFlashAttention, DFlashMLAAttention),
            (DFlashDecoderLayer, "gqa", DFlashAttention, DFlashAttention),
            (DFlashLagunaDecoderLayer, "mla", laguna, laguna),
        ):
            for positional in (False, True):
                with self.subTest(
                    layer_type=layer_type.__name__,
                    mode=mode,
                    core_cls=core_cls.__name__,
                    positional=positional,
                ):
                    got = _attention_cls(
                        layer_type, _draft_config(mode), core_cls, positional
                    )
                    self.assertIs(got, want)


if __name__ == "__main__":
    unittest.main()
