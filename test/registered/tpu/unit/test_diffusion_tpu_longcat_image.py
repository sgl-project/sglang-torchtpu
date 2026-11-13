# SPDX-License-Identifier: Apache-2.0
"""Tests for sglang_tpu's LongCat-Image patches (sglang_tpu.multimodal_gen.longcat_image)."""

import contextlib
from types import SimpleNamespace
from unittest.mock import patch

import torch
import transformers
from torch._dynamo import guards as dynamo_guards
from torch._dynamo import source as dynamo_source

from sglang.multimodal_gen.configs.models.dits.longcat_image import (
    LongCatImageArchConfig,
    LongCatImageDitConfig,
)
from sglang.multimodal_gen.configs.models.encoders.qwen_image import Qwen2_5VLConfig
from sglang.multimodal_gen.configs.pipeline_configs import base as config_base
from sglang.multimodal_gen.configs.pipeline_configs import (
    longcat_image as longcat_config,
)
from sglang.multimodal_gen.runtime import server_args as server_args_module
from sglang.multimodal_gen.runtime.cache.conditioning import ConditioningCache
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    maybe_init_distributed_environment_and_model_parallel,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.dits import longcat_image as dit
from sglang.multimodal_gen.runtime.models.encoders import qwen2_5vl
from sglang.multimodal_gen.runtime.server_args import set_global_server_args
from sglang.multimodal_gen.test.single_test_file.component_accuracy.utils import (
    ensure_distributed_env_defaults,
)
from sglang.multimodal_gen.test.unit.conftest import _make_unit_server_args
from sglang.srt.layers.quantization import unquant
from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.multimodal_gen import longcat_image, worker

# Whole node, not chips=1: the parallel-state setup runs the TPU worker
# bootstrap, which opens a chip outside the chip-parallel TPU_VISIBLE_CHIPS pin.
register_tpu_ci(est_time=40, suite="stage-a-test-tpu-unit")


@contextlib.contextmanager
def _installed(backend="eager"):
    """The TPU paths on CPU, with every patched name restored afterwards."""
    with contextlib.ExitStack() as stack:
        for owner, name in (
            (qwen2_5vl.Qwen2_5_VLForConditionalGeneration, "generate"),
            (qwen2_5vl.Qwen2_5_VLModel, "get_rope_index"),
            (qwen2_5vl, "create_causal_mask"),
            (dit.LongCatImageTransformer2DModel, "__call__"),
        ):
            stack.enter_context(patch.object(owner, name, getattr(owner, name)))
        stack.enter_context(patch.object(longcat_image, "_DEVICE_TYPES", ("cpu",)))
        stack.enter_context(
            patch.object(
                longcat_image,
                "build_torch_compile_kwargs",
                lambda mode=None: {"backend": backend},
            )
        )
        stack.enter_context(patch.dict(longcat_image._ROPE_CACHE, clear=True))
        longcat_image.install()
        longcat_image.install()  # idempotent
        yield


@contextlib.contextmanager
def _sp_rank(rank, world_size=8):
    """The pipeline configs' view of one rank in a sequence-parallel group."""
    with contextlib.ExitStack() as stack:
        for owner in (config_base, longcat_config):
            stack.enter_context(
                patch.object(owner, "get_sp_world_size", lambda: world_size)
            )
        stack.enter_context(
            patch.object(config_base, "get_sp_parallel_rank", lambda: rank)
        )
        yield


class _TestBase(CustomTestCase):
    def setUp(self):
        self._previous_args = server_args_module._global_server_args
        set_global_server_args(_make_unit_server_args())
        if not model_parallel_is_initialized():
            ensure_distributed_env_defaults()
            with patch.object(worker, "install_runtime_patches"):
                maybe_init_distributed_environment_and_model_parallel(
                    tp_size=1, sp_size=1
                )

    def tearDown(self):
        set_global_server_args(self._previous_args)


def _encoder():
    """A tiny LongCat text encoder (Qwen2.5-VL that honors the padding mask)."""
    torch.manual_seed(0)
    hf = transformers.Qwen2_5_VLConfig(
        text_config=dict(
            hidden_size=64,
            intermediate_size=96,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=300,
            max_position_embeddings=4096,
            rope_scaling={"type": "mrope", "mrope_section": [2, 3, 3]},
            rope_theta=10000.0,
        ),
        vision_config=dict(
            depth=1,
            hidden_size=32,
            intermediate_size=64,
            num_heads=2,
            out_hidden_size=64,
        ),
    )
    config = Qwen2_5VLConfig()
    for name in ("text_config", "vision_config", "image_token_id", "video_token_id"):
        setattr(config.arch_config, name, getattr(hf, name))
    config.arch_config.vision_start_token_id = hf.vision_start_token_id
    config.arch_config.text_config._attn_implementation = "sdpa"
    config.enable_image_understanding = False
    # LongCat's checkpoint generation config.
    config.generation_config = {
        "do_sample": True,
        "temperature": 0.1,
        "top_k": 1,
        "top_p": 0.001,
        "repetition_penalty": 1.05,
        "eos_token_id": [298, 299],
        "pad_token_id": 0,
    }
    config.honor_cache_free_padding_mask = True
    model = qwen2_5vl.Qwen2_5_VLForConditionalGeneration(config)
    for param in model.parameters():
        param.data.normal_(0, 0.2)
    # Close logits, so the repetition penalty changes the picks.
    model.lm_head.weight.data.normal_(0, 0.002)
    return model.float().eval()


def _generate(model, ids, max_new_tokens, mask=None, **kwargs):
    mask = torch.ones_like(ids) if mask is None else mask
    with set_forward_context(current_timestep=0, attn_metadata=None), torch.no_grad():
        return model.generate(ids, mask, max_new_tokens=max_new_tokens, **kwargs)


class TestLongCatPromptRewrite(_TestBase):
    def test_static_cache_decode_matches_stock_generate(self):
        model = _encoder()
        gen = torch.Generator().manual_seed(0)
        graphs, calls = [], []

        def counting(gm, example_inputs):
            graphs.append(gm)

            def run(*args):
                calls.append(gm)
                return gm.forward(*args)

            return run

        # As in a fresh worker, the TPU path runs before any GEMM backend is picked.
        with patch.object(unquant, "_BF16_GEMM_BACKEND", None):
            for n in (9, 300):
                ids = torch.randint(1, 290, (1, n), generator=gen)
                calls.clear()
                with _installed(counting):
                    actual = _generate(model, ids, 24)
                expected = _generate(model, ids, 24)
                self.assertTrue(torch.equal(actual, expected), (n, actual, expected))
                # The prompt chunks, then one step per token but the last.
                generated = actual.shape[1] - n
                self.assertEqual(len(calls), -(-n // 256) + generated - 1)
        # A 256-token prompt chunk and one token, across prompt lengths.
        self.assertEqual(len(graphs), 2)

    def test_padded_batch_matches_stock_generate(self):
        model = _encoder()
        gen = torch.Generator().manual_seed(2)
        # Real tokens per row: one, some, a whole prompt chunk, the full width.
        width, lengths = 290, (1, 30, 256, 290)
        graphs = []

        def counting(gm, example_inputs):
            graphs.append(gm)
            return gm.forward

        for side in ("left", "right"):
            ids = torch.randint(1, 290, (len(lengths), width), generator=gen)
            mask = torch.ones_like(ids)
            for row, n in enumerate(lengths):
                pad = slice(n, None) if side == "right" else slice(0, width - n)
                ids[row, pad], mask[row, pad] = 0, 0
            free = _generate(model, ids, 24, mask)[:, width:].tolist()

            # A row stops first, at a token the other rows pick later or never
            # (one of them never), so they run on.
            first = [{t: row.index(t) for t in row} for row in free]
            done, stop = next(
                (r, t)
                for r, row in enumerate(free)
                for k, t in enumerate(row[:-1])
                if k > 0
                and first[r][t] == k
                and all(k < f.get(t, 24) for f in first if f is not first[r])
                and any(t not in f for f in first)
            )
            with _installed(counting):
                actual = _generate(model, ids, 24, mask, eos_token_id=stop)
            expected = _generate(model, ids, 24, mask, eos_token_id=stop)
            self.assertTrue(torch.equal(actual, expected), (side, actual, expected))
            self.assertEqual(actual.shape, (len(lengths), width + 24))
            self.assertEqual(actual[done, -1].item(), 0)  # the pad token id
        # One prompt-chunk graph and one token graph serve every row's cache.
        self.assertEqual(len(graphs), 2)
        mask[0] = 0
        with _installed(), self.assertRaisesRegex(ValueError, "a real token"):
            _generate(model, ids, 24, mask)

    def test_padded_prompt_encodes_as_stock(self):
        model = _encoder()
        gen = torch.Generator().manual_seed(1)
        ids = torch.randint(1, 290, (1, 40), generator=gen)
        # LongCat's layout: prefix, a padded body, then the suffix.
        mask = torch.ones(1, 40, dtype=torch.long)
        mask[0, 20:32] = 0

        def encode():
            with set_forward_context(current_timestep=0, attn_metadata=None):
                with torch.no_grad():
                    return model(
                        input_ids=ids, attention_mask=mask, output_hidden_states=True
                    ).hidden_states[-1]

        expected = encode()
        with _installed():
            actual = encode()
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def _dit():
    torch.manual_seed(0)
    arch = LongCatImageArchConfig(
        in_channels=16,
        num_layers=1,
        num_single_layers=1,
        attention_head_dim=16,
        num_attention_heads=2,
        joint_attention_dim=24,
        pooled_projection_dim=24,
        axes_dims_rope=[4, 6, 6],
    )
    model = dit.LongCatImageTransformer2DModel(LongCatImageDitConfig(arch_config=arch))
    for param in model.parameters():
        param.data.normal_(0, 0.1)
    return model.float().eval()


def _offset_guarded(graphs):
    """A compile backend that guards input storage offsets as torch_tpu's does.

    It records each graph's nonzero input offsets; a new offset adds a graph.
    """

    def backend(gm, example_inputs):
        nodes = gm.graph.find_nodes(op="placeholder")
        for node in nodes:
            source = getattr(node, "_dynamo_source", None)
            if source is None or not isinstance(
                node.meta.get("example_value"), torch.Tensor
            ):
                continue
            offset = dynamo_source.TensorPropertySource(
                source, dynamo_source.TensorProperty.STORAGE_OFFSET
            )
            guard = offset.make_guard(dynamo_guards.GuardBuilder.EQUALS_MATCH)
            dynamo_guards.install_guard(guard)
        offsets = {n.name: x.storage_offset() for n, x in zip(nodes, example_inputs)}
        graphs.append({name: o for name, o in offsets.items() if o})
        return gm.forward

    return backend


class TestLongCatDiT(_TestBase):
    def test_host_rope_tables_match_stock(self):
        model = _dit()
        gen = torch.Generator().manual_seed(0)
        h, w = 4, 6
        img_ids = torch.zeros(h, w, 3)
        img_ids[..., 1] += torch.arange(h)[:, None] + 512
        img_ids[..., 2] += torch.arange(w)[None, :] + 512
        img_ids = img_ids.reshape(-1, 3)
        txt_ids = torch.zeros(10, 3)
        txt_ids[:, 1] = txt_ids[:, 2] = torch.arange(10)
        kwargs = dict(
            hidden_states=torch.randn(1, h * w, 16, generator=gen),
            encoder_hidden_states=torch.randn(1, 10, 24, generator=gen),
            timestep=torch.full((1,), 500.0),
            img_ids=img_ids,
            txt_ids=txt_ids,
        )
        with set_forward_context(current_timestep=0, attn_metadata=None):
            with torch.no_grad():
                expected = model(**kwargs)
                with _installed():
                    actual = model(**kwargs)
                    cached = longcat_image._rope_tables(model, txt_ids, img_ids, "cpu")
        stock = model.pos_embed(torch.cat((txt_ids, img_ids)))
        for a, e in zip(cached, stock, strict=True):
            self.assertTrue(torch.equal(a, e))
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)

    def test_sp_rope_tables_follow_the_latent_shard(self):
        model = _dit()
        config = longcat_config.LongCatImagePipelineConfig()
        gen = torch.Generator().manual_seed(0)
        latent_ids = config.maybe_prepare_latent_ids(torch.zeros(1, 4, 8, 16))
        hidden = torch.randn(1, 32, 16, generator=gen)
        embeds = torch.randn(1, 10, 24, generator=gen)
        batch = SimpleNamespace(
            enable_sequence_shard=False,
            latent_ids=latent_ids,
            prompt_embeds=[embeds],
            negative_prompt_embeds=[embeds],
        )

        def run(kwargs):
            with set_forward_context(current_timestep=0, attn_metadata=None):
                with torch.no_grad():
                    return model(
                        hidden_states=kwargs.pop("hidden_states"),
                        encoder_hidden_states=embeds,
                        timestep=torch.full((1,), 500.0),
                        **kwargs,
                    )

        # Without SP the DiT gets the latent ids themselves.
        pos = config.prepare_pos_cond_kwargs(batch, "cpu", None, torch.float32)
        neg = config.prepare_neg_cond_kwargs(batch, "cpu", None, torch.float32)
        self.assertIs(pos["img_ids"], latent_ids)
        self.assertIs(neg["img_ids"], latent_ids)
        stock = model.pos_embed(torch.cat((pos["txt_ids"], latent_ids)))
        for rank in (0, 5):
            with _sp_rank(rank):
                local, sharded = config.shard_latents_for_sp(batch, hidden)
                pos = config.prepare_pos_cond_kwargs(batch, "cpu", None, torch.float32)
                neg = config.prepare_neg_cond_kwargs(batch, "cpu", None, torch.float32)
            rows = slice(4 * rank, 4 * rank + 4)
            self.assertTrue(sharded)
            self.assertTrue(torch.equal(local, hidden[:, rows]))
            for kwargs in (pos, neg):
                self.assertTrue(torch.equal(kwargs["img_ids"], latent_ids[rows]))
            expected = run(dict(pos, hidden_states=local))
            with _installed():
                # The plugin builds each branch's tables once per request.
                for _ in range(2):
                    actual = run(dict(pos, hidden_states=local))
                    run(dict(neg, hidden_states=local))
                self.assertEqual(len(longcat_image._ROPE_CACHE), 2)
                tables = longcat_image._rope_tables(
                    model, pos["txt_ids"], pos["img_ids"], "cpu"
                )
                # Both branches' shards hold the same ids, so share one table.
                neg_tables = longcat_image._rope_tables(
                    model, neg["txt_ids"], neg["img_ids"], "cpu"
                )
                self.assertIsNot(pos["img_ids"], neg["img_ids"])
                self.assertIs(neg_tables[0], tables[0])
            # The text rows, then this rank's image rows of the full tables.
            for t, full in zip(tables, stock, strict=True):
                self.assertTrue(torch.equal(t[:10], full[:10]))
                self.assertTrue(torch.equal(t[10:], full[10:][rows]))
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)

    def test_compiled_dit_inputs_start_at_offset_zero(self):
        # The prompt embeddings are a slice of the encoder output on a
        # conditioning-cache miss, and a copy on a hit.
        model = _dit().to(torch.bfloat16)
        config = longcat_config.LongCatImagePipelineConfig()
        config._encode_prefix_len, config._encode_suffix_len = 3, 2
        gen = torch.Generator().manual_seed(0)
        hidden = torch.randn(1, 15, 24, generator=gen).bfloat16()

        def encode():
            outputs = SimpleNamespace(hidden_states=[hidden])
            post = longcat_config.longcat_postprocess_text(outputs, None, config)
            return post.prompt_embeds

        cache = ConditioningCache(1 << 20)
        with cache.scope(cross_request=True):
            miss, hit = (cache.run(model, "encode", (), {}, encode) for _ in "ab")
        self.assertEqual((miss.storage_offset(), hit.storage_offset()), (72, 0))
        latents = torch.randn(1, 32, 16, generator=gen)
        batch = SimpleNamespace(
            enable_sequence_shard=False,
            latent_ids=config.maybe_prepare_latent_ids(torch.zeros(1, 4, 8, 16)),
            prompt_embeds=[miss],
            negative_prompt_embeds=[hit],
        )
        torch._dynamo.reset()
        graphs = []
        model.compile(backend=_offset_guarded(graphs))
        with _installed(), set_forward_context(0, None), torch.no_grad():
            for rank in (0, 5):
                with _sp_rank(rank):
                    local, _ = config.shard_latents_for_sp(batch, latents)
                    pos = config.prepare_pos_cond_kwargs(batch, "cpu", None, None)
                    neg = config.prepare_neg_cond_kwargs(batch, "cpu", None, None)
                pos["encoder_hidden_states"] = config.get_pos_prompt_embeds(batch)
                neg["encoder_hidden_states"] = config.get_neg_prompt_embeds(batch)
                for t in (1000.0, 500.0):  # two denoising steps, both CFG branches
                    for kwargs in (pos, neg):
                        model(
                            hidden_states=local.to(torch.bfloat16),
                            timestep=torch.full((1,), t, dtype=torch.bfloat16),
                            **kwargs,
                        )
        torch._dynamo.reset()
        # One graph for both branches, steps and ranks, all inputs at offset 0.
        self.assertEqual(graphs, [{}])

    def test_sp_rejects_uneven_t2i_tokens(self):
        batch = SimpleNamespace(enable_sequence_shard=False)
        latents = torch.zeros(1, 9, 16)
        with _sp_rank(0):
            with self.assertRaisesRegex(ValueError, "do not split evenly"):
                longcat_config.LongCatImagePipelineConfig().shard_latents_for_sp(
                    batch, latents
                )
            # Edit pads its tokens to a multiple of the SP degree first.
            edit = longcat_config.LongCatImageEditPipelineConfig()
            local, sharded = edit.shard_latents_for_sp(batch, latents)
        self.assertTrue(sharded)
        self.assertEqual(local.shape, (1, 2, 16))


if __name__ == "__main__":
    import unittest

    unittest.main()
