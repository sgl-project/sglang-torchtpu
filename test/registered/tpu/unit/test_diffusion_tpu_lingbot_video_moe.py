# SPDX-License-Identifier: Apache-2.0
"""Tests for sglang_tpu's LingBot-Video-MoE patches (sglang_tpu.multimodal_gen.lingbot_video_moe)."""

import contextlib
import functools
import importlib.util
import sys
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F
from transformers.models.qwen3_vl.configuration_qwen3_vl import (
    Qwen3VLConfig as HFQwen3VLConfig,
)

from sglang.multimodal_gen.configs.models.dits.lingbot_video_moe import (
    LingBotVideoMoEArchConfig,
    LingBotVideoMoEConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs import base as pipeline_base
from sglang.multimodal_gen.configs.pipeline_configs.lingbot_video_moe import (
    LingBotVideoMoEPipelineConfig,
)
from sglang.multimodal_gen.configs.sample.lingbot_video_moe import (
    LingBotVideoMoESamplingParams,
)
from sglang.multimodal_gen.runtime import server_args as server_args_module
from sglang.multimodal_gen.runtime import warmup_request_builder
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    maybe_init_distributed_environment_and_model_parallel,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.layers.moe import (
    LingBotVideoRouter,
    LingBotVideoSparseMoeBlock,
)
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.dits import lingbot_video_moe as dit
from sglang.multimodal_gen.runtime.models.encoders.qwen3vl import (
    Qwen3VLForConditionalGeneration,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.lingbot_video_moe.text_encoding import (
    LingBotVideoTextEncodingStage,
)
from sglang.multimodal_gen.runtime.server_args import set_global_server_args
from sglang.multimodal_gen.test.single_test_file.component_accuracy.utils import (
    ensure_distributed_env_defaults,
)
from sglang.multimodal_gen.test.unit.conftest import _make_unit_server_args
from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.utils import is_in_ci
from sglang_tpu.multimodal_gen import lingbot_video_moe, worker

# Nightly, one job on a whole node: the parallel-state setup runs the TPU worker
# bootstrap, which opens a chip outside a chips=1 pin.
register_tpu_ci(est_time=120, suite="nightly-tpu-4-v7", nightly=True)

_TRITON_MODULE = "sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe"


def _moe(x, w1, w2, weights, ids):
    """Routed experts with (experts, in, out) weights, summed in float64."""
    out = 0
    for slot in range(ids.shape[1]):
        expert = ids[:, slot].long()
        h = torch.einsum("th,thn->tn", x.double(), w1[expert].double())
        gate, up = h.chunk(2, -1)
        down = torch.einsum("ti,tih->th", F.silu(gate) * up, w2[expert].double())
        out = out + weights[:, slot, None].double() * down
    return out.to(x.dtype)


def _fused_reference(*, hidden_states, w1, w2, topk_weights, topk_ids, **kw):
    """The Pallas ``fused_moe_call`` contract: SwiGLU over gate || up."""
    assert kw == {"topk": topk_ids.shape[1], "activation": "silu"}, kw
    return _moe(hidden_states, w1, w2, topk_weights, topk_ids)


def _triton_reference(x, w13, w2, topk_output, _config):
    """srt's Triton ``fused_experts`` contract: (experts, out, in) weights."""
    weights, ids = topk_output.topk_weights, topk_output.topk_ids
    return _moe(x, w13.transpose(1, 2), w2.transpose(1, 2), weights, ids)


def _held_experts(self, tokens, top_scores, top_indices):
    """The routed experts from either weight layout, the same on both paths."""
    w13, w2 = self.experts.w13_weight, self.experts.w2
    if not getattr(self.experts, "_tpu_layout", False):
        w13, w2 = w13.transpose(1, 2), w2.transpose(1, 2)
    return _moe(tokens, w13, w2, top_scores, top_indices)


_holding_experts = functools.partial(
    patch.object,
    LingBotVideoSparseMoeBlock,
    "_run_sglang_triton_experts",
    _held_experts,
)


_PATCHED = (
    (LingBotVideoSparseMoeBlock, "_run_sglang_triton_experts"),
    (dit.LingBotVideoAttention, "forward"),
    (dit.LingBotVideoTransformer3DModel, "forward"),
    (dit.LingBotVideoTransformer3DModel, "_compile_regions_spmd_safe"),
    (LingBotVideoMoEPipelineConfig, "shard_latents_for_sp"),
    (LingBotVideoMoESamplingParams, "__post_init__"),
    (LingBotVideoTextEncodingStage, "_encode_inputs"),
)


@contextlib.contextmanager
def _on_cpu(held_experts=False):
    """The TPU paths on CPU (the reference MoE above, 64-token buckets); yields
    the device drains, op builds and TPU metadata enters and exits, in order."""
    from sglang_tpu.moe.pallas import fused_moe_op

    events = []

    @contextlib.contextmanager
    def tpu_metadata():
        events.append("enter")
        yield
        events.append("exit")

    with contextlib.ExitStack() as stack:
        for owner, name in _PATCHED:
            stack.enter_context(patch.object(owner, name, getattr(owner, name)))
        for owner, name, value in (
            (lingbot_video_moe, "_DEVICE_TYPES", ("cpu",)),
            (lingbot_video_moe, "SEQUENCE_ALIGNMENT", 64),
            (lingbot_video_moe, "_tpu_metadata", tpu_metadata),
            (fused_moe_op, "fused_moe_call", _fused_reference),
            (fused_moe_op, "_build_fused_moe_op", lambda: events.append("build")),
            (torch.accelerator, "synchronize", lambda: events.append("drain")),
        ):
            stack.enter_context(patch.object(owner, name, value))
        lingbot_video_moe._layout.cache_clear()
        stack.callback(lingbot_video_moe._layout.cache_clear)
        lingbot_video_moe.install()
        lingbot_video_moe.install()  # idempotent
        if held_experts:
            stack.enter_context(_holding_experts())
        yield events


def _dit(depth=2):
    torch.manual_seed(0)
    arch = LingBotVideoMoEArchConfig(
        hidden_size=256,
        num_attention_heads=2,  # head_dim 128 = 32 + 48 + 48
        depth=depth,
        text_dim=64,
        num_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=64,
        n_group=2,
        topk_group=1,
    )
    model = dit.LingBotVideoTransformer3DModel(
        LingBotVideoMoEConfig(arch_config=arch), hf_config={}
    )
    for param in model.parameters():
        param.data.normal_(0, 0.05)
    for module in model.modules():
        if isinstance(module, dit.LingBotVideoRMSNorm):
            # Unit gains, as initialized: the attention stays peaked, so the
            # positions matter.
            module.weight.data.fill_(1.0)
    return model.float().eval()


def _dit_inputs(text_len):
    # Latent grid (2, 4, 4): 32 video tokens.
    gen = torch.Generator().manual_seed(text_len)
    return dict(
        hidden_states=torch.randn(1, 16, 2, 8, 8, generator=gen),
        timestep=torch.full((1,), 500.0),
        encoder_hidden_states=torch.randn(1, text_len, 64, generator=gen),
    )


def _forward_batch(inputs, negative_len):
    return SimpleNamespace(
        prompt_embeds=[inputs["encoder_hidden_states"]],
        negative_prompt_embeds=[torch.zeros(1, negative_len, 64)],
    )


def _run(model, inputs, forward_batch=None):
    with set_forward_context(0, None, forward_batch), torch.no_grad():
        return model(**inputs)


class _ThreadRanks:
    """SP ranks as threads of one process; collectives meet at a barrier."""

    def __init__(self, sp):
        self.sp, self.slots = sp, [None] * sp
        self.barrier, self.local = threading.Barrier(sp, timeout=60), threading.local()

    def rank(self):
        return self.local.rank

    def _all(self, x):
        self.slots[self.rank()] = x
        self.barrier.wait()
        parts = list(self.slots)
        self.barrier.wait()
        return parts

    def input_a2a(self, *qkv):  # [1, S / sp, H, D] -> [1, S, H / sp, D]
        heads = qkv[0].shape[2] // self.sp
        own = slice(self.rank() * heads, (self.rank() + 1) * heads)
        return [torch.cat([p[:, :, own] for p in self._all(t)], 1) for t in qkv]

    def output_a2a(self, x, head_dim):  # [1, S, H / sp, D] -> [1, S / sp, H, D]
        rows = x.shape[1] // self.sp
        own = slice(self.rank() * rows, (self.rank() + 1) * rows)
        return torch.cat([p[:, own] for p in self._all(x)], head_dim)

    def gather(self, x, length):
        return torch.cat(self._all(x), 1)[:, :length]

    def run(self, fn):
        def rank_fn(rank):
            self.local.rank = rank
            with torch.no_grad():
                return fn()

        with ThreadPoolExecutor(self.sp) as pool:
            futures = [pool.submit(rank_fn, rank) for rank in range(self.sp)]
            return [future.result() for future in futures]


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


class TestLingBotVideoMoEExperts(_TestBase):
    def test_experts_adapter_matches_the_stock_triton_call(self):
        block = LingBotVideoSparseMoeBlock(256, 64, 8, 2, "sigmoid", True, 2, 1, 2.5, 1)
        torch.manual_seed(1)
        for param in block.parameters():
            param.data.normal_(0, 0.05)
        block = block.bfloat16()
        tokens = torch.randn(24, 256).bfloat16()
        top_indices, top_scores = block.router(tokens)
        triton = ModuleType(_TRITON_MODULE)
        triton.fused_experts = _triton_reference
        with patch.dict(sys.modules, {_TRITON_MODULE: triton}):
            expected = block._run_sglang_triton_experts(tokens, top_scores, top_indices)
        with _on_cpu() as events:
            with self.assertRaisesRegex(RuntimeError, "TPU layout"):
                block._run_sglang_triton_experts(tokens, top_scores, top_indices)
            lingbot_video_moe._to_tpu_layout(block)
            lingbot_video_moe._to_tpu_layout(block)  # moves the weights once
            actual = block._run_sglang_triton_experts(tokens, top_scores, top_indices)
        self.assertEqual(events, ["drain"])
        self.assertEqual(tuple(block.experts.w13_weight.shape), (8, 256, 128))
        self.assertEqual(tuple(block.experts.w2.shape), (8, 64, 256))
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=0)


class TestLingBotVideoMoEDiT(_TestBase):
    def test_host_rope_rows_match_the_stock_table(self):
        rotary = _dit(depth=1).rotary_emb
        # 32 video and 5 text tokens in a 128-token bucket.
        grid, n_real = (2, 4, 4), 37
        positions = dit._joint_position_ids(torch.tensor([5]), *grid, 5, "cpu")
        stock = rotary.forward_uncached(positions)
        cos, sin, kv_info = lingbot_video_moe._layout(rotary, grid, 5, 128, 1, 0, "cpu")
        self.assertTrue(torch.equal(cos[:n_real], stock[0]))
        self.assertTrue(torch.equal(sin[:n_real], stock[1]))
        # Padding rows sit at position (0, 0, 0).
        self.assertTrue(bool((cos[n_real:] == 1).all() and (sin[n_real:] == 0).all()))
        self.assertEqual((kv_info.tolist(), kv_info.dtype), ([0, n_real], torch.int32))

    def test_cfg_branches_share_a_512_token_bucket(self):
        # 480x480x81: 18900 video tokens. The CI prompt and the default negative
        # share a bucket; either branch's longer prompt sets the other's too.
        cases = ((249, 234, 19456), (1262, 234, 20480), (249, 1262, 20480))
        for prompt, negative, expected in cases:
            batch = SimpleNamespace(
                prompt_embeds=[torch.zeros(1, prompt)],
                negative_prompt_embeds=[torch.zeros(1, negative)],
            )
            with set_forward_context(0, None, forward_batch=batch):
                buckets = {
                    lingbot_video_moe._bucket(18900, length, sp)
                    for length in (prompt, negative)
                    for sp in (1, 2, 4, 8)
                }
            self.assertEqual(buckets, {expected})

    def test_bucketed_forward_matches_stock(self):
        model = _dit()
        # (prompt, negative) lengths over 32 video tokens: two in the 64-token
        # bucket, then two in the 128-token bucket either branch can set.
        for text_len, negative_len in ((5, 5), (20, 5), (5, 40), (40, 5)):
            inputs = _dit_inputs(text_len)
            batch = _forward_batch(inputs, negative_len)
            with _holding_experts():
                expected = _run(model, inputs, batch)
            with _on_cpu(held_experts=True):
                actual = _run(model, inputs, batch)
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
        self.assertTrue(model.blocks[0].ffn.experts._tpu_layout)

    def test_blocks_run_under_the_tpu_metadata_after_the_op_is_built(self):
        model = _dit(depth=1)
        with _on_cpu() as events:
            self.assertTrue(model._compile_regions_spmd_safe)
            model.blocks[0].register_forward_pre_hook(lambda *_: events.append("block"))
            _run(model, _dit_inputs(5))
            args = server_args_module.get_global_server_args()
            args.enable_torch_compile, args.regional_compile = True, False
            with self.assertRaisesRegex(ValueError, "--regional-compile"):
                _run(model, _dit_inputs(5))
        self.assertEqual(events, ["drain", "build", "enter", "block", "exit"])

    def test_sequence_parallel_ranks_match_one_rank(self):
        model = _dit()
        inputs = _dit_inputs(20)
        batch = _forward_batch(inputs, 5)
        with _holding_experts():
            expected = _run(model, inputs, batch)
        lingbot_video_moe._to_tpu_layout(model)  # before the rank threads share it
        ranks = _ThreadRanks(2)
        with _on_cpu(held_experts=True), contextlib.ExitStack() as stack:
            for name, value in (
                ("get_sp_world_size", lambda: ranks.sp),
                ("get_sp_parallel_rank", ranks.rank),
                ("_usp_input_all_to_all_qkv", ranks.input_a2a),
                ("_usp_output_all_to_all", ranks.output_a2a),
                ("gather_seq", ranks.gather),
            ):
                stack.enter_context(patch.object(lingbot_video_moe, name, value))
            with set_forward_context(0, None, forward_batch=batch):
                outputs = ranks.run(lambda: model(**inputs))
        for output in outputs:
            torch.testing.assert_close(output, expected, atol=1e-5, rtol=1e-4)

    def test_latents_stay_whole_under_sequence_parallelism(self):
        config = LingBotVideoMoEPipelineConfig()
        latents = torch.randn(1, 16, 4, 2, 2)
        batch = SimpleNamespace(enable_sequence_shard=False)
        with (
            patch.object(pipeline_base, "get_sp_world_size", return_value=2),
            patch.object(pipeline_base, "get_sp_parallel_rank", return_value=1),
        ):
            # The stock shards the 4 latent frames over 2 ranks.
            self.assertTrue(config.shard_latents_for_sp(batch, latents)[1])
            with _on_cpu():
                whole, did_shard = config.shard_latents_for_sp(batch, latents)
        self.assertIs(whole, latents)
        self.assertFalse(did_shard)

    def test_requests_and_the_warmup_keep_their_frame_count(self):
        args = SimpleNamespace(num_gpus=8, output_path=None, comfyui_mode=True)
        args.pipeline_config = LingBotVideoMoEPipelineConfig()
        contract = warmup_request_builder._apply_warmup_frame_contract
        # The stock rounds 21 latent frames up to 24 for 8 GPUs: 93 frames.
        for installed, frames in ((contextlib.nullcontext(), 93), (_on_cpu(), 81)):
            with installed:
                params = LingBotVideoMoESamplingParams()
                params._adjust(args)
                warmup = contract(args, LingBotVideoMoESamplingParams(), num_frames=81)
            self.assertEqual((params.num_frames, warmup), (frames, frames))

    def test_block_regions_compile_once_per_bucket(self):
        model = _dit()
        graphs = []

        def counting(gm, example_inputs):
            graphs.append(gm)
            return gm.forward

        with _on_cpu():
            _run(model, _dit_inputs(5))  # resolves lazy kernel-registry state
            for block in model.blocks:
                block.compile(backend=counting, dynamic=False)
            for text_len in (5, 20):  # the 64-token bucket, both blocks
                _run(model, _dit_inputs(text_len))
            self.assertEqual(len(graphs), 1)
            _run(model, _dit_inputs(40))  # the 128-token bucket
        self.assertEqual(len(graphs), 2)


class _Encoding(dict):
    def to(self, _device):
        return self


def _text_encoder():
    text = dict(hidden_size=64, intermediate_size=128, num_hidden_layers=2)
    text.update(num_attention_heads=4, num_key_value_heads=2, head_dim=16)
    rope = dict(rope_type="default", mrope_section=[2, 3, 3])
    text.update(vocab_size=64, rope_scaling=rope)
    vision = dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=4)
    vision.update(out_hidden_size=64, deepstack_visual_indexes=[])
    config = HFQwen3VLConfig(text_config=text, vision_config=vision)
    config._fsdp_shard_conditions = []
    config.stacked_params_mapping = []
    torch.manual_seed(0)
    model = Qwen3VLForConditionalGeneration(
        SimpleNamespace(arch_config=config, quant_config=None)
    )
    for param in model.parameters():
        param.data.normal_(0, 0.02)
    return model.eval()


class TestLingBotVideoMoETextEncoder(_TestBase):
    def test_padded_text_encoder_matches_stock(self):
        stage = object.__new__(LingBotVideoTextEncodingStage)
        stage._component_residency_manager = None
        stage.text_encoders, stage.tokenizers = [_text_encoder()], [None]
        stage.hidden_state_skip_layer, stage._crop_start = 0, 3
        calls = []
        forward = stage._forward_text_encoder
        stage._forward_text_encoder = lambda enc, kw: (
            calls.append(kw) or forward(enc, kw)
        )
        cpu = torch.device("cpu")
        gen = torch.Generator().manual_seed(0)
        # One and two 256-token buckets; the second prompt has a padded mask.
        for length, real in ((40, 40), (300, 290)):
            inputs = _Encoding(
                input_ids=torch.randint(1, 64, (1, length), generator=gen),
                attention_mask=(torch.arange(length) < real).long()[None],
            )
            expected = stage._encode_inputs(inputs, cpu, torch.float32)
            with _on_cpu():
                actual = stage._encode_inputs(inputs, cpu, torch.float32)
            kw = calls[-1]
            ids, positions = kw["input_ids"], kw["position_ids"]
            self.assertEqual((ids.dtype, positions.dtype), (torch.int32,) * 2)
            self.assertEqual(positions.shape, (3, 1, ids.shape[1]))
            self.assertEqual(ids.shape[1] % 256, 0)
            options = (kw["attention_mask"], kw["use_cache"], kw["logits_to_keep"])
            self.assertEqual(options, (None, False, 1))
            self.assertEqual(actual[0].shape, (1, real - 3, 64))
            torch.testing.assert_close(actual[0], expected[0], atol=1e-5, rtol=1e-4)
            self.assertTrue(torch.equal(actual[1], expected[1]))


def _moe_float32(x, w13, w2, weights, ids):
    out = torch.zeros_like(x)
    for expert in range(w13.shape[0]):
        token, slot = (ids == expert).nonzero(as_tuple=True)
        gate, up = (x[token] @ w13[expert].T).chunk(2, -1)
        down = (F.silu(gate) * up) @ w2[expert].T
        out.index_add_(0, token, weights[token, slot, None] * down)
    return out


class TestLingBotVideoMoEOnTpu(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not (
            importlib.util.find_spec("torch_tpu") and torch.accelerator.device_count()
        ):
            if is_in_ci():
                raise RuntimeError("TPU CI requires torch_tpu and a TPU device")
            raise unittest.SkipTest("requires a TPU")
        from sglang_tpu.env_override import apply_tpu_env_overrides

        apply_tpu_env_overrides()
        import torch_tpu._internal.compile  # noqa: F401

        torch.zeros(1, device="tpu").cpu()

    def test_fused_moe_matches_float32_at_lingbot_shapes(self):
        from sglang_tpu.moe.pallas import fused_moe_op

        experts, hidden, inter, top_k, tokens = 128, 2048, 768, 8, 4864
        block = LingBotVideoSparseMoeBlock(
            hidden, inter, experts, top_k, "sigmoid", True, 4, 2, 2.5, 1
        )
        gen = torch.Generator().manual_seed(0)
        w13 = (torch.randn(experts, 2 * inter, hidden, generator=gen) * 0.02).bfloat16()
        w2 = (torch.randn(experts, hidden, inter, generator=gen) * 0.02).bfloat16()
        x = torch.randn(tokens, hidden, generator=gen).bfloat16()
        weights = torch.rand(tokens, top_k, generator=gen).bfloat16()
        ids = torch.rand(tokens, experts, generator=gen).argsort(-1)[:, :top_k]
        expected = _moe_float32(
            x.float(), w13.float(), w2.float(), weights.float(), ids
        )
        block.experts.w13_weight.data, block.experts.w2.data = w13, w2
        block = block.to("tpu")
        lingbot_video_moe._to_tpu_layout(block)
        # Compiled, as inside a block region.
        compiled = torch.compile(
            lambda *args: lingbot_video_moe.run_experts(block, *args),
            backend="tpu",
            dynamic=False,
        )
        # No autograd, as the workers serve: the fused MoE op has no backward.
        with lingbot_video_moe._tpu_metadata(), torch.no_grad():
            fused_moe_op._build_fused_moe_op()
            actual = compiled(x.to("tpu"), weights.to("tpu"), ids.int().to("tpu"))
        actual = actual.float().cpu()
        self.assertTrue(torch.isfinite(actual).all())
        error = (actual - expected).norm() / expected.norm()
        self.assertLess(error.item(), 5e-2)

    def test_compiled_router_matches_the_host(self):
        router = LingBotVideoRouter(2048, 128, 8, "sigmoid", True, 4, 2, 2.5)
        gen = torch.Generator().manual_seed(1)
        router.weight.data = torch.randn(128, 2048, generator=gen) * 0.02
        router.e_score_correction_bias.data = torch.randn(128, generator=gen) * 0.01
        tokens = torch.randn(4864, 2048, generator=gen).bfloat16()
        ids, scores = router(tokens)
        # The worker's float32 matmul precision.
        precision = torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision("highest")
        self.addCleanup(torch.set_float32_matmul_precision, precision)
        compiled = torch.compile(router.to("tpu"), backend="tpu", dynamic=False)
        tpu_ids, tpu_scores = (t.cpu() for t in compiled(tokens.to("tpu")))
        tpu_ids, tpu_order = tpu_ids.long().sort(-1)
        ids, order = ids.sort(-1)
        same = (tpu_ids == ids).all(-1)
        self.assertGreaterEqual(same.float().mean().item(), 0.999)
        torch.testing.assert_close(
            tpu_scores.gather(-1, tpu_order)[same].float(),
            scores.gather(-1, order)[same].float(),
            atol=1e-2,
            rtol=1e-2,
        )


if __name__ == "__main__":
    unittest.main()
