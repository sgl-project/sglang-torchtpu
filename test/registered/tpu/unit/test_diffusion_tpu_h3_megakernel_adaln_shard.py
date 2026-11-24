# SPDX-License-Identifier: Apache-2.0
"""CPU tests for SGLANG_TPU_OPT_H3_MEGAKERNEL_SHARD_ADALN (h3_megakernel).

With the option every rank keeps the AdaLN projections of a contiguous share
of the blocks, frees the others' weights at load, and each step all-gathers
the tables over the sequence-parallel group's device group. The tests check,
torch only:

- the block ranges for every block and rank count up to 64 x 16;
- the tables, the stack's operands and the op inputs of its forward bitwise
  against the replicated path, on 2, 4 and 8 ranks: one process a rank over
  sglang's own groups on gloo (``all_gather_into_tensor``, the call the hook
  makes on TPU's ``tpu_dist``), for 50 blocks (25 a rank; 13, 13, 13, 11;
  seven 7s and a 1) and 6 blocks over 4 ranks (a rank with none); and in one
  process through a gather that concatenates every rank's input in rank
  order (the all-gather's contract);
- ``enable`` end to end on those ranks: the vote, the freed weights (storage
  weak references, parameter bytes, ``state_dict``), the startup check of the
  all-gather on the device, packing, and, flag off against flag on, the op
  inputs bit for bit and the same all-gathers in the same order on every
  rank for plain, vouched, static-structure (with JAX) and out-of-request
  forwards and for requests through the denoise-loop wrapper (warmup and
  measured);
- the startup check: a rank failing anywhere before the all-gather (probe,
  replicated tables, its share, a device error at the sync) or seeing wrong
  tables from a peer makes every rank raise within seconds, on 2 and 4
  ranks, and no rank enters the all-gather after a failure before it; in one
  process, wrong tables of the rank's own blocks or of a peer's are refused;
- with the lean cut lever and the static structure (with JAX), ``enable`` as
  one rank of 2 and of 8 at the lever's row counts: the op inputs and the
  all-gathers bit for bit as with the option alone, plain and vouched; the
  lever reaches only the op's build and the static structure's cut tile;
- with the option set on some ranks only, every rank leaves the kernel off;
- off, the replicated path's calls exactly; on one rank, a no-op;
- with the FL2VA checkpoint (a directory in ``MINIMAX_H3_CHECKPOINT``), the
  tables of four real blocks' projections at H3's shapes, sharded 2, 3, 4 and
  8 ways.
"""

import contextlib
import gc
import importlib.util
import json
import logging
import os
import socket
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

# Host-CPU tests: in the TPU image the diffusion platform and JAX would
# otherwise select the TPU.
os.environ.setdefault("SGLANG_DIFFUSION_PLATFORM_OVERRIDE", "cpu")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import torch
from torch.multiprocessing.reductions import StorageWeakRef

from sglang.multimodal_gen.configs.models.dits.minimax_h3 import (
    MINIMAX_H3_ADALN_MODALITY_NUM,
    MiniMaxH3DiTArchConfig,
    MiniMaxH3DiTConfig,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_sp_group,
    get_world_group,
    maybe_init_distributed_environment_and_model_parallel,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.models.dits import minimax_h3 as dit
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3 import (
    packed_sequence,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages.denoising import (
    MiniMaxH3DenoisingStage,
)
from sglang.multimodal_gen.test.single_test_file.component_accuracy.utils import (
    ensure_distributed_env_defaults,
)
from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase

# Five multi-process runs (2, 4 and 8 ranks; failures on 2 and 4), up to about
# two minutes each on 4 cores.
register_tpu_ci(
    est_time=300,
    suite="nightly-tpu-4-v7",
    nightly=True,
    labels=["singlehost", "diffusion"],
)

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir, os.pardir)
)
# sglang_tpu from the checkout where it is not installed.
if _REPO_ROOT not in sys.path:
    sys.path.append(_REPO_ROOT)

from sglang_tpu.envs import envs as tpu_envs  # noqa: E402
from sglang_tpu.multimodal_gen.h3_megakernel import hook, pack  # noqa: E402

_HAS_JAX = importlib.util.find_spec("jax") is not None
_CKPT = os.environ.get("MINIMAX_H3_CHECKPOINT", "")
_TRANSFORMER = os.path.join(_CKPT, "transformer")
_HAS_CKPT = bool(_CKPT) and os.path.isfile(
    os.path.join(_TRANSFORMER, "model.safetensors.index.json")
)
_BF16 = torch.bfloat16
_SHARD = tpu_envs.SGLANG_TPU_OPT_H3_MEGAKERNEL_SHARD_ADALN
# The tiny arch's tile widths (pack.tile_widths imports JAX).
_TILES = {"qkv_heads": 2, "out_cols": 256, "ffn_cols": 256, "down_cols": 256}
# A hung collective fails the run instead of hanging it (sglang's groups keep
# gloo's 30-minute default; a normal 8-rank run takes a few minutes).
_DIST_TIMEOUT_S = 120
_SPAWN_TIMEOUT_S = 600
# The failure runs: a few small cases, seconds each; a hang (a rank in a
# collective its peers skipped) sits in sglang's untimed subgroups.
_FAIL_SPAWN_TIMEOUT_S = 240
# Each rank of a failure case must have raised within this.
_FAIL_RAISE_S = 60


def _arch(layers: int, heads: int = 2) -> MiniMaxH3DiTArchConfig:
    hidden = 256
    return MiniMaxH3DiTArchConfig(
        num_layers=layers,
        token_refiner_num_layers=1,
        hidden_size=hidden,
        num_attention_heads=heads,
        attention_head_dim=128,
        ffn_hidden_size=512,
        text_dim=64,
        time_embed_hidden_size=64,
        time_embed_dim=64,
        adaln_out_features=18 * hidden,
        final_adaln_out_features=2 * hidden,
    )


def _init_parallel():
    if not model_parallel_is_initialized():
        ensure_distributed_env_defaults()
        maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)


def _projs(arch, layers: int, seed: int = 0) -> list:
    """``layers`` AdaLN projections with distinct random weights (bf16)."""
    torch.manual_seed(seed)
    projs = []
    for index in range(layers):
        proj = dit.MiniMaxH3AdalnProj(
            arch,
            arch.adaln_out_features,
            None,
            prefix=f"blocks.{index}.adaln_proj",
            expand_ratio=6,
            modality_num=MINIMAX_H3_ADALN_MODALITY_NUM,
        ).requires_grad_(False)
        proj.linear.weight.data.normal_(0, 0.08)
        proj.linear.bias.data.normal_(0, 0.3)
        projs.append(proj)
    return projs


def _adaln_input(arch, timesteps: int = 2, seed: int = 1) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    embedding = torch.randn(timesteps, arch.time_embed_dim, generator=generator)
    return torch.nn.functional.silu(embedding).to(_BF16)


def _table_shape(proj) -> tuple[int, int, int]:
    return (proj.expand_ratio, proj.modality_num, proj.hidden_size)


def _kernel_config(arch, layers: int):
    """``pack.config_from_arch`` without JAX (a stand-in ``Config``)."""
    if _HAS_JAX:
        return pack.config_from_arch(arch, layers)
    return SimpleNamespace(
        hidden=arch.hidden_size,
        heads=arch.num_attention_heads,
        head_dim=arch.attention_head_dim,
        ffn=arch.ffn_hidden_size,
        layers=layers,
        rope_dims=6 * arch.rope_inv_freq_len,
        width=arch.num_attention_heads * arch.attention_head_dim,
    )


def _bits(tensor: torch.Tensor) -> torch.Tensor:
    """bf16 as its raw int16 bits (so -0.0 and +0.0 differ); others as is."""
    tensor = tensor.detach().contiguous()
    return tensor.view(torch.int16) if tensor.dtype == _BF16 else tensor


def _same(a, b) -> bool:
    """Bitwise equality of tensors (shape, dtype and bits) or plain values."""
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        return (
            isinstance(a, torch.Tensor)
            and isinstance(b, torch.Tensor)
            and a.shape == b.shape
            and a.dtype == b.dtype
            and torch.equal(_bits(a), _bits(b))
        )
    return a == b


def _same_args(a, b) -> bool:
    return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))


def _bare_model():
    model = dit.MiniMaxH3DiTModel.__new__(dit.MiniMaxH3DiTModel)
    torch.nn.Module.__init__(model)
    model.blocks = torch.nn.ModuleList()
    model.adaln_cache = None
    model.adaln_t_table = None
    model._adaln_precomputed = False
    return model


def _tiny_model(arch, seed: int = 0):
    """An H3 DiT with ``arch.num_layers`` random blocks (same seed, same
    weights)."""
    torch.manual_seed(seed)
    model = dit.MiniMaxH3DiTModel(
        config=MiniMaxH3DiTConfig(arch_config=arch), hf_config={}, quant_config=None
    )
    model.requires_grad_(False)
    for param in model.parameters():
        param.data.normal_(0, 0.02)
    for block in model.blocks:
        block.adaln_proj.linear.weight.data.normal_(0, 0.08)
        block.adaln_proj.linear.bias.data.normal_(0, 0.3)
        block.attn.q_norm.weight.data.uniform_(0.5, 1.5)
        block.attn.k_norm.weight.data.uniform_(0.5, 1.5)
    return model


def _block_model(arch, seed: int = 0):
    """A DiT holding ``arch.num_layers`` random sglang blocks (same seed, same
    weights) and nothing else: the multi-process runs keep Ulysses over the
    whole world, which the DiT constructor checks against the 2 heads."""
    torch.manual_seed(seed)
    model = _bare_model()
    model.arch = arch
    blocks = []
    for index in range(arch.num_layers):
        block = dit.MiniMaxH3DiTBlock(arch, None, prefix=f"blocks.{index}")
        block.requires_grad_(False)
        for param in block.parameters():
            param.data.normal_(0, 0.02)
        block.adaln_proj.linear.weight.data.normal_(0, 0.08)
        block.adaln_proj.linear.bias.data.normal_(0, 0.3)
        block.attn.q_norm.weight.data.uniform_(0.5, 1.5)
        block.attn.k_norm.weight.data.uniform_(0.5, 1.5)
        blocks.append(block)
    model.blocks = torch.nn.ModuleList(blocks)
    return model


def _block_kwargs(arch, tokens: int = 128, used: int = 100, timesteps: int = 2):
    """One forward's rows and block keyword arguments for the stack."""
    generator = torch.Generator().manual_seed(5)
    rows = timesteps * MINIMAX_H3_ADALN_MODALITY_NUM
    freqs = torch.rand(tokens, arch.rope_inv_freq_len * 6, generator=generator) * 30
    x = torch.randn(tokens, arch.hidden_size, generator=generator).to(_BF16)
    return x, dict(
        adaln_input=_adaln_input(arch, timesteps, seed=7),
        combined_indices=torch.randint(0, rows, (tokens,), generator=generator),
        rope_cache=(dit._rope_cos_sin_cache(freqs, dtype=_BF16), torch.arange(tokens)),
        cu_seqlens=torch.tensor([0, used, tokens], dtype=torch.int32),
        cu_seqlens_host=(0, used, tokens),
        ulysses_active=True,
    )


class _RecordingOp:
    """The op's stand-in: records every call's arguments; the plain op returns
    the rows, the vouched one also every head vouched and the tally plus one."""

    def __init__(self, vouched: bool = False):
        self.vouched = vouched
        self.calls = []

    def __call__(self, *args):
        self.calls.append(args)
        if not self.vouched:
            return args[0].clone()
        heads, tally = args[6], args[7]
        return args[0].clone(), torch.ones_like(heads), tally + 1


@contextlib.contextmanager
def _padding_restored():
    with patch.object(
        packed_sequence,
        "minimax_h3_padded_seq_len",
        packed_sequence.minimax_h3_padded_seq_len,
    ):
        yield


@contextlib.contextmanager
def _enable_patches(op, *, shard: bool, vouch: bool = False, structured=False):
    """``hook.enable`` without TPU: the support check passes, the op is
    ``op``; votes, ranks, packing (through host memory, in waves) and
    barriers are the real ones. Without JAX the kernel config and tile widths
    are stand-ins."""
    with contextlib.ExitStack() as stack:
        stack.enter_context(_SHARD.override(shard))
        stack.enter_context(tpu_envs.SGLANG_TPU_OPT_H3_MEGAKERNEL_VOUCH.override(vouch))
        stack.enter_context(
            tpu_envs.SGLANG_TPU_OPT_H3_MEGAKERNEL_STATIC_STRUCTURE.override(structured)
        )
        stack.enter_context(patch.object(hook, "unsupported_reason", return_value=None))
        stack.enter_context(patch.object(hook, "_op_for", return_value=op))
        if not _HAS_JAX:
            stack.enter_context(
                patch.object(pack, "config_from_arch", side_effect=_kernel_config)
            )
            stack.enter_context(patch.object(pack, "tile_widths", return_value=_TILES))
        stack.enter_context(_padding_restored())
        yield


class _LogLines(logging.Handler):
    def __init__(self):
        super().__init__(logging.INFO)
        self.lines = []

    def emit(self, record):
        self.lines.append(record.getMessage())


@contextlib.contextmanager
def _hook_log():
    handler = _LogLines()
    level = hook.logger.level
    hook.logger.addHandler(handler)
    hook.logger.setLevel(logging.INFO)
    try:
        yield handler.lines
    finally:
        hook.logger.removeHandler(handler)
        hook.logger.setLevel(level)


@contextlib.contextmanager
def _recorded_gathers(log: list):
    """Record each all-gather of the tables (shape, dtype) and run it."""
    real = hook._all_gather_tables

    def gather(local, out=None):
        log.append([list(local.shape), str(local.dtype)])
        return real(local, out)

    with patch.object(hook, "_all_gather_tables", gather):
        yield


def _group(world: int, rank: int):
    """A stand-in for the sequence-parallel group of ``world`` ranks."""
    return SimpleNamespace(
        world_size=world, rank_in_group=rank, device_group=object(), cpu_group=None
    )


def _proj_bytes(proj) -> int:
    return sum(p.numel() * p.element_size() for p in proj.parameters())


# ---------------------------------------------------------------- one process


class TestBlockRange(CustomTestCase):
    def test_every_block_once_in_rank_order(self):
        for layers in range(1, 65):
            for devices in range(1, 17):
                ranges = [
                    pack.adaln_block_range(layers, devices, rank)
                    for rank in range(devices)
                ]
                per_rank = -(-layers // devices)
                with self.subTest(layers=layers, devices=devices):
                    self.assertEqual({k for _, _, k in ranges}, {per_rank})
                    # Consecutive and in rank order: rank r starts where r - 1
                    # ends, so the gather's rank order is block order.
                    self.assertEqual(ranges[0][0], 0)
                    for (_, end, _), (first, _, _) in zip(ranges, ranges[1:]):
                        self.assertEqual(first, end)
                    self.assertEqual(ranges[-1][1], layers)
                    sizes = [end - first for first, end, _ in ranges]
                    self.assertTrue(all(0 <= size <= per_rank for size in sizes))
                    # Only the last ranks hold fewer (at most one a part).
                    self.assertEqual(sizes, sorted(sizes, reverse=True))
                    self.assertLessEqual(sum(0 < s < per_rank for s in sizes), 1)

    def test_h3_shares(self):
        def sizes(devices):
            return [
                end - first
                for first, end, _ in (
                    pack.adaln_block_range(50, devices, rank) for rank in range(devices)
                )
            ]

        self.assertEqual(sizes(2), [25, 25])
        self.assertEqual(sizes(4), [13, 13, 13, 11])
        self.assertEqual(sizes(8), [7] * 7 + [1])
        self.assertEqual(sizes(1), [50])
        self.assertEqual(pack.adaln_block_range(6, 4, 3), (6, 6, 2))


def _emulated_shards(projs, adaln_input, layers: int, devices: int):
    """Every rank's sharded tables in one process. Each rank's gather input is
    what ``sharded_adaln_tables`` hands its gather; the gather returns every
    rank's input concatenated in rank order, the contract of
    ``all_gather_into_tensor``."""
    inputs = []

    def capture(local):
        inputs.append(local)
        return local.new_zeros((devices * local.shape[0], *local.shape[1:]))

    def shard(gather):
        out = []
        for rank in range(devices):
            first, end, per_rank = pack.adaln_block_range(layers, devices, rank)
            out.append(
                pack.sharded_adaln_tables(
                    projs[first:end],
                    adaln_input,
                    per_rank=per_rank,
                    layers=layers,
                    table_shape=_table_shape(projs[0]),
                    gather=gather,
                )
            )
        return out

    shard(capture)
    gathered = torch.cat(inputs)
    return inputs, shard(lambda local: gathered)


class TestShardedTables(CustomTestCase):
    def test_emulated_ranks_give_the_replicated_tables_bitwise(self):
        _init_parallel()
        arch = _arch(50)
        projs = _projs(arch, 50)
        cases = [(50, 2), (50, 4), (50, 8), (50, 1), (6, 4), (7, 3), (50, 16)]
        for timesteps in (2, 5, 6):
            adaln_input = _adaln_input(arch, timesteps)
            for layers, devices in cases:
                with self.subTest(layers=layers, devices=devices, timesteps=timesteps):
                    want = pack.adaln_tables(projs[:layers], adaln_input)
                    inputs, got = _emulated_shards(
                        projs[:layers], adaln_input, layers, devices
                    )
                    # Every rank hands the gather one tensor of one shape.
                    per_rank = -(-layers // devices)
                    self.assertEqual(
                        {tuple(t.shape) for t in inputs},
                        {(per_rank, *want.shape[1:])},
                    )
                    self.assertEqual({t.dtype for t in inputs}, {_BF16})
                    for rank, tables in enumerate(got):
                        self.assertTrue(_same(tables, want), rank)
                    # The padding is zero tables after the last block.
                    tail = torch.cat(inputs)[layers:]
                    self.assertFalse(tail.any())

    def test_a_rank_without_blocks_refuses_too_many_rows_like_the_others(self):
        _init_parallel()
        arch = _arch(2)
        projs = _projs(arch, 2)
        too_many = _adaln_input(arch, timesteps=43)  # 129 rows
        for adaln_projs in (projs, []):
            with self.assertRaisesRegex(ValueError, "AdaLN rows"):
                pack.sharded_adaln_tables(
                    adaln_projs,
                    too_many,
                    per_rank=2,
                    layers=2,
                    table_shape=_table_shape(projs[0]),
                    gather=lambda local: self.fail("gathered"),
                )


def _stack(projs, op, config, devices: int, shard=None):
    """A packed stack over ``projs`` (zero weights) and the model holding it."""
    model = _bare_model()
    stack = hook.H3MegakernelStack(
        model,
        projs,
        {name: torch.zeros(1) for name in pack.WEIGHT_ORDER},
        op,
        config,
        devices=devices,
        adaln_shard=shard,
    )
    model.blocks = torch.nn.ModuleList([stack])
    return model, stack


def _operands(stack, kwargs):
    with torch.no_grad():
        return stack.operands(
            kwargs["adaln_input"],
            kwargs["combined_indices"],
            kwargs["rope_cache"][0],
            kwargs["cu_seqlens"],
        )


@unittest.skipUnless(_HAS_CKPT, "needs the FL2VA checkpoint")
class TestRealProjections(CustomTestCase):
    def test_sharded_tables_equal_the_replicated_ones_bitwise(self):
        # Four real blocks' projections (2688 -> 96768, bf16) through
        # sglang's weight loaders, two timesteps: [4, 6, 16, 5376] tables.
        from safetensors import safe_open

        _init_parallel()
        arch = MiniMaxH3DiTArchConfig()
        with open(os.path.join(_TRANSFORMER, "model.safetensors.index.json")) as f:
            where = json.load(f)["weight_map"]
        projs = []
        for layer in (0, 1, 35, 49):
            proj = dit.MiniMaxH3AdalnProj(
                arch,
                arch.adaln_out_features,
                None,
                prefix=f"blocks.{layer}.adaln_proj",
                expand_ratio=6,
                modality_num=MINIMAX_H3_ADALN_MODALITY_NUM,
            ).requires_grad_(False)
            for name, param in proj.linear.named_parameters():
                key = f"blocks.{layer}.adaln_proj.linear.{name}"
                with safe_open(os.path.join(_TRANSFORMER, where[key]), "pt") as f:
                    tensor = f.get_tensor(key)
                loader = getattr(param, "weight_loader", None)
                if loader is None:
                    param.data.copy_(tensor)
                else:
                    loader(param, tensor)
            projs.append(proj)
        adaln_input = _adaln_input(arch)
        with torch.no_grad():
            want = pack.adaln_tables(projs, adaln_input)
            self.assertEqual(tuple(want.shape), (4, 6, 16, 5376))
            # Real projections: every block's tables carry signal (some
            # modality rows of a block can be zero in the checkpoint).
            self.assertTrue(all(tables.any() for tables in want))
            self.assertFalse(want[:, :, 6:].any())
            for devices in (2, 3, 4, 8):
                _, got = _emulated_shards(projs, adaln_input, 4, devices)
                for rank, tables in enumerate(got):
                    self.assertTrue(_same(tables, want), (devices, rank))


class TestShardedStackOneProcess(CustomTestCase):
    """The stack with a share of the projections and the emulated gather."""

    def test_operands_and_op_inputs_match_the_replicated_stack(self):
        _init_parallel()
        for layers, devices in ((50, 4), (50, 8), (6, 4)):
            arch = _arch(layers)
            projs = _projs(arch, layers)
            x, kwargs = _block_kwargs(arch)
            config = _kernel_config(arch, layers)
            inputs, _ = _emulated_shards(projs, kwargs["adaln_input"], layers, devices)
            full_op = _RecordingOp()
            _, full = _stack(projs, full_op, config, devices)
            want = _operands(full, kwargs)
            with torch.no_grad():
                full(x, **kwargs)
            for rank in range(devices):
                first, end, per_rank = pack.adaln_block_range(layers, devices, rank)
                shard = hook.AdalnShard(
                    rank=rank,
                    devices=devices,
                    layers=layers,
                    first=first,
                    end=end,
                    per_rank=per_rank,
                    table_shape=_table_shape(projs[0]),
                    kept=0,
                    freed=0,
                )
                op = _RecordingOp()
                _, stack = _stack(projs[first:end], op, config, devices, shard)
                seen = []

                def gather(local):
                    seen.append(local)
                    return torch.cat(inputs)

                with patch.object(hook, "_all_gather_tables", gather):
                    got = _operands(stack, kwargs)
                    with torch.no_grad():
                        stack(x, **kwargs)
                with self.subTest(layers=layers, devices=devices, rank=rank):
                    self.assertEqual(len(stack.adaln), end - first)
                    self.assertTrue(_same_args(got, want))
                    self.assertTrue(_same_args(op.calls[0], full_op.calls[0]))
                    # One gather a call (operands, then the forward), each
                    # with this rank's own share.
                    self.assertEqual(len(seen), 2)
                    for local in seen:
                        self.assertTrue(_same(local, inputs[rank]))


class TestAllGather(CustomTestCase):
    def test_gathers_over_the_sequence_parallel_device_group(self):
        # The sequence-parallel group's device group (tpu_dist on TPU; the
        # DiT's per-step output all-gather runs on it), not its gloo CPU group
        # and not the world group's, into one [ranks * k, ...] tensor.
        sp = SimpleNamespace(
            world_size=4, device_group=object(), cpu_group=object(), rank_in_group=1
        )
        world = SimpleNamespace(
            world_size=4, device_group=object(), cpu_group=object(), rank_in_group=1
        )
        local = torch.arange(2 * 3, dtype=torch.float32).view(2, 3).to(_BF16)
        calls = []

        def all_gather(output, tensor, group):
            calls.append((output, tensor, group))
            output.copy_(tensor.repeat(4, 1))

        state = "sglang.multimodal_gen.runtime.distributed.parallel_state"
        with (
            patch(f"{state}.get_sp_group", return_value=sp),
            patch(f"{state}.get_world_group", return_value=world),
            patch(
                "sglang.multimodal_gen.runtime.distributed.utils.all_gather_single",
                all_gather,
            ),
        ):
            self.assertIs(hook._shard_group(), sp)
            out = hook._all_gather_tables(local)
            # The startup check's preallocated buffer is the one filled.
            buffer = torch.empty((8, 3), dtype=_BF16)
            into = hook._all_gather_tables(local, buffer)
        (output, tensor, used), (output2, _, used2) = calls
        self.assertIs(used, sp.device_group)
        self.assertIs(used2, sp.device_group)
        self.assertIs(tensor, local)
        self.assertEqual(tuple(output.shape), (8, 3))
        self.assertIs(out, output)
        self.assertTrue(_same(out, local.repeat(4, 1)))
        self.assertIs(output2, buffer)
        self.assertIs(into, buffer)
        self.assertTrue(_same(into, local.repeat(4, 1)))


class TestOptionOff(CustomTestCase):
    def test_off_by_default(self):
        with _SHARD.override(True):
            _SHARD.clear()
            self.assertFalse(_SHARD.is_set())
            self.assertFalse(_SHARD.get())
            self.assertFalse(hook._shard_adaln())

    def test_off_the_stack_runs_the_replicated_tables(self):
        _init_parallel()
        arch = _arch(3)
        projs = _projs(arch, 3)
        x, kwargs = _block_kwargs(arch)
        model, stack = _stack(projs, _RecordingOp(), _kernel_config(arch, 3), 2)
        self.assertIsNone(stack.adaln_shard)
        with (
            patch.object(pack, "adaln_tables", wraps=pack.adaln_tables) as tables,
            patch.object(pack, "sharded_adaln_tables") as sharded,
            patch.object(hook, "_all_gather_tables") as gather,
            torch.no_grad(),
        ):
            stack(x, **kwargs)
        tables.assert_called_once()
        self.assertIs(tables.call_args.args[0], stack.adaln)
        self.assertIs(tables.call_args.args[1], kwargs["adaln_input"])
        sharded.assert_not_called()
        gather.assert_not_called()

    def test_off_enable_votes_and_builds_as_before(self):
        # The support vote is _ranks_agree's alone, with the check's reason;
        # nothing is freed, every projection stays in block order.
        _init_parallel()
        arch = _arch(4)
        model = _tiny_model(arch)
        adaln = [block.adaln_proj for block in model.blocks]
        votes = []
        real_agree = hook._ranks_agree
        with (
            _enable_patches(_RecordingOp(), shard=False),
            patch.object(hook, "_world_size", return_value=2),
            patch.object(
                hook,
                "_ranks_agree",
                side_effect=lambda why: votes.append(why) or real_agree(why),
            ),
            patch.object(hook, "_ranks_agree_sharded") as sharded_vote,
            patch.object(hook, "_shard_on_every_rank") as shard_projs,
            patch.object(hook, "_gather", side_effect=lambda value: [value, value]),
        ):
            self.assertTrue(hook.enable(model))
        # The support check, the op registration and the packing, as before.
        self.assertEqual(votes, [None, None, None])
        sharded_vote.assert_not_called()
        shard_projs.assert_not_called()
        (stack,) = model.blocks
        self.assertIsNone(stack.adaln_shard)
        self.assertEqual([id(p) for p in stack.adaln], [id(p) for p in adaln])
        self.assertTrue(all(p.linear.weight.numel() > 0 for p in stack.adaln))


class TestVote(CustomTestCase):
    def test_sharded_vote(self):
        vote = hook._SHARD_VOTE
        cases = [
            ([vote, vote], (None, True)),
            ([vote], (None, True)),
            (
                [vote, None],
                (
                    "SGLANG_TPU_OPT_H3_MEGAKERNEL_SHARD_ADALN is set on some ranks only",
                    False,
                ),
            ),
            ([vote, "no TPU"], ("rank 1: no TPU", False)),
            (["no TPU", vote, None], ("rank 0: no TPU", False)),
        ]
        for votes, want in cases:
            with (
                self.subTest(votes=votes),
                patch.object(hook, "_gather", return_value=votes),
            ):
                self.assertEqual(hook._ranks_agree_sharded(None), want)
        # A rank without the option reads the vote as a refusal.
        with patch.object(hook, "_gather", return_value=[vote, None]):
            self.assertEqual(hook._ranks_agree(None), f"rank 0: {vote}")
        # A rank's own refusal is its vote.
        with patch.object(hook, "_gather", side_effect=lambda v: [v]) as gather:
            self.assertEqual(hook._ranks_agree_sharded("x"), ("rank 0: x", False))
            self.assertEqual(hook._ranks_agree_sharded(None), (None, True))
        self.assertEqual([c.args[0] for c in gather.call_args_list], ["x", vote])

    def test_a_failing_step_makes_the_rank_raise(self):
        # One process: each step's failure is voted on and raised (the
        # multi-process runs check that every rank raises, without a hang).
        _init_parallel()
        arch = _arch(4)
        for step in ("_shard_adaln_projs", "_check_shard"):
            model = _tiny_model(arch)
            with (
                self.subTest(step=step),
                _enable_patches(_RecordingOp(), shard=True),
                patch.object(hook, "_world_size", return_value=2),
                patch.object(hook, "_shard_group", return_value=_group(2, 0)),
                patch.object(hook, step, side_effect=OSError("device")),
                patch.object(hook, "_pack_on_every_rank") as packing,
                self.assertRaisesRegex(
                    RuntimeError,
                    "sharding the AdaLN projections failed .rank 0: .*OSError: device",
                ),
            ):
                hook.enable(model)
            packing.assert_not_called()

    def test_the_sequence_parallel_group_must_hold_every_rank(self):
        _init_parallel()
        model = _tiny_model(_arch(4))
        with (
            _enable_patches(_RecordingOp(), shard=True),
            patch.object(hook, "_world_size", return_value=4),
            patch.object(hook, "_shard_group", return_value=_group(3, 0)),
            patch.object(hook, "_gather", side_effect=lambda v: [v] * 4),
            patch.object(hook, "_all_gather_tables") as gather,
            self.assertRaisesRegex(
                RuntimeError, "the sequence-parallel group has 3 ranks, the world 4"
            ),
        ):
            hook.enable(model)
        gather.assert_not_called()
        # Nothing was freed: the check comes first.
        self.assertTrue(all(b.adaln_proj.linear.weight.numel() for b in model.blocks))


def _probe_gather(arch, devices: int, rank: int, corrupt=None):
    """The startup check's all-gather as rank ``rank`` of ``devices`` would
    see it: every rank's share of the probe's tables in rank order, from a
    copy of the model (same seed, same weights), written into the check's
    buffer. ``corrupt`` edits the gathered tables in place first."""
    projs = [block.adaln_proj for block in _tiny_model(arch).blocks]
    probe = hook._shard_probe(projs[0])
    inputs, _ = _emulated_shards(projs, probe, len(projs), devices)
    gathered = torch.cat(inputs)
    if corrupt is not None:
        corrupt(gathered)

    def gather(local, out=None):
        assert _same(local, inputs[rank]), "not this rank's share"
        if out is None:
            return gathered.clone()
        assert out.shape == gathered.shape and out.dtype == gathered.dtype
        return out.copy_(gathered)

    return gather


def _swap(i: int, j: int):
    def corrupt(gathered):
        gathered[[i, j]] = gathered[[j, i]]

    return corrupt


def _flip_bit(block: int):
    """Flip the low bit of one value of a block's first table, row 0."""

    def corrupt(gathered):
        gathered[block, 0, 0, :1].view(torch.int16).bitwise_xor_(1)

    return corrupt


class TestShardOneProcess(CustomTestCase):
    """``enable`` as rank ``r`` of ``P`` (votes of agreeing peers; the
    startup check's all-gather emulated)."""

    def _enable(self, layers, devices, rank, *, hold=True, corrupt=None):
        _init_parallel()
        arch = _arch(layers)
        model = _tiny_model(arch)
        before = sum(p.numel() * p.element_size() for p in model.parameters())
        block_bytes = sum(
            p.numel() * p.element_size() for b in model.blocks for p in b.parameters()
        )
        per_block = _proj_bytes(model.blocks[0].adaln_proj)
        refs = [
            [StorageWeakRef(p.untyped_storage()) for p in b.adaln_proj.parameters()]
            for b in model.blocks
        ]
        # Something still holding the blocks must not keep the freed memory.
        held = list(model.blocks) if hold else None
        with (
            _enable_patches(_RecordingOp(), shard=True),
            patch.object(hook, "_world_size", return_value=devices),
            patch.object(hook, "_shard_group", return_value=_group(devices, rank)),
            patch.object(hook, "_gather", side_effect=lambda v: [v] * devices),
            patch.object(
                hook, "_all_gather_tables", _probe_gather(arch, devices, rank, corrupt)
            ),
            _hook_log() as lines,
        ):
            self.assertTrue(hook.enable(model))
        return SimpleNamespace(
            model=model,
            stack=model.blocks[0],
            refs=refs,
            held=held,
            lines=lines,
            before=before,
            block_bytes=block_bytes,
            per_block=per_block,
        )

    def test_rank_keeps_its_share_and_frees_the_rest(self):
        for layers, devices, rank in ((50, 4, 3), (50, 8, 0), (50, 2, 1), (6, 4, 3)):
            with self.subTest(layers=layers, devices=devices, rank=rank):
                run = self._enable(layers, devices, rank)
                first, end, per_rank = pack.adaln_block_range(layers, devices, rank)
                shard = run.stack.adaln_shard
                self.assertEqual(
                    (shard.rank, shard.devices, shard.layers),
                    (rank, devices, layers),
                )
                self.assertEqual(
                    (shard.first, shard.end, shard.per_rank), (first, end, per_rank)
                )
                self.assertEqual(
                    [id(p) for p in run.stack.adaln],
                    [id(b.adaln_proj) for b in run.held[first:end]],
                )
                for index, refs in enumerate(run.refs):
                    local = first <= index < end
                    self.assertEqual(
                        [not ref.expired() for ref in refs], [local] * 2, index
                    )
                    weight = run.held[index].adaln_proj.linear.weight
                    self.assertEqual(weight.numel() > 0, local, index)
                kept = (end - first) * run.per_block
                freed = (layers - (end - first)) * run.per_block
                self.assertEqual(sum(_proj_bytes(p) for p in run.stack.adaln), kept)
                names = [n for n in run.model.state_dict() if ".adaln." in n]
                self.assertEqual(
                    sorted(names),
                    sorted(
                        f"blocks.0.adaln.{i}.linear.{kind}"
                        for i in range(end - first)
                        for kind in ("weight", "bias")
                    ),
                )
                (line,) = [line for line in run.lines if "sharded over" in line]
                share = f"blocks {first}-{end - 1}" if end > first else "no block"
                self.assertIn(
                    f"sharded over {devices} ranks, {per_rank} blocks a rank: "
                    f"rank {rank} keeps {share} ({kept / 2**30:.2f} GiB on the "
                    f"device) and freed {freed / 2**30:.2f} GiB",
                    line,
                )
                self.assertIn("checked: the replicated tables", line)
                self.assertEqual((shard.kept, shard.freed), (kept, freed))
                self.assertTrue(any("megakernel on" in line for line in run.lines))

    def test_nothing_else_holds_the_freed_weights(self):
        run = self._enable(50, 4, 1, hold=False)
        first, end, _ = pack.adaln_block_range(50, 4, 1)
        alive = [not ref.expired() for refs in run.refs for ref in refs]
        self.assertEqual(alive, [first <= i < end for i in range(50) for _ in range(2)])
        after = sum(p.numel() * p.element_size() for p in run.model.parameters())
        # The packed weights are buffers: of the blocks' parameters only the
        # rank's 13 projections are left, beside the embeddings, refiner and
        # final layer.
        self.assertEqual(
            after, run.before - run.block_bytes + (end - first) * run.per_block
        )
        self.assertEqual(end - first, 13)

    def test_the_startup_check_refuses_wrong_tables(self):
        # 6 blocks over 2 ranks: rank 0 holds blocks 0-2, rank 1 blocks 3-5.
        # The check compares every block's table, its own and its peers'.
        cases = [
            (0, "own blocks swapped", _swap(0, 1)),
            (0, "a peer's blocks swapped", _swap(3, 4)),
            (0, "one bit of a peer's block", _flip_bit(5)),
            (0, "one bit of its own block", _flip_bit(2)),
            (1, "a peer's blocks swapped", _swap(0, 1)),
            (1, "one bit of a peer's block", _flip_bit(0)),
            (1, "a block of each", _swap(2, 3)),
        ]
        for rank, what, corrupt in cases:
            with (
                self.subTest(rank=rank, corrupt=what),
                self.assertRaisesRegex(RuntimeError, "differ from the replicated"),
            ):
                self._enable(6, 2, rank, corrupt=corrupt)

    def test_one_rank_is_a_no_op(self):
        _init_parallel()
        arch = _arch(4)
        model = _tiny_model(arch)
        adaln = [block.adaln_proj for block in model.blocks]
        with (
            _enable_patches(_RecordingOp(), shard=True),
            patch.object(hook, "_world_size", return_value=1),
            patch.object(hook, "_all_gather_tables") as gather,
            _hook_log() as lines,
        ):
            self.assertTrue(hook.enable(model))
            x, kwargs = _block_kwargs(_arch(4))
            with torch.no_grad():
                model.blocks[0](x, **kwargs)
        (stack,) = model.blocks
        self.assertIsNone(stack.adaln_shard)
        self.assertEqual([id(p) for p in stack.adaln], [id(p) for p in adaln])
        self.assertTrue(any("no-op on one rank" in line for line in lines))
        self.assertFalse(any("sharded over" in line for line in lines))
        gather.assert_not_called()


def _lever_kwargs(arch, rows: int, devices: int, used: int, timesteps: int = 2):
    """One forward's rows and block keyword arguments for a rank of
    ``devices`` holding ``rows``: the bounds ``[0, used, rows * devices]`` of
    the whole sequence, as the DiT passes them under Ulysses."""
    generator = torch.Generator().manual_seed(11)
    table_rows = timesteps * MINIMAX_H3_ADALN_MODALITY_NUM
    total = rows * devices
    freqs = torch.rand(rows, arch.rope_inv_freq_len * 6, generator=generator) * 30
    x = torch.randn(rows, arch.hidden_size, generator=generator).to(_BF16)
    return x, dict(
        adaln_input=_adaln_input(arch, timesteps, seed=7),
        combined_indices=torch.randint(0, table_rows, (rows,), generator=generator),
        rope_cache=(dit._rope_cos_sin_cache(freqs, dtype=_BF16), torch.arange(rows)),
        cu_seqlens=torch.tensor([0, used, total], dtype=torch.int32),
        cu_seqlens_host=(0, used, total),
        ulysses_active=True,
    )


@unittest.skipUnless(_HAS_JAX, "needs JAX (pack.config_from_arch, loop_structure)")
class TestShardWithKernelLevers(CustomTestCase):
    """The option with the lean cut lever (SGLANG_TPU_OPT_H3_MEGAKERNEL_LEAN_CUT,
    under the static structure it shapes): ``enable`` as rank ``r`` of ``P``
    at a row count the lever covers (``op.LEVER_SHAPES``), against the same
    rank with the option alone. The lever changes only the kernel program:
    the op's build takes it and the static structure is found with its cut
    tile, while the op receives the same inputs bit for bit after the same
    all-gathers."""

    # (devices, rows a device, rank, blocks): 3 blocks a rank of 2; 2 blocks
    # a rank of 8, rank 3 holding blocks 6-7 (ranks 5-7 hold none).
    _CASES = ((2, 19456, 1, 6), (8, 4864, 3, 10))
    # The 1344x768x124 request's live length, a shorter and a longer one.
    _USED = (37_749, 36_000, 38_500)

    def _enable(self, arch, devices: int, rank: int, *, vouch: bool, lean: bool):
        """``enable`` with the flags; the real ``_op_for``, its op build
        recorded (``build_stack_op``'s keyword arguments) and stubbed."""
        from sglang_tpu.multimodal_gen.h3_megakernel import op as stack_op

        builds = []

        def build(config, op_devices, **kwargs):
            builds.append((op_devices, kwargs))
            return _RecordingOp(vouched=kwargs["vouch_margin"] is not None)

        model = _tiny_model(arch)
        flags = (
            (_SHARD, True),
            (tpu_envs.SGLANG_TPU_OPT_H3_MEGAKERNEL_STATIC_STRUCTURE, True),
            (tpu_envs.SGLANG_TPU_OPT_H3_MEGAKERNEL_VOUCH, vouch),
            (tpu_envs.SGLANG_TPU_OPT_H3_MEGAKERNEL_LEAN_CUT, lean),
            (tpu_envs.SGLANG_TPU_OPT_H3_MEGAKERNEL_SPLIT_ITEMS, False),
            (tpu_envs.SGLANG_TPU_OPT_H3_MEGAKERNEL_EPILOGUE_CHUNKS, False),
        )
        with contextlib.ExitStack() as stack:
            for flag, value in flags:
                stack.enter_context(flag.override(value))
            stack.enter_context(
                patch.object(hook, "unsupported_reason", return_value=None)
            )
            stack.enter_context(patch.dict(hook._OPS, clear=True))
            stack.enter_context(
                patch.object(stack_op, "build_stack_op", side_effect=build)
            )
            stack.enter_context(patch.object(hook, "_world_size", return_value=devices))
            stack.enter_context(
                patch.object(hook, "_shard_group", return_value=_group(devices, rank))
            )
            stack.enter_context(
                patch.object(hook, "_gather", side_effect=lambda v: [v] * devices)
            )
            stack.enter_context(
                patch.object(
                    hook, "_all_gather_tables", _probe_gather(arch, devices, rank)
                )
            )
            stack.enter_context(_padding_restored())
            stack.enter_context(_hook_log())
            self.assertTrue(hook.enable(model))
        return model, builds

    def _forwards(self, stack, arch, projs, rows, devices, rank, vouch):
        """``_forward_sequence`` at each live length of ``_USED`` through the
        emulated per-step all-gather: the cut tile each call's static
        structure was found with, the gathers (this rank's share each time)
        and the replicated tables each call must see."""
        layers = len(projs)
        cuts, gathers, wants = [], [], []
        real_structure = hook.static_structure

        def structure(*args, **kwargs):
            cuts.append(kwargs.get("cut_kv_compute", 0))
            return real_structure(*args, **kwargs)

        for used in self._USED:
            x, kwargs = _lever_kwargs(arch, rows, devices, used)
            adaln_input = kwargs["adaln_input"]
            inputs, _ = _emulated_shards(projs, adaln_input, layers, devices)
            gathered = torch.cat(inputs)
            with torch.no_grad():
                want = pack.adaln_tables(projs, adaln_input)

            def gather(local, out=None, share=inputs[rank], gathered=gathered):
                assert out is None and _same(local, share), "not this rank's share"
                gathers.append([list(local.shape), str(local.dtype)])
                return gathered.clone()

            calls = len(stack.op.calls)
            with (
                patch.object(hook, "_all_gather_tables", gather),
                patch.object(hook, "static_structure", side_effect=structure),
            ):
                _forward_sequence(stack, x, kwargs, vouched=vouch, request=f"r{used}")
            wants += [want] * (len(stack.op.calls) - calls)
        return SimpleNamespace(cuts=cuts, gathers=gathers, wants=wants)

    def test_lean_cut_changes_only_the_program(self):
        from sglang_tpu.multimodal_gen.h3_megakernel import op as stack_op

        _init_parallel()
        for devices, rows, rank, layers in self._CASES:
            self.assertIn((devices, rows), stack_op.LEVER_SHAPES["lean_cut"])
            arch = _arch(layers, heads=devices)
            projs = [block.adaln_proj for block in _tiny_model(arch).blocks]
            for vouch in (False, True):
                margin = 1.1 if vouch else None
                with self.subTest(devices=devices, rank=rank, vouch=vouch):
                    runs = {}
                    for lean in (False, True):
                        model, builds = self._enable(
                            arch, devices, rank, vouch=vouch, lean=lean
                        )
                        (stack,) = model.blocks
                        seen = self._forwards(
                            stack, arch, projs, rows, devices, rank, vouch
                        )
                        runs[lean] = SimpleNamespace(
                            model=model, stack=stack, builds=builds, **vars(seen)
                        )
                    alone, both = runs[False], runs[True]
                    # The same share of the projections, sharded and checked.
                    first, end, per_rank = pack.adaln_block_range(layers, devices, rank)
                    for run in (alone, both):
                        shard = run.stack.adaln_shard
                        self.assertEqual(
                            (shard.first, shard.end, shard.per_rank),
                            (first, end, per_rank),
                        )
                        self.assertEqual(len(run.stack.adaln), end - first)
                    # The lever reaches the op's build and the structure only.
                    self.assertEqual(alone.stack.levers, frozenset())
                    self.assertEqual(both.stack.levers, {"lean_cut"})
                    plain = {"structured": True, "vouch_margin": margin}
                    self.assertEqual(alone.builds, [(devices, plain)])
                    self.assertEqual(
                        both.builds,
                        [(devices, dict(plain, levers=frozenset({"lean_cut"})))],
                    )
                    calls = len(alone.stack.op.calls)
                    self.assertEqual(calls, len(self._USED) * (5 if vouch else 2))
                    self.assertEqual(alone.cuts, [0] * calls)
                    self.assertEqual(both.cuts, [stack_op.CUT_KV_COMPUTE] * calls)
                    # The op inputs bit for bit (rows, operands, gathered
                    # tables, vouched heads and tally, packed weights, static
                    # structure), after one all-gather a call of this rank's
                    # share; the tables are the replicated path's.
                    self.assertEqual(len(both.stack.op.calls), calls)
                    for index, (a, b) in enumerate(
                        zip(alone.stack.op.calls, both.stack.op.calls)
                    ):
                        self.assertTrue(_same_args(a, b), index)
                        self.assertTrue(_same(b[4], both.wants[index]), index)
                    self.assertEqual(both.gathers, alone.gathers)
                    self.assertEqual(len(both.gathers), calls)


# ------------------------------------------------------------- many processes


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _gloo_barrier(group=None, async_op=False, device_ids=None):
    """torch.distributed.barrier without the accelerator lookup: the barrier
    calls torch._C._get_accelerator(), which on a TPU host starts torch_tpu,
    and these gloo-only ranks have no TPU launch environment. A one-element
    all-reduce over the group is the same barrier."""
    torch.distributed.all_reduce(torch.zeros(1), group=group)


def _rank_main(rank: int, world: int, port: int, directory: str, cases: list):
    """One rank: sglang's groups over gloo (Ulysses over the whole world), then
    each case; the results so far go to ``<directory>/<rank>.json`` after
    each case."""
    os.environ.update(
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE=str(world),
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
    )
    torch.set_num_threads(1)
    torch.distributed.barrier = _gloo_barrier
    maybe_init_distributed_environment_and_model_parallel(
        tp_size=1, sp_size=world, ulysses_degree=world, dist_timeout=_DIST_TIMEOUT_S
    )
    results = {}
    for name, kwargs in cases:
        results[name] = _CASES[name](rank, world, **kwargs)
        with open(os.path.join(directory, f"{rank}.json"), "w") as f:
            json.dump(results, f)


def _spawn(world: int, cases: list, timeout: float = _SPAWN_TIMEOUT_S) -> list[dict]:
    """Run ``cases`` on ``world`` ranks, one process each; every rank's
    results. A rank that raises or hangs fails the run."""
    with tempfile.TemporaryDirectory() as directory:
        context = torch.multiprocessing.start_processes(
            _rank_main,
            args=(world, _free_port(), directory, cases),
            nprocs=world,
            join=False,
            start_method="spawn",
        )
        deadline = time.monotonic() + timeout
        while not context.join(timeout=5):
            if time.monotonic() > deadline:
                alive = [r for r, p in enumerate(context.processes) if p.is_alive()]
                for process in context.processes:
                    process.kill()
                done = []
                for rank in range(world):
                    path = os.path.join(directory, f"{rank}.json")
                    try:
                        with open(path) as f:
                            finished = list(json.load(f))
                    except (OSError, ValueError):
                        finished = []
                    done.append(f"rank {rank} finished {finished}")
                raise AssertionError(
                    f"ranks {alive} of {world} did not finish in {timeout:.0f} s "
                    f"(hung?); {'; '.join(done)}"
                )
        out = []
        for rank in range(world):
            with open(os.path.join(directory, f"{rank}.json")) as f:
                out.append(json.load(f))
        return out


def _case_tables(rank, world, layers_list, timesteps_list=(2,)):
    """The sharded tables over sglang's real groups vs the replicated ones."""
    sp, everyone = get_sp_group(), get_world_group()
    out = [
        dict(
            group=dict(
                is_sp=hook._shard_group() is sp,
                sp=[sp.rank_in_group, sp.world_size, list(sp.ranks)],
                world=[everyone.rank_in_group, everyone.world_size],
            )
        )
    ]
    for layers in layers_list:
        arch = _arch(layers)
        projs = _projs(arch, layers)
        first, end, per_rank = pack.adaln_block_range(layers, world, rank)
        for timesteps in timesteps_list:
            adaln_input = _adaln_input(arch, timesteps)
            want = pack.adaln_tables(projs, adaln_input)
            gathers = []
            with _recorded_gathers(gathers):
                got = pack.sharded_adaln_tables(
                    projs[first:end],
                    adaln_input,
                    per_rank=per_rank,
                    layers=layers,
                    table_shape=_table_shape(projs[0]),
                    gather=hook._all_gather_tables,
                )
            out.append(
                dict(
                    layers=layers,
                    timesteps=timesteps,
                    share=[first, end, per_rank],
                    equal=_same(got, want),
                    shape=list(got.shape),
                    want_shape=list(want.shape),
                    dtype=str(got.dtype),
                    gathers=gathers,
                )
            )
    return out


def _forward_sequence(stack, x, kwargs, *, vouched: bool, request=None):
    """Two forwards outside a request, then (vouched) a request of three."""
    from sglang.multimodal_gen.runtime.managers.forward_context import (
        set_forward_context,
    )

    with torch.no_grad():
        for step in range(2):
            stack(x, **kwargs)
        if vouched:
            stack.begin_request(request)
            for step in range(3):
                batch = SimpleNamespace(request_id=request)
                with set_forward_context(step, None, forward_batch=batch):
                    stack(x, **kwargs)
            stack.end_request()


def _case_enable(rank, world, layers):
    """``enable`` off and on, on the same weights: the vote, the freed
    weights, and the op inputs and all-gathers of plain, vouched, structured
    and loop-wrapped forwards."""
    arch = _arch(layers)
    x, kwargs = _block_kwargs(arch)
    result = {}
    variants = [("plain", False, False), ("vouched", True, False)]
    if _HAS_JAX:
        variants.append(("structured", False, True))
    first, end, per_rank = pack.adaln_block_range(layers, world, rank)
    for name, vouch, structured in variants:
        runs = {}
        for shard in (False, True):
            model = _block_model(arch)
            per_block = _proj_bytes(model.blocks[0].adaln_proj)
            refs = [
                [StorageWeakRef(p.untyped_storage()) for p in b.adaln_proj.parameters()]
                for b in model.blocks
            ]
            held = list(model.blocks)
            op = _RecordingOp(vouched=vouch)
            gathers = []
            with (
                _enable_patches(op, shard=shard, vouch=vouch, structured=structured),
                _hook_log() as lines,
            ):
                enabled = hook.enable(model)
            stack = model.blocks[0]
            with _recorded_gathers(gathers):
                _forward_sequence(stack, x, kwargs, vouched=vouch, request="r1")
            runs[shard] = SimpleNamespace(
                enabled=enabled,
                stack=stack,
                op=op,
                gathers=gathers,
                lines=lines,
                alive=[[not r.expired() for r in pair] for pair in refs],
                held_numel=[b.adaln_proj.linear.weight.numel() for b in held],
                kept=sum(_proj_bytes(p) for p in stack.adaln),
                per_block=per_block,
            )
        off, on = runs[False], runs[True]
        calls = len(on.op.calls)
        result[name] = dict(
            enabled=[off.enabled, on.enabled],
            calls=[len(off.op.calls), calls],
            same_op_inputs=len(off.op.calls) == calls
            and all(_same_args(a, b) for a, b in zip(off.op.calls, on.op.calls)),
            tables_shape=list(on.op.calls[0][4].shape),
            gathers_off=off.gathers,
            gathers_on=on.gathers,
            share=[on.stack.adaln_shard.first, on.stack.adaln_shard.end, per_rank],
            want_share=[first, end, per_rank],
            adaln_counts=[len(off.stack.adaln), len(on.stack.adaln)],
            alive_on=on.alive,
            alive_off=off.alive,
            held_numel_on=on.held_numel,
            kept=[off.kept, on.kept],
            per_block=on.per_block,
            sharded_lines=[line for line in on.lines if "sharded over" in line],
            sharded_lines_off=[line for line in off.lines if "sharded over" in line],
        )
        del runs, off, on
        gc.collect()
    off, on = (_case_loop(rank, world, layers, shard) for shard in (False, True))
    result["loop"] = dict(
        calls=[len(off.op.calls), len(on.op.calls)],
        same_op_inputs=len(off.op.calls) == len(on.op.calls)
        and all(_same_args(a, b) for a, b in zip(off.op.calls, on.op.calls)),
        gathers_off=off.gathers,
        gathers=on.gathers,
    )
    return result


def _case_loop(rank, world, layers, shard: bool):
    """Requests through the hook's denoise-loop wrapper (a warmup request,
    then a measured one) on the vouched stack, replicated or sharded: the op
    and the all-gathers it saw (sharded: one a forward, as on every other
    path)."""
    arch = _arch(layers)
    x, kwargs = _block_kwargs(arch)
    model = _block_model(arch)
    op = _RecordingOp(vouched=True)
    with _enable_patches(op, shard=shard, vouch=True), _hook_log():
        assert hook.enable(model)
    stack = model.blocks[0]
    gathers = []

    def loop(stage, batch, server_args):
        from sglang.multimodal_gen.runtime.managers.forward_context import (
            set_forward_context,
        )

        with torch.no_grad():
            for step in range(3):
                with set_forward_context(step, None, forward_batch=batch):
                    stack(x, **kwargs)

    stage = SimpleNamespace(
        transformer=model, _cache_dit_requested_for_batch=lambda batch: False
    )
    params = SimpleNamespace(
        enable_spectrum=False,
        enable_teacache=False,
        enable_cache_dit=None,
        quality="lossless",
    )
    cls = dit.MiniMaxH3DiTModel
    names = (
        "post_load_weights",
        "forward",
        "validate_weight_update_source",
        "validate_lora_layers",
    )
    from sglang.multimodal_gen.runtime.pipelines_core.lora.pipeline import (
        LoRAPipeline,
    )

    with (
        patch.multiple(cls, **{name: getattr(cls, name) for name in names}),
        patch.object(MiniMaxH3DenoisingStage, "_run_full_loop", loop),
        patch.object(
            LoRAPipeline,
            "_reject_lora_on_packed_weights",
            LoRAPipeline._reject_lora_on_packed_weights,
        ),
        _recorded_gathers(gathers),
    ):
        hook.install()
        try:
            run = MiniMaxH3DenoisingStage._run_full_loop
            for rid, warmup in (("warm", True), ("measured", False)):
                batch = SimpleNamespace(
                    request_id=rid, is_warmup=warmup, sampling_params=params
                )
                run(stage, batch, None)
        finally:
            del cls._tpu_h3_megakernel_installed
    return SimpleNamespace(op=op, gathers=gathers)


def _case_mixed(rank, world, layers, on_ranks):
    """The option on ``on_ranks`` only: every rank leaves the kernel off."""
    model = _block_model(_arch(layers))
    with (
        _enable_patches(_RecordingOp(), shard=rank in on_ranks),
        _hook_log() as lines,
    ):
        enabled = hook.enable(model)
    return dict(
        enabled=enabled,
        packed=hook.is_packed(model),
        blocks=len(model.blocks),
        off_lines=[line for line in lines if "megakernel off" in line],
        sharded_lines=[line for line in lines if "sharded over" in line],
    )


def _failing(where: str, rank: int, world: int):
    """A patch that makes this rank fail at ``where`` of the startup sharding
    (``"peers"``: its all-gather returns unrelated data for every peer's
    share, the fault torch_tpu showed in compiled TP all-gathers,
    sglang_tpu/all_gather_via_all_reduce.py)."""
    if where == "probe":
        return patch.object(hook, "_shard_probe", side_effect=OSError("injected"))
    if where == "reference":
        # The replicated tables of the probe, the first adaln_tables call.
        real, calls = pack.adaln_tables, []

        def tables(projs, adaln_input):
            calls.append(len(projs))
            if len(calls) == 1:
                raise OSError("injected")
            return real(projs, adaln_input)

        return patch.object(pack, "adaln_tables", tables)
    if where == "share":
        return patch.object(pack, "adaln_share", side_effect=OSError("injected"))
    if where == "sync":
        # An asynchronous device error that surfaces at the second sync (after
        # the share is built, before the vote).
        real, calls = pack._sync, []

        def sync(device):
            calls.append(device)
            if len(calls) == 2:
                raise OSError("injected")
            return real(device)

        return patch.object(pack, "_sync", sync)
    if where == "peers":
        real = hook._all_gather_tables

        def gather(local, out=None):
            got = real(local, out)
            share = local.shape[0]
            for peer in range(world):
                if peer != rank:
                    got[peer * share : (peer + 1) * share] = 0.5
            return got

        return patch.object(hook, "_all_gather_tables", gather)
    raise ValueError(where)


def _case_fail(rank, world, runs):
    """``enable`` with the option on, where for each ``(layers, failing rank,
    where)`` of ``runs`` that rank fails at ``where`` (:func:`_failing`): what
    every rank raised, how long it took and the all-gathers it entered."""
    out = []
    for layers, failing, where in runs:
        model = _block_model(_arch(layers))
        gathers = []
        start = time.monotonic()
        with contextlib.ExitStack() as stack:
            stack.enter_context(_enable_patches(_RecordingOp(), shard=True))
            if rank == failing:
                stack.enter_context(_failing(where, rank, world))
            stack.enter_context(_recorded_gathers(gathers))
            stack.enter_context(_hook_log())
            try:
                hook.enable(model)
                raised = None
            except Exception as error:  # noqa: BLE001 - recorded for the check
                raised = f"{type(error).__name__}: {error}"
        out.append(
            dict(
                run=[layers, failing, where],
                raised=raised,
                seconds=time.monotonic() - start,
                gathers=len(gathers),
                packed=hook.is_packed(model),
            )
        )
    return out


# Case names of _spawn: one function can run under several names (a run
# keeps each name's result).
_CASES = {
    "tables": _case_tables,
    "enable": _case_enable,
    "enable6": _case_enable,
    "mixed": _case_mixed,
    "mixed_one_off": _case_mixed,
    "fail": _case_fail,
}


class TestShardOverRanks(CustomTestCase):
    """2, 4 and 8 ranks, one process each, over gloo."""

    def check_tables(self, results, world):
        for rank, result in enumerate(results):
            group, *cases = result["tables"]
            with self.subTest(world=world, rank=rank, check="group"):
                # The tables gather over the sequence-parallel group, which
                # here (Ulysses over the world) is every rank in world order.
                self.assertEqual(
                    group["group"],
                    dict(
                        is_sp=True,
                        sp=[rank, world, list(range(world))],
                        world=[rank, world],
                    ),
                )
            for case in cases:
                layers = case["layers"]
                with self.subTest(world=world, rank=rank, layers=layers):
                    first, end, per_rank = pack.adaln_block_range(layers, world, rank)
                    self.assertEqual(case["share"], [first, end, per_rank])
                    self.assertTrue(case["equal"])
                    self.assertEqual(case["shape"], case["want_shape"])
                    self.assertEqual(case["shape"][0], layers)
                    self.assertEqual(case["dtype"], str(_BF16))
                    self.assertEqual(
                        case["gathers"],
                        [[[per_rank, *case["want_shape"][1:]], str(_BF16)]],
                    )

    def check_enable(self, results, world, layers):
        sequences = []
        for rank, result in enumerate(results):
            first, end, per_rank = pack.adaln_block_range(layers, world, rank)
            enabled = result["enable"]
            for name, case in enabled.items():
                if name == "loop":
                    continue
                with self.subTest(world=world, rank=rank, variant=name):
                    self.assertEqual(case["enabled"], [True, True])
                    self.assertEqual(case["share"], case["want_share"])
                    self.assertEqual(case["adaln_counts"], [layers, end - first])
                    # The op saw the same inputs, bit for bit, every call.
                    self.assertGreater(case["calls"][1], 0)
                    self.assertEqual(case["calls"][0], case["calls"][1])
                    self.assertTrue(case["same_op_inputs"])
                    self.assertEqual(case["tables_shape"][0], layers)
                    # Off: no gather. On: one a forward, the same everywhere.
                    self.assertEqual(case["gathers_off"], [])
                    self.assertEqual(len(case["gathers_on"]), case["calls"][1])
                    # The other blocks' weights are gone, with the blocks
                    # still held; off, nothing was freed.
                    self.assertEqual(
                        case["alive_on"],
                        [[first <= i < end] * 2 for i in range(layers)],
                    )
                    self.assertEqual(case["alive_off"], [[True, True]] * layers)
                    self.assertEqual(
                        [n > 0 for n in case["held_numel_on"]],
                        [first <= i < end for i in range(layers)],
                    )
                    self.assertEqual(
                        case["kept"],
                        [
                            layers * case["per_block"],
                            (end - first) * case["per_block"],
                        ],
                    )
                    self.assertEqual(len(case["sharded_lines"]), 1)
                    self.assertEqual(case["sharded_lines_off"], [])
                    sequences.append((name, json.dumps(case["gathers_on"])))
            loop = enabled["loop"]
            with self.subTest(world=world, rank=rank, variant="loop"):
                # Two requests of 3 steps through the wrapper: the op saw the
                # same inputs flag off and on (tables, vouched heads, tally).
                self.assertEqual(loop["calls"], [6, 6])
                self.assertTrue(loop["same_op_inputs"])
                self.assertEqual(loop["gathers_off"], [])
                self.assertEqual(len(loop["gathers"]), 6)
                sequences.append(("loop", json.dumps(loop["gathers"])))
        # Every rank issued the same all-gathers in the same order.
        by_variant = {}
        for name, sequence in sequences:
            by_variant.setdefault(name, set()).add(sequence)
        for name, distinct in by_variant.items():
            self.assertEqual(len(distinct), 1, (world, name))

    def test_two_ranks(self):
        results = _spawn(
            2,
            [
                ("tables", dict(layers_list=[50, 3], timesteps_list=(2, 5))),
                ("enable", dict(layers=50)),
            ],
        )
        self.check_tables(results, 2)
        self.check_enable(results, 2, 50)

    def test_four_ranks(self):
        # 50 blocks: 13, 13, 13, 11; 6 blocks: 2, 2, 2 and a rank with none.
        results = _spawn(
            4,
            [
                ("tables", dict(layers_list=[50, 6], timesteps_list=(2, 6))),
                ("enable", dict(layers=50)),
                ("enable6", dict(layers=6)),
                ("mixed", dict(layers=6, on_ranks=[0, 2])),
                ("mixed_one_off", dict(layers=6, on_ranks=[0, 1, 2])),
            ],
        )
        self.check_tables(results, 4)
        self.check_enable(results, 4, 50)
        self.check_enable([dict(enable=result["enable6"]) for result in results], 4, 6)
        for name in ("mixed", "mixed_one_off"):
            for rank, result in enumerate(results):
                case = result[name]
                with self.subTest(case=name, rank=rank):
                    self.assertFalse(case["enabled"])
                    self.assertFalse(case["packed"])
                    self.assertEqual(case["blocks"], 6)
                    self.assertEqual(case["sharded_lines"], [])
                    (line,) = case["off_lines"]
                    self.assertTrue(
                        "set on some ranks only" in line or hook._SHARD_VOTE in line,
                        line,
                    )

    def check_failures(self, results, world, runs):
        for rank, result in enumerate(results):
            self.assertEqual(len(result["fail"]), len(runs), rank)
            for case, (layers, failing, where) in zip(result["fail"], runs):
                with self.subTest(world=world, rank=rank, run=case["run"]):
                    self.assertEqual(case["run"], [layers, failing, where])
                    raised = case["raised"]
                    self.assertIsNotNone(raised)
                    self.assertTrue(
                        raised.startswith(
                            "RuntimeError: MiniMax H3 megakernel: sharding the "
                            f"AdaLN projections failed (rank {failing}: "
                        ),
                        raised,
                    )
                    if where == "peers":
                        # The check itself: every rank entered the all-gather,
                        # the failing rank saw wrong tables from its peers.
                        self.assertIn("differ from the replicated", raised)
                        self.assertEqual(case["gathers"], 1)
                    else:
                        # Before the all-gather: no rank entered it.
                        self.assertIn("OSError: injected", raised)
                        self.assertIn("sharding the AdaLN projections: ", raised)
                        self.assertEqual(case["gathers"], 0)
                    self.assertLess(case["seconds"], _FAIL_RAISE_S)
                    self.assertFalse(case["packed"])

    def test_a_failing_rank_makes_every_rank_raise_two_ranks(self):
        # Rank 1 fails at each point before the all-gather, or sees its
        # peer's tables wrong: rank 0 must raise too, not wait in a
        # collective rank 1 skipped.
        runs = [
            (6, 1, "probe"),
            (6, 1, "reference"),
            (6, 1, "share"),
            (6, 1, "sync"),
            (6, 1, "peers"),
            (6, 0, "share"),
        ]
        results = _spawn(2, [("fail", dict(runs=runs))], _FAIL_SPAWN_TIMEOUT_S)
        self.check_failures(results, 2, runs)

    def test_a_failing_rank_makes_every_rank_raise_four_ranks(self):
        # 6 blocks: rank 3 holds none (its share is zero tables); 50 blocks:
        # 13, 13, 13, 11.
        runs = [
            (6, 3, "share"),
            (6, 3, "sync"),
            (50, 3, "reference"),
            (6, 2, "probe"),
            (6, 0, "peers"),
            (50, 3, "peers"),
        ]
        results = _spawn(4, [("fail", dict(runs=runs))], _FAIL_SPAWN_TIMEOUT_S)
        self.check_failures(results, 4, runs)

    def test_eight_ranks(self):
        # 50 blocks: seven ranks of 7, the last with 1 and 6 zero tables.
        results = _spawn(
            8,
            [
                ("tables", dict(layers_list=[50])),
                ("enable", dict(layers=50)),
            ],
        )
        self.check_tables(results, 8)
        self.check_enable(results, 8, 50)


if __name__ == "__main__":
    unittest.main()
