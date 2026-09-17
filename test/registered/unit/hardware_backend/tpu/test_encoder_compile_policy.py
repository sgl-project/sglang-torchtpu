"""The encoder Protocols must keep tracking the encoders they describe.

A Protocol member is only checked where the policy or runner uses it, and both
are built from an untyped owner, so nothing structurally verifies that the real
encoder satisfies the Protocol. A rename on the model side would leave the
Protocol silently stale and the failure would surface at serve time instead.
"""

import ast
import inspect
import unittest
from types import SimpleNamespace
from typing import Generic, Protocol

import torch

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.tpu.compilation.encoder_compile_policy import (
    MoonVitBlock,
    MoonVitEncoder,
    MoonVitRope2D,
    MoonVitTpuCompilePolicy,
    Qwen3VLPatchEmbed,
    Qwen3VLVisionEncoder,
)
from sglang.srt.hardware_backend.tpu.graph_runner.moonvit_tpu_compile_runner import (
    MoonVitBlockEncoder,
)
from sglang.srt.hardware_backend.tpu.graph_runner.vit_tpu_compile_runner import (
    Qwen3VLRotaryPosEmb,
    Qwen3VLViTEncoder,
)
from sglang.srt.layers.rotary_embedding.base import RotaryEmbedding
from sglang.srt.models.kimi_k25 import (
    MoonViT3dEncoder,
    MoonViTEncoderLayer,
    Rope2DPosEmbRepeated,
)
from sglang.srt.models.qwen3_vl import Qwen3VLMoeVisionModel, Qwen3VLVisionPatchEmbed
from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase

register_tpu_ci(est_time=10, suite="stage-a-test-tpu-unit")

# Each Protocol paired with the class it describes. Adding a member to a
# Protocol extends the check automatically; adding a Protocol does not. A model
# appears twice when the runner and the policy each declare their own face of
# it -- that separation is deliberate, so both faces get checked.
PROTOCOL_IMPLEMENTORS = [
    # Policy side.
    (Qwen3VLVisionEncoder, Qwen3VLMoeVisionModel),
    (Qwen3VLPatchEmbed, Qwen3VLVisionPatchEmbed),
    (MoonVitEncoder, MoonViT3dEncoder),
    (MoonVitBlock, MoonViTEncoderLayer),
    (MoonVitRope2D, Rope2DPosEmbRepeated),
    # Runner side.
    (Qwen3VLViTEncoder, Qwen3VLMoeVisionModel),
    (Qwen3VLRotaryPosEmb, RotaryEmbedding),
    (MoonVitBlockEncoder, MoonViT3dEncoder),
]


def _protocol_members(protocol) -> set[str]:
    """Declared members, including those inherited from a base Protocol."""
    members = set()
    for klass in protocol.__mro__:
        if klass in (object, Protocol, Generic):
            continue
        members |= set(getattr(klass, "__annotations__", {}))
        members |= {
            name
            for name, value in vars(klass).items()
            if not name.startswith("_") and callable(value)
        }
    return members


def _instance_members(klass) -> set[str]:
    """Names klass binds only on an instance: self.X and nn.Module buffers."""
    try:
        tree = ast.parse(inspect.getsource(klass))
    except (OSError, TypeError, SyntaxError):
        return set()

    found = set()
    for node in ast.walk(tree):
        # nn.Module binds buffers and parameters by name, not by assignment.
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in ("register_buffer", "register_parameter")
            and node.args
            and isinstance(node.args[0], ast.Constant)
        ):
            found.add(node.args[0].value)
        targets = (
            node.targets
            if isinstance(node, ast.Assign)
            else [node.target] if isinstance(node, ast.AnnAssign) else []
        )
        for target in targets:
            if isinstance(target, ast.Attribute) and (
                getattr(target.value, "id", None) == "self"
            ):
                found.add(target.attr)
    return found


def _provides(klass, name: str) -> bool:
    """Whether klass binds name anywhere in its MRO.

    hasattr covers methods, mixins, properties, and class attributes; the AST
    pass covers instance attributes, which exist only after __init__ runs and
    so cannot be probed without building the model.
    """
    if hasattr(klass, name):
        return True
    return any(name in _instance_members(base) for base in klass.__mro__)


class TestEncoderProtocolsMatchTheirEncoders(CustomTestCase):
    def test_every_protocol_member_exists_on_the_encoder(self):
        for protocol, klass in PROTOCOL_IMPLEMENTORS:
            with self.subTest(protocol=protocol.__name__, encoder=klass.__name__):
                declared = _protocol_members(protocol)
                self.assertTrue(declared, f"{protocol.__name__} declares nothing")
                missing = sorted(n for n in declared if not _provides(klass, n))
                self.assertEqual(
                    missing,
                    [],
                    f"{klass.__name__} no longer provides {missing}, which "
                    f"{protocol.__name__} declares.",
                )


class TestMoonVitOversizePatchCount(CustomTestCase):
    def test_patch_count_past_the_ladder_warns_once_instead_of_raising(self):
        """An image above the top bucket must degrade to a one-time recompile.

        prepare_call de-duplicates that warning through a set no __init__ ever
        built, so the first oversize image raised AttributeError and took the
        whole vision forward down instead of falling back.
        """
        policy = MoonVitTpuCompilePolicy(SimpleNamespace(blocks=[]))
        oversize = 400
        cu_seqlens = torch.tensor([0, oversize], dtype=torch.int32)

        with envs.SGLANG_TPU_VIT_PRECOMPILE_SIZES.override("196"), self.assertLogs(
            "sglang.srt.hardware_backend.tpu.compilation.encoder_compile_policy",
            level="WARNING",
        ) as logs:
            for _ in range(2):
                _, ctx = policy.prepare_call(
                    torch.zeros(oversize, 8),
                    torch.zeros(oversize, 4),
                    torch.zeros(oversize, 4),
                    cu_seqlens,
                )

        self.assertIsNone(ctx.bucket_len)
        self.assertFalse(ctx.padded)
        self.assertEqual(ctx.real_seq_len, oversize)
        # Warned on the first call, stayed quiet on the second.
        self.assertEqual(len(logs.output), 1)


if __name__ == "__main__":
    unittest.main()
