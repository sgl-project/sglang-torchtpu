# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Unit tests for the MoE-CP allgather / scatter branches in the
communicator, gated on the plain-int ``moe_cp_*`` mirrors that the CP-v2
strategy sets on ForwardBatch in eager mode.

The gather/scatter math is what routes MoE tokens across CP ranks; any
regression to the ``moe_cp_pad_max_tokens`` vs ``moe_cp_local_tokens``
math produces silently mismatched ranks (either wrong shapes, or worse:
wrong slices of hidden states given to MoE experts).

The collective (``moe_cp_all_gather_into_tensor``) is stubbed with a
scripted fake that fills the output tensor with per-rank sentinels — we
verify the scatter arithmetic against those sentinels rather than
against a real NCCL/PJRT round-trip.
"""

from __future__ import annotations

import types
import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers.communicator import (
    CommunicateSummableTensorPairFn,
    CommunicateWithAllReduceAndLayerNormFn,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _NoopLayerNorm(torch.nn.Module):
    """LayerNorm stand-in: returns inputs unchanged so gather math is
    isolated from normalization."""

    def forward(self, hidden_states, residual=None):  # noqa: D401
        return hidden_states, residual


def _fake_forward_batch(
    *,
    moe_cp_active: bool,
    moe_cp_pad_max_tokens: int,
    moe_cp_local_tokens: int,
    attn_cp_metadata=None,
    is_context_parallel_extend: bool = True,
):
    forward_mode = types.SimpleNamespace(
        is_context_parallel_extend=lambda: is_context_parallel_extend
    )
    return types.SimpleNamespace(
        moe_cp_active=moe_cp_active,
        moe_cp_pad_max_tokens=moe_cp_pad_max_tokens,
        moe_cp_local_tokens=moe_cp_local_tokens,
        attn_cp_metadata=attn_cp_metadata,
        forward_mode=forward_mode,
    )


def _make_context():
    """The gather/scatter branches only look at ``context.attn_dp_size``."""
    return types.SimpleNamespace(attn_dp_size=1)


class MoeCpGatherTest(CustomTestCase):
    def _patch_get_moe_cp_size(self, size: int):
        return patch(
            "sglang.srt.layers.communicator.get_moe_cp_size", return_value=size
        )

    def _patch_prior_gather(self):
        """Bypass the standard TP all-reduce + DP allgather layer so we
        can isolate the MoE-CP branch behavior."""

        def _passthrough(*, hidden_states, residual, **_kw):
            return hidden_states, residual

        return patch.object(
            CommunicateWithAllReduceAndLayerNormFn,
            "_gather_hidden_states_and_residual",
            staticmethod(_passthrough),
        )

    def _patch_allgather(self):
        """Simulate an in-place allgather: fill the destination with
        ``input`` for our rank, sentinel-1 elsewhere. Test only checks
        that the destination shape matches ``pad_max_tokens * moe_cp_size``.
        """

        def _fake(output, hidden_states):
            output.zero_()
            output[: hidden_states.shape[0]] = hidden_states
            # Mark "other rank" rows with a sentinel so the scatter test
            # can verify the correct slice was pulled back.
            output[hidden_states.shape[0] :] = 1.0

        return patch(
            "sglang.srt.layers.communicator.moe_cp_all_gather_into_tensor",
            side_effect=_fake,
        )

    def test_gather_noop_when_moe_cp_inactive(self) -> None:
        # moe_cp_size=2 but moe_cp_active=False -> gather returns input.
        hidden = torch.arange(8 * 4, dtype=torch.float32).reshape(8, 4)
        residual = torch.zeros_like(hidden)
        fb = _fake_forward_batch(
            moe_cp_active=False,
            moe_cp_pad_max_tokens=0,
            moe_cp_local_tokens=0,
        )
        with self._patch_get_moe_cp_size(2), self._patch_prior_gather():
            out_h, out_r = (
                CommunicateWithAllReduceAndLayerNormFn._gather_hidden_states_and_residual_moe(
                    hidden_states=hidden,
                    residual=residual,
                    forward_batch=fb,
                    layernorm=_NoopLayerNorm(),
                    context=_make_context(),
                    residual_input_mode=None,
                )
            )
        self.assertIs(out_h, hidden)
        self.assertIs(out_r, residual)

    def test_gather_pads_and_allgathers_when_active(self) -> None:
        # 6 real tokens, pad_max=8 (2-token pad), moe_cp_size=2 → output
        # shape should be (16, 4).
        hidden = torch.arange(6 * 4, dtype=torch.float32).reshape(6, 4)
        residual = torch.zeros_like(hidden)
        fb = _fake_forward_batch(
            moe_cp_active=True,
            moe_cp_pad_max_tokens=8,
            moe_cp_local_tokens=6,
        )
        with (
            self._patch_get_moe_cp_size(2),
            self._patch_prior_gather(),
            self._patch_allgather() as spy,
        ):
            out_h, _out_r = (
                CommunicateWithAllReduceAndLayerNormFn._gather_hidden_states_and_residual_moe(
                    hidden_states=hidden,
                    residual=residual,
                    forward_batch=fb,
                    layernorm=_NoopLayerNorm(),
                    context=_make_context(),
                    residual_input_mode=None,
                )
            )
        self.assertEqual(out_h.shape, (16, 4))
        spy.assert_called_once()
        # The pad-then-allgather sequence must feed the allgather a tensor
        # of shape (pad_max=8, 4), NOT the raw (6, 4).
        (called_output, called_hidden), _kw = spy.call_args
        self.assertEqual(called_hidden.shape, (8, 4))
        self.assertEqual(called_output.shape, (16, 4))
        # First 6 rows must equal the original hidden; last 2 are zero pad.
        torch.testing.assert_close(called_hidden[:6], hidden)
        torch.testing.assert_close(called_hidden[6:], torch.zeros(2, 4))

    def test_gather_skips_when_moe_cp_size_1(self) -> None:
        # moe_cp_size=1 means no cross-rank gather is needed regardless of
        # the active flag.
        hidden = torch.arange(8 * 4, dtype=torch.float32).reshape(8, 4)
        residual = torch.zeros_like(hidden)
        fb = _fake_forward_batch(
            moe_cp_active=True,
            moe_cp_pad_max_tokens=8,
            moe_cp_local_tokens=8,
        )
        with (
            self._patch_get_moe_cp_size(1),
            self._patch_prior_gather(),
            self._patch_allgather() as spy,
        ):
            out_h, _ = (
                CommunicateWithAllReduceAndLayerNormFn._gather_hidden_states_and_residual_moe(
                    hidden_states=hidden,
                    residual=residual,
                    forward_batch=fb,
                    layernorm=_NoopLayerNorm(),
                    context=_make_context(),
                    residual_input_mode=None,
                )
            )
        self.assertIs(out_h, hidden)
        spy.assert_not_called()


class MoeCpScatterTest(CustomTestCase):
    def _patch_get_moe_cp_size(self, size: int):
        return patch(
            "sglang.srt.layers.communicator.get_moe_cp_size", return_value=size
        )

    def _patch_get_moe_cp_rank(self, rank: int):
        return patch(
            "sglang.srt.layers.communicator.get_moe_cp_rank", return_value=rank
        )

    def test_scatter_extracts_this_rank_local_slice(self) -> None:
        # After the (fake) allgather each rank holds the padded stack of
        # rank chunks. Rank 1's slice starts at index rank * pad_max = 8.
        # local_tokens=5 -> take rows [8, 13).
        pad_max = 8
        moe_cp_size = 2
        # Construct a deterministic stacked buffer: rank 0 rows are 0..7,
        # rank 1 rows are 100..107.
        rank_0 = (
            torch.arange(0, pad_max, dtype=torch.float32)
            .unsqueeze(1)
            .expand(pad_max, 4)
        )
        rank_1 = (
            torch.arange(100, 100 + pad_max, dtype=torch.float32)
            .unsqueeze(1)
            .expand(pad_max, 4)
        )
        stacked = torch.cat([rank_0, rank_1], dim=0).contiguous()
        fb = _fake_forward_batch(
            moe_cp_active=True,
            moe_cp_pad_max_tokens=pad_max,
            moe_cp_local_tokens=5,
        )
        with (
            self._patch_get_moe_cp_size(moe_cp_size),
            self._patch_get_moe_cp_rank(1),
        ):
            out_h, _ = CommunicateSummableTensorPairFn._scatter_hidden_states_moe(
                hidden_states=stacked,
                residual=None,
                forward_batch=fb,
                context=_make_context(),
            )
        self.assertEqual(out_h.shape, (5, 4))
        # Rank 1's rows 8..12 map to values 100..104.
        expected = torch.arange(100, 105, dtype=torch.float32).unsqueeze(1).expand(5, 4)
        torch.testing.assert_close(out_h, expected)

    def test_scatter_noop_when_moe_cp_inactive(self) -> None:
        hidden = torch.arange(6 * 4, dtype=torch.float32).reshape(6, 4)
        fb = _fake_forward_batch(
            moe_cp_active=False,
            moe_cp_pad_max_tokens=0,
            moe_cp_local_tokens=0,
        )
        with (
            self._patch_get_moe_cp_size(2),
            self._patch_get_moe_cp_rank(0),
        ):
            out_h, _ = CommunicateSummableTensorPairFn._scatter_hidden_states_moe(
                hidden_states=hidden,
                residual=None,
                forward_batch=fb,
                context=_make_context(),
            )
        self.assertIs(out_h, hidden)


if __name__ == "__main__":
    unittest.main()
