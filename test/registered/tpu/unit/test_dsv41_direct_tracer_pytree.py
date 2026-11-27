# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1 attention metadata as a direct-tracer graph input.

The direct tracer flattens the backend metadata with JAX tree_util and, for a
deferred call, fetches every leaf back through its key path. Guards that the
V4.1 metadata flattens to the same structure whatever per-forward scratch a
previous forward left on it, that unflattening starts that scratch empty, and
that every key path resolves the way direct_compile resolves it. Runs on the
host; JAX comes with the TPU stack.
"""

import dataclasses
import unittest
from contextlib import ExitStack, nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from jax import tree_util
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase

from sglang_tpu.models.deepseek_v41.attention_backend import (
    _METADATA_INPUT_FIELDS,
    DecoderTail,
    DSV41TPUForwardMetadata,
    LowRatioMetadata,
    build_forward_metadata,
)
from sglang_tpu.models.deepseek_v41.decoder_replay import (
    decoder_prompt_bounds,
    plan_decoder_replay,
)
from sglang_tpu.models.deepseek_v41.decoder_replay_forward import (
    configure_decoder_replay,
    forward_decoder_replay,
)
from sglang_tpu.models.deepseek_v41.request_window import RESERVED_PAGES, WINDOW_TOKENS

register_tpu_ci(est_time=20, suite="stage-a-test-tpu-unit", chips=1)


def _metadata(engram=None) -> DSV41TPUForwardMetadata:
    inputs = {
        name: torch.arange(4, dtype=torch.int32) + i
        for i, name in enumerate(_METADATA_INPUT_FIELDS)
        if name
        not in (
            "low_ratio",
            "engram",
            "swa_window_plan",
            "late_layer_tail",
            "decoder_metadata",
        )
    }
    low = LowRatioMetadata(*(torch.zeros(4, dtype=torch.int32) for _ in range(3)))
    return DSV41TPUForwardMetadata(**inputs, low_ratio={1: low, 2: low}, engram=engram)


def _engram():
    staged = tuple((torch.zeros(6, 4), torch.zeros(6, 1)) for _ in range(2))
    return torch.zeros(3, 2, 2, dtype=torch.int32), staged


def _packed_engram():
    staged = tuple(torch.zeros(16, dtype=torch.int32) for _ in range(2))
    return torch.zeros(3, 2, 2, dtype=torch.int32), staged


class TestDSV41MetadataPytree(CustomTestCase):
    def test_input_fields_are_the_host_built_fields(self):
        required = [
            f.name
            for f in dataclasses.fields(DSV41TPUForwardMetadata)
            if f.default is dataclasses.MISSING
            and f.default_factory is dataclasses.MISSING
        ]
        self.assertEqual(
            list(_METADATA_INPUT_FIELDS),
            [
                *required,
                "engram",
                "swa_window_plan",
                "late_layer_tail",
                "decoder_metadata",
            ],
        )

    def test_scratch_does_not_change_the_structure(self):
        clean = _metadata()
        used = _metadata()
        used.swa_meta = torch.ones(3)
        used.selected[2] = torch.ones(3)
        used.prepared_candidates[(2, 8)] = (torch.ones(1),)
        self.assertEqual(
            tree_util.tree_structure(clean), tree_util.tree_structure(used)
        )

    def test_unflatten_starts_scratch_empty(self):
        used = _metadata(engram=_engram())
        used.selected[2] = torch.ones(3)
        leaves, treedef = tree_util.tree_flatten(used)
        rebuilt = tree_util.tree_unflatten(treedef, leaves)
        self.assertIsNone(rebuilt.swa_meta)
        self.assertEqual(rebuilt.selected, {})
        self.assertEqual(rebuilt.prepared_candidates, {})
        self.assertIs(rebuilt.positions, used.positions)
        self.assertIs(rebuilt.low_ratio[2].out_loc, used.low_ratio[2].out_loc)
        self.assertIs(rebuilt.engram[1][1][0], used.engram[1][1][0])

    def test_decoder_replay_lifts_rank_specific_metadata(self):
        def replica(rows):
            metadata = _metadata()
            metadata.decoder_metadata = _metadata()
            indices = torch.zeros(128, dtype=torch.int32)
            metadata.late_layer_tail = DecoderTail(
                indices,
                indices.clone(),
                indices.clone(),
                torch.tensor(rows, dtype=torch.int32),
                (rows,),
                False,
            )
            return metadata

        idle, live = replica(0), replica(128)
        self.assertEqual(tree_util.tree_structure(idle), tree_util.tree_structure(live))
        paths, definition = tree_util.tree_flatten_with_path(live)
        leaves = {tree_util.keystr(path): leaf for path, leaf in paths}
        self.assertEqual(int(leaves[".late_layer_tail.num_real_rows"]), 128)
        self.assertIs(
            leaves[".decoder_metadata.positions"], live.decoder_metadata.positions
        )
        rebuilt = tree_util.tree_unflatten(definition, [leaf for _, leaf in paths])
        self.assertEqual(int(rebuilt.late_layer_tail.num_real_rows), 128)
        self.assertFalse(rebuilt.late_layer_tail.skip_decoder)

    def test_key_paths_resolve_like_direct_compile(self):
        """A deferred direct_compile call fetches each leaf by its key path and
        accepts only GetAttr / Dict / Sequence keys (direct_compile._get_path)."""
        supported = (tree_util.GetAttrKey, tree_util.DictKey, tree_util.SequenceKey)
        for engram in (_engram(), _packed_engram()):
            metadata = _metadata(engram=engram)
            paths = tree_util.tree_flatten_with_path(metadata)[0]
            self.assertEqual(len(paths), len(tree_util.tree_leaves(metadata)))
            for path, leaf in paths:
                value = metadata
                for key in path:
                    self.assertIsInstance(key, supported, msg=str(path))
                    if isinstance(key, tree_util.GetAttrKey):
                        value = getattr(value, key.name)
                    elif isinstance(key, tree_util.DictKey):
                        value = value[key.key]
                    else:
                        value = value[key.idx]
                self.assertIs(value, leaf)


class TestDecoderReplayMetadata(CustomTestCase):
    def test_only_prompt_logprobs_require_full_decoder_history(self):
        prompt = range(4096)
        generated_only = SimpleNamespace(
            full_untruncated_fill_ids=range(4100),
            origin_input_ids=prompt,
            extend_end=2048,
            return_logprob=True,
            logprob_start_len=len(prompt),
            return_hidden_states_mode=SimpleNamespace(is_full=lambda: False),
        )
        prompt_logprobs = SimpleNamespace(**vars(generated_only))
        prompt_logprobs.logprob_start_len = len(prompt) - 2
        for req, whole in ((generated_only, False), (prompt_logprobs, True)):
            with self.subTest(logprob_start_len=req.logprob_start_len):
                batch = SimpleNamespace(reqs=[req], chunked_req=req)
                self.assertEqual(
                    decoder_prompt_bounds(batch, keep_hidden=False),
                    ((4100,), (whole,)),
                )

    def test_middle_and_crossing_chunks_use_same_prompt_window(self):
        plan = plan_decoder_replay(
            prefix_lengths=(98304 - 128, 2048, 2048, 2048),
            extend_lengths=(2048, 64, 64, 64),
            prompt_ends=(114688, 2210, 2210, 2210),
            whole_extend=(False, False, True, False),
            window=WINDOW_TOKENS,
        )
        self.assertEqual(plan.lengths, (0, 30, 64, 30))
        self.assertEqual(plan.swa_floors, (114560, 2082, 0, 2082))
        self.assertEqual(plan.row_indices[:30], tuple(range(2082, 2112)))
        self.assertEqual(plan.row_indices[30:94], tuple(range(2112, 2176)))

    def test_folded_slots_stay_read_only_and_tail_history_starts_at_window(self):
        # The first two rows are folded replay. The decoder starts at position
        # 128, where it may see no ring history until this step publishes it.
        batch = SimpleNamespace(
            batch_size=1,
            forward_mode=ForwardMode.EXTEND,
            positions=torch.arange(128, 134),
            out_cache_loc=torch.arange(640, 646),
            extend_seq_lens_cpu=[6],
            req_pool_indices=torch.tensor([1]),
            encoder_swa_row_floor=torch.full((6,), 128),
            encoder_swa_compress_rows=torch.arange(2, 6),
        )
        plan = plan_decoder_replay(
            prefix_lengths=(128,),
            extend_lengths=(6,),
            prompt_ends=(256,),
            whole_extend=(False,),
            window=WINDOW_TOKENS,
        )
        decoder = build_forward_metadata(
            batch,
            req_to_token=torch.arange(512, 1024).reshape(2, 256),
            full_to_swa_index_mapping=None,
            request_window=SimpleNamespace(
                workspace_first_page=RESERVED_PAGES + 8, workspace_pages=8
            ),
            swa_ring_valid_from=None,
            max_context_len=256,
            low_ratios=(1,),
            device=torch.device("cpu"),
            decoder_plan=plan,
            decoder_rows=128,
        )
        self.assertEqual(decoder.query_start_loc.tolist(), [0, 6])
        self.assertEqual(decoder.swa_minimum.tolist(), [128])
        self.assertEqual(
            decoder.low_ratio[1].out_loc[:8].tolist(),
            [-1, -1, 642, 643, 644, 645, -1, -1],
        )
        self.assertTrue(torch.all(decoder.swa_window_plan.history_dst == -1))
        self.assertEqual(int(decoder.valid.sum()), 6)


class TestDecoderReplayForward(CustomTestCase):
    def test_restore_rows_ignores_padding_on_idle_and_live_ranks(self):
        values = torch.tensor([[7.0], [float("nan")], [float("nan")]])
        for count, output_indices, expected in (
            (0, [3, 3, 3, 3], [[0.0], [0.0], [0.0], [0.0]]),
            (1, [3, 0, 3, 3], [[0.0], [7.0], [0.0], [0.0]]),
        ):
            tail = DecoderTail(
                token_indices=torch.tensor([1, 0, 0], dtype=torch.int32),
                output_indices=torch.tensor(output_indices, dtype=torch.int32),
                positions=torch.tensor([1, 0, 0], dtype=torch.int32),
                num_real_rows=torch.tensor(count, dtype=torch.int32),
                extend_seq_lens_cpu=(count,),
                skip_decoder=False,
            )
            torch.testing.assert_close(tail.restore_rows(values), torch.tensor(expected))

    def test_decoder_replay_rejects_unshifted_mhc(self):
        from sglang.srt import runtime_context

        model = SimpleNamespace(late_layer_start=1, hc_pre_from_prev_sublayer=False)
        config = SimpleNamespace(model_type="deepseek_v41")
        with patch.object(
            runtime_context, "get_platform", lambda: SimpleNamespace(is_tpu=True)
        ):
            with self.assertRaisesRegex(ValueError, "requires shifted mHC"):
                configure_decoder_replay(Mock(), model, config)

    def test_hook_keeps_source_write_full_and_restores_output_layout(self):
        from sglang.kernels.ops.layernorm import mhc as ops
        from sglang.srt import runtime_context
        from sglang.srt.eplb import expert_distribution
        from sglang.srt.layers import moe
        from sglang.srt.model_executor import forward_context
        from sglang.srt.models import deepseek_v4_mhc as mhc

        from sglang_tpu.models.deepseek_v41 import decoder_replay_forward

        def forward_layer(*, state, input_ids, input_ids_global, **_):
            torch.testing.assert_close(input_ids_global, input_ids.repeat(2))
            return mhc.HcState(
                state.residual + 1,
                state.residual.new_full((state.residual.shape[0], 2), 0.5),
            )

        def record_source(*, x, positions, **_):
            torch.testing.assert_close(x.squeeze(1), positions.float() + 1)

        before = Mock(side_effect=forward_layer)
        after = Mock(side_effect=forward_layer)
        model = SimpleNamespace(
            late_layer_start=1,
            hc_pre_from_prev_sublayer=True,
            dspark_layers_to_capture=(0, 1),
            pp_group=SimpleNamespace(world_size=1),
            hidden_size=1,
            hc_mult=2,
            start_layer=0,
            end_layer=2,
            engram_hasher=None,
            embed_tokens=lambda ids: ids[:, None].float(),
            norm=lambda x: x,
            layers=[
                SimpleNamespace(engram=None, forward_hc_pre_from_prev=before),
                SimpleNamespace(
                    engram=None,
                    self_attn=object(),
                    attn_hc=object(),
                    hc_cfg=None,
                    forward_hc_pre_from_prev=after,
                ),
            ],
        )
        backend = SimpleNamespace(
            decoder_replay_enabled=True,
            write_tail_source_kv=Mock(side_effect=record_source),
            enter_late_layer_tail=Mock(),
            exit_late_layer_tail=Mock(),
        )
        original = Mock()
        ids = torch.arange(4)
        batch = SimpleNamespace(forward_mode=ForwardMode.EXTEND)

        with ExitStack() as stack:
            patches = [
                (runtime_context, "get_platform", lambda: SimpleNamespace(is_tpu=True)),
                (
                    runtime_context,
                    "get_parallel",
                    lambda: SimpleNamespace(attn_dp_size=2),
                ),
                (
                    runtime_context,
                    "get_forward",
                    lambda: SimpleNamespace(sp_active=False),
                ),
                (forward_context, "get_attn_backend", lambda: backend),
                (decoder_replay_forward, "_gather_ids", lambda ids, _: ids.repeat(2)),
                (
                    moe,
                    "get_moe_a2a_backend",
                    lambda: SimpleNamespace(is_none=lambda: True),
                ),
                (
                    expert_distribution,
                    "get_global_expert_distribution_recorder",
                    lambda: SimpleNamespace(with_current_layer=lambda _: nullcontext()),
                ),
                (mhc, "combine", lambda _, state: state.residual.mean(dim=1)),
                (ops, "hc_broadcast", lambda x, hc: x[:, None, :].expand(-1, hc, -1)),
                (ops, "mhc_stream_mean", lambda x, *, out: out.copy_(x.mean(dim=1))),
                (
                    ops,
                    "mhc_combine",
                    lambda x, pre, hc, _: (x.view(-1, hc, 1) * pre[..., None]).sum(
                        dim=1
                    ),
                ),
            ]
            for module, name, fn in patches:
                stack.enter_context(patch.object(module, name, fn))

            configure_decoder_replay(
                Mock(), model, SimpleNamespace(model_type="deepseek_v41")
            )
            self.assertIsNone(model.late_layer_start)
            for skip in (True, False):
                with self.subTest(skip=skip):
                    tail = DecoderTail(
                        torch.tensor([2, 0]),
                        torch.tensor([2, 2, 0, 2], dtype=torch.int32),
                        torch.tensor([2, 0]),
                        torch.tensor(0 if skip else 1),
                        (0 if skip else 1,),
                        skip,
                    )
                    backend.tail_forward_metadata = SimpleNamespace(
                        late_layer_tail=tail
                    )
                    (hidden, pre), aux = forward_decoder_replay(
                        original, model, ids, ids, batch, None
                    )
                    self.assertEqual(hidden[:, 0].tolist(), [0, 0, 0 if skip else 4, 0])
                    self.assertEqual(pre[:, 0].tolist(), hidden[:, 0].tolist())
                    self.assertEqual(aux[2].tolist(), [0, 0] if skip else [2, 3])
                    self.assertEqual(after.call_count, int(not skip))
                    self.assertEqual(
                        backend.exit_late_layer_tail.call_count, int(not skip)
                    )
            self.assertEqual(before.call_count, 2)
            self.assertEqual(backend.write_tail_source_kv.call_count, 2)
            original.assert_not_called()
            batch.forward_mode = ForwardMode.TARGET_VERIFY
            forward_decoder_replay(original, model, ids, ids, batch, None)
            original.assert_called_once()


if __name__ == "__main__":
    unittest.main()
