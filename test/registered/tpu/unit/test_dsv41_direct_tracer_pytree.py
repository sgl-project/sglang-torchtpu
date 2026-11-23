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
from types import SimpleNamespace

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
from sglang_tpu.models.deepseek_v41.decoder_replay import plan_decoder_replay
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


if __name__ == "__main__":
    unittest.main()
