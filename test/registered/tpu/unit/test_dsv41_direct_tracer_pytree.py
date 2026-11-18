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

import torch
from jax import tree_util

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.models.deepseek_v41.attention_backend import (
    _METADATA_INPUT_FIELDS,
    DSV41TPUForwardMetadata,
    LowRatioMetadata,
)

register_tpu_ci(est_time=20, suite="stage-a-test-tpu-unit", chips=1)


def _metadata(engram=None) -> DSV41TPUForwardMetadata:
    inputs = {
        name: torch.arange(4, dtype=torch.int32) + i
        for i, name in enumerate(_METADATA_INPUT_FIELDS)
        if name not in ("low_ratio", "engram")
    }
    low = LowRatioMetadata(*(torch.zeros(4, dtype=torch.int32) for _ in range(3)))
    return DSV41TPUForwardMetadata(**inputs, low_ratio={1: low, 2: low}, engram=engram)


def _engram():
    staged = tuple((torch.zeros(6, 4), torch.zeros(6, 1)) for _ in range(2))
    return torch.zeros(3, 2, 2, dtype=torch.int32), staged


class TestDSV41MetadataPytree(CustomTestCase):
    def test_input_fields_are_the_host_built_fields(self):
        required = [
            f.name
            for f in dataclasses.fields(DSV41TPUForwardMetadata)
            if f.default is dataclasses.MISSING
            and f.default_factory is dataclasses.MISSING
        ]
        self.assertEqual(list(_METADATA_INPUT_FIELDS), [*required, "engram"])

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

    def test_key_paths_resolve_like_direct_compile(self):
        """A deferred direct_compile call fetches each leaf by its key path and
        accepts only GetAttr / Dict / Sequence keys (direct_compile._get_path)."""
        metadata = _metadata(engram=_engram())
        supported = (tree_util.GetAttrKey, tree_util.DictKey, tree_util.SequenceKey)
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


if __name__ == "__main__":
    unittest.main()
