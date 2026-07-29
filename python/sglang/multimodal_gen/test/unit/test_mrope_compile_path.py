import unittest
from unittest.mock import patch

import torch

from sglang.multimodal_gen.runtime.layers.rotary_embedding.mrope import (
    OneDRotaryEmbedding,
)


class TestOneDRotaryEmbeddingCompilePath(unittest.TestCase):
    def test_eager_path_uses_tolist_cache(self):
        # The lru_cache lives on the class; count misses/hits as deltas.
        rope = OneDRotaryEmbedding(dim=8)
        pos = torch.arange(4, dtype=torch.float32)
        before = rope._forward_cached.cache_info()
        cos, sin = rope.forward(pos)
        self.assertEqual(cos.shape, (4, 4))
        self.assertEqual(rope._forward_cached.cache_info().misses, before.misses + 1)
        rope.forward(pos)
        self.assertEqual(rope._forward_cached.cache_info().hits, before.hits + 1)

    def test_compile_path_bypasses_tolist_cache(self):
        # Under dynamo tracing, pos.tolist() is both a device sync and a graph
        # break; forward must route to the pure-tensor computation instead.
        rope = OneDRotaryEmbedding(dim=8)
        pos = torch.arange(4, dtype=torch.float32)
        before = rope._forward_cached.cache_info()
        with patch.object(torch.compiler, "is_compiling", return_value=True):
            cos_compiled, sin_compiled = rope.forward(pos)
        self.assertEqual(rope._forward_cached.cache_info(), before)
        cos_eager, sin_eager = rope.forward(pos)
        torch.testing.assert_close(cos_compiled, cos_eager)
        torch.testing.assert_close(sin_compiled, sin_eager)


if __name__ == "__main__":
    unittest.main()
