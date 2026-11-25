"""The vocab-split DSpark Markov drafts equal the unsharded greedy drafts on CPU,
including exact ties, which must go to the lowest vocab index (torch.argmax's
first maximum) whether the tied entries sit in one shard or in different ones."""

import os
import subprocess
import sys
import textwrap
import unittest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="base-a-test-cpu")

_CHECK = textwrap.dedent(
    """
    import os, sys
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
    import importlib.util
    import jax, jax.numpy as jnp, numpy as np
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    spec = importlib.util.spec_from_file_location("markov_shard", sys.argv[1])
    ms = importlib.util.module_from_spec(spec)
    sys.modules["torch"] = type(sys)("torch")  # the module only needs torch at call sites
    sys.modules["torch"].Tensor = object
    spec.loader.exec_module(ms)

    W, V, D, R, G = 8, 64, 32, 16, 3

    def reference(x, prev, w, w1, w2, bs):
        # greedy_markov_drafts: head logits rounded to bf16, one-hot w1 row,
        # bias rounded to w2's dtype, first-maximum argmax over the whole vocab.
        base = jnp.dot(x, w.T, preferred_element_type=jnp.float32)
        base = base.astype(jnp.bfloat16).astype(jnp.float32).reshape(bs, G, V)
        out = []
        for step in range(G):
            latent = w1[prev]
            bias = jnp.dot(latent.astype(w2.dtype), w2.T, preferred_element_type=jnp.float32)
            bias = bias.astype(w2.dtype).astype(jnp.float32)
            prev = jnp.argmax(base[:, step] + bias, axis=-1).astype(jnp.int32)
            out.append(prev)
        return jnp.stack(out, 1)

    mesh = Mesh(np.array(jax.devices()[:W]), ("model",))
    fn = jax.jit(ms.make_markov(mesh, W, V, G))
    rng = np.random.default_rng(0)
    for bs, tie in ((1, None), (2, None), (3, "across"), (2, "within")):
        x = jnp.asarray(rng.standard_normal((W * bs * G, D)), jnp.bfloat16)
        w = jnp.asarray(rng.standard_normal((V, D)) * 0.1, jnp.bfloat16)
        w1 = jnp.asarray(rng.standard_normal((V, R)), jnp.bfloat16)
        w2 = jnp.asarray(rng.standard_normal((V, R)) * 0.1, jnp.bfloat16)
        if tie is not None:
            # Make every score exactly equal on two vocab entries: zero head and
            # bias except two identical rows dominating everything else.
            w = jnp.zeros_like(w)
            w2 = jnp.zeros_like(w2)
            a, b = (5, 5 + 2 * (V // W)) if tie == "across" else (V // W + 1, V // W + 3)
            w2 = w2.at[a].set(1.0).at[b].set(1.0)
            w1 = jnp.abs(w1)
        prev = jnp.asarray(rng.integers(0, V, W * bs), jnp.int32)
        rank = jnp.arange(W, dtype=jnp.int32)
        shard = lambda a, s: jax.device_put(a, NamedSharding(mesh, s))
        got = fn(shard(x, P("model", None)), shard(prev, P("model")), shard(rank, P("model")),
                 shard(w, P("model", None)), shard(w1, P("model", None)), shard(w2, P("model", None)))
        # Rank r's rows are its bs requests; the reference takes all W*bs rows.
        want = reference(x, prev, w, w1, w2, W * bs)
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
        if tie is not None:
            assert int(np.asarray(got)[0, 0]) == min(a, b), (tie, np.asarray(got)[0])
    print("MARKOV_SHARD_OK")
    """
)


class MarkovShardTest(unittest.TestCase):
    def test_sharded_drafts_equal_unsharded_including_ties(self):
        path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "..",
            "..",
            "..",
            "sglang_tpu",
            "models",
            "deepseek_v41",
            "markov_shard.py",
        )
        proc = subprocess.run(
            [sys.executable, "-c", _CHECK, os.path.abspath(path)],
            capture_output=True,
            text=True,
            timeout=600,
        )
        self.assertIn("MARKOV_SHARD_OK", proc.stdout, proc.stderr[-3000:])

    def test_direct_update_refreshes_slices_in_place(self):
        import torch

        from sglang_tpu.models.deepseek_v41 import markov_shard as ms

        vocab, latent, world, rank = 8, 4, 2, 1
        shard = vocab // world
        model = torch.nn.Module()
        head = model.markov_head = torch.nn.Module()
        head.markov_w1 = torch.nn.Linear(latent, vocab, bias=False)
        head.markov_w2 = torch.nn.Linear(latent, vocab, bias=False)
        rows = slice(rank * shard, (rank + 1) * shard)
        head.register_buffer(ms._W1, head.markov_w1.weight.data[rows].clone())
        head.register_buffer(ms._W2, head.markov_w2.weight.data[rows].clone())
        head.register_buffer(ms._RANK, torch.tensor([rank], dtype=torch.int32))
        w1_buffer = getattr(head, ms._W1)

        new_w1 = torch.randn(vocab, latent)
        head.markov_w1.weight.data.copy_(new_w1)
        ms.tpu_dsv41_dspark_refresh_markov_after_direct_update(
            None, model, [("markov_head.markov_w1.weight", new_w1)]
        )
        self.assertIs(getattr(head, ms._W1), w1_buffer)
        torch.testing.assert_close(w1_buffer, new_w1[rows])
        torch.testing.assert_close(
            getattr(head, ms._W2), head.markov_w2.weight.data[rows]
        )
        # Unrelated names leave the slices alone.
        head.markov_w1.weight.data.zero_()
        ms.tpu_dsv41_dspark_refresh_markov_after_direct_update(
            None, model, [("lm_head.weight", new_w1)]
        )
        torch.testing.assert_close(w1_buffer, new_w1[rows])


if __name__ == "__main__":
    unittest.main()
