"""The TPU int32 Qwen4-Exp n-gram hash matches the int64 reference bit for bit.

``sglang_tpu.hooks.qwen4_exp.ngram_hash_int32`` replaces
``Qwen4ExpNGramEmbedding._hash_contexts`` on TPU, whose int64 vector arithmetic
aborts the TPU compiler on decode-sized graphs. The reference here restates the
core hash (multiplier XOR over EOS-segmented shifts, mod each head's prime).
"""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=10, suite="stage-a-test-tpu-unit", chips=1)

import unittest
from types import SimpleNamespace

import sympy
import torch

from sglang_tpu.hooks.qwen4_exp import ngram_hash_int32

MASK64 = (1 << 64) - 1
GAMMA, M1, M2, PRIME = 0x9E3779B97F4A7C15, 0xBF58476D1CE4E5B9, 0x94D049BB133111EB, 10007


def _splitmix(x):
    x = (x + GAMMA) & MASK64
    x = ((x ^ (x >> 30)) * M1) & MASK64
    x = ((x ^ (x >> 27)) * M2) & MASK64
    return (x ^ (x >> 31)) & MASK64


def _ngram(vocab=248320, ngram_size=3, heads_per_ngram=8, eos=248044, layer=0):
    half = (((1 << 63) - 1) // vocab) // 2
    multipliers = [
        2 * (_splitmix((1234 + PRIME * layer + GAMMA * (i + 1)) & MASK64) % half) + 1
        for i in range(ngram_size)
    ]
    sizes, offsets, total, prime = [], [], 0, 20000000 - 1
    for _ in range((ngram_size - 1) * heads_per_ngram):
        prime = int(sympy.nextprime(prime))
        sizes.append(prime)
        offsets.append(total)
        total += prime
    return SimpleNamespace(
        unigram_vocab_size=vocab,
        ngram_size=ngram_size,
        heads_per_ngram=heads_per_ngram,
        eos_token_id=eos,
        layer_multipliers=torch.tensor(multipliers, dtype=torch.long),
        ngram_heads_vocab_sizes=torch.tensor(sizes),
        ngram_heads_offsets=torch.tensor(offsets),
    )


def _shift(ngram, t, k):
    if k == 0:
        return t
    b, length = t.shape
    idx = torch.arange(length)
    eos_pos = torch.where(t == ngram.eos_token_id, idx, -1)
    prev = torch.cat(
        [eos_pos.new_full((b, 1), -1), torch.cummax(eos_pos, 1).values[:, :-1]], 1
    )
    in_segment = idx.unsqueeze(0) - (prev + 1)
    src = idx - k
    shifted = t.gather(1, torch.clamp(src, min=0).unsqueeze(0).expand(b, -1))
    keep = (in_segment >= k) & (src.unsqueeze(0) >= 0)
    return torch.where(keep, shifted, t.new_full((), ngram.eos_token_id))


def _reference(ngram, contexts):
    contexts = contexts.long()
    shifted = [_shift(ngram, contexts, k) for k in range(ngram.ngram_size)]
    blocks = []
    for g in range(2, ngram.ngram_size + 1):
        s = (g - 2) * ngram.heads_per_ngram
        e = s + ngram.heads_per_ngram
        mix = shifted[0] * ngram.layer_multipliers[0]
        for pos in range(1, g):
            mix = torch.bitwise_xor(mix, shifted[pos] * ngram.layer_multipliers[pos])
        ids = torch.remainder(
            mix[:, -1:].unsqueeze(-1), ngram.ngram_heads_vocab_sizes[s:e].view(1, 1, -1)
        )
        blocks.append((ids + ngram.ngram_heads_offsets[s:e].view(1, 1, -1))[:, 0])
    return torch.cat(blocks, -1)


class TestNgramHashInt32(unittest.TestCase):
    def _check(self, ngram, n=4000, eos_rate=0.25, seed=0):
        torch.manual_seed(seed)
        ctx = torch.randint(0, ngram.unigram_vocab_size, (n, ngram.ngram_size))
        ctx = torch.where(
            torch.rand(ctx.shape) < eos_rate,
            torch.full_like(ctx, ngram.eos_token_id),
            ctx,
        )
        ctx[:16] = ngram.unigram_vocab_size - 1
        got = ngram_hash_int32(ngram, ctx)
        self.assertEqual(got.dtype, torch.int32)
        self.assertTrue(torch.equal(got.long(), _reference(ngram, ctx)))

    def test_qwen3_8_flash_next_geometry(self):
        self._check(_ngram())

    def test_other_ple_layer_multipliers(self):
        self._check(_ngram(layer=3), seed=1)

    def test_no_eos(self):
        self._check(_ngram(), eos_rate=0.0, seed=2)


if __name__ == "__main__":
    unittest.main()
