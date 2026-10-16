"""Warm the bounded eager sampling tail after the decoder bucket grid."""

from itertools import product

import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.sampling.sampling_params import TOP_K_ALL
from sglang_tpu.sampler import MAX_DISTRIBUTED_TOP_K


def _sampling_variants():
    """(greedy, top_k, top_p, min_p, mixed_k) combinations that reach distinct kernels."""
    filters = list(product((False, True), repeat=2))
    variants = [(True, False, False, False, False), (False, False, False, False, False)]
    # Uniform small k: gathered top-k block.
    variants += [(False, True, p, m, False) for p, m in filters]
    # No top-k but top-p/min-p: full-vocabulary filter without the top-k mask.
    variants += [(False, False, p, m, False) for p, m in filters if p or m]
    # Greedy and sampled rows in one batch: no uniform k, full-vocabulary filter.
    variants += [(False, True, p, m, True) for p, m in filters]
    return variants


@torch.no_grad()
def warmup_host_sampling(sampler, buckets, vocab_size, device, *, make_logits):
    width = min(MAX_DISTRIBUTED_TOP_K, vocab_size)
    for rows in buckets:
        positions = torch.zeros(rows, dtype=torch.int64, device=device)
        for greedy, top_k, top_p, min_p, mixed_k in _sampling_variants():
            # Use the real eager LM head when it lives outside the decoder graph.
            logits = make_logits(rows)
            k = 1 if greedy else width if top_k else TOP_K_ALL
            top_ks = torch.full((rows,), k, dtype=torch.int32)
            if mixed_k:
                top_ks[::2] = 1
            info = SamplingBatchInfo(
                temperatures=torch.ones((rows, 1)),
                top_ps=torch.full((rows,), 0.9 if top_p else 1.0),
                top_ks=top_ks,
                min_ps=torch.full((rows,), 0.05 if min_p else 0.0),
                is_all_greedy=greedy,
                is_any_greedy=greedy or mixed_k,
                need_top_p_sampling=top_p,
                need_top_k_sampling=top_k,
                need_min_p_sampling=min_p,
                vocab_size=vocab_size,
                return_sampling_masks=[False] * rows,
                device=str(device),
                uniform_top_k=None if mixed_k else k,
                sampling_params_on_cpu=True,
            )
            result = sampler(
                LogitsProcessorOutput(next_token_logits=logits),
                info,
                False,
                None,
                None,
                positions,
            )
            result.cpu()
