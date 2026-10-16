"""Warm the bounded eager sampling tail after the decoder bucket grid."""

from itertools import product

import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.sampling.sampling_params import TOP_K_ALL
from sglang_tpu.sampler import MAX_DISTRIBUTED_TOP_K


@torch.no_grad()
def warmup_host_sampling(sampler, buckets, vocab_size, device, *, make_logits):
    for rows in buckets:
        positions = torch.zeros(rows, dtype=torch.int64, device=device)
        width = min(MAX_DISTRIBUTED_TOP_K, vocab_size)
        variants = [(True, False, False, False), (False, False, False, False)]
        variants += [(False, True, p, m) for p, m in product((False, True), repeat=2)]
        for greedy, top_k, top_p, min_p in variants:
            # Use the real eager LM head when it lives outside the decoder graph.
            logits = make_logits(rows)
            k = 1 if greedy else width if top_k else TOP_K_ALL
            info = SamplingBatchInfo(
                temperatures=torch.ones((rows, 1)),
                top_ps=torch.full((rows,), 0.9 if top_p else 1.0),
                top_ks=torch.full((rows,), k, dtype=torch.int32),
                min_ps=torch.full((rows,), 0.05 if min_p else 0.0),
                is_all_greedy=greedy,
                is_any_greedy=greedy,
                need_top_p_sampling=top_p,
                need_top_k_sampling=top_k,
                need_min_p_sampling=min_p,
                vocab_size=vocab_size,
                return_sampling_masks=[False] * rows,
                device=str(device),
                uniform_top_k=k,
                params_on_cpu=True,
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
