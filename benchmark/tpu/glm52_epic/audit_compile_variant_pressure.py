#!/usr/bin/env python3
"""Static audit of TPU decoder precompile grid versus historical evidence.

No torch/TPU import. This reports requested graph-family cardinality and keeps
OOM/exit137 observations separate from compiler RSS attribution.
"""
import argparse
import json
import re
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--log', type=Path, required=True)
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())
    a = cfg['server_args']
    tokens = a.get('tpu_compile_tokens') or []
    bs = a.get('tpu_compile_bs') or []
    extend = len([t for t in tokens if t >= a.get('chunked_prefill_size', 0)]) or 1
    decode = len(bs) or 1
    text = args.log.read_text()
    extend_seen = sorted(set(re.findall(r'Compiled EXTEND \(tokens=(\d+), bs=(\d+)', text)))
    decode_grid = sorted(set(re.findall(r'DECODE: \d+ graph variants \(bs=\[([^]]*)\]', text)))
    result = {
        'config': str(args.config),
        'requested': {'extend_token_buckets': tokens, 'decode_batch_buckets': bs,
                      'extend_grid_lower_bound': extend, 'decode_grid_lower_bound': decode,
                      'total_grid_lower_bound': extend + decode},
        'historical_log': {'extend_compiled_shapes': [list(x) for x in extend_seen],
                           'decode_grid_messages': decode_grid,
                           'reported_compile_time_hbm_oom': 'CompileTimeHbmOom' in text,
                           'reported_exit137': bool(re.search(r'\b137\b', text))},
        'interpretation': [
            'Cardinality is a compile-request lower bound, not compiler subprocess count or RSS.',
            'exit137 with DP child EOF identifies a killed/terminated child but not kill cause.',
            'HBM usage lines are device HBM, not host compiler memory.'
        ],
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
