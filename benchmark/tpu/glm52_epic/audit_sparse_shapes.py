#!/usr/bin/env python3
"""Offline shape audit only; does not establish TPU lowering or numerical validity.

Execute the production pure-Python block selector via AST extraction so this
can run on a host without JAX/Torch. Run from any working directory.
"""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def main():
    source = ROOT / 'sglang_tpu/attention/pallas/mla_sparse_qblock_tokens.py'
    tree = ast.parse(source.read_text())
    constants = {'_MAX_ROWS', '_SCORE_ELEMS', '_MAX_PAGES_PER_STEP'}
    selected = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in constants
            for target in node.targets
        ):
            selected.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name == 'block_shape':
            selected.append(node)
    scope = {}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), 'exec'), scope)
    assert constants <= scope.keys() and 'block_shape' in scope
    heads = json.loads((HERE / 'model_config_evidence.json').read_text())
    # Evidence schema is inspected explicitly rather than assuming a default.
    def find_heads(value):
        if isinstance(value, dict):
            if 'num_attention_heads' in value:
                return value['num_attention_heads']
            for child in value.values():
                result = find_heads(child)
                if result is not None:
                    return result
        return None
    total_heads = find_heads(heads)
    assert isinstance(total_heads, int) and total_heads > 0
    results = []
    for name in ('torch_runnable_baseline.json', 'torch_compile_smoke.json'):
        config = json.loads((HERE / name).read_text())
        args = config['server_args']
        tp, dp = args['tp_size'], args['attn_dp_size']
        cp = args.get('attn_cp_size', 1)
        assert tp % (dp * cp) == 0
        attention_tp = tp // (dp * cp)
        assert total_heads % attention_tp == 0
        local_heads = total_heads // attention_tp
        local_chunk = args['chunked_prefill_size'] // dp
        for tokens in (128, 512, 2048):
            qb, pps = scope['block_shape'](local_heads, tokens, args['page_size'])
            results.append(dict(candidate=name, attention_tp=attention_tp,
                                local_heads=local_heads, tokens=tokens,
                                configured_local_chunk_ceiling=local_chunk,
                                within_eager_chunk_ceiling=tokens <= local_chunk,
                                qblock_length_eligible=tokens >= 512 and tokens % 128 == 0,
                                # Compile padding can exceed the eager live chunk;
                                # neither field establishes actual runtime routing.
                                runtime_path_verified=False,
                                query_block=qb, pages_per_step=pps,
                                head_query_rows=local_heads * qb,
                                exceeds_selector_row_budget=local_heads * qb > scope['_MAX_ROWS']))
    print(json.dumps({'scope': 'offline shape analysis; NOT a TPU compile test',
                      'selector_row_budget': scope['_MAX_ROWS'], 'cases': results}, indent=2))


if __name__ == '__main__':
    main()
