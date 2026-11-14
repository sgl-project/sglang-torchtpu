"""CPU-only selector regression audit; does not establish TPU layout/VMEM fit.

Extracts the actual pure selector and its constants without importing JAX.
Compares unchanged low-head outcomes to the pre-fix committed implementation.
"""
import ast
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
SOURCE = 'sglang_tpu/attention/pallas/mla_sparse_qblock_tokens.py'
BEFORE = '1e9935c863b13653ec19adbb9d3cffb190fe75b5'
CONSTANTS = {'_MAX_ROWS', '_SCORE_ELEMS', '_MAX_PAGES_PER_STEP'}


def load_selector(text):
    tree = ast.parse(text)
    constants = [node for node in tree.body if isinstance(node, ast.Assign)
                 and len(node.targets) == 1
                 and isinstance(node.targets[0], ast.Name)
                 and node.targets[0].id in CONSTANTS]
    assert len(constants) == len(CONSTANTS)
    fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
              and node.name == 'block_shape')
    namespace = {}
    exec(compile(ast.Module(body=constants + [fn], type_ignores=[]), SOURCE, 'exec'), namespace)
    return namespace['block_shape']


def main():
    old = load_selector(subprocess.check_output(
        ['git', 'show', f'{BEFORE}:{SOURCE}'], cwd=ROOT, text=True))
    current = load_selector((ROOT / SOURCE).read_text())
    count = 0
    for heads in range(1, 25):
        for tokens in (128, 256, 384, 512, 1024, 2048):
            for page in (128, 256, 512):
                assert current(heads, tokens, page) == old(heads, tokens, page)
                count += 1
    assert old(64, 2048, 128) == (128, 2)
    assert current(64, 2048, 128) == (32, 2)
    for heads in range(1, 129):
        for tokens in (128, 256, 384, 512, 2048):
            qb, pps = current(heads, tokens, 128)
            assert tokens % qb == 0 and qb >= 32
            assert pps == old(heads, tokens, 128)[1]
    print(f'PASS {count} unchanged low-head cases; H64 default32/2; legacy pps retained')
    print('LIMIT: H>96 still exceeds row budget at floor32; no generic VMEM guarantee')


if __name__ == '__main__':
    main()
