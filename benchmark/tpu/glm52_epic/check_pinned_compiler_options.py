"""Inspect exact pinned wheel source without importing or executing TPU code."""
import hashlib
import json
from pathlib import Path
import zipfile


def main():
    path=Path('/wheels/torch-tpu/torch_tpu-0.2.1.dev20261028201649-cp312-cp312-manylinux_2_31_x86_64.whl')
    assert hashlib.sha256(path.read_bytes()).hexdigest()=='6e12816e92f9a5d9649358541f332f06583ccdfee5b6a74434dbf2141757fd82'
    with zipfile.ZipFile(path) as wheel:
        candidates=[name for name in wheel.namelist() if name.endswith('compiler_options.py') or name.endswith('compiler_options/__init__.py')]
        assert candidates,candidates
        for name in candidates:
            raw=wheel.read(name);text=raw.decode()
            print('PINNED_COMPILER_OPTIONS_SOURCE '+json.dumps(dict(path=name,sha256=hashlib.sha256(raw).hexdigest(),source=text)),flush=True)
            assert 'custom_compiler_options' in text
            # Execute ONLY exact packaged function with nativepush/pop boundary
            # supplied. No torch_tpu import or claim about native implementation.
            import ast
            import contextlib
            import types
            tree=ast.parse(text)
            function=next(node for node in tree.body if isinstance(node,ast.FunctionDef)
                          and node.name=='custom_compiler_options')
            function.decorator_list=[]
            events=[]
            namespace=dict(compiler=types.SimpleNamespace(
                push_compiler_options=lambda options:events.append(('push',dict(options))),
                pop_compiler_options=lambda:events.append(('pop',None))))
            exec(compile(ast.Module(body=[function],type_ignores=[]),name,'exec'),namespace)
            context=contextlib.contextmanager(namespace['custom_compiler_options'])
            try:
                with context({'xla_dump_hlo_as_text':'true'}):
                    events.append(('materialize',None))
                    raise RuntimeError('intentional')
            except RuntimeError as error:
                assert str(error)=='intentional'
            assert [event[0] for event in events]==['push','materialize','pop']
            print('PINNED_COMPILER_CONTEXT_FINALLY_CPU_PASS nativeboundaryfakeonly',flush=True)
    print('PINNED_COMPILER_OPTIONS_SOURCE_PASS; sourcecontract only no runtime/thread/materialization proof',flush=True)


if __name__=='__main__':main()
