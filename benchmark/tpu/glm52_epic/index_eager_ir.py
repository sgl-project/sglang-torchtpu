"""Bounded offline HLO candidate index; lexical hits are NOT causal attribution.

Run on Falcon artifact-mounted eager-ir directory from an analyzer. No TPU,
backend imports, payload values, file deletion or compilation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

PATTERN=re.compile(r'\bu64\[|custom-call\(|op_name=|source_file=')


def index(root, max_bytes=4*1024**3, max_files=12500, max_hits=20000):
    root=Path(root).resolve()
    total=0;count=0;hits=[];modules=[];truncated=False
    for path in root.rglob('*'):
        if path.is_symlink() or not path.is_file():continue
        count+=1;size=path.stat().st_size;total+=size
        if count>max_files or total>max_bytes:
            raise ValueError('offline IR input budget exceeded')
        if path.suffix not in ('.txt','.hlo','.mlir'):continue
        # Keep per-file allocation bounded, stream hash/lines. Refuse giantlines.
        digest=hashlib.sha256();local=[]
        with path.open('rb') as stream:
            lineno=0
            while raw := stream.readline(1024*1024+1):
                lineno+=1
                if len(raw)>1024*1024:
                    raise ValueError('IR line exceeds1MiB bound')
                digest.update(raw)
                text=raw.decode('utf-8',errors='replace')
                if PATTERN.search(text):
                    if len(hits)+len(local)<max_hits:
                        local.append(dict(file=str(path.relative_to(root)),line=lineno,
                                          text=text.rstrip()[:4096],u64=bool(re.search(r'\bu64\[',text))))
                    else:truncated=True
        hits.extend(local);modules.append(dict(file=str(path.relative_to(root)),size_bytes=size,sha256=digest.hexdigest()))
    return dict(status='LEXICAL_CANDIDATES_ONLY',scope='No originatingIR/rank/fatal correlation inferred from strings',
                files_seen=count,bytes_seen=total,text_modules=modules,hits=hits,hits_truncated=truncated)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('directory');parser.add_argument('--output',required=True)
    args=parser.parse_args();Path(args.output).write_text(json.dumps(index(args.directory),indent=2)+'\n')


if __name__=='__main__':main()
