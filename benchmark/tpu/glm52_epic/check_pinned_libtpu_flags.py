"""Inspect installed pinned libtpu bytes, without loading TPU runtime."""
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path


def main():
    assert metadata.version('libtpu') == '0.0.50'
    dist=metadata.distribution('libtpu')
    matches=[Path(dist.locate_file(p)) for p in dist.files if str(p).endswith('/libtpu.so')]
    assert len(matches)==1,matches
    data=matches[0].read_bytes()
    flags=['xla_dump_to','xla_dump_hlo_as_text','xla_dump_hlo_as_proto',
           'xla_jf_dump_to','xla_jf_dump_llo_text']
    found={flag:flag.encode() in data for flag in flags}
    assert all(found[k] for k in flags[:3]),found
    print('PINNED_LIBTPU_FLAG_BYTES_PASS '+json.dumps(dict(version=metadata.version('libtpu'),
        sha256=hashlib.sha256(data).hexdigest(),size_bytes=len(data),flags=found,
        scope='exactinstalledbinary flagstrings; not parseracceptance/IRemission or TPUexecution')),flush=True)


if __name__=='__main__':
    main()
