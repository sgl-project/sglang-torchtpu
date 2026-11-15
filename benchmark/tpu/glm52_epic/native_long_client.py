"""Bounded native long-request baseline, not a matched JAX benchmark."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
import urllib.request
import uuid


def run(base_url, output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    def http(path,payload=None):
        data=None if payload is None else json.dumps(payload).encode()
        req=urllib.request.Request(base_url.rstrip('/')+path,data,{'Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=600) as response:
            assert response.status==200
            return response.read()
    info=json.loads(http('/server_info'))
    (output/'server-info.json').write_text(json.dumps(info,indent=2)+'\n')
    def execute(name,rank,seed):
        payload=dict(input_ids=[seed]+[1]*12287,routed_dp_rank=rank,rid=str(uuid.uuid4()),
                     sampling_params=dict(temperature=0,max_new_tokens=16,ignore_eos=True),stream=False)
        (output/f'{name}-request.json').write_text(json.dumps(payload)+'\n')
        start=time.perf_counter();raw=http('/generate',payload);elapsed=time.perf_counter()-start
        (output/f'{name}-response.json').write_bytes(raw)
        response=json.loads(raw);meta=response['meta_info'];ids=response['output_ids']
        assert meta['id']==payload['rid'] and meta['dp_rank']==rank
        assert meta['prompt_tokens']==12288 and meta['completion_tokens']==16
        assert meta.get('finish_reason',{}).get('type')!='abort'
        assert len(ids)==16 and all(isinstance(i,int) for i in ids)
        return dict(name=name,dp_rank=rank,wall_seconds=elapsed,meta_info=meta,output_ids=ids)
    warm=execute('warmup',0,1)
    sequential=[execute(f'sequential-{i}',0,1) for i in range(3)]
    start=time.perf_counter()
    with ThreadPoolExecutor(max_workers=16) as pool:
        pending=[pool.submit(execute,f'dp-{rank}',rank,rank+2) for rank in range(16)]
        concurrent=[future.result() for future in pending]
    cohort_seconds=time.perf_counter()-start
    result=dict(status='NATIVE_LONG_HTTP_PASS',warmup=warm,sequential=sequential,
                concurrent=concurrent,cohort_wall_seconds=cohort_seconds,
                input_tokens=16*12288,output_tokens=16*16,
                scope='16 routed long requests; cached_tokens reported, not forced cold; hostwall only; no simultaneous C16/admission/kernel-time/JAX-match/precision claim')
    (output/'result.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--base-url',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();run(args.base_url,args.output)
