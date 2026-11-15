"""Plain native HTTP functional checks; no observers or scheduler patches."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import uuid
import json
from pathlib import Path
import urllib.request
import urllib.error


def run(base_url, output, tokenizer_path=None):
    spec=json.loads(Path(__file__).with_name('text_e2e_requests.json').read_text())
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    tokenizer=None
    if tokenizer_path is not None:
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(tokenizer_path,trust_remote_code=True)
        assert tokenizer.chat_template, 'model authored chat template required'
    checks=[]
    def execute(name,text,expected,rank=None):
        if tokenizer is not None:
            assert isinstance(text,str), 'chat checks require scalar input'
            text=tokenizer.apply_chat_template([dict(role='user',content=text)],
                tokenize=False,add_generation_prompt=True,enable_thinking=False)
        payload=dict(text=text,sampling_params=spec['sampling_params'],stream=False,rid=str(uuid.uuid4()))
        if rank is not None:payload['routed_dp_rank']=rank
        (output/f'{name}-request.json').write_text(json.dumps(payload,ensure_ascii=False)+'\n')
        request=urllib.request.Request(base_url.rstrip('/')+'/generate',json.dumps(payload).encode(),{'Content-Type':'application/json'})
        try:
            with urllib.request.urlopen(request,timeout=900) as response:
                assert response.status==200
                raw=response.read()
        except urllib.error.HTTPError as error:
            raw=error.read()
            (output/f'{name}-response.json').write_bytes(raw)
            (output/f'{name}-http-error.json').write_text(json.dumps(dict(
                status=error.code,reason=str(error.reason),body=raw.decode(errors='replace')),indent=2)+'\n')
            raise
        (output/f'{name}-response.json').write_bytes(raw)
        records=json.loads(raw);records=records if isinstance(records,list) else [records]
        assert len(records)==(len(text) if isinstance(text,list) else 1)
        for record in records:
            meta=record['meta_info'];assert meta['completion_tokens']>0 and meta['prompt_tokens']>0
            assert meta.get('finish_reason',{}).get('type')!='abort'
            if rank is not None:
                assert meta.get('dp_rank') == rank, f"routing receipt mismatch: requested {rank}, got {meta.get('dp_rank')}"
            assert isinstance(record.get('text'),str)
            checks.append(dict(case=name,routed_dp_rank=rank,returned_dp_rank=meta.get('dp_rank'),rid=payload['rid'],text=record['text'],expected=expected,matched=expected in record['text'].casefold(),
                               prompt_tokens=meta['prompt_tokens'],completion_tokens=meta['completion_tokens']))
    for case in spec['cases']:execute(case['id'],case['text'],case['expected_substring_casefold'])
    batch=spec['parallel_batch']
    with ThreadPoolExecutor(max_workers=16) as pool:
        pending=[pool.submit(execute,f'dp-{rank}',batch['text_template'].format(index=rank),
                             batch['expected_substring_casefold'],rank) for rank in range(batch['count'])]
        for future in pending:future.result()
    checks.sort(key=lambda item:item['case'])
    result=dict(status='TEXT_E2E_PASS' if all(c['matched'] for c in checks) else 'TEXT_E2E_SANITY_MISMATCH',checks=checks,
                scope='20 functional responses; no measured rankadmission/C16performance/accuracy benchmark proof')
    (output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    assert result['status']=='TEXT_E2E_PASS',result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--base-url',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--tokenizer-path',help='model authored chat template, nonthinking mode; omitted preserves raw completion')
    args=parser.parse_args()
    try:
        run(args.base_url,args.output,args.tokenizer_path)
    except Exception as error:
        import traceback
        output=Path(args.output)
        if output.is_dir():
            (output/'client-error.json').write_text(json.dumps(dict(
                status='CLIENT_FAILED',type=type(error).__name__,message=str(error),
                traceback=traceback.format_exc()),indent=2)+'\n')
        raise
