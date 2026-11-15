"""Plain native HTTP functional checks; no observers or scheduler patches."""
import argparse
import json
from pathlib import Path
import urllib.request


def run(base_url, output):
    spec=json.loads(Path(__file__).with_name('text_e2e_requests.json').read_text())
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    checks=[]
    def execute(name,text,expected):
        payload=dict(text=text,sampling_params=spec['sampling_params'],stream=False)
        (output/f'{name}-request.json').write_text(json.dumps(payload,ensure_ascii=False)+'\n')
        request=urllib.request.Request(base_url.rstrip('/')+'/generate',json.dumps(payload).encode(),{'Content-Type':'application/json'})
        with urllib.request.urlopen(request,timeout=900) as response:
            assert response.status==200
            raw=response.read()
        (output/f'{name}-response.json').write_bytes(raw)
        records=json.loads(raw);records=records if isinstance(records,list) else [records]
        assert len(records)==(len(text) if isinstance(text,list) else 1)
        for record in records:
            meta=record['meta_info'];assert meta['completion_tokens']>0 and meta['prompt_tokens']>0
            assert meta.get('finish_reason',{}).get('type')!='abort'
            assert isinstance(record.get('text'),str)
            checks.append(dict(case=name,expected=expected,matched=expected in record['text'].casefold(),
                               prompt_tokens=meta['prompt_tokens'],completion_tokens=meta['completion_tokens']))
    for case in spec['cases']:execute(case['id'],case['text'],case['expected_substring_casefold'])
    batch=spec['parallel_batch'];execute('batch16',[batch['text_template'].format(index=i) for i in range(batch['count'])],batch['expected_substring_casefold'])
    result=dict(status='TEXT_E2E_PASS' if all(c['matched'] for c in checks) else 'TEXT_E2E_SANITY_MISMATCH',checks=checks,
                scope='20 functional responses; no measured rankadmission/C16performance/accuracy benchmark proof')
    (output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    assert result['status']=='TEXT_E2E_PASS',result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--base-url',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();run(args.base_url,args.output)
