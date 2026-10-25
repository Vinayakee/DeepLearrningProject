"""Real localhost satellite bloom-event inference checks; no repeat test evaluation."""
import argparse
import csv
import io
import json
from pathlib import Path
from urllib.request import Request,urlopen
from urllib.error import HTTPError


def run(url,run_id=None):
    checks=[]
    query='?run_id='+run_id if run_id else ''
    def body(date,demo=None):
        result={'input_end_date':date}
        if demo is not None:result['historical_demo']=demo
        if run_id:result['run_id']=run_id
        return result
    def request(path,body=None):
        try:
            response=urlopen(Request(url+path,data=json.dumps(body).encode() if body is not None else None,
                                    headers={'Content-Type':'application/json'}),timeout=120)
        except HTTPError as error:response=error
        with response:
            raw=response.read();checks.append({'path':path,'http_status':response.status});return response.status,raw
    for path in ('/dashboard/','/dashboard/hab.html','/dashboard/bloom.html','/dashboard/bloom.js'):
        status,_=request(path);assert status==200
    status,raw=request('/api/bloom/status'+query);state=json.loads(raw)
    assert status==200 and state['status']=='ready_research_only'
    status,raw=request('/api/hab/status');assert status==200 and json.loads(raw)['status']=='not_configured'
    status,_=request('/api/bloom/predict',body('invalid'));assert status==422
    status,_=request('/api/bloom/predict',body('2024-02-15'));assert status==422
    status,raw=request('/api/bloom/predict',body('2024-02-15',True))
    forecast=json.loads(raw);assert status==200 and forecast['status']=='success',forecast
    assert len(forecast['maps'])==len(forecast['daily_predictions'])==len(forecast['forecast_dates'])==5
    assert not forecast['operational_warnings_available']
    assert forecast['probability_calibrated']==state['probability_calibrated']
    if run_id:assert forecast['run_id']==run_id
    for path in forecast['maps']:
        status,raw=request(path);assert status==200 and raw.startswith(b'\x89PNG')
    route='/api/bloom/forecasts/'+forecast['forecast_id']+'/summary.csv'
    status,raw=request(route+'?demo=true');assert status==200
    rows=list(csv.DictReader(io.StringIO(raw.decode())));assert len(rows)==5
    assert [row['forecast_date'] for row in rows]==forecast['forecast_dates']
    assert all(row['operational_warning']=='False' for row in rows)
    status,_=request(route);assert status==409
    status,_=request('/api/bloom/forecasts/invalid/summary.csv?demo=true');assert status==409
    status,raw=request('/api/bloom/predict',body('2025-12-31',True))
    missing=json.loads(raw);assert status==200 and missing['status']=='insufficient_data'
    assert not missing['maps'] and not missing['daily_predictions']
    status,raw=request('/api/bloom/evaluation'+query);evaluation=json.loads(raw)
    assert status==200 and evaluation['model_sha256']==forecast['model_sha256']
    return {'checks':checks,'forecast':forecast,'model_status':state,'insufficient_data':missing['status'],
            'evaluation_repeated':False,'verified_hab_state':'not_configured'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url',default='http://127.0.0.1:8014')
    parser.add_argument('--output',required=True)
    parser.add_argument('--run-id',choices=('run_01','run_02','run_03'))
    args=parser.parse_args()
    if Path(args.output).exists():parser.error('Choose a fresh evidence path.')
    report=run(args.url.rstrip('/'),args.run_id)
    with Path(args.output).open('x') as handle:json.dump(report,handle,indent=2,allow_nan=False);handle.write('\n')
    print(json.dumps({'status':'passed','http_assertions':len(report['checks']),'output':args.output}))


if __name__=='__main__':main()
