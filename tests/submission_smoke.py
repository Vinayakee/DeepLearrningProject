"""Read/write localhost verification with real archived inputs; no retraining/evaluation."""
import argparse
import csv
import io
import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def run(base_url):
    checks = []

    def request(path, body=None):
        data = json.dumps(body).encode() if body is not None else None
        try:
            response = urlopen(Request(base_url+path, data=data, headers={'Content-Type':'application/json'}), timeout=120)
        except HTTPError as error:
            response = error
        with response:
            payload = response.read()
            checks.append({'path':path,'http_status':response.status})
            return response.status, payload

    for path in ('/dashboard/','/dashboard/script.js','/dashboard/hab.html','/dashboard/hab.js','/dashboard/hab.css'):
        status,_ = request(path); assert status == 200,(path,status)
    status,raw = request('/api/covered-dates'); assert status == 200
    covered=json.loads(raw)['dates']; assert covered
    chosen=next((item for item in covered if item['input_end_date']=='2024-02-15'),covered[0])['input_end_date']
    status,_=request('/api/predict',{'input_end_date':'invalid'}); assert status==422
    status,_=request('/api/hab/predict',{'input_end_date':'invalid'}); assert status==422
    status,raw=request('/api/hab/status'); assert status==200
    hab=json.loads(raw); assert not hab['operational_warnings_available']
    if hab['status']=='not_configured':
        status,raw=request('/api/hab/predict',{'input_end_date':chosen,'historical_demo':True})
        missing=json.loads(raw);assert status==200 and missing['status']=='not_configured'
        assert 'daily_predictions' not in missing and missing['alerts']==[]
    # Real historical baseline inference, not a rerun of held-out evaluation.
    status,raw=request('/api/predict',{'input_end_date':chosen,'historical_demo':True})
    forecast=json.loads(raw); assert status==200 and forecast['status']=='success',forecast
    assert len(forecast['maps'])==len(forecast['daily_predictions'])==len(forecast['forecast_dates'])==5
    assert not forecast['operational_warnings_available']
    for path in forecast['maps']:
        status,raw=request(path);assert status==200 and raw.startswith(b'\x89PNG')
    route='/api/forecasts/'+forecast['forecast_id']
    status,raw=request(route+'/summary.csv?demo=true');assert status==200
    rows=list(csv.DictReader(io.StringIO(raw.decode())));assert len(rows)==5
    assert [row['forecast_date'] for row in rows]==forecast['forecast_dates']
    assert all(row['operational_warning_available']=='False' for row in rows)
    status,raw=request(route+'/observations.png?demo=true');assert status==200 and raw.startswith(b'\x89PNG')
    status,_=request(route+'/summary.csv');assert status==409
    status,_=request('/api/forecasts/invalid/summary.csv?demo=true');assert status==409
    status,raw=request('/api/prediction');assert status==200 and json.loads(raw)['status'] in ('stale','unavailable')
    status,raw=request('/api/prediction?demo=true');assert status==200 and json.loads(raw)['forecast_id']==forecast['forecast_id']
    status,raw=request('/api/evaluation');assert status==200 and json.loads(raw)['model_sha256']==forecast['model_sha256']
    status,raw=request('/api/predict',{'input_end_date':'2025-12-31','historical_demo':True})
    missing=json.loads(raw);assert status==200 and missing['status']=='insufficient_data'
    assert missing['maps']==[] and missing['daily_predictions']==[]
    return {'checks':checks,'eligible_dates_count':len(covered),'real_chlorophyll_forecast':forecast,
            'missing_data_status':missing['status'],'hab_status':hab,
            'new_hab_training':False,'new_test_evaluation':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url',default='http://127.0.0.1:8000')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    output=Path(args.output)
    if output.exists():
        parser.error('Choose a fresh report path; existing evidence cannot be overwritten.')
    report=run(args.url.rstrip('/'))
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as handle:json.dump(report,handle,indent=2,allow_nan=False);handle.write('\n')
    print(json.dumps({'status':'passed','checks':len(report['checks']),'eligible_dates':report['eligible_dates_count'],
                      'hab_status':report['hab_status']['status'],'output':str(output)}))


if __name__=='__main__':
    main()
