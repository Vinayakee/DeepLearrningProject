"""Explicit local notification outbox; external delivery stays disabled."""
from pathlib import Path
from datetime import datetime, timezone
import json
import hashlib


def eligible(bundle):
    return (bundle.get('status')=='success' and bundle.get('operational_warning_validated') is True
            and bundle.get('operational_warnings_available') is True
            and bundle.get('historical_demo') is False
            and bundle.get('freshness',{}).get('state')=='fresh'
            and bundle.get('data_quality',{}).get('state')=='observed')


def local_outbox(bundle, directory, demo=False):
    directory=Path(directory); entries=[]
    if not demo and not eligible(bundle):
        return {'state':'blocked','reason':'Requires validated regional skill, fresh sufficient inputs and a non-demo warning.','entries':[]}
    if demo and (bundle.get('status')!='success' or bundle.get('data_quality',{}).get('state')!='observed'):
        return {'state':'blocked','reason':'Demo notifications also require available coverage-supported predictions.','entries':[]}
    for row in bundle.get('daily_predictions',[]):
        if row['warning_level'] not in ('ELEVATED RISK','STRONG BLOOM RISK'):
            continue
        day=row['day']; date=bundle['forecast_dates'][day-1]
        # Idempotent across repeated processing of the same model/date/level/demo.
        key=hashlib.sha256(f"{bundle['model_sha256']}:{date}:{row['warning_level']}:{demo}".encode()).hexdigest()
        entry={'id':key,'date':date,'level':row['warning_level'],'delivery':'local_demo' if demo else 'local_outbox',
               'demo':bool(demo),'external_sent':False,'created_at_utc':datetime.now(timezone.utc).isoformat()}
        directory.mkdir(parents=True,exist_ok=True)
        try:
            with (directory/(key+'.json')).open('x') as handle:
                json.dump(entry,handle,indent=2)
            entries.append(entry)
        except FileExistsError:
            pass
    return {'state':'local_only','entries':entries,'external_sent':False}
