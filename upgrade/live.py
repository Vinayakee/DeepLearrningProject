"""Separate daily-input acquisition/staging/inference; frozen archive stays intact."""
import argparse
import importlib
import json
import os
from pathlib import Path
from datetime import datetime, timezone
import uuid
import numpy as np
import config
from experiments.common import write_json, read_json, load_model
from experiments.finalize import verify_final, FINAL_DIR
from preprocessing.pipeline import EnvironmentalPipeline, value_transform
from preprocessing.features import add_seasonal_features
from prediction.coverage import assess_coverage
from prediction.service import data_freshness, summarize_forecast, render_maps
from prediction.regional import warning_level
from preprocessing.data_utils import month_numbers, file_sha256
from upgrade.notifications import local_outbox

ROOT=config.ROOT/'outputs/live'


class DataTimingError(ValueError):
    pass


def write_status(path,status):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name('status_'+uuid.uuid4().hex+'.tmp')
    write_json(temporary,status)
    os.replace(temporary,path)


def prepare_inputs(directory,pipeline):
    directory=Path(directory)
    metadata=read_json(directory/'metadata.json')
    with np.load(directory/'inputs.npz',allow_pickle=False) as data:
        arrays={k:data[k] for k in ['values','masks','dates','latitude','longitude']}
    values,masks=arrays['values'],arrays['masks']
    if values.shape!=(5,120,84,7) or masks.shape!=values.shape or not np.isin(masks,[0,1]).all():
        raise ValueError('Require five 120×84 frames, seven original channels and binary masks.')
    if metadata.get('channel_order')!=list(config.CHANNEL_NAMES) or metadata.get('crs')!='EPSG:4326':
        raise ValueError('Channel order and WGS84 coordinates must match fitted preprocessing.')
    for name in ['latitude','longitude']:
        expected=getattr(pipeline,name)
        if arrays[name].shape!=expected.shape or not np.allclose(arrays[name],expected,atol=1e-7,rtol=0):
            raise ValueError('Live observations are not on the actual fitted grid.')
    sources=metadata.get('sources',{})
    if any(not sources.get(name,{}).get('product_ids') or not sources[name].get('quality_reference') or sources[name].get('native_qc_applied') is not True for name in config.CHANNEL_NAMES):
        raise ValueError('Every source requires product IDs, quality references and native quality control.')
    expected_units=['mg/m3','degC','m/s','m/s','mm/day','m/s','m/s']
    if metadata.get('units')!=expected_units:
        raise ValueError('Decode source scale/offset and convert units before ingestion.')
    dates=arrays['dates'].astype('datetime64[D]')
    if dates.shape!=(5,) or np.isnat(dates).any() or not np.all(np.diff(dates).astype(int)==1):
        raise ValueError('Require five consecutive actual UTC observation dates.')
    target_start=dates[-1]+np.timedelta64(1,'D')
    now=np.datetime64(datetime.now(timezone.utc).replace(tzinfo=None),'s')
    for source in sources.values():
        for key in ['observation_end_utc','available_at_utc']:
            value=source.get(key,'')
            if not isinstance(value,str) or not value.endswith('Z'):
                raise DataTimingError('Verified source acquisition/publication times are required.')
            time=np.datetime64(value[:-1],'s')
            if np.isnat(time) or time>=target_start or time>now:
                raise DataTimingError('Source acquisition/availability overlaps the first nominal target day.')
    if np.any((masks==1)&~np.isfinite(values)) or np.any((masks[...,0]==1)&(values[...,0]<=0)):
        raise ValueError('Invalid observations cannot carry a valid mask.')
    fields=[];flags=[]
    for c,name in enumerate(config.CHANNEL_NAMES):
        valid=(masks[...,c]==1)&pipeline.ocean
        stats=pipeline.stats['channels'][name]
        transformed=value_transform(np.where(valid,values[...,c],0),name)
        scaled=np.clip((transformed-stats['median'])/stats['scale'],-8,8)
        fields.append(np.where(valid,scaled,0).astype(np.float32));flags.append(valid.astype(np.float32))
    X=np.stack(fields+flags,axis=-1)
    if pipeline.stats.get('seasonal_features'):
        X=add_seasonal_features(X,dates)
    return X,dates,metadata


def observation_summary(directory,pipeline,date,calibration):
    with np.load(Path(directory)/'inputs.npz',allow_pickle=False) as arrays:
        values=arrays['values'][-1];masks=arrays['masks'][-1]
    month=int(month_numbers([date])[0])-1
    baseline=pipeline.climatology[month]
    valid=(masks[...,0]==1)&np.isfinite(values[...,0])&(values[...,0]>0)&pipeline.ocean&np.isfinite(baseline)&(baseline>0)
    p75=np.load(pipeline.label_dir/'hab_p75_monthly.npy')[month]
    p90=np.load(pipeline.label_dir/'hab_p90_monthly.npy')[month]
    valid &= np.isfinite(p75)&np.isfinite(p90)
    anomaly=np.divide(values[...,0]-baseline,baseline,out=np.full(baseline.shape,np.nan),where=valid)
    strong=int((valid&(anomaly>=p90)).sum());count=int(valid.sum())
    coverage=count/pipeline.ocean.sum()*100;fraction=strong/count*100 if count else 0.
    observed={'valid_pixels':count,'coverage_percent':float(coverage),'strong_fraction_percent':float(fraction),
              'warning_level':warning_level(fraction,calibration) if coverage>=5 else 'INSUFFICIENT OBSERVATIONS',
              'label_definition':'Observed chlorophyll proxy using unchanged training monthly thresholds; not field-confirmed HAB.'}
    environment=[]
    for c,(name,unit) in enumerate(zip(config.CHANNEL_NAMES,['mg/m³','°C','m/s','m/s','mm/day','m/s','m/s'])):
        available=(masks[...,c]==1)&np.isfinite(values[...,c])&pipeline.ocean
        if c==0:available &= values[...,c]>0
        environment.append({'name':name,'unit':unit,'observed_mean':float(values[...,c][available].mean()) if available.any() else None,
                            'coverage_percent':float(available.sum()/pipeline.ocean.sum()*100)})
    return observed,environment


def update(directory=None):
    started=datetime.now(timezone.utc).isoformat()
    status={'last_attempt_utc':started,'state':'unavailable','live':False,'last_successful_update_utc':None}
    path=ROOT/'status.json'
    if path.exists():
        try:
            previous=read_json(path)
            if not isinstance(previous,dict):raise ValueError('Invalid status schema')
            status['last_successful_update_utc']=previous.get('last_successful_update_utc')
            status['latest_successful_bundle']=previous.get('latest_successful_bundle')
        except (OSError,ValueError):
            pass
    try:
        verify_final()
        if directory is None:
            adapter=os.getenv('HAB_LIVE_ADAPTER','')
            if not adapter:
                status.update(state='not_configured',reason='No acquisition adapter configured. Static archive is not live.')
                write_status(path,status);return status
            module,function=adapter.split(':',1)
            # Adapter returns a staged real-input directory; credentials remain in its environment.
            directory=getattr(importlib.import_module(module),function)()
        pipeline=EnvironmentalPipeline()
        X,dates,metadata=prepare_inputs(directory,pipeline)
        freshness=data_freshness(str(dates[-1]))
        quality=assess_coverage(X,pipeline.ocean,dates)
        source_freshness={name:data_freshness(str(np.datetime64(source['observation_end_utc'][:-1],'s').astype('datetime64[D]'))) for name,source in metadata['sources'].items()}
        status.update(input_end_date=str(dates[-1]),freshness=freshness,source_freshness=source_freshness,data_quality=quality,sources=metadata['sources'])
        stale=freshness['state']!='fresh' or any(state['state']!='fresh' for state in source_freshness.values())
        if stale or quality['state']!='observed':
            status.update(state='stale_data' if stale else 'insufficient_data',reason='Freshness/coverage gate withheld live inference.')
            write_status(path,status);return status
        if dates[-1]+np.timedelta64(1,'D')<=np.datetime64(datetime.now(timezone.utc).date()):
            raise DataTimingError('The first nominal target day has already begun; no prospective daily lead time.')
        model=load_model(FINAL_DIR/'model.keras')
        scores=model(X[None],training=False).numpy()[0]
        dates_out=[str(dates[-1]+np.timedelta64(i,'D')) for i in range(1,6)]
        support=pipeline.ocean[None]&np.isfinite(pipeline.climatology[month_numbers(dates_out)-1])
        calibration=read_json(FINAL_DIR/'calibration.json')
        risk,days=summarize_forecast(scores,support,calibration)
        observed,environment=observation_summary(directory,pipeline,str(dates[-1]),calibration)
        directory_out=ROOT/'bundles'/uuid.uuid4().hex;directory_out.mkdir(parents=True,exist_ok=False)
        np.save(directory_out/'prediction.npy',scores);np.save(directory_out/'risk_classes.npy',risk)
        render_maps(scores,risk,pipeline.latitude,pipeline.longitude,dates_out,directory_out)
        validated=calibration.get('operational_warning_validated',False)
        bundle={'status':'success','input_dates':[str(d) for d in dates],'forecast_dates':dates_out,'daily_predictions':days,
                'freshness':freshness,'data_quality':quality,'sources':metadata['sources'],'historical_demo':False,
                'model_sha256':file_sha256(FINAL_DIR/'model.keras'),'decision_rule':calibration['decision_rule'],
                'operational_warning_validated':validated,'operational_warnings_available':validated,
                'maps':[f'/live/bundles/{directory_out.name}/day_{i}.png' for i in range(1,6)],
                'array_sha256':{name:file_sha256(directory_out/name) for name in ['prediction.npy','risk_classes.npy']},
                'warning_reason':'Regional warning validation failed; fresh maps remain research predictions.',
                'calibrated_probability_available':False,'regional_warning_status':'research_only' if not validated else 'validated'}
        bundle.update(generated_at_utc=datetime.now(timezone.utc).isoformat(),days=5,height=120,width=84,
                      warning_parameters={k:calibration[k] for k in ['regional_elevated_percent','regional_strong_percent','regional_method']},
                      threshold=calibration.get('strong_probability_threshold'),current_observed_risk=observed,environment=environment,
                      prediction_availability=quality['prediction_availability'],source_mode='live_staged',
                      model_description={'environmental_channels':7,'availability_channels':7,'seasonal_channels':2,'classes':3,'parameter_count':model.count_params()})
        bundle['source_inputs_sha256']=file_sha256(Path(directory)/'inputs.npz')
        bundle['source_metadata_sha256']=file_sha256(Path(directory)/'metadata.json')
        write_json(directory_out/'bundle.json',bundle)
        status.update(state='fresh_research_prediction',live=True,last_successful_update_utc=datetime.now(timezone.utc).isoformat(),latest_successful_bundle=directory_out.name,
                      notifications=local_outbox(bundle,ROOT/'notifications'))
    except DataTimingError:
        status.update(state='input_timing_invalid',reason='Verified source acquisition/publication must precede the first target day; live inference withheld.')
    except Exception as error:
        # Never log credentials, tokens or untrusted acquisition response bodies.
        status.update(state='update_failed',reason=f'Acquisition/validation/inference failed ({type(error).__name__}); previous successful data retained.')
    write_status(path,status)
    return status


def current_status():
    path=ROOT/'status.json'
    if not path.exists():
        return {'state':'not_configured','live':False,'last_successful_update_utc':None,'reason':'Static archive; no live acquisition has succeeded.'}
    try:
        status=read_json(path)
        if not isinstance(status,dict):raise ValueError('Invalid status schema')
        if 'state' not in status or 'live' not in status:raise ValueError('Incomplete status record')
        if status.get('input_end_date'):
            status['freshness']=data_freshness(status['input_end_date'])
            if status['freshness']['state']!='fresh':
                status['live']=False
                if status['state']=='fresh_research_prediction':status['state']='stale_data'
        if status.get('sources'):
            status['source_freshness']={name:data_freshness(str(np.datetime64(source['observation_end_utc'][:-1],'s').astype('datetime64[D]'))) for name,source in status['sources'].items()}
            if any(source['state']!='fresh' for source in status['source_freshness'].values()):
                status['live']=False
                if status['state']=='fresh_research_prediction':status['state']='stale_data'
    except (OSError,ValueError,KeyError,TypeError):
        return {'state':'update_failed','live':False,'last_successful_update_utc':None,'reason':'Update status is missing or invalid; no live prediction is assumed.'}
    return status


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--input-dir',type=Path)
    args=parser.parse_args();print(json.dumps(update(args.input_dir),indent=2))

if __name__=='__main__':main()
