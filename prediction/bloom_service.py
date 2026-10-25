"""Actual satellite bloom-event inference, maps and exports, separate from HABs."""
from functools import lru_cache
from datetime import datetime,timezone
import csv
import io
import json
import os
from pathlib import Path
import uuid
import numpy as np
import config
from preprocessing.bloom_events import write_json,TITLE
from preprocessing.data_utils import file_sha256,month_numbers
from prediction.coverage import assess_coverage
from prediction.service import data_freshness
from training.train_bloom_events import DEFAULT_RUN

FORECAST_ROOT=config.ROOT/'outputs/bloom_events/forecasts'


def run_dir(identifier=None):
    if identifier is not None:
        if identifier not in ('run_01','run_02','run_03'):raise ValueError('Unknown satellite-event run.')
        return config.ROOT/'outputs/bloom_events'/identifier
    path=Path(os.getenv('BLOOM_EVENT_RUN_DIR',str(DEFAULT_RUN))).expanduser()
    return path if path.is_absolute() else config.ROOT/path


def status(identifier=None):
    directory=run_dir(identifier)
    base={'title':TITLE,'operational_warnings_available':False,'probability_calibrated':False,
          'run_id':directory.name,'label_semantics':'Chlorophyll-derived satellite event, not confirmed harmful species or toxicity.'}
    try:
        record=json.loads((directory/'training.json').read_text())
        if record['state']!='trained':return {**base,'status':'training','message':'The separate satellite bloom-event model is training.'}
        if not all((directory/name).exists() for name in ('freeze.json','evaluation.json','decision.json')):
            return {**base,'status':'assessment_pending','message':'Training finished; frozen validation selection and test reporting are pending.'}
        report=json.loads((directory/'evaluation.json').read_text())
        relaxed=record.get('label_schema')=='satellite-bloom-events-v2'
        inputs=bool(record.get('input_variant'))
        return {**base,'status':'ready_research_only','message':('Controlled input experiment available; compare its saved matched-persistence results. No operational or verified harmful-occurrence claim.' if inputs else 'Relaxed satellite-event research model available. Satellite-label metrics do not establish verified-HAB or operational skill.' if relaxed else 'Strict research model available. No positive validation/test event targets; detection skill cannot be established.'),
                'label_definition':record.get('label_definition','Strict P90 three-day patch'),
                'probability_calibrated':report['decision'].get('probability_calibrated',False),
                'epochs_completed':record['epochs_completed'],'best_epoch':record['best_epoch'],
                'test_event_components':report['test_event_components'],'event_detection_assessable':report['event_detection_assessable'],
                'input_days':5,'output_days':5,'latest_input_date':(config.PROCESSED_DIR/'dates_2021_2025.txt').read_text().splitlines()[-1]}
    except (OSError,ValueError,KeyError):
        return {**base,'status':'unavailable','message':'Complete the separate satellite bloom-event training and frozen evaluation workflow.'}


def ready_run(identifier=None):
    from prediction.bloom_events import verify
    record,pipeline,decision=verify(run_dir(identifier))
    if record.get('input_variant'):
        from preprocessing.bloom_inputs_v3 import InputPipeline,ROOT as feature_root
        if (file_sha256(feature_root/'features.json')!=record['input_features_manifest_sha256']
                or file_sha256(feature_root/'features.npy')!=record['feature_sha256']):
            raise ValueError('Frozen input features changed.')
        pipeline=InputPipeline(record['dataset_dir'],record['input_variant'])
    report=json.loads((run_dir(identifier)/'evaluation.json').read_text())
    if report['model_sha256']!=record['model_sha256'] or report['decision']!=decision:
        raise ValueError('Satellite-event evaluation differs from the frozen model/decision.')
    return record,pipeline,decision,report


@lru_cache(maxsize=1)
def cached_model(directory,digest):
    from prediction.bloom_events import load_model
    return load_model(directory)


def render_maps(scores,classes,support,pipeline,dates,directory,score_label='Uncalibrated model score'):
    from matplotlib.figure import Figure
    from matplotlib.colors import ListedColormap,BoundaryNorm
    cmap=ListedColormap(['#c9e6df','#e6a357']).with_extremes(bad='#e1e5e9')
    for i,date in enumerate(dates):
        fig=Figure(figsize=(10,5),layout='constrained');axes=fig.subplots(1,2)
        left=axes[0].pcolormesh(pipeline.longitude,pipeline.latitude,np.ma.masked_where(~support[i],classes[i]),
                              cmap=cmap,norm=BoundaryNorm([-.5,.5,1.5],2),shading='nearest')
        right=axes[1].pcolormesh(pipeline.longitude,pipeline.latitude,np.ma.masked_where(~support[i],scores[i]),
                               vmin=0,vmax=1,cmap='viridis',shading='nearest')
        fig.colorbar(left,ax=axes[0],ticks=[0,1],shrink=.75).ax.set_yticklabels(['No event forecast','Satellite event'])
        fig.colorbar(right,ax=axes[1],shrink=.75,label=score_label)
        axes[0].set_title('Satellite event class');axes[1].set_title(score_label)
        for ax in axes:ax.set(xlabel='Longitude (°E)',ylabel='Latitude (°N)')
        fig.suptitle(f'{TITLE} · Day +{i+1} · {date}\nResearch: chlorophyll-derived; not confirmed HAB. Grey = unsupported.')
        fig.savefig(directory/f'day_{i+1}.png',dpi=120)


def generate(input_end_date=None,historical_demo=False,identifier=None):
    record,p,decision,report=ready_run(identifier)
    directory_run=run_dir(identifier)
    calibrated=decision.get('probability_calibrated',False)
    relaxed=record.get('label_schema')=='satellite-bloom-events-v2'
    inputs=bool(record.get('input_variant'))
    end=str(input_end_date or p.dates[-1]);freshness=data_freshness(end)
    if freshness['state']=='future' or (freshness['state']!='fresh' and not historical_demo):
        raise ValueError('Stale inputs require explicit historical demonstration; future inputs are rejected.')
    found=np.flatnonzero(p.dates==np.datetime64(end,'D'))
    if not len(found) or found[0]<4:raise ValueError('Choose an archived date with five preceding input days.')
    indices=np.arange(found[0]-4,found[0]+1);X=p.transform(indices)
    dates=[str(day) for day in p.dates[indices]]
    targets=[str(np.datetime64(end)+np.timedelta64(i,'D')) for i in range(1,6)]
    quality=assess_coverage(X[...,:16],p.ocean,dates)
    base={'title':TITLE,'status':'insufficient_data','historical_demo':historical_demo,
          'input_dates':dates,'forecast_dates':targets,'freshness':freshness,'data_quality':quality,
          'maps':[],'daily_predictions':[],'alerts':[],'operational_warnings_available':False,
          'model_sha256':record['model_sha256'],'threshold':decision['threshold'],'probability_calibrated':calibrated,
          'run_id':directory_run.name,'label_definition':record.get('label_definition','Strict P90 three-day patch'),
          'score_kind':'validation_calibrated' if calibrated else 'raw',
          'validation_limit':('Controlled input experiment; past-only state/history channels. 2024 decision fit is development evidence; 2025 previously inspected. Compare matched persistence in the saved evaluation.' if inputs else 'Relaxed single-day satellite patches; 2024 calibration is development evidence and 2025 was previously inspected. No verified-HAB or operational approval.' if relaxed else 'Strict definition: no positive observed event targets in 2024/2025; detection performance is not established.'),
          'label_semantics':report['limitations'][0]}
    if quality['state']!='observed':return base
    raw=cached_model(str(directory_run),record['model_sha256'])(X[None],training=False).numpy()[0]
    if raw.shape!=(5,*p.shape[1:]) or not np.isfinite(raw).all() or np.any((raw<0)|(raw>1)):
        raise RuntimeError('Invalid satellite-event model output.')
    from prediction.bloom_events_v2 import apply_scores
    scores=apply_scores(raw,decision)
    support=p.ocean[None]&np.isfinite(p.climatology[month_numbers(targets)-1])
    classes=(scores>=decision['threshold']).astype(np.uint8);classes[~support]=255
    identifier=uuid.uuid4().hex;directory=FORECAST_ROOT/identifier;directory.mkdir(parents=True)
    np.save(directory/'scores.npy',scores);np.save(directory/'classes.npy',classes)
    np.save(directory/'support.npy',support)
    render_maps(scores,classes,support,p,targets,directory,'2024 validation-calibrated score' if calibrated else 'Uncalibrated model score')
    rows=[]
    for i,date in enumerate(targets):
        values=scores[i][support[i]];n=int(support[i].sum());positive=int((classes[i]==1).sum())
        rows.append({'day':i+1,'date':date,'event_pixels':positive,'supported_pixels':n,
                     'event_fraction_percent':positive/max(1,n)*100,
                     'max_score':float(values.max()) if n else None,'mean_score':float(values.mean()) if n else None,
                     'max_raw_score':float(raw[i][support[i]].max()) if n else None})
    bundle={**base,'status':'success','forecast_id':identifier,'generated_at_utc':datetime.now(timezone.utc).isoformat(),
            'daily_predictions':rows,'maps':[f'/bloom-forecasts/{identifier}/day_{i}.png' for i in range(1,6)],
            'decision_sha256':file_sha256(directory_run/'decision.json'),
            'array_sha256':{name:file_sha256(directory/name) for name in ('scores.npy','classes.npy','support.npy')}}
    write_json(directory/'bundle.json',bundle)
    return bundle


def export_csv(identifier,demo=False):
    if len(identifier)!=32 or any(c not in '0123456789abcdef' for c in identifier):raise ValueError('Invalid forecast ID.')
    directory=FORECAST_ROOT/identifier
    bundle=json.loads((directory/'bundle.json').read_text())
    identifier_run=bundle.get('run_id','run_01')
    record,_,_,_=ready_run(identifier_run)
    if bundle['model_sha256']!=record['model_sha256'] or bundle['decision_sha256']!=file_sha256(run_dir(identifier_run)/'decision.json'):
        raise ValueError('Forecast differs from the frozen satellite-event run.')
    if (bundle['historical_demo'] or data_freshness(bundle['input_dates'][-1])['state']!='fresh') and not demo:
        raise ValueError('Historical exports require demonstration mode.')
    for name in ('scores.npy','classes.npy','support.npy'):
        if bundle['array_sha256'][name]!=file_sha256(directory/name):raise ValueError('Forecast arrays changed.')
    out=io.StringIO(newline='');writer=csv.writer(out)
    writer.writerow(['forecast_date','day','input_end_date','event_pixels','supported_pixels','event_fraction_percent',
                     'max_model_score','mean_model_score','chlorophyll_input_coverage_percent','historical_demo','operational_warning','label_semantics','run_id','score_kind'])
    for row in bundle['daily_predictions']:
        writer.writerow([row['date'],row['day'],bundle['input_dates'][-1],row['event_pixels'],row['supported_pixels'],
                         row['event_fraction_percent'],row['max_score'],row['mean_score'],
                         bundle['data_quality']['chlorophyll_mean_input_coverage_percent'],bundle['historical_demo'],False,
                         'Chlorophyll-derived satellite bloom-event forecast; not confirmed HAB',identifier_run,bundle.get('score_kind','raw')])
    return out.getvalue()
