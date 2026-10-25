"""Controlled V0–V3 input ablation; validation-only choice, one selected test pass."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import time
import numpy as np
import config
from preprocessing.bloom_events import write_json
from preprocessing.data_utils import file_sha256
from preprocessing.bloom_inputs_v3 import InputPipeline,GROUPS,ROOT as DATA,BASE,prepare
from prediction.bloom_events import counts,metrics,persistence
from prediction.bloom_events_v2 import fit_decision,validation_parts,apply_scores

REPORT=config.ROOT/'docs/results/bloom_events_v3'
RUN=config.ROOT/'outputs/bloom_events/run_03'
DEADLINE=datetime.fromisoformat('2026-10-24T23:33:48+00:00').timestamp()


def check_time():
    if time.time()>DEADLINE:raise TimeoutError('Three-hour limit reached; preserve run_02 as submission result.')


def build_model(channels):
    import tensorflow as tf
    from models.convlstm import build_convlstm
    base=build_convlstm(channels=channels)
    trunk=base.get_layer('forecast_logits').input
    scores=tf.keras.layers.Conv2D(5,1,activation='sigmoid',name='satellite_event_scores')(trunk)
    outputs=tf.keras.layers.Permute((3,1,2),name='five_daily_event_maps')(scores)
    return tf.keras.Model(base.input,outputs,name='Satellite_Bloom_Event_ConvLSTM')


def train_variant(variant):
    import tensorflow as tf
    import training.train_bloom_events as original
    from unittest.mock import patch
    check_time();destination=RUN/'variants'/variant
    channels=16+len(GROUPS[variant])
    OriginalPilot=original.RuntimePilot
    class BudgetPilot(OriginalPilot):
        def on_epoch_end(self,epoch,logs=None):
            super().on_epoch_end(epoch,logs)
            if epoch==2:
                predicted=float(np.mean(self.elapsed)*15)
                report={'variant':variant,'three_epoch_seconds':self.elapsed,'max_epochs':15,
                        'estimated_variant_seconds':predicted,'remaining_budget_seconds':DEADLINE-time.time(),
                        'decision':'Keep identical 15-epoch ceiling when four runs plus evaluation fit remaining three-hour budget.'}
                if variant=='V1' and predicted*4+600>DEADLINE-time.time():
                    self.model.stop_training=True
                    raise TimeoutError('Pilot cannot fit four comparable 15-epoch runs within time box. Stop without test.')
                write_json(destination/'experiment_budget.json',report)
            check_time()
    with patch.object(original,'BloomPipeline',lambda dataset:InputPipeline(dataset,variant)),patch.object(original,'build_bloom_event_model',lambda:build_model(channels)),patch.object(original,'RuntimePilot',BudgetPilot):
        record=original.train(BASE,destination,15)
    record.update(input_shape=[5,120,84,channels],input_variant=variant,
                  added_channels=GROUPS[variant],input_features_dir=str(DATA),
                  input_features_manifest_sha256=file_sha256(DATA/'features.json'),
                  feature_sha256=json.loads((DATA/'features.json').read_text())['features_sha256'],
                  experimental_change='Inputs only; original filter counts, heads, loss, weights, optimizer, seed, batch size and stopping/checkpoint recipe retained.')
    for source in ('preprocessing/bloom_inputs_v3.py','training/train_bloom_inputs_v3.py','prediction/bloom_events_v2.py','tests/test_bloom_inputs_v3.py'):
        target=destination/'source'/source;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(config.ROOT/source,target)
        record['source_sha256'][source]=file_sha256(target)
    write_json(destination/'training.json',record)
    return record


def collect(run,split):
    import tensorflow as tf
    from training.train_bloom_events import BloomGenerator
    import models.bloom_events
    check_time();run=Path(run);record=json.loads((run/'training.json').read_text())
    p=InputPipeline(BASE,record['input_variant'])
    output=run/('scores_'+split);output.mkdir(exist_ok=False)
    generator=BloomGenerator(split,2,False,p)
    scores=np.lib.format.open_memmap(output/'scores.npy',mode='w+',dtype=np.float32,shape=(generator.num_samples,5,*p.shape[1:]))
    model=tf.keras.models.load_model(run/'model.keras',compile=False)
    for i in range(len(generator)):
        check_time();X,y=generator[i];pred=model(X,training=False).numpy()
        if pred.shape!=y.shape or not np.isfinite(pred).all():raise ValueError('Invalid predictions.')
        scores[i*2:i*2+len(X)]=pred
    scores.flush();generator.indices.to_csv(output/'index.csv',index=False)
    write_json(output/'metadata.json',{'split':split,'model_sha256':file_sha256(run/'model.keras'),'scores_sha256':file_sha256(output/'scores.npy')})
    return scores,generator.indices,p


def block_counts(y,pred,valid):
    """10x10 fixed blocks; >=10 valid target pixels; any valid event is positive."""
    h,w=y.shape
    padded=(int(np.ceil(h/10))*10,int(np.ceil(w/10))*10)
    def reshape(a):return np.pad(a,((0,padded[0]-h),(0,padded[1]-w))).reshape(padded[0]//10,10,padded[1]//10,10)
    v=reshape(valid);enough=v.sum(axis=(1,3))>=10
    truth=reshape((y==1)&valid).any(axis=(1,3));prediction=reshape(pred&valid).any(axis=(1,3))
    return counts(truth,prediction,enough)


def assess(scores,index,p,decision):
    totals={name:np.zeros((5,4),np.int64) for name in ('pixel','matched_model','matched_persistence','block','block_matched_model','block_matched_persistence')}
    brier=np.zeros(5);abstained=np.zeros(5,np.int64)
    for i,row in enumerate(index.itertuples()):
        y=np.asarray(p.labels[row.target_start_idx:row.target_end_idx+1]);prob=apply_scores(scores[i],decision)
        prediction=prob>=decision['threshold'];baseline=persistence(p,row.input_start_idx)
        for h in range(5):
            valid=y[h]!=255;common=valid&(baseline!=255);base=baseline==1
            for name,pred,mask in [('pixel',prediction[h],valid),('matched_model',prediction[h],common),('matched_persistence',base,common)]:
                totals[name][h]+=counts(y[h],pred,mask)
            for name,pred,mask in [('block',prediction[h],valid),('block_matched_model',prediction[h],common),('block_matched_persistence',base,common)]:
                totals[name][h]+=block_counts(y[h],pred,mask)
            brier[h]+=np.sum((prob[h][valid]-y[h][valid])**2,dtype=np.float64)
            abstained[h]+=int((valid&~common).sum())
    result={name:[{'horizon':h+1,**metrics(array[h],brier[h] if name=='pixel' else None)} for h in range(5)] for name,array in totals.items()}
    result['persistence_abstained_valid_targets']=abstained.tolist()
    result['mean_pixel_f1']=float(np.mean([x['f1'] for x in result['pixel']]))
    return result


def validate_variant(variant):
    run=RUN/'variants'/variant
    scores,index,p=collect(run,'validation')
    parts=validation_parts(scores,index,p)
    decision=fit_decision(parts['fit']['truth'],parts['fit']['scores'],parts['assessment']['truth'],parts['assessment']['scores'],len(parts['fit']['positive_days']),len(parts['assessment']['positive_days']))
    record=json.loads((run/'training.json').read_text())
    decision.update(model_sha256=record['model_sha256'],dataset_manifest_sha256=record['dataset_manifest_sha256'],
                    definition=p.event_metadata['chosen_definition'],test_disclosure=p.event_metadata['test_disclosure'])
    write_json(run/'decision.json',decision)
    result=assess(scores,index,p,decision);result.update(variant=variant,decision=decision,training=record)
    write_json(run/'validation.json',result)
    return result


def write_evaluation(report):
    rows=['# Satellite bloom-event forecast — input-only run_03','',
          '**Satellite-derived proxy targets; 2025 previously inspected. Rolling pixel forecasts are correlated, not independent biological events.**','',
          f"Chosen on 2024 mean observed-target F1: {report['chosen_variant']}. No 2025 scores used for variant or decision selection.",'',
          '| Day | Valid pixel N | P | R | F1 | IoU/CSI | Shared pixel N | Matched model F1 | Persistence F1 | Matched model IoU | Persistence IoU |',
          '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for full,m,b in zip(report['pixel'],report['matched_model'],report['matched_persistence']):
        rows.append(f"| +{full['horizon']} | {full['valid_target_count']} | {full['precision']:.4f} | {full['recall']:.4f} | {full['f1']:.4f} | {full['iou']:.4f} | {m['valid_target_count']} | {m['f1']:.4f} | {b['f1']:.4f} | {m['iou']:.4f} | {b['iou']:.4f} |")
    rows+=['','## 10×10 blocks','', 'Fixed non-overlapping grid blocks (edge blocks padded invalid), at least 10 valid target pixels. Positive if any valid target pixel is an event; prediction any positive on exactly those pixels. Matched blocks restrict both truth and predictions to pixels where persistence is known; require at least 10 of those pixels. Block aggregation can greatly inflate apparent performance and is not independent event detection.','',
           '| Day | Valid blocks | P | R | F1 | IoU/CSI | Shared blocks | Matched model F1 | Persistence F1 | Matched model IoU | Persistence IoU |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    fmt=lambda value:'N/A' if value is None else f'{value:.4f}'
    for full,m,b in zip(report['block'],report['block_matched_model'],report['block_matched_persistence']):
        rows.append(f"| +{full['horizon']} | {full['valid_target_count']} | {fmt(full['precision'])} | {fmt(full['recall'])} | {fmt(full['f1'])} | {fmt(full['iou'])} | {m['valid_target_count']} | {fmt(m['f1'])} | {fmt(b['f1'])} | {fmt(m['iou'])} | {fmt(b['iou'])} |")
    rows+=['',report['submission_decision'],'','Full JSON includes all P/R/F1/IoU/CSI, confusion counts, pixel Brier and abstentions for both matched cohorts. Known-label state input can carry observations beyond five input days; the preserved persistence comparator uses the original five-day window. Added recurrent input kernels increase parameter count with channel count; filters and heads are fixed. Calibration assessment shares 2024 development data with checkpoint selection. Nominal horizons do not prove data publication latency.']
    (REPORT/'evaluation.md').write_text('\n'.join(rows)+'\n')


def main():
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(4);tf.config.threading.set_inter_op_parallelism_threads(2)
    if RUN.exists():raise ValueError('Fresh run_03 required; no repeated experiment.')
    RUN.mkdir();REPORT.mkdir(exist_ok=True)
    protocol={'created_at_utc':datetime.now(timezone.utc).isoformat(),'deadline_utc':datetime.fromtimestamp(DEADLINE,timezone.utc).isoformat(),
      'variants':GROUPS,'order':['V1','V0','V2','V3'],'epochs':15,'seed':42,
      'selection':'Highest mean per-horizon pixel F1 on full 2024 development targets; first variant in V0,V1,V2,V3 order breaks ties. No test selection.',
      'block_rule':'Nonoverlapping 10x10, top-left anchored, pad partial edges invalid. >=10 observed valid target pixels per block/horizon. True/predicted event if any valid pixel is event. Matched block: same >=10 requirement on intersection of target-known and persistence-known pixels.',
      'test_disclosure':'2025 previously inspected including run_01/run_02. No new 2025 metrics until single best validation variant is frozen.',
      'stretch':'V4 skipped to prioritize controlled V0–V3 comparisons within three hours.'}
    write_json(REPORT/'protocol_before_test.json',protocol)
    write_json(REPORT/'progress.json',{'state':'preparing','deadline_utc':protocol['deadline_utc']})
    prepare()
    results={}
    for variant in protocol['order']:
        check_time();write_json(REPORT/'progress.json',{'state':'training','variant':variant,'completed':list(results)})
        train_variant(variant);results[variant]=validate_variant(variant)
        tf.keras.backend.clear_session()
    rows=['# Input-only ablation on 2024 development data','',
          '| Variant | Added input group | Channels | Parameters | Best epoch | Mean pixel F1 | Change from previous | Mean matched model F1 | Mean matched persistence F1 | Mean block F1 |',
          '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    previous=None
    for variant in GROUPS:
        r=results[variant];mean=r['mean_pixel_f1'];difference=None if previous is None else mean-previous
        rows.append(f"| {variant} | {['Control','Persistence state + age','Dynamics','Physical coast distance'][int(variant[1])]} | {r['training']['input_shape'][-1]} | {r['training']['parameter_count']} | {r['training']['best_epoch']} | {mean:.6f} | {'—' if difference is None else f'{difference:+.6f}'} | {np.mean([x['f1'] for x in r['matched_model']]):.6f} | {np.mean([x['f1'] for x in r['matched_persistence']]):.6f} | {np.mean([x['f1'] for x in r['block']]):.6f} |")
        if difference is not None and difference<=0:rows.append(f'\n{variant} added group did not improve over its preceding nested variant; its standalone addition is not supported.\n')
        previous=mean
    chosen=max(GROUPS,key=lambda v:results[v]['mean_pixel_f1'])
    selection={'chosen_variant':chosen,'mean_validation_f1':{v:results[v]['mean_pixel_f1'] for v in GROUPS},'test_used':False,'created_at_utc':datetime.now(timezone.utc).isoformat(),
       'limitation':'Nested ablation; a selected later variant cannot isolate any intermediate non-improving group without an undeclared extra variant. Such attribution is not claimed.'}
    write_json(REPORT/'selection_before_test.json',selection)
    rows+=['',f'Chosen: {chosen}, by maximum mean 2024 pixel F1. Calibration/thresholds fitted as run_02: first-half 2024, assessed second-half; checkpoint full 2024, hence development only.','',selection['limitation']]
    (REPORT/'candidate_comparison.md').write_text('\n'.join(rows)+'\n')
    write_json(REPORT/'candidate_comparison.json',results)
    selected=RUN/'variants'/chosen
    for name in ('model.keras','training.json','history.csv','decision.json'):shutil.copy2(selected/name,RUN/name)
    shutil.copytree(selected/'source',RUN/'source')
    # Freeze the selected model, decision, input manifest and source snapshots before test.
    frozen={str(p.relative_to(RUN)):file_sha256(p) for p in (RUN/'source').rglob('*') if p.is_file()}
    for name in ('model.keras','training.json','decision.json'):frozen[name]=file_sha256(RUN/name)
    write_json(RUN/'freeze.json',{'files':frozen,'input_features_manifest_sha256':file_sha256(DATA/'features.json'),'feature_sha256':file_sha256(DATA/'features.npy'),'test_disclosure':protocol['test_disclosure']})
    check_time()
    with (RUN/'test_started.json').open('x') as f:json.dump({'created_at_utc':datetime.now(timezone.utc).isoformat(),'selection_sha256':file_sha256(REPORT/'selection_before_test.json'),'block_protocol_sha256':file_sha256(REPORT/'protocol_before_test.json')},f,indent=2)
    write_json(REPORT/'progress.json',{'state':'evaluating_selected_once','chosen_variant':chosen})
    scores,index,p=collect(RUN,'test');decision=json.loads((RUN/'decision.json').read_text())
    result=assess(scores,index,p,decision)
    # Prior test evidence is read only now, after all input/decision choices are frozen.
    old=json.loads((config.ROOT/'outputs/bloom_events/run_02/evaluation.json').read_text())
    improves=all(np.mean([r[k] for r in result['pixel']])>np.mean([r[k] for r in old['model']]) for k in ('f1','iou'))
    beats=all(m['f1']>b['f1'] and m['iou']>b['iou'] for m,b in zip(result['matched_model'],result['matched_persistence']))
    headline=bool(improves and beats)
    submission='Run_03 improves mean F1/IoU over run_02 and beats matched persistence at every horizon; report as additional improved-input evidence.' if headline else 'Keep run_02 as headline satellite bloom-event result. Run_03 is an additional controlled input experiment: improvement over run_02 and consistent superiority to matched persistence were not both established. Do not tune further.'
    result.update(run_id='run_03',title='Satellite bloom-event forecast',schema='satellite-bloom-event-input-evaluation-v3',chosen_variant=chosen,
                  model_sha256=file_sha256(RUN/'model.keras'),decision=decision,definition=p.event_metadata['chosen_definition'],
                  test_event_components=p.event_metadata['split_reports']['test']['events'],event_detection_assessable=True,
                  label_reports=p.event_metadata['split_reports'],test_disclosure=protocol['test_disclosure'],
                  submission_decision=submission,headline_recommendation='run_03' if headline else 'run_02',
                  limitations=['Satellite chlorophyll-derived patches, not confirmed harmful species/toxicity or field events.','2025 previously inspected; rolling pixel forecasts are correlated.','Calibration assessment is development evidence.','Fixed physical-coast distance is approximate at local reference latitude.'])
    write_json(RUN/'evaluation.json',result);write_json(REPORT/'evaluation.json',result);write_evaluation(result)
    write_json(REPORT/'progress.json',{'state':'evaluation_complete','chosen_variant':chosen,'headline_recommendation':result['headline_recommendation'],'finished_utc':datetime.now(timezone.utc).isoformat()})


if __name__=='__main__':
    try:main()
    except Exception as error:
        write_json(REPORT/'progress.json',{'state':'stopped','error_type':type(error).__name__,'message':str(error),'submission_result':'run_02'})
        raise
