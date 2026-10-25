"""Validation selection and one frozen satellite bloom-event test assessment."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
from sklearn.metrics import precision_recall_curve
import config
from preprocessing.bloom_events import BloomPipeline,write_json
from preprocessing.data_utils import file_sha256
from training.train_bloom_events import BloomGenerator,DEFAULT_RUN


def load_run(run):
    run=Path(run);record=json.loads((run/'training.json').read_text())
    if record.get('state')!='trained' or not record.get('epochs_completed'):
        raise ValueError('Complete satellite bloom-event training first.')
    if file_sha256(run/'model.keras')!=record['model_sha256']:
        raise ValueError('Satellite-event checkpoint changed.')
    dataset=Path(record['dataset_dir'])
    if file_sha256(dataset/'manifest.json')!=record['dataset_manifest_sha256']:
        raise ValueError('Satellite-event fitting data changed.')
    return record,BloomPipeline(dataset)


def load_model(run):
    import tensorflow as tf
    import models.bloom_events
    return tf.keras.models.load_model(Path(run)/'model.keras',compile=False)


def collect(run,split):
    if split not in ('validation','test'):raise ValueError('Only declared validation/test collection is supported.')
    run=Path(run);record,p=load_run(run)
    directory=run/('scores_'+split)
    if directory.exists():raise ValueError('Prediction collection exists; it cannot be overwritten.')
    directory.mkdir()
    generator=BloomGenerator(split,2,False,p)
    array=np.lib.format.open_memmap(directory/'scores.npy',mode='w+',dtype=np.float32,
                                  shape=(generator.num_samples,5,*p.shape[1:]))
    model=load_model(run)
    for batch in range(len(generator)):
        X,y=generator[batch]
        scores=model(X,training=False).numpy()
        if scores.shape!=y.shape or not np.isfinite(scores).all() or np.any((scores<0)|(scores>1)):
            raise ValueError('Invalid satellite-event model scores.')
        start=batch*generator.batch_size
        array[start:start+len(X)]=scores
    array.flush();generator.indices.to_csv(directory/'index.csv',index=False)
    write_json(directory/'metadata.json',{'split':split,'shape':list(array.shape),'model_sha256':record['model_sha256'],
               'scores_sha256':file_sha256(directory/'scores.npy'),'index_sha256':file_sha256(directory/'index.csv')})
    return array,generator.indices,p


def persistence(pipeline,start):
    """Latest causally confirmed event/non-high state in the five input days."""
    state=np.full(pipeline.shape[1:],255,np.uint8)
    for day in pipeline.asof[start:start+5]:
        known=day!=255
        state[known]=day[known]
    return state


def counts(truth,prediction,valid):
    return np.bincount((truth[valid].astype(int)*2+prediction[valid].astype(int)).ravel(),minlength=4).astype(np.int64)


def metrics(confusion,squared_error=None):
    tn,fp,fn,tp=map(int,confusion)
    n=tn+fp+fn+tp
    return {'valid_target_count':n,'positive_target_count':tp+fn,'predicted_positive_count':tp+fp,
            'tn':tn,'fp':fp,'fn':fn,'tp':tp,'precision':tp/(tp+fp) if tp+fp else None,
            'recall':tp/(tp+fn) if tp+fn else None,'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
            'iou':tp/(tp+fp+fn) if tp+fp+fn else None,'csi':tp/(tp+fp+fn) if tp+fp+fn else None,
            'false_positive_rate':fp/(fp+tn) if fp+tn else None,
            'brier_score':squared_error/n if squared_error is not None and n else None,
            'detection_skill_assessable':bool(tp+fn),
            'undefined_metrics_note':'Recall is undefined without positive truth; precision is undefined without positive predictions; F1/IoU/CSI are undefined if their denominators are zero.'}


def select(run):
    run=Path(run)
    if any((run/name).exists() for name in ('decision.json','freeze.json','test_started.json')):
        raise ValueError('Satellite-event decisions already exist; use a separate declared run.')
    scores,index,p=collect(run,'validation')
    truth,values=[],[]
    for i,row in enumerate(index.itertuples()):
        y=p.labels[row.target_start_idx:row.target_end_idx+1]
        valid=y!=255
        truth.append(y[valid]);values.append(np.asarray(scores[i])[valid])
    y,s=np.concatenate(truth),np.concatenate(values)
    if len(np.unique(y))==2:
        precision,recall,thresholds=precision_recall_curve(y,s)
        f1=2*precision[:-1]*recall[:-1]/np.maximum(precision[:-1]+recall[:-1],1e-12)
        threshold=float(thresholds[np.argmax(f1)])
        method='Maximum binary F1 on observed 2024 validation targets; no test scores used.'
    else:
        threshold=.5
        method='Fixed sigmoid threshold 0.5: validation has no positive events, so detection threshold optimization and probability calibration are unsupported.'
    record=json.loads((run/'training.json').read_text())
    decision={'schema':'satellite-event-decision-v1','threshold':threshold,'selection':method,
              'fit_split':'validation (2024)','validation_positive_forecasts':int((y==1).sum()),
              'validation_valid_forecasts':len(y),'probability_calibrated':False,
              'model_sha256':record['model_sha256'],'dataset_manifest_sha256':record['dataset_manifest_sha256']}
    with (run/'decision.json').open('x') as handle:json.dump(decision,handle,indent=2)
    source=Path(__file__).resolve();destination=run/'source/prediction/bloom_events.py';destination.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source,destination)
    freeze={'files':{name:file_sha256(run/name) for name in ('training.json','model.keras','decision.json','source/prediction/bloom_events.py')},
            'test_disclosure':p.event_metadata['test_disclosure']}
    with (run/'freeze.json').open('x') as handle:json.dump(freeze,handle,indent=2)
    return decision


def verify(run):
    run=Path(run);freeze=json.loads((run/'freeze.json').read_text())
    for name,digest in freeze['files'].items():
        if file_sha256(run/name)!=digest:raise ValueError('Frozen satellite-event run changed.')
    record,p=load_run(run)
    decision=json.loads((run/'decision.json').read_text())
    if decision['model_sha256']!=record['model_sha256'] or decision['dataset_manifest_sha256']!=record['dataset_manifest_sha256']:
        raise ValueError('Satellite-event decisions do not match the model/dataset.')
    return record,p,decision


def evaluate(run,report_dir):
    run,report_dir=Path(run),Path(report_dir)
    if report_dir.exists():raise ValueError('Choose a fresh test report directory.')
    record,p,decision=verify(run)
    with (run/'test_started.json').open('x') as handle:
        json.dump({'decision_sha256':file_sha256(run/'decision.json'),'disclosure':p.event_metadata['test_disclosure']},handle)
    scores,index,p=collect(run,'test')
    sums={name:np.zeros((5,4),np.int64) for name in ('model_all_observed','persistence_available','model_common_cohort','persistence_common_cohort')}
    brier=np.zeros(5);positive_truth=np.zeros(5,np.int64);persistence_missed=np.zeros(5,np.int64)
    unavailable=np.zeros(5,np.int64)
    for i,row in enumerate(index.itertuples()):
        truth=np.asarray(p.labels[row.target_start_idx:row.target_end_idx+1])
        pred=np.asarray(scores[i])>=decision['threshold']
        baseline=persistence(p,row.input_start_idx)
        for h in range(5):
            observed=truth[h]!=255
            known=observed&(baseline!=255)
            sums['model_all_observed'][h]+=counts(truth[h],pred[h],observed)
            sums['persistence_available'][h]+=counts(truth[h],baseline,known)
            sums['model_common_cohort'][h]+=counts(truth[h],pred[h],known)
            sums['persistence_common_cohort'][h]+=counts(truth[h],baseline,known)
            brier[h]+=float(np.sum((np.asarray(scores[i,h])[observed]-truth[h][observed])**2,dtype=np.float64))
            positive_truth[h]+=int((observed&(truth[h]==1)).sum())
            persistence_missed[h]+=int((observed&(truth[h]==1)&((baseline==255)|(baseline==0))).sum())
            unavailable[h]+=int((observed&(baseline==255)).sum())
    report={'schema':'satellite-bloom-event-evaluation-v1','title':'Satellite bloom-event forecast',
            'split':'test (2025)','model_sha256':record['model_sha256'],'decision':decision,
            'label_reports':p.event_metadata['split_reports'],'model':[], 'persistence':[], 'common_cohort':[],
            'test_event_components':p.event_metadata['split_reports']['test']['events'],
            'event_detection_assessable':bool(p.event_metadata['split_reports']['test']['events']),
            'pr_auc':None,'pr_auc_note':'No positive observed test event labels; PR AUC is not estimable.' if not positive_truth.sum() else 'Not computed in this bounded report.',
            'disclosure':p.event_metadata['test_disclosure'],
            'limitations':['Chlorophyll-derived satellite event labels, not confirmed HAB occurrence or toxicity.',
                           'Strict observed three-day common-patch rule; missing candidates stay unknown, not no-event.',
                           'Rolling pixel forecasts overlap and are correlated, not independent events.',
                           'Primary model metrics include all observed targets, including weak input windows; dashboard availability is separately coverage-gated.',
                           'No positive validation/test targets means detection generalization and calibration cannot be established.',
                           'Input publication latency is unverified; nominal horizons do not prove operational warning lead time.']}
    for h in range(5):
        report['model'].append({'horizon':h+1,**metrics(sums['model_all_observed'][h],brier[h])})
        report['persistence'].append({'horizon':h+1,**metrics(sums['persistence_available'][h]),
                                      'abstained_valid_targets':int(unavailable[h]),
                                      'recall_including_abstentions':float((positive_truth[h]-persistence_missed[h])/positive_truth[h]) if positive_truth[h] else None})
        report['common_cohort'].append({'horizon':h+1,'model':metrics(sums['model_common_cohort'][h]),
                                        'persistence':metrics(sums['persistence_common_cohort'][h])})
    report_dir.mkdir(parents=True)
    write_json(run/'evaluation.json',report);write_json(report_dir/'evaluation.json',report)
    rows=['# Satellite bloom-event forecast — frozen 2025 evaluation','',
          '**2025 was previously inspected. These labels are satellite-derived; no confirmed HAB skill is claimed.**','',
          f"Observed test components: {report['test_event_components']}. Recall and PR AUC are not estimable without positive truth.",'',
          '| Horizon | Model valid targets | Precision | Recall | F1 | IoU / CSI | Persistence valid targets | Persistence F1 |',
          '|---|---:|---:|---:|---:|---:|---:|---:|']
    display=lambda v:'N/A' if v is None else f'{v:.6f}'
    for model,base in zip(report['model'],report['persistence']):
        rows.append(f"| +{model['horizon']} | {model['valid_target_count']} | {display(model['precision'])} | {display(model['recall'])} | {display(model['f1'])} | {display(model['csi'])} | {base['valid_target_count']} | {display(base['f1'])} |")
    rows+=['','Model and standalone persistence use different availability cohorts. Use the paired `common_cohort` JSON counts for matched comparisons. Undefined metrics are N/A, never invented zeros.','',*['- '+item for item in report['limitations']]]
    (report_dir/'evaluation.md').write_text('\n'.join(rows)+'\n')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('select','evaluate'))
    parser.add_argument('--run-dir',type=Path,default=DEFAULT_RUN)
    parser.add_argument('--report-dir',type=Path,default=config.ROOT/'docs/results/bloom_events/run_01')
    args=parser.parse_args()
    result=select(args.run_dir) if args.action=='select' else evaluate(args.run_dir,args.report_dir)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
