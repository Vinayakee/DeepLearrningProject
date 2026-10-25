"""Validation-only calibration/threshold selection and one relaxed-run test evaluation."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
from scipy.special import expit,logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve,average_precision_score
import config
from preprocessing.bloom_events import write_json
from preprocessing.data_utils import file_sha256
from prediction.bloom_events import collect,load_run,verify,counts,metrics,persistence

DEFAULT_RUN=config.ROOT/'outputs/bloom_events/run_02'
DEFAULT_REPORT=config.ROOT/'docs/results/bloom_events_v2'


def apply_scores(scores,decision):
    calibration=decision.get('calibration',{})
    if not calibration.get('active'):return np.asarray(scores)
    return expit(calibration['slope']*logit(np.clip(scores,1e-7,1-1e-7))+calibration['intercept'])


def validation_parts(scores,index,pipeline):
    parts={key:{'truth':[],'scores':[],'positive_days':set()} for key in ('fit','assessment')}
    for i,row in enumerate(index.itertuples()):
        if row.input_start>='2024-01-01' and row.target_end<='2024-06-30':key='fit'
        elif row.input_start>='2024-07-01' and row.target_end<='2024-12-31':key='assessment'
        else:continue
        target=np.asarray(pipeline.labels[row.target_start_idx:row.target_end_idx+1]);valid=target!=255
        parts[key]['truth'].append(target[valid]);parts[key]['scores'].append(np.asarray(scores[i])[valid])
        for h in range(5):
            if np.any(target[h]==1):parts[key]['positive_days'].add(str(pipeline.dates[row.target_start_idx+h]))
    for part in parts.values():
        part['truth']=np.concatenate(part['truth']) if part['truth'] else np.array([],np.uint8)
        part['scores']=np.concatenate(part['scores']) if part['scores'] else np.array([],float)
    return parts


def fit_decision(fit_y,fit_scores,assessment_y,assessment_scores,fit_positive_days,assessment_positive_days):
    fit_y,assessment_y=np.asarray(fit_y),np.asarray(assessment_y)
    enough=(int((fit_y==1).sum())>=100 and int((fit_y==0).sum())>=100 and fit_positive_days>=10)
    assessment_enough=(int((assessment_y==1).sum())>=100 and int((assessment_y==0).sum())>=100
                       and assessment_positive_days>=5)
    calibration={'active':False,'kind':'platt','fit_period':['2024-01-01','2024-06-30'],
                 'assessment_period':['2024-07-01','2024-12-31'],
                 'fit_valid_forecasts':len(fit_y),'fit_positive_forecasts':int((fit_y==1).sum()),
                 'fit_positive_target_days':fit_positive_days,'assessment_valid_forecasts':len(assessment_y),
                 'assessment_positive_forecasts':int((assessment_y==1).sum()),'assessment_positive_target_days':assessment_positive_days,
                 'requirements':'Fit >=100 observations of each class and >=10 positive days; assessment >=100 of each class and >=5 positive days. Retain only positive slope and non-worse second-half Brier.',
                 'independent_validation':False,
                 'limitation':'The model checkpoint used full 2024 validation loss. These chronological calibration diagnostics remain development evidence, not independent validation.'}
    if enough and assessment_enough:
        model=LogisticRegression(C=1.,max_iter=1000,random_state=42).fit(logit(np.clip(fit_scores,1e-7,1-1e-7))[:,None],fit_y)
        slope,intercept=float(model.coef_[0,0]),float(model.intercept_[0])
        calibrated=expit(slope*logit(np.clip(assessment_scores,1e-7,1-1e-7))+intercept)
        raw_brier=float(np.mean((assessment_scores-assessment_y)**2))
        calibrated_brier=float(np.mean((calibrated-assessment_y)**2))
        calibration.update(slope=slope,intercept=intercept,raw_assessment_brier=raw_brier,
                           calibrated_assessment_brier=calibrated_brier,
                           active=bool(slope>0 and calibrated_brier<=raw_brier))
        calibration['reason']='Retained: second-half development Brier did not worsen.' if calibration['active'] else 'Rejected: nonpositive discrimination or worse second-half development Brier; raw scores retained.'
    else:calibration['reason']='Insufficient positive/negative validation support for the declared calibration criteria.'
    decision={'schema':'satellite-event-decision-v2','calibration':calibration,'probability_calibrated':calibration['active'],
              'fit_split':'first-half 2024 validation, complete windows only','threshold_objective':'Maximum binary F1 on observed first-half 2024 fit targets; higher threshold breaks exact ties.'}
    transformed=apply_scores(fit_scores,decision)
    if enough:
        precision,recall,thresholds=precision_recall_curve(fit_y,transformed)
        f1=2*precision[:-1]*recall[:-1]/np.maximum(precision[:-1]+recall[:-1],1e-12)
        best=max(range(len(thresholds)),key=lambda i:(f1[i],thresholds[i]))
        decision.update(threshold=float(thresholds[best]),threshold_tuned=True)
    else:decision.update(threshold=.5,threshold_tuned=False)
    for name,y,s in [('fit',fit_y,fit_scores),('assessment',assessment_y,assessment_scores)]:
        p=apply_scores(s,decision);valid=np.ones(y.shape,bool)
        decision[name+'_metrics']=metrics(counts(y,p>=decision['threshold'],valid),float(np.sum((p-y)**2)))
    return decision


def select(run=DEFAULT_RUN,report_dir=DEFAULT_REPORT):
    run,report_dir=Path(run),Path(report_dir)
    if any((run/name).exists() for name in ('decision.json','freeze.json','test_started.json')):
        raise ValueError('Decisions already fitted; no overwrite or repeat selection.')
    scores,index,p=collect(run,'validation')
    parts=validation_parts(scores,index,p)
    decision=fit_decision(parts['fit']['truth'],parts['fit']['scores'],parts['assessment']['truth'],parts['assessment']['scores'],
                          len(parts['fit']['positive_days']),len(parts['assessment']['positive_days']))
    record=json.loads((run/'training.json').read_text())
    decision.update(model_sha256=record['model_sha256'],dataset_manifest_sha256=record['dataset_manifest_sha256'],
                    definition=p.event_metadata['chosen_definition'],test_disclosure=p.event_metadata['test_disclosure'])
    with (run/'decision.json').open('x') as handle:json.dump(decision,handle,indent=2,allow_nan=False)
    sources=('prediction/bloom_events_v2.py','prediction/bloom_events.py','preprocessing/bloom_events_v2.py')
    for name in sources:
        destination=run/'source'/name;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(config.ROOT/name,destination)
    files=['training.json','model.keras','decision.json']+['source/'+name for name in sources]
    with (run/'freeze.json').open('x') as handle:json.dump({'files':{name:file_sha256(run/name) for name in files},'test_disclosure':decision['test_disclosure']},handle,indent=2)
    write_json(report_dir/'decision.json',decision)
    return decision


def evaluate(run=DEFAULT_RUN,report_dir=DEFAULT_REPORT):
    run,report_dir=Path(run),Path(report_dir)
    if (report_dir/'evaluation.json').exists():raise ValueError('Existing relaxed evaluation cannot be overwritten.')
    record,p,decision=verify(run)
    with (run/'test_started.json').open('x') as handle:json.dump({'decision_sha256':file_sha256(run/'decision.json'),'test_disclosure':decision['test_disclosure']},handle)
    scores,index,p=collect(run,'test')
    model_counts=np.zeros((5,4),np.int64);base_counts=np.zeros_like(model_counts);common_model=np.zeros_like(model_counts)
    unavailable=np.zeros(5,np.int64);brier=np.zeros(5);truths=[[] for _ in range(5)];probs=[[] for _ in range(5)]
    missing_positive=np.zeros(5,np.int64);positive=np.zeros(5,np.int64)
    for i,row in enumerate(index.itertuples()):
        y=np.asarray(p.labels[row.target_start_idx:row.target_end_idx+1]);prob=apply_scores(scores[i],decision)
        prediction=prob>=decision['threshold'];baseline=persistence(p,row.input_start_idx)
        for h in range(5):
            valid=y[h]!=255;common=valid&(baseline!=255)
            model_counts[h]+=counts(y[h],prediction[h],valid)
            base_counts[h]+=counts(y[h],baseline,common);common_model[h]+=counts(y[h],prediction[h],common)
            brier[h]+=float(np.sum((prob[h][valid]-y[h][valid])**2,dtype=np.float64))
            unavailable[h]+=int((valid&~common).sum());positive[h]+=int((valid&(y[h]==1)).sum())
            missing_positive[h]+=int((valid&(y[h]==1)&((baseline==255)|(baseline==0))).sum())
            truths[h].append(y[h][valid]);probs[h].append(prob[h][valid])
    report={'schema':'satellite-bloom-event-evaluation-v2','title':'Satellite bloom-event forecast','run_id':'run_02',
            'model_sha256':record['model_sha256'],'decision':decision,'definition':p.event_metadata['chosen_definition'],
            'label_reports':p.event_metadata['split_reports'],'test_event_components':p.event_metadata['split_reports']['test']['events'],
            'event_detection_assessable':bool(positive.sum()),'model':[],'persistence':[],'common_cohort':[],
            'disclosure':decision['test_disclosure'],
            'limitations':['Chlorophyll-derived satellite patches, not verified HAB species, toxicity or harmful occurrence.',
                           'The selected definition allows one observed day: it detects spatial patches, not persistent multi-day blooms.',
                           'Unobserved and uncertain target pixels stay unknown; monitoring absence is not inferred.',
                           'Rolling pixel forecasts and connected components are correlated; component counts are not independent biological events.',
                           'Model metrics on all observed targets include sparse input windows; dashboard inference is separately coverage-gated.',
                           'Calibration and threshold selection are 2024 development fits; the checkpoint also used 2024, so calibration assessment is not independent.',
                           '2025 was previously inspected, including strict-run results; this is not an untouched holdout.',
                           'Satellite publication latency is unverified; nominal forecast horizons do not establish operational warning lead.']}
    for h in range(5):
        y,prob=np.concatenate(truths[h]),np.concatenate(probs[h])
        report['model'].append({'horizon':h+1,**metrics(model_counts[h],brier[h]),
                               'pr_auc':float(average_precision_score(y,prob)) if len(np.unique(y))==2 else None})
        report['persistence'].append({'horizon':h+1,**metrics(base_counts[h]),'abstained_valid_targets':int(unavailable[h]),
                                     'recall_including_abstentions':float((positive[h]-missing_positive[h])/positive[h]) if positive[h] else None})
        report['common_cohort'].append({'horizon':h+1,'model':metrics(common_model[h]),'persistence':metrics(base_counts[h])})
    write_json(run/'evaluation.json',report);write_json(report_dir/'evaluation.json',report)
    rows=['# Satellite bloom-event forecast — relaxed run_02','',
          '**2025 was previously inspected. Satellite-derived proxy labels; no verified-HAB or operational claim.**','',
          '| Day | Model valid N | Precision | Recall | F1 | IoU / CSI | PR AUC | Persistence available N | Persistence F1 |',
          '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    fmt=lambda x:'N/A' if x is None else f'{x:.4f}'
    for model,base in zip(report['model'],report['persistence']):
        rows.append(f"| +{model['horizon']} | {model['valid_target_count']} | {fmt(model['precision'])} | {fmt(model['recall'])} | {fmt(model['f1'])} | {fmt(model['csi'])} | {fmt(model['pr_auc'])} | {base['valid_target_count']} | {fmt(base['f1'])} |")
    rows+=['','## Matched-target comparison','',
           '| Day | Shared N | Model P | Model R | Model F1 | Model IoU / CSI | Persistence P | Persistence R | Persistence F1 | Persistence IoU / CSI |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in report['common_cohort']:
        m,b=row['model'],row['persistence']
        rows.append(f"| +{row['horizon']} | {m['valid_target_count']} | {fmt(m['precision'])} | {fmt(m['recall'])} | {fmt(m['f1'])} | {fmt(m['csi'])} | {fmt(b['precision'])} | {fmt(b['recall'])} | {fmt(b['f1'])} | {fmt(b['csi'])} |")
    rows+=['','Standalone denominators differ. The second table compares identical observed targets. Full JSON retains TP/FP/FN/TN, abstentions, Brier and per-metric counts.','',*['- '+item for item in report['limitations']]]
    (report_dir/'evaluation.md').write_text('\n'.join(rows)+'\n')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('select','evaluate'))
    parser.add_argument('--run-dir',type=Path,default=DEFAULT_RUN);parser.add_argument('--report-dir',type=Path,default=DEFAULT_REPORT)
    args=parser.parse_args();print(json.dumps((select if args.action=='select' else evaluate)(args.run_dir,args.report_dir),indent=2))


if __name__=='__main__':main()
