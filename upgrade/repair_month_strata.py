"""Correct month grouping at forecast target date; no new experiment/inference."""
import numpy as np
import config
from experiments.common import baseline_pipeline,cached_samples,read_json,write_json
from experiments.finalize import FINAL_DIR
from prediction.decision import apply_rule
from prediction.metrics import confusion_matrix,classification_metrics


def main():
    path=config.ROOT/'outputs/upgrade/strong_diagnostics_2024.json';report=read_json(path)
    if report.get('month_grouping')=='actual_target_date':
        print('Month grouping already correct.');return
    pipeline=baseline_pipeline();rule=read_json(FINAL_DIR/'calibration.json')['decision_rule'];groups={}
    for row,p,target in cached_samples(FINAL_DIR/'validation',pipeline):
        predicted=apply_rule(p,rule)
        for d in range(5):
            month=str(pipeline.dates[row.target_start_idx+d])[5:7];valid=target[d]!=255
            groups[month]=groups.get(month,np.zeros((3,3),dtype=np.int64))+confusion_matrix(target[d][valid],predicted[d][valid])
    report['metrics_by_stratum']={k:v for k,v in report['metrics_by_stratum'].items() if not k.startswith('month/')}
    report['metrics_by_stratum'].update({'month/'+m:classification_metrics(v) for m,v in groups.items()})
    report['month_grouping']='actual_target_date'
    write_json(path,report);print('Corrected only month grouping; original diagnostics and cached scores retained.')

if __name__=='__main__':main()
