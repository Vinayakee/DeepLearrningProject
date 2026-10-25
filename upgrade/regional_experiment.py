"""One declared 2024 spatial-support experiment; no threshold fitting."""
import numpy as np
import pandas as pd
import config
from experiments.common import baseline_pipeline,cached_samples,read_json,write_json
from experiments.finalize import FINAL_DIR
from prediction.decision import apply_rule
from prediction.regional import warning_level
from experiments.regional import summarize_rows


def main():
    output=config.ROOT/'outputs/upgrade/regional_support_2024.json'
    if output.exists():print('Existing spatial-support experiment retained.');return
    pipeline=baseline_pipeline();calibration=read_json(FINAL_DIR/'calibration.json')
    frame=pd.read_csv(FINAL_DIR/'regional_2024/daily_cases.csv')
    # Keep the same target-truth qualification, input gates, thresholds and dates.
    candidate=frame.copy();cursor=0
    for row,p,target in cached_samples(FINAL_DIR/'validation',pipeline):
        _, observed=pipeline.valid('chlorophyll',np.arange(row.input_start_idx,row.input_end_idx+1))
        input_support=observed.any(0)&pipeline.ocean
        risk=apply_rule(p,calibration['decision_rule'])
        for h in range(5):
            percent=float((risk[h][input_support]==2).mean()*100) if input_support.any() else None
            available=bool(candidate.loc[cursor,'available']) and input_support.sum()/pipeline.ocean.sum()>=.05
            level=warning_level(percent,calibration) if percent is not None else 'INSUFFICIENT DATA'
            candidate.loc[cursor,'available']=available
            candidate.loc[cursor,'issued_level']=level if available else 'INSUFFICIENT DATA'
            candidate.loc[cursor,'predicted_strong_percent']=percent
            candidate.loc[cursor,'aggregation_supported_pixels']=int(input_support.sum())
            cursor+=1
    original=summarize_rows(frame[frame.horizon==1]);new=summarize_rows(candidate[candidate.horizon==1])
    a=original['any_warning'];b=new['any_warning'];sa=original['strong_warning'];sb=new['strong_warning']
    retained=(b['recall']>a['recall'] and sb['recall']>=sa['recall'] and b['false_warning_days']<=a['false_warning_days'])
    report={'period':'2024 development only','changed_variable':'Aggregate predicted labels on cells observed at least once in the five input days rather than dense monthly domain.',
            'motivation':'Dense extrapolation and sparsely sampled proxy truth use different support; compare an input-observed domain without looking at future target masks.',
            'fixed':'Model, score factors, P75/P90 training thresholds, target coverage qualification and input gates.',
            'declared_criterion':'Higher any-risk recall, no strong-recall loss and no increase in false-warning days.',
            'baseline':original,'candidate':new,'criterion_passed':bool(retained),'deployed':False,
            'reason':'Support changes alter the estimand; development evidence requires independent validation. Frozen historical model and evaluation remain unchanged.'}
    candidate.to_csv(output.with_suffix('.csv'),index=False);write_json(output,report)
    print(f"Regional observed-input support: TP={b['correctly_warned_high_risk_days']} FP={b['false_warning_days']} FN={b['missed_high_risk_days']}; criterion={retained}",flush=True)

if __name__=='__main__':main()
