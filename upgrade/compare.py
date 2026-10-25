"""Assess the new sampling checkpoint with held rules and preserve all old results."""
import numpy as np
import config
from experiments.common import baseline_pipeline,cached_samples,read_json,write_json
from experiments.finalize import FINAL_DIR,verify_final
from experiments.development import metrics
from experiments.regional import evaluate_cache
from prediction.coverage import DEFAULT_POLICY

ROOT=config.ROOT/'outputs/upgrade'


def main():
    verify_final(); output=ROOT/'sampling_comparison_2024.json'
    if output.exists():print('Existing sampling comparison retained.');return
    training=read_json(ROOT/'balanced_sampling/training_config.json')
    if not training.get('epochs_completed') or not (ROOT/'balanced_sampling/validation/cache.json').exists():
        raise ValueError('Full sampling experiment and real prediction cache must be complete.')
    pipeline=baseline_pipeline();y=[];p=[]
    for row,scores,target in cached_samples(ROOT/'balanced_sampling/validation',pipeline):
        valid=target!=255;y.append(target[valid]);p.append(scores[valid])
    y=np.concatenate(y);p=np.concatenate(p)
    calibration=read_json(FINAL_DIR/'calibration.json')
    candidate=metrics(y,p,calibration['decision_rule'])
    direct=metrics(y,p,{'type':'argmax'})
    region=evaluate_cache(ROOT/'balanced_sampling/validation',pipeline,calibration['decision_rule'],calibration,DEFAULT_POLICY,ROOT/'balanced_sampling/regional_2024')
    previous=read_json(FINAL_DIR/'evaluation.json')['validation']
    old=previous['model'];a=previous['regional']['daily_horizon_1'];b=region['daily_horizon_1']
    retained=(candidate['macro_f1']>=old['macro_f1']+.005 and all(candidate['classes'][c]['f1']>=old['classes'][c]['f1'] for c in [1,2])
              and b['any_warning']['recall']>=a['any_warning']['recall'] and b['strong_warning']['recall']>=a['strong_warning']['recall']
              and b['any_warning']['false_warning_days']<=a['any_warning']['false_warning_days'])
    result={'period':'2024 development only','criterion':training['retention_criterion'],'candidate':{'model':candidate,'argmax':direct,'regional':region},
            'reference':previous,'criterion_passed':bool(retained),'deployed':False,
            'decision':'Retain as a development candidate requiring new independent evaluation.' if retained else 'Reject sampling change; keep existing frozen model.',
            'independent_evaluation':'Existing 2025 results are preserved, not recomputed. No new independent observations are available.'}
    write_json(output,result)
    print(f"Sampling macro F1={candidate['macro_f1']:.4f}, elevated F1={candidate['classes'][1]['f1']:.4f}, strong F1={candidate['classes'][2]['f1']:.4f}, retained={retained}",flush=True)

if __name__=='__main__':main()
