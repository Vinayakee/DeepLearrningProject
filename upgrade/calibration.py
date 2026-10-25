"""Temperature experiment with disjoint 2024 temporal fitting/assessment blocks."""
import json
import numpy as np
from scipy.optimize import minimize_scalar
import config
from experiments.common import baseline_pipeline, cached_samples, write_json, read_json
from experiments.finalize import FINAL_DIR, verify_final


def temperature_scale(scores, temperature):
    p = np.asarray(scores, dtype=float)
    if not np.isfinite(temperature) or temperature <= 0 or p.shape[-1] != 3 or not np.isfinite(p).all() or np.any(p < 0) or not np.allclose(p.sum(-1),1,atol=1e-5):
        raise ValueError('Finite normalized three-class scores and positive temperature required.')
    log = np.log(np.clip(p,1e-12,1))/temperature
    log -= log.max(-1,keepdims=True)
    exp = np.exp(log)
    return exp/exp.sum(-1,keepdims=True)


def calibration_metrics(y, p, bins=15):
    y = np.asarray(y,dtype=int); p=np.asarray(p,dtype=float)
    if not len(y):
        raise ValueError('Calibration assessment requires observed target labels.')
    truth = np.eye(3)[y]
    result={'pixels':len(y),'nll':float(-np.log(np.clip(p[np.arange(len(y)),y],1e-12,1)).mean()),
            'brier':float(np.square(p-truth).sum(-1).mean()),
            'class_brier':np.square(p-truth).mean(0).tolist(), 'class_counts':np.bincount(y,minlength=3).tolist()}
    reliability={}
    for name, score, event in [('top_label',p.max(-1),p.argmax(-1)==y)] + [(str(c),p[:,c],y==c) for c in range(3)]:
        bucket=np.minimum((score*bins).astype(int),bins-1)
        rows=[]; ece=0.
        for b in range(bins):
            keep=bucket==b
            n=int(keep.sum())
            if n:
                mean_score=float(score[keep].mean()); frequency=float(event[keep].mean())
                ece+=n/len(y)*abs(mean_score-frequency)
                rows.append({'bin':b,'count':n,'mean_score':mean_score,'observed_frequency':frequency})
        reliability[name]={'ece':float(ece),'bins':rows}
    result['reliability']=reliability
    return result


def fit_temperature(y,p):
    result=minimize_scalar(lambda log_t: -np.log(np.clip(temperature_scale(p,np.exp(log_t))[np.arange(len(y)),y],1e-12,1)).mean(),
                           bounds=(-2,2), method='bounded',options={'xatol':1e-5})
    if not result.success:
        raise ValueError('Temperature optimization failed.')
    return float(np.exp(result.x))


def main():
    verify_final()
    output=config.ROOT/'outputs/upgrade/calibration_2024.json'
    if output.exists():
        print('Existing calibration experiment retained.'); return
    pipeline=baseline_pipeline(); fit_y=[]; fit_p=[]; assessment_y=[]; assessment_p=[]; excluded=0
    for row,p,target in cached_samples(FINAL_DIR/'validation',pipeline):
        valid=target!=255
        if str(row.target_end) <= '2024-06-30':
            fit_y.append(target[valid]); fit_p.append(p[valid])
        elif str(row.input_start) >= '2024-07-01':
            assessment_y.append(target[valid]); assessment_p.append(p[valid])
        else:
            excluded+=1
    y=np.concatenate(fit_y); p=np.concatenate(fit_p); t=fit_temperature(y,p)
    ay=np.concatenate(assessment_y); ap=np.concatenate(assessment_p)
    before=calibration_metrics(ay,ap); after=calibration_metrics(ay,temperature_scale(ap,t))
    passes=after['nll'] < before['nll'] and after['brier'] < before['brier'] and all(a<=b for a,b in zip(after['class_brier'],before['class_brier']))
    report={'period':'2024 only','model_sha256':read_json(FINAL_DIR/'training_config.json')['model_sha256'],
            'method':'Scalar temperature on log clipped softmax scores; class decisions remain on original scores.',
            'fit_complete_windows':'2024-01-01 through 2024-06-30','assessment_complete_windows':'2024-07-01 through 2024-12-31',
            'excluded_boundary_origins':excluded,'fit_pixels':len(y),'temperature':t,'before':before,'after':after,
            'declared_criterion':'Lower temporal assessment NLL and Brier, with no per-class Brier degradation.',
            'criterion_passed':bool(passes),'active_probability_calibration':False,
            'reason':'Temporal assessment is development evidence only; no new independent observations exist. Frozen production probabilities and 2025 evaluation are unchanged.',
            'limits':'Pixels/horizons are correlated; ECE depends on bins and prevalence. Global improvement does not establish reliable event confidence.'}
    write_json(output,report)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    for c,ax in enumerate(axes):
        ax.plot([0,1],[0,1],'k--')
        for label,r in [('Raw',before),('Temperature',after)]:
            rows=r['reliability'][str(c)]['bins']
            ax.plot([b['mean_score'] for b in rows],[b['observed_frequency'] for b in rows],marker='o',label=label)
        ax.set(title=['Normal','Elevated','Strong'][c],xlabel='Mean score',ylabel='Observed class frequency',xlim=(0,1),ylim=(0,1)); ax.legend()
    fig.suptitle('Disjoint 2024 assessment; development evidence, not independent calibration')
    fig.tight_layout(); fig.savefig(output.with_suffix('.png'),dpi=140); plt.close(fig)
    print(json.dumps({'temperature':t,'before_brier':before['brier'],'after_brier':after['brier'],'criterion_passed':bool(passes)},indent=2))

if __name__=='__main__':
    main()
