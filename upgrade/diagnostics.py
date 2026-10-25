"""New STRONG diagnostics and development strata from existing 2024 scores."""
import json
from pathlib import Path
import numpy as np
import config
from experiments.common import baseline_pipeline, cached_samples, read_json, write_json, verify_baseline
from experiments.finalize import verify_final, FINAL_DIR
from prediction.decision import apply_rule
from prediction.metrics import confusion_matrix, classification_metrics
from prediction.coverage import assess_coverage

OUT = config.ROOT / 'outputs/upgrade'


def main():
    verify_baseline(); verify_final()
    output = OUT / 'strong_diagnostics_2024.json'
    if output.exists():
        print('Existing STRONG development diagnostics retained.'); return
    pipeline = baseline_pipeline()
    rule = read_json(FINAL_DIR / 'calibration.json')['decision_rule']
    groups = {}
    examples = []; true_scores = []; cell_counts = []
    for row, p, target in cached_samples(FINAL_DIR / 'validation', pipeline):
        indices = np.arange(row.input_start_idx, row.input_end_idx+1)
        quality = assess_coverage(pipeline.transform(indices), pipeline.ocean, pipeline.dates[indices])
        _, chl_valid = pipeline.valid('chlorophyll', indices)
        observed_days = chl_valid.sum(0)
        predicted = apply_rule(p, rule)
        for name, selection in [('all', target != 255),
                                ('coverage/'+quality['state'], target != 255),
                                ('cell_input/missing', (target != 255) & (observed_days[None] == 0)),
                                ('cell_input/observed', (target != 255) & (observed_days[None] > 0))]:
            groups[name] = groups.get(name, np.zeros((3,3), dtype=np.int64)) + confusion_matrix(target[selection], predicted[selection])
        for d in range(5):
            month=str(pipeline.dates[row.target_start_idx+d])[5:7]
            valid=target[d]!=255;name='month/'+month
            groups[name]=groups.get(name,np.zeros((3,3),dtype=np.int64))+confusion_matrix(target[d][valid],predicted[d][valid])
        for lo in range(8,13):
            selection = (target != 255) & ((pipeline.latitude[:,None] >= lo) & (pipeline.latitude[:,None] < lo+1))[None]
            name = f'latitude/{lo}-{lo+1}N'
            groups[name] = groups.get(name, np.zeros((3,3),dtype=np.int64)) + confusion_matrix(target[selection], predicted[selection])
        strong = target == 2
        true_scores.append(p[strong]); cell_counts.append(np.broadcast_to(observed_days, target.shape)[strong])
        missed = strong & (predicted != 2)
        points = np.argwhere(missed)
        if len(points):
            # One severe miss per origin; honest representative examples, not invented events.
            d,y,x = map(int, points[np.argmin(p[missed,2])])
            environment = {}
            for feature in config.CHANNEL_NAMES:
                v, mask = pipeline.valid(feature, np.array([row.input_end_idx]))
                environment[feature] = float(v[0,y,x]) if mask[0,y,x] else None
            examples.append({'input_end_date':str(row.input_end), 'target_date':str(pipeline.dates[row.target_start_idx+d]),
                             'horizon':d+1,'latitude':float(pipeline.latitude[y]),'longitude':float(pipeline.longitude[x]),
                             'true_class':2,'predicted_class':int(predicted[d,y,x]),'raw_scores':p[d,y,x].tolist(),
                             'input_coverage_percent':quality['chlorophyll_mean_input_coverage_percent'],
                             'input_state':quality['state'],'cell_observed_days':int(observed_days[y,x]),
                             'environment':environment})
    scores = np.concatenate(true_scores); observations = np.concatenate(cell_counts)
    train = (pipeline.dates >= np.datetime64('2021-01-01')) & (pipeline.dates <= np.datetime64('2023-12-31'))
    validation = (pipeline.dates >= np.datetime64('2024-01-01')) & (pipeline.dates <= np.datetime64('2024-12-31'))
    report = {'period':'2024 development only', 'model_sha256':read_json(FINAL_DIR/'training_config.json')['model_sha256'],
              'unique_training_label_counts':[int((pipeline.labels[train]==c).sum()) for c in range(3)],
              'unique_2024_label_counts':[int((pipeline.labels[validation]==c).sum()) for c in range(3)],
              'metrics_by_stratum':{k:classification_metrics(v) for k,v in groups.items()},
              'strong_score_quantiles':np.percentile(scores,[5,25,50,75,95],axis=0).tolist(),
              'strong_forecasts_without_cell_input_observations':int((observations==0).sum()),
              'strong_forecasts':int(len(scores)), 'miss_examples':examples[:30],
              'interpretation':'Adjacent proxy quantiles, class imbalance and missing observations are possible contributors. Strata are associations, not causal diagnoses. These data do not establish a model capacity limit. No architecture expansion is justified by this report alone.',
              'independent_data':'No separate region, field HAB records or genuinely uninspected period identified. Historical 2025 evaluation is not rerun.'}
    write_json(output,report)
    print(json.dumps({k:report[k] for k in ['unique_training_label_counts','unique_2024_label_counts','strong_forecasts','strong_forecasts_without_cell_input_observations']},indent=2))

if __name__ == '__main__':
    main()
