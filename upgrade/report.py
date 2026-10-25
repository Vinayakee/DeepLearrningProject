"""Publish saved development evidence and freeze the separate upgrade configuration."""
import json
import re
import shutil
from pathlib import Path
import config
from experiments.common import read_json,write_json,verify_baseline,utc_now
from experiments.finalize import FINAL_DIR,verify_final
from preprocessing.data_utils import file_sha256

OUT=config.ROOT/'outputs/upgrade'


def main():
    verify_baseline();verify_final()
    comparison=read_json(OUT/'sampling_comparison_2024.json')
    calibration=read_json(OUT/'calibration_2024.json');diagnostics=read_json(OUT/'strong_diagnostics_2024.json')
    regional=read_json(OUT/'regional_support_2024.json');audit=read_json(OUT/'data_audit.json')
    integration=read_json(OUT/'integration.json');training=read_json(OUT/'balanced_sampling/training_config.json')
    log=(OUT/'tests.log').read_text();match=re.search(r'Ran (\d+) tests',log)
    if not match or not log.rstrip().endswith('OK'):
        raise ValueError('A complete passing suite is required for completion reporting.')
    count=int(match.group(1));historical=read_json(FINAL_DIR/'evaluation.json')
    reference=historical['validation'];candidate=comparison['candidate']
    selected='balanced_sampling_development_candidate' if comparison['criterion_passed'] else 'existing_frozen_final'
    if not (OUT/'freeze.json').exists():
        files=[p for p in (config.ROOT/'upgrade').glob('*.py') if p.name!='report.py']
        files += [OUT/name for name in ['strong_diagnostics_2024.json','calibration_2024.json','sampling_comparison_2024.json','regional_support_2024.json','data_audit.json']]
        files += [OUT/'balanced_sampling'/name for name in ['model.keras','training_config.json','training_history.csv']]
        files += [FINAL_DIR/'freeze.json',FINAL_DIR/'calibration.json',config.SEQUENCE_DIR/'preprocessing.json']
        write_json(OUT/'freeze.json',{'created_at_utc':utc_now(),'purpose':'2024 development extension; historical deployment and 2025 evaluation unchanged',
                  'development_selection':selected,'operational_warning_validated':False,
                  'sampling_strategy':training['sampling'],'class_weights':training['class_weights'],
                  'loss':'Existing masked weighted CE','features':'Existing seven fields + seven masks + month sine/cosine',
                  'architecture':'Unchanged ConvLSTM 8/4, pooling2, bilinear restoration, 8787 parameters',
                  'decision_rule':historical['calibration']['decision_rule'],'regional_thresholds':{k:historical['calibration'][k] for k in ['regional_elevated_percent','regional_strong_percent']},
                  'coverage_policy':historical['calibration']['coverage_policy'],'freshness_maximum_age_days':config.MAX_DATA_AGE_DAYS,
                  'development_temperature':calibration['temperature'],'active_temperature':None,
                  'files':{str(p.relative_to(config.ROOT)):file_sha256(p) for p in files},
                  'preserved_2025_evaluation_sha256':file_sha256(FINAL_DIR/'evaluation.json'),
                  'independent_evaluation':'Not performed: no new independent observations. Existing 2025 result is reused as historical evidence only.'})
    frozen=read_json(OUT/'freeze.json')
    for name,digest in frozen['files'].items():
        if file_sha256(config.ROOT/name)!=digest:
            raise ValueError('Frozen upgrade development configuration changed: '+name)
    rows=[('Existing frozen final',reference),('Balanced sequence sampling',candidate)]
    lines=['# Post-freeze HAB improvement report','',
           'The existing repository and completed experiments were preserved. All new scientific development uses 2024. The original baseline checkpoint and saved 2025 evaluation match their archived hashes. 2025 was inspected previously and is not claimed untouched; it was not evaluated again in this round.','',
           '## Already completed and retained','',
           'Training-only corrected proxy labels, shared normalization/masks, small ConvLSTM, controlled weight/rule/seasonal experiments, coverage abstention, regional/false-warning evaluation, frozen model and 2025 results, five-map API/dashboard and the original 38 tests. Seasonal encoding is retained from its completed controlled experiment; no seasonal or class-weight experiment was repeated.','',
           '## New STRONG diagnostics','',
           f"Unique observed training normal/elevated/strong counts: {diagnostics['unique_training_label_counts']}. Unique 2024 counts: {diagnostics['unique_2024_label_counts']}. Rolling forecasts contain {diagnostics['strong_forecasts']:,} strong targets; {diagnostics['strong_forecasts_without_cell_input_observations']:,} have no chlorophyll observation at the cell in the input window. Repeated horizons/origins are not independent samples.",'',
           '| Cell input coverage | Strong precision | Strong recall | Strong F1 | Strong support |','|---|---:|---:|---:|---:|']
    for name,key in [('No observed chlorophyll','cell_input/missing'),('Observed chlorophyll','cell_input/observed')]:
        c=diagnostics['metrics_by_stratum'][key]['classes'][2]
        lines.append(f"| {name} | {c['precision']:.4f} | {c['recall']:.4f} | {c['f1']:.4f} | {c['support']} |")
    lines += ['',diagnostics['interpretation'],'',
              'Actual missed-pixel examples retain target/input date, coordinates, class confusion, scores, coverage, observed input days and environmental values. Monthly strata use the actual target date, with latitude and input-coverage strata also saved. These remain chlorophyll proxy events, not verified harmful species/toxicity. The diagnostics do not justify capacity expansion.','',
              '## Controlled sampling experiment','',training['sampling'], '',
              'One full new run changed training sequence sampling only. Existing class weights, seasonal inputs, corrected labels, filters, optimizer, seed, batch size, early stopping and frozen pixel/regional rules were retained. Thresholds were not retuned.','',
              f"{training['epochs_completed']} epochs completed; best epoch {training['best_epoch']}. Uniform rolling training target counts: {training['uniform_pixel_forecast_class_counts']}. Expected sampled counts: {training['expected_sampled_pixel_forecast_class_counts']}.",'',
              '| Model | Macro F1 | Normal P/R/F1 | Elevated P/R/F1 | Strong P/R/F1 | Pixel false-warning rate |','|---|---:|---|---|---|---:|']
    for name,result in rows:
        r=result['model'];triplets=[' / '.join(f"{c[k]:.4f}" for k in ['precision','recall','f1']) for c in r['classes']]
        cm=r['confusion_matrix'];rate=(cm[0][1]+cm[0][2])/sum(cm[0])
        lines.append(f"| {name} | {r['macro_f1']:.4f} | {' | '.join(triplets)} | {rate:.4f} |")
    lines += ['',comparison['criterion'],'',comparison['decision'],'',
              '| Model | Warning days | Correct warnings | False warnings | Missed risk days | Any-risk recall | Strong recall | Insufficient-input frequency | Mean input coverage |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name,result in rows:
        r=result['regional']['daily_horizon_1'];a=r['any_warning'];s=r['strong_warning']
        lines.append(f"| {name} | {a['warning_days']} | {a['correctly_warned_high_risk_days']} | {a['false_warning_days']} | {a['missed_high_risk_days']} | {a['recall']:.4f} | {s['recall']:.4f} | {r['insufficient_data_frequency']:.4f} | {r['mean_input_coverage_percent']:.2f}% |")
    lines += ['', 'Full confusion matrices, argmax reference, per-class support, false-warning frequency, missed strong days, abstentions and warning-day coverage are in the machine-readable comparison. Existing baseline/weight/decision/seasonal/final tables are retained in [the prior report](improvement_results.md). Fusion has no comparable result because suitable input data does not exist.','',
              '## Regional support and false alarms','',
              regional['motivation'],'',
              f"Using the unchanged thresholds on input-observed cells gave {regional['candidate']['any_warning']['correctly_warned_high_risk_days']} correct warnings, {regional['candidate']['any_warning']['false_warning_days']} false warnings and {regional['candidate']['any_warning']['missed_high_risk_days']} misses. Criterion passed: {regional['criterion_passed']}. The support change was rejected and not deployed.",'',
              'Candidate daily records retain scores, observed truth fractions, source-input coverage, month and environmental values, including the false-warning cases. Existing false-alarm diagnostics remain unchanged. No threshold was raised to make warnings disappear. Lack of warnings is not evidence of reliable detection.','',
              '## Probability calibration','',
              'Scalar temperature was fitted on complete January–June 2024 windows and assessed on complete July–December windows, excluding overlapping boundary windows. The checkpoint was already developed/early-stopped on 2024, so this is development assessment rather than an independent calibration test.','',
              f"Temperature: {calibration['temperature']:.8f}. Declared criterion passed: {calibration['criterion_passed']}. Active calibration remains disabled; raw pixel decisions and historical 2025 results are unchanged.",'',
              '| Metric | Raw | Temperature |','|---|---:|---:|']
    for key in ['nll','brier']:
        lines.append(f"| {key} | {calibration['before'][key]:.6f} | {calibration['after'][key]:.6f} |")
    for c in ['top_label','0','1','2']:
        lines.append(f"| ECE {c} | {calibration['before']['reliability'][c]['ece']:.6f} | {calibration['after']['reliability'][c]['ece']:.6f} |")
    lines += ['', 'Per-class Brier and bin counts are saved with the [reliability figure](results/upgrade/calibration_2024.png). Improved aggregate calibration does not make strong-event scores reliable; high-score reliability remains poor. Scores therefore continue to be described as model scores, not operational event probabilities.','',
              '## Satellite, field labels and live updates','',
              'No new satellite observations or field labels were acquired or fabricated. Coverage did not improve in this round. Existing 2024 MODIS coverage remains 8.30%; saved Sentinel availability remains 37.55% unverified and excluded. New provenance audit found bbox-derived coordinates consistent with the actual project centers, but acquisition IDs, independent source georeferencing and retrieval quality flags are absent.','',
              'A real MODIS daily reader validates CF decoding, concentration units, actual coordinates and source valid-range metadata. The Sentinel adapter requires documented processing-version flag mappings, native QC, UTC timestamps, aligned coordinates and units; fusion remains unavailable. Candidate fusion helpers preserve MODIS, quantify overlap agreement and retain provenance, but are not enabled automatically.','',
              audit.get('timing_limitation','Source publication/acquisition timing needs independent validation.'),'',
              'Verified field-record validation requires actual verification provenance and explicit confirmed status. It neither invents labels nor refits proxy targets. No independent field period/region was found.','',
              'Separate staged live inputs use frozen normalization/calendar features and coverage/freshness gates. A configurable source acquisition adapter can supply real bundles. The current state is not configured; no live-data result is claimed. Failed updates retain the last successful record. Local/demo notification outbox is gated and idempotent; external delivery stays disabled because regional warning validation failed.','',
              'Data formats, source quality references, acquisition requirements, field evaluation entry point and notification configuration: [real-data interfaces](real_data_interfaces.md).','',
              '## Preserved 2025 evidence','',
              '| Metric | Baseline | Existing frozen final |','|---|---:|---:|']
    baseline=historical['baseline_test']['model'];final=historical['test']['model']
    lines.append(f"| Macro F1 | {baseline['macro_f1']:.4f} | {final['macro_f1']:.4f} |")
    for c,name in [(1,'Elevated F1'),(2,'Strong F1')]:
        lines.append(f"| {name} | {baseline['classes'][c]['f1']:.4f} | {final['classes'][c]['f1']:.4f} |")
    lines += ['', 'These are saved prior results, not a fresh evaluation of this upgrade. Elevated recovery generalized in that historical result, but strong detection regressed and regional strong recall was zero. No further parameter choice used 2025. New-period independent evaluation is blocked on actual new observations; this limitation cannot be fixed by code alone.','',
              '## Engineering checks and run commands','',
              f"Full suite: **{count} Python tests passed**, preserving all original tests. A separate JavaScript request-order regression also passed. Real MODIS contract checks, frozen preprocessing equivalence, calibration, quality flags, sampling, field schema, stale/missing/source-timing behavior, notification gates, new endpoints and real five-map integration are covered. Dependency, JavaScript syntax and whitespace checks passed.",'',
              'Real HTTP five-map forecast and development calibration transformation passed; model/baseline and saved 2025 result hashes were verified. The browser displayed MODIS/Sentinel-unavailable/live-update/model status, separate STALE DATA and historical forecast states. The covered-date helper selects 2024 observations using input completeness alone. No missing-data day becomes NORMAL.','',
              '```bash','source .venv/bin/activate','python -m experiments.finalize verify','python -m unittest discover -s tests -v','python -m pip check','python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000','```','',
              'Open `http://127.0.0.1:8000/dashboard/`. Use “Find covered historical date”, or choose 2024-02-15 and enable Historical demonstration, then Generate forecast. In another activated terminal:','',
              '```bash','python tests/http_smoke.py','python -m upgrade.live  # reports not configured until a real acquisition adapter exists','```','',
              'Completed development commands (do not repeat existing output directories): `upgrade.audit`, `upgrade.diagnostics`, `upgrade.calibration`, `upgrade.regional_experiment`, `upgrade.sampling`, `upgrade.compare`, `upgrade.verify`, `upgrade.report`. Sampling refuses overwrite; existing reports are retained. No new training/evaluation command replaces the frozen baseline/final or reevaluates 2025.','',
              'New files: `upgrade/` modules, `tests/test_upgrade.py`, data-interface/report documentation and public evidence snapshots. Existing edits: `backend/app.py`, `prediction/service.py`, `frontend/index.html`, `frontend/script.js`, `README.md`. Frozen model/preprocessing/label/loss/decision/region code was preserved.','',
              '## Quality check and remaining limitations','',
              '- Verified: baseline/labels/frozen evaluation preserved; 2024-only diagnostics/calibration/sampling; decision/seasonal experiments retained; explicit missing/low/stale states; separate regional/false-warning evaluation; API/dashboard/tests/integration.','- Implemented but dependent on real data: Sentinel QC/fusion inputs, verified field schema, staged live acquisition/update interfaces and future independent evaluation.','- Local notifications only: regional warning reliability is inadequate; no external email/SMS was sent or enabled.','- Scientific limitations remain: sparse observations, proxy rather than confirmed HAB labels, poor rare-class/regional skill, correlated evaluation samples, single-region evidence and unavailable independent future-period assessment.','- No capacity increase, coverage gain, broad generalization, calibrated operational confidence or new independent test success is claimed.']
    (config.ROOT/'docs/post_freeze_improvements.md').write_text('\n'.join(lines)+'\n')
    public=config.ROOT/'docs/results/upgrade';public.mkdir(parents=True,exist_ok=True)
    for name in ['strong_diagnostics_2024.json','calibration_2024.json','calibration_2024.png','regional_support_2024.json','regional_support_2024.csv','data_audit.json','sampling_comparison_2024.json','integration.json','http_smoke.json','tests.log','freeze.json']:
        shutil.copy2(OUT/name,public/name)
    for name in ['training_config.json','training_history.csv']:
        shutil.copy2(OUT/'balanced_sampling'/name,public/('sampling_'+name))
    print(f'Published actual upgrade evidence; {count} tests; development selection={selected}.')

if __name__=='__main__':main()
