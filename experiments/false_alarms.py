"""Describe false-warning associations without inferring causality."""
import json
import numpy as np
import pandas as pd
import config
from experiments.common import EXPERIMENT_DIR, read_json, write_json


def group_statistics(frame, column):
    rows = []
    for value, part in frame.groupby(column, observed=True):
        q = part[part.truth_qualified]
        actual = q.actual_level.isin(['ELEVATED RISK', 'STRONG BLOOM RISK'])
        issued = q.issued_level.isin(['ELEVATED RISK', 'STRONG BLOOM RISK'])
        rows.append({'group': str(value), 'forecast_days': len(part), 'qualified_days': len(q),
                     'warning_days': int(issued.sum()), 'false_warning_days': int((issued & ~actual).sum()),
                     'missed_warning_days': int((~issued & actual).sum()),
                     'false_warning_frequency': float((issued & ~actual).mean()) if len(q) else None,
                     'mean_input_coverage_percent': float(part.input_coverage_percent.mean()),
                     'unavailable_days': int((~part.available).sum())})
    return rows


def analyze(path):
    frame = pd.read_csv(path)
    frame = frame[frame.horizon == 1].copy()
    frame['coverage_bin'] = pd.cut(frame.input_coverage_percent, [-.01, 5., 10., 20., 100.], right=False)
    false = frame[frame.truth_qualified & (frame.actual_level == 'NORMAL') & frame.issued_level.isin(['ELEVATED RISK', 'STRONG BLOOM RISK'])]
    environment = {}
    for name in config.CHANNEL_NAMES:
        if frame[name].notna().any():
            environment[name] = {'mean_on_false_warning_days': float(false[name].mean()) if false[name].notna().any() else None,
                                 'mean_on_all_qualified_days': float(frame[frame.truth_qualified][name].mean()) if frame[frame.truth_qualified][name].notna().any() else None}
    cases = json.loads(false.drop(columns=['coverage_bin']).to_json(orient='records'))
    return {'false_warning_cases': cases, 'by_month': group_statistics(frame, 'month'),
            'by_input_coverage': group_statistics(frame, 'coverage_bin'), 'environment_associations': environment,
            'limitations': 'Descriptive associations only: labels are sampled chlorophyll anomaly proxies, observed coverage differs from dense forecast support, and reanalysis delivery latency is unknown. Model scores are not calibrated uncertainty. Individual cloud/glint/algorithm failure causes cannot be established from retained processed masks.'}


def main():
    comparisons = read_json(EXPERIMENT_DIR / 'regional_comparison_2024.json')
    results = {}
    lines = ['# 2024 false-warning analysis', '', '| Candidate | False-warning days | Qualified days | Warning precision | Warning recall | Missed strong days |', '|---|---:|---:|---:|---:|---:|']
    for key, report in comparisons.items():
        name, stage = key.split('/')
        analysis = analyze(EXPERIMENT_DIR / name / 'regional_2024' / stage / 'daily_cases.csv')
        results[key] = analysis
        r = report['daily_horizon_1']['any_warning']; strong = report['daily_horizon_1']['strong_warning']
        lines.append(f"| {key} | {r['false_warning_days']} | {r['qualified_days']} | {r['precision']:.4f} | {r['recall']:.4f} | {strong['missed_high_risk_days']} |")
    write_json(EXPERIMENT_DIR / 'false_alarm_analysis_2024.json', results)
    lines += ['', 'Exact false-warning dates, scores, input/target coverage and observed environmental means are in the JSON and each daily_cases.csv. Month/coverage associations do not establish a causal reason for individual false alarms. Unknown target days are excluded from truth scoring and reported separately. Abstentions on qualified high-risk days remain misses.']
    (EXPERIMENT_DIR / 'false_alarm_analysis_2024.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines), flush=True)

if __name__ == '__main__':
    main()
