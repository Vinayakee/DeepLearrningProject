"""Transparent prototype input safeguards, not a calibrated confidence estimate."""
import math
import numpy as np
import config

DEFAULT_POLICY = {'minimum_chlorophyll_mean_percent': 5., 'minimum_chlorophyll_observed_days': 2,
                  'minimum_environment_observed_days': 1, 'minimum_target_coverage_percent': 5.}


def assess_coverage(X, ocean, input_dates, policy=None):
    policy = DEFAULT_POLICY if policy is None else policy
    X, ocean = np.asarray(X), np.asarray(ocean, bool)
    if X.ndim != 4 or X.shape[-1] not in (14, 16) or X.shape[:3] != (len(input_dates), *ocean.shape) or not ocean.any():
        raise ValueError('Input/marine support shapes are incompatible.')
    masks = X[..., 7:14]
    if not np.isfinite(masks).all() or np.any((masks != 0) & (masks != 1)):
        raise ValueError('Availability channels must be binary.')
    counts = (masks * ocean[None, ..., None]).sum((1, 2)).astype(int)
    support_count = int(ocean.sum())
    features = {}
    for i, name in enumerate(config.CHANNEL_NAMES):
        observed = np.flatnonzero(counts[:, i])
        required_days = policy['minimum_chlorophyll_observed_days'] if i == 0 else policy['minimum_environment_observed_days']
        features[name] = {'daily_valid_observations': counts[:, i].tolist(),
                          'valid_observations': int(counts[:, i].sum()), 'observed_input_days': len(observed),
                          'required_minimum_observed_days': required_days,
                          'daily_coverage_percent': (counts[:, i] / support_count * 100).tolist(),
                          'most_recent_observed_input_date': str(input_dates[observed[-1]]) if len(observed) else None}
    chl = features['chlorophyll']
    minimum = math.ceil(len(input_dates) * support_count * policy['minimum_chlorophyll_mean_percent'] / 100)
    missing = [name for name, value in features.items() if not value['observed_input_days']]
    insufficient = [name for name, value in features.items() if value['observed_input_days'] < value['required_minimum_observed_days']]
    mean_percent = chl['valid_observations'] / (len(input_dates) * support_count) * 100
    state = 'missing_data' if chl['valid_observations'] == 0 else 'insufficient_inputs' if insufficient else 'low_coverage' if chl['valid_observations'] < minimum else 'observed'
    display = {'missing_data': 'MISSING DATA', 'insufficient_inputs': 'INSUFFICIENT DATA',
               'low_coverage': 'LOW COVERAGE', 'observed': 'VALID OBSERVATION'}[state]
    return {'state': state, 'observation_status': display,
            'prediction_availability': 'supported_prototype' if state == 'observed' else 'exploratory_only' if state == 'low_coverage' else 'unavailable',
            'features': features, 'missing_input_features': missing, 'insufficient_input_features': insufficient,
            'chlorophyll_daily_coverage_percent': chl['daily_coverage_percent'], 'chlorophyll_mean_input_coverage_percent': float(mean_percent),
            'valid_observations': chl['valid_observations'], 'required_minimum_observations': minimum,
            'marine_supported_pixels': support_count, 'minimum_demo_quality_percent': policy['minimum_chlorophyll_mean_percent'],
            'policy': dict(policy), 'note': 'Valid observations count cell-days, not independent samples. 5% coverage and two observed chlorophyll days are explicit prototype safeguards; their regional tradeoffs are reported, not claimed as validated reliability/confidence.'}
