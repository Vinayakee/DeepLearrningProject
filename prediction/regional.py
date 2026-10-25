"""Regional aggregation and warning metrics, separate from pixel classification."""
import numpy as np


def warning_level(percent, calibration):
    if percent >= calibration['regional_strong_percent']:
        return 'STRONG BLOOM RISK'
    if percent >= calibration['regional_elevated_percent']:
        return 'ELEVATED RISK'
    return 'NORMAL'


def regional_summary(risk, support, calibration):
    risk, support = np.asarray(risk), np.asarray(support, bool)
    if risk.shape != support.shape:
        raise ValueError('Risk and regional support shapes differ.')
    count = int(support.sum())
    strong = int(((risk == 2) & support).sum())
    elevated = int(((risk == 1) & support).sum())
    percent = strong / count * 100 if count else None
    return {'supported_pixels': count, 'high_risk_pixels': strong, 'elevated_risk_pixels': elevated,
            'high_risk_percentage': percent,
            'warning_level': warning_level(percent, calibration) if count else 'UNAVAILABLE'}


def binary_warning_metrics(actual, predicted):
    actual, predicted = np.asarray(actual, bool), np.asarray(predicted, bool)
    if actual.shape != predicted.shape:
        raise ValueError('Regional actual/predicted lengths differ.')
    tp = int((actual & predicted).sum()); fp = int((~actual & predicted).sum()); fn = int((actual & ~predicted).sum())
    tn = int((~actual & ~predicted).sum())
    precision = tp / (tp + fp) if tp + fp else 0.
    recall = tp / (tp + fn) if tp + fn else 0.
    return {'qualified_days': len(actual), 'warning_days': int(predicted.sum()), 'actual_high_risk_days': int(actual.sum()),
            'correctly_warned_high_risk_days': tp, 'missed_high_risk_days': fn, 'false_warning_days': fp, 'true_negative_days': tn,
            'false_warning_frequency': fp / len(actual) if len(actual) else None,
            'false_warning_rate_on_nonrisk_days': fp / (fp + tn) if fp + tn else None,
            'precision': precision, 'recall': recall, 'detection_rate': recall,
            'f1': 2 * precision * recall / (precision + recall) if precision + recall else 0.}
