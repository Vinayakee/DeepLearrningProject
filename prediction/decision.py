"""Explicit class decisions; adjusted scores remain distinct from probabilities."""
import numpy as np


def validate_rule(rule):
    kind = rule.get('type')
    if kind not in ('argmax', 'strong_override', 'score_ratios'):
        raise ValueError('Unknown decision rule.')
    if kind == 'strong_override':
        threshold = rule.get('strong_threshold')
        if threshold is None or not np.isfinite(threshold) or not 0 < threshold < 1:
            raise ValueError('Strong threshold must lie strictly between zero and one.')
    if kind == 'score_ratios':
        scales = np.asarray(rule.get('scales', []), dtype=float)
        if scales.shape != (3,) or not np.isfinite(scales).all() or np.any(scales <= 0) or scales[0] != 1:
            raise ValueError('Three positive decision scales with normal scale 1 are required.')
    return rule


def apply_rule(probabilities, rule):
    validate_rule(rule)
    if probabilities.shape[-1] != 3:
        raise ValueError('Three class scores are required.')
    if rule['type'] == 'argmax':
        return probabilities.argmax(-1).astype(np.uint8)
    if rule['type'] == 'strong_override':
        result = probabilities[..., :2].argmax(-1).astype(np.uint8)
        result[probabilities[..., 2] >= rule['strong_threshold']] = 2
        return result
    # This adjusts decision boundaries, not the displayed/calibrated probabilities.
    return (probabilities * np.asarray(rule['scales'], dtype=np.float32)).argmax(-1).astype(np.uint8)
