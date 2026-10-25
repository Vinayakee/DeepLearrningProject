"""Optional cyclical calendar features; no statistics fitted to held-out dates."""
import numpy as np
from preprocessing.data_utils import month_numbers


def seasonal_features(dates, height, width):
    month = month_numbers(dates).astype(np.float32)
    angle = 2 * np.pi * month / 12
    values = np.stack([np.sin(angle), np.cos(angle)], -1).astype(np.float32)
    return np.broadcast_to(values[:, None, None, :], (len(dates), height, width, 2)).copy()


def add_seasonal_features(X, dates):
    if X.shape[-4] != len(dates):
        raise ValueError('Season dates do not match input time steps.')
    features = seasonal_features(dates, X.shape[-3], X.shape[-2])
    if X.ndim == 5:
        features = np.broadcast_to(features[None], X.shape[:-1] + (2,))
    return np.concatenate([X, features], -1)
