"""Metrics and the same class decisions used by evaluation and the API."""
import numpy as np
import config


def classify(probabilities, threshold):
    result = np.argmax(probabilities[..., :2], axis=-1).astype(np.uint8)
    result[probabilities[..., 2] >= threshold] = 2
    return result


def confusion_matrix(actual, predicted, classes=3):
    return np.bincount(classes * np.asarray(actual, int).ravel() + np.asarray(predicted, int).ravel(),
                       minlength=classes * classes).reshape(classes, classes)


def classification_metrics(cm):
    cm = np.asarray(cm)
    precision = np.diag(cm) / np.maximum(cm.sum(axis=0), 1)
    recall = np.diag(cm) / np.maximum(cm.sum(axis=1), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return {"accuracy": float(np.trace(cm) / max(cm.sum(), 1)), "macro_f1": float(f1.mean()),
            "valid_pixel_forecasts": int(cm.sum()), "confusion_matrix": cm.tolist(),
            "classes": [{"name": name, "precision": float(precision[i]), "recall": float(recall[i]),
                         "f1": float(f1[i]), "support": int(cm[i].sum())} for i, name in enumerate(config.CLASS_NAMES)]}
