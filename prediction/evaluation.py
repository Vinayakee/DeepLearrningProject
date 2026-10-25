"""Legacy baseline evaluation; frozen final runs cannot be retuned."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np
import config
from preprocessing.pipeline import EnvironmentalPipeline
from preprocessing.data_utils import file_sha256
from training.data_generator import HABDataGenerator
from prediction.metrics import classify, classification_metrics, confusion_matrix


def load_trained_model(path=config.MODEL_PATH):
    import tensorflow as tf
    from training import losses  # register serializable loss/metric
    path = Path(path)
    record = json.loads((config.RUN_DIR / "training_config.json").read_text())
    if record.get("smoke") or not record.get("epochs_completed"):
        raise ValueError("Full-data training has not completed.")
    if record["model_sha256"] != file_sha256(path):
        raise ValueError("Model differs from the completed training record.")
    if (record["preprocessing_sha256"] != file_sha256(config.SEQUENCE_DIR / "preprocessing.json")
            or record["label_metadata_sha256"] != file_sha256(config.LABEL_DIR / "label_metadata.json")):
        raise ValueError("Model, preprocessing, and corrected labels are from different runs.")
    stats = json.loads((config.SEQUENCE_DIR / "preprocessing.json").read_text())
    channels = 16 if stats.get("seasonal_features", False) else 14
    if record.get("channels", 14) != channels:
        raise ValueError("Model and preprocessing feature configurations differ.")
    if (config.RUN_DIR / "freeze.json").exists():
        from experiments.finalize import verify_final
        verify_final()
    model = tf.keras.models.load_model(path, compile=False)
    if model.input_shape[1:] != (5, 120, 84, channels) or model.output_shape[1:] != (5, 120, 84, 3):
        raise ValueError("Saved model has an incompatible input/output contract.")
    return model, record


def calibrate(model, pipeline):
    generator = HABDataGenerator("validation", shuffle=False, pipeline=pipeline)
    thresholds = np.linspace(.1, .9, 33)
    counts = np.zeros((len(thresholds), 3), dtype=np.int64)
    for i in range(len(generator)):
        X, y = generator[i]
        probabilities = model.predict_on_batch(X)
        valid = y.sum(-1) > 0
        actual = np.argmax(y, -1)[valid] == 2
        scores = probabilities[..., 2][valid]
        for i, threshold in enumerate(thresholds):
            predicted = scores >= threshold
            counts[i] += [int((actual & predicted).sum()), int((~actual & predicted).sum()), int((actual & ~predicted).sum())]
    tp, fp, fn = counts.T
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    best = int(np.argmax(f1))
    fractions = []
    selected = (pipeline.dates >= np.datetime64(config.TRAIN_START)) & (pipeline.dates <= np.datetime64(config.TRAIN_END))
    for day in pipeline.labels[selected]:
        valid = day != 255
        # Avoid estimating regional fractions from a handful of cloudy pixels.
        if valid.sum() >= .05 * pipeline.ocean.sum():
            fractions.append(float((day == 2).sum() / valid.sum() * 100))
    if not fractions:
        raise ValueError("Not enough training observations to derive regional warnings.")
    elevated, strong = np.percentile(fractions, [75, 90])
    if strong <= elevated or elevated <= 0:
        raise ValueError("Training data do not support distinct regional warning thresholds.")
    return {"strong_probability_threshold": float(thresholds[best]), "threshold_fit_split": "validation (2024)",
            "threshold_selection": "Maximum strong-class binary F1 on validation; first/lower threshold wins ties.",
            "threshold_sweep": [{"threshold": float(t), "f1": float(score), "tp_fp_fn": c.tolist()} for t, score, c in zip(thresholds, f1, counts)],
            "regional_elevated_percent": float(elevated), "regional_strong_percent": float(strong),
            "regional_fit_days": len(fractions), "regional_fit_split": "train (2021-2023)",
            "regional_method": "P75/P90 of daily observed strong-label fractions on training days with >=5% support coverage. Demonstration heuristic; not operationally validated."}


def persistence_labels(pipeline, start):
    previous = np.asarray(pipeline.labels[start:start + 5])
    latest = np.zeros(previous.shape[1:], dtype=np.uint8)  # training majority fallback
    for day in previous:
        valid = day != 255
        latest[valid] = day[valid]
    return np.repeat(latest[None, None], 5, axis=1)


def evaluate(model, pipeline, split, threshold):
    print(f"Evaluating {split} with fixed validation threshold...", flush=True)
    generator = HABDataGenerator(split, shuffle=False, pipeline=pipeline)
    cms = {name: np.zeros((3, 3), np.int64) for name in ("model", "argmax", "persistence", "majority")}
    horizons = np.zeros((5, 3, 3), np.int64)
    squared_error, total = 0., 0
    bins = np.zeros((10, 3), dtype=float)
    for i in range(len(generator)):
        X, y = generator[i]
        p = model.predict_on_batch(X)
        valid = y.sum(-1) > 0
        actual = np.argmax(y, -1)
        decisions = {"model": classify(p, threshold), "argmax": np.argmax(p, -1),
                     "persistence": persistence_labels(pipeline, int(generator.indices.iloc[i].input_start_idx)),
                     "majority": np.zeros(actual.shape, dtype=np.uint8)}
        for name, predicted in decisions.items():
            cms[name] += confusion_matrix(actual[valid], predicted[valid])
        for day in range(5):
            v = valid[:, day]
            horizons[day] += confusion_matrix(actual[:, day][v], decisions["model"][:, day][v])
        scores, truth = p[..., 2][valid], (actual[valid] == 2).astype(float)
        squared_error += float(np.square(scores - truth).sum())
        total += len(scores)
        for b in range(10):
            keep = np.minimum((scores * 10).astype(int), 9) == b
            bins[b] += [int(keep.sum()), float(scores[keep].sum()), float(truth[keep].sum())]
    result = {name: classification_metrics(cm) for name, cm in cms.items()}
    result.update({"split": split, "sequences": len(generator), "horizons": [classification_metrics(cm) for cm in horizons],
                   "strong_probability_brier_score": squared_error / max(total, 1),
                   "reliability_bins": [{"count": int(c), "mean_score": float(p / c) if c else None,
                                         "observed_frequency": float(t / c) if c else None} for c, p, t in bins],
                   "scoring_note": "Pixel-forecast metrics over rolling windows; overlapping target dates are scored at distinct origins/horizons, not independent observations. Probabilities are model scores, not guaranteed calibrated."})
    return result


def main():
    if (config.RUN_DIR / "freeze.json").exists():
        raise ValueError("Final run is frozen. Use python -m experiments.finalize verify; no retuning or repeat test evaluation.")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print("Loading completed checkpoint and shared preprocessing...", flush=True)
    model, training = load_trained_model()
    pipeline = EnvironmentalPipeline()
    print("Selecting threshold on 2024 validation only...", flush=True)
    calibration = calibrate(model, pipeline)
    calibration["model_sha256"] = training["model_sha256"]
    calibration["preprocessing_sha256"] = training["preprocessing_sha256"]
    calibration["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    config.RUN_DIR.mkdir(parents=True, exist_ok=True)
    (config.RUN_DIR / "calibration.json").write_text(json.dumps(calibration, indent=2) + "\n")
    results = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "model_sha256": training["model_sha256"],
               "calibration": calibration,
               "validation": evaluate(model, pipeline, "validation", calibration["strong_probability_threshold"]),
               "test": evaluate(model, pipeline, "test", calibration["strong_probability_threshold"])}
    (config.RUN_DIR / "evaluation.json").write_text(json.dumps(results, indent=2) + "\n")
    lines = ["# Corrected ConvLSTM evaluation", "", "Results generated from held-out predictions; threshold chosen on 2024 only.", "",
             "| 2025 method | Accuracy | Macro F1 | Strong precision | Strong recall | Strong F1 |", "|---|---:|---:|---:|---:|---:|"]
    for name in ("model", "argmax", "persistence", "majority"):
        r = results["test"][name]
        c = r["classes"][2]
        lines.append(f"| {name} | {r['accuracy']:.4f} | {r['macro_f1']:.4f} | {c['precision']:.4f} | {c['recall']:.4f} | {c['f1']:.4f} |")
    lines += ["", results["test"]["scoring_note"], "", f"Strong-probability Brier score: {results['test']['strong_probability_brier_score']:.4f}", "",
              "Legacy results used a different label definition/support and preprocessing. They are not a controlled comparison with this corrected test set."]
    (config.RUN_DIR / "evaluation.md").write_text("\n".join(lines) + "\n")
    from prediction.plots import plot_results
    plot_results()
    print(json.dumps({name: results["test"][name] for name in ("model", "persistence", "majority")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
