"""Dated, reproducible forecasts and honest historical demonstration mode."""
from datetime import date, datetime, timezone
import json
import os
from pathlib import Path
import uuid
import numpy as np
import config
from preprocessing.pipeline import EnvironmentalPipeline
from preprocessing.data_utils import file_sha256, month_numbers, environment_fingerprint
from prediction.evaluation import load_trained_model
from prediction.decision import apply_rule
from prediction.coverage import assess_coverage, DEFAULT_POLICY
from prediction.regional import warning_level, regional_summary


def data_freshness(input_end_date, now=None):
    today = now or datetime.now(timezone.utc).date()
    end = date.fromisoformat(str(input_end_date))
    age = (today - end).days
    return {"input_end_date": str(end), "checked_at_utc": datetime.now(timezone.utc).isoformat(),
            "age_days": age, "maximum_age_days": config.MAX_DATA_AGE_DAYS,
            "state": "future" if age < 0 else "stale" if age > config.MAX_DATA_AGE_DAYS else "fresh"}


def summarize_forecast(probabilities, support, calibration):
    if probabilities.ndim != 4 or probabilities.shape[-1] != 3 or support.shape != probabilities.shape[:-1]:
        raise ValueError("Forecast probabilities and support shapes differ.")
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError("Forecast contains invalid probability values.")
    if not np.allclose(probabilities.sum(-1), 1., atol=1e-5):
        raise ValueError("Forecast probabilities do not sum to one.")
    rule = calibration.get("decision_rule") or {"type": "strong_override", "strong_threshold": calibration["strong_probability_threshold"]}
    risk = apply_rule(probabilities, rule)
    risk[~support] = 255
    days = []
    for day, mask in enumerate(support):
        count = int(mask.sum())
        strong = int((risk[day] == 2).sum())
        percent = strong / count * 100 if count else 0.
        p = probabilities[day, ..., 2][mask]
        days.append({**regional_summary(risk[day], mask, calibration), "day": day + 1,
                     "max_elevated_probability": float(probabilities[day, ..., 1][mask].max()) if count else None,
                     "mean_elevated_probability": float(probabilities[day, ..., 1][mask].mean()) if count else None, "max_probability": float(p.max()) if count else None,
                     "mean_probability": float(p.mean()) if count else None,
                     "high_risk_pixels": strong, "high_risk_percentage": float(percent),
                     "elevated_risk_pixels": int((risk[day] == 1).sum()), "supported_pixels": count,
                     "warning_level": warning_level(percent, calibration) if count else "UNAVAILABLE"})
    return risk, days


def render_maps(probabilities, risk, latitude, longitude, forecast_dates, directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    cmap = ListedColormap(["#4caf50", "#f2c94c", "#e74c3c"]).with_extremes(bad="#e5e7eb")
    norm = BoundaryNorm([-.5, .5, 1.5, 2.5], 3)
    for i, day in enumerate(forecast_dates):
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        masked_risk = np.ma.masked_where(risk[i] == 255, risk[i])
        masked_probability = np.ma.masked_where(risk[i] == 255, probabilities[i, ..., 2])
        a = axes[0].pcolormesh(longitude, latitude, masked_risk, cmap=cmap, norm=norm, shading="nearest")
        b = axes[1].pcolormesh(longitude, latitude, masked_probability, vmin=0, vmax=1, cmap="viridis", shading="nearest")
        axes[0].set_title("Risk class (shared decision rule)")
        axes[1].set_title("Strong-class model score")
        c = fig.colorbar(a, ax=axes[0], ticks=[0, 1, 2], shrink=.8)
        c.ax.set_yticklabels(["Normal", "Elevated", "Strong"])
        fig.colorbar(b, ax=axes[1], shrink=.8)
        for ax in axes:
            ax.set(xlabel="Longitude (°E)", ylabel="Latitude (°N)", facecolor="#e5e7eb")
        fig.suptitle(f"HAB proxy forecast · Day +{i + 1} · {day}\nGrey: outside supported label domain")
        fig.tight_layout()
        fig.savefig(Path(directory) / f"day_{i + 1}.png", dpi=120)
        plt.close(fig)


def validate_current_contract(training):
    """Detect regenerated fitting artifacts before serving an older run."""
    if (training["label_metadata_sha256"] != file_sha256(config.LABEL_DIR / "label_metadata.json")
            or training["preprocessing_sha256"] != file_sha256(config.SEQUENCE_DIR / "preprocessing.json")):
        raise ValueError("Fitted labels/preprocessing changed; complete retraining and evaluation.")
    labels = json.loads((config.LABEL_DIR / "label_metadata.json").read_text())
    if labels["climatology_metadata_sha256"] != file_sha256(config.LABEL_DIR / "climatology_metadata.json"):
        raise ValueError("Climatology changed; regenerate labels, retrain and evaluate.")


class ForecastService:
    def __init__(self):
        self.pipeline = EnvironmentalPipeline()
        self.model, self.training = load_trained_model()
        self.calibration = json.loads((config.RUN_DIR / "calibration.json").read_text())
        self.calibration_identity = file_sha256(config.RUN_DIR / "calibration.json")
        if (self.calibration["model_sha256"] != self.training["model_sha256"]
                or self.calibration["preprocessing_sha256"] != self.training["preprocessing_sha256"]):
            raise ValueError("Calibration differs from the trained model; rerun evaluation.")

    def generate(self, input_end_date=None, historical_demo=False):
        validate_current_contract(self.training)
        if self.calibration_identity != file_sha256(config.RUN_DIR / "calibration.json"):
            raise ValueError("Calibration changed; restart the API to load updated thresholds.")
        if self.pipeline.environment_identity != environment_fingerprint(self.pipeline.processed_dir):
            raise ValueError("Environmental inputs changed. Regenerate preprocessing, retrain and restart the API.")
        end = str(input_end_date or self.pipeline.dates[-1])
        end_day = np.datetime64(end, "D")
        where = np.flatnonzero(self.pipeline.dates == end_day)
        if not len(where) or where[0] < 4:
            raise ValueError("Choose an available date with five preceding daily observations.")
        freshness = data_freshness(end)
        if freshness["state"] != "fresh" and not historical_demo:
            raise ValueError("Input data are outdated or future-dated; request historical_demo explicitly for a demonstration.")
        indices = np.arange(where[0] - 4, where[0] + 1)
        X = self.pipeline.transform(indices)
        input_dates = [str(d) for d in self.pipeline.dates[indices]]
        forecast_dates = [str(end_day + np.timedelta64(i, "D")) for i in range(1, 6)]
        support = self.pipeline.ocean[None] & np.isfinite(self.pipeline.climatology[month_numbers(forecast_dates) - 1])
        if not support.any():
            raise ValueError("No supported forecast cells for this season.")
        quality = assess_coverage(X, self.pipeline.ocean, input_dates, self.calibration.get("coverage_policy", DEFAULT_POLICY))
        if quality["prediction_availability"] == "unavailable" or (quality["state"] == "low_coverage" and not historical_demo):
            return self.unavailable_bundle(indices[-1], input_dates, forecast_dates, quality, freshness, historical_demo)
        p = self.model(X[None], training=False).numpy()[0]
        risk, days = summarize_forecast(p, support, self.calibration)
        if quality["state"] != "observed":
            for row in days:
                row["exploratory_warning_level"] = row["warning_level"]
                row["warning_level"] = "INSUFFICIENT DATA"
        forecast_id = uuid.uuid4().hex
        directory = config.FORECAST_DIR / forecast_id
        directory.mkdir(parents=True, exist_ok=False)
        np.save(directory / "prediction.npy", p)
        np.save(directory / "risk_classes.npy", risk)
        np.save(directory / "support.npy", support)
        render_maps(p, risk, self.pipeline.latitude, self.pipeline.longitude, forecast_dates, directory)
        current_observed_risk, environment = observed_summary(self.pipeline, indices[-1], self.calibration)
        bundle = {"status": "success", "forecast_id": forecast_id,
                  "generated_at_utc": datetime.now(timezone.utc).isoformat(), "historical_demo": bool(historical_demo),
                  "model_sha256": self.training["model_sha256"], "preprocessing_sha256": self.training["preprocessing_sha256"],
                  "calibration_sha256": file_sha256(config.RUN_DIR / "calibration.json"),
                  "days": 5, "height": config.HEIGHT, "width": config.WIDTH,
                  "threshold": self.calibration.get("strong_probability_threshold"),
                  "decision_rule": self.calibration.get("decision_rule") or {"type": "strong_override", "strong_threshold": self.calibration["strong_probability_threshold"]},
                  "prediction_availability": quality["prediction_availability"],
                  "operational_warning_validated": self.calibration.get("operational_warning_validated", False),
                  "regional_validation_status": self.calibration.get("regional_validation_status", "Research prototype; operational warning skill not established"),
                  "warning_parameters": {k: self.calibration[k] for k in ("regional_elevated_percent", "regional_strong_percent", "regional_method")},
                  "input_dates": input_dates, "forecast_dates": forecast_dates, "daily_predictions": days,
                  "freshness": freshness, "data_quality": quality, "environment": environment,
                  "current_observed_risk": current_observed_risk,
                  "maps": [f"/forecasts/{forecast_id}/day_{i}.png" for i in range(1, 6)],
                  "model_description": {"environmental_channels": 7, "availability_channels": 7, "input_days": 5,
                                        "output_days": 5, "seasonal_channels": 2 if self.training.get("seasonal_features") else 0, "classes": 3, "parameter_count": self.model.count_params()},
                  "limitations": "Chlorophyll anomaly proxy, not confirmed toxicity/HAB. Model scores are not guaranteed calibrated. No automatic external data refresh or SMS/push delivery.",
                  "array_sha256": {name: file_sha256(directory / name) for name in ("prediction.npy", "risk_classes.npy", "support.npy")}}
        store_bundle(bundle, directory)
        return decorate_bundle(bundle, allow_demo=historical_demo)

    def unavailable_bundle(self, index, input_dates, forecast_dates, quality, freshness, historical_demo):
        observed, environment = observed_summary(self.pipeline, index, self.calibration)
        forecast_id = uuid.uuid4().hex
        directory = config.FORECAST_DIR / forecast_id
        directory.mkdir(parents=True, exist_ok=False)
        bundle = {"status": "insufficient_data", "forecast_id": forecast_id,
                  "message": "Insufficient data for reliable prediction", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                  "historical_demo": bool(historical_demo), "input_dates": input_dates, "forecast_dates": forecast_dates,
                  "model_sha256": self.training["model_sha256"], "preprocessing_sha256": self.training["preprocessing_sha256"],
                  "calibration_sha256": self.calibration_identity, "data_quality": quality, "freshness": freshness,
                  "prediction_availability": "unavailable", "daily_predictions": [], "maps": [], "array_sha256": {},
                  "current_observed_risk": observed, "environment": environment,
                  "operational_warnings_available": False, "alerts": []}
        store_bundle(bundle, directory)
        return decorate_bundle(bundle, allow_demo=historical_demo)

def observed_summary(pipeline, index, calibration):
    environment = []
    observed_labels = np.asarray(pipeline.labels[index])
    observed_valid = observed_labels != 255
    observed_count = int(observed_valid.sum())
    observed_coverage = observed_count / pipeline.ocean.sum() * 100
    observed_percent = float((observed_labels == 2).sum() / observed_count * 100) if observed_count else 0.
    current_observed_risk = {"valid_pixels": observed_count, "coverage_percent": float(observed_coverage),
                 "strong_fraction_percent": observed_percent,
                 "warning_level": warning_level(observed_percent, calibration) if observed_coverage >= 5 else "INSUFFICIENT OBSERVATIONS"}
    units = ("mg/m³", "°C", "m/s", "m/s", "mm/day", "m/s", "m/s")
    for name, unit in zip(config.CHANNEL_NAMES, units):
        values, valid = pipeline.valid(name, np.array([index]))
        observed = values[valid]
        environment.append({"name": name, "unit": unit, "observed_mean": float(observed.mean()) if len(observed) else None,
                "coverage_percent": float(valid.sum() / pipeline.ocean.sum() * 100)})
    return current_observed_risk, environment

def store_bundle(bundle, directory):
    (directory / "bundle.json").write_text(json.dumps(bundle, indent=2, allow_nan=False) + "\n")
    pointer = config.FORECAST_DIR / ("latest_" + bundle["forecast_id"] + ".tmp")
    pointer.write_text(json.dumps({"forecast_id": bundle["forecast_id"]}) + "\n")
    os.replace(pointer, config.FORECAST_DIR / "latest.json")


def decorate_bundle(bundle, allow_demo=False):
    # Recheck age on every request, not only when the forecast was generated.
    bundle = dict(bundle)
    freshness = data_freshness(bundle["input_dates"][-1])
    bundle["freshness"] = freshness
    bundle["satellite_source"] = "MODIS"
    bundle["calibrated_probability_available"] = False
    bundle["model_version"] = "frozen_final" if config.DEPLOYMENT else "baseline"
    bundle["warning_reason"] = (
        "Input coverage/required channels insufficient; risk is unknown."
        if bundle["data_quality"]["state"] != "observed" else
        "Stale input observations; historical research predictions only."
        if freshness["state"] != "fresh" else
        "Regional warning reliability is not validated; research predictions only."
        if not bundle.get("operational_warning_validated", True) else
        "Coverage/freshness requirements met; frozen regional decision rule applied."
    )
    if bundle.get("daily_predictions"):
        bundle["daily_predictions"] = [
            {**row, "max_strong_model_score": row.get("max_probability"),
             "max_elevated_model_score": row.get("max_elevated_probability"),
             "calibrated_probability": None, "warning_reason": bundle["warning_reason"]}
            for row in bundle["daily_predictions"]
        ]
    if bundle.get("status", "success") != "success":
        bundle["operational_warnings_available"] = False
        bundle["alerts"] = []
        return bundle
    unavailable = freshness["state"] != "fresh" or bundle["data_quality"]["state"] != "observed" or not bundle.get("operational_warning_validated", True)
    operational = not unavailable and not bundle["historical_demo"]
    bundle["operational_warnings_available"] = operational
    if unavailable or bundle["historical_demo"]:
        bundle["message"] = "Insufficient data for reliable prediction. Exploratory historical maps only." if bundle["data_quality"]["state"] != "observed" else "Research prototype. Regional warnings have not passed operational validation; these are not current coastal warnings."
        if not allow_demo:
            return {"status": "stale" if freshness["state"] == "stale" else "unavailable", "message": bundle["message"],
                    "freshness": freshness, "data_quality": bundle["data_quality"], "generated_at_utc": bundle["generated_at_utc"],
                    "satellite_source": bundle["satellite_source"], "warning_reason": bundle["warning_reason"],
                    "model_version": bundle["model_version"], "calibrated_probability_available": False}
    bundle["alerts"] = [{"day": row["day"], "date": bundle["forecast_dates"][row["day"] - 1],
                         "level": row["warning_level"], "delivery": "dashboard demonstration" if not operational else "dashboard",
                         "operational": operational} for row in bundle["daily_predictions"] if row["warning_level"] in ("ELEVATED RISK", "STRONG BLOOM RISK")]
    return bundle


def cached_forecast(allow_demo=False):
    pointer = json.loads((config.FORECAST_DIR / "latest.json").read_text())
    forecast_id = pointer["forecast_id"]
    if len(forecast_id) != 32 or any(c not in "0123456789abcdef" for c in forecast_id):
        raise ValueError("Invalid forecast bundle identifier.")
    directory = config.FORECAST_DIR / forecast_id
    bundle = json.loads((directory / "bundle.json").read_text())
    training = json.loads((config.RUN_DIR / "training_config.json").read_text())
    validate_current_contract(training)
    if (not training.get("epochs_completed") or training.get("smoke")
            or training["model_sha256"] != bundle["model_sha256"]
            or bundle["model_sha256"] != file_sha256(config.MODEL_PATH)):
        raise ValueError("Cached forecast does not match a completed model run.")
    if bundle["preprocessing_sha256"] != file_sha256(config.SEQUENCE_DIR / "preprocessing.json"):
        raise ValueError("Cached forecast preprocessing has changed.")
    manifest = json.loads((config.SEQUENCE_DIR / "manifest.json").read_text())
    if manifest["environment_fingerprint"] != environment_fingerprint(config.PROCESSED_DIR):
        raise ValueError("Environmental inputs changed; regenerate preprocessing and retrain.")
    if bundle["calibration_sha256"] != file_sha256(config.RUN_DIR / "calibration.json"):
        raise ValueError("Forecast decision thresholds changed; regenerate forecast.")
    for name, digest in bundle["array_sha256"].items():
        if file_sha256(directory / name) != digest:
            raise ValueError("Cached forecast arrays have changed; regenerate forecast.")
    return decorate_bundle(bundle, allow_demo)
