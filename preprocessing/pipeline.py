"""One transformation for training, validation, testing, and inference."""
import json
from pathlib import Path
import numpy as np

import config
from preprocessing import data_utils as data


def build_sequence_index(dates, labels):
    import pandas as pd
    rows = []
    for start in range(len(dates) - config.INPUT_DAYS - config.OUTPUT_DAYS + 1):
        end = start + config.INPUT_DAYS + config.OUTPUT_DAYS - 1
        split = next((name for name, (first, last) in config.SPLITS.items()
                      if dates[start] >= np.datetime64(first) and dates[end] <= np.datetime64(last)), None)
        if split is None:
            continue
        target = start + config.INPUT_DAYS
        count = int((labels[target:end + 1] != 255).sum())
        if count:
            rows.append({"split": split, "input_start_idx": start, "input_end_idx": target - 1,
                         "target_start_idx": target, "target_end_idx": end,
                         "input_start": str(dates[start]), "input_end": str(dates[target - 1]),
                         "target_start": str(dates[target]), "target_end": str(dates[end]),
                         "target_valid_pixels": count})
    return pd.DataFrame(rows)


def value_transform(values, name):
    return np.log1p(np.maximum(values, 0)) if name in ("chlorophyll", "rainfall") else values


class EnvironmentalPipeline:
    def __init__(self, processed_dir=config.PROCESSED_DIR, label_dir=config.LABEL_DIR,
                 sequence_dir=config.SEQUENCE_DIR, require_stats=True):
        self.processed_dir, self.label_dir, self.sequence_dir = map(Path, (processed_dir, label_dir, sequence_dir))
        self.dates = data.load_dates(self.processed_dir)
        self.latitude, self.longitude, self.shape = data.load_grid(self.processed_dir, self.dates)
        if self.shape[1:] != (config.HEIGHT, config.WIDTH):
            raise ValueError("The model requires the existing 120 x 84 grid.")
        self.values, self.masks = {}, {}
        for name in config.CHANNEL_NAMES:
            self.values[name], self.masks[name] = data.load_pair(self.processed_dir, name, self.shape)
        self.labels = np.load(self.label_dir / "hab_labels_2021_2025.npy", mmap_mode="r", allow_pickle=False)
        self.ocean = np.load(self.label_dir / "stable_ocean_mask.npy", allow_pickle=False).astype(bool)
        self.climatology = np.load(self.label_dir / "monthly_chl_climatology_2021_2025.npy", allow_pickle=False)
        self.metadata = json.loads((self.label_dir / "label_metadata.json").read_text())
        climatology_metadata = json.loads((self.label_dir / "climatology_metadata.json").read_text())
        if (self.metadata["train_start"], self.metadata["train_end"]) != (config.TRAIN_START, config.TRAIN_END):
            raise ValueError("Labels were not fitted on the required training years.")
        if climatology_metadata["source_fingerprint"] != data.source_fingerprint(self.processed_dir):
            raise ValueError("Processed data changed since label fitting; regenerate corrected labels.")
        if self.metadata["climatology_metadata_sha256"] != data.file_sha256(self.label_dir / "climatology_metadata.json"):
            raise ValueError("Labels and climatology come from different runs; regenerate labels.")
        for name, digest in {**climatology_metadata["array_sha256"], **self.metadata["array_sha256"]}.items():
            if data.file_sha256(self.label_dir / name) != digest:
                raise ValueError(f"Corrected artifact changed: {name}")
        if self.labels.shape != self.shape or self.ocean.shape != self.shape[1:]:
            raise ValueError("Corrected labels/support do not match observations.")
        if not np.isin(self.labels, [0, 1, 2, 255]).all() or np.any(self.labels[:, ~self.ocean] != 255):
            raise ValueError("Invalid classes or labels outside the support mask.")
        self.stats = None
        self.environment_identity = data.environment_fingerprint(self.processed_dir)
        if require_stats:
            self.stats = json.loads((self.sequence_dir / "preprocessing.json").read_text())
            if (self.stats["channel_order"] != list(config.CHANNEL_NAMES)
                    or not self.stats["include_observation_masks"]
                    or (self.stats["train_start"], self.stats["train_end"]) != (config.TRAIN_START, config.TRAIN_END)):
                raise ValueError("Preprocessing channel order, masks or fitting years differ.")
            manifest = json.loads((self.sequence_dir / "manifest.json").read_text())
            if manifest["environment_fingerprint"] != self.environment_identity:
                raise ValueError("Environmental inputs changed since preprocessing; regenerate and retrain.")
            if self.stats["label_metadata_sha256"] != data.file_sha256(self.label_dir / "label_metadata.json"):
                raise ValueError("Preprocessing and labels differ; regenerate corrected sequences.")

    def valid(self, name, indices):
        values = np.asarray(self.values[name][indices])
        valid = (self.masks[name][indices] > 0) & np.isfinite(values) & self.ocean
        if name == "chlorophyll":
            valid &= values > 0
        return values, valid

    def fit_statistics(self):
        selected = data.training_selection(self.dates)
        stats = {"train_start": config.TRAIN_START, "train_end": config.TRAIN_END,
                 "label_metadata_sha256": data.file_sha256(self.label_dir / "label_metadata.json"),
                 "channel_order": list(config.CHANNEL_NAMES), "include_observation_masks": True,
                 "clip_standardized": 8., "channels": {}}
        for name in config.CHANNEL_NAMES:
            values, valid = self.valid(name, selected)
            observed = value_transform(values[valid], name)
            if not len(observed):
                raise ValueError(f"No training observations for {name}.")
            q25, median, q75 = np.percentile(observed, [25, 50, 75])
            stats["channels"][name] = {"median": float(median), "scale": float(max(q75 - q25, 1e-6)),
                                       "transform": "log1p" if name in ("chlorophyll", "rainfall") else "identity"}
        self.stats = stats
        return stats

    def transform(self, indices):
        if self.stats is None:
            raise ValueError("Training statistics must be fitted/loaded before transformation.")
        indices = np.asarray(indices, dtype=int)
        if len(indices) != config.INPUT_DAYS or np.any(np.diff(indices) != 1) or indices.min() < 0 or indices.max() >= len(self.dates):
            raise ValueError("Input must contain five consecutive in-range daily indices.")
        values_out, masks_out = [], []
        for name in config.CHANNEL_NAMES:
            values, valid = self.valid(name, indices)
            stats = self.stats["channels"][name]
            filled = np.where(valid, value_transform(np.where(valid, values, 0), name), stats["median"])
            scaled = np.clip((filled - stats["median"]) / stats["scale"], -8, 8)
            values_out.append(np.where(valid, scaled, 0).astype(np.float32))
            masks_out.append(valid.astype(np.float32))
        # Seven original environmental channels plus seven availability flags.
        result = np.stack(values_out + masks_out, axis=-1)
        if self.stats.get("seasonal_features", False):
            from preprocessing.features import add_seasonal_features
            result = add_seasonal_features(result, self.dates[indices])
        return result

    def forecast_support(self, indices):
        months = data.month_numbers(self.dates[indices])
        return self.ocean[None] & np.isfinite(self.climatology[months - 1])


def prepare_sequences():
    pipeline = EnvironmentalPipeline(require_stats=False)
    config.SEQUENCE_DIR.mkdir(parents=True, exist_ok=True)
    index = build_sequence_index(pipeline.dates, pipeline.labels)
    if any((index.split == split).sum() == 0 for split in config.SPLITS):
        raise ValueError("Each split must contain valid sequences.")
    index.to_csv(config.SEQUENCE_DIR / "sequence_index.csv", index=False)
    stats = pipeline.fit_statistics()
    (config.SEQUENCE_DIR / "preprocessing.json").write_text(json.dumps(stats, indent=2) + "\n")
    manifest = {"label_metadata_sha256": stats["label_metadata_sha256"],
                "sequence_index_sha256": data.file_sha256(config.SEQUENCE_DIR / "sequence_index.csv"),
                "splits": {}, "shape": list(pipeline.shape),
                "environment_fingerprint": pipeline.environment_identity}
    for split, (first, last) in config.SPLITS.items():
        target = pipeline.labels[(pipeline.dates >= np.datetime64(first)) & (pipeline.dates <= np.datetime64(last))]
        manifest["splits"][split] = {"sequences": int((index.split == split).sum()),
                                     "date_start": first, "date_end": last,
                                     "class_counts": [int((target == c).sum()) for c in range(3)],
                                     "missing_labels": int((target == 255).sum())}
    (config.SEQUENCE_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["splits"], indent=2))


if __name__ == "__main__":
    prepare_sequences()
