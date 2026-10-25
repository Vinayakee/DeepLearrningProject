"""Memory-mapped batches using the corrected labels and shared preprocessing."""
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import config
from preprocessing.pipeline import EnvironmentalPipeline
from preprocessing.data_utils import file_sha256

CHANNEL_NAMES = list(config.CHANNEL_NAMES)


class HABDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, split="train", batch_size=1, shuffle=True, pipeline=None, limit=None, **kwargs):
        super().__init__(**kwargs)
        if split not in config.SPLITS or batch_size < 1:
            raise ValueError("Invalid split or batch size.")
        self.pipeline = pipeline or EnvironmentalPipeline()
        self.batch_size, self.shuffle, self.split = batch_size, shuffle, split
        directory = self.pipeline.sequence_dir
        manifest = json.loads((directory / "manifest.json").read_text())
        if (manifest["label_metadata_sha256"] != file_sha256(self.pipeline.label_dir / "label_metadata.json")
                or manifest["sequence_index_sha256"] != file_sha256(directory / "sequence_index.csv")):
            raise ValueError("Sequence index and corrected labels differ; regenerate sequences.")
        index = pd.read_csv(directory / "sequence_index.csv")
        self.indices = index[index.split == split].reset_index(drop=True)
        if limit:
            # Spread smoke samples through the split, never take test samples for fitting.
            positions = np.linspace(0, len(self.indices) - 1, min(limit, len(self.indices)), dtype=int)
            self.indices = self.indices.iloc[positions].reset_index(drop=True)
        first, last = map(np.datetime64, config.SPLITS[split])
        for row in self.indices.itertuples():
            if (row.input_end_idx - row.input_start_idx != 4 or row.target_start_idx != row.input_end_idx + 1
                    or row.target_end_idx - row.target_start_idx != 4
                    or self.pipeline.dates[row.input_start_idx] < first or self.pipeline.dates[row.target_end_idx] > last):
                raise ValueError("Sequence crosses split boundaries or has incorrect windows.")
        self.num_samples = len(self.indices)
        if not self.num_samples:
            raise ValueError(f"No valid sequences in {split}.")
        self.rng = np.random.default_rng(42)
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(self.num_samples / self.batch_size))

    def on_epoch_end(self):
        self.order = np.arange(self.num_samples)
        if self.shuffle:
            self.rng.shuffle(self.order)

    def _load_sample(self, row):
        start, target = int(row["input_start_idx"]), int(row["target_start_idx"])
        X = self.pipeline.transform(np.arange(start, start + 5))
        raw = np.asarray(self.pipeline.labels[target:target + 5])
        valid = raw != 255
        y = np.eye(3, dtype=np.float32)[np.where(valid, raw, 0)]
        y[~valid] = 0
        return X, y

    def __getitem__(self, batch_index):
        if batch_index < 0 or batch_index >= len(self):
            raise IndexError("Batch index is out of range.")
        rows = self.indices.iloc[self.order[batch_index * self.batch_size:(batch_index + 1) * self.batch_size]]
        samples = [self._load_sample(row) for _, row in rows.iterrows()]
        return tuple(np.stack(part) for part in zip(*samples))


if __name__ == "__main__":
    generator = HABDataGenerator(shuffle=False)
    X, y = generator[0]
    print("Input:", X.shape, "Target:", y.shape, "Valid pixels:", y.sum())
    assert np.isfinite(X).all() and X.shape[1:] == (5, 120, 84, 16 if generator.pipeline.stats.get("seasonal_features", False) else 14)
