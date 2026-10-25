"""Train corrected ConvLSTM; smoke runs never overwrite the full checkpoint."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import config
import tensorflow as tf
from models.convlstm import build_convlstm
from preprocessing.pipeline import EnvironmentalPipeline
from preprocessing.data_utils import file_sha256
from training.data_generator import HABDataGenerator
from training.losses import MaskedWeightedLoss, MaskedAccuracy


def main():
    baseline_freeze = config.ROOT / "outputs/improvement/baseline/freeze.json"
    if baseline_freeze.exists() and "--smoke" not in sys.argv:
        frozen = json.loads(baseline_freeze.read_text())
        if config.MODEL_PATH.resolve() == Path(frozen["checkpoint_original"]).resolve():
            raise ValueError("The original baseline checkpoint is frozen. Train into a separate experiment; do not overwrite it.")
    if (config.RUN_DIR / "freeze.json").exists():
        raise ValueError("Active final run is frozen and cannot be overwritten. Use HAB_USE_BASELINE=1 for baseline work, or a separate experiment.")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.threads) < 1:
        parser.error("Epochs, batch size and threads must be positive.")
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.keras.utils.set_random_seed(42)
    pipeline = EnvironmentalPipeline()
    train = HABDataGenerator("train", args.batch_size, True, pipeline, limit=12 if args.smoke else None)
    validation = HABDataGenerator("validation", args.batch_size, False, pipeline, limit=8 if args.smoke else None)
    manifest = json.loads((config.SEQUENCE_DIR / "manifest.json").read_text())
    counts = np.array(manifest["splits"]["train"]["class_counts"], dtype=float)
    weights = np.sqrt(counts.max() / np.maximum(counts, 1)).tolist()
    output = config.RUN_DIR / "smoke" if args.smoke else config.RUN_DIR
    model_path = output / "smoke.keras" if args.smoke else config.MODEL_PATH
    if not args.smoke and model_path.resolve() in {
            (config.ROOT / "models" / name).resolve() for name in ("best_convlstm.keras", "best_convlstm_backup.keras")} :
        parser.error("Choose a separate corrected checkpoint; original models are preserved.")
    output.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    # Archive an earlier corrected run before deliberately training a new one.
    if model_path.exists():
        import shutil
        archive = output / ("previous_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f"))
        archive.mkdir()
        shutil.copy2(model_path, archive / model_path.name)
        for filename in ("training_config.json", "training_history.csv"):
            if (output / filename).exists():
                shutil.copy2(output / filename, archive / filename)
    record = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "smoke": args.smoke,
              "epochs_requested": args.epochs, "batch_size": args.batch_size, "learning_rate": .001,
              "class_weights": weights, "weight_method": "sqrt inverse training class frequency relative to normal",
              "filters": [8, 4], "spatial_downsample": 2, "seed": 42,
              "input_shape": [5, 120, 84, 14], "output_shape": [5, 120, 84, 3],
              "train_sequences": train.num_samples, "validation_sequences": validation.num_samples,
              "label_metadata_sha256": file_sha256(config.LABEL_DIR / "label_metadata.json"),
              "preprocessing_sha256": file_sha256(config.SEQUENCE_DIR / "preprocessing.json"),
              "tensorflow_version": tf.__version__, "platform": platform.platform(),
              "model_path": str(model_path), "devices": [str(d) for d in tf.config.list_physical_devices()]}
    (output / "training_config.json").write_text(json.dumps(record, indent=2) + "\n")
    model = build_convlstm()
    model.compile(optimizer=tf.keras.optimizers.Adam(.001, clipnorm=1.),
                  loss=MaskedWeightedLoss(weights), metrics=[MaskedAccuracy()])
    print(json.dumps(record, indent=2), flush=True)
    started = time.monotonic()
    history = model.fit(train, validation_data=validation, epochs=args.epochs, verbose=2, callbacks=[
        tf.keras.callbacks.ModelCheckpoint(str(model_path), monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=2, factor=.5, min_lr=1e-5),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        tf.keras.callbacks.CSVLogger(str(output / "training_history.csv")),
        tf.keras.callbacks.TerminateOnNaN()])
    record.update({"epochs_completed": len(history.history["loss"]),
                   "best_epoch": int(np.argmin(history.history["val_loss"])) + 1,
                   "duration_seconds": time.monotonic() - started,
                   "model_sha256": file_sha256(model_path)})
    (output / "training_config.json").write_text(json.dumps(record, indent=2) + "\n")
    restored = tf.keras.models.load_model(model_path, compile=False)
    X, y = validation[0]
    prediction = restored(X, training=False).numpy()
    assert prediction.shape == y.shape and np.isfinite(prediction).all()
    np.testing.assert_allclose(prediction.sum(-1), 1., atol=1e-6)
    print("Saved checkpoint reload and real-batch inference passed.", flush=True)


if __name__ == "__main__":
    main()
