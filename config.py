"""Paths and small, shared settings for the existing project."""
import os
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = ROOT / "data" / "processed"
DEPLOYMENT_PATH = ROOT / "outputs" / "improvement" / "final" / "deployment.json"
DEPLOYMENT = json.loads(DEPLOYMENT_PATH.read_text()) if DEPLOYMENT_PATH.exists() and os.getenv("HAB_USE_BASELINE") != "1" else {}

def configured_path(key, default, environment):
    path = Path(os.getenv(environment, DEPLOYMENT.get(key, default))).expanduser()
    return path if path.is_absolute() else ROOT / path

LABEL_DIR = configured_path("label_dir", "data/processed/training_only_labels", "HAB_LABEL_DIR")
SEQUENCE_DIR = configured_path("sequence_dir", "data/sequences/corrected", "HAB_SEQUENCE_DIR")
RUN_DIR = configured_path("run_dir", "outputs/corrected", "HAB_RUN_DIR")
os.environ.setdefault("MPLCONFIGDIR", str(RUN_DIR / ".matplotlib"))
MODEL_PATH = configured_path("model_path", "models/corrected_convlstm.keras", "HAB_MODEL_PATH")
FORECAST_DIR = ROOT / "outputs" / "forecasts"
INPUT_DAYS = OUTPUT_DAYS = 5
HEIGHT, WIDTH = 120, 84
CHANNEL_NAMES = ("chlorophyll", "sst", "wind_u", "wind_v", "rainfall", "current_u", "current_v")
CLASS_NAMES = ("Normal", "Elevated risk", "Strong bloom risk")
TRAIN_START, TRAIN_END = "2021-01-01", "2023-12-31"
SPLITS = {"train": (TRAIN_START, TRAIN_END), "validation": ("2024-01-01", "2024-12-31"), "test": ("2025-01-01", "2025-12-31")}
MAX_DATA_AGE_DAYS = int(os.getenv("HAB_MAX_DATA_AGE_DAYS", "3"))

if MAX_DATA_AGE_DAYS < 0:
    raise ValueError("HAB_MAX_DATA_AGE_DAYS must be nonnegative.")
