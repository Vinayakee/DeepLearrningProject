"""Shared validation for the existing daily environmental arrays."""

import hashlib
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
LABEL_DIR = PROCESSED_DIR / "training_only_labels"
TRAIN_START = "2021-01-01"
TRAIN_END = "2023-12-31"
CHANNEL_NAMES = (
    "chlorophyll", "sst", "wind_u", "wind_v", "rainfall", "current_u", "current_v"
)


def load_dates(directory, filename="dates_2021_2025.txt"):
    dates = np.atleast_1d(np.loadtxt(Path(directory) / filename, dtype="datetime64[D]"))
    if not len(dates) or np.isnat(dates).any():
        raise ValueError("Dates must be nonempty and contain no missing dates.")
    if not np.all(np.diff(dates) == np.timedelta64(1, "D")):
        raise ValueError("Dates must be unique, ordered, and consecutive daily observations.")
    return dates


def training_selection(dates, start=TRAIN_START, end=TRAIN_END):
    dates = np.asarray(dates, dtype="datetime64[D]")
    start, end = np.datetime64(start, "D"), np.datetime64(end, "D")
    if np.isnat(start) or np.isnat(end) or start > end:
        raise ValueError("Invalid training date range.")
    selected = (dates >= start) & (dates <= end)
    if not selected.any():
        raise ValueError("No observations fall inside the training date range.")
    return selected


def month_numbers(dates):
    return np.asarray(dates, dtype="datetime64[M]").astype(int) % 12 + 1


def validate_pair(values, mask, expected_shape=None):
    if values.ndim != 3 or values.shape != mask.shape:
        raise ValueError(f"Expected matching (time, latitude, longitude) arrays, got {values.shape}, {mask.shape}.")
    if expected_shape is not None and values.shape != expected_shape:
        raise ValueError(f"Array shape {values.shape} does not match grid/date shape {expected_shape}.")
    if not np.all((mask == 0) | (mask == 1)):
        raise ValueError("Observation masks must contain only 0 and 1.")


def load_pair(directory, name, expected_shape):
    directory = Path(directory)
    values = np.load(directory / f"{name}_2021_2025.npy", mmap_mode="r", allow_pickle=False)
    mask = np.load(directory / f"{name}_mask_2021_2025.npy", mmap_mode="r", allow_pickle=False)
    validate_pair(values, mask, expected_shape)
    return values, mask


def load_grid(directory, dates):
    directory = Path(directory)
    lat = np.load(directory / "latitude_kerala_2021_2025.npy", allow_pickle=False)
    lon = np.load(directory / "longitude_kerala_2021_2025.npy", allow_pickle=False)
    for name, coord in (("Latitude", lat), ("Longitude", lon)):
        if coord.ndim != 1 or not len(coord) or not np.isfinite(coord).all():
            raise ValueError(f"{name} must be a finite, nonempty one-dimensional coordinate.")
        if not (np.all(np.diff(coord) > 0) or np.all(np.diff(coord) < 0)):
            raise ValueError(f"{name} must be strictly monotonic.")
    return lat, lon, (len(dates), len(lat), len(lon))


def derive_ocean_support(directory, expected_shape, selected):
    """Use training SST/current availability as ocean support, not a coastline map.

    The union avoids excluding coastal cells solely because one interpolated
    marine product is missing there. Held-out observations are never used.
    """
    ocean = np.zeros(expected_shape[1:], dtype=bool)
    for name in ("sst", "current_u", "current_v"):
        values, mask = load_pair(directory, name, expected_shape)
        for index in np.flatnonzero(selected):
            ocean |= (mask[index] > 0) & np.isfinite(values[index])
    if not ocean.any():
        raise ValueError("Training marine products provide no ocean support.")
    return ocean


def source_fingerprint(directory):
    """Detect changed inputs between the climatology and label commands."""
    directory = Path(directory)
    names = ["dates_2021_2025.txt", "latitude_kerala_2021_2025.npy", "longitude_kerala_2021_2025.npy"]
    names += [f"{name}{suffix}_2021_2025.npy" for name in ("chlorophyll", "sst", "current_u", "current_v") for suffix in ("", "_mask")]
    return {name: {"bytes": (directory / name).stat().st_size, "mtime_ns": (directory / name).stat().st_mtime_ns} for name in names}


def environment_fingerprint(directory):
    """Lightweight identity of every aligned feature and availability mask."""
    directory = Path(directory)
    names = [f"{name}{suffix}_2021_2025.npy" for name in CHANNEL_NAMES for suffix in ("", "_mask")]
    return {name: {"bytes": (directory / name).stat().st_size, "mtime_ns": (directory / name).stat().st_mtime_ns} for name in names}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_output_directory(processed_dir, output_dir):
    if Path(processed_dir).resolve() == Path(output_dir).resolve():
        raise ValueError("Use a separate output directory to preserve labels used by the existing checkpoints.")
