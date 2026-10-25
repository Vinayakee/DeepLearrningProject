"""Original January diagnostic, now selecting January rather than all five years."""
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

chl_file = "data/processed/sentinel3_chl_2021_2025.npy"
mask_file = "data/processed/sentinel3_mask_2021_2025.npy"

dates = np.loadtxt(ROOT / "data/processed/dates_2021_2025.txt", dtype="datetime64[D]")
selected = (dates >= np.datetime64("2021-01-01")) & (dates <= np.datetime64("2021-01-31"))
chl = np.load(ROOT / chl_file, mmap_mode="r")[selected]
mask = np.load(ROOT / mask_file, mmap_mode="r")[selected]

valid = (mask > 0) & np.isfinite(chl) & (chl > 0)
values = chl[valid]

print("=" * 60)
print("SENTINEL-3 JANUARY 2021 QUALITY CHECK")
print("=" * 60)

print("Array shape:", chl.shape)

print("Total pixels:", chl.size)

print("Valid pixels:", np.sum(valid))

print(
    "Overall coverage:",
    f"{np.mean(mask) * 100:.2f}%"
)

print()

print("NaN values:", np.sum(np.isnan(chl)))

print("Infinite values:", np.sum(np.isinf(chl)))

print()

if len(values) > 0:

    print("Minimum CHL:", np.min(values))
    print("Maximum CHL:", np.max(values))
    print("Mean CHL:", np.mean(values))
    print("Median CHL:", np.median(values))

    print()

    print(
        "CHL > 10 mg/m³:",
        np.sum(values > 10)
    )

    print(
        "CHL > 50 mg/m³:",
        np.sum(values > 50)
    )

    print(
        "CHL >= 100 mg/m³:",
        np.sum(values >= 100)
    )

    print()

    print(
        "Percentage > 10:",
        f"{np.mean(values > 10) * 100:.4f}%"
    )

    print(
        "Percentage > 50:",
        f"{np.mean(values > 50) * 100:.4f}%"
    )

    print(
        "Percentage >= 100:",
        f"{np.mean(values >= 100) * 100:.4f}%"
    )

print("=" * 60)