"""Report coverage and integrity without imputing or replacing source data."""

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

if __package__:
    from . import data_utils as data
else:
    import data_utils as data


SENTINEL_DOC = "https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Data/S3OLCIL2.html"
OLCI_DOC = "https://sentiwiki.copernicus.eu/web/olci-products"


def longest_missing_run(counts):
    longest = current = 0
    for count in counts:
        current = current + 1 if count == 0 else 0
        longest = max(longest, current)
    return longest


def summarize_source(name, values, mask, dates, ocean):
    data.validate_pair(values, mask, (len(dates), *ocean.shape))
    positive_required = name in ("chlorophyll", "sentinel3")
    daily = []
    samples = []
    issues = {"masked_nonfinite_pixels": 0, "finite_unmasked_pixels": 0,
              "masked_nonpositive_chlorophyll_pixels": 0}
    ocean_pixels = int(ocean.sum())
    if not ocean_pixels:
        raise ValueError("Ocean support is empty; coverage has no valid denominator.")
    for index, date in enumerate(dates):
        day = values[index]
        finite = np.isfinite(day)
        available = mask[index] > 0
        issues["masked_nonfinite_pixels"] += int((available & ~finite).sum())
        issues["finite_unmasked_pixels"] += int((finite & ~available).sum())
        if positive_required:
            issues["masked_nonpositive_chlorophyll_pixels"] += int((available & finite & (day <= 0)).sum())
        valid = available & finite
        if positive_required:
            valid &= day > 0
        samples.append(day[valid & ocean])
        daily.append({
            "source": name, "date": str(date), "valid_grid_pixels": int(valid.sum()),
            "valid_ocean_pixels": int((valid & ocean).sum()),
            "valid_outside_ocean_pixels": int((valid & ~ocean).sum()),
            "grid_coverage_percent": float(valid.mean() * 100),
            "ocean_coverage_percent": float((valid & ocean).sum() / ocean_pixels * 100),
        })
    observed_values = np.concatenate(samples)
    quantiles = np.percentile(observed_values, [0, 1, 50, 99, 100]).tolist() if len(observed_values) else [None] * 5
    grid_counts = np.array([row["valid_grid_pixels"] for row in daily])
    ocean_counts = np.array([row["valid_ocean_pixels"] for row in daily])
    metrics = {
        "shape": list(values.shape), "integrity_issues": issues,
        "grid_coverage_percent": float(grid_counts.sum() / values.size * 100),
        "ocean_coverage_percent": float(ocean_counts.sum() / (len(dates) * ocean_pixels) * 100),
        "days_without_ocean_observations": int((ocean_counts == 0).sum()),
        "longest_run_without_ocean_observations_days": longest_missing_run(ocean_counts),
        "valid_pixels_outside_ocean_support": sum(row["valid_outside_ocean_pixels"] for row in daily),
        "ocean_value_quantiles": dict(zip(("min", "p01", "median", "p99", "max"), quantiles)),
        "yearly_coverage": {}, "monthly_coverage": {}, "latest_five_days": daily[-5:],
    }
    for period_type, unit in (("yearly_coverage", "Y"), ("monthly_coverage", "M")):
        periods = dates.astype(f"datetime64[{unit}]")
        for period in np.unique(periods):
            selected = periods == period
            metrics[period_type][str(period)] = {
                "days": int(selected.sum()),
                "ocean_coverage_percent": float(ocean_counts[selected].sum() / (selected.sum() * ocean_pixels) * 100),
                "days_without_ocean_observations": int((ocean_counts[selected] == 0).sum()),
            }
    if name == "sentinel3":
        metrics["quality_status"] = "Retrieval quality unverified; saved chlorophyll/dataMask arrays contain no quality/science flags."
        metrics["boundary_diagnostics"] = {
            # Diagnostic only: typical ranges and boundary piles are not proof
            # that individual observations are invalid. Do not clip them.
            "at_0_01_mg_m3_pixels": int(np.isclose(observed_values, 0.01, rtol=0, atol=1e-7).sum()),
            "at_100_mg_m3_pixels": int(np.isclose(observed_values, 100, rtol=0, atol=1e-5).sum()),
            "above_100_mg_m3_pixels": int((observed_values > 100).sum()),
        }
    return metrics, daily


def paired_chlorophyll_comparison(directory, dates, ocean):
    """Describe source agreement on jointly observed cells, not label accuracy."""
    shape = (len(dates), *ocean.shape)
    modis, modis_mask = data.load_pair(directory, "chlorophyll", shape)
    sentinel = np.load(directory / "sentinel3_chl_2021_2025.npy", mmap_mode="r", allow_pickle=False)
    sentinel_mask = np.load(directory / "sentinel3_mask_2021_2025.npy", mmap_mode="r", allow_pickle=False)
    differences = {"train": [], "validation": [], "test": []}
    selected = data.training_selection(dates)
    for index, date in enumerate(dates):
        a, b = modis[index], sentinel[index]
        valid = (ocean & (modis_mask[index] > 0) & (sentinel_mask[index] > 0)
                 & np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0))
        split = "train" if selected[index] else "validation" if str(date).startswith("2024") else "test"
        differences[split].append(np.log10(b[valid]) - np.log10(a[valid]))
    report = {}
    for split, samples in differences.items():
        delta = np.concatenate(samples)
        report[split] = {"jointly_observed_pixels": len(delta),
                         "median_log10_sentinel_minus_modis": float(np.median(delta)) if len(delta) else None,
                         "median_absolute_log10_difference": float(np.median(np.abs(delta))) if len(delta) else None}
    return report


def build_report(directory):
    directory = Path(directory)
    dates = data.load_dates(directory)
    lat, lon, shape = data.load_grid(directory, dates)
    ocean = data.derive_ocean_support(directory, shape, data.training_selection(dates))
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "date_start": str(dates[0]), "date_end": str(dates[-1]),
        "ocean_support_pixels": int(ocean.sum()), "total_grid_pixels": int(ocean.size),
        "ocean_mask_method": "Union of training-year SST/current availability; a support proxy, not an authoritative coastline.",
        "coverage_definition": "Observed means mask=1 and finite; chlorophyll must also be positive. Grid coverage includes the whole rectangle; ocean coverage uses the fixed training marine support mask.",
        "sources": {}, "notes": [
            "Cloud/observation gaps remain missing; coverage is measured before imputation.",
            "Chlorophyll anomaly labels do not establish species, toxicity, or confirmed HAB events.",
        ],
    }
    rows = []
    for name in data.CHANNEL_NAMES:
        values, mask = data.load_pair(directory, name, shape)
        metrics, source_rows = summarize_source(name, values, mask, dates, ocean)
        report["sources"][name] = metrics
        rows.extend(source_rows)
    sentinel_files = [directory / name for name in (
        "sentinel3_chl_2021_2025.npy", "sentinel3_mask_2021_2025.npy", "sentinel3_dates_2021_2025.txt")]
    if all(path.exists() for path in sentinel_files):
        sentinel_dates = data.load_dates(directory, sentinel_files[2].name)
        if not np.array_equal(dates, sentinel_dates):
            raise ValueError("Sentinel-3 dates do not match the environmental calendar.")
        values = np.load(sentinel_files[0], mmap_mode="r", allow_pickle=False)
        mask = np.load(sentinel_files[1], mmap_mode="r", allow_pickle=False)
        metrics, source_rows = summarize_source("sentinel3", values, mask, dates, ocean)
        # Existing downloader uses BBOX [74, 8, 77.5, 13] with north-first rows.
        # This checks its inferred cell centres, not independently saved georeferencing.
        expected_lat = 13 - (np.arange(len(lat)) + 0.5) * 5 / len(lat)
        expected_lon = 74 + (np.arange(len(lon)) + 0.5) * 3.5 / len(lon)
        metrics["inferred_grid_centres_match_modis"] = bool(
            np.allclose(lat, expected_lat, rtol=0, atol=1e-4)
            and np.allclose(lon, expected_lon, rtol=0, atol=1e-4))
        report["sources"]["sentinel3"] = metrics
        rows.extend(source_rows)
        report["paired_chlorophyll_comparison"] = paired_chlorophyll_comparison(directory, dates, ocean)
        report["notes"].append("Sentinel-3 remains excluded from modelling until retrieval quality is checked; boundary values are diagnostic and have not been removed.")
        report["references"] = [SENTINEL_DOC, OLCI_DOC]
    elif any(path.exists() for path in sentinel_files):
        raise ValueError("Incomplete Sentinel-3 bundle: values, mask, and dates must all be present.")
    else:
        report["notes"].append("Sentinel-3 arrays are unavailable; assessed the existing seven-channel dataset only.")
    return report, rows


def render_markdown(report):
    lines = ["# HAB data quality report", "", f"Period: {report['date_start']} to {report['date_end']}", "",
             f"Ocean support: {report['ocean_support_pixels']} / {report['total_grid_pixels']} grid pixels, derived from training-year marine availability.", "",
             "| Source | Grid coverage | Ocean coverage | Empty ocean days | Longest gap (days) |",
             "|---|---:|---:|---:|---:|"]
    for name, metrics in report["sources"].items():
        lines.append(f"| {name} | {metrics['grid_coverage_percent']:.2f}% | {metrics['ocean_coverage_percent']:.2f}% | {metrics['days_without_ocean_observations']} | {metrics['longest_run_without_ocean_observations_days']} |")
    lines += ["", "Coverage is measured before imputation. Ocean support is an availability proxy, not a coastline map.", ""]
    for name, metrics in report["sources"].items():
        lines += [f"## {name}: latest five input days", "",
                  "| Date | Ocean coverage |", "|---|---:|"]
        lines += [f"| {row['date']} | {row['ocean_coverage_percent']:.2f}% |" for row in metrics["latest_five_days"]]
        lines += ["", f"Integrity issue counts: {metrics['integrity_issues']}", ""]
        if name == "sentinel3":
            lines += [metrics["quality_status"], "", f"Boundary diagnostics within ocean support: {metrics['boundary_diagnostics']}", "",
                      f"Inferred downloader grid centres match MODIS: {metrics['inferred_grid_centres_match_modis']}. Independent per-source georeferencing was not saved.", ""]
    lines += ["## Interpretation", ""] + [f"- {note}" for note in report["notes"]]
    if "paired_chlorophyll_comparison" in report:
        lines += ["", "Paired source agreement (jointly observed ocean pixels; not HAB validation):", "",
                  "| Split | Paired pixels | Median log10 difference (Sentinel − MODIS) | Median absolute log10 difference |",
                  "|---|---:|---:|---:|"]
        for split, metrics in report["paired_chlorophyll_comparison"].items():
            lines.append(f"| {split} | {metrics['jointly_observed_pixels']} | {metrics['median_log10_sentinel_minus_modis']} | {metrics['median_absolute_log10_difference']} |")
    if "references" in report:
        lines += ["", "Sentinel Hub documents CHL_OC4ME in log10(mg/m³) and dataMask as data availability. OLCI product documentation describes separate quality/science information. The current downloader retains only chlorophyll and dataMask.", "",
                  f"Sources: [Sentinel Hub OLCI L2]({SENTINEL_DOC}), [Copernicus OLCI products]({OLCI_DOC})."]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=data.PROCESSED_DIR)
    parser.add_argument("--output-dir", type=Path, default=data.PROJECT_ROOT / "outputs" / "data_quality")
    args = parser.parse_args()
    report, rows = build_report(args.processed_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (args.output_dir / "report.md").write_text(render_markdown(report))
    with (args.output_dir / "daily_coverage.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved quality report and daily coverage to {args.output_dir}")
    for name, metrics in report["sources"].items():
        print(f"{name}: {metrics['ocean_coverage_percent']:.2f}% ocean coverage; {metrics['days_without_ocean_observations']} empty days")
    if any(any(metrics["integrity_issues"].values()) for metrics in report["sources"].values()):
        raise SystemExit("Data integrity problems found; see report.json for counts.")


if __name__ == "__main__":
    main()
