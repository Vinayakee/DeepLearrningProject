"""Regenerate all five maps from the current dated forecast bundle."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import config
from prediction.service import cached_forecast, render_maps

if __name__ == "__main__":
    bundle = cached_forecast(allow_demo=True)
    if bundle.get("status") != "success":
        print(bundle.get("message", "No prediction maps are available."))
        raise SystemExit(0)
    directory = config.FORECAST_DIR / bundle["forecast_id"]
    render_maps(np.load(directory / "prediction.npy"), np.load(directory / "risk_classes.npy"),
                np.load(config.PROCESSED_DIR / "latitude_kerala_2021_2025.npy"),
                np.load(config.PROCESSED_DIR / "longitude_kerala_2021_2025.npy"),
                bundle["forecast_dates"], directory)
    print("Five dated maps regenerated:", directory)
