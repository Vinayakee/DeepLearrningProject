"""Generate a forecast through the same service used by the API."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prediction.service import ForecastService


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-end-date")
    parser.add_argument("--historical-demo", action="store_true")
    args = parser.parse_args()
    result = ForecastService().generate(args.input_end_date, args.historical_demo)
    print("Forecast:", result["forecast_id"], "Input:", result["input_dates"], "Future:", result["forecast_dates"])
    print("Freshness:", result["freshness"], "Quality:", result["data_quality"])
    for day in result["daily_predictions"]:
        print(day)


if __name__ == "__main__":
    main()
