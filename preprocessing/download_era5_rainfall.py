import cdsapi
import os
import calendar

# ==========================================
# ERA5 DAILY TOTAL PRECIPITATION
# Kerala Coast
# 2021-2025
# One month per request
# ==========================================

OUTPUT_DIR = "data/raw/rainfall"

os.makedirs(OUTPUT_DIR, exist_ok=True)

YEARS = [2021, 2022, 2023, 2024, 2025]

DATASET = "derived-era5-single-levels-daily-statistics"

client = cdsapi.Client()

for year in YEARS:

    for month in range(1, 13):

        month_str = f"{month:02d}"

        output_file = os.path.join(
            OUTPUT_DIR,
            f"era5_rainfall_{year}_{month_str}.zip"
        )

        # Skip already downloaded files
        if os.path.exists(output_file):
            print(f"✓ {year}-{month_str} already exists. Skipping.")
            continue

        # Correct number of days for this month
        number_of_days = calendar.monthrange(
            year,
            month
        )[1]

        days = [
            f"{day:02d}"
            for day in range(1, number_of_days + 1)
        ]

        print("\n" + "=" * 60)
        print(f"DOWNLOADING ERA5 RAINFALL: {year}-{month_str}")
        print("=" * 60)

        request = {

            "product_type": "reanalysis",

            "variable": [
                "total_precipitation"
            ],

            "year": str(year),

            "month": [
                month_str
            ],

            "day": days,

            # Precipitation is accumulated,
            # so use daily SUM.
            "daily_statistic": "daily_sum",

            "time_zone": "utc+00:00",

            "frequency": "1_hourly",

            # North, West, South, East
            "area": [
                13,
                74,
                8,
                77.5
            ]
        }

        try:

            client.retrieve(
                DATASET,
                request,
                output_file
            )

            print(
                f"✓ {year}-{month_str} COMPLETE"
            )

            print(
                f"Saved: {output_file}"
            )

        except Exception as e:

            print(
                f"✗ {year}-{month_str} FAILED"
            )

            print(
                "Error:",
                e
            )

print("\n" + "=" * 60)
print("ERA5 RAINFALL DOWNLOAD FINISHED")
print("=" * 60)