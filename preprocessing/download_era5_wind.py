import cdsapi
import os
import calendar

# ==========================================
# ERA5 DAILY WIND
# Kerala Coast
# 2021-2025
# One month per request
# ==========================================

OUTPUT_DIR = "data/raw/wind"

os.makedirs(OUTPUT_DIR, exist_ok=True)

YEARS = [2021, 2022, 2023, 2024, 2025]

DATASET = "derived-era5-single-levels-daily-statistics"

client = cdsapi.Client()

for year in YEARS:

    for month in range(1, 13):

        month_str = f"{month:02d}"

        output_file = os.path.join(
            OUTPUT_DIR,
            f"era5_wind_{year}_{month_str}.zip"
        )

        # Skip if already downloaded
        if os.path.exists(output_file):
            print(f"✓ {year}-{month_str} already exists. Skipping.")
            continue

        # Get correct number of days
        number_of_days = calendar.monthrange(year, month)[1]

        days = [
            f"{day:02d}"
            for day in range(1, number_of_days + 1)
        ]

        print("\n" + "=" * 60)
        print(f"DOWNLOADING ERA5 WIND: {year}-{month_str}")
        print("=" * 60)

        request = {
            "product_type": "reanalysis",

            "variable": [
                "10m_u_component_of_wind",
                "10m_v_component_of_wind"
            ],

            "year": str(year),

            "month": [month_str],

            "day": days,

            "daily_statistic": "daily_mean",

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

            print(f"✓ {year}-{month_str} COMPLETE")
            print(f"Saved: {output_file}")

        except Exception as e:

            print(f"✗ {year}-{month_str} FAILED")
            print("Error:", e)

print("\n" + "=" * 60)
print("ERA5 WIND DOWNLOAD FINISHED")
print("=" * 60)