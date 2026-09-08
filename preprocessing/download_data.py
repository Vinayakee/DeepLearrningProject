import os
import subprocess
from datetime import date, timedelta

import earthaccess


# ============================================================
# SETTINGS
# ============================================================

START_DATE = date(2021, 1, 1)
END_DATE = date(2025, 12, 31)

CHL_DIR = "data/raw/Chlorophyll-a"
SST_DIR = "data/raw/sst"

# IMPORTANT:
# Keep this True for the first test.
# After the test succeeds, change it to False.
TEST_MODE = False

# One test day
TEST_DATE = date(2021, 1, 15)


# ============================================================
# CREATE DIRECTORIES
# ============================================================

os.makedirs(CHL_DIR, exist_ok=True)
os.makedirs(SST_DIR, exist_ok=True)


# ============================================================
# NASA EARTHDATA LOGIN
# ============================================================

print("=" * 60)
print("NASA EARTHDATA LOGIN")
print("=" * 60)

auth = earthaccess.login(
    strategy="netrc",
    persist=True
)

if not auth.authenticated:
    raise RuntimeError("NASA Earthdata login failed.")

print("NASA login successful!")


# ============================================================
# DOWNLOAD ONE MODIS CHL DAY
# ============================================================

def download_chlorophyll_day(download_date):

    date_string = download_date.strftime("%Y%m%d")

    filename = (
        f"AQUA_MODIS.{date_string}."
        f"L3m.DAY.CHL.chlor_a.4km.nc"
    )

    output_path = os.path.join(CHL_DIR, filename)

    # Skip if already downloaded
    if os.path.exists(output_path):
        print(f"[SKIP] {filename}")
        return True

    print(f"\n[SEARCH] Chlorophyll {date_string}")

    try:

        results = earthaccess.search_data(
            short_name="MODISA_L3m_CHL",
            granule_name=f"*.{date_string}.L3m.DAY.CHL.chlor_a.4km.nc",
            temporal=(date_string, date_string),
            count=10
        )

        if not results:
            print(f"[NOT FOUND] {date_string}")
            return False

        print(f"[FOUND] {filename}")
        print("[DOWNLOAD] Starting...")

        earthaccess.download(
            results[0],
            local_path=CHL_DIR
        )

        if os.path.exists(output_path):
            print(f"[SUCCESS] {filename}")
            return True

        print(f"[WARNING] Download finished but file not found: {filename}")
        return False

    except Exception as e:

        print(f"[ERROR] {date_string}: {e}")
        return False


# ============================================================
# TEST MODE
# ============================================================

if TEST_MODE:

    print()
    print("=" * 60)
    print("TEST MODE")
    print("=" * 60)
    print(f"Testing date: {TEST_DATE}")

    success = download_chlorophyll_day(TEST_DATE)

    if success:
        print()
        print("=" * 60)
        print("TEST SUCCESSFUL!")
        print("=" * 60)
        print()
        print("If the file exists in:")
        print(CHL_DIR)
        print()
        print("change:")
        print("TEST_MODE = False")
        print()
        print("Then run this script again for the full dataset.")
    else:
        print()
        print("TEST FAILED.")
        print("Do NOT start the 5-year download.")

    raise SystemExit


# ============================================================
# FULL 5-YEAR MODIS DOWNLOAD
# ============================================================

print()
print("=" * 60)
print("STARTING 5-YEAR MODIS CHL DOWNLOAD")
print("=" * 60)

current_date = START_DATE

successful = 0
failed_dates = []

total_days = (END_DATE - START_DATE).days + 1
day_number = 0

while current_date <= END_DATE:

    day_number += 1

    print()
    print("-" * 60)
    print(f"DAY {day_number}/{total_days}")
    print(f"DATE: {current_date}")
    print("-" * 60)

    success = download_chlorophyll_day(current_date)

    if success:
        successful += 1
    else:
        failed_dates.append(
            current_date.strftime("%Y-%m-%d")
        )

    current_date += timedelta(days=1)


# ============================================================
# SAVE FAILED DATES
# ============================================================

failed_file = "data/raw/failed_chlorophyll_dates.txt"

with open(failed_file, "w") as f:

    for failed_date in failed_dates:
        f.write(failed_date + "\n")


# ============================================================
# FINAL REPORT
# ============================================================

print()
print("=" * 60)
print("MODIS DOWNLOAD COMPLETE")
print("=" * 60)

print(f"Total days:       {total_days}")
print(f"Successful:       {successful}")
print(f"Failed:           {len(failed_dates)}")

print()
print("Failed dates saved to:")
print(failed_file)

print()
print("Chlorophyll directory:")
print(CHL_DIR)