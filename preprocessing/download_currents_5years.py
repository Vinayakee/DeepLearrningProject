import copernicusmarine
import os

# ============================================================
# SETTINGS
# ============================================================

OUTPUT_DIR = "data/raw/currents"
os.makedirs(OUTPUT_DIR, exist_ok=True)

YEARS = [2021, 2022, 2023, 2024, 2025]

DATASET_ID = "cmems_mod_glo_phy_my_0.083deg_P1D-m"

# Kerala / Arabian Sea region
MIN_LON = 74.0
MAX_LON = 77.5
MIN_LAT = 8.0
MAX_LAT = 13.0

# ============================================================
# DOWNLOAD
# ============================================================

print("=" * 70)
print("5-YEAR CMEMS SURFACE CURRENT DOWNLOAD")
print("=" * 70)

for year in YEARS:

    output_file = os.path.join(
        OUTPUT_DIR,
        f"currents_{year}.nc"
    )

    # --------------------------------------------------------
    # Skip existing files
    # --------------------------------------------------------

    if os.path.exists(output_file):

        print(
            f"\n✓ {year} already exists. Skipping."
        )

        continue

    print("\n" + "=" * 70)
    print(f"DOWNLOADING CURRENTS: {year}")
    print("=" * 70)

    try:

        result = copernicusmarine.subset(

            dataset_id=DATASET_ID,

            # Eastward and northward currents
            variables=[
                "uo",
                "vo"
            ],

            # Kerala region
            minimum_longitude=MIN_LON,
            maximum_longitude=MAX_LON,

            minimum_latitude=MIN_LAT,
            maximum_latitude=MAX_LAT,

            # Full year
            start_datetime=f"{year}-01-01",
            end_datetime=f"{year}-12-31",

            # Surface layer
            minimum_depth=0,
            maximum_depth=1,

            # Output
            output_directory=OUTPUT_DIR,
            output_filename=f"currents_{year}.nc",

            overwrite=True
        )

        print(f"\n✓ {year} DOWNLOAD COMPLETE")
        print(f"Saved: {output_file}")

    except Exception as e:

        print(f"\n✗ {year} DOWNLOAD FAILED")
        print("Error:", e)
        print(
            "The next year will still be attempted."
        )

# ============================================================
# FINAL CHECK
# ============================================================

print("\n" + "=" * 70)
print("CURRENT DOWNLOAD FINISHED")
print("=" * 70)

for year in YEARS:

    filepath = os.path.join(
        OUTPUT_DIR,
        f"currents_{year}.nc"
    )

    if os.path.exists(filepath):

        size_mb = os.path.getsize(filepath) / (
            1024 * 1024
        )

        print(
            f"✓ {year}: {size_mb:.2f} MB"
        )

    else:

        print(
            f"✗ {year}: NOT FOUND"
        )

print("=" * 70)