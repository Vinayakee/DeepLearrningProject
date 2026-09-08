import os
import subprocess
from datetime import date


# ============================================================
# SETTINGS
# ============================================================

START_YEAR = 2021
END_YEAR = 2025

OUTPUT_DIR = "data/raw/sst"

DATASET_ID = "METOFFICE-GLO-SST-L4-REP-OBS-SST"

# Kerala + surrounding Arabian Sea region
MIN_LON = 74.0
MAX_LON = 77.5
MIN_LAT = 8.0
MAX_LAT = 13.0


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


print("=" * 65)
print("5-YEAR SST DOWNLOAD")
print("=" * 65)

print(f"Years: {START_YEAR} - {END_YEAR}")
print(f"Region: {MIN_LAT}–{MAX_LAT} N")
print(f"        {MIN_LON}–{MAX_LON} E")
print()


# ============================================================
# DOWNLOAD YEAR BY YEAR
# ============================================================

successful = 0
failed = 0


for year in range(START_YEAR, END_YEAR + 1):

    start_date = f"{year}-01-01T00:00:00"
    end_date = f"{year}-12-31T23:59:59"

    output_file = (
        f"sst_{year}_kerala.nc"
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        output_file
    )

    print()
    print("=" * 65)
    print(f"DOWNLOADING SST: {year}")
    print("=" * 65)

    # --------------------------------------------------------
    # Skip if already downloaded
    # --------------------------------------------------------

    if os.path.exists(output_path):

        size_mb = (
            os.path.getsize(output_path)
            / (1024 * 1024)
        )

        print(
            f"Already exists: {output_file}"
        )

        print(
            f"Size: {size_mb:.2f} MB"
        )

        successful += 1

        continue


    # --------------------------------------------------------
    # Copernicus Marine command
    # --------------------------------------------------------

    command = [
        "copernicusmarine",
        "subset",

        "--dataset-id",
        DATASET_ID,

        "--variable",
        "analysed_sst",

        "--start-datetime",
        start_date,

        "--end-datetime",
        end_date,

        "--minimum-longitude",
        str(MIN_LON),

        "--maximum-longitude",
        str(MAX_LON),

        "--minimum-latitude",
        str(MIN_LAT),

        "--maximum-latitude",
        str(MAX_LAT),

        "--output-directory",
        OUTPUT_DIR,

        "--output-filename",
        output_file,
    ]


    # --------------------------------------------------------
    # RUN DOWNLOAD
    # --------------------------------------------------------

    try:

        result = subprocess.run(
            command,
            check=True
        )

        if os.path.exists(output_path):

            size_mb = (
                os.path.getsize(output_path)
                / (1024 * 1024)
            )

            print()
            print(
                f"SUCCESS: {year}"
            )

            print(
                f"File: {output_file}"
            )

            print(
                f"Size: {size_mb:.2f} MB"
            )

            successful += 1

        else:

            print(
                f"WARNING: {year} command completed "
                "but file was not found."
            )

            failed += 1


    except subprocess.CalledProcessError as e:

        print()
        print(
            f"FAILED: {year}"
        )

        print(
            "We will continue with the next year."
        )

        failed += 1


# ============================================================
# FINAL REPORT
# ============================================================

print()
print("=" * 65)
print("DOWNLOAD COMPLETE")
print("=" * 65)

print(
    f"Successful: {successful}"
)

print(
    f"Failed: {failed}"
)

print()

print("SST files:")

for year in range(START_YEAR, END_YEAR + 1):

    filename = (
        f"sst_{year}_kerala.nc"
    )

    path = os.path.join(
        OUTPUT_DIR,
        filename
    )

    if os.path.exists(path):

        size_mb = (
            os.path.getsize(path)
            / (1024 * 1024)
        )

        print(
            f"  {year}: {filename} "
            f"({size_mb:.2f} MB)"
        )

    else:

        print(
            f"  {year}: MISSING"
        )


print()
print("DONE!")