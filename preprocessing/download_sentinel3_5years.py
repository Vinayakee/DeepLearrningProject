import os
import time
import requests
import numpy as np

from dotenv import load_dotenv
from rasterio.io import MemoryFile
from rasterio.warp import reproject, Resampling
from datetime import datetime, timedelta


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

CLIENT_ID = os.getenv("CDSE_CLIENT_ID")
CLIENT_SECRET = os.getenv("CDSE_CLIENT_SECRET")

if not CLIENT_ID or not CLIENT_SECRET:
    raise ValueError(
        "CDSE_CLIENT_ID or CDSE_CLIENT_SECRET is missing from .env"
    )


# Kerala project region
BBOX = [74.0, 8.0, 77.5, 13.0]

# Sentinel-3 request resolution
SOURCE_WIDTH = 140
SOURCE_HEIGHT = 200

# Our existing project grid
TARGET_WIDTH = 84
TARGET_HEIGHT = 120

# Project dates
START_DATE = datetime(2021, 1, 1)
END_DATE = datetime(2025, 12, 31)

PROCESS_URL = "https://sh.dataspace.copernicus.eu/process/v1"

TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/"
    "auth/realms/CDSE/protocol/openid-connect/token"
)

OUTPUT_DIR = "data/processed"

OLCI_FILE = os.path.join(
    OUTPUT_DIR,
    "sentinel3_chl_2021_2025.npy"
)

MASK_FILE = os.path.join(
    OUTPUT_DIR,
    "sentinel3_mask_2021_2025.npy"
)

DATES_FILE = os.path.join(
    OUTPUT_DIR,
    "sentinel3_dates_2021_2025.txt"
)

COVERAGE_FILE = os.path.join(
    OUTPUT_DIR,
    "sentinel3_daily_coverage.csv"
)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# GET ACCESS TOKEN
# ============================================================

print("Authenticating with CDSE...")

token_response = requests.post(
    TOKEN_URL,
    data={
        "grant_type": "client_credentials",
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
    },
    timeout=60,
)

if token_response.status_code != 200:
    print(token_response.text)
    raise RuntimeError("CDSE authentication failed")

ACCESS_TOKEN = token_response.json()["access_token"]

print("OAuth authentication: SUCCESS")


# ============================================================
# REQUEST HEADERS
# ============================================================

headers = {
    "Authorization": f"Bearer {ACCESS_TOKEN}",
    "Content-Type": "application/json",
    "Accept": "image/tiff",
}


# ============================================================
# EVALSCRIPT
# ============================================================

evalscript = """
//VERSION=3

function setup() {

    return {
        input: [
            {
                bands: ["CHL_OC4ME", "dataMask"],
                units: [
                    "LOG10_MILLIGRAMS_M3",
                    "DN"
                ]
            }
        ],

        output: {
            bands: 2,
            sampleType: "FLOAT32"
        }
    };
}


function evaluatePixel(sample) {

    return [
        sample.CHL_OC4ME,
        sample.dataMask
    ];
}
"""


# ============================================================
# GENERATE ALL DATES
# ============================================================

dates = []

current = START_DATE

while current <= END_DATE:

    dates.append(
        current.strftime("%Y-%m-%d")
    )

    current += timedelta(days=1)


print()
print("Total dates:", len(dates))
print(
    f"Period: {dates[0]} → {dates[-1]}"
)


# ============================================================
# PREALLOCATE ARRAYS
# ============================================================

chl_data = np.full(
    (
        len(dates),
        TARGET_HEIGHT,
        TARGET_WIDTH
    ),
    np.nan,
    dtype=np.float32
)

valid_mask = np.zeros(
    (
        len(dates),
        TARGET_HEIGHT,
        TARGET_WIDTH
    ),
    dtype=np.uint8
)


# ============================================================
# COVERAGE RESULTS
# ============================================================

coverage_results = []


# ============================================================
# DOWNLOAD EACH DAY
# ============================================================

for index, date in enumerate(dates):

    print()
    print(
        f"[{index + 1}/{len(dates)}] {date}"
    )

    start_time = f"{date}T00:00:00Z"
    end_time = f"{date}T23:59:59Z"


    request_body = {

        "input": {

            "bounds": {

                "bbox": BBOX,

                "properties": {
                    "crs":
                    "http://www.opengis.net/def/crs/OGC/1.3/CRS84"
                }
            },


            "data": [

                {

                    "type":
                    "sentinel-3-olci-l2",


                    "dataFilter": {

                        "timeRange": {

                            "from": start_time,

                            "to": end_time
                        },

                        "maxCloudCoverage": 80,

                        "mosaickingOrder": "leastCC"
                    },


                    "processing": {

                        "upsampling":
                        "BILINEAR"
                    }
                }
            ]
        },


        "output": {

            "width": SOURCE_WIDTH,

            "height": SOURCE_HEIGHT,

            "responses": [

                {

                    "identifier":
                    "default",

                    "format": {

                        "type":
                        "image/tiff"
                    }
                }
            ]
        },


        "evalscript":
        evalscript
    }


    # ========================================================
    # REQUEST WITH RETRIES
    # ========================================================

    response = None

    for attempt in range(4):

        try:

            response = requests.post(
                PROCESS_URL,
                headers=headers,
                json=request_body,
                timeout=180
            )

            if response.status_code == 200:
                break

            print(
                f"Request failed "
                f"(attempt {attempt + 1}/4): "
                f"{response.status_code}"
            )

            if response.status_code == 401:

                print("Refreshing authentication token...")

                token_response = requests.post(
                    TOKEN_URL,
                    data={
                        "grant_type":
                        "client_credentials",

                        "client_id":
                        CLIENT_ID,

                        "client_secret":
                        CLIENT_SECRET,
                    },
                    timeout=60,
                )

                if token_response.status_code == 200:

                    ACCESS_TOKEN = (
                        token_response
                        .json()["access_token"]
                    )

                    headers["Authorization"] = (
                        f"Bearer {ACCESS_TOKEN}"
                    )

            time.sleep(5 * (attempt + 1))

        except Exception as e:

            print("Request error:", e)

            time.sleep(
                5 * (attempt + 1)
            )


    # ========================================================
    # FAILED REQUEST
    # ========================================================

    if response is None or response.status_code != 200:

        print("No data returned.")

        coverage_results.append(
            {
                "date": date,
                "status": "FAILED",
                "coverage_percent": 0.0
            }
        )

        continue


    # ========================================================
    # READ TIFF FROM MEMORY
    # ========================================================

    try:

        with MemoryFile(
            response.content
        ) as memfile:

            with memfile.open() as src:

                chl_source = src.read(1)

                mask_source = src.read(2)

                source_transform = src.transform

                source_crs = src.crs


        # ----------------------------------------------------
        # VALIDITY MASK
        # ----------------------------------------------------

        source_valid = (
            (mask_source >= 0.5)
            & np.isfinite(chl_source)
        )


        # ----------------------------------------------------
        # INVALID VALUES → NaN
        # ----------------------------------------------------

        chl_source = chl_source.astype(
            np.float32
        )

        chl_source[
            ~source_valid
        ] = np.nan


        # ----------------------------------------------------
        # TARGET TRANSFORM
        #
        # Same geographic bounding box but 120 × 84
        # ----------------------------------------------------

        min_lon, min_lat, max_lon, max_lat = BBOX

        target_pixel_width = (
            (max_lon - min_lon)
            / TARGET_WIDTH
        )

        target_pixel_height = (
            (max_lat - min_lat)
            / TARGET_HEIGHT
        )

        from rasterio.transform import from_origin

        target_transform = from_origin(
            min_lon,
            max_lat,
            target_pixel_width,
            target_pixel_height
        )


        # ----------------------------------------------------
        # RESAMPLE CHL
        # ----------------------------------------------------

        chl_target = np.full(
            (
                TARGET_HEIGHT,
                TARGET_WIDTH
            ),
            np.nan,
            dtype=np.float32
        )


        reproject(

            source=chl_source,

            destination=chl_target,

            src_transform=source_transform,

            src_crs=source_crs,

            dst_transform=target_transform,

            dst_crs="EPSG:4326",

            src_nodata=np.nan,

            dst_nodata=np.nan,

            resampling=Resampling.bilinear
        )


        # ----------------------------------------------------
        # RESAMPLE VALIDITY MASK
        # ----------------------------------------------------

        mask_target_float = np.zeros(
            (
                TARGET_HEIGHT,
                TARGET_WIDTH
            ),
            dtype=np.float32
        )


        reproject(

            source=source_valid.astype(
                np.float32
            ),

            destination=mask_target_float,

            src_transform=source_transform,

            src_crs=source_crs,

            dst_transform=target_transform,

            dst_crs="EPSG:4326",

            src_nodata=0,

            dst_nodata=0,

            resampling=Resampling.nearest
        )


        mask_target = (
            mask_target_float >= 0.5
        )


        # ----------------------------------------------------
        # IMPORTANT:
        # CHL_OC4ME is LOG10(mg/m3)
        #
        # Convert:
        #
        # log10(CHL) → CHL
        # ----------------------------------------------------

        chl_linear = np.full_like(
            chl_target,
            np.nan
        )


        valid = (
            mask_target
            & np.isfinite(chl_target)
        )


        chl_linear[valid] = (
            10.0 ** chl_target[valid]
        )


        # ----------------------------------------------------
        # STORE
        # ----------------------------------------------------

        chl_data[
            index
        ] = chl_linear

        valid_mask[
            index
        ] = mask_target.astype(
            np.uint8
        )


        # ----------------------------------------------------
        # COVERAGE
        # ----------------------------------------------------

        total_pixels = (
            TARGET_HEIGHT
            * TARGET_WIDTH
        )

        valid_pixels = int(
            np.sum(mask_target)
        )

        coverage = (
            valid_pixels
            / total_pixels
        ) * 100


        coverage_results.append(
            {
                "date": date,
                "status": "SUCCESS",
                "coverage_percent":
                coverage
            }
        )


        print(
            f"Coverage: {coverage:.2f}%"
        )

        if valid_pixels > 0:

            values = chl_linear[
                valid
            ]

            print(
                "Mean CHL:",
                f"{np.mean(values):.4f}",
                "mg/m³"
            )

            print(
                "Max CHL:",
                f"{np.max(values):.4f}",
                "mg/m³"
            )


    except Exception as e:

        print(
            "Processing error:",
            e
        )

        coverage_results.append(
            {
                "date": date,
                "status": "PROCESSING_ERROR",
                "coverage_percent": 0.0
            }
        )


    # ========================================================
    # SMALL DELAY
    # ========================================================

    time.sleep(0.2)


# ============================================================
# SAVE DATA
# ============================================================

print()
print("=" * 70)
print("SAVING SENTINEL-3 DATA")
print("=" * 70)


np.save(
    OLCI_FILE,
    chl_data
)

np.save(
    MASK_FILE,
    valid_mask
)


with open(
    DATES_FILE,
    "w"
) as f:

    for date in dates:

        f.write(
            date + "\n"
        )


with open(
    COVERAGE_FILE,
    "w"
) as f:

    f.write(
        "date,status,coverage_percent\n"
    )

    for r in coverage_results:

        f.write(
            f"{r['date']},"
            f"{r['status']},"
            f"{r['coverage_percent']:.4f}\n"
        )


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("Sentinel-3 processing complete.")

print()
print("CHL shape:")
print(chl_data.shape)

print()
print("Mask shape:")
print(valid_mask.shape)

overall_coverage = (
    np.mean(valid_mask)
    * 100
)

print()
print(
    f"Overall pixel coverage: "
    f"{overall_coverage:.2f}%"
)

print()
print("Saved:")
print(OLCI_FILE)
print(MASK_FILE)
print(DATES_FILE)
print(COVERAGE_FILE)