import os
import requests
import numpy as np
import rasterio
from rasterio.io import MemoryFile
from dotenv import load_dotenv
from datetime import datetime

# ============================================================
# LOAD CREDENTIALS
# ============================================================

load_dotenv()

CLIENT_ID = os.getenv("CDSE_CLIENT_ID")
CLIENT_SECRET = os.getenv("CDSE_CLIENT_SECRET")

if not CLIENT_ID or not CLIENT_SECRET:
    raise ValueError(
        "CDSE_CLIENT_ID or CDSE_CLIENT_SECRET missing from .env"
    )


# ============================================================
# GET ACCESS TOKEN
# ============================================================

TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/"
    "auth/realms/CDSE/protocol/openid-connect/token"
)

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
    raise RuntimeError("Could not authenticate with CDSE")

ACCESS_TOKEN = token_response.json()["access_token"]

print("OAuth authentication: SUCCESS")


# ============================================================
# SETTINGS
# ============================================================

PROCESS_URL = "https://sh.dataspace.copernicus.eu/process/v1"

# Kerala coastline region used in our project
BBOX = [74.0, 8.0, 77.5, 13.0]

# Same test resolution that successfully worked
WIDTH = 140
HEIGHT = 200

# Test four representative months per year
TEST_DATES = [
    "2021-01-15",
    "2021-04-15",
    "2021-07-15",
    "2021-10-15",

    "2022-01-15",
    "2022-04-15",
    "2022-07-15",
    "2022-10-15",

    "2023-01-15",
    "2023-04-15",
    "2023-07-15",
    "2023-10-15",

    "2024-01-15",
    "2024-04-15",
    "2024-07-15",
    "2024-10-15",

    "2025-01-15",
    "2025-04-15",
    "2025-07-15",
    "2025-10-15",
]


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
                units: ["LOG10_MILLIGRAMS_M3", "DN"]
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
# HEADERS
# ============================================================

headers = {
    "Authorization": f"Bearer {ACCESS_TOKEN}",
    "Content-Type": "application/json",
    "Accept": "image/tiff",
}


# ============================================================
# RESULTS
# ============================================================

results = []


# ============================================================
# TEST EACH DATE
# ============================================================

for i, date in enumerate(TEST_DATES, start=1):

    print()
    print("=" * 60)
    print(f"Testing {i}/{len(TEST_DATES)}: {date}")
    print("=" * 60)

    start_date = f"{date}T00:00:00Z"

    # Request a 24-hour period
    end_datetime = datetime.strptime(
        date, "%Y-%m-%d"
    ).replace(hour=23, minute=59, second=59)

    end_date = end_datetime.strftime("%Y-%m-%dT%H:%M:%SZ")

    request_body = {
        "input": {
            "bounds": {
                "bbox": BBOX,
                "properties": {
                    "crs": "http://www.opengis.net/def/crs/OGC/1.3/CRS84"
                }
            },

            "data": [
                {
                    "type": "sentinel-3-olci-l2",

                    "dataFilter": {
                        "timeRange": {
                            "from": start_date,
                            "to": end_date
                        },

                        "maxCloudCoverage": 80,

                        "mosaickingOrder": "leastCC"
                    },

                    "processing": {
                        "upsampling": "BILINEAR"
                    }
                }
            ]
        },

        "output": {
            "width": WIDTH,
            "height": HEIGHT,

            "responses": [
                {
                    "identifier": "default",

                    "format": {
                        "type": "image/tiff"
                    }
                }
            ]
        },

        "evalscript": evalscript
    }

    try:

        response = requests.post(
            PROCESS_URL,
            headers=headers,
            json=request_body,
            timeout=180
        )

        print("API status:", response.status_code)

        if response.status_code != 200:
            print("No usable data.")
            print(response.text[:500])

            results.append({
                "date": date,
                "status": "FAILED",
                "valid_pixels": 0,
                "coverage_percent": 0,
                "mean_chl": np.nan
            })

            continue


        # ----------------------------------------------------
        # Read TIFF directly from memory
        # ----------------------------------------------------

        with MemoryFile(response.content) as memfile:

            with memfile.open() as src:

                chl = src.read(1)
                mask = src.read(2)

        valid = (
            (mask > 0)
            & np.isfinite(chl)
        )

        valid_values = chl[valid]

        total_pixels = WIDTH * HEIGHT

        valid_pixels = int(np.sum(valid))

        coverage = (
            valid_pixels / total_pixels
        ) * 100


        if valid_pixels > 0:
            mean_chl = float(np.mean(valid_values))
            min_chl = float(np.min(valid_values))
            max_chl = float(np.max(valid_values))
        else:
            mean_chl = np.nan
            min_chl = np.nan
            max_chl = np.nan


        print("Valid pixels:", valid_pixels)
        print(
            f"Coverage: {coverage:.2f}%"
        )

        if valid_pixels > 0:
            print(
                f"Mean CHL (log10): {mean_chl:.4f}"
            )

            print(
                f"Min CHL (log10): {min_chl:.4f}"
            )

            print(
                f"Max CHL (log10): {max_chl:.4f}"
            )


        results.append({
            "date": date,
            "status": "SUCCESS",
            "valid_pixels": valid_pixels,
            "coverage_percent": coverage,
            "mean_chl": mean_chl
        })


    except Exception as e:

        print("ERROR:", e)

        results.append({
            "date": date,
            "status": "ERROR",
            "valid_pixels": 0,
            "coverage_percent": 0,
            "mean_chl": np.nan
        })


# ============================================================
# SUMMARY
# ============================================================

print()
print()
print("#" * 70)
print("SENTINEL-3 OLCI COVERAGE SUMMARY")
print("#" * 70)

successful = [
    r for r in results
    if r["status"] == "SUCCESS"
]

print()
print(
    f"Successful dates: {len(successful)} / {len(results)}"
)

if successful:

    coverages = [
        r["coverage_percent"]
        for r in successful
    ]

    print(
        f"Average coverage: {np.mean(coverages):.2f}%"
    )

    print(
        f"Minimum coverage: {np.min(coverages):.2f}%"
    )

    print(
        f"Maximum coverage: {np.max(coverages):.2f}%"
    )


print()
print("DATE              STATUS       COVERAGE")
print("-" * 45)

for r in results:

    print(
        f"{r['date']}    "
        f"{r['status']:8s}     "
        f"{r['coverage_percent']:6.2f}%"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

os.makedirs("outputs", exist_ok=True)

output_file = "outputs/sentinel3_coverage_test.csv"

with open(output_file, "w") as f:

    f.write(
        "date,status,valid_pixels,coverage_percent,mean_chl\n"
    )

    for r in results:

        f.write(
            f"{r['date']},"
            f"{r['status']},"
            f"{r['valid_pixels']},"
            f"{r['coverage_percent']:.4f},"
            f"{r['mean_chl']}\n"
        )

print()
print("Results saved to:")
print(output_file)
