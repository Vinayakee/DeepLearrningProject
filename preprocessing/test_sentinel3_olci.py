import os
import requests
import numpy as np
from dotenv import load_dotenv

# Load .env
load_dotenv()

CLIENT_ID = os.getenv("CDSE_CLIENT_ID")
CLIENT_SECRET = os.getenv("CDSE_CLIENT_SECRET")

if not CLIENT_ID or not CLIENT_SECRET:
    raise ValueError("CDSE_CLIENT_ID or CDSE_CLIENT_SECRET is missing from .env")

# ---------------------------------------------------------
# 1. Get OAuth access token
# ---------------------------------------------------------

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

print("Token status:", token_response.status_code)

if token_response.status_code != 200:
    print(token_response.text)
    raise RuntimeError("Failed to get CDSE access token")

access_token = token_response.json()["access_token"]

print("OAuth authentication: SUCCESS")


# ---------------------------------------------------------
# 2. Request Sentinel-3 OLCI CHL_OC4ME
# ---------------------------------------------------------

# Kerala region
bbox = [74.0, 8.0, 77.5, 13.0]

# Test date
start_date = "2025-01-16T00:00:00Z"
end_date = "2025-01-17T00:00:00Z"

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

request_body = {
    "input": {
        "bounds": {
            "bbox": bbox,
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
        "width": 140,
        "height": 200,
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


# ---------------------------------------------------------
# 3. Send request
# ---------------------------------------------------------

PROCESS_URL = "https://sh.dataspace.copernicus.eu/process/v1"

headers = {
    "Authorization": f"Bearer {access_token}",
    "Content-Type": "application/json",
    "Accept": "image/tiff"
}

print("Requesting Sentinel-3 OLCI data...")
print("Region:", bbox)
print("Date:", start_date)

response = requests.post(
    PROCESS_URL,
    headers=headers,
    json=request_body,
    timeout=180
)

print("Process API status:", response.status_code)


# ---------------------------------------------------------
# 4. Save result
# ---------------------------------------------------------

if response.status_code != 200:
    print(response.text)
    raise RuntimeError("Sentinel-3 Process API request failed")

output_file = "data/raw/sentinel3_test.tiff"

os.makedirs("data/raw", exist_ok=True)

with open(output_file, "wb") as f:
    f.write(response.content)

print()
print("SUCCESS!")
print("Saved:", output_file)
print("File size:", round(len(response.content) / 1024, 2), "KB")
print()

# ---------------------------------------------------------
# 5. Basic validation
# ---------------------------------------------------------

try:
    import rasterio

    with rasterio.open(output_file) as src:
        print("Raster width:", src.width)
        print("Raster height:", src.height)
        print("Raster bands:", src.count)

        chl = src.read(1)
        mask = src.read(2)

        valid = mask > 0
        values = chl[valid]

        print("Valid pixels:", len(values))

        if len(values) > 0:
            print("Minimum CHL:", np.nanmin(values))
            print("Maximum CHL:", np.nanmax(values))
            print("Mean CHL:", np.nanmean(values))

except Exception as e:
    print("Raster validation error:", e)