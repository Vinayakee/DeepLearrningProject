import os
import numpy as np
import xarray as xr

# ============================================================
# SETTINGS
# ============================================================

RAW_DIR = "data/raw/currents"
PROCESSED_DIR = "data/processed"

YEARS = [2021, 2022, 2023, 2024, 2025]

# ============================================================
# LOAD TARGET GRID AND DATES
# ============================================================

print("=" * 70)
print("CMEMS CURRENT PREPROCESSING")
print("=" * 70)

print("\nLoading target CHL grid...")

target_lat = np.load(
    os.path.join(
        PROCESSED_DIR,
        "latitude_kerala_2021_2025.npy"
    )
)

target_lon = np.load(
    os.path.join(
        PROCESSED_DIR,
        "longitude_kerala_2021_2025.npy"
    )
)

dates = np.loadtxt(
    os.path.join(
        PROCESSED_DIR,
        "dates_2021_2025.txt"
    ),
    dtype=str
)

print("Target latitude:", target_lat.shape)
print("Target longitude:", target_lon.shape)
print("Number of dates:", len(dates))

# ============================================================
# CREATE OUTPUT ARRAYS
# ============================================================

num_days = len(dates)
height = len(target_lat)
width = len(target_lon)

current_u = np.full(
    (num_days, height, width),
    np.nan,
    dtype=np.float32
)

current_v = np.full(
    (num_days, height, width),
    np.nan,
    dtype=np.float32
)

current_u_mask = np.zeros(
    (num_days, height, width),
    dtype=np.uint8
)

current_v_mask = np.zeros(
    (num_days, height, width),
    dtype=np.uint8
)

# Date lookup
date_to_index = {
    date: i
    for i, date in enumerate(dates)
}

# ============================================================
# PROCESS EACH YEAR
# ============================================================

processed_files = 0
failed_files = 0

for year in YEARS:

    filename = f"currents_{year}.nc"

    filepath = os.path.join(
        RAW_DIR,
        filename
    )

    print("\n" + "=" * 70)
    print(f"PROCESSING CURRENTS: {year}")
    print("=" * 70)

    if not os.path.exists(filepath):

        print(
            f"✗ File not found: {filepath}"
        )

        failed_files += 1
        continue

    try:

        # ----------------------------------------------------
        # OPEN CMEMS FILE
        # ----------------------------------------------------

        ds = xr.open_dataset(
            filepath,
            engine="h5netcdf"
        )

        print(
            "Variables:",
            list(ds.data_vars)
        )

        # ----------------------------------------------------
        # CHECK REQUIRED VARIABLES
        # ----------------------------------------------------

        if "uo" not in ds.data_vars:
            raise ValueError(
                "uo variable not found"
            )

        if "vo" not in ds.data_vars:
            raise ValueError(
                "vo variable not found"
            )

        uo = ds["uo"]
        vo = ds["vo"]

        # ----------------------------------------------------
        # REMOVE DEPTH DIMENSION
        # ----------------------------------------------------
        #
        # The download requested the surface layer.
        # Keep the first available depth level.
        #

        if "depth" in uo.dims:

            print(
                "Depth levels:",
                uo.depth.values
            )

            uo = uo.isel(depth=0)
            vo = vo.isel(depth=0)

        # ----------------------------------------------------
        # SORT COORDINATES
        # ----------------------------------------------------

        uo = uo.sortby("latitude")
        uo = uo.sortby("longitude")

        vo = vo.sortby("latitude")
        vo = vo.sortby("longitude")

        # ----------------------------------------------------
        # INTERPOLATE TO CHL GRID
        # ----------------------------------------------------

        print(
            "Interpolating to 120 × 84 grid..."
        )

        uo_interp = uo.interp(
            latitude=target_lat,
            longitude=target_lon,
            method="linear"
        )

        vo_interp = vo.interp(
            latitude=target_lat,
            longitude=target_lon,
            method="linear"
        )

        # ----------------------------------------------------
        # SAVE EACH DAY
        # ----------------------------------------------------

        for time_value in uo_interp.time.values:

            date_string = str(
                time_value
            )[:10]

            if date_string not in date_to_index:
                continue

            index = date_to_index[
                date_string
            ]

            u_day = uo_interp.sel(
                time=time_value
            ).values

            v_day = vo_interp.sel(
                time=time_value
            ).values

            current_u[index] = (
                u_day.astype(np.float32)
            )

            current_v[index] = (
                v_day.astype(np.float32)
            )

            current_u_mask[index] = (
                ~np.isnan(u_day)
            ).astype(np.uint8)

            current_v_mask[index] = (
                ~np.isnan(v_day)
            ).astype(np.uint8)

        ds.close()

        processed_files += 1

        print(
            f"✓ {year} processed successfully"
        )

    except Exception as e:

        failed_files += 1

        print(
            f"✗ {year} FAILED"
        )

        print(
            "Error:",
            e
        )

# ============================================================
# SAVE PROCESSED DATA
# ============================================================

print("\n" + "=" * 70)
print("SAVING PROCESSED CURRENTS")
print("=" * 70)

os.makedirs(
    PROCESSED_DIR,
    exist_ok=True
)

np.save(
    os.path.join(
        PROCESSED_DIR,
        "current_u_2021_2025.npy"
    ),
    current_u
)

np.save(
    os.path.join(
        PROCESSED_DIR,
        "current_v_2021_2025.npy"
    ),
    current_v
)

np.save(
    os.path.join(
        PROCESSED_DIR,
        "current_u_mask_2021_2025.npy"
    ),
    current_u_mask
)

np.save(
    os.path.join(
        PROCESSED_DIR,
        "current_v_mask_2021_2025.npy"
    ),
    current_v_mask
)

# ============================================================
# DIAGNOSTICS
# ============================================================

print("\n" + "=" * 70)
print("FINAL CURRENT CHECK")
print("=" * 70)

print(
    "Current U shape:",
    current_u.shape
)

print(
    "Current V shape:",
    current_v.shape
)

print(
    "Current U mask:",
    current_u_mask.shape
)

print(
    "Current V mask:",
    current_v_mask.shape
)

# Missing percentages

u_missing = (
    np.isnan(current_u).sum()
    / current_u.size
) * 100

v_missing = (
    np.isnan(current_v).sum()
    / current_v.size
) * 100

print(
    f"Current U missing: "
    f"{u_missing:.2f}%"
)

print(
    f"Current V missing: "
    f"{v_missing:.2f}%"
)

# Valid values

u_valid = current_u[
    ~np.isnan(current_u)
]

v_valid = current_v[
    ~np.isnan(current_v)
]

if len(u_valid) > 0:

    print(
        f"Current U min: "
        f"{u_valid.min():.4f} m/s"
    )

    print(
        f"Current U max: "
        f"{u_valid.max():.4f} m/s"
    )

    print(
        f"Current U mean: "
        f"{u_valid.mean():.4f} m/s"
    )

if len(v_valid) > 0:

    print(
        f"Current V min: "
        f"{v_valid.min():.4f} m/s"
    )

    print(
        f"Current V max: "
        f"{v_valid.max():.4f} m/s"
    )

    print(
        f"Current V mean: "
        f"{v_valid.mean():.4f} m/s"
    )

print(
    "Processed files:",
    processed_files
)

print(
    "Failed files:",
    failed_files
)

# ============================================================
# FINAL RESULT
# ============================================================

if (
    processed_files == 5
    and failed_files == 0
    and current_u.shape == (1826, 120, 84)
    and current_v.shape == (1826, 120, 84)
):

    print("\n" + "=" * 70)
    print("✓ CURRENT PREPROCESSING COMPLETE!")
    print("✓ U CURRENT ALIGNED TO CHL GRID")
    print("✓ V CURRENT ALIGNED TO CHL GRID")
    print("✓ 1826 DAYS CONFIRMED")
    print("✓ 120 × 84 GRID CONFIRMED")
    print("=" * 70)

else:

    print("\n" + "=" * 70)
    print("✗ SOMETHING NEEDS TO BE CHECKED")
    print("=" * 70)