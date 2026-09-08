import copernicusmarine
import os

OUTPUT_DIR = "data/raw/currents"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("DOWNLOADING TEST OCEAN CURRENTS")
print("=" * 60)

result = copernicusmarine.subset(
    dataset_id="cmems_mod_glo_phy_my_0.083deg_P1D-m",

    variables=[
        "uo",
        "vo"
    ],

    minimum_longitude=74.0,
    maximum_longitude=77.5,

    minimum_latitude=8.0,
    maximum_latitude=13.0,

    start_datetime="2021-01-01",
    end_datetime="2021-01-31",

    # Surface layer
    minimum_depth=0,
    maximum_depth=1,

    output_directory=OUTPUT_DIR,
    output_filename="currents_2021_01.nc",

    overwrite=True
)

print("\n" + "=" * 60)
print("TEST DOWNLOAD COMPLETE")
print("=" * 60)

print(result)