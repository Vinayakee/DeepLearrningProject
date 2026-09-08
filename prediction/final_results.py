import os


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# FINAL RESULTS
# ============================================================

results = """
============================================================
HAB EARLY WARNING SYSTEM
FINAL MODEL RESULTS
============================================================

DATASET
------------------------------------------------------------
Period                  : 2021 - 2025
Total days              : 1826
Spatial grid            : 120 × 84
Input time window       : 5 days
Forecast horizon        : 5 days
Environmental channels  : 7

Channels:
1. Chlorophyll-a
2. Sea Surface Temperature
3. Wind U
4. Wind V
5. Rainfall
6. Current U
7. Current V


MODEL
------------------------------------------------------------
Architecture            : ConvLSTM
Input shape              : 5 × 120 × 84 × 7
Output shape             : 5 × 120 × 84 × 3

Risk classes:
0 = Normal
1 = Elevated Risk
2 = Strong Bloom Risk

Strong Bloom threshold  : 0.50


TEST SET
------------------------------------------------------------
Test period              : 2025

Confusion Matrix:

                    Predicted
                 Normal  Elevated  Strong

Actual Normal     388621    10281    186934
Actual Elevated    36525      883     26408
Actual Strong      13123      443     18717


IMPORTANT METRICS
------------------------------------------------------------
Strong Bloom Recall    : 57.98%
Strong Bloom F1        : 14.16%
Macro F1               : 30.80%


INTERPRETATION
------------------------------------------------------------
The model detects a substantial portion of Strong Bloom Risk
pixels, achieving a Strong Bloom Recall of 57.98%.

However, the confusion matrix shows a relatively high number
of false positive Strong Bloom predictions. Therefore, the
current model should be considered an early-warning prototype
rather than a highly accurate operational HAB forecasting
system.

The model is designed to prioritize detection of potentially
high-risk bloom regions while providing a five-day forecast.


PREDICTION SYSTEM
------------------------------------------------------------
Input window            : Previous 5 days
Forecast                : Next 5 days
Grid resolution         : 120 × 84
Strong Bloom threshold  : 50%

Dashboard:
- Five-day forecast maps
- Maximum bloom probability
- High-risk pixel count
- Regional warning level
- Environmental information


============================================================
END OF RESULTS
============================================================
"""


# ============================================================
# SAVE
# ============================================================

output_file = os.path.join(
    OUTPUT_DIR,
    "final_model_results.txt"
)

with open(
    output_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(results)


print(results)

print(
    "\n✓ Final results saved:"
)

print(
    output_file
)