from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import os


app = FastAPI(title="HAB Early Warning API")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)


OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs"
)


@app.get("/")
def home():

    return {
        "message": "HAB Early Warning API is running",
        "status": "success"
    }


@app.get("/api/prediction")
def get_prediction():

    prediction_file = os.path.join(
        OUTPUT_DIR,
        "strong_bloom_probability.npy"
    )


    if not os.path.exists(prediction_file):

        return {
            "status": "error",
            "message": "Prediction file not found"
        }


    prediction = np.load(prediction_file)


    # Remove batch dimension if present
    if prediction.ndim == 4 and prediction.shape[0] == 1:

        prediction = prediction[0]


    # Expected shape: (5, 120, 84)
    if prediction.ndim != 3:

        return {
            "status": "error",
            "message": f"Unexpected prediction shape: {prediction.shape}"
        }


    days = prediction.shape[0]


    # Strong bloom probability threshold
    threshold = 0.50
        # ========================================================
    # FORECAST DATES
    # ========================================================

    dates_file = os.path.join(
        PROJECT_ROOT,
        "data",
        "processed",
        "dates_2021_2025.txt"
    )

    dates = np.loadtxt(
        dates_file,
        dtype=str
    )

    # Last 5 days are used as model input
    input_dates = dates[-5:].tolist()

    # Forecast starts on the day after the last input day
    last_input_date = np.datetime64(
        dates[-1],
        "D"
    )

    forecast_dates = [
        str(
            last_input_date +
            np.timedelta64(day, "D")
        )
        for day in range(1, 6)
    ]


    daily_predictions = []


    for day in range(prediction.shape[0]):

        day_prediction = prediction[day]


        max_probability = float(
            np.max(day_prediction)
        )


        mean_probability = float(
            np.mean(day_prediction)
        )


        # Count pixels above strong-bloom threshold
        high_risk_pixels = int(
            np.sum(day_prediction >= threshold)
        )


        total_pixels = (
            day_prediction.shape[0]
            * day_prediction.shape[1]
        )


        high_risk_percentage = (
            high_risk_pixels
            / total_pixels
        ) * 100


        # Regional warning level
        if high_risk_percentage >= 20:

            warning_level = "STRONG BLOOM RISK"

        elif high_risk_percentage >= 5:

            warning_level = "ELEVATED RISK"

        else:

            warning_level = "NORMAL"


        daily_predictions.append({

            "day": day + 1,

            "max_probability":
                round(max_probability, 4),

            "mean_probability":
                round(mean_probability, 4),

            "high_risk_pixels":
                high_risk_pixels,

            "high_risk_percentage":
                round(high_risk_percentage, 2),

            "warning_level":
                warning_level
        })


    return {

        "status": "success",

        "days": days,

        "height":
            int(prediction.shape[1]),

        "width":
            int(prediction.shape[2]),

        "threshold":
            threshold,

        "input_dates":
            input_dates,

        "forecast_dates":
            forecast_dates,

        "daily_predictions":
            daily_predictions
    }