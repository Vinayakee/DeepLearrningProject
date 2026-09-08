// ============================================================
// HAB EARLY WARNING DASHBOARD
// ============================================================

let currentDay = 1;
let predictionData = null;


// ============================================================
// LOAD DATA FROM BACKEND
// ============================================================

async function loadPredictionData() {

    try {

        const response = await fetch(
            "http://127.0.0.1:8000/api/prediction"
        );

        const data = await response.json();

        console.log("Prediction data:", data);


        if (data.status !== "success") {

            console.error(data.message);
            return;

        }


        predictionData = data;


        console.log(
            "Backend connected successfully"
        );


        showDay(1);

    }

    catch (error) {

        console.error(
            "Backend connection failed:",
            error
        );


        const warning =
            document.getElementById(
                "warning-level"
            );


        if (warning) {

            warning.textContent =
                "Backend Offline";

        }

    }

}


// ============================================================
// SHOW FORECAST DAY
// ============================================================

function showDay(day) {

    currentDay = day;


    // ----------------------------------------
    // Update map
    // ----------------------------------------

    const map =
        document.getElementById(
            "prediction-map"
        );


    if (map) {

        map.src =
            `../outputs/maps/day_${day}.png`;

    }


    // ----------------------------------------
    // Update day
    // ----------------------------------------

    const forecastDay =
        document.getElementById(
            "forecast-day"
        );


    if (forecastDay) {

        forecastDay.textContent =
            `Day +${day}`;

    }


    // ----------------------------------------
    // Update forecast date
    // ----------------------------------------

    const forecastDate =
        document.getElementById(
            "forecast-date"
        );


    if (
        forecastDate &&
        predictionData.forecast_dates
    ) {

        const date =
            predictionData.forecast_dates[
                day - 1
            ];


        const formattedDate =
            new Date(
                date + "T00:00:00"
            ).toLocaleDateString(
                "en-GB",
                {
                    day: "2-digit",
                    month: "short",
                    year: "numeric"
                }
            );


        forecastDate.textContent =
            formattedDate;

    }


    // ----------------------------------------
    // Update active button
    // ----------------------------------------

    const buttons =
        document.querySelectorAll(
            ".day-button"
        );


    buttons.forEach(
        (button, index) => {

            if (index + 1 === day) {

                button.classList.add(
                    "active"
                );

            }

            else {

                button.classList.remove(
                    "active"
                );

            }

        }
    );


    // ----------------------------------------
    // Update prediction information
    // ----------------------------------------

    updatePredictionInfo(day);

}


// ============================================================
// UPDATE PREDICTION INFORMATION
// ============================================================

function updatePredictionInfo(day) {

    if (!predictionData) {

        return;

    }


    const daily =
        predictionData.daily_predictions[
            day - 1
        ];


    if (!daily) {

        return;

    }


    console.log(
        `Day +${day}:`,
        daily
    );


    // ----------------------------------------
    // Maximum probability
    // ----------------------------------------

    const probability =
        daily.max_probability;


    const probabilityElement =
        document.getElementById(
            "max-probability"
        );


    if (probabilityElement) {

        probabilityElement.textContent =
            `${(probability * 100).toFixed(1)}%`;

    }


    // ----------------------------------------
    // High-risk pixels
    // ----------------------------------------

    const riskPixels =
        document.getElementById(
            "high-risk-pixels"
        );


    if (riskPixels) {

        riskPixels.textContent =
            `${daily.high_risk_pixels} (${daily.high_risk_percentage}%)`;

    }
    // ----------------------------------------
// Regional risk coverage
// ----------------------------------------

    const riskCoverage =
        document.getElementById(
            "risk-coverage"
        );


    if (riskCoverage) {

        riskCoverage.textContent =
            `${daily.high_risk_percentage}%`;

    }


    // ----------------------------------------
    // Warning level
    // ----------------------------------------

    const warning =
        document.getElementById(
            "warning-level"
        );


    const warningLevel =
        daily.warning_level;


    if (warning) {

        warning.classList.remove(
            "warning-normal",
            "warning-elevated",
            "warning-strong"
        );


        warning.textContent =
            warningLevel;


        if (
            warningLevel ===
            "STRONG BLOOM RISK"
        ) {

            warning.classList.add(
                "warning-strong"
            );

        }

        else if (
            warningLevel ===
            "ELEVATED RISK"
        ) {

            warning.classList.add(
                "warning-elevated"
            );

        }

        else {

            warning.classList.add(
                "warning-normal"
            );

        }

    }

}


// ============================================================
// INITIALIZE
// ============================================================

document.addEventListener(
    "DOMContentLoaded",
    function () {

        loadPredictionData();

    }
);