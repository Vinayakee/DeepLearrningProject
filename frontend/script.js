// ============================================================
// HAB EARLY WARNING DASHBOARD
// ============================================================

let currentDay = 1;
let predictionData = null;
let demoMode = false;
let requestSerial = 0;
let mapView = "forecast";
let mapZoom = 1;
let reloadRequest = 0;

function setForecastControls(available) {
    document.querySelectorAll(".day-button").forEach(button => {
        button.disabled = !available;
        button.title = available ? "Show this forecast day" : "Load a forecast to enable day switching";
    });
    ["map-zoom-in", "map-zoom-out", "map-zoom-reset"].forEach(id => {
        const button = document.getElementById(id);
        if (button) button.disabled = !available;
    });
}

async function fetchWithTimeout(url, options = {}, timeoutMs = 60000) {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), timeoutMs);
    try {
        return await fetch(url, {...options, signal: controller.signal});
    } catch (error) {
        if (error.name === "AbortError") throw new Error("The server is taking too long. Please retry, or check that the local server is running.");
        throw error;
    } finally {
        clearTimeout(timeout);
    }
}

function forecastError(message) {
    return /outdated|historical_demo|future-dated/i.test(message)
        ? "Current forecasts are unavailable because the observations are old. Click Load historical example, or enable Historical demonstration for your selected date."
        : message;
}

function setText(id, value) {
    const element = document.getElementById(id);
    if (element) element.textContent = value;
}

function clearForecast(message) {
    requestSerial += 1;
    currentDay = 1;
    setText("forecast-day", "Unavailable");
    predictionData = null;
    setForecastControls(false);
    mapView = "forecast";
    mapZoom = 1;
    const view = document.getElementById("map-view");
    if (view) { view.value = "forecast"; view.disabled = true; }
    const download = document.getElementById("forecast-export");
    if (download) { download.hidden = true; download.removeAttribute("href"); }
    setText("map-context-note", "");
    const map = document.getElementById("prediction-map");
    map.hidden = true;
    map.removeAttribute("src");
    const link = document.getElementById("map-link");
    link.hidden = true;
    link.removeAttribute("href");
    ["max-probability", "high-risk-pixels", "risk-coverage", "forecast-date", "latest-input-date", "model-threshold", "model-grid", "supported-pixels"].forEach(id => setText(id, "--"));
    setText("elevated-summary", "Elevated predictions unavailable.");
    setText("coverage-details", "");
    setText("forecast-period", "Unavailable");
    setText("generated-at", "");
    setText("warning-level", "UNAVAILABLE");
    document.getElementById("warning-level").className = "";
    setText("observed-status", "Unavailable");
    setText("warning-rule", "Warning parameters unavailable.");
    setText("observed-coverage", "No reliable observation summary loaded.");
    document.getElementById("environment-values").replaceChildren();
    setText("system-status", "Results unavailable");
    document.querySelector(".status").classList.add("unavailable");
    setText("data-notice", message);
    document.getElementById("data-notice").classList.add("attention");
}

function updateSystemInfo(data) {
    setForecastControls(true);
    const archived = data.maps?.[0]?.startsWith("/forecasts/");
    const view = document.getElementById("map-view");
    if (view) view.disabled = !archived;
    const download = document.getElementById("forecast-export");
    if (download) {
        download.hidden = !archived;
        if (archived) {
            download.href = `/api/forecasts/${encodeURIComponent(data.forecast_id)}/summary.csv?demo=${demoMode}`;
            download.download = `forecast-${data.forecast_id}.csv`;
        }
    }
    const demonstration = !data.operational_warnings_available;
    const historical = data.historical_demo || data.freshness.state !== "fresh";
    setText("system-status", demonstration ? (historical ? "Historical prototype" : "Research prototype") : "Forecast available");
    document.querySelector(".status").classList.toggle("unavailable", demonstration);
    const coverage = data.data_quality.chlorophyll_mean_input_coverage_percent;
    setText("data-notice", `${demonstration ? (historical ? "DEMONSTRATION — these are not current coastal warnings. " : "RESEARCH PROTOTYPE — regional warnings are not operationally validated. ") : "Prototype forecast. "}${data.data_quality.state !== "observed" ? "Insufficient data for reliable prediction. Exploratory maps only. " : ""}Input data: ${data.freshness.state}, ${data.freshness.age_days} days old. Mean chlorophyll input coverage: ${coverage.toFixed(2)}%. ${data.data_quality.state !== "observed" ? "Input coverage is insufficient; missing fields are flagged in the model." : ""}`);
    document.getElementById("data-notice").classList.toggle("attention", demonstration);
    setText("generated-at", `Prediction generated: ${new Date(data.generated_at_utc).toLocaleString()}`);
    const rule = data.decision_rule;
    setText("model-threshold", rule?.type === "score_ratios" ? `argmax(p × [${rule.scales.map(value => value.toFixed(4)).join(", ")}])` : rule?.type === "argmax" ? "Direct argmax" : `Strong score ≥ ${(data.threshold * 100).toFixed(1)}%`);
    updateCoverageDetails(data);
    setText("model-grid", `${data.height} × ${data.width}`);
    setText("warning-rule", `Regional warning rule: elevated when strong-class coverage reaches ${data.warning_parameters.regional_elevated_percent.toFixed(2)}%; strong at ${data.warning_parameters.regional_strong_percent.toFixed(2)}%. Derived from training daily fractions; demonstration thresholds.`);
    setText("observed-status", data.current_observed_risk.warning_level);
    setText("observed-coverage", `On ${data.input_dates.at(-1)}: ${data.current_observed_risk.valid_pixels} labelled cells (${data.current_observed_risk.coverage_percent.toFixed(2)}% support coverage). This is the latest observed chlorophyll proxy, not a live measurement.`);
    const table = document.createElement("table");
    const header = table.insertRow();
    ["Latest input variable", "Observed regional mean", "Support coverage"].forEach(value => { const th = document.createElement("th"); th.textContent = value; header.append(th); });
    data.environment.forEach(item => {
        const row = table.insertRow();
        [item.name.replaceAll("_", " "), item.observed_mean === null ? "Not observed" : `${item.observed_mean.toFixed(3)} ${item.unit}`, `${item.coverage_percent.toFixed(2)}%`].forEach(value => { row.insertCell().textContent = value; });
    });
    document.getElementById("environment-values").replaceChildren(table);
}

function updateCoverageDetails(data) {
    const quality = data.data_quality;
    if (!quality) return;
    const satellite = quality.features?.chlorophyll;
    const counts = quality.valid_observations === undefined ? "" : ` ${quality.valid_observations} valid chlorophyll cell-days; minimum ${quality.required_minimum_observations}.`;
    setText("coverage-details", `${quality.observation_status || quality.state}. Prediction: ${data.prediction_availability || quality.prediction_availability || "prototype"}.${counts} ${data.satellite_source || "MODIS"} mean input coverage: ${quality.chlorophyll_mean_input_coverage_percent.toFixed(2)}%. Observed on ${satellite?.observed_input_days ?? "unknown"} of 5 input days; required ${satellite?.required_minimum_observed_days ?? "unknown"}. Latest satellite input: ${satellite?.most_recent_observed_input_date || "missing"}. ${data.model_description?.seasonal_channels ? "Two month sine/cosine channels enabled." : ""} ${data.sentinel_fusion?.used ? "Experimental QC Sentinel gap filling; operational warnings disabled." : "No Sentinel observations used."} Raw model scores; no operational confidence estimate.`);
}

function showUnavailable(data) {
    const message = data.status === "stale" || (!demoMode && data.status === "unavailable")
        ? "No current forecast is available. Click Load historical example to explore real archived inputs and enable the five forecast days."
        : data.message || "Forecast unavailable. Load historical example to try a real archived forecast.";
    clearForecast(message);
    if (data.status === "insufficient_data") setText("warning-level", "INSUFFICIENT DATA");
    if (data.status === "stale") setText("warning-level", "STALE DATA");
    updateCoverageDetails(data);
    setText("latest-input-date", formatDate(data.input_dates?.at(-1) || data.freshness?.input_end_date));
    if (data.current_observed_risk) {
        setText("observed-status", data.current_observed_risk.warning_level);
        setText("observed-coverage", `${data.current_observed_risk.valid_pixels} observed proxy cells (${data.current_observed_risk.coverage_percent.toFixed(2)}% coverage). Insufficient inputs prevent a reliable forecast.`);
    }
}

async function generateForecast() {
    const button = document.getElementById("generate-forecast");
    if (button.disabled) return;
    button.disabled = true;
    button.textContent = "Generating…";
    setText("forecast-progress", "Running the model and drawing five maps…");
    const requestedDemo = demoMode;
    let serial;
    try {
        clearForecast("Generating a forecast from the selected actual inputs…");
        serial = requestSerial;
        const selected = document.getElementById("input-end-date").value;
        const response = await fetchWithTimeout("/api/predict", { method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({input_end_date: selected || null, historical_demo: requestedDemo}) });
        const result = await response.json();
        if (!response.ok) throw new Error(result.detail || "Prediction failed.");
        if (serial !== requestSerial || requestedDemo !== demoMode) return;
        if (result.status !== "success") { showUnavailable(result); return; }
        predictionData = result;
        updateSystemInfo(result);
        updateLatestInputDate();
        updateForecastPeriod();
        showDay(1);
    } catch (error) {
        if (serial === requestSerial) clearForecast(forecastError(error.message));
    } finally {
        button.disabled = false;
        button.textContent = "Generate forecast";
        setText("forecast-progress", "");
    }
}


// ============================================================
// FORMAT DATE
// ============================================================

function formatDate(dateString) {

    if (!dateString) {

        return "--";

    }

    const date = new Date(
        dateString + "T00:00:00"
    );

    if (isNaN(date.getTime())) {

        return "--";

    }

    return date.toLocaleDateString(
        "en-GB",
        {
            day: "2-digit",
            month: "short",
            year: "numeric"
        }
    );
}


// ============================================================
// LOAD DATA FROM BACKEND
// ============================================================

async function loadPredictionData() {
    refreshSystemStatus();
    const requestedDemo = demoMode;
    const serial = ++requestSerial;
    const loading = ++reloadRequest;
    const reload = document.getElementById("refresh-forecast");
    reload.disabled = true;
    reload.textContent = "Loading…";

    try {

        let response = await fetchWithTimeout(requestedDemo ? '/api/prediction?demo=true' : '/api/live/prediction', {}, 15000);
        if (!requestedDemo && response.ok) {
            const live = await response.clone().json();
            if (live.status === 'unavailable') response = await fetchWithTimeout('/api/prediction?demo=false', {}, 15000);
        }


        if (!response.ok) {

            const failure = await response.json().catch(() => ({}));
            throw new Error(failure.detail || `HTTP error: ${response.status}`);

        }


        const data = await response.json();
        if (serial !== requestSerial || requestedDemo !== demoMode) return;





        if (data.status !== "success") {



            showUnavailable(data);

            return;

        }


        predictionData = data;
        updateSystemInfo(data);





        // ----------------------------------------
        // Update latest input date
        // ----------------------------------------

        updateLatestInputDate();


        // ----------------------------------------
        // Update forecast period
        // ----------------------------------------

        updateForecastPeriod();


        // ----------------------------------------
        // Show first forecast day
        // ----------------------------------------

        showDay(1);

    }


    catch (error) {
        if (serial !== requestSerial) return;

        console.error(
            "Backend connection failed:",
            error
        );


        if (error instanceof TypeError) showBackendError();
        else clearForecast(forecastError(error.message));

    } finally {
        if (loading === reloadRequest) {
            reload.disabled = false;
            reload.textContent = "Reload saved forecast";
        }
    }

}


// ============================================================
// UPDATE LATEST INPUT DATE
// ============================================================

function updateLatestInputDate() {

    const latestInputDate =
        document.getElementById(
            "latest-input-date"
        );


    if (
        !latestInputDate ||
        !predictionData ||
        !predictionData.input_dates ||
        predictionData.input_dates.length === 0
    ) {

        return;

    }


    const latestDate =
        predictionData.input_dates[
            predictionData.input_dates.length - 1
        ];


    latestInputDate.textContent =
        formatDate(latestDate);

}


// ============================================================
// UPDATE FORECAST PERIOD
// ============================================================

function updateForecastPeriod() {

    const forecastPeriod =
        document.getElementById(
            "forecast-period"
        );


    if (
        !forecastPeriod ||
        !predictionData ||
        !predictionData.forecast_dates ||
        predictionData.forecast_dates.length === 0
    ) {

        return;

    }


    const firstDate =
        predictionData.forecast_dates[0];


    const lastDate =
        predictionData.forecast_dates[
            predictionData.forecast_dates.length - 1
        ];


    const first =
        new Date(
            firstDate + "T00:00:00"
        );


    const last =
        new Date(
            lastDate + "T00:00:00"
        );


    const firstFormatted =
        first.toLocaleDateString(
            "en-GB",
            {
                day: "2-digit",
                month: "short"
            }
        );


    const lastFormatted =
        last.toLocaleDateString(
            "en-GB",
            {
                day: "2-digit",
                month: "short",
                year: "numeric"
            }
        );


    forecastPeriod.textContent =
        `${firstFormatted} – ${lastFormatted}`;

}


// ============================================================
// SHOW FORECAST DAY
// ============================================================

function showDay(day) {
    if (!predictionData || day < 1 || day > predictionData.days) return;
    currentDay = day;


    // ----------------------------------------
    // Update map
    // ----------------------------------------

    const map =
        document.getElementById(
            "prediction-map"
        );


    if (map) {
        if (!predictionData || day < 1 || day > predictionData.days) return;
        map.hidden = false;
        const context = mapView === "observations";
        map.src = context ? `/api/forecasts/${encodeURIComponent(predictionData.forecast_id)}/observations.png?demo=${demoMode}` : predictionData.maps[day - 1];

        const link = document.getElementById("map-link");
        link.href = map.src;
        link.hidden = false;

        map.alt = context ? "Latest observed chlorophyll proxy labels and five-day input observation coverage" : `Chlorophyll anomaly prediction map for Day +${day}`;
        if (map.style) {
            map.style.width = `${mapZoom * 100}%`;
            map.style.maxHeight = mapZoom > 1 ? "none" : "600px";
        }
        setText("map-context-note", context ? "Observation context ends on the last input date and stays fixed as forecast days change. Grey cells have no observed label or lie outside support. These are chlorophyll proxies." : "Forecast classes and raw model scores for the selected day. Zoom to inspect the map; scroll to pan.");

    }


    // ----------------------------------------
    // Update day label
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
        predictionData &&
        predictionData.forecast_dates &&
        predictionData.forecast_dates.length >= day
    ) {

        const date =
            predictionData.forecast_dates[
                day - 1
            ];


        forecastDate.textContent =
            formatDate(date);

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


    if (
        !predictionData.daily_predictions ||
        predictionData.daily_predictions.length < day
    ) {

        return;

    }


    const daily =
        predictionData.daily_predictions[
            day - 1
        ];


    if (!daily) {

        return;

    }


    setText("supported-pixels", daily.supported_pixels);
    setText("elevated-summary", `Day +${day}: ${daily.elevated_risk_pixels} elevated-risk pixels. Maximum raw elevated score: ${daily.max_elevated_probability === null || daily.max_elevated_probability === undefined ? "unavailable" : (daily.max_elevated_probability * 100).toFixed(1) + "%"}. ${daily.exploratory_warning_level ? "Regional warnings withheld because input coverage is insufficient." : "Scores are not calibrated confidence."}`);




    // ----------------------------------------
    // Maximum probability
    // ----------------------------------------

    const probability = daily.max_probability === null ? NaN : Number(daily.max_probability);


    const probabilityElement =
        document.getElementById(
            "max-probability"
        );


    if (
        probabilityElement &&
        !isNaN(probability)
    ) {

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
            `${daily.high_risk_pixels} (${daily.high_risk_percentage.toFixed(2)}%)`;

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
            `${daily.high_risk_percentage.toFixed(2)}%`;

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


        else if (warningLevel === "NORMAL") {

            warning.classList.add(
                "warning-normal"
            );

        }

    }

}


// ============================================================
// BACKEND ERROR
// ============================================================

function showBackendError() {
    clearForecast("Backend connection failed. Start the API and open /dashboard/ through it.");

    const warning =
        document.getElementById(
            "warning-level"
        );


    if (warning) {

        warning.classList.remove(
            "warning-normal",
            "warning-elevated",
            "warning-strong"
        );


        warning.textContent =
            "Backend Offline";

    }


    const forecastDate =
        document.getElementById(
            "forecast-date"
        );


    if (forecastDate) {

        forecastDate.textContent =
            "--";

    }


    const latestInputDate =
        document.getElementById(
            "latest-input-date"
        );


    if (latestInputDate) {

        latestInputDate.textContent =
            "--";

    }


    const forecastPeriod =
        document.getElementById(
            "forecast-period"
        );


    if (forecastPeriod) {

        forecastPeriod.textContent =
            "Unavailable";

    }

}


// ============================================================
// INITIALIZE
// ============================================================

document.addEventListener(
    "DOMContentLoaded",
    function () {
        setForecastControls(false);
        document.getElementById("map-view").addEventListener("change", event => {
            mapView = event.target.value;
            showDay(currentDay);
        });
        [["map-zoom-in", .5], ["map-zoom-out", -.5], ["map-zoom-reset", 0]].forEach(([id, step]) => {
            document.getElementById(id).addEventListener("click", () => {
                mapZoom = step === 0 ? 1 : Math.max(1, Math.min(3, mapZoom + step));
                if (predictionData) showDay(currentDay);
            });
        });
        demoMode = document.getElementById("historical-demo").checked;
        document.getElementById("historical-demo").addEventListener("change", event => {
            demoMode = event.target.checked;
            clearForecast("Loading the selected forecast mode…");
            loadPredictionData();
        });
        document.getElementById("generate-forecast").addEventListener("click", generateForecast);
        document.getElementById("refresh-forecast").addEventListener("click", loadPredictionData);
        const eligibleButton = document.getElementById("list-covered-dates");
        const eligibleSelect = document.getElementById("eligible-input-date");
        if (eligibleButton && eligibleSelect) {
            eligibleButton.addEventListener("click", async () => {
                eligibleButton.disabled = true;
                eligibleButton.textContent = "Checking dates…";
                try {
                    const response = await fetchWithTimeout("/api/covered-dates", {}, 20000);
                    if (!response.ok) throw new Error("Eligible dates could not be loaded. Check the server and retry.");
                    const result = await response.json();
                    const placeholder = document.createElement("option");
                    placeholder.value = "";
                    placeholder.textContent = "Choose an eligible input date";
                    const options = result.dates.map(item => {
                        const option = document.createElement("option");
                        option.value = item.input_end_date;
                        option.textContent = `${item.input_end_date} · ${item.mean_chlorophyll_coverage_percent.toFixed(1)}% coverage · ${item.chlorophyll_observed_days}/5 observed days`;
                        return option;
                    });
                    eligibleSelect.replaceChildren(placeholder, ...options);
                    eligibleSelect.disabled = options.length === 0;
                    setText("eligible-date-help", `${options.length} coverage-supported historical dates available (2024 demonstrations). Input eligibility does not establish HAB forecasting skill.`);
                } catch (error) {
                    setText("eligible-date-help", error.message);
                } finally {
                    eligibleButton.disabled = false;
                    eligibleButton.textContent = "Show eligible dates";
                }
            });
            eligibleSelect.addEventListener("change", event => {
                if (!event.target.value) return;
                ++requestSerial;
                document.getElementById("input-end-date").value = event.target.value;
                document.getElementById("historical-demo").checked = true;
                demoMode = true;
                clearForecast("Eligible historical date selected. Click Generate forecast.");
            });
        }
        document.getElementById("covered-date").addEventListener("click", async event => {
            const button = event.currentTarget;
            button.disabled = true;
            button.textContent = "Finding an example…";
            setText("forecast-progress", "Finding an archived date with enough observations…");
            const serial = ++requestSerial;
            try {
                const response = await fetchWithTimeout("/api/covered-dates", {}, 20000);
                if (!response.ok) throw new Error("Could not load covered dates.");
                const result = await response.json();
                if (serial !== requestSerial) return;
                const examples = result.dates.filter(item => item.input_end_date >= "2024-01-05" && item.input_end_date <= "2024-12-26");
                if (!examples.length) throw new Error("No coverage-supported 2024 input window is available.");
                const chosen = document.getElementById("input-end-date").value;
                const selected = new Date(chosen?.startsWith("2024-") ? chosen : "2024-02-15").getTime();
                const nearest = examples.reduce((best, item) => Math.abs(new Date(item.input_end_date).getTime() - selected) < Math.abs(new Date(best.input_end_date).getTime() - selected) ? item : best);
                document.getElementById("input-end-date").value = nearest.input_end_date;
                document.getElementById("historical-demo").checked = true;
                demoMode = true;
                await generateForecast();
            } catch (error) {
                if (serial === requestSerial) clearForecast(error.message);
            } finally {
                button.disabled = false;
                button.textContent = "Load historical example";
                setText("forecast-progress", "");
            }
        });
        document.getElementById("prediction-map").addEventListener("error", () => {
            if (mapView === "observations") {
                mapView = "forecast";
                document.getElementById("map-view").value = "forecast";
                showDay(currentDay);
                setText("map-context-note", "Observation context is unavailable for this forecast. Showing the forecast map.");
            } else clearForecast("Forecast map unavailable. Regenerate the forecast.");
        });
        fetch("/api/health").then(response => response.json()).then(data => {
            if (data.latest_data_date) {
                if (!document.getElementById("input-end-date").value) document.getElementById("input-end-date").value = data.latest_data_date;
                document.getElementById("input-end-date").max = data.latest_data_date;
            }
        }).catch(() => {});
        loadPredictionData();
        setInterval(() => {
            if (!demoMode && !document.getElementById("generate-forecast").disabled) loadPredictionData();
        }, 60000);

    }
);

function refreshSystemStatus() {
    fetch("/api/system").then(response => {
        if (!response.ok) throw new Error("Source status unavailable");
        return response.json();
    }).then(data => {
        const update = data.live_update;
        const source = update.sentinel_fusion?.source || data.satellite_sources.active;
        setText("source-update-status", `Satellite source: ${source}. Sentinel-3: ${update.sentinel_fusion?.state || data.satellite_sources.sentinel3.state}. Live update: ${update.state.replaceAll("_", " ")}. Last successful update: ${update.last_successful_update_utc || "none — static archive"}. Data age: ${update.freshness?.age_days ?? "unavailable"} days. Model: ${data.model.version}. Calibration tested: ${data.calibration.tested}; active: ${data.calibration.active}; operationally approved: ${data.calibration.operationally_approved}. External notifications disabled.`);
    }).catch(() => setText("source-update-status", "Source/update status unavailable. No live data refresh is assumed."));
}
