"""Existing API with shared inference, freshness checks and dashboard serving."""
from functools import lru_cache
from datetime import date
import json
import logging
from threading import Lock
from typing import Literal
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse, Response
from pydantic import BaseModel, ConfigDict
import config

app = FastAPI(title="HAB Early Warning API", version="2.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://127.0.0.1:8000", "http://localhost:8000"],
                   allow_methods=["GET", "POST"], allow_headers=["Content-Type"])
logger = logging.getLogger(__name__)
prediction_lock = Lock()


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    input_end_date: date | None = None
    historical_demo: bool = False


class BloomPredictionRequest(PredictionRequest):
    run_id: Literal['run_01','run_02','run_03'] | None = None


@lru_cache(maxsize=1)
def forecast_service(model_digest):
    from prediction.service import ForecastService
    return ForecastService()


@app.get("/")
def home():
    return {"message": "HAB Early Warning API is running", "status": "success", "dashboard": "/dashboard/"}


@app.get("/api/health")
def health():
    from operations.metadata import calibration_status, notification_status
    complete = False
    seasonal = False
    try:
        training = json.loads((config.RUN_DIR / "training_config.json").read_text())
        complete = bool(training.get("epochs_completed")) and not training.get("smoke", False)
        seasonal = bool(training.get("seasonal_features", False))
    except (OSError, ValueError):
        pass
    calibration = calibration_status()
    notifications = notification_status()
    return {"status": "success", "active_model": "final" if config.DEPLOYMENT else "baseline", "seasonal_features": seasonal, "model_file_available": config.MODEL_PATH.exists(),
            "training_complete": complete, "calibration_available": calibration['active'], "probability_calibration": calibration,
            "latest_data_date": latest_date(), "external_notifications": notifications['mode'], "notifications": notifications}


@app.get("/api/system")
def system_status():
    from operations.metadata import calibration_status, notification_status
    from upgrade.live import current_status
    from upgrade.satellite import availability
    update = current_status()
    return {"status": "success", "live_update": update,
            "satellite_sources": {"active": update.get('sentinel_fusion', {}).get('source', 'MODIS') if update.get('live') else 'MODIS', "sentinel3": availability(config.ROOT / "data/incoming_qc/sentinel3")},
            "calibration": calibration_status(),
            "model": {"version": "frozen_final" if config.DEPLOYMENT else "baseline", "independent_evaluation": "Saved 2025 results preserved; no repeat evaluation"},
            "notifications": notification_status()}


@app.get('/api/forecasts/{identifier}/summary.csv')
def forecast_export(identifier: str, demo: bool = False):
    from operations.forecast_views import load_bundle, summary_csv
    try:
        bundle = load_bundle(identifier, demo)
        return Response(summary_csv(bundle), media_type='text/csv',
                        headers={'Content-Disposition': f'attachment; filename="forecast-{identifier}.csv"', 'Cache-Control': 'no-store'})
    except FileNotFoundError:
        raise HTTPException(404, 'Forecast export is unavailable.')
    except (ValueError, KeyError, OSError):
        raise HTTPException(409, 'Forecast is stale, inconsistent or requires explicit demonstration mode.')


@app.get('/api/forecasts/{identifier}/observations.png')
def forecast_observations(identifier: str, demo: bool = False):
    from operations.forecast_views import load_bundle, observation_map
    try:
        return Response(observation_map(load_bundle(identifier, demo)), media_type='image/png', headers={'Cache-Control': 'no-store'})
    except FileNotFoundError:
        raise HTTPException(404, 'Archived observation context is unavailable.')
    except (ValueError, KeyError, OSError):
        raise HTTPException(409, 'Observation context is unavailable for this forecast or mode.')


@app.get("/api/covered-dates")
def covered_dates():
    from upgrade.coverage_windows import covered_windows
    from preprocessing.pipeline import EnvironmentalPipeline
    try:
        return {"status": "success", "period": "2024 development demonstrations", "dates": covered_windows(EnvironmentalPipeline()),
                "reason": "Input coverage only; no model scores or future target observations used. These dates still provide research predictions, not validated warnings."}
    except (OSError, ValueError, KeyError):
        raise HTTPException(503, "Coverage information is unavailable because fitted input artifacts are inconsistent.")


@app.get("/api/live/prediction")
def live_prediction():
    from upgrade.live import current_status, ROOT
    state = current_status()
    if state["state"] != "fresh_research_prediction" or not state.get("live"):
        status = 'stale' if state['state'] == 'stale_data' else 'insufficient_data' if state['state'] in ('missing_data', 'low_coverage', 'insufficient_data') else 'unavailable'
        return {"status": status, "warning_level": "STALE DATA" if status == 'stale' else "INSUFFICIENT DATA",
                "update": state, "data_quality": state.get('data_quality'), "freshness": state.get('freshness'),
                "message": state.get('reason', 'No valid fresh live prediction is available.'), "maps": [], "alerts": []}
    try:
        identifier = state["latest_successful_bundle"]
        if len(identifier) != 32 or any(c not in "0123456789abcdef" for c in identifier):
            raise ValueError("Invalid live bundle identifier")
        result = json.loads((ROOT / "bundles" / identifier / "bundle.json").read_text())
        from experiments.finalize import verify_final
        from preprocessing.data_utils import file_sha256
        verify_final()
        if result['model_sha256'] != file_sha256(config.MODEL_PATH):
            raise ValueError('Live prediction model differs from the active frozen checkpoint')
        for name, digest in result['array_sha256'].items():
            if name not in ('prediction.npy', 'risk_classes.npy') or file_sha256(ROOT / 'bundles' / identifier / name) != digest:
                raise ValueError('Live prediction arrays changed')
        if result.get('chlorophyll_provenance_sha256') and file_sha256(ROOT / 'bundles' / identifier / 'chlorophyll_provenance.npy') != result['chlorophyll_provenance_sha256']:
            raise ValueError('Live chlorophyll source provenance changed')
        result["alerts"] = []  # Operational validation remains disabled.
        return result
    except (OSError, ValueError, KeyError):
        return {"status": "unavailable", "maps": [], "alerts": [], "message": "No valid live bundle is available."}


def latest_date():
    try:
        return (config.PROCESSED_DIR / "dates_2021_2025.txt").read_text().splitlines()[-1].strip()
    except (OSError, IndexError):
        return None


@app.get("/api/prediction")
def get_prediction(demo: bool = False):
    from prediction.service import cached_forecast, data_freshness
    try:
        return cached_forecast(allow_demo=demo)
    except FileNotFoundError:
        return {"status": "unavailable", "message": "No complete forecast is available. Prepare data, train, evaluate, then generate a forecast.",
                "freshness": data_freshness(latest_date()) if latest_date() else None}
    except (ValueError, KeyError, OSError) as error:
        logger.warning("Forecast unavailable: %s", error)
        raise HTTPException(503, "Forecast artifacts are inconsistent. Regenerate the corrected forecast.")


@app.post("/api/predict")
def predict(request: PredictionRequest):
    if not prediction_lock.acquire(blocking=False):
        raise HTTPException(409, "A prediction is already running. Try again after it finishes.")
    try:
        from preprocessing.data_utils import file_sha256
        return forecast_service((file_sha256(config.MODEL_PATH), file_sha256(config.RUN_DIR / "calibration.json"))).generate(request.input_end_date, request.historical_demo)
    except FileNotFoundError:
        raise HTTPException(503, "Prepare corrected data, complete training and run evaluation before predicting.")
    except ValueError as error:
        raise HTTPException(422, str(error))
    except Exception:
        logger.exception("Prediction failed")
        raise HTTPException(503, "Prediction could not be completed. Check the backend log.")
    finally:
        prediction_lock.release()


@app.get("/api/alerts")
def alerts(demo: bool = False):
    result = get_prediction(demo)
    return {"status": result["status"], "alerts": result.get("alerts", []),
            "operational_warnings_available": result.get("operational_warnings_available", False)}


@app.get("/api/evaluation")
def evaluation():
    try:
        from preprocessing.data_utils import file_sha256
        result = json.loads((config.RUN_DIR / "evaluation.json").read_text())
        training = json.loads((config.RUN_DIR / "training_config.json").read_text())
        from prediction.service import validate_current_contract
        validate_current_contract(training)
        if (not training.get("epochs_completed") or result["model_sha256"] != file_sha256(config.MODEL_PATH)
                or result["calibration"]["preprocessing_sha256"] != file_sha256(config.SEQUENCE_DIR / "preprocessing.json")):
            return {"status": "unavailable", "message": "Evaluation differs from the current model/preprocessing. Rerun evaluation."}
        return {"status": "success", **result}
    except (OSError, ValueError, KeyError):
        return {"status": "unavailable", "message": "Evaluation has not been generated."}


@app.get('/api/hab/status')
def hab_status():
    from prediction.hab_service import status
    return status()


@app.get('/api/hab/evaluation')
def hab_evaluation():
    from prediction.hab_service import evaluation as hab_report
    try:
        return hab_report()
    except (OSError, ValueError, KeyError):
        raise HTTPException(503, 'The verified-HAB evaluation is unavailable or inconsistent.')


@app.post('/api/hab/predict')
def hab_predict(request: PredictionRequest):
    from prediction.hab_service import HABArtifactError
    if not prediction_lock.acquire(blocking=False):
        raise HTTPException(409, 'A prediction is already running.')
    try:
        from prediction.hab_service import generate
        return generate(request.input_end_date, request.historical_demo)
    except FileNotFoundError:
        raise HTTPException(503, 'Complete the verified-HAB preparation, training, calibration and evaluation workflow.')
    except HABArtifactError:
        raise HTTPException(503, 'The verified-HAB model or fitted artifacts are unavailable or inconsistent.')
    except ValueError as error:
        raise HTTPException(422, str(error))
    except Exception:
        logger.exception('Verified-HAB prediction failed')
        raise HTTPException(503, 'The verified-HAB forecast could not be generated.')
    finally:
        prediction_lock.release()


@app.get("/dashboard")
def dashboard_redirect():
    return RedirectResponse("/dashboard/")


@app.get('/api/bloom/status')
def bloom_status(run_id:Literal['run_01','run_02','run_03']|None=None):
    from prediction.bloom_service import status
    return status(run_id)


@app.get('/api/bloom/evaluation')
def bloom_evaluation(run_id:Literal['run_01','run_02','run_03']|None=None):
    from prediction.bloom_service import ready_run
    try:
        return {'status':'success',**ready_run(run_id)[3]}
    except (OSError,ValueError,KeyError):
        raise HTTPException(503,'Satellite bloom-event evaluation is unavailable or inconsistent.')


@app.post('/api/bloom/predict')
def bloom_predict(request:BloomPredictionRequest):
    from prediction.bloom_service import status,generate,ready_run
    if status(request.run_id)['status']!='ready_research_only':return status(request.run_id)
    if not prediction_lock.acquire(blocking=False):raise HTTPException(409,'A prediction is already running.')
    try:
        try:ready_run(request.run_id)
        except (OSError,ValueError,KeyError):
            raise HTTPException(503,'Satellite bloom-event artifacts are unavailable or inconsistent.')
        return generate(request.input_end_date,request.historical_demo,request.run_id)
    except ValueError as error:raise HTTPException(422,str(error))
    except (OSError,KeyError,RuntimeError):
        logger.exception('Satellite bloom-event inference failed')
        raise HTTPException(503,'Satellite bloom-event inference failed; check the server log.')
    finally:prediction_lock.release()


@app.get('/api/bloom/forecasts/{identifier}/summary.csv')
def bloom_export(identifier:str,demo:bool=False):
    from prediction.bloom_service import export_csv
    try:
        return Response(export_csv(identifier,demo),media_type='text/csv',headers={'Cache-Control':'no-store',
                        'Content-Disposition':f'attachment; filename="satellite-bloom-event-{identifier}.csv"'})
    except FileNotFoundError:raise HTTPException(404,'Satellite bloom-event export unavailable.')
    except (OSError,ValueError,KeyError):raise HTTPException(409,'Forecast is inconsistent or requires demonstration mode.')


app.mount("/dashboard", StaticFiles(directory=str(config.ROOT / "frontend"), html=True), name="dashboard")
app.mount("/forecasts", StaticFiles(directory=str(config.FORECAST_DIR), check_dir=False), name="forecasts")
app.mount("/live/bundles", StaticFiles(directory=str(config.ROOT / "outputs/live/bundles"), check_dir=False), name="live_maps")
app.mount('/bloom-forecasts',StaticFiles(directory=str(config.ROOT/'outputs/bloom_events/forecasts'),check_dir=False),name='bloom_event_maps')
