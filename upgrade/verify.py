"""Verify engineering integration and frozen result preservation, never reevaluate 2025."""
import json
import urllib.request
import numpy as np
import config
from experiments.common import read_json,write_json,verify_baseline
from experiments.finalize import verify_final,FINAL_DIR
from preprocessing.data_utils import file_sha256
from upgrade.calibration import temperature_scale


def main():
    verify_baseline();verify_final()
    archived=config.ROOT/'docs/results/improvement/evaluation.json'
    if file_sha256(archived)!=file_sha256(FINAL_DIR/'evaluation.json'):
        raise ValueError('The previously saved final evaluation changed.')
    record=read_json(config.ROOT/'outputs/upgrade/http_smoke.json')
    bundle=read_json(config.FORECAST_DIR/record['forecast_id']/'bundle.json')
    if not all('2024-01-01'<=d<='2024-12-31' for d in bundle['input_dates']):
        raise ValueError('Engineering verification must use development dates.')
    scores=np.load(config.FORECAST_DIR/record['forecast_id']/'prediction.npy')
    temperature=read_json(config.ROOT/'outputs/upgrade/calibration_2024.json')['temperature']
    calibrated=temperature_scale(scores,temperature)
    if not np.isfinite(calibrated).all() or not np.allclose(calibrated.sum(-1),1,atol=1e-6):
        raise ValueError('Development temperature transformation failed.')
    checks={}
    for route in ['/api/system','/api/live/prediction']:
        with urllib.request.urlopen('http://127.0.0.1:8000'+route,timeout=30) as response:
            checks[route]=json.load(response)
    if checks['/api/system']['calibration']['active'] or checks['/api/system']['notifications']['external_delivery']:
        raise ValueError('Unvalidated calibration or external delivery was activated.')
    if checks['/api/live/prediction']['status']!='unavailable':
        raise ValueError('No configured live adapter exists; must not claim live predictions.')
    result={'baseline_verified':True,'frozen_training_contract_verified':True,
            'saved_2025_evaluation_sha256':file_sha256(FINAL_DIR/'evaluation.json'),'saved_2025_evaluation_preserved':True,
            'real_http_forecast_id':record['forecast_id'],'input_dates':bundle['input_dates'],
            'real_five_map_http_integration':True,'development_calibration_transform_finite_normalized':True,
            'production_calibration_changed':False,'new_independent_evaluation_performed':False,'system_endpoints':checks}
    write_json(config.ROOT/'outputs/upgrade/integration.json',result)
    print('Frozen baseline/results preserved; real five-map HTTP, development calibration and unavailable live-state integration verified.')

if __name__=='__main__':main()
