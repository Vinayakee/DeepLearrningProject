"""Small synthetic fixtures test contracts only; never enter scientific results."""
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from upgrade.satellite import quality_control,compare_and_fill,load_bundle,availability,COMMON_REJECT
from upgrade.calibration import temperature_scale,calibration_metrics,fit_temperature
from upgrade.sampling import sampling_distribution
from upgrade.field_records import load_verified_records,COLUMNS
from upgrade.notifications import eligible,local_outbox
from upgrade.live import prepare_inputs,update,current_status,observation_summary
import config


def qc_metadata():
    names=['WATER',*COMMON_REJECT,'OC4ME_FAIL','MEGLINT']
    return {'product_id':'unit_test_only','processing_baseline':'fixture','quality_reference':'fixture',
            'flag_masks':{n:1<<i for i,n in enumerate(names)},'reject_flags':names[1:],'units':'log10(mg/m3)',
            'crs':'EPSG:4326','native_qc_before_resampling':True}


class SatelliteTests(unittest.TestCase):
    @unittest.skipUnless((config.ROOT/'data/raw/Chlorophyll-a/AQUA_MODIS.20240201.L3m.DAY.CHL.chlor_a.4km.nc').exists(), 'Requires the actual local MODIS source product.')
    def test_real_modis_source_grid_units_and_cf_validity(self):
        from upgrade.modis import read_daily
        from preprocessing.data_utils import load_dates,load_grid
        dates=load_dates(config.PROCESSED_DIR);lat,lon,_=load_grid(config.PROCESSED_DIR,dates)
        values,valid,metadata=read_daily(config.ROOT/'data/raw/Chlorophyll-a/AQUA_MODIS.20240201.L3m.DAY.CHL.chlor_a.4km.nc',lat,lon)
        self.assertEqual(values.shape,(120,84));self.assertTrue(np.isnan(values[~valid]).all())
        self.assertTrue((values[valid]>=metadata['valid_min']).all());self.assertTrue((values[valid]<=metadata['valid_max']).all())

    def test_flags_and_units(self):
        metadata=qc_metadata();water=metadata['flag_masks']['WATER'];cloud=metadata['flag_masks']['CLOUD']
        values,valid,report=quality_control(np.array([0.,1.,2.,np.nan]),np.array([water,water|cloud,0,water],dtype=np.uint64),metadata)
        self.assertEqual(valid.tolist(),[True,False,False,False]);self.assertEqual(values[0],1.)
        self.assertEqual(report['valid_observations'],1)

    def test_missing_flag_policy_rejected(self):
        metadata=qc_metadata();metadata['reject_flags'].remove('OC4ME_FAIL')
        with self.assertRaises(ValueError):quality_control(np.ones(2),np.ones(2,dtype=np.uint64),metadata)

    def test_modern_flags_accepted_without_hardcoded_bits(self):
        metadata=qc_metadata();metadata['flag_masks']['CHLOR_A_FAIL']=metadata['flag_masks'].pop('OC4ME_FAIL')
        metadata['reject_flags']=[n.replace('OC4ME_FAIL','CHLOR_A_FAIL') for n in metadata['reject_flags']]
        values,valid,_=quality_control(np.ones(1),np.ones(1,dtype=np.uint64),metadata)
        self.assertTrue(valid[0]);self.assertEqual(values[0],10.)

    def test_negative_or_unknown_units_invalid(self):
        metadata=qc_metadata();metadata['units']='mg/m3'
        _,valid,_=quality_control(np.array([-1.,0.,2.]),np.ones(3,dtype=np.uint64),metadata)
        self.assertEqual(valid.tolist(),[False,False,True])
        metadata['units']='unknown'
        with self.assertRaises(ValueError):quality_control(np.ones(1),np.ones(1,dtype=np.uint64),metadata)

    def test_fill_preserves_modis_and_source_flags(self):
        modis=np.array([[2.,np.nan]]);sentinel=np.array([[3.,4.]])
        fused,valid,source,report=compare_and_fill(modis,np.isfinite(modis),sentinel,np.ones_like(sentinel,bool),np.ones((1,2),bool))
        np.testing.assert_equal(fused,[[2.,4.]]);np.testing.assert_equal(source,[[1,2]])
        self.assertEqual(report['additional_cell_days'],1);self.assertFalse(report['retained_for_training'])

    def test_coordinates_and_timestamps_required(self):
        with TemporaryDirectory() as folder:
            path=Path(folder);(path/'metadata.json').write_text(json.dumps(qc_metadata()))
            np.savez(path/'observations.npz',chlorophyll=np.zeros((1,1,1)),quality_flags=np.ones((1,1,1),dtype=np.uint64),latitude=[10.],longitude=[75.],timestamps=['2024-01-01T12:00:00Z'])
            self.assertEqual(load_bundle(path,[10.],[75.])[0][0],np.datetime64('2024-01-01'))
            with self.assertRaises(ValueError):load_bundle(path,[11.],[75.])
            self.assertFalse(availability(path)['quality_controlled'])


class CalibrationTests(unittest.TestCase):
    def test_identity_and_normalization(self):
        p=np.array([[.2,.3,.5],[.8,.1,.1]])
        np.testing.assert_allclose(temperature_scale(p,1),p)
        np.testing.assert_allclose(temperature_scale(p,.5).sum(-1),1)
        with self.assertRaises(ValueError):temperature_scale(p,0)

    def test_calibration_reduces_overconfidence_in_fixture(self):
        p=np.tile([.98,.01,.01],(30,1));y=np.tile([0,1,2],10)
        t=fit_temperature(y,p)
        self.assertLess(calibration_metrics(y,temperature_scale(p,t))['nll'],calibration_metrics(y,p)['nll'])


class SamplingAndFieldsTests(unittest.TestCase):
    def test_balanced_sampling_increases_rare_window_mass(self):
        counts=np.array([[100,0,0],[100,0,0],[20,10,10]])
        p=sampling_distribution(counts)
        self.assertAlmostEqual(p.sum(),1);self.assertGreater(p[2],p[0])
        with self.assertRaises(ValueError):sampling_distribution(np.array([[1,0,0]]))

    def test_verified_records_reject_known_holdout_and_missing_status(self):
        with TemporaryDirectory() as folder:
            path=Path(folder)/'field.csv'
            row=dict(zip(COLUMNS,['fixture','2024-01-01T00:00:00Z','10','75','true','fixture_species','fixture_method','fixture_reference','fixture_reviewer']))
            with path.open('w',newline='') as handle:
                writer=csv.DictWriter(handle,fieldnames=COLUMNS);writer.writeheader();writer.writerow(row)
            with self.assertRaises(ValueError):load_verified_records(path)
            self.assertEqual(len(load_verified_records(path,purpose='development')),1)
            row['confirmed_hab']=''
            with path.open('w',newline='') as handle:
                writer=csv.DictWriter(handle,fieldnames=COLUMNS);writer.writeheader();writer.writerow(row)
            with self.assertRaises(ValueError):load_verified_records(path,purpose='development')


class NotificationTests(unittest.TestCase):
    def bundle(self):
        return {'status':'success','operational_warning_validated':True,'operational_warnings_available':True,
                'historical_demo':False,'freshness':{'state':'fresh'},'data_quality':{'state':'observed'},
                'model_sha256':'fixture','forecast_dates':['2026-01-01'],'daily_predictions':[{'day':1,'warning_level':'STRONG BLOOM RISK'}]}

    def test_gate_and_idempotency(self):
        bundle=self.bundle()
        with TemporaryDirectory() as folder:
            self.assertTrue(eligible(bundle));self.assertEqual(len(local_outbox(bundle,folder)['entries']),1)
            self.assertEqual(len(local_outbox(bundle,folder)['entries']),0)
            for key,value in [('operational_warning_validated',False),('historical_demo',True),('status','insufficient_data')]:
                blocked={**bundle,key:value};self.assertFalse(eligible(blocked));self.assertEqual(local_outbox(blocked,folder)['state'],'blocked')

    def test_low_coverage_stale_normal_and_demo(self):
        bundle=self.bundle()
        with TemporaryDirectory() as folder:
            self.assertFalse(eligible({**bundle,'freshness':{'state':'stale'}}))
            self.assertEqual(local_outbox({**bundle,'data_quality':{'state':'low_coverage'}},folder,demo=True)['state'],'blocked')
            normal={**bundle,'daily_predictions':[{'day':1,'warning_level':'NORMAL'}]}
            self.assertEqual(local_outbox(normal,folder)['entries'],[])
            result=local_outbox({**bundle,'operational_warning_validated':False},folder,demo=True)
            self.assertTrue(result['entries'][0]['demo']);self.assertFalse(result['entries'][0]['external_sent'])


class LiveTests(unittest.TestCase):
    @unittest.skipUnless((config.SEQUENCE_DIR/'manifest.json').exists(), 'Requires real fitted environmental artifacts.')
    def test_transform_matches_real_shared_pipeline(self):
        from preprocessing.pipeline import EnvironmentalPipeline
        pipeline=EnvironmentalPipeline();where=int(np.flatnonzero(pipeline.dates==np.datetime64('2024-02-15'))[0]);indices=np.arange(where-4,where+1)
        with TemporaryDirectory() as folder:
            path=Path(folder)
            np.savez(path/'inputs.npz',values=np.stack([pipeline.values[n][indices] for n in config.CHANNEL_NAMES],-1),masks=np.stack([pipeline.masks[n][indices] for n in config.CHANNEL_NAMES],-1),dates=pipeline.dates[indices],latitude=pipeline.latitude,longitude=pipeline.longitude)
            metadata={'channel_order':list(config.CHANNEL_NAMES),'crs':'EPSG:4326','units':['mg/m3','degC','m/s','m/s','mm/day','m/s','m/s'],
                      'sources':{n:{'product_ids':['unit_contract_check'],'quality_reference':'unit_contract_check','native_qc_applied':True,'observation_end_utc':'2024-02-15T23:00:00Z','available_at_utc':'2024-02-15T23:30:00Z'} for n in config.CHANNEL_NAMES}}
            (path/'metadata.json').write_text(json.dumps(metadata))
            actual,dates,_=prepare_inputs(path,pipeline)
            np.testing.assert_allclose(actual,pipeline.transform(indices),atol=1e-6)
            from prediction.service import observed_summary
            calibration=json.loads((config.RUN_DIR/'calibration.json').read_text())
            observed,environment=observation_summary(path,pipeline,str(dates[-1]),calibration)
            expected_observed,expected_environment=observed_summary(pipeline,where,calibration)
            self.assertEqual(observed['valid_pixels'],expected_observed['valid_pixels'])
            self.assertEqual(environment,expected_environment)
            with patch('upgrade.live.ROOT',path/'output'),patch('upgrade.live.load_model') as model:
                result=update(path)
                self.assertEqual(result['state'],'stale_data');model.assert_not_called()
            metadata['sources']['chlorophyll']['available_at_utc']='2024-02-16T05:00:00Z'
            (path/'metadata.json').write_text(json.dumps(metadata))
            with patch('upgrade.live.ROOT',path/'output'),patch('upgrade.live.load_model') as model:
                self.assertEqual(update(path)['state'],'input_timing_invalid');model.assert_not_called()
            metadata['sources']['chlorophyll']['available_at_utc']='2024-02-15T23:30:00Z'
            metadata['units'][0]='unknown';(path/'metadata.json').write_text(json.dumps(metadata))
            with self.assertRaises(ValueError):prepare_inputs(path,pipeline)

    def test_missing_acquisition_and_failure_preserve_last_success(self):
        with TemporaryDirectory() as folder,patch('upgrade.live.ROOT',Path(folder)),patch.dict('os.environ',{'HAB_LIVE_ADAPTER':''}),patch('upgrade.live.verify_final'):
            previous={'last_successful_update_utc':'2026-01-01T00:00:00Z','latest_successful_bundle':'fixture'}
            (Path(folder)/'status.json').write_text(json.dumps(previous))
            result=update();self.assertEqual(result['state'],'not_configured');self.assertEqual(result['last_successful_update_utc'],previous['last_successful_update_utc'])
            result=update(Path(folder)/'absent');self.assertEqual(result['state'],'update_failed');self.assertFalse(result['live'])

    def test_new_api_metadata_and_live_unavailability(self):
        from fastapi.testclient import TestClient
        from backend.app import app
        with TestClient(app) as client:
            response=client.get('/api/system');self.assertEqual(response.status_code,200)
            self.assertFalse(response.json()['calibration']['active'])
            self.assertFalse(response.json()['satellite_sources']['sentinel3']['fusion_active'])
            result=client.get('/api/live/prediction').json();self.assertEqual(result['status'],'unavailable');self.assertEqual(result['maps'],[])
            coverage=client.get('/api/covered-dates')
            if (config.SEQUENCE_DIR/'manifest.json').exists():
                dates=coverage.json()['dates']
                self.assertTrue(dates);self.assertTrue(all(row['input_end_date'].startswith('2024-') and row['mean_chlorophyll_coverage_percent']>=5 and row['chlorophyll_observed_days']>=2 for row in dates))
            else:
                self.assertEqual(coverage.status_code,503)

    def test_covered_window_helper_rejects_holdout_selection(self):
        from upgrade.coverage_windows import covered_windows
        with self.assertRaises(ValueError):covered_windows(None,2025)

    def test_corrupt_and_stale_live_status_fail_closed(self):
        with TemporaryDirectory() as folder,patch('upgrade.live.ROOT',Path(folder)):
            path=Path(folder)/'status.json';path.write_text('{invalid')
            self.assertFalse(current_status()['live']);self.assertEqual(current_status()['state'],'update_failed')
            path.write_text(json.dumps({'state':'fresh_research_prediction','live':True,'sources':{'chlorophyll':None}}))
            self.assertFalse(current_status()['live']);self.assertEqual(current_status()['state'],'update_failed')
            path.write_text(json.dumps({'state':'fresh_research_prediction','live':True,'input_end_date':'2024-02-15'}))
            self.assertFalse(current_status()['live']);self.assertEqual(current_status()['state'],'stale_data')

if __name__=='__main__':unittest.main()
