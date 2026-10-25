"""Integration contracts use labeled test fixtures, never scientific observations."""
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from datetime import datetime, timezone, timedelta
from types import SimpleNamespace
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import config
from operations.acquisition import acquire, AcquisitionUnavailable
from operations.fusion import stage_fusion
from operations.provenance import validate_manifest
from operations.update import run_update
from operations.notifications import dispatch, send_email, send_sms
from operations.validation import grid_index, match_fields, evaluate_arrays, independent_report, load_forecast, load_fields
from operations.metadata import calibration_status
from upgrade.satellite import COMMON_REJECT


def staged(root, chlorophyll=True):
    dates = np.arange(np.datetime64('2024-02-11'), np.datetime64('2024-02-16'))
    values = np.ones((5,120,84,7), np.float32)
    masks = np.ones_like(values, np.uint8)
    if not chlorophyll:
        values[...,0] = np.nan; masks[...,0] = 0
    lat = np.linspace(13,8,120); lon = np.linspace(74,77.5,84)
    root.mkdir(parents=True)
    np.savez(root/'inputs.npz', values=values, masks=masks, dates=dates, latitude=lat, longitude=lon)
    metadata = {'channel_order':list(config.CHANNEL_NAMES),'units':['mg/m3','degC','m/s','m/s','mm/day','m/s','m/s'], 'crs':'EPSG:4326',
        'sources':{n:{'product_ids':['CONTRACT_FIXTURE_NOT_OBSERVATION'], 'quality_reference':'unit fixture only', 'native_qc_applied':True,
                     'observation_end_utc':'2024-02-15T12:00:00Z','available_at_utc':'2024-02-15T13:00:00Z'} for n in config.CHANNEL_NAMES}}
    (root/'metadata.json').write_text(json.dumps(metadata))
    pipeline = SimpleNamespace(latitude=lat,longitude=lon,ocean=np.ones((120,84),bool),
        stats={'channels':{n:{'median':0.,'scale':1.} for n in config.CHANNEL_NAMES}})
    return pipeline, metadata


def sentinel(root, pipeline, cloud=False, date='2024-02-15'):
    root.mkdir()
    flags = ['WATER',*COMMON_REJECT,'OC4ME_FAIL','MEGLINT']
    mapping = {n:1<<i for i,n in enumerate(flags)}
    metadata = {'product_id':'CONTRACT_SENTINEL_FIXTURE', 'processing_baseline':'fixture', 'quality_reference':'fixture only',
        'flag_masks':mapping, 'reject_flags':flags[1:], 'units':'mg/m3', 'crs':'EPSG:4326', 'native_qc_before_resampling':True,
        'observation_end_utc':date+'T12:00:00Z','available_at_utc':date+'T13:00:00Z'}
    np.savez(root/'observations.npz',chlorophyll=np.full((1,120,84),2.,np.float32),
        quality_flags=np.full((1,120,84),mapping['WATER']|(mapping['CLOUD'] if cloud else 0),np.uint64),
        latitude=pipeline.latitude,longitude=pipeline.longitude,timestamps=[date+'T11:00:00Z'])
    (root/'metadata.json').write_text(json.dumps(metadata))


class AcquisitionAndFusionTests(unittest.TestCase):
    def test_acquisition_config_missing_and_invalid(self):
        with patch.dict(os.environ,{'HAB_LIVE_INPUT_DIR':'','HAB_LIVE_ADAPTER':''}):
            with self.assertRaises(AcquisitionUnavailable): acquire()
        with TemporaryDirectory() as folder:
            path=Path(folder)/'stage';staged(path)
            with patch.dict(os.environ,{'HAB_LIVE_INPUT_DIR':str(path)}):self.assertEqual(acquire(),path.resolve())
            with self.assertRaises(AcquisitionUnavailable):acquire(path/'missing')
        with patch.dict(os.environ,{'HAB_LIVE_INPUT_DIR':'','HAB_LIVE_ADAPTER':'bad'}):
            with self.assertRaises(ValueError):acquire()

    def test_manifest_rejects_false_publication_and_product_ids(self):
        with TemporaryDirectory() as folder:
            root=Path(folder)/'stage';_,meta=staged(root)
            validate_manifest(root)
            meta['sources']['sst']['available_at_utc']='2024-02-15T11:00:00Z'
            (root/'metadata.json').write_text(json.dumps(meta))
            with self.assertRaises(ValueError):validate_manifest(root)
            meta['sources']['sst']['available_at_utc']='2024-02-15T13:00:00Z';meta['sources']['sst']['product_ids']='not-a-list'
            (root/'metadata.json').write_text(json.dumps(meta))
            with self.assertRaises(ValueError):validate_manifest(root)

    def test_valid_qc_sentinel_gap_filling_and_provenance(self):
        with TemporaryDirectory() as folder:
            root=Path(folder);pipe,_=staged(root/'stage',False);sentinel(root/'s3',pipe)
            path,report=stage_fusion(root/'stage',root/'s3',root/'fused',pipe)
            self.assertTrue(report['used']);self.assertEqual(report['source'],'MODIS + Sentinel-3')
            with np.load(path/'inputs.npz') as data:
                self.assertTrue((data['values'][-1,...,0]==2).all());self.assertTrue((data['masks'][:-1,...,0]==0).all())
            provenance=np.load(path/'chlorophyll_provenance.npy');self.assertTrue((provenance[-1]==2).all())
            self.assertFalse(report['operationally_approved']);validate_manifest(path)

    def test_existing_modis_never_replaced(self):
        with TemporaryDirectory() as folder:
            root=Path(folder);pipe,_=staged(root/'stage');sentinel(root/'s3',pipe)
            path,report=stage_fusion(root/'stage',root/'s3',root/'fused',pipe)
            self.assertEqual(path,root/'stage');self.assertFalse(report['used'])

    def test_invalid_missing_stale_and_cloud_sentinel_fallback(self):
        with TemporaryDirectory() as folder:
            root=Path(folder);pipe,_=staged(root/'stage',False)
            self.assertEqual(stage_fusion(root/'stage',None,root/'fused',pipe)[1]['state'],'sentinel3_unavailable')
            sentinel(root/'s3',pipe,cloud=True)
            self.assertEqual(stage_fusion(root/'stage',root/'s3',root/'fused',pipe)[1]['state'],'sentinel3_insufficient')
            meta=json.loads((root/'s3/metadata.json').read_text());meta['units']='bad'
            (root/'s3/metadata.json').write_text(json.dumps(meta))
            self.assertEqual(stage_fusion(root/'stage',root/'s3',root/'fused',pipe)[1]['state'],'sentinel3_rejected')
            sentinel(root/'old',pipe,date='2024-02-11')
            self.assertEqual(stage_fusion(root/'stage',root/'old',root/'fused',pipe)[1]['state'],'sentinel3_stale_or_unaligned')

    def test_future_sentinel_publication_rejected(self):
        with TemporaryDirectory() as folder:
            root=Path(folder);pipe,_=staged(root/'stage',False);sentinel(root/'s3',pipe)
            meta=json.loads((root/'s3/metadata.json').read_text());meta['available_at_utc']='2024-02-16T01:00:00Z'
            (root/'s3/metadata.json').write_text(json.dumps(meta))
            self.assertFalse(stage_fusion(root/'stage',root/'s3',root/'fused',pipe)[1]['used'])

    def test_worker_not_configured_and_overlap(self):
        with TemporaryDirectory() as folder,patch('upgrade.live.ROOT',Path(folder)),patch.dict(os.environ,{'HAB_LIVE_INPUT_DIR':'','HAB_LIVE_ADAPTER':''}):
            self.assertEqual(run_update()['state'],'not_configured')
            (Path(folder)/'update.lock').write_text('test')
            self.assertEqual(run_update()['state'],'update_in_progress')
            self.assertTrue((Path(folder)/'update.lock').exists())

    def test_worker_preserves_actual_coverage_state(self):
        with TemporaryDirectory() as folder,patch('upgrade.live.ROOT',Path(folder)/'output'):
            root=Path(folder)/'stage';staged(root)
            with patch('upgrade.live.update',return_value={'state':'insufficient_data','live':False,'data_quality':{'state':'low_coverage'}}):
                self.assertEqual(run_update(root)['state'],'low_coverage')


class FieldAndEvaluationTests(unittest.TestCase):
    def inputs(self):
        record={'record_id':'FIXTURE','date':'2026-02-02','latitude':10.,'longitude':75.,'confirmed_hab':True,
                'verification_method':'fixture','source_reference':'fixture'}
        bundle={'forecast_id':'fixture','forecast_dates':['2026-02-02'],'input_dates':['2026-02-01'],'data_quality':{'state':'observed'}}
        risk=np.array([[[2,0],[1,255]]],np.uint8)
        return record,(bundle,risk,np.array([10.,11.]),np.array([75.,76.]))

    def test_field_date_grid_and_class_matching(self):
        record,forecast=self.inputs();r=match_fields([record],[forecast])
        self.assertEqual(r['status'],'success');self.assertEqual(r['matches'][0]['model_prediction'],2)
        self.assertEqual(r['by_horizon']['1']['available_forecast_metrics']['tp'],1)
        self.assertIsNone(r['regional_metrics'])
        self.assertIsNone(grid_index([10,11],[75,76],20,75))
        r=match_fields([{**record,'latitude':20}],[forecast]);self.assertEqual(len(r['unmatched']),1)

    def test_field_abstention_counts_event_miss(self):
        record,forecast=self.inputs();forecast[0]['data_quality']['state']='low_coverage'
        r=match_fields([record],[forecast])
        self.assertEqual(r['status'],'insufficient_data');self.assertEqual(r['by_horizon']['1']['abstentions'],1)
        self.assertEqual(r['by_horizon']['1']['event_detection_including_abstentions']['fn'],1)

    def test_field_duplicates_and_target_leak_rejected(self):
        record,forecast=self.inputs()
        with self.assertRaises(ValueError):match_fields([record],[forecast,forecast])
        forecast[0]['input_dates']=['2026-02-02']
        with self.assertRaises(ValueError):match_fields([record],[forecast])

    def test_evaluation_per_class_and_regional_metrics(self):
        actual=np.array([[[0,1],[2,255]],[[0,0],[0,0]]],np.uint8)
        predicted=actual.copy();predicted[1]=255
        cal={'regional_elevated_percent':10.,'regional_strong_percent':20.}
        result=evaluate_arrays(actual,predicted,[1,0],np.ones((2,2),bool),cal)
        self.assertEqual(result['pixel_metrics']['macro_f1'],1.)
        self.assertEqual(result['forecast_available_days'],1);self.assertEqual(result['regional_abstentions'],1)
        self.assertEqual(result['regional_strong']['tp'],1)

    def test_empty_and_invalid_evaluation(self):
        a=np.full((1,2,2),255,np.uint8);cal={'regional_elevated_percent':10.,'regional_strong_percent':20.}
        result=evaluate_arrays(a,a,[0],np.ones((2,2),bool),cal)
        self.assertEqual(result['status'],'insufficient_data');self.assertIsNone(result['pixel_metrics'])
        with self.assertRaises(ValueError):evaluate_arrays(a,a,[2],np.ones((2,2),bool),cal)

    def test_independent_known_period_rejected(self):
        with TemporaryDirectory() as folder,patch('experiments.finalize.verify_final'):
            path=Path(folder);(path/'metadata.json').write_text(json.dumps({'independent':True,'label_definition':'fixture','label_provenance':'fixture'}))
            np.savez(path/'evaluation.npz',dates=['2025-01-01'],latitude=[10,11],longitude=[75,76],actual=np.zeros((1,2,2),np.uint8),predicted=np.zeros((1,2,2),np.uint8),available=[1],ocean=np.ones((2,2),bool))
            with self.assertRaises(ValueError):independent_report(path)

    @unittest.skipUnless((config.SEQUENCE_DIR/'manifest.json').exists(),'Requires actual fitted grid/freeze.')
    def test_independent_contract_success_and_tampered_prediction_rejection(self):
        # Supplied-label contract fixture, NOT independent scientific evidence.
        from preprocessing.data_utils import file_sha256
        lat=np.load(config.PROCESSED_DIR/'latitude_kerala_2021_2025.npy');lon=np.load(config.PROCESSED_DIR/'longitude_kerala_2021_2025.npy')
        ocean=np.load(config.LABEL_DIR/'stable_ocean_mask.npy').astype(bool)
        labels=np.where(ocean,np.indices(ocean.shape).sum(0)%3,255).astype(np.uint8)[None]
        bundle={'status':'success','forecast_dates':['2026-01-02'],'input_dates':['2026-01-01'],'data_quality':{'state':'observed'}}
        with TemporaryDirectory() as folder:
            root=Path(folder)
            metadata={'independent':True,'label_definition':'CONTRACT FIXTURE ONLY','label_provenance':'CONTRACT FIXTURE ONLY',
                'model_sha256':file_sha256(config.MODEL_PATH),'decision_rule_sha256':file_sha256(config.RUN_DIR/'calibration.json'), 'forecast_directories':['fixture-not-observation']}
            (root/'metadata.json').write_text(json.dumps(metadata))
            def write(prediction):np.savez(root/'evaluation.npz',dates=['2026-01-02'],latitude=lat,longitude=lon,actual=labels,predicted=prediction,available=[1],ocean=ocean)
            write(labels)
            with patch('operations.validation.load_forecast',return_value=(bundle,labels,lat,lon)):
                result=independent_report(root)
                self.assertEqual(result['status'],'success');self.assertFalse(result['trained']);self.assertFalse(result['tuned'])
                self.assertEqual(len(result['pixel_metrics']['classes']),3)
                tampered=labels.copy();row,col=np.argwhere(ocean)[0];tampered[0,row,col]=(tampered[0,row,col]+1)%3;write(tampered)
                with self.assertRaises(ValueError):independent_report(root)

    def test_field_malformed_csv_and_toxin_units_rejected(self):
        with TemporaryDirectory() as folder:
            path=Path(folder)/'field.csv'
            path.write_text('record_id,observed_at_utc\nfixture\n')
            with self.assertRaises(ValueError):load_fields(path,'development')
            path.write_text('toxin_value,toxin_units\nNaN,mg/L\n')
            with self.assertRaises(ValueError):load_fields(path,'development')


class ProviderAndMetadataTests(unittest.TestCase):
    def bundle(self):
        today=datetime.now(timezone.utc).date()
        return {'status':'success','operational_warning_validated':False,'operational_warnings_available':False,'historical_demo':False,
            'freshness':{'state':'fresh'},'data_quality':{'state':'observed'},'input_dates':[str(today)],
            'model_sha256':'FIXTURE','generated_at_utc':datetime.now(timezone.utc).isoformat(),'forecast_dates':[str(today+timedelta(days=1))],
            'daily_predictions':[{'day':1,'warning_level':'STRONG BLOOM RISK'}]}

    def test_truthful_health_calibration(self):
        from fastapi.testclient import TestClient
        from backend.app import app
        with TestClient(app) as client:
            health=client.get('/api/health').json();system=client.get('/api/system').json()
            self.assertEqual(health['probability_calibration'],system['calibration'])
            self.assertFalse(health['calibration_available']);self.assertFalse(system['calibration']['operationally_approved'])

    def test_notification_dryrun_and_disabled_gate(self):
        with TemporaryDirectory() as folder:
            bundle=self.bundle();r=dispatch(bundle,folder)
            self.assertEqual(r['state'],'dry_run');self.assertFalse(r['external_sent']);self.assertEqual(r['entries'][0]['horizon_days'],1)
            self.assertIn('region',r['entries'][0]);self.assertFalse(r['operationally_eligible'])
            with patch.dict(os.environ,{'HAB_NOTIFICATIONS_ENABLED':'1','HAB_NOTIFICATION_PROVIDER':'email'}),patch('operations.notifications.send_email') as send:
                self.assertEqual(dispatch(bundle,folder,dry_run=False)['state'],'disabled');send.assert_not_called()

    def test_notification_missing_stale_low_and_normal(self):
        with TemporaryDirectory() as folder:
            bundle=self.bundle()
            for change in [{'status':'insufficient_data'},{'data_quality':{'state':'low_coverage'}},{'input_dates':['2024-01-01']}]:
                self.assertEqual(dispatch({**bundle,**change},folder)['state'],'blocked')
            self.assertEqual(dispatch({**bundle,'daily_predictions':[{'day':1,'warning_level':'NORMAL'}]},folder)['entries'],[])

    def test_mock_provider_transport_contract_and_idempotency(self):
        # No recipient is contacted; this validates dispatch only with a mocked provider.
        with TemporaryDirectory() as folder,patch.dict(os.environ,{'HAB_NOTIFICATIONS_ENABLED':'1','HAB_NOTIFICATION_PROVIDER':'email'}),patch('operations.notifications.operational_approval',return_value=True),patch('operations.notifications.send_email',return_value=None) as send:
            b={**self.bundle(),'operational_warning_validated':True,'operational_warnings_available':True}
            self.assertTrue(dispatch(b,folder,False)['external_sent']);dispatch(b,folder,False);self.assertEqual(send.call_count,1)

    def test_forged_bundle_approval_cannot_enable_delivery(self):
        with TemporaryDirectory() as folder,patch.dict(os.environ,{'HAB_NOTIFICATIONS_ENABLED':'1','HAB_NOTIFICATION_PROVIDER':'email'}),patch('operations.notifications.send_email') as send:
            b={**self.bundle(),'operational_warning_validated':True,'operational_warnings_available':True}
            self.assertEqual(dispatch(b,folder,False)['state'],'disabled');send.assert_not_called()

    def test_unconfigured_providers_fail_without_network(self):
        with patch.dict(os.environ,{},clear=True):
            with self.assertRaises(ValueError):send_email({'risk_level':'fixture','forecast_date':'fixture'})
            with self.assertRaises(ValueError):send_sms({'id':'fixture'})

    def test_smtp_transport_tls_login_and_message(self):
        with patch.dict(os.environ,{'HAB_SMTP_HOST':'fixture.invalid','HAB_EMAIL_FROM':'fixture@invalid','HAB_EMAIL_TO':'fixture@invalid','HAB_SMTP_USER':'fixture','HAB_SMTP_PASSWORD':'fixture'}),patch('operations.notifications.smtplib.SMTP') as smtp:
            smtp.return_value.__enter__.return_value.send_message.return_value={}
            send_email({'id':'fixture','risk_level':'fixture','forecast_date':'fixture'})
            client=smtp.return_value.__enter__.return_value
            client.starttls.assert_called_once();client.login.assert_called_once();client.send_message.assert_called_once()

    def test_sms_transport_acceptance_required(self):
        with patch.dict(os.environ,{'HAB_SMS_ENDPOINT':'https://fixture.invalid','HAB_SMS_TOKEN':'fixture','HAB_SMS_TO':'fixture'}),patch('operations.notifications.urlopen') as request:
            request.return_value.__enter__.return_value.read.return_value=b'{"accepted":true,"message_id":"fixture-id"}'
            self.assertEqual(send_sms({'id':'fixture'}),'fixture-id')
            request.return_value.__enter__.return_value.read.return_value=b'{"accepted":false}'
            with self.assertRaises(ValueError):send_sms({'id':'fixture'})


class RealStagedIntegrationTests(unittest.TestCase):
    @unittest.skipUnless((config.SEQUENCE_DIR/'manifest.json').exists(),'Requires actual project data/model artifacts.')
    def test_real_low_insufficient_missing_and_stale_inputs_withhold_model(self):
        from preprocessing.pipeline import EnvironmentalPipeline
        from prediction.service import data_freshness
        pipe=EnvironmentalPipeline()
        cases=[('2024-03-11','low_coverage'),('2024-03-07','insufficient_data'),('2024-05-08','missing_data')]
        with TemporaryDirectory() as folder:
            root=Path(folder)
            for date,state in cases:
                stage=root/date;stage.mkdir()
                end=int(np.flatnonzero(pipe.dates==np.datetime64(date))[0]);indices=np.arange(end-4,end+1)
                np.savez(stage/'inputs.npz',values=np.stack([pipe.values[n][indices] for n in config.CHANNEL_NAMES],-1),masks=np.stack([pipe.masks[n][indices] for n in config.CHANNEL_NAMES],-1),dates=pipe.dates[indices],latitude=pipe.latitude,longitude=pipe.longitude)
                meta={'channel_order':list(config.CHANNEL_NAMES),'units':['mg/m3','degC','m/s','m/s','mm/day','m/s','m/s'],'crs':'EPSG:4326',
                    'sources':{n:{'product_ids':['TEST_CONTRACT_ONLY_REAL_ARCHIVE'],'quality_reference':'test manifest only','native_qc_applied':True,
                                 'observation_end_utc':date+'T22:00:00Z','available_at_utc':date+'T23:00:00Z'} for n in config.CHANNEL_NAMES}}
                (stage/'metadata.json').write_text(json.dumps(meta));fixed=datetime.fromisoformat(date+'T23:45:00+00:00')
                with patch('upgrade.live.ROOT',root/'live'),patch('upgrade.live.datetime') as clock,patch('upgrade.live.data_freshness',side_effect=lambda end:data_freshness(end,now=fixed.date())),patch('upgrade.live.load_model') as model:
                    clock.now.return_value=fixed;result=run_update(stage)
                    self.assertEqual(result['state'],state);model.assert_not_called()
            with patch('upgrade.live.ROOT',root/'live'),patch('upgrade.live.load_model') as model:
                self.assertEqual(run_update(root/'2024-03-11')['state'],'stale_data');model.assert_not_called()

    @unittest.skipUnless((config.SEQUENCE_DIR/'manifest.json').exists(),'Requires actual project data/model artifacts.')
    def test_real_staged_data_model_maps_and_live_api_under_test_clock(self):
        # Real historical arrays/model; fixed test clock and a TEST manifest.
        # This is engineering verification, not fresh acquisition/field evidence.
        from preprocessing.pipeline import EnvironmentalPipeline
        from prediction.service import data_freshness
        from upgrade.live import update
        from fastapi.testclient import TestClient
        from backend.app import app
        pipe=EnvironmentalPipeline();end=int(np.flatnonzero(pipe.dates==np.datetime64('2024-02-15'))[0]);indices=np.arange(end-4,end+1)
        with TemporaryDirectory() as folder:
            root=Path(folder);stage=root/'stage';stage.mkdir()
            np.savez(stage/'inputs.npz',values=np.stack([pipe.values[n][indices] for n in config.CHANNEL_NAMES],-1),
                masks=np.stack([pipe.masks[n][indices] for n in config.CHANNEL_NAMES],-1),dates=pipe.dates[indices],latitude=pipe.latitude,longitude=pipe.longitude)
            metadata={'channel_order':list(config.CHANNEL_NAMES),'units':['mg/m3','degC','m/s','m/s','mm/day','m/s','m/s'],'crs':'EPSG:4326',
                'sources':{n:{'product_ids':['TEST_MANIFEST_REAL_ARCHIVE_CONTRACT_NOT_ACQUISITION'], 'quality_reference':'test manifest, not independent QC evidence','native_qc_applied':True,
                             'observation_end_utc':'2024-02-15T22:00:00Z','available_at_utc':'2024-02-15T23:00:00Z'} for n in config.CHANNEL_NAMES}}
            (stage/'metadata.json').write_text(json.dumps(metadata))
            fixed=datetime(2024,2,15,23,45,tzinfo=timezone.utc)
            with patch('upgrade.live.ROOT',root/'live'),patch('upgrade.live.datetime') as clock,patch('upgrade.live.data_freshness',side_effect=lambda end:data_freshness(end,now=fixed.date())):
                clock.now.return_value=fixed
                result=update(stage);self.assertEqual(result['state'],'fresh_research_prediction')
                directory=root/'live/bundles'/result['latest_successful_bundle']
                bundle=json.loads((directory/'bundle.json').read_text())
                self.assertEqual(len(bundle['daily_predictions']),5)
                for i in range(1,6):self.assertTrue((directory/f'day_{i}.png').read_bytes().startswith(b'\x89PNG'))
                with TestClient(app) as client:
                    response=client.get('/api/live/prediction');self.assertEqual(response.json()['status'],'success')
                    self.assertEqual(response.json()['alerts'],[]);self.assertFalse(response.json()['operational_warnings_available'])
                # Independent post-2025 evaluation is unavailable; never relabel
                # these 2024 historical arrays as independent observations.
                with self.assertRaises(ValueError):load_forecast(directory,independent=True)
            from operations.hindcast import generate
            hindcast=generate(stage,root/'hindcast')
            self.assertEqual(hindcast['status'],'success');self.assertTrue(hindcast['historical_demo'])
            _,risk,_,_=load_forecast(root/'hindcast')
            np.testing.assert_array_equal(risk,np.load(directory/'risk_classes.npy'))
            self.assertEqual(len(hindcast['maps']),5)


if __name__=='__main__':unittest.main()
