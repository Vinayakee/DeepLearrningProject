"""Scientific integrity and failure handling for the roadmap extension."""
import hashlib
import json
import os
import socket
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest
import netCDF4
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
import config
from operations.sentinel_native import read_native, aggregate_granules, grid_bins, flag_policy
from operations.sentinel_pilot import load_daily, SCHEMA
from operations.geography import land_fraction, compare
from operations.acquisition import acquire_with_retry, AcquisitionUnavailable
from operations.update import recover_lock, run_update
from operations.forecast_views import summary_csv
from preprocessing.data_utils import file_sha256
from experiments.roadmap_evaluation import seasonal_predictions, block_intervals, event_report


def native_fixture(root, cloud=False):
    """Small native contract fixture, never used as scientific evidence."""
    name = 'S3A_OL_2_WFR____CONTRACT_FIXTURE_003.SEN3'
    path = root / name; path.mkdir()
    names = ['WATER','CLOUD','CLOUD_AMBIGUOUS','CLOUD_MARGIN','INVALID','COSMETIC','SATURATED','SUSPECT',
             'HISOLZEN','SNOW_ICE','HIGHGLINT','AC_FAIL','WHITECAPS','ADJAC',
             *[f'RWNEG_O{i}' for i in range(2,9)],'OC4ME_FAIL','MEGLINT']
    masks = np.array([1 << i for i in range(len(names))], np.uint64)
    with netCDF4.Dataset(path/'geo_coordinates.nc','w') as ds:
        ds.createDimension('rows',3);ds.createDimension('cols',4)
        ds.createVariable('latitude','f8',('rows','cols'))[:] = 11.
        ds.createVariable('longitude','f8',('rows','cols'))[:] = 75.
    with netCDF4.Dataset(path/'wqsf.nc','w') as ds:
        ds.createDimension('rows',3);ds.createDimension('cols',4)
        v=ds.createVariable('WQSF','u8',('rows','cols'));v.flag_meanings=' '.join(names);v.flag_masks=masks
        v[:]=np.uint64(1 | (2 if cloud else 0))
    with netCDF4.Dataset(path/'chl_oc4me.nc','w') as ds:
        ds.createDimension('rows',3);ds.createDimension('cols',4)
        v=ds.createVariable('CHL_OC4ME','i2',('rows','cols'),fill_value=-32768)
        v.scale_factor=.1;v.add_offset=-2.;v.units='lg(re mg.m-3)';v[:]=0.
    with netCDF4.Dataset(path/'time_coordinates.nc','w') as ds:
        ds.createDimension('rows',3)
        v=ds.createVariable('time_stamp','i8',('rows',));v.units='microseconds since 2024-02-11 00:00:00';v[:]=[0,1000000,2000000]
    names = ['geo_coordinates.nc','wqsf.nc','chl_oc4me.nc','time_coordinates.nc']
    objects = ''.join(f'<dataObject><fileLocation href="{n}"/><checksum checksumName="MD5">{hashlib.md5((path/n).read_bytes()).hexdigest()}</checksum></dataObject>' for n in names)
    (path/'xfdumanifest.xml').write_text(f'<root>{objects}</root>')
    metadata={'catalogue':{'Name':name,'PublicationDate':'2024-02-12T12:00:00Z','ContentDate':{'Start':'2024-02-11T00:00:00Z'}},
              'files':{n:{'sha256':file_sha256(path/n)} for n in [*names,'xfdumanifest.xml']}}
    (path/'source_provenance.json').write_text(json.dumps(metadata))
    return path


class NativeQCTests(unittest.TestCase):
    def test_actual_decoding_qc_and_conservative_aggregation_contract(self):
        with TemporaryDirectory() as tmp:
            path=native_fixture(Path(tmp))
            sums,good,total,report=read_native(path,[11,10],[75,76])
            values,valid=aggregate_granules([(sums,good,total)],np.ones((2,2),bool))
            self.assertEqual(values[0,0],1.)  # CF packing and log10 decoded once each.
            self.assertEqual(int(valid.sum()),1)
            self.assertEqual(report['observation_end_utc'],'2024-02-11T00:00:02Z')
            self.assertNotIn('MEGLINT',report['reject_flags'])
            self.assertEqual(int(good[0,0]),12)

    def test_cloud_pixels_stay_missing_and_tampered_sources_fail(self):
        with TemporaryDirectory() as tmp:
            path=native_fixture(Path(tmp),cloud=True)
            sums,good,total,_=read_native(path,[11,10],[75,76])
            values,valid=aggregate_granules([(sums,good,total)],np.ones((2,2),bool))
            self.assertFalse(valid.any());self.assertTrue(np.isnan(values).all())
            with (path/'wqsf.nc').open('ab') as stream:stream.write(b'changed')
            with self.assertRaises(ValueError):read_native(path,[11,10],[75,76])

    def test_native_grid_order_and_duplicate_flag_bits_rejected(self):
        with self.assertRaises(ValueError):grid_bins(np.array([10]),np.array([75]),[10,11],[75,76])
        with self.assertRaises(ValueError):flag_policy('WATER CLOUD',[1,1])
        bins, inside = grid_bins(np.array([11.,10.,9.4]),np.array([75.,76.,75.]),[11,10],[75,76])
        np.testing.assert_array_equal(bins[:2],[0,3]);np.testing.assert_array_equal(inside,[True,True,False])

    def test_publication_gate_and_export_integrity(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            np.savez(root/'observations.npz',latitude=[11,10],longitude=[75,76],dates=['2024-02-11'],
                     chlorophyll=np.ones((1,2,2)),valid=np.ones((1,2,2),bool),
                     valid_native_samples=np.full((1,2,2),3),total_native_samples=np.full((1,2,2),4))
            metadata={'schema':SCHEMA,'crs':'EPSG:4326','units':'mg/m3','native_qc_before_resampling':True,
                      'observations_sha256':file_sha256(root/'observations.npz'),
                      'daily':[{'date':'2024-02-11','observation_end_utc':'2024-02-11T04:00:00Z','available_at_utc':'2024-02-12T11:00:00Z'}]}
            (root/'metadata.json').write_text(json.dumps(metadata))
            _,values,valid,_,qc=load_daily(root,[11,10],[75,76],'2024-02-12')
            self.assertFalse(valid.any());self.assertTrue(np.isnan(values).all());self.assertEqual(qc['late_publication_days_excluded'],1)
            self.assertTrue(load_daily(root,[11,10],[75,76],'2024-02-13')[2].all())
            with (root/'observations.npz').open('ab') as stream:stream.write(b'changed')
            with self.assertRaises(ValueError):load_daily(root,[11,10],[75,76])


class GeographicAndEvaluationTests(unittest.TestCase):
    def test_physical_polygon_holes_and_coordinate_order(self):
        geometry={'type':'FeatureCollection','features':[{'type':'Feature','properties':{},'geometry':{'type':'Polygon','coordinates':[
            [[74.5,9.5],[75.5,9.5],[75.5,11.5],[74.5,11.5],[74.5,9.5]],
            [[74.6,10.6],[74.6,11.4],[75.4,11.4],[75.4,10.6],[74.6,10.6]]]}}]}
        fraction=land_fraction(geometry,[11,10],[75,76],10)
        self.assertAlmostEqual(fraction[0,0],.36);self.assertEqual(fraction[1,0],1.)
        self.assertTrue((fraction[:,1]==0).all())
        report=compare(np.ones((2,2),bool),fraction)
        self.assertEqual(report['supported_land_cells'],1)
        with self.assertRaises(ValueError):land_fraction(geometry,[10,11],[75,76])

    def test_seasonal_predictions_do_not_fit_heldout_values(self):
        dates=np.array(['2023-02-01','2023-02-02','2024-02-01'],dtype='datetime64[D]')
        labels=np.array([[[1]],[[1]],[[2]]],np.uint8)
        first=seasonal_predictions(labels,dates);labels[-1]=0
        np.testing.assert_array_equal(first,seasonal_predictions(labels,dates))
        self.assertEqual(first[1,0,0],1);self.assertEqual(first[0,0,0],255)

    def test_bootstrap_is_deterministic_and_keeps_daily_counts(self):
        counts=np.tile(np.eye(3,dtype=int),(30,1,1))
        first=block_intervals(counts,50)
        self.assertEqual(first,block_intervals(counts,50))
        self.assertEqual(first['intervals_95_percent']['macro_f1'],[1.,1.])
        with self.assertRaises(ValueError):block_intervals(counts,50,1)

    def test_unknown_truth_splits_proxy_episodes_and_lead_time(self):
        frame=pd.DataFrame({'horizon':[1,1,1], 'target_date':['2024-02-02','2024-02-03','2024-02-04'],
                            'truth_qualified':[True,False,True],'actual_level':['STRONG BLOOM RISK','UNKNOWN','ELEVATED RISK'],
                            'available':[True,False,False],'issued_level':['STRONG BLOOM RISK','INSUFFICIENT DATA','INSUFFICIENT DATA'],
                            'input_end_date':['2024-02-01','2024-02-02','2024-02-03']})
        result=event_report(frame)
        self.assertEqual(result['episode_count'],2);self.assertEqual(result['detected_episodes'],1)
        self.assertEqual(result['episodes'][0]['lead_days_before_onset'],1)


class RuntimeAndExportTests(unittest.TestCase):
    def test_retry_only_transient_acquisition(self):
        with patch('operations.acquisition.acquire',side_effect=[TimeoutError(),Path('real-stage')]) as acquire,patch('operations.acquisition.time.sleep'):
            self.assertEqual(acquire_with_retry(),Path('real-stage'));self.assertEqual(acquire.call_count,2)
        with patch('operations.acquisition.acquire',side_effect=AcquisitionUnavailable()) as acquire:
            with self.assertRaises(AcquisitionUnavailable):acquire_with_retry()
            self.assertEqual(acquire.call_count,1)

    def test_orphan_lock_requires_same_host_and_dead_process(self):
        with TemporaryDirectory() as tmp,patch('upgrade.live.ROOT',Path(tmp)):
            lock=Path(tmp)/'update.lock'
            lock.write_text(json.dumps({'hostname':socket.gethostname(),'pid':123456}))
            with patch('operations.update.os.kill',side_effect=ProcessLookupError):self.assertEqual(recover_lock()['state'],'orphan_lock_recovered')
            lock.write_text(json.dumps({'hostname':socket.gethostname(),'pid':os.getpid()}))
            with self.assertRaises(ValueError):recover_lock()
            self.assertTrue(lock.exists())

    def test_changed_lock_owner_is_not_removed(self):
        with TemporaryDirectory() as tmp,patch('upgrade.live.ROOT',Path(tmp)),patch('operations.update.acquire_with_retry') as acquire:
            lock=Path(tmp)/'update.lock'
            def replace(*args,**kwargs):
                lock.write_text(json.dumps({'token':'different-owner'}));raise AcquisitionUnavailable()
            acquire.side_effect=replace
            run_update();self.assertTrue(lock.exists())

    def test_export_keeps_research_status_and_dates(self):
        bundle={'forecast_id':'a'*32,'input_dates':['2024-02-15'],'forecast_dates':['2024-02-16'],
                'daily_predictions':[{'day':1,'warning_level':'NORMAL','high_risk_percentage':0.,'mean_probability':.1}],
                'data_quality':{'chlorophyll_mean_input_coverage_percent':10.},'freshness':{'state':'stale'},'operational_warnings_available':False}
        result=summary_csv(bundle)
        self.assertIn('2024-02-16',result);self.assertIn('False,chlorophyll anomaly proxy',result)

    def test_export_and_observation_endpoints_reject_untrusted_ids(self):
        from backend.app import app
        with TestClient(app) as client:
            for suffix in ('summary.csv','observations.png'):
                self.assertEqual(client.get(f'/api/forecasts/not-an-id/{suffix}?demo=true').status_code,409)


@unittest.skipUnless((config.RUN_DIR/'training_config.json').exists(), 'Requires actual fitted model and environmental inputs.')
class RealForecastViewsTests(unittest.TestCase):
    def test_real_forecast_context_export_staleness_and_tampering(self):
        from backend.app import app
        from prediction.service import ForecastService
        with TemporaryDirectory() as tmp,patch('config.FORECAST_DIR',Path(tmp)),TestClient(app) as client:
            bundle=ForecastService().generate('2024-02-15',historical_demo=True)
            identifier=bundle['forecast_id']
            route=f'/api/forecasts/{identifier}'
            self.assertEqual(client.get(route+'/summary.csv').status_code,409)
            csv=client.get(route+'/summary.csv?demo=true')
            self.assertEqual(csv.status_code,200);self.assertIn('2024-02-20',csv.text)
            image=client.get(route+'/observations.png?demo=true')
            self.assertEqual(image.status_code,200);self.assertTrue(image.content.startswith(b'\x89PNG'))
            with (Path(tmp)/identifier/'risk_classes.npy').open('ab') as stream:stream.write(b'tampered')
            self.assertEqual(client.get(route+'/summary.csv?demo=true').status_code,409)


if __name__ == '__main__':
    unittest.main()
