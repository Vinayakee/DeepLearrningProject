"""Software contracts using synthetic temporary fixtures, never HAB skill evidence."""
import csv
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import numpy as np
import tensorflow as tf
from fastapi.testclient import TestClient
import config
from backend.app import app
from preprocessing.hab_labels import RECORD_COLUMNS, build_labels, sequence_rows, validate_spec
from preprocessing.hab_sequences import prepare, load_dataset
from models.hab_convlstm import build_hab_convlstm, MaskedBinaryLoss
from training.train_hab import train
from prediction.hab_evaluation import (fit_decision, calibrated, binary_metrics, persistence,
                                       calibrate, evaluate, verify_decision, collect, summarize)


def fixture(root, days_per_split=18):
    """72 deliberately synthetic days on a tiny grid for regression testing only."""
    processed = root/'processed'; processed.mkdir()
    first = np.datetime64('2021-01-01')
    dates = np.arange(first, first+np.timedelta64(4*days_per_split,'D'))
    np.savetxt(processed/'dates_2021_2025.txt', dates, fmt='%s')
    np.save(processed/'latitude_kerala_2021_2025.npy', [12., 11., 10., 9.])
    np.save(processed/'longitude_kerala_2021_2025.npy', [74., 75., 76., 77.])
    for name in config.CHANNEL_NAMES:
        values = np.broadcast_to((1+np.arange(len(dates)) % 2)[:, None, None], (len(dates), 4, 4)).astype(np.float32).copy()
        np.save(processed/f'{name}_2021_2025.npy', values)
        np.save(processed/f'{name}_mask_2021_2025.npy', np.ones_like(values, np.uint8))
    spec = {'schema': 'hab-occurrence-v1', 'definition_id': 'synthetic-test-only',
            'event_definition': 'Synthetic test occurrence; not scientific evidence',
            'negative_definition': 'Synthetic complete-zone monitoring for software tests', 'input_days': 2, 'output_days': 5,
            'splits': {split: [str(dates[i*days_per_split]), str(dates[(i+1)*days_per_split-1])] for i, split in enumerate(('train','validation','calibration','test'))},
            'zones': [{'id':'test-zone','name':'Synthetic zone','bounds':[73.9, 8.9, 77.1, 12.1]}]}
    observations = root/'observations.csv'
    records = []
    for i, day in enumerate(dates):
        records.append(dict(zip(RECORD_COLUMNS, [f'r{i}', f'event{i}' if i % 2 else '', 'test-zone',
            f'{day}T12:00:00Z', f'{day}T13:00:00Z', '11', '75', 'true' if i % 2 else 'false',
            'point' if i % 2 else 'complete_zone', 'synthetic-test-only', 'Synthetic species',
            'synthetic fixture', 'synthetic fixture, not real observations', 'unit test'])))
    save_records(observations, records)
    spec_path=root/'spec.json'; spec_path.write_text(json.dumps(spec))
    return processed, dates, spec, spec_path, observations, records


def save_records(path, records):
    with path.open('w', newline='') as handle:
        writer=csv.DictWriter(handle, fieldnames=RECORD_COLUMNS); writer.writeheader(); writer.writerows(records)


class HABLabelTests(unittest.TestCase):
    def test_unknown_days_stay_unknown_and_point_absence_is_rejected(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp); _,dates,spec,_,path,records=fixture(root)
            save_records(path, records[:2])
            labels,_,_=build_labels(path,dates,spec)
            np.testing.assert_array_equal(labels[:3,0],[0,1,255])
            records[0]['observation_scope']='point';save_records(path,records[:1])
            with self.assertRaisesRegex(ValueError,'Point absence'):build_labels(path,dates,spec)

    def test_conflicts_event_leakage_provenance_and_publication_fail(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp); _,dates,spec,_,path,records=fixture(root)
            conflicting={**records[0],'record_id':'conflict','confirmed_hab':'true','event_id':'same'}
            save_records(path,[records[0],conflicting])
            with self.assertRaisesRegex(ValueError,'Conflicting'):build_labels(path,dates,spec)
            records[19]['event_id']=records[1]['event_id'];save_records(path,records)
            with self.assertRaisesRegex(ValueError,'crosses dataset splits'):build_labels(path,dates,spec)
            record={**records[1],'available_at_utc':'2020-01-01T00:00:00Z'};save_records(path,[record])
            with self.assertRaisesRegex(ValueError,'Publication'):build_labels(path,dates,spec)
            record={**records[1],'source_reference':''};save_records(path,[record])
            with self.assertRaisesRegex(ValueError,'Missing observation'):build_labels(path,dates,spec)

    def test_spec_template_cannot_train_and_windows_do_not_cross_splits(self):
        template=json.loads((config.ROOT/'docs/examples/hab_spec.template.json').read_text())
        with self.assertRaises(ValueError):validate_spec(template)
        with TemporaryDirectory() as tmp:
            _,dates,spec,_,path,_=fixture(Path(tmp))
            labels,_,_=build_labels(path,dates,spec)
            rows=sequence_rows(dates,labels,spec)
            for row in rows:
                first,last=spec['splits'][row['split']]
                self.assertGreaterEqual(str(dates[row['input_start']]),first)
                self.assertLessEqual(str(dates[row['target_start']+4]),last)
                self.assertEqual(row['target_start']-row['input_start'],2)

    def test_training_only_statistics_artifact_integrity_and_empty_records(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);processed,dates,spec,spec_path,path,_=fixture(root)
            a=prepare(spec_path,path,root/'dataset',processed)
            self.assertEqual(a['audit']['train']['positive_zone_days'],9)
            pipeline,_,_,_,_=load_dataset(root/'dataset')
            old=pipeline.stats
            values=np.load(processed/'sst_2021_2025.npy');values[18:]=999
            np.save(processed/'sst_2021_2025.npy',values)
            with self.assertRaisesRegex(ValueError,'archive changed'):load_dataset(root/'dataset')
            prepare(spec_path,path,root/'dataset2',processed)
            new=json.loads((root/'dataset2/preprocessing.json').read_text())
            self.assertEqual(old,new)
            save_records(path,[])
            with self.assertRaisesRegex(ValueError,'No verified'):build_labels(path,dates,spec)


class HABModelTests(unittest.TestCase):
    def test_lead_time_uses_full_event_catalogue_and_distinguishes_ongoing_detection(self):
        data={'truth':np.ones((1,5,1),np.uint8),'available':np.ones((1,5,1),bool),
              'events':np.full((1,5,1),'event'),'origins':np.array(['2021-01-02']),
              'targets':np.array([['2021-01-03','2021-01-04','2021-01-05','2021-01-06','2021-01-07']]),
              'event_first_recorded_day':{'event':'2021-01-01'}}
        result=summarize(data,np.ones((1,5,1)),.5)
        self.assertEqual(result['event_recall'],1.)
        self.assertEqual(result['advance_warning_event_recall'],0.)
        self.assertEqual(result['event_lead_times'][0]['nominal_days_before_first_recorded_day'],-1)
        self.assertEqual(result['event_lead_times'][0]['timing'],'ongoing_event_detection')
        del data['event_first_recorded_day']
        missing=summarize(data,np.ones((1,5,1)),.5)
        self.assertIsNone(missing['event_lead_times'][0]['nominal_days_before_first_recorded_day'])

    def test_fourteen_day_inputs_train_statistics_and_produce_five_future_days(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);processed,_,spec,spec_path,path,_=fixture(root,days_per_split=24)
            spec['input_days']=14;spec_path.write_text(json.dumps(spec))
            prepare(spec_path,path,root/'dataset',processed)
            pipeline,labels,events,rows,_=load_dataset(root/'dataset')
            row=rows[0]
            self.assertEqual(row['target_start']-row['input_start'],14)
            X=pipeline.transform(np.arange(row['input_start'],row['target_start']))
            model=build_hab_convlstm(X.shape,pipeline.zones)
            model.compile(optimizer='adam',loss=MaskedBinaryLoss())
            y=labels[row['target_start']:row['target_start']+5].astype(np.float32)
            loss=model.train_on_batch(X[None],y[None])
            self.assertTrue(np.isfinite(loss))
            self.assertEqual(model(X[None],training=False).shape,(1,5,1))

    def test_masked_loss_zero_gradient_on_unknown_and_model_reload(self):
        loss=MaskedBinaryLoss(2.)
        p=tf.Variable([[.8,.4]],dtype=tf.float32)
        with tf.GradientTape() as tape:
            value=loss(tf.constant([[1.,255.]]),p)
        gradient=tape.gradient(value,p).numpy()
        self.assertAlmostEqual(float(value),float(loss(tf.constant([[1.]]),tf.constant([[.8]]))),places=6)
        self.assertEqual(float(gradient[0,1]),0.)
        self.assertEqual(float(loss(tf.constant([[255.,255.]]),p)),0.)
        tf.keras.utils.set_random_seed(42)
        model=build_hab_convlstm((2,4,4,16),[np.ones((4,4),bool)])
        X=np.zeros((1,2,4,4,16),np.float32); altered=X.copy();altered[:,-1]=1
        a=model(X,training=False).numpy(); b=model(altered,training=False).numpy()
        self.assertEqual(b.shape,(1,5,1));self.assertTrue(np.all(np.abs(a-b)>1e-7))
        with TemporaryDirectory() as tmp:
            path=Path(tmp)/'model.keras';model.save(path)
            restored=tf.keras.models.load_model(path,compile=False)
            np.testing.assert_allclose(restored(altered,training=False),b,atol=1e-6)

    def test_calibration_uses_both_classes_and_explicit_threshold_objectives(self):
        y=np.array([0,0,0,1,1,1]);scores=np.array([.1,.15,.2,.65,.8,.9])
        decision=fit_decision(y,scores)
        p=calibrated(scores,decision)
        self.assertLessEqual(decision['low_threshold'],decision['high_threshold'])
        self.assertEqual(binary_metrics(y,p,decision['high_threshold'])['f1'],1.)
        with self.assertRaises(ValueError):fit_decision(np.zeros(6),scores)
        with self.assertRaisesRegex(ValueError,'nonpositive'):fit_decision(y,1-scores)
        fallback=fit_decision(y,1-scores,allow_uncalibrated=True)
        self.assertFalse(fallback['calibration_applied'])

    def test_persistence_cannot_use_unpublished_observations(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);processed,_,spec,spec_path,path,_=fixture(root)
            prepare(spec_path,path,root/'dataset',processed)
            pipeline,_,_,rows,_=load_dataset(root/'dataset')
            records=json.loads((root/'dataset/records.json').read_text())
            records[1]['available_at_utc']='2021-02-20T00:00:00Z'
            self.assertEqual(persistence(pipeline,records,rows[0])[0],0.)
            records[0]['available_at_utc']='2021-02-20T00:00:00Z'
            self.assertTrue(np.isnan(persistence(pipeline,records,rows[0])[0]))

    def test_event_recall_counts_unavailable_positive_forecasts_as_missed(self):
        data={'truth':np.ones((1,5,1),np.uint8),'available':np.zeros((1,5,1),bool),
              'events':np.full((1,5,1),'event'),'origins':np.array(['2021-01-01']),
              'targets':np.array([['2021-01-02','2021-01-03','2021-01-04','2021-01-05','2021-01-06']])}
        report=summarize(data,np.full((1,5,1),np.nan),.5)
        self.assertEqual(report['event_recall'],0.)
        self.assertEqual(report['abstentions'],5)
        self.assertEqual(report['positive_target_recall_including_abstentions'],0.)


class HABWorkflowTests(unittest.TestCase):
    def test_unconfigured_api_has_no_chlorophyll_fallback(self):
        with patch.dict(os.environ,{},clear=True),TestClient(app) as client:
            state=client.get('/api/hab/status').json()
            self.assertEqual(state['status'],'not_configured')
            result=client.post('/api/hab/predict',json={'input_end_date':'2024-02-15','historical_demo':True}).json()
            self.assertEqual(result['status'],'not_configured')
            self.assertFalse(result['operational_warnings_available'])
            self.assertNotIn('daily_predictions',result)

    def test_synthetic_training_freeze_evaluation_api_and_input_abstention(self):
        # Real tiny TensorFlow training tests plumbing, never scientific performance.
        with TemporaryDirectory() as tmp:
            root=Path(tmp);processed,_,_,spec_path,path,_=fixture(root)
            dataset,run=root/'dataset',root/'run'
            prepare(spec_path,path,dataset,processed)
            record=train(dataset,run,epochs=1,batch_size=2)
            self.assertEqual(record['state'],'trained')
            with self.assertRaises(ValueError):train(dataset,run,epochs=1)
            # Use explicitly controlled software-test scores for deterministic
            # calibration/freeze tests; they are not trained-model skill evidence.
            calibration_data=collect(run,'calibration')
            calibration_data['scores']=np.where(calibration_data['truth']==1,.9,.1)
            calibration_data['logistic']=calibration_data['scores'].copy()
            with patch('prediction.hab_evaluation.collect',return_value=calibration_data):
                calibrate(run)
            with self.assertRaises(ValueError):calibrate(run)
            report=evaluate(run)
            self.assertEqual(report['split'],'test');self.assertFalse(report['operational_warning_validated'])
            self.assertIn('test-zone',report['methods']['model']['by_zone'])
            self.assertEqual(report['assessed_event_count']+report['events_without_split_safe_target_windows'],9)
            with self.assertRaises(FileExistsError):evaluate(run)
            verify_decision(run)
            with patch.dict(os.environ,{'HAB_OCCURRENCE_RUN_DIR':str(run)}),TestClient(app) as client:
                self.assertEqual(client.get('/api/hab/status').json()['status'],'configured')
                self.assertEqual(client.post('/api/hab/predict',json={'input_end_date':'2021-03-01'}).status_code,422)
                response=client.post('/api/hab/predict',json={'input_end_date':'2021-03-01','historical_demo':True})
                self.assertEqual(response.status_code,200,response.text)
                result=response.json();self.assertEqual(len(result['daily_predictions']),5)
                self.assertEqual(result['forecast_dates'][-1],'2021-03-06')
                self.assertFalse(result['operational_warnings_available'])
                self.assertIsNotNone(result['daily_predictions'][0]['zones'][0]['probability'])
                self.assertEqual(client.get('/api/hab/evaluation').status_code,200)
                with patch('preprocessing.hab_sequences.HABInputs.transform',return_value=np.zeros((2,4,4,16),np.float32)),patch('prediction.hab_evaluation.load_model') as model:
                    missing=client.post('/api/hab/predict',json={'input_end_date':'2021-03-01','historical_demo':True}).json()
                    model.assert_not_called();self.assertEqual(missing['status'],'insufficient_data')
                    self.assertIsNone(missing['daily_predictions'][0]['zones'][0]['probability'])
                with (run/'model.keras').open('ab') as handle:handle.write(b'tampered')
                self.assertEqual(client.get('/api/hab/evaluation').status_code,503)
                self.assertEqual(client.post('/api/hab/predict',json={'input_end_date':'2021-03-01','historical_demo':True}).status_code,503)


if __name__=='__main__':
    unittest.main()
