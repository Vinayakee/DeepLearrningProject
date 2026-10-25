"""Synthetic software fixtures only; never satellite bloom-event skill results."""
import unittest
import numpy as np
import tensorflow as tf
from preprocessing.bloom_events import patch_filter,confirmed_membership,event_labels,event_summary
from models.bloom_events import build_bloom_event_model,MaskedBinaryLoss


class SatelliteEventLabelsTests(unittest.TestCase):
    def test_missing_inputs_do_not_load_or_execute_the_satellite_model(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from prediction.bloom_service import generate
        pipeline=SimpleNamespace(dates=np.arange(np.datetime64('2024-02-11'),np.datetime64('2024-02-16')),
                                 shape=(5,4,4),ocean=np.ones((4,4),bool),
                                 transform=lambda indices:np.zeros((5,4,4,16),np.float32))
        run=({'model_sha256':'software-fixture'},pipeline,{'threshold':.5},{'limitations':['Software fixture only.']})
        with patch('prediction.bloom_service.ready_run',return_value=run),patch('prediction.bloom_service.cached_model') as model:
            result=generate('2024-02-15',historical_demo=True)
            model.assert_not_called()
            self.assertEqual(result['status'],'insufficient_data')
            self.assertEqual(result['maps'],[])

    def test_zero_positive_truth_has_undefined_recall_and_no_invented_perfect_f1(self):
        from prediction.bloom_events import metrics
        quiet=metrics([100,0,0,0])
        self.assertIsNone(quiet['recall']);self.assertIsNone(quiet['f1']);self.assertIsNone(quiet['csi'])
        self.assertEqual(quiet['valid_target_count'],100)
        false_alarms=metrics([99,1,0,0])
        self.assertEqual(false_alarms['precision'],0.)
        self.assertIsNone(false_alarms['recall']);self.assertEqual(false_alarms['false_positive_rate'],.01)

    def test_satellite_event_api_rejects_invalid_request_and_bad_artifacts(self):
        from unittest.mock import patch
        from fastapi.testclient import TestClient
        from backend.app import app
        with TestClient(app) as client:
            self.assertEqual(client.post('/api/bloom/predict',json={'input_end_date':'invalid'}).status_code,422)
            with patch('prediction.bloom_service.status',return_value={'status':'ready_research_only'}),patch('prediction.bloom_service.ready_run',side_effect=ValueError('changed')):
                self.assertEqual(client.post('/api/bloom/predict',json={'input_end_date':'2024-02-15','historical_demo':True}).status_code,503)
            with patch('prediction.bloom_service.status',return_value={'status':'training'}):
                self.assertEqual(client.post('/api/bloom/predict',json={'historical_demo':True}).json()['status'],'training')

    def test_common_three_day_connected_patch_and_causal_confirmation(self):
        high=np.zeros((6,4,4),bool);high[1:5,1,0:3]=True
        labels,asof=event_labels(high,np.ones_like(high),np.ones((4,4),bool),3)
        self.assertEqual(int((labels==1).sum()),12)
        self.assertTrue((labels[1:5,1,:3]==1).all())
        self.assertTrue((asof[1:3,1,:3]==255).all())
        self.assertTrue((asof[3:5,1,:3]==1).all())
        self.assertEqual(event_summary(labels)['events'],1)

    def test_missing_day_does_not_become_negative_or_create_event(self):
        high=np.zeros((5,4,4),bool);high[1:4,1,:3]=True
        observed=np.ones_like(high);observed[2,1,:3]=False;high[~observed]=False
        labels,_=event_labels(high,observed,np.ones((4,4),bool),3)
        self.assertEqual(int((labels==1).sum()),0)
        self.assertTrue((labels[1:4,1,:3]==255).all())
        self.assertEqual(labels[2,0,0],0)

    def test_two_day_run_and_unresolved_split_edge(self):
        high=np.zeros((7,4,4),bool);high[2:4,1,:3]=True
        labels,_=event_labels(high,np.ones_like(high),np.ones((4,4),bool),3)
        self.assertEqual(int((labels==1).sum()),0)
        self.assertTrue((labels[2:4,1,:3]==0).all())
        high[:2,1,:3]=True
        edge,_=event_labels(high[:2],np.ones_like(high[:2]),np.ones((4,4),bool),3)
        self.assertTrue((edge[:,1,:3]==255).all())

    def test_diagonal_pixels_do_not_form_a_four_connected_patch(self):
        mask=np.eye(4,dtype=bool)
        self.assertFalse(patch_filter(mask,2).any())
        high=np.ones((3,2,2),bool);support=np.array([[True,False],[True,True]])
        observed=np.broadcast_to(support,high.shape).copy();high&=observed
        labels,_=event_labels(high,observed,support,3)
        self.assertTrue((labels[:,:,1][:,0]==255).all())

    def test_binary_heads_and_masked_loss(self):
        model=build_bloom_event_model()
        X=np.zeros((1,5,120,84,16),np.float32)
        prediction=model(X,training=False).numpy()
        self.assertEqual(prediction.shape,(1,5,120,84))
        self.assertTrue(((prediction>=0)&(prediction<=1)).all())
        loss=MaskedBinaryLoss(3.)
        self.assertEqual(float(loss(np.full_like(prediction,255),prediction)),0.)


if __name__=='__main__':unittest.main()
