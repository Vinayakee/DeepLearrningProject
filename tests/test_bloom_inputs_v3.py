"""Synthetic causality tests; no biological validation claim."""
import unittest
from copy import deepcopy
from types import SimpleNamespace
import numpy as np
from preprocessing.bloom_inputs_v3 import daily_channels,GROUPS,coastal_distance

class InputCausalityTests(unittest.TestCase):
    def fixture(self):
        rng=np.random.default_rng(42);shape=(12,4,5)
        p=SimpleNamespace(ocean=np.ones((4,5),bool),dates=np.arange('2024-02-01','2024-02-13',dtype='datetime64[D]'),
            asof=rng.choice([0,1,255],shape),latitude=np.arange(4)+8.,longitude=np.arange(5)+74.)
        p.values={n:rng.uniform(1,10,shape) for n in ('chlorophyll','sst','wind_u','wind_v')}
        p.masks={n:rng.random(shape)>.25 for n in p.values}
        p.valid=lambda name,indices:(p.values[name][indices],p.masks[name][indices]&p.ocean)
        return p
    def compute(self,p,origin):
        state=np.zeros((4,5),np.uint8);seen=np.full((4,5),-1)
        coast=np.arange(20).reshape(4,5)
        for day in range(origin+1):features,state,seen=daily_channels(p,day,state,seen,coast)
        return features
    def test_each_new_channel_ignores_every_future_value_mask_and_label(self):
        p=self.fixture();before=self.compute(p,6)
        p.asof[7:]=1-p.asof[7:]
        for n in p.values:p.values[n][7:]=1e9;p.masks[n][7:]=False
        after=self.compute(p,6)
        for name in GROUPS['V3']:
            with self.subTest(channel=name):np.testing.assert_array_equal(before[name],after[name])
    def test_missing_history_not_absence_and_no_future_backfill(self):
        p=self.fixture();p.asof[:7]=255
        result=self.compute(p,6)
        self.assertFalse(result['last_event_known'].any());self.assertTrue(np.isnan(result['observation_age']).all())
        p.asof[2,0,0]=1;result=self.compute(p,6)
        self.assertEqual(result['last_event'][0,0],1);self.assertEqual(result['observation_age'][0,0],4)
    def test_chlorophyll_change_uses_only_observed_past_window(self):
        p=self.fixture();p.masks['chlorophyll'][:]=False
        p.masks['chlorophyll'][[3,5],0,0]=True;p.values['chlorophyll'][3,0,0]=2;p.values['chlorophyll'][5,0,0]=6
        result=self.compute(p,6)
        self.assertEqual(result['chl_change'][0,0],2)
        self.assertFalse(result['chl_change_valid'][1,1]);self.assertTrue(np.isnan(result['chl_change'][1,1]))
    def test_imputed_extra_channels_do_not_count_as_original_observations(self):
        from unittest.mock import patch
        from prediction.bloom_service import generate
        X=np.zeros((5,4,5,31),np.float32);X[...,16:]=1
        pipeline=SimpleNamespace(dates=np.arange('2024-02-11','2024-02-16',dtype='datetime64[D]'),
            shape=(5,4,5),ocean=np.ones((4,5),bool),transform=lambda indices:X)
        run=({'model_sha256':'fixture','input_variant':'V3'},pipeline,{'threshold':.5},{'limitations':['Software test only.']})
        with patch('prediction.bloom_service.ready_run',return_value=run),patch('prediction.bloom_service.cached_model') as model:
            result=generate('2024-02-15',True,'run_03')
        model.assert_not_called()
        self.assertEqual(result['status'],'insufficient_data')
        self.assertEqual(result['data_quality']['valid_observations'],0)

    def test_physical_land_distance_not_observation_mask(self):
        geo={'type':'FeatureCollection','features':[{'type':'Feature','geometry':{'type':'Polygon','coordinates':[[[75,8],[77,8],[77,12],[75,12],[75,8]]]}}]}
        result=coastal_distance(np.array([9.,10.]),np.array([74.,76.]),geo)
        self.assertTrue((result[:,0]>100).all());self.assertTrue((result[:,1]==0).all())

if __name__=='__main__':unittest.main()
