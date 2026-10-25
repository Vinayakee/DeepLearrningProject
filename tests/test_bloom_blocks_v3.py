"""Block aggregation counts only matched observed targets, never unknowns."""
import unittest
import numpy as np
from training.train_bloom_inputs_v3 import block_counts

class BlockTargetTests(unittest.TestCase):
    def test_at_least_ten_known_targets_and_unknown_predictions_excluded(self):
        y=np.full((10,10),255,np.uint8);pred=np.ones((10,10),bool)
        y.flat[:9]=0
        self.assertEqual(int(block_counts(y,pred,y!=255).sum()),0)
        y.flat[9]=0;pred.flat[:10]=False
        np.testing.assert_array_equal(block_counts(y,pred,y!=255),[1,0,0,0])
        y.flat[0]=1;pred.flat[0]=True
        np.testing.assert_array_equal(block_counts(y,pred,y!=255),[0,0,0,1])
    def test_matched_blocks_restrict_truth_to_same_persistence_known_pixels(self):
        y=np.zeros((10,10),np.uint8);y[9,9]=1
        common=np.zeros((10,10),bool);common.flat[:10]=True
        pred=np.zeros((10,10),bool);pred[9,9]=True
        np.testing.assert_array_equal(block_counts(y,pred,common),[1,0,0,0])
    def test_partial_edge_blocks_padding_remains_invalid(self):
        y=np.zeros((12,14),np.uint8);pred=np.zeros(y.shape,bool)
        result=block_counts(y,pred,np.ones(y.shape,bool))
        # 100,40,20 valid pixels pass, final 2x4=8 does not.
        np.testing.assert_array_equal(result,[3,0,0,0])

if __name__=='__main__':unittest.main()
