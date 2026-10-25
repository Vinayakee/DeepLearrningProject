"""Synthetic software contracts for relaxed labels/calibration, not field evidence."""
import unittest
import numpy as np
from preprocessing.bloom_events_v2 import relaxed_labels,component_count,choose_candidate,CANDIDATES
from prediction.bloom_events_v2 import fit_decision,apply_scores


class RelaxedLabelsTests(unittest.TestCase):
    def definition(self,duration=2,gap=1):
        return {'observed_duration':duration,'max_unobserved_gap':gap,'minimum_patch_pixels':2}

    def test_gap_neither_becomes_positive_nor_counts_toward_duration(self):
        observed=np.ones((4,2,2),bool);observed[1]=False
        high=np.zeros_like(observed);high[0,0,:]=True;high[2,0,:]=True
        labels,asof=relaxed_labels(high,observed,np.ones((2,2),bool),self.definition())
        self.assertTrue((labels[0,0,:]==1).all());self.assertTrue((labels[2,0,:]==1).all())
        self.assertTrue((labels[1]==255).all())
        self.assertTrue((asof[0,0,:]==255).all());self.assertTrue((asof[2,0,:]==1).all())
        self.assertEqual(component_count(labels==1,observed,1),1)
        high[2]=False
        one,_=relaxed_labels(high,observed,np.ones((2,2),bool),self.definition())
        self.assertFalse((one==1).any())

    def test_observed_low_breaks_event_and_gap_cap_is_enforced(self):
        high=np.zeros((5,2,2),bool);high[[0,3],0,:]=True
        observed=np.ones_like(high);observed[1:3]=False
        too_far,_=relaxed_labels(high,observed,np.ones((2,2),bool),self.definition(gap=1))
        self.assertFalse((too_far==1).any())
        linked,_=relaxed_labels(high,observed,np.ones((2,2),bool),self.definition(gap=2))
        self.assertEqual(int((linked==1).sum()),4)
        observed[1]=True
        interrupted,_=relaxed_labels(high,observed,np.ones((2,2),bool),self.definition(gap=2))
        self.assertFalse((interrupted==1).any())

    def test_single_day_is_causal_and_unknown_cells_stay_unknown(self):
        high=np.zeros((2,3,3),bool);high[0,1,:2]=True
        observed=np.ones_like(high);observed[1]=False
        labels,asof=relaxed_labels(high,observed,np.ones((3,3),bool),self.definition(duration=1,gap=0))
        self.assertEqual(int((labels==1).sum()),2)
        np.testing.assert_array_equal(labels,asof)
        self.assertTrue((labels[1]==255).all())

    def test_first_candidate_selection_ignores_test_counts(self):
        good={'events':100,'positive_pixel_days':2000,'positive_fraction_of_valid':.1}
        rows=[{'definition':CANDIDATES[0],'train':good,'validation':good,'test_label_presence_only':{'event_component_count':0}},
              {'definition':CANDIDATES[1],'train':good,'validation':good,'test_label_presence_only':{'event_component_count':100000}}]
        self.assertEqual(choose_candidate(rows)['id'],'A')
        rows[0]['validation']={**good,'events':0}
        self.assertEqual(choose_candidate(rows)['id'],'B')
        rows[1]['validation']={**good,'events':0}
        self.assertIsNone(choose_candidate(rows))


class RelaxedCalibrationTests(unittest.TestCase):
    def test_insufficient_validation_does_not_claim_calibration(self):
        y=np.array([0,1]);s=np.array([.1,.9])
        decision=fit_decision(y,s,y,s,1,1)
        self.assertFalse(decision['calibration']['active']);self.assertFalse(decision['threshold_tuned'])
        np.testing.assert_array_equal(apply_scores(s,decision),s)

    def test_platt_and_threshold_use_observed_validation_classes(self):
        y=np.tile([0,0,0,1],200)
        scores=np.where(y==1,.65,.35)
        decision=fit_decision(y,scores,y,scores,20,10)
        self.assertTrue(decision['calibration']['active']);self.assertTrue(decision['threshold_tuned'])
        self.assertLessEqual(decision['calibration']['calibrated_assessment_brier'],decision['calibration']['raw_assessment_brier'])
        self.assertEqual(decision['fit_metrics']['f1'],1.)
        self.assertTrue(np.isfinite(apply_scores(np.array([0.,1.]),decision)).all())


if __name__=='__main__':unittest.main()
