"""Independent constrained regressions and shared-protocol invariants for EASE."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from data_utils import FitData
from evaluation import EvaluationContext, evaluate_model
from models.ease import EASE


class EASETests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame([
            ('a','A',5,1),('a','B',4,2),('a','D',1,3),
            ('b','B',5,2),('b','C',4,3),('c','A',5,4),
            ('c','C',5,5),('negative','D',2,6)],
            columns=['user_id','item_id','rating','timestamp'])
        self.data=FitData.from_frame(self.frame)

    def test_hand_computed_two_item_solution(self):
        frame=pd.DataFrame([('a','A',5,1),('a','B',5,2),('b','A',4,3)],columns=self.frame.columns)
        model=EASE(regularization=1).fit(FitData.from_frame(frame))
        # G=[[2,1],[1,1]]. Predict B from A: 1/(2+1); A from B: 1/(1+1).
        np.testing.assert_allclose(model.weights_,[[0,1/3],[1/2,0]],atol=1e-14)
        self.assertFalse(np.allclose(model.weights_,model.weights_.T))
        self.assertTrue(model.weights_.flags.c_contiguous)

    def test_matches_independent_leave_target_out_ridge_regressions(self):
        x=self.data.positive.toarray();n=x.shape[1]
        for regularization in (.01,1.,1000.):
            model=EASE(regularization).fit(self.data)
            expected=np.zeros((n,n))
            for target in range(n):
                sources=np.arange(n)!=target
                design=x[:,sources]
                expected[sources,target]=np.linalg.solve(design.T@design+regularization*np.eye(n-1),design.T@x[:,target])
            np.testing.assert_allclose(model.weights_,expected,rtol=1e-11,atol=1e-12)
            loss=np.sum((x-x@expected)**2)+regularization*np.sum(expected**2)
            self.assertAlmostEqual(loss,model.diagnostics_['training_objective'],places=10)
            np.testing.assert_array_equal(model.weights_.diagonal(),0)

    def test_signed_weights_are_kept_and_zero_profiles_fallback(self):
        frame=pd.DataFrame([('a','A',5,1),('a','B',5,2),('b','B',5,3),('b','C',5,4),('n','D',1,5)],columns=self.frame.columns)
        data=FitData.from_frame(frame);model=EASE(1).fit(data)
        self.assertLess(model.weights_[data.item_index['A'],data.item_index['C']],0)
        self.assertTrue(np.isnan(model.score_batch(['n'])).all())
        np.testing.assert_array_equal(model.weights_[:,data.item_index['D']],0)
        np.testing.assert_array_equal(model.weights_[data.item_index['D'],:],0)
        # Explicit negative candidate evidence is not a missing profile.
        model.weights_[:]=-1.;np.fill_diagonal(model.weights_,0)
        self.assertTrue(np.isfinite(model.score_batch(['a'])).all())

    def test_mask_chronology_catalog_and_same_cohort(self):
        hold=pd.DataFrame([('a','C',5,10),('b','A',4,10),('negative','A',5,10)],columns=self.frame.columns)
        context=EvaluationContext.from_frames(self.frame,hold,k=1,query_timestamp=10)
        model=EASE(10).fit(self.data)
        first=evaluate_model(model,context,batch_size=1)
        second=evaluate_model(model,context,batch_size=128)
        self.assertEqual(first.metrics['users'],3)
        self.assertTrue(first.per_user.loc['negative','fallback'])
        np.testing.assert_array_equal(first.recommendations,second.recommendations)
        for row,chosen in zip(context.user_indices,first.recommendations):
            self.assertTrue(set(chosen).isdisjoint(self.data.seen_items(row)))
        with self.assertRaises(ValueError):model.score_batch(['new'])
        with self.assertRaises(ValueError):model.score_batch(['a'],query_timestamp=5)
        with self.assertRaises(RuntimeError):EASE().score_batch(['a'])

    def test_regularization_domain(self):
        for value in (0,-1,np.nan,np.inf,True):
            with self.assertRaises(ValueError):EASE(value)


if __name__=='__main__':unittest.main()
