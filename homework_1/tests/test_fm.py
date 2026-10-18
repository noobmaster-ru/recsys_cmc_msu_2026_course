"""FM mathematical, metadata-preprocessing, sampling and protocol oracles."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from data_utils import FitData
from evaluation import EvaluationContext,evaluate_model
from models.fm import FM,MetadataFeatures,fm_predict,fm_loss_gradient,sample_training


class FMTests(unittest.TestCase):
    def setUp(self):
        self.frame=pd.DataFrame([('a','A',5,1),('a','B',1,2),('b','B',4,3),('b','C',5,4),('c','C',2,5),('c','D',5,6)],columns=['user_id','item_id','rating','timestamp'])
        self.data=FitData.from_frame(self.frame)
        self.records={'A':{'brand':'One','categories':['Video Games','PC','Games']},'B':{'brand':'Two','categories':['PC','Games']},'C':{'brand':'One','categories':['Console']},'D':{}}
        self.features=MetadataFeatures.from_items(self.data,self.records,minimum_items=1)

    def test_fm_formula_matches_explicit_pairwise_sum(self):
        rng=np.random.default_rng(5);w=rng.normal(size=8);v=rng.normal(size=(8,3))
        ix=np.array([[1,3,5,0],[2,4,6,7]]);x=np.array([[1.,1.,.5,0.],[1.,1.,.7,.7]])
        actual=fm_predict(ix,x,.2,w,v);expected=[]
        for ids,values in zip(ix,x):
            score=.2+sum(w[j]*a for j,a in zip(ids,values))
            for a in range(len(ids)):
                for b in range(a+1,len(ids)):score+=np.dot(v[ids[a]],v[ids[b]])*values[a]*values[b]
            expected.append(score)
        np.testing.assert_allclose(actual,expected,rtol=1e-13,atol=1e-13)

    def test_gradient_matches_finite_differences_with_repeated_features(self):
        rng=np.random.default_rng(7);w=rng.normal(size=7);v=rng.normal(size=(7,2));bias=.3
        ix=np.array([[1,3,5,0],[2,3,6,0],[1,4,6,5]]);x=np.array([[1,1,.7,0],[1,1,1,0],[1,1,.5,.5]])
        y=np.array([1.,0.,1.]);reg=.13;eps=1e-6
        loss,unique,gw,gv,gb=fm_loss_gradient(ix,x,y,bias,w,v,reg)
        def objective():return fm_loss_gradient(ix,x,y,bias,w,v,reg)[0]
        bias+=eps;plus=objective();bias-=2*eps;minus=objective();bias+=eps
        self.assertAlmostEqual(gb,(plus-minus)/(2*eps),places=8)
        for pos,j in enumerate(unique):
            w[j]+=eps;plus=objective();w[j]-=2*eps;minus=objective();w[j]+=eps
            self.assertAlmostEqual(gw[pos],(plus-minus)/(2*eps),places=8)
            for f in range(v.shape[1]):
                v[j,f]+=eps;plus=objective();v[j,f]-=2*eps;minus=objective();v[j,f]+=eps
                self.assertAlmostEqual(gv[pos,f],(plus-minus)/(2*eps),places=8)

    def test_metadata_vocabulary_is_fit_only_with_category_normalization(self):
        expanded={**self.records,'FUTURE':{'brand':'NeverTrain','categories':['FutureCategory']}}
        other=MetadataFeatures.from_items(self.data,expanded,minimum_items=1)
        self.assertEqual(other.fingerprint,self.features.fingerprint)
        self.assertFalse(any('future' in name for name in other.feature_names))
        self.assertFalse(any(name=='category:video games' for name in other.feature_names))
        for row,values in zip(other.item_indices,other.item_values):
            cats=[p for p,j in enumerate(row) if other.feature_names[j].startswith('category:')]
            self.assertAlmostEqual(float(np.sum(values[cats]**2)),1.)
        rare=MetadataFeatures.from_items(self.data,self.records,minimum_items=3)
        self.assertEqual(rare.diagnostics['brand_vocabulary'],0)
        self.assertIn('brand:UNK',rare.feature_names)

    def test_negatives_exclude_all_fit_ratings_and_are_reproducible(self):
        first=sample_training(self.data,self.frame,negatives_per_positive=2,seed=137)
        second=sample_training(self.data,self.frame,negatives_per_positive=2,seed=137)
        for a,b in zip(first[:3],second[:3]):np.testing.assert_array_equal(a,b)
        for u,i in zip(first[0][len(self.frame):],first[1][len(self.frame):]):
            self.assertNotIn(i,self.data.seen_items(u))
        self.assertEqual(first[3]['observed_negative'],2)
        self.assertEqual(first[3]['sampled_unobserved'],8)
        self.assertEqual(first[3]['fingerprint'],second[3]['fingerprint'])

    def test_fast_catalog_scores_equal_full_fm_formula(self):
        training=sample_training(self.data,self.frame)
        model=FM(n_factors=3,n_epochs=3,batch_size=4).fit(self.data,self.features,training)
        users=np.repeat(np.arange(3),4);items=np.tile(np.arange(4),3)
        ix,x=self.features.batch(users,items)
        expected=fm_predict(ix,x,model.bias_,model.linear_,model.factors_).reshape(3,4)
        actual=model.score_batch(self.data.user_ids,query_timestamp=10)
        np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-12)
        np.testing.assert_array_equal(model.factors_[0],0)
        self.assertEqual(model.linear_[0],0)

    def test_training_reproducibility_loss_and_common_protocol(self):
        training=sample_training(self.data,self.frame)
        models=[FM(n_factors=3,n_epochs=15,batch_size=4).fit(self.data,self.features,training) for _ in range(2)]
        np.testing.assert_array_equal(models[0].factors_,models[1].factors_)
        self.assertLess(models[0].curve_[-1]['probe_bce'],models[0].curve_[0]['probe_bce'])
        hold=pd.DataFrame([('a','C',5,10),('b','A',5,10),('c','A',4,10)],columns=self.frame.columns)
        context=EvaluationContext.from_frames(self.frame,hold,k=1,query_timestamp=10)
        a=evaluate_model(models[0],context,batch_size=1);b=evaluate_model(models[0],context,batch_size=128)
        np.testing.assert_array_equal(a.recommendations,b.recommendations)
        self.assertEqual(a.metrics['users'],3)
        for row,chosen in zip(context.user_indices,a.recommendations):self.assertTrue(set(chosen).isdisjoint(self.data.seen_items(row)))
        with self.assertRaises(ValueError):models[0].score_batch(['new'])
        with self.assertRaises(ValueError):models[0].score_batch(['a'],query_timestamp=5)
        with self.assertRaises(RuntimeError):FM().score_batch(['a'])


if __name__=='__main__':unittest.main()
