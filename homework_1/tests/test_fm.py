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
        self.features=MetadataFeatures.from_history(self.data,self.frame,10,minimum_items=1)

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

    def test_metadata_is_fit_only_and_rating_independent(self):
        other=MetadataFeatures.from_history(self.data,self.frame,10,minimum_items=1)
        self.assertEqual(other.fingerprint,self.features.fingerprint)
        self.assertTrue(other.diagnostics['all_source_timestamps_before_query'])
        self.assertFalse(other.diagnostics['snapshot_fields_used'])
        self.assertFalse(other.diagnostics['rating_values_used_in_metadata'])
        self.assertFalse(any(k in n for n in other.feature_names for k in ['brand:','category:']))
        poisoned=self.frame.assign(brand='FUTURE_SNAPSHOT',reviewText='future edited review')
        ignored=MetadataFeatures.from_history(self.data,poisoned,10,minimum_items=1)
        self.assertEqual(ignored.fingerprint,other.fingerprint)
        # Changing known rating labels does not change the numerical side features.
        relabeled=self.frame.copy();relabeled['rating']=6-relabeled.rating
        relabeled_data=FitData.from_frame(relabeled)
        rebuilt=MetadataFeatures.from_history(relabeled_data,relabeled,10,minimum_items=1)
        np.testing.assert_array_equal(rebuilt.item_values,other.item_values)
        self.assertEqual(rebuilt.feature_names,other.feature_names)
        rare=MetadataFeatures.from_history(self.data,self.frame,10,minimum_items=5)
        self.assertEqual(rare.diagnostics['first_year_vocabulary'],0)
        self.assertIn('first_year:UNK',rare.feature_names)

    def test_metadata_time_guard_and_hand_computed_timestamp_features(self):
        with self.assertRaises(ValueError):
            MetadataFeatures.from_history(self.data,self.frame,6)
        future=self.frame.copy();future.loc[0,'timestamp']=20
        with self.assertRaises(ValueError):
            MetadataFeatures.from_history(FitData.from_frame(future),future,10)
        with self.assertRaises(ValueError):
            MetadataFeatures.from_history(self.data,future,30)
        row=self.data.item_index['B'];features=self.features
        values={features.feature_names[j]:v for j,v in zip(features.item_indices[row],features.item_values[row])}
        self.assertEqual(values['first_year:1970'],1.)
        self.assertEqual(values['first_month:01'],1.)
        self.assertAlmostEqual(values['history_count_log'],np.log1p(2)/np.log1p(1000))
        self.assertAlmostEqual(values['history_age_log_days'],np.log1p(8/86400)/np.log1p(3650))
        self.assertAlmostEqual(values['recency_log_days'],np.log1p(7/86400)/np.log1p(3650))
        ids=MetadataFeatures.from_history(self.data,self.frame,10,use_metadata=False)
        self.assertEqual(ids.item_indices.shape[1],1)
        self.assertEqual(len(ids.feature_names),1+3+4)

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
