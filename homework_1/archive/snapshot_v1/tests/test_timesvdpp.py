"""TimeSVD++ independent equations, full-history gradients and time-boundary tests."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from data_utils import FitData
from evaluation import EvaluationContext,evaluate_model
from models.timesvdpp import TemporalData,TimeSVDpp

DAY=86400
class TimeSVDppTests(unittest.TestCase):
    def setUp(self):
        self.frame=pd.DataFrame([('a','A',5,DAY),('a','B',1,2*DAY),('a','C',4,2*DAY),('b','B',4,3*DAY),('b','D',2,4*DAY),('c','D',1,4*DAY)],columns=['user_id','item_id','rating','timestamp'])
        self.data=FitData.from_frame(self.frame);self.time=TemporalData.from_frame(self.data,self.frame)

    def random_model(self):
        m=TimeSVDpp(n_factors=2,n_bins=2,regularization=.13,n_epochs=3).initialize(self.time)
        rng=np.random.default_rng(9)
        for p in m.parameters_.values():p[:]=rng.normal(0,.1,size=p.shape)
        m.prepare_scoring();return m

    def test_formula_matches_explicit_components_and_all_rating_history(self):
        m=self.random_model();p=m.parameters_;expected=[]
        for u,i,day in zip(self.time.users,self.time.items,self.time.days):
            mean=self.time.user_mean_day[u];dev=np.sign(day-mean)*(abs(day-mean)/365.25)**m.beta
            d=self.time.lookup_user_days(np.array([u]),np.array([day]))[0]
            b=self.time.bins([day],m.n_bins)[0];history=self.data.seen_items(u)
            z=p['p'][u]+p['a'][u]*dev+p['day_p'][d]+sum(p['y'][j] for j in history)/np.sqrt(len(history))
            expected.append(m.mean_+p['bu'][u]+p['alpha_bias'][u]*dev+p['day_bias'][d]+p['bi'][i]+p['bin_bias'][i,b]+np.dot(p['q'][i],z))
        actual=m.predict_pairs(self.time.users,self.time.items,self.time.days*DAY)
        np.testing.assert_allclose(actual,expected,atol=1e-13)
        # Low rating B and the target A both belong to a's implicit history.
        np.testing.assert_allclose(self.time.history.toarray()[0],[1/np.sqrt(3)]*3+[0])

    def test_all_gradients_match_finite_differences(self):
        m=self.random_model();loss,mse,grads=m.loss_and_gradients();eps=1e-6
        for key,p in m.parameters_.items():
            for index in np.ndindex(p.shape):
                original=p[index];p[index]=original+eps;plus=m.loss_and_gradients(gradients=False)[0]
                p[index]=original-eps;minus=m.loss_and_gradients(gradients=False)[0];p[index]=original
                self.assertAlmostEqual(grads[key][index],(plus-minus)/(2*eps),places=8,msg=f'{key}{index}')

    def test_implicit_gradient_includes_every_history_item_with_normalization(self):
        m=self.random_model();p=m.parameters_;_,_,g=m.loss_and_gradients()
        prediction=m.predict_pairs(self.time.users,self.time.items,self.time.days*DAY)
        oracle=np.zeros_like(p['y'])
        for u,i,error in zip(self.time.users,self.time.items,(prediction-self.time.ratings)/len(prediction)):
            history=self.data.seen_items(u)
            for j in history:oracle[j]+=error*p['q'][i]/np.sqrt(len(history))
        oracle+=m.regularization*m.penalty_weights_['y']*p['y']
        np.testing.assert_allclose(g['y'],oracle,atol=1e-13)

    def test_fit_only_time_features_and_future_day_no_alias(self):
        np.testing.assert_allclose(self.time.user_mean_day,[5/3,3.5,4])
        np.testing.assert_array_equal(self.time.bins([-10,1,2,3,4,5,100],2),[0,0,0,1,1,1,1])
        # a at day 5 would alias b at day 1 if boundary validation were omitted.
        np.testing.assert_array_equal(self.time.lookup_user_days([0,0,1,2],[5,2,1,4]),[-1,1,-1,4])
        extended=pd.concat([self.frame,pd.DataFrame([('c','A',5,10*DAY)],columns=self.frame.columns)],ignore_index=True)
        with self.assertRaises(ValueError):TemporalData.from_frame(self.data,extended)

    def test_unseen_daily_terms_zero_but_smooth_drift_and_edge_bin_used(self):
        m=self.random_model();p=m.parameters_;users=np.array([0,1,2]);days=np.repeat(10,3)
        vector,bias,bins=m.components(users,days)
        dev=self.time.deviation(users,days,m.beta)
        expected=p['p']+p['a']*dev[:,None]+self.time.history@p['y']
        np.testing.assert_allclose(vector,expected,atol=1e-13)
        np.testing.assert_allclose(bias,p['bu']+p['alpha_bias']*dev,atol=1e-13)
        np.testing.assert_array_equal(bins,[1,1,1])
        old=m.score_batch(self.data.user_ids,query_timestamp=10*DAY)
        p['day_p'][:]=1000;p['day_bias'][:]=1000
        np.testing.assert_array_equal(old,m.score_batch(self.data.user_ids,query_timestamp=10*DAY))

    def test_training_deterministic_and_loss_prediction_consistency(self):
        models=[TimeSVDpp(n_factors=3,n_bins=2,n_epochs=20).fit_temporal(self.time) for _ in range(2)]
        for key in models[0].parameters_:np.testing.assert_array_equal(models[0].parameters_[key],models[1].parameters_[key])
        self.assertLess(models[0].curve_[-1]['training_rmse'],models[0].curve_[0]['training_rmse'])
        pred=models[0].predict_pairs(self.time.users,self.time.items,self.time.days*DAY)
        self.assertAlmostEqual(np.sqrt(np.mean((pred-self.time.ratings)**2)),models[0].curve_[-1]['training_rmse'],places=12)

    def test_static_ablation_equals_svdpp_and_temporal_terms_remain_zero(self):
        m=TimeSVDpp(n_factors=2,n_bins=2,n_epochs=5,temporal=False).fit_temporal(self.time);p=m.parameters_
        for key in ['a','alpha_bias','bin_bias','day_p','day_bias']:np.testing.assert_array_equal(p[key],0)
        expected=np.dot(p['p']+self.time.history@p['y'],p['q'].T)+m.mean_+p['bu'][:,None]+p['bi'][None,:]
        np.testing.assert_allclose(m.score_batch(self.data.user_ids,query_timestamp=10*DAY),expected,atol=1e-12)
        np.testing.assert_array_equal(m.score_batch(self.data.user_ids,query_timestamp=10*DAY),m.score_batch(self.data.user_ids,query_timestamp=100*DAY))

    def test_shared_evaluator_mask_same_users_batch_and_time_guard(self):
        hold=pd.DataFrame([('a','D',5,10*DAY),('b','A',4,10*DAY),('c','A',5,10*DAY)],columns=self.frame.columns)
        context=EvaluationContext.from_frames(self.frame,hold,k=1,query_timestamp=10*DAY)
        m=TimeSVDpp(n_factors=2,n_bins=2,n_epochs=5).fit_temporal(self.time)
        a=evaluate_model(m,context,batch_size=1);b=evaluate_model(m,context,batch_size=128)
        np.testing.assert_array_equal(a.recommendations,b.recommendations);self.assertEqual(a.metrics['users'],3)
        for row,chosen in zip(context.user_indices,a.recommendations):self.assertTrue(set(chosen).isdisjoint(self.data.seen_items(row)))
        with self.assertRaises(ValueError):m.score_batch(['new'],query_timestamp=10*DAY)
        with self.assertRaises(ValueError):m.score_batch(['a'],query_timestamp=DAY)
        with self.assertRaises(ValueError):m.score_batch(['a'])
        with self.assertRaises(RuntimeError):TimeSVDpp().score_batch(['a'],query_timestamp=10*DAY)


if __name__=='__main__':unittest.main()
