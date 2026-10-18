"""Independent training, sampling, pairwise derivative and resumption oracles."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import tempfile
import unittest
import numpy as np
import pandas as pd
from data_utils import FitData
from models.fm import FM,MetadataFeatures,sample_training,fm_predict
from models.fm_training import TunableFM,NegativeSampler,bpr_loss_gradient
from scripts.tune_fm_extended import ValidationOnlyGuard


class FMTrainingTests(unittest.TestCase):
    def setUp(self):
        self.frame=pd.DataFrame([('a','A',5,1),('a','B',1,2),('b','B',4,3),('b','C',5,4),('c','C',2,5),('c','D',5,6)],columns=['user_id','item_id','rating','timestamp'])
        self.data=FitData.from_frame(self.frame)
        self.features=MetadataFeatures.from_history(self.data,self.frame,10,minimum_items=1)

    def test_fixed_bce_matches_v2_updates_exactly(self):
        params={'n_factors':3,'n_epochs':4,'batch_size':4,'learning_rate':.05,'regularization':1e-5}
        original=FM(**params).fit(self.data,self.features,sample_training(self.data,self.frame))
        current=TunableFM(**params,resample=False).initialize_training(self.data,self.features,self.frame).train_until(4)
        np.testing.assert_array_equal(original.factors_,current.factors_)
        np.testing.assert_array_equal(original.linear_,current.linear_)
        self.assertEqual(original.bias_,current.bias_)

    def test_popularity_sampler_has_full_support_excludes_all_seen_and_matches_conditional_law(self):
        extra=pd.DataFrame([('d','C',5,7),('e','E',1,8)],columns=self.frame.columns)
        unequal=FitData.from_frame(pd.concat([self.frame,extra],ignore_index=True))
        sampler=NegativeSampler(unequal,'popularity');users=np.repeat(0,30000)
        draws=sampler.draw(users,np.random.default_rng(9))
        self.assertTrue(set(draws).issubset({2,3,4})) # A positive and B dislike both excluded; E has no positives.
        expected=sampler.probabilities[2]/sampler.probabilities[[2,3,4]].sum()
        self.assertAlmostEqual(float(np.mean(draws==2)),expected,delta=.015)
        self.assertTrue(np.all(sampler.probabilities>0))
        np.testing.assert_array_equal(draws,sampler.draw(users,np.random.default_rng(9)))

    def test_resampling_and_checkpoint_resume_are_identical_to_uninterrupted_training(self):
        for loss in ['bce','bpr']:
            params={'loss':loss,'sampler':'popularity','n_factors':3,'batch_size':3,'resample':True}
            first=TunableFM(**params).initialize_training(self.data,self.features,self.frame).train_until(4)
            split=TunableFM(**params).initialize_training(self.data,self.features,self.frame).train_until(2)
            with tempfile.TemporaryDirectory() as temp:
                path=Path(temp)/'state.npz';split.save_state(path)
                restored=TunableFM(**params).initialize_training(self.data,self.features,self.frame).load_state(path).train_until(4)
            np.testing.assert_array_equal(first.factors_,restored.factors_)
            np.testing.assert_array_equal(first.linear_,restored.linear_)
            np.testing.assert_array_equal(first.acc_v_,restored.acc_v_)
            self.assertEqual(first.bias_,restored.bias_)
            self.assertEqual([r['negative_seed'] for r in first.sampling_history_],[137,138,139,140])

    def test_bpr_gradients_match_finite_differences_with_shared_metadata(self):
        rng=np.random.default_rng(8);w=rng.normal(size=8);v=rng.normal(size=(8,3))
        pi=np.array([[1,3,5],[2,4,5]]);ni=np.array([[1,4,6],[2,3,6]])
        px=np.array([[1.,1.,.7],[1.,1.,.3]]);nx=np.array([[1.,1.,.5],[1.,1.,.8]])
        reg=.1;eps=1e-6
        loss,unique,gw,gv,gb=bpr_loss_gradient(pi,px,ni,nx,.2,w,v,reg)
        expected=float(np.logaddexp(0.,-(fm_predict(pi,px,.2,w,v)-fm_predict(ni,nx,.2,w,v))).mean())
        self.assertGreater(loss,expected);self.assertEqual(gb,0.)
        def objective():return bpr_loss_gradient(pi,px,ni,nx,.2,w,v,reg)[0]
        for pos,j in enumerate(unique):
            w[j]+=eps;plus=objective();w[j]-=2*eps;minus=objective();w[j]+=eps
            self.assertAlmostEqual(gw[pos],(plus-minus)/(2*eps),places=8)
            for f in range(3):
                v[j,f]+=eps;plus=objective();v[j,f]-=2*eps;minus=objective();v[j,f]+=eps
                self.assertAlmostEqual(gv[pos,f],(plus-minus)/(2*eps),places=8)

    def test_bpr_users_without_positive_history_use_the_common_fallback(self):
        from evaluation import EvaluationContext,evaluate_model,evaluate_popularity
        frame=pd.concat([self.frame,pd.DataFrame([('d','A',2,7)],columns=self.frame.columns)],ignore_index=True)
        data=FitData.from_frame(frame);features=MetadataFeatures.from_history(data,frame,10,minimum_items=1)
        model=TunableFM(loss='bpr',n_factors=3,batch_size=3).initialize_training(data,features,frame).train_until(2)
        self.assertTrue(np.isnan(model.score_batch(['d'],query_timestamp=10)).all())
        self.assertTrue(np.isfinite(model.score_batch(['a'],query_timestamp=10)).all())
        self.assertEqual(model.score_batch([],query_timestamp=10).shape,(0,4))
        hold=pd.DataFrame([('d','C',5,10)],columns=frame.columns)
        context=EvaluationContext.from_frames(frame,hold,k=1,query_timestamp=10)
        result=evaluate_model(model,context);reference=evaluate_popularity(context)
        self.assertEqual(result.metrics['users'],1)
        self.assertEqual(result.metrics['fallback_rate'],1.)
        np.testing.assert_array_equal(result.recommendations,reference.recommendations)

    def test_fit_api_uses_bpr_and_checkpoint_rejects_other_training_configuration(self):
        params={'loss':'bpr','n_factors':3,'n_epochs':3,'batch_size':3}
        direct=TunableFM(**params).fit(self.data,self.features,self.frame)
        explicit=TunableFM(**params).initialize_training(self.data,self.features,self.frame).train_until(3)
        np.testing.assert_array_equal(direct.factors_,explicit.factors_)
        self.assertEqual(direct.bias_,0.)
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'state.npz';direct.save_state(path)
            other=TunableFM(**params,sampler='popularity').initialize_training(self.data,self.features,self.frame)
            with self.assertRaisesRegex(ValueError,'configuration'):other.load_state(path)

    def test_training_rejects_a_dataset_without_positive_anchors(self):
        frame=self.frame.assign(rating=1);data=FitData.from_frame(frame)
        features=MetadataFeatures.from_history(data,frame,10)
        with self.assertRaisesRegex(ValueError,'positive fit events'):
            TunableFM(loss='bpr').initialize_training(data,features,frame)

    def test_guard_blocks_test_final_and_snapshot_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);guard=ValidationOnlyGuard(root)
            for relative in ['data/splits/test.csv.gz','reports/final_test/summary.json','data/final_test/fm/parameters.npz',
                             'data/processed/items.jsonl.gz','archive/snapshot_v1/experiments.ipynb','configs/final_models.json']:
                with self.assertRaises(RuntimeError):guard('open',(str(root/relative),'r',0))
            guard('open',(str(root/'data/splits/train.csv.gz'),'r',0))
            guard('open',(str(root/'data/splits/validation.csv.gz'),'r',0))
            self.assertEqual(len(guard.blocked),6)


if __name__=='__main__':unittest.main()
