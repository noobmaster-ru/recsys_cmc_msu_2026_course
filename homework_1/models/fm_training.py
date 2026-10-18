"""Scratch FM training variants; frozen v2 implementation remains unchanged.

BCE uses the exact v2 loss/gradient. BPR uses the complete FM score difference,
with mean active-feature L2 across positive and negative feature rows. Both use
sparse AdaGrad; sampler support is the fit catalog minus ALL observed fit pairs.
"""
from pathlib import Path
import hashlib
import json
from time import perf_counter
import numpy as np
from models.fm import FM,fm_predict,fm_loss_gradient,sample_training


class NegativeSampler:
    def __init__(self,data,scheme='uniform'):
        if scheme not in ['uniform','popularity']:raise ValueError('Unknown sampler')
        self.data=data;self.scheme=scheme;self.n_items=len(data.item_ids)
        coo=data.observed.tocoo()
        self.observed=np.sort(coo.row.astype(np.int64)*self.n_items+coo.col)
        weights=(np.asarray(data.positive.sum(axis=0)).ravel()+1.)**.75
        self.probabilities=weights/weights.sum()

    def draw(self,users,rng):
        users=np.asarray(users,dtype=np.int32)
        if np.any(self.data.history_sizes[users]>=self.n_items):raise ValueError('No unseen negative exists')
        def proposal(n):
            if self.scheme=='uniform':return rng.integers(self.n_items,size=n,dtype=np.int32)
            return rng.choice(self.n_items,size=n,p=self.probabilities).astype(np.int32)
        items=proposal(len(users));pending=np.arange(len(users))
        while len(pending):
            codes=users[pending].astype(np.int64)*self.n_items+items[pending]
            positions=np.searchsorted(self.observed,codes)
            inside=positions<len(self.observed);collision=np.zeros(len(codes),dtype=bool)
            collision[inside]=self.observed[positions[inside]]==codes[inside]
            pending=pending[collision]
            items[pending]=proposal(len(pending))
        return items


def bpr_loss_gradient(pos_indices,pos_x,neg_indices,neg_x,bias,linear,factors,regularization):
    """Full-FM pairwise logistic loss; repeated feature gradients are summed."""
    n=len(pos_indices)
    positive=fm_predict(pos_indices,pos_x,bias,linear,factors)
    negative=fm_predict(neg_indices,neg_x,bias,linear,factors)
    difference=positive-negative
    loss=float(np.logaddexp(0.,-difference).mean())
    derivative=-np.exp(-np.logaddexp(0.,difference))/n
    indices=np.concatenate([pos_indices,neg_indices]);x=np.concatenate([pos_x,neg_x])
    dz=np.r_[derivative,-derivative]
    v=factors[indices];weighted=v*x[:,:,None];summed=weighted.sum(axis=1)
    loss+=.5*regularization*np.mean(np.sum(x*x*(linear[indices]**2+np.sum(v*v,axis=2)),axis=1))
    gw=dz[:,None]*x+regularization/(2*n)*x*x*linear[indices]
    gv=dz[:,None,None]*x[:,:,None]*(summed[:,None,:]-weighted)+regularization/(2*n)*x[:,:,None]**2*v
    unique,inverse=np.unique(indices.ravel(),return_inverse=True)
    grad_w=np.zeros(len(unique));grad_v=np.zeros((len(unique),factors.shape[1]))
    np.add.at(grad_w,inverse,gw.ravel());np.add.at(grad_v,inverse,gv.reshape(-1,factors.shape[1]))
    return loss,unique,grad_w,grad_v,0.


class TunableFM(FM):
    def __init__(self,*,loss='bce',sampler='uniform',negatives_per_positive=2,
                 resample=True,init_scale=.01,negative_seed=137,**kwargs):
        super().__init__(**kwargs)
        if loss not in ['bce','bpr']:raise ValueError('Unknown loss')
        if isinstance(negatives_per_positive,bool) or not isinstance(negatives_per_positive,int) or negatives_per_positive<1:raise ValueError('negative ratio')
        if not np.isfinite(init_scale) or init_scale<=0:raise ValueError('init_scale')
        self.loss=loss;self.sampler_scheme=sampler;self.ratio=negatives_per_positive
        self.resample=bool(resample);self.init_scale=init_scale;self.negative_seed=negative_seed

    def initialize_training(self,data,features,frame):
        self.data_=data;self.features_=features;self.item_ids=data.item_ids.copy();self.fit_fingerprint=data.fingerprint
        if features.fit_fingerprint!=data.fingerprint:raise ValueError('Fit/feature mismatch')
        # Reuse the audited v2 sampler to obtain identical ordering and labels.
        original=sample_training(data,frame,negatives_per_positive=self.ratio,seed=self.negative_seed)
        self.observed_users_=original[0][:len(frame)];self.observed_items_=original[1][:len(frame)]
        self.observed_y_=original[2][:len(frame)]
        self.positive_users_=self.observed_users_[self.observed_y_==1]
        self.positive_items_=self.observed_items_[self.observed_y_==1]
        if not len(self.positive_users_):raise ValueError('Training requires positive fit events')
        self.negative_sampler_=NegativeSampler(data,self.sampler_scheme)
        self.fixed_examples_=original if self.sampler_scheme=='uniform' else self._examples(0)
        count=len(features.feature_names);rng=np.random.default_rng(self.seed)
        self.linear_=np.zeros(count);self.factors_=rng.normal(0,self.init_scale,size=(count,self.n_factors));self.factors_[0]=0
        y=self.fixed_examples_[2];self.bias_=float(np.log(y.mean()/(1-y.mean()))) if self.loss=='bce' else 0.
        self.acc_w_=np.zeros(count);self.acc_v_=np.zeros_like(self.factors_);self.acc_b_=0.
        self.shuffle_=np.random.default_rng(self.shuffle_seed);self.epoch_=0
        probe=np.random.default_rng(991).choice(len(y),size=min(8192,len(y)),replace=False)
        self.probe_indices_,self.probe_x_=features.batch(self.fixed_examples_[0][probe],self.fixed_examples_[1][probe]);self.probe_y_=y[probe]
        self.curve_=[];self.sampling_history_=[]
        return self

    def fit(self,data,features,frame,*,progress_callback=None):
        """The tunable API accepts the dated fit DataFrame, not fixed triples."""
        if not hasattr(frame,'columns'):raise ValueError('TunableFM.fit expects a fit DataFrame')
        return self.initialize_training(data,features,frame).train_until(self.n_epochs,progress_callback=progress_callback)

    def score_batch(self,user_ids,*,query_timestamp=None):
        user_ids=list(user_ids)
        scores=super().score_batch(user_ids,query_timestamp=query_timestamp)
        if self.loss=='bpr':
            rows=np.array([self.data_.user_index[u] for u in user_ids],dtype=np.int64)
            unavailable=np.diff(self.data_.positive.indptr)[rows]==0
            scores[unavailable]=np.nan  # Entire ranking uses the common fit-only fallback.
        return scores

    def training_signature(self):
        return {'loss':self.loss,'sampler':self.sampler_scheme,'ratio':self.ratio,'resample':self.resample,
            'init_scale':self.init_scale,'negative_seed':self.negative_seed,'n_factors':self.n_factors,
            'learning_rate':self.learning_rate,'regularization':self.regularization,'batch_size':self.batch_size,
            'seed':self.seed,'shuffle_seed':self.shuffle_seed}

    def _examples(self,epoch):
        users=np.repeat(self.positive_users_,self.ratio)
        seed=self.negative_seed+epoch if self.resample else self.negative_seed
        negative=self.negative_sampler_.draw(users,np.random.default_rng(seed))
        return (np.r_[self.observed_users_,users],np.r_[self.observed_items_,negative],
                np.r_[self.observed_y_,np.zeros(len(users))],{})

    def train_until(self,epoch,*,progress_callback=None):
        if epoch<self.epoch_:raise ValueError('Cannot reverse training')
        for step in range(self.epoch_,epoch):
            started=perf_counter()
            examples=self._examples(step) if self.resample else self.fixed_examples_
            users,items,y=examples[:3]
            unseen=items[len(self.observed_y_):]
            negative_users=users[len(self.observed_y_):]
            codes=negative_users.astype(np.int64)*len(self.item_ids)+unseen
            sample_hash=hashlib.sha256(unseen.tobytes()).hexdigest()
            self.sampling_history_.append({'epoch':step+1,'negative_seed':self.negative_seed+step if self.resample else self.negative_seed,
                'sampled_draws':len(unseen),'unique_unseen_pairs':len(np.unique(codes)),'negative_items_sha256':sample_hash})
            if self.loss=='bce':
                order=self.shuffle_.permutation(len(y));total=len(y)
            else:
                pair_users=np.repeat(self.positive_users_,self.ratio)
                pair_positive=np.repeat(self.positive_items_,self.ratio)
                order=self.shuffle_.permutation(len(unseen));total=len(unseen)
            online=0.
            for start in range(0,total,self.batch_size):
                batch=order[start:start+self.batch_size]
                if self.loss=='bce':
                    ix,x=self.features_.batch(users[batch],items[batch])
                    objective,unique,gw,gv,gb=fm_loss_gradient(ix,x,y[batch],self.bias_,self.linear_,self.factors_,self.regularization)
                else:
                    pi,px=self.features_.batch(pair_users[batch],pair_positive[batch])
                    ni,nx=self.features_.batch(pair_users[batch],unseen[batch])
                    objective,unique,gw,gv,gb=bpr_loss_gradient(pi,px,ni,nx,self.bias_,self.linear_,self.factors_,self.regularization)
                online+=len(batch)*objective
                active=unique!=0;unique=unique[active];gw=gw[active];gv=gv[active]
                self.acc_w_[unique]+=gw*gw;self.acc_v_[unique]+=gv*gv;self.acc_b_+=gb*gb
                self.linear_[unique]-=self.learning_rate*gw/(np.sqrt(self.acc_w_[unique])+1e-8)
                self.factors_[unique]-=self.learning_rate*gv/(np.sqrt(self.acc_v_[unique])+1e-8)
                self.bias_-=self.learning_rate*gb/(np.sqrt(self.acc_b_)+1e-8)
            self.epoch_=step+1
            if not np.isfinite(self.factors_).all() or not np.isfinite(self.linear_).all():raise FloatingPointError('Diverged FM')
            probe=fm_predict(self.probe_indices_,self.probe_x_,self.bias_,self.linear_,self.factors_)
            row={'epoch':self.epoch_,'online_objective':online/total,
                'probe_bce':float(np.mean(np.logaddexp(0.,probe)-self.probe_y_*probe)),
                'seconds':perf_counter()-started,'mean_user_factor_norm':float(np.mean(np.linalg.norm(self.factors_[1:1+self.features_.n_users],axis=1)))}
            self.curve_.append(row)
            if progress_callback:progress_callback(row)
        self.prepare_scoring()
        return self

    def save_state(self,path):
        np.savez(path,linear=self.linear_,factors=self.factors_,bias=np.array(self.bias_),
            acc_w=self.acc_w_,acc_v=self.acc_v_,acc_b=np.array(self.acc_b_),epoch=np.array(self.epoch_),
            shuffle_state=np.array(json.dumps(self.shuffle_.bit_generator.state)),
            fit_fingerprint=np.array(self.fit_fingerprint),feature_fingerprint=np.array(self.features_.fingerprint),
            training_signature=np.array(json.dumps(self.training_signature(),sort_keys=True)))

    def load_state(self,path):
        with np.load(path,allow_pickle=False) as z:
            if json.loads(str(z['training_signature']))!=self.training_signature():raise ValueError('Checkpoint training configuration mismatch')
            if str(z['fit_fingerprint'])!=self.fit_fingerprint or str(z['feature_fingerprint'])!=self.features_.fingerprint:
                raise ValueError('Checkpoint fit/features mismatch')
            self.linear_=z['linear'];self.factors_=z['factors'];self.bias_=float(z['bias'])
            self.acc_w_=z['acc_w'];self.acc_v_=z['acc_v'];self.acc_b_=float(z['acc_b']);self.epoch_=int(z['epoch'])
            self.shuffle_.bit_generator.state=json.loads(str(z['shuffle_state']))
        self.prepare_scoring()
        return self
