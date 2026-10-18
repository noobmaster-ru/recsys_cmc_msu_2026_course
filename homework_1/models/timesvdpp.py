"""TimeSVD++ prediction model with exact full-gradient NumPy/SciPy AdaGrad.

Includes item-bin bias, user bias drift and day bias, user factor drift/day
factors, and the normalized sum of ALL rated-item implicit vectors. Ratings
1..5 are fitted directly; no negative sampling or output clipping.
"""
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy import sparse
from data_utils import FitData


@dataclass
class TemporalData:
    fit: FitData
    users: np.ndarray
    items: np.ndarray
    ratings: np.ndarray
    days: np.ndarray
    user_mean_day: np.ndarray
    user_day_codes: np.ndarray
    user_days: np.ndarray
    user_design: sparse.csr_matrix
    item_design: sparse.csr_matrix
    day_design: sparse.csr_matrix
    history: sparse.csr_matrix
    min_day: int
    max_day: int

    @classmethod
    def from_frame(cls,data,frame):
        if not isinstance(data,FitData) or FitData.from_frame(frame).fingerprint!=data.fingerprint:
            raise ValueError('Temporal frame must match the shared fit index')
        users=frame.user_id.map(data.user_index).to_numpy(dtype=np.int32)
        items=frame.item_id.map(data.item_index).to_numpy(dtype=np.int32)
        days=frame.timestamp.to_numpy(dtype=np.int64)//86400
        ratings=frame.rating.to_numpy(dtype=np.float64)
        minimum=int(days.min());maximum=int(days.max());stride=maximum-minimum+1
        codes=users.astype(np.int64)*stride+(days-minimum)
        unique,inverse=np.unique(codes,return_inverse=True)
        n=len(frame);rows=np.arange(n);ones=np.ones(n)
        ud=sparse.csr_matrix((ones,(rows,users)),shape=(n,len(data.user_ids)))
        it=sparse.csr_matrix((ones,(rows,items)),shape=(n,len(data.item_ids)))
        dd=sparse.csr_matrix((ones,(rows,inverse)),shape=(n,len(unique)))
        mean=np.bincount(users,weights=days,minlength=len(data.user_ids))/data.history_sizes
        history=data.observed.multiply((1/np.sqrt(data.history_sizes))[:,None]).tocsr()
        return cls(data,users,items,ratings,days,mean,unique,inverse.astype(np.int32),ud,it,dd,history,minimum,maximum)

    def bins(self,days,n_bins):
        raw=np.floor((np.asarray(days)-self.min_day)*n_bins/(self.max_day-self.min_day+1)).astype(np.int64)
        return np.clip(raw,0,n_bins-1)

    def deviation(self,users,days,beta):
        delta=(np.asarray(days,dtype=np.float64)-self.user_mean_day[users])/365.25
        return np.sign(delta)*np.abs(delta)**beta

    def lookup_user_days(self,users,days):
        users=np.asarray(users);days=np.broadcast_to(np.asarray(days,dtype=np.int64),users.shape)
        result=np.full(users.shape,-1,dtype=np.int64)
        # Outside-range days must not alias another user's code after stride wrapping.
        valid=(days>=self.min_day)&(days<=self.max_day)
        codes=users[valid].astype(np.int64)*(self.max_day-self.min_day+1)+(days[valid]-self.min_day)
        position=np.searchsorted(self.user_day_codes,codes)
        match=position<len(self.user_day_codes)
        match[match]=self.user_day_codes[position[match]]==codes[match]
        slots=np.flatnonzero(valid);result[slots[match]]=position[match]
        return result


class TimeSVDpp:
    def __init__(self,n_factors=16,n_bins=12,regularization=.01,learning_rate=.05,n_epochs=50,beta=.4,seed=42,temporal=True):
        for name,value in [('n_factors',n_factors),('n_bins',n_bins),('n_epochs',n_epochs)]:
            if isinstance(value,bool) or not isinstance(value,(int,np.integer)) or value<1:raise ValueError(name)
        if not np.isfinite(regularization) or regularization<0:raise ValueError('regularization')
        if not np.isfinite(learning_rate) or learning_rate<=0:raise ValueError('learning_rate')
        if not np.isfinite(beta) or not 0<beta<=1:raise ValueError('beta')
        self.n_factors=int(n_factors);self.n_bins=int(n_bins);self.n_epochs=int(n_epochs)
        self.regularization=float(regularization);self.learning_rate=float(learning_rate);self.beta=float(beta)
        self.seed=seed;self.temporal=bool(temporal)

    def initialize(self,temporal_data):
        self.temporal_data_=t=temporal_data;self.data_=t.fit
        self.item_ids=t.fit.item_ids.copy();self.fit_fingerprint=t.fit.fingerprint
        n_users=len(t.fit.user_ids);n_items=len(t.fit.item_ids);n_days=len(t.user_day_codes)
        rng=np.random.default_rng(self.seed);f=self.n_factors
        self.mean_=float(t.ratings.mean())
        self.parameters_={'bu':np.zeros(n_users),'bi':np.zeros(n_items),'alpha_bias':np.zeros(n_users),
            'bin_bias':np.zeros((n_items,self.n_bins)),'day_bias':np.zeros(n_days),
            'p':rng.normal(0,.05,size=(n_users,f)),'q':rng.normal(0,.05,size=(n_items,f)),
            'a':np.zeros((n_users,f)),'day_p':np.zeros((n_days,f)),'y':np.zeros((n_items,f))}
        self.training_bins_=t.bins(t.days,self.n_bins)
        self.training_deviation_=t.deviation(t.users,t.days,self.beta)
        n=len(t.users);wu=np.bincount(t.users,minlength=n_users)/n;wi=np.bincount(t.items,minlength=n_items)/n
        wd=np.bincount(t.user_days,minlength=n_days)/n
        wb=np.bincount(t.items*self.n_bins+self.training_bins_,minlength=n_items*self.n_bins).reshape(n_items,self.n_bins)/n
        # Per-observation ridge; implicit Y penalty averages over normalized history.
        # With one fit pair per user/item, sum_u n_u H_uj^2 / N equals item frequency / N.
        self.penalty_weights_={'bu':wu,'bi':wi,'alpha_bias':wu,'bin_bias':wb,'day_bias':wd,
            'p':wu[:,None],'q':wi[:,None],'a':wu[:,None],'day_p':wd[:,None],'y':wi[:,None]}
        return self

    def loss_and_gradients(self,*,gradients=True):
        t=self.temporal_data_;p=self.parameters_;u=t.users;i=t.items;d=t.user_days
        implicit=t.history@p['y']
        effective=p['p'][u]+implicit[u]
        score=self.mean_+p['bu'][u]+p['bi'][i]
        if self.temporal:
            effective+=p['a'][u]*self.training_deviation_[:,None]+p['day_p'][d]
            score+=p['alpha_bias'][u]*self.training_deviation_+p['day_bias'][d]+p['bin_bias'][i,self.training_bins_]
        q=p['q'][i];score+=np.einsum('ij,ij->i',q,effective)
        residual=score-t.ratings;n=len(u)
        mse=float(np.mean(residual*residual))
        penalty=sum(float(np.sum(self.penalty_weights_[key]*value*value)) for key,value in p.items())
        loss=.5*(mse+self.regularization*penalty)
        if not np.isfinite(loss):raise FloatingPointError('Nonfinite timeSVD++ objective')
        if not gradients:return loss,mse
        error=residual/n;eq=error[:,None]*q
        gp=np.asarray(t.user_design.T@eq)
        grad={'bu':np.bincount(u,weights=error,minlength=len(p['bu'])),
            'bi':np.bincount(i,weights=error,minlength=len(p['bi'])),
            'p':gp,'q':np.asarray(t.item_design.T@(error[:,None]*effective)),
            'y':np.asarray(t.history.T@gp)}
        if self.temporal:
            grad.update({'alpha_bias':np.bincount(u,weights=error*self.training_deviation_,minlength=len(p['bu'])),
                'day_bias':np.bincount(d,weights=error,minlength=len(p['day_bias'])),
                'bin_bias':np.bincount(i*self.n_bins+self.training_bins_,weights=error,minlength=p['bin_bias'].size).reshape(p['bin_bias'].shape),
                'a':np.asarray(t.user_design.T@(eq*self.training_deviation_[:,None])),
                'day_p':np.asarray(t.day_design.T@eq)})
        else:
            grad.update({key:np.zeros_like(p[key]) for key in ('alpha_bias','day_bias','bin_bias','a','day_p')})
        for key in p:grad[key]+=self.regularization*self.penalty_weights_[key]*p[key]
        return loss,mse,grad

    def fit(self,data,frame,*,progress_callback=None):
        return self.fit_temporal(TemporalData.from_frame(data,frame),progress_callback=progress_callback)

    def fit_temporal(self,temporal_data,*,progress_callback=None):
        self.initialize(temporal_data)
        accumulators={key:np.zeros_like(value) for key,value in self.parameters_.items()}
        self.curve_=[];step_seconds=0.
        # An epoch is ONE exact full-data gradient step. All parameters stay frozen
        # while computing H@Y and its complete chain-rule gradient H.T@g_p.
        for epoch in range(self.n_epochs+1):
            started=perf_counter()
            loss,mse,grad=self.loss_and_gradients(gradients=True)
            row={'epoch':epoch,'training_objective':loss,'training_rmse':float(np.sqrt(mse)),
                 'previous_step_seconds':step_seconds}
            self.curve_.append(row)
            if progress_callback is not None:progress_callback(row)
            if epoch==self.n_epochs:break
            for key,value in self.parameters_.items():
                accumulators[key]+=grad[key]*grad[key]
                value-=self.learning_rate*grad[key]/(np.sqrt(accumulators[key])+1e-8)
            step_seconds=perf_counter()-started
        self.prepare_scoring()
        return self

    def prepare_scoring(self):
        self.implicit_vectors_=self.temporal_data_.history@self.parameters_['y']

    def components(self,users,days):
        t=self.temporal_data_;p=self.parameters_;u=np.asarray(users,dtype=np.int32)
        days=np.broadcast_to(np.asarray(days,dtype=np.int64),u.shape)
        vector=p['p'][u]+self.implicit_vectors_[u]
        bias=p['bu'][u].copy();bins=t.bins(days,self.n_bins)
        if self.temporal:
            dev=t.deviation(u,days,self.beta);indices=t.lookup_user_days(u,days);known=indices>=0
            vector+=p['a'][u]*dev[:,None];bias+=p['alpha_bias'][u]*dev
            vector[known]+=p['day_p'][indices[known]];bias[known]+=p['day_bias'][indices[known]]
        return vector,bias,bins

    def predict_pairs(self,users,items,timestamps):
        u=np.asarray(users,dtype=np.int32);i=np.asarray(items,dtype=np.int32)
        vector,bias,bins=self.components(u,np.asarray(timestamps,dtype=np.int64)//86400)
        p=self.parameters_
        score=self.mean_+bias+p['bi'][i]+np.einsum('ij,ij->i',p['q'][i],vector)
        if self.temporal:score+=p['bin_bias'][i,bins]
        return score

    def score_batch(self,user_ids,*,query_timestamp=None):
        if not hasattr(self,'implicit_vectors_'):raise RuntimeError('Fit timeSVD++ before scoring')
        if query_timestamp is None or query_timestamp<=self.data_.max_timestamp:
            raise ValueError('Provide a query timestamp strictly after all fit observations')
        try:users=np.asarray([self.data_.user_index[u] for u in user_ids],dtype=np.int32)
        except KeyError as error:raise ValueError('Unknown user outside the warm-start cohort') from error
        vector,bias,bins=self.components(users,int(query_timestamp)//86400)
        p=self.parameters_
        score=np.dot(vector,p['q'].T)+self.mean_+bias[:,None]+p['bi'][None,:]
        if self.temporal:score+=p['bin_bias'][:,int(bins[0])][None,:] if len(bins) else 0.
        return score
