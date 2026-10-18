"""Second-order FM and sparse mini-batch AdaGrad implemented in NumPy.

Objective: mean BCE + reg/2 * mean_rows sum_active x_j^2*(w_j^2+||v_j||^2).
Regularization is feature-frequency weighted, not global uniform ridge.
"""
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from time import perf_counter

import numpy as np

from data_utils import FitData


def token(value):
    return ' '.join(str(value or '').casefold().split())


@dataclass
class MetadataFeatures:
    item_indices: np.ndarray
    item_values: np.ndarray
    feature_names: list
    n_users: int
    fit_fingerprint: str
    fingerprint: str
    diagnostics: dict

    @classmethod
    def from_history(cls, data, frame, query_timestamp, *, minimum_items=5, use_metadata=True):
        """Item metadata derived exclusively from dated fit observations.

        One representation is frozen at the recommendation query and is also
        used to reconstruct fit labels. This is not event-time backtesting of
        individual training rows. No rating value enters the side features.
        """
        import pandas as pd
        if minimum_items < 1:
            raise ValueError('minimum_items must be positive')
        if FitData.from_frame(frame).fingerprint != data.fingerprint:
            raise ValueError('Metadata frame/index mismatch')
        if not isinstance(query_timestamp, (int, np.integer)) or query_timestamp <= data.max_timestamp:
            raise ValueError('Metadata query must strictly follow every fit event')
        times = frame.groupby('item_id').timestamp.agg(['min','max','count']).reindex(data.item_ids)
        if times.isna().any().any():
            raise ValueError('Missing fit item timestamp history')
        first = pd.to_datetime(times['min'],unit='s',utc=True)
        years = first.dt.year.astype(str).tolist()
        months = first.dt.month.map(lambda x:f'{x:02}').tolist()
        counts = [Counter(years), Counter(months)]
        vocabs = [sorted(k for k,v in c.items() if v >= minimum_items) for c in counts]
        names = ['PAD']+['user:'+u for u in data.user_ids]+['item:'+i for i in data.item_ids]
        numeric_names = ['history_age_log_days','recency_log_days','history_count_log']
        if use_metadata:
            for prefix,vocab in zip(['first_year','first_month'],vocabs):
                names += [prefix+':UNK']+[prefix+':'+k for k in vocab]
            names += numeric_names
        mapping = {name:i for i,name in enumerate(names)}
        age = (query_timestamp-times['min'].to_numpy())/86400.
        recency = (query_timestamp-times['max'].to_numpy())/86400.
        numeric = np.column_stack((np.log1p(age)/np.log1p(3650.),
                                   np.log1p(recency)/np.log1p(3650.),
                                   np.log1p(times['count'].to_numpy())/np.log1p(1000.)))
        width = 6 if use_metadata else 1
        indices = np.zeros((len(data.item_ids),width),dtype=np.int32)
        values = np.ones(indices.shape,dtype=np.float64)
        for row,(item,year,month) in enumerate(zip(data.item_ids,years,months)):
            indices[row,0] = mapping['item:'+item]
            if use_metadata:
                indices[row,1] = mapping.get('first_year:'+year,mapping['first_year:UNK'])
                indices[row,2] = mapping.get('first_month:'+month,mapping['first_month:UNK'])
                indices[row,3:] = [mapping[name] for name in numeric_names]
                values[row,3:] = numeric[row]
        digest = hashlib.sha256(data.fingerprint.encode())
        digest.update(json.dumps(names,ensure_ascii=False).encode())
        digest.update(indices.tobytes());digest.update(values.tobytes())
        diagnostics = {'features':len(names),'user_id_features':len(data.user_ids),
            'item_id_features':len(data.item_ids),'uses_metadata':bool(use_metadata),
            'first_year_vocabulary':len(vocabs[0]) if use_metadata else 0,
            'first_month_vocabulary':len(vocabs[1]) if use_metadata else 0,
            'numeric_metadata_features':len(numeric_names) if use_metadata else 0,
            'minimum_catalog_items_per_token':minimum_items,'max_active_item_features':width,
            'query_timestamp':int(query_timestamp),'max_source_timestamp':int(frame.timestamp.max()),
            'all_source_timestamps_before_query':bool((frame.timestamp < query_timestamp).all()),
            'snapshot_fields_used':False,'rating_values_used_in_metadata':False,
            'metadata_source':'fit item identifiers and review timestamps only',
            'representation':'fixed as of stage query for fit reconstruction and recommendation',
            'numeric_scales':{'log_days':3650,'log_count':1000},
            'history_age_days_min':float(age.min()),'history_age_days_max':float(age.max()),
            'metadata_assumption':'timestamp is event availability; no historical catalog snapshot required'}
        return cls(indices,values,names,len(data.user_ids),data.fingerprint,digest.hexdigest(),diagnostics)

    def batch(self,users,items):
        indices=np.column_stack((np.asarray(users,dtype=np.int32)+1,self.item_indices[items]))
        values=np.column_stack((np.ones(len(users)),self.item_values[items]))
        return indices,values


def sample_training(data,frame,*,negatives_per_positive=2,seed=137):
    if not isinstance(data,FitData) or FitData.from_frame(frame).fingerprint!=data.fingerprint:
        raise ValueError('Training frame/index mismatch')
    if isinstance(negatives_per_positive,bool) or not isinstance(negatives_per_positive,int) or negatives_per_positive<1:
        raise ValueError('Require a positive number of negatives')
    user=frame.user_id.map(data.user_index).to_numpy(dtype=np.int32)
    item=frame.item_id.map(data.item_index).to_numpy(dtype=np.int32)
    labels=(frame.rating.to_numpy()>=data.relevance_threshold).astype(np.float64)
    n_items=len(data.item_ids)
    sampled_users=np.repeat(user[labels==1],negatives_per_positive)
    if np.any(data.history_sizes[np.unique(sampled_users)]>=n_items):
        raise ValueError('A positive user has no unobserved candidate for negative sampling')
    observed=np.sort(user.astype(np.int64)*n_items+item)
    rng=np.random.default_rng(seed);sampled_items=rng.integers(n_items,size=len(sampled_users),dtype=np.int32)
    while len(sampled_items):
        codes=sampled_users.astype(np.int64)*n_items+sampled_items
        positions=np.searchsorted(observed,codes);inside=positions<len(observed)
        collision=np.zeros(len(codes),dtype=bool);collision[inside]=observed[positions[inside]]==codes[inside]
        if not collision.any():break
        sampled_items[collision]=rng.integers(n_items,size=int(collision.sum()),dtype=np.int32)
    users=np.r_[user,sampled_users];items=np.r_[item,sampled_items];y=np.r_[labels,np.zeros(len(sampled_users))]
    h=hashlib.sha256()
    for a in (users,items,y):h.update(a.tobytes())
    diagnostics={'examples':len(y),'observed_positive':int(labels.sum()),'observed_negative':int((labels==0).sum()),
        'sampled_unobserved':len(sampled_users),'negative_sampling_seed':seed,'negatives_per_positive':negatives_per_positive,
        'fingerprint':h.hexdigest(),'sampling':'uniform with replacement, excluding ALL observed train pairs only; fixed across epochs/trials'}
    return users,items,y,diagnostics


def fm_predict(indices,x,bias,linear,factors):
    vectors=factors[indices]*x[:,:,None]
    summed=vectors.sum(axis=1)
    return (bias+(linear[indices]*x).sum(axis=1)
            +.5*(np.sum(summed*summed,axis=1)-np.sum(vectors*vectors,axis=(1,2))))


def fm_loss_gradient(indices,x,y,bias,linear,factors,regularization):
    """Exact batch objective/gradient, summing repeated feature IDs before updates."""
    n=len(y);v=factors[indices];weighted=v*x[:,:,None];summed=weighted.sum(axis=1)
    score=bias+(linear[indices]*x).sum(axis=1)+.5*(np.sum(summed**2,axis=1)-np.sum(weighted**2,axis=(1,2)))
    loss=float(np.mean(np.logaddexp(0.,score)-y*score))
    penalty=np.sum((linear[indices]**2+np.sum(v*v,axis=2))*x*x)/n
    loss+=.5*regularization*penalty
    dz=(np.exp(-np.logaddexp(0.,-score))-y)/n
    gw=dz[:,None]*x+regularization/n*x*x*linear[indices]
    gv=dz[:,None,None]*x[:,:,None]*(summed[:,None,:]-weighted)+regularization/n*x[:,:,None]**2*v
    unique,inverse=np.unique(indices.ravel(),return_inverse=True)
    grad_w=np.zeros(len(unique));grad_v=np.zeros((len(unique),factors.shape[1]))
    np.add.at(grad_w,inverse,gw.ravel());np.add.at(grad_v,inverse,gv.reshape(-1,factors.shape[1]))
    return loss,unique,grad_w,grad_v,float(dz.sum())


class FM:
    def __init__(self,n_factors=16,learning_rate=.05,regularization=.001,n_epochs=12,batch_size=1024,seed=42,shuffle_seed=43):
        for name,value in [('n_factors',n_factors),('n_epochs',n_epochs),('batch_size',batch_size)]:
            if isinstance(value,bool) or not isinstance(value,(int,np.integer)) or value<1:raise ValueError(name)
        if not np.isfinite(learning_rate) or learning_rate<=0:raise ValueError('learning_rate')
        if not np.isfinite(regularization) or regularization<0:raise ValueError('regularization')
        self.n_factors=int(n_factors);self.learning_rate=float(learning_rate);self.regularization=float(regularization)
        self.n_epochs=int(n_epochs);self.batch_size=int(batch_size);self.seed=seed;self.shuffle_seed=shuffle_seed

    def fit(self,data,features,training,*,progress_callback=None):
        if data.fingerprint!=features.fit_fingerprint:raise ValueError('Features belong to a different fit')
        self.data_=data;self.features_=features;self.item_ids=data.item_ids.copy();self.fit_fingerprint=data.fingerprint
        users,items,y=training[:3]
        if not len(y) or not np.isin(y,[0.,1.]).all() or not 0<y.mean()<1:raise ValueError('Require binary labels of both classes')
        rng=np.random.default_rng(self.seed);shuffle=np.random.default_rng(self.shuffle_seed)
        count=len(features.feature_names);self.linear_=np.zeros(count)
        self.factors_=rng.normal(0,.01,size=(count,self.n_factors));self.factors_[0]=0
        self.bias_=float(np.log(y.mean()/(1-y.mean())))
        acc_w=np.zeros(count);acc_v=np.zeros_like(self.factors_);acc_b=0.
        probe=np.random.default_rng(991).choice(len(y),size=min(8192,len(y)),replace=False)
        pi,px=features.batch(users[probe],items[probe])
        self.curve_=[]
        def record(epoch,seconds,online):
            score=fm_predict(pi,px,self.bias_,self.linear_,self.factors_)
            bce=float(np.mean(np.logaddexp(0.,score)-y[probe]*score))
            vv=self.factors_[pi];penalty=np.mean(np.sum(px*px*(self.linear_[pi]**2+np.sum(vv*vv,axis=2)),axis=1))
            objective=bce+.5*self.regularization*penalty
            if not np.isfinite(objective):raise FloatingPointError('FM diverged')
            row={'epoch':epoch,'probe_bce':bce,'probe_objective':objective,'online_objective':online,'seconds':seconds}
            self.curve_.append(row)
            if progress_callback is not None:progress_callback(row)
        record(0,0.,None)
        for epoch in range(1,self.n_epochs+1):
            started=perf_counter();order=shuffle.permutation(len(y));online=0.
            for start in range(0,len(y),self.batch_size):
                batch=order[start:start+self.batch_size];ix,x=features.batch(users[batch],items[batch])
                loss,unique,gw,gv,gb=fm_loss_gradient(ix,x,y[batch],self.bias_,self.linear_,self.factors_,self.regularization)
                online+=len(batch)*loss
                active=unique!=0;unique=unique[active];gw=gw[active];gv=gv[active]
                acc_w[unique]+=gw*gw;acc_v[unique]+=gv*gv;acc_b+=gb*gb
                self.linear_[unique]-=self.learning_rate*gw/(np.sqrt(acc_w[unique])+1e-8)
                self.factors_[unique]-=self.learning_rate*gv/(np.sqrt(acc_v[unique])+1e-8)
                self.bias_-=self.learning_rate*gb/(np.sqrt(acc_b)+1e-8)
            record(epoch,perf_counter()-started,online/len(y))
        if not np.isfinite(self.factors_).all() or not np.isfinite(self.linear_).all():raise FloatingPointError('Nonfinite FM parameters')
        self.prepare_scoring()
        return self

    def prepare_scoring(self):
        ix=self.features_.item_indices;x=self.features_.item_values
        weighted=self.factors_[ix]*x[:,:,None]
        self.item_vectors_=weighted.sum(axis=1)
        self.item_offsets_=(self.linear_[ix]*x).sum(axis=1)+.5*(np.sum(self.item_vectors_**2,axis=1)-np.sum(weighted**2,axis=(1,2)))

    def score_batch(self,user_ids,*,query_timestamp=None):
        if not hasattr(self,'item_vectors_'):raise RuntimeError('Fit FM before scoring')
        if query_timestamp is not None and query_timestamp<=self.data_.max_timestamp:raise ValueError('Query must follow fit')
        if query_timestamp is not None and query_timestamp<self.features_.diagnostics['query_timestamp']:
            raise ValueError('Metadata not available at the requested query time')
        try:rows=np.asarray([self.data_.user_index[u]+1 for u in user_ids],dtype=np.int32)
        except KeyError as error:raise ValueError('Unknown user outside the warm-start cohort') from error
        # Exact decomposition of the complete FM formula, including item-metadata pairs.
        return np.dot(self.factors_[rows],self.item_vectors_.T)+self.item_offsets_[None,:]+self.linear_[rows,None]+self.bias_
