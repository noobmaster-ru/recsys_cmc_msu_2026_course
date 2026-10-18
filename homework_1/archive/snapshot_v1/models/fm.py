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
    def from_items(cls, data, records, *, minimum_items=5, use_metadata=True):
        if minimum_items<1:raise ValueError('minimum_items must be positive')
        selected=[records.get(item,{}) for item in data.item_ids]
        brands=[token(r.get('brand','')) for r in selected]
        categories=[sorted({token(c) for c in r.get('categories',[]) if token(c) and token(c)!='video games'}) for r in selected]
        bc=Counter(b for b in brands if b);cc=Counter(c for cs in categories for c in cs)
        names=['PAD']+['user:'+u for u in data.user_ids]+['item:'+i for i in data.item_ids]
        brand_vocab=sorted(b for b,count in bc.items() if count>=minimum_items) if use_metadata else []
        category_vocab=sorted(c for c,count in cc.items() if count>=minimum_items) if use_metadata else []
        if use_metadata:names+=['brand:UNK']+['brand:'+b for b in brand_vocab]+['category:UNK']+['category:'+c for c in category_vocab]
        mapping={name:i for i,name in enumerate(names)}
        rows=[];values=[]
        for item,brand,cats in zip(data.item_ids,brands,categories):
            row=[mapping['item:'+item]];value=[1.]
            if use_metadata:
                row.append(mapping.get('brand:'+brand,mapping['brand:UNK']));value.append(1.)
                valid=[mapping['category:'+c] for c in cats if 'category:'+c in mapping]
                if not valid:valid=[mapping['category:UNK']]
                row+=valid;value+=[1/np.sqrt(len(valid))]*len(valid)
            rows.append(row);values.append(value)
        width=max(map(len,rows));indices=np.zeros((len(rows),width),dtype=np.int32);x=np.zeros(indices.shape)
        for i,(row,value) in enumerate(zip(rows,values)):
            indices[i,:len(row)]=row;x[i,:len(row)]=value
        h=hashlib.sha256(data.fingerprint.encode());h.update(json.dumps(names,ensure_ascii=False).encode())
        h.update(indices.tobytes());h.update(x.tobytes())
        diagnostics={'features':len(names),'user_id_features':len(data.user_ids),'item_id_features':len(data.item_ids),
            'brand_vocabulary':len(brand_vocab),'category_vocabulary':len(category_vocab),
            'minimum_catalog_items_per_token':minimum_items,'uses_metadata':bool(use_metadata),
            'fit_items_with_brand':sum(bool(b) for b in brands),'fit_items_with_category':sum(bool(c) for c in categories),
            'max_active_item_features':width,'category_l2_normalization':True,
            'metadata_assumption':'crawl-time static snapshot; historical availability unverified'}
        return cls(indices,x,names,len(data.user_ids),data.fingerprint,h.hexdigest(),diagnostics)

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
        try:rows=np.asarray([self.data_.user_index[u]+1 for u in user_ids],dtype=np.int32)
        except KeyError as error:raise ValueError('Unknown user outside the warm-start cohort') from error
        # Exact decomposition of the complete FM formula, including item-metadata pairs.
        return np.dot(self.factors_[rows],self.item_vectors_.T)+self.item_offsets_[None,:]+self.linear_[rows,None]+self.bias_
