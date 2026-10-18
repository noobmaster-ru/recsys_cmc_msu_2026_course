from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.tune_fm_extended import ValidationOnlyGuard
from scripts.tune_fm import prepare
from models.fm_training import TunableFM
from models.fm import fm_predict
from evaluation import evaluate_model,paired_bootstrap
from scripts.prepare_data import sha256
root=ROOT;guard=ValidationOnlyGuard(root);sys.addaudithook(guard)
out=root/'reports/fm_extended_validation';s=json.loads((out/'summary.json').read_text())
assert s['test_evaluated'] is False and s['train_validation_refit'] is False
assert not s['io_audit']['blocked_attempts']
for path,digest in s['source_hashes'].items():assert sha256(root/path)==digest,path
for path,digest in s['artifact_hashes'].items():assert sha256(root/path)==digest,path
protocol,_,data,context,baselines,features,training,frame,previous=prepare(root)
assert s['feature_fingerprint']==features.fingerprint
assert features.diagnostics['max_source_timestamp']<context.query_timestamp
assert s['cohort_fingerprint']==context.fingerprint
cfg=s['configuration'];winner=s['best_checkpoint'];spec=next(t for t in cfg['trials'] if t['trial']==winner['trial'])
params={k:v for k,v in spec.items() if k!='trial'}
model=TunableFM(**params).initialize_training(data,features,frame).load_state(root/'data/validation/fm_extended/best_state.npz')
assert model.epoch_==winner['epoch']
result=evaluate_model(model,context,batch_size=128)
saved=np.load(root/'data/validation/fm_extended/best_recommendation_indices.npy',allow_pickle=False)
np.testing.assert_array_equal(saved,result.recommendations)
computed=[]
for row,chosen,relevant in zip(context.user_indices,saved,context.relevance):
 assert len(chosen)==10 and len(set(chosen))==10
 assert set(chosen).isdisjoint(data.seen_items(row))
 hits=np.array([i in relevant for i in chosen],dtype=float)
 ideal=(1/np.log2(np.arange(2,min(10,len(relevant))+2))).sum()
 computed.append(((hits/np.log2(np.arange(2,12))).sum()/ideal,hits.sum()/len(relevant)))
computed=np.array(computed)
for pos,metric in enumerate(['ndcg_at_10','recall_at_10']):
 assert abs(computed[:,pos].mean()-s['best_validation_metrics'][metric])<1e-14
 if abs(result.metrics[metric]-s['best_validation_metrics'][metric])>1e-14:raise AssertionError(metric)
positive_users=[row for row in context.user_indices if data.positive.indptr[row+1]>data.positive.indptr[row]][:3]
for user in positive_users:
 ix,x=features.batch(np.repeat(user,len(data.item_ids)),np.arange(len(data.item_ids)))
 exact=fm_predict(ix,x,model.bias_,model.linear_,model.factors_)
 fast=model.score_batch([data.user_ids[user]],query_timestamp=context.query_timestamp)[0]
 np.testing.assert_allclose(exact,fast,rtol=1e-11,atol=1e-11)
 seen=set(data.seen_items(user));candidates=np.array([i for i in range(len(data.item_ids)) if i not in seen])
 ordered=candidates[np.lexsort((candidates,-exact[candidates]))][:10]
 np.testing.assert_array_equal(ordered,saved[np.flatnonzero(context.user_indices==user)[0]])
search=pd.read_csv(out/'search_results.csv');curves=pd.read_csv(out/'training_curves.csv');samples=pd.read_csv(out/'sampling_history.csv')
assert len(search)==cfg['maximum_validation_evaluations']==60
assert len(curves)==len(samples)==cfg['maximum_trained_epochs']==384
assert (search.users==10207).all()
ordered=search.sort_values(['ndcg_at_10','epoch','trial'],ascending=[False,True,True])
assert ordered.iloc[0].trial==winner['trial'] and int(ordered.iloc[0].epoch)==winner['epoch']
assert abs(float(ordered.iloc[0].ndcg_at_10)-winner['ndcg_at_10'])<1e-14
stage1=search[search.stage==1].sort_values(['ndcg_at_10','epoch','trial'],ascending=[False,True,True])
expected_promoted=stage1.groupby('trial',sort=False).head(1).head(4).trial.tolist()
assert expected_promoted==s['promoted']
assert set(search[search.stage==2].trial)==set(expected_promoted)
for name,group in search.groupby('trial'):
 assert sorted(group[group.stage==1].epoch.tolist())==[4,8,12]
 if name in expected_promoted:assert sorted(group[group.stage==2].epoch.tolist())==[20,36,60]

for trial,spec in [(t['trial'],t) for t in cfg['trials']]:
 records=samples[samples.trial==trial].sort_values('epoch')
 assert records.epoch.tolist()==list(range(1,len(records)+1))
 expected=np.arange(137,137+len(records)) if spec['resample'] else np.repeat(137,len(records))
 np.testing.assert_array_equal(records.negative_seed.to_numpy(),expected)
 assert (records.sampled_draws==259086*spec['negatives_per_positive']).all()
 if not spec['resample']:assert records.negative_items_sha256.nunique()==1
bpr=search[search.loss=='bpr'];expected_fallback=float(np.mean(np.diff(data.positive.indptr)[context.user_indices]==0))
np.testing.assert_allclose(bpr.fallback_rate,expected_fallback,rtol=0,atol=1e-15)
# Post-selection diagnostics, not extra candidates or a selection rule.
control_results={}
for name in ['00_control_v2','01_resample_only','02_popularity_only']:
 spec=next(t for t in cfg['trials'] if t['trial']==name)
 candidate=TunableFM(**{k:v for k,v in spec.items() if k!='trial'}).initialize_training(data,features,frame)
 candidate.load_state(root/'data/validation/fm_extended'/name/'last_state.npz')
 assert candidate.epoch_==12
 control_results[name]=evaluate_model(candidate,context,batch_size=128)
control_intervals=[];boot=protocol['confidence_intervals']
for left,right in [('01_resample_only','00_control_v2'),('02_popularity_only','01_resample_only')]:
 table=paired_bootstrap(control_results[left],control_results[right],resamples=boot['resamples'],seed=boot['seed'],confidence=boot['confidence'],result_name=left,baseline_name=right)
 table.insert(0,'comparison',left+' - '+right);control_intervals.append(table)
pd.concat(control_intervals,ignore_index=True).to_csv(out/'sampling_control_bootstrap.csv',index=False)
unit_audit=json.loads((out/'unit_test_audit.json').read_text())
assert not guard.blocked
verification={'verified_summary_sha256':sha256(out/'summary.json'),'actual_forbidden_reads':guard.blocked,
 'cohort_users':len(saved),'independently_verified_rankings':len(saved),'exact_full_fm_full_catalog_oracles':len(positive_users),
 'reloaded_checkpoint_reproduces_all_rankings':True,'validation_evaluations':len(search),'training_epoch_rows':len(curves),
 'historical_feature_availability_preserved':True,'all_bpr_zero_positive_users_use_common_fallback':True,
 'non_final_unit_tests_passed':unit_audit['tests_passed'],
 'post_selection_validation_reevaluations':4,'sampling_control_bootstrap_sha256':sha256(out/'sampling_control_bootstrap.csv'),
 'verification_source_sha256':sha256(Path(__file__)), 'unit_test_audit_sha256':sha256(out/'unit_test_audit.json'),'test_evaluated':False,'train_validation_refit':False}
(out/'verification.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2)+'\n')
print('Independent validation checks passed:',verification)
