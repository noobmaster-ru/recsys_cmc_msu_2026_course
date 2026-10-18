"""Preregistered extended FM search. Strict validation only; preserves frozen v2."""
from pathlib import Path
import json
import sys
from time import perf_counter
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from models.fm import FM
from models.fm_training import TunableFM
from scripts.tune_fm import prepare,save_model
from scripts.tune_ials import load_result,write_json
from scripts.prepare_data import sha256,deterministic_gzip
from evaluation import evaluate_model,history_slices,popularity_slices,paired_bootstrap


class ValidationOnlyGuard:
    def __init__(self,root):
        self.root=Path(root).resolve();self.active=True;self.blocked=[];self.opened=set()
        self.paths=[self.root/p for p in ['data/splits/test.csv.gz','data/processed/interactions.csv.gz',
            'data/processed/items.jsonl.gz','data/raw','data/final_test','reports/final_test','configs/final_models.json','archive']]
    def __call__(self,event,args):
        if not self.active or event!='open' or not isinstance(args[0],(str,bytes)):return
        raw=args[0].decode() if isinstance(args[0],bytes) else args[0];path=Path(raw).resolve()
        if any(path==p or p in path.parents for p in self.paths):
            self.blocked.append(str(path));raise RuntimeError('Validation-only FM search cannot read test/final/snapshot data')
        if self.root in path.parents:self.opened.add(str(path.relative_to(self.root)))


def key(row):return (-row['ndcg_at_10'],row['epoch'],row['trial'])


def restore_v2(data,features,root,summary):
    model=FM(**summary['best_params']);model.data_=data;model.features_=features
    model.item_ids=data.item_ids.copy();model.fit_fingerprint=data.fingerprint
    with np.load(root/'data/validation/fm/best_parameters.npz',allow_pickle=False) as z:
        model.bias_=float(z['bias']);model.linear_=z['linear'];model.factors_=z['factors']
    model.prepare_scoring();return model


def run(root=ROOT):
    root=Path(root).resolve();guard=ValidationOnlyGuard(root);sys.addaudithook(guard)
    output=root/'reports/fm_extended_validation';local=root/'data/validation/fm_extended'
    try:
        config=json.loads((root/'configs/fm_extended_search.json').read_text())
        if (output/'search_plan.json').exists():raise RuntimeError('Search already exists; preserve it rather than silently overwrite/restart')
        output.mkdir(parents=True,exist_ok=True);local.mkdir(parents=True,exist_ok=True)
        protocol,_,data,context,baselines,features,training,frame,previous=prepare(root)
        old=json.loads((root/'reports/fm_validation/summary.json').read_text())
        for group in ['source_hashes','artifact_hashes']:
            for relative,digest in old[group].items():
                if sha256(root/relative)!=digest:raise ValueError('Changed v2 training evidence: '+relative)
        baselines['FM v2']=load_result(root/'data/validation/fm','best',old['best_validation_metrics'],context)
        time=json.loads((root/'reports/timesvdpp_validation/summary.json').read_text())
        for relative,digest in time['artifact_hashes'].items():
            if relative.startswith('data/') and any(k in relative for k in ['per_user','recommendation_indices']):
                if sha256(root/relative)!=digest:raise ValueError('Changed time validation ranking')
        baselines['timeSVD++']=load_result(root/'data/validation/timesvdpp','best',time['best_validation_metrics'],context)
        source_paths=['models/fm.py','models/fm_training.py','scripts/tune_fm_extended.py','scripts/tune_fm.py',
            'scripts/tune_ease.py','scripts/tune_ials.py','data_utils.py','evaluation.py','configs/fm_extended_search.json','requirements.txt']
        plan={'configuration':config,'source_hashes':{p:sha256(root/p) for p in source_paths},
            'input_hashes':previous['input_hashes'],'protocol_sha256':previous['protocol_sha256'],
            'cohort_fingerprint':context.fingerprint,'cohort':context.diagnostics,'query_timestamp':context.query_timestamp,
            'feature_fingerprint':features.fingerprint,'feature_diagnostics':features.diagnostics,
            'prior_v2_validation_summary_sha256':sha256(root/'reports/fm_validation/summary.json'),
            'training_examples_v2':training[3],'test_evaluated':False,'train_validation_refit':False,
            'training_data':'train only','guard':'raw test, final metrics/weights, snapshot metadata, full history and archives blocked'}
        write_json(output/'search_plan.json',plan)
        v2=restore_v2(data,features,root,old)
        selected_users=context.user_ids[:128];scores=v2.score_batch(selected_users,query_timestamp=context.query_timestamp)
        interaction=scores-v2.item_offsets_[None,:]-v2.linear_[[data.user_index[u]+1 for u in selected_users],None]-v2.bias_
        audit={'v2_probe_bce_initial':float(pd.read_csv(root/'reports/fm_validation/training_curve.csv').query('trial == 1').iloc[0].probe_bce),
            'fixed_sampling_examples':training[3],
            'v2_item_offset_std':float(v2.item_offsets_.std()),'v2_median_personal_interaction_std':float(np.median(interaction.std(axis=1))),
            'notes':['Full FM/BCE gradients previously pass finite differences; v3 repeats these oracles plus BPR.',
                'Fixed uniform samples are legitimate but exposure repeats every epoch; resampling is compared, not assumed beneficial.',
                'Popularity proposal biases the training distribution and is not interpreted as a calibrated or unbiased estimator.']}
        write_json(output/'training_audit.json',audit)
        records=[];curves=[];sampling=[];best=None;best_result=None;total_fit=0.
        trial_specs={spec['trial']:spec for spec in config['trials']}
        def execute(spec,checkpoints,stage):
            nonlocal best,best_result,total_fit
            name=spec['trial'];params={k:v for k,v in spec.items() if k!='trial'}
            model=TunableFM(**params).initialize_training(data,features,frame)
            directory=local/name;directory.mkdir(exist_ok=True)
            if stage==2:model.load_state(directory/'last_state.npz')
            def progress(row):
                curves.append({'trial':name,'stage':stage,'loss':spec['loss'],**row})
                pd.DataFrame(curves).to_csv(output/'training_curves.csv',index=False)
            for checkpoint in checkpoints:
                started=perf_counter();model.train_until(checkpoint,progress_callback=progress);seconds=perf_counter()-started;total_fit+=seconds
                result=evaluate_model(model,context,batch_size=128)
                row={'trial':name,'stage':stage,'epoch':checkpoint,**params,**result.metrics,'additional_fit_seconds':seconds}
                records.append(row);pd.DataFrame(records).to_csv(output/'search_results.csv',index=False)
                if best is None or key(row)<key(best):
                    best=row.copy();best_result=result;save_model(local,model,features)
                    model.save_state(local/'best_state.npz')
                print(f'{stage=} {name} epoch={checkpoint}: NDCG={result.metrics["ndcg_at_10"]:.6f}, Recall={result.metrics["recall_at_10"]:.6f}; best={best["ndcg_at_10"]:.6f}',flush=True)
            sampling.extend({'trial':name,'stage':stage,**r} for r in model.sampling_history_)
            pd.DataFrame(sampling).to_csv(output/'sampling_history.csv',index=False)
            model.save_state(directory/'last_state.npz')
            if name=='00_control_v2' and stage==1:
                np.testing.assert_allclose(model.factors_,v2.factors_,rtol=0,atol=0)
                np.testing.assert_allclose(model.linear_,v2.linear_,rtol=0,atol=0)
                if abs(records[-1]['ndcg_at_10']-old['best_validation_metrics']['ndcg_at_10'])>1e-14:raise ValueError('Control failed to reproduce v2')
                audit['control_weights_identical_to_v2']=True;audit['control_epoch12_metrics']=result.metrics
                write_json(output/'training_audit.json',audit)
            del model
        for spec in config['trials']:execute(spec,config['stage1_checkpoints'],1)
        trial_best=[min([r for r in records if r['trial']==name],key=key) for name in trial_specs]
        promoted=[r['trial'] for r in sorted(trial_best,key=key)[:config['promote_trials']]]
        write_json(output/'promotion.json',{'promoted':promoted,'criterion':config['promotion'],'stage1_results_sha256':sha256(output/'search_results.csv')})
        for name in promoted:execute(trial_specs[name],config['stage2_checkpoints'],2)
        results={**baselines,'FM extended':best_result}
        pd.DataFrame([{'model':name,**result.metrics} for name,result in results.items()]).to_csv(output/'comparison.csv',index=False)
        pd.concat({name:history_slices(result) for name,result in results.items()},names=['model']).to_csv(output/'history_slices.csv')
        pd.concat({name:popularity_slices(result,context) for name,result in results.items()},names=['model']).to_csv(output/'popularity_slices.csv')
        boot=protocol['confidence_intervals'];intervals=[]
        for name in ['FM v2','Popularity','EASE']:
            table=paired_bootstrap(best_result,baselines[name],resamples=boot['resamples'],seed=boot['seed'],confidence=boot['confidence'],result_name='FM extended',baseline_name=name)
            table.insert(0,'baseline',name);intervals.append(table)
        intervals=pd.concat(intervals,ignore_index=True);intervals.to_csv(output/'bootstrap.csv',index=False)
        with deterministic_gzip(local/'best_per_user.csv.gz') as stream:best_result.per_user.to_csv(stream,lineterminator='\n')
        np.save(local/'best_recommendation_indices.npy',best_result.recommendations,allow_pickle=False)
        summary={**plan,'completed_configurations':len(config['trials']),'validation_evaluations':len(records),
            'promoted':promoted,'best_checkpoint':best,'best_validation_metrics':best_result.metrics,
            'training_seconds':total_fit,'training_audit':audit,'test_evaluated':False,'train_validation_refit':False,
            'io_audit':{'blocked_attempts':guard.blocked,'paths_opened':sorted(guard.opened)},
            'interval_interpretation':'descriptive validation after expanded HPO and checkpoint selection; not an independent test',
            'artifact_hashes':{str(p.relative_to(root)):sha256(p) for p in [*output.glob('*.csv'),output/'promotion.json',output/'training_audit.json',*local.glob('*')] if p.is_file()}}
        write_json(output/'summary.json',summary)
        print('Completed validation-only extended FM search:',best,flush=True)
        return summary
    finally:guard.active=False


if __name__=='__main__':run()
