"""Fixed FM metadata search on train/validation; frozen CF, iALS and EASE baselines."""
import argparse
import gzip
from itertools import product
import json
from pathlib import Path
import sys
from time import perf_counter
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from models.fm import FM,MetadataFeatures,sample_training
from evaluation import evaluate_model,history_slices,popularity_slices,paired_bootstrap
from scripts.prepare_data import sha256,deterministic_gzip
from scripts.tune_ials import load_result,write_json
from scripts.tune_ease import prepare as prepare_previous


def prepare(root):
    protocol,_,data,context,results,previous_plan=prepare_previous(root)
    config=json.loads((root/'configs/fm_search.json').read_text())
    path=root/'reports/ease_validation/summary.json';ease=json.loads(path.read_text())
    if ease['cohort_fingerprint']!=context.fingerprint or ease['input_hashes']!=previous_plan['input_hashes']:
        raise ValueError('EASE cohort or inputs differ')
    for group in ('source_hashes','artifact_hashes'):
        for p,digest in ease[group].items():
            if sha256(root/p)!=digest:raise ValueError(f'Changed EASE artifact: {p}')
    results['EASE']=load_result(root/'data/validation/ease','best',ease['best_validation_metrics'],context)
    meta_path=root/'data/processed/items.jsonl.gz';meta_hash=sha256(meta_path)
    if meta_hash!=protocol['input_checksums']['data/processed/items.jsonl.gz']['sha256']:
        raise ValueError('Metadata source checksum differs')
    fit_catalog=set(data.item_ids);records={}
    with gzip.open(meta_path,'rt',encoding='utf-8') as stream:
        for line in stream:
            record=json.loads(line)
            if record['item_id'] in fit_catalog:records[record['item_id']]=record
    features=MetadataFeatures.from_items(data,records,minimum_items=config['minimum_items_per_metadata_token'])
    frame=pd.read_csv(root/protocol['splits']['train']['path'],dtype={'user_id':'string','item_id':'string','rating':'int8','timestamp':'int64'})
    training=sample_training(data,frame,negatives_per_positive=config['negatives_per_positive'],seed=config['negative_sampling_seed'])
    sources=['models/fm.py','scripts/tune_fm.py','scripts/tune_ease.py','scripts/tune_ials.py',
             'scripts/prepare_data.py','data_utils.py','evaluation.py','configs/fm_search.json','requirements.txt']
    plan={'configuration':config,'input_hashes':previous_plan['input_hashes'],
        'metadata_sha256':meta_hash,'source_hashes':{p:sha256(root/p) for p in sources},
        'protocol_sha256':previous_plan['protocol_sha256'],'cohort_fingerprint':context.fingerprint,
        'cohort':context.diagnostics,'query_timestamp':context.query_timestamp,
        'baseline_summary_hashes':{**previous_plan['baseline_summary_hashes'],'EASE':sha256(path)},
        'baseline_best_params':{**previous_plan['baseline_best_params'],'EASE':ease['best_params']},
        'feature_fingerprint':features.fingerprint,'feature_diagnostics':features.diagnostics,
        'training_examples':training[3],'test_evaluated':False,'training_data':'train only, no train+validation refit'}
    return protocol,config,data,context,results,features,training,records,plan


def assemble(output,summary,context,results,cached):
    return {'summary':summary,'context':context,'results':results,'reused_cache':cached,
        'search':pd.read_csv(output/'search_results.csv'),'curve':pd.read_csv(output/'training_curve.csv'),
        'comparison':pd.read_csv(output/'comparison.csv'),'ablation':pd.read_csv(output/'metadata_ablation.csv'),
        'history_slices':pd.read_csv(output/'history_slices.csv').set_index(['model','history_slice']),
        'popularity_slices':pd.read_csv(output/'popularity_slices.csv').set_index(['model','item_slice']),
        'intervals':pd.read_csv(output/'bootstrap_vs_baselines.csv')}


def save_model(local,model,features):
    np.savez(local/'best_parameters.npz',bias=np.array(model.bias_),linear=model.linear_,factors=model.factors_)
    np.savez(local/'item_features.npz',indices=features.item_indices,values=features.item_values)
    write_json(local/'feature_schema.json',{'feature_names':features.feature_names,'fit_fingerprint':features.fit_fingerprint,
               'fingerprint':features.fingerprint,'diagnostics':features.diagnostics,'n_users':features.n_users})


def run_validation_search(root=ROOT,*,reuse_completed=False,batch_size=128):
    root=Path(root)
    protocol,config,data,context,results,features,training,records_meta,plan=prepare(root);plan['evaluation_batch_size']=batch_size
    output=root/'reports/fm_validation';local=root/'data/validation/fm'
    output.mkdir(parents=True,exist_ok=True);local.mkdir(parents=True,exist_ok=True)
    summary_path=output/'summary.json'
    if reuse_completed and summary_path.exists():
        cached=json.loads(summary_path.read_text());artifacts=cached.get('artifact_hashes',{})
        if (all(cached.get(k)==v for k,v in plan.items()) and artifacts and
            all((root/p).is_file() and sha256(root/p)==h for p,h in artifacts.items()) and cached.get('completed_trials')==config['budget']):
            print('Loaded verified FM search: same code, inputs, metadata, samples, cohort and artifacts.',flush=True)
            results['FM']=load_result(local,'best',cached['best_validation_metrics'],context)
            return assemble(output,cached,context,results,True)
        raise ValueError('FM cache differs; rerun explicitly without reuse')
    grid=list(product(config['grid']['n_factors'],config['grid']['learning_rate'],config['grid']['regularization']))
    if len(grid)!=config['budget']:raise ValueError('Grid/budget mismatch')
    write_json(output/'search_plan.json',plan)  # before the first FM fit on real data
    records=[];curve=[];best_key=best=best_params=None;best_seconds=None
    for trial,(dimensions,lr,reg) in enumerate(grid,1):
        params={'n_factors':dimensions,'learning_rate':lr,'regularization':reg,
                'n_epochs':config['n_epochs'],'batch_size':config['batch_size'],'seed':config['seed'],'shuffle_seed':config['shuffle_seed']}
        print(f'[{trial}/{len(grid)}] FM f={dimensions}, lr={lr:g}, reg={reg:g}',flush=True)
        def progress(row):
            curve.append({'trial':trial,**params,**row})
            pd.DataFrame(curve).to_csv(output/'training_curve.csv',index=False)
        started=perf_counter();model=FM(**params).fit(data,features,training,progress_callback=progress)
        seconds=perf_counter()-started;result=evaluate_model(model,context,batch_size=batch_size)
        records.append({'trial':trial,**params,**result.metrics,'fit_seconds':seconds,
            'initial_probe_bce':model.curve_[0]['probe_bce'],'final_probe_bce':model.curve_[-1]['probe_bce']})
        pd.DataFrame(records).to_csv(output/'search_results.csv',index=False)
        key=(-result.metrics['ndcg_at_10'],dimensions,lr,-reg)
        if best_key is None or key<best_key:
            best_key=key;best=result;best_params=params;best_seconds=seconds;save_model(local,model,features)
        print(f'  NDCG@10={result.metrics["ndcg_at_10"]:.6f}, Recall@10={result.metrics["recall_at_10"]:.6f}, fit={seconds:.2f}s',flush=True)
        del model
    results['FM']=best
    # Fixed diagnostic ablation, not an additional HPO candidate or selection rule.
    id_features=MetadataFeatures.from_items(data,records_meta,minimum_items=config['minimum_items_per_metadata_token'],use_metadata=False)
    print('Diagnostic ID-only ablation at selected metadata-FM hyperparameters ...',flush=True)
    started=perf_counter();id_model=FM(**best_params).fit(data,id_features,training);id_seconds=perf_counter()-started
    id_result=evaluate_model(id_model,context,batch_size=batch_size)
    pd.DataFrame([{'features':'IDs + brand + categories',**best.metrics,'fit_seconds':best_seconds},
                  {'features':'IDs only (diagnostic)',**id_result.metrics,'fit_seconds':id_seconds}]).to_csv(output/'metadata_ablation.csv',index=False)
    pd.DataFrame(id_model.curve_).to_csv(output/'ablation_training_curve.csv',index=False)
    with deterministic_gzip(local/'id_only_per_user.csv.gz') as stream:id_result.per_user.to_csv(stream,lineterminator='\n')
    np.save(local/'id_only_recommendation_indices.npy',id_result.recommendations,allow_pickle=False)
    previous=pd.read_csv(root/'reports/ease_validation/comparison.csv').set_index('model')
    pd.DataFrame([{'model':name,**r.metrics,'fit_seconds':best_seconds if name=='FM' else float(previous.loc[name,'fit_seconds'])}
                  for name,r in results.items()]).to_csv(output/'comparison.csv',index=False)
    pd.concat({name:history_slices(r) for name,r in results.items()},names=['model']).to_csv(output/'history_slices.csv')
    pd.concat({name:popularity_slices(r,context) for name,r in results.items()},names=['model']).to_csv(output/'popularity_slices.csv')
    boot=protocol['confidence_intervals'];intervals=[]
    for baseline,ref in [('ItemKNN',results['ItemKNN']),('iALS',results['iALS']),('EASE',results['EASE']),('FM-ID-only',id_result)]:
        table=paired_bootstrap(best,ref,resamples=boot['resamples'],seed=boot['seed'],confidence=boot['confidence'],result_name='FM',baseline_name=baseline)
        table.insert(0,'baseline',baseline);intervals.append(table)
    intervals=pd.concat(intervals,ignore_index=True);intervals.to_csv(output/'bootstrap_vs_baselines.csv',index=False)
    with deterministic_gzip(local/'best_per_user.csv.gz') as stream:best.per_user.to_csv(stream,lineterminator='\n')
    np.save(local/'best_recommendation_indices.npy',best.recommendations,allow_pickle=False)
    summary={**plan,'completed_trials':len(records),'diagnostic_ablation_runs':1,'best_params':best_params,
        'best_validation_metrics':best.metrics,'id_only_validation_metrics':id_result.metrics,
        'baseline_validation_metrics':{name:r.metrics for name,r in results.items() if name!='FM'},
        'bootstrap_intervals':intervals.to_dict(orient='records'),
        'interval_interpretation':'descriptive validation after HPO; no independent test/multiple-comparison claim; ID-only ablation not separately tuned',
        'timing_note':'Prior baseline times reused; feature construction and sampling excluded from FM model fit',
        'artifact_hashes':{str(p.relative_to(root)):sha256(p) for p in [*sorted(local.iterdir()),*sorted(output.glob('*.csv'))] if p.is_file()}}
    write_json(summary_path,summary)
    return assemble(output,summary,context,results,False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--reuse-completed',action='store_true')
    args=parser.parse_args();run=run_validation_search(reuse_completed=args.reuse_completed)
    print(run['comparison'].to_string(index=False))
