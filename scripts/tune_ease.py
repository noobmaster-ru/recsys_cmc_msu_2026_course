"""Fixed EASE regularization search; train/validation only, frozen CF/iALS baselines."""
import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

import numpy as np
import pandas as pd
from data_utils import FitData
from evaluation import EvaluationContext, evaluate_model, history_slices, popularity_slices, paired_bootstrap
from models.ease import EASE
from scripts.prepare_data import sha256, deterministic_gzip
from scripts.tune_ials import load_result, write_json


def prepare(root):
    protocol=json.loads((root/'reports/evaluation_protocol.json').read_text())
    config=json.loads((root/'configs/ease_search.json').read_text())
    if protocol['k']!=10 or protocol['relevance']['rating_at_least']!=4:
        raise ValueError('Expected frozen K=10, rating>=4 protocol')
    frames,hashes={},{}
    for name in ('train','validation'):
        spec=protocol['splits'][name];path=root/spec['path'];hashes[name]=sha256(path)
        if hashes[name]!=spec['sha256']:raise ValueError(f'Changed split: {name}')
        frames[name]=pd.read_csv(path,dtype={'user_id':'string','item_id':'string','rating':'int8','timestamp':'int64'})
    data=FitData.from_frame(frames['train'])
    timestamp=int(pd.Timestamp(protocol['validation_start_utc']).timestamp())
    context=EvaluationContext.from_frames(frames['train'],frames['validation'],fit_data=data,k=10,query_timestamp=timestamp)
    baseline_summaries,baseline_hashes,results={},{},{}
    for model,folder in (('ItemKNN','item_knn'),('iALS','ials')):
        path=root/f'reports/{folder}_validation/summary.json';summary=json.loads(path.read_text())
        if summary['cohort_fingerprint']!=context.fingerprint or summary['input_hashes']!=hashes:
            raise ValueError(f'{model} cohort/inputs differ')
        for key in ('source_hashes','local_artifacts' if model=='ItemKNN' else 'artifact_hashes'):
            for artifact,digest in summary[key].items():
                if sha256(root/artifact)!=digest:raise ValueError(f'Changed baseline artifact: {artifact}')
        baseline_summaries[model]=summary;baseline_hashes[model]=sha256(path)
        results[model]=load_result(root/f'data/validation/{folder}','best',summary['best_validation_metrics'],context)
    cf=baseline_summaries['ItemKNN']
    results={'Popularity':load_result(root/'data/validation/item_knn','popularity',cf['popularity_validation_metrics'],context),**results}
    for key in ('evaluation_users','catalog_items','relevant_interactions'):
        if context.diagnostics[key]!=protocol['evaluation_diagnostics']['validation'][key]:raise ValueError(f'Changed cohort: {key}')
    sources=['models/ease.py','scripts/tune_ease.py','scripts/tune_ials.py','scripts/prepare_data.py',
             'configs/ease_search.json','data_utils.py','evaluation.py','requirements.txt']
    plan={'configuration':config,'input_hashes':hashes,'source_hashes':{p:sha256(root/p) for p in sources},
          'protocol_sha256':sha256(root/'reports/evaluation_protocol.json'),
          'cohort_fingerprint':context.fingerprint,'cohort':context.diagnostics,'query_timestamp':timestamp,
          'baseline_summary_hashes':baseline_hashes,
          'baseline_best_params':{name:s['best_params'] for name,s in baseline_summaries.items()},
          'test_evaluated':False,'training_data':'train only; no train+validation refit'}
    return protocol,config,data,context,results,plan


def assemble(output,summary,context,results,cached):
    return {'summary':summary,'context':context,'results':results,'reused_cache':cached,
            'search':pd.read_csv(output/'search_results.csv'),
            'comparison':pd.read_csv(output/'comparison.csv'),
            'history_slices':pd.read_csv(output/'history_slices.csv').set_index(['model','history_slice']),
            'popularity_slices':pd.read_csv(output/'popularity_slices.csv').set_index(['model','item_slice']),
            'intervals':pd.read_csv(output/'bootstrap_vs_baselines.csv')}


def run_validation_search(root=ROOT,*,reuse_completed=False,batch_size=128):
    root=Path(root)
    protocol,config,data,context,results,plan=prepare(root);plan['batch_size']=batch_size
    grid=config['grid']['regularization']
    if len(grid)!=config['budget'] or len(set(grid))!=len(grid):raise ValueError('Invalid fixed grid/budget')
    output=root/'reports/ease_validation';local=root/'data/validation/ease'
    output.mkdir(parents=True,exist_ok=True);local.mkdir(parents=True,exist_ok=True)
    summary_path=output/'summary.json'
    if reuse_completed and summary_path.exists():
        summary=json.loads(summary_path.read_text());artifacts=summary.get('artifact_hashes',{})
        if (all(summary.get(k)==v for k,v in plan.items()) and artifacts
            and all((root/p).is_file() and sha256(root/p)==h for p,h in artifacts.items())
            and summary.get('completed_trials')==config['budget']):
            print('Loaded verified EASE search: same code, inputs, cohort and artifacts.',flush=True)
            results['EASE']=load_result(local,'best',summary['best_validation_metrics'],context)
            return assemble(output,summary,context,results,True)
        raise ValueError('EASE cache differs; explicitly rerun the fixed search without reuse')
    write_json(output/'search_plan.json',plan)  # saved before the first EASE fit on real data
    records=[];best_key=best=best_params=best_diagnostics=None;best_seconds=None
    for trial,regularization in enumerate(grid,1):
        print(f'[{trial}/{len(grid)}] fitting EASE lambda={regularization:g} ...',flush=True)
        started=perf_counter();model=EASE(regularization).fit(data);fit_seconds=perf_counter()-started
        result=evaluate_model(model,context,batch_size=batch_size)
        records.append({'trial':trial,'regularization':regularization,**result.metrics,
                        'fit_seconds':fit_seconds,**model.diagnostics_})
        pd.DataFrame(records).to_csv(output/'search_results.csv',index=False)
        key=(-result.metrics['ndcg_at_10'],-regularization)
        if best_key is None or key<best_key:
            best_key=key;best=result;best_params={'regularization':regularization}
            best_diagnostics=model.diagnostics_.copy();best_seconds=fit_seconds
            # Keep only one dense model in memory. Best weights are local and ignored by Git.
            np.save(local/'best_weights.npy',model.weights_,allow_pickle=False)
            np.save(local/'best_precision_diagonal.npy',model.precision_diagonal_,allow_pickle=False)
        print(f'  NDCG@10={result.metrics["ndcg_at_10"]:.6f}, Recall@10={result.metrics["recall_at_10"]:.6f}, fit={fit_seconds:.2f}s',flush=True)
        del model
    results['EASE']=best
    previous=pd.read_csv(root/'reports/ials_validation/comparison.csv').set_index('model')
    comparison=pd.DataFrame([{'model':name,**result.metrics,
        'fit_seconds':best_seconds if name=='EASE' else float(previous.loc[name,'fit_seconds'])}
        for name,result in results.items()])
    comparison.to_csv(output/'comparison.csv',index=False)
    pd.concat({name:history_slices(r) for name,r in results.items()},names=['model']).to_csv(output/'history_slices.csv')
    pd.concat({name:popularity_slices(r,context) for name,r in results.items()},names=['model']).to_csv(output/'popularity_slices.csv')
    boot=protocol['confidence_intervals'];intervals=[]
    for baseline in ('ItemKNN','iALS'):
        table=paired_bootstrap(best,results[baseline],resamples=boot['resamples'],seed=boot['seed'],
                               confidence=boot['confidence'],result_name='EASE',baseline_name=baseline)
        table.insert(0,'baseline',baseline);intervals.append(table)
    intervals=pd.concat(intervals,ignore_index=True);intervals.to_csv(output/'bootstrap_vs_baselines.csv',index=False)
    with deterministic_gzip(local/'best_per_user.csv.gz') as stream:best.per_user.to_csv(stream,lineterminator='\n')
    np.save(local/'best_recommendation_indices.npy',best.recommendations,allow_pickle=False)
    summary={**plan,'completed_trials':len(records),'best_params':best_params,
             'best_validation_metrics':best.metrics,'best_model_diagnostics':best_diagnostics,
             'baseline_validation_metrics':{name:r.metrics for name,r in results.items() if name!='EASE'},
             'bootstrap_intervals':intervals.to_dict(orient='records'),
             'interval_interpretation':'descriptive validation after HPO; no independent test or multiple-comparison claim',
             'timing_note':'Prior CF/iALS times reused; indexing excluded from model fit, hardware/run noise applies',
             'artifact_hashes':{str(p.relative_to(root)):sha256(p) for p in
                 [*sorted(local.iterdir()),*sorted(output.glob('*.csv'))] if p.is_file()}}
    write_json(summary_path,summary)
    return assemble(output,summary,context,results,False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reuse-completed',action='store_true');args=parser.parse_args()
    run=run_validation_search(reuse_completed=args.reuse_completed)
    print(run['comparison'].to_string(index=False))
