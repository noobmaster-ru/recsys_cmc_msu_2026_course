"""Fixed timeSVD++ validation search and frozen comparison with all four models."""
import argparse
from itertools import product
import json
from pathlib import Path
import sys
from time import perf_counter
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from models.timesvdpp import TemporalData,TimeSVDpp
from evaluation import evaluate_model,history_slices,popularity_slices,paired_bootstrap
from scripts.prepare_data import sha256,deterministic_gzip
from scripts.tune_ials import load_result,write_json
from scripts.tune_ease import prepare as prepare_previous


def prepare(root):
    protocol,_,data,context,results,previous=prepare_previous(root)
    config=json.loads((root/'configs/timesvdpp_search.json').read_text())
    baseline_hashes=previous['baseline_summary_hashes'].copy();baseline_params=previous['baseline_best_params'].copy()
    for name,folder in [('EASE','ease'),('FM','fm')]:
        path=root/f'reports/{folder}_validation/summary.json';summary=json.loads(path.read_text())
        if summary['cohort_fingerprint']!=context.fingerprint or summary['input_hashes']!=previous['input_hashes']:
            raise ValueError(f'{name} cohort or inputs differ')
        for group in ('source_hashes','artifact_hashes'):
            for p,digest in summary[group].items():
                if sha256(root/p)!=digest:raise ValueError(f'Changed baseline artifact: {p}')
        baseline_hashes[name]=sha256(path);baseline_params[name]=summary['best_params']
        results[name]=load_result(root/f'data/validation/{folder}','best',summary['best_validation_metrics'],context)
    frame=pd.read_csv(root/protocol['splits']['train']['path'],dtype={'user_id':'string','item_id':'string','rating':'int8','timestamp':'int64'})
    temporal=TemporalData.from_frame(data,frame)
    files=['models/timesvdpp.py','scripts/tune_timesvdpp.py','scripts/tune_ease.py','scripts/tune_ials.py',
           'scripts/prepare_data.py','data_utils.py','evaluation.py','configs/timesvdpp_search.json','requirements.txt']
    plan={'configuration':config,'input_hashes':previous['input_hashes'],
        'source_hashes':{p:sha256(root/p) for p in files},'protocol_sha256':previous['protocol_sha256'],
        'cohort_fingerprint':context.fingerprint,'cohort':context.diagnostics,'query_timestamp':context.query_timestamp,
        'baseline_summary_hashes':baseline_hashes,'baseline_best_params':baseline_params,
        'time_statistics':{'min_day':temporal.min_day,'max_day':temporal.max_day,
            'user_day_pairs':len(temporal.user_day_codes),'observed_ratings':len(frame),'mean_rating':float(frame.rating.mean()),
            'implicit_history':'all observed fit ratings, including 1-3; target included during fit; frozen at query'},
        'test_evaluated':False,'training_data':'train only; no train+validation refit'}
    return protocol,config,data,context,results,temporal,plan


def assemble(output,summary,context,results,cached):
    return {'summary':summary,'context':context,'results':results,'reused_cache':cached,
        'search':pd.read_csv(output/'search_results.csv'),'curve':pd.read_csv(output/'training_curve.csv'),
        'comparison':pd.read_csv(output/'comparison.csv'),'ablation':pd.read_csv(output/'temporal_ablation.csv'),
        'history_slices':pd.read_csv(output/'history_slices.csv').set_index(['model','history_slice']),
        'popularity_slices':pd.read_csv(output/'popularity_slices.csv').set_index(['model','item_slice']),
        'intervals':pd.read_csv(output/'bootstrap_vs_baselines.csv')}


def run_validation_search(root=ROOT,*,reuse_completed=False,batch_size=128):
    root=Path(root)
    protocol,config,data,context,results,temporal,plan=prepare(root);plan['evaluation_batch_size']=batch_size
    output=root/'reports/timesvdpp_validation';local=root/'data/validation/timesvdpp'
    output.mkdir(parents=True,exist_ok=True);local.mkdir(parents=True,exist_ok=True)
    summary_path=output/'summary.json'
    if reuse_completed and summary_path.exists():
        cached=json.loads(summary_path.read_text());artifacts=cached.get('artifact_hashes',{})
        if (all(cached.get(k)==v for k,v in plan.items()) and artifacts and
            all((root/p).is_file() and sha256(root/p)==h for p,h in artifacts.items()) and cached.get('completed_trials')==config['budget']):
            print('Loaded verified timeSVD++ results: identical code, fit dates, inputs, cohort and artifacts.',flush=True)
            results['timeSVD++']=load_result(local,'best',cached['best_validation_metrics'],context)
            return assemble(output,cached,context,results,True)
        raise ValueError('timeSVD++ cache differs; explicitly rerun without reuse')
    grid=list(product(config['grid']['n_factors'],config['grid']['n_bins'],config['grid']['regularization']))
    if len(grid)!=config['budget']:raise ValueError('Grid/budget mismatch')
    write_json(output/'search_plan.json',plan)  # before first real-data timeSVD++ fit
    records=[];curve=[];best_key=best=best_params=best_diagnostics=None;best_seconds=None
    for trial,(dimensions,bins,reg) in enumerate(grid,1):
        params={'n_factors':dimensions,'n_bins':bins,'regularization':reg,'learning_rate':config['learning_rate'],
                'n_epochs':config['n_epochs'],'beta':config['beta'],'seed':config['seed']}
        print(f'[{trial}/{len(grid)}] timeSVD++ f={dimensions}, bins={bins}, reg={reg:g}',flush=True)
        def progress(row):
            curve.append({'trial':trial,**params,**row})
            pd.DataFrame(curve).to_csv(output/'training_curve.csv',index=False)
        started=perf_counter();model=TimeSVDpp(**params).fit_temporal(temporal,progress_callback=progress)
        seconds=perf_counter()-started;result=evaluate_model(model,context,batch_size=batch_size)
        records.append({'trial':trial,**params,**result.metrics,'fit_seconds':seconds,
            'initial_training_rmse':model.curve_[0]['training_rmse'],'final_training_rmse':model.curve_[-1]['training_rmse'],
            'final_training_objective':model.curve_[-1]['training_objective']})
        pd.DataFrame(records).to_csv(output/'search_results.csv',index=False)
        key=(-result.metrics['ndcg_at_10'],dimensions,bins,-reg)
        if best_key is None or key<best_key:
            best_key=key;best=result;best_params=params;best_seconds=seconds
            np.savez(local/'best_parameters.npz',**model.parameters_,mean=np.array(model.mean_))
            np.savez(local/'time_schema.npz',user_mean_day=temporal.user_mean_day,user_day_codes=temporal.user_day_codes,
                     min_day=np.array(temporal.min_day),max_day=np.array(temporal.max_day))
            day=context.query_timestamp//86400;rows=context.user_indices;query_bin=int(temporal.bins([day],bins)[0])
            counts=np.bincount(temporal.items*bins+model.training_bins_,minlength=len(data.item_ids)*bins).reshape(len(data.item_ids),bins)
            dev=temporal.deviation(rows,day,model.beta)
            best_diagnostics={'query_day':day,'query_bin':query_bin,'n_bins':bins,
                'query_user_day_unseen':int(np.sum(temporal.lookup_user_days(rows,np.repeat(day,len(rows)))<0)),
                'items_without_fit_observations_in_query_bin':int(np.sum(counts[:,query_bin]==0)),
                'fit_bin_counts':counts.sum(axis=0).tolist(),
                'query_deviation_min':float(dev.min()),'query_deviation_median':float(np.median(dev)),
                'query_deviation_max':float(dev.max()),'parameter_bytes':sum(p.nbytes for p in model.parameters_.values())}
        print(f'  NDCG@10={result.metrics["ndcg_at_10"]:.6f}, Recall@10={result.metrics["recall_at_10"]:.6f}, fit={seconds:.2f}s',flush=True)
        del model
    results['timeSVD++']=best
    print('Diagnostic static SVD++ ablation at selected hyperparameters ...',flush=True)
    started=perf_counter();static=TimeSVDpp(**best_params,temporal=False).fit_temporal(temporal);static_seconds=perf_counter()-started
    static_result=evaluate_model(static,context,batch_size=batch_size)
    pd.DataFrame([{'model':'timeSVD++',**best.metrics,'fit_seconds':best_seconds},
                  {'model':'SVD++ static (diagnostic)',**static_result.metrics,'fit_seconds':static_seconds}]).to_csv(output/'temporal_ablation.csv',index=False)
    pd.DataFrame(static.curve_).to_csv(output/'ablation_training_curve.csv',index=False)
    with deterministic_gzip(local/'static_per_user.csv.gz') as stream:static_result.per_user.to_csv(stream,lineterminator='\n')
    np.save(local/'static_recommendation_indices.npy',static_result.recommendations,allow_pickle=False)
    previous=pd.read_csv(root/'reports/fm_validation/comparison.csv').set_index('model')
    pd.DataFrame([{'model':name,**r.metrics,'fit_seconds':best_seconds if name=='timeSVD++' else float(previous.loc[name,'fit_seconds'])}
                  for name,r in results.items()]).to_csv(output/'comparison.csv',index=False)
    pd.concat({name:history_slices(r) for name,r in results.items()},names=['model']).to_csv(output/'history_slices.csv')
    pd.concat({name:popularity_slices(r,context) for name,r in results.items()},names=['model']).to_csv(output/'popularity_slices.csv')
    boot=protocol['confidence_intervals'];intervals=[]
    for baseline,ref in [(name,results[name]) for name in ('ItemKNN','iALS','EASE','FM')]+[('SVD++ static',static_result)]:
        table=paired_bootstrap(best,ref,resamples=boot['resamples'],seed=boot['seed'],confidence=boot['confidence'],result_name='timeSVD++',baseline_name=baseline)
        table.insert(0,'baseline',baseline);intervals.append(table)
    intervals=pd.concat(intervals,ignore_index=True);intervals.to_csv(output/'bootstrap_vs_baselines.csv',index=False)
    with deterministic_gzip(local/'best_per_user.csv.gz') as stream:best.per_user.to_csv(stream,lineterminator='\n')
    np.save(local/'best_recommendation_indices.npy',best.recommendations,allow_pickle=False)
    summary={**plan,'completed_trials':len(records),'diagnostic_ablation_runs':1,'best_params':best_params,
        'best_validation_metrics':best.metrics,'best_temporal_diagnostics':best_diagnostics,
        'static_validation_metrics':static_result.metrics,
        'baseline_validation_metrics':{name:r.metrics for name,r in results.items() if name!='timeSVD++'},
        'bootstrap_intervals':intervals.to_dict(orient='records'),
        'interval_interpretation':'descriptive validation after HPO; no independent test or multiple-comparison claim; static ablation not retuned',
        'timing_note':'Prior baseline times reused; indexing and time schema construction excluded from timeSVD++ fit',
        'artifact_hashes':{str(p.relative_to(root)):sha256(p) for p in [*sorted(local.iterdir()),*sorted(output.glob('*.csv'))] if p.is_file()}}
    write_json(summary_path,summary)
    return assemble(output,summary,context,results,False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--reuse-completed',action='store_true')
    args=parser.parse_args();run=run_validation_search(reuse_completed=args.reuse_completed)
    print(run['comparison'].to_string(index=False))
