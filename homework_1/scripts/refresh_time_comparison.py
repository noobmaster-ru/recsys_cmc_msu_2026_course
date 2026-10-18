"""Refresh timeSVD++ comparison against corrected FM; do not refit timeSVD++."""
from pathlib import Path
import json
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import pandas as pd
from scripts.prepare_data import sha256
from scripts.tune_fm import prepare
from scripts.tune_ials import load_result,write_json
from evaluation import history_slices,popularity_slices,paired_bootstrap


def refresh(root=ROOT):
    root=Path(root)
    protocol,config,data,context,results,features,training,frame,fm_plan=prepare(root)
    directory=root/'reports/timesvdpp_validation';path=directory/'summary.json'
    original=json.loads(path.read_text())
    if original['cohort_fingerprint']!=context.fingerprint:raise ValueError('Changed cohort')
    for relative,digest in original['source_hashes'].items():
        if sha256(root/relative)!=digest:raise ValueError(f'Changed original time training source: {relative}')
    for relative,digest in original['artifact_hashes'].items():
        if relative.startswith('data/') and any(key in relative for key in ['per_user','recommendation_indices']):
            if sha256(root/relative)!=digest:raise ValueError(f'Changed original time ranking: {relative}')
    fm=json.loads((root/'reports/fm_validation/summary.json').read_text())
    for relative,digest in fm['artifact_hashes'].items():
        if sha256(root/relative)!=digest:raise ValueError(f'Changed corrected FM evidence: {relative}')
    results['FM']=load_result(root/'data/validation/fm','best',fm['best_validation_metrics'],context)
    results['timeSVD++']=load_result(root/'data/validation/timesvdpp','best',original['best_validation_metrics'],context)
    static=load_result(root/'data/validation/timesvdpp','static',original['static_validation_metrics'],context)
    previous=pd.read_csv(root/'reports/fm_validation/comparison.csv').set_index('model')
    original_table=pd.read_csv(directory/'comparison.csv').set_index('model')
    rows=[{'model':name,**result.metrics,'fit_seconds':float(original_table.loc[name,'fit_seconds']) if name=='timeSVD++' else float(previous.loc[name,'fit_seconds'])}
          for name,result in results.items()]
    pd.DataFrame(rows).to_csv(directory/'comparison.csv',index=False)
    pd.concat({name:history_slices(result) for name,result in results.items()},names=['model']).to_csv(directory/'history_slices.csv')
    pd.concat({name:popularity_slices(result,context) for name,result in results.items()},names=['model']).to_csv(directory/'popularity_slices.csv')
    intervals=[];boot=protocol['confidence_intervals']
    for name,result in [(name,results[name]) for name in ['ItemKNN','iALS','EASE','FM']]+[('SVD++ static',static)]:
        table=paired_bootstrap(results['timeSVD++'],result,resamples=boot['resamples'],seed=boot['seed'],confidence=boot['confidence'],result_name='timeSVD++',baseline_name=name)
        table.insert(0,'baseline',name);intervals.append(table)
    intervals=pd.concat(intervals,ignore_index=True);intervals.to_csv(directory/'bootstrap_vs_baselines.csv',index=False)
    # Preserve exact original training/search evidence; identify comparison-only revision.
    original['source_hashes']['scripts/refresh_time_comparison.py']=sha256(root/'scripts/refresh_time_comparison.py')
    original['comparison_revision']={'experiment':'temporal_metadata_v2','time_model_retrained':False,
        'original_summary_sha256':sha256(root/'archive/snapshot_v1/reports/timesvdpp_validation/summary.json'),
        'correction_plan_sha256':sha256(root/'reports/correction_plan.json')}
    original['baseline_summary_hashes']['FM']=sha256(root/'reports/fm_validation/summary.json')
    original['baseline_best_params']['FM']=fm['best_params']
    original['baseline_validation_metrics']={name:r.metrics for name,r in results.items() if name!='timeSVD++'}
    original['bootstrap_intervals']=intervals.to_dict(orient='records')
    for csv in directory.glob('*.csv'):original['artifact_hashes'][str(csv.relative_to(root))]=sha256(csv)
    original['timing_note']='Original four validation searches retained; only comparison with corrected FM refreshed. No timeSVD++ retraining or selection.'
    write_json(path,original)
    print('Refreshed five-model validation comparison without timeSVD++ retraining',flush=True)


if __name__=='__main__':refresh()
