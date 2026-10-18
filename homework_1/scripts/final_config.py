"""Freeze final parameters from completed validation searches, before test access."""
import json
from pathlib import Path
import pandas as pd
from scripts.prepare_data import sha256
from scripts.tune_ials import write_json

MODEL_FOLDERS={'ItemKNN':'item_knn','iALS':'ials','EASE':'ease','FM':'fm','timeSVD++':'timesvdpp'}
PARAMETER_KEYS={
 'ItemKNN':['n_neighbors','shrinkage'],
 'iALS':['alpha','n_factors','n_iterations','regularization','seed','solver_batch_size'],
 'EASE':['regularization'],
 'FM':['batch_size','learning_rate','n_epochs','n_factors','regularization','seed','shuffle_seed'],
 'timeSVD++':['beta','learning_rate','n_bins','n_epochs','n_factors','regularization','seed']}


def validation_winner(name,table):
    keys={
     'ItemKNN':(['ndcg_at_10','n_neighbors','shrinkage'],[False,True,True]),
     'iALS':(['ndcg_at_10','n_factors','alpha','regularization'],[False,True,True,False]),
     'EASE':(['ndcg_at_10','regularization'],[False,False]),
     'FM':(['ndcg_at_10','n_factors','learning_rate','regularization'],[False,True,True,False]),
     'timeSVD++':(['ndcg_at_10','n_factors','n_bins','regularization'],[False,True,True,False])}
    columns,ascending=keys[name]
    if table.empty or not table.ndcg_at_10.notna().all():raise ValueError('Invalid validation table')
    return table.sort_values(columns,ascending=ascending,kind='stable').iloc[0]


def write_immutable_json(path,payload):
    path=Path(path)
    if path.exists():
        if json.loads(path.read_text())!=payload:raise ValueError(f'Frozen file differs; refusing overwrite: {path}')
    else:
        path.parent.mkdir(parents=True,exist_ok=True);write_json(path,payload)


def freeze_models(root):
    root=Path(root);reports={};params={};metrics={};hashes={}
    for name,folder in MODEL_FOLDERS.items():
        relative=f'reports/{folder}_validation/summary.json';path=root/relative
        summary=json.loads(path.read_text())
        if summary['test_evaluated'] is not False:raise ValueError('Parameters must come from validation only')
        for group in ('source_hashes','local_artifacts' if name=='ItemKNN' else 'artifact_hashes'):
            for artifact,digest in summary[group].items():
                # Selection/refits need verified results, not old fitted weights.
                if group!='source_hashes' and artifact.startswith('data/') and not any(k in artifact for k in ['per_user','recommendation_indices']):continue
                if sha256(root/artifact)!=digest:raise ValueError(f'Changed validation artifact: {artifact}')
        search=pd.read_csv(root/f'reports/{folder}_validation/search_results.csv')
        if len(search)!=summary['completed_trials']:raise ValueError('Incomplete search')
        winner=validation_winner(name,search)
        for key in PARAMETER_KEYS[name]:
            if winner[key]!=summary['best_params'][key]:raise ValueError(f'Validation selection mismatch: {name}/{key}')
        if abs(winner.ndcg_at_10-summary['best_validation_metrics']['ndcg_at_10'])>1e-14:
            raise ValueError('Validation metric mismatch')
        reports[name]=relative;params[name]=summary['best_params'];metrics[name]=summary['best_validation_metrics']
        hashes[name]=sha256(path)
    champion=max(metrics,key=lambda name:metrics[name]['ndcg_at_10'])
    fm=json.loads((root/'configs/fm_search.json').read_text())
    payload={'version':2,'experiment':'temporal_metadata_v2','prior_test_already_observed':True,
      'correction_plan_sha256':sha256(root/'reports/correction_plan.json'),'selection':'completed validation macro NDCG@10; no test-based choices',
      'models':params,'validation_reference':champion,'validation_reports':reports,'validation_summary_hashes':hashes,
      'validation_metrics':metrics,'fit':'train + validation, fresh initialization, no warm start',
      'query_time_utc':'2017-01-01T00:00:00+00:00','evaluation_batch_size':128,
      'fm_preprocessing':{key:fm[key] for key in ['minimum_items_per_metadata_token','negatives_per_positive','negative_sampling_seed']},
      'feature_policy':'rebuild indices, metadata vocabularies, negative samples, temporal schema from train+validation only',
      'statistics':'paired bootstrap vs validation-selected reference; 2000 resamples, seed 42, 95%; unadjusted intervals',
      'post_test_policy':'corrective reevaluation on previously used holdout; after v2 freeze no hyperparameter, epoch, feature or protocol changes; no diagnostic ablation fits on test'}
    write_immutable_json(root/'configs/final_models.json',payload)
    return payload
