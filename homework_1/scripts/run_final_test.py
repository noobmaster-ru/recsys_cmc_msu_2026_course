"""Freeze validation choices, refit ALL models, then unseal and evaluate test once."""
import gzip
import json
from datetime import datetime,timezone
from pathlib import Path
import sys
from time import perf_counter
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from scipy import sparse
from data_utils import FitData
from evaluation import EvaluationContext,evaluate_model,evaluate_popularity,history_slices,popularity_slices,paired_bootstrap
from models.item_knn import ItemKNN
from models.ials import IALS
from models.ease import EASE
from models.fm import FM,MetadataFeatures,sample_training
from models.timesvdpp import TimeSVDpp,TemporalData
from scripts.final_config import freeze_models,write_immutable_json,MODEL_FOLDERS
from scripts.prepare_data import sha256,deterministic_gzip
from scripts.tune_ials import write_json,load_result


class TestReadGate:
    def __init__(self,root):
        self.test=(Path(root)/'data/splits/test.csv.gz').resolve()
        self.full={(Path(root)/p).resolve() for p in ['data/processed/interactions.csv.gz','data/raw/Video_Games_5.json.gz',
            'data/processed/items.jsonl.gz','data/raw/meta_Video_Games.json.gz']}
        self.active=True;self.allow_test=False;self.opened=False;self.open_count=0
    def __call__(self,event,args):
        if not self.active or event!='open' or not isinstance(args[0],(str,bytes)):return
        value=args[0].decode() if isinstance(args[0],bytes) else args[0];path=Path(value).resolve()
        if path in self.full:raise RuntimeError('Final pipeline must use split files, not full-history data')
        if path==self.test:
            if not self.allow_test:raise RuntimeError('Test is sealed until all five refits are complete')
            self.opened=True;self.open_count+=1


def journal(output,event,**fields):
    row={'event':event,'utc':datetime.now(timezone.utc).isoformat(),**fields}
    with (output/'execution_journal.jsonl').open('a',encoding='utf-8') as stream:
        stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')


def prepare_fit(root,protocol):
    frames=[];hashes={}
    for split in ['train','validation']:
        spec=protocol['splits'][split];path=root/spec['path'];hashes[split]=sha256(path)
        if hashes[split]!=spec['sha256']:raise ValueError(f'Changed {split}')
        frames.append(pd.read_csv(path,dtype={'user_id':'string','item_id':'string','rating':'int8','timestamp':'int64'}))
    frame=pd.concat(frames,ignore_index=True);data=FitData.from_frame(frame)
    query=int(pd.Timestamp(protocol['test_start_utc']).timestamp())
    if data.max_timestamp>=query:raise ValueError('Fit reaches test time')
    expected=protocol['evaluation_diagnostics']['test']
    for actual,key in [(len(frame),'fit_interactions'),(len(data.user_ids),'fit_users'),(len(data.item_ids),'catalog_items')]:
        if actual!=expected[key]:raise ValueError(f'Final fit changed: {key}')
    # Snapshot metadata are deliberately not read by the corrected pipeline.
    return frame,data,{},hashes,None,query


def refit_all(data,frame,metadata,lock,output,local,*,event=lambda *a,**k:None):
    """No test/context argument: all five fits depend solely on the combined fit."""
    manifest={};output.mkdir(parents=True,exist_ok=True);local.mkdir(parents=True,exist_ok=True)
    fm_spec=lock['fm_preprocessing']
    for name,folder in MODEL_FOLDERS.items():
        params=lock['models'][name];directory=local/folder;directory.mkdir(parents=True,exist_ok=True)
        event('refit_started',model=name);print(f'Refitting {name} on train+validation ...',flush=True)
        started=perf_counter();details={};curve=None
        if name=='ItemKNN':model=ItemKNN(**params).fit(data)
        elif name=='iALS':model=IALS(**params).fit(data)
        elif name=='EASE':model=EASE(**params).fit(data)
        elif name=='FM':
            query=int(pd.Timestamp(lock['query_time_utc']).timestamp())
            features=MetadataFeatures.from_history(data,frame,query,minimum_items=fm_spec['minimum_items_per_metadata_token'])
            training=sample_training(data,frame,negatives_per_positive=fm_spec['negatives_per_positive'],seed=fm_spec['negative_sampling_seed'])
            preparation=perf_counter()-started;started=perf_counter()
            model=FM(**params).fit(data,features,training)
            details={'feature_fingerprint':features.fingerprint,'feature_diagnostics':features.diagnostics,
                     'training_examples':training[3],'preparation_seconds':preparation}
        else:
            temporal=TemporalData.from_frame(data,frame);preparation=perf_counter()-started;started=perf_counter()
            model=TimeSVDpp(**params).fit_temporal(temporal)
            details={'preparation_seconds':preparation,'min_day':temporal.min_day,'max_day':temporal.max_day,
                     'user_day_pairs':len(temporal.user_day_codes),'mean_rating':model.mean_}
        fit_seconds=perf_counter()-started
        if name=='ItemKNN':sparse.save_npz(directory/'neighbors.npz',model.neighbors_)
        elif name=='iALS':
            np.savez(directory/'parameters.npz',user_factors=model.user_factors_,item_factors=model.item_factors_)
            curve=pd.DataFrame({'iteration':range(len(model.loss_history_)),'objective':model.loss_history_})
        elif name=='EASE':
            np.save(directory/'weights.npy',model.weights_,allow_pickle=False)
            np.save(directory/'precision_diagonal.npy',model.precision_diagonal_,allow_pickle=False)
            details=model.diagnostics_
        elif name=='FM':
            np.savez(directory/'parameters.npz',bias=np.array(model.bias_),linear=model.linear_,factors=model.factors_)
            np.savez(directory/'item_features.npz',indices=features.item_indices,values=features.item_values)
            write_json(directory/'feature_schema.json',{'feature_names':features.feature_names,'n_users':features.n_users,
                'fit_fingerprint':features.fit_fingerprint,'fingerprint':features.fingerprint,'diagnostics':features.diagnostics})
            curve=pd.DataFrame(model.curve_)
        else:
            np.savez(directory/'parameters.npz',**model.parameters_,mean=np.array(model.mean_))
            np.savez(directory/'time_schema.npz',user_mean_day=temporal.user_mean_day,user_day_codes=temporal.user_day_codes,
                     min_day=np.array(temporal.min_day),max_day=np.array(temporal.max_day))
            curve=pd.DataFrame(model.curve_)
        if curve is not None:curve.to_csv(output/f'training_{folder}.csv',index=False)
        info={'params':params,'fit_fingerprint':data.fingerprint,'fit_seconds':fit_seconds,'diagnostics':details,
              'files':{p.name:sha256(p) for p in sorted(directory.iterdir()) if p.is_file()}}
        manifest[name]=info
        event('refit_completed',model=name,fit_fingerprint=data.fingerprint,fit_seconds=fit_seconds)
        del model
    write_json(local/'fit_manifest.json',manifest)
    return manifest


def restore_model(name,data,frame,lock,directory):
    params=lock['models'][name]
    if name=='ItemKNN':model=ItemKNN(**params);model.neighbors_=sparse.load_npz(directory/'neighbors.npz')
    elif name=='iALS':
        model=IALS(**params);z=np.load(directory/'parameters.npz',allow_pickle=False)
        model.user_factors_=z['user_factors'];model.item_factors_=z['item_factors']
        model.positive_history_sizes_=np.diff(data.positive.indptr)
    elif name=='EASE':model=EASE(**params);model.weights_=np.load(directory/'weights.npy',mmap_mode='r',allow_pickle=False)
    elif name=='FM':
        model=FM(**params);s=json.loads((directory/'feature_schema.json').read_text());x=np.load(directory/'item_features.npz',allow_pickle=False)
        model.features_=MetadataFeatures(x['indices'],x['values'],s['feature_names'],s['n_users'],s['fit_fingerprint'],s['fingerprint'],s['diagnostics'])
        if model.features_.fit_fingerprint!=data.fingerprint:raise ValueError('FM features mismatch')
        z=np.load(directory/'parameters.npz',allow_pickle=False);model.bias_=float(z['bias']);model.linear_=z['linear'];model.factors_=z['factors'];model.prepare_scoring()
    else:
        model=TimeSVDpp(**params).initialize(TemporalData.from_frame(data,frame))
        z=np.load(directory/'parameters.npz',allow_pickle=False)
        model.parameters_={key:z[key] for key in model.parameters_};model.mean_=float(z['mean']);model.prepare_scoring()
    model.data_=data;model.item_ids=data.item_ids.copy();model.fit_fingerprint=data.fingerprint
    return model


def load_completed_final(root=ROOT):
    """Read verified frozen reports/artifacts only; never opens test or trains."""
    root=Path(root);output=root/'reports/final_test';path=output/'summary.json'
    if not path.exists():raise RuntimeError('Run scripts/run_final_test.py first; notebook only loads completed final results')
    s=json.loads(path.read_text())
    if not s['test_evaluated'] or s['completed_refits']!=5 or s['completed_test_evaluations']!=6:raise ValueError('Incomplete final result')
    if sha256(root/'configs/final_models.json')!=s['locked_config_sha256']:raise ValueError('Changed final lock')
    if sha256(root/'reports/evaluation_protocol.json')!=s['protocol_sha256']:raise ValueError('Changed protocol')
    for group in ['source_hashes','artifact_hashes']:
        for p,h in s[group].items():
            if not (root/p).is_file() or sha256(root/p)!=h:raise ValueError(f'Changed final artifact: {p}')
    for name,path in s['validation_reports'].items():
        if sha256(root/path)!=s['validation_summary_hashes'][name]:raise ValueError('Changed validation evidence')
    return {'summary':s,'comparison':pd.read_csv(output/'comparison.csv'),
        'intervals':pd.read_csv(output/'bootstrap_vs_validation_reference.csv'),
        'history_slices':pd.read_csv(output/'history_slices.csv').set_index(['model','history_slice']),
        'popularity_slices':pd.read_csv(output/'popularity_slices.csv').set_index(['model','item_slice']),
        'validation_vs_test':pd.read_csv(output/'validation_vs_test.csv'),
        'training_curves':{name:pd.read_csv(output/f'training_{folder}.csv') for name,folder in MODEL_FOLDERS.items() if name in ['iALS','FM','timeSVD++']}}


def run_final(root=ROOT):
    root=Path(root);lock=freeze_models(root)
    output=root/'reports/final_test';local=root/'data/final_test'
    if (output/'summary.json').exists():return load_completed_final(root)
    if (output/'execution_journal.jsonl').exists():raise RuntimeError('An incomplete final run exists; inspect its journal before any restart')
    output.mkdir(parents=True,exist_ok=True);local.mkdir(parents=True,exist_ok=True)
    gate=TestReadGate(root);sys.addaudithook(gate)
    try:
        protocol=json.loads((root/'reports/evaluation_protocol.json').read_text())
        frame,data,metadata,input_hashes,metadata_hash,query=prepare_fit(root,protocol)
        files=['scripts/run_final_test.py','scripts/final_config.py','scripts/prepare_data.py','scripts/tune_ials.py',
               'models/item_knn.py','models/ials.py','models/ease.py','models/fm.py','models/timesvdpp.py',
               'data_utils.py','evaluation.py','requirements.txt']
        plan={'locked_config_sha256':sha256(root/'configs/final_models.json'),'models':lock['models'],
            'validation_reference':lock['validation_reference'],'validation_reports':lock['validation_reports'],
            'validation_summary_hashes':lock['validation_summary_hashes'],
            'source_hashes':{p:sha256(root/p) for p in files},'protocol_sha256':sha256(root/'reports/evaluation_protocol.json'),
            'input_hashes':{**input_hashes,'test_expected':protocol['splits']['test']['sha256']},
            'metadata_source':'fit split timestamps only','prior_test_already_observed':True,
            'correction_plan_sha256':sha256(root/'reports/correction_plan.json'),'fit_fingerprint':data.fingerprint,
            'fit':{'interactions':len(frame),'users':len(data.user_ids),'items':len(data.item_ids),'max_timestamp':data.max_timestamp},
            'query_timestamp':query,'fit_policy':lock['fit'],'test_policy':lock['post_test_policy'],
            'confidence_intervals':protocol['confidence_intervals']}
        if query!=int(pd.Timestamp(lock['query_time_utc']).timestamp()):raise ValueError('Query differs from frozen choice')
        write_immutable_json(output/'final_plan.json',plan)
        journal(output,'choices_frozen_test_sealed',locked_config_sha256=plan['locked_config_sha256'])
        manifest=refit_all(data,frame,metadata,lock,output,local,event=lambda e,**kw:journal(output,e,**kw))
        if len(manifest)!=5 or gate.opened:raise RuntimeError('All refits must finish before any test access')
        journal(output,'all_five_refits_completed_test_still_sealed')
        gate.allow_test=True;journal(output,'test_unsealed_for_fixed_evaluation')
        test_path=root/protocol['splits']['test']['path'];test_hash=sha256(test_path)
        if test_hash!=protocol['splits']['test']['sha256']:raise ValueError('Changed test source')
        test=pd.read_csv(test_path,dtype={'user_id':'string','item_id':'string','rating':'int8','timestamp':'int64'})
        journal(output,'test_read',sha256=test_hash,rows=len(test))
        context=EvaluationContext.from_frames(frame,test,fit_data=data,k=protocol['k'],query_timestamp=query)
        expected=protocol['evaluation_diagnostics']['test']
        for key in ['evaluation_users','catalog_items','relevant_interactions']:
            if context.diagnostics[key]!=expected[key]:raise ValueError(f'Test cohort differs: {key}')
        results={'Popularity':evaluate_popularity(context,batch_size=lock['evaluation_batch_size'])}
        for name,folder in MODEL_FOLDERS.items():
            print(f'Final test evaluation: {name} ...',flush=True)
            model=restore_model(name,data,frame,lock,local/folder)
            results[name]=evaluate_model(model,context,batch_size=lock['evaluation_batch_size'])
            journal(output,'test_evaluated',model=name,cohort_fingerprint=context.fingerprint)
            del model
        rows=[]
        for name,result in results.items():
            folder=MODEL_FOLDERS.get(name,'popularity');directory=local/folder;directory.mkdir(parents=True,exist_ok=True)
            with deterministic_gzip(directory/'test_per_user.csv.gz') as stream:result.per_user.to_csv(stream,lineterminator='\n')
            np.save(directory/'test_recommendation_indices.npy',result.recommendations,allow_pickle=False)
            rows.append({'model':name,**result.metrics,'fit_seconds':manifest[name]['fit_seconds'] if name in manifest else 0.})
        comparison=pd.DataFrame(rows);comparison.to_csv(output/'comparison.csv',index=False)
        pd.concat({name:history_slices(r) for name,r in results.items()},names=['model']).to_csv(output/'history_slices.csv')
        pd.concat({name:popularity_slices(r,context) for name,r in results.items()},names=['model']).to_csv(output/'popularity_slices.csv')
        intervals=[];reference=lock['validation_reference'];boot=protocol['confidence_intervals']
        for name,result in results.items():
            if name==reference:continue
            table=paired_bootstrap(result,results[reference],resamples=boot['resamples'],seed=boot['seed'],confidence=boot['confidence'],result_name=name,baseline_name=reference)
            table.insert(0,'comparison',name+' - '+reference);intervals.append(table)
        intervals=pd.concat(intervals,ignore_index=True);intervals.to_csv(output/'bootstrap_vs_validation_reference.csv',index=False)
        shifts=[]
        for name in MODEL_FOLDERS:
            for metric in ['ndcg_at_10','recall_at_10','catalog_coverage_at_10']:
                shifts.append({'model':name,'metric':metric,'validation':lock['validation_metrics'][name][metric],
                               'test':results[name].metrics[metric]})
        pd.DataFrame(shifts).to_csv(output/'validation_vs_test.csv',index=False)
        journal(output,'final_evaluation_completed_no_choices_changed',evaluated_models=5,control='Popularity')
        summary={**plan,'test_evaluated':True,'completed_refits':5,'completed_test_evaluations':len(results),
            'test_sha256':test_hash,'test_rows':len(test),'cohort_fingerprint':context.fingerprint,'cohort':context.diagnostics,
            'results':{name:r.metrics for name,r in results.items()},'refit_manifest':manifest,
            'bootstrap_intervals':intervals.to_dict(orient='records'),
            'interval_interpretation':'corrective reevaluation of previously observed holdout after v2 choices frozen; not a new independent test; paired user bootstrap conditional on fixed fitted models; unadjusted multiple comparisons',
            'test_open_operations':gate.open_count,
            'artifact_hashes':{str(p.relative_to(root)):sha256(p) for p in
                [*sorted(local.rglob('*')),*sorted(output.glob('*.csv')),output/'final_plan.json',output/'execution_journal.jsonl'] if p.is_file()}}
        write_json(output/'summary.json',summary)
        print(comparison.to_string(index=False),flush=True)
        return load_completed_final(root)
    finally:gate.active=False


if __name__=='__main__':run_final()
