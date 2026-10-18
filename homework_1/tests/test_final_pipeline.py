"""Final choice lock, access order, fresh refits and serialization invariants."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from data_utils import FitData
from evaluation import EvaluationContext,evaluate_model
from scripts.final_config import write_immutable_json,validation_winner
from scripts.run_final_test import TestReadGate,refit_all,restore_model,MODEL_FOLDERS


class FinalPipelineTests(unittest.TestCase):
    def test_lock_is_immutable_and_validation_ties_follow_original_rules(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'frozen.json';write_immutable_json(p,{'models':{'a':1}})
            write_immutable_json(p,{'models':{'a':1}})
            with self.assertRaises(ValueError):write_immutable_json(p,{'models':{'a':2}})
            self.assertEqual(json.loads(p.read_text()),{'models':{'a':1}})
        table=pd.DataFrame([{'ndcg_at_10':.1,'regularization':1},{'ndcg_at_10':.1,'regularization':10}])
        self.assertEqual(validation_winner('EASE',table).regularization,10)
        table=pd.DataFrame([{'ndcg_at_10':.2,'n_factors':32,'n_bins':12,'regularization':.1},
                            {'ndcg_at_10':.2,'n_factors':16,'n_bins':36,'regularization':.01}])
        self.assertEqual(validation_winner('timeSVD++',table).n_factors,16)

    def test_test_access_gate_blocks_test_until_refits_and_always_blocks_full_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            gate=TestReadGate(tmp)
            with self.assertRaises(RuntimeError):gate('open',(str(gate.test),'r',0))
            self.assertFalse(gate.opened)
            gate.allow_test=True;gate('open',(str(gate.test),'r',0));self.assertTrue(gate.opened)
            with self.assertRaises(RuntimeError):gate('open',(str(next(iter(gate.full))),'r',0))
            gate.active=False;gate('open',(str(gate.test),'r',0));self.assertEqual(gate.open_count,1)

    def test_all_five_refits_restore_and_use_union_mask_and_catalog(self):
        rows=[]
        for u in range(10):
            for j in range(3):rows.append((f'u{u}',f'i{u+j:02}',5 if j!=1 else 2,1+j))
        train=pd.DataFrame(rows,columns=['user_id','item_id','rating','timestamp'])
        validation=pd.DataFrame([(f'u{u}',f'i{u+3:02}',4,10) for u in range(10)],columns=train.columns)
        frame=pd.concat([train,validation],ignore_index=True);data=FitData.from_frame(frame)
        hold=pd.DataFrame([(f'u{u}',f'i{u+4:02}',5,20) for u in range(9)],columns=train.columns)
        context=EvaluationContext.from_frames(frame,hold,fit_data=data,k=2,query_timestamp=20)
        lock={'query_time_utc':pd.Timestamp(20,unit='s',tz='UTC').isoformat(),'models':{
          'ItemKNN':{'n_neighbors':3,'shrinkage':1.},
          'iALS':{'n_factors':2,'n_iterations':2,'alpha':10.,'regularization':1.},
          'EASE':{'regularization':10.},
          'FM':{'n_factors':2,'n_epochs':2,'batch_size':8},
          'timeSVD++':{'n_factors':2,'n_bins':2,'n_epochs':2}},
          'fm_preprocessing':{'minimum_items_per_metadata_token':1,'negatives_per_positive':2,'negative_sampling_seed':137}}
        metadata={item:{'brand':'maker','categories':['Video Games','PC']} for item in data.item_ids}
        events=[]
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'reports';local=Path(tmp)/'data'
            manifest=refit_all(data,frame,metadata,lock,output,local,event=lambda e,**kw:events.append((e,kw['model'])))
            self.assertEqual(len(manifest),5)
            self.assertEqual(sum(e=='refit_completed' for e,n in events),5)
            for name,folder in MODEL_FOLDERS.items():
                self.assertEqual(manifest[name]['fit_fingerprint'],data.fingerprint)
                model=restore_model(name,data,frame,lock,local/folder)
                first=evaluate_model(model,context,batch_size=1);second=evaluate_model(model,context,batch_size=128)
                np.testing.assert_array_equal(first.recommendations,second.recommendations)
                self.assertEqual(first.metrics['users'],9)
                for row,chosen in zip(context.user_indices,first.recommendations):
                    self.assertTrue(set(chosen).isdisjoint(data.seen_items(row)))
                # Validation-only items must now be in the fit vocabulary/catalog.
                self.assertIn('i12',set(model.item_ids))
                if name=='FM':self.assertEqual(model.features_.fit_fingerprint,data.fingerprint)


if __name__=='__main__':unittest.main()
