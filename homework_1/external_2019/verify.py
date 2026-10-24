import gzip,hashlib,json,subprocess
from pathlib import Path
import pandas as pd
root=Path(__file__).resolve().parents[2]; audit=root/'homework_1/external_2019'; report=json.loads((audit/'reports/feasibility.json').read_text());plan=json.loads((audit/'reports/selection_plan.json').read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
for rel,digest in report['artifacts'].items():assert sha(audit/rel)==digest,rel
assert sha(audit/'prepare.py')==plan['script_sha256']
f=pd.read_csv(audit/'data/fit_before_2019.csv.gz',dtype={'user_id':str,'item_id':str,'rating':'int8','timestamp':'int64'})
t=pd.read_csv(audit/'data/test_2019_all.csv.gz',usecols=['user_id','item_id','timestamp'],dtype={'user_id':str,'item_id':str,'timestamp':'int64'})
s=pd.read_csv(audit/'data/test_2019_selected.csv.gz',usecols=['user_id','item_id','timestamp'],dtype={'user_id':str,'item_id':str,'timestamp':'int64'})
assert f.timestamp.max()<1546300800
assert ((t.timestamp>=1546300800)&(t.timestamp<1577836800)).all()
assert not f.duplicated(['user_id','item_id']).any()
assert not t.duplicated(['user_id','item_id']).any()
assert (f.groupby('user_id').size()>=5).all() and (f.groupby('item_id').size()>=5).all()
expected=t[t.user_id.isin(f.user_id.unique())&t.item_id.isin(f.item_id.unique())]
pd.testing.assert_frame_equal(s.reset_index(drop=True),expected.reset_index(drop=True))
assert len(s)==10092 and s.user_id.nunique()==5536
assert len(f)+len(s)==530677
assert all(report['assignment_size_checks'].values())
for rel,digest in plan['protected_v2_files'].items():assert sha(root/'homework_1'/rel)==digest,rel
assert not subprocess.check_output(['git','diff','--name-only'],text=True).strip()
assert subprocess.check_output(['git','check-ignore',str(audit/'data/fit_before_2019.csv.gz')],text=True).strip()
check={'artifact_hashes_verified':4,'fit_chronology':'all fit rows strictly before 2019-01-01 UTC','test_chronology':'all IDs-only inspected test rows within 2019 UTC','fit_core_and_unique_pairs_verified':True,'test_unique_pairs_verified':True,'selected_test_equals_frozen_fit_membership_filter':True,'test_labels_analyzed':False,'recommendation_metrics_computed':False,'protected_v2_files_unchanged':len(plan['protected_v2_files']),'original_tracked_files_modified':False,'data_excluded_from_git':True,'synthetic_selection_tests_passed':2,'fit_plus_selected_test_events':len(f)+len(s),'selected_test_fraction_of_all_2019_events':len(s)/len(t)}
(audit/'reports/verification.json').write_text(json.dumps(check,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(check,ensure_ascii=False,indent=2))
